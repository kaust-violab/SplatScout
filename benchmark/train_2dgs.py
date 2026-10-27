"""2DGS scene training with gsplat's rasterization_2dgs (recipe = gsplat v1.5.1 examples/simple_trainer_2dgs.py defaults,
with normalize off and the normal / distortion regularizers on). Self-contained: reads cameras.json / images / masks /
init_points.npz written by prep_scene.py, trains on the train split only (test frames never enter training) and writes
<SPLATSCOUT_DATA>/ckpts_2dgs/<scene>/ckpt_29999.pt (keys: step + splats{means, quats, scales, opacities, sh0, shN}).

Usage:  CUDA_VISIBLE_DEVICES=0 python benchmark/train_2dgs.py --scene 578511c8a9
"""

import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import SCENES_ROOT, DATA_ROOT  # noqa: E402

OUT_ROOT = os.path.join(DATA_ROOT, "ckpts_2dgs")


def scene_dir(scene):
    """Converted scene directory (prep_scene.py output)."""
    return os.path.join(SCENES_ROOT, scene)
C0 = 0.28209479177387814          # SH 0 阶系数



# ── 相机 + 图像: 以管线的 cameras.json 为准 (坐标系/内参约定与下游一致) ──
# ScanNet++ 扩展 (2026-08-18, 经 prep_scene.py 转换): cameras.json 可带
# cx/cy(主点)、split(test 帧不参与训练)、mask(隐私遮罩, 白=有效); images/masks 软链在场景目录下.
def load_views(scene, device):
    sdir = scene_dir(scene)
    cams = json.load(open(os.path.join(sdir, "cameras.json")))
    img_dir = os.path.join(sdir, "images")
    if not os.path.isdir(img_dir):
        raise SystemExit(f"{img_dir} not found: run benchmark/prep_scene.py --scene {scene} first")
    mask_dir = os.path.join(sdir, "masks")
    stem2file = {fn.rsplit(".", 1)[0]: fn for fn in os.listdir(img_dir)}
    views, n_masked = [], 0
    for c in cams:
        if c.get("split") == "test":
            continue                                    # 考试帧, 严禁入训练
        W, H = int(c["width"]), int(c["height"])
        K = torch.tensor([[c["fx"], 0, c.get("cx", W / 2)],
                          [0, c["fy"], c.get("cy", H / 2)], [0, 0, 1]],
                         dtype=torch.float32, device=device)
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = np.array(c["rotation"], np.float64)
        c2w[:3, 3] = np.array(c["position"], np.float64)
        img = cv2.imread(os.path.join(img_dir, stem2file[c["img_name"]]))[:, :, ::-1]
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        valid = None                                    # 隐私遮罩: 只有真有涂抹的帧才存
        if c.get("mask") and os.path.isdir(mask_dir):
            m = cv2.imread(os.path.join(mask_dir, c["mask"]), cv2.IMREAD_GRAYSCALE)
            if m is not None and (m <= 127).any():
                if m.shape != (H, W):
                    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                valid = torch.from_numpy((m > 127).astype(np.float32)).to(device)
                n_masked += 1
        views.append({
            "c2w": torch.from_numpy(c2w).float().to(device),
            "K": K, "W": W, "H": H, "valid": valid,
            "img": torch.from_numpy(np.ascontiguousarray(img)).to(device),  # uint8, 常驻显存
        })
    if n_masked:
        print(f"[load] {n_masked} 帧带隐私遮罩, 损失将剔除涂抹区")
    return views


def knn_mean_dist(points, k=3):
    """每点到最近 k 个邻居的均方距 (初始化尺度用), sklearn 与官方示例同源."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1).fit(points)
    d, _ = nn.kneighbors(points)                        # (N, k+1), 首列为自身 0
    return (d[:, 1:] ** 2).mean(1)


def create_splats(points, rgbs, scene_scale, device):
    """官方 sfm 初始化配方: knn 尺度, 不透明度 0.1, SH0 由点色而来."""
    N = points.shape[0]
    dist_avg = np.sqrt(knn_mean_dist(points))
    sh = np.zeros((N, 16, 3), np.float32)
    sh[:, 0, :] = (rgbs / 255.0 - 0.5) / C0
    t = lambda a: torch.from_numpy(np.asarray(a, np.float32)).to(device)
    splats = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(t(points)),
        "scales": torch.nn.Parameter(t(np.log(dist_avg))[:, None].repeat(1, 3)),
        "quats": torch.nn.Parameter(torch.rand(N, 4, device=device)),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((N,), 0.1, device=device))),
        "sh0": torch.nn.Parameter(t(sh[:, :1, :])),
        "shN": torch.nn.Parameter(t(sh[:, 1:, :])),
    })
    lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3,
           "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    optims = {k: torch.optim.Adam([{"params": splats[k], "lr": lr}], eps=1e-15)
              for k, lr in lrs.items()}
    return splats, optims


def render(splats, view, sh_degree, dist_loss):
    from gsplat import rasterization_2dgs
    return rasterization_2dgs(
        means=splats["means"], quats=splats["quats"],
        scales=torch.exp(splats["scales"]),
        opacities=torch.sigmoid(splats["opacities"]),
        colors=torch.cat([splats["sh0"], splats["shN"]], 1),
        viewmats=torch.linalg.inv(view["c2w"])[None], Ks=view["K"][None],
        width=view["W"], height=view["H"], sh_degree=sh_degree,
        near_plane=0.2, far_plane=200.0, render_mode="RGB+ED", distloss=dist_loss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="ScanNet++ scene id converted by prep_scene.py")
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--normal-lambda", type=float, default=5e-2)   # 官方默认, >7000 步生效
    ap.add_argument("--dist-lambda", type=float, default=1e-2)     # 官方默认, >3000 步生效
    ap.add_argument("--fixed-points", action="store_true",
                    help="OpenGaussian 的 ScanNet 口径: 高斯位置钉在初始化点上, 不稠密化不剪枝不重置, 高斯数 == 点数 (点级评测可直接对号)")
    args = ap.parse_args()
    from gsplat.strategy import DefaultStrategy
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

    torch.manual_seed(42)
    np.random.seed(42)
    device = "cuda"
    out_dir = os.path.join(OUT_ROOT, args.scene)
    os.makedirs(os.path.join(out_dir, "previews"), exist_ok=True)

    views = load_views(args.scene, device)
    npz_path = os.path.join(scene_dir(args.scene), "init_points.npz")
    if not os.path.exists(npz_path):
        raise SystemExit(f"{npz_path} not found: run benchmark/prep_scene.py --scene {args.scene} first")
    npz = np.load(npz_path)                             # laser-mesh vertices as initialization
    pts, rgbs = npz["xyz"], npz["rgb"]
    cam_pos = np.stack([v["c2w"][:3, 3].cpu().numpy() for v in views])
    # 稳健尺度 (2026-08-13): p95 而非 max —— figurines 有一台 COLMAP 配准坏帧
    # (frame_00162 距中心 149.6, 其余 298 台 <=6.1), max 会把位置学习率撑大 20 倍
    dists = np.linalg.norm(cam_pos - cam_pos.mean(0), axis=1)
    scene_scale = float(np.percentile(dists, 95)) * 1.1
    print(f"[init] {args.scene}: {len(views)} 帧 {views[0]['W']}x{views[0]['H']}, "
          f"初始点 {len(pts)}, scene_scale {scene_scale:.3f} (坐标系 = COLMAP 原始, 未归一化)")

    splats, optims = create_splats(pts, rgbs, scene_scale, device)
    strategy = DefaultStrategy(                          # 官方 2DGS 示例同款参数
        verbose=False, prune_opa=0.05, grow_grad2d=2e-4, grow_scale3d=0.01,
        prune_scale3d=0.1, refine_start_iter=500, refine_stop_iter=15_000,
        reset_every=3000, refine_every=100, key_for_gradient="gradient_2dgs")
    if args.fixed_points:                                # 全程不进 refine 分支 (含不透明度重置); 位置学习率归零
        strategy.refine_start_iter, strategy.refine_stop_iter = 0, 0
        optims["means"].param_groups[0]["lr"] = 0.0
        print(f"[init] --fixed-points: {len(pts)} 个高斯位置固定, 不稠密化不剪枝")
    strategy.check_sanity(splats, optims)
    sstate = strategy.initialize_state()
    sched = torch.optim.lr_scheduler.ExponentialLR(
        optims["means"], gamma=0.01 ** (1.0 / args.steps))
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)

    with open(os.path.join(out_dir, "cfg.json"), "w") as f:
        json.dump({**vars(args), "scene_scale": scene_scale, "normalize": False, "n_init_points": int(len(pts)),
                   "recipe": "gsplat v1.5.1 simple_trainer_2dgs defaults"}, f, indent=1)

    tic = time.time()
    order = np.random.permutation(len(views))
    cursor = 0
    pbar = tqdm(range(args.steps), desc=args.scene)
    for step in pbar:
        if cursor == 0:
            order = np.random.permutation(len(views))
        view = views[order[cursor]]
        cursor = (cursor + 1) % len(views)
        gt = view["img"].float() / 255.0                 # (H,W,3)

        sh_deg = min(step // 1000, 3)
        renders, alphas, normals, normals_surf, distort, _median, info = render(
            splats, view, sh_deg, dist_loss=True)
        colors = renders[..., :3]

        strategy.step_pre_backward(splats, optims, sstate, step, info)

        if view["valid"] is not None:                    # 隐私涂抹区不进损失
            m = view["valid"][..., None]
            l1 = ((colors[0] - gt).abs() * m).sum() / (m.sum() * 3 + 1e-8)
            sloss = 1.0 - ssim((gt * m).permute(2, 0, 1)[None],
                               (colors[0] * m).permute(2, 0, 1)[None])
        else:
            l1 = F.l1_loss(colors[0], gt)
            sloss = 1.0 - ssim(gt.permute(2, 0, 1)[None], colors[0].permute(2, 0, 1)[None])
        loss = l1 * 0.8 + sloss * 0.2
        if step > 7000:                                  # 法线一致正则 (官方起步点)
            ns = (normals_surf * alphas.detach()).squeeze(0)
            loss = loss + args.normal_lambda * \
                (1 - (normals.squeeze(0) * ns).sum(-1)).mean()
        if step > 3000:                                  # 深度失真正则 (官方起步点)
            loss = loss + args.dist_lambda * distort.mean()

        loss.backward()
        strategy.step_post_backward(splats, optims, sstate, step, info, packed=False)
        for o in optims.values():
            o.step()
            o.zero_grad(set_to_none=True)
        sched.step()

        if step % 100 == 0:
            pbar.set_postfix(loss=f"{loss.item():.3f}", GS=len(splats["means"]))
        if (step + 1) % 5000 == 0 or step == args.steps - 1:
            with torch.no_grad():
                p = psnr(colors[0].permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])
            side = np.concatenate([(colors[0].clamp(0, 1) * 255).byte().cpu().numpy(),
                                   view["img"].cpu().numpy()], axis=1)
            cv2.imwrite(os.path.join(out_dir, "previews", f"step{step+1}.jpg"),
                        side[:, :, ::-1])
            tqdm.write(f"  step {step+1}: 训练视角 PSNR {p.item():.2f}, "
                       f"GS {len(splats['means'])}, 用时 {(time.time()-tic)/60:.1f} 分")

    ck_path = os.path.join(out_dir, f"ckpt_{args.steps-1}.pt")
    torch.save({"step": args.steps - 1, "splats": splats.state_dict()}, ck_path)
    print(f"[done] ckpt -> {ck_path}\n       预览图 -> {out_dir}/previews/ (左渲染右实拍, 检查对齐)")


if __name__ == "__main__":
    main()
