"""场景预处理包 (施工方案 §5): 每场景离线一次, 落 cache/agent_prep/<scene>/.

产物:
  meta.json      场景名 / 帧名表 / 图像宽高 / 高斯数 / 重力 up / 总权重
  cams.npz       c2w (F,4,4), K (F,3,3) —— 图像分辨率下的内参
  visible.npy    (F, ceil(N/8)) uint8 packbits: 每帧看得见的高斯 (未遮挡 + 投影半径 >= 3 px)
  weight.npy     (N,) float32: 不透明度 x 最大尺度^2 (近似表面积), 覆盖率按它加权
  overlap.npy    (F,F) float32: 帧间可见集合的加权交并比
  thumbs/f%05d.jpg  长边 768 的低清图, 左上印帧号

用法 (normlift 环境, 小卡即可, f9f9 755 帧约 3-4 分钟):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  python -m splatscout.scene_prep --scene f9f95681fd
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import torch

from . import config
from . import legacy


def list_frames(scene_dir, exclude=(), keep_test=False):
    """images/ 下按文件名排序 = 帧号. 返回 [(stem, path)]. 没有相机位姿的图跳过 (f9f9 有 1 张).
    评估协议 v2.1 §2.1: 帧池只含训练帧 —— cameras.json 里 split == "test" 的考试帧一律剔除 (keep_test=True 才保留, 只为复现旧数);
    exclude 是额外剔除的帧名 (LERF 场景没有 split 字段, Ref-LERF 的测试帧由调用方传进来)."""
    with open(os.path.join(scene_dir, "cameras.json")) as f:
        cams = json.load(f)
    posed = {c["img_name"] for c in cams}
    test = set() if keep_test else {c["img_name"] for c in cams if c.get("split") == "test"}
    exclude = set(exclude)
    d = os.path.join(scene_dir, "images")
    out, skipped, dropped = [], [], []
    for fn in sorted(os.listdir(d)):
        stem, ext = os.path.splitext(fn)
        if ext.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        if stem not in posed:
            skipped.append(stem)
            continue
        if stem in test or stem in exclude:
            dropped.append(stem)
            continue
        out.append((stem, os.path.join(d, fn)))
    if skipped:
        print(f"[prep] {len(skipped)} 张图没有相机位姿, 跳过: {skipped[:5]}{' ...' if len(skipped) > 5 else ''}")
    if dropped:
        print(f"[prep] {len(dropped)} 张考试帧 / 指定剔除帧不进帧池 (方法只许看训练帧)")
    list_frames.dropped = dropped
    return out



def load_cams(scene_dir, frames, img_wh):
    """cameras.json -> 每帧 (c2w, K@图像分辨率). 缺相机的帧报错 (基准场景应全有)."""
    with open(os.path.join(scene_dir, "cameras.json")) as f:
        cams = {c["img_name"]: c for c in json.load(f)}
    c2w = np.zeros((len(frames), 4, 4), np.float64)
    K = np.zeros((len(frames), 3, 3), np.float64)
    for i, (stem, _) in enumerate(frames):
        c = cams[stem]
        sx, sy = img_wh[0] / c["width"], img_wh[1] / c["height"]
        K[i] = [[c["fx"] * sx, 0, img_wh[0] / 2], [0, c["fy"] * sy, img_wh[1] / 2], [0, 0, 1]]
        m = np.eye(4)
        m[:3, :3] = np.array(c["rotation"])
        m[:3, 3] = np.array(c["position"])
        c2w[i] = m
    return c2w, K


def gravity_up(scene, c2w):
    """优先读 lift 目录已有的重力估计; 没有就用相机 -y 轴平均 (pre_relation.camera_up_prior 同法)."""
    sp = legacy.scene_paths()
    p = os.path.join(sp.lift_dir(scene), "gravity.json")
    if os.path.exists(p):
        with open(p) as f:
            return np.array(json.load(f)["up"], np.float64), "lift/gravity.json"
    ups = -c2w[:, :3, 1]
    ups /= np.linalg.norm(ups, axis=1, keepdims=True)
    u = ups.mean(axis=0)
    return u / np.linalg.norm(u), "camera_mean"


@torch.no_grad()
def visible_bits(bp, splats, viewmat, K, W, H):
    """一帧的可见高斯位 (bool N,). 判据见 config.VIS_*."""
    D, A = bp.render_depth_alpha(splats, viewmat, K, W, H)
    u, v, z = bp.project_centers(splats["means"], viewmat, K)
    ui, vi = u.round().long(), v.round().long()
    inb = (z > 1e-3) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ui_c, vi_c = ui.clamp(0, W - 1), vi.clamp(0, H - 1)
    d_at = D[vi_c, ui_c]
    a_at = A[vi_c, ui_c]
    smax = splats["scales"].max(dim=1).values
    r_px = smax * K[0, 0] / z.clamp(min=1e-3)
    vis = (inb & (a_at >= config.VIS_MIN_ALPHA)
           & ((z - d_at).abs() < config.VIS_DEPTH_TOL * z)
           & (r_px >= config.VIS_MIN_RADIUS_PX))
    return vis


def overlap_matrix(packed, weight, n_gauss, device):
    """帧间加权交并比: 分块解包到 GPU 做矩阵乘."""
    F = packed.shape[0]
    C = config.PREP_FRAME_CHUNK
    w = torch.from_numpy(weight).to(device)
    tot = torch.zeros(F, device=device)
    inter = torch.zeros(F, F, device=device)

    def chunk(a):
        bits = np.unpackbits(packed[a:a + C], axis=1)[:, :n_gauss]
        return torch.from_numpy(bits).to(device).float()

    for a in range(0, F, C):          # 两层都现场解包, 显存只占两块 (大场景也撑得住)
        ca = chunk(a) * w[None]
        tot[a:a + ca.shape[0]] = ca.sum(dim=1)
        for b in range(a, F, C):
            cb = chunk(b)
            blk = ca @ cb.T
            inter[a:a + ca.shape[0], b:b + cb.shape[0]] = blk
            inter[b:b + cb.shape[0], a:a + ca.shape[0]] = blk.T
            del cb
        del ca
    union = tot[:, None] + tot[None, :] - inter
    jac = torch.where(union > 0, inter / union.clamp(min=1e-9), torch.zeros_like(inter))
    return jac.cpu().numpy().astype(np.float32), tot.cpu().numpy()


def make_thumb(path, fid, out_path):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    s = config.THUMB_LONG / max(h, w)
    t = cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    cv2.rectangle(t, (0, 0), (92, 30), (0, 0, 0), -1)
    cv2.putText(t, f"f{fid}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(out_path, t, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return (w, h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--skip-thumbs", action="store_true")
    ap.add_argument("--only-thumbs", action="store_true")
    ap.add_argument("--keep-test", action="store_true", help="保留考试帧 (只为复现 0905/0908 的旧数; 协议不允许)")
    args = ap.parse_args()
    scene = args.scene
    legacy.bind_scene(scene)
    sp = legacy.scene_paths()
    scene_dir = sp.scene_dir(scene)
    out = config.prep_dir(scene)
    os.makedirs(os.path.join(out, "thumbs"), exist_ok=True)

    exclude = []
    frames = list_frames(scene_dir, exclude=exclude, keep_test=args.keep_test)
    if not frames:
        raise SystemExit(f"{scene_dir}/images 为空")
    img0 = cv2.imread(frames[0][1])
    H, W = img0.shape[:2]
    print(f"[prep] {scene}: {len(frames)} 帧, 图像 {W}x{H}")

    if not args.skip_thumbs:
        t0 = time.time()
        for fid, (_, p) in enumerate(frames):
            make_thumb(p, fid, os.path.join(out, "thumbs", f"f{fid:05d}.jpg"))
        print(f"[prep] 低清图 {len(frames)} 张, {time.time() - t0:.0f}s")
    if args.only_thumbs:
        return

    c2w, K = load_cams(scene_dir, frames, (W, H))
    up, up_src = gravity_up(scene, c2w)
    np.savez(os.path.join(out, "cams.npz"), c2w=c2w, K=K)

    bp = legacy.bp()
    splats = bp.load_splats()
    N = splats["means"].shape[0]
    smax = splats["scales"].max(dim=1).values
    weight = (splats["opacities"] * smax * smax).cpu().numpy().astype(np.float32)
    np.save(os.path.join(out, "weight.npy"), weight)

    t0 = time.time()
    packed = np.zeros((len(frames), (N + 7) // 8), np.uint8)
    for fid in range(len(frames)):
        viewmat = torch.from_numpy(np.linalg.inv(c2w[fid])).float().to(config.DEVICE)
        Kt = torch.from_numpy(K[fid]).float().to(config.DEVICE)
        vis = visible_bits(bp, splats, viewmat, Kt, W, H).cpu().numpy()
        packed[fid] = np.packbits(vis)
        if (fid + 1) % 100 == 0:
            print(f"[prep]   可见位图 {fid + 1}/{len(frames)}  {time.time() - t0:.0f}s")
    np.save(os.path.join(out, "visible.npy"), packed)

    jac, tot = overlap_matrix(packed, weight, N, config.DEVICE)
    np.save(os.path.join(out, "overlap.npy"), jac)
    total_w = float(weight.sum())
    meta = {
        "scene": scene, "n_frames": len(frames), "n_gaussians": int(N),
        "frame_names": [s for s, _ in frames], "image_dir": os.path.join(scene_dir, "images"),
        "image_ext": os.path.splitext(frames[0][1])[1], "width": W, "height": H,
        "up": up.tolist(), "up_source": up_src, "total_weight": total_w,
        "per_frame_weight_frac": (tot / total_w).tolist(),
        "vis_params": {"depth_tol": config.VIS_DEPTH_TOL, "min_radius_px": config.VIS_MIN_RADIUS_PX,
                       "min_alpha": config.VIS_MIN_ALPHA},
        "train_only": not args.keep_test, "excluded_frames": getattr(list_frames, "dropped", []),
    }
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"[prep] 完成 -> {out}  单帧覆盖中位 {np.median(tot) / total_w:.1%}, "
          f"最大 {tot.max() / total_w:.1%}, 总耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
