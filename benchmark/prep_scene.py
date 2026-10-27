"""ScanNet++ 场景 -> 管线标准布局转换器.

坐标系结论 (2026-08-18 实测 f9f95681fd):
  - dslr/colmap/images.txt 的位姿在激光网格坐标系 (OpenCV 约定), 直接可用;
  - dslr/nerfstudio/*.json 的位姿被 nerfstudio 换过世界轴 ((x,y,z)->(y,x,-z)) 且是
    OpenGL 相机约定 —— **不用它的位姿**, 只借去畸变内参与坏帧标记.

Output (<SPLATSCOUT_DATA>/scenes/<scene>/):
  cameras.json     全部好帧, OpenCV c2w + fx/fy/cx/cy + split(train/test) + mask 名
  images/ masks/   软链 -> 官方去畸变图 / 隐私遮罩 (白=有效)
  init_points.npz  激光网格顶点 (xyz+rgb), 训练初始化用

硬自检: 抽帧把网格顶点投进画面, 命中率过低直接报错拒绝输出.

Usage: SCANNETPP_DATA=/path/to/scannetpp/data python benchmark/prep_scene.py --scene f9f95681fd
"""

import argparse
import json
import os

import numpy as np

from paths import SCANNETPP, SCENES_ROOT as OUT_ROOT  # noqa: E402

MAX_INIT_PTS = 1_000_000
# 自检判据 (2026-08-18 修): 大场景里个别帧怼墙角、命中率天然只有几个百分点,
# 不能拿最小值当门; 位姿真错是"所有帧全部趋零". 改看统计形态:
PROJ_MED_MIN = 0.10       # 抽 8 帧命中率的中位数下限
PROJ_MAX_MIN = 0.25       # 且至少一帧要看到全场 25% 以上


def quat2rot(qw, qx, qy, qz):
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)]])


def colmap_c2w(images_txt):
    """images.txt -> {图名: c2w (OpenCV, 网格坐标系)}. 位姿行以 .JPG 结尾, 观测行跳过."""
    out = {}
    with open(images_txt) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or not line.upper().endswith(".JPG"):
                continue
            t = line.split()
            qw, qx, qy, qz, tx, ty, tz = map(float, t[1:8])
            R_w2c = quat2rot(qw, qx, qy, qz)
            c2w = np.eye(4)
            c2w[:3, :3] = R_w2c.T
            c2w[:3, 3] = -R_w2c.T @ np.array([tx, ty, tz])
            out[t[9]] = c2w
    return out


def mesh_vertices_rgb(scene):
    from plyfile import PlyData
    ply = PlyData.read(os.path.join(SCANNETPP, scene, "scans", "mesh_aligned_0.05.ply"))
    v = ply["vertex"]
    return (np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64),
            np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.uint8))


def proj_frac(xyz, c2w, fx, fy, cx, cy, W, H):
    """网格点投影命中率 (正深度且落在画面内)."""
    pc = (xyz - c2w[:3, 3]) @ c2w[:3, :3]
    z = pc[:, 2]
    ok = z > 0.05
    u = fx * pc[ok, 0] / z[ok] + cx
    v = fy * pc[ok, 1] / z[ok] + cy
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return float(inb.sum()) / max(len(xyz), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    args = ap.parse_args()
    scene = args.scene
    dslr = os.path.join(SCANNETPP, scene, "dslr")
    t = json.load(open(os.path.join(dslr, "nerfstudio", "transforms_undistorted.json")))
    assert t["camera_model"] == "PINHOLE", f"非针孔模型: {t['camera_model']}"
    poses = colmap_c2w(os.path.join(dslr, "colmap", "images.txt"))
    splits = json.load(open(os.path.join(dslr, "train_test_lists.json")))
    bad = {f["file_path"] for fr in ("frames", "test_frames")
           for f in t.get(fr, []) if f.get("is_bad")}
    W, H = int(t["w"]), int(t["h"])
    fx, fy, cx, cy = t["fl_x"], t["fl_y"], t["cx"], t["cy"]

    xyz, rgb = mesh_vertices_rgb(scene)

    cams, miss = [], []
    for split in ("train", "test"):
        for name in splits[split]:
            if name in bad:
                continue
            if name not in poses:
                miss.append(name)
                continue
            c2w = poses[name]
            cams.append({
                "img_name": name.rsplit(".", 1)[0],
                "fx": fx, "fy": fy, "cx": cx, "cy": cy, "width": W, "height": H,
                "position": [round(float(x), 8) for x in c2w[:3, 3]],
                "rotation": [[round(float(x), 8) for x in row] for row in c2w[:3, :3]],
                "split": split,
                "mask": name.rsplit(".", 1)[0] + ".png",
            })

    # 硬自检: 抽 8 帧投影网格, 看统计形态 (中位数 + 最大值)
    rs = np.random.RandomState(0)
    sub = xyz[rs.choice(len(xyz), min(30000, len(xyz)), replace=False)]
    fracs = []
    for c in [cams[i] for i in rs.choice(len(cams), min(8, len(cams)), replace=False)]:
        c2w = np.eye(4)
        c2w[:3, :3] = np.array(c["rotation"])
        c2w[:3, 3] = np.array(c["position"])
        fracs.append(proj_frac(sub, c2w, fx, fy, cx, cy, W, H))
    med, mx = float(np.median(fracs)), max(fracs)
    if med < PROJ_MED_MIN or mx < PROJ_MAX_MIN:
        raise SystemExit(f"投影自检失败: 命中率 {[round(f,3) for f in fracs]} "
                         f"(中位 {med:.3f}<{PROJ_MED_MIN} 或 最大 {mx:.3f}<{PROJ_MAX_MIN}) "
                         f"— 位姿/内参对不上网格, 拒绝输出")

    out = os.path.join(OUT_ROOT, scene)
    os.makedirs(out, exist_ok=True)
    json.dump(cams, open(os.path.join(out, "cameras.json"), "w"))
    for link, src in (("images", "resized_undistorted_images"),
                      ("masks", "resized_undistorted_masks")):
        dst = os.path.join(out, link)
        if not os.path.exists(dst):
            os.symlink(os.path.join(dslr, src), dst)
    if len(xyz) > MAX_INIT_PTS:
        sel = np.random.RandomState(0).choice(len(xyz), MAX_INIT_PTS, replace=False)
        xyz, rgb = xyz[sel], rgb[sel]
    np.savez_compressed(os.path.join(out, "init_points.npz"), xyz=xyz, rgb=rgb)

    n_tr = sum(c["split"] == "train" for c in cams)
    print(f"[{scene}] 帧 train {n_tr} / test {len(cams) - n_tr} (剔坏帧 {len(bad)}, "
          f"colmap 缺位姿 {len(miss)}) | 投影自检命中率 {[round(f, 3) for f in fracs]} | "
          f"初始化点 {len(xyz)}")
    print(f"-> {out}/  (cameras.json / images / masks / init_points.npz)")


if __name__ == "__main__":
    main()
