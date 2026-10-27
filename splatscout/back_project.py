"""Rendering and geometry helpers on top of gsplat (2DGS): load a checkpoint, rasterize per-Gaussian attributes,
expected depth / alpha, project Gaussian centers, PCA bounding boxes and farthest-point sampling.
Reads the SCENE environment variable at import time (see legacy.bind_scene); one process serves one scene."""
import os

import numpy as np
import torch

from . import scene_paths

SCENE = os.environ.get("SCENE", "")
SCENE_DIR = scene_paths.scene_dir(SCENE)
IMAGES_DIR = os.path.join(SCENE_DIR, "images")
CAMERAS_JSON = os.path.join(SCENE_DIR, "cameras.json")
OUT_DIR = scene_paths.lift_dir(SCENE)

GS2D = os.environ.get("GS2D", "1") == "1"       # rasterize() is the single dispatch point between 2DGS and 3DGS
CKPT_PATH = scene_paths.ckpt_path(SCENE, GS2D)

DEVICE = "cuda"
SH_DEGREE = 3


def scale_K(K, from_wh, to_wh):
    """mask 分辨率和 cameras.json 不同时, 内参同比缩."""
    K = K.copy()
    K[0] *= to_wh[0] / from_wh[0]
    K[1] *= to_wh[1] / from_wh[1]
    return K


def load_splats():
    """gsplat ckpt -> 激活后的参数 (scales=exp, opacities=sigmoid)."""
    ck = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    sp = ck["splats"]
    means = sp["means"].to(DEVICE).float()
    quats = sp["quats"].to(DEVICE).float()
    scales = torch.exp(sp["scales"].to(DEVICE).float())
    opacities = torch.sigmoid(sp["opacities"].to(DEVICE).float())
    sh = torch.cat([sp["sh0"], sp["shN"]], dim=1).to(DEVICE).float()  # (N,16,3)
    print(f"[load] {means.shape[0]} 个高斯 ({'2DGS' if GS2D else '3DGS'}) <- {CKPT_PATH}")
    return dict(means=means, quats=quats, scales=scales, opacities=opacities, sh=sh)


def rasterize(means, quats, scales, opacities, colors, viewmats, Ks, W, H, **kw):
    """统一光栅化入口: 3DGS/2DGS 按 GS2D 分发, 返回 (colors, alphas), 语义一致."""
    if GS2D:
        from gsplat import rasterization_2dgs
        out = rasterization_2dgs(means, quats, scales, opacities, colors,
                                 viewmats, Ks, W, H, **kw)
        return out[0], out[1]
    from gsplat import rasterization
    c, a, _ = rasterization(means, quats, scales, opacities, colors,
                            viewmats, Ks, W, H, **kw)
    return c, a


@torch.no_grad()
def render_rgb(splats, viewmat, K, W, H):
    colors, _ = rasterize(
        splats["means"], splats["quats"], splats["scales"], splats["opacities"],
        splats["sh"], viewmat[None], K[None], W, H,
        sh_degree=SH_DEGREE, render_mode="RGB",
    )
    return colors[0].clamp(0, 1).cpu().numpy()  # (H,W,3)


@torch.no_grad()
def render_depth_alpha(splats, viewmat, K, W, H, sel=None):
    """期望深度 D (H,W) 与 alpha A (H,W). sel 给定时只渲该子集高斯."""
    means, quats = splats["means"], splats["quats"]
    scales, opac = splats["scales"], splats["opacities"]
    if sel is not None:
        means, quats, scales, opac = means[sel], quats[sel], scales[sel], opac[sel]
    dummy = torch.ones(means.shape[0], 1, device=DEVICE)
    colors, alphas = rasterize(
        means, quats, scales, opac, dummy, viewmat[None], K[None], W, H,
        render_mode="ED",  # expected depth (按 alpha 归一)
    )
    return colors[0, :, :, -1], alphas[0, :, :, 0]  # D, A


def project_centers(means, viewmat, K):
    """高斯中心 -> 像素坐标. 返回 (u, v, z) 各 (N,)."""
    R, t = viewmat[:3, :3], viewmat[:3, 3]
    p = means @ R.T + t
    z = p[:, 2]
    zs = z.clamp(min=1e-6)
    u = K[0, 0] * p[:, 0] / zs + K[0, 2]
    v = K[1, 1] * p[:, 1] / zs + K[1, 2]
    return u, v, z


# ──────────────────────────────────────────────────────────────────────

def _pca_obb(P):
    """2-98 分位 OBB: PCA 主轴 + 分位裁剪, 返回 center/axes/extent/axis_conf."""
    mu = P.mean(axis=0)
    X = P - mu
    _, S, Vt = np.linalg.svd(X[np.random.RandomState(0).choice(len(X), min(len(X), 20000), replace=False)], full_matrices=False)
    proj = X @ Vt.T
    lo, hi = np.percentile(proj, 2, axis=0), np.percentile(proj, 98, axis=0)
    center = mu + ((lo + hi) / 2) @ Vt
    return {
        "center": center.tolist(),
        "axes": Vt.tolist(),           # 行向量 = 主轴方向
        "extent": (hi - lo).tolist(),  # 各主轴方向的边长
        # 主轴置信度 (奇异值比): 近对称物体 (苹果/球) 比值 ~1, 轴方向不稳定,
        # 下游用轴方向的谓词 (facing 等) 应据此降级
        "axis_conf": [round(float(S[0] / max(S[1], 1e-9)), 3),
                      round(float(S[1] / max(S[2], 1e-9)), 3)],
    }


def _fps(P, k, seed=0):
    """最远点采样: 从 P (N,3) 里选 k 个覆盖性好的点, 返回下标."""
    n = len(P)
    if n <= k:
        return np.arange(n)
    rs = np.random.RandomState(seed)
    if n > 4096:  # 先随机降到 4096 再 FPS, 控制 O(kN)
        base = rs.choice(n, 4096, replace=False)
    else:
        base = np.arange(n)
    Q = P[base]
    sel = [int(rs.randint(len(Q)))]
    d = np.linalg.norm(Q - Q[sel[0]], axis=1)
    for _ in range(k - 1):
        i = int(d.argmax())
        sel.append(i)
        d = np.minimum(d, np.linalg.norm(Q - Q[i], axis=1))
    return base[np.array(sel)]

