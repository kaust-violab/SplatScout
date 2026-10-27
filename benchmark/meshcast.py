"""Mesh-side helpers of the evaluator: load the laser-scanned mesh, load the ground-truth entities (parts + official
objects), attach faces, cast whole-frame rays (embree) and render the occlusion-aware membership share of a Gaussian subset."""
import os

import numpy as np

from paths import ANNO_ROOT, SCANNETPP
from scannetpp_anno import load_official

STRUCTURAL = {"wall", "floor", "ceiling", "remove", "split", "object", "structure"}
ALPHA_THR = 0.5         # the pixel must really show a surface
SHARE_THR = 0.5         # membership-share threshold


def load_mesh(scene):
    import trimesh
    m = trimesh.load(os.path.join(SCANNETPP, scene, "scans", "mesh_aligned_0.05.ply"),
                     process=False)   # process=False: 保持顶点顺序, 标注索引才对得上
    return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64), m


def load_entities(scene, n_verts):
    """返回 (parts, objs). 每个实体自带 verts(顶点索引数组).

    **不建全局顶点标签数组**: 部件之间本来就会重叠 —— A3D 的门包含它自己的把手, 同一个盖子
    还可能被标两遍(见 GT_ISSUES.md)。若强行"一个顶点只归一个部件", 后写的会把先写的吃掉,
    实测 737 个部件里 68 个因此丢掉 >30% 的真值、8 个几乎被吃光。故各实体独立成集。
    """
    parts, objs = [], []
    pj = json.load(open(os.path.join(ANNO_ROOT, scene, "parts.json")))
    assert pj["n_vertices"] == n_verts, f"网格顶点数不符: {pj['n_vertices']} vs {n_verts}"
    for p in pj["parts"]:
        idx = np.asarray(p["vert_indices"], np.int64)
        parts.append({"kind": "part", "name": p["part_id"], "label": p["label"],
                      "n_verts": len(idx), "verts": idx})
    for o in load_official(scene):
        lab = o["label"].strip().lower()
        if o["status"] != "kept" or lab in STRUCTURAL or not o["verts"]:
            continue
        idx = np.fromiter(o["verts"], np.int64)
        objs.append({"kind": "obj", "name": str(o["obj_id"]), "label": o["label"],
                     "n_verts": len(idx), "verts": idx})
    return parts, objs


def attach_faces(ents, faces, n_verts):
    """给每个实体算出属于它的面片索引 (3 顶点里 >=2 属于该实体), 存在 ent['faces']."""
    vmask = np.zeros(n_verts, bool)
    for e in ents:
        vmask[:] = False
        vmask[e["verts"]] = True
        e["faces"] = np.nonzero(vmask[faces].sum(1) >= 2)[0].astype(np.int64)


def cast_frame(mesh, cam):
    """整帧光线求交 -> 命中面片 id 图 (H,W), 未命中 -1."""
    import trimesh.ray.ray_pyembree as rp
    H, W = cam["height"], cam["width"]
    R = np.array(cam["rotation"]); t = np.array(cam["position"])
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    d = np.stack([(u - cam["cx"]) / cam["fx"], (v - cam["cy"]) / cam["fy"],
                  np.ones_like(u)], -1).reshape(-1, 3)
    d = d @ np.asarray(R).T
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    inter = getattr(cast_frame, "_inter", None)
    if inter is None or cast_frame._mesh is not mesh:
        inter = rp.RayMeshIntersector(mesh)
        cast_frame._inter, cast_frame._mesh = inter, mesh
    tri = inter.intersects_first(np.repeat(t[None], len(d), 0), d)
    return tri.reshape(H, W)


def render_share(bp, torch, splats, sel, viewmat, K, W, H):
    """全场景合成渲染, 子集=1 其余=0 -> (归属份额图, alpha 图), numpy (H,W)."""
    with torch.no_grad():
        m = sel.float().unsqueeze(1)
        colors, alphas = bp.rasterize(
            splats["means"], splats["quats"], splats["scales"], splats["opacities"],
            m, viewmat[None], K[None], W, H, render_mode="RGB")
        M, A = colors[0, :, :, 0], alphas[0, :, :, 0]
        return (M / A.clamp(min=1e-6)).cpu().numpy(), A.cpu().numpy()


def claim_gaussians(splats, verts):
    """每个高斯认领到最近网格顶点: 返回 (nn_idx, nn_dist) numpy."""
    from scipy.spatial import cKDTree
    means = splats["means"].detach().cpu().numpy().astype(np.float64)
    dist, idx = cKDTree(verts).query(means, workers=-1)
    return idx, dist

