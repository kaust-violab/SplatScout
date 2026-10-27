"""Common pieces of the PartRef-ScanNet++ evaluation protocol: queries and targets, test-frame cameras, mesh ray casting
(occluded first hit / unoccluded + z depth), the Gaussian model (our 2DGS checkpoint or a 3DGS ply) with the two renderings
(track A: occlusion-aware membership share in the full scene; track B: isolated rendering + median z depth), and submission parsing.
Used by partref_gt.py / partref_score.py. Needs the SCENE environment variable for the Gaussian renderer (back_project)."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
from splatscout import scene_paths                                              # noqa: E402,F401
from paths import ANNO_ROOT, SCENES_ROOT, EVAL_DIR, QDIR, A3D_DIR               # noqa: E402
import meshcast as oc                                                           # noqa: E402
from scannetpp_anno import load_official                                        # noqa: E402

MIN_PX = 300                                            # a test frame counts only if >= this many target pixels are visible
FALSE_APPEAR_PX = 300                                   # §8 误现帧: 真值为空的帧里预测掩码超过此像素数


# ─────────────────────────── 题目 / 相机 / 真值 ───────────────────────────
def load_queries(scene):
    doc = json.load(open(os.path.join(QDIR, f"{scene}.json")))
    return [q for q in doc["queries"] if q["status"] == "active"]


def test_cams(scene):
    return [c for c in json.load(open(os.path.join(SCENES_ROOT, scene, "cameras.json"))) if c.get("split") == "test"]


def cam_mats(cam):
    R = np.asarray(cam["rotation"], np.float64)
    t = np.asarray(cam["position"], np.float64)
    K = np.array([[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1]], np.float64)
    return R, t, K, int(cam["width"]), int(cam["height"])


def valid_mask(scene, cam):
    """隐私涂抹: 白 = 有效. 两侧同时剔除 (与 score.py 同源)."""
    import cv2
    H, W = cam["height"], cam["width"]
    valid = np.ones((H, W), bool)
    if cam.get("mask"):
        m = cv2.imread(os.path.join(SCENES_ROOT, scene, "masks", cam["mask"]), cv2.IMREAD_GRAYSCALE)
        if m is not None:
            if m.shape != (H, W):
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            valid = m > 127
    return valid


def project(cam, P):
    """世界点 -> (u, v, z_cam)."""
    R, t, K, _, _ = cam_mats(cam)
    pc = (P - t) @ R                                     # R 列 = 相机轴 -> R^T (p - t)
    z = pc[:, 2]
    zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
    return K[0, 0] * pc[:, 0] / zs + K[0, 2], K[1, 1] * pc[:, 1] / zs + K[1, 2], z


class Caster:
    """网格光线求交. 整帧首命中 (有遮挡) 走 meshcast.cast_frame; 只对一组面片求交 (无遮挡) 在这里."""

    def __init__(self, scene):
        self.verts, self.faces, self.mesh = oc.load_mesh(scene)
        self._first = {}

    def first_hit(self, cam):
        """整场景首命中面片图 (H,W), 未命中 -1. 按帧缓存."""
        k = cam["img_name"]
        if k not in self._first:
            self._first[k] = oc.cast_frame(self.mesh, cam)
        return self._first[k]

    def faces_of(self, vert_idx):
        vmask = np.zeros(len(self.verts), bool)
        vmask[vert_idx] = True
        return np.nonzero(vmask[self.faces].sum(1) >= 2)[0]

    def unoccluded(self, face_idx, cam, margin=16):
        """只拿这组面片求交 -> (掩码 (H,W) bool, z 深度 (H,W) float32, 掩码外 0). 射线只发目标投影框内的, 省时间."""
        import trimesh
        import trimesh.ray.ray_pyembree as rp
        R, t, K, W, H = cam_mats(cam)
        mask = np.zeros((H, W), bool)
        depth = np.zeros((H, W), np.float32)
        if len(face_idx) == 0:
            return mask, depth
        sub_faces = self.faces[face_idx]
        vids = np.unique(sub_faces)
        u, v, z = project(cam, self.verts[vids])
        front = z > 1e-3
        if not front.any():
            return mask, depth
        u0, u1 = int(np.floor(u[front].min())) - margin, int(np.ceil(u[front].max())) + margin
        v0, v1 = int(np.floor(v[front].min())) - margin, int(np.ceil(v[front].max())) + margin
        if (~front).any():                               # 有顶点在相机后面: 投影框不可信, 全帧发射
            u0, v0, u1, v1 = 0, 0, W, H
        u0, v0, u1, v1 = max(u0, 0), max(v0, 0), min(u1, W), min(v1, H)
        if u1 <= u0 or v1 <= v0:
            return mask, depth
        sub = trimesh.Trimesh(vertices=self.verts, faces=sub_faces, process=False)
        inter = rp.RayMeshIntersector(sub)
        uu, vv = np.meshgrid(np.arange(u0, u1) + 0.5, np.arange(v0, v1) + 0.5)
        d = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu)], -1).reshape(-1, 3)
        d = d @ R.T
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        loc, idx_ray, _ = inter.intersects_location(np.repeat(t[None], len(d), 0), d, multiple_hits=False)
        if len(idx_ray) == 0:
            return mask, depth
        zc = ((loc - t) @ R)[:, 2].astype(np.float32)   # 相机坐标系 z (§4)
        py, px = np.divmod(idx_ray, u1 - u0)
        rows, cols = py + v0, px + u0
        mask[rows, cols] = True
        depth[rows, cols] = zc
        return mask, depth


# ─────────────────────────── 真值缓存 ───────────────────────────
def gt_dir(scene):
    tag = os.environ.get("PARTREF_GT_TAG", "")                                     # 另一套题集的真值另存 (partref_gt_<tag>)
    return os.path.join(EVAL_DIR, scene, "partref_gt" + (("_" + tag) if tag else ""))


def pack(m):
    return np.packbits(m.reshape(-1))


def unpack(b, shape):
    return np.unpackbits(b)[: shape[0] * shape[1]].reshape(shape).astype(bool)


def save_gt(scene, qid, frames_gt):
    """frames_gt: {frame_name: (occ_mask, unocc_mask, z_depth)} -> npz."""
    os.makedirs(gt_dir(scene), exist_ok=True)
    out = {}
    for name, (occ, unocc, z) in frames_gt.items():
        out[f"{name}__occ"] = pack(occ)
        out[f"{name}__unocc"] = pack(unocc)
        out[f"{name}__z"] = z[unocc].astype(np.float32)       # float16 在 8 m 外只有 8 mm 精度, 比 τ=1 cm 还粗, 用 float32
        out[f"{name}__shape"] = np.asarray(occ.shape, np.int32)
    np.savez_compressed(os.path.join(gt_dir(scene), f"{qid}.npz"), **out)


def load_gt(scene, qid):
    p = os.path.join(gt_dir(scene), f"{qid}.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p)
    names = sorted({k.split("__")[0] for k in d.files})
    out = {}
    for n in names:
        shape = tuple(int(x) for x in d[f"{n}__shape"])
        occ, unocc = unpack(d[f"{n}__occ"], shape), unpack(d[f"{n}__unocc"], shape)
        z = np.zeros(shape, np.float32)
        z[unocc] = d[f"{n}__z"].astype(np.float32)
        out[n] = (occ, unocc, z)
    return out


# ─────────────────────────── 高斯模型 ───────────────────────────
class SplatModel:
    """kind = '2dgs' (我们的 ckpt, 走 gsplat.rasterization_2dgs, 自带中值深度) 或 '3dgs' (同行的 INRIA ply, 走 gsplat.rasterization,
    中值深度用切片扫描逼近, §6.6). 张量都在 cuda 上."""

    def __init__(self, kind, means, quats, scales, opacities):
        import torch
        self.kind = kind
        self.means, self.quats, self.scales, self.opacities = means, quats, scales, opacities
        self.N = int(means.shape[0])
        self.device = means.device
        self.torch = torch

    @classmethod
    def ours_2dgs(cls, bp):
        sp = bp.load_splats()
        return cls("2dgs", sp["means"], sp["quats"], sp["scales"], sp["opacities"])

    @classmethod
    def from_ply(cls, path, device="cuda"):
        import torch
        from plyfile import PlyData
        v = PlyData.read(path)["vertex"]
        means = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
        quats = np.stack([v[f"rot_{i}"] for i in range(4)], 1).astype(np.float32)      # INRIA: w,x,y,z
        scales = np.exp(np.stack([v[f"scale_{i}"] for i in range(3)], 1).astype(np.float32))
        opac = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float32)))
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
        return cls("3dgs", t(means), t(quats), t(scales), t(opac))

    def _raster(self, means, quats, scales, opac, colors, vm, K, W, H, median=False):
        """-> (colors [H,W,C], alpha [H,W], median_depth [H,W] 或 None)."""
        if self.kind == "2dgs":
            from gsplat import rasterization_2dgs
            # 要中值深度必须用带 D 的模式: render_mode="RGB" 时 out[5] 是常数 1 (实测), "ED" 下 out[5] 与网格 z 差 1.5 cm (p50)
            out = rasterization_2dgs(means, quats, scales, opac, colors, vm[None], K[None], W, H,
                                     render_mode="ED" if median else "RGB", depth_mode="median")
            return out[0][0], out[1][0, :, :, 0], (out[5][0, :, :, 0] if median else None)
        from gsplat import rasterization
        c, a, _ = rasterization(means, quats, scales, opac, colors, vm[None], K[None], W, H, render_mode="RGB", packed=False)
        return c[0], a[0, :, :, 0], None

    def share(self, weights, vm, K, W, H):
        """赛道 A: 权重当属性值全场景合成 -> (份额 (H,W), alpha (H,W)) numpy."""
        torch = self.torch
        with torch.no_grad():
            c, a, _ = self._raster(self.means, self.quats, self.scales, self.opacities, weights[:, None], vm, K, W, H)
            return (c[:, :, 0] / a.clamp(min=1e-6)).cpu().numpy(), a.cpu().numpy()

    def isolated(self, idx, weights, vm, K, W, H, thr=0.5, n_slices=32):
        """赛道 B: 只渲提交集合 (软权重乘到不透明度) -> (alpha (H,W), 中值 z 深度 (H,W), 有定义处 bool) numpy.
        2DGS 且 thr == 0.5 用光栅化器自带的中值; 其余 (3DGS, 或 0.3 变体) 用切片扫描找累计不透明度首次过 thr 的深度."""
        torch = self.torch
        with torch.no_grad():
            if len(idx) == 0:
                z = np.zeros((H, W), np.float32)
                return z.copy(), z, np.zeros((H, W), bool)
            sel = torch.as_tensor(np.asarray(idx, np.int64), device=self.device)
            means, quats, scales = self.means[sel], self.quats[sel], self.scales[sel]
            opac = self.opacities[sel] * weights[sel]
            ones = torch.ones((len(sel), 1), device=self.device)
            if self.kind == "2dgs" and abs(thr - 0.5) < 1e-6:
                _, a, med = self._raster(means, quats, scales, opac, ones, vm, K, W, H, median=True)
                a_np = a.cpu().numpy()
                return a_np, med.cpu().numpy().astype(np.float32), a_np >= thr
            a_full = self._raster(means, quats, scales, opac, ones, vm, K, W, H)[1]
            zc = (means @ vm[:3, :3].T + vm[:3, 3])[:, 2]
            near = float(max(zc.min().item() - 0.05, 0.01))
            far = float(zc.max().item() + 0.05)
            ts = torch.linspace(near, far, n_slices, device=self.device)
            A_prev = torch.zeros((H, W), device=self.device)
            t_prev = torch.full((H, W), near, device=self.device)
            med = torch.zeros((H, W), device=self.device)
            done = torch.zeros((H, W), dtype=torch.bool, device=self.device)
            for tval in ts:
                m = zc < tval
                if not bool(m.any()):
                    continue
                A = self._raster(means[m], quats[m], scales[m], opac[m], ones[m], vm, K, W, H)[1]
                cross = (~done) & (A >= thr)
                if bool(cross.any()):
                    frac = ((thr - A_prev) / (A - A_prev).clamp_min(1e-8)).clamp(0.0, 1.0)
                    med = torch.where(cross, t_prev + frac * (tval - t_prev), med)
                    done |= cross
                A_prev = A
                t_prev = torch.full_like(t_prev, float(tval))
                if bool(done.all()):
                    break
            return a_full.cpu().numpy(), med.cpu().numpy().astype(np.float32), done.cpu().numpy()


def view_tensors(cam, device="cuda"):
    import torch
    R, t, K, W, H = cam_mats(cam)
    c2w = np.eye(4)
    c2w[:3, :3], c2w[:3, 3] = R, t
    vm = torch.from_numpy(np.linalg.inv(c2w)).float().to(device)
    return vm, torch.from_numpy(K).float().to(device), W, H


# ─────────────────────────── 提交解析 ───────────────────────────
def parse_submission(ans, n_gauss, membership=None, strict_status=False):
    """-> ("gauss", idx (m,), weights (n_gauss,) float32) | ("verts", idx) | None.
    支持: gaussians 列表; weights {gid: w} 或 [[gid, w], ...] (软提交); instance_gid (走库); vertices.
    strict_status: status 不是 answered 就当空答 (协议的 strict 口径; 默认与 score.py 一致, 有高斯就评)."""
    if not ans:
        return None
    if strict_status and ans.get("status") not in (None, "answered"):
        return None
    if ans.get("weights"):
        w = ans["weights"]
        pairs = list(w.items()) if isinstance(w, dict) else [(p[0], p[1]) for p in w]
        idx = np.asarray([int(g) for g, _ in pairs], np.int64)
        val = np.asarray([float(x) for _, x in pairs], np.float32)
        keep = (val > 0) & (idx >= 0) & (idx < n_gauss)
        idx, val = idx[keep], np.clip(val[keep], 0, 1)
        if len(idx) == 0:
            return None
        weights = np.zeros(n_gauss, np.float32)
        weights[idx] = val
        return "gauss", idx, weights
    if ans.get("gaussians"):
        idx = np.unique(np.asarray(ans["gaussians"], np.int64))
    elif ans.get("instance_gid") is not None and membership is not None:
        key = f"idx_{int(ans['instance_gid'])}"
        if key not in membership.files:
            return None
        idx = np.unique(np.asarray(membership[key], np.int64))
    elif ans.get("vertices"):
        return "verts", np.unique(np.asarray(ans["vertices"], np.int64))
    else:
        return None
    idx = idx[(idx >= 0) & (idx < n_gauss)]
    if len(idx) == 0:
        return None
    weights = np.zeros(n_gauss, np.float32)
    weights[idx] = 1.0
    return "gauss", idx, weights


def iou(p, g):
    u = int((p | g).sum())
    return float((p & g).sum() / u) if u > 0 else 0.0


# ─────────────────────────── query targets -> mesh vertices ───────────────────────────
def target_vertices(scene, targets, n_verts):
    """Union of the vertex indices of all targets of a query; None if any target cannot be resolved."""
    cache = target_vertices.__dict__.setdefault("_c", {})
    if scene not in cache:
        pj = json.load(open(os.path.join(ANNO_ROOT, scene, "parts.json")))
        parts = {p["part_id"]: np.asarray(p["vert_indices"], np.int64) for p in pj["parts"]}
        objs = {}
        for o in load_official(scene):
            if o["verts"]:
                objs[int(o["obj_id"])] = np.fromiter(o["verts"], np.int64)
        cache[scene] = (parts, objs)
    parts, objs = cache[scene]
    out = []
    for t in targets:
        if t["kind"] == "a3part":                                   # raw Articulate3D partId (e.g. OpenTarget queries)
            a3 = target_vertices.__dict__.setdefault("_a3", {})
            if scene not in a3:
                pj = json.load(open(os.path.join(A3D_DIR, f"{scene}_parts.json")))
                a3[scene] = {int(a["partId"]): np.asarray(a["vertIndices"], np.int64) for a in pj["data"]["annotations"]}
            v = a3[scene].get(int(t["part_id"]))
        elif t["kind"] == "vertices":
            v = np.asarray(t["vertices"], np.int64)
        else:
            v = parts.get(t.get("part_id")) if t["kind"] == "part" else objs.get(t.get("obj_id"))
        if v is None or len(v) == 0:
            return None
        out.append(v)
    return np.unique(np.concatenate(out)) if out else None
