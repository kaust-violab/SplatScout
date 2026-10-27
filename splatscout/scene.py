"""运行期场景: 预处理包 + GPU 上的高斯 + 渲染 / 投影 / 投票. 工具层只通过这里碰几何.

帧号 fid = 预处理包 meta.frame_names 的下标 (发给模型时写成 f123).
"""
import json
import os

import cv2
import numpy as np
import torch

from . import config
from . import legacy


class Scene:
    def __init__(self, scene, device=config.DEVICE, load_gpu=True):
        self.name = scene
        self.device = device
        d = config.prep_dir(scene)
        if not os.path.exists(os.path.join(d, "meta.json")):
            raise FileNotFoundError(f"预处理包不存在: {d}  (先跑 python -m splatscout.scene_prep --scene {scene})")
        with open(os.path.join(d, "meta.json")) as f:
            self.meta = json.load(f)
        self.prep_dir = d
        self.n_frames = self.meta["n_frames"]
        self.N = self.meta["n_gaussians"]
        self.W, self.H = self.meta["width"], self.meta["height"]
        self.up = np.array(self.meta["up"], np.float64)
        cams = np.load(os.path.join(d, "cams.npz"))
        self.c2w, self.K = cams["c2w"], cams["K"]
        self.packed = np.load(os.path.join(d, "visible.npy"))           # (F, ceil(N/8)) uint8
        self.weight = np.load(os.path.join(d, "weight.npy"))            # (N,)
        # 覆盖率的分母 = 任何一帧能看见的高斯的总权重, 不是全部高斯: f9f9 有 7.5 万个高斯没有一帧按"半径 >= 3 px"
        # 看得见 (天花板外的浮渣、远处碎点), 按全部算的话看遍所有帧也只有 73.9%, 模型永远等不到"看完了"
        reach = np.unpackbits(np.bitwise_or.reduce(self.packed, axis=0))[:self.N].astype(bool)
        self.total_weight = float(self.weight[reach].sum())
        self.total_weight_all = float(self.meta["total_weight"])       # 全部高斯的权重, 只作参考
        self.overlap = np.load(os.path.join(d, "overlap.npy"))          # (F,F)
        self.frame_weight = np.array(self.meta["per_frame_weight_frac"])
        self.splats = None
        if load_gpu:
            legacy.bind_scene(scene)
            self.bp = legacy.bp()
            self.splats = self.bp.load_splats()
            self._smax = self.splats["scales"].max(dim=1).values

    # ── 帧与图 ──
    def frame_name(self, fid):
        return self.meta["frame_names"][fid]

    def image_path(self, fid):
        return os.path.join(self.meta["image_dir"], self.frame_name(fid) + self.meta["image_ext"])

    def image(self, fid):
        img = cv2.imread(self.image_path(fid))
        if img.shape[1] != self.W or img.shape[0] != self.H:
            img = cv2.resize(img, (self.W, self.H))
        return img

    def thumb_path(self, fid):
        return os.path.join(self.prep_dir, "thumbs", f"f{fid:05d}.jpg")

    def thumb_bytes(self, fid):
        with open(self.thumb_path(fid), "rb") as f:
            return f.read()

    # ── 可见集合与覆盖 (§4.1) ──
    def visible(self, fid):
        return np.unpackbits(self.packed[fid])[:self.N].astype(bool)

    def union_visible(self, fids):
        if not fids:
            return np.zeros(self.N, bool)
        bits = np.bitwise_or.reduce(self.packed[list(fids)], axis=0)
        return np.unpackbits(bits)[:self.N].astype(bool)

    def coverage(self, fids):
        """已看帧可见并集的权重 / 总权重."""
        if not fids:
            return 0.0
        return float(self.weight[self.union_visible(fids)].sum() / self.total_weight)

    def explore_pick(self, seen, n):
        """贪心: 每次挑新增权重最大的帧. 返回 [(fid, 选后累计覆盖)]."""
        seen_bits = np.bitwise_or.reduce(self.packed[list(seen)], axis=0) if seen else np.zeros_like(self.packed[0])
        picks = []
        cand = np.ones(self.n_frames, bool)
        cand[list(seen)] = False
        for _ in range(n):
            best, best_gain = -1, 0.0
            not_seen = np.unpackbits(~seen_bits)[:self.N].astype(bool)
            w_ns = self.weight * not_seen
            # 新增 = 候选可见 & 未看 的权重和; 分块解包避免一次占太大内存
            for a in range(0, self.n_frames, 64):
                idx = np.arange(a, min(a + 64, self.n_frames))
                idx = idx[cand[idx]]
                if len(idx) == 0:
                    continue
                bits = np.unpackbits(self.packed[idx], axis=1)[:, :self.N]
                gain = bits @ w_ns
                j = int(gain.argmax())
                if gain[j] > best_gain:
                    best, best_gain = int(idx[j]), float(gain[j])
            if best < 0 or best_gain <= 0:
                break
            cand[best] = False
            seen_bits = np.bitwise_or(seen_bits, self.packed[best])
            cov = float(self.weight[np.unpackbits(seen_bits)[:self.N].astype(bool)].sum() / self.total_weight)
            picks.append((best, cov))
        return picks

    def nearby_frames(self, fid, seen, n):
        """与 fid 可见集合交并比在区间内的帧, 排除已看, 按新增覆盖排序."""
        lo, hi = config.NEARBY_JACCARD
        j = self.overlap[fid]
        cand = [int(f) for f in np.argsort(-j) if lo <= j[f] <= hi and f not in seen and f != fid][:96]
        if not cand:
            return []
        # 贪心逐张挑: 每挑一张就并进"已看", 下一张按新增算 —— 否则连号帧 (f371 f372 f373) 一起上, 四张几乎一样
        seen_bits = np.bitwise_or.reduce(self.packed[list(seen) + [fid]], axis=0)
        vis = {f: self.visible(f) for f in cand}
        picked = []
        while len(picked) < n and cand:
            not_seen = ~np.unpackbits(seen_bits)[:self.N].astype(bool)
            w_ns = self.weight * not_seen
            best = max(cand, key=lambda f: float(vis[f] @ w_ns))
            if float(vis[best] @ w_ns) <= 0:
                break
            picked.append(best)
            cand.remove(best)
            seen_bits = np.bitwise_or(seen_bits, self.packed[best])
        return picked

    # ── 成员集合的可见性 (§4.2 对准某个东西 / §4.4 精档) ──
    def visible_fraction(self, members):
        """每帧看得见的成员比例 (F,). members: 高斯下标数组."""
        members = np.asarray(members, np.int64)
        if len(members) == 0:
            return np.zeros(self.n_frames)
        byte = self.packed[:, members >> 3]
        bit = (byte >> (7 - (members & 7)).astype(np.uint8)) & 1
        return bit.sum(axis=1) / len(members)

    def project_np(self, pts, fid):
        """世界点 -> 像素 (u,v,z) numpy."""
        w2c = np.linalg.inv(self.c2w[fid])
        p = pts @ w2c[:3, :3].T + w2c[:3, 3]
        z = p[:, 2]
        zs = np.maximum(z, 1e-6)
        K = self.K[fid]
        return K[0, 0] * p[:, 0] / zs + K[0, 2], K[1, 1] * p[:, 1] / zs + K[1, 2], z

    def members_np(self, members):
        return self.splats["means"][torch.as_tensor(members, device=self.device)].cpu().numpy().astype(np.float64)

    def member_box(self, members, fid, pad=0.0):
        """成员在该帧的投影框 (只用该帧可见的成员); 无可见成员返回 None."""
        vis = self.visible(fid)[members]
        if not vis.any():
            return None
        P = self.members_np(np.asarray(members)[vis])
        u, v, _ = self.project_np(P, fid)
        x0, x1 = np.percentile(u, 1), np.percentile(u, 99)
        y0, y1 = np.percentile(v, 1), np.percentile(v, 99)
        if pad:
            dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
            x0, x1, y0, y1 = x0 - dx, x1 + dx, y0 - dy, y1 + dy
        x0, y0 = int(max(0, x0)), int(max(0, y0))
        x1, y1 = int(min(self.W, x1 + 1)), int(min(self.H, y1 + 1))
        return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None

    def region_members(self, fid, box):
        """该帧看得见、且中心投影落在框内的高斯下标. box = 原图像素 (x0,y0,x1,y1). 给 nearby(frame=, box=) 用:
        模型框住一块区域, 这些高斯就是那块区域的三维立足点, 不需要分割."""
        vis = np.nonzero(self.visible(fid))[0]
        if len(vis) == 0:
            return vis.astype(np.int64)
        P = self.members_np(vis)
        u, v, z = self.project_np(P, fid)
        x0, y0, x1, y1 = box
        ok = (z > 0) & (u >= x0) & (u < x1) & (v >= y0) & (v < y1)
        return vis[ok].astype(np.int64)

    def frames_seeing(self, members, min_frac, exclude=()):
        """[(fid, 可见比例, 投影面积像素)] 按面积降序. 面积 = 可见成员投影框面积."""
        frac = self.visible_fraction(members)
        out = []
        for fid in np.nonzero(frac >= min_frac)[0]:
            if fid in exclude:
                continue
            box = self.member_box(members, int(fid))
            if box is None:
                continue
            out.append((int(fid), float(frac[fid]), (box[2] - box[0]) * (box[3] - box[1])))
        out.sort(key=lambda t: -t[2])
        return out

    def spread_by_direction(self, members, cands, n):
        """从候选帧里挑 n 张观察方向分散的: 先取面积最大的, 再迭代取与已选夹角最小值最大者."""
        if not cands:
            return []
        c = self.members_np(members).mean(axis=0)
        dirs = {}
        for fid, _, _ in cands:
            d = self.c2w[fid][:3, 3] - c
            dirs[fid] = d / max(np.linalg.norm(d), 1e-9)
        max_area = cands[0][2]
        pool = [t for t in cands if t[2] >= config.NEARBY_AREA_FLOOR * max_area]
        picked = [pool[0][0]]
        while len(picked) < n and len(picked) < len(pool):
            best, best_ang = None, -1
            for fid, _, _ in pool:
                if fid in picked:
                    continue
                ang = min(float(np.arccos(np.clip(dirs[fid] @ dirs[p], -1, 1))) for p in picked)
                if ang > best_ang:
                    best, best_ang = fid, ang
            picked.append(best)
        return picked

    def view_label(self, members, fid, ref_fid=None):
        """观察方向的粗描述: 俯仰 (above/level/below) + 相对首帧的方位角."""
        c = self.members_np(members).mean(axis=0)
        d = self.c2w[fid][:3, 3] - c
        d /= max(np.linalg.norm(d), 1e-9)
        elev = float(np.degrees(np.arcsin(np.clip(d @ self.up, -1, 1))))
        tag = "from above" if elev > 45 else ("from below" if elev < -20 else "side view")
        if ref_fid is None or ref_fid == fid:
            return tag
        d0 = self.c2w[ref_fid][:3, 3] - c
        d0 -= (d0 @ self.up) * self.up
        dh = d - (d @ self.up) * self.up
        if np.linalg.norm(d0) < 1e-6 or np.linalg.norm(dh) < 1e-6:
            return tag
        cosang = np.clip((d0 @ dh) / (np.linalg.norm(d0) * np.linalg.norm(dh)), -1, 1)
        sign = np.sign(np.cross(d0, dh) @ self.up)
        az = float(np.degrees(np.arccos(cosang))) * (1 if sign >= 0 else -1)
        return f"{tag}, rotated {az:+.0f}deg from f{ref_fid}"

    # ── 渲染 / 反投影 (GPU) ──
    def view(self, fid):
        viewmat = torch.from_numpy(np.linalg.inv(self.c2w[fid])).float().to(self.device)
        K = torch.from_numpy(self.K[fid]).float().to(self.device)
        return viewmat, K

    @torch.no_grad()
    def render_alpha(self, members, fid):
        """只渲这些成员的 alpha 图 (H,W) numpy."""
        viewmat, K = self.view(fid)
        sel = torch.as_tensor(np.asarray(members, np.int64), device=self.device)
        with torch.autocast(device_type="cuda", enabled=False):
            _, A = self.bp.render_depth_alpha(self.splats, viewmat, K, self.W, self.H, sel=sel)
        return A.float().cpu().numpy()

    def vote(self, per_frame, n_slots):
        """反投影投票 (merge3d.claims_for 同法).

        per_frame: {fid: [(slot, mask bool HxW), ...]}; 返回 W1, Tot (N, n_slots) torch.
        每帧一次光栅化: 通道 = 各 slot 掩码 + 全 1 通道 (Tot), 对 (渲染 x 掩码) 之和反传,
        colors.grad 即 Σ T·α 的票数.
        """
        with torch.autocast(device_type="cuda", enabled=False):   # 投票必须 fp32 (SAM3 可能留下开启的 autocast)
            return self._vote(per_frame, n_slots)

    def _vote(self, per_frame, n_slots):
        sp = self.splats
        N = self.N
        W1 = torch.zeros(N, n_slots, device=self.device)
        Tot = torch.zeros(N, n_slots, device=self.device)
        ones = torch.ones(self.H, self.W, device=self.device)
        for fid, items in per_frame.items():
            items = [(s, m) for s, m in items if m.any()]
            if not items:
                continue
            viewmat, K = self.view(fid)
            for s0 in range(0, len(items), config.CH_MAX - 1):
                sub = items[s0:s0 + config.CH_MAX - 1]
                M = torch.stack([torch.from_numpy(m.astype(np.float32)).to(self.device) for _, m in sub]
                                + [ones], dim=-1)
                C = len(sub) + 1
                colors = torch.ones(N, C, device=self.device, requires_grad=True)
                O, _ = self.bp.rasterize(sp["means"], sp["quats"], sp["scales"], sp["opacities"],
                                         colors, viewmat[None], K[None], self.W, self.H, render_mode="RGB")
                (O[0] * M).sum().backward()
                g = colors.grad
                idx = torch.tensor([s for s, _ in sub], device=self.device)
                W1.index_add_(1, idx, g[:, :-1])
                Tot.index_add_(1, idx, g[:, -1:].expand(-1, len(sub)))
                del O, colors, M
        torch.cuda.empty_cache()
        return W1, Tot

    def centers_in_mask_count(self, members, frame_masks):
        """成员中心落在各帧掩码内的次数 (M,); 只数该帧看得见的成员."""
        members = np.asarray(members, np.int64)
        P = self.members_np(members)
        cnt = np.zeros(len(members), np.int64)
        for fid, mask in frame_masks:
            u, v, z = self.project_np(P, fid)
            ui, vi = np.round(u).astype(int), np.round(v).astype(int)
            ok = (z > 0) & (ui >= 0) & (ui < self.W) & (vi >= 0) & (vi < self.H)
            hit = np.zeros(len(members), bool)
            hit[ok] = mask[vi[ok], ui[ok]]
            hit &= self.visible(fid)[members]
            cnt += hit
        return cnt

    def obb_and_points(self, members):
        """质心 / OBB / FPS 点 (格式与 back_project.stage_export 一致)."""
        P = self.members_np(members)
        gscale = self._smax[torch.as_tensor(members, device=self.device)].cpu().numpy()
        sel = self.bp._fps(P, config.FPS_POINTS)
        return {"centroid": P.mean(axis=0).tolist(), "obb": self.bp._pca_obb(P),
                "pts": P[sel].astype(np.float32), "pscale": gscale[sel].astype(np.float32)}
