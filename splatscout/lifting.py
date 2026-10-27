"""切段 / 粗认领 / 并组 / 全票提升 / 精档循环 (施工方案 §4.3 步 2-3, §4.4).

规则全部沿生产管线 (merge3d / back_project), 只是把"全帧"换成"模型点的几帧".
"""
import numpy as np
import torch

from . import config


# ── 切段 ──
def cut_segments(sam_out, fids, word):
    """SAM3 输出 -> 段. 同一 SAM3 编号在帧号上空缺 >= GAP_SPLIT 帧处剪断 (跨大跳可能跟错, 并组只合不拆).

    sam_out: {局部下标: [(oid, prob, mask)]}; fids: 局部下标 -> 全局帧号.
    返回 [{"word", "frames": [(fid, mask)], "scores": [prob]}] 按首帧排序.
    """
    by_oid = {}
    for li in sorted(sam_out):
        for oid, prob, mask in sam_out[li]:
            by_oid.setdefault(oid, []).append((fids[li], prob, mask))
    segs = []
    for oid, rows in by_oid.items():
        rows.sort(key=lambda r: r[0])
        cur = []
        for r in rows:
            if cur and r[0] - cur[-1][0] - 1 >= config.GAP_SPLIT:
                segs.append(cur)
                cur = []
            cur.append(r)
        if cur:
            segs.append(cur)
    out = []
    for rows in segs:
        out.append({"word": word, "frames": [(f, m) for f, _, m in rows],
                    "scores": [p for _, p, _ in rows]})
    out.sort(key=lambda s: s["frames"][0][0])
    return out


# ── 粗认领 ──
def rough_claims(scene, segments):
    """每段一次单视角 (或段内几帧) 投票: Tot>0 & 2W1>Tot & W1>=0.3 -> (N,K) bool."""
    per_frame = {}
    for k, s in enumerate(segments):
        for fid, m in s["frames"]:
            per_frame.setdefault(fid, []).append((k, m))
    W1, Tot = scene.vote(per_frame, len(segments))
    claims = (Tot > 1e-6) & (2.0 * W1 > Tot) & (W1 >= config.CLAIM_MIN_W1)
    return claims


def merge_groups(claims, iom=config.MERGE_IOM):
    """并查集: 认领集合 交/较小者 >= iom 的段并成一组. claims (N,K) bool torch. 返回 [[k,...]]."""
    K = claims.shape[1]
    parent = list(range(K))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    with torch.autocast(device_type="cuda", enabled=False):   # 进程级 bf16 autocast (SAM3 留下的) 下矩阵乘会变 bf16
        c = claims.float()
        sizes = c.sum(dim=0).cpu().numpy()
        inter = (c.T @ c).float().cpu().numpy()
    for i in range(K):
        for j in range(i + 1, K):
            m = min(sizes[i], sizes[j])
            if m > 0 and inter[i, j] / m >= iom:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
    groups = {}
    for k in range(K):
        groups.setdefault(find(k), []).append(k)
    return sorted(groups.values(), key=lambda g: min(g))


def build_groups(segments, groups, claims):
    """组 = 同一物体: 逐帧掩码取并 (同帧两段 = SAM3 重复检出), 粗成员 = 认领并集."""
    out = []
    for g in groups:
        frames = {}
        scores = []
        for k in g:
            for fid, m in segments[k]["frames"]:
                frames[fid] = (frames[fid] | m) if fid in frames else m.copy()
            scores += segments[k]["scores"]
        rough = claims[:, torch.tensor(g, device=claims.device)].any(dim=1).nonzero().flatten().cpu().numpy()
        out.append({"word": segments[g[0]]["word"], "frames": frames, "score": float(np.mean(scores)),
                    "rough_members": rough.astype(np.int64)})
    out.sort(key=lambda gr: -sum(int(m.sum()) for m in gr["frames"].values()))
    return out


# ── 全票提升 ──
def full_lift(scene, frames, acc=None):
    """frames: {fid: mask}. 全票 W1 > W0 -> 帧数>=2 时中心共识 >= 2 帧 -> 碎块判定.

    acc = (W1, Tot, 已投过的帧集合): 精档逐轮补帧时只给新帧投票, 票数累加 (第五次 demo 每轮全量重投,
    一次精提 40 s). 返回 (members np.int64 或 None, info dict, acc).
    """
    W1, Tot, done = acc if acc is not None else (None, None, set())
    valid = {fid: m for fid, m in frames.items() if m.any()}
    if not valid:
        return None, {"reason": "empty masks"}, (W1, Tot, done)
    new = {fid: [(0, m)] for fid, m in valid.items() if fid not in done}
    if new:
        w1, tot = scene.vote(new, 1)
        W1 = w1[:, 0] if W1 is None else W1 + w1[:, 0]
        Tot = tot[:, 0] if Tot is None else Tot + tot[:, 0]
        done = done | set(new)
    acc = (W1, Tot, done)
    members = ((Tot > 1e-6) & (W1 > Tot - W1)).nonzero().flatten().cpu().numpy().astype(np.int64)
    info = {"n_vote": int(len(members)), "n_frames": len(valid), "single_view": len(valid) < config.MIN_CONSENSUS_VIEWS}
    if len(members) == 0:
        return None, dict(info, reason="no gaussian got a majority vote"), acc
    if len(valid) >= config.MIN_CONSENSUS_VIEWS:
        cnt = scene.centers_in_mask_count(members, list(valid.items()))
        members = members[cnt >= config.MIN_CONSENSUS_VIEWS]
        info["n_consensus"] = int(len(members))
    if len(members) < config.MIN_INST_GAUSSIANS:
        return None, dict(info, reason=f"only {len(members)} gaussians (< {config.MIN_INST_GAUSSIANS}), treated as debris"), acc
    return members, info, acc


def thorough_lift(scene, word, frames, members, seg_more, acc=None, banned=()):
    """精档: 反投影补帧到成员饱和. seg_more(word, fids, boxes, members) -> {fid: mask 或 None} 由工具层
    提供 (在这些帧上切词, 挑出与当前成员投影最贴合的掩码). 返回 (members, frames, curve, info)."""
    curve = [int(len(members))]
    frames = dict(frames)
    info = {"rounds": 0}
    for _ in range(config.THOROUGH_ROUNDS):
        cands = scene.frames_seeing(members, config.THOROUGH_MIN_VISIBLE, exclude=set(frames) | set(banned))   # banned: 观察帧不补
        if not cands:
            info["stop"] = "no more frames see it"
            break
        fids = scene.spread_by_direction(members, cands, config.THOROUGH_FRAMES_PER_ROUND)
        boxes = {f: scene.member_box(members, f, pad=config.THOROUGH_BOX_PAD) for f in fids}
        boxes = {f: b for f, b in boxes.items() if b is not None}
        got = seg_more(word, list(boxes), boxes, members)
        new = {f: m for f, m in got.items() if m is not None and m.any()}
        for f in boxes:                     # 切过但没找到的帧也记下, 下一轮不再选
            frames.setdefault(f, new.get(f, np.zeros((scene.H, scene.W), bool)))
        info["rounds"] += 1
        if not new:
            curve.append(curve[-1])
            continue
        m2, inf2, acc = full_lift(scene, frames, acc)      # 只给新帧投票, 旧票累加
        if m2 is None:
            info["stop"] = "re-vote collapsed: " + inf2.get("reason", "")
            break
        members = m2
        curve.append(int(len(members)))
        if abs(curve[-1] - curve[-2]) <= config.THOROUGH_STOP_DELTA * max(curve[-2], 1):
            info["stop"] = "saturated"
            break
    else:
        info["stop"] = "round limit"
    return members, {f: m for f, m in frames.items() if m.any()}, curve, info


def best_overlap(proj_mask, cands, min_iou=0.3):
    """cands: [(prob, mask)] -> 与 proj_mask IoU 最高的掩码, 低于 min_iou 返回 None."""
    best, best_iou = None, min_iou
    pa = proj_mask.sum()
    for _, m in cands:
        inter = np.logical_and(proj_mask, m).sum()
        union = pa + m.sum() - inter
        iou = inter / union if union > 0 else 0.0
        if iou >= best_iou:
            best, best_iou = m, iou
    return best


def iom(a, b):
    """交 / 较小者, a b 为下标数组."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    inter = len(np.intersect1d(a, b, assume_unique=True))
    return inter / min(len(a), len(b))


def mutual_fit(a, b):
    """交 / 较大者 (merge3d.MERGE_MUTUAL_FIT 同义): 两边互相覆盖都高才接近 1."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    inter = len(np.intersect1d(a, b, assume_unique=True))
    return inter / max(len(a), len(b))


def inside_fraction(mask, parent):
    """候选掩码落在父掩码内的像素比例."""
    n = mask.sum()
    return float(np.logical_and(mask, parent).sum() / n) if n else 0.0


def box_mask(box, shape):
    m = np.zeros(shape, bool)
    x0, y0, x1, y1 = box
    m[y0:y1, x0:x1] = True
    return m
