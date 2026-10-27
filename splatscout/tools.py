"""九件工具 (施工方案 v2 §4 + v3 §7): explore / nearby / segment / same / lift / relate / recall / answer / report.
每件: 参数解析 -> 确定性机械 -> ToolResult(文字 + 图 + 一行摘要); 返回时顺手把结论写进场景图 (cache.SceneGraph), 模型不改图.

记法: 帧 f123 / 分割组 s3 / 物体 i2 / 实例 #12 / 框 [x0,y0,x1,y1] 按所见图的 0-1000 网格.
分工: 感知归 explore / nearby / recall, 分割归 segment (模型画框, 分割器只切框里的东西), 三维归 lift / relate.
工具只感知或动作, 不下判断; 失败也是观察, 不抛给模型.
"""
import re
import time
from dataclasses import dataclass, field

import numpy as np

from . import config
from . import imaging
from . import lifting
from . import legacy
from . import memory_match
from . import render
from .cache import SceneGraph, norm_word


@dataclass
class ToolResult:
    text: str
    images: list = field(default_factory=list)      # [(jpeg bytes, caption)]
    summary: str = ""                               # 图被裁掉后留在历史里的一行
    progress: bool = True                           # 有没有带来新信息 (熔断用)
    error: bool = False                             # 参数错 / 用法错: 不算"无进展", 只是白走一步
    final: dict = None                              # answer / report 的收尾载荷
    files: list = field(default_factory=list)       # 落盘用: [(name, bytes)]
    payload: dict = field(default_factory=dict)     # 给可视化的结构化数据: frames / ref_frame / groups (粗成员) /
                                                    # lift (成员) / box + box_frame (模型画的框, 原图像素)


class SessionState:
    """会话状态 (v3 §3): 一个场景一份, 从第一题活到最后一题. s / i / # 的账在场景图里 (self.g), 这里只放模型看图画框
    要用的窗口、看过的帧, 和本题的临时账 (begin_query 清)."""

    def __init__(self, graph=None):
        self.g = graph if graph is not None else SceneGraph("_mem", mode="session", shape=None, root="")
        self.shown = {}                              # fid -> 最近一次发给模型的该帧图片对应的原图窗口 (x0,y0,x1,y1)
        self.seen_scene = set(self.g.looked_frames())  # 会话里看过的帧 (explore 的"没看过"按它算)
        self.begin_query(None, 0, "")

    def begin_query(self, qid, order_index=0, text=""):
        self.qid, self.order_index, self.text = qid, order_index, text
        self.seen = set()                            # 本题看过的帧
        self.view_frame = None                       # 观察帧 (Ref-LERF strict 口径): 只供 relate(view=) 与读左右前后, 不许切 / 放大 / 提升
        self.lifted = []                             # 本题提升出的实例号
        self.touched = set()                         # 本题提到过的编号 (简化图裁剪用)
        self.measurements = 0
        self.step = 0
        self.n_recall = 0
        self.fixed_frame = None                      # 消融设置 (2): 首次 segment 的帧, 之后只许在这帧上切
        self.n_explore = 0                           # 消融设置 (5): 本题 explore 次数 (只有第一次返图)

    def banned(self):
        """感知与分割都不许碰的帧 (观察帧)."""
        return {self.view_frame} if self.view_frame is not None else set()

    # 账本在图里
    @property
    def groups(self):
        return self.g.segs

    @property
    def objects(self):
        return self.g.objs

    @property
    def n_groups(self):
        return self.g.n_segs

    @property
    def n_objects(self):
        return self.g.n_objs

    def new_group(self, g, **meta):
        name = self.g.add_seg(g, self.step, **meta)
        self.touched.add(name)
        return name

    def new_object(self, gnames):
        name = self.g.new_object(gnames, self.step)
        self.touched.add(name)
        return name

    def put_in_object(self, oname, gnames):
        self.touched.add(oname)
        return self.g.put_in_object(oname, gnames, self.step)

    def look(self, fids, tool, window=None):
        """看过这些帧: 本题 + 会话 + 图 (looked 事件). window 只在整帧展示时为 None."""
        fids = [int(f) for f in fids]
        self.seen.update(fids)
        self.seen_scene.update(fids)
        self.g.note_looked(fids, tool, self.step, window)

    def resolve(self, name):
        """s3 -> 那一组 (带 groups=[s3]); i2 -> 它的几组合成一个; #12 -> 库里实例当一组 (帧掩码 + 成员). 不存在返回 None."""
        name = str(name).strip().lower()
        self.touched.add(name)
        if name.startswith("s"):
            g = self.groups.get(name)
            if g is not None:
                self.g.confirm(name)
            return dict(g, groups=[name]) if g is not None else None
        if name.startswith("#"):
            try:
                iid = int(name[1:])
            except ValueError:
                return None
            rec = self.g.get(iid)
            if rec is None:
                return None
            frames = {}
            for f in self.g.instance_frames(iid):
                m = self.g.get_instance_mask(iid, f)
                if m is not None:
                    frames[f] = m
            return {"word": rec["names"][0], "frames": frames, "rough_members": np.asarray(self.g.members[iid], np.int64),
                    "score": 1.0, "parent_id": None, "groups": list(rec.get("groups") or []), "instance": iid}
        o = self.objects.get(name)
        if o is None or not o["groups"]:
            return None
        return merge_group_dicts([self.groups[n] for n in o["groups"]], o["groups"])

    def known_ids(self):
        ids = list(self.groups) + [o for o, v in self.objects.items() if v["groups"]] + [f"#{i}" for i in self.g.ids()]
        return ", ".join(ids) or "none"

    def window(self, fid, W, H):
        """模型对该帧画框时的参照: 最近展示过的那张图的窗口; 没展示过就当整帧. 返回 (窗口, 是否整帧兜底)."""
        w = self.shown.get(fid)
        return (w, False) if w is not None else ((0, 0, W, H), True)


QueryState = SessionState      # 旧名


# ── 参数解析 (宽松: 收 "f12" / 12 / "#12" / "s3" / [x0,y0,x1,y1]) ──
def parse_frame(x):
    if isinstance(x, (int, np.integer)):
        return int(x)
    m = re.search(r"(\d+)", str(x))
    if not m:
        raise ValueError(f"bad frame id {x!r}, expected like f123")
    return int(m.group(1))


def parse_instance(x):
    if isinstance(x, (int, np.integer)):
        return int(x)
    m = re.search(r"(\d+)", str(x))
    if not m:
        raise ValueError(f"bad instance id {x!r}, expected like #12")
    return int(m.group(1))


def parse_group(x):
    s = str(x).strip().lower()
    if not re.fullmatch(r"s\d+", s):
        raise ValueError(f"bad segment-group id {x!r}, expected like s3")
    return s


def parse_target(x, allow_inst=False):
    """组号 s3 或物体号 i2 (模型宣布过是同一个东西的几组); allow_inst 时也收库里实例 #12."""
    s = str(x).strip().lower()
    if allow_inst and re.fullmatch(r"#\d+", s):
        return s
    if not re.fullmatch(r"[si]\d+", s):
        raise ValueError(f"bad id {x!r}, expected a group like s3 or an object like i2" + (" or an instance like #12" if allow_inst else ""))
    return s


def merge_group_dicts(gs, names):
    """几组当一个东西: 同帧掩码取并, 粗成员取并, 词取多数 (并列取字典序). 给 lift / nearby / within 用."""
    frames = {}
    for g in gs:
        for fid, m in g["frames"].items():
            frames[fid] = (frames[fid] | m) if fid in frames else m.copy()
    rough = (np.unique(np.concatenate([np.asarray(g["rough_members"], np.int64) for g in gs]))
             if gs else np.zeros(0, np.int64))
    words = [g["word"] for g in gs]
    word = max(sorted(set(words)), key=words.count) if words else ""
    parents = {g.get("parent_id") for g in gs}
    return {"word": word, "frames": frames, "rough_members": rough.astype(np.int64),
            "score": float(np.mean([g["score"] for g in gs])) if gs else 0.0,
            "parent_id": parents.pop() if len(parents) == 1 else None, "groups": list(names)}


def parse_box(x):
    """[x0,y0,x1,y1] 按所见图的 0-1000 网格 (全都 <=1 时当 0-1 比例). 返回四个 0-1 比例, 角点顺序已理正."""
    if isinstance(x, str):
        vals = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", x)]
    else:
        try:
            vals = [float(v) for v in x]
        except (TypeError, ValueError):
            raise ValueError(f"bad box {x!r}, expected [x0,y0,x1,y1] on a 0-{config.BOX_GRID} grid of the image")
    if len(vals) != 4:
        raise ValueError(f"bad box {x!r}, expected [x0,y0,x1,y1] on a 0-{config.BOX_GRID} grid of the image")
    fr = vals if max(vals) <= 1.0 else [v / config.BOX_GRID for v in vals]
    x0, x1 = sorted((fr[0], fr[2]))
    y0, y1 = sorted((fr[1], fr[3]))
    x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(1.0, x1), min(1.0, y1)
    if x1 - x0 < 0.005 or y1 - y0 < 0.005:
        raise ValueError(f"box {x!r} is empty or too thin: give [x0,y0,x1,y1] with x1 > x0 and y1 > y0")
    return x0, y0, x1, y1


def box_to_frame(frac, window):
    """所见图上的比例框 -> 原图像素框. window = 那张图对应的原图窗口 (整帧就是 0,0,W,H)."""
    wx0, wy0, wx1, wy1 = window
    x0 = wx0 + frac[0] * (wx1 - wx0)
    x1 = wx0 + frac[2] * (wx1 - wx0)
    y0 = wy0 + frac[1] * (wy1 - wy0)
    y1 = wy0 + frac[3] * (wy1 - wy0)
    x0, y0 = int(np.floor(x0)), int(np.floor(y0))
    x1, y1 = max(int(np.ceil(x1)), x0 + 1), max(int(np.ceil(y1)), y0 + 1)
    return x0, y0, x1, y1


def order_by_overlap(overlap_row, fids):
    """与提示帧视野重叠度从高到低排: SAM3 从提示帧往后跟踪, 相邻的先来, 跟踪更稳."""
    return sorted(fids, key=lambda f: -float(overlap_row[f]))


def mask_box_iou(mask, boxmask):
    inter = np.logical_and(mask, boxmask).sum()
    union = mask.sum() + boxmask.sum() - inter
    return float(inter / union) if union else 0.0


def fmt_frames(fids):
    return " ".join(f"f{f}" for f in fids)


def parse_evidence(x):
    """answer 的 evidence: 三元组列表 [{subject, relation, object, how}] 或一段文字. 返回 (三元组, 文字)."""
    if isinstance(x, str):
        try:
            import json as _json
            y = _json.loads(x)
            if isinstance(y, list):
                x = y
        except (ValueError, TypeError):
            pass
    if isinstance(x, list):
        triples, lines = [], []
        for t in x:
            if isinstance(t, dict):
                subj, rel, obj = str(t.get("subject", "")).strip(), str(t.get("relation", "")).strip(), str(t.get("object", "")).strip()
                how = str(t.get("how", "")).strip()
                if subj and rel and obj:
                    triples.append({"subject": subj, "relation": rel, "object": obj, "how": how})
                    lines.append(f"{subj} {rel} {obj} ({how})" if how else f"{subj} {rel} {obj}")
                elif how or subj:
                    lines.append(how or subj)
            else:
                lines.append(str(t))
        return triples, "\n".join(lines)
    return [], str(x or "")


def parse_dispute(x):
    out = []
    for d in (x if isinstance(x, list) else ([x] if x else [])):
        if isinstance(d, dict) and d.get("subject") and d.get("object"):
            out.append({"subject": str(d["subject"]), "relation": str(d.get("relation", "")), "object": str(d["object"]),
                        "why": str(d.get("why", ""))[:200]})
    return out


# 谓词库 (pre_relation) 把不过几何门的候选直接丢出表; 这里用一句话说明每条谓词的门是什么, 让"没进表"读成"不满足"而不是"没算"
RELATE_REQUIRES = {
    "above": "the candidate's footprint to overlap the anchor's by at least 30% and its bottom to sit over the anchor's top",
    "below": "the candidate's footprint to overlap the anchor's by at least 30% and its top to sit under the anchor's bottom",
    "on": "the candidate to touch the anchor, overlap it in footprint, and rest its bottom on the anchor's top surface",
    "higher": "the candidate's center to be higher than the anchor's (heights only, no horizontal constraint)",
    "lower": "the candidate's center to be lower than the anchor's (heights only, no horizontal constraint)",
    "inside": "at least 70% of the candidate to lie within the anchor",
    "inserted": "30-70% of the candidate to lie within the anchor, or its long axis to pass through the anchor",
    "contains": "most of the anchor to lie within the candidate",
    "between": "the candidate to sit between the two anchors' centers, close to the line joining them",
    "touching": "the gap between candidate and anchor to be under a tenth of the smaller one's size",
    "left": "the candidate to be to the left of the anchor as seen from the given camera",
    "right": "the candidate to be to the right of the anchor as seen from the given camera",
    "front": "the candidate to be nearer to the given camera than the anchor",
    "behind": "the candidate to be farther from the given camera than the anchor",
}
RELATE_SIBLING = {
    "below": "If you only meant 'lower in height' without being under it, use lower.",
    "above": "If you only meant 'higher in height' without being over it, use higher.",
    "lower": "If you meant 'directly under' (vertically aligned), use below.",
    "higher": "If you meant 'directly over' (vertically aligned), use above.",
    "on": "If it only touches or is merely above, try touching or above.",
    "inside": "If it is only partly inserted, try inserted.",
}


def k(n):
    return f"{n / 1000:.1f}k" if n >= 1000 else str(int(n))


class Toolbox:
    def __init__(self, scene, graph, sam3):
        self.scene, self.g, self.sam3 = scene, graph, sam3
        self.mem = graph                                  # 旧名
        self.pr = legacy.pre_relation()
        self.full = (0, 0, scene.W, scene.H)
        self.st = SessionState(graph)                     # 会话状态, 一个场景一份
        self.memory = True                                # False = 记忆停用 (09-12): segment 不算 CLIP 向量

    # ─────────────────────────── 4.1 explore ───────────────────────────
    def explore(self, st, n=config.EXPLORE_N):
        n = max(1, min(int(n), config.EXPLORE_MAX))
        before = self.scene.coverage(sorted(st.seen))                 # "没看过"按本题算 (v3 09-09: 旧题看过的帧对新题算新)
        picks = self.scene.explore_pick(st.seen, n)
        if not picks:
            return ToolResult(f"nothing left to explore: coverage {before:.0%}, every frame adds ~0 new area. "
                              f"Zoom with nearby(frame=, box=), look at a memory entry with nearby(group=), or recall(frame=).",
                              summary=f"explore: nothing new (coverage {before:.0%})", progress=False)
        fids = [f for f, _ in picks]
        st.look(fids, "explore")
        after = picks[-1][1]
        no_img = config.ABLATION == "5" and st.n_explore >= 1      # 消融 (5): 开题那次 explore 照常返图, 之后只列帧号
        st.n_explore += 1
        if no_img:
            text = (f"frames retrieved: {fmt_frames(fids)}  (images are not returned in this setting; a box on one of "
                    f"these frames is read on the whole frame)\n"
                    f"coverage {before:.0%} -> {after:.0%} of the visible scene seen so far in this query "
                    f"({len(st.seen)}/{self.scene.n_frames} frames)")
            return ToolResult(text, [], summary=f"explore: listed {fmt_frames(fids)} (no images), coverage {before:.0%}->{after:.0%}",
                              payload={"frames": fids})
        for f in fids:
            st.shown[f] = self.full
        imgs = [(self.scene.thumb_bytes(f), f"frame f{f}") for f in fids]
        text = (f"frames: {fmt_frames(fids)}  (full frames, low-res)\n"
                f"coverage {before:.0%} -> {after:.0%} of the visible scene seen so far in this query "
                f"({len(st.seen)}/{self.scene.n_frames} frames)")
        return ToolResult(text, imgs, summary=f"explore: showed {fmt_frames(fids)}, coverage {before:.0%}->{after:.0%}",
                          payload={"frames": fids})

    # ─────────────────────────── 4.2 nearby ───────────────────────────
    def nearby(self, st, frame=None, group=None, n=config.EXPLORE_N, box=None):
        n = max(1, min(int(n), config.EXPLORE_MAX))
        if group is not None:
            if box is not None:
                return ToolResult("box= goes with frame= (zoom on a region of that frame); with group= leave it out.",
                                  progress=False, error=True)
            return self._nearby_group(st, parse_target(group, allow_inst=True), n)
        if frame is None:
            return ToolResult("nearby needs frame='f102' (frames around that view), frame + box (zoom on a region), "
                              "or group='s3' (a segmented thing from other directions).", progress=False, error=True)
        fid = parse_frame(frame)
        if not 0 <= fid < self.scene.n_frames:
            return ToolResult(f"frame f{fid} does not exist (scene has f0..f{self.scene.n_frames - 1}).",
                              progress=False, error=True)
        if box is not None:
            if config.ABLATION == "5":                # 消融 (5): 没有放大; 明确的错误观测, 仍算一步
                return ToolResult("zooming (nearby with box=) is off in this setting: the navigator only lists frames. Box the thing on a "
                                  "frame you have seen (the opening frames, or a segmenter overlay), or list frames with nearby(frame=) / explore().",
                                  summary="nearby zoom rejected (off in this setting)", progress=False, error=True)
            return self._nearby_zoom(st, fid, parse_box(box), n)
        before = self.scene.coverage(sorted(st.seen))
        fids = self.scene.nearby_frames(fid, st.seen, n)
        if not fids:
            return ToolResult(f"no unseen frame overlaps f{fid} in the {config.NEARBY_JACCARD} range; "
                              f"try nearby(frame=f{fid}, box=...) to zoom on a region, nearby(group=...) on something "
                              f"segmented there, or explore().",
                              summary=f"nearby f{fid}: nothing", progress=False)
        st.look(fids, "nearby")
        after = self.scene.coverage(sorted(st.seen))
        if config.ABLATION == "5":                    # 消融 (5): 只列帧号与重叠, 不返图
            text = (f"frames near f{fid}: {fmt_frames(fids)}  (view-overlap with f{fid}: "
                    + ", ".join(f"{self.scene.overlap[fid, f]:.0%}" for f in fids)
                    + f"; images are not returned in this setting)\ncoverage {before:.0%} -> {after:.0%}")
            return ToolResult(text, [], summary=f"nearby f{fid}: listed {fmt_frames(fids)} (no images), coverage {before:.0%}->{after:.0%}",
                              payload={"frames": fids, "ref_frame": fid})
        for f in fids:
            st.shown[f] = self.full
        imgs = [(self.scene.thumb_bytes(f), f"frame f{f} (overlaps f{fid} {self.scene.overlap[fid, f]:.0%})") for f in fids]
        text = (f"frames near f{fid}: {fmt_frames(fids)}  (view-overlap with f{fid}: "
                + ", ".join(f"{self.scene.overlap[fid, f]:.0%}" for f in fids)
                + f")\ncoverage {before:.0%} -> {after:.0%}")
        return ToolResult(text, imgs, summary=f"nearby f{fid}: showed {fmt_frames(fids)}, coverage {before:.0%}->{after:.0%}",
                          payload={"frames": fids, "ref_frame": fid})

    def _nearby_zoom(self, st, fid, frac, n):
        """放大看一块区域: 本帧按框裁到原分辨率, 再用框内高斯当立足点找别的帧从别的方向看它. 不碰 SAM3."""
        if fid in st.banned():
            return ToolResult(f"frame f{fid} is the viewpoint frame: it cannot be zoomed into; use it only to read the viewpoint, "
                              f"and look at the region in other frames (nearby(frame='f{fid}') without box).", progress=False, error=True)
        H, W = self.scene.H, self.scene.W
        window, whole = st.window(fid, W, H)
        pbox = box_to_frame(frac, window)
        cw = imaging.pad_box(pbox, (H, W))
        st.look([fid], "zoom", cw)
        img = imaging.crop(self.scene.image(fid), cw)
        imaging.draw_box(img, (pbox[0] - cw[0], pbox[1] - cw[1], pbox[2] - cw[0], pbox[3] - cw[1]))
        imaging.stamp(img, f"f{fid}")
        st.shown[fid] = cw
        imgs = [(imaging.to_jpeg(img), f"f{fid} zoomed on your box (box drawn in yellow)")]
        note = f" (f{fid} had not been shown to you before; the box was read on the whole frame)" if whole else ""
        members = self.scene.region_members(fid, pbox)
        if len(members) == 0:
            return ToolResult(f"zoom on f{fid}{note}: crop of your box at full resolution. The boxed region has no "
                              f"reconstructed surface (glass, mirror, or empty), so no other view can be found by geometry.",
                              imgs, summary=f"zoom f{fid}: crop only (no surface in box)",
                              payload={"frames": [], "ref_frame": fid, "box": pbox, "box_frame": fid})
        others = []
        if n > 1:
            cands = self.scene.frames_seeing(members, config.NEARBY_MIN_VISIBLE, exclude={fid} | st.seen | st.banned())
            if not cands:
                cands = self.scene.frames_seeing(members, config.NEARBY_MIN_VISIBLE, exclude={fid} | st.banned())
            others = self.scene.spread_by_direction(members, cands, n - 1) if cands else []
        lines, shown = [], []
        frac_vis = self.scene.visible_fraction(members) if others else None
        for f in others:
            box = self.scene.member_box(members, f, pad=0.0)
            if box is None:
                continue
            cw2 = imaging.pad_box(box, (H, W))
            crop = imaging.crop(self.scene.image(f), cw2)
            imaging.stamp(crop, f"f{f}")
            st.shown[f] = cw2
            st.look([f], "zoom", cw2)
            shown.append(f)
            label = self.scene.view_label(members, f, fid)
            imgs.append((imaging.to_jpeg(crop), f"f{f}: the same region, {label}"))
            lines.append(f"  f{f}: {label}, sees {frac_vis[f]:.0%} of the region")
        text = (f"zoom on f{fid}{note}: crop of your box at full resolution; the region holds {k(len(members))} gaussians "
                f"of surface.\n" + (("other frames that see this region (cropped to it):\n" + "\n".join(lines)) if lines
                                     else f"no other frame sees {config.NEARBY_MIN_VISIBLE:.0%} or more of this region."))
        return ToolResult(text, imgs, summary=f"zoom f{fid} box: showed " + fmt_frames([fid] + shown),
                          payload={"frames": shown, "ref_frame": fid, "groups": {"region": members},
                                   "box": pbox, "box_frame": fid})

    def _nearby_group(self, st, gname, n):
        g = st.resolve(gname)
        if g is None:
            return ToolResult(f"unknown id {gname}; known so far: {st.known_ids()}", progress=False, error=True)
        members = g["rough_members"]
        if len(members) == 0:
            return ToolResult(f"{gname} has no 3D footprint yet (masks claimed nothing); segment it in another frame.", progress=False)
        cands = self.scene.frames_seeing(members, config.NEARBY_MIN_VISIBLE, exclude=set(g["frames"]) | st.seen | st.banned())
        if not cands:
            cands = self.scene.frames_seeing(members, config.NEARBY_MIN_VISIBLE, exclude=set(g["frames"]) | st.banned())
        if not cands:
            return ToolResult(f"no other frame sees >= {config.NEARBY_MIN_VISIBLE:.0%} of {gname}.",
                              summary=f"nearby {gname}: nothing", progress=False)
        fids = self.scene.spread_by_direction(members, cands, n)
        st.look(fids, "nearby")
        ref = sorted(g["frames"])[0] if g["frames"] else fids[0]
        if config.ABLATION == "5":                    # 消融 (5): 导航器不返图, 只列哪些帧从哪个方向看得见它
            lines = [f"  f{f}: {self.scene.view_label(members, f, ref)}, sees {self.scene.visible_fraction(members)[f]:.0%} of {gname}" for f in fids]
            return ToolResult(f"frames that see {gname} ('{g['word']}') from different directions (images are not returned in this setting):\n"
                              + "\n".join(lines), [], summary=f"nearby {gname}: listed {fmt_frames(fids)} (no images)",
                              payload={"frames": fids, "groups": {gname: g["rough_members"]}})
        # 消融 (6): 分割结果不可见 -> 组 / 物体 (s, i) 的轮廓不画; 消融 (7): 提升结果不可见 -> 实例 (#) 的轮廓不画
        outline = not ((config.ABLATION == "6" and not gname.startswith("#")) or (config.ABLATION == "7" and gname.startswith("#")))
        imgs, lines = [], []
        for f in fids:
            box = self.scene.member_box(members, f, pad=0.0)
            box = imaging.pad_box(box, (self.scene.H, self.scene.W))
            img = self.scene.image(f)
            if outline:
                proj = self.scene.render_alpha(members, f) > 0.5
                crop = imaging.contour_only(imaging.crop(img, box), imaging.crop(proj, box), color=(40, 200, 240), lw=2)
            else:
                crop = imaging.crop(img, box).copy()
            imaging.stamp(crop, f"f{f}")
            st.shown[f] = box
            label = self.scene.view_label(members, f, ref)
            imgs.append((imaging.to_jpeg(crop), f"f{f}: {gname} {'outlined' if outline else 'sits in this crop (not outlined in this setting)'}, {label}"))
            lines.append(f"  f{f}: {label}, sees {self.scene.visible_fraction(members)[f]:.0%} of {gname}")
        how = f"cropped, {gname} outlined in cyan" if outline else f"cropped to where {gname} sits; not outlined in this setting"
        text = (f"frames that see {gname} ('{g['word']}') from different directions ({how}):\n"
                + "\n".join(lines))
        return ToolResult(text, imgs, summary=f"nearby {gname}: showed {fmt_frames(fids)} (crops around it)",
                          payload={"frames": fids, "groups": {gname: g["rough_members"]}})

    # ─────────────────────────── 4.3 segment (框提示) ───────────────────────────
    def segment(self, st, frame, box, word, frames=None, within=None, same_as=None):
        """模型在看过的图上框住一样东西, 说它叫什么; 分割器切框里的东西, 并跟踪到模型另点的几帧.
        文字+框 先试 (概念+示例), 框里没切到再只按长相试一次 (词不对也能切到, 比如透明沥水架).
        same_as=: 模型断定框里的就是此前某组/某物体 -> 框中那组直接归入那个物体 (省一步 same())."""
        t0 = time.time()
        word = str(word).strip()
        if not word:
            return ToolResult("segment needs a word: what you call the thing you boxed (1-2 plain nouns).", progress=False, error=True)
        fid0 = parse_frame(frame)
        if not 0 <= fid0 < self.scene.n_frames:
            return ToolResult(f"frame f{fid0} does not exist (scene has f0..f{self.scene.n_frames - 1}).", progress=False, error=True)
        if fid0 in st.banned():
            return ToolResult(f"frame f{fid0} is the viewpoint frame: it only defines the viewer for left/right/front/behind and cannot be "
                              f"segmented. Box the thing on another frame (nearby(frame='f{fid0}') lists frames around it).", progress=False, error=True)
        abl_note = ""
        if config.ABLATION == "2":                    # 消融 (2): 切割帧在首次 segment 后固定, 换帧返回明确的错误观测 (仍算一步)
            if st.fixed_frame is None:
                st.fixed_frame = fid0
            elif fid0 != st.fixed_frame:
                return ToolResult(f"frame is fixed to f{st.fixed_frame} in this setting: every segment() call in this sentence must box on "
                                  f"f{st.fixed_frame} (zoom into it with nearby(frame='f{st.fixed_frame}', box=) and box there). Other frames can only be looked at.",
                                  summary=f"segment rejected: frame fixed to f{st.fixed_frame}, asked f{fid0}", progress=False, error=True)
        if config.ABLATION == "4" and within is not None:   # 消融 (4): 不给父物体范围, within 忽略并明说
            abl_note = f"\n(within={within} is ignored in this setting: the part was cut without confining it to the parent)"
            within = None
        if config.ABLATION == "6" and same_as is not None:  # 消融 (6): 分割反馈不可见, same_as 的"框中之组"无从谈起
            return ToolResult("same_as is not available in this setting: the segmenter does not report which group is in your box. "
                              "Decide afterwards and call same(groups=[...]).", summary="segment rejected: same_as off in this setting",
                              progress=False, error=True)
        frac = parse_box(box)
        H, W = self.scene.H, self.scene.W
        window, whole = st.window(fid0, W, H)
        pbox = box_to_frame(frac, window)
        extras = []
        for f in (frames if isinstance(frames, list) else [frames]) if frames else []:
            g = parse_frame(f)
            if g != fid0 and 0 <= g < self.scene.n_frames and g not in extras and g not in st.banned():
                extras.append(g)
        extras = order_by_overlap(self.scene.overlap[fid0], extras)[:config.SEG_MAX_FRAMES - 1]
        fids = [fid0] + extras
        st.look(fids, "segment")

        parent, parent_tag, parent_id = None, "", None
        if within is not None:
            parent, parent_tag, parent_id, err = self._resolve_parent(st, within)
            if err:
                return ToolResult(err, progress=False, error=True)
        parent_masks, skipped = {}, []
        if parent is not None:
            for f in fids:
                pm = self._parent_mask(parent, f)
                if pm is None or pm.sum() < 400:
                    skipped.append(f)
                else:
                    parent_masks[f] = pm
            if fid0 not in parent_masks:
                return ToolResult(f"{parent_tag} is not visible in f{fid0}, the frame you boxed; box the part in a frame "
                                  f"where the parent is visible (its own frames, or nearby(group=)).", progress=False)
            fids = [f for f in fids if f in parent_masks]
        whiten = {i: parent_masks[f] for i, f in enumerate(fids)} if parent is not None else None
        boxmask = lifting.box_mask(pbox, (H, W))
        paths = [self.scene.image_path(f) for f in fids]

        sam_out = self.sam3.run(paths, word, whiten, wh=(W, H), box=pbox)
        kept = self._kept_segments(sam_out, fids, word, parent_masks if parent is not None else None)
        best_iou = self._best_box_iou(kept, fid0, boxmask)
        n_pass, appearance_only = 1, False
        if best_iou < config.BOX_MATCH_IOU:
            sam_out2 = self.sam3.run(paths, None, whiten, wh=(W, H), box=pbox)
            kept2 = self._kept_segments(sam_out2, fids, word, parent_masks if parent is not None else None)
            best2 = self._best_box_iou(kept2, fid0, boxmask)
            n_pass = 2
            if best2 > best_iou:
                kept, best_iou, appearance_only = kept2, best2, True
        cost = f"SAM3 {len(fids)} frames x {n_pass} pass(es), {time.time() - t0:.1f}s"
        note = f"\n(f{fid0} had not been shown to you before; the box was read on the whole frame)" if whole else ""
        where = f" inside {parent_tag}" if parent is not None else ""
        zoom_hint = (f"Zoom in with nearby(frame=f{fid0}, box=...) and draw a tighter box" if config.ABLATION != "5"
                     else "Draw a tighter box on a frame you have seen (zooming is off in this setting)")
        if not kept:
            if config.ABLATION == "6":                # 消融 (6): 只报执行状态, 不解释为什么
                return ToolResult(f"segment ran: '{word}' on f{fid0}{where}: 0 groups created.{note}\n{cost}",
                                  summary=f"segment '{word}' on f{fid0}: 0 groups", progress=False,
                                  payload={"frames": fids, "box": pbox, "box_frame": fid0})
            return ToolResult(f"segmenter found nothing for '{word}' in your box on f{fid0}{where}, nor any look-alike in "
                              f"{fmt_frames(fids)}.{note}\n{zoom_hint}, or the region may hold no clear object.\n{cost}",
                              summary=f"segment '{word}' box on f{fid0}: nothing", progress=False,
                              payload={"frames": fids, "box": pbox, "box_frame": fid0})

        claims = lifting.rough_claims(self.scene, kept)
        groups = lifting.build_groups(kept, lifting.merge_groups(claims), claims)
        ious = [mask_box_iou(g["frames"][fid0], boxmask) if fid0 in g["frames"] else 0.0 for g in groups]
        bi = int(np.argmax(ious))
        matched = ious[bi] >= config.BOX_MATCH_IOU
        if matched and config.ABLATION != "6":           # 框里那组排第一 (消融 (6) 不重排: 排序不能泄漏哪组在框里)
            order = [bi] + [i for i in range(len(groups)) if i != bi]
            groups, ious = [groups[i] for i in order], [ious[i] for i in order]
        names = []
        for i, g in enumerate(groups):
            g["within"] = parent_tag or None
            g["parent_id"] = parent_id
            names.append(st.new_group(g, box=pbox, box_frame=fid0, box_iou=ious[i], boxed=(matched and i == 0)))

        lines = []
        for i, (nm, g) in enumerate(zip(names, groups)):
            fr = sorted(g["frames"])
            size = max(int(m.sum()) for m in g["frames"].values()) / (W * H)
            tag = "BOXED     " if (matched and i == 0) else "look-alike"
            lines.append(f"  {nm}  {tag}  frames {fmt_frames(fr)}  score {g['score']:.2f}  size {size:.1%} of frame  "
                         f"rough-lift {k(len(g['rough_members']))} gaussians")
        if matched:
            head = (f"'{word}' boxed on f{fid0}{where}: {names[0]} is the thing in your box (mask/box IoU {ious[0]:.2f}); "
                    f"{len(groups) - 1} look-alike(s) also found in {fmt_frames(fids)}")
        else:
            head = (f"'{word}' boxed on f{fid0}{where}: nothing matched your box (best mask/box IoU {max(ious):.2f}); "
                    f"{len(groups)} look-alike(s) found elsewhere in {fmt_frames(fids)}")
        text = head + "\n" + "\n".join(lines)
        if appearance_only:
            text += f"\n(note: '{word}' + box found nothing in the box; matched by appearance only, so the label may not fit)"
        if not matched:
            text += (f"\n{zoom_hint if config.ABLATION == '5' else f'Zoom with nearby(frame=f{fid0}, box=...) and box the thing tightly'}, "
                     f"or use a look-alike above if it is what you meant.")
        if skipped:
            text += f"\n(skipped {fmt_frames(skipped)}: {parent_tag} not visible there)"
        # 与此前各组的三维重叠 (粗成员 交/较小者): 同一个东西再切一次会重叠高. 只报数, 是不是同一个由模型判断.
        # 只和模型用过的组 (确认过的 / 提升过的) 比, 分割器顺带带回的相似块不算; 每组最多报 3 条
        earlier = [(n2, g2) for n2, g2 in st.groups.items()
                   if n2 not in names and (g2.get("confirmed") or g2.get("instance") is not None)]
        overlap_lines = []
        for nm, g in zip(names, groups):
            hits = []
            for n2, g2 in earlier:
                v = lifting.iom(g["rough_members"], g2["rough_members"])
                if v >= config.SAME_HINT_IOM:
                    hits.append((v, f"{n2} {v:.0%}" + (f" (in {g2['object']})" if g2.get("object") else "")
                                 + (f" (= #{g2['instance']})" if g2.get("instance") else "")))
            hits.sort(key=lambda t: -t[0])
            if hits:
                overlap_lines.append(f"  {nm} ~ " + ", ".join(h for _, h in hits[:3]))
        if overlap_lines:
            text += ("\n3D overlap with earlier groups (same place; check the images, then same(groups=[...]) if it is "
                     "the same thing):\n" + "\n".join(overlap_lines))
        if same_as is not None:
            text += "\n" + self._join_same(st, same_as, names[0] if matched else None)
        text += note + abl_note + f"\n{cost}"

        # 图: 提示帧叠图 (模型的框描黄) + 框中组的抠图 + 第二帧叠图 (多帧时) + 其余组抠图, 共 <= 4
        imgs = [self._overlay_image(st, fid0, names, groups, word, box=pbox)]
        cut_idx = list(range(len(groups)))
        if matched:
            imgs.append(self._cutout_image(names[0], groups[0]))
            cut_idx = cut_idx[1:]
        second = [f for f in fids if f != fid0 and any(f in g["frames"] for g in groups)]
        second.sort(key=lambda f: -sum(int(g["frames"][f].sum()) for g in groups if f in g["frames"]))
        if second and len(imgs) < config.MAX_IMAGES_PER_TOOL:
            imgs.append(self._overlay_image(st, second[0], names, groups, word))
        for i in cut_idx:
            if len(imgs) >= config.MAX_IMAGES_PER_TOOL:
                break
            imgs.append(self._cutout_image(names[i], groups[i]))
        for nm, g in zip(names, groups):
            jpg, f_cut = self._cutout_jpeg(nm, g)
            self.g.put_cutout(nm, f_cut, jpg)
            if not self.memory:
                continue
            try:                                          # 每组一个 CLIP 向量 (腐蚀掩码抠图), 查询按图文相似度找节点
                self.g.put_feature(nm, memory_match.group_feature(self.scene, g))
            except Exception as e:                        # noqa: BLE001  CLIP 出错不影响切
                print(f"[clip] feature failed for {nm}: {type(e).__name__}: {e}", flush=True)
        payload = {"frames": fids, "groups": {nm: g["rough_members"] for nm, g in zip(names, groups)}, "box": pbox, "box_frame": fid0,
                   "boxed": names[bi] if matched else None}      # 框中之组只进轨迹记录 (消融 (6) 的事后统计用), 不进模型所见
        if config.ABLATION == "6":                       # 消融 (6): 分割结果对模型不可见 -> 只报状态与组号 (无 BOXED / 帧数 / 面积 / 重叠 / 图)
            text6 = (f"segment ran: '{word}' on f{fid0}{where}: {len(groups)} group(s) created: {', '.join(names)}. "
                     f"Use these ids in lift(), same(), nearby(group=).{note}{abl_note}\n{cost}")
            return ToolResult(text6, [], summary=f"segment '{word}' on f{fid0}: groups {', '.join(names)}", payload=payload)
        summary = (f"segment '{word}' box on f{fid0}{where}: groups {', '.join(names)}"
                   + (f" ({names[0]} = boxed)" if matched else " (none matched the box)"))
        return ToolResult(text, imgs, summary=summary, payload=payload)

    def segment_label_only(self, st, frame, word, frames=None, **rejected):
        """消融 (3): 只收帧号与标签, SAM3 文本提示切出该标签的全部候选, 编号叠图 (OpenGround 式: 投影框 + 编号) 返回, 模型再 lift(编号)."""
        t0 = time.time()
        if rejected:
            return ToolResult(f"this setting takes no {', '.join(sorted(rejected))}: call segment(frame, word[, frames]) and choose among the "
                              f"numbered candidates it returns.", progress=False, error=True)
        word = str(word).strip()
        if not word:
            return ToolResult("segment needs a word: the label of what you want cut (1-2 plain nouns).", progress=False, error=True)
        fid0 = parse_frame(frame)
        if not 0 <= fid0 < self.scene.n_frames:
            return ToolResult(f"frame f{fid0} does not exist (scene has f0..f{self.scene.n_frames - 1}).", progress=False, error=True)
        if fid0 in st.banned():
            return ToolResult(f"frame f{fid0} is the viewpoint frame and cannot be segmented.", progress=False, error=True)
        H, W = self.scene.H, self.scene.W
        extras = []
        for f in (frames if isinstance(frames, list) else [frames]) if frames else []:
            g = parse_frame(f)
            if g != fid0 and 0 <= g < self.scene.n_frames and g not in extras and g not in st.banned():
                extras.append(g)
        extras = order_by_overlap(self.scene.overlap[fid0], extras)[:config.SEG_MAX_FRAMES - 1]
        fids = [fid0] + extras
        st.look(fids, "segment")
        paths = [self.scene.image_path(f) for f in fids]
        sam_out = self.sam3.run(paths, word, None, wh=(W, H), box=None)
        kept = self._kept_segments(sam_out, fids, word, None)
        cost = f"SAM3 {len(fids)} frames x 1 pass (label only), {time.time() - t0:.1f}s"
        if not kept:
            return ToolResult(f"segmenter found nothing for '{word}' in {fmt_frames(fids)}. Try another frame (nearby(frame=) / explore()) or a "
                              f"plainer noun.\n{cost}", summary=f"segment '{word}' label-only on f{fid0}: nothing", progress=False,
                              payload={"frames": fids})
        claims = lifting.rough_claims(self.scene, kept)
        groups = lifting.build_groups(kept, lifting.merge_groups(claims), claims)
        names = []
        for g in groups:
            g["within"] = None
            g["parent_id"] = None
            names.append(st.new_group(g, box=None, box_frame=fid0, box_iou=0.0, boxed=False))
        lines = []
        for nm, g in zip(names, groups):
            fr = sorted(g["frames"])
            size = max(int(m.sum()) for m in g["frames"].values()) / (W * H)
            lines.append(f"  {nm}  candidate  frames {fmt_frames(fr)}  score {g['score']:.2f}  size {size:.1%} of frame  "
                         f"rough-lift {k(len(g['rough_members']))} gaussians")
        text = (f"'{word}' on f{fid0}: {len(groups)} candidate(s) found in {fmt_frames(fids)}, numbered in the overlay. Pick the one you mean and "
                f"lift it (nearby(group=) shows a candidate from other directions).\n" + "\n".join(lines))
        earlier = [(n2, g2) for n2, g2 in st.groups.items()
                   if n2 not in names and (g2.get("confirmed") or g2.get("instance") is not None)]
        overlap_lines = []
        for nm, g in zip(names, groups):
            hits = []
            for n2, g2 in earlier:
                v = lifting.iom(g["rough_members"], g2["rough_members"])
                if v >= config.SAME_HINT_IOM:
                    hits.append((v, f"{n2} {v:.0%}" + (f" (in {g2['object']})" if g2.get("object") else "")
                                 + (f" (= #{g2['instance']})" if g2.get("instance") else "")))
            hits.sort(key=lambda t_: -t_[0])
            if hits:
                overlap_lines.append(f"  {nm} ~ " + ", ".join(h for _, h in hits[:3]))
        if overlap_lines:
            text += ("\n3D overlap with earlier groups (same place; check the images, then same(groups=[...]) if it is "
                     "the same thing):\n" + "\n".join(overlap_lines))
        text += f"\n{cost}"
        imgs = [self._candidate_overlay(st, fid0, names, groups, word)]
        second = [f for f in fids if f != fid0 and any(f in g["frames"] for g in groups)]
        second.sort(key=lambda f: -sum(int(g["frames"][f].sum()) for g in groups if f in g["frames"]))
        if second and len(imgs) < config.MAX_IMAGES_PER_TOOL:
            imgs.append(self._candidate_overlay(st, second[0], names, groups, word))
        for nm, g in zip(names, groups):
            if len(imgs) >= config.MAX_IMAGES_PER_TOOL:
                break
            imgs.append(self._cutout_image(nm, g))
        for nm, g in zip(names, groups):
            jpg, f_cut = self._cutout_jpeg(nm, g)
            self.g.put_cutout(nm, f_cut, jpg)
        return ToolResult(text, imgs, summary=f"segment '{word}' label-only on f{fid0}: candidates {', '.join(names)}",
                          payload={"frames": fids, "groups": {nm: g["rough_members"] for nm, g in zip(names, groups)}})

    def _candidate_overlay(self, st, f, names, groups, word):
        """消融 (3) 的候选叠图, 照 OpenGround 的画法: 每个候选画投影外接框 + 编号标签 (实底底色, 白字), 裁到候选并集的外接框."""
        import cv2
        H, W = self.scene.H, self.scene.W
        items = [(nm, g["frames"][f]) for nm, g in zip(names, groups) if f in g["frames"]]
        boxes = [(nm, imaging.bbox_of(m)) for nm, m in items]
        boxes = [(nm, b) for nm, b in boxes if b is not None]
        bb = None
        for _, b in boxes:
            bb = b if bb is None else (min(bb[0], b[0]), min(bb[1], b[1]), max(bb[2], b[2]), max(bb[3], b[3]))
        cw = imaging.pad_box(bb if bb is not None else (0, 0, W, H), (H, W))
        img = self.scene.image(f).copy()
        palette = [(230, 80, 40), (40, 160, 230), (60, 200, 90), (220, 170, 30), (200, 60, 200), (30, 200, 200), (240, 120, 160), (120, 120, 240)]
        for i, (nm, (x0, y0, x1, y1)) in enumerate(boxes):
            color = palette[i % len(palette)]
            cv2.rectangle(img, (int(x0), int(y0)), (int(x1) - 1, int(y1) - 1), color, 3)
            (tw, th), _ = cv2.getTextSize(nm, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            ly = max(int(y0) - 5, th + 5)
            cv2.rectangle(img, (int(x0), ly - th - 5), (int(x0) + tw + 5, ly + 5), color, -1)
            cv2.putText(img, nm, (int(x0) + 2, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        crop = imaging.crop(img, cw)
        imaging.stamp(crop, f"f{f}")
        st.shown[f] = cw
        return imaging.to_jpeg(crop), f"f{f}: '{word}' candidates, each with its box and number: " + ", ".join(nm for nm, _ in boxes)

    def _join_same(self, st, same_as, boxed):
        """segment 的 same_as=: 框中那组归入 same_as 所在的物体 (same_as 是还没物体号的组就新开一个)."""
        try:
            t = parse_target(same_as)
        except ValueError as e:
            return f"(same_as ignored: {e})"
        if boxed is None:
            return f"(same_as={t} ignored: nothing matched your box, so there is no group to join)"
        if t.startswith("i"):
            if t not in st.objects or not st.objects[t]["groups"]:
                return f"(same_as={t} ignored: unknown object; known: {st.known_ids()})"
            oname = t
        elif t in st.groups:
            oname = st.groups[t].get("object") or st.new_object([t])
        else:
            return f"(same_as={t} ignored: unknown group; known: {st.known_ids()})"
        st.put_in_object(oname, [boxed])
        return f"{boxed} joined {oname} = {' '.join(st.objects[oname]['groups'])} (declared the same thing as {t})"

    # ─────────────────────────── 4.3b same ───────────────────────────
    def same(self, st, groups, into=None):
        """模型宣布几组是同一个东西 -> 物体号 i1 (或并入已有物体). 工具只合账不判断; 顺带报三维重叠供它自查."""
        raw = groups if isinstance(groups, list) else [groups]
        try:
            names = [parse_target(x) for x in raw]
            oname = parse_target(into) if into is not None else None
        except ValueError as e:
            return ToolResult(str(e), progress=False, error=True)
        gnames, bad = [], []
        for n in names:
            if n.startswith("i"):
                if n in st.objects and st.objects[n]["groups"]:
                    gnames += st.objects[n]["groups"]
                else:
                    bad.append(n)
            elif n in st.groups:
                gnames.append(n)
            else:
                bad.append(n)
        if oname is not None and (not oname.startswith("i") or oname not in st.objects):
            bad.append(oname)
        if bad:
            return ToolResult(f"unknown id(s) {' '.join(bad)}; known so far: {st.known_ids()}", progress=False, error=True)
        gnames = list(dict.fromkeys(gnames))
        if not gnames:
            return ToolResult("same needs at least one group id.", progress=False, error=True)
        for n in gnames:
            self.g.confirm(n)
            st.touched.add(n)
        if oname is None:                                # 没指定就并入第一个已有归属的物体, 都没有就新开
            owners = [st.groups[n]["object"] for n in gnames if st.groups[n].get("object")]
            oname = owners[0] if owners else None
        moved = st.put_in_object(oname, gnames) if oname is not None else []
        if oname is None:
            oname = st.new_object(gnames)
        gl = st.objects[oname]["groups"]
        g = st.resolve(oname)
        pairs, lonely = [], []
        for a in range(len(gl)):
            best = 0.0
            for b in range(len(gl)):
                if a == b:
                    continue
                v = lifting.iom(st.groups[gl[a]]["rough_members"], st.groups[gl[b]]["rough_members"])
                best = max(best, v)
                if b > a:
                    pairs.append(f"{gl[a]}~{gl[b]} {v:.0%}")
            if len(gl) > 1 and best == 0.0:
                lonely.append(gl[a])
        words = sorted({st.groups[n]["word"] for n in gl})
        text = (f"{oname} = {' '.join(gl)} ('{g['word']}'" + (f"; words {', '.join(words)}" if len(words) > 1 else "")
                + f"): {len(gl)} group(s), frames {fmt_frames(sorted(g['frames']))}, rough members union "
                f"{k(len(g['rough_members']))} gaussians. lift({oname}) votes with all these frames.")
        if pairs:
            text += "\n3D overlap between them (rough members, inter/min): " + ", ".join(pairs)
        if lonely:
            text += (f"\nnote: {' '.join(lonely)} share no gaussians with the rest; if you are not sure they are the same "
                     f"thing, call same() again without them")
        if moved:
            text += "\n(moved: " + ", ".join(f"{gn} from {o}" for gn, o in moved) + ")"
        if st.objects[oname].get("instance"):
            text += f"\n{oname} was already lifted as #{st.objects[oname]['instance']}; lift it again to include the new groups"
        return ToolResult(text, summary=f"same: {oname} = {' '.join(gl)}",
                          payload={"groups": {oname: g["rough_members"]}, "frames": sorted(g["frames"])})

    def _kept_segments(self, sam_out, fids, word, parent_masks):
        """SAM3 输出 -> 切段 -> 过滤 (置信度; 有父物体时掩码须 >= 80% 在父掩码内)."""
        kept = []
        for s in lifting.cut_segments(sam_out, fids, word):
            if np.mean(s["scores"]) < config.SEG_MIN_SCORE:
                continue
            if parent_masks is not None:
                s["frames"] = [(f, m) for f, m in s["frames"]
                               if lifting.inside_fraction(m, parent_masks[f]) >= config.WITHIN_MIN_INSIDE]
                if not s["frames"]:
                    continue
            kept.append(s)
        return kept

    @staticmethod
    def _best_box_iou(segs, fid0, boxmask):
        best = 0.0
        for s in segs:
            for f, m in s["frames"]:
                if f == fid0:
                    best = max(best, mask_box_iou(m, boxmask))
        return best

    def _overlay_image(self, st, f, names, groups, word, box=None):
        """一帧的叠图: 各组掩码上色标号, 裁到掩码 (和模型的框) 的外接框; 框描黄. 记下窗口供模型再画框."""
        H, W = self.scene.H, self.scene.W
        items = [(g["frames"][f], nm, i) for i, (nm, g) in enumerate(zip(names, groups)) if f in g["frames"]]
        bb = imaging.bbox_of(np.any([m for m, _, _ in items], axis=0)) if items else None
        if box is not None:
            bb = box if bb is None else (min(bb[0], box[0]), min(bb[1], box[1]), max(bb[2], box[2]), max(bb[3], box[3]))
        cw = imaging.pad_box(bb, (H, W))
        ov = imaging.overlay(self.scene.image(f), items) if items else self.scene.image(f)
        if box is not None:
            imaging.draw_box(ov, box)
        crop = imaging.crop(ov, cw)
        imaging.stamp(crop, f"f{f}")
        st.shown[f] = cw
        cap = f"f{f}: '{word}' masks labeled " + ", ".join(nm for _, nm, _ in items)
        if box is not None:
            cap += "; your box in yellow"
        return imaging.to_jpeg(crop), cap

    def _cutout_jpeg(self, nm, g):
        f = max(g["frames"], key=lambda ff: int(g["frames"][ff].sum()))
        cut = imaging.cutout(self.scene.image(f), g["frames"][f])
        imaging.stamp(cut, nm, (6, 22), 0.7)
        return imaging.to_jpeg(cut, long_side=512), f

    def _cutout_image(self, nm, g):
        jpg, f = self._cutout_jpeg(nm, g)
        return jpg, f"{nm} cut out on white (from f{f})"

    def _resolve_parent(self, st, within):
        s = str(within).strip().lower()
        if re.fullmatch(r"[si]\d+", s):
            g = st.resolve(s)
            if g is None:
                return None, "", None, f"unknown group/object {s}; known so far: {st.known_ids()}"
            return {"kind": "group", "group": g}, s, None, None
        iid = parse_instance(within)
        rec = self.g.get(iid)
        if rec is None:
            return None, "", None, f"unknown instance #{iid}; known: {' '.join(f'#{i}' for i in self.g.ids()) or 'none'}"
        st.touched.add(f"#{iid}")
        return {"kind": "instance", "rec": rec}, f"#{iid}", iid, None

    def _parent_mask(self, parent, fid):
        """父掩码: 组自己的掩码 > 实例来源帧的真实掩码 > 渲染成员 alpha (外扩 5 px)."""
        if parent["kind"] == "group":
            g = parent["group"]
            if fid in g["frames"]:
                return g["frames"][fid]
            members = g["rough_members"]
        else:
            rec = parent["rec"]
            m = self.g.get_instance_mask(rec["id"], fid)
            if m is not None and m.any():
                return m
            members = self.g.members[rec["id"]]
        if len(members) == 0 or self.scene.visible_fraction(members)[fid] < 0.05:
            return None
        return imaging.dilate(self.scene.render_alpha(members, fid) > 0.5, config.WITHIN_DILATE_PX)

    # ─────────────────────────── 4.4 lift ───────────────────────────
    def lift(self, st, group, thorough=False):
        """group: 组号 s3, 或物体号 i1 (模型宣布过是同一个东西的几组; 它们的掩码按帧取并, 所有帧一起投票)."""
        t0 = time.time()
        gname = parse_target(group)
        g = st.resolve(gname)
        if g is None:
            return ToolResult(f"unknown id {gname}; known so far: {st.known_ids()}", progress=False, error=True)
        if not g["frames"]:
            return ToolResult(f"{gname} has no masks to lift.", progress=False, error=True)
        label = gname if gname.startswith("s") else f"{gname} (= {' '.join(g['groups'])})"
        frames = dict(g["frames"])
        members, info, acc = lifting.full_lift(self.scene, frames)
        if members is None:
            return ToolResult(f"lift {gname} failed: {info.get('reason')}. Segment it in more/better frames first.",
                              summary=f"lift {gname}: failed ({info.get('reason')})", progress=False)
        curve, tinfo = None, {}
        grade = "rough"
        if thorough:
            members, frames, curve, tinfo = lifting.thorough_lift(
                self.scene, g["word"], frames, members, self._seg_more_fn(g), acc, banned=st.banned())
            grade = "thorough"
        # 单视角标记按最终帧数算 (第五次 demo: 精提补到 24 帧仍带着首帧的 SINGLE-VIEW, 模型因此多跑了 4 步)
        single_view = len(frames) < config.MIN_CONSENSUS_VIEWS
        geo = self.scene.obb_and_points(members)
        sources = {f: {"word": g["word"], "within": g.get("parent_id")} for f in frames}
        # 验收图的帧 = 看得最全的帧, 也是实例的代表帧 (缩略图从它来)
        fid = max(frames, key=lambda f: self.scene.visible_fraction(members)[f])
        iid, how = self.g.add_instance(g["word"], members, geo, list(frames), grade, sources, step=st.step, source=gname,
                                       groups=list(g["groups"]), single_view=single_view, exemplar_fid=fid)
        if how != "alias":
            self.g.put_instance_masks(iid, frames)
        for n in g["groups"]:
            if n in st.groups:
                st.groups[n]["instance"] = iid
        if gname.startswith("i"):
            st.objects[gname]["instance"] = iid
        st.lifted.append(iid)
        st.touched.add(f"#{iid}")

        box = self.scene.member_box(members, fid, pad=0.0) or imaging.bbox_of(frames[fid])
        box = imaging.pad_box(box, (self.scene.H, self.scene.W))
        proj = self.scene.render_alpha(members, fid) > 0.5
        crop = imaging.contour_only(imaging.crop(self.scene.image(fid), box), imaging.crop(proj, box))
        imaging.stamp(crop, f"#{iid} f{fid}")
        if how != "alias":
            self.g.put_thumb(iid, imaging.to_jpeg(crop, long_side=config.THUMB_SIDE))
        payload = {"frames": sorted(frames), "lift": {"id": iid, "word": g["word"], "grade": grade, "members": members, "source": gname,
                                                      "groups": list(g["groups"]), "rounds": int(tinfo.get("rounds", 0)) if thorough else 0,
                                                      "curve": [int(c) for c in curve] if curve else None, "single_view": bool(single_view)}}
        if config.ABLATION == "7":                       # 消融 (7): 提升结果对模型不可见 -> 只回实例号与状态 (无尺寸 / 帧数 / 单视角标记 / 曲线 / 验收图)
            text7 = f"lifted {label} -> instance #{iid} (grade {grade})"
            if how == "alias":
                text7 += f"\nnote: same object already in library as #{iid} (alias added)"
            elif how == "upgraded":
                text7 += f"\nnote: replaced the earlier version of #{iid}"
            text7 += f"\n{time.time() - t0:.1f}s"
            return ToolResult(text7, [], summary=f"lift {gname} -> #{iid} ({grade}, {how})", payload=payload)
        st.shown[fid] = box
        st.look([fid], "lift", box)
        ext = np.array(geo["obb"]["extent"])
        text = (f"instance #{iid} '{g['word']}' from {label} . {k(len(members))} gaussians . {len(frames)} frames . grade {grade}"
                + (" . SINGLE-VIEW (position ok, extent incomplete; fine as an anchor, not as an answer)" if single_view else "")
                + f"\nsize {ext[0]:.2f} x {ext[1]:.2f} x {ext[2]:.2f} m (oriented box), centroid "
                + "(" + ", ".join(f"{c:.2f}" for c in geo["centroid"]) + ")")
        if how == "alias":
            text += f"\nnote: same object already in library as #{iid} (alias added)"
        elif how == "upgraded":
            text += f"\nnote: replaced the earlier version of #{iid}"
        if curve:
            text += f"\nthorough rounds: {' -> '.join(k(c) for c in curve)} gaussians ({tinfo.get('stop', '')})"
        text += f"\nred outline = lifted gaussians projected on f{fid}; check it matches the thing.\n{time.time() - t0:.1f}s"
        return ToolResult(text, [(imaging.to_jpeg(crop), f"#{iid} lifted, projected in red on f{fid}")],
                          summary=f"lift {gname} -> #{iid} ({k(len(members))} gaussians, {grade}, {how})", payload=payload)

    def _seg_more_fn(self, g):
        """精档补帧: 当前成员投影框最大的那帧当提示帧, 框当示例 + 组的词当概念, 一次会话跟踪到其余新帧;
        父物体存在时父掩码外刷白. 每帧挑与当前成员投影最贴合的掩码. 不走词备忘 (结果依赖示例框)."""
        parent_members = None
        if g.get("parent_id") is not None and g["parent_id"] in self.g.members:
            parent_members = self.g.members[g["parent_id"]]

        def seg_more(word, fids, boxes, members_now):
            out = {}
            if not fids:
                return out
            order = sorted(fids, key=lambda f: -(boxes[f][2] - boxes[f][0]) * (boxes[f][3] - boxes[f][1]))
            whiten = None
            if parent_members is not None:
                whiten = {i: imaging.dilate(self.scene.render_alpha(parent_members, f) > 0.5, config.WITHIN_DILATE_PX)
                          for i, f in enumerate(order)}
            sam_out = self.sam3.run([self.scene.image_path(f) for f in order], word, whiten,
                                    wh=(self.scene.W, self.scene.H), box=boxes[order[0]])
            for li, f in enumerate(order):
                rows = [(p, m) for _, p, m in sam_out.get(li, [])]
                proj = self.scene.render_alpha(members_now, f) > 0.5
                out[f] = lifting.best_overlap(proj, [r for r in rows if r[0] >= config.SEG_MIN_SCORE])
            return out
        return seg_more

    # ─────────────────────────── 4.5 relate ───────────────────────────
    def relate(self, st, pred, anchors=None, subjects=None, anchors2=None, view=None):
        raw = str(pred).strip().lower()
        pred = self.pr.CANON.get(raw)                             # 收同义词 (under -> below 等)
        if pred not in config.PREDICATES:
            return ToolResult(f"unknown predicate '{raw}'. Use one of: {', '.join(config.PREDICATES)}", progress=False, error=True)
        objs = self.g.objs_for_relation()
        if not objs:
            return ToolResult("no lifted instances yet; segment + lift the things you want to relate first.", progress=False)
        known = {o["id"] for o in objs}

        def ids_of(x):
            if x is None:
                return []
            xs = x if isinstance(x, list) else [x]
            out = [parse_instance(v) for v in xs]
            bad = [i for i in out if i not in known]
            if bad:
                raise ValueError(f"unknown instance id(s) {' '.join(f'#{b}' for b in bad)}; lifted so far: "
                                 f"{' '.join(f'#{i}' for i in sorted(known))}")
            return out

        try:
            A, S, A2 = ids_of(anchors), ids_of(subjects), ids_of(anchors2)
        except ValueError as e:
            return ToolResult(str(e), progress=False, error=True)
        c2w = None
        if pred in config.VIEW_PREDICATES:
            if view is None:
                return ToolResult(f"'{pred}' depends on the viewer: pass view=f<frame> (observer = that camera).", progress=False)
            c2w = self.scene.c2w[parse_frame(view)]
        extreme = pred in ("largest", "smallest", "highest", "lowest", "leftmost", "rightmost", "middle")
        rows = {}
        try:
            if extreme:
                pool = [o for o in objs if o["id"] in S] if S else objs
                for r in self.pr.relate(pred, "all", topk=len(pool), objs=pool, view=c2w):
                    rows[r["id"]] = (r, None)
            else:
                if not A:
                    return ToolResult(f"'{pred}' needs anchors=[instance ids].", progress=False)
                if pred == "between" and not A2:
                    return ToolResult("'between' needs anchors and anchors2.", progress=False)
                for a in A:
                    for a2 in (A2 or [None]):
                        for r in self.pr.relate(pred, f"#{a}", anchor2_name=(f"#{a2}" if a2 else None),
                                                topk=len(objs), objs=objs, view=c2w):
                            if r["id"] in A or r["id"] in A2:
                                continue
                            if r["id"] not in rows or r["conf"] > rows[r["id"]][0]["conf"]:
                                rows[r["id"]] = (r, a if a2 is None else f"{a}&{a2}")
        except SystemExit as e:                      # pre_relation 用 SystemExit 报参数错
            return ToolResult(f"relate failed: {e}", progress=False)
        if S:
            rows = {i: v for i, v in rows.items() if i in S}
        ranked = sorted(rows.values(), key=lambda t: -t[0]["conf"])
        # 被评估但没进表的实例 = 几何上不满足这条谓词 (谓词库把不过门的候选直接丢掉). 这是结果, 不是"没算"
        evaluated = [o["id"] for o in objs if o["id"] not in A and o["id"] not in A2]
        if S:
            evaluated = [i for i in evaluated if i in S]
        missing = [i for i in evaluated if i not in rows]
        ids_txt = lambda xs: " ".join(f"#{i}" for i in xs)                 # noqa: E731
        if A or A2:
            self.g.note_anchor_use(list(A) + list(A2), st.step)
            st.touched.update(f"#{a}" for a in list(A) + list(A2))
        if not ranked:
            if not evaluated:
                return ToolResult("nothing lifted besides the anchors; segment + lift a candidate first.",
                                  summary=f"relate {pred}: no candidates", progress=False)
            st.measurements += 1
            self.g.note_relate(st.step, pred, A, [], missing)
            req = RELATE_REQUIRES.get(pred, f"the geometry of '{pred}'")
            sib = RELATE_SIBLING.get(pred, "")
            return ToolResult(f"'{pred}' evaluated for {ids_txt(evaluated)}" + (f" against {ids_txt(A)}" if A else "")
                              + f": none meets it. '{pred}' requires {req}. Not meeting it is a result you can use as "
                              f"evidence (they are not '{pred}')." + (f" {sib}" if sib else ""),
                              summary=f"relate {pred}: none of {ids_txt(evaluated)} meets it")
        st.measurements += 1
        self.g.note_relate(st.step, pred, A, [(r["id"], r["conf"], r["dist"], a) for r, a in ranked], missing)
        st.touched.update(f"#{r['id']}" for r, _ in ranked[:3])
        unit = {"closest": "dist m", "farthest": "dist m", "touching": "gap m", "inside": "inside frac",
                "inserted": "inside frac", "contains": "contain frac", "between": "offset", "largest": "volume m3",
                "smallest": "volume m3", "above": "vertical gap m", "below": "vertical gap m", "on": "align err",
                "higher": "height diff m", "lower": "height diff m", "highest": "height m", "lowest": "height m",
                "left": "angle deg", "right": "angle deg", "front": "depth m", "behind": "depth m",
                "leftmost": "screen x", "rightmost": "screen x", "middle": "screen x"}.get(pred, "measure")
        lines = [f"{'instance':<22} {'conf':>5}  {unit:>14}  anchor"]
        for r, a in ranked[:config.RELATE_MAX_ROWS]:
            flags = ",".join(f"{kk}={vv}" for kk, vv in (r.get("flags") or {}).items()
                             if kk in ("zero_dist_tie", "contains_anchor", "scene_scale"))
            lines.append(f"#{r['id']:<4} {r['name'][:16]:<16} {r['conf']:>5.2f}  {r['dist']:>14.3f}  "
                         f"{('#' + str(a)) if a is not None else '-'}  {flags}")
        if len(ranked) > config.RELATE_MAX_ROWS:
            lines.append(f"... {len(ranked) - config.RELATE_MAX_ROWS} more rows truncated")
        if missing:
            shown = missing[:config.RELATE_MAX_ROWS]
            lines.append(f"not meeting '{pred}' (evaluated, absent above): {ids_txt(shown)}"
                         + (f" ... {len(missing) - len(shown)} more" if len(missing) > len(shown) else "")
                         + f"  [{pred} requires {RELATE_REQUIRES.get(pred, 'its geometry')}]")
        text = (f"relate '{pred}'" + (f" anchors {' '.join(f'#{a}' for a in A)}" if A else "")
                + (f" x {' '.join(f'#{a}' for a in A2)}" if A2 else "")
                + (f" viewer f{parse_frame(view)}" if c2w is not None else "")
                + f" -- conf = how well each candidate satisfies it (1 = fully); judge the gaps yourself:\n"
                + "\n".join(lines))
        return ToolResult(text, summary=f"relate {pred}: top " + ", ".join(f"#{r['id']}({r['conf']:.2f})" for r, _ in ranked[:3]))

    # ─────────────────────────── 4.6 answer / report ───────────────────────────
    def answer(self, st, ids, evidence="", dispute=None):
        ids_raw = ids
        try:
            ids = [parse_instance(i) for i in (ids if isinstance(ids, list) else [ids])]
        except ValueError as e:
            return ToolResult(str(e), progress=False, error=True)
        missing = [i for i in ids if i not in self.g.members]
        if missing:
            raw = [str(x).strip().lower() for x in (ids_raw if isinstance(ids_raw, list) else [ids_raw])]
            segs = [x for x in raw if re.fullmatch(r"[si]\d+", x)]
            hint = (f" {' '.join(segs)} are segment groups / objects, not instances: lift them first (lift(group, thorough=true))."
                    if segs else "")
            return ToolResult(f"unknown instance id(s) {' '.join(f'#{m}' for m in missing)}; answer must be lifted instances.{hint}",
                              progress=False)
        triples, text = parse_evidence(evidence)
        if not text.strip():
            return ToolResult("answer needs evidence: one triple per condition in the sentence "
                              "({subject, relation, object, how}), or one line per condition with the step it came from.",
                              progress=False)
        gauss = np.unique(np.concatenate([self.g.members[i] for i in ids])).astype(np.int64)
        return ToolResult(f"answer recorded: {' '.join(f'#{i}' for i in ids)} ({k(len(gauss))} gaussians).",
                          summary="answer", final={"status": "answered", "ids": ids, "evidence": text, "triples": triples,
                                                   "dispute": parse_dispute(dispute), "gaussians": gauss.tolist()})

    def report(self, st, status, detail="", candidates=None, dispute=None):
        status = str(status).strip().lower()
        if status not in ("ambiguous", "not_found"):
            return ToolResult("report status must be 'ambiguous' or 'not_found'.", progress=False)
        cands = []
        if candidates:
            try:
                cands = [parse_instance(c) for c in (candidates if isinstance(candidates, list) else [candidates])]
            except ValueError as e:
                return ToolResult(str(e), progress=False, error=True)
        if status == "ambiguous" and not cands:
            return ToolResult("ambiguous report must list the tied candidates.", progress=False)
        cov = self.scene.coverage(sorted(st.seen))
        arrs = [self.g.members[i] for i in cands if i in self.g.members]
        gauss = np.unique(np.concatenate(arrs)).tolist() if arrs else []
        return ToolResult(f"report recorded: {status}.", summary=f"report {status}",
                          final={"status": status, "detail": str(detail), "candidates": cands, "triples": [],
                                 "dispute": parse_dispute(dispute), "coverage_seen": cov, "gaussians": gauss})

    # ─────────────────────────── 4.7 recall (v3 §7.3, 兜底) ───────────────────────────
    def recall(self, st, ids=None, word=None, frame=None, level=None):
        given = [x for x in (ids, word, frame) if x is not None]
        if len(given) != 1:
            return ToolResult("recall takes exactly one of ids=[...], word='...', frame='f12'.", progress=False, error=True)
        level = str(level).strip().lower() if level else None
        if level not in (None, "summary", "full", "image"):
            return ToolResult("level must be summary, full or image.", progress=False, error=True)
        if frame is not None:
            return self._recall_frame(st, parse_frame(frame))
        if word is not None:
            return self._recall_word(st, str(word), level or "summary")
        labels = ids if isinstance(ids, list) else [ids]
        return self._recall_ids(st, [str(x).strip().lower() for x in labels][:config.RECALL_MAX_THUMBS], level or "image")

    def _record_lines(self, label, full):
        g = self.g
        nid = g.node_of(label)
        if nid is None:
            return None, []
        n = g.nodes[nid]
        lines = [render.node_line(g, nid).strip()]
        lines.append(f"    names: {', '.join(n['names'])}; frames: {fmt_frames(sorted(n['frames']))}"
                     + (f"; parent {n['parent']}" if n.get("parent") else ""))
        for sname in n["s"]:
            sg = g.segs.get(sname)
            if sg is None:
                continue
            lines.append(f"    {sname}: '{sg['word']}' frames {fmt_frames(sorted(sg['frames']))}, score {sg['score']:.2f}"
                         + (f", box on f{sg['box_frame']} IoU {sg['box_iou']}" if sg.get("box_frame") is not None else "")
                         + (" BOXED" if sg.get("boxed") else "") + f" ({sg['q']})")
        if n["inst"] is not None:
            r = g.insts[n["inst"]]
            ext = r["obb"]["extent"]
            lines.append(f"    #{n['inst']}: {r['n_gaussians']} gaussians, {len(r['frames'])} views, {r['grade']}, "
                         f"size {ext[0]:.2f}x{ext[1]:.2f}x{ext[2]:.2f} m, exemplar f{r.get('exemplar_fid')}; "
                         + (g.claims_text(n["inst"], with_sentence=True, maxlen=80) or "no claims yet"))
        for e in g.edges_of(nid)[:6]:
            lines.append("  " + render.edge_line(g, e).strip())
        if full:
            for h in n["history"][-6:]:
                lines.append(f"    history {h.get('q')}: {h.get('what')}")
        return nid, lines

    def _thumb_for(self, nid):
        n = self.g.nodes[nid]
        if n["inst"] is not None:
            b = self.g.thumb_bytes(n["inst"])
            if b:
                return b
        for sname in n["s"]:
            b = self.g.cutout_bytes(sname)
            if b:
                return b
        return None

    def _recall_ids(self, st, labels, level):
        lines, thumbs, unknown = [], [], []
        for lab in labels:
            nid, rec = self._record_lines(lab, level == "full")
            if nid is None:
                unknown.append(lab)
                continue
            st.touched.add(lab)
            if lab.startswith("s"):
                self.g.confirm(lab)
            lines += rec
            if level == "image":
                b = self._thumb_for(nid)
                if b:
                    thumbs.append((b, self.g.label(nid)))
        if not lines:
            return ToolResult(f"unknown id(s) {' '.join(unknown)}; known: {st.known_ids()}", progress=False, error=True)
        text = "memory records:\n" + "\n".join(lines)
        if unknown:
            text += f"\n(unknown: {' '.join(unknown)})"
        imgs = []
        if thumbs:
            imgs.append((imaging.contact_sheet(thumbs), "memory thumbnails (as seen earlier): " + ", ".join(l for _, l in thumbs)))
        self.g.note_recall(st.step, "ids", labels, [l for _, l in thumbs])
        st.n_recall += 1
        return ToolResult(text, imgs, summary=f"recall {' '.join(labels)}", progress=True)

    def _recall_word(self, st, word, level):
        w = norm_word(word)
        hits = memory_match.match(self.g, [{"name": w, "desc": ""}], self.g.feature_of_node)
        if not hits:
            names = sorted(self.g.names_index())
            return ToolResult(f"nothing in memory matches '{w}' by name or appearance."
                              + (f" Names in memory: {', '.join(names[:40])}" if names else " Memory is empty."), progress=False)
        lines, thumbs = [], []
        for nid, score, phrase, how in hits[:12]:
            lines.append(render.node_line(self.g, nid).strip() + f"   [{how}{'' if how == 'name' else f' {score:.2f}'}]")
            for e in self.g.edges_of(nid)[:3]:
                lines.append("  " + render.edge_line(self.g, e).strip())
            if level == "image" and len(thumbs) < config.RECALL_MAX_THUMBS:
                b = self._thumb_for(nid)
                if b:
                    thumbs.append((b, self.g.label(nid)))
        keys = {nm for nid, _, _, _ in hits for nm in self.g.nodes[nid]["names"]}
        qs = [q for q in self.g.queries if any(nm in norm_word(q["text"]) for nm in keys) and q.get("status")]
        for q in qs[-3:]:
            lines.append(f"  {q['qid']} \"{q['text']}\" -> {q['status']} {' '.join(f'#{i}' for i in q['ids'])}")
        imgs = [(imaging.contact_sheet(thumbs), "memory thumbnails: " + ", ".join(l for _, l in thumbs))] if thumbs else []
        self.g.note_recall(st.step, "word", [w], [l for _, l in thumbs])
        st.n_recall += 1
        return ToolResult(f"memory for '{w}':\n" + "\n".join(lines), imgs, summary=f"recall word '{w}'", progress=True)

    def _recall_frame(self, st, fid):
        if not 0 <= fid < self.scene.n_frames:
            return ToolResult(f"frame f{fid} does not exist (scene has f0..f{self.scene.n_frames - 1}).", progress=False, error=True)
        jpeg, cap, labs, qs = render.frame_view(self.g, self.scene, st, fid)
        self.g.note_looked([fid], "recall", st.step, None)
        text = (f"f{fid} (whole frame, low-res)" + (f"; outlined from memory: {', '.join(labs)}" if labs else "; nothing from memory outlined")
                + (f"; looked at in {' '.join(qs[:4])}" if qs else "; never looked at before")
                + ". You can draw boxes on this image (whole-frame coordinates).")
        self.g.note_recall(st.step, "frame", [f"f{fid}"], labs)
        st.n_recall += 1
        return ToolResult(text, [(jpeg, cap)], summary=f"recall f{fid}", progress=True)

    # ─────────────────────────── 分发 ───────────────────────────
    def dispatch(self, name, args, st):
        fn = {"explore": self.explore, "nearby": self.nearby, "segment": self.segment, "same": self.same,
              "lift": self.lift, "relate": self.relate, "recall": self.recall, "answer": self.answer,
              "report": self.report}.get(name)
        if name == "segment" and config.ABLATION == "3":   # 消融 (3): 只给标签
            fn = self.segment_label_only
        if name == "relate" and config.ABLATION == "8":    # 消融 (8): 没有尺子; 明确的错误观测, 仍算一步
            return ToolResult("relate() is not available in this setting: judge the relation from the images (find a frame showing the candidates "
                              "together, or nearby(group=) on each), then answer() or report().", summary="relate rejected (off in this setting)",
                              progress=False, error=True)
        if fn is None:
            return ToolResult(f"unknown tool {name}", progress=False, error=True)
        # 模型常把没用的可选参数填成 "" / null / [] (首次 demo: group="" within="" 连错三步), 一律当没传
        args = {k_: v for k_, v in (args or {}).items() if v not in ("", None, [])}
        try:
            return fn(st, **args)
        except TypeError as e:
            msg = str(e)
            if "argument" in msg and ("unexpected" in msg or "missing" in msg or "positional" in msg):
                return ToolResult(f"bad arguments for {name}: {msg}", progress=False, error=True)
            return self._crash(name, e)
        except ValueError as e:
            return ToolResult(f"{name}: {e}", progress=False, error=True)
        except Exception as e:                                   # noqa: BLE001  工具内部崩了: 如实报, 不冒充参数错
            return self._crash(name, e)

    def _crash(self, name, e):
        import traceback
        tb = traceback.format_exc()
        self.last_traceback = tb
        print(f"[tool crash] {name}: {type(e).__name__}: {e}\n{tb}", flush=True)
        return ToolResult(f"tool {name} crashed internally ({type(e).__name__}: {e}). This is a bug on the tool side, "
                          f"not in your arguments; retrying the same call will fail again.",
                          progress=False, error=True)
