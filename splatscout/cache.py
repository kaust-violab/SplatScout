"""场景图记忆 (施工方案 v3 §2-3): 磁盘上一张场景图是唯一真相, 每个工具返回时由程序写, 模型从不改它.

节点 = 物体, 一种类型, 三级成熟度: 有 s (segment 切出) -> 有 i (same 合并) -> 有 # (lift 提升).
模型看到的编号仍是 s / i / #, 节点号 n 只在内部用. 每组带一个 CLIP 图像向量 (腐蚀掩码抠图), 查询按图文相似度找节点.
(2026-09-09 删掉了模型 seen 写的名字级记录: 记忆里不再有模型的自由文本; 旧日志里的 seen 事件回放时忽略.)
边 = 题目问出来的关系 (answer 的三元组建边, relate 补数字证据), 数字是证据, 只追加, 不删.
落盘: <dir>/events.jsonl 只追加 (唯一真相); 掩码 / 高斯 / 抠图 另存 npz / jpg; 启动时回放日志重建内存态.
"""
import json
import os
import re
import shutil
import threading
import time
from collections.abc import MutableMapping

import cv2
import numpy as np

from . import config
from . import lifting


def slug(word):
    return re.sub(r"[^a-z0-9]+", "_", word.lower()).strip("_") or "x"


def norm_word(word):
    w = str(word).strip().lower()
    w = re.sub(r"^(the|a|an)\s+", "", w)
    return re.sub(r"\s+", " ", w)


def norm_label(x):
    """外部编号规范化: '#12' / 's3' / 'i2' 原样 (小写); 编号后面带描述 ('#4 silver lid', 's3 (cup)') 只取编号; 其余当名字."""
    s = str(x).strip().lower()
    m = re.match(r"^(#\s*\d+|[si]\d+)(?![a-z0-9])", s)
    if m:
        return m.group(1).replace(" ", "")
    return norm_word(s.lstrip("#").strip())


class LazyMasks(MutableMapping):
    """fid -> bool 掩码 (HxW); 内存里按位压缩, 取的时候解包. 省 8 倍内存, 一场几十组掩码常驻不吃力."""

    def __init__(self, shape, packed=None):
        self.shape = tuple(int(v) for v in shape)
        self._p = dict(packed or {})

    def __getitem__(self, fid):
        H, W = self.shape
        return np.unpackbits(self._p[int(fid)])[:H * W].reshape(H, W).astype(bool)

    def __setitem__(self, fid, mask):
        self._p[int(fid)] = np.packbits(np.asarray(mask, bool).reshape(-1))

    def __delitem__(self, fid):
        del self._p[int(fid)]

    def __iter__(self):
        return iter(self._p)

    def __len__(self):
        return len(self._p)

    def __contains__(self, fid):
        if isinstance(fid, (bool, str)) or not isinstance(fid, (int, np.integer)):
            return False
        return int(fid) in self._p

    def packed(self):
        return self._p


class SceneGraph:
    def __init__(self, scene, mode="session", shape=None, tag=None, root=None):
        assert mode in config.MODES, mode
        self.scene, self.mode, self.shape = scene, mode, shape
        self.dir = root if root is not None else config.mem_dir(scene, tag)   # root="" -> 纯内存 (测试)
        self.lock = threading.RLock()            # 主 agent 的工具线程与记忆 agent 线程都写图: 写入单线程化
        self._init_empty()
        if self.dir:
            os.makedirs(self.dir, exist_ok=True)
            self._load()

    # ─────────────────────────── 内存态 ───────────────────────────
    def _init_empty(self):
        self.nodes, self.n_nodes = {}, 0
        self.segs, self.n_segs = {}, 0
        self.objs, self.n_objs = {}, 0
        self.insts, self.members, self.points, self.next_inst = {}, {}, {}, 1
        self.feats = {}                          # sname -> (D,) CLIP 图像向量 (磁盘 feats/<s>.npy)
        self.edges, self.n_edges = [], 0
        self.queries = []
        self.looked = {}                         # fid -> [{"q","step","tool","window"}]
        self.relates = []
        self.seq = 0
        self.cur_q = None
        self._inst_masks = {}

    def _p(self, *a):
        return os.path.join(self.dir, *a)

    def reset(self):
        """换一张空图 (cold 口径每题用). 只清本目录, 真值永远不在这里."""
        if self.dir:
            shutil.rmtree(self.dir, ignore_errors=True)
            os.makedirs(self.dir, exist_ok=True)
        self._init_empty()

    # ─────────────────────────── 事件 ───────────────────────────
    def _emit(self, type_, live=None, **fields):
        with self.lock:
            ev = {"seq": self.seq, "t": round(time.time(), 3), "q": self.cur_q, "type": type_, **fields}
            self.seq += 1
            if self.dir:
                with open(self._p("events.jsonl"), "a") as f:
                    f.write(json.dumps(ev, ensure_ascii=False) + "\n")
            self._apply(ev, live)
            return ev

    def _load(self):
        p = self._p("events.jsonl")
        if os.path.exists(self._p("membership.npz")):
            m = np.load(self._p("membership.npz"))
            self.members = {int(k[4:]): m[k].astype(np.int64) for k in m.files}
        if os.path.exists(self._p("instance_points.npz")):
            pz = np.load(self._p("instance_points.npz"))
            for k_ in pz.files:
                if k_.startswith("pts_"):
                    i = int(k_[4:])
                    self.points[i] = (pz[k_], pz[f"scale_{i}"])
        if not os.path.exists(p):
            return
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ev = json.loads(line)
                self.seq = max(self.seq, int(ev["seq"]) + 1)
                self._apply(ev, None)
        self.cur_q = None

    def _apply(self, ev, live):
        fn = getattr(self, "_on_" + ev["type"], None)
        if fn is None:
            return
        fn(ev, live)

    # ─────────────────────────── 题目 ───────────────────────────
    def begin_query(self, qid, order_index, text, role=None):
        self.cur_q = qid
        self._emit("query_begin", qid=qid, order_index=int(order_index), text=str(text), role=role)

    def _on_query_begin(self, ev, live):
        self.cur_q = ev["qid"]
        self.queries.append({"qid": ev["qid"], "order_index": ev["order_index"], "text": ev["text"], "role": ev.get("role"),
                             "status": None, "ids": [], "evidence": [], "evidence_text": "", "candidates": [],
                             "detail": "", "steps": None, "seconds": None, "in_tok": None, "out_tok": None,
                             "opened": {}, "n_recall": 0, "looked": [], "relates": 0, "dispute": []})

    def query(self, qid=None):
        qid = qid or self.cur_q
        for q in reversed(self.queries):
            if q["qid"] == qid:
                return q
        return None

    def note_opening(self, fids, hit_labels, exact_qid=None, things=None, how=None):
        self._emit("opening", fids=[int(f) for f in fids], hits=list(hit_labels), exact=exact_qid,
                   things=list(things or []), how=how)

    def _on_opening(self, ev, live):
        q = self.query(ev["q"])
        if q is not None:
            q["opened"] = {"fids": ev["fids"], "hits": ev["hits"], "exact": ev.get("exact"),
                           "things": ev.get("things", []), "how": ev.get("how")}

    def end_query(self, steps, seconds, in_tok, out_tok, status):
        self._emit("query_end", steps=int(steps), seconds=float(seconds), in_tok=int(in_tok or 0),
                   out_tok=int(out_tok or 0), status=status)
        self.cur_q = None

    def _on_query_end(self, ev, live):
        q = self.query(ev["q"])
        if q is not None:
            q.update({"steps": ev["steps"], "seconds": ev["seconds"], "in_tok": ev["in_tok"], "out_tok": ev["out_tok"]})
            if q["status"] is None:
                q["status"] = ev.get("status")

    def exact_match(self, text):
        """句子一字不差答过 -> 那条题目记录 (最近的一条); 没有返回 None."""
        key = norm_word(text)
        for q in reversed(self.queries):
            if q["status"] == "answered" and norm_word(q["text"]) == key and q["ids"]:
                return q
        return None

    # ─────────────────────────── 看过的帧 / seen ───────────────────────────
    def note_looked(self, fids, tool, step, window=None):
        fids = [int(f) for f in fids]
        if not fids:
            return
        self._emit("looked", fids=fids, tool=tool, step=int(step),
                   window=None if window is None else [int(v) for v in window])

    def _on_looked(self, ev, live):
        for f in ev["fids"]:
            self.looked.setdefault(int(f), []).append({"q": ev["q"], "step": ev["step"], "tool": ev["tool"],
                                                       "window": ev.get("window")})
        q = self.query(ev["q"])
        if q is not None:
            for f in ev["fids"]:
                if f not in q["looked"]:
                    q["looked"].append(int(f))

    def looked_frames(self):
        return set(self.looked)

    def _on_seen(self, ev, live):
        """旧日志里的 seen 事件: 忽略 (名字级记录已取消)."""
        return

    # ─────────────────────────── CLIP 特征 ───────────────────────────
    def put_feature(self, sname, vec):
        if sname not in self.segs or vec is None:
            return
        vec = np.asarray(vec, np.float32)
        if self.dir:
            os.makedirs(self._p("feats"), exist_ok=True)
            np.save(self._p("feats", f"{sname}.npy"), vec)
        self._emit("feat", live={"vec": vec}, sname=sname, dim=int(vec.shape[0]))

    def _on_feat(self, ev, live):
        if live is not None:
            self.feats[ev["sname"]] = live["vec"]
        elif self.dir and os.path.exists(self._p("feats", f"{ev['sname']}.npy")):
            self.feats[ev["sname"]] = np.load(self._p("feats", f"{ev['sname']}.npy")).astype(np.float32)

    def feature_of_node(self, nid):
        """节点向量 = 它各组向量的平均 (归一); 没有返回 None."""
        n = self.nodes.get(nid)
        if n is None:
            return None
        vs = [self.feats[s] for s in n["s"] if s in self.feats]
        if not vs:
            return None
        v = np.mean(vs, axis=0)
        return v / max(float(np.linalg.norm(v)), 1e-8)

    # ─────────────────────────── 节点 ───────────────────────────
    def _new_node(self, names, q, frames=()):
        self.n_nodes += 1
        nid = self.n_nodes
        self.nodes[nid] = {"id": nid, "names": list(dict.fromkeys(norm_word(n) for n in names if n)),
                           "frames": set(int(f) for f in frames), "s": [], "i": None, "inst": None, "parent": None,
                           "history": [], "q": q, "merged_into": None}
        return nid

    def _merge_nodes(self, keep, other, q, why):
        if keep == other or other not in self.nodes or keep not in self.nodes:
            return keep
        a, b = self.nodes[keep], self.nodes[other]
        for nm in b["names"]:
            if nm not in a["names"]:
                a["names"].append(nm)
        a["frames"] |= b["frames"]
        for s in b["s"]:
            if s not in a["s"]:
                a["s"].append(s)
                if s in self.segs:
                    self.segs[s]["node"] = keep
        if a["i"] is None:
            a["i"] = b["i"]
        if b["i"] is not None and b["i"] in self.objs:
            self.objs[b["i"]]["node"] = keep
        if a["inst"] is None:
            a["inst"] = b["inst"]
        if b["inst"] is not None and b["inst"] in self.insts:
            self.insts[b["inst"]]["node"] = keep
        if a["parent"] is None:
            a["parent"] = b["parent"]
        a["history"].append({"q": q, "what": f"merged n{other} ({why})"})
        b["merged_into"] = keep
        for e in self.edges:
            for side in ("a_node", "b_node"):
                if e.get(side) == other:
                    e[side] = keep
        return keep

    def active_nodes(self):
        return [n for n in self.nodes.values() if n["merged_into"] is None]

    def node_of(self, label):
        """'#12' / 'i2' / 's3' -> 节点号; 名字 / 未知 -> None."""
        s = norm_label(label)
        if not re.fullmatch(r"#\d+|[si]\d+", s):
            return None
        if s.startswith("#"):
            rec = self.insts.get(int(s[1:]))
            nid = rec["node"] if rec else None
        elif s.startswith("i"):
            o = self.objs.get(s)
            nid = o["node"] if o else None
        else:
            g = self.segs.get(s)
            nid = g["node"] if g else None
        while nid is not None and self.nodes[nid]["merged_into"] is not None:
            nid = self.nodes[nid]["merged_into"]
        return nid

    def label(self, nid):
        """节点的对外编号: # > i > s."""
        n = self.nodes[nid]
        if n["inst"] is not None:
            return f"#{n['inst']}"
        if n["i"] is not None:
            return n["i"]
        return n["s"][0] if n["s"] else f"n{nid}"

    def maturity(self, nid):
        n = self.nodes[nid]
        if n["inst"] is not None:
            rec = self.insts[n["inst"]]
            tag = f"lifted {rec.get('grade', 'rough')} {len(rec.get('frames', []))}v"
            if rec.get("single_view"):
                tag += " SINGLE-VIEW"
            if rec.get("by") == "memory":
                tag += " (memory agent)"
            return tag
        if n["s"]:
            fr = sorted(n["frames"])
            return f"segmented {' '.join(n['s'][:3])}" + (f" f{fr[0]}" if fr else "")
        return "seen only"

    def word_of(self, nid):
        n = self.nodes[nid]
        return n["names"][0] if n["names"] else "?"

    # ─────────────────────────── s 组 ───────────────────────────
    def add_seg(self, g, step, box=None, box_frame=None, box_iou=None, boxed=False):
        """segment 切出的一组 -> s 号 + 自己的节点. g: word / frames{fid: mask} / rough_members / score / parent_id / within."""
        with self.lock:
            return self._add_seg(g, step, box, box_frame, box_iou, boxed)

    def _add_seg(self, g, step, box, box_frame, box_iou, boxed):
        self.n_segs += 1
        sname = f"s{self.n_segs}"
        fids = sorted(int(f) for f in g["frames"])
        H, W = self.shape or next(iter(g["frames"].values())).shape
        masks = LazyMasks((H, W))
        for f in fids:
            masks[f] = g["frames"][f]
        if self.dir:
            os.makedirs(self._p("masks"), exist_ok=True)
            np.savez_compressed(self._p("masks", f"{sname}.npz"), shape=np.array([H, W], np.int64),
                                fids=np.array(fids, np.int64), **{f"m_{f}": masks.packed()[f] for f in fids})
            np.save(self._p("masks", f"{sname}_rough.npy"), np.asarray(g["rough_members"], np.int64))
        live = {"masks": masks, "rough": np.asarray(g["rough_members"], np.int64)}
        parent_label = g.get("within")
        self._emit("seg_new", live=live, sname=sname, step=int(step), word=norm_word(g["word"]), fids=fids,
                   score=float(g.get("score", 0.0)), parent=parent_label, parent_inst=g.get("parent_id"),
                   rough_n=int(len(g["rough_members"])), box=None if box is None else [int(v) for v in box],
                   box_frame=box_frame, box_iou=None if box_iou is None else round(float(box_iou), 3), boxed=bool(boxed))
        return sname

    def confirm(self, sname, name=None, by=None):
        """模型用过这一组 (框中的、nearby / lift / same / within / recall 点过名的), 或记忆 agent 看图确认过 -> 它才算图里的东西.
        分割器顺带带回的相似块只存不显, 免得一次 segment 带出十几个 cabinet 把视图淹了. name: 记忆 agent 给它的名字."""
        g = self.segs.get(sname)
        if g is None or (g.get("confirmed") and not name):
            return
        self._emit("seg_confirm", sname=sname, name=norm_word(name) if name else None, by=by)

    def _on_seg_confirm(self, ev, live):
        g = self.segs.get(ev["sname"])
        if g is None:
            return
        g["confirmed"] = True
        g["discarded"] = False
        if ev.get("name"):
            n = self.nodes.get(g["node"])
            if n is not None and ev["name"] not in n["names"]:
                n["names"].insert(0, ev["name"])

    def discard(self, sname, reason="", by=None):
        """记忆 agent 看图判定: 碎块 / 背景 / 误检. 只存不显, 不删."""
        g = self.segs.get(sname)
        if g is None or g.get("instance") is not None:
            return
        self._emit("seg_discard", sname=sname, reason=str(reason)[:120], by=by)

    def _on_seg_discard(self, ev, live):
        g = self.segs.get(ev["sname"])
        if g is not None:
            g["discarded"] = True
            g["confirmed"] = False

    def alias(self, label, name, by=None):
        name = norm_word(name)
        nid = self.node_of(label)
        if nid is None or not name or name in self.nodes[nid]["names"]:
            return
        self._emit("alias", label=norm_label(label), name=name, by=by)

    def _on_alias(self, ev, live):
        nid = self.node_of(ev["label"])
        if nid is not None and ev["name"] not in self.nodes[nid]["names"]:
            self.nodes[nid]["names"].append(ev["name"])

    def merge(self, a, b, why="", by=None):
        """记忆 agent 判定两个节点是同一个东西 (带图为证): 节点合并, 只追加."""
        na, nb = self.node_of(a), self.node_of(b)
        if na is None or nb is None or na == nb:
            return None
        self._emit("node_merge", a=norm_label(a), b=norm_label(b), why=str(why)[:120], by=by)
        return self.node_of(a)

    def _on_node_merge(self, ev, live):
        na, nb = self.node_of(ev["a"]), self.node_of(ev["b"])
        if na is None or nb is None or na == nb:
            return
        keep, other = (na, nb) if (self.nodes[na]["inst"] is not None or self.nodes[nb]["inst"] is None) else (nb, na)
        self._merge_nodes(keep, other, ev["q"], f"memory agent: {ev.get('why', '')}")
        for s in self.nodes[keep]["s"]:
            if s in self.segs:
                self.segs[s]["confirmed"] = self.segs[s].get("confirmed") or not self.segs[s].get("discarded")

    def exclude(self, ids, why="", by=None):
        """本题考虑过又排除的候选 (记忆 agent 或主 agent 给的), 记进题目记录与实例的断言."""
        labs = [norm_label(i) for i in ids if self.node_of(i) is not None]
        if labs:
            self._emit("excluded", ids=labs, why=str(why)[:120], by=by)

    def _on_excluded(self, ev, live):
        q = self.query(ev["q"])
        if q is not None:
            q.setdefault("excluded", []).extend(ev["ids"])
        for lab in ev["ids"]:
            nid = self.node_of(lab)
            if nid is not None and self.nodes[nid]["inst"] is not None:
                self.insts[self.nodes[nid]["inst"]].setdefault("claims", []).append({"q": ev["q"], "kind": "excluded"})

    def node_visible(self, nid):
        """进视图 (交接页 / 开题帧 / 名册) 的节点: 提升过的, 或有确认过 (且没被丢弃) 的段."""
        n = self.nodes.get(nid)
        if n is None or n["merged_into"] is not None:
            return False
        return n["inst"] is not None or any(self.segs.get(s, {}).get("confirmed") and not self.segs.get(s, {}).get("discarded")
                                            for s in n["s"])

    def unconfirmed_segs(self, word=None):
        """没确认也没丢弃的相似块 (记忆 agent 的待办)."""
        return [s for s, g in self.segs.items() if not g.get("confirmed") and not g.get("discarded")
                and g.get("instance") is None and (word is None or g["word"] == word)]

    def visible_nodes(self):
        return [n for n in self.active_nodes() if self.node_visible(n["id"])]

    def _on_seg_new(self, ev, live):
        sname = ev["sname"]
        self.n_segs = max(self.n_segs, int(sname[1:]))
        if live is not None:
            masks, rough = live["masks"], live["rough"]
        else:
            masks, rough = self._load_masks(sname, ev["fids"])
        nid = self._new_node([ev["word"]], ev["q"], ev["fids"])
        self.nodes[nid]["s"].append(sname)
        self.nodes[nid]["history"].append({"q": ev["q"], "what": f"segmented as {sname}"})
        parent = ev.get("parent")
        if parent:
            self.nodes[nid]["parent"] = parent
        self.segs[sname] = {"word": ev["word"], "frames": masks, "rough_members": rough, "score": ev["score"],
                            "parent_id": ev.get("parent_inst"), "within": parent, "object": None, "instance": None,
                            "node": nid, "q": ev["q"], "step": ev["step"], "box": ev.get("box"),
                            "box_frame": ev.get("box_frame"), "box_iou": ev.get("box_iou"), "boxed": ev.get("boxed", False),
                            "confirmed": bool(ev.get("boxed", False)), "cutouts": {}}

    def _load_masks(self, sname, fids):
        H, W = self.shape
        packed = {}
        p = self._p("masks", f"{sname}.npz") if self.dir else None
        if p and os.path.exists(p):
            z = np.load(p)
            H, W = z["shape"].tolist()
            for f in z["fids"].tolist():
                packed[int(f)] = z[f"m_{f}"]
        rp = self._p("masks", f"{sname}_rough.npy") if self.dir else None
        rough = np.load(rp).astype(np.int64) if rp and os.path.exists(rp) else np.zeros(0, np.int64)
        return LazyMasks((H, W), packed), rough

    def put_cutout(self, sname, fid, jpeg):
        """模型看到的白底抠图, 原样存; recall / 开题原样给回去."""
        if sname not in self.segs:
            return
        if self.dir:
            os.makedirs(self._p("cutouts"), exist_ok=True)
            with open(self._p("cutouts", f"{sname}_f{int(fid)}.jpg"), "wb") as f:
                f.write(jpeg)
        self.segs[sname]["cutouts"][int(fid)] = jpeg if not self.dir else None

    def cutout_bytes(self, sname, fid=None):
        g = self.segs.get(sname)
        if g is None:
            return None
        fids = list(g["cutouts"]) if g["cutouts"] else []
        if not fids and self.dir and os.path.isdir(self._p("cutouts")):
            fids = [int(m.group(1)) for fn in os.listdir(self._p("cutouts"))
                    for m in [re.fullmatch(rf"{sname}_f(\d+)\.jpg", fn)] if m]
        if not fids:
            return None
        f = int(fid) if fid is not None and int(fid) in fids else fids[0]
        if g["cutouts"].get(f) is not None:
            return g["cutouts"][f]
        p = self._p("cutouts", f"{sname}_f{f}.jpg")
        return open(p, "rb").read() if os.path.exists(p) else None

    # ─────────────────────────── i 物体 ───────────────────────────
    def new_object(self, gnames, step):
        self.n_objs += 1
        iname = f"i{self.n_objs}"
        self._emit("obj_new", iname=iname, step=int(step))
        self.put_in_object(iname, gnames, step)
        return iname

    def _on_obj_new(self, ev, live):
        iname = ev["iname"]
        self.n_objs = max(self.n_objs, int(iname[1:]))
        self.objs[iname] = {"groups": [], "instance": None, "node": None, "q": ev["q"]}

    def put_in_object(self, oname, gnames, step):
        """几组放进物体 oname; 一组只属于一个物体. 返回搬动记录 [(组, 原物体)]. 节点随之合并."""
        moved = []
        todo = []
        for gn in gnames:
            old = self.segs[gn].get("object")
            if old == oname:
                continue
            if old is not None:
                moved.append((gn, old))
            todo.append(gn)
        if todo:
            self._emit("obj_put", iname=oname, groups=todo, step=int(step))
        return moved

    def _on_obj_put(self, ev, live):
        o = self.objs[ev["iname"]]
        for gn in ev["groups"]:
            g = self.segs[gn]
            g["confirmed"] = True
            old = g.get("object")
            if old is not None and old in self.objs and gn in self.objs[old]["groups"]:
                self.objs[old]["groups"].remove(gn)
            g["object"] = ev["iname"]
            o["groups"].append(gn)
            gnode = self.node_of(gn)
            if o["node"] is None:
                o["node"] = gnode
                self.nodes[gnode]["i"] = ev["iname"]
            elif gnode is not None and gnode != o["node"]:
                self._merge_nodes(o["node"], gnode, ev["q"], f"same -> {ev['iname']}")

    # ─────────────────────────── # 实例 ───────────────────────────
    def find_overlap(self, members):
        best, best_v = None, config.DEDUP_MUTUAL_FIT
        for i, m in self.members.items():
            v = lifting.mutual_fit(members, m)
            if v >= best_v:
                best, best_v = i, v
        return best, best_v

    def add_instance(self, word, members, geo, frames, grade, sources, step=0, source=None, groups=(),
                     single_view=False, exemplar_fid=None, by=None):
        """入库 + 去重 (互吻合 >= 0.7 -> alias / upgraded). 返回 (id, how). by='memory' = 记忆 agent 的粗提."""
        with self.lock:
            return self._add_instance(word, members, geo, frames, grade, sources, step, source, groups, single_view,
                                      exemplar_fid, by)

    def _add_instance(self, word, members, geo, frames, grade, sources, step, source, groups, single_view, exemplar_fid, by):
        w = norm_word(word)
        members = np.asarray(members, np.int64)
        hit, _ = self.find_overlap(members)
        src_node = self.node_of(source) if source else None
        rec_fields = {"word": w, "n_gaussians": int(len(members)), "centroid": [float(v) for v in geo["centroid"]],
                      "obb": geo["obb"], "grade": grade, "frames": sorted(int(f) for f in frames),
                      "sources": {str(k): v for k, v in sources.items()}, "single_view": bool(single_view),
                      "exemplar_fid": None if exemplar_fid is None else int(exemplar_fid), "source": source,
                      "groups": list(groups), "step": int(step), "by": by}
        if hit is not None:
            rec = self.insts[hit]
            if grade == "thorough" and (rec.get("grade") != "thorough" or len(members) > len(self.members[hit])):
                self.members[hit] = members
                self.points[hit] = (geo["pts"], geo["pscale"])
                self._save_arrays()
                self._emit("inst_upgrade", iid=hit, src_node=src_node, **rec_fields)
                return hit, "upgraded"
            self._emit("inst_alias", iid=hit, src_node=src_node, **rec_fields)
            return hit, "alias"
        iid = self.next_inst
        self.members[iid] = members
        self.points[iid] = (geo["pts"], geo["pscale"])
        self._save_arrays()
        self._emit("inst_new", iid=iid, src_node=src_node, **rec_fields)
        return iid, "new"

    def _save_arrays(self):
        if not self.dir:
            return
        np.savez(self._p("membership.npz"), **{f"idx_{i}": v for i, v in self.members.items()})
        arr = {}
        for i, (pts, sc) in self.points.items():
            arr[f"pts_{i}"], arr[f"scale_{i}"] = pts, sc
        if arr:
            np.savez(self._p("instance_points.npz"), **arr)

    def _inst_node(self, ev):
        nid = ev.get("src_node")
        if nid is not None and nid in self.nodes:
            while self.nodes[nid]["merged_into"] is not None:
                nid = self.nodes[nid]["merged_into"]
            return nid
        return self._new_node([ev["word"]], ev["q"], ev["frames"])

    def _on_inst_new(self, ev, live):
        iid = ev["iid"]
        self.next_inst = max(self.next_inst, iid + 1)
        nid = self._inst_node(ev)
        rec = {k: ev[k] for k in ("word", "n_gaussians", "centroid", "obb", "grade", "frames", "sources", "single_view",
                                  "exemplar_fid", "source", "groups")}
        rec.update({"id": iid, "names": [ev["word"]], "status": "healthy", "node": nid, "q": ev["q"], "by": ev.get("by"),
                    "history": [{"q": ev["q"], "what": f"lifted {ev['grade']}" + (f" by {ev['by']}" if ev.get("by") else ""),
                                 "step": ev.get("step")}]})
        self.insts[iid] = rec
        n = self.nodes[nid]
        n["inst"] = iid
        n["frames"] |= set(ev["frames"])
        if ev["word"] not in n["names"]:
            n["names"].append(ev["word"])
        n["history"].append({"q": ev["q"], "what": f"lifted -> #{iid} ({ev['grade']})"})
        for gn in ev.get("groups") or []:
            if gn in self.segs:
                self.segs[gn]["instance"] = iid
        src = ev.get("source")
        if src and src in self.objs:
            self.objs[src]["instance"] = iid
        if iid not in self.members and live is None:
            pass                                          # 数组在 _load 时已从 npz 读入

    def _on_inst_upgrade(self, ev, live):
        rec = self.insts[ev["iid"]]
        rec.update({k: ev[k] for k in ("n_gaussians", "centroid", "obb", "grade", "frames", "sources", "single_view",
                                       "exemplar_fid")})
        if ev["word"] not in rec["names"]:
            rec["names"].append(ev["word"])
        rec["history"].append({"q": ev["q"], "what": f"upgraded to {ev['grade']} ({ev['n_gaussians']} gaussians)",
                               "step": ev.get("step")})
        self._absorb_source(ev, rec)

    def _on_inst_alias(self, ev, live):
        rec = self.insts[ev["iid"]]
        if ev["word"] not in rec["names"]:
            rec["names"].append(ev["word"])
        rec["frames"] = sorted(set(rec.get("frames", [])) | set(ev["frames"]))
        rec.setdefault("sources", {}).update(ev["sources"])
        rec["history"].append({"q": ev["q"], "what": f"seen again as '{ev['word']}' (alias)", "step": ev.get("step")})
        self._absorb_source(ev, rec)

    def _absorb_source(self, ev, rec):
        """升级 / 别名: 来源组的节点并进实例的节点."""
        nid = rec["node"]
        src = ev.get("src_node")
        if src is not None and src in self.nodes:
            while self.nodes[src]["merged_into"] is not None:
                src = self.nodes[src]["merged_into"]
            if src != nid:
                self._merge_nodes(nid, src, ev["q"], f"lift dedup -> #{rec['id']}")
        n = self.nodes[nid]
        n["frames"] |= set(ev["frames"])
        if ev["word"] not in n["names"]:
            n["names"].append(ev["word"])
        for gn in ev.get("groups") or []:
            if gn in self.segs:
                self.segs[gn]["instance"] = rec["id"]
        if ev.get("source") in self.objs:
            self.objs[ev["source"]]["instance"] = rec["id"]

    def put_instance_masks(self, iid, frames):
        arr = {"fids": np.array(sorted(int(f) for f in frames), np.int64), "shape": np.array(self.shape, np.int64)}
        for f, m in frames.items():
            arr[f"mask_{int(f)}"] = np.packbits(np.asarray(m, bool).reshape(-1))
        self._inst_masks[iid] = arr
        if self.dir:
            os.makedirs(self._p("inst_masks"), exist_ok=True)
            np.savez_compressed(self._p("inst_masks", f"{iid}.npz"), **arr)

    def get_instance_mask(self, iid, fid):
        z = self._inst_masks.get(iid)
        if z is None:
            p = self._p("inst_masks", f"{iid}.npz") if self.dir else None
            if not p or not os.path.exists(p):
                return None
            z = dict(np.load(p))
            self._inst_masks[iid] = z
        if f"mask_{int(fid)}" not in z:
            return None
        H, W = [int(v) for v in z["shape"]]
        return np.unpackbits(z[f"mask_{int(fid)}"])[:H * W].reshape(H, W).astype(bool)

    def instance_frames(self, iid):
        z = self._inst_masks.get(iid)
        if z is None and self.dir and os.path.exists(self._p("inst_masks", f"{iid}.npz")):
            z = dict(np.load(self._p("inst_masks", f"{iid}.npz")))
            self._inst_masks[iid] = z
        return [int(f) for f in z["fids"]] if z is not None else list(self.insts.get(iid, {}).get("frames", []))

    def put_thumb(self, iid, jpeg):
        if self.dir:
            os.makedirs(self._p("thumbs"), exist_ok=True)
            with open(self._p("thumbs", f"{iid}.jpg"), "wb") as f:
                f.write(jpeg)
        else:
            self.insts[iid]["_thumb"] = jpeg

    def thumb_bytes(self, iid):
        rec = self.insts.get(iid)
        if rec is None:
            return None
        if rec.get("_thumb") is not None:
            return rec["_thumb"]
        p = self._p("thumbs", f"{iid}.jpg") if self.dir else None
        return open(p, "rb").read() if p and os.path.exists(p) else None

    def get(self, iid):
        return self.insts.get(iid)

    def ids(self):
        return sorted(self.insts)

    def objs_for_relation(self):
        objs = []
        for iid in sorted(self.insts):
            r = self.insts[iid]
            obb = r["obb"]
            h = np.array(obb["extent"], np.float64) / 2
            pts = self.points.get(iid)
            objs.append({"id": iid, "name": r["names"][0], "names": list(r["names"]), "status": r.get("status", "healthy"),
                         "centroid": np.array(r["centroid"], np.float64), "c": np.array(obb["center"], np.float64),
                         "R": np.array(obb["axes"], np.float64), "h": h,
                         "s": float(np.cbrt(np.prod(np.maximum(h * 2, 1e-6)))),
                         "pts": pts[0].astype(np.float64) if pts else None,
                         "pscale": pts[1].astype(np.float64) if pts else None})
        return objs

    # ─────────────────────────── 边 ───────────────────────────
    def find_edge(self, a, b, relation=None):
        a, b = norm_label(a), norm_label(b)
        na, nb = self.node_of(a), self.node_of(b)
        for e in self.edges:
            same_ab = ((e["a_node"] is not None and e["a_node"] == na) or e["a"] == a) and \
                      ((e["b_node"] is not None and e["b_node"] == nb) or e["b"] == b)
            if same_ab and (relation is None or e["relation"] == relation):
                return e
        return None

    def add_edge(self, a, b, relation, role, evidence, step):
        """a relation b (来自第 q 题); 已有同一条就只补证据."""
        relation = norm_word(relation)
        e = self.find_edge(a, b, relation)
        if e is not None:
            return self.add_edge_evidence(e["id"], evidence, step, role)
        self.n_edges += 1
        eid = self.n_edges
        self._emit("edge_new", eid=eid, a=norm_label(a), b=norm_label(b), relation=relation, role=role,
                   evidence=[dict(evidence, step=int(step))] if evidence else [], step=int(step))
        return eid

    def _on_edge_new(self, ev, live):
        self.n_edges = max(self.n_edges, ev["eid"])
        self.edges.append({"id": ev["eid"], "a": ev["a"], "b": ev["b"], "a_node": self.node_of(ev["a"]),
                           "b_node": self.node_of(ev["b"]), "relation": ev["relation"], "q": ev["q"], "role": ev["role"],
                           "evidence": list(ev["evidence"]), "disputes": []})

    def add_edge_evidence(self, eid, evidence, step, role=None):
        self._emit("edge_evidence", eid=int(eid), evidence=dict(evidence, step=int(step)), role=role)
        return eid

    def _on_edge_evidence(self, ev, live):
        for e in self.edges:
            if e["id"] == ev["eid"]:
                e["evidence"].append(dict(ev["evidence"], q=ev["q"]))
                if ev.get("role") == "answer" and e["role"] != "answer":
                    e["role"] = "answer"

    def dispute(self, a, b, relation, why, step):
        e = self.find_edge(a, b, norm_word(relation) if relation else None)
        if e is None:
            return None
        self._emit("edge_dispute", eid=e["id"], why=str(why)[:200], step=int(step))
        return e["id"]

    def _on_edge_dispute(self, ev, live):
        for e in self.edges:
            if e["id"] == ev["eid"]:
                e["disputes"].append({"q": ev["q"], "why": ev["why"], "step": ev["step"]})

    def edges_of(self, nid_or_label):
        nid = nid_or_label if isinstance(nid_or_label, int) else self.node_of(nid_or_label)
        lab = None if isinstance(nid_or_label, int) else norm_label(nid_or_label)
        out = []
        for e in self.edges:
            if (nid is not None and nid in (e["a_node"], e["b_node"])) or (lab is not None and lab in (e["a"], e["b"])):
                out.append(e)
        return out

    def edge_label(self, e, side):
        nid = e[f"{side}_node"]
        return self.label(nid) if nid is not None and nid in self.nodes else e[side]

    # ─────────────────────────── relate / answer / report / recall ───────────────────────────
    def note_relate(self, step, pred, anchors, rows, missing):
        """rows: [(iid, conf, dist, anchor)] 按分排序. 锚点与赢家之间的边补数字证据; 整张表存进题目记录."""
        self._emit("relate", step=int(step), pred=pred, anchors=[int(a) for a in anchors],
                   rows=[[int(i), round(float(c), 3), round(float(d), 3), None if a is None else str(a)] for i, c, d, a in rows],
                   missing=[int(m) for m in missing])
        if rows and anchors:
            win = rows[0]
            rivals = {f"#{i}": round(float(d), 3) for i, _, d, _ in rows[1:4]}
            for a in anchors:
                self.add_edge(f"#{win[0]}", f"#{a}", pred, "measured",
                              {"kind": "relate", "pred": pred, "value": round(float(win[2]), 3),
                               "conf": round(float(win[1]), 3), "rivals": rivals}, step)

    def _on_relate(self, ev, live):
        self.relates.append({k: ev[k] for k in ("q", "step", "pred", "anchors", "rows", "missing")})
        q = self.query(ev["q"])
        if q is not None:
            q["relates"] += 1

    def finish_query(self, step, final, triples, seen_text="", dispute=None):
        """answer / report 收尾: 题目记录 + 按三元组建边 + dispute 追加否定证据."""
        status = final.get("status")
        ids = [int(i) for i in final.get("ids", [])]
        for t in triples or []:
            subj, rel, obj, how = t.get("subject"), t.get("relation"), t.get("object"), t.get("how", "")
            if not (subj and rel and obj):
                continue
            role = "answer" if status == "answered" else "tie"
            self.add_edge(subj, obj, rel, role, {"kind": "image" if "relate" not in str(how).lower() else "relate-ref",
                                                 "how": str(how)[:120]}, step)
        for d in dispute or []:
            try:
                a, rel, b = d.get("subject"), d.get("relation"), d.get("object")
                if a and b:
                    self.dispute(a, b, rel, d.get("why", ""), step)
            except (AttributeError, TypeError):
                continue
        self._emit("finish", step=int(step), status=status, ids=ids, triples=list(triples or []),
                   evidence_text=str(final.get("evidence", ""))[:2000], candidates=[int(c) for c in final.get("candidates", [])],
                   detail=str(final.get("detail", ""))[:500], seen_text=str(seen_text)[:200])

    def _on_finish(self, ev, live):
        q = self.query(ev["q"])
        if q is None:
            return
        q.update({"status": ev["status"], "ids": ev["ids"], "evidence": ev["triples"], "evidence_text": ev["evidence_text"],
                  "candidates": ev["candidates"], "detail": ev["detail"], "final_seen": ev.get("seen_text", "")})
        for iid in ev["ids"]:
            if iid in self.insts:
                self.insts[iid].setdefault("claims", []).append({"q": ev["q"], "kind": "answer"})
        for iid in ev["candidates"]:
            if iid in self.insts:
                self.insts[iid].setdefault("claims", []).append({"q": ev["q"], "kind": "candidate"})

    def note_recall(self, step, kind, keys, shown):
        self._emit("recall", step=int(step), kind=kind, keys=[str(k) for k in keys], shown=[str(s) for s in shown])

    def _on_recall(self, ev, live):
        q = self.query(ev["q"])
        if q is not None:
            q["n_recall"] += 1

    def note_anchor_use(self, iids, step):
        """relate 用过的锚点 (给汇总的 mem_anchor 用)."""
        self._emit("anchor", step=int(step), ids=[int(i) for i in iids])

    def _on_anchor(self, ev, live):
        for iid in ev["ids"]:
            if iid in self.insts:
                self.insts[iid].setdefault("claims", []).append({"q": ev["q"], "kind": "anchor"})

    # ─────────────────────────── 查询辅助 ───────────────────────────
    def claims_text(self, iid, with_sentence=False, maxlen=48):
        """'answer q3, anchor q7'; with_sentence 时答案断言带那题的句子, 模型才能判断是不是同一个东西."""
        out = []
        for c in self.insts.get(iid, {}).get("claims", []):
            item = f"{c['kind']} {c['q']}"
            if with_sentence and c["kind"] == "answer":
                q = self.query(c["q"])
                if q is not None:
                    t = q["text"]
                    item += f' "{t[:maxlen]}{"..." if len(t) > maxlen else ""}"'
            out.append(item)
        return ", ".join(dict.fromkeys(out))

    def names_index(self):
        """名字 -> 该名字的节点列表 (进视图的节点)."""
        idx = {}
        for n in self.visible_nodes():
            for nm in n["names"]:
                idx.setdefault(nm, {"nodes": []})["nodes"].append(n["id"])
        return idx

    def frames_of_node(self, nid):
        return sorted(self.nodes[nid]["frames"])

    def stats(self):
        act = self.visible_nodes()
        return {"nodes": len(act), "lifted": sum(1 for n in act if n["inst"] is not None),
                "segmented": sum(1 for n in act if n["s"] and n["inst"] is None),
                "hidden": len(self.active_nodes()) - len(act), "edges": len(self.edges), "queries": len(self.queries),
                "looked": len(self.looked), "feats": len(self.feats)}

    def prewarm_from_lift(self, n_frames):
        """把旧管线全场注册库当 grade=thorough 灌进来 (对照口径), 每条一个节点."""
        from . import legacy
        sp = legacy.scene_paths()
        d = sp.lift_dir(self.scene)
        with open(os.path.join(d, "instances.json")) as f:
            insts = json.load(f)["instances"]
        mem = np.load(os.path.join(d, "membership.npz"))
        pts = np.load(os.path.join(d, "instance_points.npz")) if os.path.exists(os.path.join(d, "instance_points.npz")) else None
        for inst in insts:
            key = f"idx_{inst['id']}"
            if key not in mem.files:
                continue
            members = mem[key].astype(np.int64)
            geo = {"centroid": inst["centroid"], "obb": inst["obb"],
                   "pts": pts[f"pts_{inst['id']}"] if pts is not None and f"pts_{inst['id']}" in pts.files else np.zeros((1, 3), np.float32),
                   "pscale": pts[f"scale_{inst['id']}"] if pts is not None and f"scale_{inst['id']}" in pts.files else np.ones(1, np.float32)}
            iid, how = self.add_instance(inst["names"][0], members, geo, [], "thorough", {}, step=0)
            for nm in inst["names"][1:]:
                if iid in self.insts and norm_word(nm) not in self.insts[iid]["names"]:
                    self.insts[iid]["names"].append(norm_word(nm))


SceneMemory = SceneGraph      # 旧名
