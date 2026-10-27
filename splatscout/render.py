"""场景图 -> 给模型看的东西 (v3, 2026-09-09 改): 开题消息里放"和这句话有关的节点" (小模型改写句子 -> CLIP 找节点),
命中的节点所在的帧描边给图; 没命中给房间总览 (固定关键帧拼图). 每步末尾只剩本题的账 (几行). 不再有全图的简化视图."""
import numpy as np

from . import config
from . import imaging
from . import memory_match


# ─────────────────────────── 可见性 (CPU) ───────────────────────────
def node_visible_in(graph, scene, nid, fid):
    n = graph.nodes[nid]
    if fid in n["frames"]:
        return True
    if n["inst"] is not None and n["inst"] in graph.members and scene is not None:
        return float(scene.visible_fraction(graph.members[n["inst"]])[fid]) >= config.SNAPSHOT_MIN_VISIBLE
    if n["s"] and scene is not None:
        rough = graph.segs[n["s"][0]]["rough_members"]
        if len(rough):
            return float(scene.visible_fraction(rough)[fid]) >= config.SNAPSHOT_MIN_VISIBLE
    return False


def _node_mask(graph, scene, nid, fid):
    """节点在该帧的掩码: 段自己的掩码 > 实例成员投影 > 粗成员投影. 没有返回 None."""
    n = graph.nodes[nid]
    for s in n["s"]:
        g = graph.segs.get(s)
        if g is not None and fid in g["frames"]:
            return g["frames"][fid], "mask"
    if scene is None or getattr(scene, "splats", None) is None:
        return None, None
    if n["inst"] is not None and n["inst"] in graph.members:
        return scene.render_alpha(graph.members[n["inst"]], fid) > 0.5, "lifted"
    if n["s"]:
        rough = graph.segs[n["s"][0]]["rough_members"]
        if len(rough):
            return scene.render_alpha(rough, fid) > 0.5, "approx"
    return None, None


def outline_frame(graph, scene, fid, node_ids, long_side=config.THUMB_LONG):
    """整帧低清, 各节点描边 + 标编号. 返回 (jpeg, [(label, how)])."""
    img = scene.image(fid).copy()
    drawn = []
    for i, nid in enumerate(node_ids):
        mask, how = _node_mask(graph, scene, nid, fid)
        if mask is None or not mask.any():
            continue
        color = imaging.PALETTE[i % len(imaging.PALETTE)]
        img = imaging.contour_only(img, mask, color=color, lw=3, tint=0.15)
        ys, xs = np.nonzero(mask)
        lab = graph.label(nid) + ("~" if how == "approx" else "")
        imaging.stamp(img, lab, (max(6, int(np.median(xs)) - 20), max(22, int(np.median(ys)))), 0.9, color)
        drawn.append((graph.label(nid), how))
    imaging.stamp(img, f"f{fid}")
    return imaging.to_jpeg(img, long_side=long_side), drawn


def visible_nodes_in(graph, scene, fid, limit=8):
    """该帧看得见的节点 (提升过的优先), 给 recall(frame=) 描边用."""
    out = [n["id"] for n in graph.visible_nodes() if node_visible_in(graph, scene, n["id"], fid)]
    out.sort(key=lambda nid: (graph.nodes[nid]["inst"] is None, nid))
    return out[:limit]


# ─────────────────────────── 文字 ───────────────────────────
def node_line(graph, nid):
    n = graph.nodes[nid]
    fr = sorted(n["frames"])
    frames = " ".join(f"f{f}" for f in fr[:3]) + (f" +{len(fr) - 3}" if len(fr) > 3 else "")
    parent = f", in {n['parent']}" if n.get("parent") else ""
    claims = graph.claims_text(n["inst"], with_sentence=True) if n["inst"] is not None else ""
    return (f"  {graph.label(nid):<5} {graph.word_of(nid)[:18]:<18} {graph.maturity(nid) + parent:<30} "
            f"{frames:<18} {claims}").rstrip()


def edge_line(graph, e):
    a, b = graph.edge_label(e, "a"), graph.edge_label(e, "b")
    parts = []
    for ev in e["evidence"][-2:]:
        if ev.get("kind") == "relate":
            riv = ", ".join(f"{k} {v}" for k, v in list((ev.get("rivals") or {}).items())[:3])
            parts.append(f"relate {ev.get('pred')} {ev.get('value')} m" + (f" (others: {riv})" if riv else ""))
        else:
            parts.append((ev.get("how") or ev.get("kind", ""))[:60])
    disp = f"  DISPUTED ({e['disputes'][-1]['q']}: {e['disputes'][-1]['why'][:40]})" if e.get("disputes") else ""
    return f"  {a} {e['relation']} {b}   ({e['q'] or '?'}; {'; '.join(p for p in parts if p)}){disp}"


def measured_line(graph, a, b):
    ra, rb = graph.insts.get(graph.nodes[a]["inst"]), graph.insts.get(graph.nodes[b]["inst"])
    if ra is None or rb is None:
        return None
    ca, cb = np.array(ra["centroid"], float), np.array(rb["centroid"], float)
    d = float(np.linalg.norm(ca - cb))
    dz = float(cb[2] - ca[2])
    up = "level" if abs(dz) < 0.1 else (f"{graph.label(b)} {abs(dz):.1f} m higher" if dz > 0 else f"{graph.label(b)} {abs(dz):.1f} m lower")
    return f"  {graph.label(a)} -- {graph.label(b)}   measured: {d:.1f} m apart, {up}"


def near_lifted(graph, nid, radius=config.NEAR_M, exclude=()):
    """提升过的节点一米以内的其它提升过的节点 (几何算)."""
    n = graph.nodes[nid]
    if n["inst"] is None or n["inst"] not in graph.insts:
        return []
    c = np.array(graph.insts[n["inst"]]["centroid"], float)
    out = []
    for m in graph.visible_nodes():
        if m["id"] == nid or m["id"] in exclude or m["inst"] is None or m["inst"] not in graph.insts:
            continue
        d = float(np.linalg.norm(np.array(graph.insts[m["inst"]]["centroid"], float) - c))
        if d <= radius:
            out.append((d, m["id"]))
    return [nid2 for _, nid2 in sorted(out)]


def ledger(graph, st, step, max_steps, tokens_used, token_budget):
    """每步末尾: 本题的账, 几行."""
    qid = getattr(st, "qid", None)
    lines = [f"[this query] step {step}/{max_steps}; input tokens {tokens_used // 1000}k/{token_budget // 1000}k"]
    made = []
    for s, g in graph.segs.items():
        if g["q"] == qid and (g.get("confirmed") or g.get("instance")):
            tail = (f" -> {g['object']}" if g.get("object") else "") + (f" -> #{g['instance']}" if g.get("instance") else "")
            made.append(f"{s} '{g['word']}'{' BOXED' if g.get('boxed') and config.ABLATION != '6' else ''}{tail}")   # 消融 (6): 不漏 BOXED
    for iid in getattr(st, "lifted", []):
        r = graph.insts.get(iid)
        if r:
            if config.ABLATION == "7":                # 消融 (7): 提升诊断 (帧数 / 单视角) 不进账目行
                made.append(f"#{iid} '{r['word']}'")
            else:
                made.append(f"#{iid} '{r['word']}' {r['grade']} {len(r['frames'])}v" + (" SINGLE-VIEW" if r.get("single_view") else ""))
    if made:
        lines.append("made: " + "; ".join(made[-12:]))
    looked = sorted(getattr(st, "seen", ()))
    if looked:
        lines.append("looked at: " + " ".join(f"f{f}" for f in looked[:14]) + (f" +{len(looked) - 14}" if len(looked) > 14 else ""))
    return "\n".join(lines)


# ─────────────────────────── 开题 ───────────────────────────
def keyframes(scene, n=config.KEYFRAMES):
    """固定的总览关键帧: 按覆盖贪心挑 (和 explore 同一套), 场景对象上缓存."""
    kf = getattr(scene, "_keyframes", None)
    if kf is None:
        kf = [f for f, _ in scene.explore_pick(set(), n)]
        scene._keyframes = kf
    return kf


def overview_sheet(scene, fids):
    items = []
    for f in fids:
        img = scene.image(f).copy()
        imaging.stamp(img, f"f{f}", (10, 40), 1.4)
        items.append((imaging.to_jpeg(img, long_side=config.THUMB_SIDE * 2), f"f{f}"))
    return imaging.contact_sheet(items, side=config.THUMB_SIDE, cols=3)


def pick_frames_for(graph, scene, nids, n):
    """命中节点在哪几帧看得最全: 贪心, 每次挑新盖住的命中节点最多的帧."""
    cands = set(graph.looked_frames())
    for nid in nids:
        cands |= graph.nodes[nid]["frames"]
    if not cands or n <= 0 or not nids:
        return [], {}
    vis_map = {f: [nid for nid in nids if node_visible_in(graph, scene, nid, f)] for f in sorted(cands)}
    weight = (lambda f: float(scene.frame_weight[f])) if scene is not None else (lambda f: 0.0)
    picked, covered = [], set()
    while len(picked) < n:
        best, best_key = None, None
        for f in sorted(cands):
            if f in picked:
                continue
            new = len([nid for nid in vis_map[f] if nid not in covered])
            key = (new, len(vis_map[f]), weight(f), -f)
            if new > 0 and (best_key is None or key > best_key):
                best, best_key = f, key
        if best is None:
            break
        picked.append(best)
        covered |= set(vis_map[best])
    return picked, vis_map


def opening(graph, scene, st, sentence, parser=None, n=config.OPEN_IMAGES, use_clip=True):
    """开题消息: 句子 -> 短语 (小模型) -> 命中节点 (名字 / CLIP) -> 节点行 + 边 + 邻居 + 描边帧; 没命中 -> 总览拼图.
    返回 dict(lines, images[(jpeg, caption)], fids, hits(labels), things, how, exact)."""
    out = {"lines": [], "images": [], "fids": [], "hits": [], "things": [], "how": None, "exact": None}
    if graph.visible_nodes():
        parsed = parser.parse(sentence) if parser is not None else {"things": memory_match.ngram_phrases(sentence), "how": "ngram"}
        out["things"], out["how"] = parsed["things"], parsed["how"]
        hits = memory_match.match(graph, parsed["things"], graph.feature_of_node, use_clip=use_clip)[:config.OPEN_MAX_HITS]
    else:
        hits = []
    hit_ids = [nid for nid, _, _, _ in hits]
    out["hits"] = [graph.label(nid) for nid in hit_ids]
    if hits:
        out["lines"].append("memory entries matching this sentence (our own earlier work, not ground truth):")
        shown_e = set()
        for nid, score, phrase, how in hits:
            out["lines"].append(node_line(graph, nid) + f"   [{how} '{phrase}'{'' if how == 'name' else f' {score:.2f}'}]")
            for e in graph.edges_of(nid)[:config.GRAPH_MAX_EDGES_PER_NODE]:
                if e["id"] not in shown_e:
                    shown_e.add(e["id"])
                    out["lines"].append(edge_line(graph, e))
        extra = []
        for nid in hit_ids:
            for m in near_lifted(graph, nid, exclude=hit_ids):
                if m not in extra:
                    extra.append(m)
        for m in extra[:4]:
            out["lines"].append(node_line(graph, m) + "   [near a matched entry]")
            for nid in hit_ids:
                ml = measured_line(graph, nid, m) if graph.nodes[nid]["inst"] is not None else None
                if ml:
                    out["lines"].append(ml)
        fids, vis_map = pick_frames_for(graph, scene, hit_ids, n)
        for f in fids:
            jpeg, drawn = outline_frame(graph, scene, f, vis_map.get(f, []))
            what = ", ".join(f"{lab} {graph.word_of(graph.node_of(lab))}" + (" (approx)" if how == "approx" else "") for lab, how in drawn)
            cap = f"memory: f{f}, outlined: {what}" if what else f"memory: f{f}"
            out["images"].append((jpeg, cap))
            out["lines"].append(cap)
            st.shown[f] = (0, 0, scene.W, scene.H)
            st.seen.add(f)
            st.seen_scene.add(f)
        out["fids"] = fids
    else:
        if graph.visible_nodes():
            out["lines"].append("memory: nothing recorded matches this sentence" + (f" (looked for: {', '.join(t['name'] for t in out['things'][:6])})" if out["things"] else "") + ".")
        kf = keyframes(scene)
        if kf and n > 0:
            out["images"].append((overview_sheet(scene, kf), "room overview: " + " ".join(f"f{f}" for f in kf)))
            out["lines"].append(f"room overview attached: frames {' '.join(f'f{f}' for f in kf)} (whole frames, low-res); "
                                f"zoom with nearby(frame=, box=) or explore() from there.")
            for f in kf:                                  # 总览里的帧对本题算看过 (explore 不再重复给)
                st.shown[f] = (0, 0, scene.W, scene.H)
                st.seen.add(f)
                st.seen_scene.add(f)
            out["fids"] = list(kf)
    exact = graph.exact_match(sentence)
    if exact is not None:
        out["exact"] = exact["qid"]
        out["lines"].append(f"This exact sentence was answered before ({exact['qid']}) as "
                            + " ".join(f"#{i}" for i in exact["ids"]) + "; evidence then: " + (exact.get("evidence_text") or "")[:200])
    return out


# ─────────────────────────── 记忆 agent 的交接页 -> 开题消息 ───────────────────────────
def handoff_lines(graph, page):
    """交接页 -> 文字: 查询图、每个实体在记忆里的匹配 (子图)、排除的同类、已有的边、先看哪帧."""
    lines = ["memory (matching subgraph from the memory keeper; hypotheses, verify before use):"]
    q = page.get("query") or {}
    if q.get("entities"):
        ents = " ; ".join(f"[{e['role']}] {e['name']}" for e in q["entities"])
        rels = " ; ".join(f"{a} --{r}--> {b}" for a, r, b in q.get("relations", []))
        lines.append("query: " + ents + (f"   relations: {rels}" if rels else ""))
    matches = page.get("matches") or []
    if matches:
        lines.append("matched in memory:")
        for m in matches:
            nid = graph.node_of(m["id"])
            if nid is not None:
                lines.append(f"  {m['entity']} -> " + node_line(graph, nid).strip())
    if page.get("missing"):
        lines.append("not in memory: " + ", ".join(page["missing"]))
    if not matches and not page.get("missing"):
        lines.append("matched in memory: nothing")
    if page.get("ruled_out"):
        lines.append("ruled out (same kind, not it):")
        for c in page["ruled_out"]:
            nid = graph.node_of(c["id"])
            if nid is not None:
                lines.append("  " + node_line(graph, nid).strip())
    shown_e = set()
    ids = [x["id"] for x in matches]
    edge_lines = []
    for lab in ids:
        nid = graph.node_of(lab)
        for e in graph.edges_of(nid)[:config.GRAPH_MAX_EDGES_PER_NODE] if nid is not None else []:
            if e["id"] not in shown_e:
                shown_e.add(e["id"])
                edge_lines.append(edge_line(graph, e))
    if edge_lines:
        lines.append("relations recorded:")
        lines += edge_lines
    if page.get("look_first"):
        lines.append("look first: " + " ".join(f"f{f}" for f in page["look_first"]))
    return lines


def handoff_images(graph, scene, st, page, n=config.OPEN_IMAGES):
    """交接页 -> 描边帧 (候选 + 锚点); 没帧没候选 -> 房间总览."""
    ids = [x["id"] for x in (page.get("matches") or page.get("candidates", []) + page.get("anchors", []))]
    nids = list(dict.fromkeys(graph.node_of(l) for l in ids if graph.node_of(l) is not None))
    fids = list(page.get("look_first") or [])
    if not fids and nids:
        fids, _ = pick_frames_for(graph, scene, nids, n)
    images = []
    for f in fids[:n]:
        vis = [nid for nid in nids if node_visible_in(graph, scene, nid, f)]
        jpeg, drawn = outline_frame(graph, scene, f, vis)
        what = ", ".join(f"{lab} {graph.word_of(graph.node_of(lab))}" for lab, _ in drawn)
        images.append((jpeg, f"memory: f{f}" + (f", outlined: {what}" if what else "")))
        st.shown[f] = (0, 0, scene.W, scene.H)
        st.seen.add(f)
        st.seen_scene.add(f)
    if not images and n > 0:
        kf = keyframes(scene)
        if kf:
            images.append((overview_sheet(scene, kf), "room overview: " + " ".join(f"f{f}" for f in kf)))
            for f in kf:
                st.shown[f] = (0, 0, scene.W, scene.H)
                st.seen.add(f)
                st.seen_scene.add(f)
            fids = list(kf)
    return images, fids[:n] if images and not images[-1][1].startswith("room overview") else fids


def frame_view(graph, scene, st, fid):
    nodes = visible_nodes_in(graph, scene, fid)
    jpeg, drawn = outline_frame(graph, scene, fid, nodes)
    labs = [lab for lab, _ in drawn]
    st.shown[fid] = (0, 0, scene.W, scene.H)
    st.seen.add(fid)
    st.seen_scene.add(fid)
    qs = sorted({h["q"] for h in graph.looked.get(fid, []) if h["q"]})
    cap = f"f{fid} from memory" + (f", outlined: {', '.join(labs)}" if labs else "")
    return jpeg, cap, labs, qs
