"""记忆 agent (施工方案 v4, 2026-09-10): 一个常驻的小 VLM, 只对场景图和缓存工作, 不看新帧、不切、不答.

两份工作:
  1. 题中跟着看 (后台线程, 在主 agent 思考的空档): segment 带回的相似块逐个看图 -> 确认 (给名字) / 丢弃 / 并入已有节点;
     确认过没提升的组做一次粗提 (显卡空闲时); 作答后把没选的候选记为"排除".
  2. 换题时交接: 读新句子, 用 look / frame / measure 在图里找, 交一页固定格式的假设 (候选、锚点、缺的、先看哪几帧) 给主 agent.
规矩: 图是唯一真相; 它只做带证据的结构操作 (confirm / discard / merge / alias / exclude), 只追加; 它的上下文一题一清; 交接是假设.
模型: DashScope 兼容接口 (默认 qwen3-vl-30b-a3b-instruct, 环境变量 MEMORY_AGENT_MODEL 可换), 以后可换本地服务.
"""
import base64
import json
import os
import queue
import re
import threading
import time

import numpy as np

from . import config
from . import legacy
from . import imaging
from . import lifting
from . import render
from .cache import norm_label, norm_word


# ─────────────────────────── 小 VLM 客户端 ───────────────────────────
class VlmClient:
    """记忆 agent 的模型: 文字 (+ 若干 jpeg) -> 文字. 按模型名选后端:
    gpt* -> 主 agent 同一个 key / 网址 (agent/config.py), chat/completions, 带 reasoning_effort;
    其余 -> DashScope 兼容接口 (环境变量 DASHSCOPE_API_KEY / DASHSCOPE_BASE_URL)."""

    def __init__(self, model=None, base=None, timeout=None, reasoning=None):
        self.model = model or config.MEMORY_AGENT_MODEL
        self.openai = self.model.lower().startswith("gpt")
        if self.openai:
            cfg = legacy.llm_config()
            self.base = base or cfg.LLM_URL.rsplit("/chat/completions", 1)[0]
            self.key = cfg.LLM_API_KEY
        else:
            self.base = base or os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
            self.key = os.environ.get("DASHSCOPE_API_KEY")
        self.reasoning = config.MEMORY_AGENT_REASONING if reasoning is None else reasoning
        self.timeout = timeout or config.MEMORY_AGENT_HTTP_TIMEOUT
        self.usage = {"calls": 0, "seconds": 0.0, "input_tokens": 0, "output_tokens": 0, "errors": 0}

    def chat(self, prompt, images=(), max_tokens=700):
        """文字 + 若干 jpeg -> 文字. 出错抛 RuntimeError."""
        import requests
        if not self.key:
            raise RuntimeError("OPENAI_API_KEY not set" if self.openai else "DASHSCOPE_API_KEY not set")
        content = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(j).decode()}}
                   for j in images]
        content.append({"type": "text", "text": prompt})
        body = {"model": self.model, "messages": [{"role": "user", "content": content}]}
        if self.openai:                                   # 5.x: 不带 temperature; 思考的 token 也算在上限里, 留出余量
            thinking = self.reasoning not in (None, "", "none")
            body["max_completion_tokens"] = max_tokens + (config.MEMORY_AGENT_REASONING_HEADROOM if thinking else 0)
            if self.reasoning:
                body["reasoning_effort"] = self.reasoning
        else:
            body["temperature"] = 0
            body["max_tokens"] = max_tokens
        t0 = time.time()
        try:
            r = requests.post(self.base + "/chat/completions", json=body, timeout=self.timeout,
                              headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
            j = r.json()
            if "choices" not in j:
                raise RuntimeError(str(j.get("error", j))[:300])
            u = j.get("usage") or {}
            self.usage["input_tokens"] += int(u.get("prompt_tokens") or 0)
            self.usage["output_tokens"] += int(u.get("completion_tokens") or 0)
            return j["choices"][0]["message"]["content"]
        except Exception as e:                            # noqa: BLE001
            self.usage["errors"] += 1
            raise RuntimeError(f"vlm: {type(e).__name__}: {e}") from e
        finally:
            self.usage["calls"] += 1
            self.usage["seconds"] += time.time() - t0


def parse_json(text):
    m = re.search(r"\{.*\}", str(text), re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except ValueError:
        try:                                              # 去掉尾随逗号这类小毛病
            return json.loads(re.sub(r",\s*([}\]])", r"\1", m.group(0)))
        except ValueError:
            return {}


HANDOFF_PROMPT = """You keep the memory of a 3D indoor scene for a grounding agent. The memory is a scene graph built while answering earlier sentences in this room. Nodes are objects and parts (id, names, state, frames it appears in, what the earlier sentence said about it). Edges are relations between nodes: geometric ones (beside, above, under, ...) or part-of ones (a handle of a cabinet, a lid of a bin). The graph is incomplete, may contain mistakes, and one name can cover several different objects.

Scene graph:
{listing}

New sentence: "{sentence}"

1. Build the query graph of the sentence: every object or part the sentence mentions is a node (mark which one is the target); every relation between them is an edge, geometric ("above the stove") or part-of ("the handle of the cabinet").
2. Match the query graph against the scene graph and find the subgraph they share: a query node matches a scene node that is the same object or part; a query edge matches a scene edge with the same meaning between the matched nodes. Names and relation words may differ between the two graphs (aliases: cooktop / stove, beside / next to), so judge by meaning. A node is a match only if nothing recorded about it (its earlier sentence, its edges) contradicts what this sentence says about that entity; a node of the same kind whose record contradicts the sentence is a different object: put it under "ruled_out", never in matches. Leave a node unmatched rather than guess; if several scene nodes fit equally, list them all.
3. look_first: 1-2 frames where the matched nodes are best seen; the agent is shown these frames first.
{tools}Return the shared subgraph as JSON in this shape (fill it from THIS sentence and THIS graph):
  {{"query": {{"entities": [{{"name": "<object or part>", "role": "target"}}, {{"name": "<object or part>", "role": "anchor"}}, ...],
              "relations": [["<entity>", "<relation word>", "<entity>"], ...]}},
   "matches": [{{"entity": "<entity>", "id": "<node id>"}}, ...],
   "ruled_out": ["<node id>", ...],
   "look_first": ["<frame id>"]}}
No explanations. Ids must come from the scene graph. JSON only.
{extra}"""

HANDOFF_TOOLS = """You may first ask for evidence (one JSON per reply):
  {"look": ["#14", "s60"]}      thumbnails of up to 6 entries
  {"frame": "f12"}              that frame with the entries outlined
  {"measure": [["#14", "#16"]]} distance and height difference between two lifted entries
Or """

SEGMENT_PROMPT = """You keep the memory of a 3D indoor scene. A segmentation call with the word "{word}" just returned the segments in sheet A (labels printed on each tile){boxed_note}. Sheet B shows existing memory entries that might be the same objects (labels printed).
For EACH segment in sheet A decide:
- keep: true if the tile shows one real, whole object (any kind), false if it is a fragment, background, or a misdetection;
- name: 1-2 plain words for what it is (may differ from "{word}");
- same_as: the label of an existing entry in sheet B that is the SAME physical object (same place, same thing), or another sheet-A label, else null.
Reply with JSON only: {{"segments": [{{"id": "s10", "keep": true, "name": "cabinet", "same_as": null}}, ...]}}"""


class MemoryAgent:
    def __init__(self, graph, scene, vlm, enabled=True, watch=True, rough_lift=None, log_path=None,
                 max_handoff_calls=None, event_timeout=None):
        self.g, self.scene, self.vlm = graph, scene, vlm
        self.enabled, self.watch = enabled, watch
        self.rough_lift_on = config.MEMORY_AGENT_ROUGH_LIFT if rough_lift is None else rough_lift
        self.max_calls = max_handoff_calls or config.MEMORY_AGENT_MAX_HANDOFF_CALLS
        self.event_timeout = event_timeout or config.MEMORY_AGENT_EVENT_TIMEOUT
        self.log_path = log_path
        self.main_busy = False                            # 主 agent 正在跑工具 (显卡在用): 粗提等它
        self.stats = {"handoffs": 0, "handoff_calls": 0, "handoff_seconds": 0.0, "handoff_fallbacks": 0,
                      "events": 0, "event_seconds": 0.0, "event_timeouts": 0, "event_errors": 0,
                      "confirmed": 0, "discarded": 0, "merged": 0, "excluded": 0, "rough_lifts": 0}
        self.last_page = None
        self.qid = None
        self._q = queue.Queue()
        self._pending = 0
        self._cv = threading.Condition()
        self._stop = False
        self._thread = threading.Thread(target=self._consume, daemon=True, name="memory-agent")
        self._thread.start()

    # ─────────────────────────── 记录 ───────────────────────────
    def _log(self, rec):
        rec = dict(rec, t=round(time.time(), 2), q=self.qid)
        if self.log_path:
            try:
                with open(self.log_path, "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except OSError:
                pass

    def begin_query(self, qid):
        self.qid = qid
        self.last_page = None

    # ─────────────────────────── 图 -> 文字 / 图 ───────────────────────────
    def listing(self, limit=80):
        g = self.g
        lines = []
        for n in sorted(g.visible_nodes(), key=lambda n: (n["inst"] is None, n["inst"] or 0, n["id"]))[:limit]:
            nid = n["id"]
            fr = sorted(n["frames"])
            frames = " ".join(f"f{f}" for f in fr[:3]) + (f" +{len(fr) - 3}" if len(fr) > 3 else "")
            claims = g.claims_text(n["inst"], with_sentence=True, maxlen=50) if n["inst"] is not None else ""
            parent = f", in {n['parent']}" if n.get("parent") else ""
            line = f"{g.label(nid):<5} {'/'.join(n['names'][:3]):<22} {g.maturity(nid) + parent:<34} {frames:<18} {claims}"
            edges = []
            for e in g.edges_of(nid)[:3]:
                edges.append(f"{g.edge_label(e, 'a')} {e['relation']} {g.edge_label(e, 'b')} ({e['q']})")
            if edges:
                line += "   | " + "; ".join(edges)
            lines.append(line.rstrip())
        return "\n".join(lines) if lines else "(memory is empty)"

    def thumb(self, label):
        nid = self.g.node_of(label)
        if nid is None:
            return None
        n = self.g.nodes[nid]
        if n["inst"] is not None:
            b = self.g.thumb_bytes(n["inst"])
            if b:
                return b
        for s in n["s"]:
            b = self.g.cutout_bytes(s)
            if b:
                return b
        return None

    def look(self, labels):
        items, missing = [], []
        for lab in [norm_label(x) for x in labels][:config.RECALL_MAX_THUMBS]:
            b = self.thumb(lab)
            if b:
                items.append((b, self.g.label(self.g.node_of(lab))))
            else:
                missing.append(lab)
        if not items:
            return None, f"no thumbnails for {' '.join(missing)}"
        return imaging.contact_sheet(items), "thumbnails: " + ", ".join(l for _, l in items) + (f" (none for {' '.join(missing)})" if missing else "")

    def frame(self, fid):
        fid = int(re.search(r"\d+", str(fid)).group(0))
        if not 0 <= fid < self.scene.n_frames:
            return None, f"frame f{fid} does not exist"
        nodes = render.visible_nodes_in(self.g, self.scene, fid)
        jpeg, drawn = render.outline_frame(self.g, self.scene, fid, nodes)
        return jpeg, f"f{fid}: outlined " + (", ".join(l for l, _ in drawn) or "nothing")

    def measure(self, a, b):
        g = self.g
        na, nb = g.node_of(a), g.node_of(b)
        if na is None or nb is None:
            return f"unknown id in {a}, {b}"
        ia, ib = g.nodes[na]["inst"], g.nodes[nb]["inst"]
        if ia is None or ib is None:
            return f"{g.label(na)} -- {g.label(nb)}: one of them is not lifted, no geometry"
        ra, rb = g.insts[ia], g.insts[ib]
        ca, cb = np.array(ra["centroid"], float), np.array(rb["centroid"], float)
        d = float(np.linalg.norm(ca - cb))
        dz = float(cb[2] - ca[2])
        ea, eb = np.array(ra["obb"]["extent"], float), np.array(rb["obb"]["extent"], float)
        gap = d - 0.5 * (float(np.max(ea)) + float(np.max(eb)))
        return (f"{g.label(na)} -- {g.label(nb)}: centers {d:.2f} m apart, {g.label(nb)} is {abs(dz):.2f} m "
                f"{'higher' if dz > 0 else 'lower'}; boxes {'overlap/touch' if gap <= 0 else f'{gap:.2f} m apart'}")

    # ─────────────────────────── 换题: 交接 ───────────────────────────
    def handoff(self, sentence, st=None):
        """-> page dict 或 None (失败 / 关闭 / 图空). page = {query, matches, ruled_out, look_first, candidates, anchors, missing, calls}."""
        if not self.enabled or not self.g.visible_nodes():
            return None
        t0 = time.time()
        self.stats["handoffs"] += 1
        listing = self.listing()
        evidence, images = [], []
        page = None
        for k in range(self.max_calls):
            extra = ("\nEvidence gathered so far:\n" + "\n".join(evidence)) if evidence else ""
            if self.max_calls > 1 and k == self.max_calls - 1:
                extra += "\nThis is your last reply: give the final page now."
            prompt = HANDOFF_PROMPT.format(listing=listing, sentence=sentence, extra=extra,
                                           tools=HANDOFF_TOOLS if self.max_calls > 1 else "")
            try:
                text = self.vlm.chat(prompt, images[-6:], max_tokens=config.MEMORY_AGENT_MAX_TOKENS)
            except RuntimeError as e:
                self._log({"kind": "handoff_error", "error": str(e)[:200]})
                break
            self.stats["handoff_calls"] += 1
            j = parse_json(text)
            if not j:
                self._log({"kind": "handoff_error", "error": "reply not JSON (truncated?)", "raw": str(text)[-300:]})
            self._log({"kind": "handoff_turn", "turn": k, "reply": j})
            if "final" not in j and ("matches" in j or "query" in j):     # 没包 final 也收
                j = {"final": j}
            if "final" in j:
                page = self._validate_page(j["final"])
                break
            if self.max_calls == 1:                       # 单次口径: 没交页就算失败 (外壳退到程序检索)
                self._log({"kind": "handoff_error", "error": "no final page in single-call mode", "reply": j})
                break
            asked = False
            if j.get("look"):
                sheet, txt = self.look(j["look"] if isinstance(j["look"], list) else [j["look"]])
                evidence.append("look -> " + txt)
                if sheet:
                    images.append(sheet)
                asked = True
            if j.get("frame"):
                jpeg, txt = self.frame(j["frame"])
                evidence.append("frame -> " + txt)
                if jpeg:
                    images.append(jpeg)
                asked = True
            if j.get("measure"):
                pairs = j["measure"] if isinstance(j["measure"], list) else []
                for p in pairs[:4]:
                    if isinstance(p, (list, tuple)) and len(p) == 2:
                        evidence.append("measure -> " + self.measure(p[0], p[1]))
                asked = True
            if not asked:
                evidence.append("(your reply was not understood; give the final page)")
        self.stats["handoff_seconds"] += time.time() - t0
        if page is None:
            self.stats["handoff_fallbacks"] += 1
            return None
        page["calls"] = k + 1
        page["seconds"] = round(time.time() - t0, 2)
        self.last_page = page
        self._log({"kind": "handoff", "page": page})
        return page

    def _validate_page(self, fin):
        """交接页 = 查询图 + 匹配 (实体 -> 节点) + 排除 (同类但不是) + 先看哪帧, 都不带理由 (用户 09-10 定: 少写省时间). 只做结构校验: 编号必须在图里且进视图, 编造的剔掉;
        不评判匹配对不对 (那是它的活, 主 agent 作答前再核). 兼容字段: candidates = 目标实体的匹配, anchors = 其余的, missing = 没匹配上的."""
        g = self.g
        q = fin.get("query") or {}
        ents = []
        for e in (q.get("entities") or [])[:8]:
            if isinstance(e, str):
                e = {"name": e, "role": "anchor"}
            if not isinstance(e, dict) or not str(e.get("name", "")).strip():
                continue
            role = "target" if str(e.get("role", "")).lower().startswith("t") else "anchor"
            ents.append({"name": norm_word(str(e["name"]))[:40], "role": role})
        if ents and not any(e["role"] == "target" for e in ents):
            ents[0]["role"] = "target"
        names = {e["name"] for e in ents}
        rels = []
        for r in (q.get("relations") or [])[:8]:
            if isinstance(r, (list, tuple)) and len(r) == 3:
                a, rel, b = (norm_word(str(r[0]))[:40], str(r[1])[:30], norm_word(str(r[2]))[:40])
                if a in names and b in names:
                    rels.append([a, rel, b])
        matches, seen = [], set()
        for it in (fin.get("matches") or [])[:16]:
            if not isinstance(it, dict):
                continue
            nid = g.node_of(str(it.get("id", "")))
            ent = norm_word(str(it.get("entity", "")))[:40]
            if nid is None or not g.node_visible(nid) or (ent, nid) in seen:
                continue
            if ent not in names:                          # 实体名没在查询图里: 当作锚点补上
                ents.append({"name": ent or g.word_of(nid), "role": "anchor"})
                names.add(ents[-1]["name"])
                ent = ents[-1]["name"]
            seen.add((ent, nid))
            matches.append({"entity": ent, "id": g.label(nid)})
        matched_ids = {m["id"] for m in matches}
        ruled_out = []
        for it in (fin.get("ruled_out") or [])[:8]:
            lab = it.get("id", "") if isinstance(it, dict) else it        # 裸编号或 {"id": ...} 都收
            nid = g.node_of(str(lab))
            if nid is None or not g.node_visible(nid) or g.label(nid) in matched_ids or any(x["id"] == g.label(nid) for x in ruled_out):
                continue
            ruled_out.append({"id": g.label(nid)})
        look = []
        for f in (fin.get("look_first") or [])[:4]:
            m = re.search(r"\d+", str(f))
            if m and 0 <= int(m.group(0)) < self.scene.n_frames and int(m.group(0)) not in look:
                look.append(int(m.group(0)))
        target_names = {e["name"] for e in ents if e["role"] == "target"}
        cands = [{"id": m["id"]} for m in matches if m["entity"] in target_names][:6]
        anchors = [{"id": m["id"]} for m in matches if m["entity"] not in target_names][:6]
        missing = [e["name"] for e in ents if not any(m["entity"] == e["name"] for m in matches)]
        return {"query": {"entities": ents, "relations": rels}, "matches": matches, "ruled_out": ruled_out,
                "look_first": look[:2], "candidates": cands, "anchors": anchors, "missing": missing[:6]}

    # ─────────────────────────── 题中: 事件 ───────────────────────────
    def notify(self, kind, payload):
        if not self.enabled or not self.watch:
            return
        with self._cv:
            self._pending += 1
        self._q.put((kind, payload, time.time()))

    def drain(self, timeout=None):
        """等后台把本题的事件处理完 (换题前调用). 超时就不等了, 剩下的事件照常处理."""
        timeout = config.MEMORY_AGENT_DRAIN_TIMEOUT if timeout is None else timeout
        with self._cv:
            self._cv.wait_for(lambda: self._pending == 0, timeout=timeout)
            return self._pending == 0

    def close(self):
        self._stop = True
        self._q.put(("stop", None, time.time()))

    def _consume(self):
        while not self._stop:
            kind, payload, t_in = self._q.get()
            if kind == "stop":
                break
            t0 = time.time()
            try:
                if time.time() - t_in > self.event_timeout * 3:       # 排队太久的丢掉 (下次碰到这个节点再补)
                    self.stats["event_timeouts"] += 1
                elif kind == "segment":
                    self._on_segment(payload)
                elif kind == "answer":
                    self._on_answer(payload)
            except Exception as e:                        # noqa: BLE001  记忆 agent 出错不影响跑题
                self.stats["event_errors"] += 1
                self._log({"kind": "event_error", "event": kind, "error": f"{type(e).__name__}: {e}"[:200]})
            finally:
                self.stats["events"] += 1
                self.stats["event_seconds"] += time.time() - t0
                with self._cv:
                    self._pending -= 1
                    self._cv.notify_all()

    def _on_segment(self, p):
        """p: {word, names, boxed, qid}. 相似块逐个看: 确认 (给名字) / 丢弃 / 并入."""
        g = self.g
        news = [s for s in p["names"] if s in g.segs and s != p.get("boxed") and not g.segs[s].get("confirmed")]
        if not news:
            return
        items_a = [(g.cutout_bytes(s), s) for s in news if g.cutout_bytes(s)]
        if not items_a:
            return
        head = p["word"].split()[-1]
        existing = [n for n in g.visible_nodes() if any(head in nm.split() for nm in n["names"])]
        items_b = []
        for n in existing[:config.RECALL_MAX_THUMBS]:
            b = self.thumb(g.label(n["id"]))
            if b:
                items_b.append((b, g.label(n["id"])))
        images = [imaging.contact_sheet(items_a[:6])]
        if items_b:
            images.append(imaging.contact_sheet(items_b))
        boxed_note = f' (the boxed one, {p["boxed"]}, is already confirmed and is not in sheet A)' if p.get("boxed") else ""
        prompt = SEGMENT_PROMPT.format(word=p["word"], boxed_note=boxed_note)
        if not items_b:
            prompt = prompt.replace("Sheet B shows existing memory entries that might be the same objects (labels printed).",
                                    "There is no sheet B (no similar entries in memory yet).")
        text = self.vlm.chat(prompt, images, max_tokens=600)
        j = parse_json(text)
        decisions = j.get("segments") or []
        self._log({"kind": "segment_review", "word": p["word"], "new": [s for _, s in items_a], "decisions": decisions})
        allowed = {s for _, s in items_a}
        for d in decisions:
            if not isinstance(d, dict):
                continue
            s = str(d.get("id", "")).strip().lower()
            if s not in allowed:
                continue
            same = d.get("same_as")
            if d.get("keep") is False:
                g.discard(s, d.get("name") or "fragment/misdetection", by="memory")
                self.stats["discarded"] += 1
                continue
            name = norm_word(str(d.get("name") or p["word"]))
            g.confirm(s, name=name, by="memory")
            self.stats["confirmed"] += 1
            if same and str(same).strip().lower() != s:
                tgt = norm_label(str(same))
                if g.node_of(tgt) is not None and g.node_of(tgt) != g.node_of(s):
                    g.merge(tgt, s, why=f"memory agent saw the same object ({p['word']})", by="memory")
                    self.stats["merged"] += 1
        if self.rough_lift_on:
            for s in news:
                sg = g.segs.get(s)
                if sg and sg.get("confirmed") and sg.get("instance") is None and g.nodes[sg["node"]]["inst"] is None:
                    self._rough_lift(s)

    def _rough_lift(self, sname):
        """确认过没提升的组做一次粗提 (显卡空闲时), 让它有三维. 主 agent 在跑工具就等一会."""
        waited = 0.0
        while self.main_busy and waited < 30.0:
            time.sleep(0.2)
            waited += 0.2
        if self.main_busy:
            self.stats["event_timeouts"] += 1
            return
        g = self.g
        sg = g.segs[sname]
        frames = dict(sg["frames"])
        if not frames:
            return
        members, info, _ = lifting.full_lift(self.scene, frames)
        if members is None or len(members) < config.MIN_INST_GAUSSIANS:
            self._log({"kind": "rough_lift_failed", "s": sname, "reason": (info or {}).get("reason")})
            return
        geo = self.scene.obb_and_points(members)
        fid = max(frames, key=lambda f: int(np.asarray(frames[f]).sum()))
        with g.lock:
            iid, how = g.add_instance(sg["word"], members, geo, list(frames), "rough",
                                      {f: {"word": sg["word"], "within": sg.get("parent_id")} for f in frames},
                                      step=0, source=sname, groups=[sname], single_view=len(frames) < config.MIN_CONSENSUS_VIEWS,
                                      exemplar_fid=fid, by="memory")
            if how != "alias":
                g.put_instance_masks(iid, frames)
                cut = g.cutout_bytes(sname)
                if cut:
                    g.put_thumb(iid, cut)
        self.stats["rough_lifts"] += 1
        self._log({"kind": "rough_lift", "s": sname, "inst": iid, "how": how, "n": int(len(members))})

    def _on_answer(self, p):
        """作答后: 交接页里排除的 (ruled_out), 和给过又没被答的目标匹配, 都记为"排除"."""
        page = p.get("page") or {}
        ids = {f"#{i}" for i in (p.get("ids") or [])}
        left = [c["id"] for c in page.get("candidates", []) if c["id"] not in ids]
        not_it = [c["id"] for c in page.get("ruled_out", []) if c["id"] not in ids]
        if p.get("status") == "answered":
            if left:
                self.g.exclude(left, why=f"not chosen for \"{p.get('text', '')[:60]}\"", by="memory")
            if not_it:
                self.g.exclude(not_it, why=f"ruled out for \"{p.get('text', '')[:60]}\"", by="memory")
            self.stats["excluded"] += len(left) + len(not_it)

    def summary(self):
        return dict(self.stats, vlm=dict(self.vlm.usage))
