"""查询 -> 记忆里的节点 (v3 改, 2026-09-09): 句子先由一个小模型改写成"提到的东西"的短语 (名字 + 描述),
每条短语算 CLIP 文本向量, 和节点的 CLIP 图像向量 (腐蚀后的掩码抠图) 比余弦; 名字字面对上的也算命中.
相似块 (模型没用过的组) 不参与. 小模型不可用时退到片段法 (句子切成 1-3 词的片段)."""
import json
import os
import re
import time

import numpy as np

from . import config
from . import imaging
from .cache import norm_word

STOP = set("the a an of on in at to by with from under over above below beside near next between behind front "
           "left right into onto and or that which is are was were this these those it its there here up down "
           "closest nearest farthest furthest highest lowest largest smallest biggest top bottom middle two three "
           "standing sitting hanging lying placed put set one".split())


# ─────────────────────────── CLIP ───────────────────────────
def _clip():
    raise RuntimeError("the CLIP channel of the memory module is not included in this release (memory is off by default)")


def image_feature(image, mask):
    """一帧一组 -> (D,) ℓ2 归一向量: 掩码腐蚀后白底抠图给 CLIP."""
    import torch
    from PIL import Image
    cc = _clip()
    m = imaging.erode(mask, config.CLIP_ERODE_PX)
    if not m.any():
        m = mask.astype(bool)
    cut = imaging.cutout(image, m)
    if cut.size == 0 or min(cut.shape[:2]) < 4:
        return None
    pil = Image.fromarray(cut[:, :, ::-1])
    with torch.no_grad():
        batch = cc._PREPROCESS(pil).unsqueeze(0).to(cc._DEVICE)
        emb = cc._MODEL.encode_image(batch).float().cpu().numpy()[0]
    return emb / max(float(np.linalg.norm(emb)), 1e-8)


def group_feature(scene, g, n_frames=config.CLIP_FEAT_FRAMES):
    """一组 -> 掩码最大的几帧各算一个向量取平均再归一."""
    fids = sorted(g["frames"], key=lambda f: -int(np.asarray(g["frames"][f]).sum()))[:n_frames]
    vecs = []
    for f in fids:
        v = image_feature(scene.image(f), g["frames"][f])
        if v is not None:
            vecs.append(v)
    if not vecs:
        return None
    v = np.mean(vecs, axis=0)
    return v / max(float(np.linalg.norm(v)), 1e-8)


def text_features(phrases):
    cc = _clip()
    return np.stack([cc.encode_query(p) for p in phrases]) if phrases else np.zeros((0, 512), np.float32)


# ─────────────────────────── 句子 -> 短语 ───────────────────────────
PARSE_PROMPT = """Sentence: "{text}"
List every physical thing this sentence refers to (the target and the things it is described relative to).
For each give: "name" = the most likely plain name (1-3 words; if the sentence only describes it, guess the kind, e.g. "cup");
"aliases" = up to 3 other common names for the same kind of thing (e.g. ["mug"]); "desc" = only its color / material / shape words from the sentence (e.g. "white inside", "red-striped"), never a location or relation; else "".
Return JSON only: {{"things": [{{"name": "...", "aliases": ["..."], "desc": "..."}}]}}"""


def ngram_phrases(text, max_n=3):
    """兜底: 去掉虚词后的 1-3 词片段."""
    toks = [t for t in re.findall(r"[a-z0-9-]+", str(text).lower()) if t not in STOP]
    out = []
    for n in range(1, max_n + 1):
        for i in range(len(toks) - n + 1):
            p = " ".join(toks[i:i + n])
            if p not in out:
                out.append(p)
    return [{"name": p, "aliases": [], "desc": ""} for p in out]


class QueryParser:
    """句子 -> [{"name", "desc"}]. 后端: dashscope 小模型 / main 主模型 (无图) / none. 逐句缓存 (文件 + 内存)."""

    def __init__(self, backend=None, model=None, cache_path=None, main_client=None):
        self.backend = backend or config.QUERY_PARSER
        self.model = model or config.QUERY_PARSER_MODEL
        self.cache_path = cache_path
        self.main = main_client
        self.cache = {}
        self.usage = {"calls": 0, "seconds": 0.0, "input_tokens": 0, "output_tokens": 0, "fallbacks": 0}
        if cache_path and os.path.exists(cache_path):
            try:
                self.cache = json.load(open(cache_path))
            except (ValueError, OSError):
                self.cache = {}

    def parse(self, text):
        key = norm_word(text)
        if key in self.cache:
            return self.cache[key]
        things, how = None, self.backend
        if self.backend == "dashscope":
            things = self._dashscope(text)
        elif self.backend == "main":
            things = self._main(text)
        if not things:
            things, how = ngram_phrases(text), "ngram"
            if self.backend != "none":
                self.usage["fallbacks"] += 1
        out = {"things": things, "how": how}
        self.cache[key] = out
        if self.cache_path:
            try:
                json.dump(self.cache, open(self.cache_path, "w"), ensure_ascii=False, indent=1)
            except OSError:
                pass
        return out

    @staticmethod
    def _clean(things):
        out = []
        for t in things or []:
            if not isinstance(t, dict):
                continue
            nm = norm_word(str(t.get("name", "")))
            if not nm or len(nm.split()) > 4:
                continue
            al = t.get("aliases") or []
            if isinstance(al, str):
                al = re.split(r"[,/;]", al)
            al = [norm_word(a) for a in al if str(a).strip()][:3]
            out.append({"name": nm, "aliases": [a for a in al if a and a != nm], "desc": str(t.get("desc", "")).strip()[:120]})
        return out

    def _dashscope(self, text):
        import requests
        key = os.environ.get("DASHSCOPE_API_KEY")
        if not key:
            return None
        base = os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        body = {"model": self.model, "temperature": 0, "max_tokens": 300,
                "messages": [{"role": "user", "content": PARSE_PROMPT.format(text=text)}]}
        t0 = time.time()
        try:
            r = requests.post(base + "/chat/completions", json=body, timeout=config.QUERY_PARSER_TIMEOUT,
                              headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            j = r.json()
            content = j["choices"][0]["message"]["content"]
            u = j.get("usage") or {}
            self.usage["input_tokens"] += int(u.get("prompt_tokens") or 0)
            self.usage["output_tokens"] += int(u.get("completion_tokens") or 0)
        except Exception as e:                          # noqa: BLE001  小模型挂了就退片段法, 不影响跑题
            print(f"[parser] dashscope failed: {type(e).__name__}: {e}", flush=True)
            return None
        finally:
            self.usage["calls"] += 1
            self.usage["seconds"] += time.time() - t0
        return self._clean(_json_block(content).get("things"))

    def _main(self, text):
        if self.main is None:
            return None
        from .llm_client import History
        h = History()
        h.user_text(PARSE_PROMPT.format(text=text))
        t0 = time.time()
        try:
            resp = self.main.step("You extract the physical things a sentence refers to. Reply with JSON only.", h, [])
        except Exception as e:                          # noqa: BLE001
            print(f"[parser] main model failed: {e}", flush=True)
            return None
        finally:
            self.usage["calls"] += 1
            self.usage["seconds"] += time.time() - t0
        return self._clean(_json_block(resp.get("text", "")).get("things"))


def _json_block(text):
    m = re.search(r"\{.*\}", str(text), re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except ValueError:
        return {}


# ─────────────────────────── 匹配 ───────────────────────────
def name_hits(phrase, names):
    """字面: 短语的每个词 (去尾 s) 都在名字里, 或名字的每个词都在短语里."""
    pw = [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"[a-z0-9]+", phrase)]
    out = []
    for nm in names:
        nw = [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"[a-z0-9]+", nm)]
        if pw and nw and (all(w in nw for w in pw) or all(w in pw for w in nw)):
            out.append(nm)
    return out


def relevancy(text_vec, node_vec):
    """CLIP 相关度 0-1: 和四个空泛词 (object / things / stuff / texture) 比, 比它们都更像才高 (agent.clip_channel 的标定)."""
    clip_channel = _clip()
    return float(clip_channel.calibrate_sim(np.asarray(text_vec, np.float32), np.asarray(node_vec, np.float32)[None]))


def match(graph, things, feats_fn, thresh=config.CLIP_MATCH_T, margin=config.CLIP_MARGIN, use_clip=True):
    """things: [{"name","aliases","desc"}]; feats_fn(nid) -> 节点向量或 None.
    第一通道: 名字或别名字面对上 (分 1.0, how=name). 第二通道: 字面没对上的短语, 按 CLIP 相关度找最像的一个节点,
    过线且比第二名高出 margin 才算 (how=clip). 返回 [(nid, score, phrase, how)] 按分降序; 相似块不参与."""
    nodes = graph.visible_nodes()
    if not nodes or not things:
        return []
    best = {}
    unmatched = []
    for t in things:
        cands = [t["name"]] + list(t.get("aliases") or [])
        hit = False
        for n in nodes:
            for c in cands:
                if name_hits(c, n["names"]):
                    hit = True
                    if n["id"] not in best or best[n["id"]][0] < 1.0:
                        best[n["id"]] = (1.0, t["name"], "name")
                    break
        if not hit:
            unmatched += [p for p in ([t["name"]] + ([t["desc"]] if t.get("desc") else [])) if p and p not in ("thing", "object")]
    feats = {n["id"]: feats_fn(n["id"]) for n in nodes}
    if use_clip and unmatched and any(v is not None for v in feats.values()):
        try:
            T = text_features(unmatched)
        except Exception as e:                          # noqa: BLE001  CLIP 不可用就只靠名字
            print(f"[match] clip unavailable: {e}", flush=True)
            T = None
        if T is not None:
            for k, phrase in enumerate(unmatched):
                scored = sorted(((relevancy(T[k], v), nid) for nid, v in feats.items() if v is not None), reverse=True)
                if not scored or scored[0][0] < thresh:
                    continue
                if len(scored) > 1 and scored[0][0] - scored[1][0] < margin:
                    continue                            # 几个节点差不多像: 分不清是谁, 不猜
                s, nid = scored[0]
                if nid not in best:
                    best[nid] = (s, phrase, "clip")
    return sorted([(nid, s, p, h) for nid, (s, p, h) in best.items()], key=lambda t: (-t[1], t[0]))
