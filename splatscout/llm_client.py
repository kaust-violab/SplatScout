"""Responses 接口客户端 (施工方案 §2). 实测结论: sol 在 chat/completions 上不能 工具+思考 并用;
Responses 可以; 图片必须放在 user 消息里 (放进工具返回会被静默丢弃).

历史用"条目"保存, 每轮重建成 input. v3: 一题之内图全留 (KEEP_IMAGE_GROUPS=0); 只有图超过 IMAGE_HARD_CAP 才略去最早的组,
替换文字优先用模型自己写的 seen (工具摘要兜底). build_input(suffix=) 把末尾简化图临时接在最后, 不进历史.
"""
import base64
import json
import time

import requests

from . import config
from . import legacy


def _backoff(status, attempt):
    """限流 (429) 是按分钟算的配额, 要等得久一点再试; 5xx 短等即可. 09-14: 百炼 30B 跑 refbench 时 16 步撞 429, 原来只等 3/6 s 两次就放弃, 整题作废."""
    if status == 429:
        return min(90.0, 20.0 * (attempt + 1))
    return 3.0 * (attempt + 1)


class LLMError(RuntimeError):
    pass


class History:
    def __init__(self):
        self.entries = []

    def user_text(self, text):
        self.entries.append({"kind": "user_text", "text": text})

    def model_items(self, items):
        self.entries.append({"kind": "model_items", "items": items})

    def tool_output(self, call_id, text):
        self.entries.append({"kind": "tool_output", "call_id": call_id, "output": text})

    def images(self, step, images, summary):
        """images: [(jpeg bytes, caption)]."""
        self.entries.append({"kind": "images", "step": step, "images": images, "summary": summary, "seen": None})

    def user_images(self, text, images):
        """开题消息: 文字 + 图 (永远保留, 不受略去规则约束)."""
        self.entries.append({"kind": "user_images", "text": text, "images": images})

    def set_seen(self, step, text):
        """模型对第 step 步那组图写的 seen, 图被略去时用它当替换文字."""
        for e in self.entries:
            if e["kind"] == "images" and e["step"] == step:
                e["seen"] = text

    def last_image_step(self):
        for e in reversed(self.entries):
            if e["kind"] == "images":
                return e["step"]
        return None

    def n_images(self):
        return sum(len(e["images"]) for e in self.entries if e["kind"] in ("images", "user_images"))

    def build_input(self, keep_groups=config.KEEP_IMAGE_GROUPS, hard_cap=config.IMAGE_HARD_CAP, suffix=None):
        img_idx = [i for i, e in enumerate(self.entries) if e["kind"] == "images"]
        keep = set(img_idx[-keep_groups:]) if keep_groups > 0 else set(img_idx)
        fixed = sum(len(e["images"]) for e in self.entries if e["kind"] == "user_images")
        total = fixed + sum(len(self.entries[i]["images"]) for i in keep)
        for i in img_idx:                                    # 超硬上限: 从最早的组开始略去
            if total <= hard_cap:
                break
            if i in keep:
                keep.discard(i)
                total -= len(self.entries[i]["images"])
        out = []
        for i, e in enumerate(self.entries):
            if e["kind"] == "user_text":
                out.append({"role": "user", "content": e["text"]})
            elif e["kind"] == "user_images":
                content = [{"type": "input_text", "text": e["text"]}]
                for jpg, cap in e["images"]:
                    content.append({"type": "input_text", "text": f"[{cap}]"})
                    content.append({"type": "input_image", "detail": config.IMAGE_DETAIL,
                                    "image_url": "data:image/jpeg;base64," + base64.b64encode(jpg).decode()})
                out.append({"role": "user", "content": content})
            elif e["kind"] == "model_items":
                out.extend(e["items"])
            elif e["kind"] == "tool_output":
                out.append({"type": "function_call_output", "call_id": e["call_id"], "output": e["output"]})
            elif e["kind"] == "images":
                if i in keep:
                    content = [{"type": "input_text", "text": f"[images from step {e['step']}] "
                                + "; ".join(c for _, c in e["images"])}]
                    for jpg, _ in e["images"]:
                        content.append({"type": "input_image", "detail": config.IMAGE_DETAIL,
                                        "image_url": "data:image/jpeg;base64," + base64.b64encode(jpg).decode()})
                    out.append({"role": "user", "content": content})
                else:
                    gist = e.get("seen") or e["summary"]
                    out.append({"role": "user", "content": f"[images from step {e['step']} omitted] {gist}"})
        if suffix:
            out.append({"role": "user", "content": suffix})
        return out

    def loggable(self):
        """落盘版: 图片只留字节数."""
        out = []
        for e in self.entries:
            if e["kind"] == "images":
                out.append({"kind": "images", "step": e["step"], "summary": e["summary"], "seen": e.get("seen"),
                            "images": [{"caption": c, "bytes": len(j)} for j, c in e["images"]]})
            elif e["kind"] == "user_images":
                out.append({"kind": "user_images", "text": e["text"],
                            "images": [{"caption": c, "bytes": len(j)} for j, c in e["images"]]})
            else:
                out.append(e)
        return out


class ResponsesClient:
    def __init__(self, model=None, effort=config.REASONING_EFFORT, log_path=None):
        cfg = legacy.llm_config()
        self.url = cfg.LLM_URL.rsplit("/chat/completions", 1)[0] + "/responses"
        self.key = cfg.LLM_API_KEY
        self.model = model or config.MODEL or cfg.LLM_MODEL
        self.effort = effort
        self.log_path = log_path
        self.want_summary = bool(config.REASONING_SUMMARY)
        self.usage = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "calls": 0, "seconds": 0.0}
        if not self.key:
            raise LLMError("OPENAI_API_KEY is not set")

    def step(self, instructions, history, tools, max_retries=config.LLM_MAX_RETRIES, suffix=None, keep_groups=config.KEEP_IMAGE_GROUPS):
        body = {"model": self.model, "instructions": instructions,
                "input": history.build_input(keep_groups=keep_groups, suffix=suffix),
                "tools": tools, "tool_choice": "auto", "parallel_tool_calls": False}
        if self.effort and self.effort != "none":
            body["reasoning"] = {"effort": self.effort}
            if self.want_summary:                      # 要思考摘要 (可视化 "想了啥"); 端点不支持就自动放弃
                body["reasoning"]["summary"] = config.REASONING_SUMMARY
        last = None
        for attempt in range(max_retries + 1):
            t0 = time.time()
            try:
                r = requests.post(self.url, headers={"Authorization": f"Bearer {self.key}",
                                                     "Content-Type": "application/json"},
                                  json=body, timeout=config.REQUEST_TIMEOUT)
                j = r.json()
            except Exception as e:                       # noqa: BLE001
                last = f"request failed: {e}"
                time.sleep(2 * (attempt + 1))
                continue
            dt = time.time() - t0
            if r.status_code >= 400 or j.get("error"):
                last = f"HTTP {r.status_code}: {str(j.get('error'))[:400]}"
                if self.want_summary and "summary" in last.lower() and r.status_code == 400:
                    self.want_summary = False
                    body["reasoning"].pop("summary", None)
                    continue
                if r.status_code in (429, 500, 502, 503, 504) or ("InternalError" in last or "timed out" in last.lower()):   # 服务端瞬时错 (百炼有时以 400 InternalError 返回) 也重试
                    time.sleep(_backoff(r.status_code, attempt) if r.status_code in (429, 500, 502, 503, 504) else 5.0 * (attempt + 1))
                    continue
                raise LLMError(last)
            u = j.get("usage", {})
            self.usage["input_tokens"] += u.get("input_tokens", 0)
            self.usage["output_tokens"] += u.get("output_tokens", 0)
            self.usage["reasoning_tokens"] += (u.get("output_tokens_details") or {}).get("reasoning_tokens", 0)
            self.usage["calls"] += 1
            self.usage["seconds"] += dt
            items = j.get("output", [])
            calls, text, thoughts = [], [], []
            for it in items:
                if it.get("type") == "reasoning":
                    for part in it.get("summary") or []:
                        if part.get("text"):
                            thoughts.append(part["text"])
                if it.get("type") == "function_call":
                    try:
                        args = json.loads(it.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        args = {"_raw": it.get("arguments")}
                    calls.append((it["call_id"], it["name"], args))
                elif it.get("type") == "message":
                    for c in it.get("content", []):
                        if c.get("type") == "output_text":
                            text.append(c["text"])
            self._log({"t": time.time(), "dt": dt, "usage": u, "n_input_items": len(body["input"]),
                       "output": [{k: v for k, v in it.items() if k not in ("content", "encrypted_content")} | (
                           {"text": "".join(c.get("text", "") for c in it.get("content", []))}
                           if it.get("type") == "message" else {}) for it in items]})
            return {"items": items, "calls": calls, "text": "\n".join(text), "thoughts": thoughts,
                    "usage": u, "seconds": dt}
        raise LLMError(last or "unknown error")

    def _log(self, rec):
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _collect_stream(r):
    """把 chat/completions 的 SSE 流拼回一个非流式的回复: content / reasoning_content 逐段接, tool_calls 按 index 接参数, usage 取最后一块."""
    content, reasoning, calls, usage, finish = [], [], {}, None, None
    for raw in r.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith("data:"):
            continue
        data = raw[5:].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        if chunk.get("error"):
            return chunk
        if chunk.get("usage"):
            usage = chunk["usage"]
        for ch in chunk.get("choices") or []:
            d = ch.get("delta") or {}
            if d.get("content"):
                content.append(d["content"])
            if d.get("reasoning_content"):
                reasoning.append(d["reasoning_content"])
            for tc in d.get("tool_calls") or []:
                k = tc.get("index", 0)
                slot = calls.setdefault(k, {"id": None, "name": "", "arguments": ""})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] += fn["name"]
                if fn.get("arguments"):
                    slot["arguments"] += fn["arguments"]
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
    msg = {"role": "assistant", "content": "".join(content), "reasoning_content": "".join(reasoning)}
    if calls:
        msg["tool_calls"] = [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                             for _, c in sorted(calls.items())]
    return {"choices": [{"message": msg, "finish_reason": finish}], "usage": usage or {}}


class ChatClient:
    """chat/completions 后端 (本地 Qwen 服务 local_vlm_server.py, 或任何 OpenAI 兼容端点). 与 ResponsesClient 同一个 step() 契约:
    history 里存的 model_items 是 chat 格式的 assistant 消息, build_input 原样接回; 工具结果映射成 role=tool."""

    def __init__(self, model=None, base_url=None, key=None, log_path=None, max_tokens=None, effort=None):
        cfg = legacy.llm_config()
        base = (base_url or config.LLM_BASE_URL or cfg.LLM_URL.rsplit("/chat/completions", 1)[0]).rstrip("/")
        self.url = base + "/chat/completions"
        self.key = key or config.LLM_KEY or cfg.LLM_API_KEY or "EMPTY"
        self.model = model or config.MODEL or cfg.LLM_MODEL
        self.max_tokens = max_tokens or config.CHAT_MAX_TOKENS
        self.effort = effort
        self.log_path = log_path
        self.usage = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "calls": 0, "seconds": 0.0}

    @staticmethod
    def to_chat_tools(tools):
        return [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""), "parameters": t.get("parameters", {})}}
                for t in tools]

    @staticmethod
    def to_messages(instructions, items):
        msgs = [{"role": "system", "content": instructions}]
        for it in items:
            if it.get("type") == "function_call_output":
                msgs.append({"role": "tool", "tool_call_id": it["call_id"], "content": str(it["output"])})
            elif it.get("role") == "assistant":
                msgs.append(it)
            elif it.get("role") == "user":
                c = it["content"]
                if isinstance(c, str):
                    msgs.append({"role": "user", "content": c})
                else:
                    parts = []
                    for p in c:
                        if p.get("type") == "input_text":
                            parts.append({"type": "text", "text": p["text"]})
                        elif p.get("type") == "input_image":
                            parts.append({"type": "image_url", "image_url": {"url": p["image_url"], "detail": p.get("detail", "low")}})
                    msgs.append({"role": "user", "content": parts})
        return msgs

    def step(self, instructions, history, tools, max_retries=config.LLM_MAX_RETRIES, suffix=None, keep_groups=config.KEEP_IMAGE_GROUPS):
        msgs = self.to_messages(instructions, history.build_input(keep_groups=keep_groups, suffix=suffix))
        # 思考: LLM_THINKING=on|off 明确指定 (混合模型如 qwen3.5 用 enable_thinking 开关); 不给则按模型名 (*-thinking 的固定思考版).
        # 思考版只走流式, 思考正文在 reasoning_content, max_tokens 含思考正文所以给大.
        mode = (config.LLM_THINKING or "").lower()
        thinking = mode == "on" or (mode == "" and "thinking" in str(self.model).lower())
        body = {"model": self.model, "messages": msgs, "tools": self.to_chat_tools(tools), "tool_choice": "auto",
                "max_tokens": config.CHAT_MAX_TOKENS_THINKING if thinking else self.max_tokens}
        if mode in ("on", "off"):
            body["enable_thinking"] = mode == "on"
        if not str(self.model).lower().startswith("gpt"):
            if config.LLM_SAMPLING == "card":
                body.update({"temperature": 0.7, "top_p": 0.8, "presence_penalty": 1.5, "seed": 0})
            else:
                body["temperature"] = 0
        stream = thinking or config.LLM_STREAM
        if stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
        last = None
        for attempt in range(max_retries + 1):
            t0 = time.time()
            try:
                r = requests.post(self.url, headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
                                  json=body, timeout=config.REQUEST_TIMEOUT, stream=stream)
                if stream and r.status_code < 400:
                    j = _collect_stream(r)
                else:
                    j = r.json()
            except Exception as e:                       # noqa: BLE001
                last = f"request failed: {e}"
                time.sleep(2 * (attempt + 1))
                continue
            dt = time.time() - t0
            if r.status_code >= 400 or j.get("error"):
                last = f"HTTP {r.status_code}: {str(j.get('error'))[:400]}"
                if r.status_code in (429, 500, 502, 503, 504) or ("InternalError" in last or "timed out" in last.lower()):   # 服务端瞬时错 (百炼有时以 400 InternalError 返回) 也重试
                    time.sleep(_backoff(r.status_code, attempt) if r.status_code in (429, 500, 502, 503, 504) else 5.0 * (attempt + 1))
                    continue
                raise LLMError(last)
            msg = (j.get("choices") or [{}])[0].get("message") or {}
            calls = []
            for k, tc in enumerate(msg.get("tool_calls") or []):
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                    if not isinstance(args, dict):
                        args = {"_raw": args}
                except json.JSONDecodeError:
                    args = {"_raw": fn.get("arguments")}
                calls.append((tc.get("id") or f"call_{self.usage['calls']}_{k}", fn.get("name"), args))
            calls = calls[:1]                             # 一轮只执行一个调用 (小模型会连发一串同样的)
            u = j.get("usage") or {}
            rt = int(((u.get("completion_tokens_details") or {}).get("reasoning_tokens")) or 0)
            usage = {"input_tokens": int(u.get("prompt_tokens") or 0), "output_tokens": int(u.get("completion_tokens") or 0),
                     "output_tokens_details": {"reasoning_tokens": rt}}
            self.usage["input_tokens"] += usage["input_tokens"]
            self.usage["output_tokens"] += usage["output_tokens"]
            self.usage["reasoning_tokens"] += rt
            self.usage["calls"] += 1
            self.usage["seconds"] += dt
            thoughts = [str(msg["reasoning_content"])[:6000]] if msg.get("reasoning_content") else []   # 思考正文只进轨迹, 不回传
            item = {"role": "assistant", "content": msg.get("content") or ""}
            if msg.get("tool_calls"):
                item["tool_calls"] = [{"id": cid, "type": "function", "function": {"name": n, "arguments": json.dumps(a, ensure_ascii=False)}}
                                      for cid, n, a in calls]
            self._log({"t": time.time(), "dt": dt, "usage": u, "n_messages": len(msgs), "text": item["content"][:2000],
                       "calls": [(n, a) for _, n, a in calls], "reasoning_chars": len(msg.get("reasoning_content") or "")})
            return {"items": [item], "calls": calls, "text": item["content"], "thoughts": thoughts, "usage": usage, "seconds": dt}
        raise LLMError(last or "unknown error")

    def _log(self, rec):
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
