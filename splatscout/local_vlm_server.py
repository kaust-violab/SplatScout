"""本地 Qwen2.5-VL (transformers) 的最小 OpenAI 兼容服务: POST /v1/chat/completions (messages + tools -> 一个回复, 可带 tool_calls).
给 exp3 agent 的 ChatClient 用 (LLM_BACKEND=chat, LLM_BASE_URL=http://127.0.0.1:8008/v1). 跑在 laga 环境 (有 transformers 5 + qwen_vl_utils).
Qwen2.5-VL 的对话模板不带工具, 这里按 Qwen2.5 的 Hermes 格式把函数签名写进 system, 模型用 <tool_call>{json}</tool_call> 调用, 工具结果以
<tool_response> 放在 user 轮里. 单线程, 贪心解码. 图片按 max_pixels 缩到 <= 640 个视觉 token.
用法: CUDA_VISIBLE_DEVICES=1 python -m splatscout.local_vlm_server --port 8008 [--model Qwen/Qwen2.5-VL-7B-Instruct] [--max-pixels 501760]"""
import argparse
import base64
import io
import json
import re
import sys
import time

# ~/.local 里的 torchaudio 2.7 与 torch 2.8 不配, transformers 一 import 就炸; Qwen2.5-VL 用不到音频, 用假模块顶掉
sys.modules["torchaudio"] = None          # find_spec 返回 None -> transformers 认为没装 torchaudio, 走可选依赖的分支
from http.server import BaseHTTPRequestHandler, HTTPServer

TOOLS_HEADER = ("\n\n# Tools\n\nYou may call one function to assist with the user query.\n\n"
                "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{sigs}\n</tools>\n\n"
                "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
                "<tool_call>\n{{\"name\": <function-name>, \"arguments\": <args-json-object>}}\n</tool_call>\n"
                "Call exactly one function per turn. Arguments must be a JSON object using the parameter names given.")
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*(?:</tool_call>|$)", re.S)


class Engine:
    def __init__(self, model_id, max_pixels, min_pixels, device="cuda:0"):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16, device_map=device, attn_implementation="sdpa")
        self.model.eval()
        self.device = device
        torch.cuda.set_per_process_memory_fraction(float(__import__("os").environ.get("VLM_MEM_FRACTION", "0.55")), 0)   # 与 agent 的 SAM3 / 提升同卡, 不许吃满 (上次涨到 39 GB 把 SAM3 挤 OOM)
        self.model_id = model_id

    @staticmethod
    def _img(url):
        from PIL import Image
        if url.startswith("data:"):
            url = url.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(url))).convert("RGB")

    def build(self, messages, tools):
        """OpenAI 消息 -> Qwen 消息 + PIL 列表 (按出现顺序)."""
        out, images = [], []
        system = ""
        for m in messages:
            role, content = m.get("role"), m.get("content")
            if role == "system":
                system += (content if isinstance(content, str) else "".join(c.get("text", "") for c in content))
                continue
            if role == "tool":
                out.append({"role": "user", "content": f"<tool_response>\n{content}\n</tool_response>"})
                continue
            if role == "assistant":
                text = content if isinstance(content, str) else "".join(c.get("text", "") for c in (content or []))
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function", tc)
                    args = fn.get("arguments", "{}")
                    if not isinstance(args, str):
                        args = json.dumps(args, ensure_ascii=False)
                    text = (text or "") + f"\n<tool_call>\n{json.dumps({'name': fn.get('name'), 'arguments': json.loads(args) if args.strip().startswith('{') else args}, ensure_ascii=False)}\n</tool_call>"
                out.append({"role": "assistant", "content": text or ""})
                continue
            if isinstance(content, str):
                out.append({"role": "user", "content": content})
                continue
            parts = []
            for c in content or []:
                t = c.get("type")
                if t in ("text", "input_text"):
                    parts.append({"type": "text", "text": c.get("text", "")})
                elif t in ("image_url", "input_image"):
                    u = c.get("image_url")
                    u = u.get("url") if isinstance(u, dict) else u
                    images.append(self._img(u))
                    parts.append({"type": "image"})
            out.append({"role": "user", "content": parts})
        if tools:
            sigs = "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)
            system += TOOLS_HEADER.format(sigs=sigs)
        if system:
            out.insert(0, {"role": "system", "content": system})
        return out, images

    def generate(self, messages, tools, max_new_tokens):
        torch = self.torch
        qmsgs, images = self.build(messages, tools)
        text = self.processor.apply_chat_template(qmsgs, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=images or None, return_tensors="pt").to(self.device)
        n_in = int(inputs["input_ids"].shape[1])
        with torch.inference_mode():
            gen = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                      stop_strings=["</tool_call>"], tokenizer=self.processor.tokenizer)   # 一次只要一个调用
        gen = gen[:, n_in:]
        raw = self.processor.batch_decode(gen, skip_special_tokens=True)[0]
        n_out = int(gen.shape[1])
        del inputs, gen
        torch.cuda.empty_cache()                              # 每次请求后把缓存还回去, 别让分配器越攒越大
        return raw, n_in, n_out, len(images)


def parse_reply(raw):
    calls = []
    for k, m in enumerate(TOOL_CALL_RE.finditer(raw)):
        try:
            j = json.loads(m.group(1))
            calls.append({"id": f"call_{int(time.time() * 1000) % 100000}_{k}", "type": "function",
                          "function": {"name": j.get("name"), "arguments": json.dumps(j.get("arguments", {}), ensure_ascii=False)}})
        except json.JSONDecodeError:
            continue
    text = TOOL_CALL_RE.sub("", raw).strip()
    return text, calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--port", type=int, default=8008)
    ap.add_argument("--max-pixels", type=int, default=640 * 28 * 28)
    ap.add_argument("--min-pixels", type=int, default=64 * 28 * 28)
    ap.add_argument("--max-new-tokens", type=int, default=700)
    args = ap.parse_args()
    t0 = time.time()
    eng = Engine(args.model, args.max_pixels, args.min_pixels)
    print(f"[server] {args.model} loaded in {time.time() - t0:.0f}s, port {args.port}", flush=True)
    stats = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            b = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            self._send(200, {"status": "ok", "model": args.model, **stats})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(n))
                t1 = time.time()
                raw, n_in, n_out, n_img = eng.generate(body.get("messages", []), body.get("tools") or [], int(body.get("max_tokens") or args.max_new_tokens))
                dt = time.time() - t1
                text, calls = parse_reply(raw)
                stats["calls"] += 1; stats["prompt_tokens"] += n_in; stats["completion_tokens"] += n_out; stats["seconds"] += dt
                print(f"[server] {n_in} in / {n_out} out tok, {n_img} img, {dt:.1f}s, calls={[c['function']['name'] for c in calls]}", flush=True)
                msg = {"role": "assistant", "content": text}
                if calls:
                    msg["tool_calls"] = calls
                self._send(200, {"id": f"chatcmpl-{int(t1)}", "object": "chat.completion", "model": args.model,
                                 "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if calls else "stop"}],
                                 "usage": {"prompt_tokens": n_in, "completion_tokens": n_out, "total_tokens": n_in + n_out}})
            except Exception as e:                        # noqa: BLE001
                import traceback
                traceback.print_exc()
                self._send(500, {"error": {"message": f"{type(e).__name__}: {e}"}})

    HTTPServer(("127.0.0.1", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
