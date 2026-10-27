"""把一题的 trajectory.json 渲成一页静态 HTML: 每一步三栏 —— 看到了啥 / 想了啥 / 干了啥.

  看到: 这一步发给模型时上下文里还带着图的那几组 (缩略图, 点开放大), 已被换成一行摘要的组, 外壳注入的提示
  想到: 模型的思考摘要 (端点支持 reasoning.summary 时才有) + 模型直接说的话 + 思考 token 数
  干了: 调了哪件工具、参数、工具返回的全文、工具带回的图 (新图带 NEW 标)

用法: python -m splatscout.viewer <一题目录 | 场景口径目录>
  一题目录 (有 trajectory.json) -> 该题的 trajectory.html
  口径目录 (如 runs/f9f95681fd/cold) -> 每题都渲 + index.html 总表
旧轨迹 (没有 saw/thought 字段) 也能渲, 对应栏留空.
"""
import html
import json
import os
import sys

CSS = """
body{font-family:system-ui,-apple-system,sans-serif;max-width:1500px;margin:20px auto;padding:0 16px;color:#222;background:#f5f6f8}
h1{font-size:20px;margin:0 0 6px} .meta{color:#555;font-size:13px;margin-bottom:14px}
.roster{background:#fff;border:1px solid #ddd;padding:10px 14px;white-space:pre;font-family:ui-monospace,Menlo,monospace;font-size:12.5px;overflow-x:auto}
.step{background:#fff;border:1px solid #ddd;border-left:6px solid #0b579c;margin:16px 0;padding:0}
.step.final{border-left-color:#1f7a33} .step.err{border-left-color:#b3261e} .step.text{border-left-color:#888}
.head{display:flex;justify-content:space-between;align-items:baseline;padding:8px 14px;border-bottom:1px solid #eee;background:#fafbfc}
.head .n{font-weight:700;font-size:15px} .head .cost{color:#666;font-size:12px}
.cols{display:grid;grid-template-columns:1fr 1fr 1.4fr;gap:0}
.col{padding:10px 14px;border-right:1px solid #eee;min-width:0} .col:last-child{border-right:none}
.col h4{margin:0 0 8px;font-size:12px;letter-spacing:.06em;color:#555;text-transform:uppercase}
.imgs{display:flex;flex-wrap:wrap;gap:8px} .imgs figure{margin:0;width:170px}
.imgs img{width:100%;border:1px solid #ccc;cursor:zoom-in;background:#eee} figcaption{font-size:11px;color:#555;line-height:1.25}
.imgs.big figure{width:300px}
.tag{display:inline-block;font-size:10.5px;padding:1px 6px;border-radius:9px;background:#e8eef5;color:#0b579c;margin-right:4px}
.tag.new{background:#e2f3e6;color:#1f7a33} .tag.gone{background:#f1f1f1;color:#777}
.note{background:#fff6e0;border-left:3px solid #e0a800;padding:6px 8px;margin:6px 0;font-size:12.5px}
.think{white-space:pre-wrap;font-size:13px;line-height:1.45;background:#f6f3fb;border-left:3px solid #7e57c2;padding:8px}
.think.empty{color:#888;font-style:italic;background:none;border:none;padding:0}
.call{font-family:ui-monospace,Menlo,monospace;font-size:13.5px;font-weight:600;color:#0b579c;word-break:break-all}
.call.err{color:#b3261e}
.res{white-space:pre-wrap;font-family:ui-monospace,Menlo,monospace;font-size:12.3px;background:#f4f6f8;padding:8px;margin:8px 0;overflow-x:auto;max-height:340px;overflow-y:auto}
.final-box{background:#eaf3ec;border:1px solid #b7d8bf;padding:10px 14px;margin-top:16px;white-space:pre-wrap;font-family:ui-monospace,Menlo,monospace;font-size:13px}
.small{font-size:11.5px;color:#666}
#lb{position:fixed;inset:0;background:rgba(0,0,0,.85);display:none;align-items:center;justify-content:center;z-index:9;cursor:zoom-out}
#lb img{max-width:96vw;max-height:92vh;border:2px solid #fff}
#lb div{position:absolute;bottom:12px;left:0;right:0;text-align:center;color:#eee;font-size:13px}
"""

JS = """
const lb=document.getElementById('lb'),lbi=lb.querySelector('img'),lbc=lb.querySelector('div');
document.querySelectorAll('.imgs img').forEach(im=>im.onclick=()=>{lbi.src=im.src;lbc.textContent=im.alt;lb.style.display='flex';});
lb.onclick=()=>lb.style.display='none';
document.addEventListener('keydown',e=>{if(e.key==='Escape')lb.style.display='none';});
"""


def _fig(im, tag=None, esc=html.escape):
    t = f"<span class='tag {tag[1]}'>{tag[0]}</span>" if tag else ""
    return (f"<figure><img src='{esc(im['file'])}' alt='{esc(im['caption'])}' loading='lazy'>"
            f"<figcaption>{t}{esc(im['caption'])}</figcaption></figure>")


def render(out_dir):
    with open(os.path.join(out_dir, "trajectory.json")) as f:
        d = json.load(f)
    r, steps = d["result"], d["steps"]
    esc = html.escape
    imgs_by_step = {s["step"]: s.get("images", []) for s in steps if s.get("images")}
    fin = r["final"]
    parts = [f"<title>{esc(r['qid'])}</title><style>{CSS}</style>",
             f"<h1>&ldquo;{esc(r['text'])}&rdquo;</h1>",
             f"<div class='meta'>{esc(r['qid'])} · mode {esc(str(r.get('mode')))} · {r['steps']} steps · {r['seconds']} s · "
             f"input tokens {r['usage'].get('input_tokens_this_query')} · reasoning tokens {r['usage'].get('reasoning_tokens')} · "
             f"coverage seen {r.get('coverage_seen', 0):.0%} · frames seen {len(r.get('frames_seen', []))} · "
             f"<b>final: {esc(str(fin.get('status')))} {esc(' '.join('#' + str(i) for i in fin.get('ids', fin.get('candidates', []) or [])))}</b></div>",
             "<div class='small'>开题消息 (系统提示词见 prompts.py):</div>",
             f"<div class='roster'>{esc(d.get('roster', ''))}</div>"]
    opens = sorted(fn for fn in os.listdir(os.path.join(out_dir, "images")) if fn.startswith("open_")) \
        if os.path.isdir(os.path.join(out_dir, "images")) else []
    if opens:
        parts.append("<div class='imgs'>" + "".join(_fig({"file": "images/" + fn, "caption": f"开题描边帧 {fn}"}, ("opening", ""))
                                                   for fn in opens) + "</div>")
    mem = r.get("memory") or {}
    if mem:
        parts.append(f"<div class='small'>memory: opened frames {mem.get('opened')} · name hits {mem.get('hits')} · "
                     f"recall {mem.get('n_recall')} · answer from earlier query: {mem.get('mem_answer')} · "
                     f"anchor from earlier query: {mem.get('mem_anchor')} · graph {esc(str(mem.get('graph')))}</div>")

    for s in steps:
        if "error" in s and "tool" not in s:
            parts.append(f"<div class='step err'><div class='head'><span class='n'>LLM error</span></div>"
                         f"<div class='col'><div class='res'>{esc(s['error'])}</div></div></div>")
            continue
        is_text = "model_text" in s and "tool" not in s
        cls = "step text" if is_text else ("step final" if s["tool"] in ("answer", "report") else
                                            ("step err" if s.get("error") else "step"))
        cost = (f"tool {s.get('tool_seconds')} s · llm {s.get('llm_seconds', s.get('seconds'))} s · "
                f"in {s.get('input_tokens')} / out {s.get('output_tokens')} tokens")
        title = "model text (no tool call)" if is_text else f"{s['tool']}"
        parts.append(f"<div class='{cls}'><div class='head'><span class='n'>step {s['step']} · {esc(title)}</span>"
                     f"<span class='cost'>{esc(cost)}</span></div><div class='cols'>")

        # ── 看到 ──
        saw = s.get("saw") or {}
        parts.append("<div class='col'><h4>看到 · in context</h4>")
        if not saw:
            parts.append("<div class='small'>(旧轨迹, 未记录)</div>")
        else:
            for n in saw.get("notes", []):
                parts.append(f"<div class='note'>shell: {esc(n)}</div>")
            shown = saw.get("image_steps", [])
            if shown:
                parts.append("<div class='imgs'>")
                for st_ in shown:
                    for im in imgs_by_step.get(st_, []):
                        parts.append(_fig(im, (f"step {st_}", "")))
                parts.append("</div>")
            else:
                parts.append("<div class='small'>no images in context yet</div>")
            if saw.get("omitted_steps"):
                parts.append("<div class='small' style='margin-top:6px'>"
                             + "".join(f"<span class='tag gone'>step {x} images → one-line summary</span>" for x in saw["omitted_steps"])
                             + "</div>")
            if saw.get("graph"):
                parts.append(f"<details><summary class='small'>末尾简化图 [scene memory] ({len(saw['graph'])} chars)</summary>"
                             f"<div class='roster'>{esc(saw['graph'])}</div></details>")
        parts.append("</div>")

        # ── 想到 ──
        th = s.get("thought") or {}
        parts.append("<div class='col'><h4>想到 · reasoning</h4>")
        if th.get("reasoning"):
            parts.append("<div class='think'>" + esc("\n\n".join(th["reasoning"])) + "</div>")
        elif th:
            parts.append(f"<div class='think empty'>(no summary returned; {th.get('reasoning_tokens') or 0} reasoning tokens)</div>")
        else:
            parts.append("<div class='think empty'>(旧轨迹, 未记录)</div>")
        txt = th.get("text") or s.get("model_text")
        if txt:
            parts.append(f"<div class='res' style='background:#fff;border:1px dashed #bbb'>model says: {esc(txt)}</div>")
        parts.append("</div>")

        # ── 干了 ──
        parts.append("<div class='col'><h4>干了 · action</h4>")
        if is_text:
            parts.append("<div class='small'>(no tool call this step)</div>")
        else:
            args = json.dumps(s["args"], ensure_ascii=False)
            parts.append(f"<div class='call{' err' if s.get('error') else ''}'>{esc(s['tool'])}({esc(args)})</div>")
            if s.get("seen"):
                parts.append("<div class='note'>seen (写进记忆): " + esc("; ".join(
                    f"f{e['frame']}: {', '.join(e['names'])}" for e in s["seen"])) + "</div>")
            parts.append(f"<div class='res'>{esc(s['result'])}</div>")
            if s.get("images"):
                parts.append("<div class='imgs big'>" + "".join(_fig(im, ("NEW", "new")) for im in s["images"]) + "</div>")
        parts.append("</div></div></div>")

    fin2 = dict(fin)
    fin2.pop("gaussians", None)
    parts.append(f"<div class='final-box'>final: {esc(json.dumps(fin2, ensure_ascii=False, indent=1))}\n"
                 f"answer gaussians: {len(fin.get('gaussians', []))}</div>")
    parts.append(f"<div id='lb'><img><div></div></div><script>{JS}</script>")
    path = os.path.join(out_dir, "trajectory.html")
    with open(path, "w") as f:
        f.write("\n".join(parts))
    return path


def render_index(scene_dir):
    """口径目录下所有题的索引页."""
    rows = []
    for q in sorted(os.listdir(scene_dir)):
        p = os.path.join(scene_dir, q, "trajectory.json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            r = json.load(f)["result"]
        fin = r["final"]
        ids = " ".join(f"#{i}" for i in (fin.get("ids") or fin.get("candidates") or []))
        m = r.get("memory") or {}
        rows.append(f"<tr><td>{r.get('order_index', '')}</td><td><a href='{q}/trajectory.html'>{html.escape(q)}</a></td><td>{html.escape(r['text'])}</td>"
                    f"<td>{html.escape(str(fin.get('status')))} {html.escape(ids)}</td><td>{r['steps']}</td><td>{r['seconds']}</td>"
                    f"<td>{r['usage'].get('input_tokens_this_query')}</td><td>{r.get('coverage_seen', 0):.0%}</td>"
                    f"<td>{html.escape(' '.join(m.get('hits') or []))}</td><td>{m.get('n_recall', '')}</td>"
                    f"<td>{'yes' if m.get('mem_answer') else ''}</td></tr>")
    with open(os.path.join(scene_dir, "index.html"), "w") as f:
        f.write(f"<style>{CSS} table{{border-collapse:collapse;background:#fff}} td,th{{border:1px solid #ddd;padding:4px 8px;font-size:13px}}</style>"
                f"<h1>{html.escape(os.path.basename(os.path.dirname(scene_dir)))} / {html.escape(os.path.basename(scene_dir))}</h1>"
                f"<table><tr><th>#</th><th>qid</th><th>sentence</th><th>final</th><th>steps</th><th>s</th><th>in tokens</th><th>coverage</th>"
                f"<th>memory hits</th><th>recall</th><th>answer from memory</th></tr>"
                f"{''.join(rows)}</table>")
    return os.path.join(scene_dir, "index.html")


if __name__ == "__main__":
    target = sys.argv[1]
    if os.path.exists(os.path.join(target, "trajectory.json")):
        print(render(target))
    else:
        n = 0
        for q in os.listdir(target):
            if os.path.exists(os.path.join(target, q, "trajectory.json")):
                render(os.path.join(target, q))
                n += 1
        print(render_index(target), f"({n} queries)")
