"""循环外壳 (施工方案 v2 §3 + v3 §4, §9): 一题一个循环, 会话状态与场景图跨题活着.
预算 / 熔断 / 开题描边帧 / 末尾简化图 / seen 与 evidence 写图 / 答案固化 / 轨迹落盘 都在这里.

run_query() 不管口径 (cold 的换空图由 run_bench 在题前做).
"""
import json
import os
import re
import time

from . import config
from . import prompts
from . import render
from .llm_client import History, LLMError


def _notify(watcher, event, *a):
    if watcher is None:
        return
    try:
        getattr(watcher, event)(*a)
    except Exception as e:                               # noqa: BLE001  可视化出错不影响跑题
        print(f"[watcher] {event}: {type(e).__name__}: {e}", flush=True)


def _save_payload(out_dir, step, payload):
    """工具的结构化载荷落盘 (npy), 轨迹里只记文件名. 冷启动每题清库, 不存这里的话提升结果跑完就没了."""
    if not payload:
        return {}
    import numpy as np
    d = os.path.join(out_dir, "gaussians")
    os.makedirs(d, exist_ok=True)
    viz = {"frames": [int(f) for f in payload.get("frames", [])]}
    if payload.get("ref_frame") is not None:
        viz["ref_frame"] = int(payload["ref_frame"])
    if payload.get("box") is not None:                   # 模型画的框 (原图像素) 和它画在哪一帧
        viz["box"] = [int(v) for v in payload["box"]]
        viz["box_frame"] = int(payload.get("box_frame", viz.get("ref_frame", -1)))
    if payload.get("groups"):
        viz["groups"] = {}
        for nm, idx in payload["groups"].items():
            fn = f"step{step:02d}_{nm}.npy"
            np.save(os.path.join(d, fn), np.asarray(idx, np.int64))
            viz["groups"][nm] = {"file": "gaussians/" + fn, "n": int(len(idx))}
    if payload.get("lift"):
        L = payload["lift"]
        fn = f"step{step:02d}_inst{L['id']}.npy"
        np.save(os.path.join(d, fn), np.asarray(L["members"], np.int64))
        viz["lift"] = {"id": int(L["id"]), "word": L["word"], "grade": L["grade"], "file": "gaussians/" + fn,
                       "n": int(len(L["members"])), "source": L.get("source"), "groups": list(L.get("groups", [])),
                       "rounds": L.get("rounds"), "curve": L.get("curve"), "single_view": L.get("single_view")}   # 精档轮数 (内部迭代, 不算模型步数)
    if payload.get("boxed") is not None:
        viz["boxed"] = payload["boxed"]                  # 框中之组 (消融 (6) 里模型看不到, 只留在记录里)
    return viz


def run_query(scene, graph, toolbox, client, text, qid, out_dir, order_index=0, role=None, max_steps=config.MAX_STEPS,
              token_budget=config.TOKEN_BUDGET, verbose=True, watcher=None, tools=None, open_images=config.OPEN_IMAGES,
              keep_groups=config.KEEP_IMAGE_GROUPS, parser=None, memory_agent=None, use_memory=True, view_frame=None,
              view_mode="strict"):
    """一题. graph = cache.SceneGraph (跨题), toolbox.st = 会话状态 (跨题). tools=None 用全部工具; 传 prompts.MEMORY_ONLY 跑只靠记忆基线.
    watcher: 可选的实时观察者, 有 on_start(run_dir, text, qid, roster, mode) / on_step(step_record) / on_done(result).
    view_frame: 观察帧号 (Ref-LERF 口径: 句子按这帧的机位写). view_mode: strict = 附低清图只供读左右前后, 该帧不许切 / 放大 / 提升 /
    当补帧 (评分帧不能当分割输入, 不然是泄漏); pose = 只给帧号不给图; loose = 附图且不设限 (0912 那次的口径, 只留作对照)."""
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    client.log_path = os.path.join(out_dir, "llm_calls.jsonl")
    st = toolbox.st
    st.begin_query(qid, order_index, text)
    graph.begin_query(qid, order_index, text, role)
    hist = History()
    instructions = prompts.system_prompt(max_steps, memory=use_memory)
    all_tools = tools or (prompts.TOOLS if use_memory else prompts.TOOLS_NO_MEMORY)
    if config.ABLATION in ("2", "3", "4", "5", "6", "7", "8"):   # 消融 (09-22 / 09-25): 只换提示词与工具 schema, 循环本身不动
        from . import prompts_ablation
        instructions = prompts_ablation.system_prompt(config.ABLATION, max_steps)
        all_tools = tools or prompts_ablation.tools(config.ABLATION)

    # 开题: 记忆 agent 读句子在图里找, 交假设页 (候选 / 锚点 / 缺的 / 先看哪几帧) -> 文字 + 描边帧;
    # 它关着或失败 -> 程序检索 (名字 / 别名 / CLIP); 什么都没有 -> 房间总览
    t_open = time.time()
    page = None
    if not use_memory:                                   # 记忆停用 (09-12): 没有记忆 agent, 开题只有句子
        memory_agent = None
    if memory_agent is not None:
        memory_agent.begin_query(qid)
        try:
            page = memory_agent.handoff(text, st)
        except Exception as e:                           # noqa: BLE001  记忆 agent 出错不影响跑题
            print(f"[memory-agent] handoff failed: {type(e).__name__}: {e}", flush=True)
            page = None
    if not use_memory:
        op = {"lines": [], "images": [], "fids": [], "hits": [], "things": [], "how": None, "exact": None}
    elif page is not None:
        images, fids = render.handoff_images(graph, scene, st, page, n=open_images)
        op = {"lines": render.handoff_lines(graph, page), "images": images, "fids": fids,
              "hits": [c["id"] for c in page["candidates"]] + [a["id"] for a in page["anchors"]],
              "things": [{"name": m, "aliases": [], "desc": ""} for m in page.get("missing", [])], "how": "agent",
              "exact": None, "page": page}
        exact = graph.exact_match(text)
        if exact is not None:
            op["exact"] = exact["qid"]
            op["lines"].append(f"This exact sentence was answered before ({exact['qid']}) as "
                               + " ".join(f"#{i}" for i in exact["ids"]))
    else:
        op = render.opening(graph, scene, st, text, parser=parser, n=open_images)
    if view_frame is not None:                           # 观察帧: 一句话 (+ 低清图), 算本题看过 (explore / nearby 不再给它)
        if view_mode == "loose":
            op["lines"].append(f"Viewpoint: the sentence describes the scene as seen from frame f{view_frame} (attached). "
                               f"Judge left / right / in front / behind from this view; for relate() with those predicates pass view='f{view_frame}'.")
        elif view_mode == "pose":
            op["lines"].append(f"Viewpoint: the sentence is written from the camera of frame f{view_frame}. You are not shown that frame and "
                               f"cannot segment, zoom into or lift with it; for relate() with left/right/front/behind pass view='f{view_frame}', "
                               f"and look at the scene through other frames (nearby(frame='f{view_frame}') lists frames around it).")
            st.view_frame = view_frame
        else:                                            # strict
            op["lines"].append(f"Viewpoint: the sentence describes the scene as seen from frame f{view_frame} (attached, low-res, for reading "
                               f"the viewpoint only). Judge left / right / in front / behind from this view and pass view='f{view_frame}' to "
                               f"relate() for those predicates. This frame cannot be segmented, zoomed into or used for lifting: find and box "
                               f"things in other frames (nearby(frame='f{view_frame}') lists frames around it).")
            st.view_frame = view_frame
        if view_mode != "pose":
            op["images"] = list(op["images"]) + [(scene.thumb_bytes(view_frame), f"viewpoint frame f{view_frame}")]
            st.shown[view_frame] = (0, 0, scene.W, scene.H)
        op["fids"] = list(op["fids"]) + [view_frame]
        st.seen.add(view_frame)
        st.seen_scene.add(view_frame)
    if use_memory:
        graph.note_opening(op["fids"], op["hits"], op["exact"], things=[t["name"] for t in op["things"]], how=op["how"])
        graph.note_looked(op["fids"], "opening", 0)
    stats = graph.stats()
    if not use_memory:
        intro = ""
    elif stats["nodes"] == 0 and len(graph.queries) <= 1:
        intro = "Scene memory is empty: this is the first sentence in this scene."
    else:
        intro = (f"Scene memory holds {stats['nodes']} things ({stats['lifted']} lifted) from {len(graph.queries) - 1} "
                 f"earlier sentence(s).")
    open_text = (intro + ("\n" + "\n".join(op["lines"]) if op["lines"] else "")
                 + f"\nScene has {scene.n_frames} frames.\n\nSentence: \"{text}\"").lstrip("\n")
    if op["images"]:
        hist.user_images(open_text, op["images"])
        for i, (jpg, cap) in enumerate(op["images"]):
            with open(os.path.join(out_dir, "images", f"open_{i}.jpg"), "wb") as f:
                f.write(jpg)
    else:
        hist.user_text(open_text)
    roster = open_text
    traj, final = [], None
    t_start = time.time()
    tokens_at_start = client.usage["input_tokens"]
    out_tokens_at_start = client.usage["output_tokens"]
    last_key, no_progress, text_only, forced = None, 0, 0, False
    step = 0
    notes_from = 1

    def say(msg):
        if verbose:
            print(msg, flush=True)

    open_secs = round(time.time() - t_open, 2)                  # 开题耗时 (交接 + 描边帧), 到这里为止
    say(f"\n=== {qid}  \"{text}\"\n{roster}  (opening {open_secs:.1f}s)")
    _notify(watcher, "on_start", out_dir, text, qid, roster, graph.mode)
    while True:
        used = client.usage["input_tokens"] - tokens_at_start
        over = step >= max_steps or used >= token_budget
        tools_now = prompts.FINAL_ONLY if over else all_tools
        if over and not forced:
            hist.user_text("Budget exhausted. Call answer() if the evidence supports it, otherwise report(). No other tools now.")
            forced = True
        if forced and step >= max_steps + 2:            # 逼两次还不收尾 -> 外壳代报
            break
        block = render.ledger(graph, st, step, max_steps, used, token_budget)
        img_entries = [e for e in hist.entries if e["kind"] == "images"]
        kept = img_entries[-keep_groups:] if keep_groups > 0 else img_entries
        saw = {"image_steps": [e["step"] for e in kept], "omitted_steps": [e["step"] for e in img_entries if e not in kept],
               "notes": [e["text"] for e in hist.entries[notes_from:] if e["kind"] == "user_text"], "graph": block}
        try:
            resp = client.step(instructions, hist, tools_now, suffix=block, keep_groups=keep_groups)
        except LLMError as e:
            traj.append({"step": step, "error": str(e)})
            say(f"[llm error] {e}")
            break
        hist.model_items(resp["items"])
        step += 1
        st.step = step
        thought = {"reasoning": resp.get("thoughts", []), "text": resp["text"],
                   "reasoning_tokens": (resp["usage"].get("output_tokens_details") or {}).get("reasoning_tokens")}
        notes_from = len(hist.entries)
        if not resp["calls"]:
            text_only += 1
            traj.append({"step": step, "model_text": resp["text"], "seconds": resp["seconds"],
                         "saw": saw, "thought": thought})
            _notify(watcher, "on_step", traj[-1])
            say(f"[{step}] model: {resp['text'][:300]}")
            hist.user_text("Continue with a tool call. Finish with answer() or report()." if text_only < 2 else
                           "You must call answer() or report() now.")
            if text_only >= 3:
                break
            continue
        text_only = 0
        for call_id, name, args in resp["calls"]:
            args = dict(args or {})
            args.pop("seen", None)                        # 旧提示词的遗留参数, 不再收
            t0 = time.time()
            if memory_agent is not None:
                memory_agent.main_busy = True
            try:
                res = toolbox.dispatch(name, args, st)
            finally:
                if memory_agent is not None:
                    memory_agent.main_busy = False
            dt = time.time() - t0
            if memory_agent is not None and name == "segment" and res.payload.get("groups") and not res.error:
                m = re.search(r"\((s\d+) = boxed\)", res.summary or "")
                memory_agent.notify("segment", {"word": str(args.get("word", "")), "names": list(res.payload["groups"]),
                                                "boxed": m.group(1) if m else None, "qid": qid})
            hist.tool_output(call_id, res.text)
            img_files = []
            for i, (jpg, cap) in enumerate(res.images[:config.MAX_IMAGES_PER_TOOL]):
                fn = f"step{step:02d}_{i}.jpg"
                with open(os.path.join(out_dir, "images", fn), "wb") as f:
                    f.write(jpg)
                img_files.append({"file": "images/" + fn, "caption": cap})
            if res.images:
                hist.images(step, res.images[:config.MAX_IMAGES_PER_TOOL], res.summary or res.text.splitlines()[0])
            viz = _save_payload(out_dir, step, res.payload)
            traj.append({"step": step, "tool": name, "args": args, "result": res.text, "images": img_files,
                         "saw": saw, "thought": thought, "error": res.error, "viz": viz,
                         "summary": res.summary, "progress": res.progress, "tool_seconds": round(dt, 2),
                         "llm_seconds": round(resp["seconds"], 2),
                         "input_tokens": resp["usage"].get("input_tokens"),
                         "output_tokens": resp["usage"].get("output_tokens")})
            _notify(watcher, "on_step", traj[-1])
            if thought["reasoning"]:
                say("    (thinking) " + " | ".join(thought["reasoning"])[:600])
            say(f"[{step}] {name}({json.dumps(args, ensure_ascii=False)[:300]})  {dt:.1f}s\n    "
                + res.text.replace("\n", "\n    ")[:900])
            if res.final:
                final = res.final
                if memory_agent is not None:
                    memory_agent.notify("answer", {"page": op.get("page"), "ids": final.get("ids", []),
                                                   "status": final.get("status"), "text": text})
                try:
                    graph.finish_query(step, final, final.get("triples"), dispute=final.get("dispute"))
                except Exception as e:                   # noqa: BLE001  写图出错不该带崩已经答完的题; 打出来修
                    import traceback
                    print(f"[memory] finish_query failed: {type(e).__name__}: {e}\n{traceback.format_exc()}", flush=True)
                break
            key = (name, json.dumps(args, sort_keys=True))
            if key == last_key:
                hist.user_text("That is the same call with the same arguments as before; it brings nothing new. "
                               "Decide with the current evidence or try a different step.")
            last_key = key
            if res.error:
                pass
            elif res.progress:
                no_progress = 0
            else:
                no_progress += 1
            if no_progress >= config.NO_PROGRESS_STEPS:
                zoom = ("box the thing tightly on a frame you have seen" if config.ABLATION == "5"
                        else "zoom with nearby(frame=, box=) and box the thing tightly")     # 消融 (5) 没有放大
                hist.user_text(f"The last {no_progress} steps brought no new information. Change approach "
                               f"({zoom}, nearby(group=) on what "
                               f"you found, recall(frame=) on a memory frame, explore elsewhere), or if the evidence "
                               f"is already sufficient, answer()/report().")
                no_progress = 0
        if final:
            break

    if final is None:
        final = {"status": "not_found", "detail": "shell: budget exhausted without answer/report",
                 "gaussians": [], "forced": True}
        graph.finish_query(step, final, [])
    if memory_agent is not None:
        drained = memory_agent.drain()
        if not drained:
            say("[memory-agent] background work still running at query end (continues in the next query)")
    usage = dict(client.usage)
    usage["input_tokens_this_query"] = client.usage["input_tokens"] - tokens_at_start
    usage["output_tokens_this_query"] = client.usage["output_tokens"] - out_tokens_at_start
    seconds = round(time.time() - t_start, 1)
    graph.end_query(step, seconds, usage["input_tokens_this_query"], usage["output_tokens_this_query"], final.get("status"))
    ids = [int(i) for i in final.get("ids", [])]
    anchors = [iid for iid, rec in graph.insts.items()
               if any(c.get("q") == qid and c.get("kind") == "anchor" for c in rec.get("claims", []))]
    memory = None if not use_memory else {"opened": op["fids"], "hits": op["hits"], "things": op["things"], "parse": op["how"], "exact": op["exact"],
              "n_recall": st.n_recall, "open_seconds": open_secs,
              "page": op.get("page"), "agent": memory_agent.summary() if memory_agent is not None else None,
              "mem_answer": bool(ids) and all(graph.insts[i].get("q") != qid for i in ids if i in graph.insts),
              "mem_anchor": bool(anchors) and any(graph.insts[i].get("q") != qid for i in anchors),
              "graph": graph.stats()}
    result = {"qid": qid, "scene": scene.name, "text": text, "final": final, "steps": step, "seconds": seconds,
              "order_index": order_index, "frames_seen": sorted(st.seen),
              "coverage_seen": scene.coverage(sorted(st.seen)), "coverage_session": scene.coverage(sorted(st.seen_scene)),
              "lifted": st.lifted, "usage": usage, "mode": graph.mode, "memory": memory}
    with open(os.path.join(out_dir, "trajectory.json"), "w") as f:
        json.dump({"result": result, "roster": roster, "steps": traj}, f, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(hist.loggable(), f, ensure_ascii=False)
    try:
        from . import viewer
        viewer.render(out_dir)
    except Exception as e:                               # noqa: BLE001  轨迹页失败不影响结果
        say(f"[viewer] {e}")
    _notify(watcher, "on_done", result)
    say(f"=== done: {final.get('status')} ids={final.get('ids', final.get('candidates'))} "
        f"steps={step} {seconds}s tokens_in={usage['input_tokens_this_query']} "
        f"memory: opened {op['fids']} hits {op['hits']} recall {st.n_recall}")
    return result
