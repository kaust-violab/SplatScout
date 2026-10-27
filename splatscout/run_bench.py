"""Entry point: run SplatScout on one query or on the PartRef-ScanNet++ queries of one scene.

  # one query (demo)
  python -m splatscout.run_bench --scene 578511c8a9 --query "the pedestal fan standing beside the open lid of the big cardboard box"

  # all active queries of the scene (the paper setting: no memory, every query starts from an empty state)
  python -m splatscout.run_bench --scene 578511c8a9 --tag v1
  -> runs/<scene>/cold_given_v1/{answers.json, rows.csv, <qid>/...}

  # another query file in the same format (e.g. OpenTarget), a subset of qids, a different model / backend
  python -m splatscout.run_bench --scene 578511c8a9 --tag ot --queries my_queries.json
  python -m splatscout.run_bench --scene 578511c8a9 --tag v1 --qid q460182c99f
  AGENT_MODEL=... python -m splatscout.run_bench --scene 578511c8a9 --tag qwen --backend chat --base-url http://127.0.0.1:8008/v1

  # live viewer (viser) in the same process; without --query, type questions in the web page
  python -m splatscout.run_bench --scene 578511c8a9 --live --port 8080

Before the first run: benchmark/prep_scene.py, benchmark/train_2dgs.py, then python -m splatscout.scene_prep --scene <scene>.
Scoring: SCENE=<scene> GS2D=1 python benchmark/partref_score.py --scene <scene> --answers runs/<scene>/cold_given_v1/answers.json --tag v1
The memory / caching modes (--memory on, --mode session|prewarm, --memory-only) are kept for reference but are not used by the paper.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import time

from . import config
from . import legacy
from . import prompts
from .cache import SceneGraph
from .llm_client import ChatClient, ResponsesClient
from .loop import run_query
from .memory_agent import MemoryAgent, VlmClient
from .memory_match import QueryParser
from .sam3_runner import Sam3Runner
from .scene import Scene
from .tools import SessionState, Toolbox


def load_queries(scene, path=None):
    path = path or os.path.join(config.QUERIES_DIR, f"{scene}.json")
    with open(path) as f:
        doc = json.load(f)
    return [{"qid": q["qid"], "text": q["text"], "role": q.get("role")} for q in doc["queries"]
            if q.get("status") == "active"]


def order_queries(queries, order):
    """given / reverse / shuffle:<seed>. 返回新列表 (order_index 按执行顺序记)."""
    if order == "given":
        return list(queries)
    if order == "reverse":
        return list(reversed(queries))
    if order.startswith("shuffle"):
        seed = int(order.split(":", 1)[1]) if ":" in order else 0
        qs = list(queries)
        random.Random(seed).shuffle(qs)
        return qs
    raise SystemExit(f"bad --order {order}: use given | reverse | shuffle:<seed>")


def run_tag(mode, order, tag):
    o = order.replace(":", "")
    return f"{mode}_{o}" + (f"_{tag}" if tag else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--memory", default="on" if config.MEMORY else "off", choices=["on", "off"],
                    help="记忆与缓存总开关 (默认 off, 09-12 停用): off = 每题空图不落盘、开题只给句子、无 recall、不算 CLIP、记忆 agent 不起, --mode 强制 cold")
    ap.add_argument("--mode", default="session", choices=config.MODES)
    ap.add_argument("--order", default="given", help="given | reverse | shuffle:<seed>")
    ap.add_argument("--query", default=None, help="单题 demo: 直接给句子")
    ap.add_argument("--qid", nargs="*", default=None, help="只跑这些题号")
    ap.add_argument("--queries", default=None, help="another query file in the benchmark/data/queries format (e.g. OpenTarget queries)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="", help="跑次标签: 输出目录与记忆目录都带它")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-steps", type=int, default=config.MAX_STEPS)
    ap.add_argument("--effort", default=config.REASONING_EFFORT)
    ap.add_argument("--model", default=None)
    ap.add_argument("--backend", default=config.LLM_BACKEND, choices=["responses", "chat"], help="responses = OpenAI Responses (默认); chat = chat/completions (本地 Qwen 服务 / 兼容端点)")
    ap.add_argument("--base-url", default=None, help="chat 后端网址, 如 http://127.0.0.1:8008/v1")
    ap.add_argument("--open-images", type=int, default=config.OPEN_IMAGES, help="开题带几张描边帧 (0 关)")
    ap.add_argument("--keep-image-groups", type=int, default=config.KEEP_IMAGE_GROUPS,
                    help="上下文里保留最近几组工具图 (0 = 一题之内全留; 更早的组换成模型写的 seen)")
    ap.add_argument("--memory-only", action="store_true", help="只靠记忆基线: 工具只留 recall/nearby/relate/answer/report")
    ap.add_argument("--from", dest="from_mem", default=None, help="--memory-only 用: 第一遍的记忆目录 (只读复制)")
    ap.add_argument("--exact-memo", action="store_true", help="句子一字不差答过就直接复用答案, 不调模型 (部署用, 评测默认关)")
    ap.add_argument("--parser", default=config.QUERY_PARSER, choices=["dashscope", "main", "none"],
                    help="记忆 agent 关着或失败时的兜底: 句子 -> 物体短语 的小模型 (dashscope / main / none)")
    ap.add_argument("--memory-agent", default="on" if config.MEMORY_AGENT else "off", choices=["on", "off"],
                    help="常驻的记忆 agent (小 VLM): 题中跟看整理图, 换题交假设页; off = v3 的程序检索")
    ap.add_argument("--memory-agent-model", default=config.MEMORY_AGENT_MODEL)
    ap.add_argument("--no-watch", action="store_true", help="记忆 agent 只做换题交接, 不在题中跟看 (M2 口径)")
    ap.add_argument("--no-rough-lift", action="store_true", help="记忆 agent 不做相似块粗提")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--live", action="store_true", help="同进程起 viser, 边跑边看; 不给 --query 时在网页里提问")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--points", action="store_true", help="live: 场景用点云渲染 (大场景更轻)")
    args = ap.parse_args()
    use_memory = args.memory == "on"
    if not use_memory:
        if args.memory_only:
            raise SystemExit("--memory-only needs --memory on")
        args.mode, args.memory_agent = "cold", "off"

    legacy.bind_scene(args.scene)
    scene = Scene(args.scene)
    rtag = run_tag(args.mode, args.order, args.tag) + ("_memonly" if args.memory_only else "")
    mem_dir = config.mem_dir(args.scene, rtag)
    if args.memory_only:
        if not args.from_mem or not os.path.isdir(args.from_mem):
            raise SystemExit("--memory-only needs --from <memory dir of the first pass>")
        if os.path.exists(mem_dir):
            shutil.rmtree(mem_dir)
        shutil.copytree(args.from_mem, mem_dir)
    graph = SceneGraph(args.scene, args.mode, shape=(scene.H, scene.W), tag=rtag, root=None if use_memory else "")   # 记忆停用: 纯内存, 不落盘
    if args.mode == "prewarm" and not graph.insts:
        graph.prewarm_from_lift(scene.n_frames)
        print(f"[prewarm] {len(graph.insts)} instances from the legacy library", flush=True)
    sam3 = Sam3Runner()
    print(f"[sam3] warm-up (load weights + first inference) {sam3.warmup(scene.image_path(0)):.0f}s", flush=True)
    toolbox = Toolbox(scene, graph, sam3)
    toolbox.memory = use_memory
    client = ChatClient(model=args.model, base_url=args.base_url) if args.backend == "chat" else ResponsesClient(model=args.model, effort=args.effort)
    print(f"[llm] backend={args.backend} model={client.model} url={client.url}", flush=True)
    tools = prompts.MEMORY_ONLY if args.memory_only else None
    out_root = args.out or os.path.join(config.RUNS_ROOT, args.scene, rtag)
    os.makedirs(out_root, exist_ok=True)
    parser = QueryParser(backend=args.parser, cache_path=os.path.join(out_root, "query_parse.json"), main_client=client) if use_memory else None
    mem_agent = None
    if args.memory_agent == "on" and not args.memory_only:
        mem_agent = MemoryAgent(graph, scene, VlmClient(model=args.memory_agent_model), watch=not args.no_watch,
                                rough_lift=not args.no_rough_lift, log_path=os.path.join(out_root, "memory_agent.jsonl"))
        print(f"[memory-agent] on: model={args.memory_agent_model} watch={not args.no_watch} rough_lift={not args.no_rough_lift}", flush=True)
    print(f"[run] mode={args.mode} order={args.order} out={out_root} memory={mem_dir if use_memory else 'off'} "
          f"graph={graph.stats()}", flush=True)

    if config.ABLATION:
        print(f"[ablation] setting {config.ABLATION} (AGENT_ABLATION); prompts from splatscout/prompts_ablation.py", flush=True)

    def run_one(q, order_index):
        if args.mode == "cold":                       # 对照口径: 每题换空图 + 新会话状态
            graph.reset()
            toolbox.st = SessionState(graph)
        if config.ABLATION == "1":                    # 消融 (1) 看一次切一次: 两回合流程, 不走 run_query
            from .ablation import run_query_oneshot
            return run_query_oneshot(scene, graph, toolbox, client, q["text"], q["qid"], os.path.join(out_root, q["qid"]),
                                     order_index=order_index, role=q.get("role"), verbose=not args.quiet)
        return run_query(scene, graph, toolbox, client, q["text"], q["qid"], os.path.join(out_root, q["qid"]),
                         order_index=order_index, role=q.get("role"), max_steps=args.max_steps, verbose=not args.quiet,
                         watcher=watcher, tools=tools, open_images=args.open_images, keep_groups=args.keep_image_groups,
                         parser=parser, memory_agent=mem_agent, use_memory=use_memory)

    watcher = None
    if args.live:
        from .viser_view import Viewer

        def runner(text, mode):                       # 网页输入框提问 -> 后台跑一题 (与命令行同一套对象)
            qid = "demo_" + hashlib.md5(text.encode()).hexdigest()[:8]
            res = run_one({"qid": qid, "text": text, "role": "demo"}, len(graph.queries))
            ans_path = os.path.join(out_root, "answers.json")
            ans = json.load(open(ans_path)) if os.path.exists(ans_path) else {}
            ans[qid] = {"gaussians": res["final"].get("gaussians", []), "status": res["final"].get("status"),
                        "ids": res["final"].get("ids", res["final"].get("candidates", []))}
            json.dump(ans, open(ans_path, "w"))
        watcher = Viewer(args.scene, port=args.port, points=args.points, runner=runner)

    if args.query:
        qid = "demo_" + hashlib.md5(args.query.encode()).hexdigest()[:8]
        queries = [{"qid": qid, "text": args.query, "role": "demo"}]
    elif args.live:
        queries = []
    else:
        queries = order_queries(load_queries(args.scene, args.queries), args.order)
        if args.qid:
            queries = [q for q in queries if q["qid"] in set(args.qid)]
        if args.limit:
            queries = queries[:args.limit]

    ans_path = os.path.join(out_root, "answers.json")
    answers = json.load(open(ans_path)) if os.path.exists(ans_path) and not args.query else {}
    rows_path = os.path.join(out_root, "rows.csv")
    new_rows = not os.path.exists(rows_path)
    with open(rows_path, "a", newline="") as fcsv:
        wr = csv.writer(fcsv)
        if new_rows:
            wr.writerow(["qid", "role", "status", "ids", "steps", "seconds", "input_tokens", "output_tokens",
                         "coverage_seen", "frames_seen", "text", "order_index", "coverage_session", "n_recall",
                         "open_hit", "mem_answer", "mem_anchor", "exact"])
        for i, q in enumerate(queries):
            if q["qid"] in answers and not args.query:
                print(f"[skip] {q['qid']} 已有答案")
                continue
            t0 = time.time()
            exact = graph.exact_match(q["text"]) if args.exact_memo else None
            if exact is not None:                     # 查表, 不调模型
                fin = {"status": "answered", "ids": list(exact["ids"]),
                       "gaussians": sorted(set(int(v) for i in exact["ids"] for v in graph.members[i].tolist()))}
                res = {"steps": 0, "seconds": 0.0, "usage": {"input_tokens_this_query": 0, "output_tokens_this_query": 0},
                       "coverage_seen": 0.0, "coverage_session": scene.coverage(sorted(graph.looked_frames())),
                       "frames_seen": [], "final": fin,
                       "memory": {"n_recall": 0, "hits": [], "exact": exact["qid"], "mem_answer": True, "mem_anchor": False}}
                print(f"[memo] {q['qid']} same sentence as {exact['qid']} -> {fin['ids']}")
            else:
                res = run_one(q, i)
                fin = res["final"]
            answers[q["qid"]] = {"gaussians": fin.get("gaussians", []), "status": fin.get("status"),
                                 "ids": fin.get("ids", fin.get("candidates", []))}
            with open(ans_path, "w") as f:
                json.dump(answers, f)                     # 每题落盘, 崩了不丢
            m = res.get("memory") or {}                   # 记忆停用时 memory 是 None
            wr.writerow([q["qid"], q["role"], fin.get("status"), " ".join(f"#{x}" for x in answers[q["qid"]]["ids"]),
                         res["steps"], res["seconds"], res["usage"]["input_tokens_this_query"],
                         res["usage"]["output_tokens_this_query"], f"{res['coverage_seen']:.3f}",
                         len(res["frames_seen"]), q["text"], i, f"{res.get('coverage_session', 0):.3f}",
                         m.get("n_recall", 0), int(bool(m.get("hits"))), int(bool(m.get("mem_answer"))),
                         int(bool(m.get("mem_anchor"))), m.get("exact") or ""])
            fcsv.flush()
            print(f"[{i + 1}/{len(queries)}] {q['qid']} {fin.get('status')} {time.time() - t0:.0f}s  graph {graph.stats()}")
    try:
        from . import viewer
        viewer.render_index(out_root)
    except Exception as e:                               # noqa: BLE001
        print(f"[viewer] {e}")
    if mem_agent is not None:
        mem_agent.drain()
        print(f"memory agent: {mem_agent.summary()}")
        mem_agent.close()
    print(f"answers -> {ans_path}\nmemory -> {mem_dir if use_memory else 'off'}  {graph.stats()}\nusage total: {client.usage}\n"
          f"parser: {parser.usage if parser is not None else None}")
    if watcher is not None:
        print(f"[live] viser 继续服务于 http://localhost:{args.port}; 网页里可以继续提问, Ctrl+C 退出", flush=True)
        watcher.serve_forever()


if __name__ == "__main__":
    main()
