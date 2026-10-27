"""汇总一次多场景评测 (run_all.sh 的最后一步): 逐题 / 逐场景的时间、token、三维分.

用法: python -m splatscout.summarize --tag 0907 [--mode session --order given] [--price-in 2.5 --price-out 15]
读: runs/<scene>/<mode>_<order>_<tag>/rows.csv (老跑次没有 order 段: runs/<scene>/<mode>_<tag>/) (run_bench 逐题一行), 每题目录的 trajectory.json (每步 LLM 秒 / 工具秒) 与
    llm_calls.jsonl (思考 token), refbench/eval/score3d/agent_<mode>_<tag>.json (score3d 产物, 没有就留空),
    runs/logs/<tag>/timeline.csv (run_all.sh 各阶段墙钟, 没有就留空)
写: runs/summary_<mode>_<tag>/per_query.csv 与 per_scene.csv; 终端打两张表.
"""
import argparse
import csv
import json
import os

from . import config

Q_FIELDS = ["scene", "qid", "order_index", "role", "status", "ids", "iou3d", "steps", "wall_s", "llm_s", "tool_s", "segment_s", "lift_s",
            "n_explore", "n_nearby", "n_segment", "n_same", "n_lift", "n_relate", "n_recall", "open_hit", "mem_answer",
            "mem_anchor", "exact", "in_tok", "out_tok", "reason_tok", "coverage", "coverage_session", "frames", "forced", "text"]
S_FIELDS = ["scene", "n", "answered", "ambiguous", "not_found", "forced", "n_scored", "miou3d", "acc25", "acc50",
            "wall_s", "mean_wall_s", "llm_s", "tool_s", "segment_s", "lift_s", "mean_steps", "in_tok", "out_tok",
            "reason_tok", "mean_in_tok", "n_recall", "open_hit", "mem_answer", "mem_anchor",
            "third1_steps", "third2_steps", "third3_steps", "third1_s", "third2_s", "third3_s",
            "third1_miou", "third2_miou", "third3_miou", "prep_wall_s", "bench_wall_s", "score_wall_s"]


def query_detail(qdir):
    """一题目录 -> LLM 秒 / 工具秒 (segment, lift 单列) / 思考 token / 各工具调用次数 / 是否外壳代报."""
    det = {"llm_s": 0.0, "tool_s": 0.0, "segment_s": 0.0, "lift_s": 0.0, "reason_tok": 0, "forced": False,
           "n_explore": 0, "n_nearby": 0, "n_segment": 0, "n_same": 0, "n_lift": 0, "n_relate": 0, "n_recall": 0,
           "open_hit": None, "mem_answer": None, "mem_anchor": None, "exact": None, "order_index": None,
           "coverage_session": None}
    tp = os.path.join(qdir, "trajectory.json")
    if os.path.exists(tp):
        with open(tp) as f:
            d = json.load(f)
        res = d.get("result", {})
        det["forced"] = bool((res.get("final") or {}).get("forced"))
        m = res.get("memory") or {}
        if m:
            det.update({"open_hit": int(bool(m.get("hits"))), "mem_answer": int(bool(m.get("mem_answer"))),
                        "mem_anchor": int(bool(m.get("mem_anchor"))), "exact": m.get("exact")})
        det["order_index"] = res.get("order_index")
        det["coverage_session"] = res.get("coverage_session")
        for s in d.get("steps", []):
            det["llm_s"] += float(s.get("llm_seconds") or s.get("seconds") or 0)
            ts = float(s.get("tool_seconds") or 0)
            det["tool_s"] += ts
            t = s.get("tool")
            if t in ("segment", "lift"):
                det[f"{t}_s"] += ts
            if f"n_{t}" in det:
                det[f"n_{t}"] += 1
    lp = os.path.join(qdir, "llm_calls.jsonl")
    if os.path.exists(lp):
        with open(lp) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                det["reason_tok"] += int(((rec.get("usage") or {}).get("output_tokens_details") or {})
                                         .get("reasoning_tokens") or 0)
    return det


def load_scores(path):
    """score3d 的 tag 文件 -> {scene: {qid: iou}}, {scene: 宏指标}."""
    if not path or not os.path.exists(path):
        return {}, {}
    with open(path) as f:
        doc = json.load(f)
    per_q, macro = {}, {}
    for scene, rep in doc.items():
        per_q[scene] = {r["qid"]: r.get("iou3d") for r in rep.get("rows", [])}
        macro[scene] = {"n_scored": rep.get("n"), "miou3d": rep.get("miou3d"),
                        "acc25": rep.get("acc@0.25"), "acc50": rep.get("acc@0.5")}
    return per_q, macro


def load_timeline(path):
    """run_all.sh 的 timeline.csv -> {(stage, scene): seconds} (同一阶段跑了几次就累加)."""
    out = {}
    if not path or not os.path.exists(path):
        return out
    with open(path) as f:
        for r in csv.DictReader(f):
            k = (r["stage"], r["scene"])
            out[k] = out.get(k, 0.0) + float(r["seconds"] or 0)
    return out


def run_suffix(mode, tag, order=None):
    o = (order or "").replace(":", "")
    return f"{mode}_{o}_{tag}" if o else f"{mode}_{tag}"


def collect(runs_root, mode, tag, per_q_score, order=None):
    """所有场景的逐题记录 (rows.csv 去重, 同 qid 取最后一行) + 每题目录里的细账. 目录 <mode>_<order>_<tag>, 找不到退到 <mode>_<tag>."""
    rows = []
    for scene in sorted(os.listdir(runs_root)):
        d = os.path.join(runs_root, scene, run_suffix(mode, tag, order))
        if not os.path.exists(os.path.join(d, "rows.csv")):
            d = os.path.join(runs_root, scene, run_suffix(mode, tag))
        rp = os.path.join(d, "rows.csv")
        if not os.path.exists(rp):
            continue
        last = {}
        with open(rp, newline="") as f:
            for r in csv.DictReader(f):
                last[r["qid"]] = r
        for qid, r in last.items():
            det = query_detail(os.path.join(d, qid))
            iou = per_q_score.get(scene, {}).get(qid)
            oi = r.get("order_index")
            rows.append({"scene": scene, "qid": qid, "role": r.get("role"), "status": r.get("status"), "ids": r.get("ids"),
                         "order_index": int(float(oi)) if oi not in (None, "") else det["order_index"],
                         "n_recall": int(float(r.get("n_recall") or det["n_recall"] or 0)),
                         "open_hit": _int_or(r.get("open_hit"), det["open_hit"]),
                         "mem_answer": _int_or(r.get("mem_answer"), det["mem_answer"]),
                         "mem_anchor": _int_or(r.get("mem_anchor"), det["mem_anchor"]),
                         "exact": r.get("exact") or det["exact"] or "",
                         "coverage_session": r.get("coverage_session") or det["coverage_session"],
                         "iou3d": None if iou is None else round(float(iou), 4), "steps": int(float(r.get("steps") or 0)),
                         "wall_s": float(r.get("seconds") or 0), "llm_s": round(det["llm_s"], 1),
                         "tool_s": round(det["tool_s"], 1), "segment_s": round(det["segment_s"], 1),
                         "lift_s": round(det["lift_s"], 1), "n_explore": det["n_explore"], "n_nearby": det["n_nearby"],
                         "n_segment": det["n_segment"], "n_same": det["n_same"], "n_lift": det["n_lift"],
                         "n_relate": det["n_relate"], "in_tok": int(float(r.get("input_tokens") or 0)),
                         "out_tok": int(float(r.get("output_tokens") or 0)), "reason_tok": det["reason_tok"],
                         "coverage": r.get("coverage_seen"), "frames": r.get("frames_seen"), "forced": det["forced"],
                         "text": r.get("text")})
    return rows


def _int_or(v, fallback):
    if v in (None, ""):
        return fallback
    try:
        return int(float(v))
    except ValueError:
        return fallback


def _thirds(rs):
    """按题序 (order_index, 没有就按 rows 顺序) 分前 / 中 / 后三段, 每段的平均步数 / 秒 / mIoU."""
    ordered = sorted(rs, key=lambda r: (r["order_index"] if r["order_index"] is not None else 0))
    n = len(ordered)
    out = {}
    for k in range(3):
        seg = ordered[k * n // 3:(k + 1) * n // 3] if n >= 3 else (ordered if k == 0 else [])
        sc = [r["iou3d"] for r in seg if r["iou3d"] is not None]
        out[f"third{k + 1}_steps"] = round(sum(r["steps"] for r in seg) / len(seg), 1) if seg else None
        out[f"third{k + 1}_s"] = round(sum(r["wall_s"] for r in seg) / len(seg), 0) if seg else None
        out[f"third{k + 1}_miou"] = round(sum(sc) / len(sc), 3) if sc else None
    return out


def per_scene(rows, macro, timeline):
    out = []
    scenes = sorted({r["scene"] for r in rows})
    for scene in scenes + ["ALL"]:
        rs = rows if scene == "ALL" else [r for r in rows if r["scene"] == scene]
        n = len(rs)
        scored = [r["iou3d"] for r in rs if r["iou3d"] is not None]
        rec = {"scene": scene, "n": n,
               "answered": sum(r["status"] == "answered" for r in rs),
               "ambiguous": sum(r["status"] == "ambiguous" for r in rs),
               "not_found": sum(r["status"] == "not_found" for r in rs),
               "forced": sum(bool(r["forced"]) for r in rs),
               "n_scored": len(scored),
               "miou3d": round(sum(scored) / len(scored), 4) if scored else None,
               "acc25": round(sum(v >= 0.25 for v in scored) / len(scored), 4) if scored else None,
               "acc50": round(sum(v >= 0.5 for v in scored) / len(scored), 4) if scored else None,
               "wall_s": round(sum(r["wall_s"] for r in rs), 1),
               "mean_wall_s": round(sum(r["wall_s"] for r in rs) / n, 1) if n else None,
               "llm_s": round(sum(r["llm_s"] for r in rs), 1), "tool_s": round(sum(r["tool_s"] for r in rs), 1),
               "segment_s": round(sum(r["segment_s"] for r in rs), 1), "lift_s": round(sum(r["lift_s"] for r in rs), 1),
               "mean_steps": round(sum(r["steps"] for r in rs) / n, 1) if n else None,
               "in_tok": sum(r["in_tok"] for r in rs), "out_tok": sum(r["out_tok"] for r in rs),
               "reason_tok": sum(r["reason_tok"] for r in rs),
               "mean_in_tok": int(sum(r["in_tok"] for r in rs) / n) if n else None,
               "n_recall": sum(int(r.get("n_recall") or 0) for r in rs),
               "open_hit": sum(int(r.get("open_hit") or 0) for r in rs),
               "mem_answer": sum(int(r.get("mem_answer") or 0) for r in rs),
               "mem_anchor": sum(int(r.get("mem_anchor") or 0) for r in rs)}
        rec.update(_thirds(rs))
        if scene != "ALL" and scene in macro and macro[scene].get("miou3d") is not None:
            rec.update({k: macro[scene][k] for k in ("n_scored", "miou3d", "acc25", "acc50")})   # 以评分器的宏平均为准
        for stage in ("prep", "bench", "score"):
            if scene == "ALL":
                v = sum(s for (st, _), s in timeline.items() if st == stage)
            else:
                v = timeline.get((stage, scene))
            rec[f"{stage}_wall_s"] = None if v is None else round(v, 0)
        out.append(rec)
    return out


def fmt(v, w, nd=None):
    if v is None:
        s = "-"
    elif isinstance(v, float):
        s = f"{v:.{nd if nd is not None else 3}f}" if nd != 0 else f"{v:.0f}"
    else:
        s = str(v)
    return s.rjust(w)


def print_tables(rows, scenes, price_in=None, price_out=None):
    print(f"\n== per scene ==  ({len(rows)} queries)")
    head = ["scene", "n", "ans", "amb", "nf", "forced", "mIoU", "a@.25", "a@.5", "s/q", "steps", "in_tok/q",
            "reason", "seg_s", "lift_s", "bench_wall"]
    print(" ".join(h.rjust(w) for h, w in zip(head, [10, 4, 4, 4, 4, 6, 6, 6, 6, 6, 5, 9, 8, 7, 7, 10])))
    for s in scenes:
        print(" ".join([fmt(s["scene"], 10), fmt(s["n"], 4), fmt(s["answered"], 4), fmt(s["ambiguous"], 4),
                        fmt(s["not_found"], 4), fmt(s["forced"], 6), fmt(s["miou3d"], 6), fmt(s["acc25"], 6, 2),
                        fmt(s["acc50"], 6, 2), fmt(s["mean_wall_s"], 6, 0), fmt(s["mean_steps"], 5, 1),
                        fmt(s["mean_in_tok"], 9), fmt(s["reason_tok"], 8), fmt(s["segment_s"], 7, 0),
                        fmt(s["lift_s"], 7, 0), fmt(s["bench_wall_s"], 10, 0)]))
    tot = scenes[-1]
    print(f"\ntotal: {tot['n']} queries, {tot['wall_s'] / 3600:.1f} h of query time "
          f"(LLM {tot['llm_s'] / 3600:.1f} h, tools {tot['tool_s'] / 3600:.1f} h), "
          f"input {tot['in_tok'] / 1e6:.2f} M tokens, output {tot['out_tok'] / 1e6:.2f} M (reasoning {tot['reason_tok'] / 1e6:.2f} M)")
    if price_in is not None and price_out is not None:
        cost = tot["in_tok"] / 1e6 * price_in + tot["out_tok"] / 1e6 * price_out
        print(f"cost at ${price_in}/M in, ${price_out}/M out: ${cost:.2f}")
    print("\n== by position in the scene (thirds of the query order) ==")
    print(" ".join(h.rjust(w) for h, w in zip(["scene", "steps 1/2/3", "s/q 1/2/3", "mIoU 1/2/3", "recall", "open_hit", "mem_ans", "mem_anc"],
                                              [10, 16, 16, 20, 7, 8, 8, 8])))
    for s in scenes:
        st = "/".join(fmt(s.get(f"third{k}_steps"), 0, 1).strip() for k in (1, 2, 3))
        sec = "/".join(fmt(s.get(f"third{k}_s"), 0, 0).strip() for k in (1, 2, 3))
        mi = "/".join(fmt(s.get(f"third{k}_miou"), 0, 3).strip() for k in (1, 2, 3))
        print(" ".join([fmt(s["scene"], 10), st.rjust(16), sec.rjust(16), mi.rjust(20), fmt(s.get("n_recall"), 7),
                        fmt(s.get("open_hit"), 8), fmt(s.get("mem_answer"), 8), fmt(s.get("mem_anchor"), 8)]))
    by_role = {}
    for r in rows:
        if r["iou3d"] is not None:
            by_role.setdefault(r["role"], []).append(r["iou3d"])
    if by_role:
        print("\n== by role (all scenes) ==")
        for role, v in sorted(by_role.items()):
            print(f"  {role:<20} n={len(v):<4} mIoU={sum(v) / len(v):.3f}  acc@.5={sum(x >= 0.5 for x in v) / len(v):.2f}")
    worst = sorted([r for r in rows if r["iou3d"] is not None], key=lambda r: r["iou3d"])[:15]
    if worst:
        print("\n== lowest 15 ==")
        for r in worst:
            print(f"  {r['iou3d']:.2f}  {r['scene']} {r['qid']} {r['status']:<9} {r['ids'] or '-':<10} {r['text'][:60]}")


def write_csv(path, fields, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mode", default="session")
    ap.add_argument("--order", default="given", help="given | reverse | shuffle<seed>; 老跑次 (没有 order 段) 传 ''")
    ap.add_argument("--runs-root", default=config.RUNS_ROOT)
    ap.add_argument("--score-json", default=None, help="optional per-query score json to join (default <SPLATSCOUT_DATA>/eval/score3d/agent_<mode>_<tag>.json)")
    ap.add_argument("--timeline", default=None, help="默认 runs/logs/<tag>/timeline.csv")
    ap.add_argument("--price-in", type=float, default=None, help="美元 / 1M 输入 token (可选, 只为估成本)")
    ap.add_argument("--price-out", type=float, default=None)
    args = ap.parse_args()
    suffix = run_suffix(args.mode, args.tag, args.order)
    score_json = args.score_json or os.path.join(config.DATA_ROOT, "eval", "score3d", f"agent_{suffix}.json")
    if not os.path.exists(score_json) and args.order:
        alt = os.path.join(config.DATA_ROOT, "eval", "score3d", f"agent_{run_suffix(args.mode, args.tag)}.json")
        score_json = alt if os.path.exists(alt) else score_json
    timeline = args.timeline or os.path.join(args.runs_root, "logs", args.tag, "timeline.csv")
    per_q_score, macro = load_scores(score_json)
    rows = collect(args.runs_root, args.mode, args.tag, per_q_score, args.order)
    if not rows:
        raise SystemExit(f"no rows.csv under {args.runs_root}/<scene>/{suffix}/")
    scenes = per_scene(rows, macro, load_timeline(timeline))
    out = os.path.join(args.runs_root, f"summary_{suffix}")
    os.makedirs(out, exist_ok=True)
    write_csv(os.path.join(out, "per_query.csv"), Q_FIELDS, rows)
    write_csv(os.path.join(out, "per_scene.csv"), S_FIELDS, scenes)
    print_tables(rows, scenes, args.price_in, args.price_out)
    if not os.path.exists(score_json):
        print(f"\n(no score file {score_json}: iou columns empty; run the score stage)")
    print(f"\nwrote {out}/per_query.csv and per_scene.csv")


if __name__ == "__main__":
    main()
