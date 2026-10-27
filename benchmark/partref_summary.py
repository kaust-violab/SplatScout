"""Aggregate partref_score.py outputs over scenes -> the benchmark table (all queries pooled, per role, per scene).

Usage:  python benchmark/partref_summary.py --tag v1 [--tag other_run ...] [--scenes a,b,c] [--out summary.md]
Reads <SPLATSCOUT_DATA>/eval/<scene>/partref/<tag>/summary.json for every scene in benchmark/data/scenes.txt.
Track A (occlusion-aware share rendering on the held-out test frames) is the metric reported in the paper:
mIoU = mean over evaluable queries, Acc@0.25 / Acc@0.5 = fraction of queries with IoU above the threshold."""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import EVAL_DIR  # noqa: E402
from partref_score import agg  # noqa: E402

ROLE_GROUPS = {"Obj": ("object",), "Part": ("part_target", "part_discriminator", "part_anchor"), "Multi": ("multi",)}


def fmt(v):
    return "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", action="append", required=True, help="score tag(s) given to partref_score.py; repeatable")
    ap.add_argument("--scenes", default=None, help="comma-separated scene ids (default benchmark/data/scenes.txt)")
    ap.add_argument("--main-tau", type=float, default=0.05)
    ap.add_argument("--out", default=None, help="markdown output (default <SPLATSCOUT_DATA>/eval/partref_summary.md)")
    args = ap.parse_args()
    scenes = args.scenes.split(",") if args.scenes else [s.strip() for s in open(os.path.join(HERE, "data", "scenes.txt")) if s.strip()]
    out_md = args.out or os.path.join(EVAL_DIR, "partref_summary.md")
    L = ["# PartRef-ScanNet++ results", "", f"Scenes: {len(scenes)}. Pooled = all evaluable queries of all scenes averaged (paper protocol).", ""]
    hdr = "| run | queries | mIoU | Acc@0.25 | Acc@0.5 | Obj mIoU | Part mIoU | Multi mIoU | scene-mean mIoU | track B mIoU |"
    L += [hdr, "|" + "---|" * (hdr.count("|") - 1)]
    result, pooled_queries = {}, {}
    for tag in args.tag:
        pqs, macro, taus, missing = [], [], None, []
        for s in scenes:
            p = os.path.join(EVAL_DIR, s, "partref", tag, "summary.json")
            if not os.path.exists(p):
                missing.append(s)
                continue
            d = json.load(open(p))
            taus = d["taus"]
            pqs += [q for q in d["per_query"].values() if "reason" not in q]
            if d["all"]["A_mIoU"] is not None:
                macro.append(d["all"]["A_mIoU"])
        if not pqs:
            L.append(f"| {tag} | no summary.json found |" + " |" * (hdr.count("|") - 3))
            continue
        pooled_queries[tag] = pqs
        a = agg(pqs, taus, args.main_tau)
        groups = {}
        for g, roles in ROLE_GROUPS.items():
            sub = [q for q in pqs if q["role"] in roles]
            groups[g] = agg(sub, taus, args.main_tau) if sub else None
        L.append(f"| {tag} | {a['n_A']} | {fmt(a['A_mIoU'])} | {fmt(a['A_acc@0.25'])} | {fmt(a['A_acc@0.5'])} | "
                 + " | ".join(fmt(groups[g]["A_mIoU"] if groups[g] else None) for g in ROLE_GROUPS) + " | "
                 f"{fmt(float(np.mean(macro)) if macro else None)} | {fmt(a['B_mIoU'])} |")
        if missing:
            L.append(f"|  | missing scenes for {tag}: {', '.join(missing)} |" + " |" * (hdr.count("|") - 3))
        result[tag] = {"pooled": a, "groups": {g: v for g, v in groups.items()}, "scene_mean_A": float(np.mean(macro)) if macro else None,
                       "n_scenes": len(macro), "missing_scenes": missing}
    L += ["", "## Per role (track A mIoU, number of queries)", ""]
    roles = ["object", "part_target", "part_discriminator", "part_anchor", "multi"]
    L.append("| run | " + " | ".join(roles) + " |")
    L.append("|---|" + "---|" * len(roles))
    for tag, pqs in pooled_queries.items():
        cells = []
        for r in roles:
            sub = [q for q in pqs if q["role"] == r]
            a = agg(sub, [args.main_tau], args.main_tau) if sub else None
            cells.append("-" if a is None else f"{fmt(a['A_mIoU'])} ({a['n_A']})")
        L.append(f"| {tag} | " + " | ".join(cells) + " |")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    open(out_md, "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump(result, open(out_md.replace(".md", ".json"), "w"), indent=1)
    print("\n".join(L))
    print(f"\nwrote {out_md}")


if __name__ == "__main__":
    main()
