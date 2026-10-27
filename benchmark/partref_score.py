"""PartRef 评估协议 v2.1 评分器: 赛道 A (有遮挡, 全场景份额渲染, = 旧 score.py 口径) + 赛道 B (无遮挡, 只渲提交集合, 轮廓交并比 + 深度感知交并比).
提交: 高斯硬集合 / 软权重 / 库实例号 / 网格顶点集 (见 partref_common.parse_submission).
模型: 默认我们的 2DGS ckpt (需 SCENE 环境变量, GS2D=1); --ply 给同行的 INRIA 3DGS (ZeroSplat / GaussDet 交的是它们自己场景的高斯), 中值深度用切片扫描逼近 (§6.6).
Ground truth: the partref_gt.py cache, computed on the fly if missing. Track A (occlusion-aware, full-scene share rendering) is the metric of the paper.
产出: <SPLATSCOUT_DATA>/eval/<scene>/partref/<tag>/{rows.csv, summary.json, overlays/}
用法: SCENE=f9f95681fd GS2D=1 python benchmark/partref_score.py --scene f9f95681fd --answers <answers.json> --tag agent_0915 [--ply x.ply] [--taus 0.01,0.02,0.05] [--alpha-thr 0.5] [--overlays]"""
import argparse
import csv
import json
import os
import time

import numpy as np

from partref_common import (EVAL_DIR, SCENES_ROOT, FALSE_APPEAR_PX, MIN_PX, Caster, SplatModel, iou, load_gt, load_queries,
                            parse_submission, target_vertices, test_cams, valid_mask, view_tensors)


def per_query_summary(qrec):
    return {k: v for k, v in qrec.items() if k != "frames"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--answers", required=True)
    ap.add_argument("--tag", required=True, help="输出子目录 partref/<tag>")
    ap.add_argument("--ply", default=None, help="同行方法自己的 3DGS ply (提交的高斯号指向它); 不给 = 我们的 2DGS ckpt")
    ap.add_argument("--taus", default="0.01,0.02,0.05", help="深度容差 (米), 逗号分隔; 主表用 --main-tau")
    ap.add_argument("--main-tau", type=float, default=0.05)
    ap.add_argument("--alpha-thr", type=float, default=0.5, help="赛道 B 掩码阈值 (0.5 主口径; 0.3 薄部件变体)")
    ap.add_argument("--strict-status", action="store_true", help="status 不是 answered 的题当空答 (默认与 score.py 一致: 有高斯就评)")
    ap.add_argument("--tracks", default="A,B")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overlays", action="store_true")
    ap.add_argument("--n-slices", type=int, default=32)
    args = ap.parse_args()
    import torch
    scene = args.scene
    taus = [float(x) for x in args.taus.split(",") if x.strip()]
    if args.main_tau not in taus:
        taus.append(args.main_tau)
    tracks = set(args.tracks.split(","))
    out_dir = os.path.join(EVAL_DIR, scene, "partref", args.tag)
    os.makedirs(os.path.join(out_dir, "overlays"), exist_ok=True)

    os.environ.setdefault("SCENE", scene)
    os.environ.setdefault("GS2D", "1")
    from splatscout import back_project as bp
    assert bp.SCENE == scene, f"SCENE 环境变量 ({bp.SCENE}) 与 --scene ({scene}) 不一致"
    model = SplatModel.from_ply(args.ply) if args.ply else SplatModel.ours_2dgs(bp)
    mem_path = os.path.join(bp.OUT_DIR, "membership.npz")
    membership = np.load(mem_path) if (not args.ply and os.path.exists(mem_path)) else None
    caster = Caster(scene)
    cams = test_cams(scene)
    queries = load_queries(scene)[: args.limit] if args.limit else load_queries(scene)
    answers = json.load(open(args.answers))
    print(f"[partref] {scene}: {len(queries)} 题, {len(cams)} 帧, 模型 {model.kind} {model.N} 高斯, 赛道 {sorted(tracks)}, taus {taus}, alpha_thr {args.alpha_thr}", flush=True)
    frames = []
    for cam in cams:
        vm, K, W, H = view_tensors(cam, model.device)
        frames.append({"cam": cam, "name": cam["img_name"], "valid": valid_mask(scene, cam), "vm": vm, "K": K, "W": W, "H": H})

    rows, per_query = [], {}
    t_all = time.time()
    for qi, q in enumerate(queries):
        gt = load_gt(scene, q["qid"])
        if gt is None:
            tv = target_vertices(scene, q["targets"], len(caster.verts))
            if tv is None:
                per_query[q["qid"]] = {"role": q["role"], "text": q["text"], "reason": "no_gt"}
                rows.append([q["qid"], q["role"], "", 0, 0, 0, 0] + [""] * (4 + len(taus)) + ["no_gt", q["text"]])
                continue
            face_idx = caster.faces_of(tv)
            flab = np.zeros(len(caster.faces), bool)
            flab[face_idx] = True
            gt = {}
            for f in frames:
                tri = caster.first_hit(f["cam"])
                occ = np.where(tri >= 0, flab[np.clip(tri, 0, None)], False)
                unocc, z = caster.unoccluded(face_idx, f["cam"])
                gt[f["name"]] = (occ, unocc, z)
        a_ent = answers.get(q["qid"])
        if isinstance(a_ent, dict) and a_ent.get("blocked"):        # 被服务商内容过滤拦下的题: 单列, 不计入平均
            per_query[q["qid"]] = {"role": q["role"], "text": q["text"], "reason": "blocked:" + str(a_ent["blocked"])[:80]}
            rows.append([q["qid"], q["role"], "", 0, 0, 0, 0] + [""] * (4 + len(taus)) + ["blocked", q["text"]])
            continue
        sub = parse_submission(a_ent, model.N, membership, args.strict_status)
        sub_faces = None
        if sub is not None and sub[0] == "verts":
            sub_faces = caster.faces_of(sub[1])
            sub_flab = np.zeros(len(caster.faces), bool)
            sub_flab[sub_faces] = True
        qrec = {"role": q["role"], "text": q["text"], "frames": [], "n_frames_A": 0, "n_frames_B": 0, "false_appear_frames": 0, "empty_gt_frames": 0}
        A_list, B_list, Bt_list, derr_list = [], [], {t: [] for t in taus}, []
        best = {"A": (-1, None), "B": (-1, None)}
        for f in frames:
            occ, unocc, gz = gt[f["name"]]
            occ, unocc = occ & f["valid"], unocc & f["valid"]
            occ_px, un_px = int(occ.sum()), int(unocc.sum())
            validA = occ_px >= MIN_PX
            validB = un_px >= MIN_PX and occ_px > 0
            empty_gt = un_px == 0
            rec = {"frame": f["name"], "validA": validA, "validB": validB, "gt_occ_px": occ_px, "gt_unocc_px": un_px,
                   "iouA": None, "iouB": None, "iouB_tau": {}, "depth_err_cm": None, "predB_px": None}
            need_B = ("B" in tracks) and (validB or empty_gt)
            if sub is None:
                if validA:
                    A_list.append(0.0); rec["iouA"] = 0.0
                if validB:
                    B_list.append(0.0); rec["iouB"] = 0.0
                    for t in taus:
                        Bt_list[t].append(0.0); rec["iouB_tau"][t] = 0.0
                if empty_gt:
                    qrec["empty_gt_frames"] += 1
                rec["status"] = "no_answer"
            else:
                W, H = f["W"], f["H"]
                if validA and "A" in tracks:
                    if sub[0] == "verts":
                        tri = caster.first_hit(f["cam"])
                        predA = np.where(tri >= 0, sub_flab[np.clip(tri, 0, None)], False) & f["valid"]
                    else:
                        share, A = model.share(torch.from_numpy(sub[2]).to(model.device), f["vm"], f["K"], W, H)
                        predA = (share > 0.5) & (A > 0.5) & f["valid"]
                    ia = iou(predA, occ)
                    A_list.append(ia); rec["iouA"] = round(ia, 4)
                    if args.overlays and ia > best["A"][0]:
                        best["A"] = (ia, (f, predA, occ))
                if need_B:
                    if sub[0] == "verts":
                        predB, pz = caster.unoccluded(sub_faces, f["cam"])
                        defined = predB.copy()
                    else:
                        a, pz, defined = model.isolated(sub[1], torch.from_numpy(sub[2]).to(model.device), f["vm"], f["K"], W, H,
                                                        thr=args.alpha_thr, n_slices=args.n_slices)
                        predB = (a >= args.alpha_thr) & defined
                    predB = predB & f["valid"]
                    rec["predB_px"] = int(predB.sum())
                    if empty_gt:
                        qrec["empty_gt_frames"] += 1
                        if rec["predB_px"] > FALSE_APPEAR_PX:
                            qrec["false_appear_frames"] += 1
                    if validB:
                        ib = iou(predB, unocc)
                        B_list.append(ib); rec["iouB"] = round(ib, 4)
                        inter = predB & unocc
                        union = int((predB | unocc).sum())
                        if inter.any():
                            dd = np.abs(pz[inter] - gz[inter])
                            rec["depth_err_cm"] = round(float(dd.mean()) * 100, 2)
                            derr_list.append(float(dd.mean()) * 100)
                            for t in taus:
                                v = float((dd <= t).sum() / max(union, 1))
                                Bt_list[t].append(v); rec["iouB_tau"][t] = round(v, 4)
                        else:
                            for t in taus:
                                Bt_list[t].append(0.0); rec["iouB_tau"][t] = 0.0
                        if args.overlays and ib > best["B"][0]:
                            best["B"] = (ib, (f, predB, unocc))
                rec["status"] = "ok"
            qrec["frames"].append(rec)
            rows.append([q["qid"], q["role"], f["name"], int(validA), int(validB), occ_px, un_px,
                         "" if rec["iouA"] is None else rec["iouA"], "" if rec["iouB"] is None else rec["iouB"]]
                        + ["" if rec["iouB_tau"].get(t) is None else rec["iouB_tau"][t] for t in taus]
                        + ["" if rec["depth_err_cm"] is None else rec["depth_err_cm"], "" if rec["predB_px"] is None else rec["predB_px"],
                           rec["status"], q["text"]])
        qrec["n_frames_A"], qrec["n_frames_B"] = len(A_list), len(B_list)
        qrec["A"] = round(float(np.mean(A_list)), 4) if A_list else None
        qrec["B"] = round(float(np.mean(B_list)), 4) if B_list else None
        qrec["B_tau"] = {str(t): (round(float(np.mean(Bt_list[t])), 4) if Bt_list[t] else None) for t in taus}
        qrec["depth_err_cm"] = round(float(np.mean(derr_list)), 2) if derr_list else None
        qrec["answered"] = sub is not None
        qrec["wrong_instance"] = (qrec["A"] is not None and qrec["A"] < 0.1)        # §7.3 选错实例
        per_query[q["qid"]] = qrec
        if args.overlays:
            import cv2
            for tr in ("A", "B"):
                if best[tr][1] is None:
                    continue
                f, pred, g = best[tr][1]
                img = cv2.imread(scene_paths_image(scene, f["name"]))
                img = cv2.resize(img, (f["W"], f["H"]))
                img[pred] = (0.45 * img[pred] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
                cnts, _ = cv2.findContours(g.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)
                cv2.putText(img, f"{tr} {best[tr][0]:.2f} {q['text'][:60]}", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imwrite(os.path.join(out_dir, "overlays", f"{qi:03d}_{q['qid']}_{tr}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"[partref {qi + 1}/{len(queries)}] A {qrec['A']} B {qrec['B']} B@{args.main_tau:.2f} {qrec['B_tau'].get(str(args.main_tau))} "
              f"derr {qrec['depth_err_cm']} cm | {q['text'][:55]}", flush=True)

    with open(os.path.join(out_dir, "rows.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["qid", "role", "frame", "validA", "validB", "gt_occ_px", "gt_unocc_px", "iouA", "iouB"] + [f"iouB_tau{t}" for t in taus]
                   + ["depth_err_cm", "predB_px", "status", "text"])
        w.writerows(rows)
    summary = summarize(per_query, taus, args.main_tau)
    summary.update({"scene": scene, "answers": args.answers, "tag": args.tag, "model": args.ply or "ours_2dgs", "alpha_thr": args.alpha_thr,
                    "strict_status": args.strict_status, "taus": taus, "main_tau": args.main_tau, "n_queries": len(queries),
                    "seconds": round(time.time() - t_all), "per_query": {k: per_query_summary(v) for k, v in per_query.items()}})
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), ensure_ascii=False, indent=1)
    a, b = summary["all"], summary["all"]
    print(f"\n[partref] {scene} {args.tag}: A mIoU {a['A_mIoU']} acc@.5 {a['A_acc@0.5']} (n={a['n_A']}) | B mIoU {b['B_mIoU']} "
          f"B@tau{args.main_tau} {b['B_tau_mIoU'].get(str(args.main_tau))} acc@.5 {b['B_acc@0.5']} (n={b['n_B']}) | 错层 {b['mislayer_ratio']} "
          f"深度误差 {b['depth_err_cm']} cm | 误现 {b['false_appear_rate']} | 选错实例 {b['wrong_instance']} 题 | {summary['seconds']} s")
    for k, v in summary["by_role"].items():
        print(f"   {k:<20} nA={v['n_A']:<3} A {v['A_mIoU']}  nB={v['n_B']:<3} B {v['B_mIoU']}  B@tau {v['B_tau_mIoU'].get(str(args.main_tau))}")
    print(f"-> {out_dir}")


def scene_paths_image(scene, name):
    from splatscout import scene_paths
    return scene_paths.image_file(os.path.join(SCENES_ROOT, scene, "images"), name)


def agg(qs, taus, main_tau):
    A = [q["A"] for q in qs if q.get("A") is not None]
    B = [q["B"] for q in qs if q.get("B") is not None]
    out = {"n_A": len(A), "n_B": len(B),
           "A_mIoU": round(float(np.mean(A)), 4) if A else None,
           "A_acc@0.25": round(float(np.mean([x >= .25 for x in A])), 4) if A else None,
           "A_acc@0.5": round(float(np.mean([x >= .5 for x in A])), 4) if A else None,
           "B_mIoU": round(float(np.mean(B)), 4) if B else None,
           "B_acc@0.25": round(float(np.mean([x >= .25 for x in B])), 4) if B else None,
           "B_acc@0.5": round(float(np.mean([x >= .5 for x in B])), 4) if B else None,
           "B_tau_mIoU": {}, "B_tau_acc@0.5": {}}
    for t in taus:
        v = [q["B_tau"][str(t)] for q in qs if q.get("B_tau", {}).get(str(t)) is not None]
        out["B_tau_mIoU"][str(t)] = round(float(np.mean(v)), 4) if v else None
        out["B_tau_acc@0.5"][str(t)] = round(float(np.mean([x >= .5 for x in v])), 4) if v else None
    d = [q["depth_err_cm"] for q in qs if q.get("depth_err_cm") is not None]
    out["depth_err_cm"] = round(float(np.mean(d)), 2) if d else None
    bt = out["B_tau_mIoU"].get(str(main_tau))
    out["mislayer_ratio"] = round((out["B_mIoU"] - bt) / out["B_mIoU"], 4) if (out["B_mIoU"] and bt is not None) else None
    eg = sum(q.get("empty_gt_frames", 0) for q in qs)
    out["false_appear_rate"] = round(sum(q.get("false_appear_frames", 0) for q in qs) / eg, 4) if eg else None
    out["wrong_instance"] = int(sum(bool(q.get("wrong_instance")) for q in qs))
    out["n_unanswered"] = int(sum(not q.get("answered", False) for q in qs if "reason" not in q))
    return out


def summarize(per_query, taus, main_tau):
    scored = [q for q in per_query.values() if "reason" not in q]
    by_role = {}
    for q in scored:
        by_role.setdefault(q["role"], []).append(q)
    return {"all": agg(scored, taus, main_tau), "by_role": {k: agg(v, taus, main_tau) for k, v in sorted(by_role.items())},
            "n_no_gt": sum("reason" in q for q in per_query.values()),
            "n_blocked": sum(str(q.get("reason", "")).startswith("blocked") for q in per_query.values()),
            "blocked": [k for k, q in per_query.items() if str(q.get("reason", "")).startswith("blocked")]}


if __name__ == "__main__":
    main()
