"""PartRef v2.1 §4 真值生成: 每题每张考试帧三张图 —— 有遮挡掩码 (整场景首命中落在目标面片)、无遮挡掩码 (只对目标面片求交)、
无遮挡 z 深度 (相机坐标系 z, 米). -> <SPLATSCOUT_DATA>/eval/<scene>/partref_gt/<qid>.npz, 评分器按需读.
用法: python benchmark/partref_gt.py --scene f9f95681fd [--force]   (纯 CPU, embree 光线求交; 一场几分钟到十几分钟)"""
import argparse
import os
import time

import numpy as np

from partref_common import Caster, load_queries, test_cams, target_vertices, save_gt, gt_dir, cam_mats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    scene = args.scene
    caster = Caster(scene)
    cams = test_cams(scene)
    queries = load_queries(scene)[: args.limit] if args.limit else load_queries(scene)
    os.makedirs(gt_dir(scene), exist_ok=True)
    t0 = time.time()
    first = {c["img_name"]: caster.first_hit(c) for c in cams}
    print(f"[gt] {scene}: {len(queries)} 题 x {len(cams)} 帧; 整帧求交 {time.time() - t0:.0f}s", flush=True)
    n_done = n_skip = 0
    for qi, q in enumerate(queries):
        p = os.path.join(gt_dir(scene), f"{q['qid']}.npz")
        if os.path.exists(p) and not args.force:
            n_skip += 1
            continue
        tv = target_vertices(scene, q["targets"], len(caster.verts))
        if tv is None:
            continue                                     # 真值缺失的题 (needs_id / 缺部件): 不写文件, 评分器记 no_gt
        face_idx = caster.faces_of(tv)
        flab = np.zeros(len(caster.faces), bool)
        flab[face_idx] = True
        frames = {}
        t1 = time.time()
        for cam in cams:
            tri = first[cam["img_name"]]
            occ = np.where(tri >= 0, flab[np.clip(tri, 0, None)], False)
            unocc, z = caster.unoccluded(face_idx, cam)
            frames[cam["img_name"]] = (occ, unocc, z)
        save_gt(scene, q["qid"], frames)
        n_done += 1
        n_occ = sum(int(v[0].sum() >= 300) for v in frames.values())
        n_un = sum(int(v[1].sum() >= 300 and v[0].sum() > 0) for v in frames.values())
        print(f"[gt {qi + 1}/{len(queries)}] {q['qid']} 有效帧 A {n_occ} / B {n_un}  {time.time() - t1:.1f}s | {q['text'][:50]}", flush=True)
    print(f"[gt] {scene}: 写 {n_done} 题, 跳过已有 {n_skip}, 总 {time.time() - t0:.0f}s -> {gt_dir(scene)}")


if __name__ == "__main__":
    main()
