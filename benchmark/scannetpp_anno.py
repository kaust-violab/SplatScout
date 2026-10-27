"""Read the official ScanNet++ instance annotation of a scene (scans/segments.json + scans/segments_anno.json)."""
import json
import os

import numpy as np

from paths import SCANNETPP

WORK_LABELS = {"remove", "split", "object", "structure", "unknown"}   # annotation scaffolding, not objects


def mesh_n_vertices(scene):
    """Vertex count from the ply header only."""
    with open(os.path.join(SCANNETPP, scene, "scans", "mesh_aligned_0.05.ply"), "rb") as f:
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex"):
                return int(line.split()[-1])
            if line == "end_header":
                raise RuntimeError("no vertex count in the ply header")


def load_official(scene):
    """Official instances -> [{obj_id, label, verts (set of mesh vertex indices), status}]."""
    seg = json.load(open(os.path.join(SCANNETPP, scene, "scans", "segments.json")))
    si = np.asarray(seg["segIndices"], dtype=np.int64)
    order = np.argsort(si, kind="stable")
    svals = si[order]
    anno = json.load(open(os.path.join(SCANNETPP, scene, "scans", "segments_anno.json")))
    objs = []
    for g in anno["segGroups"]:
        segs = np.asarray(g["segments"], dtype=np.int64)
        lo = np.searchsorted(svals, segs, side="left")
        hi = np.searchsorted(svals, segs, side="right")
        verts = np.concatenate([order[a:b] for a, b in zip(lo, hi)]) if len(segs) else \
            np.empty(0, np.int64)
        status = "dropped" if g["label"].strip().lower() in WORK_LABELS else "kept"
        objs.append({"obj_id": int(g["id"]), "label": g["label"],
                     "verts": set(verts.tolist()), "status": status})
    return objs
