"""All external paths of the benchmark scripts, from environment variables.

  SCANNETPP_DATA     the `data` directory of the official ScanNet++ release (<scene>/scans, <scene>/dslr)
  ARTICULATE3D_DIR   optional, only for queries whose targets are raw Articulate3D part ids (kind "a3part")
  SPLATSCOUT_DATA    working directory for converted scenes, 2DGS checkpoints, ground truth and scores (default <repo>/data)
The annotations and queries ship with the repository under benchmark/data/.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
from splatscout.config import DATA_ROOT  # noqa: E402

SCANNETPP = os.environ.get("SCANNETPP_DATA", "/path/to/scannetpp/data")
A3D_DIR = os.environ.get("ARTICULATE3D_DIR", "")

ANNO_ROOT = os.path.join(HERE, "data", "anno")                                   # objects.json / parts.json per scene
QDIR = os.environ.get("PARTREF_QDIR") or os.path.join(HERE, "data", "queries")    # another query set in the same format
SCENES_ROOT = os.path.join(DATA_ROOT, "scenes")                                  # prep_scene.py output
EVAL_DIR = os.path.join(DATA_ROOT, "eval")                                       # ground truth cache and scores
