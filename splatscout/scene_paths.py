"""Scene data layout under SPLATSCOUT_DATA (default <repo>/data). Only 2DGS reconstructions are supported.

  scenes/<scene>/        cameras.json, images/, masks/, init_points.npz   (benchmark/prep_scene.py)
  ckpts_2dgs/<scene>/    ckpt_29999.pt                                    (benchmark/train_2dgs.py)
  lift/<scene>/          gravity.json and other per-scene caches          (created on demand)
  cache/agent_prep/      preprocessing package of the agent               (python -m splatscout.scene_prep)
  eval/<scene>/          ground-truth masks and scores                    (benchmark/partref_gt.py, partref_score.py)
"""
import os

from .config import DATA_ROOT


def is_bench(scene):
    return True


def scene_dir(scene):
    return os.path.join(DATA_ROOT, "scenes", scene)


def lift_dir(scene):
    return os.path.join(DATA_ROOT, "lift", scene)


_STEM_CACHE = {}


def image_file(images_dir, stem):
    """Resolve an image path by its stem (the extension varies, e.g. .JPG for ScanNet++)."""
    m = _STEM_CACHE.get(images_dir)
    if m is None or stem not in m:
        m = {os.path.splitext(f)[0]: f for f in os.listdir(images_dir)}
        _STEM_CACHE[images_dir] = m
    if stem not in m:
        raise FileNotFoundError(f"{images_dir}/{stem}.*")
    return os.path.join(images_dir, m[stem])


def ckpt_path(scene, gs2d=True):
    if not gs2d:
        raise SystemExit("only 2DGS reconstructions are supported (GS2D=1)")
    return os.path.join(DATA_ROOT, "ckpts_2dgs", scene, "ckpt_29999.pt")
