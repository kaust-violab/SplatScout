"""Bridge to the modules that read the SCENE environment variable at import time (back_project, pre_relation).
Call bind_scene(scene) before the first use; one process serves one scene."""
import os

from . import config  # noqa: F401  (import order: config has no scene dependency)

_LOADED = {}


def bind_scene(scene):
    """Set SCENE / GS2D before the geometry modules are imported."""
    prev = os.environ.get("SCENE")
    if prev is not None and prev != scene and _LOADED:
        raise RuntimeError(f"this process is already bound to scene {prev}; it cannot switch to {scene}")
    os.environ["SCENE"] = scene
    os.environ.setdefault("GS2D", "1")


def bp():
    """back_project: rendering / projection / OBB / FPS."""
    if "bp" not in _LOADED:
        from . import back_project
        _LOADED["bp"] = back_project
    return _LOADED["bp"]


def pre_relation():
    """Geometric predicate library (relate tool)."""
    if "pr" not in _LOADED:
        from . import pre_relation as pr
        _LOADED["pr"] = pr
    return _LOADED["pr"]


def scene_paths():
    if "sp" not in _LOADED:
        from . import scene_paths as sp
        _LOADED["sp"] = sp
    return _LOADED["sp"]


def llm_config():
    """Endpoint / key / model name (llm_config.py, environment variables)."""
    if "cfg" not in _LOADED:
        from . import llm_config as cfg
        _LOADED["cfg"] = cfg
    return _LOADED["cfg"]
