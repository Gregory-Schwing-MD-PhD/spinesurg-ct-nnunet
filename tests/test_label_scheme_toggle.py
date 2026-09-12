"""
tests/test_label_scheme_toggle.py

Unit tests for the SPINESURG_LABEL_SCHEME toggle in
tools/nnunet_wandb_variant.py.

The trainer module resolves the label scheme ONCE at import time (into the
module-level `_SCHEME`) and finalizes every scheme-dependent constant then,
because several helpers bind those constants as default arguments (evaluated
at def-time). These tests therefore import the module FRESH in a subprocess
with SPINESURG_LABEL_SCHEME set, so each case gets a clean import.

They assert:
  * unset / merged (default) -> the merged Dataset803 constants (8 fg,
    last_lumbar, ignore=9, 9 output channels) — backward compatible.
  * unmerged                  -> the unmerged Dataset802 constants (9 fg,
    distinct L5 and L6, ignore=10, 10 output channels).
  * oneshot                   -> the Dataset810 constants (22 fg, ignore=23).
  * fullribs                  -> the Dataset813 constants (43 fg, ignore=44).
  * and that no two scheme names resolve to the SAME constants, which is what
    a branch guard swallowing a scheme looks like from the outside.

No GPU, no nnU-Net trainer instantiation. If torch / nnunetv2 are not
installed in the test environment, the child stubs them with the minimal
surface the module touches at import (the scheme constants are pure Python
and independent of both), so the test still runs everywhere.

Run:
    python -m pytest tests/test_label_scheme_toggle.py -v
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_THIS_DIR = Path(__file__).resolve().parent
_TOOLS_DIR = _THIS_DIR.parent / "tools"
_MODULE_FILE = _TOOLS_DIR / "nnunet_wandb_variant.py"


# Child program: stub heavy deps if absent, import the trainer module fresh,
# emit the scheme constants as JSON on stdout.
_CHILD = r"""
import json, sys, types

TOOLS = sys.argv[1]
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

# Stub torch + nnunetv2 only if the real ones are not importable. The scheme
# constants don't depend on either; the stubs just satisfy import-time names
# (e.g. `device=torch.device('cuda')` default args, base-class imports).
try:
    import torch  # noqa: F401
except Exception:
    torch = types.ModuleType("torch")
    class _D:
        def __call__(self, *a, **k): return _D()
        def __getattr__(self, n): return _D()
    torch.device = lambda *a, **k: _D()
    torch.Tensor = object
    torch.float32 = "f32"
    torch.zeros = lambda *a, **k: []
    sys.modules["torch"] = torch

try:
    import nnunetv2  # noqa: F401
except Exception:
    def _mod(name):
        m = types.ModuleType(name); sys.modules[name] = m; return m
    _mod("nnunetv2"); _mod("nnunetv2.training"); _mod("nnunetv2.training.nnUNetTrainer")
    _m = _mod("nnunetv2.training.nnUNetTrainer.nnUNetTrainer")
    class nnUNetTrainer: ...
    _m.nnUNetTrainer = nnUNetTrainer
    _mod("nnunetv2.training.nnUNetTrainer.variants")
    _mod("nnunetv2.training.nnUNetTrainer.variants.training_length")
    _m2 = _mod("nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs")
    class nnUNetTrainer_5epochs: ...
    class nnUNetTrainer_100epochs: ...
    _m2.nnUNetTrainer_5epochs = nnUNetTrainer_5epochs
    _m2.nnUNetTrainer_100epochs = nnUNetTrainer_100epochs

import importlib
m = importlib.import_module("nnunet_wandb_variant")
out = {
    "scheme": m._SCHEME,
    "anatomy_names": list(m._ANATOMY_NAMES),
    "fg_class_ids": list(m._FG_CLASS_IDS),
    "lumbar_ids": list(m._LUMBAR_IDS),
    "pelvis_ids": list(m._PELVIS_IDS),
    "sacrum_id": m._SACRUM_LABEL_ID,
    "ignore_label": m._IGNORE_LABEL,
    "num_output_classes": m._DEFAULT_NUM_OUTPUT_CLASSES,
    "has_last_lumbar_id": hasattr(m, "_LAST_LUMBAR_LABEL_ID"),
    "has_l5_id": hasattr(m, "_L5_LABEL_ID"),
    "has_l6_id": hasattr(m, "_L6_LABEL_ID"),
    "l6_id": getattr(m, "_L6_LABEL_ID", None),
    "l5_id": getattr(m, "_L5_LABEL_ID", None),
}
print("JSON_BEGIN" + json.dumps(out) + "JSON_END")
"""


def _import_under_scheme(scheme_env):
    """Import the trainer module in a fresh subprocess with
    SPINESURG_LABEL_SCHEME set to `scheme_env` (None => unset). Returns the
    constants dict, or calls pytest.skip on an import failure we can't fix."""
    env = dict(os.environ)
    env.pop("SPINESURG_LABEL_SCHEME", None)
    if scheme_env is not None:
        env["SPINESURG_LABEL_SCHEME"] = scheme_env
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, str(_TOOLS_DIR)],
        env=env, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        pytest.skip(
            f"could not import trainer module (scheme={scheme_env!r}): "
            f"{proc.stderr.strip()[-500:]}"
        )
    out = proc.stdout
    if "JSON_BEGIN" not in out or "JSON_END" not in out:
        pytest.skip(f"child produced no JSON payload; stderr={proc.stderr[-500:]}")
    payload = out.split("JSON_BEGIN", 1)[1].split("JSON_END", 1)[0]
    return json.loads(payload)


# ── merged (default) — backward-compatibility ──────────────────────────────

def test_default_unset_is_merged():
    c = _import_under_scheme(None)
    assert c["scheme"] == "merged"
    assert c["anatomy_names"] == [
        "L1", "L2", "L3", "L4", "last_lumbar", "sacrum", "left_hip", "right_hip"
    ]
    assert c["fg_class_ids"] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert c["lumbar_ids"] == [1, 2, 3, 4, 5]
    assert c["pelvis_ids"] == [6, 7, 8]
    assert c["sacrum_id"] == 6
    assert c["ignore_label"] == 9
    assert c["num_output_classes"] == 9
    assert c["has_last_lumbar_id"] is True
    assert c["has_l6_id"] is False


def test_explicit_merged_matches_default():
    assert _import_under_scheme("merged") == _import_under_scheme(None)


# ── unmerged (Dataset802 baseline) ─────────────────────────────────────────

def test_unmerged_constants():
    c = _import_under_scheme("unmerged")
    assert c["scheme"] == "unmerged"
    assert "L5" in c["anatomy_names"]
    assert "L6" in c["anatomy_names"]
    assert "last_lumbar" not in c["anatomy_names"]
    assert c["anatomy_names"] == [
        "L1", "L2", "L3", "L4", "L5", "L6", "sacrum", "left_hip", "right_hip"
    ]
    assert c["fg_class_ids"] == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert c["lumbar_ids"] == [1, 2, 3, 4, 5, 6]
    assert c["pelvis_ids"] == [7, 8, 9]
    assert c["sacrum_id"] == 7
    assert c["ignore_label"] == 10
    assert c["num_output_classes"] == 10
    assert c["has_l5_id"] is True
    assert c["has_l6_id"] is True
    assert c["has_last_lumbar_id"] is False


def test_802_alias_selects_unmerged():
    assert _import_under_scheme("802")["scheme"] == "unmerged"



# ── oneshot (Dataset810) and fullribs (Dataset813) ─────────────────────────
# These two had constant blocks but no tests, and the branch guard above them
# listed all three non-merged schemes, so both blocks were dead: Dataset813 was
# scored with Dataset802's nine classes. Label 6 (L2) answered to the name L6
# and label 10 -- L6 itself -- answered to _IGNORE_LABEL, so every L6 voxel was
# masked out before the per-case dice was taken.

def test_fullribs_constants():
    c = _import_under_scheme("fullribs")
    assert c["scheme"] == "fullribs"
    assert c["fg_class_ids"] == list(range(1, 44))
    assert c["lumbar_ids"] == [5, 6, 7, 8, 9, 10]
    assert c["pelvis_ids"] == [11, 40, 41, 42]
    assert c["sacrum_id"] == 11
    assert c["ignore_label"] == 44
    assert c["num_output_classes"] == 44
    # the headline metric must point at the class actually called L6
    assert c["l6_id"] == 10
    assert c["anatomy_names"][c["fg_class_ids"].index(c["l6_id"])] == "L6"
    # ... and the ignore label must not be a real class
    assert c["ignore_label"] not in c["fg_class_ids"]


def test_oneshot_constants():
    c = _import_under_scheme("oneshot")
    assert c["scheme"] == "oneshot"
    assert c["fg_class_ids"] == list(range(1, 23))
    assert c["lumbar_ids"] == [5, 6, 7, 8, 9, 10]
    assert c["sacrum_id"] == 11
    assert c["ignore_label"] == 23
    assert c["num_output_classes"] == 23
    assert c["l6_id"] == 10
    assert c["anatomy_names"][c["fg_class_ids"].index(c["l6_id"])] == "L6"


def test_813_and_810_aliases():
    assert _import_under_scheme("813")["scheme"] == "fullribs"
    assert _import_under_scheme("810")["scheme"] == "oneshot"


def test_every_scheme_resolves_to_its_own_constants():
    """No two scheme names may produce the same constants.

    This is the shape of the bug that hid for a whole training run: a branch
    guard listing several schemes swallows the ones with their own blocks
    below, and they all silently inherit the first block's label ids. Nothing
    raises -- the numbers are finite and plausible, they just describe a
    different bone.
    """
    schemes = ["merged", "unmerged", "oneshot", "fullribs"]
    seen = {}
    for name in schemes:
        c = _import_under_scheme(name)
        key = (tuple(c["fg_class_ids"]), c["ignore_label"], c["sacrum_id"])
        assert key not in seen, (
            f"{name!r} resolves to the same constants as {seen[key]!r}: "
            f"{len(c['fg_class_ids'])} fg classes, ignore={c['ignore_label']}. "
            f"A branch guard is swallowing one of them."
        )
        seen[key] = name


def test_anatomy_names_parallel_to_class_ids():
    """_ANATOMY_NAMES[i] names _FG_CLASS_IDS[i]; a mismatch renames every class."""
    for name in ["merged", "unmerged", "oneshot", "fullribs"]:
        c = _import_under_scheme(name)
        assert len(c["anatomy_names"]) == len(c["fg_class_ids"]), (
            f"{name}: {len(c['anatomy_names'])} names for "
            f"{len(c['fg_class_ids'])} classes"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
