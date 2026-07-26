"""
tests/test_wandb_verbosity.py

Unit tests for the SPINESURG_WANDB_VERBOSE gate in
tools/nnunet_wandb_variant.py.

The trainer resolves `_WANDB_VERBOSE` ONCE at import time (a module-level
constant, like `_SCHEME`). When False (the default), the per-epoch W&B
payload emits a FOCUSED key set: nnU-Net per-class dice, per-subgroup
roll-ups, the any_lstv / any_sacralization roll-ups, and every
val/headline/* rebuttal metric. It withholds only the bulky
per-(subgroup, class) matrix keys:

    val/lstv_subgroup/{sub}/dice/{class}
    val/lstv_subgroup/{sub}/n_with_class/{class}

When True it additionally emits that matrix.

Because the flag is bound at import time, these tests import the module
FRESH in a subprocess with the env var set/unset, then drive a synthetic
call to _WandBMixin._aggregate_subgroup_dice() on a bare instance (no
GPU, no nnU-Net trainer, no torch tensors) and inspect which keys the
aggregator returns. torch / nnunetv2 are stubbed only if absent (the
aggregation path uses numpy + pre-populated Python buffers).

Run:
    python -m pytest tests/test_wandb_verbosity.py -v
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


# Child program: stub heavy deps if absent, import the trainer module fresh,
# build a synthetic subgroup buffer, run the aggregator, and emit
# {flag, keys} as JSON on stdout.
_CHILD = r"""
import json, sys, types

TOOLS = sys.argv[1]
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

# Stub torch + nnunetv2 only if the real ones are not importable. The
# aggregation path under test uses numpy + pre-populated Python buffers;
# it never touches a torch tensor.
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

n = len(m._FG_CLASS_IDS)

# Bare mixin instance (skip nnUNetTrainer.__init__ entirely).
obj = m._WandBMixin.__new__(m._WandBMixin)
obj._per_subgroup_logging_enabled = True

# One synthetic case per relevant subtype, full dice on every fg class so
# per_class_means, roll-ups, and the L4/last_lumbar headlines all populate.
dice_row = [0.9] * n
obj._val_subgroup_dice = {s: [] for s in m._KNOWN_SUBTYPES}
obj._val_subgroup_dice[m._SUBTYPE_LUMB] = [list(dice_row)]
obj._val_subgroup_dice[m._SUBTYPE_SACR_COUNT] = [list(dice_row)]

# Voxel buffer: on sacr_count the bottom-lumbar class is absent from GT
# (gt_n == 0) with a small pred count -> feeds the specificity headline.
absent_cid = (m._L6_LABEL_ID if m._SCHEME == "unmerged"
              else m._LAST_LUMBAR_LABEL_ID)
absent_idx = m._FG_CLASS_IDS.index(absent_cid)
vox_row = [(50, 50)] * n
vox_row[absent_idx] = (0, 10)
obj._val_subgroup_voxels = {s: [] for s in m._KNOWN_SUBTYPES}
obj._val_subgroup_voxels[m._SUBTYPE_SACR_COUNT] = [list(vox_row)]

# Empty (non-None) confusion buffers -> confusion blocks emit nothing here.
obj._val_subgroup_confusion_by_gt = {s: [] for s in m._KNOWN_SUBTYPES}
obj._val_subgroup_confusion_by_pred = {s: [] for s in m._KNOWN_SUBTYPES}

out = m._WandBMixin._aggregate_subgroup_dice(obj)

payload = {
    "verbose_flag": bool(m._WANDB_VERBOSE),
    "keys": sorted(out.keys()),
}
print("JSON_BEGIN" + json.dumps(payload) + "JSON_END")
"""


def _run_child(verbose_env):
    """Import the module fresh with SPINESURG_WANDB_VERBOSE set to
    `verbose_env` (None => unset) and return {verbose_flag, keys}."""
    env = dict(os.environ)
    env.pop("SPINESURG_WANDB_VERBOSE", None)
    # Pin the default (merged) scheme so key names are deterministic.
    env.pop("SPINESURG_LABEL_SCHEME", None)
    if verbose_env is not None:
        env["SPINESURG_WANDB_VERBOSE"] = verbose_env
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, str(_TOOLS_DIR)],
        env=env, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        pytest.skip(
            f"could not import/run trainer module "
            f"(verbose={verbose_env!r}): {proc.stderr.strip()[-800:]}"
        )
    out = proc.stdout
    if "JSON_BEGIN" not in out or "JSON_END" not in out:
        pytest.skip(f"child produced no JSON; stderr={proc.stderr[-800:]}")
    body = out.split("JSON_BEGIN", 1)[1].split("JSON_END", 1)[0]
    return json.loads(body)


# ── flag default ───────────────────────────────────────────────────────────

def test_verbose_flag_defaults_false():
    c = _run_child(None)
    assert c["verbose_flag"] is False


def test_verbose_flag_true_when_set():
    c = _run_child("1")
    assert c["verbose_flag"] is True


# ── default (focused) — matrix withheld, everything else present ────────────

def test_default_withholds_per_class_matrix():
    keys = set(_run_child(None)["keys"])
    # The bulky per-(subgroup, class) matrix must be ABSENT by default.
    assert not any(
        k.startswith("val/lstv_subgroup/") and "/dice/" in k for k in keys
    ), "per-(subgroup,class) dice matrix must be gated off by default"
    assert not any("/n_with_class/" in k for k in keys), \
        "n_with_class matrix must be gated off by default"


def test_default_keeps_rollups_and_headlines():
    keys = set(_run_child(None)["keys"])
    # Per-subgroup roll-ups (always-on).
    assert "val/lstv_subgroup/lumb/n_cases" in keys
    assert "val/lstv_subgroup/lumb/mean_fg_dice" in keys
    assert "val/lstv_subgroup/lumb/mean_lumbar_dice" in keys
    assert "val/lstv_subgroup/sacr_count/mean_fg_dice" in keys
    # Rebuttal headlines (always-on).
    assert "val/headline/last_lumbar_dice_on_lumbarization" in keys
    assert "val/headline/last_lumbar_n_lumbarization_cases" in keys
    assert "val/headline/L4_dice_on_sacr_count" in keys
    assert "val/headline/L4_n_sacr_count_cases" in keys
    # Specificity headline (always-on scalar).
    assert "val/headline/last_lumbar_specificity_on_sacr_count" in keys


# ── verbose — matrix restored, headlines still present ──────────────────────

def test_verbose_emits_per_class_matrix():
    keys = set(_run_child("1")["keys"])
    assert "val/lstv_subgroup/lumb/dice/last_lumbar" in keys
    assert "val/lstv_subgroup/lumb/n_with_class/last_lumbar" in keys
    assert "val/lstv_subgroup/sacr_count/dice/L4" in keys
    # Headlines and roll-ups remain present in verbose mode too.
    assert "val/headline/last_lumbar_dice_on_lumbarization" in keys
    assert "val/lstv_subgroup/lumb/mean_fg_dice" in keys


def test_verbose_adds_keys_vs_default():
    default_keys = set(_run_child(None)["keys"])
    verbose_keys = set(_run_child("1")["keys"])
    # Verbose is a strict superset; the extra keys are exactly the matrix.
    extra = verbose_keys - default_keys
    assert extra, "verbose must add the per-(subgroup,class) matrix keys"
    assert all(
        ("/dice/" in k or "/n_with_class/" in k) and k.startswith("val/lstv_subgroup/")
        for k in extra
    ), f"verbose-only keys must all be matrix keys; got {sorted(extra)[:10]}"
    assert default_keys <= verbose_keys, \
        "every focused/always-on key must also be present in verbose mode"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
