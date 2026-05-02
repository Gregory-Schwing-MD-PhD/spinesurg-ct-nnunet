"""
tests/test_lstv_biased_dataloader.py

Tests for tools/lstv_biased_dataloader.py.

Standalone-runner style (matches tests/test_subtype_refinement.py and
others). Does NOT require pytest. If pytest IS installed, the file is
also pytest-compatible — pytest auto-discovers module-level test_*
functions and runs them the same way.

Test layers
===========
1. Pure-function tests (no nnU-Net dependency).
2. Mixin tests (LSTVBiasMixin composed with a fake parent class).
3. Production loader tests — only run if nnU-Net is importable.

Running
=======
    /opt/conda/bin/python3 tests/test_lstv_biased_dataloader.py
or
    python -m pytest tests/test_lstv_biased_dataloader.py -v   (if pytest installed)
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from typing import List

import numpy as np

# Layout: tests/ and tools/ are siblings under the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = PROJECT_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from lstv_biased_dataloader import (  # noqa: E402
    L1_LABEL, L2_LABEL, L3_LABEL, L4_LABEL, L5_LABEL, L6_LABEL,
    SACRUM_LABEL, LEFT_HIP_LABEL, RIGHT_HIP_LABEL,
    LSTVBiasMixin,
    _has_voxels,
    _is_lumbarization_signature,
    _is_sacralization_count_signature,
    select_lstv_biased_class,
    _NNUNET_AVAILABLE,
)


# =============================================================================
# Helpers
# =============================================================================

class _EnvOverride:
    """Save/restore env vars. Use as a context manager OR as a drop-in
    replacement for pytest's monkeypatch fixture (has a setenv method)."""

    def __init__(self):
        self._saved = {}

    def setenv(self, name, value):
        if name not in self._saved:
            self._saved[name] = os.environ.get(name)
        os.environ[name] = value

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.teardown()

    def teardown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self._saved.clear()


def _voxels(n: int = 10) -> np.ndarray:
    """Return a fake voxel-coordinates array with `n` (z,y,x) entries."""
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 100, size=(n, 3))


# =============================================================================
# Test fixtures: synthetic class_locations dicts mimicking real cases.
# =============================================================================

NORMAL_CASE = {
    L1_LABEL: _voxels(),
    L2_LABEL: _voxels(),
    L3_LABEL: _voxels(),
    L4_LABEL: _voxels(),
    L5_LABEL: _voxels(),
    SACRUM_LABEL: _voxels(),
    LEFT_HIP_LABEL: _voxels(),
    RIGHT_HIP_LABEL: _voxels(),
}

LUMB_CASE = {**NORMAL_CASE, L6_LABEL: _voxels()}

SACR_COUNT_CASE = {
    L1_LABEL: _voxels(),
    L2_LABEL: _voxels(),
    L3_LABEL: _voxels(),
    L4_LABEL: _voxels(),
    # NO L5
    # NO L6
    SACRUM_LABEL: _voxels(),
    LEFT_HIP_LABEL: _voxels(),
    RIGHT_HIP_LABEL: _voxels(),
}

PELVIC_ONLY_CASE = {
    SACRUM_LABEL: _voxels(),
    LEFT_HIP_LABEL: _voxels(),
    RIGHT_HIP_LABEL: _voxels(),
}

SPINE_ONLY_LUMB = {
    L1_LABEL: _voxels(), L2_LABEL: _voxels(), L3_LABEL: _voxels(),
    L4_LABEL: _voxels(), L5_LABEL: _voxels(), L6_LABEL: _voxels(),
}

LUMB_BUT_EMPTY_L6 = {
    **NORMAL_CASE,
    L6_LABEL: np.empty((0, 3), dtype=np.int64),
}


# =============================================================================
# Layer 1: Pure-function tests
# =============================================================================

def test_has_voxels_present_with_data():
    assert _has_voxels(NORMAL_CASE, L1_LABEL) is True

def test_has_voxels_absent_key():
    assert _has_voxels(NORMAL_CASE, L6_LABEL) is False

def test_has_voxels_empty_dict():
    assert _has_voxels({}, L6_LABEL) is False

def test_has_voxels_none_value():
    assert _has_voxels({L6_LABEL: None}, L6_LABEL) is False

def test_has_voxels_empty_array():
    assert _has_voxels({L6_LABEL: np.empty((0, 3))}, L6_LABEL) is False

def test_has_voxels_empty_list():
    assert _has_voxels({L6_LABEL: []}, L6_LABEL) is False

def test_has_voxels_lumb_with_empty_array():
    """L6 key exists but with no voxels -> 'no L6'."""
    assert _has_voxels(LUMB_BUT_EMPTY_L6, L6_LABEL) is False

def test_has_voxels_tuple_key_ignored_when_querying_int():
    """Region-based training uses tuple keys; querying for int 6 must
    not match a tuple key like (5, 6, 7)."""
    cl = {(5, 6, 7): _voxels()}
    assert _has_voxels(cl, L6_LABEL) is False


def test_lumb_signature_normal():
    assert _is_lumbarization_signature(NORMAL_CASE) is False

def test_lumb_signature_lumb():
    assert _is_lumbarization_signature(LUMB_CASE) is True

def test_lumb_signature_sacr_count():
    assert _is_lumbarization_signature(SACR_COUNT_CASE) is False

def test_lumb_signature_pelvic_only():
    """Pelvic-only views never trigger lumb (no lumbar labels at all)."""
    assert _is_lumbarization_signature(PELVIC_ONLY_CASE) is False

def test_lumb_signature_spine_only_still_triggers():
    """Spine-only view of a lumb patient: L6 voxels still present,
    so the signature still fires."""
    assert _is_lumbarization_signature(SPINE_ONLY_LUMB) is True

def test_lumb_signature_empty_l6_does_not_trigger():
    """L6 key present but empty -> no real L6 voxels -> not lumb."""
    assert _is_lumbarization_signature(LUMB_BUT_EMPTY_L6) is False


def test_sacr_count_signature_normal():
    """Normal has L5, so sacr_count never fires."""
    assert _is_sacralization_count_signature(NORMAL_CASE) is False

def test_sacr_count_signature_yes():
    assert _is_sacralization_count_signature(SACR_COUNT_CASE) is True

def test_sacr_count_signature_lumb():
    """Lumb has L5, so sacr_count never fires (even though L6 also
    present)."""
    assert _is_sacralization_count_signature(LUMB_CASE) is False

def test_sacr_count_signature_pelvic_only_no_l4():
    """Pelvic-only views have no L5 either, but ALSO no L4 — that's
    what disambiguates them from real sacr_count cases. The L4
    requirement in the signature is exactly this disambiguation."""
    assert _is_sacralization_count_signature(PELVIC_ONLY_CASE) is False

def test_sacr_count_signature_spine_only_no_sacrum():
    """Spine-only sacr_count: would need sacrum voxels but there are
    none in spine-only views. Falls through."""
    spine_only_sacr_count = {
        L1_LABEL: _voxels(), L2_LABEL: _voxels(),
        L3_LABEL: _voxels(), L4_LABEL: _voxels(),
        # NO L5 (the actual finding) but ALSO no sacrum (annotation mode)
    }
    assert _is_sacralization_count_signature(spine_only_sacr_count) is False

def test_sacr_count_signature_empty():
    assert _is_sacralization_count_signature({}) is False


def test_select_class_normal_returns_none():
    rng = np.random.default_rng(0)
    for _ in range(20):
        assert select_lstv_biased_class(
            NORMAL_CASE, bias_l6_prob=1.0, bias_sacrum_prob=1.0, rng=rng
        ) is None

def test_select_class_lumb_at_prob_1_always_l6():
    rng = np.random.default_rng(0)
    for _ in range(20):
        assert select_lstv_biased_class(
            LUMB_CASE, bias_l6_prob=1.0, bias_sacrum_prob=1.0, rng=rng
        ) == L6_LABEL

def test_select_class_lumb_at_prob_0_never_l6():
    rng = np.random.default_rng(0)
    for _ in range(20):
        assert select_lstv_biased_class(
            LUMB_CASE, bias_l6_prob=0.0, bias_sacrum_prob=0.0, rng=rng
        ) is None

def test_select_class_sacr_count_at_prob_1_always_sacrum():
    rng = np.random.default_rng(0)
    for _ in range(20):
        assert select_lstv_biased_class(
            SACR_COUNT_CASE, bias_l6_prob=0.0, bias_sacrum_prob=1.0, rng=rng
        ) == SACRUM_LABEL

def test_select_class_pelvic_only_returns_none():
    """Pelvic-only views match neither signature; always None."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        assert select_lstv_biased_class(
            PELVIC_ONLY_CASE, bias_l6_prob=1.0, bias_sacrum_prob=1.0, rng=rng
        ) is None

def test_select_class_lumb_takes_priority_over_sacr_count():
    """Pathological case satisfying both signatures. Lumb wins because
    L6 is the rarer/harder finding."""
    weird = {
        L1_LABEL: _voxels(), L2_LABEL: _voxels(), L3_LABEL: _voxels(),
        L4_LABEL: _voxels(),
        # NO L5
        L6_LABEL: _voxels(),  # but L6 present!
        SACRUM_LABEL: _voxels(),
    }
    rng = np.random.default_rng(0)
    result = select_lstv_biased_class(
        weird, bias_l6_prob=1.0, bias_sacrum_prob=1.0, rng=rng
    )
    assert result == L6_LABEL

def test_select_class_empty_class_locations():
    rng = np.random.default_rng(0)
    assert select_lstv_biased_class(
        {}, bias_l6_prob=1.0, bias_sacrum_prob=1.0, rng=rng
    ) is None


# Statistical: bias rate should match configured probability within a
# 5σ binomial band. With n=10000 and p=0.6, σ ≈ 49, 5σ ≈ 245.

_N_STAT_TRIALS = 10_000

def _binomial_5sigma_band(n, p):
    mean = n * p
    sigma = np.sqrt(n * p * (1 - p))
    return mean - 5 * sigma, mean + 5 * sigma

def test_select_class_lumb_bias_rate_60():
    rng = np.random.default_rng(seed=2026)
    n_l6 = sum(
        1 for _ in range(_N_STAT_TRIALS)
        if select_lstv_biased_class(
            LUMB_CASE, bias_l6_prob=0.60, bias_sacrum_prob=0.0, rng=rng
        ) == L6_LABEL
    )
    lo, hi = _binomial_5sigma_band(_N_STAT_TRIALS, 0.60)
    assert lo <= n_l6 <= hi, (
        f"L6 hit rate {n_l6}/{_N_STAT_TRIALS} = {n_l6/_N_STAT_TRIALS:.4f} "
        f"outside 5σ band [{lo:.0f}, {hi:.0f}] for p=0.60"
    )

def test_select_class_sacr_count_bias_rate_50():
    rng = np.random.default_rng(seed=2026)
    n_sacrum = sum(
        1 for _ in range(_N_STAT_TRIALS)
        if select_lstv_biased_class(
            SACR_COUNT_CASE, bias_l6_prob=0.0, bias_sacrum_prob=0.50, rng=rng
        ) == SACRUM_LABEL
    )
    lo, hi = _binomial_5sigma_band(_N_STAT_TRIALS, 0.50)
    assert lo <= n_sacrum <= hi, (
        f"sacrum hit rate {n_sacrum}/{_N_STAT_TRIALS} outside 5σ band "
        f"[{lo:.0f}, {hi:.0f}] for p=0.50"
    )

def test_select_class_normal_never_biases():
    rng = np.random.default_rng(seed=2026)
    for _ in range(_N_STAT_TRIALS):
        r = select_lstv_biased_class(
            NORMAL_CASE, bias_l6_prob=1.0, bias_sacrum_prob=1.0, rng=rng
        )
        assert r is None  # MUST never bias on normal cases


# =============================================================================
# Layer 2: Mixin tests via a fake parent class
# =============================================================================

class _FakeBaseLoader:
    """Stand-in for nnUNetDataLoader3D. Records every get_bbox call so
    tests can assert what was passed through to the parent."""

    def __init__(self, *args, **kwargs):
        self.get_bbox_calls: List[dict] = []

    def get_bbox(self, data_shape, force_fg, class_locations, overwrite_class=None):
        self.get_bbox_calls.append({
            "data_shape": data_shape,
            "force_fg": force_fg,
            "class_locations": class_locations,
            "overwrite_class": overwrite_class,
        })
        return ([0, 0, 0], list(data_shape))


class _TestableBiasedLoader(LSTVBiasMixin, _FakeBaseLoader):
    """Mixin composed with the fake parent — testable without nnU-Net."""
    pass


def test_mixin_force_fg_false_no_bias():
    """force_fg=False: parent does random cropping, no bias logic."""
    loader = _TestableBiasedLoader(bias_l6_prob=1.0, bias_sacrum_prob=1.0)
    loader.get_bbox(data_shape=(100, 100, 100), force_fg=False,
                     class_locations=LUMB_CASE)
    call = loader.get_bbox_calls[0]
    assert call["overwrite_class"] is None
    assert loader._bias_n_force_fg == 0  # not even counted

def test_mixin_caller_already_set_overwrite_class():
    """If caller passes overwrite_class explicitly, don't override."""
    loader = _TestableBiasedLoader(bias_l6_prob=1.0, bias_sacrum_prob=1.0)
    loader.get_bbox(data_shape=(100, 100, 100), force_fg=True,
                     class_locations=LUMB_CASE, overwrite_class=L4_LABEL)
    call = loader.get_bbox_calls[0]
    assert call["overwrite_class"] == L4_LABEL  # unchanged

def test_mixin_bias_disabled():
    """bias_enabled=False: pure pass-through, no counter increment."""
    loader = _TestableBiasedLoader(
        bias_l6_prob=1.0, bias_sacrum_prob=1.0, bias_enabled=False
    )
    loader.get_bbox(data_shape=(100, 100, 100), force_fg=True,
                     class_locations=LUMB_CASE)
    call = loader.get_bbox_calls[0]
    assert call["overwrite_class"] is None
    assert loader._bias_n_force_fg == 0

def test_mixin_class_locations_none():
    """class_locations=None means case has no foreground; pass through."""
    loader = _TestableBiasedLoader(bias_l6_prob=1.0, bias_sacrum_prob=1.0)
    loader.get_bbox(data_shape=(100, 100, 100), force_fg=True,
                     class_locations=None)
    call = loader.get_bbox_calls[0]
    assert call["overwrite_class"] is None


def test_mixin_lumb_at_prob_1_forces_l6():
    loader = _TestableBiasedLoader(bias_l6_prob=1.0, bias_sacrum_prob=0.0)
    loader.get_bbox(data_shape=(100, 100, 100), force_fg=True,
                     class_locations=LUMB_CASE)
    call = loader.get_bbox_calls[0]
    assert call["overwrite_class"] == L6_LABEL
    assert loader._bias_n_force_fg == 1
    assert loader._bias_n_l6_returned == 1
    assert loader._bias_n_sacrum_returned == 0

def test_mixin_sacr_count_at_prob_1_forces_sacrum():
    loader = _TestableBiasedLoader(bias_l6_prob=0.0, bias_sacrum_prob=1.0)
    loader.get_bbox(data_shape=(100, 100, 100), force_fg=True,
                     class_locations=SACR_COUNT_CASE)
    call = loader.get_bbox_calls[0]
    assert call["overwrite_class"] == SACRUM_LABEL
    assert loader._bias_n_sacrum_returned == 1
    assert loader._bias_n_l6_returned == 0

def test_mixin_normal_with_prob_1_no_force():
    """Even with bias_l6_prob=1.0, normal cases never satisfy the
    signature, so overwrite_class stays None and we count a uniform
    fallback."""
    loader = _TestableBiasedLoader(bias_l6_prob=1.0, bias_sacrum_prob=1.0)
    loader.get_bbox(data_shape=(100, 100, 100), force_fg=True,
                     class_locations=NORMAL_CASE)
    call = loader.get_bbox_calls[0]
    assert call["overwrite_class"] is None
    assert loader._bias_n_force_fg == 1
    assert loader._bias_n_l6_returned == 0
    assert loader._bias_n_uniform_fallback == 1


def test_mixin_diagnostic_summary_keys():
    loader = _TestableBiasedLoader()
    d = loader.bias_diagnostic_summary()
    assert set(d.keys()) >= {
        "n_force_fg", "n_l6_returned", "n_sacrum_returned",
        "n_uniform_fallback", "bias_enabled",
        "bias_l6_prob", "bias_sacrum_prob",
    }
    assert d["n_force_fg"] == 0
    assert d["bias_enabled"] is True
    assert d["bias_l6_prob"] == 0.60  # default

def test_mixin_counters_accumulate_across_calls():
    loader = _TestableBiasedLoader(bias_l6_prob=1.0, bias_sacrum_prob=1.0)
    # 3 lumb, 2 sacr_count, 1 normal (uniform fallback)
    for _ in range(3):
        loader.get_bbox((100,)*3, True, LUMB_CASE)
    for _ in range(2):
        loader.get_bbox((100,)*3, True, SACR_COUNT_CASE)
    loader.get_bbox((100,)*3, True, NORMAL_CASE)

    assert loader._bias_n_force_fg == 6
    assert loader._bias_n_l6_returned == 3
    assert loader._bias_n_sacrum_returned == 2
    assert loader._bias_n_uniform_fallback == 1


# v19.1: shared cross-process counter tests
from lstv_biased_dataloader import (  # noqa: E402
    read_shared_bias_counters,
    reset_shared_bias_counters,
    _SHARED_COUNTERS_AVAILABLE,
)


def test_shared_counters_module_helpers_present():
    """The new module-level helpers should be importable and callable."""
    assert callable(read_shared_bias_counters)
    assert callable(reset_shared_bias_counters)
    # On any reasonable Python install, multiprocessing.Value should
    # have succeeded.
    assert _SHARED_COUNTERS_AVAILABLE is True


def test_shared_counters_increment_on_bias_trigger():
    """Hitting the mixin's get_bbox with a triggering signature should
    bump the SHARED counter, not just the per-instance one."""
    reset_shared_bias_counters()
    before = read_shared_bias_counters()
    assert before == {
        "n_force_fg": 0, "n_l6_returned": 0,
        "n_sacrum_returned": 0, "n_uniform_fallback": 0,
    }

    loader = _TestableBiasedLoader(bias_l6_prob=1.0, bias_sacrum_prob=0.0)
    for _ in range(7):
        loader.get_bbox((100,)*3, True, LUMB_CASE)

    after = read_shared_bias_counters()
    # Diff against snapshot — no global state assumptions
    assert after["n_force_fg"] - before["n_force_fg"] == 7
    assert after["n_l6_returned"] - before["n_l6_returned"] == 7
    assert after["n_sacrum_returned"] - before["n_sacrum_returned"] == 0


def test_shared_counters_reset():
    """reset_shared_bias_counters should zero everything atomically."""
    loader = _TestableBiasedLoader(bias_l6_prob=1.0, bias_sacrum_prob=0.0)
    # Pump some counts
    for _ in range(5):
        loader.get_bbox((100,)*3, True, LUMB_CASE)
    after_pump = read_shared_bias_counters()
    assert after_pump["n_force_fg"] >= 5

    reset_shared_bias_counters()
    after_reset = read_shared_bias_counters()
    assert all(v == 0 for v in after_reset.values()), (
        f"reset should zero everything; got {after_reset}"
    )


def test_shared_counters_pass_through_when_disabled():
    """bias_enabled=False -> no shared counter increments."""
    reset_shared_bias_counters()
    loader = _TestableBiasedLoader(
        bias_l6_prob=1.0, bias_sacrum_prob=1.0, bias_enabled=False
    )
    for _ in range(10):
        loader.get_bbox((100,)*3, True, LUMB_CASE)

    after = read_shared_bias_counters()
    assert all(v == 0 for v in after.values()), (
        f"with bias_enabled=False shared counters should not increment; "
        f"got {after}"
    )


# Env-var resolution tests. Use _EnvOverride context manager so we
# don't pollute the test environment.

def test_env_explicit_arg_overrides_env():
    with _EnvOverride() as env:
        env.setenv("SPINESURG_LSTV_BIAS_L6_PROB", "0.99")
        loader = _TestableBiasedLoader(bias_l6_prob=0.10)
        assert loader.bias_l6_prob == 0.10

def test_env_overrides_default():
    with _EnvOverride() as env:
        env.setenv("SPINESURG_LSTV_BIAS_L6_PROB", "0.85")
        loader = _TestableBiasedLoader()
        assert loader.bias_l6_prob == 0.85

def test_env_default_applies():
    # No env set; expect documented defaults
    with _EnvOverride() as env:
        env.setenv("SPINESURG_LSTV_BIAS_L6_PROB", "")  # cleared
        env.setenv("SPINESURG_LSTV_BIAS_SACRUM_PROB", "")
        loader = _TestableBiasedLoader()
        assert loader.bias_l6_prob == 0.60
        assert loader.bias_sacrum_prob == 0.50

def test_env_disable_bias():
    with _EnvOverride() as env:
        env.setenv("SPINESURG_LSTV_BIAS_ENABLED", "0")
        loader = _TestableBiasedLoader()
        assert loader.bias_enabled is False

def test_env_truthy_variants():
    for v in ("1", "true", "yes", "ON"):
        with _EnvOverride() as env:
            env.setenv("SPINESURG_LSTV_BIAS_ENABLED", v)
            loader = _TestableBiasedLoader()
            assert loader.bias_enabled is True, f"failed for env value {v!r}"

def test_env_falsy_variants():
    for v in ("0", "false", "no", "OFF"):
        with _EnvOverride() as env:
            env.setenv("SPINESURG_LSTV_BIAS_ENABLED", v)
            loader = _TestableBiasedLoader()
            assert loader.bias_enabled is False, f"failed for env value {v!r}"

def test_env_invalid_value_falls_to_default():
    """Garbage env value shouldn't crash; falls through to default."""
    with _EnvOverride() as env:
        env.setenv("SPINESURG_LSTV_BIAS_L6_PROB", "not_a_number")
        loader = _TestableBiasedLoader()
        assert loader.bias_l6_prob == 0.60


# =============================================================================
# Layer 3: Production loader tests (require nnU-Net)
# =============================================================================
# These tests early-return with a SKIP message if nnU-Net isn't available.
# In pytest mode they'd be marked with @pytest.mark.skipif; without pytest
# we just check the flag and return.

class _SkipTest(Exception):
    """Raised by tests to signal 'skip' to the standalone runner.
    pytest also recognizes this if pytest.skip is unavailable in the env."""
    pass


def _skip_if_no_nnunet():
    if not _NNUNET_AVAILABLE:
        raise _SkipTest("nnU-Net v2 not importable")


def test_production_loader_class_exists():
    _skip_if_no_nnunet()
    from lstv_biased_dataloader import LSTVBiasedDataLoader3D
    assert LSTVBiasedDataLoader3D is not None

def test_production_loader_mro_contains_mixin_and_base():
    _skip_if_no_nnunet()
    from lstv_biased_dataloader import LSTVBiasedDataLoader3D
    from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
    mro = LSTVBiasedDataLoader3D.__mro__
    assert LSTVBiasMixin in mro
    assert nnUNetDataLoader3D in mro
    # Mixin must come BEFORE nnU-Net base in MRO so its get_bbox runs first.
    assert mro.index(LSTVBiasMixin) < mro.index(nnUNetDataLoader3D)

def test_production_loader_class_is_picklable():
    """MultiThreadedAugmenter pickles the loader to send to workers.
    Class itself must be picklable for fork/spawn to succeed."""
    _skip_if_no_nnunet()
    from lstv_biased_dataloader import LSTVBiasedDataLoader3D
    b = pickle.dumps(LSTVBiasedDataLoader3D)
    cls = pickle.loads(b)
    assert cls is LSTVBiasedDataLoader3D

def test_production_loader_get_bbox_method_resolution():
    """LSTVBiasedDataLoader3D.get_bbox must resolve to the mixin's
    override, not the base's method."""
    _skip_if_no_nnunet()
    from lstv_biased_dataloader import LSTVBiasedDataLoader3D
    for cls in LSTVBiasedDataLoader3D.__mro__:
        if "get_bbox" in cls.__dict__:
            assert cls is LSTVBiasMixin, (
                f"get_bbox resolved to {cls.__name__}, expected LSTVBiasMixin")
            return
    raise AssertionError("get_bbox not found anywhere in MRO")


# =============================================================================
# Standalone runner (matches tests/test_subtype_refinement.py style)
# =============================================================================

if __name__ == "__main__":
    tests = [(name, obj) for name, obj in globals().items()
              if name.startswith("test_") and callable(obj)]
    tests.sort()
    failed = []
    skipped = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except _SkipTest as e:
            print(f"  SKIP  {name}: {e}")
            skipped.append(name)
        except AssertionError as e:
            msg = str(e).splitlines()[0] if str(e) else "(no message)"
            print(f"  FAIL  {name}: {msg}")
            failed.append(name)
        except Exception as e:
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
            failed.append(name)
    print()
    print(f"  ran {len(tests)} tests, {len(failed)} failed, "
          f"{len(skipped)} skipped")
    sys.exit(1 if failed else 0)
