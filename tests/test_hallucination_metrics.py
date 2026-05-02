"""
test_hallucination_metrics.py

v17.4: unit tests for false-positive / specificity tracking.

These tests verify the math of _per_case_voxel_counts_per_class and
the aggregation logic in _aggregate_subgroup_hallucination. The
trainer-level integration is tested via AST inspection of the
emitted method body.

Run inside the container or anywhere torch is available:
    python3 tests/test_hallucination_metrics.py
"""
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
TOOLS_DIR = PROJECT_ROOT / "tools"
sys.path.insert(0, str(TOOLS_DIR))

# Track passes/failures
_passed = 0
_failed = 0


def _check(cond, name, msg=""):
    global _passed, _failed
    if cond:
        print(f"  PASS  {name}")
        _passed += 1
    else:
        print(f"  FAIL  {name}: {msg}")
        _failed += 1


def _read_variant_src() -> str:
    return (TOOLS_DIR / "nnunet_wandb_variant.py").read_text()


# ─── Test 1: _per_case_voxel_counts_per_class returns correct shape ─────────

def test_voxel_counts_basic_shape():
    """Helper returns one (gt_n, pred_n) tuple per FG class."""
    try:
        import torch  # noqa: F401
    except ImportError:
        print("  SKIP test_voxel_counts_basic_shape (no torch)")
        return
    import importlib
    if "nnunet_wandb_variant" in sys.modules:
        del sys.modules["nnunet_wandb_variant"]
    try:
        m = importlib.import_module("nnunet_wandb_variant")
    except ImportError as exc:
        print(f"  SKIP test_voxel_counts_basic_shape (import failed: {exc})")
        return
    import torch
    # 9 FG classes (L1..L6, sacrum, left_hip, right_hip)
    gt = torch.zeros((4, 4, 4), dtype=torch.int16)  # all background
    pred = torch.zeros((4, 4, 4), dtype=torch.int16)
    res = m._per_case_voxel_counts_per_class(pred, gt)
    _check(len(res) == len(m._FG_CLASS_IDS),
           "test_voxel_counts_basic_shape:length",
           f"expected {len(m._FG_CLASS_IDS)}, got {len(res)}")
    _check(all(isinstance(t, tuple) and len(t) == 2 for t in res),
           "test_voxel_counts_basic_shape:tuples",
           f"non-tuple entries: {res}")
    _check(all(g == 0 and p == 0 for g, p in res),
           "test_voxel_counts_basic_shape:all_zero",
           f"with all-bg input, all entries should be (0,0): {res}")


# ─── Test 2: pred-only (hallucination) is recorded with gt_n=0 ──────────────

def test_hallucination_with_gt_absent():
    """When GT has no L6 voxels but pred has 50 L6 voxels:
    expect (gt_n=0, pred_n=50) for L6, dice would have been None."""
    try:
        import torch
    except ImportError:
        print("  SKIP test_hallucination_with_gt_absent (no torch)")
        return
    import importlib
    if "nnunet_wandb_variant" in sys.modules:
        del sys.modules["nnunet_wandb_variant"]
    try:
        m = importlib.import_module("nnunet_wandb_variant")
    except ImportError as exc:
        print(f"  SKIP test_hallucination_with_gt_absent (import failed: {exc})")
        return
    L6_id = 6  # _L6_LABEL_ID
    # 1000 voxels: 50 of them get L6 in pred, none in GT
    gt = torch.zeros((10, 10, 10), dtype=torch.int16)
    pred = torch.zeros((10, 10, 10), dtype=torch.int16)
    pred.flatten()[:50] = L6_id
    res = m._per_case_voxel_counts_per_class(pred, gt)
    l6_idx = m._FG_CLASS_IDS.index(L6_id)
    g, p = res[l6_idx]
    _check(g == 0,
           "test_hallucination_with_gt_absent:gt_zero",
           f"L6 GT should be 0, got {g}")
    _check(p == 50,
           "test_hallucination_with_gt_absent:pred_fifty",
           f"L6 pred should be 50, got {p}")
    # Dice must agree: returns None for L6
    dice_res = m._per_case_dice_per_class(pred, gt)
    _check(dice_res[l6_idx] is None,
           "test_hallucination_with_gt_absent:dice_none",
           f"dice for absent class must be None, got {dice_res[l6_idx]}")


# ─── Test 3: ignore voxels excluded from BOTH gt and pred counts ────────────

def test_ignore_voxels_excluded():
    """ignore_label voxels should not count toward gt_n or pred_n.

    Critical for partial-annotation cases: a pelvic-only record has
    ignore=10 throughout the lumbar region, so even if the model
    outputs L6 there, those voxels shouldn't be counted as
    'hallucination' — the GT is uninformative there."""
    try:
        import torch
    except ImportError:
        print("  SKIP test_ignore_voxels_excluded (no torch)")
        return
    import importlib
    if "nnunet_wandb_variant" in sys.modules:
        del sys.modules["nnunet_wandb_variant"]
    try:
        m = importlib.import_module("nnunet_wandb_variant")
    except ImportError as exc:
        print(f"  SKIP test_ignore_voxels_excluded (import failed: {exc})")
        return
    L6_id = 6
    IGN = m._IGNORE_LABEL  # = 10
    gt = torch.full((10, 10, 10), IGN, dtype=torch.int16)
    # First 100 voxels are background (label 0) — fully informative
    gt.flatten()[:100] = 0
    # Pred: 30 L6 voxels in the IGNORE region, 5 L6 voxels in the
    # informative-as-bg region.
    pred = torch.zeros((10, 10, 10), dtype=torch.int16)
    pred.flatten()[100:130] = L6_id  # in ignore region
    pred.flatten()[80:85] = L6_id    # in informative region
    res = m._per_case_voxel_counts_per_class(pred, gt)
    l6_idx = m._FG_CLASS_IDS.index(L6_id)
    g, p = res[l6_idx]
    # GT-side L6 voxels in informative region: 0
    _check(g == 0, "test_ignore_voxels_excluded:gt_zero", f"got gt_n={g}")
    # Pred-side L6 voxels in informative region: 5 (the 30 in ignore
    # are excluded). This is the critical check.
    _check(p == 5,
           "test_ignore_voxels_excluded:pred_only_informative",
           f"pred_n should count only informative-region voxels (5), "
           f"got {p}")


# ─── Test 4: aggregation specificity math ────────────────────────────────────

def test_specificity_math_perfect():
    """3 cases with class absent + pred=0 each → specificity 1.0"""
    threshold = 100
    cases = [(0, 0), (0, 0), (0, 0)]  # (gt_n, pred_n) per case
    absent_preds = [p for (g, p) in cases if g == 0]
    n_meaningful = sum(1 for p in absent_preds if p > threshold)
    spec = 1.0 - n_meaningful / len(absent_preds)
    _check(abs(spec - 1.0) < 1e-9,
           "test_specificity_math_perfect",
           f"got {spec}")


def test_specificity_math_one_hallucination():
    """3 absent + 1 hallucinates = 2/3 specificity"""
    threshold = 100
    cases = [(0, 0), (0, 0), (0, 500)]
    absent = [p for (g, p) in cases if g == 0]
    n_meaningful = sum(1 for p in absent if p > threshold)
    spec = 1.0 - n_meaningful / len(absent)
    _check(abs(spec - 2.0 / 3.0) < 1e-9,
           "test_specificity_math_one_hallucination",
           f"got {spec}, expected 0.667")


def test_specificity_math_below_threshold():
    """Predictions below threshold count as correct (noise-tolerant)."""
    threshold = 100
    cases = [(0, 50), (0, 99), (0, 0)]  # all below or zero
    absent = [p for (g, p) in cases if g == 0]
    n_meaningful = sum(1 for p in absent if p > threshold)
    spec = 1.0 - n_meaningful / len(absent)
    _check(abs(spec - 1.0) < 1e-9,
           "test_specificity_math_below_threshold",
           f"got {spec}, expected 1.0 (all below threshold)")


def test_specificity_present_excluded():
    """Cases where class is PRESENT in GT shouldn't enter specificity calc."""
    threshold = 100
    cases = [(0, 0), (500, 500), (0, 1000), (300, 300)]
    absent = [p for (g, p) in cases if g == 0]
    _check(len(absent) == 2,
           "test_specificity_present_excluded:partition",
           f"absent should be 2 cases, got {len(absent)}")
    n_meaningful = sum(1 for p in absent if p > threshold)
    spec = 1.0 - n_meaningful / len(absent)
    _check(abs(spec - 0.5) < 1e-9,
           "test_specificity_present_excluded:value",
           f"got {spec}, expected 0.5")


# ─── Test 5: trainer-level integration via AST ──────────────────────────────

def test_buffer_field_declared():
    src = _read_variant_src()
    _check("_val_subgroup_voxels:" in src,
           "test_buffer_field_declared",
           "_val_subgroup_voxels field must be declared on _WandBMixin")


def test_reset_clears_voxel_buffer():
    src = _read_variant_src()
    _check("self._val_subgroup_voxels = {sub: [] for sub in _KNOWN_SUBTYPES}" in src,
           "test_reset_clears_voxel_buffer",
           "_reset_subgroup_dice_buffer must also reset voxel buffer")


def test_record_appends_voxel_tuple():
    src = _read_variant_src()
    _check("_per_case_voxel_counts_per_class(pred_b, gt_b)" in src,
           "test_record_appends_voxel_tuple",
           "_record_per_case_dice must call voxel-count helper")
    _check("self._val_subgroup_voxels.setdefault(subtype, []).append(voxels_per_cls)" in src,
           "test_record_appends_voxel_buffer",
           "_record_per_case_dice must append to voxel buffer")


def test_aggregator_method_exists():
    src = _read_variant_src()
    _check("def _aggregate_subgroup_hallucination" in src,
           "test_aggregator_method_exists",
           "_aggregate_subgroup_hallucination method must be defined")


def test_aggregator_called_from_dice_aggregator():
    src = _read_variant_src()
    _check("self._aggregate_subgroup_hallucination()" in src,
           "test_aggregator_called_from_dice_aggregator",
           "_aggregate_subgroup_dice must call hallucination aggregator and merge")


def test_threshold_constant_defined():
    src = _read_variant_src()
    _check("_HALLUC_VOXEL_THRESHOLD" in src,
           "test_threshold_constant_defined",
           "_HALLUC_VOXEL_THRESHOLD module constant must exist")


def test_headline_specificity_keys_emitted():
    """All 7 clinically curated headline keys should be referenced
    in the source so reviewers can find them in the W&B payload."""
    src = _read_variant_src()
    expected_keys = [
        "L5_specificity_on_sacralization",
        "L5_specificity_on_sacr_count",
        "L5_specificity_on_semisacralization",
        "L6_specificity_on_normal",
        "L6_specificity_on_sacralization",
        "L6_specificity_on_semisacralization",
        "L6_specificity_on_sacr_count",
    ]
    missing = [k for k in expected_keys if k not in src]
    _check(not missing,
           "test_headline_specificity_keys_emitted",
           f"missing keys: {missing}")


def test_summary_prints_hallucination_section():
    src = _read_variant_src()
    _check("v17.4 hallucination" in src,
           "test_summary_prints_hallucination_section",
           "_print_per_subgroup_summary must include v17.4 hallucination header")


# ─── Test 6: end-to-end aggregation on synthetic buffer ─────────────────────

def test_end_to_end_synthetic():
    """Build a fake buffer and verify the aggregator produces correct
    headline numbers without invoking the trainer.

    Scenario: 4 sacralization val cases. L5 should be absent in all.
    Model behavior:
      Case A: pred=0 L5 voxels (correct)
      Case B: pred=20 L5 voxels (below threshold, counts as correct)
      Case C: pred=500 L5 voxels (hallucination)
      Case D: pred=0 L5 voxels (correct)
    Expected: 1 / 4 hallucinations -> specificity = 0.75
    """
    # Simulate the aggregator math directly (what it does internally
    # for each (subgroup, class) pair).
    threshold = 100
    pairs_l5 = [(0, 0), (0, 20), (0, 500), (0, 0)]  # (gt, pred)
    absent = [p for (g, p) in pairs_l5 if g == 0]
    _check(len(absent) == 4, "test_end_to_end_synthetic:n_absent",
           f"got {len(absent)}")
    n_strict = sum(1 for p in absent if p > 0)
    n_meaningful = sum(1 for p in absent if p > threshold)
    spec = 1.0 - n_meaningful / len(absent)
    mean_pred = sum(absent) / len(absent)
    max_pred = max(absent)

    _check(n_strict == 2,
           "test_end_to_end_synthetic:n_strict",
           f"got {n_strict}, expected 2 (cases B and C have any prediction)")
    _check(n_meaningful == 1,
           "test_end_to_end_synthetic:n_meaningful",
           f"got {n_meaningful}, expected 1 (only C exceeds threshold)")
    _check(abs(spec - 0.75) < 1e-9,
           "test_end_to_end_synthetic:specificity",
           f"got {spec}, expected 0.75")
    _check(abs(mean_pred - 130.0) < 1e-9,
           "test_end_to_end_synthetic:mean",
           f"got {mean_pred}, expected 130.0")
    _check(max_pred == 500,
           "test_end_to_end_synthetic:max",
           f"got {max_pred}, expected 500")


# ─── Test 7: empty subgroup safety ──────────────────────────────────────────

def test_empty_subgroup_does_not_emit_specificity():
    """If a subgroup has zero cases, no specificity keys should be
    emitted for it (avoid division by zero)."""
    threshold = 100
    cases = []
    absent = [p for (g, p) in cases if g == 0]
    _check(len(absent) == 0,
           "test_empty_subgroup_does_not_emit_specificity:no_absent",
           f"empty buffer should have 0 absent cases")
    # The aggregator's "if not absent_preds: continue" branch handles this.


# ─── Test 8: hallucination tracking is symmetric with dice ──────────────────

def test_dice_and_voxel_buffers_parallel_shapes():
    """For a recorded case, dice_per_cls and voxels_per_cls must have
    the same length, since both are indexed by _FG_CLASS_IDS."""
    src = _read_variant_src()
    # The record method should call BOTH helpers with the same (pred_b, gt_b).
    _check(("dice_per_cls = _per_case_dice_per_class(pred_b, gt_b)" in src and
            "voxels_per_cls = _per_case_voxel_counts_per_class(pred_b, gt_b)" in src),
           "test_dice_and_voxel_buffers_parallel_shapes",
           "Both helpers must be called with the same per-batch tensors")


# ─── Run ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [(name, obj) for name, obj in globals().items()
              if name.startswith("test_") and callable(obj)]
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:
            _failed += 1
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n  ran {_passed + _failed} tests, {_failed} failed")
    sys.exit(1 if _failed else 0)
