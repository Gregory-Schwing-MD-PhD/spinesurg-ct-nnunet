"""
test_dataset803_pipeline.py — tests for the Dataset803 (merged-label)
convert + trainer changeset.

Coverage
--------
1. Convert script (VerSe-native source):
   - _build_verse_remap_lut()  — LUT construction is correct under the
                            published LABEL_REMAP_MERGED_FROM_VERSE, and
                            every UNLISTED VerSe id defaults to 0
                            (background), not passthrough.
   - _remap_and_write_label() — end-to-end NIfTI rewrite preserves
                                 affine, collapses VerSe L5 (24) + L6 (25)
                                 into last_lumbar (5), drops non-training
                                 VerSe ids (e.g. thoracic) to background.
   - LABEL_NAMES + LABEL_REMAP_MERGED_FROM_VERSE are MUTUALLY CONSISTENT
     (every LUT destination value appears as a LABEL_NAMES value).

2. Trainer aggregation:
   - _build_ce_class_weights() — returns a 9-element tensor with the
                                  documented v20 weights at the
                                  documented positions.
   - _per_case_dice_per_class() — produces correct per-class Dice
                                   under the contiguous-ID label
                                   scheme (background, L1-L4,
                                   last_lumbar, sacrum, hips).
   - _per_case_confusion_by_gt() — produces correct GT-direction
                                    confusion vectors for synthetic
                                    pred/GT volumes.
   - _per_case_confusion_by_pred() — symmetric PRED-direction case.
   - _aggregate_block_by_gt() / _aggregate_block_by_pred() — produce
     the documented W&B key set on a synthetic per-case buffer.
   - End-to-end: build a buffer with two synthetic lumb cases and one
     sacr_count case, run the three v20 confusion blocks, and assert
     the W&B payload contains EXACTLY the documented headline keys —
     no per-(subtype, class) spam.
   - Headline-target spec validation: every target_cid in
     _HEADLINE_TARGETS_* references a class that exists in the v20
     label scheme.

Run
---
    python test_dataset803_pipeline.py
or with pytest:
    pytest test_dataset803_pipeline.py -v

Both modes work because each test is a self-contained function with
plain assert statements; the __main__ block discovers and runs them
when pytest is unavailable.

These tests do NOT require nnU-Net or W&B to be installed. The
convert tests need nibabel + numpy; the trainer tests need numpy +
torch. Both run in seconds on CPU.
"""
from __future__ import annotations

import importlib
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Test fixture: import the modules under test
# ──────────────────────────────────────────────────────────────────────
#
# These fixtures are declared two ways so the file works with both:
#
#   1. `python test_dataset803_pipeline.py`
#      The bottom-of-file _run_tests() runner inspects each test's
#      signature and passes the cached fixtures by name.
#
#   2. `pytest test_dataset803_pipeline.py`
#      The pytest @pytest.fixture decorators below register
#      `convert_mod` and `trainer_mod` as session-scoped fixtures
#      that pytest auto-injects.
#
# If pytest isn't installed, the @pytest.fixture decorators are
# replaced with no-op decorators so the standalone runner still works.

try:
    import pytest as _pytest
    _PYTEST_FIXTURE = _pytest.fixture(scope="session")
except ImportError:
    def _PYTEST_FIXTURE(fn):
        return fn


def _import_under_test():
    """Import convert_hf_to_nnunet and nnunet_wandb_variant. Returns
    (convert_mod, trainer_mod).

    Module discovery: searches a list of candidate directories in order:
      1. The directory containing this test file
      2. <test_dir>/tools/        (project root layout)
      3. <test_dir>/../tools/     (test moved into tools/)
      4. CWD/tools/               (running from project root)

    On import, the loaded path of each module is printed so a stale-
    version-on-PATH problem is immediately visible. Then a v20 marker
    check verifies key attributes exist; if not, an explicit error is
    raised that names which file was loaded and what's missing,
    instead of letting 25 individual tests fail with confusing
    AttributeErrors.
    """
    here = Path(__file__).resolve().parent
    candidates: List[Path] = [here]
    for sub in (here / "tools", here.parent / "tools",
                 Path.cwd() / "tools"):
        if sub != here and sub not in candidates and sub.is_dir():
            candidates.append(sub)
    # Front-load all candidates onto sys.path so importlib finds the
    # most local copy first.
    for cand in reversed(candidates):
        cand_str = str(cand)
        if cand_str not in sys.path:
            sys.path.insert(0, cand_str)

    # Import convert script (no nnU-Net dependency)
    convert_mod = importlib.import_module("convert_hf_to_nnunet")
    print(f"  loaded convert_hf_to_nnunet from: {convert_mod.__file__}")

    # nnunetv2 import strategy: prefer the real installed package
    # (inside the Singularity training container) so the test
    # validates the trainer file against the actual installed
    # nnunetv2 version. Fall back to stubs only when nnunetv2 is
    # unavailable (e.g., running tests on a login node before
    # launching a SLURM job). The stubs install enough surface for
    # `nnunet_wandb_variant.py` to import cleanly; tests still
    # exercise pure-Python helpers, not the trainer subclasses.
    if "nnunetv2" not in sys.modules:
        try:
            # Real install path (container)
            import nnunetv2  # noqa: F401
            from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer  # noqa: F401
            from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import (  # noqa: F401
                nnUNetTrainer_5epochs, nnUNetTrainer_100epochs,
            )
        except ImportError:
            # Stub fallback (login node / dev)
            nnunetv2 = type(sys)("nnunetv2")
            trainer_pkg = type(sys)("nnunetv2.training.nnUNetTrainer.nnUNetTrainer")
            class _StubTrainer: pass
            trainer_pkg.nnUNetTrainer = _StubTrainer
            xep_pkg = type(sys)("nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs")
            class _StubT5(_StubTrainer): pass
            class _StubT100(_StubTrainer): pass
            xep_pkg.nnUNetTrainer_5epochs = _StubT5
            xep_pkg.nnUNetTrainer_100epochs = _StubT100
            for name, mod in [
                ("nnunetv2", nnunetv2),
                ("nnunetv2.training", type(sys)("nnunetv2.training")),
                ("nnunetv2.training.nnUNetTrainer", type(sys)("nnunetv2.training.nnUNetTrainer")),
                ("nnunetv2.training.nnUNetTrainer.nnUNetTrainer", trainer_pkg),
                ("nnunetv2.training.nnUNetTrainer.variants", type(sys)("nnunetv2.training.nnUNetTrainer.variants")),
                ("nnunetv2.training.nnUNetTrainer.variants.training_length", type(sys)("nnunetv2.training.nnUNetTrainer.variants.training_length")),
                ("nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs", xep_pkg),
            ]:
                sys.modules[name] = mod

    trainer_mod = importlib.import_module("nnunet_wandb_variant")
    print(f"  loaded nnunet_wandb_variant from: {trainer_mod.__file__}")

    # ── v20 marker check ──────────────────────────────────────────────
    # Each loaded module must carry markers that uniquely identify the
    # v20 release. If any are missing, the user has an older version on
    # sys.path (e.g. masking tools/ from the project root). Surface the
    # exact problem instead of letting 25 test failures bury it.
    convert_required = ["_build_verse_remap_lut", "LABEL_REMAP_MERGED_FROM_VERSE", "LABEL_NAMES"]
    trainer_required = [
        "_L4_LABEL_ID", "_LAST_LUMBAR_LABEL_ID",
        "_HEADLINE_TARGETS_LL_GT_ON_LUMB",
        "_HEADLINE_TARGETS_L4_GT_ON_SACR_COUNT",
        "_HEADLINE_TARGETS_PRED_LL_ON_SACR_COUNT",
        "_aggregate_block_by_gt", "_aggregate_block_by_pred",
        "_per_case_confusion_by_gt", "_per_case_confusion_by_pred",
    ]
    missing_convert = [a for a in convert_required if not hasattr(convert_mod, a)]
    missing_trainer = [a for a in trainer_required if not hasattr(trainer_mod, a)]

    if missing_convert or missing_trainer:
        msg_lines = [
            "",
            "=" * 72,
            "ERROR: stale version of test fixtures detected on sys.path",
            "=" * 72,
            "",
            "Loaded files:",
            f"  convert_hf_to_nnunet : {convert_mod.__file__}",
            f"  nnunet_wandb_variant : {trainer_mod.__file__}",
            "",
        ]
        if missing_convert:
            msg_lines.append(f"convert_hf_to_nnunet is MISSING attrs: {missing_convert}")
            msg_lines.append("  -> the loaded convert script is older than v20.")
        if missing_trainer:
            msg_lines.append(f"nnunet_wandb_variant is MISSING attrs: {missing_trainer}")
            msg_lines.append("  -> the loaded trainer is older than v20 OR is a local fork.")
        msg_lines += [
            "",
            "What to do:",
            "  1. Make sure the v20 files are in your tools/ directory:",
            "       <project_root>/tools/convert_hf_to_nnunet.py",
            "       <project_root>/tools/nnunet_wandb_variant.py",
            "  2. Remove any older copies that may shadow them, e.g.:",
            "       rm <project_root>/convert_hf_to_nnunet.py",
            "       rm <project_root>/nnunet_wandb_variant.py",
            "     (these would be picked up before tools/ when running the",
            "      test from the project root)",
            "  3. Re-run the test. The test searches tools/ automatically.",
            "",
            "Search order used by this test:",
        ]
        for c in candidates:
            msg_lines.append(f"    {c}")
        msg_lines.append("=" * 72)
        raise RuntimeError("\n".join(msg_lines))

    print(f"  ✓ v20 markers present in both modules")
    return convert_mod, trainer_mod


# Cached imports so we don't re-import per test (the fork stubs are
# module-level state and only need to happen once).
_CACHED_MODS: List = []


def _modules():
    if not _CACHED_MODS:
        _CACHED_MODS.extend(_import_under_test())
    return _CACHED_MODS[0], _CACHED_MODS[1]


@_PYTEST_FIXTURE
def convert_mod():
    return _modules()[0]


@_PYTEST_FIXTURE
def trainer_mod():
    return _modules()[1]


# ──────────────────────────────────────────────────────────────────────
# Convert script tests
# ──────────────────────────────────────────────────────────────────────

def test_lut_published_remap_correct(convert_mod):
    """LUT correctly implements LABEL_REMAP_MERGED_FROM_VERSE: the
    VerSe-native lumbar bodies (20-25), sacrum (26), hips (30/31) and
    ignore (255) map to the merged 9-class scheme; EVERY other id -> 0."""
    lut = convert_mod._build_verse_remap_lut(
        convert_mod.LABEL_REMAP_MERGED_FROM_VERSE)
    # LUT spans the full VerSe range so ignore (255) is indexable.
    assert len(lut) == 256, f"LUT length should be 256 got {len(lut)}"
    expected = {
        0: 0, 20: 1, 21: 2, 22: 3, 23: 4, 24: 5, 25: 5,
        26: 6, 30: 7, 31: 8, 255: 9,
    }
    for src, dst in expected.items():
        assert int(lut[src]) == dst, (
            f"VerSe {src} -> {int(lut[src])}, expected {dst}")
    # Non-training VerSe ids (thoracic/cervical/coccyx/T13/S1/femur/rib/
    # soft-tissue/lumbar-rib) MUST drop to background, not pass through.
    for src in (5, 15, 27, 28, 29, 32, 33, 40, 60, 74, 75):
        assert int(lut[src]) == 0, (
            f"non-training VerSe {src} must -> 0, got {int(lut[src])}")


def test_lut_round_trip_through_array(convert_mod):
    """Vectorized indexing arr -> lut[arr] yields the documented VerSe
    merge: L5 (24) and L6 (25) both end up at label 5; sacrum 26->6 etc.,
    and a stray thoracic id -> 0."""
    lut = convert_mod._build_verse_remap_lut(
        convert_mod.LABEL_REMAP_MERGED_FROM_VERSE)
    arr = np.array([0, 20, 21, 22, 23, 24, 25, 26, 30, 31, 255, 15],
                   dtype=np.int32)
    out = lut[arr]
    assert out.tolist() == [0, 1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 0]


def test_lut_rejects_out_of_range_remap_source(convert_mod):
    """Building a LUT with a remap key beyond _MAX_SOURCE_LABEL (255)
    should raise — defensive against silent off-by-one in label edits."""
    bogus = {999: 1}  # 999 way above _MAX_SOURCE_LABEL=255
    try:
        convert_mod._build_verse_remap_lut(bogus)
    except ValueError as exc:
        assert "outside expected range" in str(exc), (
            f"unexpected message: {exc}"
        )
        return
    raise AssertionError("expected ValueError for out-of-range remap source")


def test_label_names_contiguous_and_consistent_with_remap(convert_mod):
    """LABEL_NAMES values + ignore form a contiguous range 0..N, and
    every LUT destination is a valid foreground class id under the
    merged scheme. This catches the subtle bug of leaving a gap in label
    IDs (which would break the trainer's per-class dice computation —
    pred_argmax channel indices have to match GT label values)."""
    lns = convert_mod.LABEL_NAMES
    fg_values = sorted(v for k, v in lns.items() if k != "ignore")
    assert fg_values[0] == 0, f"background should be 0, got {fg_values[0]}"
    expected_contiguous = list(range(len(fg_values)))
    assert fg_values == expected_contiguous, (
        f"foreground label IDs are not contiguous: {fg_values}; "
        f"this WILL break per-class dice in the trainer"
    )
    # Every VerSe remap destination ends up at a defined label value.
    lut_dsts = set(convert_mod.LABEL_REMAP_MERGED_FROM_VERSE.values())
    all_dsts = set(lns.values())
    missing = lut_dsts - all_dsts
    assert not missing, (
        f"remap destinations {sorted(missing)} have no entry "
        f"in LABEL_NAMES; rebuilds would mint orphan label values"
    )


def test_remap_and_write_label_nifti_end_to_end(convert_mod):
    """Synthetic VerSe-labeled volume → remap → reload → assertions on
    each label's voxel count. Exercises the full disk-to-disk path
    (nibabel read/write + LUT indexing + uint8 dtype downcast). Includes
    a non-training VerSe id (thoracic T8 = 15) that must vanish to
    background, and VerSe L5 (24) + L6 (25) that must collapse to 5."""
    import nibabel as nib
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / "src.nii.gz"
        dst = tmp / "dst.nii.gz"
        # Each source VerSe id gets two slabs (2 voxels along W) so a
        # miscount is easy to detect. Order: bg + the 10 training ids +
        # one non-training thoracic id (15).
        src_ids = [0, 20, 21, 22, 23, 24, 25, 26, 30, 31, 255, 15]
        arr = np.zeros((10, 10, 2 * len(src_ids)), dtype=np.int32)
        for i, sid in enumerate(src_ids):
            arr[:, :, i * 2:(i + 1) * 2] = sid
        affine = np.eye(4)
        nib.save(nib.Nifti1Image(arr, affine), str(src))

        lut = convert_mod._build_verse_remap_lut(
            convert_mod.LABEL_REMAP_MERGED_FROM_VERSE)
        convert_mod._remap_and_write_label(src, dst, lut)

        out_img = nib.load(str(dst))
        out = np.asarray(out_img.get_fdata(), dtype=np.int32)

        # Affine + dtype assertions
        assert np.allclose(out_img.affine, affine), "affine not preserved"
        assert out_img.get_data_dtype() == np.uint8, (
            f"output dtype should be uint8, got {out_img.get_data_dtype()}"
        )
        # Destination voxel counts under the merged VerSe remap.
        # per source slab-pair = 10*10*2 = 200 voxels.
        #   bg(0): its own 2 slabs + T8(15) 2 slabs = 400
        #   5 (last_lumbar): L5(24) + L6(25) = 400
        #   1-4,6,7,8,9: 200 each
        per_slab = 10 * 10 * 2  # 200
        expected = {
            0: 2 * per_slab,  # background + dropped thoracic (15)
            1: per_slab, 2: per_slab, 3: per_slab, 4: per_slab,
            5: 2 * per_slab,  # VerSe L5 + VerSe L6
            6: per_slab,      # sacrum
            7: per_slab,      # left_hip
            8: per_slab,      # right_hip
            9: per_slab,      # ignore
        }
        actual = {int(v): int((out == v).sum()) for v in np.unique(out)}
        assert actual == expected, (
            f"voxel-count mismatch:\n  expected: {expected}\n  actual:   {actual}"
        )
        # No value above 9; the non-training thoracic id is gone.
        assert int(out.max()) == 9
        assert int((out == 6).sum()) == 200, (
            "channel 6 should contain VerSe sacrum voxels (200)"
        )


# ──────────────────────────────────────────────────────────────────────
# Trainer constants — the foundation everything else builds on
# ──────────────────────────────────────────────────────────────────────

def test_trainer_label_constants_v20(trainer_mod):
    """Module-level label-ID constants match the v20 contiguous scheme."""
    assert trainer_mod._L4_LABEL_ID == 4
    assert trainer_mod._LAST_LUMBAR_LABEL_ID == 5
    assert trainer_mod._SACRUM_LABEL_ID == 6
    assert trainer_mod._IGNORE_LABEL == 9
    assert trainer_mod._DEFAULT_NUM_OUTPUT_CLASSES == 9
    assert trainer_mod._FG_CLASS_IDS == [1, 2, 3, 4, 5, 6, 7, 8]
    assert trainer_mod._LUMBAR_IDS == [1, 2, 3, 4, 5]
    assert trainer_mod._PELVIS_IDS == [6, 7, 8]
    assert trainer_mod._ANATOMY_NAMES == [
        "L1", "L2", "L3", "L4", "last_lumbar",
        "sacrum", "left_hip", "right_hip"
    ]
    # Defensive: ensure no stray L6 reference survives
    assert not hasattr(trainer_mod, "_L6_LABEL_ID"), (
        "_L6_LABEL_ID should be REMOVED in v20 (L6 class no longer exists); "
        "any code still using it is referencing a class that doesn't exist"
    )


def test_anatomy_names_index_alignment(trainer_mod):
    """_ANATOMY_NAMES[cid - 1] must give the right name for every
    foreground class id; the per-class dice loop relies on this."""
    expected = {
        1: "L1", 2: "L2", 3: "L3", 4: "L4",
        5: "last_lumbar", 6: "sacrum",
        7: "left_hip", 8: "right_hip",
    }
    for cid, name in expected.items():
        idx = trainer_mod._FG_CLASS_IDS.index(cid)
        assert trainer_mod._ANATOMY_NAMES[idx] == name, (
            f"cid={cid}: _ANATOMY_NAMES[{idx}] = "
            f"{trainer_mod._ANATOMY_NAMES[idx]!r}, expected {name!r}"
        )


def test_class_id_to_name(trainer_mod):
    assert trainer_mod._class_id_to_name(0) == "background"
    assert trainer_mod._class_id_to_name(1) == "L1"
    assert trainer_mod._class_id_to_name(4) == "L4"
    assert trainer_mod._class_id_to_name(5) == "last_lumbar"
    assert trainer_mod._class_id_to_name(6) == "sacrum"
    assert trainer_mod._class_id_to_name(99) == "label99"


# ──────────────────────────────────────────────────────────────────────
# CE weights
# ──────────────────────────────────────────────────────────────────────

def test_ce_weights_shape_and_values(trainer_mod):
    """v20 CE weights: 9-element tensor, with last_lumbar=1.5 and
    sacrum=2.0 boost, no L6=4.0 boost (no L6 channel)."""
    import torch
    import os

    # Force-enable CE reweight regardless of test environment
    prev = os.environ.pop("SPINESURG_CE_REWEIGHT", None)
    try:
        os.environ["SPINESURG_CE_REWEIGHT"] = "1"
        w = trainer_mod._build_ce_class_weights(num_classes=9)
        assert w is not None, "weights should not be None when SPINESURG_CE_REWEIGHT=1"
        assert isinstance(w, torch.Tensor)
        assert w.numel() == 9, f"weight tensor size should be 9, got {w.numel()}"

        # Documented values (see _build_ce_class_weights docstring).
        expected = [0.5, 1.0, 1.0, 1.0, 1.0, 1.5, 2.0, 1.0, 1.0]
        actual = [float(x) for x in w.cpu().numpy()]
        assert actual == expected, (
            f"weight tensor mismatch:\n  expected: {expected}\n  actual:   {actual}"
        )
    finally:
        if prev is None:
            os.environ.pop("SPINESURG_CE_REWEIGHT", None)
        else:
            os.environ["SPINESURG_CE_REWEIGHT"] = prev


def test_ce_weights_disabled_returns_none(trainer_mod):
    import os
    prev = os.environ.pop("SPINESURG_CE_REWEIGHT", None)
    try:
        os.environ["SPINESURG_CE_REWEIGHT"] = "0"
        w = trainer_mod._build_ce_class_weights(num_classes=9)
        assert w is None, "weights should be None when reweight disabled"
    finally:
        if prev is None:
            os.environ.pop("SPINESURG_CE_REWEIGHT", None)
        else:
            os.environ["SPINESURG_CE_REWEIGHT"] = prev


def test_ce_weights_env_override(trainer_mod):
    """SPINESURG_CE_WEIGHTS=name:value overrides per-class weights."""
    import os
    prev_re = os.environ.pop("SPINESURG_CE_REWEIGHT", None)
    prev_w = os.environ.pop("SPINESURG_CE_WEIGHTS", None)
    try:
        os.environ["SPINESURG_CE_REWEIGHT"] = "1"
        os.environ["SPINESURG_CE_WEIGHTS"] = "last_lumbar:3.0,sacrum:5.0"
        w = trainer_mod._build_ce_class_weights(num_classes=9)
        assert float(w[5]) == 3.0, f"last_lumbar override failed: {w[5]}"
        assert float(w[6]) == 5.0, f"sacrum override failed: {w[6]}"
        # Untouched weights stay at default
        assert float(w[0]) == 0.5
        assert float(w[1]) == 1.0
    finally:
        for k, v in [("SPINESURG_CE_REWEIGHT", prev_re),
                     ("SPINESURG_CE_WEIGHTS", prev_w)]:
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ──────────────────────────────────────────────────────────────────────
# Per-case dice
# ──────────────────────────────────────────────────────────────────────

def _make_synthetic_case(label_blocks: Dict[int, slice],
                          shape: Tuple[int, int, int] = (4, 4, 32)
                          ) -> "torch.Tensor":
    """Build a synthetic (D, H, W) tensor where each label_id occupies
    the given slice along the W axis. Voxels outside any block are 0.
    """
    import torch
    arr = np.zeros(shape, dtype=np.int16)
    for label_id, sl in label_blocks.items():
        arr[:, :, sl] = label_id
    return torch.from_numpy(arr)


def test_per_case_dice_perfect_prediction(trainer_mod):
    """Identical pred and GT → Dice == 1.0 for every present class."""
    gt = _make_synthetic_case({
        1: slice(0, 4),    # L1
        2: slice(4, 8),    # L2
        3: slice(8, 12),   # L3
        4: slice(12, 16),  # L4
        5: slice(16, 20),  # last_lumbar
        6: slice(20, 24),  # sacrum
        7: slice(24, 28),  # left_hip
        8: slice(28, 32),  # right_hip
    })
    pred = gt.clone()
    out = trainer_mod._per_case_dice_per_class(pred, gt)
    assert len(out) == 8, f"expected 8 entries, got {len(out)}"
    for cid_idx, dice in enumerate(out):
        assert dice is not None, f"class {cid_idx+1} dice is None"
        assert abs(dice - 1.0) < 1e-9, (
            f"class {cid_idx+1}: dice should be 1.0, got {dice}"
        )


def test_per_case_dice_absent_class_is_none(trainer_mod):
    """Class with no GT voxels → Dice is None (undefined)."""
    # Only L1 and L2 in GT
    gt = _make_synthetic_case({1: slice(0, 16), 2: slice(16, 32)})
    pred = gt.clone()
    out = trainer_mod._per_case_dice_per_class(pred, gt)
    assert out[0] is not None  # L1 present
    assert out[1] is not None  # L2 present
    for i in range(2, 8):
        assert out[i] is None, (
            f"class {trainer_mod._FG_CLASS_IDS[i]} absent in GT but "
            f"got non-None dice {out[i]}"
        )


def test_per_case_dice_ignores_label_9_voxels(trainer_mod):
    """ignore=9 voxels should be masked out of both num and denom."""
    # GT: L1 in [0:8], ignore in [8:16], L2 in [16:32]
    gt = _make_synthetic_case({
        1: slice(0, 8),
        9: slice(8, 16),    # ignore voxels — should be masked
        2: slice(16, 32),
    })
    # Pred: L1 in [0:8], L1 (WRONG) in [8:16], L2 in [16:32].
    # If ignore is honored, the [8:16] block is masked out → Dice = 1.0
    # for both L1 and L2. If NOT honored, L1 gets extra false positives
    # from [8:16] being predicted L1 where GT is "ignore" treated as 0.
    pred = _make_synthetic_case({
        1: slice(0, 16),    # extends across the ignore region
        2: slice(16, 32),
    })
    out = trainer_mod._per_case_dice_per_class(pred, gt)
    # L1 dice (idx 0): only the [0:8] region is "valid"; both pred and
    # GT match perfectly there → 1.0
    assert abs(out[0] - 1.0) < 1e-9, (
        f"L1 dice with ignore region should be 1.0, got {out[0]}; "
        f"ignore label is not being masked correctly"
    )
    # L2 dice (idx 1): perfect match in [16:32] → 1.0
    assert abs(out[1] - 1.0) < 1e-9


# ──────────────────────────────────────────────────────────────────────
# Per-case confusion vectors
# ──────────────────────────────────────────────────────────────────────

def test_per_case_confusion_by_gt_l4_to_last_lumbar(trainer_mod):
    """Synthetic case: GT has L4 in [0:16], pred has 12 vox L4 +
    4 vox last_lumbar in that region. Confusion[L4] should split
    accordingly. This tests the 'L4 bottom-vertebra labeled
    last_lumbar' failure mode directly."""
    import torch
    gt = _make_synthetic_case({4: slice(0, 16)})  # 16*16 = 256 L4 vox
    pred_arr = np.zeros((4, 4, 32), dtype=np.int16)
    pred_arr[:, :, 0:12] = 4   # L4 correct
    pred_arr[:, :, 12:16] = 5  # mislabeled as last_lumbar
    pred = torch.from_numpy(pred_arr)

    conf = trainer_mod._per_case_confusion_by_gt(pred, gt, num_classes=9)
    # GT class 4 should be in dict
    assert 4 in conf, f"GT class 4 not in confusion dict: {list(conf)}"
    vec = conf[4]  # length 9, indexed by predicted class
    assert vec.shape == (9,)
    # Total = total L4 GT voxels = 4*4*16 = 256
    assert int(vec.sum()) == 256
    # 4*4*12 = 192 voxels predicted as L4
    assert int(vec[4]) == 192
    # 4*4*4 = 64 voxels predicted as last_lumbar
    assert int(vec[5]) == 64
    # All other channels are zero
    for k in (0, 1, 2, 3, 6, 7, 8):
        assert int(vec[k]) == 0, f"unexpected mass at channel {k}: {vec[k]}"


def test_per_case_confusion_by_pred_last_lumbar_actually_l4(trainer_mod):
    """Symmetric PRED-direction case: pred says last_lumbar everywhere
    in a region whose GT is half L4, half background. The
    pred-direction confusion of last_lumbar should split 50/50 between
    L4 and background."""
    import torch
    gt_arr = np.zeros((4, 4, 32), dtype=np.int16)
    gt_arr[:, :, 0:16] = 4   # L4 in first half
    # Last half is GT background
    gt = torch.from_numpy(gt_arr)
    pred_arr = np.zeros((4, 4, 32), dtype=np.int16)
    pred_arr[:, :, 0:32] = 5  # all last_lumbar
    pred = torch.from_numpy(pred_arr)

    conf = trainer_mod._per_case_confusion_by_pred(pred, gt, num_classes=9)
    assert 5 in conf, f"PRED class 5 (last_lumbar) not in dict: {list(conf)}"
    vec = conf[5]
    assert int(vec.sum()) == 4 * 4 * 32  # 512 voxels total predicted last_lumbar
    assert int(vec[4]) == 4 * 4 * 16     # 256 actually L4
    assert int(vec[0]) == 4 * 4 * 16     # 256 actually background
    for k in (1, 2, 3, 5, 6, 7, 8):
        assert int(vec[k]) == 0


def test_per_case_confusion_excludes_ignore_voxels(trainer_mod):
    """Ignore voxels (label 9) must be excluded from confusion buffers
    in BOTH directions. A pelvic-only case has ignore in the lumbar
    region; we must not accumulate false 'last_lumbar hallucination' at
    those voxels."""
    import torch
    # GT: ignore in first half, sacrum in second half
    gt_arr = np.zeros((4, 4, 32), dtype=np.int16)
    gt_arr[:, :, 0:16] = 9   # ignore
    gt_arr[:, :, 16:32] = 6  # sacrum
    gt = torch.from_numpy(gt_arr)
    # Pred: last_lumbar everywhere (would look like a hallucination
    # in the ignore region if we counted it)
    pred_arr = np.full((4, 4, 32), 5, dtype=np.int16)
    pred = torch.from_numpy(pred_arr)

    # GT-direction: only sacrum should appear in dict (ignore is masked)
    conf_gt = trainer_mod._per_case_confusion_by_gt(pred, gt, num_classes=9)
    assert 9 not in conf_gt
    assert 6 in conf_gt
    # All sacrum GT predicted as last_lumbar
    assert int(conf_gt[6][5]) == 4 * 4 * 16

    # PRED-direction: pred class 5 has 256 voxels (only the ones in the
    # second half count; first half is ignored)
    conf_pr = trainer_mod._per_case_confusion_by_pred(pred, gt, num_classes=9)
    assert 5 in conf_pr
    assert int(conf_pr[5].sum()) == 4 * 4 * 16, (
        f"sum should be 256 (post-ignore-mask), got {int(conf_pr[5].sum())}"
    )
    # Of those, all are actually GT-sacrum
    assert int(conf_pr[5][6]) == 4 * 4 * 16


# ──────────────────────────────────────────────────────────────────────
# v20 confusion block aggregation — the headline of this changeset
# ──────────────────────────────────────────────────────────────────────

def _build_synthetic_buffer(per_subtype_cases: Dict[str, List[Dict[int, np.ndarray]]]):
    """Helper: build a confusion buffer dict like the trainer maintains."""
    return {sub: list(cases) for sub, cases in per_subtype_cases.items()}


def test_aggregate_block_by_gt_emits_documented_keys(trainer_mod):
    """Single lumb case where GT-last_lumbar (5) splits 80/20 between
    pred=last_lumbar and pred=sacrum. Block 1 should emit total_voxels
    + the 4 documented fraction keys with the right values."""
    # Confusion vector for GT class 5 in this case: 80 vox -> pred 5,
    # 20 vox -> pred 6 (sacrum). All other pred channels zero.
    vec = np.zeros(9, dtype=np.int64)
    vec[5] = 80
    vec[6] = 20
    case_dict = {5: vec}
    buf = _build_synthetic_buffer({trainer_mod._SUBTYPE_LUMB: [case_dict]})

    out: Dict[str, float] = {}
    trainer_mod._aggregate_block_by_gt(
        buf,
        subtype=trainer_mod._SUBTYPE_LUMB,
        anchor_gt_cid=trainer_mod._LAST_LUMBAR_LABEL_ID,
        target_specs=trainer_mod._HEADLINE_TARGETS_LL_GT_ON_LUMB,
        block_id="last_lumbar_GT_on_lumb",
        out=out,
    )

    base = "val/headline/last_lumbar_GT_on_lumb"
    assert out[f"{base}/total_voxels"] == 100.0
    assert abs(out[f"{base}/as_last_lumbar_frac"] - 0.80) < 1e-9
    assert abs(out[f"{base}/as_sacrum_frac"] - 0.20) < 1e-9
    assert abs(out[f"{base}/as_L4_frac"] - 0.0) < 1e-9
    assert abs(out[f"{base}/as_background_frac"] - 0.0) < 1e-9


def test_aggregate_block_by_gt_no_eligible_cases_emits_nothing(trainer_mod):
    """Empty subtype buffer → no keys emitted (caller distinguishes 'no
    cases' from 'zero fraction')."""
    buf = _build_synthetic_buffer({trainer_mod._SUBTYPE_LUMB: []})
    out: Dict[str, float] = {}
    trainer_mod._aggregate_block_by_gt(
        buf,
        subtype=trainer_mod._SUBTYPE_LUMB,
        anchor_gt_cid=trainer_mod._LAST_LUMBAR_LABEL_ID,
        target_specs=trainer_mod._HEADLINE_TARGETS_LL_GT_ON_LUMB,
        block_id="last_lumbar_GT_on_lumb",
        out=out,
    )
    assert out == {}, f"empty buffer should emit no keys, got {out}"


def test_aggregate_block_by_pred_actually_l4(trainer_mod):
    """sacr_count case: pred has 200 voxels of last_lumbar (which
    SHOULD NOT exist on sacr_count). Of those, 150 actually L4 and
    50 actually background. Block 3 should reflect this split."""
    vec = np.zeros(9, dtype=np.int64)
    vec[4] = 150  # actually L4
    vec[0] = 50   # actually background
    case_dict = {5: vec}  # PRED class 5 = last_lumbar
    buf = _build_synthetic_buffer({trainer_mod._SUBTYPE_SACR_COUNT: [case_dict]})

    out: Dict[str, float] = {}
    trainer_mod._aggregate_block_by_pred(
        buf,
        subtype=trainer_mod._SUBTYPE_SACR_COUNT,
        anchor_pred_cid=trainer_mod._LAST_LUMBAR_LABEL_ID,
        target_specs=trainer_mod._HEADLINE_TARGETS_PRED_LL_ON_SACR_COUNT,
        block_id="pred_last_lumbar_on_sacr_count",
        out=out,
    )
    base = "val/headline/pred_last_lumbar_on_sacr_count"
    assert out[f"{base}/total_voxels"] == 200.0
    assert abs(out[f"{base}/actually_L4_frac"] - 0.75) < 1e-9
    assert abs(out[f"{base}/actually_background_frac"] - 0.25) < 1e-9
    assert abs(out[f"{base}/actually_sacrum_frac"] - 0.0) < 1e-9


def test_aggregate_block_sums_across_cases(trainer_mod):
    """Two lumb cases — confusion vectors should be summed before
    fractioning. Case A: 80 LL + 20 sacrum. Case B: 100 LL + 0 sacrum.
    Sum: 180 LL + 20 sacrum = 90% / 10%."""
    vec_a = np.zeros(9, dtype=np.int64); vec_a[5] = 80; vec_a[6] = 20
    vec_b = np.zeros(9, dtype=np.int64); vec_b[5] = 100
    buf = _build_synthetic_buffer({
        trainer_mod._SUBTYPE_LUMB: [{5: vec_a}, {5: vec_b}]
    })
    out: Dict[str, float] = {}
    trainer_mod._aggregate_block_by_gt(
        buf,
        subtype=trainer_mod._SUBTYPE_LUMB,
        anchor_gt_cid=trainer_mod._LAST_LUMBAR_LABEL_ID,
        target_specs=trainer_mod._HEADLINE_TARGETS_LL_GT_ON_LUMB,
        block_id="last_lumbar_GT_on_lumb",
        out=out,
    )
    base = "val/headline/last_lumbar_GT_on_lumb"
    assert out[f"{base}/total_voxels"] == 200.0
    assert abs(out[f"{base}/as_last_lumbar_frac"] - 0.90) < 1e-9
    assert abs(out[f"{base}/as_sacrum_frac"] - 0.10) < 1e-9


def test_headline_target_specs_only_reference_valid_classes(trainer_mod):
    """Every (name, cid) tuple in the three _HEADLINE_TARGETS_* tuples
    must reference a valid class id under the v20 scheme: either
    background (0) or a member of _FG_CLASS_IDS. Catches typos or
    stale references to label 6 / label 7 from before the renumbering."""
    valid_cids = {0} | set(trainer_mod._FG_CLASS_IDS)
    for spec_name, spec in [
        ("LL_GT_ON_LUMB",            trainer_mod._HEADLINE_TARGETS_LL_GT_ON_LUMB),
        ("L4_GT_ON_SACR_COUNT",      trainer_mod._HEADLINE_TARGETS_L4_GT_ON_SACR_COUNT),
        ("PRED_LL_ON_SACR_COUNT",    trainer_mod._HEADLINE_TARGETS_PRED_LL_ON_SACR_COUNT),
    ]:
        for name, cid, leaf, rationale in spec:
            assert cid in valid_cids, (
                f"{spec_name}: target ({name!r}, cid={cid}) is not a valid "
                f"v20 class id; valid set = {sorted(valid_cids)}"
            )
            assert leaf, f"{spec_name}: empty leaf for {name!r}"
            assert rationale, f"{spec_name}: empty rationale for {name!r}"


# ──────────────────────────────────────────────────────────────────────
# End-to-end: synthetic _aggregate_subgroup_confusion + key set audit
# ──────────────────────────────────────────────────────────────────────

class _MinimalMixinHost:
    """A bare object with just the buffers _aggregate_subgroup_confusion
    needs, so we can call it without instantiating a full trainer."""
    def __init__(self):
        self._val_subgroup_confusion_by_gt = None
        self._val_subgroup_confusion_by_pred = None


def test_v20_confusion_full_payload_and_no_legacy_spam(trainer_mod):
    """The big one: build a buffer with synthetic lumb + sacr_count
    cases, call the trainer's _aggregate_subgroup_confusion, and assert
    that the full output dict contains EXACTLY the documented headline
    keys and ZERO of the legacy v19 'val/lstv_subgroup/{sub}/conf_by_*/'
    spam keys. Catches accidental regression where someone re-enables
    the verbose emitter."""
    # Lumb case: 90% LL, 5% sacrum, 5% L4
    vec_lumb_gt_5 = np.zeros(9, dtype=np.int64)
    vec_lumb_gt_5[5] = 90; vec_lumb_gt_5[6] = 5; vec_lumb_gt_5[4] = 5

    # sacr_count case: GT-L4 confusion: 80% L4, 15% last_lumbar, 5% bg
    vec_sc_gt_4 = np.zeros(9, dtype=np.int64)
    vec_sc_gt_4[4] = 80; vec_sc_gt_4[5] = 15; vec_sc_gt_4[0] = 5

    # sacr_count case: PRED-last_lumbar confusion: 60% L4, 30% bg, 10% sacrum
    vec_sc_pred_5 = np.zeros(9, dtype=np.int64)
    vec_sc_pred_5[4] = 60; vec_sc_pred_5[0] = 30; vec_sc_pred_5[6] = 10

    host = _MinimalMixinHost()
    host._val_subgroup_confusion_by_gt = {
        sub: [] for sub in trainer_mod._KNOWN_SUBTYPES
    }
    host._val_subgroup_confusion_by_pred = {
        sub: [] for sub in trainer_mod._KNOWN_SUBTYPES
    }
    host._val_subgroup_confusion_by_gt[trainer_mod._SUBTYPE_LUMB].append({5: vec_lumb_gt_5})
    host._val_subgroup_confusion_by_gt[trainer_mod._SUBTYPE_SACR_COUNT].append({4: vec_sc_gt_4})
    host._val_subgroup_confusion_by_pred[trainer_mod._SUBTYPE_SACR_COUNT].append({5: vec_sc_pred_5})

    out = trainer_mod._WandBMixin._aggregate_subgroup_confusion(host)

    # Expected key set
    expected_keys = {
        # Block 1: last_lumbar GT on lumb
        "val/headline/last_lumbar_GT_on_lumb/total_voxels",
        "val/headline/last_lumbar_GT_on_lumb/as_last_lumbar_frac",
        "val/headline/last_lumbar_GT_on_lumb/as_sacrum_frac",
        "val/headline/last_lumbar_GT_on_lumb/as_L4_frac",
        "val/headline/last_lumbar_GT_on_lumb/as_background_frac",
        # Block 2: L4 GT on sacr_count
        "val/headline/L4_GT_on_sacr_count/total_voxels",
        "val/headline/L4_GT_on_sacr_count/as_L4_frac",
        "val/headline/L4_GT_on_sacr_count/as_last_lumbar_frac",
        "val/headline/L4_GT_on_sacr_count/as_sacrum_frac",
        "val/headline/L4_GT_on_sacr_count/as_background_frac",
        # Block 3: predicted last_lumbar on sacr_count
        "val/headline/pred_last_lumbar_on_sacr_count/total_voxels",
        "val/headline/pred_last_lumbar_on_sacr_count/actually_L4_frac",
        "val/headline/pred_last_lumbar_on_sacr_count/actually_sacrum_frac",
        "val/headline/pred_last_lumbar_on_sacr_count/actually_background_frac",
    }
    actual_keys = set(out.keys())
    missing = expected_keys - actual_keys
    extra = actual_keys - expected_keys
    assert not missing, f"missing expected headline keys: {sorted(missing)}"
    assert not extra, (
        f"v20 confusion emitted UNEXPECTED keys (legacy spam regression?): "
        f"{sorted(extra)}"
    )

    # Spot-check values (Block 1)
    assert out["val/headline/last_lumbar_GT_on_lumb/total_voxels"] == 100.0
    assert abs(out["val/headline/last_lumbar_GT_on_lumb/as_last_lumbar_frac"] - 0.90) < 1e-9
    assert abs(out["val/headline/last_lumbar_GT_on_lumb/as_sacrum_frac"] - 0.05) < 1e-9
    assert abs(out["val/headline/last_lumbar_GT_on_lumb/as_L4_frac"] - 0.05) < 1e-9

    # Spot-check Block 3 — the "model still says last_lumbar where it
    # should be L4" failure mode.
    assert out["val/headline/pred_last_lumbar_on_sacr_count/total_voxels"] == 100.0
    assert abs(out["val/headline/pred_last_lumbar_on_sacr_count/actually_L4_frac"] - 0.60) < 1e-9


def test_v20_confusion_total_key_count_is_bounded(trainer_mod):
    """Sanity: even with all 5 LSTV subtypes populated, the v20 confusion
    output stays at ≤ 14 keys. This is the whole point of the v20 cut."""
    host = _MinimalMixinHost()
    host._val_subgroup_confusion_by_gt = {sub: [] for sub in trainer_mod._KNOWN_SUBTYPES}
    host._val_subgroup_confusion_by_pred = {sub: [] for sub in trainer_mod._KNOWN_SUBTYPES}
    # Populate every subtype with a synthetic case to be defensive.
    for sub in trainer_mod._KNOWN_SUBTYPES:
        gt_dict = {}
        pred_dict = {}
        for cid in trainer_mod._FG_CLASS_IDS:
            v = np.zeros(9, dtype=np.int64); v[cid] = 100
            gt_dict[cid] = v
            pred_dict[cid] = v.copy()
        host._val_subgroup_confusion_by_gt[sub].append(gt_dict)
        host._val_subgroup_confusion_by_pred[sub].append(pred_dict)

    out = trainer_mod._WandBMixin._aggregate_subgroup_confusion(host)
    assert len(out) <= 14, (
        f"v20 confusion emitted {len(out)} keys; should be ≤ 14 "
        f"(3 blocks × ~5 keys each). Keys:\n  " +
        "\n  ".join(sorted(out.keys()))
    )


# ──────────────────────────────────────────────────────────────────────
# Hallucination headline (single specificity number)
# ──────────────────────────────────────────────────────────────────────

def test_hallucination_specificity_perfect_score(trainer_mod):
    """sacr_count cases with no last_lumbar GT and no last_lumbar pred
    → specificity = 1.0."""
    host = _MinimalMixinHost()
    host._per_subgroup_logging_enabled = True
    host._val_subgroup_voxels = {sub: [] for sub in trainer_mod._KNOWN_SUBTYPES}
    # Each case: per-class (gt_n, pred_n). 8 fg classes (1..8) → 8 entries.
    # last_lumbar (cid=5) is at index _FG_CLASS_IDS.index(5) = 4.
    case_perfect = [(0, 0)] * 8
    host._val_subgroup_voxels[trainer_mod._SUBTYPE_SACR_COUNT] = [
        case_perfect, case_perfect, case_perfect
    ]
    out = trainer_mod._WandBMixin._aggregate_subgroup_hallucination(host)
    assert out["val/headline/last_lumbar_specificity_on_sacr_count"] == 1.0
    assert out["val/headline/last_lumbar_specificity_on_sacr_count_n_cases"] == 3.0


def test_hallucination_specificity_partial_failure(trainer_mod):
    """3 sacr_count cases: 2 with no last_lumbar pred (good), 1 with
    500 voxels of last_lumbar pred (above threshold 100, counts as
    meaningful hallucination). Expect specificity = 2/3."""
    host = _MinimalMixinHost()
    host._per_subgroup_logging_enabled = True
    host._val_subgroup_voxels = {sub: [] for sub in trainer_mod._KNOWN_SUBTYPES}
    # last_lumbar index in _FG_CLASS_IDS
    ll_idx = trainer_mod._FG_CLASS_IDS.index(trainer_mod._LAST_LUMBAR_LABEL_ID)
    perfect = [(0, 0)] * 8
    halluc = list(perfect); halluc[ll_idx] = (0, 500)  # GT-absent, pred=500
    host._val_subgroup_voxels[trainer_mod._SUBTYPE_SACR_COUNT] = [
        perfect, perfect, halluc
    ]
    out = trainer_mod._WandBMixin._aggregate_subgroup_hallucination(host)
    assert abs(out["val/headline/last_lumbar_specificity_on_sacr_count"] - (2/3)) < 1e-9
    assert out["val/headline/last_lumbar_specificity_on_sacr_count_n_cases"] == 3.0


def test_hallucination_below_threshold_counts_as_clean(trainer_mod):
    """Predictions of ≤ _HALLUC_VOXEL_THRESHOLD voxels are noise-floor;
    they should NOT count as a hallucination event. Specificity should
    stay at 1.0."""
    host = _MinimalMixinHost()
    host._per_subgroup_logging_enabled = True
    host._val_subgroup_voxels = {sub: [] for sub in trainer_mod._KNOWN_SUBTYPES}
    ll_idx = trainer_mod._FG_CLASS_IDS.index(trainer_mod._LAST_LUMBAR_LABEL_ID)
    perfect = [(0, 0)] * 8
    just_under = list(perfect); just_under[ll_idx] = (0, trainer_mod._HALLUC_VOXEL_THRESHOLD)
    host._val_subgroup_voxels[trainer_mod._SUBTYPE_SACR_COUNT] = [
        perfect, just_under
    ]
    out = trainer_mod._WandBMixin._aggregate_subgroup_hallucination(host)
    assert out["val/headline/last_lumbar_specificity_on_sacr_count"] == 1.0


# ──────────────────────────────────────────────────────────────────────
# Test runner
# ──────────────────────────────────────────────────────────────────────

def _all_tests():
    """Discover this module's test functions in source order."""
    import inspect
    here = sys.modules[__name__]
    tests = []
    for name, obj in inspect.getmembers(here, inspect.isfunction):
        if name.startswith("test_"):
            tests.append((name, obj))
    # Preserve source-order
    tests.sort(key=lambda nf: inspect.getsourcelines(nf[1])[1])
    return tests


def _run_tests() -> int:
    convert_m, trainer_m = _modules()
    fixtures = {"convert_mod": convert_m, "trainer_mod": trainer_m}

    n_pass = n_fail = 0
    failures: List[Tuple[str, str]] = []
    for name, fn in _all_tests():
        # Only pass the fixtures the test actually accepts
        import inspect
        sig = inspect.signature(fn)
        kwargs = {p: fixtures[p] for p in sig.parameters if p in fixtures}
        try:
            fn(**kwargs)
            print(f"PASS  {name}")
            n_pass += 1
        except Exception as exc:
            print(f"FAIL  {name}")
            tb = traceback.format_exc()
            failures.append((name, tb))
            n_fail += 1

    print()
    print("=" * 60)
    print(f"  {n_pass} passed, {n_fail} failed")
    print("=" * 60)
    if failures:
        for name, tb in failures:
            print()
            print(f"--- {name} ---")
            print(tb)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(_run_tests())
