"""
tests/test_verse_remap.py

Unit tests for the VerSe-native -> training-scheme label remap in
tools/convert_hf_to_nnunet.py.

The HF export was migrated to VerSe-native label ids (the dataset's single
source of truth, label_scheme.py). The convert script remaps those ids to
the training scheme:

  * MERGED   (Dataset803, default):        VerSe L5+L6 -> last_lumbar (5)
  * UNMERGED (Dataset802, --no_remap):     VerSe L5 -> 5, L6 -> 6

CRITICAL invariant tested here: every VerSe id that is NOT a training class
(thoracic/cervical vertebrae, coccyx, T13, S1, femurs, ribs, soft-tissue,
lumbar ribs) must map to 0 (background) — NOT pass through. A passthrough
would leak ids like 13-24, 26, 30, 31, 255 into the produced labels and
nnU-Net's --verify_dataset_integrity would reject them.

Functional test: needs numpy only (no torch / nnU-Net). Builds the real
LUTs from the real remap dicts and asserts exact mappings.

Run:
    python -m pytest tests/test_verse_remap.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

# Import the convert script from tools/
_THIS_DIR = Path(__file__).resolve().parent
_TOOLS_DIR = _THIS_DIR.parent / "tools"
sys.path.insert(0, str(_TOOLS_DIR))

import convert_hf_to_nnunet as convert  # noqa: E402


# VerSe-native source ids (mirrors the frozen constants in the convert
# script, which in turn mirror label_scheme.py). Kept literal here on
# purpose so this test would catch an accidental edit to those constants.
_VERSE = {
    0: "background",
    20: "L1", 21: "L2", 22: "L3", 23: "L4", 24: "L5", 25: "L6",
    26: "sacrum", 30: "left_hip", 31: "right_hip", 255: "ignore",
}

# The canonical source-value sample from the task sanity spec.
_SAMPLE_SRC = [0, 20, 21, 22, 23, 24, 25, 26, 30, 31, 255]

# A spread of VerSe ids that are NOT training classes and MUST drop to 0.
#   5   = C5 (cervical)
#   15  = T8 (thoracic)
#   27  = coccyx
#   28  = T13
#   29  = S1
#   32  = femur_left
#   33  = femur_right
#   40  = a rib
#   59  = rib_right_13
#   60  = lumbar rib
#   62, 68 = hardware
_NON_TRAINING_SRC = [5, 15, 27, 28, 29, 32, 33, 40, 59, 60, 61, 62, 68]


def _merged_lut():
    return convert._build_verse_remap_lut(convert.LABEL_REMAP_MERGED_FROM_VERSE)


def _unmerged_lut():
    return convert._build_verse_remap_lut(convert.LABEL_REMAP_UNMERGED_FROM_VERSE)


# ── constants sanity ───────────────────────────────────────────────────

def test_verse_source_constants_are_frozen_ids():
    assert convert.VERSE_L1 == 20
    assert convert.VERSE_L2 == 21
    assert convert.VERSE_L3 == 22
    assert convert.VERSE_L4 == 23
    assert convert.VERSE_L5 == 24
    assert convert.VERSE_L6 == 25
    assert convert.VERSE_SACRUM == 26
    assert convert.VERSE_LEFT_HIP == 30
    assert convert.VERSE_RIGHT_HIP == 31
    assert convert.VERSE_IGNORE == 255


def test_lut_spans_full_verse_range():
    """The LUT must cover 0..255 so ignore (255) is indexable and every
    non-training VerSe id has an explicit background destination."""
    lut = _merged_lut()
    assert len(lut) == 256


# ── merged (Dataset803) ────────────────────────────────────────────────

def test_merged_exact_mapping():
    lut = _merged_lut()
    expected = {
        20: 1, 21: 2, 22: 3, 23: 4,
        24: 5, 25: 5,   # L5 + L6 -> last_lumbar
        26: 6, 30: 7, 31: 8, 255: 9,
    }
    for src, dst in expected.items():
        assert int(lut[src]) == dst, f"VerSe {src} ({_VERSE.get(src)}) -> {int(lut[src])}, expected {dst}"


def test_merged_sample_unique_range():
    """{0,20,...,255} -> unique {0,1,2,3,4,5,6,7,8,9}."""
    lut = _merged_lut()
    mapped = [int(lut[v]) for v in _SAMPLE_SRC]
    assert mapped == [0, 1, 2, 3, 4, 5, 5, 6, 7, 8, 9]
    assert sorted(set(mapped)) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_merged_background_stays_background():
    assert int(_merged_lut()[0]) == 0


def test_merged_non_training_ids_drop_to_background():
    lut = _merged_lut()
    for src in _NON_TRAINING_SRC:
        assert int(lut[src]) == 0, f"non-training VerSe {src} must -> 0, got {int(lut[src])}"


# ── unmerged (Dataset802, --no_remap) ──────────────────────────────────

def test_unmerged_exact_mapping():
    lut = _unmerged_lut()
    expected = {
        20: 1, 21: 2, 22: 3, 23: 4,
        24: 5, 25: 6,   # L5 and L6 distinct
        26: 7, 30: 8, 31: 9, 255: 10,
    }
    for src, dst in expected.items():
        assert int(lut[src]) == dst, f"VerSe {src} ({_VERSE.get(src)}) -> {int(lut[src])}, expected {dst}"


def test_unmerged_sample_range():
    """{0,20,...,255} -> {0,1,2,3,4,5,6,7,8,9,10}."""
    lut = _unmerged_lut()
    mapped = [int(lut[v]) for v in _SAMPLE_SRC]
    assert mapped == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def test_unmerged_background_stays_background():
    assert int(_unmerged_lut()[0]) == 0


def test_unmerged_non_training_ids_drop_to_background():
    lut = _unmerged_lut()
    for src in _NON_TRAINING_SRC:
        assert int(lut[src]) == 0, f"non-training VerSe {src} must -> 0, got {int(lut[src])}"


# ── no-ignore variant (fused-only ablation) ────────────────────────────

def test_no_ignore_variant_maps_ignore_to_background():
    """The merged no-ignore variant is identical to merged, except VerSe
    ignore (255) -> background (0) instead of the ignore class."""
    lut = convert._build_verse_remap_lut(
        convert.LABEL_REMAP_MERGED_NO_IGNORE_FROM_VERSE)
    assert int(lut[255]) == 0
    # Foreground mappings match the merged scheme.
    for src, dst in {20: 1, 24: 5, 25: 5, 26: 6, 30: 7, 31: 8}.items():
        assert int(lut[src]) == dst
    for src in _NON_TRAINING_SRC:
        assert int(lut[src]) == 0


# ── builder contract ───────────────────────────────────────────────────

def test_builder_defaults_unlisted_to_zero_not_identity():
    """Regression guard: the VerSe builder must NOT pass unlisted ids
    through (identity). A single-entry remap must leave everything else
    at background."""
    lut = convert._build_verse_remap_lut({24: 5})
    assert int(lut[24]) == 5
    assert int(lut[23]) == 0   # would be 23 under a passthrough LUT
    assert int(lut[100]) == 0
    assert int(lut[0]) == 0


def test_builder_rejects_out_of_range_source():
    import pytest
    with pytest.raises(ValueError):
        convert._build_verse_remap_lut({999: 1})  # 999 > 255


# ── label-name dicts unchanged (must match dataset.json sanity checks) ──

def test_merged_label_names_exact():
    assert convert.LABEL_NAMES == {
        "background": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4,
        "last_lumbar": 5, "sacrum": 6, "left_hip": 7, "right_hip": 8,
        "ignore": 9,
    }


def test_unmerged_label_names_exact():
    assert convert.LABEL_NAMES_UNMERGED == {
        "background": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4,
        "L5": 5, "L6": 6, "sacrum": 7, "left_hip": 8, "right_hip": 9,
        "ignore": 10,
    }


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
