"""
tests/test_subtype_refinement.py

Tests for the per-record subtype refinement in convert_hf_to_nnunet.py.

The refinement is SYMMETRIC by anatomical region:

  pelvic_native  -> demote {lumb, sacr_count} (these need lumbar coverage)
  spine_only     -> demote {sacralization, semisacralization}
                    (these need pelvis coverage to see sacral wing)
  fused          -> preserve all subtypes
  ambiguous      -> preserved on every config (don't hide it)

This mirrors the partial-annotation contract: a spine_only label has
the pelvis masked as ignore=10, and a pelvic_native label has the
lumbar region masked as ignore=10. A subtype's tag is only
trustworthy when its defining anatomy is actually visible in the
record's label file.

Run:
    /opt/conda/bin/python3 -m pytest tests/test_subtype_refinement.py -v

Or, without pytest (stdlib):
    /opt/conda/bin/python3 tests/test_subtype_refinement.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add tools/ to path so we can import convert_hf_to_nnunet
_THIS_DIR = Path(__file__).resolve().parent
_TOOLS_DIR = _THIS_DIR.parent / "tools"
sys.path.insert(0, str(_TOOLS_DIR))

from convert_hf_to_nnunet import (  # noqa: E402
    _refine_subtype_for_record_config,
    _resolve_subtype_for_case,
    _LUMBAR_REGION_SUBTYPES,
    _PELVIS_REGION_SUBTYPES,
    _PELVIC_ONLY_CONFIG,
    _SPINE_ONLY_CONFIG,
    SUBTYPES,
)


# pelvic_native: demote LUMBAR-region subtypes

def test_pelvic_only_lumb_demotes_to_normal():
    assert _refine_subtype_for_record_config("lumb", "pelvic_native") == "normal"


def test_pelvic_only_sacr_count_demotes_to_normal():
    assert _refine_subtype_for_record_config("sacr_count", "pelvic_native") == "normal"


def test_pelvic_only_semi_preserved():
    # semisacralization is sacral-wing morphology, visible at the
    # lumbosacral junction in pelvic_native scans. Preserve.
    assert _refine_subtype_for_record_config("semisacralization", "pelvic_native") == "semisacralization"


def test_pelvic_only_sacr_preserved():
    # sacralization (full sacral-wing fusion) is also visible in
    # pelvic_native views. Preserve.
    assert _refine_subtype_for_record_config("sacralization", "pelvic_native") == "sacralization"


def test_pelvic_only_normal_stays_normal():
    assert _refine_subtype_for_record_config("normal", "pelvic_native") == "normal"


def test_pelvic_only_ambiguous_stays_ambiguous():
    # Preserved on every config so the case still appears in
    # per-subgroup metrics. Don't silently relabel as normal.
    assert _refine_subtype_for_record_config("ambiguous", "pelvic_native") == "ambiguous"


# spine_only: demote PELVIS-region subtypes

def test_spine_only_preserves_lumb():
    # Spine-only view of lumb patient still shows L6 in the lumbar
    # region (the defining anatomy). Preserve.
    assert _refine_subtype_for_record_config("lumb", "spine_only") == "lumb"


def test_spine_only_preserves_sacr_count():
    # Spine-only view of sacr_count: lumbar shows L1-L4 (no L5).
    # The LUMBAR-side count signal is present. Preserve.
    assert _refine_subtype_for_record_config("sacr_count", "spine_only") == "sacr_count"


def test_spine_only_semi_demotes_to_normal():
    # Spine-only view masks the pelvis as ignore=10. Sacral-wing
    # morphology is invisible.
    assert _refine_subtype_for_record_config("semisacralization", "spine_only") == "normal"


def test_spine_only_sacr_demotes_to_normal():
    assert _refine_subtype_for_record_config("sacralization", "spine_only") == "normal"


def test_spine_only_normal_stays_normal():
    assert _refine_subtype_for_record_config("normal", "spine_only") == "normal"


def test_spine_only_ambiguous_stays_ambiguous():
    assert _refine_subtype_for_record_config("ambiguous", "spine_only") == "ambiguous"


# fused: preserve everything

def test_fused_preserves_lumb():
    assert _refine_subtype_for_record_config("lumb", "fused") == "lumb"


def test_fused_preserves_sacr_count():
    assert _refine_subtype_for_record_config("sacr_count", "fused") == "sacr_count"


def test_fused_preserves_sacralization():
    assert _refine_subtype_for_record_config("sacralization", "fused") == "sacralization"


def test_fused_preserves_semisacralization():
    assert _refine_subtype_for_record_config("semisacralization", "fused") == "semisacralization"


def test_fused_preserves_ambiguous():
    assert _refine_subtype_for_record_config("ambiguous", "fused") == "ambiguous"


def test_fused_normal_stays_normal():
    assert _refine_subtype_for_record_config("normal", "fused") == "normal"


# Defensive parsing

def test_unknown_config_preserves_subtype():
    assert _refine_subtype_for_record_config("lumb", "") == "lumb"
    assert _refine_subtype_for_record_config("lumb", "unknown_config") == "lumb"
    assert _refine_subtype_for_record_config("lumb", None) == "lumb"
    assert _refine_subtype_for_record_config("sacralization", "") == "sacralization"


def test_case_insensitive_pelvic_native():
    assert _refine_subtype_for_record_config("lumb", "Pelvic_Native") == "normal"
    assert _refine_subtype_for_record_config("lumb", "PELVIC_NATIVE") == "normal"


def test_case_insensitive_spine_only():
    assert _refine_subtype_for_record_config("sacralization", "Spine_Only") == "normal"
    assert _refine_subtype_for_record_config("sacralization", "SPINE_ONLY") == "normal"


def test_whitespace_in_config_handled():
    assert _refine_subtype_for_record_config("lumb", "  pelvic_native  ") == "normal"
    assert _refine_subtype_for_record_config("sacralization", "  spine_only  ") == "normal"


# Constants sanity

def test_lumbar_region_subtypes_contains_expected():
    assert _LUMBAR_REGION_SUBTYPES == {"lumb", "sacr_count"}


def test_pelvis_region_subtypes_contains_expected():
    assert _PELVIS_REGION_SUBTYPES == {"sacralization", "semisacralization"}


def test_lumbar_region_excludes_normal_ambig():
    assert "normal" not in _LUMBAR_REGION_SUBTYPES
    assert "ambiguous" not in _LUMBAR_REGION_SUBTYPES


def test_pelvis_region_excludes_normal_ambig():
    assert "normal" not in _PELVIS_REGION_SUBTYPES
    assert "ambiguous" not in _PELVIS_REGION_SUBTYPES


def test_lumbar_and_pelvis_regions_disjoint():
    assert _LUMBAR_REGION_SUBTYPES.isdisjoint(_PELVIS_REGION_SUBTYPES)


def test_config_constants():
    assert _PELVIC_ONLY_CONFIG == "pelvic_native"
    assert _SPINE_ONLY_CONFIG == "spine_only"


# End-to-end via _resolve_subtype_for_case

def test_resolve_uses_splits_subtypes_then_refines_pelvic_native():
    record = {
        "token": "267",
        "config": "pelvic_native",
        "lstv_label": "LUMBARIZATION",
    }
    splits_subtypes = {"267": "lumb"}
    assert _resolve_subtype_for_case(record, splits_subtypes) == "normal"


def test_resolve_sacralization_patient_spine_only_demoted():
    record = {
        "token": "999_sacr",
        "config": "spine_only",
        "lstv_label": "SACRALIZATION",
    }
    splits_subtypes = {"999_sacr": "sacralization"}
    assert _resolve_subtype_for_case(record, splits_subtypes) == "normal"


def test_resolve_sacralization_patient_pelvic_native_preserved():
    record = {
        "token": "999_sacr",
        "config": "pelvic_native",
        "lstv_label": "SACRALIZATION",
    }
    splits_subtypes = {"999_sacr": "sacralization"}
    assert _resolve_subtype_for_case(record, splits_subtypes) == "sacralization"


def test_resolve_semi_patient_spine_only_demoted():
    record = {
        "token": "999_semi",
        "config": "spine_only",
        "lstv_label": "SEMI_SACRAL",
    }
    splits_subtypes = {"999_semi": "semisacralization"}
    assert _resolve_subtype_for_case(record, splits_subtypes) == "normal"


def test_resolve_semi_patient_pelvic_native_preserved():
    record = {
        "token": "999_semi",
        "config": "pelvic_native",
        "lstv_label": "SEMI_SACRAL",
    }
    splits_subtypes = {"999_semi": "semisacralization"}
    assert _resolve_subtype_for_case(record, splits_subtypes) == "semisacralization"


def test_resolve_lumb_patient_spine_only_preserved():
    record = {
        "token": "267",
        "config": "spine_only",
        "lstv_label": "LUMBARIZATION",
    }
    splits_subtypes = {"267": "lumb"}
    assert _resolve_subtype_for_case(record, splits_subtypes) == "lumb"


def test_resolve_uses_splits_subtypes_keeps_for_fused():
    record = {
        "token": "267",
        "config": "fused",
        "lstv_label": "LUMBARIZATION",
    }
    splits_subtypes = {"267": "lumb"}
    assert _resolve_subtype_for_case(record, splits_subtypes) == "lumb"


def test_resolve_falls_back_to_record_fields_when_no_splits_entry():
    record = {
        "token": "999_unknown",
        "config": "pelvic_native",
        "lstv_label": "LUMBARIZATION",
        "n_lumbar_labels": 6,
        "has_l6": True,
    }
    splits_subtypes = {}
    assert _resolve_subtype_for_case(record, splits_subtypes) == "normal"


def test_resolve_fallback_normal_patient_pelvic_view():
    record = {
        "token": "999_normal",
        "config": "pelvic_native",
        "lstv_label": "",
        "n_lumbar_labels": 5,
        "has_l6": False,
    }
    splits_subtypes = {}
    assert _resolve_subtype_for_case(record, splits_subtypes) == "normal"


def test_resolve_fallback_sacr_count():
    record = {
        "token": "999_sc",
        "config": "fused",
        "lstv_label": "",
        "lstv_vertebral": "SACRALIZATION",
        "n_lumbar_labels": 4,
    }
    splits_subtypes = {}
    assert _resolve_subtype_for_case(record, splits_subtypes) == "sacr_count"


def test_resolve_fallback_sacr_count_pelvic_view_demoted():
    record = {
        "token": "999_sc",
        "config": "pelvic_native",
        "lstv_label": "",
        "lstv_vertebral": "SACRALIZATION",
        "n_lumbar_labels": 4,
    }
    splits_subtypes = {}
    assert _resolve_subtype_for_case(record, splits_subtypes) == "normal"


# Combinatorial completeness

def test_every_subtype_handled_for_every_config():
    for sub in SUBTYPES:
        for cfg in ("fused", "spine_only", "pelvic_native"):
            result = _refine_subtype_for_record_config(sub, cfg)
            assert result in SUBTYPES, (
                f"refinement of ({sub}, {cfg}) returned {result!r}, not in SUBTYPES"
            )


# Real-world regression cases

def test_real_world_colonog_267_pelvic_native():
    """COLONOG_267: lumb patient, pelvic_native record's label has
    only {7,8,9,10}. Refinement demotes to 'normal'."""
    record = {
        "token": "267",
        "config": "pelvic_native",
        "lstv_label": "LUMBARIZATION",
        "n_lumbar_labels": 6,
        "has_l6": True,
    }
    splits_subtypes = {"267": "lumb"}
    refined = _resolve_subtype_for_case(record, splits_subtypes)
    assert refined == "normal", (
        f"COLONOG_267 pelvic_native should be 'normal', got {refined!r}"
    )


def test_real_world_colonog_149_spine_only():
    """COLONOG_149: spine-only view of lumb patient (label set
    {1,2,3,4,5,6,10}). The lumbar region is fully present. spine_only
    must NOT demote lumb. (Regression test ensuring the new symmetric
    rule doesn't overshoot.)"""
    record = {
        "token": "149",
        "config": "spine_only",
        "lstv_label": "LUMBARIZATION",
        "n_lumbar_labels": 6,
        "has_l6": True,
    }
    splits_subtypes = {"149": "lumb"}
    refined = _resolve_subtype_for_case(record, splits_subtypes)
    assert refined == "lumb", (
        f"COLONOG_149 spine_only of lumb patient should stay 'lumb', got {refined!r}"
    )


# Standalone runner

if __name__ == "__main__":
    tests = [(name, obj) for name, obj in globals().items()
              if name.startswith("test_") and callable(obj)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            failed.append(name)
        except Exception as e:
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
            failed.append(name)
    print()
    print(f"  ran {len(tests)} tests, {len(failed)} failed")
    if failed:
        sys.exit(1)
