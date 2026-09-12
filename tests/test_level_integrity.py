"""Every failure mode, built by hand, so the detector is known to fire on each.

A metric that has never been shown a failure it is supposed to catch is a decoration. Each
test below constructs the exact geometry it names -- six stacked lumbar bodies in a small
volume -- breaks it one way, and asserts both that the intended flag fires and that the
others stay quiet. The last part matters as much: a smear detector that also calls every
clean case smeared tells you nothing.

    python -m pytest tests/test_level_integrity.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import level_integrity as li  # noqa: E402

# Dataset813's lumbar ids, so the test exercises the real numbering.
L1, L2, L3, L4, L5, L6 = 5, 6, 7, 8, 9, 10
LUMBAR = [L1, L2, L3, L4, L5, L6]
NAMES = {5: "L1", 6: "L2", 7: "L3", 8: "L4", 9: "L5", 10: "L6", 11: "sacrum"}

# Six bodies stacked along axis 0, each 10 slices tall in a 64-slice volume, none of them
# touching the array boundary, each ~10*12*12 = 1440 voxels (over MIN_VOX).
Z0, H, SHAPE = 2, 10, (64, 12, 12)


def stack(ids=LUMBAR) -> np.ndarray:
    v = np.zeros(SHAPE, np.int32)
    for k, cid in enumerate(ids):
        v[Z0 + k * H: Z0 + (k + 1) * H] = cid
    return v


def test_perfect_prediction_is_quiet():
    gt = stack()
    r = li.report(gt.copy(), gt, LUMBAR, L6, NAMES)
    assert r["collapse"]["self_frac"] == pytest.approx(1.0)
    assert r["collapse"]["collapsed"] is False
    assert r["purity"]["n_smeared"] == 0
    assert r["purity"]["mean_purity"] == pytest.approx(1.0)
    assert r["spread"]["n_merged"] == 0
    assert r["count"]["deficit"] == 0
    assert r["count"]["n_pred_lumbar"] == 6


def test_collapse_l6_into_l5():
    """Every L6 voxel called L5 -- the rebuttal claim."""
    gt = stack()
    pred = gt.copy()
    pred[pred == L6] = L5
    r = li.report(pred, gt, LUMBAR, L6, NAMES)

    c = r["collapse"]
    assert c["self_frac"] == pytest.approx(0.0)
    assert c["top_other"]["name"] == "L5"
    assert c["top_other"]["frac"] == pytest.approx(1.0)
    assert c["collapsed"] is True

    # and the count is short by one: five names for six bodies
    assert r["count"]["n_gt_lumbar"] == 6
    assert r["count"]["n_pred_lumbar"] == 5
    assert r["count"]["deficit"] == 1
    assert r["count"]["missing"] == ["L6"]

    # one name now covers two bodies
    assert r["spread"]["n_merged"] == 1
    assert r["spread"]["merged"][0]["pred_name"] == "L5"
    assert r["spread"]["merged"][0]["covers_names"] == ["L5", "L6"]

    # L6's body is not SMEARED -- it is wholly and consistently misnamed
    l6body = [b for b in r["purity"]["bodies"] if b["cid"] == L6][0]
    assert l6body["smeared"] is False
    assert l6body["misnamed"] is True
    assert l6body["modal_name"] == "L5"


def test_smear_one_body_across_two_names():
    """A level boundary through the middle of a vertebra: half L5, half L6."""
    gt = stack()
    pred = gt.copy()
    lo = Z0 + 4 * H                      # the L5 body
    pred[lo: lo + H // 2] = L6           # its top half renamed
    r = li.report(pred, gt, LUMBAR, L6, NAMES)

    sm = r["purity"]["smeared"]
    assert len(sm) == 1
    assert sm[0]["name"] == "L5"
    assert sm[0]["purity"] == pytest.approx(0.5)
    assert sm[0]["second_frac"] == pytest.approx(0.5)
    assert sm[0]["second_name"] in ("L6", "L5")
    assert sm[0]["n_names_over_smear"] == 2

    # every lumbar name is still in use, so a count check alone would MISS this
    assert r["count"]["deficit"] == 0
    # which is exactly why the smear flag has to exist
    assert r["purity"]["n_smeared"] == 1


def test_merge_two_bodies_under_one_name():
    """Two neighbouring bodies both called L5 -- what split_tall_component exists for."""
    gt = stack()
    pred = gt.copy()
    pred[pred == L4] = L5
    r = li.report(pred, gt, LUMBAR, L6, NAMES)

    assert r["spread"]["n_merged"] == 1
    m = r["spread"]["merged"][0]
    assert m["pred_name"] == "L5"
    assert sorted(m["covers_names"]) == ["L4", "L5"]
    assert r["count"]["deficit"] == 1
    assert r["count"]["missing"] == ["L4"]
    assert r["purity"]["n_smeared"] == 0          # both bodies are internally consistent


def test_clipped_body_is_not_a_smear():
    """A body cut by the patch edge is partial, not broken, and must not be flagged.

    This is the case the boundary guard was written for, and it does not need one: the
    visible part of a clipped body still carries a single predicted name, so purity stays
    high on its own.
    """
    gt = stack()
    gt[0:Z0] = L1                       # L1 runs off the top of the array
    pred = gt.copy()

    p = li.body_purity(pred, gt, LUMBAR, NAMES)
    assert p["n_bodies"] == 6, "a clipped body is still measurable"
    l1 = [b for b in p["bodies"] if b["cid"] == L1][0]
    assert l1["purity"] == pytest.approx(1.0)
    assert l1["smeared"] is False
    assert p["n_smeared"] == 0


def test_sliver_below_floor_is_dropped():
    """Only a few voxels of a body survive the patch edge: too little to judge."""
    gt = stack()
    gt[gt == L1] = 0
    gt[Z0, 0:3, 0:3] = L1               # 9 voxels, far under MIN_VOX
    pred = gt.copy()
    p = li.body_purity(pred, gt, LUMBAR, NAMES)
    assert L1 not in [b["cid"] for b in p["bodies"]]
    assert p["n_bodies"] == 5


def test_absent_focus_class_reports_absent_not_zero():
    """A case with no L6 must not read as 'L6 collapsed'."""
    gt = stack([L1, L2, L3, L4, L5, L5])          # five names, no L6
    r = li.report(gt.copy(), gt, LUMBAR, L6, NAMES)
    assert r["collapse"]["present"] is False
    assert "collapsed" not in r["collapse"]


def test_specks_are_ignored():
    """A handful of stray voxels is not a lumbar name in use."""
    gt = stack()
    pred = gt.copy()
    pred[pred == L6] = L5
    pred[0, 0, 0] = L6                            # one voxel of L6
    r = li.report(pred, gt, LUMBAR, L6, NAMES)
    assert r["count"]["n_pred_lumbar"] == 5, "a single voxel must not count as a name in use"
    assert r["count"]["deficit"] == 1


def test_aggregate_reports_rates_and_the_thief():
    gt = stack()
    collapsed = gt.copy(); collapsed[collapsed == L6] = L5
    clean = gt.copy()
    reps = [li.report(collapsed, gt, LUMBAR, L6, NAMES),
            li.report(clean, gt, LUMBAR, L6, NAMES)]
    agg = li.aggregate(reps)
    assert agg["level_integrity/n_cases"] == 2
    assert agg["level_integrity/focus_collapse_rate"] == pytest.approx(0.5)
    assert agg["level_integrity/count_wrong_rate"] == pytest.approx(0.5)
    assert agg["level_integrity/lumbar_count_deficit"] == pytest.approx(0.5)
    # the metric names the class that is eating L6, rather than leaving it to be guessed
    assert agg["level_integrity/focus_taken_by/L5"] == pytest.approx(0.5)


def test_format_report_names_the_failure_in_words():
    gt = stack()
    pred = gt.copy(); pred[pred == L6] = L5
    text = "\n".join(li.format_report(li.report(pred, gt, LUMBAR, L6, NAMES)))
    assert "COLLAPSE" in text
    assert "L5 takes 100.0%" in text
    assert "COUNT WRONG" in text
    assert "never predicted: L6" in text


def test_smear_format_names_both_classes():
    gt = stack()
    pred = gt.copy()
    lo = Z0 + 4 * H
    pred[lo: lo + H // 2] = L6
    text = "\n".join(li.format_report(li.report(pred, gt, LUMBAR, L6, NAMES)))
    assert "SMEARED" in text
    assert "one body, two names" in text
    # the numeric id must never leak into the human-readable line
    assert "% 10 " not in text and "% 9 " not in text
