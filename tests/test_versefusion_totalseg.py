"""The composition order, tested without a GPU.

Every question here is about PRECEDENCE -- who wins where two sources disagree -- and that
is pure array arithmetic. TotalSegmentator's output is injected as plain boolean masks, so
the rules can be exercised exactly, including the one case the whole dataset exists for:
a sixth lumbar body that a model calls sacrum.

    python -m pytest tests/test_versefusion_totalseg.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import versefusion_totalseg as vt  # noqa: E402
from tools.convert_versefusion import IGNORE  # noqa: E402

SHAPE = (6, 6, 12)
SOFT, BONE = 40.0, 400.0


def blank():
    return (np.zeros(SHAPE, np.float32),            # verse
            {},                                      # ts
            np.full(SHAPE, SOFT, np.float32))        # ct


def test_the_label_ids_match_dataset813_exactly():
    """Pinned against the dataset.json, because a rib off by one is invisible."""
    assert vt.TS_TO_813["sacrum"] == 11
    assert vt.TS_TO_813["rib_left_1"] == 12
    assert vt.TS_TO_813["rib_left_11"] == 22
    assert vt.TS_TO_813["rib_left_12"] == 23      # rib12_left, not rib_left_12's slot
    assert vt.TS_TO_813["rib_right_1"] == 25
    assert vt.TS_TO_813["rib_right_11"] == 35
    assert vt.TS_TO_813["rib_right_12"] == 36
    assert vt.TS_TO_813["hip_left"] == 40 and vt.TS_TO_813["hip_right"] == 41
    # one femur class for both sides, which is what the scheme carries
    assert vt.TS_TO_813["femur_left"] == vt.TS_TO_813["femur_right"] == 42
    # and nothing maps onto a rib13 slot, which TS cannot produce
    assert 24 not in vt.TS_TO_813.values() and 37 not in vt.TS_TO_813.values()


def test_verse_outranks_totalsegmentator_on_a_lumbarized_body():
    """THE case this dataset is about: TS calls the L6 a sacrum, VerSe says L6."""
    verse, ts, ct = blank()
    body = (slice(None), slice(None), slice(4, 8))
    verse[body] = 25                                  # VerSe: this is L6
    m = np.zeros(SHAPE, bool); m[body] = True
    ts["sacrum"] = m                                  # TS: this is sacrum
    out, stats = vt.compose(verse, ts, ct, BONE)

    assert np.unique(out[body]).tolist() == [10], "VerSe's L6 must survive TS's sacrum"
    assert stats["verse_overwrote_ts_voxels"] == int(m.sum())


def test_totalsegmentator_fills_what_verse_leaves_out():
    verse, ts, ct = blank()
    for name, sl in (("sacrum", 0), ("hip_left", 1), ("hip_right", 2),
                     ("femur_left", 3), ("rib_left_1", 4), ("rib_right_12", 5)):
        m = np.zeros(SHAPE, bool); m[:, :, sl] = True
        ts[name] = m
    out, _ = vt.compose(verse, ts, ct, BONE)
    assert np.unique(out[:, :, 0]).tolist() == [11]
    assert np.unique(out[:, :, 1]).tolist() == [40]
    assert np.unique(out[:, :, 2]).tolist() == [41]
    assert np.unique(out[:, :, 3]).tolist() == [42]
    assert np.unique(out[:, :, 4]).tolist() == [12]
    assert np.unique(out[:, :, 5]).tolist() == [36]


def test_bones_the_scheme_cannot_name_are_ignored_not_background():
    """Clavicle, scapula, humerus, sternum: real bone, no class, so ignore."""
    verse, ts, ct = blank()
    for i, name in enumerate(vt.TS_UNNAMEABLE):
        m = np.zeros(SHAPE, bool); m[:, :, i] = True
        ts[name] = m
    out, stats = vt.compose(verse, ts, ct, BONE)
    for i in range(len(vt.TS_UNNAMEABLE)):
        assert np.unique(out[:, :, i]).tolist() == [IGNORE], vt.TS_UNNAMEABLE[i]
    assert all(f"ts_ignored/{n}" in stats for n in vt.TS_UNNAMEABLE)


def test_named_ignore_beats_the_hu_guess_and_the_report_separates_them():
    """A scapula TS found must be counted as named, not as 'dense but unknown'."""
    verse, ts, ct = blank()
    m = np.zeros(SHAPE, bool); m[:, :, 0] = True
    ts["scapula_left"] = m
    ct[:, :, 0] = BONE                                 # it is also dense
    ct[:, :, 1] = BONE                                 # and here is bone TS did not name
    out, stats = vt.compose(verse, ts, ct, 150.0)
    assert np.unique(out[:, :, 0]).tolist() == [IGNORE]
    assert np.unique(out[:, :, 1]).tolist() == [IGNORE]
    assert stats["ts_ignored/scapula_left"] == 36
    assert stats["ignore/dense_but_unnamed"] == 36, "the two rules stay distinguishable"


def test_soft_tissue_is_background_even_beside_bone():
    verse, ts, ct = blank()
    out, stats = vt.compose(verse, ts, ct, 150.0)
    assert np.unique(out).tolist() == [0]
    assert stats["background"] == int(np.prod(SHAPE))
    assert stats["ignore/dense_but_unnamed"] == 0


def test_cervical_vertebrae_are_ignored_even_when_ts_named_a_bone_there():
    """VerSe's C-spine has no class; it must not fall back to a TS class either."""
    verse, ts, ct = blank()
    verse[:, :, 0] = 3                                 # C3
    m = np.zeros(SHAPE, bool); m[:, :, 0] = True
    ts["sternum"] = m
    out, _ = vt.compose(verse, ts, ct, BONE)
    assert np.unique(out[:, :, 0]).tolist() == [IGNORE]


def test_rois_requested_exclude_vertebrae():
    """TS vertebrae are never asked for: VerSe's are better and every ROI costs time."""
    assert not any(r.startswith("vertebrae") for r in vt.TS_ROIS)
    assert set(vt.TS_ROIS) == set(vt.TS_TO_813) | set(vt.TS_UNNAMEABLE)


def test_every_ts_target_is_a_real_dataset813_id():
    from tools.convert_versefusion import IGNORE as IG
    for name, cid in vt.TS_TO_813.items():
        assert 1 <= cid <= 43, f"{name} -> {cid} is outside the foreground range"
        assert cid != IG
