"""The two operations that can silently ruin this corpus, each shown to work and to fail.

A reorientation that transposes a label away from its CT, and a label map that sends a
vertebra to the wrong class, both produce a dataset that trains happily and means nothing.
Neither raises on its own. So both are exercised here against volumes built by hand, where
the right answer is known by construction.

    python -m pytest tests/test_convert_versefusion.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import convert_versefusion as cv  # noqa: E402

BONE, SOFT = 400.0, 40.0


def pair(axcodes=("L", "A", "S"), shape=(8, 9, 10)):
    """A CT and a mask sharing one affine, in a chosen orientation."""
    aff = nib.orientations.inv_ornt_aff(
        nib.orientations.axcodes2ornt(axcodes), shape) @ np.eye(4)
    aff = np.linalg.inv(aff)
    ct = np.full(shape, SOFT, np.float32)
    msk = np.zeros(shape, np.float64)
    return ct, msk, aff


# ---------------------------------------------------------------- label mapping

def test_every_named_level_lands_on_its_dataset813_class():
    shape = (4, 4, 12)
    verse = np.zeros(shape, np.float64)
    ct = np.full(shape, SOFT, np.float32)
    order = [17, 18, 19, 28, 20, 21, 22, 23, 24, 25, 26]      # T10..sacrum
    for k, v in enumerate(order):
        verse[:, :, k] = v
    out, stats = cv.map_labels(verse, ct, BONE)
    for k, v in enumerate(order):
        want = cv.VERSE_TO_813[v]
        got = np.unique(out[:, :, k])
        assert got.tolist() == [want], f"VerSe {v} -> {got}, wanted {want}"
    assert out.dtype == np.uint8
    # the two classes this whole exercise exists for
    assert stats["L6"] > 0 and stats["T13"] > 0


def test_l6_lands_on_ten_not_on_l5():
    """The single mapping whose failure would be invisible and fatal."""
    verse = np.zeros((4, 4, 4), np.float64)
    verse[:] = 25
    out, _ = cv.map_labels(verse, np.full((4, 4, 4), SOFT, np.float32), BONE)
    assert np.unique(out).tolist() == [10], "L6 must be 10; 9 is L5 and would be silent"


def test_cervical_and_coccyx_are_ignored_not_background():
    """Real bone with no class in the scheme. Background would teach a falsehood."""
    shape = (4, 4, 6)
    verse = np.zeros(shape, np.float64)
    verse[:, :, 0] = 1        # C1
    verse[:, :, 1] = 16       # T9
    verse[:, :, 2] = 27       # coccyx
    verse[:, :, 3] = 20       # L1, which IS nameable
    out, stats = cv.map_labels(verse, np.full(shape, SOFT, np.float32), BONE)
    assert np.unique(out[:, :, 0]).tolist() == [cv.IGNORE]
    assert np.unique(out[:, :, 1]).tolist() == [cv.IGNORE]
    assert np.unique(out[:, :, 2]).tolist() == [cv.IGNORE]
    assert np.unique(out[:, :, 3]).tolist() == [5]
    assert stats["ignore_unnameable_vertebrae"] == 3 * 16


def test_unlabelled_bone_is_ignored_and_soft_tissue_is_background():
    """A rib must never be called background; bowel gas safely can be."""
    shape = (4, 4, 4)
    verse = np.zeros(shape, np.float64)
    ct = np.full(shape, SOFT, np.float32)
    ct[:, :, 0] = BONE                      # unlabelled bone: a rib, say
    out, stats = cv.map_labels(verse, ct, 150.0)
    assert np.unique(out[:, :, 0]).tolist() == [cv.IGNORE], "unlabelled bone -> ignore"
    assert np.unique(out[:, :, 1:]).tolist() == [0], "soft tissue -> background"
    assert stats["ignore_unlabelled_bone"] == 16
    assert stats["background"] == 48


def test_labelled_vertebra_survives_the_bone_rule():
    """The bone rule must not overwrite a class it sits on top of."""
    shape = (4, 4, 4)
    verse = np.full(shape, 25.0)            # all L6
    ct = np.full(shape, BONE, np.float32)   # and all of it is bone-dense
    out, _ = cv.map_labels(verse, ct, 150.0)
    assert np.unique(out).tolist() == [10], "a named vertebra outranks the maybe-bone rule"


def test_float_ids_are_rounded_not_truncated():
    """359 of 374 masks are float64; 24.999999 is L5, not L4."""
    verse = np.full((3, 3, 3), 24.0 - 1e-9)
    out, _ = cv.map_labels(verse, np.full((3, 3, 3), SOFT, np.float32), BONE)
    assert np.unique(out).tolist() == [9], "L5"


# ---------------------------------------------------------------- reorientation

@pytest.mark.parametrize("axcodes", [("L", "A", "S"), ("P", "S", "R"), ("L", "P", "S"),
                                     ("L", "S", "P"), ("P", "I", "R")])
def test_reorientation_lands_on_pir_and_keeps_the_pair_together(axcodes):
    shape = (5, 6, 7)
    rng = np.random.default_rng(0)
    ct_a = rng.normal(0, 300, shape).astype(np.float32)
    msk_a = np.zeros(shape, np.float64)
    msk_a[1:3, 2:4, 3:6] = 25                      # an L6 somewhere off-centre

    ornt = nib.orientations.axcodes2ornt(axcodes)
    aff = np.linalg.inv(nib.orientations.inv_ornt_aff(ornt, shape))
    ct_i = nib.Nifti1Image(ct_a, aff)
    msk_i = nib.Nifti1Image(msk_a, aff)
    assert nib.aff2axcodes(ct_i.affine) == axcodes

    ct2, msk2 = cv.reorient_pair(ct_i, msk_i)
    assert nib.aff2axcodes(ct2.affine) == cv.TARGET_AXCODES
    assert nib.aff2axcodes(msk2.affine) == cv.TARGET_AXCODES
    assert ct2.shape == msk2.shape
    assert np.allclose(ct2.affine, msk2.affine)

    # EXACT, not approximate: reorientation is a transpose and a flip, so every voxel
    # value survives and the label histogram cannot change.
    a2 = np.asanyarray(msk2.dataobj)
    assert sorted(np.unique(a2).tolist()) == [0.0, 25.0]
    assert int((a2 == 25).sum()) == int((msk_a == 25).sum())
    assert np.allclose(np.sort(np.asanyarray(ct2.dataobj).ravel()), np.sort(ct_a.ravel()))


def test_reorientation_keeps_the_label_on_the_same_anatomy():
    """The failure this guards: a label transposed away from its CT.

    A marker is written into BOTH volumes at the same voxel. After reorientation it must
    still be at the same voxel in both -- wherever that voxel has moved to.
    """
    shape = (5, 6, 7)
    ct_a = np.zeros(shape, np.float32)
    msk_a = np.zeros(shape, np.float64)
    ct_a[1, 2, 3] = 1000.0
    msk_a[1, 2, 3] = 25

    ornt = nib.orientations.axcodes2ornt(("L", "A", "S"))
    aff = np.linalg.inv(nib.orientations.inv_ornt_aff(ornt, shape))
    ct2, msk2 = cv.reorient_pair(nib.Nifti1Image(ct_a, aff),
                                 nib.Nifti1Image(msk_a, aff))
    c = np.argwhere(np.asanyarray(ct2.dataobj) == 1000.0)
    m = np.argwhere(np.asanyarray(msk2.dataobj) == 25)
    assert len(c) == 1 and len(m) == 1
    assert c.tolist() == m.tolist(), "label and CT parted company"


def test_float32_roundoff_in_the_affine_is_not_a_mismatch():
    """The check that rejected six good cases, now pinned.

    VerSeFusion's masks were written by a different tool from its CTs (sform_code 2 against
    1), so the two affines differ by float32 round-off -- 6e-6 mm on one case. Reorienting
    composes the affine with inv_ornt_aff(ornt, shape), multiplying that by the array
    dimension, so on a 1119-slice volume it surfaces as 0.041 mm. That is 4% of a voxel and
    must not be an error.
    """
    shape = (64, 64, 1119)
    ornt = nib.orientations.axcodes2ornt(("L", "A", "S"))
    aff = np.linalg.inv(nib.orientations.inv_ornt_aff(ornt, shape))
    noisy = aff.copy()
    noisy[:3, :3] += 3.7e-5                    # the real magnitude, in the rotation block

    ct2, msk2 = cv.reorient_pair(nib.Nifti1Image(np.zeros(shape, np.float32), aff),
                                 nib.Nifti1Image(np.zeros(shape, np.float64), noisy))
    assert nib.aff2axcodes(ct2.affine) == cv.TARGET_AXCODES
    # and the element-wise comparison that used to gate this would indeed have failed
    assert not np.allclose(ct2.affine, msk2.affine, atol=1e-3)


def test_a_real_shift_is_still_refused_at_every_scale():
    """Half a voxel is a real disagreement whether the voxel is 0.27 mm or 3 mm."""
    shape = (32, 32, 32)
    ornt = nib.orientations.axcodes2ornt(("L", "A", "S"))
    for vox in (0.27, 1.0, 3.0):
        aff = np.linalg.inv(nib.orientations.inv_ornt_aff(ornt, shape))
        aff[:3, :3] *= vox
        bad = aff.copy()
        bad[0, 3] += 0.5 * vox
        with pytest.raises(ValueError, match="disagree"):
            cv.reorient_pair(nib.Nifti1Image(np.zeros(shape, np.float32), aff),
                             nib.Nifti1Image(np.zeros(shape, np.float64), bad))


def test_mismatched_affines_are_refused():
    """If a pair does not describe the same space, no reorientation can save it."""
    shape = (4, 4, 4)
    ornt = nib.orientations.axcodes2ornt(("L", "A", "S"))
    aff = np.linalg.inv(nib.orientations.inv_ornt_aff(ornt, shape))
    bad = aff.copy()
    bad[0, 3] += 25.0                              # shifted 25 mm
    with pytest.raises(ValueError):
        cv.reorient_pair(nib.Nifti1Image(np.zeros(shape, np.float32), aff),
                         nib.Nifti1Image(np.zeros(shape, np.float64), bad))
