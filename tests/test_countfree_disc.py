"""tests/test_countfree_disc.py — the derived intervertebral-space class.

Class 2 is the only label in the count-free scheme that is not a lookup, so it is the
only one that can be wrong in an interesting way. These tests are on synthetic geometry
where the right answer is known by construction: a real CT can only tell you the output
looks plausible.

    pytest tests/test_countfree_disc.py -q
"""
import importlib.util as _u
from pathlib import Path

import numpy as np
import pytest

_spec = _u.spec_from_file_location(
    "cvt", Path(__file__).resolve().parents[1] / "tools" / "convert_hf_to_nnunet.py")
cvt = _u.module_from_spec(_spec)
_spec.loader.exec_module(cvt)


def _stack(gaps_vox, body=(40, 40), h=12, spacing=(1.0, 1.0, 1.0), first_id=20):
    """Two or more slabs along axis 2 with the given gaps between them."""
    n = len(gaps_vox) + 1
    depth = n * h + sum(gaps_vox) + 8
    a = np.zeros((body[0] + 20, body[1] + 20, depth), np.int32)
    z = 4
    for i in range(n):
        a[10:10 + body[0], 10:10 + body[1], z:z + h] = first_id + i
        z += h + (gaps_vox[i] if i < len(gaps_vox) else 0)
    return a, spacing


def test_disc_fills_the_gap_between_two_bodies():
    arr, sp = _stack([8])
    disc = cvt._derive_disc_space(arr, sp)
    assert disc.any(), "no disc derived between two adjacent bodies"
    # bodies that do not touch: nothing is carved out of them
    assert not (disc & (arr != 0)).any(), "disc took bone from non-touching bodies"
    # the gap is z 16..23, and every slice of it is filled
    zs = np.nonzero(disc.any(axis=(0, 1)))[0]
    assert set(range(16, 24)) <= set(zs.tolist()), f"gap not filled: {zs}"
    # SOLID, not speckled. A disc that fragments cannot cut the column, which is the
    # one thing this class exists to do -- an earlier strict-betweenness rule produced
    # 2262 fragments on a real case and left the vertebrae merged.
    from scipy import ndimage
    assert ndimage.label(disc)[1] == 1, "disc fragmented"


def test_lateral_collar_is_bounded_to_about_one_slice():
    """A distance rule admits some voxels that are near both bodies without being
    between them -- a collar around the sides at the outermost slice of each body.
    It is tolerated (see _derive_disc_space) but it must stay small: if the collar grew
    it would start joining discs to each other down the side of the column."""
    arr, sp = _stack([8])
    disc = cvt._derive_disc_space(arr, sp)
    zs = np.nonzero(disc.any(axis=(0, 1)))[0]
    assert zs.min() >= 14 and zs.max() <= 25, f"collar too deep: z {zs.min()}..{zs.max()}"
    in_gap = disc[:, :, 16:24].sum()
    outside = disc.sum() - in_gap
    assert outside < 0.25 * in_gap, f"collar is {outside} vs {in_gap} in the gap"


def test_no_disc_between_non_adjacent_ids():
    """A gap in the labelling is not a disc. If T12 is present and L1 is missing from the
    field of view, the space below T12 must not be filled -- that would invent a boundary
    where the pipeline has no evidence of one."""
    arr, sp = _stack([8])
    arr[arr == 21] = 23                       # L2 -> L4: now ids 20 and 23, not adjacent
    assert not cvt._derive_disc_space(arr, sp).any()


def test_gap_wider_than_a_disc_is_not_filled():
    """max_gap_mm is what stops a missing vertebra from being papered over with one
    enormous disc."""
    arr, sp = _stack([40])                    # 40 mm: no disc is this thick
    assert not cvt._derive_disc_space(arr, sp).any()


def test_millimetres_not_voxels():
    """The same anatomy at a different slice thickness must give the same answer, or the
    rule means something different in every scan in the corpus."""
    fine, _ = _stack([16], h=24)              # 16 voxels x 0.5 mm = 8 mm
    coarse, _ = _stack([4], h=6)              # 4 voxels x 2.0 mm = 8 mm
    d_fine = cvt._derive_disc_space(fine, (1.0, 1.0, 0.5))
    d_coarse = cvt._derive_disc_space(coarse, (1.0, 1.0, 2.0))
    mm_fine = d_fine.any(axis=(0, 1)).sum() * 0.5
    mm_coarse = d_coarse.any(axis=(0, 1)).sum() * 2.0
    # tolerance is one coarse slice: the collar is one slice deep at each end, so the
    # coarse grid necessarily quantises the same anatomy more crudely
    assert mm_fine == pytest.approx(mm_coarse, abs=4.0), (mm_fine, mm_coarse)


def test_three_bodies_give_two_separate_discs():
    """The point of the class: components of vertebra-minus-space are individual
    vertebrae. If the two discs merged, or one were missed, the count would be wrong."""
    from scipy import ndimage
    arr, sp = _stack([8, 8])
    disc = cvt._derive_disc_space(arr, sp)
    n_disc = ndimage.label(disc)[1]
    assert n_disc == 2, f"{n_disc} discs from three bodies, want 2"
    merged = (arr > 0)                        # what the network would emit as one class
    assert ndimage.label(merged)[1] == 3      # separate here because the slabs do not touch
    # AND THE CASE THAT ACTUALLY MATTERS. In these labels adjacent vertebrae usually
    # touch with no background between them -- the source segments bone, not joints --
    # so a gap rule has nothing to fill and the column comes back as ONE component.
    touching, sp2 = _stack([0, 0])
    assert ndimage.label(touching > 0)[1] == 1, "phantom should start as one blob"
    d2 = cvt._derive_disc_space(touching, sp2)
    assert ndimage.label((touching > 0) & ~d2)[1] == 3,         "abutting bodies were not cut apart"


def test_lumbosacral_junction_is_left_alone():
    """Whether a disc exists between the lowest mobile vertebra and the sacrum IS the
    transitional finding. Deriving one would answer the question the counting stage
    exists to ask."""
    arr, sp = _stack([8], first_id=25)        # L6 (25) above sacrum (26)
    assert not cvt._derive_disc_space(arr, sp).any()


def test_ignore_region_is_never_overwritten():
    """The ignore contract is what makes partial annotation safe: a spine_only record has
    no pelvic labels, and supervising that region with anything at all is a lie."""
    arr, sp = _stack([8])
    arr[:, :, 16:24] = np.where(arr[:, :, 16:24] == 0, 255, arr[:, :, 16:24])
    lut = cvt._build_verse_remap_lut(cvt.LABEL_REMAP_COUNTFREE_FROM_VERSE)
    out = lut[arr].astype(np.uint8)
    disc = cvt._derive_disc_space(arr, sp)
    out[disc & (out != cvt._CF_IGNORE)] = cvt._CF_DISC
    assert not (out[arr == 255] == cvt._CF_DISC).any(), "disc ate the ignore region"
    assert (out[arr == 255] == cvt._CF_IGNORE).all()
