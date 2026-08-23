"""tests/test_rib_regions.py — rib-bearing status, and the region scheme around it.

The promotion is the only part of this scheme that is geometry rather than a lookup, so
it is the only part that can be wrong in an interesting way. The cases below are the ones
that decide a transitional phenotype.
"""
import importlib.util as _u
from pathlib import Path

import numpy as np

_spec = _u.spec_from_file_location(
    "cvt", Path(__file__).resolve().parents[1] / "tools" / "convert_hf_to_nnunet.py")
cvt = _u.module_from_spec(_spec)
_spec.loader.exec_module(cvt)

SP = (1.0, 1.0, 1.0)


def _spine(n=4, first=19, gap=4, with_rib_on=(), rib_id=34, rib_gap=2):
    """n stacked vertebra bodies; ribs placed beside the ones named."""
    h, depth = 12, n * (12 + gap) + 10
    a = np.zeros((80, 60, depth), np.int32)
    z = 4
    for i in range(n):
        vid = first + i
        a[20:60, 20:40, z:z + h] = vid
        if vid in with_rib_on:
            # a rib head a few voxels lateral, separated by a joint space
            a[60 + rib_gap:70 + rib_gap, 25:35, z + 2:z + 8] = rib_id
        z += h + gap
    return a


def test_a_vertebra_with_a_rib_beside_it_is_rib_bearing():
    arr = _spine(with_rib_on=(19,))
    assert cvt._rib_bearing_vertebrae(arr, SP) == {19}


def test_a_vertebra_without_one_is_not():
    arr = _spine()
    assert cvt._rib_bearing_vertebrae(arr, SP) == set()


def test_a_lumbar_rib_makes_its_vertebra_rib_bearing():
    """THE CASE THE WHOLE SCHEME EXISTS FOR. Ids 74/75 are ribs borne on a lumbar body.
    Treating them as anything but ribs would decide the transitional question inside the
    label, which is exactly what must not happen: a rib on the first lumbar-type vertebra
    shortens the rib-free interval to four, and that IS the phenotype."""
    arr = _spine(first=20, with_rib_on=(20,), rib_id=74)
    assert cvt._rib_bearing_vertebrae(arr, SP) == {20}


def test_the_next_vertebra_down_is_not_dragged_in():
    """A rib head sits within millimetres of its own body and a whole body height from
    the next one. If the threshold could not tell those apart it would promote the level
    below every rib and the count would be wrong everywhere, not just at the junction."""
    arr = _spine(n=4, first=17, with_rib_on=(19,))
    assert cvt._rib_bearing_vertebrae(arr, SP) == {19}


def test_promotion_reaches_the_written_label():
    arr = _spine(first=19, with_rib_on=(19,))
    lut = cvt._build_verse_remap_lut(cvt.LABEL_REMAP_RIB_REGIONS_FROM_VERSE)
    out = lut[arr].astype(np.uint8)
    bearing = cvt._rib_bearing_vertebrae(arr, SP)
    out[np.isin(arr, list(bearing))] = cvt._RR_VERT_RIB
    assert (out[arr == 19] == cvt._RR_VERT_RIB).all()
    assert (out[arr == 20] == cvt._RR_VERT_PLAIN).all()
    # and the nesting the regions rely on actually holds
    vert = np.isin(out, cvt.LABEL_NAMES_RIB_REGIONS["vertebra"])
    rib_bearing = np.isin(out, cvt.LABEL_NAMES_RIB_REGIONS["rib_bearing"])
    assert (rib_bearing & ~vert).sum() == 0, "rib_bearing must be nested inside vertebra"


def test_regions_are_ordered_encompassing_first():
    """regions_class_order paints in order and later wins, so a substructure listed
    before the region containing it would be painted over and erased."""
    keys = [k for k in cvt.LABEL_NAMES_RIB_REGIONS if k not in ("background", "ignore")]
    assert keys.index("vertebra") < keys.index("rib_bearing")
    assert len(cvt.REGIONS_CLASS_ORDER_RIB) == len(keys)
    assert cvt.LABEL_NAMES_RIB_REGIONS["ignore"] not in cvt.REGIONS_CLASS_ORDER_RIB


def test_ignore_is_the_highest_value():
    vals = []
    for v in cvt.LABEL_NAMES_RIB_REGIONS.values():
        vals.extend(v if isinstance(v, list) else [v])
    assert cvt.LABEL_NAMES_RIB_REGIONS["ignore"] == max(vals)


# ---------------------------------------------------------------------------
# --oneshot: the 21-class target that makes the hypoplastic-twelfth-rib versus
# lumbar-rib discrimination measurable at all.
# ---------------------------------------------------------------------------

def test_oneshot_keeps_the_confusable_pair_apart():
    """THE WHOLE POINT OF THE SCHEME. rib12 and lumbar_rib must be different classes.
    Pooling them would hide the comparison inside a class that is easy for other
    reasons, and the model could then be neither right nor wrong about it."""
    lut = cvt._build_verse_remap_lut(cvt.LABEL_REMAP_ONESHOT_FROM_VERSE)
    n = cvt.LABEL_NAMES_ONESHOT
    assert lut[45] == n["rib12_left"] and lut[57] == n["rib12_right"]
    assert lut[74] == n["lumbar_rib_left"] and lut[75] == n["lumbar_rib_right"]
    assert lut[45] != lut[74] and lut[57] != lut[75]


def test_oneshot_names_the_vertebra_that_names_the_rib():
    """T12 and L1 must be separable classes: the costal facet on the BODY is what says
    whether the small rib beside it is a twelfth or a lumbar one."""
    lut = cvt._build_verse_remap_lut(cvt.LABEL_REMAP_ONESHOT_FROM_VERSE)
    n = cvt.LABEL_NAMES_ONESHOT
    assert lut[19] == n["T12"] and lut[20] == n["L1"] and lut[19] != lut[20]
    for src, name in ((17, "T10"), (18, "T11"), (24, "L5"), (25, "L6")):
        assert lut[src] == n[name]


def test_oneshot_pools_ribs_one_to_eleven():
    """Only the twelfth is confusable. Ribs 1-11 are pooled so the network is not asked
    to number them, which is the same non-local problem as vertebral identity."""
    lut = cvt._build_verse_remap_lut(cvt.LABEL_REMAP_ONESHOT_FROM_VERSE)
    n = cvt.LABEL_NAMES_ONESHOT
    for src in range(34, 45):
        assert lut[src] == n["rib_left"]
    for src in range(46, 57):
        assert lut[src] == n["rib_right"]


def test_oneshot_label_values_are_contiguous_and_ignore_is_last():
    """nnU-Net builds its output head from the sorted label values, so a gap or a
    non-maximal ignore label silently misaligns channels against class ids."""
    vals = sorted(cvt.LABEL_NAMES_ONESHOT.values())
    assert vals == list(range(len(vals))), vals
    assert cvt.LABEL_NAMES_ONESHOT["ignore"] == max(vals)


def test_oneshot_drops_what_is_outside_the_question():
    """Cervical and upper thoracic vertebrae, coccyx, soft tissue and hardware are not
    classes here and must fall to background rather than leak through as stray ids that
    nnU-Net's integrity check would reject."""
    lut = cvt._build_verse_remap_lut(cvt.LABEL_REMAP_ONESHOT_FROM_VERSE)
    for src in (1, 7, 16, 27, 60, 73, 76, 79):
        assert lut[src] == 0, src
