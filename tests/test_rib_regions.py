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
