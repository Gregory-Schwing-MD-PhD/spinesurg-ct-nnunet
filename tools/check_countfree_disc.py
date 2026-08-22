"""tools/check_countfree_disc.py — does the derived disc class actually let you count?

The disc-space class earns its place only if connected components of
vertebra-minus-space come out as individual vertebrae. Dice against nothing, or a
plausible-looking render, would not answer that. This asks the functional question
directly: how many components do you get, and is it the number of vertebrae the source
labels say are there?

A component count that is too LOW means two vertebrae merged -- the failure the class
exists to prevent, and the one that silently costs a level in the count. Too HIGH means
the disc cut a vertebra in half, which is the opposite failure and just as bad.

    python tools/check_countfree_disc.py --labels /path/to/v5_final --n 12
"""
from __future__ import annotations

import argparse
import importlib.util as _u
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

_spec = _u.spec_from_file_location(
    "cvt", Path(__file__).resolve().parent / "convert_hf_to_nnunet.py")
cvt = _u.module_from_spec(_spec)
_spec.loader.exec_module(cvt)

MIN_VOX = 200            # below this a "vertebra" is a field-of-view speck


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--cases", default="")
    a = ap.parse_args()

    files = sorted(Path(a.labels).glob("*_label.nii.gz"))
    if a.cases:
        want = {c.strip() for c in a.cases.split(",") if c.strip()}
        files = [f for f in files if f.name.split("_")[0] in want]
    files = files[: a.n]
    if not files:
        print("no label files found")
        return 1

    lut = cvt._build_verse_remap_lut(cvt.LABEL_REMAP_COUNTFREE_FROM_VERSE)
    print(f"  {'case':6} {'verts':>5} {'comps':>5} {'match':>6} {'discs':>5} "
          f"{'disc mm':>8} {'stray':>6}")
    ok = 0
    for fp in files:
        img = nib.load(str(fp))
        arr = np.asarray(img.dataobj).astype(np.int32)
        zooms = tuple(float(z) for z in img.header.get_zooms()[:3])

        # how many vertebrae the SOURCE says are there, ignoring FOV specks
        ids = [v for v in cvt._CF_VERTEBRA_IDS if (arr == v).sum() >= MIN_VOX]
        n_src = len(ids)

        disc = cvt._derive_disc_space(arr, zooms)
        out = lut[np.clip(arr, 0, len(lut) - 1)].astype(np.uint8)
        out[disc & (out != cvt._CF_IGNORE)] = cvt._CF_DISC

        vert = out == cvt._CF_VERTEBRA
        lab_c, n_comp = ndimage.label(vert)
        sizes = ndimage.sum(vert, lab_c, range(1, n_comp + 1))
        n_big = int((sizes >= MIN_VOX).sum())

        n_disc = ndimage.label(disc)[1]
        thick = (disc.sum() * float(np.prod(zooms)) / max(n_disc, 1)) ** (1 / 3)

        # disc voxels that ended up nowhere near two vertebrae would be a leak
        # disc voxels that did not land, i.e. were held back by the ignore guard
        stray = int((disc & (out != cvt._CF_DISC)).sum())

        hit = "yes" if n_big == n_src else "NO"
        ok += n_big == n_src
        print(f"  {fp.name.split('_')[0]:6} {n_src:5d} {n_big:5d} {hit:>6} "
              f"{n_disc:5d} {thick:8.1f} {stray:6d}")

    print(f"\n  components match the source vertebra count in {ok}/{len(files)} cases")
    print("  a LOW count means two vertebrae merged -- the silent off-by-one this class")
    print("  exists to prevent. A HIGH count means a disc cut a vertebra in two.")
    return 0 if ok == len(files) else 2


if __name__ == "__main__":
    sys.exit(main())
