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
    print(f"  {'case':6} {'verts':>5} {'pieces':>7} {'comps':>5} {'match':>6} "
          f"{'discs':>5} {'disc mm':>8} {'stray':>6}")
    ok = 0
    for fp in files:
        img = nib.load(str(fp))
        arr = np.asarray(img.dataobj).astype(np.int32)
        zooms = tuple(float(z) for z in img.header.get_zooms()[:3])

        # How many vertebrae the SOURCE says are there, ignoring FOV specks.
        #
        # AND how many CONNECTED PIECES those ids are already in. The two are not the
        # same, and conflating them made this check blame the disc derivation for
        # damage it had not done: on case 0001 the topmost vertebra came out as two
        # components, and it was two components in the source before anything was
        # derived -- a 391-voxel speck detached from the label. Counting components
        # against ids compares a connectivity fact to a naming fact. The derivation can
        # only be held responsible for the difference between them.
        #
        # And that speck is DELIBERATE. strip_vertebra_speckle.py in the
        # CTSpinoPelvic1K repo keeps a disconnected piece when it touches a face of the
        # volume, because the scan cut it rather than the annotator -- largest-component-
        # wins was tried and would have deleted 40% of a vertebra a human had drawn. Case
        # 0001's id 15 is the topmost vertebra in its field of view. So a source in more
        # than one piece is a property of the release, not damage, and the counting stage
        # downstream has to cope with it either way.
        ids = [v for v in cvt._CF_VERTEBRA_IDS if (arr == v).sum() >= MIN_VOX]
        n_src = len(ids)
        n_src_pieces = 0
        broken = []
        for v in ids:
            lv, nv = ndimage.label(arr == v)
            szv = ndimage.sum(arr == v, lv, range(1, nv + 1))
            k = int((szv >= MIN_VOX).sum())
            n_src_pieces += k
            if k > 1:
                broken.append(f"{v}x{k}")

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

        # the derivation's job is to leave the source's own pieces intact and to add
        # no new ones, so THAT is what is scored
        hit = "yes" if n_big == n_src_pieces else "NO"
        ok += n_big == n_src_pieces
        note = ("already split in source: " + ",".join(broken)) if broken else ""
        print(f"  {fp.name.split('_')[0]:6} {n_src:5d} {n_src_pieces:7d} {n_big:5d} "
              f"{hit:>6} {n_disc:5d} {thick:8.1f} {stray:6d}  {note}")

        if n_big != n_src_pieces:
            # A count that is off says nothing about WHY. Each surviving component is
            # named by the source ids it covers: a component spanning two ids is a
            # failed cut, one covering a fraction of a single id is a severed piece,
            # and one covering an id the source count never included is a structure
            # this check forgot about.
            keep = {i for i, sz in enumerate(sizes, 1) if sz >= MIN_VOX}
            print(f"         {'vox':>8}  source ids covered (voxels of each)")
            for i in sorted(keep, key=lambda k: -sizes[k - 1]):
                m = lab_c == i
                cov = {}
                for v in np.unique(arr[m]):
                    if v > 0:
                        cov[int(v)] = int((arr[m] == v).sum())
                top = ", ".join(f"{k}:{n}" for k, n in
                                sorted(cov.items(), key=lambda kv: -kv[1])[:4])
                print(f"         {int(sizes[i - 1]):8d}  {top}")

    print(f"\n  components match the source vertebra count in {ok}/{len(files)} cases")
    print("  a LOW count means two vertebrae merged -- the silent off-by-one this class")
    print("  exists to prevent. A HIGH count means a disc cut a vertebra in two.")
    return 0 if ok == len(files) else 2


if __name__ == "__main__":
    sys.exit(main())
