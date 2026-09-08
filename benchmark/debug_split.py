#!/usr/bin/env python3
"""Diagnose split_tall_component on a ground-truth record with L6 renamed L5: for the
merged L5 component, print the height, the case's body height, and per erosion radius the
seeds, their sizes and z-centroids, and what the guard decides.

    python benchmark/debug_split.py --labels ~/data/CTSpinoPelvic1K/labels --case 0376
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "tools"))
import convert_hf_to_nnunet as cvt  # noqa: E402
import decode_sequence as ds  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--case", required=True)
    a = ap.parse_args()
    names = cvt.LABEL_NAMES_ONESHOT
    lut = cvt._build_verse_remap_lut(cvt.LABEL_REMAP_ONESHOT_FROM_VERSE)
    img = nib.load(str(Path(a.labels) / f"{a.case}_label.nii.gz"))
    lab = np.asanyarray(img.dataobj).astype(np.int64)
    seg = lut[np.clip(lab, 0, 255)].astype(np.int16)
    seg[seg == names["ignore"]] = 0
    zooms = np.asarray(img.header.get_zooms()[:3], float)
    axis, sign = ds.sup_axis(img)
    print("axcodes", nib.aff2axcodes(img.affine), "zooms", zooms, "sup axis", axis, sign)
    l5, l6 = names["L5"], names["L6"]
    for nm, vid in (("L5", l5), ("L6", l6)):
        m = seg == vid
        if m.any():
            lo, hi = ds._height_vox(m, axis)
            print(f"{nm}: {int(m.sum())} vox, height {(hi-lo+1)*zooms[axis]:.1f} mm, z-centroid {np.argwhere(m)[:, axis].mean()*zooms[axis]*sign:.1f}")
    seg[seg == l6] = l5                                    # the collapse
    m_all = seg == l5
    cc, n = ndimage.label(m_all, structure=np.ones((3, 3, 3)))
    sizes = ndimage.sum(m_all, cc, range(1, n + 1))
    print(f"merged L5: {n} components, sizes {sorted((int(s) for s in sizes), reverse=True)[:6]}")
    # typical body height as decode_case computes it
    vert_ids = [names[k] for k in ds.THORACIC + ds.LUMBAR if k in names]
    heights = []
    for vid in vert_ids:
        mm = seg == vid
        if mm.sum() < ds.MIN_VOX:
            continue
        c2, n2 = ndimage.label(mm, structure=np.ones((3, 3, 3)))
        for k in range(1, n2 + 1):
            p = c2 == k
            if p.sum() >= ds.MIN_VOX:
                lo, hi = ds._height_vox(p, axis); heights.append((hi - lo + 1) * zooms[axis])
    body_mm = float(np.median(heights)) if len(heights) >= 3 else 35.0
    print("component heights", [round(h, 1) for h in sorted(heights)], "body_mm", round(body_mm, 1))
    big = cc == (int(np.argmax(sizes)) + 1)
    lo, hi = ds._height_vox(big, axis)
    h = (hi - lo + 1) * zooms[axis]
    print(f"largest merged component: height {h:.1f} mm = {h/body_mm:.2f} bodies (TALL_RATIO {ds.TALL_RATIO})")
    vox = float(min(zooms))
    for r_mm in (2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
        it = max(1, int(round(r_mm / vox)))
        er = ndimage.binary_erosion(big, structure=np.ones((3, 3, 3)), iterations=it)
        c3, n3 = ndimage.label(er, structure=np.ones((3, 3, 3)))
        s3 = ndimage.sum(er, c3, range(1, n3 + 1)) if n3 else []
        seeds = [(int(s), round(float(np.argwhere(c3 == k + 1)[:, axis].mean()) * zooms[axis] * sign, 1))
                 for k, s in enumerate(s3) if s >= ds.MIN_VOX // 3]
        seeds.sort(key=lambda t: -t[1])
        gaps = [round(a_ - b_, 1) for (_, a_), (_, b_) in zip(seeds, seeds[1:])]
        print(f"r={r_mm} mm ({it} it): {n3} comps, seeds (vox, z) {seeds[:6]}, gaps {gaps}, need >= {0.5*body_mm:.1f}")
    pieces = ds.split_tall_component(big, axis, zooms, body_mm)
    print("split_tall_component ->", len(pieces), "pieces", [int(p.sum()) for p in pieces])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
