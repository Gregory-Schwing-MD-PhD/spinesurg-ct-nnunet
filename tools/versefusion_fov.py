"""How much unlabelled volume lies BELOW the lowest labelled vertebra?

This is the number that sizes the annotation job. The audit CSV says has_sacrum is False
for all 374, and a 40-case label scan agrees -- VerSe labels vertebrae and not the sacrum.
But "not labelled" is not "not in the scan": the pelvis may well be in the field of view
and simply unannotated, which is precisely what annotating would fix.

So measure the scan, not the label. For every case, find the inferior-most labelled voxel
and ask how many millimetres of image lie below it. A case with 100 mm below its last
vertebra has a sacrum and probably hips to annotate; a case with 5 mm has nothing there.

Reported for the 44 L6 cases separately, because those are the ones worth annotating first.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


def inferior_gap_mm(msk_path: Path) -> tuple[float, float, int]:
    """(mm below the lowest labelled voxel, mm of labelled span, lowest label id)."""
    img = nib.load(str(msk_path))
    a = np.rint(img.get_fdata(dtype=np.float32)).astype(np.int16)
    # which axis is craniocaudal, and which way is down, from the affine
    ax = int(np.argmax(np.abs(img.affine[2, :3])))          # axis most aligned with S/I
    sign = float(np.sign(img.affine[2, ax]))                 # +1 if index increases superiorly
    zoom = float(img.header.get_zooms()[ax])
    idx = np.nonzero(a.any(axis=tuple(i for i in range(3) if i != ax)))[0]
    if not len(idx):
        return (float("nan"), 0.0, 0)
    lo, hi = int(idx.min()), int(idx.max())
    n = a.shape[ax]
    below = (lo if sign > 0 else n - 1 - hi)                 # slices between label and edge
    # the lowest labelled level, by taking the slice at the inferior end
    sl = [slice(None)] * 3
    sl[ax] = lo if sign > 0 else hi
    vals = [v for v in np.unique(a[tuple(sl)]) if v]
    return below * zoom, (hi - lo + 1) * zoom, int(max(vals) if vals else 0)


def main(root: str, out_csv: str = "") -> int:
    cases = sorted(p for p in Path(root).glob("scan-*") if p.is_dir())
    rows = []
    for i, d in enumerate(cases, 1):
        cid = d.name.replace("scan-", "")
        m = d / f"scan-{cid}_msk.nii.gz"
        if not m.exists():
            continue
        try:
            below, span, low = inferior_gap_mm(m)
            u = np.unique(np.rint(nib.load(str(m)).get_fdata(dtype=np.float32)))
            rows.append({"case": cid, "mm_below_last_vertebra": round(below, 1),
                         "labelled_span_mm": round(span, 1), "lowest_label": low,
                         "has_l6": int(25 in u), "has_t13": int(28 in u),
                         "has_sacrum_label": int(26 in u)})
        except Exception as exc:                                    # noqa: BLE001
            print(f"  {cid}: {type(exc).__name__}: {exc}")
        if i % 50 == 0:
            print(f"  ... {i}/{len(cases)}", flush=True)

    def summarise(name, sub):
        if not sub:
            print(f"\n{name}: none"); return
        v = np.array([r["mm_below_last_vertebra"] for r in sub], float)
        v = v[~np.isnan(v)]
        print(f"\n{name}: {len(sub)} cases")
        print(f"   mm below the lowest labelled vertebra: "
              f"median {np.median(v):.0f}, p25 {np.percentile(v,25):.0f}, "
              f"p75 {np.percentile(v,75):.0f}, max {v.max():.0f}")
        for thr in (40, 80, 120):
            print(f"   at least {thr} mm below (room for sacrum/pelvis): "
                  f"{int((v >= thr).sum())} of {len(v)}")

    summarise("ALL", rows)
    summarise("L6 CASES", [r for r in rows if r["has_l6"]])
    summarise("T13 CASES", [r for r in rows if r["has_t13"]])
    print(f"\ncases with a SACRUM label anywhere: "
          f"{sum(r['has_sacrum_label'] for r in rows)} of {len(rows)}")
    if out_csv:
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else ""))
