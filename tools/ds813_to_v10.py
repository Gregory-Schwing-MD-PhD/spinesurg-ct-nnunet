"""Dataset813 ids -> the v10 release ids, so the paper's own morphometrics can run on them.

    python tools/ds813_to_v10.py --src ~/vf_813/labelsTr --dst ~/vf_v10 \
        --cases verse509,verse512

WHY BRIDGE INSTEAD OF REIMPLEMENTING. CTSpinoPelvic1K's
scripts/extract_transition_morphometrics.py already measures the thoracolumbar junction the
way the manuscript reports it: _nearest_vert assigns a rib to a vertebra when their gap is
under ANCHOR_MM, which IS the articulation test, and rib12_11_ratio with its 0.33 cut is
the validated stump criterion. That code has been checked against 802 records and its
failure modes are documented. Writing a second implementation to answer the same question
on a different corpus would be a new thing to be wrong about.

It expects v10 ids, so the bridge is a lookup table and nothing else.

WHAT DOES NOT SURVIVE THE TRIP. Dataset813 carries ONE femur class where v10 has two; both
map to the left femur id and the sidedness is lost, which nothing downstream of here uses.
The disc space (43) has no v10 id and is dropped. `ignore` becomes background: the
morphometrics only ever address specific ids, so an ignore region behaves the same as
unlabelled either way -- but that makes this output fit for MEASUREMENT and not for
training, since it silently asserts that unnamed bone is background.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

# v10: T_n = 7 + n, L_n = 19 + n, T13 = 28, sacrum 26, hips 30/31, femurs 32/33,
# ribs left 34..46 (33 + n), right 47..59 (46 + n), lumbar ribs 60/61.
DS813_TO_V10: dict[int, int] = {
    1: 17, 2: 18, 3: 19,           # T10, T11, T12
    4: 28,                          # T13
    5: 20, 6: 21, 7: 22, 8: 23, 9: 24, 10: 25,     # L1..L6
    11: 26,                         # sacrum
    **{11 + k: 33 + k for k in range(1, 12)},      # rib_left_1..11  -> 34..44
    23: 45, 24: 46,                 # rib12_left, rib13_left
    **{24 + k: 46 + k for k in range(1, 12)},      # rib_right_1..11 -> 47..57
    36: 58, 37: 59,                 # rib12_right, rib13_right
    38: 60, 39: 61,                 # lumbar ribs
    40: 30, 41: 31,                 # hips
    42: 32,                         # one femur class; v10's 33 is unreachable from here
    43: 0,                          # disc space: no v10 id
    44: 0,                          # ignore -> background, see the docstring
}


def convert(src: Path, dst: Path) -> dict:
    img = nib.load(str(src))
    a = np.asanyarray(img.dataobj).astype(np.uint8)
    lut = np.zeros(256, np.uint8)
    for k, v in DS813_TO_V10.items():
        lut[k] = v
    out = lut[a]
    unmapped = sorted(set(np.unique(a).tolist()) - set(DS813_TO_V10) - {0})
    dst.parent.mkdir(parents=True, exist_ok=True)
    oi = nib.Nifti1Image(out, img.affine, img.header.copy())
    oi.set_data_dtype(np.uint8)
    nib.save(oi, str(dst))
    return {"unmapped": unmapped,
            "ids_out": sorted(int(v) for v in np.unique(out) if v)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--cases", default="", help="comma-separated ids; default all")
    a = ap.parse_args()

    src, dst = Path(a.src), Path(a.dst)
    want = {c.strip() for c in a.cases.split(",") if c.strip()}
    files = sorted(src.glob("VERSE_*.nii.gz"))
    if want:
        files = [f for f in files
                 if f.name.replace("VERSE_", "").replace(".nii.gz", "") in want]
    print(f"{len(files)} labels {src} -> {dst}")
    bad = []
    for f in files:
        cid = f.name.replace("VERSE_", "").replace(".nii.gz", "")
        r = convert(f, dst / f"{cid}_label.nii.gz")
        if r["unmapped"]:
            bad.append((cid, r["unmapped"]))
        print(f"  {cid}: {len(r['ids_out'])} ids out")
    if bad:
        print(f"\n  UNMAPPED ids seen (these were dropped): {bad[:5]}")
        return 1
    print("\n  every id had a v10 home")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
