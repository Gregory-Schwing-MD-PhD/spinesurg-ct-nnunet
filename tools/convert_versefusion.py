"""Bring VerSeFusion into the Dataset813 label space, without asserting anything false.

    python tools/convert_versefusion.py --src ~/VerSeFusion/data/unified --dst <out> --dry
    python tools/convert_versefusion.py --src ... --dst ... --workers 8

WHAT IS BEING JOINED. VerSeFusion is 374 VerSe 2019+2020 scans, curated, carrying 44 cases
with an L6 and 18 with a T13 -- against 15 and ZERO in CTSpinoPelvic1K. It labels vertebrae
only: no ribs, no pelvis, no femora, which is 32 of Dataset813's 43 classes.

THE THREE THINGS THAT MAKE THIS MORE THAN A LOOKUP TABLE

1. FIVE ORIENTATIONS. The corpus is stored LAS (216), PSR (67), PIR (59), LPS (31) and
   LSP (1). Dataset813 is PIR throughout. nnU-Net does not reorient on read, so mixing
   them teaches the network that its patch axes mean different anatomical directions in
   different cases. Every pair is reoriented to PIR here.

   This repo's standing rule is "canonicalise to analyse, never on a write path", because
   as_closest_canonical + nib.save silently transposes a label away from its CT. That rule
   is about reorienting ONE of a pair. Here the transform is derived once per case and
   applied to BOTH, and the result is verified -- axcodes equal, affines equal, label
   histogram unchanged -- before anything is written. Reorientation is a transpose and a
   flip, so it is exact: no interpolation, no resampling, no label invented or lost.

2. WHAT TO DO WITH BONE THAT HAS NO NAME. Ribs, hips, femora and the sacrum below the
   promontory are unlabelled here, and both obvious choices are wrong. Calling them
   background teaches the network that a rib is background, which it will believe, and
   Dataset813 has 26 rib classes. Calling the whole volume ignore throws away the
   background supervision that makes a class boundary mean anything.

   So the split is made on the CT, not on the mask: a voxel that is not a labelled vertebra
   is IGNORED if it could be bone (>= --bone_hu) and BACKGROUND otherwise. Air, fat, muscle
   and bowel are safely background because they are not any class in the scheme. A rib is
   never called background and never called a rib. Nothing is asserted that is not known.

3. VERTEBRAE THE SCHEME CANNOT NAME. VerSe labels C1-T9 (1-16) and the coccyx (27), and
   Dataset813 starts at T10. Those are real bone with no class, so they are ignored for
   the same reason -- not background.

WHAT THIS DELIBERATELY DOES NOT DO. It does not pseudo-label the ribs and pelvis with
TotalSegmentator. That would give these cases full supervision and is the obvious next
step, but it puts a model's guesses into the ground truth of a dataset whose entire claim
is careful labelling. Ignore first, measure what it buys, then decide.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, Tuple

import nibabel as nib
import numpy as np

TARGET_AXCODES = ("P", "I", "R")
IGNORE = 44                                 # Dataset813's ignore label

# VerSe id -> Dataset813 id. Only the levels Dataset813 names; everything else is handled
# by the rules below rather than by a silent default, because a silent default is how this
# project has produced wrong-but-plausible labels before.
VERSE_TO_813: Dict[int, int] = {
    17: 1,    # T10
    18: 2,    # T11
    19: 3,    # T12
    28: 4,    # T13   <- Dataset813 has this class and zero examples of it
    20: 5,    # L1
    21: 6,    # L2
    22: 7,    # L3
    23: 8,    # L4
    24: 9,    # L5
    25: 10,   # L6    <- the class this whole exercise is for
    26: 11,   # sacrum
}
# Named bone with no class in Dataset813: C1-C7, T1-T9, coccyx. Ignored, never background.
VERSE_UNNAMEABLE = set(range(1, 17)) | {27}

NAMES_813 = {1: "T10", 2: "T11", 3: "T12", 4: "T13", 5: "L1", 6: "L2", 7: "L3",
             8: "L4", 9: "L5", 10: "L6", 11: "sacrum", IGNORE: "ignore"}


def corner_disagreement_mm(aff_a, aff_b, shape) -> float:
    """How far apart do two affines put the same voxel, in millimetres?

    Comparing affine matrices element-wise with an absolute tolerance is the wrong test and
    it rejected six good cases. The masks here were written by a different tool from the
    CTs (sform_code 2 against 1), so the matrices differ by float32 round-off -- 6e-6 on
    one case. `as_reoriented` composes the affine with inv_ornt_aff(ornt, shape), which
    multiplies that round-off by the array dimension, so on a 1119-slice volume 3.7e-5
    becomes 0.041. Nothing moved; the arithmetic simply has more digits.

    What actually matters is whether the two put the same voxel in the same place, so this
    measures exactly that, at the corners where any rotation difference is largest.
    """
    i, j, k = (np.array([0, s - 1]) for s in shape)
    corners = np.array([[x, y, z, 1.0] for x in i for y in j for z in k]).T
    return float(np.linalg.norm((aff_a @ corners - aff_b @ corners)[:3], axis=0).max())


def reorient_pair(ct_img, msk_img):
    """Put both images in TARGET_AXCODES using ONE transform, and prove it worked.

    The transform comes from the CT. Applying the mask's own transform would be equivalent
    here -- every pair in this corpus has identical affines, checked -- but deriving it
    once and applying it twice is what makes that equivalence a property of the code rather
    than of the data.
    """
    start = nib.orientations.io_orientation(ct_img.affine)
    want = nib.orientations.axcodes2ornt(TARGET_AXCODES)
    xfm = nib.orientations.ornt_transform(start, want)
    ct2, msk2 = ct_img.as_reoriented(xfm), msk_img.as_reoriented(xfm)

    got = nib.aff2axcodes(ct2.affine)
    if got != TARGET_AXCODES:
        raise ValueError(f"reorientation gave {got}, wanted {TARGET_AXCODES}")
    if nib.aff2axcodes(msk2.affine) != TARGET_AXCODES:
        raise ValueError("mask did not land in the target orientation")
    if ct2.shape != msk2.shape:
        raise ValueError(f"shapes diverged: {ct2.shape} vs {msk2.shape}")
    # A fraction of a voxel, not a fixed number of millimetres: the tolerance has to mean
    # the same thing on a 0.27 mm scan as on a 3.0 mm one.
    vox = float(min(ct2.header.get_zooms()[:3]))
    off = corner_disagreement_mm(ct2.affine, msk2.affine, ct2.shape)
    if off > 0.25 * vox:
        raise ValueError(f"CT and mask disagree by {off:.4f} mm "
                         f"({off / vox:.2f} voxels) after reorientation")
    return ct2, msk2


def map_labels(verse: np.ndarray, ct: np.ndarray, bone_hu: float) -> Tuple[np.ndarray, dict]:
    """VerSe ids -> Dataset813 ids, with unnameable bone ignored rather than zeroed."""
    verse = np.rint(verse).astype(np.int16)        # stored float64 in 359 of 374 cases
    out = np.zeros(verse.shape, np.uint8)

    stats = collections.Counter()
    for v, d in VERSE_TO_813.items():
        m = verse == v
        n = int(m.sum())
        if n:
            out[m] = d
            stats[NAMES_813[d]] = n

    # named bone with no class in this scheme
    unnameable = np.isin(verse, list(VERSE_UNNAMEABLE))
    out[unnameable] = IGNORE
    stats["ignore_unnameable_vertebrae"] = int(unnameable.sum())

    # UNLABELLED BONE. Not background -- ribs, hips, femora and the lower sacrum live here
    # and Dataset813 has classes for all of them. Not a class either, because which one is
    # unknown. Soft tissue and air stay background: they are not any class in the scheme.
    unlabelled = (out == 0)
    maybe_bone = unlabelled & (ct >= bone_hu)
    out[maybe_bone] = IGNORE
    stats["ignore_unlabelled_bone"] = int(maybe_bone.sum())
    stats["background"] = int((out == 0).sum())

    leftover = set(np.unique(verse[out == 0]).tolist()) - {0}
    if leftover:
        raise ValueError(f"VerSe ids fell through to background: {sorted(leftover)}")
    return out, dict(stats)


def convert_one(case_dir: Path, dst: Path, bone_hu: float, dry: bool) -> dict:
    cid = case_dir.name.replace("scan-", "")
    ct_p = case_dir / f"scan-{cid}_ct.nii.gz"
    msk_p = case_dir / f"scan-{cid}_msk.nii.gz"
    rec: dict = {"case": cid}
    try:
        ct_i, msk_i = nib.load(str(ct_p)), nib.load(str(msk_p))
        rec["src_axcodes"] = "".join(nib.aff2axcodes(ct_i.affine))
        ct_i, msk_i = reorient_pair(ct_i, msk_i)

        # get_fdata(dtype=float32) rather than asanyarray(dataobj).astype(float32): the
        # latter materialises a float64 array first, which is 8 bytes a voxel on volumes
        # that reach 960x960x110. Twelve of those in a process pool is what killed it.
        ct = ct_i.get_fdata(dtype=np.float32)
        verse = msk_i.get_fdata(dtype=np.float32)
        lab, stats = map_labels(verse, ct, bone_hu)

        rec["shape"] = list(lab.shape)
        rec["spacing"] = [round(float(z), 3) for z in ct_i.header.get_zooms()[:3]]
        rec["stats"] = stats
        rec["has_L6"] = stats.get("L6", 0) > 0
        rec["has_T13"] = stats.get("T13", 0) > 0
        rec["ok"] = True

        if not dry:
            dst.mkdir(parents=True, exist_ok=True)
            (dst / "imagesTr").mkdir(exist_ok=True)
            (dst / "labelsTr").mkdir(exist_ok=True)
            # int16 CT and uint8 label, matching what Dataset813 already stores. The affine
            # written is the REORIENTED one for both, so they stay in lockstep.
            nib.save(nib.Nifti1Image(ct.astype(np.int16), ct_i.affine, ct_i.header.copy()),
                     str(dst / "imagesTr" / f"VERSE_{cid}_0000.nii.gz"))
            lab_img = nib.Nifti1Image(lab, ct_i.affine, ct_i.header.copy())
            lab_img.set_data_dtype(np.uint8)
            nib.save(lab_img, str(dst / "labelsTr" / f"VERSE_{cid}.nii.gz"))
    except Exception as exc:                                        # noqa: BLE001
        rec["ok"] = False
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="VerSeFusion data/unified")
    ap.add_argument("--dst", default="", help="output root; required unless --dry")
    ap.add_argument("--bone_hu", type=float, default=150.0,
                    help="a voxel at or above this is possibly bone, so ignored rather "
                         "than called background")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry", action="store_true", help="report, write nothing")
    ap.add_argument("--report", default="", help="write the per-case record here as JSON")
    a = ap.parse_args()

    if not a.dry and not a.dst:
        print("--dst is required unless --dry", file=sys.stderr)
        return 2
    cases = sorted(p for p in Path(a.src).glob("scan-*") if p.is_dir())
    if a.limit:
        cases = cases[:a.limit]
    print(f"{len(cases)} cases under {a.src}"
          f"{'  (dry run, nothing written)' if a.dry else ''}")

    dst = Path(a.dst) if a.dst else Path(".")
    recs = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(convert_one, c, dst, a.bone_hu, a.dry): c for c in cases}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            recs.append(r)
            if not r["ok"]:
                print(f"  FAIL {r['case']}: {r['error']}")
            elif i % 50 == 0:
                print(f"  ... {i}/{len(cases)}")

    ok = [r for r in recs if r["ok"]]
    bad = [r for r in recs if not r["ok"]]
    print(f"\n  converted     : {len(ok)}")
    print(f"  failed        : {len(bad)}")
    print(f"  with L6       : {sum(r['has_L6'] for r in ok)}   (repo audit says 44)")
    print(f"  with T13      : {sum(r['has_T13'] for r in ok)}   (repo audit says 18)")
    src_ax = collections.Counter(r["src_axcodes"] for r in ok)
    print(f"  source orientations converted: {dict(src_ax)}")
    if ok:
        ig = np.mean([r["stats"].get("ignore_unlabelled_bone", 0)
                      / max(1, np.prod(r["shape"])) for r in ok])
        bg = np.mean([r["stats"].get("background", 0) / max(1, np.prod(r["shape"]))
                      for r in ok])
        print(f"  mean volume ignored as maybe-bone: {100*ig:.2f}%")
        print(f"  mean volume kept as background   : {100*bg:.2f}%")
    if a.report:
        Path(a.report).write_text(json.dumps(recs, indent=1))
        print(f"  wrote {a.report}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
