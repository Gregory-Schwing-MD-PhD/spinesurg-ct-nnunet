"""Fill in the bones VerSe does not label, with TotalSegmentator, and say which are which.

    # one case, to look at
    python tools/versefusion_totalseg.py --src ~/VerSeFusion/data/unified \
        --dst ~/vf_813 --case verse504
    # everything, as a sharded array job
    sbatch --array=0-15 -D ~/spinesurg-ct-nnunet slurm/versefusion_totalseg.sh

WHAT VERSE LABELS, AND WHAT IT DOES NOT. VerSe labels vertebrae. Nothing else -- and that
includes the SACRUM, which is worth stating because the VerSe convention reserves id 26 for
it and not one of the 374 cases uses it. Dataset813 needs 32 further classes: 24 ribs, two
hips, the femora, the sacrum, the lumbar ribs and the disc space.

WHO WINS WHERE THEY DISAGREE. VerSe is expert annotation and TotalSegmentator is a model,
so the vertebrae are painted LAST and overwrite TS wherever the two differ. That ordering
is not cosmetic: on a lumbarization case TS is quite likely to call the sixth lumbar body a
sacrum, which is the exact voxel this dataset exists to get right.

BONE THAT THE SCHEME CANNOT NAME. Dataset813 has no class for the clavicle, scapula,
humerus, sternum or costal cartilage, and VerSe's C1-T9 have none either. All of it is real
bone, so calling it background would teach a falsehood. It is marked ignore -- but now
because a segmenter NAMED it, not because it was dense. The previous rule ("at least 150
HU, so it might be bone") is kept only as the last resort for whatever TS missed, and the
report says how much of the volume each rule claimed, so the guess never hides.

WHAT TOTALSEGMENTATOR CANNOT DO. Its total task has 117 classes and no radius, ulna, carpus
or phalanges. Forearms and hands in the field of view fall through to the HU rule.

THE RAW TS OUTPUT IS KEPT. Every class TS produced is written beside the label, including
the ones this scheme discards. Adding a clavicle class later is then a relabel of a file
that already exists, not a second pass over 374 CTs on a GPU.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_versefusion import (  # noqa: E402
    IGNORE, NAMES_813, TARGET_AXCODES, VERSE_TO_813, VERSE_UNNAMEABLE, reorient_pair,
)

# TotalSegmentator 'total' name -> Dataset813 id. Verified against the dataset.json of
# Dataset813_SpineSurgLSTVFullRibs rather than derived from the trainer's name list.
TS_TO_813: Dict[str, int] = {
    "sacrum": 11,
    "hip_left": 40, "hip_right": 41,
    # Dataset813 carries ONE femur class, not one per side.
    "femur_left": 42, "femur_right": 42,
    **{f"rib_left_{k}": 11 + k for k in range(1, 12)},     # rib_left_1..11 -> 12..22
    "rib_left_12": 23,
    **{f"rib_right_{k}": 24 + k for k in range(1, 12)},    # rib_right_1..11 -> 25..35
    "rib_right_12": 36,
}
# Bone TS can name that this scheme cannot carry. Ignored, never background.
TS_UNNAMEABLE = ["clavicula_left", "clavicula_right", "scapula_left", "scapula_right",
                 "humerus_left", "humerus_right", "sternum", "costal_cartilages"]
# Deliberately NOT requested: TS's own vertebrae. VerSe's are better and would be
# overwritten anyway, and every ROI costs inference time.
TS_ROIS = sorted(set(TS_TO_813) | set(TS_UNNAMEABLE))

# rib13 (24, 37) has no TotalSegmentator equivalent: TS numbers ribs 1-12. The 18 T13 cases
# in this corpus should each carry a thirteenth rib, and TS will either miss it or fold it
# into rib 12. Those cases are flagged in the report for human review rather than guessed.
RIB13 = {24: "rib13_left", 37: "rib13_right"}


def run_totalseg(ct_img, device: str = "gpu"):
    """One TS pass, returning {class name: boolean mask} in the CT's array space."""
    from totalsegmentator.map_to_binary import class_map
    from totalsegmentator.python_api import totalsegmentator

    name_of = class_map["total"]
    valid = [r for r in TS_ROIS if r in name_of.values()]
    missing = sorted(set(TS_ROIS) - set(valid))
    if missing:
        print(f"    not in this TS build, skipped: {missing}")
    pred = totalsegmentator(input=ct_img, output=None, task="total", ml=True,
                            device=device, roi_subset=valid, verbose=False)
    arr = np.asanyarray(pred.dataobj).astype(np.int16)
    val_of = {v: k for k, v in name_of.items()}
    return {n: (arr == val_of[n]) for n in valid if val_of[n] in arr}, arr


def compose(verse: np.ndarray, ts: Dict[str, np.ndarray], ct: np.ndarray,
            bone_hu: float) -> tuple[np.ndarray, dict]:
    """Build the Dataset813 label. Order is the whole argument, so it is explicit.

    1. TS classes the scheme carries.
    2. TS bone the scheme cannot name  -> ignore.
    3. VerSe vertebrae                 -> LAST, so expert labels outrank the model.
    4. VerSe vertebrae with no class    -> ignore.
    5. Whatever is still bone-dense     -> ignore, the fallback for what TS missed.
    """
    verse = np.rint(verse).astype(np.int16)
    out = np.zeros(verse.shape, np.uint8)
    stats: Dict[str, int] = {}

    for name, m in ts.items():
        if name in TS_TO_813 and m.any():
            out[m] = TS_TO_813[name]
            stats[f"ts/{name}"] = int(m.sum())
    for name in TS_UNNAMEABLE:
        m = ts.get(name)
        if m is not None and m.any():
            out[m] = IGNORE
            stats[f"ts_ignored/{name}"] = int(m.sum())

    overwritten = 0
    for v, d in VERSE_TO_813.items():
        m = verse == v
        if m.any():
            overwritten += int((out[m] != 0).sum())
            out[m] = d
            stats[f"verse/{NAMES_813[d]}"] = int(m.sum())
    stats["verse_overwrote_ts_voxels"] = overwritten

    un = np.isin(verse, list(VERSE_UNNAMEABLE))
    if un.any():
        out[un] = IGNORE
        stats["ignore/unnameable_vertebrae"] = int(un.sum())

    maybe = (out == 0) & (ct >= bone_hu)
    out[maybe] = IGNORE
    stats["ignore/dense_but_unnamed"] = int(maybe.sum())
    stats["background"] = int((out == 0).sum())
    return out, stats


def process(case_dir: Path, dst: Path, bone_hu: float, device: str,
            keep_ts: bool) -> dict:
    cid = case_dir.name.replace("scan-", "")
    rec: dict = {"case": cid}
    try:
        ct_i = nib.load(str(case_dir / f"scan-{cid}_ct.nii.gz"))
        mk_i = nib.load(str(case_dir / f"scan-{cid}_msk.nii.gz"))
        rec["src_axcodes"] = "".join(nib.aff2axcodes(ct_i.affine))
        ct_i, mk_i = reorient_pair(ct_i, mk_i)

        ct = ct_i.get_fdata(dtype=np.float32)
        verse = mk_i.get_fdata(dtype=np.float32)
        ts, ts_raw = run_totalseg(ct_i, device)
        lab, stats = compose(verse, ts, ct, bone_hu)

        rec.update(ok=True, shape=list(lab.shape), stats=stats,
                   ts_classes_found=sorted(ts),
                   has_L6=bool((lab == 10).any()), has_T13=bool((lab == 4).any()))
        # A thirteenth rib cannot come from TS. Say so per case instead of leaving a gap.
        if rec["has_T13"]:
            rec["needs_rib13_review"] = True

        for d in ("imagesTr", "labelsTr", "totalseg_raw"):
            (dst / d).mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(ct.astype(np.int16), ct_i.affine, ct_i.header.copy()),
                 str(dst / "imagesTr" / f"VERSE_{cid}_0000.nii.gz"))
        li = nib.Nifti1Image(lab, ct_i.affine, ct_i.header.copy())
        li.set_data_dtype(np.uint8)
        nib.save(li, str(dst / "labelsTr" / f"VERSE_{cid}.nii.gz"))
        if keep_ts:
            ri = nib.Nifti1Image(ts_raw.astype(np.uint8), ct_i.affine, ct_i.header.copy())
            ri.set_data_dtype(np.uint8)
            nib.save(ri, str(dst / "totalseg_raw" / f"VERSE_{cid}_ts.nii.gz"))
    except Exception as exc:                                        # noqa: BLE001
        rec.update(ok=False, error=f"{type(exc).__name__}: {exc}")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--case", default="", help="one case id, for inspection")
    ap.add_argument("--bone_hu", type=float, default=150.0)
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--no_keep_ts", action="store_true",
                    help="do not keep the raw TotalSegmentator output")
    ap.add_argument("--resume", action="store_true",
                    help="skip cases whose label already exists")
    a = ap.parse_args()

    dst = Path(a.dst)
    cases = sorted(p for p in Path(a.src).glob("scan-*") if p.is_dir())
    if a.case:
        cases = [p for p in cases if p.name == f"scan-{a.case}"]
    else:
        cases = [c for i, c in enumerate(cases) if i % a.n_shards == a.shard]
    if a.resume:
        cases = [c for c in cases
                 if not (dst / "labelsTr" /
                         f"VERSE_{c.name.replace('scan-', '')}.nii.gz").exists()]
    print(f"shard {a.shard}/{a.n_shards}: {len(cases)} cases -> {dst}")

    recs = []
    for i, c in enumerate(cases, 1):
        r = process(c, dst, a.bone_hu, a.device, not a.no_keep_ts)
        recs.append(r)
        flag = "ok " if r["ok"] else "FAIL"
        extra = ""
        if r["ok"]:
            extra = (f"  L6={'y' if r['has_L6'] else '-'}"
                     f" T13={'y' if r['has_T13'] else '-'}"
                     f" ts={len(r['ts_classes_found'])}"
                     f" verse_over_ts={r['stats'].get('verse_overwrote_ts_voxels', 0)}")
        else:
            extra = "  " + r["error"]
        print(f"  [{i}/{len(cases)}] {flag} {r['case']}{extra}", flush=True)

    rep = dst / f"report_shard{a.shard}.json"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(recs, indent=1))
    ok = [r for r in recs if r["ok"]]
    print(f"\n  ok {len(ok)} / {len(recs)}")
    if ok:
        need = [r["case"] for r in ok if r.get("needs_rib13_review")]
        print(f"  T13 cases needing a rib13 by hand: {len(need)}  {need[:10]}")
        dense = np.mean([r["stats"].get("ignore/dense_but_unnamed", 0)
                         / max(1, np.prod(r["shape"])) for r in ok])
        print(f"  mean volume left to the HU fallback: {100*dense:.2f}%"
              f"   (lower is better: TS named the rest)")
    return 0 if len(ok) == len(recs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
