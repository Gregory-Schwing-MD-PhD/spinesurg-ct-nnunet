#!/usr/bin/env python3
"""benchmark/spineps_to_oneshot.py — compose the SPINEPS/VERIDAH vertebra instances and
Möller's rib assignment into one prediction per case in the one-shot space, on the label
grid, so score.sh can treat "spineps" like any other system.

Inputs per case (from run_spineps_cases.py and run_spineps_ribs.py):
  <native>/<case>_seg-vert_msk.nii.gz     VerSe ids: 17-19 T10-T12, 28 T13, 20-25 L1-L6, 26 sacrum
  <ribs>/<case>_rib-inst_msk.nii.gz       ribs carry TPTBox Vertebra_Instance.RIB ids:
                                          T1..T9 40..48, T10 49, T11 50, T12 51, L1 52, T13 53
                                          (L2 and below have no RIB id in TPTBox: his code can
                                          only hand a rib to L1, never lower)
  <ribs>/<case>_rib-sem_msk.nii.gz        side: Location.Rib_Left 63, Rib_Right 64
                                          (or the sem mask his run_all_steps returns)
Absent rib files leave the ribs empty (vertebrae-only scoring, ribs count as misses).

    python benchmark/spineps_to_oneshot.py --cases benchmark/cases.csv \
        --native ~/bench/spineps/native --ribs ~/bench/spineps/ribs \
        --dataset_json nnunet/raw/Dataset810_SpineSurgLSTVOneShot/dataset.json \
        --out_dir ~/bench/spineps/oneshot
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

VERT = {17: "T10", 18: "T11", 19: "T12", 28: "T13", 20: "L1", 21: "L2", 22: "L3", 23: "L4",
        24: "L5", 25: "L6", 26: "sacrum"}
RIB_BY_VERT = {**{i: "rib" for i in range(40, 51)}, 51: "rib12", 52: "lumbar_rib", 53: "rib13"}
RIB_LEFT, RIB_RIGHT = 63, 64


def on_grid(img, ref):
    if img.shape == ref.shape and np.allclose(img.affine, ref.affine, atol=1e-3):
        return np.asanyarray(img.dataobj).astype(np.int32)
    return np.asanyarray(resample_from_to(img, (ref.shape, ref.affine), order=0).dataobj).astype(np.int32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True, type=Path)
    ap.add_argument("--native", required=True, type=Path)
    ap.add_argument("--ribs", type=Path, default=None)
    ap.add_argument("--dataset_json", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    names = {k: int(v) for k, v in json.loads(a.dataset_json.read_text())["labels"].items() if not isinstance(v, list)}
    a.out_dir.mkdir(parents=True, exist_ok=True)
    n = n_ribs = 0
    for r in csv.DictReader(open(a.cases)):
        case = r["case"]
        vp = a.native / f"{case}_seg-vert_msk.nii.gz"
        dst = a.out_dir / f"{case}.nii.gz"
        if not vp.exists() or (dst.exists() and not a.force):
            continue
        ref = nib.load(r["label"])
        vert = on_grid(nib.load(str(vp)), ref)
        out = np.zeros(ref.shape, dtype=np.uint8)
        for vid, nm in VERT.items():
            out[vert == vid] = names[nm]
        rp = a.ribs / f"{case}_rib-inst_msk.nii.gz" if a.ribs else None
        sp = a.ribs / f"{case}_rib-sem_msk.nii.gz" if a.ribs else None
        if rp and rp.exists() and sp and sp.exists():
            rib = on_grid(nib.load(str(rp)), ref)
            sem = on_grid(nib.load(str(sp)), ref)
            for rid, base in RIB_BY_VERT.items():
                m = rib == rid
                if not m.any():
                    continue
                out[m & (sem == RIB_LEFT)] = names[f"{base}_left"]
                out[m & (sem == RIB_RIGHT)] = names[f"{base}_right"]
            n_ribs += 1
        nib.save(nib.Nifti1Image(out, ref.affine, ref.header), str(dst))
        n += 1
        print(case, "ids", sorted(int(x) for x in np.unique(out)), flush=True)
    print(f"composed {n} cases ({n_ribs} with ribs) -> {a.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
