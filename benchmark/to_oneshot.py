#!/usr/bin/env python3
"""benchmark/to_oneshot.py — put any system's prediction into the one-shot scoring space.

Every competitor predicts in its own vocabulary and often on its own grid. This resamples
the prediction onto the reference label grid (nearest neighbour, through the affines) and
maps its ids to the Dataset810 one-shot ids by NAME, so the same two scorers
(tools/eval_oneshot.py, tools/decode_sequence.py --onehot) run on everything.

Two maps:
  --class_map   json {native_id: native_name}            (e.g. TotalSegmentator's class_map)
  --name_map    json {native_name: oneshot_name}         (benchmark/mappings/<system>.json)
Native names absent from the name map go to background; oneshot names must exist in the
dataset.json labels. A native id may also be given directly as a string key in the name map
("25": "sacrum"), which is how the VerSe-id systems (SPINEPS/VERIDAH) are mapped.

    python benchmark/to_oneshot.py --cases benchmark/cases.csv \
        --pred_dir ~/bench/totalsegmentator/native --out_dir ~/bench/totalsegmentator/oneshot \
        --dataset_json nnunet/raw/Dataset810_SpineSurgLSTVOneShot/dataset.json \
        --class_map ~/bench/mappings/totalsegmentator_class_map.json \
        --name_map benchmark/mappings/totalsegmentator.json [--pattern "{case}.nii.gz"]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to


def build_lut(class_map: dict | None, name_map: dict, oneshot: dict) -> np.ndarray:
    """native id -> oneshot id, as a lookup table over native ids 0..max."""
    native_name = {int(k): v for k, v in (class_map or {}).items()}
    pairs = {}
    for key, target in name_map.items():
        if target not in oneshot:
            raise KeyError(f"{target!r} is not a one-shot label")
        if key.lstrip("-").isdigit():
            pairs[int(key)] = oneshot[target]
        else:
            ids = [i for i, n in native_name.items() if n == key]
            if not ids and class_map is not None:
                print(f"  warning: native name {key!r} not in class map")
            for i in ids:
                pairs[i] = oneshot[target]
    n = max(list(pairs) + list(native_name) + [0]) + 1
    lut = np.zeros(n, dtype=np.int16)
    for i, o in pairs.items():
        lut[i] = o
    return lut


def convert(pred_path: Path, ref_path: Path, lut: np.ndarray, out_path: Path) -> dict:
    pred = nib.load(str(pred_path))
    ref = nib.load(str(ref_path))
    same_grid = pred.shape == ref.shape and np.allclose(pred.affine, ref.affine, atol=1e-3)
    if not same_grid:
        pred = resample_from_to(pred, (ref.shape, ref.affine), order=0, mode="constant", cval=0)
    arr = np.asanyarray(pred.dataobj).astype(np.int64)
    arr = np.clip(arr, 0, len(lut) - 1)
    out = lut[arr].astype(np.uint8)
    nib.save(nib.Nifti1Image(out, ref.affine, ref.header), str(out_path))
    return {"case": out_path.name.replace(".nii.gz", ""), "resampled": int(not same_grid),
            "native_ids": int(len(np.unique(arr))), "oneshot_ids": int(len(np.unique(out)))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True, type=Path)
    ap.add_argument("--pred_dir", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--dataset_json", required=True, type=Path)
    ap.add_argument("--name_map", required=True, type=Path)
    ap.add_argument("--class_map", type=Path, default=None)
    ap.add_argument("--pattern", default="{case}.nii.gz", help="prediction file name per case")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    oneshot = {k: int(v) for k, v in json.loads(a.dataset_json.read_text())["labels"].items()
               if not isinstance(v, list)}
    class_map = json.loads(a.class_map.read_text()) if a.class_map else None
    lut = build_lut(class_map, json.loads(a.name_map.read_text()), oneshot)
    a.out_dir.mkdir(parents=True, exist_ok=True)

    rows, missing = [], []
    with open(a.cases) as fh:
        for r in csv.DictReader(fh):
            src = a.pred_dir / a.pattern.format(**r)
            dst = a.out_dir / f"{r['case']}.nii.gz"
            if not src.exists():
                missing.append(r["case"]); continue
            if dst.exists() and not a.force:
                continue
            rows.append(convert(src, Path(r["label"]), lut, dst))
            print(rows[-1], flush=True)
    print(f"converted {len(rows)}; missing predictions {len(missing)}" + (f" e.g. {missing[:5]}" if missing else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
