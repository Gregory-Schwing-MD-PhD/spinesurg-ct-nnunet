#!/usr/bin/env python3
"""benchmark/run_spineps_ribs.py — Möller's rib stack on top of the SPINEPS vertebra masks:

  stage 2  his binary rib nnU-Net (Zenodo 10.5281/zenodo.14850928; the flattened model dir
           already on the grid at ~/CTSpinoPelvic1K/models/moller_ribseg/ribseg_model_weights)
  stage 3  his rib-segmentation code (run_all_steps: ribs assigned to vertebra instances,
           rib length, stump-rib features), run verbatim from the clone at ~/rib-segmentation

Inputs per case: the CT, and SPINEPS' <case>_seg-vert_msk / <case>_seg-spine_msk written by
run_spineps_cases.py. Outputs to <out_dir>: <case>_ribmask.nii.gz (binary, CT grid),
<case>_rib-inst_msk.nii.gz (ribs carrying the id of the vertebra they attach to, SPINEPS
grid), rib_measurements_shard<k>.csv, rib_features/<case>.json. Resumable.

Runs in the `spineps` conda env (nnunetv2 and TPTBox come with spineps).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_SCALARS = (str, int, float, bool, np.integer, np.floating, np.bool_)


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer, np.floating, np.bool_)):
        return o.item()
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o if (isinstance(o, _SCALARS) or o is None) else None


def predict_ribs(cases: list[dict], work: Path, model_dir: Path, folds: tuple[int, ...],
                 checkpoint: str, out_dir: Path) -> None:
    """One predictor for the shard; Möller's zip is a flattened nnU-Net model dir."""
    in_dir = work / "rib_in"; in_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = work / "rib_pred"; pred_dir.mkdir(parents=True, exist_ok=True)
    todo = 0
    for r in cases:
        if (out_dir / f"{r['case']}_ribmask.nii.gz").exists():
            continue
        dst = in_dir / f"{r['case']}_0000.nii.gz"
        if not dst.exists():
            os.symlink(r["ct"], dst)
        todo += 1
    if not todo:
        return
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                                device=torch.device("cuda"), verbose=False, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(str(model_dir), use_folds=folds, checkpoint_name=checkpoint)
    print(f"rib nnU-Net on {todo} cases", flush=True)
    predictor.predict_from_files(str(in_dir), str(pred_dir), save_probabilities=False, overwrite=False)
    for r in cases:
        p = pred_dir / f"{r['case']}.nii.gz"
        if p.exists():
            p.replace(out_dir / f"{r['case']}_ribmask.nii.gz")


def measure(case: str, rib_path: Path, vert_path: Path, sem_path: Path, out_dir: Path, calc_orientation: bool) -> list[dict]:
    from TPTBox import NII
    from run import run_all_steps                       # ~/rib-segmentation/run.py, his orchestration
    rib = NII.load(str(rib_path), seg=True)
    vert = NII.load(str(vert_path), seg=True)
    sem = NII.load(str(sem_path), seg=True)
    if rib.shape != vert.shape:                         # nnU-Net writes on the CT grid, SPINEPS on its own
        rib = rib.resample_from_to(vert)
    results = run_all_steps(rib, vert, sem, poi=None, calc_orientation=calc_orientation, verbose=False)
    rows = []
    for d in results:
        row = {"case": case}
        for k, v in d.items():
            if k == "features":
                continue
            if isinstance(v, _SCALARS) or v is None:
                row[k] = v
            elif isinstance(v, (list, tuple)) and all(isinstance(x, _SCALARS) for x in v):
                row[k] = ";".join(str(x) for x in v)
        rows.append(row)
        # the combined rib instance mask, if his step returns one, is saved once per case
        for k, v in d.items():
            if isinstance(v, NII) and "inst" in k.lower() and not (out_dir / f"{case}_rib-inst_msk.nii.gz").exists():
                v.save(str(out_dir / f"{case}_rib-inst_msk.nii.gz"))
    (out_dir / "rib_features").mkdir(exist_ok=True)
    (out_dir / "rib_features" / f"{case}.json").write_text(json.dumps([{k: _jsonable(v) for k, v in d.items()} for d in results], indent=1))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True, type=Path)
    ap.add_argument("--spineps_dir", required=True, type=Path, help="where run_spineps_cases.py wrote *_seg-vert_msk")
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--rib_model", required=True, type=Path)
    ap.add_argument("--rib_folds", default="0")
    ap.add_argument("--rib_checkpoint", default="checkpoint_final.pth")
    ap.add_argument("--ribseg_repo", type=Path, default=Path.home() / "rib-segmentation")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--calc_orientation", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, str(a.ribseg_repo))
    a.out_dir.mkdir(parents=True, exist_ok=True); a.work.mkdir(parents=True, exist_ok=True)

    with open(a.cases) as fh:
        rows = [r for i, r in enumerate(csv.DictReader(fh)) if i % a.n_shards == a.shard]
    if a.limit:
        rows = rows[: a.limit]
    have = [r for r in rows if (a.spineps_dir / f"{r['case']}_seg-vert_msk.nii.gz").exists()
            and (a.spineps_dir / f"{r['case']}_seg-spine_msk.nii.gz").exists()]
    print(f"shard {a.shard}/{a.n_shards}: {len(rows)} cases, {len(have)} with SPINEPS masks", flush=True)

    predict_ribs(have, a.work, a.rib_model, tuple(int(f) for f in a.rib_folds.split(",")), a.rib_checkpoint, a.out_dir)

    csv_path = a.out_dir / f"rib_measurements_shard{a.shard}.csv"
    done = set()
    if csv_path.exists():
        done = {r["case"] for r in csv.DictReader(open(csv_path))}
    all_rows = list(csv.DictReader(open(csv_path))) if csv_path.exists() else []
    n_fail = 0
    for r in have:
        case = r["case"]
        if case in done:
            continue
        rib_path = a.out_dir / f"{case}_ribmask.nii.gz"
        if not rib_path.exists():
            print(case, "no rib mask", flush=True); n_fail += 1; continue
        t0 = time.time()
        try:
            new = measure(case, rib_path, a.spineps_dir / f"{case}_seg-vert_msk.nii.gz",
                          a.spineps_dir / f"{case}_seg-spine_msk.nii.gz", a.out_dir, a.calc_orientation)
            all_rows += new
            print({"case": case, "rows": len(new), "seconds": round(time.time() - t0, 1)}, flush=True)
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            print({"case": case, "error": repr(e)[:600]}, flush=True)
        fields, seen = [], set()
        for row in all_rows:
            for k in row:
                if k not in seen:
                    seen.add(k); fields.append(k)
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(all_rows)
    print(f"failed {n_fail}", flush=True)
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
