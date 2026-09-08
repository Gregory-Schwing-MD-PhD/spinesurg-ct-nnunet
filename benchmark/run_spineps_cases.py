#!/usr/bin/env python3
"""benchmark/run_spineps_cases.py — SPINEPS (CT models) with VERIDAH labeling over a shard of
the benchmark case table. Runs in the `spineps` conda env (benchmark/install_envs.sh) with
SPINEPS_SEGMENTOR_MODELS pointing at the unpacked CT models (benchmark/fetch_weights.sh).

Each case is staged alone as <work>/<case>/<case>_ct.nii.gz (spineps writes a derivatives
tree next to its input and names outputs from the input name), `spineps sample` runs on it,
and the vertebra INSTANCE mask (VerSe ids; L6 = 25, T13 = 28 when VERIDAH says so) is
harvested to <out_dir>/<case>_seg-vert_msk.nii.gz with the semantic mask and the centroid
json beside it. Resumable.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


def harvest(stage: Path, case: str, out_dir: Path) -> dict:
    found = {}
    for p in stage.rglob("*"):
        if not p.is_file():
            continue
        n = p.name
        if n.endswith(".nii.gz") and "seg-vert" in n:
            found["vert"] = p
        elif n.endswith(".nii.gz") and "seg-spine" in n:
            found["spine"] = p
        elif n.endswith(".json") and ("ctd" in n or "poi" in n):
            found["ctd"] = p
    dst = {"vert": out_dir / f"{case}_seg-vert_msk.nii.gz",
           "spine": out_dir / f"{case}_seg-spine_msk.nii.gz",
           "ctd": out_dir / f"{case}_ctd.json"}
    for k, p in found.items():
        shutil.copy2(p, dst[k])
    return {k: str(dst[k]) for k in found}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model_semantic", default="ct")
    ap.add_argument("--model_instance", default="CT_instance")
    ap.add_argument("--model_labeling", default="ct_labeling")
    ap.add_argument("--extra", default="", help="extra flags passed straight to spineps sample")
    ap.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args()

    a.out_dir.mkdir(parents=True, exist_ok=True); a.work.mkdir(parents=True, exist_ok=True)
    with open(a.cases) as fh:
        rows = [r for i, r in enumerate(csv.DictReader(fh)) if i % a.n_shards == a.shard]
    if a.limit:
        rows = rows[: a.limit]
    print(f"shard {a.shard}/{a.n_shards}: {len(rows)} cases", flush=True)
    log = open(a.out_dir / f"timing_shard{a.shard}.jsonl", "a")
    n_done = n_skip = n_fail = 0
    for r in rows:
        case = r["case"]
        if (a.out_dir / f"{case}_seg-vert_msk.nii.gz").exists():
            n_skip += 1; continue
        stage = a.work / case
        shutil.rmtree(stage, ignore_errors=True); stage.mkdir(parents=True)
        staged = stage / f"{case}_ct.nii.gz"
        os.symlink(r["ct"], staged)
        cmd = ["spineps", "sample", "-i", str(staged),
               "--model-semantic", a.model_semantic, "--model-instance", a.model_instance,
               "--model-labeling", a.model_labeling,
               "--ignore-inference-compatibility", "--no-n4"] + (a.extra.split() if a.extra else [])
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
            if proc.returncode != 0:
                raise RuntimeError(f"spineps exited {proc.returncode}: {proc.stderr[-600:]}")
            got = harvest(stage, case, a.out_dir)
            if "vert" not in got:
                raise RuntimeError("no seg-vert mask produced; stage tree: " +
                                   ", ".join(p.name for p in stage.rglob("*") if p.is_file())[:500])
            n_done += 1
            rec = {"case": case, "seconds": round(time.time() - t0, 1), "ok": True, "files": got}
            shutil.rmtree(stage, ignore_errors=True)
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            rec = {"case": case, "seconds": round(time.time() - t0, 1), "ok": False, "error": repr(e)[:800]}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(rec, flush=True)
    print(f"done {n_done}, skipped {n_skip}, failed {n_fail}", flush=True)
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
