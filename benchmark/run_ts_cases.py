#!/usr/bin/env python3
"""benchmark/run_ts_cases.py — TotalSegmentator (task 'total', full resolution, multilabel)
over a shard of the benchmark case table. Runs INSIDE containers/ctspinopelvic1k-ts.sif
(see benchmark/run_totalsegmentator.sh). Native TS ids on the CT's own grid are written to
<out_dir>/<case>.nii.gz; mapping to the one-shot space happens at scoring time.

No roi_subset: with a subset TS first localises on its 3 mm model and crops, and that crop is
a second model in the loop. The full task is what a downstream user gets by default.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fast", action="store_true", help="3 mm model (smoke tests only)")
    ap.add_argument("--ct_prefix_from", default="", help="rewrite CT path prefix (host -> container)")
    ap.add_argument("--ct_prefix_to", default="")
    a = ap.parse_args()

    import nibabel as nib
    from totalsegmentator.python_api import totalsegmentator

    a.out_dir.mkdir(parents=True, exist_ok=True)
    # the id -> name map of the installed TS version, which to_oneshot.py maps by name
    from totalsegmentator.map_to_binary import class_map
    cm = a.out_dir.parent / "class_map.json"
    if not cm.exists():
        cm.write_text(json.dumps({str(k): v for k, v in class_map["total"].items()}, indent=0))
    with open(a.cases) as fh:
        rows = [r for i, r in enumerate(csv.DictReader(fh)) if i % a.n_shards == a.shard]
    if a.limit:
        rows = rows[: a.limit]
    print(f"shard {a.shard}/{a.n_shards}: {len(rows)} cases", flush=True)
    log = open(a.out_dir / f"timing_shard{a.shard}.jsonl", "a")
    n_done = n_skip = n_fail = 0
    for r in rows:
        out = a.out_dir / f"{r['case']}.nii.gz"
        if out.exists():
            n_skip += 1; continue
        ct = r["ct"]
        if a.ct_prefix_from and ct.startswith(a.ct_prefix_from):
            ct = a.ct_prefix_to + ct[len(a.ct_prefix_from):]
        t0 = time.time()
        try:
            pred = totalsegmentator(input=nib.load(ct), output=None, task="total", ml=True,
                                    device="gpu", fast=a.fast, verbose=False, quiet=True)
            tmp = out.with_name(out.name + ".part.nii.gz")
            nib.save(pred, str(tmp)); tmp.rename(out)
            n_done += 1
            rec = {"case": r["case"], "seconds": round(time.time() - t0, 1), "ok": True}
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            rec = {"case": r["case"], "seconds": round(time.time() - t0, 1), "ok": False, "error": repr(e)[:300]}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(rec, flush=True)
    print(f"done {n_done}, skipped {n_skip}, failed {n_fail}", flush=True)
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
