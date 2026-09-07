#!/usr/bin/env python3
"""tools/eval_oneshot.py — what the one-shot LSTV arm has to be scored on.

Dice is not the endpoint. A model that is 97% accurate overall and 0% on transitional
anatomy is the null result wearing a good number, and only a split by transitional status
shows it. So this reports, per fold and pooled:

  1. per-class Dice (every class in dataset.json), and separately for typical and
     transitional cases;
  2. RIB-FREE COUNT accuracy: the number of distinct lumbar-type vertebrae the prediction
     carries (L1..L6 classes present with >= MIN_VOX voxels) against the ground truth,
     split by transitional status, with the sign of the error (off by -1, 0, +1, ...);
  3. the RIB CONFUSION read BY CLASS: how ground-truth twelfth-rib voxels, thirteenth-rib
     voxels and lumbar-rib voxels are distributed across predicted classes, so that a
     model that never says "lumbar rib" is caught (it would still score 94% on short ribs);
  4. per-case emission of the rare classes (did the model ever output a lumbar rib or a
     T13, and on which cases).

    python tools/eval_oneshot.py \
        --predictions_dir nnunet/results/Dataset810_.../fold_0/validation \
        --labels_dir nnunet/raw/Dataset810_SpineSurgLSTVOneShot/labelsTr \
        --dataset_json nnunet/raw/Dataset810_SpineSurgLSTVOneShot/dataset.json \
        --lstv_cases_json nnunet/preprocessed/Dataset810_SpineSurgLSTVOneShot/lstv_cases.json \
        --output_dir results/oneshot_fold0

Volumes are read as stored; nothing is reoriented.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np

MIN_VOX = 200            # a class "present" in a volume needs at least this many voxels


def _names(dataset_json: Path) -> dict[int, str]:
    d = json.loads(dataset_json.read_text())
    lab = d["labels"]
    out = {}
    for k, v in lab.items():
        if isinstance(v, list):      # region-style entry: skip
            continue
        out[int(v)] = k
    return dict(sorted(out.items()))


def _one(args):
    pred_path, gt_path, ids, lumbar_ids, rib_ids, ignore_id = args
    case = Path(pred_path).name.replace(".nii.gz", "")
    p = np.asanyarray(nib.load(str(pred_path)).dataobj).astype(np.int16)
    g = np.asanyarray(nib.load(str(gt_path)).dataobj).astype(np.int16)
    if p.shape != g.shape:
        return {"case": case, "error": f"shape {p.shape} vs {g.shape}"}
    valid = g != ignore_id
    p, g = p[valid], g[valid]
    out = {"case": case, "dice": {}, "gt_lumbar": 0, "pred_lumbar": 0, "rib_conf": {},
           "emit": {}, "gt_present": {}}
    for i in ids:
        gi, pi = g == i, p == i
        ng, npred = int(gi.sum()), int(pi.sum())
        out["gt_present"][i] = ng >= MIN_VOX
        out["emit"][i] = npred >= MIN_VOX
        if ng == 0 and npred == 0:
            out["dice"][i] = None
        else:
            out["dice"][i] = 2.0 * int((gi & pi).sum()) / (ng + npred)
    out["gt_lumbar"] = sum(1 for i in lumbar_ids if out["gt_present"][i])
    out["pred_lumbar"] = sum(1 for i in lumbar_ids if out["emit"][i])
    for r in rib_ids:                     # where do GT rib voxels of each rib class go?
        m = g == r
        if m.any():
            vals, cnts = np.unique(p[m], return_counts=True)
            out["rib_conf"][r] = {int(v): int(c) for v, c in zip(vals, cnts)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions_dir", required=True, type=Path)
    ap.add_argument("--labels_dir", required=True, type=Path)
    ap.add_argument("--dataset_json", required=True, type=Path)
    ap.add_argument("--lstv_cases_json", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)

    names = _names(a.dataset_json)
    by_name = {v: k for k, v in names.items()}
    ignore_id = by_name.get("ignore", -1)
    ids = [i for i in names if i != 0 and i != ignore_id]
    lumbar_ids = [by_name[n] for n in ("L1", "L2", "L3", "L4", "L5", "L6") if n in by_name]
    rib_ids = [by_name[n] for n in ("rib12_left", "rib12_right", "rib13_left", "rib13_right",
                                     "lumbar_rib_left", "lumbar_rib_right") if n in by_name]

    lc = json.loads(a.lstv_cases_json.read_text())
    subtype = lc.get("case_to_subtype", {}) or {}
    attrs = lc.get("case_to_attrs", {}) or {}

    def transitional(case):
        s = subtype.get(case, "normal")
        return s != "normal" or bool((attrs.get(case) or {}).get("has_lumbar_rib"))

    preds = sorted(a.predictions_dir.glob("*.nii.gz"))
    jobs = []
    for pp in preds:
        gp = a.labels_dir / pp.name
        if gp.exists():
            jobs.append((str(pp), str(gp), ids, lumbar_ids, rib_ids, ignore_id))
    print(f"{len(jobs)} cases with prediction and label")
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(_one, jobs, chunksize=2))
    res = [r for r in res if "error" not in r]

    # ---- 1. Dice, overall and by status -------------------------------------------------
    rows = []
    dice_acc = {"all": defaultdict(list), "typical": defaultdict(list), "transitional": defaultdict(list)}
    for r in res:
        grp = "transitional" if transitional(r["case"]) else "typical"
        for i, d in r["dice"].items():
            if d is not None and r["gt_present"][i]:
                dice_acc["all"][i].append(d); dice_acc[grp][i].append(d)
        rows.append({"case": r["case"], "status": grp, "subtype": subtype.get(r["case"], "normal"),
                     "gt_lumbar": r["gt_lumbar"], "pred_lumbar": r["pred_lumbar"],
                     "count_error": r["pred_lumbar"] - r["gt_lumbar"],
                     **{f"dice_{names[i]}": ("" if r["dice"][i] is None else f"{r['dice'][i]:.4f}") for i in ids},
                     **{f"emit_{names[i]}": int(r["emit"][i]) for i in rib_ids + [by_name.get("T13", -1)] if i in names}})
    with open(a.output_dir / "per_case.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # ---- 2. rib-free count accuracy by status -------------------------------------------
    count = {}
    for grp in ("all", "typical", "transitional"):
        sel = [r for r in rows if grp == "all" or r["status"] == grp]
        errs = Counter(r["count_error"] for r in sel)
        count[grp] = {"n": len(sel), "exact": sum(1 for r in sel if r["count_error"] == 0) / max(1, len(sel)),
                      "error_histogram": {str(k): v for k, v in sorted(errs.items())}}

    # ---- 3. rib confusion by class -------------------------------------------------------
    conf = {}
    for r in res:
        for rid, dist in r["rib_conf"].items():
            c = conf.setdefault(names[rid], Counter())
            for pid, n in dist.items():
                c[names.get(pid, str(pid))] += n
    conf_pct = {k: {pn: round(100 * n / sum(v.values()), 2) for pn, n in v.most_common()} for k, v in conf.items()}

    # ---- 4. rare-class emission -----------------------------------------------------------
    emit = {}
    for i in rib_ids + ([by_name["T13"]] if "T13" in by_name else []):
        gt_cases = [r["case"] for r in res if r["gt_present"][i]]
        hit = [c for c in gt_cases if next(x for x in res if x["case"] == c)["emit"][i]]
        false = [r["case"] for r in res if r["emit"][i] and not r["gt_present"][i]]
        emit[names[i]] = {"gt_cases": len(gt_cases), "emitted_on_gt_cases": len(hit),
                          "emitted_where_absent": len(false), "missed": sorted(set(gt_cases) - set(hit))}

    summary = {
        "n_cases": len(res),
        "n_transitional": sum(1 for r in rows if r["status"] == "transitional"),
        "dice_mean": {grp: {names[i]: round(float(np.mean(v)), 4) for i, v in acc.items()} for grp, acc in dice_acc.items()},
        "rib_free_count": count,
        "rib_confusion_pct_of_gt_voxels": conf_pct,
        "rare_class_emission": emit,
    }
    (a.output_dir / "summary.json").write_text(json.dumps(summary, indent=1))

    lines = [f"cases {len(res)} (transitional {summary['n_transitional']})",
             "", "rib-free count exact-match rate:"]
    for grp, c in count.items():
        lines.append(f"  {grp:13s} n={c['n']:4d}  exact {100*c['exact']:5.1f}%  errors {c['error_histogram']}")
    lines += ["", "where the ground-truth rib voxels went (percent of GT voxels of that class):"]
    for k, v in conf_pct.items():
        lines.append(f"  {k:18s} " + ", ".join(f"{pn} {pct}%" for pn, pct in list(v.items())[:5]))
    lines += ["", "rare classes:"]
    for k, v in emit.items():
        lines.append(f"  {k:18s} GT in {v['gt_cases']:3d} cases, emitted on {v['emitted_on_gt_cases']:3d} of them, "
                     f"emitted where absent in {v['emitted_where_absent']:3d}")
    lines += ["", "mean Dice, typical vs transitional:"]
    for i in ids:
        t, tr = summary["dice_mean"]["typical"].get(names[i]), summary["dice_mean"]["transitional"].get(names[i])
        lines.append(f"  {names[i]:18s} {'' if t is None else f'{t:.3f}':>6s}  {'' if tr is None else f'{tr:.3f}':>6s}")
    text = "\n".join(lines)
    (a.output_dir / "report.txt").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
