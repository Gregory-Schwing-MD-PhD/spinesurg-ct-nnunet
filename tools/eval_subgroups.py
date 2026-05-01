#!/usr/bin/env python3
"""
tools/eval_subgroups.py -- 4-way LSTV subgroup analysis for SpineSurg-CT.

Generates the headline benchmark table broken down by LSTV subtype
(lumb / sacr / ambiguous / normal). Reads case-level dice from a CSV and
joins with patient-token subtype labels from splits_5fold.json.

Inputs
------
  --splits_file       splits_5fold.json (schema v5 for 4-way; v4 OK as 3-way)
  --predictions_csv   one row per case_id with at least:
                          case_id, dice_l1..dice_l6, dice_sacrum,
                          dice_left_hip, dice_right_hip, dice_overall
                      Extra columns OK; missing class columns -> NaN.
  --output_dir        where to write tables.

Outputs
-------
  subgroup_dice_summary.csv     headline table (rows: subtype)
  subgroup_dice_summary.md      markdown for paste-into-paper
  subgroup_dice_summary.json    machine-readable mean/std/n/CI95
  subgroup_dice_per_case.csv    every case_id with subtype + dice

Usage
-----
  python3 tools/eval_subgroups.py \
      --splits_file     data/hf_export/splits_5fold.json \
      --predictions_csv path/to/dice_scores.csv \
      --output_dir      results/subgroup_eval/ \
      --label           nnunet

  # Side-by-side comparison (e.g. nnU-Net vs TotalSegmentator):
  python3 tools/eval_subgroups.py \
      --splits_file     data/hf_export/splits_5fold.json \
      --predictions_csv results/nnunet_dice.csv \
      --compare_with    results/ts_dice.csv \
      --label nnunet --compare_label ts \
      --output_dir      results/subgroup_eval/

Author: Gregory Schwing, MD-PhD  |  Wayne State University / DMC
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


SUBTYPES = ("normal", "lumb", "sacr", "ambiguous")
CLASS_COLS = ("dice_l1", "dice_l2", "dice_l3", "dice_l4", "dice_l5",
              "dice_l6", "dice_sacrum", "dice_left_hip", "dice_right_hip",
              "dice_overall")
PRETTY_SUBTYPE = {
    "normal": "Normal",
    "lumb": "Lumbarization",
    "sacr": "Sacralization",
    "ambiguous": "Ambiguous (Castellvi II/III)",
}
PRETTY_CLASS = {
    "dice_l1": "L1", "dice_l2": "L2", "dice_l3": "L3",
    "dice_l4": "L4", "dice_l5": "L5", "dice_l6": "L6",
    "dice_sacrum": "Sacrum",
    "dice_left_hip": "L Hip", "dice_right_hip": "R Hip",
    "dice_overall": "Overall",
}


def load_splits_token_info(path: Path) -> Dict[str, Dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    data = json.loads(path.read_text())
    schema = data.get("schema_version", 0)
    if schema < 5:
        print(f"WARN: splits schema v{schema}; v5+ has 'ambiguous' subtype. "
              f"Falling back to 3-way.", file=sys.stderr)
    info = data.get("token_info") or {}
    if not info:
        raise RuntimeError(f"{path}: no token_info")
    return {str(k): v for k, v in info.items()}


def case_id_to_token(case_id: str) -> str:
    return case_id.split("__", 1)[0] if "__" in case_id else case_id


def load_predictions_csv(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    rows: List[Dict] = []
    with open(path) as f:
        rdr = csv.DictReader(f)
        if "case_id" not in (rdr.fieldnames or []):
            raise ValueError(f"{path}: 'case_id' column required. Got: {rdr.fieldnames}")
        for r in rdr:
            cleaned = {"case_id": str(r["case_id"]).strip()}
            for c in CLASS_COLS:
                v = r.get(c)
                if v is None or v == "" or str(v).lower() in ("nan", "none"):
                    cleaned[c] = float("nan")
                else:
                    try: cleaned[c] = float(v)
                    except ValueError: cleaned[c] = float("nan")
            for k, v in r.items():
                if k not in cleaned and k != "case_id":
                    cleaned[k] = v
            rows.append(cleaned)
    return rows


def aggregate_by_subtype(rows: List[Dict], token_info: Dict[str, Dict]
                          ) -> Tuple[Dict[str, Dict[str, Dict]], List[Dict]]:
    per_case: List[Dict] = []
    by_subtype: Dict[str, List[Dict]] = {s: [] for s in SUBTYPES}
    n_unmatched = 0
    sample_unmatched: List[str] = []
    for r in rows:
        cid = r["case_id"]
        tok = case_id_to_token(cid)
        info = token_info.get(tok)
        if info is None:
            n_unmatched += 1
            if len(sample_unmatched) < 5:
                sample_unmatched.append(f"  {cid} (token={tok})")
            sub = "unmatched"
        else:
            sub = info.get("lstv_subtype") or "normal"
            if sub not in SUBTYPES:
                sub = "normal"
        per_case.append({"token": tok, "lstv_subtype": sub, **r})
        if sub in by_subtype:
            by_subtype[sub].append(r)
    if n_unmatched:
        print(f"WARN: {n_unmatched} cases had no token in splits file:", file=sys.stderr)
        for s in sample_unmatched: print(s, file=sys.stderr)

    summary: Dict[str, Dict[str, Dict]] = {}
    for sub in SUBTYPES:
        per_class: Dict[str, Dict] = {}
        for c in CLASS_COLS:
            vals = np.array([r[c] for r in by_subtype[sub] if not np.isnan(r[c])], dtype=float)
            n = int(len(vals))
            if n == 0:
                per_class[c] = {"n": 0, "mean": None, "std": None,
                                  "ci95_lo": None, "ci95_hi": None, "values": []}
                continue
            mean = float(vals.mean())
            std = float(vals.std(ddof=1)) if n > 1 else 0.0
            ci95 = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
            per_class[c] = {
                "n": n, "mean": mean, "std": std,
                "ci95_lo": float(mean - ci95), "ci95_hi": float(mean + ci95),
                "values": [float(v) for v in vals],
            }
        per_class["_n_cases"] = len(by_subtype[sub])
        summary[sub] = per_class
    return summary, per_case


def write_summary_csv(summary, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subtype", "n_cases"] + list(CLASS_COLS))
        for sub in SUBTYPES:
            row = [PRETTY_SUBTYPE[sub], summary[sub]["_n_cases"]]
            for c in CLASS_COLS:
                cell = summary[sub][c]
                row.append("" if cell["n"] == 0 else
                            f"{cell['mean']:.3f} ± {cell['std']:.3f} (n={cell['n']})")
            w.writerow(row)
    print(f"  wrote {out_path}")


def write_summary_md(summary, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["dice_l6", "dice_overall"]
    lines = []
    header = ["Subtype", "n"] + [PRETTY_CLASS[c] for c in cols]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for sub in SUBTYPES:
        row = [PRETTY_SUBTYPE[sub], str(summary[sub]["_n_cases"])]
        for c in cols:
            cell = summary[sub][c]
            row.append("—" if cell["n"] == 0 else f"{cell['mean']:.3f} ± {cell['std']:.3f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("Full per-class breakdown in `subgroup_dice_summary.csv`.")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"  wrote {out_path}")


def write_summary_json(summary, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"  wrote {out_path}")


def write_per_case_csv(per_case, out_path):
    if not per_case: return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["token", "lstv_subtype", "case_id"] + list(CLASS_COLS)
    extra = sorted({k for r in per_case for k in r.keys() if k not in keys})
    keys += extra
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in per_case: w.writerow(r)
    print(f"  wrote {out_path}")


def print_summary_to_stdout(summary, label=""):
    if label:
        print(f"\n{'=' * 70}\n {label}\n{'=' * 70}")
    print(f"\n  4-way LSTV subgroup dice:\n")
    print(f"  {'Subtype':<32s} {'n':>4s}  {'L6 dice':<20s} {'Overall':<20s}")
    print(f"  {'-' * 32} {'-' * 4}  {'-' * 20} {'-' * 20}")
    for sub in SUBTYPES:
        n = summary[sub]["_n_cases"]
        l6 = summary[sub]["dice_l6"]
        ov = summary[sub]["dice_overall"]
        l6s = f"{l6['mean']:.3f} ± {l6['std']:.3f}" if l6["n"] > 0 else "—"
        ovs = f"{ov['mean']:.3f} ± {ov['std']:.3f}" if ov["n"] > 0 else "—"
        print(f"  {PRETTY_SUBTYPE[sub]:<32s} {n:>4d}  {l6s:<20s} {ovs:<20s}")


def compare_summaries(summary_a, summary_b, label_a, label_b, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["dice_l6", "dice_overall"]
    lines = []
    header = ["Subtype", "n",
               f"{label_a} L6", f"{label_b} L6",
               f"{label_a} Overall", f"{label_b} Overall"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for sub in SUBTYPES:
        n = summary_a[sub]["_n_cases"]
        row = [PRETTY_SUBTYPE[sub], str(n)]
        for c in cols:
            for s in (summary_a, summary_b):
                cell = s[sub][c]
                row.append("—" if cell["n"] == 0 else f"{cell['mean']:.3f} ± {cell['std']:.3f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append(f"**{label_a}** vs **{label_b}**, per-LSTV-subgroup dice (mean ± SD).")
    lines.append("")
    lines.append("'Ambiguous' = 2 patients (tokens 4, 67) with annotator disagreement "
                 "at the lumbosacral transition consistent with Castellvi II/III "
                 "asymmetric morphology.")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"  wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--splits_file", required=True, type=Path)
    ap.add_argument("--predictions_csv", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--label", default="model")
    ap.add_argument("--compare_with", type=Path, default=None)
    ap.add_argument("--compare_label", default="other")
    args = ap.parse_args()

    print(f"Loading splits from {args.splits_file}")
    token_info = load_splits_token_info(args.splits_file)
    print(f"  token_info entries: {len(token_info)}")
    sub_counts = {s: 0 for s in SUBTYPES}
    for tok, info in token_info.items():
        s = info.get("lstv_subtype") or "normal"
        if s in sub_counts: sub_counts[s] += 1
    print(f"  patient-level subtype counts: {sub_counts}")
    if sub_counts.get("ambiguous", 0) > 0:
        amb = sorted((tok for tok, info in token_info.items()
                       if info.get("lstv_subtype") == "ambiguous"),
                      key=lambda x: int(x) if x.isdigit() else 99999)
        print(f"  ambiguous tokens: {amb}")

    print(f"\nLoading predictions from {args.predictions_csv}")
    rows = load_predictions_csv(args.predictions_csv)
    print(f"  rows: {len(rows)}")

    print(f"\nAggregating...")
    summary, per_case = aggregate_by_subtype(rows, token_info)
    print_summary_to_stdout(summary, label=args.label)

    print(f"\nWriting outputs to {args.output_dir}")
    out = args.output_dir
    write_summary_csv(summary, out / "subgroup_dice_summary.csv")
    write_summary_md(summary, out / "subgroup_dice_summary.md")
    write_summary_json(summary, out / "subgroup_dice_summary.json")
    write_per_case_csv(per_case, out / "subgroup_dice_per_case.csv")

    if args.compare_with:
        print(f"\nLoading comparison from {args.compare_with}")
        rows_b = load_predictions_csv(args.compare_with)
        summary_b, per_case_b = aggregate_by_subtype(rows_b, token_info)
        print_summary_to_stdout(summary_b, label=args.compare_label)
        cdir = out / f"compare_{args.label}_vs_{args.compare_label}"
        compare_summaries(summary, summary_b, args.label, args.compare_label,
                           cdir / "comparison.md")
        write_summary_csv(summary_b, cdir / f"{args.compare_label}_subgroup_dice.csv")
        write_per_case_csv(per_case_b, cdir / f"{args.compare_label}_per_case.csv")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
