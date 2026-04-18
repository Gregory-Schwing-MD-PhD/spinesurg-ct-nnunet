#!/usr/bin/env python3
"""
SpineSurg-CT -- ablation results aggregator
tools/compare_ablations.py

Scans all ablation prediction directories (named ABL_<tag>_...), loads
each ablation's metrics_paper.json and ablation_meta.json, and writes a
side-by-side comparison table.

Outputs:
    <results_root>/ablation_comparison.md      markdown table
    <results_root>/ablation_comparison.tex     LaTeX table fragment
    <results_root>/ablation_comparison.csv     wide CSV for spreadsheet analysis

Metrics compared (per ablation row):
    - L1-L5 mean DSC
    - L6 (when present) DSC + detection rate
    - Sacrum DSC, Left/Right hip DSC
    - Junction (L5/S1) DSC
    - Foreground HD95, ASSD
    - N test cases

Usage
-----
    python tools/compare_ablations.py --results_root nnunet/predictions

Author: Gregory Schwing, MD-PhD  |  Wayne State University / DMC
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional


def _fmt_pct(m: Optional[float], s: Optional[float] = None, digits: int = 1) -> str:
    if m is None or (isinstance(m, float) and math.isnan(m)):
        return "--"
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return f"{m*100:.{digits}f}"
    return f"{m*100:.{digits}f} ± {s*100:.{digits}f}"


def _fmt_mm(m: Optional[float], s: Optional[float] = None, digits: int = 2) -> str:
    if m is None or (isinstance(m, float) and math.isnan(m)):
        return "--"
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return f"{m:.{digits}f}"
    return f"{m:.{digits}f} ± {s:.{digits}f}"


def _g(d: dict, *path, default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(p)
        if cur is None:
            return default
    return cur


def load_ablation(ablation_dir: Path) -> Optional[Dict]:
    """Load one ablation's metrics + metadata into a flat record."""
    metrics_path = ablation_dir / "paper_metrics" / "metrics_paper.json"
    meta_path    = ablation_dir / "paper_metrics" / "ablation_meta.json"
    if not metrics_path.exists():
        return None
    try:
        metrics = json.loads(metrics_path.read_text())
    except Exception as e:
        print(f"  SKIP {ablation_dir.name}: could not parse metrics_paper.json ({e})")
        return None
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            pass

    # Fall back to parsing directory name for the tag if no meta file
    tag = meta.get("ablation_tag") or ablation_dir.name.replace("ABL_", "").split("_Dataset")[0]

    summary = metrics.get("summary", {})
    l15   = summary.get("_L1_L5_mean", {})
    l6    = summary.get("L6", {})
    sa    = summary.get("sacrum", {})
    lh    = summary.get("left_hip", {})
    rh    = summary.get("right_hip", {})
    jx    = summary.get("_junction", {})
    fg    = summary.get("_foreground_mean", {})
    l6d   = summary.get("_L6_detection", {})

    return {
        "tag":           tag,
        "dir":           str(ablation_dir),
        "meta":          meta,
        "n_cases":       metrics.get("n_cases", 0),
        "l15_dsc_m":     l15.get("dsc_mean"),  "l15_dsc_s": l15.get("dsc_std"),
        "l6_dsc_all_m":  l6.get("dsc_mean"),   "l6_dsc_all_s": l6.get("dsc_std"),
        "l6_dsc_pres_m": _g(l6d, "dsc_when_both_present_mean"),
        "l6_dsc_pres_s": _g(l6d, "dsc_when_both_present_std"),
        "l6_detect":     _g(l6d, "detection_rate"),
        "l6_correct":    _g(l6d, "n_cases_l6_correctly_present"),
        "l6_present_gt": _g(l6d, "n_cases_with_l6_in_gt"),
        "sacrum_dsc_m":  sa.get("dsc_mean"),   "sacrum_dsc_s": sa.get("dsc_std"),
        "lh_dsc_m":      lh.get("dsc_mean"),   "lh_dsc_s": lh.get("dsc_std"),
        "rh_dsc_m":      rh.get("dsc_mean"),   "rh_dsc_s": rh.get("dsc_std"),
        "jxn_dsc_m":     jx.get("dsc_mean"),   "jxn_dsc_s": jx.get("dsc_std"),
        "fg_dsc_m":      fg.get("dsc_mean"),   "fg_dsc_s": fg.get("dsc_std"),
        "fg_hd95_m":     fg.get("hd95_mean"),  "fg_hd95_s": fg.get("hd95_std"),
        "fg_assd_m":     fg.get("assd_mean"),  "fg_assd_s": fg.get("assd_std"),
    }


def write_markdown(records: List[Dict], out_path: Path) -> None:
    lines = [
        "# Ablation comparison",
        "",
        "Values are mean ± std across test-set cases. DSC/IoU in %, HD95/ASSD in mm.",
        "",
        "| Ablation | N | L1-L5 DSC | L6 all | L6 when present (N/M) | Sacrum DSC | Junction DSC | Foreground DSC | FG HD95 (mm) | FG ASSD (mm) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in records:
        l6_pres = "--"
        if r["l6_present_gt"] and r["l6_present_gt"] > 0:
            l6_pres = (f"{_fmt_pct(r['l6_dsc_pres_m'])} "
                       f"({r['l6_correct']}/{r['l6_present_gt']})")
        lines.append(
            f"| **{r['tag']}** | {r['n_cases']} | "
            f"{_fmt_pct(r['l15_dsc_m'], r['l15_dsc_s'])} | "
            f"{_fmt_pct(r['l6_dsc_all_m'], r['l6_dsc_all_s'])} | "
            f"{l6_pres} | "
            f"{_fmt_pct(r['sacrum_dsc_m'], r['sacrum_dsc_s'])} | "
            f"{_fmt_pct(r['jxn_dsc_m'], r['jxn_dsc_s'])} | "
            f"{_fmt_pct(r['fg_dsc_m'], r['fg_dsc_s'])} | "
            f"{_fmt_mm(r['fg_hd95_m'], r['fg_hd95_s'])} | "
            f"{_fmt_mm(r['fg_assd_m'], r['fg_assd_s'])} |"
        )

    # Per-ablation config footnote
    lines += ["", "## Configurations", ""]
    lines += ["| Ablation | Trainer | Planner | Train match_types | Test match_types | LSTV frac |",
              "|---|---|---|---|---|---:|"]
    for r in records:
        m = r.get("meta", {})
        lines.append(
            f"| {r['tag']} | "
            f"`{m.get('trainer', '?')}` | "
            f"`{m.get('planner', '?')}` | "
            f"{m.get('include_train_match_types') or 'all'} | "
            f"{m.get('test_match_types') or 'all'} | "
            f"{m.get('lstv_oversample_frac', '?')} |"
        )

    out_path.write_text("\n".join(lines) + "\n")


def write_latex(records: List[Dict], out_path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Ablation study on the held-out test set (fold 0 only). "
        r"Values are mean $\pm$ std; DSC in \%, HD95 in mm.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lrcccccc}",
        r"\toprule",
        r"Ablation & N & L1--L5 DSC & L6 (present) & Sacrum DSC & Junction DSC & Foreground DSC & FG HD95 \\",
        r"\midrule",
    ]
    for r in records:
        lines.append(
            f"{r['tag'].replace('_', r' \\_')} & "
            f"{r['n_cases']} & "
            f"{_fmt_pct(r['l15_dsc_m'])} $\\pm$ {_fmt_pct(r['l15_dsc_s'])} & "
            f"{_fmt_pct(r['l6_dsc_pres_m'])} & "
            f"{_fmt_pct(r['sacrum_dsc_m'])} $\\pm$ {_fmt_pct(r['sacrum_dsc_s'])} & "
            f"{_fmt_pct(r['jxn_dsc_m'])} $\\pm$ {_fmt_pct(r['jxn_dsc_s'])} & "
            f"{_fmt_pct(r['fg_dsc_m'])} $\\pm$ {_fmt_pct(r['fg_dsc_s'])} & "
            f"{_fmt_mm(r['fg_hd95_m'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out_path.write_text("\n".join(lines) + "\n")


def write_csv(records: List[Dict], out_path: Path) -> None:
    if not records:
        return
    # Flatten meta into top level for CSV
    rows = []
    for r in records:
        row = {k: v for k, v in r.items() if k != "meta"}
        for mk, mv in r.get("meta", {}).items():
            row[f"cfg_{mk}"] = mv
        rows.append(row)
    keys = sorted({k for row in rows for k in row.keys()})
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Aggregate ablation results into a comparison table.",
    )
    ap.add_argument("--results_root", required=True, type=Path,
                    help="Directory containing ABL_<tag>_... subdirectories "
                         "(e.g. nnunet/predictions/).")
    ap.add_argument("--glob", default="ABL_*",
                    help="Glob pattern for ablation subdirectories.")
    ap.add_argument("--out_dir", type=Path, default=None,
                    help="Where to write outputs (default: --results_root).")
    args = ap.parse_args()

    if not args.results_root.is_dir():
        raise SystemExit(f"results_root not a directory: {args.results_root}")

    out_dir = args.out_dir or args.results_root
    out_dir.mkdir(parents=True, exist_ok=True)

    ablation_dirs = sorted(args.results_root.glob(args.glob))
    print(f"Found {len(ablation_dirs)} ablation directories under {args.results_root}")

    records: List[Dict] = []
    for d in ablation_dirs:
        rec = load_ablation(d)
        if rec is None:
            print(f"  SKIP {d.name} (no paper_metrics/metrics_paper.json)")
            continue
        records.append(rec)
        l6_pres_str = "--"
        if rec.get("l6_present_gt"):
            l6_pres_str = (f"{_fmt_pct(rec['l6_dsc_pres_m'])}% "
                           f"({rec['l6_correct']}/{rec['l6_present_gt']})")
        print(f"  {rec['tag']:30s} "
              f"L1-L5={_fmt_pct(rec['l15_dsc_m'])}% "
              f"L6 (pres)={l6_pres_str} "
              f"Jxn={_fmt_pct(rec['jxn_dsc_m'])}%")

    if not records:
        print("No ablations loaded.")
        return 1

    # Sort by tag for reproducible output ordering
    records.sort(key=lambda r: r["tag"])

    md_path  = out_dir / "ablation_comparison.md"
    tex_path = out_dir / "ablation_comparison.tex"
    csv_path = out_dir / "ablation_comparison.csv"
    write_markdown(records, md_path)
    write_latex(records, tex_path)
    write_csv(records, csv_path)
    print(f"\nWrote:")
    print(f"  {md_path}")
    print(f"  {tex_path}")
    print(f"  {csv_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
