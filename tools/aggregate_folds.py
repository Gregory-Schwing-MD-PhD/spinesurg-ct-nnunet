#!/usr/bin/env python3
"""
tools/aggregate_folds.py — Combine per-fold eval outputs into mean ± std.

Input: 5 per-fold output directories (each from eval_full.py) plus
       optionally the ensemble output directory.

Output (under --output_dir):
  cross_fold_headline.json     scalar metrics: mean ± std across folds
  cross_fold_summary.csv       per-subtype × class: mean ± std across folds
  cross_fold_summary.md        paper-ready markdown
  cross_fold_confusion.json    Block 1/2/3: voxel sums across folds,
                                percentages re-derived from the sums
                                (NOT averaged percentages, which would
                                weight small-voxel cases disproportionately)

Usage:
  python tools/aggregate_folds.py \
      --fold_dirs eval/fold_0 eval/fold_1 eval/fold_2 eval/fold_3 eval/fold_4 \
      --ensemble_dir eval/ensemble \
      --output_dir eval/cross_fold

Author: Gregory Schwing, MD-PhD  |  Wayne State University / DMC
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


_KNOWN_SUBTYPES = (
    "normal", "lumb", "sacr_count", "semisacralization",
    "sacralization", "ambiguous",
)
PRETTY = {
    "normal":            "Normal",
    "lumb":              "Lumbarization",
    "sacr_count":        "Sacralization (count)",
    "semisacralization": "Semisacralization",
    "sacralization":     "Sacralization (morph)",
    "ambiguous":         "Ambiguous",
}
_SUBGROUP_CSV_CLASSES = [
    "L1", "L2", "L3", "L4", "last_lumbar",
    "sacrum", "left_hip", "right_hip",
    "hip_merged", "junction_dsc",
]
# Headline scalars we'll aggregate as mean ± std across folds.
_HEADLINE_SCALARS = [
    "mean_fg_dice_overall", "hip_merged_dice_overall",
    "left_hip_dice_overall", "right_hip_dice_overall",
    "sacrum_dice_overall", "sacrum_minus_TS_overall_pts",
    "junction_dsc_overall", "junction_dsc_normal",
    "junction_dsc_lumb", "junction_dsc_lumb_minus_TS_pts",
    "junction_dsc_sacr_count",
    "last_lumbar_dice_on_lumb", "L4_dice_on_sacr_count",
    "last_lumbar_specificity_on_sacr_count",
    "normal_mean_fg_dice", "any_lstv_mean_fg_dice",
    "any_lstv_minus_normal_fg_dice",
]


def _fmt(v: Optional[float], fmt: str = "{:.3f}") -> str:
    return fmt.format(v) if v is not None else "—"


def _load_fold(fold_dir: Path) -> Dict[str, Any]:
    """Read all JSON / CSV outputs from a single fold's eval dir."""
    out: Dict[str, Any] = {"path": str(fold_dir)}
    hl_path = fold_dir / "headline.json"
    cb_path = fold_dir / "confusion_blocks.json"
    sg_path = fold_dir / "subgroup_summary.csv"
    if hl_path.exists():
        out["headline"] = json.loads(hl_path.read_text())
    else:
        out["headline"] = {}
    if cb_path.exists():
        out["confusion"] = json.loads(cb_path.read_text())
    else:
        out["confusion"] = {}
    out["subgroup"] = {}
    if sg_path.exists():
        with open(sg_path) as f:
            for row in csv.DictReader(f):
                sub = row["subtype"]
                # cast numeric fields
                cast: Dict[str, Optional[float]] = {}
                for k, v in row.items():
                    if v == "" or v is None:
                        cast[k] = None
                    elif k in ("subtype",):
                        cast[k] = v
                    else:
                        try:
                            cast[k] = float(v)
                        except ValueError:
                            cast[k] = v
                out["subgroup"][sub] = cast
    return out


def _mean_std(vals: List[Optional[float]]) -> Dict[str, Optional[float]]:
    real = [v for v in vals if v is not None]
    if not real:
        return {"mean": None, "std": None, "n_folds": 0,
                "values": [None] * len(vals)}
    return {
        "mean":    float(np.mean(real)),
        "std":     float(np.std(real, ddof=1)) if len(real) > 1 else 0.0,
        "n_folds": len(real),
        "values":  vals,
    }


def aggregate_headline(folds: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in _HEADLINE_SCALARS:
        vals = [f["headline"].get(key) for f in folds]
        out[key] = _mean_std(vals)
    return out


def aggregate_subgroup(folds: List[Dict[str, Any]]) -> Dict[str, Any]:
    """For each (subtype, class), compute mean ± std of per-fold means."""
    agg: Dict[str, Dict[str, Any]] = {}
    for sub in _KNOWN_SUBTYPES:
        agg[sub] = {"per_class": {}, "n_cases_per_fold": []}
        # per-fold n_cases for this subtype (sanity check of fold split)
        for f in folds:
            row = f["subgroup"].get(sub, {})
            n = row.get("n_cases")
            agg[sub]["n_cases_per_fold"].append(int(n) if n is not None else 0)
        for cls in _SUBGROUP_CSV_CLASSES:
            mean_key = f"{cls}_mean"
            vals = [f["subgroup"].get(sub, {}).get(mean_key) for f in folds]
            agg[sub]["per_class"][cls] = _mean_std(vals)
        # Mean fg dice across folds
        vals = [f["subgroup"].get(sub, {}).get("mean_fg_dice") for f in folds]
        agg[sub]["mean_fg_dice"] = _mean_std(vals)
        # LL emit: integer counts -> sum across folds + per-fold values
        emits = [int(f["subgroup"].get(sub, {}).get("ll_emit_n_emit") or 0)
                 for f in folds]
        totals = [int(f["subgroup"].get(sub, {}).get("ll_emit_n_total") or 0)
                  for f in folds]
        agg[sub]["ll_emit"] = {
            "per_fold_emit":  emits,
            "per_fold_total": totals,
            "sum_emit":       sum(emits),
            "sum_total":      sum(totals),
        }
    return agg


def aggregate_confusion(folds: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Sum voxel counts across folds; re-derive percentages from sums.

    Averaging percentages across folds would over-weight folds with few
    voxels. Voxel-count summation gives a denominator-weighted mean,
    which is what the paper text describes (a single percentage across
    cases of the relevant subtype).
    """
    out: Dict[str, Any] = {}
    block_keys = [
        "block1_gt_LL_on_lumb",
        "block2_gt_L4_on_sacr_count",
        "block3_pred_LL_on_sacr_count",
    ]
    for bk in block_keys:
        # Determine row direction by inspecting the first fold that has data.
        sample = next((f["confusion"].get(bk) for f in folds
                        if f["confusion"].get(bk)), {})
        is_pred_dir = "by_pred" in (sample or {})
        sub_total = "gt_voxel_total" if is_pred_dir else "pred_voxel_total"
        per_class_key = "by_pred" if is_pred_dir else "by_gt"
        cls_voxels: Dict[str, int] = defaultdict(int)
        total_voxels = 0
        n_cases_total = 0
        for f in folds:
            blk = f["confusion"].get(bk, {})
            n_cases_total += int(blk.get("n_cases", 0))
            total_voxels += int(blk.get(sub_total, 0))
            for cname, cell in (blk.get(per_class_key) or {}).items():
                cls_voxels[cname] += int(cell.get("voxels", 0))
        merged = {
            "n_cases":       n_cases_total,
            "voxel_total":   int(total_voxels),
            per_class_key:   {},
        }
        if total_voxels > 0:
            for cname, vox in cls_voxels.items():
                merged[per_class_key][cname] = {
                    "voxels":  int(vox),
                    "percent": 100.0 * vox / total_voxels,
                }
        out[bk] = merged

    # last_lumbar specificity on sacr_count: sum TN, FP, gt_not5; recompute.
    tn_sum = fp_sum = neg_sum = 0
    n_sc_cases_total = 0
    for f in folds:
        spec = f["confusion"].get("last_lumbar_specificity_on_sacr_count", {})
        if spec.get("tn_voxels") is None:
            continue
        tn_sum  += int(spec.get("tn_voxels", 0) or 0)
        fp_sum  += int(spec.get("fp_voxels", 0) or 0)
        neg_sum += int(spec.get("gt_not5_voxels", 0) or 0)
        n_sc_cases_total += int(spec.get("n_sacr_count_cases", 0) or 0)
    if neg_sum > 0:
        out["last_lumbar_specificity_on_sacr_count"] = {
            "value": 1.0 - (fp_sum / neg_sum),
            "tn_voxels": tn_sum,
            "fp_voxels": fp_sum,
            "gt_not5_voxels": neg_sum,
            "n_sacr_count_cases_total": n_sc_cases_total,
        }
    else:
        out["last_lumbar_specificity_on_sacr_count"] = {
            "value": None, "n_sacr_count_cases_total": n_sc_cases_total,
        }

    # LL emit per subtype across folds
    emit: Dict[str, Dict[str, int]] = {}
    for f in folds:
        for sub, d in (f["confusion"].get("ll_emit_by_subtype") or {}).items():
            r = emit.setdefault(sub, {"n_total": 0, "n_emit": 0})
            r["n_total"] += int(d.get("n_total", 0))
            r["n_emit"]  += int(d.get("n_emit",  0))
    out["ll_emit_by_subtype"] = emit
    return out


# ===========================================================================
# Writers
# ===========================================================================

def _write_headline_json(headline_xf: Dict[str, Any],
                          ensemble_headline: Optional[Dict[str, Any]],
                          path: Path) -> None:
    out = {
        "cross_fold": headline_xf,
        "ensemble":   ensemble_headline or {},
    }
    path.write_text(json.dumps(out, indent=2))


def _write_subgroup_csv(subgroup_xf: Dict[str, Any], path: Path) -> None:
    cols = (["subtype", "n_cases_per_fold"]
            + [f"{c}_mean"   for c in _SUBGROUP_CSV_CLASSES]
            + [f"{c}_std"    for c in _SUBGROUP_CSV_CLASSES]
            + [f"{c}_nfolds" for c in _SUBGROUP_CSV_CLASSES]
            + ["mean_fg_dice_mean", "mean_fg_dice_std",
                "ll_emit_n_emit_total", "ll_emit_n_total"])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for sub in _KNOWN_SUBTYPES:
            row = {"subtype": sub}
            row["n_cases_per_fold"] = "/".join(
                str(n) for n in subgroup_xf[sub]["n_cases_per_fold"]
            )
            for cls in _SUBGROUP_CSV_CLASSES:
                stat = subgroup_xf[sub]["per_class"][cls]
                row[f"{cls}_mean"]   = stat["mean"]
                row[f"{cls}_std"]    = stat["std"]
                row[f"{cls}_nfolds"] = stat["n_folds"]
            row["mean_fg_dice_mean"] = subgroup_xf[sub]["mean_fg_dice"]["mean"]
            row["mean_fg_dice_std"]  = subgroup_xf[sub]["mean_fg_dice"]["std"]
            row["ll_emit_n_emit_total"] = subgroup_xf[sub]["ll_emit"]["sum_emit"]
            row["ll_emit_n_total"]      = subgroup_xf[sub]["ll_emit"]["sum_total"]
            w.writerow(row)


def _write_summary_md(headline_xf: Dict[str, Any],
                       subgroup_xf: Dict[str, Any],
                       confusion_xf: Dict[str, Any],
                       ensemble_headline: Optional[Dict[str, Any]],
                       n_folds: int,
                       path: Path) -> None:
    L: List[str] = []
    L.append(f"# CTSpinoPelvic1K — {n_folds}-fold cross-fold + ensemble eval")
    L.append("")
    L.append(f"Per-fold per-class means aggregated across {n_folds} folds; "
             f"reported as mean ± std (1σ across folds).")
    L.append("")

    # Headline scalars
    L.append("## Headline metrics")
    L.append("")
    L.append(f"| Metric | {n_folds}-fold mean ± std | Ensemble | n folds |")
    L.append("|---|---|---|---|")
    for k in _HEADLINE_SCALARS:
        s = headline_xf.get(k, {})
        if s.get("mean") is None:
            continue
        ens = (ensemble_headline or {}).get(k)
        ens_cell = _fmt(ens) if ens is not None else "—"
        L.append(f"| {k} | {s['mean']:.3f} ± {s['std']:.3f} | "
                 f"{ens_cell} | {s['n_folds']} |")
    L.append("")

    # Subgroup table
    L.append(f"## Per-subtype — {n_folds}-fold cross-fold mean ± std")
    L.append("")
    cls_show = ["L4", "last_lumbar", "sacrum", "hip_merged", "junction_dsc"]
    L.append("| Subtype | n_cases (per fold) | "
             + " | ".join(cls_show) + " | LL emit (sum) | mean fg |")
    L.append("|" + "|".join(["---"] * (3 + len(cls_show))) + "|")
    for sub in _KNOWN_SUBTYPES:
        agg = subgroup_xf[sub]
        n_str = "/".join(str(n) for n in agg["n_cases_per_fold"])
        cells = [PRETTY[sub], n_str]
        for cls in cls_show:
            s = agg["per_class"][cls]
            if s["mean"] is None:
                cells.append("—")
            else:
                cells.append(f"{s['mean']:.3f} ± {s['std']:.3f}")
        ll = agg["ll_emit"]
        cells.append(f"{ll['sum_emit']}/{ll['sum_total']}")
        m = agg["mean_fg_dice"]
        cells.append(f"{m['mean']:.3f} ± {m['std']:.3f}" if m["mean"] is not None else "—")
        L.append("| " + " | ".join(cells) + " |")
    L.append("")

    # Confusion blocks (summed across folds)
    L.append("## Voxel confusion blocks (summed voxel counts across folds)")
    L.append("")
    for bk, label in [
        ("block1_gt_LL_on_lumb",
         "Block 1 — GT==last_lumbar voxels on lumb cases"),
        ("block2_gt_L4_on_sacr_count",
         "Block 2 — GT==L4 voxels on sacr_count cases"),
        ("block3_pred_LL_on_sacr_count",
         "Block 3 — pred==last_lumbar voxels on sacr_count cases"),
    ]:
        blk = confusion_xf.get(bk, {})
        L.append(f"### {label}")
        L.append(f"n_cases (sum across folds) = {blk.get('n_cases', 0)}; "
                 f"voxel total = {blk.get('voxel_total', 0)}")
        per_class_key = "by_pred" if "by_pred" in blk else "by_gt"
        row_label = "pred" if per_class_key == "by_pred" else "GT"
        cells = blk.get(per_class_key, {})
        if not cells:
            L.append("(no data — subtype absent or no relevant voxels)")
            L.append("")
            continue
        L.append(f"| {row_label} class | voxels | percent |")
        L.append("|---|---|---|")
        for cname in ["background", "L1", "L2", "L3", "L4", "last_lumbar",
                       "sacrum", "left_hip", "right_hip"]:
            cell = cells.get(cname)
            if cell is None: continue
            L.append(f"| {cname} | {cell['voxels']} | {cell['percent']:.1f}% |")
        L.append("")
    spec = confusion_xf.get("last_lumbar_specificity_on_sacr_count", {})
    if spec.get("value") is not None:
        L.append(f"**last_lumbar specificity on sacr_count (across folds)** = "
                 f"{spec['value']:.4f}  (n_cases = "
                 f"{spec.get('n_sacr_count_cases_total', '?')})")
        L.append("")

    L.append("Per-fold raw values in `cross_fold_summary.csv` (column "
             "`<class>_nfolds` shows how many folds contributed); "
             "per-fold confusion sums in `cross_fold_confusion.json`.")
    path.write_text("\n".join(L) + "\n")


# ===========================================================================
# Main
# ===========================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold_dirs", nargs="+", required=True, type=Path,
                     help="One eval output dir per fold (must each contain "
                          "headline.json, confusion_blocks.json, "
                          "subgroup_summary.csv).")
    ap.add_argument("--ensemble_dir", type=Path, default=None,
                     help="Optional ensemble eval dir (for side-by-side "
                          "headline comparison).")
    ap.add_argument("--output_dir", required=True, type=Path)
    args = ap.parse_args()

    print("=" * 72)
    print(f" CROSS-FOLD AGGREGATION  (n_folds = {len(args.fold_dirs)})")
    print("=" * 72)
    for d in args.fold_dirs:
        print(f"   fold dir : {d}")
    if args.ensemble_dir:
        print(f"   ensemble : {args.ensemble_dir}")
    print(f"   output   : {args.output_dir}")
    print("=" * 72)

    folds = []
    for d in args.fold_dirs:
        if not d.is_dir():
            print(f"ERROR: fold dir missing: {d}", file=sys.stderr)
            return 2
        folds.append(_load_fold(d))

    ensemble_headline = None
    if args.ensemble_dir and (args.ensemble_dir / "headline.json").exists():
        ensemble_headline = json.loads(
            (args.ensemble_dir / "headline.json").read_text())

    headline_xf = aggregate_headline(folds)
    subgroup_xf = aggregate_subgroup(folds)
    confusion_xf = aggregate_confusion(folds)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_headline_json(headline_xf, ensemble_headline,
                          args.output_dir / "cross_fold_headline.json")
    _write_subgroup_csv(subgroup_xf,
                          args.output_dir / "cross_fold_summary.csv")
    (args.output_dir / "cross_fold_confusion.json").write_text(
        json.dumps(confusion_xf, indent=2))
    _write_summary_md(headline_xf, subgroup_xf, confusion_xf,
                       ensemble_headline, len(folds),
                       args.output_dir / "cross_fold_summary.md")

    print()
    print("=" * 72)
    print(" CROSS-FOLD AGGREGATION COMPLETE")
    print("=" * 72)
    for k in ("junction_dsc_lumb", "junction_dsc_overall",
              "sacrum_dice_overall", "last_lumbar_dice_on_lumb",
              "L4_dice_on_sacr_count",
              "last_lumbar_specificity_on_sacr_count"):
        s = headline_xf.get(k, {})
        if s.get("mean") is None: continue
        ens = (ensemble_headline or {}).get(k)
        ens_str = f", ensemble={_fmt(ens)}" if ens is not None else ""
        print(f"  {k:>40s}: {s['mean']:.3f} ± {s['std']:.3f} "
              f"(n_folds={s['n_folds']}){ens_str}")
    print("=" * 72)
    print(f"  cross_fold_summary.md   -> {args.output_dir/'cross_fold_summary.md'}")
    print(f"  cross_fold_summary.csv  -> {args.output_dir/'cross_fold_summary.csv'}")
    print(f"  cross_fold_headline.json-> {args.output_dir/'cross_fold_headline.json'}")
    print(f"  cross_fold_confusion.json-> {args.output_dir/'cross_fold_confusion.json'}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
