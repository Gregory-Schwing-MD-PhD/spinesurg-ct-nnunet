#!/usr/bin/env python3
"""
tools/eval_ensemble_test.py — Test-set eval for SpineSurg-CT Dataset803.

Per-case Dice using the trainer's _per_case_dice_per_class convention
(ignore_label=9 masked). Aggregates by 6-way LSTV subtype from
lstv_cases.json.

Hip-merged class (added May 2026)
---------------------------------
The CTPelvic1K source annotations have observed L/R hip label noise
(asymmetric per-side dice that can't be explained by FOV truncation
or model failure modes). To characterize hip segmentation quality
independent of laterality assignment, we additionally compute
`dice_hip_merged` — Dice computed with both label 7 (left_hip) and
label 8 (right_hip) treated as a single "hip" class on both prediction
and GT. If the per-side problem is laterality noise, hip_merged should
be substantially higher than either per-side metric (because pred and
GT agree on "this is hip bone" even when they disagree on which hip).

Outputs (under --output_dir)
---------------------------
  per_case_dice.csv        one row per case (incl. dice_hip_merged)
  subgroup_summary.csv     6-way subtype × class mean ± std
  subgroup_summary.md      paper-ready markdown
  headline.json            machine-readable headline numbers

Author: Gregory Schwing, MD-PhD  |  Wayne State University / DMC
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Pin BLAS / OpenMP / MKL BEFORE importing numpy.
os.environ.setdefault("OMP_NUM_THREADS",         "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS",    "1")
os.environ.setdefault("MKL_NUM_THREADS",         "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS",  "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS",     "1")

import numpy as np

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

# ── Anatomy constants — MUST match nnunet_wandb_variant.py exactly ─────────
_ANATOMY_NAMES   = ["L1", "L2", "L3", "L4", "last_lumbar",
                     "sacrum", "left_hip", "right_hip"]
_FG_CLASS_IDS    = [1, 2, 3, 4, 5, 6, 7, 8]
_LUMBAR_IDS      = [1, 2, 3, 4, 5]
_PELVIS_IDS      = [6, 7, 8]
_IGNORE_LABEL    = 9

_LEFT_HIP_LABEL_ID  = 7
_RIGHT_HIP_LABEL_ID = 8

_SUBTYPE_NORMAL     = "normal"
_SUBTYPE_LUMB       = "lumb"
_SUBTYPE_SACR_COUNT = "sacr_count"
_SUBTYPE_SEMI       = "semisacralization"
_SUBTYPE_SACR       = "sacralization"
_SUBTYPE_AMBIG      = "ambiguous"
_KNOWN_SUBTYPES = (
    _SUBTYPE_NORMAL, _SUBTYPE_LUMB, _SUBTYPE_SACR_COUNT,
    _SUBTYPE_SEMI, _SUBTYPE_SACR, _SUBTYPE_AMBIG,
)
_LSTV_SUBTYPES = tuple(s for s in _KNOWN_SUBTYPES if s != _SUBTYPE_NORMAL)

_LAST_LUMBAR_LABEL_ID = 5
_L4_LABEL_ID          = 4

PRETTY = {
    _SUBTYPE_NORMAL:     "Normal",
    _SUBTYPE_LUMB:       "Lumbarization",
    _SUBTYPE_SACR_COUNT: "Sacralization (count)",
    _SUBTYPE_SEMI:       "Semisacralization",
    _SUBTYPE_SACR:       "Sacralization (morph)",
    _SUBTYPE_AMBIG:      "Ambiguous",
}


def _per_case_dice(pred: np.ndarray, gt: np.ndarray
                    ) -> Tuple[List[Optional[float]], Optional[float], int]:
    """Per-fg-class Dice + merged-hip Dice; ignore voxels masked out."""
    pred = pred.ravel().astype(np.int16)
    gt   = gt.ravel().astype(np.int16)
    valid = gt != _IGNORE_LABEL
    n_valid = int(valid.sum())
    if n_valid == 0:
        return [None] * len(_FG_CLASS_IDS), None, 0
    pred = pred[valid]
    gt   = gt[valid]

    # Per-class
    out: List[Optional[float]] = []
    for cid in _FG_CLASS_IDS:
        gt_mask = gt == cid
        gt_n = int(gt_mask.sum())
        if gt_n == 0:
            out.append(None)
            continue
        pred_mask = pred == cid
        pred_n = int(pred_mask.sum())
        inter  = int((gt_mask & pred_mask).sum())
        denom  = gt_n + pred_n
        out.append(2.0 * inter / denom if denom > 0 else None)

    # Merged hip: treat labels 7 and 8 as a single class on BOTH sides.
    # If per-side dice is bad due to L/R label noise in source annotations,
    # this metric quantifies hip seg quality independent of laterality.
    gt_hip   = (gt   == _LEFT_HIP_LABEL_ID) | (gt   == _RIGHT_HIP_LABEL_ID)
    pred_hip = (pred == _LEFT_HIP_LABEL_ID) | (pred == _RIGHT_HIP_LABEL_ID)
    gt_hip_n = int(gt_hip.sum())
    if gt_hip_n == 0:
        dice_hip_merged: Optional[float] = None
    else:
        pred_hip_n = int(pred_hip.sum())
        inter_hip  = int((gt_hip & pred_hip).sum())
        denom_hip  = gt_hip_n + pred_hip_n
        dice_hip_merged = (2.0 * inter_hip / denom_hip) if denom_hip > 0 else None

    return out, dice_hip_merged, n_valid


def _eval_one_case(args: dict) -> dict:
    import nibabel as nib
    case_id   = args["case_id"]
    pred_path = args["pred_path"]
    gt_path   = args["gt_path"]
    t0 = time.time()
    try:
        pred = np.asarray(nib.load(pred_path).dataobj, dtype=np.int16)
        gt   = np.asarray(nib.load(gt_path).dataobj,   dtype=np.int16)
        if pred.shape != gt.shape:
            return {"case_id": case_id, "ok": False,
                    "error": f"shape mismatch pred={pred.shape} gt={gt.shape}",
                    "elapsed_sec": time.time() - t0}
        dices, dice_hip_merged, n_valid = _per_case_dice(pred, gt)
        out = {"case_id": case_id, "ok": True,
                "n_valid_voxels": n_valid,
                "shape": str(pred.shape),
                "elapsed_sec": round(time.time() - t0, 2),
                "dice_hip_merged": dice_hip_merged}
        for name, d in zip(_ANATOMY_NAMES, dices):
            out[f"dice_{name}"] = d
        out["dice_l1"] = out["dice_L1"]
        out["dice_l2"] = out["dice_L2"]
        out["dice_l3"] = out["dice_L3"]
        out["dice_l4"] = out["dice_L4"]
        out["dice_l5"] = out["dice_last_lumbar"]
        out["dice_l6"] = None
        lumbar = [d for d in dices[:5] if d is not None]
        pelvis = [d for d in dices[5:] if d is not None]
        all_fg = [d for d in dices    if d is not None]
        out["mean_lumbar"]  = float(np.mean(lumbar)) if lumbar else None
        out["mean_pelvis"]  = float(np.mean(pelvis)) if pelvis else None
        out["dice_overall"] = float(np.mean(all_fg)) if all_fg else None
        return out
    except Exception as exc:
        return {"case_id": case_id, "ok": False, "error": str(exc),
                "elapsed_sec": time.time() - t0}


def _load_subtype_map(lstv_cases_json: Path) -> Dict[str, str]:
    if not lstv_cases_json.exists():
        print(f"WARN: {lstv_cases_json} not found; all cases will be 'normal'.",
              file=sys.stderr)
        return {}
    try:
        data = json.loads(lstv_cases_json.read_text())
    except Exception as exc:
        print(f"WARN: cannot parse {lstv_cases_json}: {exc}", file=sys.stderr)
        return {}
    cts = data.get("case_to_subtype") or {}
    return {str(k): str(v) for k, v in cts.items()}


def _build_work(predictions_dir: Path, labels_dir: Path) -> List[dict]:
    work = []
    pred_files = sorted(predictions_dir.glob("*.nii.gz"))
    if not pred_files:
        print(f"ERROR: no .nii.gz files in {predictions_dir}", file=sys.stderr)
        sys.exit(2)
    n_skipped = 0
    for pf in pred_files:
        case_id = pf.name[:-len(".nii.gz")]
        gt_path = labels_dir / f"{case_id}.nii.gz"
        if not gt_path.exists():
            n_skipped += 1
            continue
        work.append({
            "case_id":   case_id,
            "pred_path": str(pf),
            "gt_path":   str(gt_path),
        })
    if n_skipped:
        print(f"WARN: {n_skipped} predictions had no matching label "
              f"in {labels_dir}", file=sys.stderr)
    return work


def _aggregate(rows: List[dict], case_to_subtype: Dict[str, str]
                ) -> Tuple[Dict[str, Dict], Dict[str, float]]:
    by_subtype: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        if not r.get("ok"):
            continue
        sub = case_to_subtype.get(r["case_id"], _SUBTYPE_NORMAL)
        if sub not in _KNOWN_SUBTYPES:
            sub = _SUBTYPE_NORMAL
        by_subtype[sub].append(r)

    summary: Dict[str, Dict] = {}
    for sub in _KNOWN_SUBTYPES:
        cases = by_subtype.get(sub, [])
        n = len(cases)
        per_class: Dict[str, Dict] = {}
        for name in _ANATOMY_NAMES:
            vals = [c[f"dice_{name}"] for c in cases
                    if c.get(f"dice_{name}") is not None]
            if vals:
                per_class[name] = {"n": len(vals),
                                    "mean": float(np.mean(vals)),
                                    "std":  float(np.std(vals, ddof=1))
                                              if len(vals) > 1 else 0.0}
            else:
                per_class[name] = {"n": 0, "mean": None, "std": None}
        # Merged-hip aggregate (one extra "class" outside the per-label loop).
        hip_merged_vals = [c["dice_hip_merged"] for c in cases
                            if c.get("dice_hip_merged") is not None]
        if hip_merged_vals:
            per_class["hip_merged"] = {
                "n":    len(hip_merged_vals),
                "mean": float(np.mean(hip_merged_vals)),
                "std":  float(np.std(hip_merged_vals, ddof=1))
                          if len(hip_merged_vals) > 1 else 0.0,
            }
        else:
            per_class["hip_merged"] = {"n": 0, "mean": None, "std": None}

        lumbar_means = [per_class[n]["mean"] for n in _ANATOMY_NAMES[:5]
                        if per_class[n]["mean"] is not None]
        pelvis_means = [per_class[n]["mean"] for n in _ANATOMY_NAMES[5:]
                        if per_class[n]["mean"] is not None]
        fg_means_per_case = [c["dice_overall"] for c in cases
                             if c.get("dice_overall") is not None]
        summary[sub] = {
            "n_cases":          n,
            "per_class":        per_class,
            "mean_lumbar_dice": float(np.mean(lumbar_means)) if lumbar_means else None,
            "mean_pelvis_dice": float(np.mean(pelvis_means)) if pelvis_means else None,
            "mean_fg_dice":     float(np.mean(fg_means_per_case))
                                  if fg_means_per_case else None,
            "fg_dice_std":      float(np.std(fg_means_per_case, ddof=1))
                                  if len(fg_means_per_case) > 1 else 0.0,
        }

    headline: Dict[str, float] = {}
    all_ok = [r for r in rows if r.get("ok")]
    overall_fg = [r["dice_overall"] for r in all_ok
                  if r.get("dice_overall") is not None]
    headline["n_test_cases"]         = float(len(all_ok))
    headline["mean_fg_dice_overall"] = (float(np.mean(overall_fg))
                                         if overall_fg else None)

    # Headline: hip-merged across ALL cases (not subtype-specific).
    hip_merged_all = [r["dice_hip_merged"] for r in all_ok
                      if r.get("dice_hip_merged") is not None]
    if hip_merged_all:
        headline["hip_merged_dice_overall"]   = float(np.mean(hip_merged_all))
        headline["hip_merged_n_cases"]        = float(len(hip_merged_all))
        # Also expose mean per-side for direct comparison in one place.
        l_all = [r["dice_left_hip"]  for r in all_ok if r.get("dice_left_hip")  is not None]
        r_all = [r["dice_right_hip"] for r in all_ok if r.get("dice_right_hip") is not None]
        if l_all: headline["left_hip_dice_overall"]  = float(np.mean(l_all))
        if r_all: headline["right_hip_dice_overall"] = float(np.mean(r_all))

    ll_on_lumb = [c["dice_last_lumbar"] for c in by_subtype.get(_SUBTYPE_LUMB, [])
                  if c.get("dice_last_lumbar") is not None]
    if ll_on_lumb:
        headline["last_lumbar_dice_on_lumb"] = float(np.mean(ll_on_lumb))
        headline["last_lumbar_n_lumb_cases"] = float(len(ll_on_lumb))

    l4_on_sc = [c["dice_L4"] for c in by_subtype.get(_SUBTYPE_SACR_COUNT, [])
                if c.get("dice_L4") is not None]
    if l4_on_sc:
        headline["L4_dice_on_sacr_count"] = float(np.mean(l4_on_sc))
        headline["L4_n_sacr_count_cases"] = float(len(l4_on_sc))

    normal_fg = [c["dice_overall"] for c in by_subtype.get(_SUBTYPE_NORMAL, [])
                 if c.get("dice_overall") is not None]
    any_lstv_fg = []
    for sub in _LSTV_SUBTYPES:
        any_lstv_fg.extend(c["dice_overall"] for c in by_subtype.get(sub, [])
                            if c.get("dice_overall") is not None)
    if normal_fg:
        headline["normal_mean_fg_dice"] = float(np.mean(normal_fg))
        headline["normal_n_cases"]      = float(len(normal_fg))
    if any_lstv_fg:
        headline["any_lstv_mean_fg_dice"] = float(np.mean(any_lstv_fg))
        headline["any_lstv_n_cases"]      = float(len(any_lstv_fg))
    if normal_fg and any_lstv_fg:
        headline["any_lstv_minus_normal_fg_dice"] = (
            float(np.mean(any_lstv_fg)) - float(np.mean(normal_fg))
        )

    return summary, headline


def _write_per_case_csv(rows: List[dict], path: Path,
                         case_to_subtype: Dict[str, str]) -> None:
    cols = (
        ["case_id", "subtype", "ok", "error",
         "n_valid_voxels", "shape", "elapsed_sec"]
        + [f"dice_{n}" for n in _ANATOMY_NAMES]
        + ["dice_hip_merged",
           "mean_lumbar", "mean_pelvis", "dice_overall",
           "dice_l1", "dice_l2", "dice_l3", "dice_l4",
           "dice_l5", "dice_l6"]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            sub = case_to_subtype.get(r["case_id"], _SUBTYPE_NORMAL)
            w.writerow({**r, "subtype": sub})


def _write_subgroup_csv(summary: Dict[str, Dict], path: Path) -> None:
    extra_classes = ["hip_merged"]
    all_classes = list(_ANATOMY_NAMES) + extra_classes
    cols = (["subtype", "n_cases"]
            + [f"{n}_mean" for n in all_classes]
            + [f"{n}_std"  for n in all_classes]
            + [f"{n}_n"    for n in all_classes]
            + ["mean_lumbar_dice", "mean_pelvis_dice",
                "mean_fg_dice", "fg_dice_std"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for sub in _KNOWN_SUBTYPES:
            s = summary[sub]
            row = {"subtype": sub, "n_cases": s["n_cases"]}
            for name in all_classes:
                pc = s["per_class"].get(name, {"n": 0, "mean": None, "std": None})
                row[f"{name}_mean"] = pc["mean"]
                row[f"{name}_std"]  = pc["std"]
                row[f"{name}_n"]    = pc["n"]
            row["mean_lumbar_dice"] = s["mean_lumbar_dice"]
            row["mean_pelvis_dice"] = s["mean_pelvis_dice"]
            row["mean_fg_dice"]     = s["mean_fg_dice"]
            row["fg_dice_std"]      = s["fg_dice_std"]
            w.writerow(row)


def _fmt(v: Optional[float], fmt: str = "{:.3f}") -> str:
    return fmt.format(v) if v is not None else "—"


def _write_subgroup_md(summary: Dict[str, Dict], headline: Dict[str, float],
                        path: Path) -> None:
    lines: List[str] = []
    lines.append("# CTSpinoPelvic1K — Test-set ensemble eval (Dataset803, merged)")
    lines.append("")
    n_total = int(headline.get("n_test_cases", 0))
    fg_overall = headline.get("mean_fg_dice_overall")
    lines.append(f"**Test cases evaluated:** {n_total}")
    lines.append(f"**Overall mean foreground Dice:** {_fmt(fg_overall)}")
    lines.append("")
    lines.append("## Headline metrics")
    lines.append("")
    lines.append("| Metric | Value | n |")
    lines.append("|---|---|---|")
    if "last_lumbar_dice_on_lumb" in headline:
        lines.append(f"| last_lumbar Dice on lumb cases | "
                     f"{_fmt(headline['last_lumbar_dice_on_lumb'])} | "
                     f"{int(headline['last_lumbar_n_lumb_cases'])} |")
    if "L4_dice_on_sacr_count" in headline:
        lines.append(f"| L4 Dice on sacr_count cases | "
                     f"{_fmt(headline['L4_dice_on_sacr_count'])} | "
                     f"{int(headline['L4_n_sacr_count_cases'])} |")
    if "normal_mean_fg_dice" in headline:
        lines.append(f"| Normal mean fg Dice | "
                     f"{_fmt(headline['normal_mean_fg_dice'])} | "
                     f"{int(headline['normal_n_cases'])} |")
    if "any_lstv_mean_fg_dice" in headline:
        lines.append(f"| Any-LSTV mean fg Dice | "
                     f"{_fmt(headline['any_lstv_mean_fg_dice'])} | "
                     f"{int(headline['any_lstv_n_cases'])} |")
    if "any_lstv_minus_normal_fg_dice" in headline:
        delta = headline["any_lstv_minus_normal_fg_dice"]
        sign = "+" if delta >= 0 else ""
        lines.append(f"| **Any-LSTV − Normal (fg Dice)** | "
                     f"**{sign}{delta:.3f}** | — |")
    lines.append("")
    if "hip_merged_dice_overall" in headline:
        lines.append("## Hip class — per-side vs merged")
        lines.append("")
        lines.append("Asymmetric per-side Dice with high right-hip variance "
                      "is consistent with L/R label noise in CTPelvic1K source "
                      "annotations. The merged hip class collapses labels 7 + 8 "
                      "into a single 'hip' class on both prediction and GT, "
                      "characterizing hip segmentation quality independent of "
                      "laterality assignment.")
        lines.append("")
        lines.append("| Metric | Value | n |")
        lines.append("|---|---|---|")
        if "left_hip_dice_overall" in headline:
            lines.append(f"| left_hip Dice  | {_fmt(headline['left_hip_dice_overall'])} | "
                         f"{int(headline.get('hip_merged_n_cases', 0))} |")
        if "right_hip_dice_overall" in headline:
            lines.append(f"| right_hip Dice | {_fmt(headline['right_hip_dice_overall'])} | "
                         f"{int(headline.get('hip_merged_n_cases', 0))} |")
        lines.append(f"| **hip_merged Dice (L+R as one class)** | "
                     f"**{_fmt(headline['hip_merged_dice_overall'])}** | "
                     f"{int(headline['hip_merged_n_cases'])} |")
        lines.append("")
    lines.append("## Per-subtype Dice (mean ± SD)")
    lines.append("")
    cls_show = ["L4", "last_lumbar", "sacrum",
                 "left_hip", "right_hip", "hip_merged"]
    header = (["Subtype", "n"] + cls_show + ["mean fg"])
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for sub in _KNOWN_SUBTYPES:
        s = summary[sub]
        n = s["n_cases"]
        cells: List[str] = [PRETTY[sub], str(n)]
        for name in cls_show:
            pc = s["per_class"].get(name, {"mean": None, "std": None})
            if pc["mean"] is None:
                cells.append("—")
            else:
                cells.append(f"{pc['mean']:.3f} ± {pc['std']:.3f}")
        cells.append(_fmt(s["mean_fg_dice"]))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Full per-class breakdown in `subgroup_summary.csv`; "
                  "per-case dices in `per_case_dice.csv`.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions_dir", required=True, type=Path)
    ap.add_argument("--labels_dir",      required=True, type=Path)
    ap.add_argument("--lstv_cases_json", required=True, type=Path)
    ap.add_argument("--output_dir",      required=True, type=Path)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not args.predictions_dir.is_dir():
        print(f"ERROR: predictions_dir not found: {args.predictions_dir}", file=sys.stderr)
        return 2
    if not args.labels_dir.is_dir():
        print(f"ERROR: labels_dir not found: {args.labels_dir}", file=sys.stderr)
        return 2

    print(f"=" * 72)
    print(f"SPINESURG-CT TEST EVAL — DATASET803 MERGED (+ hip_merged)")
    print(f"=" * 72)
    print(f"  predictions_dir : {args.predictions_dir}")
    print(f"  labels_dir      : {args.labels_dir}")
    print(f"  lstv_cases_json : {args.lstv_cases_json}")
    print(f"  output_dir      : {args.output_dir}")
    print(f"  workers         : {args.workers}")
    print(f"  tqdm available  : {_HAS_TQDM}")
    print(f"=" * 72)

    print(f"\nLoading subtype map from {args.lstv_cases_json}")
    case_to_subtype = _load_subtype_map(args.lstv_cases_json)
    print(f"  {len(case_to_subtype)} cases mapped")
    if case_to_subtype:
        sub_counts = Counter(case_to_subtype.values())
        print(f"  subtype distribution (full lstv_cases.json): {dict(sub_counts)}")

    print(f"\nBuilding work from {args.predictions_dir}")
    work = _build_work(args.predictions_dir, args.labels_dir)
    print(f"  {len(work)} (prediction, label) pairs to evaluate")

    rows: List[dict] = []
    n_ok = n_fail = 0
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_eval_one_case, w): w["case_id"] for w in work}
        if _HAS_TQDM:
            iterator = tqdm(as_completed(futs), total=len(futs),
                             desc="dice eval", unit="case",
                             dynamic_ncols=True, mininterval=0.5)
        else:
            iterator = as_completed(futs)
        for i, fut in enumerate(iterator, 1):
            cid = futs[fut]
            try:
                r = fut.result()
            except Exception as exc:
                r = {"case_id": cid, "ok": False, "error": str(exc),
                     "elapsed_sec": None}
            rows.append(r)
            if r.get("ok"):
                n_ok += 1
                if args.verbose:
                    msg = (f"  [{i:>3}/{len(work)}] OK   {cid:<60s} "
                            f"shape={r.get('shape','?')} "
                            f"fg={_fmt(r.get('dice_overall'))} "
                            f"hip_m={_fmt(r.get('dice_hip_merged'))} "
                            f"({r.get('elapsed_sec','?')}s)")
                    if _HAS_TQDM:
                        tqdm.write(msg)
                    else:
                        print(msg, flush=True)
            else:
                n_fail += 1
                msg = (f"  [{i:>3}/{len(work)}] FAIL {cid}: "
                        f"{r.get('error', '?')}")
                if _HAS_TQDM:
                    tqdm.write(msg)
                else:
                    print(msg, file=sys.stderr, flush=True)
            if not _HAS_TQDM and (i % 10 == 0 or i == len(work)):
                elapsed = time.time() - t_start
                rate = i / max(elapsed, 1e-9)
                eta_sec = (len(work) - i) / max(rate, 1e-9)
                print(f"  [progress] {i}/{len(work)}  "
                      f"ok={n_ok} fail={n_fail}  "
                      f"{rate:.1f} case/s  ETA {eta_sec:.0f}s",
                      flush=True)

    if n_ok == 0:
        print("ERROR: zero successful evaluations.", file=sys.stderr)
        return 3

    summary, headline = _aggregate(rows, case_to_subtype)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_per_case_csv(rows, args.output_dir / "per_case_dice.csv",
                         case_to_subtype)
    _write_subgroup_csv(summary, args.output_dir / "subgroup_summary.csv")
    _write_subgroup_md(summary, headline,
                         args.output_dir / "subgroup_summary.md")
    (args.output_dir / "headline.json").write_text(
        json.dumps(headline, indent=2))

    elapsed_total = time.time() - t_start
    print("\n" + "=" * 72)
    print(" EVAL COMPLETE")
    print("=" * 72)
    print(f"  cases ok / failed       : {n_ok} / {n_fail}")
    print(f"  total elapsed           : {elapsed_total:.1f}s "
          f"({n_ok/max(elapsed_total,1e-9):.1f} case/s with {args.workers} workers)")
    print(f"  overall mean fg Dice    : "
          f"{_fmt(headline.get('mean_fg_dice_overall'))}")
    if "left_hip_dice_overall" in headline:
        print(f"  left_hip Dice           : "
              f"{_fmt(headline['left_hip_dice_overall'])}")
    if "right_hip_dice_overall" in headline:
        print(f"  right_hip Dice          : "
              f"{_fmt(headline['right_hip_dice_overall'])}")
    if "hip_merged_dice_overall" in headline:
        print(f"  hip_merged Dice (L+R)   : "
              f"{_fmt(headline['hip_merged_dice_overall'])}  "
              f"(n={int(headline.get('hip_merged_n_cases',0))})")
    if "last_lumbar_dice_on_lumb" in headline:
        print(f"  last_lumbar on lumb     : "
              f"{_fmt(headline['last_lumbar_dice_on_lumb'])} "
              f"(n={int(headline['last_lumbar_n_lumb_cases'])})")
    if "L4_dice_on_sacr_count" in headline:
        print(f"  L4 on sacr_count        : "
              f"{_fmt(headline['L4_dice_on_sacr_count'])} "
              f"(n={int(headline['L4_n_sacr_count_cases'])})")
    if "any_lstv_minus_normal_fg_dice" in headline:
        delta = headline["any_lstv_minus_normal_fg_dice"]
        sign = "+" if delta >= 0 else ""
        print(f"  Any-LSTV − Normal       : {sign}{delta:.3f}  "
              f"(Normal n={int(headline.get('normal_n_cases', 0))}, "
              f"Any-LSTV n={int(headline.get('any_lstv_n_cases', 0))})")
    print("=" * 72)
    print(f"  per_case_dice.csv       -> {args.output_dir/'per_case_dice.csv'}")
    print(f"  subgroup_summary.csv    -> {args.output_dir/'subgroup_summary.csv'}")
    print(f"  subgroup_summary.md     -> {args.output_dir/'subgroup_summary.md'}")
    print(f"  headline.json           -> {args.output_dir/'headline.json'}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
