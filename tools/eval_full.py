#!/usr/bin/env python3
"""
tools/eval_full.py — Full test-set eval for SpineSurg-CT Dataset803.

Strict superset of eval_ensemble_test.py. Produces the same outputs PLUS:
  * Junction-DSC over a 40 mm L5/S1 axial window per case (mean over
    {L4, last_lumbar, sacrum} classes that have GT voxels in the window)
  * Per-class voxel confusion blocks aggregated per subtype:
      Block 1 — GT==last_lumbar voxels on lumb cases
                -> distribution across pred classes (paper §residual)
      Block 2 — GT==L4 voxels on sacr_count cases
                -> distribution across pred classes
      Block 3 — pred==last_lumbar voxels on sacr_count cases
                -> distribution across GT classes
  * last_lumbar specificity on sacr_count
      = TN / (TN + FP) micro-aggregated where positive = (label == 5)
  * LL emit per subtype: # cases with >= 100 last_lumbar voxels in pred

Outputs (under --output_dir)
  per_case_dice.csv          (drop-in replacement for old script;
                              extra columns: junction_dsc, ll_emit_voxels)
  subgroup_summary.csv       per-subtype dice + junction-DSC means/std
  subgroup_summary.md        paper-ready markdown
  headline.json              machine-readable headline numbers
  confusion_blocks.json      Block 1/2/3 percentages + raw voxel counts
                              + last_lumbar specificity on sacr_count
                              + LL emit counts per subtype

Junction-DSC definition
-----------------------
Per case:
  1. Find the inferior-superior (IS) axis from the GT volume's affine
     (uses nib.aff2axcodes; works for RAS, LPI, LAS, etc.)
  2. Project GT==last_lumbar and GT==sacrum onto the IS axis;
     junction_idx = mean(IS coords of last_lumbar) + mean(IS coords of
     sacrum) divided by 2.
  3. Window = junction_idx ± (20 mm / IS spacing) along that axis.
  4. Slab pred and GT to that window.
  5. Compute Dice for classes {L4=4, last_lumbar=5, sacrum=6} restricted
     to slab voxels (with ignore_label=9 masked); junction-DSC = mean
     of those classes that have >0 GT voxels in the slab.
Skipped (returns None) if GT lacks last_lumbar OR sacrum voxels.

Per-subtype junction-DSC = mean of per-case junction-DSC.

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
from typing import Dict, List, Optional, Tuple, Any

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
_BG_LABEL_ID     = 0
_L4_LABEL_ID            = 4
_LAST_LUMBAR_LABEL_ID   = 5
_SACRUM_LABEL_ID        = 6
_LEFT_HIP_LABEL_ID      = 7
_RIGHT_HIP_LABEL_ID     = 8

_JUNCTION_CLASSES       = [_L4_LABEL_ID, _LAST_LUMBAR_LABEL_ID, _SACRUM_LABEL_ID]
_JUNCTION_WINDOW_MM     = 40.0          # full window (±20 mm)
_LL_EMIT_THRESHOLD_VOX  = 100

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

# Confusion-block class universe (paper Table 5):
# pred / GT can be: bg, L1, L2, L3, L4, last_lumbar, sacrum, left_hip,
# right_hip, ignore. We tabulate against the FG classes + bg explicitly.
_CONFUSION_CLASS_IDS = [_BG_LABEL_ID] + list(_FG_CLASS_IDS)
_CONFUSION_CLASS_NAMES = {
    0: "background", 1: "L1", 2: "L2", 3: "L3", 4: "L4",
    5: "last_lumbar", 6: "sacrum", 7: "left_hip", 8: "right_hip",
}

PRETTY = {
    _SUBTYPE_NORMAL:     "Normal",
    _SUBTYPE_LUMB:       "Lumbarization",
    _SUBTYPE_SACR_COUNT: "Sacralization (count)",
    _SUBTYPE_SEMI:       "Semisacralization",
    _SUBTYPE_SACR:       "Sacralization (morph)",
    _SUBTYPE_AMBIG:      "Ambiguous",
}


# ===========================================================================
# Per-case computations
# ===========================================================================

def _per_case_dice(pred: np.ndarray, gt: np.ndarray
                    ) -> Tuple[List[Optional[float]], Optional[float], int]:
    """Per-fg-class Dice + merged-hip Dice; ignore voxels masked out.

    Identical to eval_ensemble_test.py for compatibility.
    """
    pred = pred.ravel().astype(np.int16)
    gt   = gt.ravel().astype(np.int16)
    valid = gt != _IGNORE_LABEL
    n_valid = int(valid.sum())
    if n_valid == 0:
        return [None] * len(_FG_CLASS_IDS), None, 0
    pred = pred[valid]
    gt   = gt[valid]

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


def _identify_is_axis(affine: np.ndarray) -> Tuple[int, float]:
    """Return (axis_index, mm_per_voxel_along_IS) for the GT volume.

    Uses nib.aff2axcodes to find the axis closest to inferior-superior;
    spacing along that axis is the L2-norm of the corresponding affine
    column (scanner mm per voxel along that axis).
    """
    import nibabel as nib
    axcodes = nib.aff2axcodes(affine)        # e.g. ('R', 'A', 'S')
    is_axis = next((i for i, c in enumerate(axcodes) if c in ("I", "S")), 2)
    spacing = float(np.linalg.norm(affine[:3, is_axis]))
    return is_axis, spacing


def _compute_junction_dsc(pred: np.ndarray,
                            gt: np.ndarray,
                            affine: np.ndarray) -> Optional[float]:
    """Junction-DSC over a 40 mm L5/S1 axial window.

    See module docstring for full definition. Returns None if the window
    cannot be defined (no GT last_lumbar OR no GT sacrum).
    """
    is_axis, spacing = _identify_is_axis(affine)
    if spacing <= 0:
        return None

    gt_ll  = gt == _LAST_LUMBAR_LABEL_ID
    gt_sac = gt == _SACRUM_LABEL_ID
    if not gt_ll.any() or not gt_sac.any():
        return None

    # IS-axis indices where these classes have any voxel.
    other_axes = tuple(i for i in range(3) if i != is_axis)
    ll_along  = gt_ll.any(axis=other_axes)
    sac_along = gt_sac.any(axis=other_axes)
    ll_idx_mean  = float(np.where(ll_along)[0].mean())
    sac_idx_mean = float(np.where(sac_along)[0].mean())
    junction_idx = (ll_idx_mean + sac_idx_mean) / 2.0

    half_win_voxels = max(1, int(round((_JUNCTION_WINDOW_MM / 2.0) / spacing)))
    lo = max(0, int(round(junction_idx - half_win_voxels)))
    hi = min(gt.shape[is_axis], int(round(junction_idx + half_win_voxels)) + 1)
    if hi <= lo:
        return None

    # Slab the volumes along the IS axis.
    sl = [slice(None)] * 3
    sl[is_axis] = slice(lo, hi)
    sl = tuple(sl)
    pred_w = pred[sl].ravel().astype(np.int16)
    gt_w   = gt[sl].ravel().astype(np.int16)

    valid = gt_w != _IGNORE_LABEL
    if not valid.any():
        return None
    pred_w = pred_w[valid]
    gt_w   = gt_w[valid]

    dices: List[float] = []
    for cid in _JUNCTION_CLASSES:
        gt_mask = gt_w == cid
        gt_n = int(gt_mask.sum())
        if gt_n == 0:
            continue
        pred_mask = pred_w == cid
        pred_n = int(pred_mask.sum())
        inter  = int((gt_mask & pred_mask).sum())
        denom  = gt_n + pred_n
        if denom > 0:
            dices.append(2.0 * inter / denom)

    return float(np.mean(dices)) if dices else None


def _compute_voxel_confusion(pred: np.ndarray,
                              gt: np.ndarray) -> Dict[Tuple[int, int], int]:
    """Compute (gt_class, pred_class) -> voxel count for confusion blocks.

    Restricted to the class pairs the paper's Table 5 cares about plus the
    last_lumbar specificity computation. Ignore-label voxels are excluded.
    Keys we always populate (when gt has matching voxels):
      Block 1 / Block 2 setup:
        (gt_class == 4, pred ∈ {0, 4, 5, 6})         ; (gt_class == 5, ...)
      Block 3 setup (pred-conditioned):
        (gt ∈ {0, 4, 5, 6}, pred_class == 5)
      Specificity setup:
        ("gt_not_5_total", "n/a") and ("gt_not_5_pred_eq_5", "n/a")
    For convenience, we just emit the full set of (gt, pred) pairs over
    the FG classes + bg + the totals; aggregation in _aggregate_full
    pulls out the blocks it wants.
    """
    pred = pred.ravel().astype(np.int16)
    gt   = gt.ravel().astype(np.int16)
    valid = gt != _IGNORE_LABEL
    pred = pred[valid]
    gt   = gt[valid]

    out: Dict[Tuple[int, int], int] = {}
    # Full (gt, pred) histogram over the relevant universe.
    # Cheap because we use boolean masking, not a full 9x9 confusion call.
    for g in _CONFUSION_CLASS_IDS:
        g_mask = gt == g
        g_n = int(g_mask.sum())
        out[("gt_total", g)] = g_n
        if g_n == 0:
            continue
        for p in _CONFUSION_CLASS_IDS:
            inter = int((g_mask & (pred == p)).sum())
            if inter > 0:
                out[(g, p)] = inter
    # Pred totals (for Block 3's denominator).
    for p in _CONFUSION_CLASS_IDS:
        p_n = int((pred == p).sum())
        out[("pred_total", p)] = p_n
    return out


def _eval_one_case(args: dict) -> dict:
    import nibabel as nib
    case_id   = args["case_id"]
    pred_path = args["pred_path"]
    gt_path   = args["gt_path"]
    t0 = time.time()
    try:
        pred_img = nib.load(pred_path)
        gt_img   = nib.load(gt_path)
        pred = np.asarray(pred_img.dataobj, dtype=np.int16)
        gt   = np.asarray(gt_img.dataobj,   dtype=np.int16)
        affine = gt_img.affine

        if pred.shape != gt.shape:
            return {"case_id": case_id, "ok": False,
                    "error": f"shape mismatch pred={pred.shape} gt={gt.shape}",
                    "elapsed_sec": time.time() - t0}

        dices, dice_hip_merged, n_valid = _per_case_dice(pred, gt)
        junction_dsc = _compute_junction_dsc(pred, gt, affine)
        ll_emit_voxels = int((pred == _LAST_LUMBAR_LABEL_ID).sum())
        confusion = _compute_voxel_confusion(pred, gt)

        out: Dict[str, Any] = {
            "case_id": case_id, "ok": True,
            "n_valid_voxels": n_valid,
            "shape": str(pred.shape),
            "elapsed_sec": round(time.time() - t0, 2),
            "dice_hip_merged": dice_hip_merged,
            "junction_dsc": junction_dsc,
            "ll_emit_voxels": ll_emit_voxels,
            "ll_emit": bool(ll_emit_voxels >= _LL_EMIT_THRESHOLD_VOX),
            # Confusion histogram is JSON-serialized (tuples-as-keys
            # don't survive JSON, so we stringify them).
            "_confusion": {f"{g}|{p}": v for (g, p), v in confusion.items()},
        }
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
        out["mean_pelvis"] = float(np.mean(pelvis)) if pelvis else None
        out["dice_overall"] = float(np.mean(all_fg)) if all_fg else None
        return out
    except Exception as exc:
        return {"case_id": case_id, "ok": False, "error": str(exc),
                "elapsed_sec": time.time() - t0}


# ===========================================================================
# Aggregation
# ===========================================================================

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


def _decode_confusion(d: Dict[str, int]) -> Dict[Tuple[Any, int], int]:
    """Reverse the JSON-safe stringification done in _eval_one_case."""
    out: Dict[Tuple[Any, int], int] = {}
    for k, v in d.items():
        g_str, p_str = k.split("|")
        g: Any = int(g_str) if g_str.lstrip("-").isdigit() else g_str
        p: int = int(p_str) if p_str.lstrip("-").isdigit() else p_str
        out[(g, p)] = v
    return out


def _aggregate(rows: List[dict], case_to_subtype: Dict[str, str]
                ) -> Tuple[Dict[str, Dict], Dict[str, float], Dict[str, Any]]:
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

        # Junction-DSC per subtype (per-case mean over cases that produced
        # a non-None junction_dsc).
        jvals = [c.get("junction_dsc") for c in cases
                 if c.get("junction_dsc") is not None]
        if jvals:
            per_class["junction_dsc"] = {
                "n": len(jvals),
                "mean": float(np.mean(jvals)),
                "std":  float(np.std(jvals, ddof=1)) if len(jvals) > 1 else 0.0,
            }
        else:
            per_class["junction_dsc"] = {"n": 0, "mean": None, "std": None}

        # LL emit per subtype: cases with pred-LL voxels >= threshold,
        # over cases that have any prediction (= all cases here).
        emit_total = sum(1 for c in cases if c.get("ll_emit"))
        per_class["ll_emit"] = {
            "n_total": n,
            "n_emit":  int(emit_total),
            "rate":    (emit_total / n) if n > 0 else None,
        }

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

    # ── Confusion blocks (Table 5) ────────────────────────────────────────
    def _aggregate_confusion(subtype: str) -> Dict[Tuple[Any, int], int]:
        agg: Counter = Counter()
        for c in by_subtype.get(subtype, []):
            d = _decode_confusion(c.get("_confusion", {}))
            for k, v in d.items():
                agg[k] += v
        return dict(agg)

    confusion_blocks: Dict[str, Any] = {}

    # Block 1: GT==last_lumbar voxels on lumb cases -> distribution across pred classes
    cnf_lumb = _aggregate_confusion(_SUBTYPE_LUMB)
    gt5_total_lumb = cnf_lumb.get(("gt_total", _LAST_LUMBAR_LABEL_ID), 0)
    block1: Dict[str, Dict[str, Any]] = {
        "n_cases":  len(by_subtype.get(_SUBTYPE_LUMB, [])),
        "gt_class": "last_lumbar",
        "subtype":  "lumb",
        "gt_voxel_total": int(gt5_total_lumb),
        "by_pred": {},
    }
    if gt5_total_lumb > 0:
        for p in _CONFUSION_CLASS_IDS:
            v = cnf_lumb.get((_LAST_LUMBAR_LABEL_ID, p), 0)
            block1["by_pred"][_CONFUSION_CLASS_NAMES[p]] = {
                "voxels":  int(v),
                "percent": 100.0 * v / gt5_total_lumb,
            }

    # Block 2: GT==L4 voxels on sacr_count cases -> distribution
    cnf_sc = _aggregate_confusion(_SUBTYPE_SACR_COUNT)
    gt4_total_sc = cnf_sc.get(("gt_total", _L4_LABEL_ID), 0)
    block2: Dict[str, Dict[str, Any]] = {
        "n_cases":  len(by_subtype.get(_SUBTYPE_SACR_COUNT, [])),
        "gt_class": "L4",
        "subtype":  "sacr_count",
        "gt_voxel_total": int(gt4_total_sc),
        "by_pred": {},
    }
    if gt4_total_sc > 0:
        for p in _CONFUSION_CLASS_IDS:
            v = cnf_sc.get((_L4_LABEL_ID, p), 0)
            block2["by_pred"][_CONFUSION_CLASS_NAMES[p]] = {
                "voxels":  int(v),
                "percent": 100.0 * v / gt4_total_sc,
            }

    # Block 3: pred==last_lumbar voxels on sacr_count cases -> dist over GT
    pred5_total_sc = cnf_sc.get(("pred_total", _LAST_LUMBAR_LABEL_ID), 0)
    block3: Dict[str, Dict[str, Any]] = {
        "n_cases":    len(by_subtype.get(_SUBTYPE_SACR_COUNT, [])),
        "pred_class": "last_lumbar",
        "subtype":    "sacr_count",
        "pred_voxel_total": int(pred5_total_sc),
        "by_gt": {},
    }
    if pred5_total_sc > 0:
        for g in _CONFUSION_CLASS_IDS:
            v = cnf_sc.get((g, _LAST_LUMBAR_LABEL_ID), 0)
            block3["by_gt"][_CONFUSION_CLASS_NAMES[g]] = {
                "voxels":  int(v),
                "percent": 100.0 * v / pred5_total_sc,
            }

    # last_lumbar specificity on sacr_count
    # = TN / (TN + FP) over all gt != 5 voxels in sacr_count cases.
    gt_total_all_classes_sc = sum(cnf_sc.get(("gt_total", c), 0)
                                   for c in _CONFUSION_CLASS_IDS)
    gt5_total_sc = cnf_sc.get(("gt_total", _LAST_LUMBAR_LABEL_ID), 0)
    gt_not5_total_sc = gt_total_all_classes_sc - gt5_total_sc
    fp_sc = sum(cnf_sc.get((g, _LAST_LUMBAR_LABEL_ID), 0)
                 for g in _CONFUSION_CLASS_IDS if g != _LAST_LUMBAR_LABEL_ID)
    if gt_not5_total_sc > 0:
        specificity_sc = 1.0 - (fp_sc / gt_not5_total_sc)
    else:
        specificity_sc = None

    confusion_blocks["block1_gt_LL_on_lumb"]      = block1
    confusion_blocks["block2_gt_L4_on_sacr_count"] = block2
    confusion_blocks["block3_pred_LL_on_sacr_count"] = block3
    confusion_blocks["last_lumbar_specificity_on_sacr_count"] = {
        "value":          specificity_sc,
        "tn_voxels":      int(gt_not5_total_sc - fp_sc) if gt_not5_total_sc > 0 else None,
        "fp_voxels":      int(fp_sc),
        "gt_not5_voxels": int(gt_not5_total_sc),
        "n_sacr_count_cases": len(by_subtype.get(_SUBTYPE_SACR_COUNT, [])),
    }
    confusion_blocks["ll_emit_by_subtype"] = {
        sub: {
            "n_total": len(by_subtype.get(sub, [])),
            "n_emit":  int(sum(1 for c in by_subtype.get(sub, []) if c.get("ll_emit"))),
        } for sub in _KNOWN_SUBTYPES
    }

    # ── Headline scalars ─────────────────────────────────────────────────
    headline: Dict[str, Any] = {}
    all_ok = [r for r in rows if r.get("ok")]
    overall_fg = [r["dice_overall"] for r in all_ok
                  if r.get("dice_overall") is not None]
    headline["n_test_cases"]         = float(len(all_ok))
    headline["mean_fg_dice_overall"] = (float(np.mean(overall_fg))
                                         if overall_fg else None)

    hip_merged_all = [r["dice_hip_merged"] for r in all_ok
                      if r.get("dice_hip_merged") is not None]
    if hip_merged_all:
        headline["hip_merged_dice_overall"]   = float(np.mean(hip_merged_all))
        headline["hip_merged_n_cases"]        = float(len(hip_merged_all))
        l_all = [r["dice_left_hip"]  for r in all_ok if r.get("dice_left_hip")  is not None]
        r_all = [r["dice_right_hip"] for r in all_ok if r.get("dice_right_hip") is not None]
        if l_all: headline["left_hip_dice_overall"]  = float(np.mean(l_all))
        if r_all: headline["right_hip_dice_overall"] = float(np.mean(r_all))

    # Sacrum overall (used for headline ∆ vs TS=0.817)
    sac_all = [r["dice_sacrum"] for r in all_ok if r.get("dice_sacrum") is not None]
    if sac_all:
        headline["sacrum_dice_overall"] = float(np.mean(sac_all))
        headline["sacrum_n_cases"]      = float(len(sac_all))
        headline["sacrum_minus_TS_overall_pts"] = (float(np.mean(sac_all)) - 0.817) * 100.0

    # Junction-DSC headline
    j_all  = [r.get("junction_dsc") for r in all_ok
              if r.get("junction_dsc") is not None]
    j_norm = [c.get("junction_dsc") for c in by_subtype.get(_SUBTYPE_NORMAL, [])
              if c.get("junction_dsc") is not None]
    j_lumb = [c.get("junction_dsc") for c in by_subtype.get(_SUBTYPE_LUMB, [])
              if c.get("junction_dsc") is not None]
    j_sc   = [c.get("junction_dsc") for c in by_subtype.get(_SUBTYPE_SACR_COUNT, [])
              if c.get("junction_dsc") is not None]
    if j_all:
        headline["junction_dsc_overall"] = float(np.mean(j_all))
        headline["junction_dsc_n_cases_overall"] = float(len(j_all))
    if j_norm:
        headline["junction_dsc_normal"] = float(np.mean(j_norm))
        headline["junction_dsc_n_cases_normal"] = float(len(j_norm))
    if j_lumb:
        headline["junction_dsc_lumb"] = float(np.mean(j_lumb))
        headline["junction_dsc_n_cases_lumb"] = float(len(j_lumb))
        # Δ vs TS junction lumb = 0.357 (from tab:sota_stratified)
        headline["junction_dsc_lumb_minus_TS_pts"] = (float(np.mean(j_lumb)) - 0.357) * 100.0
    if j_sc:
        headline["junction_dsc_sacr_count"] = float(np.mean(j_sc))
        headline["junction_dsc_n_cases_sacr_count"] = float(len(j_sc))

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

    if specificity_sc is not None:
        headline["last_lumbar_specificity_on_sacr_count"] = float(specificity_sc)

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

    return summary, headline, confusion_blocks


# ===========================================================================
# CSV / Markdown writers
# ===========================================================================

def _write_per_case_csv(rows: List[dict], path: Path,
                         case_to_subtype: Dict[str, str]) -> None:
    cols = (
        ["case_id", "subtype", "ok", "error",
         "n_valid_voxels", "shape", "elapsed_sec"]
        + [f"dice_{n}" for n in _ANATOMY_NAMES]
        + ["dice_hip_merged",
           "junction_dsc",
           "ll_emit_voxels", "ll_emit",
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
    extra_classes = ["hip_merged", "junction_dsc"]
    all_classes = list(_ANATOMY_NAMES) + extra_classes
    cols = (["subtype", "n_cases"]
            + [f"{n}_mean" for n in all_classes]
            + [f"{n}_std"  for n in all_classes]
            + [f"{n}_n"    for n in all_classes]
            + ["mean_lumbar_dice", "mean_pelvis_dice",
                "mean_fg_dice", "fg_dice_std",
                "ll_emit_n_total", "ll_emit_n_emit"])
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
            llem = s["per_class"].get("ll_emit", {"n_total": 0, "n_emit": 0})
            row["ll_emit_n_total"]  = llem["n_total"]
            row["ll_emit_n_emit"]   = llem["n_emit"]
            w.writerow(row)


def _fmt(v: Optional[float], fmt: str = "{:.3f}") -> str:
    return fmt.format(v) if v is not None else "—"


def _write_subgroup_md(summary: Dict[str, Dict], headline: Dict[str, float],
                        confusion_blocks: Dict[str, Any], path: Path) -> None:
    L: List[str] = []
    L.append("# CTSpinoPelvic1K — Test-set ensemble eval (Dataset803, merged)")
    L.append("")
    n_total = int(headline.get("n_test_cases", 0))
    fg_overall = headline.get("mean_fg_dice_overall")
    L.append(f"**Test cases evaluated:** {n_total}")
    L.append(f"**Overall mean foreground Dice:** {_fmt(fg_overall)}")
    L.append("")
    L.append("## Headline metrics")
    L.append("")
    L.append("| Metric | Value | n |")
    L.append("|---|---|---|")
    if "junction_dsc_overall" in headline:
        L.append(f"| Junction-DSC (overall) | {_fmt(headline['junction_dsc_overall'])} | "
                 f"{int(headline.get('junction_dsc_n_cases_overall', 0))} |")
    if "junction_dsc_normal" in headline:
        L.append(f"| Junction-DSC Normal | {_fmt(headline['junction_dsc_normal'])} | "
                 f"{int(headline.get('junction_dsc_n_cases_normal', 0))} |")
    if "junction_dsc_lumb" in headline:
        L.append(f"| Junction-DSC Lumbarization | "
                 f"**{_fmt(headline['junction_dsc_lumb'])}** | "
                 f"{int(headline.get('junction_dsc_n_cases_lumb', 0))} |")
    if "junction_dsc_lumb_minus_TS_pts" in headline:
        delta = headline["junction_dsc_lumb_minus_TS_pts"]
        sign = "+" if delta >= 0 else ""
        L.append(f"| **∆ Junction-DSC vs TS=0.357 (lumb)** | "
                 f"**{sign}{delta:.1f} pts** | — |")
    if "junction_dsc_sacr_count" in headline:
        L.append(f"| Junction-DSC Sacr_count | "
                 f"{_fmt(headline['junction_dsc_sacr_count'])} | "
                 f"{int(headline.get('junction_dsc_n_cases_sacr_count', 0))} |")
    if "last_lumbar_dice_on_lumb" in headline:
        L.append(f"| last_lumbar Dice on lumb | "
                 f"{_fmt(headline['last_lumbar_dice_on_lumb'])} | "
                 f"{int(headline['last_lumbar_n_lumb_cases'])} |")
    if "L4_dice_on_sacr_count" in headline:
        L.append(f"| L4 Dice on sacr_count | "
                 f"{_fmt(headline['L4_dice_on_sacr_count'])} | "
                 f"{int(headline['L4_n_sacr_count_cases'])} |")
    if "last_lumbar_specificity_on_sacr_count" in headline:
        L.append(f"| last_lumbar specificity on sacr_count | "
                 f"{_fmt(headline['last_lumbar_specificity_on_sacr_count'])} | — |")
    if "sacrum_dice_overall" in headline:
        L.append(f"| Sacrum overall (vs TS=0.817) | "
                 f"{_fmt(headline['sacrum_dice_overall'])} | "
                 f"{int(headline.get('sacrum_n_cases', 0))} |")
    if "sacrum_minus_TS_overall_pts" in headline:
        delta = headline["sacrum_minus_TS_overall_pts"]
        sign = "+" if delta >= 0 else ""
        L.append(f"| **∆ Sacrum vs TS** | **{sign}{delta:.1f} pts** | — |")
    if "normal_mean_fg_dice" in headline:
        L.append(f"| Normal mean fg Dice | "
                 f"{_fmt(headline['normal_mean_fg_dice'])} | "
                 f"{int(headline['normal_n_cases'])} |")
    if "any_lstv_mean_fg_dice" in headline:
        L.append(f"| Any-LSTV mean fg Dice | "
                 f"{_fmt(headline['any_lstv_mean_fg_dice'])} | "
                 f"{int(headline['any_lstv_n_cases'])} |")
    if "any_lstv_minus_normal_fg_dice" in headline:
        delta = headline["any_lstv_minus_normal_fg_dice"]
        sign = "+" if delta >= 0 else ""
        L.append(f"| **Any-LSTV − Normal (fg Dice)** | "
                 f"**{sign}{delta:.3f}** | — |")
    L.append("")

    # Hip per-side / merged
    if "hip_merged_dice_overall" in headline:
        L.append("## Hip class — per-side vs merged")
        L.append("")
        L.append("| Metric | Value | n |")
        L.append("|---|---|---|")
        if "left_hip_dice_overall" in headline:
            L.append(f"| left_hip Dice  | {_fmt(headline['left_hip_dice_overall'])} | "
                     f"{int(headline.get('hip_merged_n_cases', 0))} |")
        if "right_hip_dice_overall" in headline:
            L.append(f"| right_hip Dice | {_fmt(headline['right_hip_dice_overall'])} | "
                     f"{int(headline.get('hip_merged_n_cases', 0))} |")
        L.append(f"| **hip_merged Dice (L+R as one class)** | "
                 f"**{_fmt(headline['hip_merged_dice_overall'])}** | "
                 f"{int(headline['hip_merged_n_cases'])} |")
        L.append("")

    # Per-subtype Dice including junction-DSC and LL emit
    L.append("## Per-subtype Dice (mean ± SD)")
    L.append("")
    cls_show = ["L4", "last_lumbar", "sacrum", "hip_merged", "junction_dsc"]
    header = (["Subtype", "n"] + cls_show + ["LL emit", "mean fg"])
    L.append("| " + " | ".join(header) + " |")
    L.append("|" + "|".join(["---"] * len(header)) + "|")
    for sub in _KNOWN_SUBTYPES:
        s = summary[sub]
        n = s["n_cases"]
        cells: List[str] = [PRETTY[sub], str(n)]
        for name in cls_show:
            pc = s["per_class"].get(name, {"mean": None, "std": None, "n": 0})
            if pc.get("mean") is None:
                cells.append("—")
            else:
                cells.append(f"{pc['mean']:.3f} ± {pc['std']:.3f}")
        ll = s["per_class"].get("ll_emit", {"n_total": 0, "n_emit": 0})
        cells.append(f"{ll['n_emit']}/{ll['n_total']}")
        cells.append(_fmt(s["mean_fg_dice"]))
        L.append("| " + " | ".join(cells) + " |")
    L.append("")

    # Confusion blocks
    L.append("## Voxel confusion blocks (paper Table 5)")
    L.append("")
    for blk_key, blk_label in [
        ("block1_gt_LL_on_lumb",
         "Block 1: GT==last_lumbar voxels on lumb cases (good signal)"),
        ("block2_gt_L4_on_sacr_count",
         "Block 2: GT==L4 voxels on sacr_count cases (failure-mode signal)"),
        ("block3_pred_LL_on_sacr_count",
         "Block 3: pred==last_lumbar voxels on sacr_count cases (inverse)"),
    ]:
        blk = confusion_blocks.get(blk_key, {})
        L.append(f"### {blk_label}")
        L.append(f"n_cases = {blk.get('n_cases', 0)}")
        if blk_key in ("block1_gt_LL_on_lumb", "block2_gt_L4_on_sacr_count"):
            cells_dict = blk.get("by_pred", {})
            row_label = "pred"
        else:
            cells_dict = blk.get("by_gt", {})
            row_label = "GT"
        if not cells_dict:
            L.append("(no data — subtype absent or no relevant voxels)")
            L.append("")
            continue
        L.append(f"| {row_label} class | voxels | percent |")
        L.append("|---|---|---|")
        for cname in ["background", "L1", "L2", "L3", "L4", "last_lumbar",
                       "sacrum", "left_hip", "right_hip"]:
            cell = cells_dict.get(cname)
            if cell is None:
                continue
            L.append(f"| {cname} | {cell['voxels']} | {cell['percent']:.1f}% |")
        L.append("")
    spec = confusion_blocks.get("last_lumbar_specificity_on_sacr_count", {})
    if spec.get("value") is not None:
        L.append(f"**last_lumbar specificity on sacr_count** = "
                 f"{spec['value']:.4f}  (n_cases = {spec['n_sacr_count_cases']})")
        L.append("")

    L.append("Full per-class breakdown in `subgroup_summary.csv`; "
             "per-case dices in `per_case_dice.csv`; "
             "raw voxel-counts in `confusion_blocks.json`.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n")


# ===========================================================================
# Main
# ===========================================================================

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

    print("=" * 72)
    print(f"SPINESURG-CT FULL EVAL — DATASET803 MERGED")
    print("=" * 72)
    print(f"  predictions_dir : {args.predictions_dir}")
    print(f"  labels_dir      : {args.labels_dir}")
    print(f"  lstv_cases_json : {args.lstv_cases_json}")
    print(f"  output_dir      : {args.output_dir}")
    print(f"  workers         : {args.workers}")
    print("=" * 72)

    case_to_subtype = _load_subtype_map(args.lstv_cases_json)
    print(f"\n{len(case_to_subtype)} cases mapped from lstv_cases.json")
    if case_to_subtype:
        print(f"  full distribution: {dict(Counter(case_to_subtype.values()))}")

    work = _build_work(args.predictions_dir, args.labels_dir)
    print(f"\n{len(work)} (prediction, label) pairs to evaluate\n")

    rows: List[dict] = []
    n_ok = n_fail = 0
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_eval_one_case, w): w["case_id"] for w in work}
        if _HAS_TQDM:
            iterator = tqdm(as_completed(futs), total=len(futs),
                             desc="dice+jxn+conf", unit="case",
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
                    msg = (f"  [{i:>3}/{len(work)}] OK  {cid:<48s} "
                           f"fg={_fmt(r.get('dice_overall'))} "
                           f"hip_m={_fmt(r.get('dice_hip_merged'))} "
                           f"jxn={_fmt(r.get('junction_dsc'))} "
                           f"({r.get('elapsed_sec','?')}s)")
                    if _HAS_TQDM: tqdm.write(msg)
                    else:         print(msg, flush=True)
            else:
                n_fail += 1
                msg = f"  [{i:>3}/{len(work)}] FAIL {cid}: {r.get('error','?')}"
                if _HAS_TQDM: tqdm.write(msg)
                else:         print(msg, file=sys.stderr, flush=True)

    if n_ok == 0:
        print("ERROR: zero successful evaluations.", file=sys.stderr)
        return 3

    summary, headline, confusion_blocks = _aggregate(rows, case_to_subtype)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_per_case_csv(rows, args.output_dir / "per_case_dice.csv",
                         case_to_subtype)
    _write_subgroup_csv(summary, args.output_dir / "subgroup_summary.csv")
    _write_subgroup_md(summary, headline, confusion_blocks,
                         args.output_dir / "subgroup_summary.md")
    (args.output_dir / "headline.json").write_text(
        json.dumps(headline, indent=2))
    (args.output_dir / "confusion_blocks.json").write_text(
        json.dumps(confusion_blocks, indent=2))

    elapsed_total = time.time() - t_start
    print()
    print("=" * 72)
    print(" EVAL COMPLETE")
    print("=" * 72)
    print(f"  cases ok / failed       : {n_ok} / {n_fail}")
    print(f"  total elapsed           : {elapsed_total:.1f}s")
    print(f"  overall mean fg Dice    : "
          f"{_fmt(headline.get('mean_fg_dice_overall'))}")
    if "junction_dsc_overall" in headline:
        print(f"  junction-DSC overall    : "
              f"{_fmt(headline['junction_dsc_overall'])}")
    if "junction_dsc_lumb" in headline:
        print(f"  junction-DSC lumb       : "
              f"{_fmt(headline['junction_dsc_lumb'])} "
              f"(TS=0.357, ∆={headline.get('junction_dsc_lumb_minus_TS_pts',0):+.1f} pts)")
    if "sacrum_dice_overall" in headline:
        print(f"  sacrum overall          : "
              f"{_fmt(headline['sacrum_dice_overall'])} "
              f"(TS=0.817, ∆={headline.get('sacrum_minus_TS_overall_pts',0):+.1f} pts)")
    if "last_lumbar_specificity_on_sacr_count" in headline:
        print(f"  last_lumbar specificity : "
              f"{_fmt(headline['last_lumbar_specificity_on_sacr_count'])}")
    print("=" * 72)
    print(f"  per_case_dice.csv       -> {args.output_dir/'per_case_dice.csv'}")
    print(f"  subgroup_summary.csv    -> {args.output_dir/'subgroup_summary.csv'}")
    print(f"  subgroup_summary.md     -> {args.output_dir/'subgroup_summary.md'}")
    print(f"  headline.json           -> {args.output_dir/'headline.json'}")
    print(f"  confusion_blocks.json   -> {args.output_dir/'confusion_blocks.json'}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
