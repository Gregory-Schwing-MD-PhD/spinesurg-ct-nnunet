"""
SpineSurg-CT -- nnU-Net v2 trainers with W&B logging (v19.1)
tools/nnunet_wandb_variant.py

v19.1 — May 2026: CROSS-PROCESS BIAS COUNTERS.
==============================================
Per-epoch logging of patch-bias activity (force_fg calls served, L6
forces emitted, sacrum forces emitted, uniform-fallback count). The
counters live in multiprocessing.Value at module level in
lstv_biased_dataloader.py — created at import time before nnU-Net's
MultiThreadedAugmenter forks workers, so the shared-memory backing
is inherited across fork. Workers increment, parent reads at epoch
boundaries.

What you'll see in the SLURM log every epoch:

    --- v19.1 patch bias activity (epoch 50) ---
      this epoch: force_fg=14823, L6 forces=752, sacrum forces=237,
                  uniform fallback=13834
      rates: L6=5.07%, sacrum=1.60%, no-bias=93.33%
      cumulative: force_fg=741150, L6=37600, sacrum=11850

What you'll see in W&B:

    patch_bias/this_epoch/n_force_fg
    patch_bias/this_epoch/n_l6_returned
    patch_bias/this_epoch/n_sacrum_returned
    patch_bias/this_epoch/l6_rate
    patch_bias/this_epoch/sacrum_rate
    patch_bias/cumulative/...

If the counters stay at 0 across multiple epochs while training is
otherwise progressing normally, the bias hook isn't firing — check
the startup banner's "PATCH BIAS" section first to confirm the
LSTVBiasedDataLoader3D substitution caught.

Expected rates with default config (frac=0.25, L6_prob=0.6,
sacrum_prob=0.5):
  - lumb cases are ~34% of oversample pool, oversampled to 25% of queue
    -> ~8.5% of force_fg calls match lumb signature
    -> ~5% of force_fg calls return L6 (0.085 * 0.6)
  - sacr_count cases are ~12% of pool -> 3% of queue -> ~1.5% sacrum rate

If observed rates are within 30% of expected (e.g. L6 rate between
3.5% and 6.5%), bias is firing correctly.

v19 — May 2026: BIDIRECTIONAL CONFUSION TRACKING.
=================================================
Adds per-(subgroup, class) confusion logging in BOTH directions:

  GT-direction: "for each voxel where GT == C, what did the model
                 predict?" — answers "where do my L6 voxels go on
                 lumbarization cases?" (Are they being called L5?
                 Sacrum? Background?)

  PRED-direction: "for each voxel where PRED == C, what was the GT?"
                  — answers "when the model says L5 on a sacr_count
                  case (where L5 should be ABSENT), what is it
                  actually looking at?" (Is it sacrum? L4? Background?)

The two views are complementary: GT-direction localizes where the
correct anatomy is being misclassified TO, PRED-direction localizes
where false predictions are coming FROM. Together with the v17.4
specificity metrics they give a full picture of confusion landscape.

Headline keys emitted:
  val/headline/L6_GT_predicted_as_{name}_on_lumb        (GT direction)
  val/headline/predicted_L5_actually_{name}_on_sacr_count (PRED direction)
  val/headline/predicted_L6_actually_{name}_on_normal     (PRED direction)

Per-subgroup detail keys for every (subgroup, GT class) pair with
voxels are emitted in both directions; the headlines pull out the
clinically interesting ones for the paper.

v18 — May 2026: PATCH BIAS VIA CLASS OVERRIDE.
==============================================
The patch class biasing that v15-v17.x attempted via monkey-patching
np.random.choice has been replaced by a proper subclass of
nnUNetDataLoader3D (see tools/lstv_biased_dataloader.py).

The new approach:
  - LSTVBiasedDataLoader3D overrides get_bbox to set `overwrite_class`
    when the case has L6 voxels (lumb signature) or L4-with-no-L5-with-
    sacrum (sacr_count signature). nnU-Net's existing overwrite_class
    parameter then deterministically crops around a voxel of the chosen
    class.
  - get_dataloaders in the LSTV mixin swaps the class for the duration
    of super().get_dataloaders() so workers fork from a parent where
    the symbol is already replaced — no monkey-patching across forks,
    no frame inspection.
  - Detection uses class_locations content alone (no JSON dependency,
    no case_id lookup) — L6 voxels imply lumb, L4+sacrum-without-L5
    imply sacr_count.

Why this matters: the v15-v17.x monkey-patch never actually fired in
worker processes — the bias was advertised in startup logs but in
production we observed bias_call_count=0 across 64 epochs of fold 4
in job 35993148, and the same in every prior production run. v18's
class override is testable end-to-end (see
tests/test_lstv_biased_dataloader.py) and verifiably active at
training time (PATCH BIAS section in the startup diagnostics banner
walks to the actual loader instance and reports its class name).

Configuration knobs:
  SPINESURG_LSTV_BIAS_ENABLED       (default 1; "0" disables)
  SPINESURG_LSTV_BIAS_L6_PROB       (default 0.60)
  SPINESURG_LSTV_BIAS_SACRUM_PROB   (default 0.50)

v17.4 — hallucination metrics (specificity tracking on absent classes).
v17.1-v17.3 — dedicated LSTV val pass (forces forward pass on every
              LSTV val case each epoch instead of subsampling).
v17 — oversample pool includes ambiguous; symmetric per-record subtype
      demotion; verbose startup-diagnostic banner.
v15-v16 — multiple attempts to make the np.random.choice monkey-patch
          fire at the right call site. Superseded by v18.
v9-v14 — 6-way subgroup binning, lstv_cases.json schema v3, CE weight
         tensor sizing, oversample pool semantics, partial-annotation
         contract. Preserved in v18.

Design intent
=============
Three-tier emphasis on LSTV anatomy, partial-annotation aware:

  (a) Class-imbalance correction via queue-level oversampling.
      Applied uniformly to ALL LSTV subtypes except 'ambiguous'.
      Without this, LSTV is ~4% of training and the model defaults
      to normal anatomy.

  (b) Headline-subtype emphasis via per-case patch sampling bias.
      v18: implemented via LSTVBiasedDataLoader3D (class override,
      not monkey patch).

      Lumb       -> L6  (class 6) @ 60%  (the class TS misses)
      sacr_count -> sacrum (cls 7) @ 50%  (L4/sacrum transition zone)

      Detection from class_locations content alone:
        lumb       :  L6 has voxels in this case
        sacr_count :  L4 has voxels AND L5 does NOT AND sacrum has voxels
                      (the L4-in-fg constraint is what distinguishes
                       sacr_count from pelvic-only records, which lack
                       BOTH L5 and L4 because their lumbar region is
                       all ignore=10)

  (c) Class-level loss reweighting via CE weights.
        L6     = 4.0
        sacrum = 2.0
      Background = 0.5; rest = 1.0.

Partial-annotation contract
===========================
Some training cases (separate-mode spine-only and pelvic-only) have
labels with value 10 = IGNORE in regions outside the present
annotator's domain. nnU-Net v2's DC_and_CE_loss respects ignore_label
when configured, masking those voxels out of the gradient. Required:

  - dataset.json must have "ignore": 10 in its labels dict
    (set by convert_hf_to_nnunet.py:LABEL_NAMES, May 2026)
  - export_hf.py:merge_labels writes value 10 in un-annotated regions
    of partial-mode cases (May 2026)
  - This trainer's _IGNORE_LABEL = 10 must match
  - num_segmentation_heads from label_manager = 10 (the ignore label
    has no output head; it's purely a loss mask)

Author: Gregory Schwing, MD-PhD  |  Wayne State University / DMC
"""

from __future__ import annotations

import json
import logging
import multiprocessing as _mp
import os
import random
import sys
import time
import warnings
from collections import Counter, defaultdict
from os.path import isfile, isdir, join
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs import (
    nnUNetTrainer_5epochs, nnUNetTrainer_100epochs,
)


# ── v18: LSTV-aware patch class biasing via dataloader subclass ────────────
# Tries multiple import paths to support both installed-into-nnunetv2
# layouts (the canonical case after copying tools/*.py into
# /opt/conda/.../variants/) and source-tree layouts (running tests
# from spinesurg-ct-nnunet/tools/).
LSTVBiasedDataLoader3D = None
_LSTV_LOADER_IMPORT_ERROR: Optional[str] = None
# v19.1: cross-process shared-counter helpers. None when the loader
# module isn't importable; trainer falls through gracefully.
_read_shared_bias_counters: Optional[Callable] = None
_reset_shared_bias_counters: Optional[Callable] = None
try:
    from nnunetv2.training.nnUNetTrainer.variants.lstv_biased_dataloader import (
        LSTVBiasedDataLoader3D,
        read_shared_bias_counters as _read_shared_bias_counters,
        reset_shared_bias_counters as _reset_shared_bias_counters,
    )
except ImportError as _exc1:
    try:
        # Source-tree fallback (running from spinesurg-ct-nnunet/tools/)
        from lstv_biased_dataloader import (  # type: ignore
            LSTVBiasedDataLoader3D,
            read_shared_bias_counters as _read_shared_bias_counters,
            reset_shared_bias_counters as _reset_shared_bias_counters,
        )
    except ImportError as _exc2:
        try:
            # Sibling-package fallback
            from .lstv_biased_dataloader import (  # type: ignore
                LSTVBiasedDataLoader3D,
                read_shared_bias_counters as _read_shared_bias_counters,
                reset_shared_bias_counters as _reset_shared_bias_counters,
            )
        except (ImportError, ValueError) as _exc3:
            LSTVBiasedDataLoader3D = None
            _read_shared_bias_counters = None
            _reset_shared_bias_counters = None
            _LSTV_LOADER_IMPORT_ERROR = (
                f"installed: {_exc1}; source: {_exc2}; sibling: {_exc3}"
            )

_LSTV_BIASED_LOADER_AVAILABLE = LSTVBiasedDataLoader3D is not None


_METRIC_KEYS = (
    "train_losses", "val_losses", "mean_fg_dice", "ema_fg_dice",
    "dice_per_class_or_region", "lrs",
    "epoch_start_timestamps", "epoch_end_timestamps",
)

_ANATOMY_NAMES = ["L1", "L2", "L3", "L4", "L5", "L6",
                   "sacrum", "left_hip", "right_hip"]

_FG_CLASS_IDS    = list(range(1, 10))
_LUMBAR_IDS      = list(range(1, 7))
_PELVIS_IDS      = list(range(7, 10))
_L6_LABEL_ID     = 6
_SACRUM_LABEL_ID = 7
_IGNORE_LABEL    = 10
_VALID_CONFIGS   = ("fused", "spine_only", "pelvic_native")

# Network output channel count under the standard SpineSurg-CT label
# schema (background + 9 foreground; ignore is masked, not classified).
_DEFAULT_NUM_OUTPUT_CLASSES = 10

# 6-way taxonomy
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
# v17: ALL non-normal subtypes are oversampled. Previously
# excluded: ambiguous (n=4) on memorization concerns. v17 includes
# it per Greg's request — the downside of memorization on a tiny
# group is acceptable; the upside is the model sees rare LSTV
# anatomy more often.
_OVERSAMPLE_SUBTYPES = tuple(s for s in _KNOWN_SUBTYPES if s != _SUBTYPE_NORMAL)
# Equivalent: (lumb, sacr_count, semisacralization, sacralization, ambiguous)

LSTV_CASES_SCHEMA_MIN = 1
LSTV_CASES_SCHEMA_MAX = 3


def _install_nnunet_warning_filter() -> None:
    warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide", category=RuntimeWarning)
    np.seterr(invalid="ignore", divide="warn")
    for n in ("torch.fx.experimental.symbolic_shapes", "torch._dynamo",
              "torch._dynamo.symbolic_convert", "torch._dynamo.guards",
              "torch._dynamo.output_graph", "torch._inductor",
              "torch._inductor.compile_fx", "torch._inductor.scheduler",
              "torch._inductor.codecache", "torch._functorch"):
        logging.getLogger(n).setLevel(logging.ERROR)


def _install_perf_tuning():
    try: torch.backends.cudnn.benchmark = True
    except Exception: pass
    try: torch.set_float32_matmul_precision('high')
    except Exception: pass


def _env_truthy(name, default=False):
    v = os.environ.get(name, "").strip().lower()
    return (v in ("1", "true", "yes", "on")) if v else default


def _env_int(name, default):
    try: return int(os.environ.get(name, str(default)))
    except Exception: return default


def _env_float(name, default):
    try: return float(os.environ.get(name, str(default)))
    except Exception: return default


# ── LSTV case identification ────────────────────────────────────────────────

def _safe_id(token: str) -> str:
    """Mirrors convert_hf_to_nnunet.py:safe_id() — must stay in sync."""
    s = str(token)
    for bad in (".", "/", "\\", " ", ":", "(", ")"):
        s = s.replace(bad, "_")
    return s.strip("_")


def _canonicalize_subtype_str(sub: str) -> str:
    """Canonicalize an LSTV subtype string to one of the 6 canonical names."""
    if not sub: return _SUBTYPE_NORMAL
    s = str(sub).strip().lower()
    if s.startswith("ambig"):       return _SUBTYPE_AMBIG
    if "lumb" in s:                  return _SUBTYPE_LUMB
    if "semi" in s:                  return _SUBTYPE_SEMI
    if "sacr_count" in s or "sacrcount" in s: return _SUBTYPE_SACR_COUNT
    if "sacr" in s:                  return _SUBTYPE_SACR
    return _SUBTYPE_NORMAL


def _read_lstv_cases_json(json_path: Path
                           ) -> Optional[Tuple[Set[str], Counter, Dict[str, str], List[str]]]:
    if not json_path.exists(): return None
    try: data = json.loads(json_path.read_text())
    except Exception: return None
    if not isinstance(data, dict): return None
    schema = int(data.get("schema_version", 0))
    if schema < LSTV_CASES_SCHEMA_MIN or schema > LSTV_CASES_SCHEMA_MAX:
        return None

    case_to_subtype: Dict[str, str] = {}
    all_case_ids: Set[str] = set()

    if schema >= 3:
        cts = data.get("case_to_subtype") or {}
        if not isinstance(cts, dict): return None
        for cid, sub in cts.items():
            canon = _canonicalize_subtype_str(sub)
            case_to_subtype[cid] = canon
            all_case_ids.add(cid)
        # Always recompute pool from in-code _OVERSAMPLE_SUBTYPES.
        # The JSON's lstv_oversample_pool field, if present, reflects
        # whatever decision was current when the JSON was written and
        # is ignored on read. Single source of truth: this .py file.
        oversample_pool = sorted(
            cid for cid, sub in case_to_subtype.items()
            if sub in _OVERSAMPLE_SUBTYPES
        )
    else:
        case_ids_dict = data.get("case_ids") or {}
        if not isinstance(case_ids_dict, dict): return None
        for cid, sub in case_ids_dict.items():
            if not isinstance(sub, str): continue
            canon = _canonicalize_subtype_str(sub)
            case_to_subtype[cid] = canon
            all_case_ids.add(cid)
        oversample_pool = sorted(
            cid for cid, sub in case_to_subtype.items()
            if sub in _OVERSAMPLE_SUBTYPES
        )

    subtype_counts = Counter(case_to_subtype.values())
    return all_case_ids, subtype_counts, case_to_subtype, list(oversample_pool)


def _candidate_lstv_json_paths(dataset_name: str) -> List[Path]:
    cands: List[Path] = []
    pre = os.environ.get("nnUNet_preprocessed")
    if pre: cands.append(Path(pre) / dataset_name / "lstv_cases.json")
    raw = os.environ.get("nnUNet_raw") or os.environ.get("nnUNet_raw_data_base")
    if raw: cands.append(Path(raw) / dataset_name / "lstv_cases.json")
    return cands


def _candidate_hf_export_dirs() -> List[Path]:
    cands: List[Path] = []
    env = os.environ.get("HF_EXPORT_DIR", "").strip()
    if env: cands.append(Path(env))
    home = Path.home()
    cands.append(home / "CTSpinoPelvic1K" / "data" / "hf_export")
    cands.append(home / "spinesurg-ct-nnunet" / "data" / "hf_export")
    raw = os.environ.get("nnUNet_raw") or os.environ.get("nnUNet_raw_data_base")
    if raw:
        cands.append(Path(raw).parent.parent / "data" / "hf_export")
        cands.append(Path(raw).parent / "data" / "hf_export")
    seen, uniq = set(), []
    for c in cands:
        s = str(c)
        if s not in seen:
            seen.add(s); uniq.append(c)
    return uniq


def _read_lstv_records_from_manifest(hf_export_dir: Path
                                      ) -> Optional[Tuple[Set[str], Counter, Dict[str, str], List[str]]]:
    if not hf_export_dir.is_dir(): return None
    records: List[Dict] = []
    for fname in ("manifest_train.json", "manifest_validation.json", "manifest_test.json"):
        p = hf_export_dir / fname
        if p.exists():
            try:
                d = json.loads(p.read_text())
                if isinstance(d, list): records.extend(d)
                elif isinstance(d, dict) and "records" in d: records.extend(d["records"])
            except Exception: continue
    if not records:
        p = hf_export_dir / "manifest.json"
        if p.exists():
            try:
                d = json.loads(p.read_text())
                if isinstance(d, list): records = d
                elif isinstance(d, dict) and "records" in d: records = d["records"]
            except Exception: return None
    if not records: return None

    all_case_ids: Set[str] = set()
    subtypes: Counter = Counter()
    case_to_sub: Dict[str, str] = {}
    for r in records:
        lstv = str(r.get("lstv_label", "")).strip().lower()
        token = r.get("token")
        config = r.get("config", "fused")
        if token is None or config not in _VALID_CONFIGS: continue
        cid = f"{_safe_id(token)}__{config}"
        canon = _canonicalize_subtype_str(lstv)
        case_to_sub[cid] = canon
        subtypes[canon] += 1
        all_case_ids.add(cid)
    oversample_pool = sorted(
        cid for cid, sub in case_to_sub.items()
        if sub in _OVERSAMPLE_SUBTYPES
    )
    return (all_case_ids, subtypes, case_to_sub, oversample_pool) if all_case_ids else None


def _scan_lstv_case_ids_by_l6_voxels(labels_dir: str) -> Set[str]:
    import nibabel as nib
    out: Set[str] = set()
    if not os.path.isdir(labels_dir): return out
    for fn in sorted(os.listdir(labels_dir)):
        if not fn.endswith(".nii.gz"): continue
        try:
            arr = np.asarray(nib.load(join(labels_dir, fn)).dataobj).astype(np.int16)
            if np.any(arr == _L6_LABEL_ID):
                out.add(fn[:-len(".nii.gz")])
        except Exception: continue
    return out


# ── Per-case dice for subgroup logger ──────────────────────────────────────

def _per_case_dice_per_class(pred_argmax: torch.Tensor, gt: torch.Tensor,
                              ignore_label: int = _IGNORE_LABEL) -> List[Optional[float]]:
    pred_argmax = pred_argmax.flatten()
    gt = gt.flatten()
    valid = gt != ignore_label
    if not valid.any(): return [None] * len(_FG_CLASS_IDS)
    pred_argmax = pred_argmax[valid]
    gt = gt[valid]
    out: List[Optional[float]] = []
    for cid in _FG_CLASS_IDS:
        gt_mask = gt == cid
        gt_n = int(gt_mask.sum().item())
        if gt_n == 0:
            out.append(None); continue
        pred_mask = pred_argmax == cid
        pred_n = int(pred_mask.sum().item())
        inter = int((gt_mask & pred_mask).sum().item())
        denom = gt_n + pred_n
        out.append(2.0 * inter / denom if denom > 0 else None)
    return out


def _per_case_voxel_counts_per_class(pred_argmax: torch.Tensor, gt: torch.Tensor,
                                      ignore_label: int = _IGNORE_LABEL
                                      ) -> List[Tuple[int, int]]:
    """Return (gt_n, pred_n) per class, including classes where gt_n == 0.

    v17.4: hallucination tracking. When gt_n == 0, the case is in the
    "should not predict" condition for that class. pred_n is the number
    of voxels the model wrongly emitted. We accumulate these per
    subgroup so we can report:

        - Hallucination rate: fraction of "absent" cases where pred_n
          exceeds a clinically-meaningful threshold
        - Mean/max predicted volume when GT is absent
        - Specificity: 1 - (hallucination rate)

    Voxels with gt == ignore_label are excluded from BOTH gt_n and
    pred_n, matching the standard dice computation. This means a
    pelvic-only case (which has ignore=10 in the lumbar region) won't
    contribute spurious "L6 hallucination" stats from its uninformative
    region.

    Returns a list of (gt_n, pred_n) integer tuples in the same order
    as _FG_CLASS_IDS — so the caller can index parallel to the dice
    list.
    """
    pred_argmax = pred_argmax.flatten()
    gt = gt.flatten()
    valid = gt != ignore_label
    if not valid.any():
        return [(0, 0)] * len(_FG_CLASS_IDS)
    pred_argmax = pred_argmax[valid]
    gt = gt[valid]
    out: List[Tuple[int, int]] = []
    for cid in _FG_CLASS_IDS:
        gt_n = int((gt == cid).sum().item())
        pred_n = int((pred_argmax == cid).sum().item())
        out.append((gt_n, pred_n))
    return out


# Hallucination threshold: a per-class predicted-voxel count above which
# we consider the prediction a "meaningful hallucination" rather than
# noise at a boundary. ~1 cc at typical CT spacing of ~0.8 mm³/voxel.
_HALLUC_VOXEL_THRESHOLD = 100


# ── v19: bidirectional confusion helpers ───────────────────────────────────

def _per_case_confusion_by_gt(pred_argmax: torch.Tensor, gt: torch.Tensor,
                               num_classes: int = _DEFAULT_NUM_OUTPUT_CLASSES,
                               ignore_label: int = _IGNORE_LABEL
                               ) -> Dict[int, "np.ndarray"]:
    """GT-direction confusion: for each GT class C present in this case,
    return a length-num_classes vector where index k = count of voxels
    with (gt==C, pred==k).

    Returns {gt_class: confusion_vector}. Classes absent from GT
    (gt_n == 0) are not in the dict — there's no GT-direction confusion
    to compute when there are no GT voxels.

    Voxels with gt == ignore_label are excluded entirely (matches the
    dice / voxel-count helpers).

    Use case: "where do GT-L6 voxels end up being predicted on lumb
    cases?" — read confusion[L6] and inspect the per-pred-class fractions.
    """
    pred_argmax = pred_argmax.flatten()
    gt = gt.flatten()
    valid = gt != ignore_label
    if not valid.any():
        return {}
    pred_argmax = pred_argmax[valid]
    gt = gt[valid]

    out: Dict[int, np.ndarray] = {}
    for cid in _FG_CLASS_IDS:
        gt_mask = gt == cid
        if not gt_mask.any():
            continue
        pred_at_gt = pred_argmax[gt_mask].cpu().numpy().astype(np.int64)
        # Clamp to [0, num_classes-1] in case of out-of-range values
        # (shouldn't happen but defensive against unexpected pred values).
        pred_at_gt = np.clip(pred_at_gt, 0, num_classes - 1)
        out[cid] = np.bincount(pred_at_gt, minlength=num_classes).astype(np.int64)
    return out


def _per_case_confusion_by_pred(pred_argmax: torch.Tensor, gt: torch.Tensor,
                                 num_classes: int = _DEFAULT_NUM_OUTPUT_CLASSES,
                                 ignore_label: int = _IGNORE_LABEL
                                 ) -> Dict[int, "np.ndarray"]:
    """PRED-direction confusion: for each predicted class C with non-zero
    voxels in this case, return a length-num_classes vector where index k
    = count of voxels with (pred==C, gt==k).

    Returns {pred_class: confusion_vector}. Classes absent from PRED
    (pred_n == 0) are not in the dict.

    Voxels with gt == ignore_label are excluded entirely.

    Use case: "when the model predicts L5 on a sacr_count case (where
    L5 should NOT exist), what is it actually looking at?" — read
    confusion[L5] and inspect the per-gt-class fractions. Useful for
    diagnosing "the model is calling sacrum tissue 'L5'" vs "the model
    is calling background voxels 'L5'" — different fixes.
    """
    pred_argmax = pred_argmax.flatten()
    gt = gt.flatten()
    valid = gt != ignore_label
    if not valid.any():
        return {}
    pred_argmax = pred_argmax[valid]
    gt = gt[valid]

    out: Dict[int, np.ndarray] = {}
    for cid in _FG_CLASS_IDS:
        pred_mask = pred_argmax == cid
        if not pred_mask.any():
            continue
        gt_at_pred = gt[pred_mask].cpu().numpy().astype(np.int64)
        gt_at_pred = np.clip(gt_at_pred, 0, num_classes - 1)
        out[cid] = np.bincount(gt_at_pred, minlength=num_classes).astype(np.int64)
    return out


def _class_id_to_name(cid: int) -> str:
    """Render a class id as a human-readable name for log/W&B keys.
    Background = 0; foreground = 1..9; anything else = 'label{cid}'."""
    if cid == 0:
        return "background"
    if 1 <= cid <= 9:
        return _ANATOMY_NAMES[cid - 1]
    return f"label{cid}"


# =============================================================================
# Class reweighting
# =============================================================================

def _build_ce_class_weights(num_classes: int) -> Optional[torch.Tensor]:
    """Build a CE weight tensor of shape (num_classes,).

    `num_classes` MUST equal the number of output channels of the
    network — typically 10 here (background + 9 foreground). The
    ignore label is NOT a class in the network output (it's masked via
    ignore_index) and therefore has no entry in this weight tensor.
    """
    if not _env_truthy("SPINESURG_CE_REWEIGHT", default=True):
        return None
    if num_classes is None or num_classes <= 0:
        return None

    # Per-class weights keyed by the class index in the network output.
    # Indices 0-9: background, L1-L6, sacrum, left_hip, right_hip.
    # Do NOT add an entry for the ignore label (10) — the network does
    # not emit a channel for it.
    weights = {
        0: 0.5,  # background
        1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0,
        6: 4.0,  # L6 — emphasized for lumbarization recovery
        7: 2.0,  # sacrum — emphasized for sacr_count (missing-L5) and Castellvi
        8: 1.0, 9: 1.0,
    }
    name_to_id = {"background": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4,
                   "L5": 5, "L6": 6, "sacrum": 7,
                   "left_hip": 8, "right_hip": 9}
    overrides = os.environ.get("SPINESURG_CE_WEIGHTS", "").strip()
    if overrides:
        for spec in overrides.split(","):
            spec = spec.strip()
            if not spec or ":" not in spec: continue
            name, val = spec.split(":", 1)
            name = name.strip()
            cid = name_to_id.get(name)
            if cid is None:
                try: cid = int(name)
                except ValueError: continue
            try: weights[cid] = float(val)
            except ValueError: continue
    out = torch.zeros(num_classes, dtype=torch.float32)
    for cid in range(num_classes):
        out[cid] = float(weights.get(cid, 1.0))
    return out


def _detect_num_output_classes(trainer) -> int:
    """Return the number of channels in the network's output head.

    Resolution order:
      1. trainer.label_manager.num_segmentation_heads (canonical nnU-Net v2)
      2. count of dataset_json['labels'] excluding any 'ignore' entry
      3. _DEFAULT_NUM_OUTPUT_CLASSES (10) as final fallback
    """
    try:
        n = int(getattr(trainer.label_manager, "num_segmentation_heads"))
        if n > 0:
            return n
    except Exception:
        pass
    try:
        labels = trainer.plans_manager.dataset_json.get("labels", {})
        n = sum(1 for k in labels.keys() if str(k).lower() != "ignore")
        if n > 0:
            return n
    except Exception:
        pass
    return _DEFAULT_NUM_OUTPUT_CLASSES


# =============================================================================
# W&B mixin (6-way subgroup tracking)
# =============================================================================

class _WandBMixin:
    _wandb_run = None
    _metric_cache = None
    _intercepted_objects = None
    _profiler = None
    _profiler_total_steps = 0
    _profiler_step_count = 0
    _perf_tuning_installed = False
    _channels_last_active = False

    _val_subgroup_dice: Dict[str, List[List[Optional[float]]]] = None
    # v17.4: parallel buffer of (gt_n, pred_n) tuples per case, per class.
    # Same outer shape as _val_subgroup_dice (subtype -> list-of-cases ->
    # list-of-classes), but each per-class entry is a (int, int) tuple
    # instead of an Optional[float]. Used by _aggregate_subgroup_hallucination
    # to compute specificity / hallucination volume on cases where the
    # class is absent in GT.
    _val_subgroup_voxels: Dict[str, List[List[Tuple[int, int]]]] = None
    # v19: bidirectional confusion buffers, parallel to _val_subgroup_dice.
    # Each entry is a list-of-cases, where each case is a dict mapping
    # class_id -> length-num_classes np.int64 vector.
    #   _by_gt[sub][i][C][k]   = count of voxels with (gt==C, pred==k) in case i of subtype sub
    #   _by_pred[sub][i][C][k] = count of voxels with (pred==C, gt==k) in case i of subtype sub
    _val_subgroup_confusion_by_gt: Dict[str, List[Dict[int, "np.ndarray"]]] = None
    _val_subgroup_confusion_by_pred: Dict[str, List[Dict[int, "np.ndarray"]]] = None
    _val_case_to_subtype: Optional[Dict[str, str]] = None
    _per_subgroup_logging_enabled: bool = True
    # v19.1: snapshot of cross-process bias counters at epoch start.
    # Populated by on_train_epoch_start, diffed in on_epoch_end to
    # produce per-epoch counts.
    _bias_counters_at_epoch_start: Optional[Dict[str, int]] = None

    def _progress_json_path(self):
        out = getattr(self, "output_folder", None)
        return join(out, "progress.json") if out else None

    def _load_progress_json(self):
        path = self._progress_json_path()
        if path is None or not isfile(path): return {}
        try:
            with open(path, "r") as f: data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception: return {}

    # ── Phase logging helpers (v17 startup tracing) ──────────────────────

    def _phase(self, phase: str, msg: str) -> None:
        """Single phase log line. Every line is prefixed with
        [STARTUP] <phase>: ... so logs are grep-friendly."""
        try:
            self.print_to_log_file(f"[STARTUP] {phase}: {msg}")
        except Exception:
            print(f"[STARTUP] {phase}: {msg}", flush=True)

    def _phase_begin(self, phase: str) -> None:
        self._phase(phase, "============ BEGIN ============")

    def _phase_end(self, phase: str, ok: bool, note: str = "") -> None:
        marker = "OK ✓" if ok else "FAILED ✗"
        suffix = f" — {note}" if note else ""
        self._phase(phase, f"============ {marker}{suffix} ============")

    def _wrap_log_method(self, obj, cache):
        try:
            if not hasattr(obj, "log") or not callable(obj.log): return False
            if self._intercepted_objects is None:
                self._intercepted_objects = set()
            if id(obj) in self._intercepted_objects: return False
            original_log = obj.log
            def log_intercept(*args, **kwargs):
                try:
                    if len(args) >= 3:
                        key, value, epoch = args[0], args[1], args[2]
                    elif len(args) == 2:
                        key, value = args[0], args[1]
                        epoch = kwargs.get("epoch", None)
                    else:
                        key = kwargs.get("key", args[0] if args else None)
                        value = kwargs.get("value", None)
                        epoch = kwargs.get("epoch", None)
                    if isinstance(key, str) and epoch is not None:
                        e = int(epoch)
                        seq = cache[key]
                        while len(seq) < e + 1: seq.append(None)
                        seq[e] = value
                except Exception: pass
                return original_log(*args, **kwargs)
            obj.log = log_intercept
            self._intercepted_objects.add(id(obj))
            return True
        except Exception: return False

    def _install_log_intercepts(self):
        if self._metric_cache is None:
            self._metric_cache = defaultdict(list)
        cache = self._metric_cache
        logger = getattr(self, "logger", None)
        if logger is None: return
        if self._wrap_log_method(logger, cache):
            self.print_to_log_file(f"WandB: log intercepts installed on {type(logger).__name__}")
        for attr in dir(logger):
            if attr.startswith("_"): continue
            try: val = getattr(logger, attr)
            except Exception: continue
            if isinstance(val, (list, tuple)):
                for child in val:
                    if hasattr(child, "log") and callable(getattr(child, "log", None)):
                        self._wrap_log_method(child, cache)
            elif hasattr(val, "log") and callable(getattr(val, "log", None)) and val is not logger:
                self._wrap_log_method(val, cache)

    def _current_logs(self):
        disk = self._load_progress_json()
        if disk and any(k in disk for k in _METRIC_KEYS): return disk
        if self._metric_cache and len(self._metric_cache) > 0: return dict(self._metric_cache)
        return {}

    # ── CE reweighting ─────────────────────────────────────────────────

    def _maybe_apply_ce_reweighting(self):
        self._phase_begin("ce_reweight")
        if not _env_truthy("SPINESURG_CE_REWEIGHT", default=True):
            self._phase("ce_reweight", "DISABLED via SPINESURG_CE_REWEIGHT=0")
            self._phase_end("ce_reweight", ok=True, note="disabled")
            return
        try:
            from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
            from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
        except ImportError as e:
            self._phase("ce_reweight", f"IMPORT FAILED: {e}")
            self._phase_end("ce_reweight", ok=False, note="import failed")
            return

        num_classes = _detect_num_output_classes(self)
        self._phase("ce_reweight",
            f"num_segmentation_heads = {num_classes} (network output channels)")

        weights = _build_ce_class_weights(num_classes=num_classes)
        if weights is None:
            self._phase("ce_reweight", "weights tensor is None; skipping")
            self._phase_end("ce_reweight", ok=False, note="weights=None")
            return
        if weights.numel() != num_classes:
            self._phase("ce_reweight",
                f"weight length {weights.numel()} != num_classes "
                f"{num_classes}; skipping to avoid crash")
            self._phase_end("ce_reweight", ok=False, note="size mismatch")
            return

        device = getattr(self, "device", torch.device("cuda"))
        weights = weights.to(device)
        ds_enabled = bool(getattr(self, "enable_deep_supervision", True))
        self._phase("ce_reweight",
            f"weights = {weights.cpu().numpy().tolist()}")
        self._phase("ce_reweight", f"deep_supervision = {ds_enabled}")

        try:
            soft_dice_kwargs = {
                "batch_dice": getattr(self.configuration_manager, "batch_dice", True),
                "do_bg": False, "smooth": 1e-5, "ddp": False,
            }
            ce_kwargs = {
                "weight": weights, "ignore_index": _IGNORE_LABEL, "reduction": "mean",
            }
            new_loss = DC_and_CE_loss(soft_dice_kwargs, ce_kwargs,
                                       weight_ce=1.0, weight_dice=1.0,
                                       ignore_label=_IGNORE_LABEL,
                                       dice_class=MemoryEfficientSoftDiceLoss)
            if ds_enabled:
                from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
                ds_weights = getattr(self.loss, "weight_factors", None) if hasattr(self, "loss") else None
                if ds_weights is None:
                    n_levels = 5
                    ds_weights = np.array([1.0 / (2 ** i) for i in range(n_levels)])
                    ds_weights[-1] = 0
                    ds_weights = ds_weights / ds_weights.sum()
                self._phase("ce_reweight",
                    f"ds_weights = {[float(w) for w in ds_weights]}")
                new_loss = DeepSupervisionWrapper(new_loss, ds_weights)
            self.loss = new_loss
            self._phase("ce_reweight",
                f"installed DC_and_CE_loss with ignore_label={_IGNORE_LABEL}")
            self._phase_end("ce_reweight", ok=True)
        except Exception as exc:
            self._phase("ce_reweight",
                f"INSTALL FAILED: {exc}; using default loss")
            self._phase_end("ce_reweight", ok=False, note=str(exc))

    # ── Startup diagnostics ──────────────────────────────────────────────

    def _emit_startup_diagnostics(self) -> None:
        """Dump every relevant configuration value, pool composition,
        env-var override, and hook-install status as a single banner.

        Output is intentionally verbose: we want one grep-able block
        in the SLURM log that confirms whether oversampling, patch
        bias, CE reweighting, and per-subgroup tracking are each live.
        """
        log = self.print_to_log_file
        try:
            log("")
            log("=" * 72)
            log("SPINESURG-CT TRAINER STARTUP DIAGNOSTICS")
            log("=" * 72)

            log(f"  trainer class:    {type(self).__name__}")
            try: ds_name = self.plans_manager.dataset_name
            except Exception: ds_name = "<unknown>"
            log(f"  dataset:          {ds_name}")
            log(f"  fold:             {getattr(self, 'fold', '?')}")
            try:
                log(f"  num_epochs:       {self.num_epochs}")
                log(f"  iters/epoch:      {getattr(self, 'num_iterations_per_epoch', '?')}")
                log(f"  val iters/epoch:  {getattr(self, 'num_val_iterations_per_epoch', '?')}")
            except Exception: pass

            log("")
            log("  ENVIRONMENT OVERRIDES (SPINESURG_*, LSTV_*, nnUNet_*)")
            env_keys = sorted(
                k for k in os.environ
                if k.startswith("SPINESURG_") or k.startswith("LSTV_")
                or k in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results")
            )
            if not env_keys:
                log("    <none set; all defaults>")
            for k in env_keys:
                v = os.environ[k]
                if len(v) > 100: v = v[:97] + "..."
                log(f"    {k} = {v}")

            log("")
            log("  LSTV CASES JSON")
            json_found = False
            try:
                for jp in _candidate_lstv_json_paths(ds_name):
                    log(f"    candidate path: {jp}  (exists={jp.exists()})")
                    if jp.exists() and not json_found:
                        try:
                            data = json.loads(jp.read_text())
                            schema = data.get("schema_version", "?")
                            n_cases = (
                                len(data.get("case_to_subtype", {}))
                                if int(schema) >= 3
                                else len(data.get("case_ids", {}))
                            )
                            log(f"    -> loaded: schema_version={schema}, "
                                f"{n_cases} cases")
                            sc = data.get("subtype_counts") or {}
                            if sc:
                                log(f"    -> subtype_counts (raw, in JSON): {dict(sc)}")
                            json_found = True
                        except Exception as exc:
                            log(f"    -> load FAILED: {exc}")
            except Exception as exc:
                log(f"    candidate-path enumeration failed: {exc}")
            if not json_found:
                log("    -> no JSON found; will fall back to manifests/L6 scan")

            log("")
            log("  OVERSAMPLE POOL")
            log(f"    code-resident _OVERSAMPLE_SUBTYPES = {_OVERSAMPLE_SUBTYPES}")
            try:
                frac = float(os.environ.get("LSTV_OVERSAMPLE_FRAC", "0.25"))
            except ValueError:
                frac = 0.25
            log(f"    LSTV_OVERSAMPLE_FRAC = {frac:.3f} (target queue fraction)")
            pool_ids = getattr(self, "_lstv_case_ids", None)
            if pool_ids is None:
                log("    pool not yet identified (oversample mixin pre-run)")
            else:
                log(f"    pool size: {len(pool_ids)} cases")
                if self._val_case_to_subtype:
                    by_sub = Counter(
                        self._val_case_to_subtype.get(c, "<unmapped>")
                        for c in pool_ids
                    )
                    log(f"    pool by subtype: {dict(by_sub)}")
                src = getattr(self, "_lstv_detection_source", None)
                if src: log(f"    detection source: {src}")
                examples = sorted(pool_ids)[:5]
                log(f"    sample case IDs (first 5 sorted): {examples}")

            try:
                loader = getattr(self, "dataloader_train", None)
                fold_keys = self._extract_loader_keys_for_diagnostic(loader)
                if fold_keys is not None and pool_ids is not None:
                    fold_set = set(fold_keys)
                    n_in_fold = len(fold_set)
                    pool_in_fold = fold_set & set(pool_ids)
                    natural_frac = len(pool_in_fold) / max(1, n_in_fold)
                    log("")
                    log("  TRAIN FOLD INTERSECTION")
                    log(f"    train keys in this fold: {n_in_fold}")
                    log(f"    LSTV pool members in fold: {len(pool_in_fold)} "
                        f"({natural_frac*100:.1f}% natural)")
                    if self._val_case_to_subtype and pool_in_fold:
                        in_fold_by_sub = Counter(
                            self._val_case_to_subtype.get(c, "<unmapped>")
                            for c in pool_in_fold
                        )
                        log(f"    in-fold pool by subtype: {dict(in_fold_by_sub)}")
                    if natural_frac < frac:
                        gap = frac - natural_frac
                        log(f"    -> oversampler will duplicate to "
                            f"close gap of {gap*100:.1f}pp")
                    else:
                        log("    -> natural frac already >= target; "
                            "oversampler will be a no-op")
            except Exception as exc:
                log(f"    fold intersection diagnostic failed: {exc}")

            # ── v18: PATCH BIAS section ──────────────────────────
            log("")
            log("  PATCH BIAS (v18 — class override, not monkey-patch)")
            bias_enabled = _env_truthy("SPINESURG_LSTV_BIAS_ENABLED", default=True)
            log(f"    enabled (env SPINESURG_LSTV_BIAS_ENABLED): {bias_enabled}")
            if not _LSTV_BIASED_LOADER_AVAILABLE:
                log(f"    ✗ LSTVBiasedDataLoader3D NOT IMPORTABLE")
                log(f"      import error: {_LSTV_LOADER_IMPORT_ERROR}")
                log(f"      bias is INACTIVE — nnU-Net default uniform sampling")
            elif bias_enabled:
                l6_prob = _env_float("SPINESURG_LSTV_BIAS_L6_PROB", 0.60)
                sac_prob = _env_float("SPINESURG_LSTV_BIAS_SACRUM_PROB", 0.50)
                log(f"    L6 prob (lumb cases):        {l6_prob:.2f}  "
                    f"[env SPINESURG_LSTV_BIAS_L6_PROB]")
                log(f"    sacrum prob (sacr_count):    {sac_prob:.2f}  "
                    f"[env SPINESURG_LSTV_BIAS_SACRUM_PROB]")
                log(f"    detection: from class_locations content alone "
                    f"(no JSON/case_id lookup)")
                # Walk to the underlying train loader to confirm class.
                # If the get_dataloaders override worked, this will be
                # LSTVBiasedDataLoader3D. If not, something is broken.
                loader = getattr(self, "dataloader_train", None)
                cls_name = "<no train loader>"
                if loader is not None:
                    underlying = loader
                    for attr in ("generator", "data_loader", "_data_loader"):
                        cand = getattr(underlying, attr, None)
                        if cand is not None and (
                            hasattr(cand, "generate_train_batch")
                            or hasattr(cand, "_data")
                        ):
                            underlying = cand
                            break
                    cls_name = type(underlying).__name__
                log(f"    train loader class: {cls_name}")
                if cls_name == "LSTVBiasedDataLoader3D":
                    log(f"    ✓ bias is ACTIVE — get_bbox will set "
                        f"overwrite_class on lumb / sacr_count signatures")
                else:
                    log(f"    ✗ WARNING: train loader is {cls_name!r}, "
                        f"expected LSTVBiasedDataLoader3D.")
                    log(f"      bias is NOT active. Check that "
                        f"_LSTVOversampleMixin.get_dataloaders ran "
                        f"(it should have been called via MRO from "
                        f"super().on_train_start()).")
                # v19.1: report shared-counter status
                if _read_shared_bias_counters is not None:
                    try:
                        startup_counters = _read_shared_bias_counters()
                        if startup_counters:
                            log(f"    shared bias counters at startup: "
                                f"force_fg={startup_counters.get('n_force_fg', 0)}, "
                                f"L6={startup_counters.get('n_l6_returned', 0)}, "
                                f"sacrum={startup_counters.get('n_sacrum_returned', 0)}")
                            log(f"      (non-zero values here indicate "
                                f"resumed training; per-epoch deltas are "
                                f"computed against epoch-start snapshots)")
                        else:
                            log(f"    shared counters: read returned empty "
                                f"(multiprocessing.Value setup may have failed)")
                    except Exception as exc:
                        log(f"    shared counters: probe failed: {exc}")
                else:
                    log(f"    shared counters: not importable (per-epoch "
                        f"counter logging will be silent)")
            else:
                log(f"    bias DISABLED via env; "
                    f"nnU-Net default uniform sampling")

            log("")
            log("  PER-SUBGROUP VAL DICE")
            log(f"    enabled: {self._per_subgroup_logging_enabled}")
            if self._val_case_to_subtype is not None:
                log(f"    subtype map size: {len(self._val_case_to_subtype)} cases")
                map_counts = Counter(self._val_case_to_subtype.values())
                log(f"    subtype distribution: {dict(map_counts)}")
            log(f"    classes tracked: {_ANATOMY_NAMES}")
            # v17.1: dedicated LSTV val pass status
            try:
                ddv_enabled, ddv_every = self._resolve_lstv_dedicated_val_settings()
                ddv_str = f"every {ddv_every} epoch(s)" if ddv_enabled else "DISABLED"
                log(f"    dedicated LSTV val pass: {ddv_str} "
                    f"(env: SPINESURG_LSTV_DEDICATED_VAL, "
                    f"SPINESURG_LSTV_DEDICATED_VAL_EVERY)")
                # Show which val cases would be hit (best-effort; some
                # frameworks don't expose val keys at banner time).
                if ddv_enabled:
                    result = self._find_underlying_val_loader()
                    if (isinstance(result, tuple)
                            and len(result) == 4
                            and result[1] is not None
                            and result[2] is not None):
                        underlying, keys_owner, keys_attr, is_callable = result
                        try:
                            raw = getattr(keys_owner, keys_attr)
                            vk = list(raw() if is_callable else raw)
                            attr_kind = "method" if is_callable else "attr"
                            log(f"    val keys source: "
                                f"{type(keys_owner).__name__}.{keys_attr} "
                                f"[{attr_kind}], len={len(vk)}")
                            lstv_subs = set(_OVERSAMPLE_SUBTYPES)
                            lstv_in_val = [
                                k for k in vk
                                if (self._val_case_to_subtype or {}).get(
                                    k, _SUBTYPE_NORMAL) in lstv_subs
                            ]
                            if lstv_in_val and self._val_case_to_subtype:
                                by_sub = Counter(
                                    self._val_case_to_subtype.get(k, "?")
                                    for k in lstv_in_val
                                )
                                log(f"    LSTV val cases this fold: "
                                    f"{len(lstv_in_val)} total -> "
                                    f"{dict(by_sub)}")
                                if is_callable:
                                    log("    NOTE: keys attr is a method; "
                                        "dedicated pass will run in "
                                        "pass-through mode (regular val "
                                        "sampler still produces metrics).")
                            else:
                                log("    LSTV val cases this fold: 0 "
                                    "(dedicated pass will be a no-op)")
                        except Exception as exc:
                            log(f"    val keys probe failed: {exc}")
                    else:
                        log("    val keys probe: no usable list found "
                            "(dedicated pass will be a no-op)")
            except Exception as exc:
                log(f"    dedicated-val settings resolve failed: {exc}")

            ce_w = getattr(self, "_ce_class_weights", None)
            if ce_w is not None:
                log("")
                log("  CE CLASS REWEIGHTING")
                try:
                    ce_str = ", ".join(
                        f"{_ANATOMY_NAMES[i-1] if 1 <= i <= 9 else 'bg' if i==0 else f'lab{i}'}"
                        f"={float(w):.2f}"
                        for i, w in enumerate(ce_w)
                    )
                    log(f"    weights: [{ce_str}]")
                except Exception:
                    log(f"    weights: {ce_w}")
            else:
                log("")
                log("  CE CLASS REWEIGHTING: not visible at banner time "
                    "(check the 'CE reweighting: ENABLED' line earlier in log)")

            log("")
            log("  WANDB")
            wb_run = getattr(self, "_wandb_run", None)
            if wb_run is not None:
                try:
                    log(f"    run: {wb_run.name}  id: {wb_run.id}")
                    log(f"    project: {wb_run.project}")
                except Exception:
                    log(f"    run: {wb_run}")
            else:
                log("    no W&B run active")

            log("=" * 72)
            log("END STARTUP DIAGNOSTICS")
            log("=" * 72)
            log("")
        except Exception as exc:
            try: self.print_to_log_file(f"startup diagnostics: failed: {exc}")
            except Exception: pass

    @staticmethod
    def _extract_loader_keys_for_diagnostic(loader) -> Optional[List[str]]:
        """Best-effort key extraction; returns None if framework
        internals don't expose a key list. Mirrors the attribute-walk
        used by `_apply_lstv_sampler`."""
        if loader is None: return None
        for path in (("generator", "_data", "identifiers"),
                      ("generator", "_data", "keys"),
                      ("_data", "identifiers"), ("_data", "keys"),
                      ("data_loader", "_data", "identifiers"),
                      ("data_loader", "_data", "keys")):
            obj = loader
            for p in path[:-1]:
                obj = getattr(obj, p, None)
                if obj is None: break
            else:
                cand = getattr(obj, path[-1], None)
                if callable(cand):
                    try: cand = list(cand())
                    except Exception: continue
                else:
                    try: cand = list(cand)
                    except Exception: continue
                if cand and all(isinstance(k, str) for k in cand):
                    return cand
        return None

    # ── Dedicated LSTV validation pass (v17.1) ───────────────────────────
    # nnU-Net v2's regular val loop subsamples the val set
    # (num_val_iterations_per_epoch × batch_size = 100 cases by default,
    # vs ~190 in a fold's val set). For the dataset's natural ~3% LSTV
    # base rate, this means many epochs see zero LSTV val cases by luck
    # of the draw — the per-subgroup metric is silenced when it's
    # supposed to be the headline. The dedicated pass below runs ONE
    # extra forward pass on every LSTV case in this fold's val set,
    # every epoch (configurable via SPINESURG_LSTV_DEDICATED_VAL_EVERY).
    #
    # Cost: ~N_lstv_val * batch_size * forward_time. For 8 LSTV cases
    # and bs=2 on H200 with the 256x320x320 patch, ≈30s per epoch.
    # Cheaper than bumping num_val_iterations_per_epoch to cover the
    # full fold (~70s extra) since most of those extra cases are
    # normals we already have plenty of.
    #
    # The dedicated pass REPLACES (not appends to) any LSTV results
    # the regular sampler may have caught this epoch — clears LSTV
    # subtype slots in the buffer first, then forward-passes each
    # LSTV case once. Normal results from the regular sampler are
    # preserved untouched.

    _lstv_dedicated_val_enabled: bool = True

    def _resolve_lstv_dedicated_val_settings(self) -> Tuple[bool, int]:
        """Return (enabled, every_n_epochs). Read once per epoch."""
        enabled = _env_truthy("SPINESURG_LSTV_DEDICATED_VAL", default=True)
        every = max(1, _env_int("SPINESURG_LSTV_DEDICATED_VAL_EVERY", 1))
        return enabled, every

    def _find_underlying_val_loader(self):
        """Walk through dataloader_val's wrappers to find the actual
        sampling list. Returns (underlying_loader, keys_owner,
        keys_attr_name, is_callable) or (None, None, None, False) if
        not found.

        v17.2 changes from v17.1:
          - Prefer ``loader.indices`` and ``loader.list_of_keys`` over
            ``loader._data.identifiers`` / ``_data.keys``. The former are
            what nnUNetDataLoader3D actually samples from in
            ``get_indices()``; modifying them constrains the sampler.
            Modifying ``_data.identifiers`` after construction may
            be a no-op if the loader already snapshotted the list.
          - Returns an extra ``is_callable`` flag so callers know
            whether ``getattr(keys_owner, keys_attr)`` returns a
            method (call it) or a value (use directly).
        """
        loader = getattr(self, "dataloader_val", None)
        if loader is None: return None, None, None, False
        underlying = loader
        for attr in ("generator", "data_loader", "_data_loader"):
            cand = getattr(underlying, attr, None)
            if cand is not None and (hasattr(cand, "generate_train_batch")
                                       or hasattr(cand, "_data")):
                underlying = cand
                break
        if underlying is None or not hasattr(underlying, "generate_train_batch"):
            return None, None, None, False

        # Try in order: attributes the loader itself uses for sampling
        # (these are the right place to modify), then the dataset's
        # identifier list (modify-then-sample only works if the loader
        # re-reads on each batch — usually not the case, but try as
        # last resort).
        candidates: List[Tuple[object, str]] = []
        # Loader-level sample lists (highest priority).
        for ka in ("indices", "list_of_keys", "_indices"):
            if hasattr(underlying, ka):
                candidates.append((underlying, ka))
        # Dataset-level identifier lists (lower priority).
        data_attr = getattr(underlying, "_data", None)
        if data_attr is not None:
            for ka in ("identifiers", "keys"):
                if hasattr(data_attr, ka):
                    candidates.append((data_attr, ka))

        for owner, ka in candidates:
            try:
                raw = getattr(owner, ka)
            except Exception:
                continue
            try:
                if callable(raw):
                    materialized = list(raw())
                    is_callable = True
                else:
                    materialized = list(raw)
                    is_callable = False
            except Exception:
                continue
            if materialized and all(isinstance(k, str) for k in materialized):
                return underlying, owner, ka, is_callable
        return underlying, None, None, False

    def _validate_lstv_dedicated(self) -> None:
        """Forward-pass every LSTV case in this fold's val set and record
        per-case dice into _val_subgroup_dice. Called from on_epoch_end
        AFTER the regular val loop completes and BEFORE per-subgroup
        aggregation — so the buffer reflects the dedicated-pass results
        when _aggregate_subgroup_dice runs.
        """
        if not self._per_subgroup_logging_enabled: return
        enabled, every = self._resolve_lstv_dedicated_val_settings()
        if not enabled: return

        epoch = getattr(self, "current_epoch", 0)
        if (epoch % every) != 0:
            # Skip this epoch — leave whatever the regular sampler
            # caught (may be n=0; that's the price of skipping).
            return

        if self._val_case_to_subtype is None:
            return
        if self._val_subgroup_dice is None:
            self._reset_subgroup_dice_buffer()

        result = self._find_underlying_val_loader()
        if not isinstance(result, tuple) or len(result) != 4:
            self.print_to_log_file(
                "LSTV dedicated val: _find_underlying_val_loader returned "
                "unexpected shape; skipping.")
            return
        underlying, keys_owner, keys_attr, is_callable = result
        if underlying is None or keys_owner is None:
            self.print_to_log_file(
                "LSTV dedicated val: cannot locate val dataloader internals; "
                "skipping (regular val sampler still active).")
            return

        # v17.2: read the keys list, handling both attribute and method
        # cases. The is_callable flag came from _find_underlying_val_loader
        # which already validated the read works.
        try:
            raw = getattr(keys_owner, keys_attr)
            if is_callable:
                full_keys = list(raw())
            else:
                full_keys = list(raw)
        except Exception as exc:
            self.print_to_log_file(f"LSTV dedicated val: keys read failed: {exc}")
            return

        # Filter val keys to LSTV pool members only
        lstv_subtypes = set(_OVERSAMPLE_SUBTYPES)
        lstv_in_val = [
            k for k in full_keys
            if self._val_case_to_subtype.get(k, _SUBTYPE_NORMAL) in lstv_subtypes
        ]
        if not lstv_in_val:
            # No LSTV cases in this fold's val set — silently no-op
            # (no point logging every epoch on folds without LSTV val).
            return

        # If the keys attribute is a method (e.g., dict.keys), we
        # cannot safely setattr to constrain it to one case — that
        # would replace the method with a list and break subsequent
        # callers. Skip the dedicated forward pass and just log a
        # diagnostic. The normal val sampler still produces metrics
        # (sub-sampled).
        if is_callable:
            self.print_to_log_file(
                f"LSTV dedicated val: keys attribute "
                f"{type(keys_owner).__name__}.{keys_attr} is a method, "
                f"not a settable list; cannot constrain val sampler. "
                f"Found {len(lstv_in_val)} LSTV cases in val but skipping "
                f"forced forward passes (regular val sampler will still "
                f"catch some). Pass-through mode.")
            return

        # Clear LSTV slots — the dedicated pass replaces the regular
        # sampler's contribution for these subtypes. Normal slot
        # untouched.
        for sub in lstv_subtypes:
            self._val_subgroup_dice[sub] = []

        net = getattr(self, "network", None)
        if net is None:
            self.print_to_log_file(
                "LSTV dedicated val: self.network is None; skipping")
            return
        device = getattr(self, "device", torch.device("cuda"))

        # For each LSTV case, force the loader to see only that case
        # then pull one batch — which will be batch_size patches all
        # cropped from the same case. Forward-pass and record.
        n_attempted = 0
        n_succeeded = 0
        was_training = net.training
        net.eval()
        try:
            for cid in lstv_in_val:
                n_attempted += 1
                try:
                    setattr(keys_owner, keys_attr, [cid])
                    batch = underlying.generate_train_batch()
                    if not isinstance(batch, dict) or "data" not in batch:
                        continue
                    data = batch["data"]
                    if not isinstance(data, torch.Tensor):
                        # Numpy → Tensor
                        try: data = torch.as_tensor(data)
                        except Exception: continue
                    if data.device != device:
                        data = data.to(device, non_blocking=True)
                    with torch.no_grad():
                        with torch.amp.autocast(
                            device_type=device.type if device is not None else "cuda",
                            enabled=True
                        ):
                            output = net(data)
                    # Force the batch's keys list to this case so the
                    # per-subgroup logger tags the dice correctly.
                    batch["keys"] = [cid] * (data.shape[0] if data.dim() >= 1 else 1)
                    self._record_per_case_dice(batch, output)
                    n_succeeded += 1
                except Exception as exc:
                    try:
                        self.print_to_log_file(
                            f"LSTV dedicated val: case {cid} failed: {exc}")
                    except Exception: pass
        finally:
            # Always restore full key list even if something raised
            try:
                setattr(keys_owner, keys_attr, full_keys)
            except Exception: pass
            if was_training:
                net.train()

        try:
            self.print_to_log_file(
                f"LSTV dedicated val (epoch {epoch}): "
                f"forward-passed {n_succeeded}/{n_attempted} LSTV val cases")
        except Exception: pass

    # ── Per-subgroup setup ───────────────────────────────────────────────

    def _maybe_load_subtype_map(self):
        self._phase_begin("subtype_map")
        self._per_subgroup_logging_enabled = _env_truthy(
            "SPINESURG_LSTV_PER_GROUP_DICE", default=True)
        if not self._per_subgroup_logging_enabled:
            self._phase("subtype_map",
                "DISABLED via SPINESURG_LSTV_PER_GROUP_DICE=0")
            self._phase_end("subtype_map", ok=True, note="disabled")
            return
        if self._val_case_to_subtype is not None:
            self._phase("subtype_map",
                f"already loaded ({len(self._val_case_to_subtype)} cases)")
            self._phase_end("subtype_map", ok=True, note="cached")
            return

        try: ds_name = self.plans_manager.dataset_name
        except Exception: ds_name = None
        self._phase("subtype_map", f"dataset_name = {ds_name!r}")

        case_to_sub: Dict[str, str] = {}
        loaded_from = None
        if ds_name:
            for jp in _candidate_lstv_json_paths(ds_name):
                self._phase("subtype_map",
                    f"probe lstv_cases.json: {jp}  exists={jp.exists()}")
                result = _read_lstv_cases_json(jp)
                if result is not None:
                    _, subtypes, case_to_sub, _ = result
                    self._phase("subtype_map",
                        f"loaded {len(case_to_sub)} cases from {jp}")
                    self._phase("subtype_map",
                        f"subtype counts: {dict(subtypes)}")
                    loaded_from = str(jp)
                    break
        if not case_to_sub:
            for hf_dir in _candidate_hf_export_dirs():
                self._phase("subtype_map",
                    f"probe HF manifest dir: {hf_dir}  "
                    f"exists={hf_dir.is_dir()}")
                result = _read_lstv_records_from_manifest(hf_dir)
                if result is not None:
                    _, subtypes, case_to_sub, _ = result
                    self._phase("subtype_map",
                        f"loaded {len(case_to_sub)} cases from manifests at "
                        f"{hf_dir}")
                    self._phase("subtype_map",
                        f"subtype counts: {dict(subtypes)}")
                    loaded_from = f"manifest:{hf_dir}"
                    break
        self._val_case_to_subtype = case_to_sub
        if not case_to_sub:
            self._phase("subtype_map",
                "no subtype map found; all cases will fall back to 'normal'")
            self._phase_end("subtype_map", ok=False, note="no map found")
            return

        # Show oversample-pool composition.
        n_oversample = sum(1 for s in case_to_sub.values()
                            if s in _OVERSAMPLE_SUBTYPES)
        self._phase("subtype_map",
            f"oversample-pool size: {n_oversample} cases "
            f"(eligible subtypes: {sorted(_OVERSAMPLE_SUBTYPES)})")
        # Show a few example case_ids per subtype for sanity-checking.
        for sub in _KNOWN_SUBTYPES:
            ids_for_sub = [cid for cid, s in case_to_sub.items() if s == sub]
            if ids_for_sub:
                preview = ", ".join(ids_for_sub[:3])
                if len(ids_for_sub) > 3:
                    preview += f", ... ({len(ids_for_sub)} total)"
                self._phase("subtype_map", f"  {sub:<20s} -> {preview}")
        self._phase_end("subtype_map", ok=True, note=f"source={loaded_from}")

    def _reset_subgroup_dice_buffer(self):
        self._val_subgroup_dice = {sub: [] for sub in _KNOWN_SUBTYPES}
        # v17.4: hallucination tracking buffer, parallel to dice buffer
        self._val_subgroup_voxels = {sub: [] for sub in _KNOWN_SUBTYPES}
        # v19: bidirectional confusion buffers
        self._val_subgroup_confusion_by_gt = {sub: [] for sub in _KNOWN_SUBTYPES}
        self._val_subgroup_confusion_by_pred = {sub: [] for sub in _KNOWN_SUBTYPES}

    def _subtype_for_case(self, case_id: str) -> str:
        if not self._val_case_to_subtype: return _SUBTYPE_NORMAL
        return self._val_case_to_subtype.get(case_id, _SUBTYPE_NORMAL)

    def _record_per_case_dice(self, batch, output):
        if not self._per_subgroup_logging_enabled: return
        if self._val_subgroup_dice is None: self._reset_subgroup_dice_buffer()
        try:
            pred_logits = output[0] if isinstance(output, (list, tuple)) else output
            gt = batch.get("target")
            if isinstance(gt, (list, tuple)): gt = gt[0]
            # Explicit None checks: batch["keys"] / batch["identifier"]
            # may be numpy arrays, and `arr or fallback` raises
            # "truth value of an array with more than one element is
            # ambiguous". The previous chained `or` worked only when
            # batch.get returned a Python list.
            keys_raw = batch.get("keys")
            if keys_raw is None:
                keys_raw = batch.get("identifier")
            if keys_raw is None:
                keys_raw = []
            if isinstance(keys_raw, str): keys = [keys_raw]
            elif hasattr(keys_raw, "tolist"): keys = [str(k) for k in keys_raw.tolist()]
            else:
                try: keys = [str(k) for k in keys_raw]
                except TypeError: keys = [str(keys_raw)]
            if pred_logits is None or gt is None or len(keys) == 0: return
            if pred_logits.dim() != 5: return
            if gt.dim() == 5 and gt.size(1) == 1: gt = gt[:, 0]
            elif gt.dim() != 4: return
            with torch.no_grad():
                pred_argmax = pred_logits.argmax(dim=1)
            B = pred_argmax.size(0)
            if len(keys) != B: return
            for b in range(B):
                cid = str(keys[b])
                cid_lookup = cid[:-5] if cid.endswith("_0000") else cid
                subtype = self._subtype_for_case(cid_lookup)
                pred_b = pred_argmax[b].cpu().to(torch.int16)
                gt_b   = gt[b].cpu().to(torch.int16)
                dice_per_cls = _per_case_dice_per_class(pred_b, gt_b)
                # v17.4: also record voxel counts for hallucination tracking
                voxels_per_cls = _per_case_voxel_counts_per_class(pred_b, gt_b)
                # v19: bidirectional confusion vectors
                confusion_by_gt = _per_case_confusion_by_gt(pred_b, gt_b)
                confusion_by_pred = _per_case_confusion_by_pred(pred_b, gt_b)
                self._val_subgroup_dice.setdefault(subtype, []).append(dice_per_cls)
                self._val_subgroup_voxels.setdefault(subtype, []).append(voxels_per_cls)
                if self._val_subgroup_confusion_by_gt is None:
                    self._val_subgroup_confusion_by_gt = {s: [] for s in _KNOWN_SUBTYPES}
                if self._val_subgroup_confusion_by_pred is None:
                    self._val_subgroup_confusion_by_pred = {s: [] for s in _KNOWN_SUBTYPES}
                self._val_subgroup_confusion_by_gt.setdefault(subtype, []).append(confusion_by_gt)
                self._val_subgroup_confusion_by_pred.setdefault(subtype, []).append(confusion_by_pred)
        except Exception as exc:
            try: self.print_to_log_file(f"per-subgroup dice: batch failed: {exc}")
            except Exception: pass

    def _aggregate_subgroup_dice(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if not self._per_subgroup_logging_enabled or self._val_subgroup_dice is None:
            return out
        for subtype, cases in self._val_subgroup_dice.items():
            n = len(cases)
            out[f"val/lstv_subgroup/{subtype}/n_cases"] = float(n)
            if n == 0: continue
            per_class_means: Dict[int, float] = {}
            for ci, cls_id in enumerate(_FG_CLASS_IDS):
                vals = [c[ci] for c in cases if c[ci] is not None]
                if not vals: continue
                m = float(np.mean(vals))
                per_class_means[cls_id] = m
                out[f"val/lstv_subgroup/{subtype}/dice/{_ANATOMY_NAMES[ci]}"] = m
                out[f"val/lstv_subgroup/{subtype}/n_with_class/{_ANATOMY_NAMES[ci]}"] = float(len(vals))
            lumbar_means = [per_class_means[c] for c in _LUMBAR_IDS if c in per_class_means]
            pelvis_means = [per_class_means[c] for c in _PELVIS_IDS if c in per_class_means]
            fg_means = list(per_class_means.values())
            if lumbar_means: out[f"val/lstv_subgroup/{subtype}/mean_lumbar_dice"] = float(np.mean(lumbar_means))
            if pelvis_means: out[f"val/lstv_subgroup/{subtype}/mean_pelvis_dice"] = float(np.mean(pelvis_means))
            if fg_means:     out[f"val/lstv_subgroup/{subtype}/mean_fg_dice"] = float(np.mean(fg_means))

        lumb_cases = self._val_subgroup_dice.get(_SUBTYPE_LUMB, [])
        if lumb_cases:
            l6_idx = _FG_CLASS_IDS.index(_L6_LABEL_ID)
            l6_vals = [c[l6_idx] for c in lumb_cases if c[l6_idx] is not None]
            if l6_vals:
                out["val/headline/L6_dice_on_lumbarization"] = float(np.mean(l6_vals))
                out["val/headline/L6_n_lumbarization_cases"] = float(len(l6_vals))

        lstv_means: List[float] = []
        for sub in (_SUBTYPE_LUMB, _SUBTYPE_SACR_COUNT, _SUBTYPE_SEMI,
                     _SUBTYPE_SACR, _SUBTYPE_AMBIG):
            for case_dice in self._val_subgroup_dice.get(sub, []):
                vals = [v for v in case_dice if v is not None]
                if vals: lstv_means.append(float(np.mean(vals)))
        if lstv_means:
            out["val/lstv_subgroup/any_lstv/mean_fg_dice"] = float(np.mean(lstv_means))
            out["val/lstv_subgroup/any_lstv/n_cases"] = float(len(lstv_means))

        any_sacr_means: List[float] = []
        for sub in (_SUBTYPE_SACR_COUNT, _SUBTYPE_SEMI, _SUBTYPE_SACR):
            for case_dice in self._val_subgroup_dice.get(sub, []):
                vals = [v for v in case_dice if v is not None]
                if vals: any_sacr_means.append(float(np.mean(vals)))
        if any_sacr_means:
            out["val/lstv_subgroup/any_sacralization/mean_fg_dice"] = float(np.mean(any_sacr_means))
            out["val/lstv_subgroup/any_sacralization/n_cases"] = float(len(any_sacr_means))

        # v17.4: hallucination metrics (specificity / false-positive
        # tracking). Merge into the same payload dict.
        try:
            halluc_payload = self._aggregate_subgroup_hallucination()
            out.update(halluc_payload)
        except Exception as exc:
            try: self.print_to_log_file(f"hallucination aggregation: {exc}")
            except Exception: pass
        # v19: bidirectional confusion (where GT-class voxels go,
        # where PRED-class voxels come from)
        try:
            confusion_payload = self._aggregate_subgroup_confusion()
            out.update(confusion_payload)
        except Exception as exc:
            try: self.print_to_log_file(f"confusion aggregation: {exc}")
            except Exception: pass
        return out

    def _aggregate_subgroup_hallucination(self) -> Dict[str, float]:
        """v17.4: compute specificity and hallucination volume per
        (subgroup, class) pair from the voxel-count buffer.

        For each subgroup S and each foreground class C, partition cases
        into "absent" (gt_n == 0) and "present" (gt_n > 0). Then for
        absent cases compute:

            n_absent              : how many cases had no voxels of C in GT
            n_strict_halluc       : of those, how many had pred_n > 0
            n_meaningful_halluc   : of those, how many had pred_n > THRESHOLD
            mean_pred_when_absent : mean predicted voxel count over absent
            max_pred_when_absent  : max predicted voxel count over absent
            specificity           : 1 - n_meaningful_halluc / n_absent

        Specificity is the headline number: "given the class is NOT
        present in this anatomy variant, did the model correctly
        predict zero voxels (or noise-level voxels) of it?"

        Headline derived metrics (clinically curated):
            val/headline/L5_specificity_on_sacralization
            val/headline/L5_specificity_on_sacr_count
            val/headline/L5_specificity_on_semisacralization
            val/headline/L6_specificity_on_normal
            val/headline/L6_specificity_on_sacralization
            val/headline/L6_specificity_on_semisacralization
            val/headline/L6_specificity_on_sacr_count
        """
        out: Dict[str, float] = {}
        if self._val_subgroup_voxels is None:
            return out
        threshold = _HALLUC_VOXEL_THRESHOLD

        for subtype, cases in self._val_subgroup_voxels.items():
            if not cases:
                continue
            for ci, cls_id in enumerate(_FG_CLASS_IDS):
                cls_name = _ANATOMY_NAMES[ci]
                # Pull (gt_n, pred_n) for this class across all cases
                # in the subgroup.
                pairs = [c[ci] for c in cases]
                # Partition by GT presence.
                absent_preds = [p for (g, p) in pairs if g == 0]
                present_preds = [(g, p) for (g, p) in pairs if g > 0]

                # Per-class absent/present counts (always emitted; useful
                # for sanity-checking the partition in W&B).
                k = f"val/lstv_subgroup/{subtype}/halluc/{cls_name}"
                out[f"{k}/n_absent"]  = float(len(absent_preds))
                out[f"{k}/n_present"] = float(len(present_preds))

                if not absent_preds:
                    # Class is always present in GT for this subgroup —
                    # no specificity defined. Skip.
                    continue

                n_strict = sum(1 for p in absent_preds if p > 0)
                n_meaningful = sum(1 for p in absent_preds if p > threshold)
                mean_pred = float(np.mean(absent_preds)) if absent_preds else 0.0
                max_pred = float(max(absent_preds)) if absent_preds else 0.0
                specificity = 1.0 - (n_meaningful / len(absent_preds))

                out[f"{k}/n_strict_halluc"]      = float(n_strict)
                out[f"{k}/n_meaningful_halluc"]  = float(n_meaningful)
                out[f"{k}/mean_pred_when_absent"] = mean_pred
                out[f"{k}/max_pred_when_absent"]  = max_pred
                out[f"{k}/specificity"]          = specificity

        # ── Headline metrics: clinically meaningful (subgroup, class)
        # pairs where the class SHOULD NOT be present in GT. These are
        # the numbers that go into the paper table.
        headline_pairs = [
            ("L5", _SUBTYPE_SACR,        "L5_specificity_on_sacralization"),
            ("L5", _SUBTYPE_SACR_COUNT,  "L5_specificity_on_sacr_count"),
            ("L5", _SUBTYPE_SEMI,        "L5_specificity_on_semisacralization"),
            ("L6", _SUBTYPE_NORMAL,      "L6_specificity_on_normal"),
            ("L6", _SUBTYPE_SACR,        "L6_specificity_on_sacralization"),
            ("L6", _SUBTYPE_SEMI,        "L6_specificity_on_semisacralization"),
            ("L6", _SUBTYPE_SACR_COUNT,  "L6_specificity_on_sacr_count"),
        ]
        for cls_name, subtype, headline_key in headline_pairs:
            spec_key = f"val/lstv_subgroup/{subtype}/halluc/{cls_name}/specificity"
            n_abs_key = f"val/lstv_subgroup/{subtype}/halluc/{cls_name}/n_absent"
            mean_key = f"val/lstv_subgroup/{subtype}/halluc/{cls_name}/mean_pred_when_absent"
            if spec_key in out:
                out[f"val/headline/{headline_key}"] = out[spec_key]
                out[f"val/headline/{headline_key}_n_cases"] = out[n_abs_key]
                out[f"val/headline/{headline_key}_mean_voxels"] = out[mean_key]
        return out

    def _aggregate_subgroup_confusion(self) -> Dict[str, float]:
        """v19: aggregate bidirectional confusion across cases per subgroup.

        Emits two families of keys per (subgroup, class) pair that has
        any voxels:

          GT-DIRECTION ("where do voxels of this GT class get predicted?"):
            val/lstv_subgroup/{sub}/conf_by_gt/{gt_name}/total_voxels
            val/lstv_subgroup/{sub}/conf_by_gt/{gt_name}/as_{pred_name}_frac
            val/lstv_subgroup/{sub}/conf_by_gt/{gt_name}/as_{pred_name}_voxels

          PRED-DIRECTION ("when the model predicted this class, what
          was the GT?"):
            val/lstv_subgroup/{sub}/conf_by_pred/{pred_name}/total_voxels
            val/lstv_subgroup/{sub}/conf_by_pred/{pred_name}/actually_{gt_name}_frac
            val/lstv_subgroup/{sub}/conf_by_pred/{pred_name}/actually_{gt_name}_voxels

        Headline keys (the clinically interesting ones for the paper):

          GT-direction on lumb (where do L6 voxels go?):
            val/headline/L6_GT_predicted_as_{name}_on_lumb
            val/headline/L6_GT_total_voxels_on_lumb

          PRED-direction on sacr_count (when model says L5 but L5 is
          absent, what's actually there?):
            val/headline/predicted_L5_actually_{name}_on_sacr_count
            val/headline/predicted_L5_total_voxels_on_sacr_count

          PRED-direction on normal (when model hallucinates L6 on a
          normal patient, what's at that location?):
            val/headline/predicted_L6_actually_{name}_on_normal
            val/headline/predicted_L6_total_voxels_on_normal

        Fractions are emitted only for predicted/actual classes with
        non-zero voxel counts, so the W&B run doesn't explode with 100
        zero-valued keys per epoch.
        """
        out: Dict[str, float] = {}
        if (self._val_subgroup_confusion_by_gt is None
                or self._val_subgroup_confusion_by_pred is None):
            return out

        # ── GT-direction aggregation ──────────────────────────────
        for subtype, cases in self._val_subgroup_confusion_by_gt.items():
            if not cases:
                continue
            # Sum per-case confusion vectors per GT class
            summed_by_gt: Dict[int, np.ndarray] = {}
            for case_dict in cases:
                for gt_cid, vec in case_dict.items():
                    if gt_cid not in summed_by_gt:
                        summed_by_gt[gt_cid] = np.zeros_like(vec, dtype=np.int64)
                    summed_by_gt[gt_cid] = summed_by_gt[gt_cid] + vec

            for gt_cid, vec in summed_by_gt.items():
                gt_name = _class_id_to_name(gt_cid)
                total = int(vec.sum())
                if total == 0:
                    continue
                base = f"val/lstv_subgroup/{subtype}/conf_by_gt/{gt_name}"
                out[f"{base}/total_voxels"] = float(total)
                for pred_cid in range(len(vec)):
                    n = int(vec[pred_cid])
                    if n == 0:
                        continue
                    pred_name = _class_id_to_name(pred_cid)
                    out[f"{base}/as_{pred_name}_frac"] = n / total
                    out[f"{base}/as_{pred_name}_voxels"] = float(n)

        # ── PRED-direction aggregation ────────────────────────────
        for subtype, cases in self._val_subgroup_confusion_by_pred.items():
            if not cases:
                continue
            summed_by_pred: Dict[int, np.ndarray] = {}
            for case_dict in cases:
                for pred_cid, vec in case_dict.items():
                    if pred_cid not in summed_by_pred:
                        summed_by_pred[pred_cid] = np.zeros_like(vec, dtype=np.int64)
                    summed_by_pred[pred_cid] = summed_by_pred[pred_cid] + vec

            for pred_cid, vec in summed_by_pred.items():
                pred_name = _class_id_to_name(pred_cid)
                total = int(vec.sum())
                if total == 0:
                    continue
                base = f"val/lstv_subgroup/{subtype}/conf_by_pred/{pred_name}"
                out[f"{base}/total_voxels"] = float(total)
                for gt_cid in range(len(vec)):
                    n = int(vec[gt_cid])
                    if n == 0:
                        continue
                    gt_name = _class_id_to_name(gt_cid)
                    out[f"{base}/actually_{gt_name}_frac"] = n / total
                    out[f"{base}/actually_{gt_name}_voxels"] = float(n)

        # ── Headline: L6 GT-direction on lumbarization ────────────
        # "Where do my L6 voxels end up being classified?"
        gt_base_lumb = f"val/lstv_subgroup/{_SUBTYPE_LUMB}/conf_by_gt/L6"
        total_l6 = out.get(f"{gt_base_lumb}/total_voxels")
        if total_l6 is not None and total_l6 > 0:
            out["val/headline/L6_GT_total_voxels_on_lumb"] = total_l6
            for pred_name in ("background", *_ANATOMY_NAMES):
                frac_key = f"{gt_base_lumb}/as_{pred_name}_frac"
                if frac_key in out:
                    out[f"val/headline/L6_GT_predicted_as_{pred_name}_on_lumb"] = out[frac_key]

        # ── Headline: L5 PRED-direction on sacr_count ────────────
        # "When the model predicts L5 on a sacr_count case (where L5
        # should be ABSENT in GT), what is it actually looking at?"
        pred_base_sc = f"val/lstv_subgroup/{_SUBTYPE_SACR_COUNT}/conf_by_pred/L5"
        total_pred_l5_sc = out.get(f"{pred_base_sc}/total_voxels")
        if total_pred_l5_sc is not None and total_pred_l5_sc > 0:
            out["val/headline/predicted_L5_total_voxels_on_sacr_count"] = total_pred_l5_sc
            for gt_name in ("background", *_ANATOMY_NAMES):
                frac_key = f"{pred_base_sc}/actually_{gt_name}_frac"
                if frac_key in out:
                    out[f"val/headline/predicted_L5_actually_{gt_name}_on_sacr_count"] = out[frac_key]

        # ── Headline: L6 PRED-direction on normal ────────────────
        # "When the model predicts L6 on a normal patient (where L6
        # doesn't exist anatomically), what's actually there?"
        pred_base_n = f"val/lstv_subgroup/{_SUBTYPE_NORMAL}/conf_by_pred/L6"
        total_pred_l6_n = out.get(f"{pred_base_n}/total_voxels")
        if total_pred_l6_n is not None and total_pred_l6_n > 0:
            out["val/headline/predicted_L6_total_voxels_on_normal"] = total_pred_l6_n
            for gt_name in ("background", *_ANATOMY_NAMES):
                frac_key = f"{pred_base_n}/actually_{gt_name}_frac"
                if frac_key in out:
                    out[f"val/headline/predicted_L6_actually_{gt_name}_on_normal"] = out[frac_key]

        return out

    def _print_per_subgroup_summary(self, epoch, payload):
        try:
            self.print_to_log_file(f"--- LSTV per-subgroup val dice (epoch {epoch}) ---")
            for sub in _KNOWN_SUBTYPES:
                n = payload.get(f"val/lstv_subgroup/{sub}/n_cases")
                if n is None or n == 0:
                    self.print_to_log_file(f"  {sub:<20s} (n=0)  no val cases this epoch")
                    continue
                def _fmt(k):
                    v = payload.get(k); return f"{v:.3f}" if v is not None else "  n/a"
                self.print_to_log_file(
                    f"  {sub:<20s} (n={int(n)})  "
                    f"mean_fg={_fmt(f'val/lstv_subgroup/{sub}/mean_fg_dice')}  "
                    f"lumb={_fmt(f'val/lstv_subgroup/{sub}/mean_lumbar_dice')}  "
                    f"pelvis={_fmt(f'val/lstv_subgroup/{sub}/mean_pelvis_dice')}")
            for sub in (_SUBTYPE_LUMB, _SUBTYPE_SACR_COUNT, _SUBTYPE_SEMI,
                         _SUBTYPE_SACR, _SUBTYPE_AMBIG):
                n = payload.get(f"val/lstv_subgroup/{sub}/n_cases")
                if not n or int(n) == 0: continue
                cls_parts = []
                for cn in _ANATOMY_NAMES:
                    v = payload.get(f"val/lstv_subgroup/{sub}/dice/{cn}")
                    cls_parts.append(f"{cn}={v:.2f}" if v is not None else f"{cn}=---")
                self.print_to_log_file(f"  {sub} per-class: [{', '.join(cls_parts)}]")
            l6 = payload.get("val/headline/L6_dice_on_lumbarization")
            l6n = payload.get("val/headline/L6_n_lumbarization_cases")
            if l6 is not None and l6n is not None:
                self.print_to_log_file(
                    f"  HEADLINE: L6 dice on lumbarization cases = {l6:.3f}  (n={int(l6n)})")

            # v17.4: hallucination headline lines. Specificity = fraction
            # of cases where class is absent in GT and model correctly
            # predicted ≤ 100 voxels of it (clinically: noise floor).
            # Higher is better; 1.0 = no hallucination.
            self.print_to_log_file(
                f"  --- v17.4 hallucination (specificity = fraction of "
                f"absent-cases with pred ≤ 100 voxels) ---")
            halluc_headlines = [
                ("L5_specificity_on_sacralization",       "L5 on sacralization     "),
                ("L5_specificity_on_sacr_count",          "L5 on sacr_count        "),
                ("L5_specificity_on_semisacralization",   "L5 on semisacralization "),
                ("L6_specificity_on_normal",              "L6 on normal            "),
                ("L6_specificity_on_sacralization",       "L6 on sacralization     "),
                ("L6_specificity_on_semisacralization",   "L6 on semisacralization "),
                ("L6_specificity_on_sacr_count",          "L6 on sacr_count        "),
            ]
            any_halluc_line = False
            for key, label in halluc_headlines:
                spec = payload.get(f"val/headline/{key}")
                n = payload.get(f"val/headline/{key}_n_cases")
                mean_vox = payload.get(f"val/headline/{key}_mean_voxels")
                if spec is None or n is None or int(n) == 0:
                    continue
                any_halluc_line = True
                mean_str = (f", mean_pred={mean_vox:.0f} voxels"
                            if mean_vox is not None else "")
                self.print_to_log_file(
                    f"    {label} specificity={spec:.3f}  "
                    f"(n_absent={int(n)}{mean_str})")
            if not any_halluc_line:
                self.print_to_log_file("    (no eligible (subgroup, class) pairs this epoch)")

            # v19: bidirectional confusion blocks. Show the three
            # clinically headline (subgroup, class) pairs in both
            # directions. Each block lists predicted-class fractions
            # >= 1% so noise-level entries don't clutter the log.
            self.print_to_log_file(
                "  --- v19 confusion (where misclassifications go / come from) ---"
            )

            def _print_distribution(header_line, total_key, frac_key_template):
                """Helper: print a header and the > 1% fraction lines for a
                given headline distribution. Returns True if anything was
                printed."""
                total = payload.get(total_key)
                if total is None or int(total) == 0:
                    return False
                self.print_to_log_file(header_line.format(total=int(total)))
                printed_any = False
                for cls_name in ("background", *_ANATOMY_NAMES):
                    frac = payload.get(frac_key_template.format(name=cls_name))
                    if frac is None:
                        continue
                    if frac < 0.01:
                        continue
                    self.print_to_log_file(
                        f"      {cls_name:<12s}: {frac*100:5.1f}%"
                    )
                    printed_any = True
                if not printed_any:
                    self.print_to_log_file("      (all predicted classes < 1%)")
                return True

            # Block 1: GT-direction on lumb. "Where do L6 GT voxels end up?"
            shown = _print_distribution(
                header_line="    L6 GT-voxels on lumb cases (total={total}, "
                            "where do they end up classified?):",
                total_key="val/headline/L6_GT_total_voxels_on_lumb",
                frac_key_template="val/headline/L6_GT_predicted_as_{name}_on_lumb",
            )
            if not shown:
                self.print_to_log_file(
                    "    L6 GT-voxels on lumb cases: (no lumb cases this epoch)"
                )

            # Block 2: PRED-direction on sacr_count. "When model says L5
            # on a sacr_count case (where L5 should be missing), what's
            # actually there?"
            shown = _print_distribution(
                header_line="    Predicted-L5 voxels on sacr_count cases "
                            "(total={total}, what's actually there in GT?):",
                total_key="val/headline/predicted_L5_total_voxels_on_sacr_count",
                frac_key_template="val/headline/predicted_L5_actually_{name}_on_sacr_count",
            )
            if not shown:
                self.print_to_log_file(
                    "    Predicted-L5 voxels on sacr_count cases: "
                    "(no sacr_count cases this epoch, or model predicted no L5 there)"
                )

            # Block 3: PRED-direction on normal. "When model hallucinates
            # L6 on a normal patient, what's at that location?"
            shown = _print_distribution(
                header_line="    Predicted-L6 voxels on normal cases "
                            "(total={total}, what's actually there in GT?):",
                total_key="val/headline/predicted_L6_total_voxels_on_normal",
                frac_key_template="val/headline/predicted_L6_actually_{name}_on_normal",
            )
            if not shown:
                self.print_to_log_file(
                    "    Predicted-L6 voxels on normal cases: "
                    "(model predicted no L6 on normals — good)"
                )
        except Exception as exc:
            try: self.print_to_log_file(f"per-subgroup summary failed: {exc}")
            except Exception: pass

    # ── v19.1: cross-process bias counter reporting ───────────────────────

    def _collect_bias_counter_payload(self, epoch: int) -> Dict[str, float]:
        """Read shared bias counters, diff against epoch-start snapshot,
        emit per-epoch + cumulative numbers as a W&B payload dict, and
        print a one-block summary to the SLURM log.

        Counters are populated from worker processes via shared memory
        in lstv_biased_dataloader. If the read returns nothing (e.g.
        the bias loader isn't installed, or shared counters failed at
        import time), this method silently returns an empty dict —
        callers should not crash when bias logging is unavailable.

        Returns the W&B payload dict (caller .update()s payload with it).
        """
        out: Dict[str, float] = {}
        if _read_shared_bias_counters is None:
            return out
        try:
            now = _read_shared_bias_counters()
        except Exception as exc:
            try: self.print_to_log_file(f"v19.1 read shared counters: {exc}")
            except Exception: pass
            return out
        if not now:
            return out

        start = self._bias_counters_at_epoch_start or {}

        # Per-epoch deltas
        n_fg_e = now.get("n_force_fg", 0) - start.get("n_force_fg", 0)
        n_l6_e = now.get("n_l6_returned", 0) - start.get("n_l6_returned", 0)
        n_sac_e = now.get("n_sacrum_returned", 0) - start.get("n_sacrum_returned", 0)
        n_uf_e = now.get("n_uniform_fallback", 0) - start.get("n_uniform_fallback", 0)

        out["patch_bias/this_epoch/n_force_fg"]         = float(n_fg_e)
        out["patch_bias/this_epoch/n_l6_returned"]      = float(n_l6_e)
        out["patch_bias/this_epoch/n_sacrum_returned"]  = float(n_sac_e)
        out["patch_bias/this_epoch/n_uniform_fallback"] = float(n_uf_e)

        # Per-epoch rates (only meaningful if force_fg fired at all
        # this epoch; otherwise the denominator is zero).
        if n_fg_e > 0:
            out["patch_bias/this_epoch/l6_rate"] = n_l6_e / n_fg_e
            out["patch_bias/this_epoch/sacrum_rate"] = n_sac_e / n_fg_e
            out["patch_bias/this_epoch/uniform_fallback_rate"] = n_uf_e / n_fg_e

        # Cumulative numbers
        out["patch_bias/cumulative/n_force_fg"]         = float(now.get("n_force_fg", 0))
        out["patch_bias/cumulative/n_l6_returned"]      = float(now.get("n_l6_returned", 0))
        out["patch_bias/cumulative/n_sacrum_returned"]  = float(now.get("n_sacrum_returned", 0))
        out["patch_bias/cumulative/n_uniform_fallback"] = float(now.get("n_uniform_fallback", 0))

        # Print a compact summary block to SLURM log. Same conventions
        # as the other per-epoch summaries.
        try:
            self.print_to_log_file(
                f"--- v19.1 patch bias activity (epoch {epoch}) ---"
            )
            if n_fg_e == 0:
                self.print_to_log_file(
                    "  this epoch: force_fg=0 (no force-foreground patches "
                    "served — check that bias is enabled and oversample mixin ran)"
                )
            else:
                self.print_to_log_file(
                    f"  this epoch: force_fg={n_fg_e}, "
                    f"L6 forces={n_l6_e}, sacrum forces={n_sac_e}, "
                    f"uniform fallback={n_uf_e}"
                )
                self.print_to_log_file(
                    f"  rates: L6={(n_l6_e/n_fg_e)*100:.2f}%, "
                    f"sacrum={(n_sac_e/n_fg_e)*100:.2f}%, "
                    f"no-bias={(n_uf_e/n_fg_e)*100:.2f}%"
                )
            self.print_to_log_file(
                f"  cumulative since training start: "
                f"force_fg={int(now.get('n_force_fg', 0))}, "
                f"L6={int(now.get('n_l6_returned', 0))}, "
                f"sacrum={int(now.get('n_sacrum_returned', 0))}"
            )
        except Exception:
            pass

        return out

    # ── train_step / validation_step ─────────────────────────────────────

    def train_step(self, batch):
        if self._channels_last_active and isinstance(batch, dict) and 'data' in batch:
            try:
                d = batch['data']
                if isinstance(d, torch.Tensor) and d.dim() == 5:
                    batch['data'] = d.contiguous(memory_format=torch.channels_last_3d)
            except Exception: pass
        result = super().train_step(batch)
        prof = getattr(self, "_profiler", None)
        if prof is not None:
            try:
                prof.step()
                self._profiler_step_count += 1
                if self._profiler_step_count >= self._profiler_total_steps:
                    self._stop_profiler()
            except Exception: pass
        return result

    def validation_step(self, batch):
        if self._channels_last_active and isinstance(batch, dict) and 'data' in batch:
            try:
                d = batch['data']
                if isinstance(d, torch.Tensor) and d.dim() == 5:
                    batch['data'] = d.contiguous(memory_format=torch.channels_last_3d)
            except Exception: pass
        if self._per_subgroup_logging_enabled:
            try:
                net = getattr(self, "network", None)
                if net is not None and isinstance(batch, dict) and 'data' in batch:
                    data = batch['data']
                    if isinstance(data, torch.Tensor):
                        device = getattr(self, "device", None)
                        if device is not None and data.device != device:
                            data = data.to(device, non_blocking=True)
                        net.eval()
                        with torch.no_grad():
                            with torch.amp.autocast(
                                device_type=device.type if device is not None else 'cuda',
                                enabled=True):
                                output = net(data)
                        self._record_per_case_dice(batch, output)
            except Exception as exc:
                try: self.print_to_log_file(f"per-subgroup pre-forward: {exc}")
                except Exception: pass
        return super().validation_step(batch)

    # ── W&B init / log / finish ──────────────────────────────────────────

    def _init_wandb(self):
        if self._wandb_run is not None: return
        if not os.environ.get("WANDB_API_KEY") and not _env_truthy("WANDB_ALLOW_OFFLINE", True):
            self.print_to_log_file("WandB: disabled (no key, offline disallowed)"); return
        try: import wandb
        except ImportError: self.print_to_log_file("WandB: not installed"); return
        try:
            run_name = os.environ.get("WANDB_RUN_NAME") or (
                f"{self.plans_manager.dataset_name}__{self.__class__.__name__}__"
                f"{self.plans_manager.plans_name}__f{self.fold}")
        except Exception:
            run_name = os.environ.get("WANDB_RUN_NAME") or self.__class__.__name__
        config = {
            "trainer": self.__class__.__name__,
            "anatomy_mapping": dict(enumerate(_ANATOMY_NAMES)),
            "subgroup_taxonomy": "6-way",
            "subgroups": list(_KNOWN_SUBTYPES),
            "oversample_subtypes": list(_OVERSAMPLE_SUBTYPES),
            "ce_reweight": _env_truthy("SPINESURG_CE_REWEIGHT", default=True),
            "lstv_oversample_frac": _env_float("LSTV_OVERSAMPLE_FRAC", 0.25),
            # v18: patch bias config recorded so W&B runs are
            # comparable across different bias settings.
            "patch_bias_enabled": _env_truthy("SPINESURG_LSTV_BIAS_ENABLED", default=True),
            "patch_bias_l6_prob": _env_float("SPINESURG_LSTV_BIAS_L6_PROB", 0.60),
            "patch_bias_sacrum_prob": _env_float("SPINESURG_LSTV_BIAS_SACRUM_PROB", 0.50),
            "patch_bias_loader_available": _LSTV_BIASED_LOADER_AVAILABLE,
        }
        for attr, getter in [
            ("dataset", lambda: self.plans_manager.dataset_name),
            ("plans", lambda: self.plans_manager.plans_name),
            ("fold", lambda: self.fold),
            ("num_epochs", lambda: self.num_epochs),
            ("batch_size", lambda: self.batch_size),
            ("patch_size", lambda: list(self.configuration_manager.patch_size)),
            ("iterations_per_epoch", lambda: self.num_iterations_per_epoch),
            ("val_iterations_per_epoch", lambda: self.num_val_iterations_per_epoch),
            ("initial_lr", lambda: self.initial_lr),
            ("weight_decay", lambda: self.weight_decay),
        ]:
            try: config[attr] = getter()
            except Exception: pass

        max_retries = _env_int("WANDB_INIT_MAX_RETRIES", 3)
        init_timeout = _env_int("WANDB_INIT_TIMEOUT_SEC", 120)
        allow_offline = _env_truthy("WANDB_ALLOW_OFFLINE", True)
        for attempt in range(1, max_retries + 1):
            try:
                settings = wandb.Settings(init_timeout=init_timeout) if hasattr(wandb, "Settings") else None
                init_kwargs = dict(
                    project=os.environ.get("WANDB_PROJECT", "SpineSurg-CT"),
                    entity=os.environ.get("WANDB_ENTITY") or None,
                    name=run_name, id=os.environ.get("WANDB_RUN_ID") or None,
                    resume="allow", config=config,
                )
                if settings is not None: init_kwargs["settings"] = settings
                self._wandb_run = wandb.init(**init_kwargs)
                self.print_to_log_file(f"WandB: initialized run {self._wandb_run.name}")
                return
            except Exception as exc:
                self.print_to_log_file(f"WandB attempt {attempt}: {exc}")
                if attempt < max_retries:
                    time.sleep(min(60, 2**attempt + random.uniform(0, 3)))
        if allow_offline:
            try:
                self.print_to_log_file("WandB: falling back to OFFLINE mode")
                os.environ["WANDB_MODE"] = "offline"
                self._wandb_run = wandb.init(
                    project=os.environ.get("WANDB_PROJECT", "SpineSurg-CT"),
                    entity=os.environ.get("WANDB_ENTITY") or None,
                    name=run_name, id=os.environ.get("WANDB_RUN_ID") or None,
                    resume="allow", config=config, mode="offline")
            except Exception as exc:
                self.print_to_log_file(f"WandB offline failed: {exc}")
                self._wandb_run = None

    def _log_wandb(self, data, step=None):
        if self._wandb_run is None: return
        clean = {k: v for k, v in data.items() if v is not None}
        if not clean: return
        try:
            if step is not None: self._wandb_run.log(clean, step=step)
            else: self._wandb_run.log(clean)
        except Exception as exc: self.print_to_log_file(f"WandB log: {exc}")

    def _finish_wandb(self):
        if self._wandb_run is None: return
        try: self._wandb_run.finish()
        except Exception: pass
        self._wandb_run = None

    # ── Lifecycle ────────────────────────────────────────────────────────

    def on_train_start(self):
        try: _install_perf_tuning()
        except Exception as exc: print(f"[perf] {exc}", flush=True)
        try: self._maybe_load_subtype_map()
        except Exception as exc: self.print_to_log_file(f"subtype map (pre-fork): {exc}")

        super().on_train_start()

        try: _install_nnunet_warning_filter()
        except Exception: pass
        # _maybe_load_subtype_map already ran above; second call is a no-op.
        try: self._maybe_apply_ce_reweighting()
        except Exception as exc: self.print_to_log_file(f"CE reweight: {exc}")
        try: self._install_log_intercepts()
        except Exception: pass
        try: self._init_wandb()
        except Exception as exc: self.print_to_log_file(f"WandB init: {exc}")

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self._reset_subgroup_dice_buffer()
        # v19.1: snapshot cross-process bias counters at epoch start so
        # we can diff at epoch end and report per-epoch counts. The
        # counters themselves live in lstv_biased_dataloader and are
        # cumulative across the training run.
        if _read_shared_bias_counters is not None:
            try:
                self._bias_counters_at_epoch_start = _read_shared_bias_counters()
            except Exception as exc:
                self._bias_counters_at_epoch_start = {}
                try: self.print_to_log_file(
                    f"v19.1 bias counter snapshot at epoch start: {exc}")
                except Exception: pass

    def on_epoch_end(self):
        super().on_epoch_end()
        if self._wandb_run is None: return
        try:
            L = self._current_logs()
            epoch = self.current_epoch
            def last(key):
                seq = L.get(key, [])
                if not seq: return None
                for v in reversed(seq):
                    if v is not None: return v
                return None
            epoch_time = None
            starts, ends = L.get("epoch_start_timestamps", []), L.get("epoch_end_timestamps", [])
            if starts and ends and len(starts) == len(ends):
                try: epoch_time = float(ends[-1] - starts[-1])
                except Exception: pass
            per_class_log: Dict[str, float] = {}
            dpc = last("dice_per_class_or_region")
            lumbar_v, pelvis_v, fg_v = [], [], []
            if dpc is not None:
                for i, v in enumerate(dpc):
                    if i >= len(_ANATOMY_NAMES): continue
                    try: vf = float(v)
                    except Exception: continue
                    if np.isnan(vf): continue
                    per_class_log[f"val/dice/{_ANATOMY_NAMES[i]}"] = vf
                    fg_v.append(vf)
                    (lumbar_v if i < 6 else pelvis_v).append(vf)
                if lumbar_v: per_class_log["val/dice/mean_lumbar"] = float(np.mean(lumbar_v))
                if pelvis_v: per_class_log["val/dice/mean_pelvis"] = float(np.mean(pelvis_v))
                if fg_v: per_class_log["val/dice/mean_foreground_nanmasked"] = float(np.mean(fg_v))
            payload = {
                "epoch": epoch,
                "train/loss": _as_float(last("train_losses")),
                "val/loss": _as_float(last("val_losses")),
                "val/mean_fg_dice": _as_float(last("mean_fg_dice")),
                "val/ema_fg_dice": _as_float(last("ema_fg_dice")),
                "train/lr": _as_float(last("lrs")),
                "timing/epoch_sec": epoch_time,
                **per_class_log,
            }
            try:
                # v17.1: dedicated LSTV val pass — forward-pass every
                # LSTV case in this fold's val set so the per-subgroup
                # metric is deterministic instead of subsampling-noisy.
                # Must run BEFORE _aggregate_subgroup_dice so the
                # buffer reflects dedicated-pass results.
                self._validate_lstv_dedicated()
            except Exception as exc:
                self.print_to_log_file(f"LSTV dedicated val: {exc}")
            try:
                sg = self._aggregate_subgroup_dice()
                payload.update(sg)
                self._print_per_subgroup_summary(epoch, sg)
            except Exception as exc:
                self.print_to_log_file(f"subgroup aggregation: {exc}")
            # v19.1: read shared bias counters, diff vs epoch-start
            # snapshot, log per-epoch and cumulative numbers.
            try:
                bias_payload = self._collect_bias_counter_payload(epoch)
                payload.update(bias_payload)
            except Exception as exc:
                self.print_to_log_file(f"v19.1 bias counter logging: {exc}")
            self._log_wandb(payload, step=epoch)
        except Exception as exc:
            self.print_to_log_file(f"on_epoch_end: {exc}")

    def on_train_end(self):
        try:
            L = self._current_logs()
            ema = [v for v in L.get("ema_fg_dice", []) if v is not None]
            mean = [v for v in L.get("mean_fg_dice", []) if v is not None]
            summary = {}
            if ema: summary["summary/best_ema_fg_dice"] = max(ema)
            if mean: summary["summary/best_mean_fg_dice"] = max(mean)
            dpc_h = L.get("dice_per_class_or_region", [])
            if dpc_h:
                n_classes = max((len(row) for row in dpc_h if row), default=0)
                for ci in range(n_classes):
                    seq = []
                    for row in dpc_h:
                        if not row or ci >= len(row) or row[ci] is None: continue
                        try: vf = float(row[ci])
                        except Exception: continue
                        if not np.isnan(vf): seq.append(vf)
                    if seq and ci < len(_ANATOMY_NAMES):
                        summary[f"summary/best_dice/{_ANATOMY_NAMES[ci]}"] = max(seq)
            if summary: self._log_wandb(summary)
        except Exception: pass
        super().on_train_end()
        try: self._finish_wandb()
        except Exception: pass

    def perform_actual_validation(self, save_probabilities=False):
        if _env_truthy("SPINESURG_RUN_VAL_EXPORT"):
            return super().perform_actual_validation(save_probabilities)
        self.print_to_log_file("perform_actual_validation: SKIPPED (set SPINESURG_RUN_VAL_EXPORT=1)")
        return


def _as_float(x):
    if x is None: return None
    try: xf = float(x)
    except (TypeError, ValueError): return None
    return None if np.isnan(xf) else xf


# =============================================================================
# LSTV oversampling mixin (queue-level, 6-way) + v18 patch-bias install
# =============================================================================

class _LSTVOversampleMixin:
    """Oversample all 5 LSTV variant subtypes at the dataloader queue level,
    AND substitute LSTVBiasedDataLoader3D for nnU-Net's default loader so
    force_fg patches on lumb / sacr_count cases bias toward the headline
    anatomy (L6 / sacrum respectively).

    Pool composition (5 of 6 subtypes; 'normal' is the majority and
    by definition not oversampled):
      - lumb              (lumbarization, L6 present)
      - sacr_count        (sacralization with full L5->sacrum count change)
      - sacralization     (sacral-wing morphology, count unchanged)
      - semisacralization (unilateral sacral-wing fusion)
      - ambiguous         (uncertain Castellvi classification)

    Rationale: every non-normal LSTV variant deserves more training
    exposure than its natural frequency. ambiguous and semisacralization
    are rare (n<5 each) — they can only be over-represented through
    duplication, which is acceptable given the alternative (the model
    rarely sees them at all during training).

    v18: get_dataloaders is overridden to substitute the biased loader
    class. This replaces the v15-v17.x np.random.choice monkey-patch
    that never fired in worker processes.
    """
    _lstv_case_ids: Optional[Set[str]] = None
    _lstv_subtype_counts: Optional[Counter] = None
    _lstv_detection_source: Optional[str] = None

    def _identify_lstv_cases(self) -> Set[str]:
        if self._lstv_case_ids is not None:
            return self._lstv_case_ids
        try: ds_name = self.plans_manager.dataset_name
        except Exception: ds_name = None

        if ds_name:
            for jp in _candidate_lstv_json_paths(ds_name):
                self.print_to_log_file(f"LSTV oversample: checking {jp}")
                result = _read_lstv_cases_json(jp)
                if result is not None:
                    all_ids, subtypes, case_to_sub, oversample_pool = result
                    pool_set = set(oversample_pool)
                    self._lstv_case_ids = pool_set
                    self._lstv_subtype_counts = subtypes
                    self._lstv_detection_source = f"json:{jp}"
                    n_normal = subtypes.get(_SUBTYPE_NORMAL, 0)
                    n_other_excluded = (
                        len(case_to_sub) - len(pool_set) - n_normal
                    )
                    pool_subtype_counts = Counter(
                        case_to_sub[c] for c in pool_set if c in case_to_sub
                    )
                    self.print_to_log_file(
                        f"LSTV oversample: loaded {len(case_to_sub)} cases; "
                        f"using {len(pool_set)} for oversampling (normals "
                        f"excluded: {n_normal}; other excluded: "
                        f"{n_other_excluded}). "
                        f"Pool by subtype: {dict(pool_subtype_counts)}. "
                        f"Full subtype counts: {dict(subtypes)}")
                    return pool_set

        for hf_dir in _candidate_hf_export_dirs():
            self.print_to_log_file(f"LSTV oversample: checking manifest {hf_dir}")
            result = _read_lstv_records_from_manifest(hf_dir)
            if result is not None:
                all_ids, subtypes, case_to_sub, oversample_pool = result
                pool_set = set(oversample_pool)
                self._lstv_case_ids = pool_set
                self._lstv_subtype_counts = subtypes
                self._lstv_detection_source = f"manifest:{hf_dir}"
                self.print_to_log_file(
                    f"LSTV oversample: scanned manifests, found {len(all_ids)} cases; "
                    f"using {len(pool_set)} for oversampling")
                return pool_set

        self.print_to_log_file("LSTV oversample: falling back to L6 scan (lumb-only)")
        labels_dir = self._lstv_labels_dir_for_legacy_scan()
        if labels_dir is None:
            self._lstv_case_ids = set()
            return self._lstv_case_ids
        ids = _scan_lstv_case_ids_by_l6_voxels(labels_dir)
        self._lstv_case_ids = ids
        self._lstv_subtype_counts = Counter({"lumb_l6scan": len(ids)})
        self._lstv_detection_source = f"l6scan:{labels_dir}"
        return ids

    def _lstv_labels_dir_for_legacy_scan(self):
        override = os.environ.get("LSTV_LABELS_DIR", "").strip()
        if override: return override
        raw = os.environ.get("nnUNet_raw") or os.environ.get("nnUNet_raw_data_base")
        if raw:
            try:
                cand = join(raw, self.plans_manager.dataset_name, "labelsTr")
                if isdir(cand): return cand
            except Exception: pass
        return None

    def _apply_lstv_sampler(self):
        try: frac = float(os.environ.get("LSTV_OVERSAMPLE_FRAC", "0.25"))
        except ValueError: frac = 0.25
        frac = max(0.0, min(0.95, frac))
        if frac <= 0:
            self.print_to_log_file("LSTV oversample: frac=0, no-op."); return
        lstv_ids = self._identify_lstv_cases()
        if not lstv_ids:
            self.print_to_log_file("LSTV oversample: no LSTV cases, no-op."); return
        loader = getattr(self, "dataloader_train", None)
        if loader is None:
            self.print_to_log_file("LSTV oversample: dataloader_train is None"); return

        keys_paths = [
            ("generator", "_data", "identifiers"), ("generator", "_data", "keys"),
            ("_data", "identifiers"), ("_data", "keys"),
            ("data_loader", "_data", "identifiers"), ("data_loader", "_data", "keys"),
        ]
        keys_list, keys_owner, keys_attr = None, None, None
        for path in keys_paths:
            obj = loader; ok = True
            for p in path[:-1]:
                obj = getattr(obj, p, None)
                if obj is None: ok = False; break
            if not ok: continue
            if hasattr(obj, path[-1]):
                cand = getattr(obj, path[-1])
                if callable(cand):
                    try: cand = list(cand())
                    except Exception: continue
                else:
                    try: cand = list(cand)
                    except Exception: continue
                if cand and all(isinstance(k, str) for k in cand):
                    keys_list, keys_owner, keys_attr = cand, obj, path[-1]
                    break
        if keys_list is None:
            self.print_to_log_file("LSTV oversample: cannot locate dataloader keys; DISABLED"); return
        original_n = len(keys_list)
        lstv_in_keys = [k for k in keys_list if k in lstv_ids]
        if not lstv_in_keys:
            self.print_to_log_file("LSTV oversample: no LSTV in this fold; no-op."); return
        p = len(lstv_in_keys) / original_n
        if p >= frac:
            self.print_to_log_file(f"LSTV oversample: natural frac {p:.3f} >= target {frac:.3f}"); return
        needed = max(1, int(round((frac * original_n - len(lstv_in_keys)) / (1.0 - frac))))
        duplicates = [lstv_in_keys[i % len(lstv_in_keys)] for i in range(needed)]
        new_keys = keys_list + duplicates
        try: setattr(keys_owner, keys_attr, new_keys)
        except Exception as exc:
            self.print_to_log_file(f"LSTV oversample: setattr failed: {exc}"); return
        unique_lstv = len(set(lstv_in_keys))
        avg_dup = (len(lstv_in_keys) + len(duplicates)) / max(1, unique_lstv)
        self.print_to_log_file(
            f"LSTV oversample: source={self._lstv_detection_source}")
        self.print_to_log_file(
            f"LSTV oversample: fold {self.fold}: original_N={original_n}  "
            f"LSTV={len(lstv_in_keys)} ({p*100:.1f}%);  unique={unique_lstv};  "
            f"+{len(duplicates)} dups -> N={len(new_keys)};  avg_dup={avg_dup:.1f}x")
        if avg_dup > 25:
            self.print_to_log_file(f"LSTV oversample: WARNING avg_dup={avg_dup:.0f}x -> overfitting risk")

    # ── v18: substitute the biased loader class via get_dataloaders ──────

    def get_dataloaders(self):
        """v18: substitute LSTVBiasedDataLoader3D for nnU-Net's default
        nnUNetDataLoader3D so that force_fg patches on lumb / sacr_count
        cases bias toward the headline anatomy (L6 / sacrum).

        nnU-Net's nnUNetTrainer.get_dataloaders builds dataloaders by
        instantiating nnUNetDataLoader3D as referenced in the trainer
        module's namespace. To swap our subclass in, we rebind that
        symbol for the duration of super().get_dataloaders() and
        restore it after. This survives fork/spawn cleanly because
        workers fork AFTER the substitution and pickle the substituted
        class.

        Disable via SPINESURG_LSTV_BIAS_ENABLED=0 — the substitution
        is skipped and nnU-Net's default uniform foreground sampling
        is used. The biased loader itself also reads the env var, so
        even if substitution happens but bias_enabled=False, the
        loader behaves identically to the parent.
        """
        bias_enabled = _env_truthy("SPINESURG_LSTV_BIAS_ENABLED", default=True)

        # Short-circuit: bias disabled OR loader class missing -> no swap
        if not bias_enabled:
            self.print_to_log_file(
                "LSTV patch bias: disabled via SPINESURG_LSTV_BIAS_ENABLED=0; "
                "using nnU-Net default loader.")
            return super().get_dataloaders()

        if not _LSTV_BIASED_LOADER_AVAILABLE:
            self.print_to_log_file(
                f"LSTV patch bias: LSTVBiasedDataLoader3D not importable "
                f"({_LSTV_LOADER_IMPORT_ERROR}); using nnU-Net default loader. "
                f"Install lstv_biased_dataloader.py alongside this trainer "
                f"to enable bias.")
            return super().get_dataloaders()

        # Locate the trainer module (where nnUNetDataLoader3D is imported)
        # so we can rebind that symbol. nnU-Net's nnUNetTrainer.get_dataloaders
        # uses an unqualified `nnUNetDataLoader3D(...)` call which Python
        # resolves at runtime against the module's globals, so rebinding
        # in the module namespace is sufficient.
        try:
            import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as _trainer_mod
        except ImportError as exc:
            self.print_to_log_file(
                f"LSTV patch bias: cannot import trainer module ({exc}); "
                f"using nnU-Net default loader.")
            return super().get_dataloaders()

        original_3d = getattr(_trainer_mod, 'nnUNetDataLoader3D', None)
        if original_3d is None:
            self.print_to_log_file(
                "LSTV patch bias: nnUNetDataLoader3D symbol not found in "
                "trainer module; using nnU-Net default loader. "
                "(This indicates a nnU-Net layout change incompatible with v18.)")
            return super().get_dataloaders()

        l6_p = _env_float("SPINESURG_LSTV_BIAS_L6_PROB", 0.60)
        sac_p = _env_float("SPINESURG_LSTV_BIAS_SACRUM_PROB", 0.50)
        self.print_to_log_file(
            f"LSTV patch bias: substituting nnUNetDataLoader3D -> "
            f"LSTVBiasedDataLoader3D for get_dataloaders() "
            f"(L6 prob={l6_p:.2f}, sacrum prob={sac_p:.2f})")

        _trainer_mod.nnUNetDataLoader3D = LSTVBiasedDataLoader3D
        try:
            return super().get_dataloaders()
        finally:
            _trainer_mod.nnUNetDataLoader3D = original_3d

    def on_train_start(self):
        super().on_train_start()
        try: self._apply_lstv_sampler()
        except Exception as exc: self.print_to_log_file(f"LSTV oversample: {exc}")
        # Banner runs LAST so it captures post-sampler pool composition
        # AND can verify the train loader's class is LSTVBiasedDataLoader3D.
        # super().on_train_start() above ran _WandBMixin's setup
        # (subtype map, CE reweight, get_dataloaders [which our override
        # intercepted to swap the loader class], W&B init);
        # _apply_lstv_sampler then duplicated keys to hit the target
        # frac. Now dump everything as one grep-able block.
        try: self._emit_startup_diagnostics()
        except Exception as exc:
            self.print_to_log_file(f"startup diagnostics: {exc}")


# =============================================================================
# Concrete trainer classes
# =============================================================================

class nnUNetTrainerWandB(_WandBMixin, nnUNetTrainer):
    pass

class nnUNetTrainerWandB_5epochs(_WandBMixin, nnUNetTrainer_5epochs):
    pass

class nnUNetTrainerWandB_100epochs(_WandBMixin, nnUNetTrainer_100epochs):
    pass

class nnUNetTrainerWandB_500epochs(_WandBMixin, nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json,
                 unpack_dataset=True, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 500

class nnUNetTrainerWandB_1000epochs(_WandBMixin, nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json,
                 unpack_dataset=True, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 1000

class nnUNetTrainerWandB_1000ep_500iter(_WandBMixin, nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json,
                 unpack_dataset=True, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 1000
        self.num_iterations_per_epoch = 500
        self.num_val_iterations_per_epoch = 50


class nnUNetTrainerWandB_500ep_LSTVOversample(
    _LSTVOversampleMixin, _WandBMixin, nnUNetTrainer
):
    """500 epochs + LSTV queue oversample + class-override patch bias
    + CE reweight. DEFAULT for the SpineSurg-CT paper run.

    v18 (May 2026): Patch class biasing now via LSTVBiasedDataLoader3D
         subclass, not np.random.choice monkey-patching. The previous
         monkey-patch never fired in worker processes (verified
         bias_call_count=0 across 64 epochs of fold 4 in production
         job 35993148, and in every prior production run). v18's class
         override is testable end-to-end (see
         tests/test_lstv_biased_dataloader.py) and verifiably active
         at training time (see PATCH BIAS section in startup
         diagnostics banner).
    v17.4: Hallucination metrics (specificity tracking on absent classes).
    v17.1-v17.3: Dedicated LSTV val pass — forces forward pass on every
         LSTV case in this fold's val set every epoch, regardless of
         whether the regular val sampler caught them in its
         num_val_iterations subsample. Eliminates n=0 LSTV val epochs
         that silenced the headline metric.
    v17: Oversample pool now includes all 5 LSTV variants (added
         ambiguous, n=4). Symmetric per-record subtype refinement at
         convert time: spine_only demotes pelvis-region subtypes
         (sacralization, semisacralization), pelvic_native demotes
         lumbar-region subtypes (lumb, sacr_count). Verbose
         startup-diagnostic banner emitted via
         _emit_startup_diagnostics.
    v15-v16: Multiple attempts to make the np.random.choice monkey-patch
         fire at the right call site. Superseded by v18.
    v12: Per-subgroup dice logger no longer crashes on numpy keys array
         (was using truthy fallback chain that numpy rejects).
    v11: CE weight tensor sized to actual network output (was 11; should
         be num_segmentation_heads, typically 10 for our schema).
    v10: 6-way subgroups; semisacralization separated from sacralization.
    """
    def __init__(self, plans, configuration, fold, dataset_json,
                 unpack_dataset=True, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 500


class nnUNetTrainerWandB_1000ep_LSTVOversample(
    _LSTVOversampleMixin, _WandBMixin, nnUNetTrainer
):
    def __init__(self, plans, configuration, fold, dataset_json,
                 unpack_dataset=True, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 1000
