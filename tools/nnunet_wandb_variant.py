"""
SpineSurg-CT -- nnU-Net v2 trainers with W&B logging (v17)
tools/nnunet_wandb_variant.py

Design intent
=============
Three-tier emphasis on LSTV anatomy, partial-annotation aware:

  (a) Class-imbalance correction via queue-level oversampling.
      Applied uniformly to ALL LSTV subtypes except 'ambiguous'.
      Without this, LSTV is ~4% of training and the model defaults
      to normal anatomy.

  (b) Headline-subtype emphasis via per-case patch sampling bias.
      Lumb       -> L6  (class 6) @ 60%  (the class TS misses)
      sacr_count -> sacrum (cls 7) @ 50%  (L4/sacrum transition zone)

      Detection (partial-annotation safe):
        lumb       :  L6 in case's foreground class list
        sacr_count :  sacrum in fg AND L5 NOT in fg AND L4 in fg
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

Changes in v17 (May 2026)
=========================
1. Symmetric demotion (in convert_hf_to_nnunet.py): also demote
   sacralization / semisacralization for spine_only views, mirroring
   the lumb / sacr_count demotion for pelvic_native views. A record
   only contributes to a subtype's stats / sampling if its defining
   anatomy is supervised (not ignore=10) in that view.

2. Oversample pool widened to ALL non-normal subtypes, including
   ambiguous (was excluded as "n=4 too small"). Per Greg: even
   ambiguous cases provide LSTV-like exposure that the model
   wouldn't otherwise see.

3. Comprehensive startup logging — each install phase emits
   [STARTUP] prefixed lines for subtype_map, ce_reweight,
   patch_bias, oversample, log_intercepts, wandb. Every probe
   attempt, every fallback path, every decision is logged. BEGIN /
   OK ✓ / FAILED ✗ sentinels frame each phase.

4. Patch-bias install runs an end-to-end self-test BEFORE training:
   constructs a fake function literally named `generate_train_batch`,
   exercises the np.random.choice intercept, and confirms frame
   inspection works in this Python. If the self-test fails, the
   user sees it at startup instead of silently no-op'ing for hours.

5. Oversample install reads back the keys list AFTER setattr to
   verify the mutation actually took effect, counts LSTV duplicates
   in the readback, and aborts if the readback doesn't match.

6. Per-subgroup val dice prints individual L1-L6 + sacrum +
   left_hip + right_hip dice on dedicated lines for ALL subtypes
   (including normal). Was: one-line dump only for non-normal.

Changes in v16 (May 2026 — CRITICAL fix, preserved)
========================================
PATCH-BIAS HOOK WAS DEAD. v15 patched stdlib `random.choice`, but
nnU-Net v2's `nnUNetDataLoader{2D,3D}.generate_train_batch` calls
`np.random.choice(len(eligible_classes_or_regions))` — a different
function entirely. The "Patch-class bias: ENABLED" log line at startup
was misleading: the wrapper installed cleanly but never intercepted
anything, so all bias decisions came from the original sampler.

Fix:
  1. Patch `np.random.choice` (correct target)
  2. Frame-inspect the caller to extract `eligible_classes_or_regions`
     and confirm the call originates from `generate_train_batch`. This
     avoids hijacking unrelated np.random.choice calls (e.g., the
     subsequent `np.random.choice(len(voxels_of_that_class))` for
     within-class voxel selection, or any np.random.choice in user
     transforms or augmentations).
  3. Return an INDEX into the eligible list, not the class id —
     nnU-Net dereferences `eligible_classes_or_regions[idx]` to get
     the class.
  4. Skip tuple keys in eligible (the all-classes "any foreground"
     region key) — only consider int single-class entries.
  5. Diagnostic counters (`_bias_call_count`, `_bias_l6_returns`,
     `_bias_sacrum_returns`) so the per-subgroup log shows whether
     the hook is actually firing.
  6. The selection logic is extracted as a module-level pure function
     `_select_biased_index` so it can be unit-tested in isolation.

Changes from v13/v14 (preserved)
================================
1. Patch-bias detection refined for partial-annotation labels: requires
   L4 IN fg list to distinguish sacr_count from pelvic-only records.

2. All LSTV subtypes (lumb, sacr_count, sacralization, semisacralization)
   are oversampled. Only ambiguous excluded.

3. JSON reader (`_read_lstv_cases_json`) ignores any stored
   `lstv_oversample_pool` field; always recomputes from in-code
   `_OVERSAMPLE_SUBTYPES`.

Changes from v12 (preserved)
============================
- Generalized patch-class bias from L6-only to per-subtype rules.
- Sacrum CE weight bumped 1.0 -> 2.0.

Changes from v11 (preserved)
============================
- Fix per-subgroup dice logger numpy-truth-value bug.

Changes from v10 (preserved)
============================
- Fix CE weight tensor sizing crash. Reads num_classes from
  self.label_manager.num_segmentation_heads.

Changes from v9 (preserved)
===========================
- 6-way LSTV subgroup binning (matches splits_5fold.json v6).
- lstv_cases.json reader (schema v3).
- Per-subgroup val tracking: 6 subgroups in W&B and stdout.

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
# Patch-class bias selection logic (pure function — unit-testable)
# =============================================================================

def _select_biased_index(
    eligible: Sequence,
    rules: Dict[str, Tuple[int, str, float]],
    rand: Callable[[], float],
) -> Optional[int]:
    """Decide whether to bias the next forced-FG selection.

    Pure function with no side effects. Returns the INDEX into
    ``eligible`` of the biased target class, or None if no bias rule
    applies (in which case the caller should defer to the default
    sampler).

    Detection logic — partial-annotation safe:

      lumb        => L6 (class 6) is in eligible. Only lumb cases have
                     L6 annotated. Pelvic-only views of lumb patients
                     do NOT have 6 in eligible (their lumbar region is
                     all ignore=10 -> excluded from foreground class
                     locations).

      sacr_count  => sacrum (7) IS in eligible, L5 (5) is NOT, and L4
                     (4) IS. The L4-in-fg check is critical — pelvic-
                     only normal cases also lack L5 (it's ignore there)
                     but they also lack L4. Only true sacr_count
                     anatomy has L1-L4 + sacrum without L5.

    Args:
      eligible: the ``eligible_classes_or_regions`` list nnU-Net is
        about to sample from. Entries are class ids (ints) or region
        tuples. Tuple entries (e.g. the all-classes "any foreground"
        key) are skipped during detection.
      rules: mapping from subtype name to (target_class, env_var_name,
        frac). Same shape as ``_PATCH_BIAS_RULES``. Only target_class
        and frac are used here; env_var_name is informational.
      rand: a function returning a float in [0, 1). Pass
        ``random.random`` in production; pass a deterministic stub in
        tests.

    Returns:
      The index ``i`` such that ``eligible[i]`` is the biased target
      class, or None if no rule fires.
    """
    # Build a class_id -> index map, ignoring tuple/region entries.
    int_keys: Dict[int, int] = {}
    for i, x in enumerate(eligible):
        if isinstance(x, (int, np.integer)):
            int_keys[int(x)] = i
        # Tuple entries (region keys) are intentionally skipped.

    # Lumb takes priority: L6 is the most specific signal.
    if _L6_LABEL_ID in int_keys:
        rule = rules.get(_SUBTYPE_LUMB)
        if rule is not None:
            _, _, frac = rule
            if rand() < frac:
                return int_keys[_L6_LABEL_ID]
        return None  # L6 present means this is a lumb case; do not fall
                      # through to sacr_count heuristic (which checks L5
                      # absent — irrelevant for a lumb spine view).

    # sacr_count: sacrum present, L5 absent, L4 present.
    if (_SACRUM_LABEL_ID in int_keys
            and 5 not in int_keys
            and 4 in int_keys):
        rule = rules.get(_SUBTYPE_SACR_COUNT)
        if rule is not None:
            _, _, frac = rule
            if rand() < frac:
                return int_keys[_SACRUM_LABEL_ID]
        return None

    return None


def _selftest_patch_bias(rules: Dict[str, Tuple[int, str, float]],
                          n_trials: int = 200) -> Tuple[int, int, int]:
    """End-to-end self-test of the np.random.choice intercept mechanism.

    Constructs a fake function literally named ``generate_train_batch``,
    sets up an ``eligible_classes_or_regions`` local exactly as nnU-Net
    does, calls ``np.random.choice(len(eligible))``, and counts how many
    times the bias fires.

    The intercept relies on:
      - patching np.random.choice (correct target — not stdlib random)
      - Python's frame inspection (sys._getframe(1).f_code.co_name)
      - reading f_locals["eligible_classes_or_regions"] from caller

    If any of those break in this Python's runtime, the self-test
    catches it at startup instead of silently no-op'ing for hours.

    Returns: (n_trials_run, n_intercept_calls, n_l6_returns)
      - n_intercept_calls should equal n_trials_run (intercept fires
        on every call). If not, frame inspection is broken.
      - n_l6_returns should be ~ n_trials * lumb_frac. If 0 with
        non-zero frac, _select_biased_index is broken.
    """
    original_np_choice = np.random.choice

    n_calls = 0
    n_l6_returns = 0

    def biased_test_choice(*args, **kwargs):
        nonlocal n_calls, n_l6_returns
        try:
            if (len(args) == 1 and not kwargs
                    and isinstance(args[0], (int, np.integer))):
                frame = sys._getframe(1)
                if frame.f_code.co_name == "generate_train_batch":
                    eligible = frame.f_locals.get("eligible_classes_or_regions")
                    if (eligible is not None
                            and hasattr(eligible, "__len__")
                            and len(eligible) == int(args[0])
                            and len(eligible) > 1):
                        n_calls += 1
                        biased_idx = _select_biased_index(
                            eligible, rules, random.random)
                        if biased_idx is not None:
                            cls = eligible[biased_idx]
                            if (isinstance(cls, (int, np.integer))
                                    and int(cls) == _L6_LABEL_ID):
                                n_l6_returns += 1
                            return biased_idx
        except Exception:
            pass
        return original_np_choice(*args, **kwargs)

    def generate_train_batch():
        # Fake nnU-Net call site: same locals, same call shape.
        eligible_classes_or_regions = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        return np.random.choice(len(eligible_classes_or_regions))

    np.random.choice = biased_test_choice
    try:
        for _ in range(n_trials):
            generate_train_batch()
    finally:
        np.random.choice = original_np_choice

    return n_trials, n_calls, n_l6_returns


# =============================================================================
# Module-level bias-hook state (v17.2 fix for worker-fork bug)
#
# Why module-level instead of instance-level:
#   nnU-Net v2's NonDetMultiThreadedAugmenter spawns worker processes
#   that pickle (or fork-copy) the data loader. Patches on
#   trainer-instance attributes don't reliably reach the workers, so
#   the v15-v17 instance-level patch produced the right startup self-
#   test signal but never actually fired during real training (visible
#   as "bias hook (cumulative): calls=0" in every epoch log).
#
# Architecture:
#   - On Linux fork (default for batchgenerators on Python<3.8 and on
#     explicit fork-mode setups), child processes inherit the parent's
#     entire memory image at fork time. If we replace ``np.random.choice``
#     on the parent BEFORE workers fork, every worker sees the patched
#     version — the patch lives in the np.random module's __dict__,
#     which forks just like everything else.
#   - On spawn mode, workers re-import nnunet_wandb_variant (it's in
#     the import path) and the module-level state below is set during
#     import. The trainer also replaces np.random.choice during
#     initialize() before workers start, so spawned workers see the
#     replacement after they finish importing this module.
#   - Counters live in multiprocessing.Value so worker increments are
#     visible to the parent. Without this, the parent's counter would
#     stay at zero even with the patch firing in workers.
#
# Frame guard:
#   The patch fires ONLY when called from a function literally named
#   ``generate_train_batch`` AND when an ``eligible_classes_or_regions``
#   local variable is present AND its length matches the int arg.
#   Anything else (size=, p=, multi-arg, called from anywhere else)
#   passes through to the original numpy implementation untouched.
# =============================================================================

_BIAS_RULES_FOR_WORKERS: Optional[Dict[str, Tuple[int, str, float]]] = None
_BIAS_ORIGINAL_NP_CHOICE = None  # the real np.random.choice
_BIAS_INSTALLED: bool = False    # idempotency
_BIAS_CALL_COUNT = None          # multiprocessing.Value('i', 0)
_BIAS_L6_RETURNS = None
_BIAS_SACRUM_RETURNS = None


def _ensure_bias_counters() -> None:
    """Lazily initialize cross-process atomic counters.

    Must be called BEFORE any worker fork. After fork, child processes
    share the same ``Value`` objects as the parent (they live in shared
    memory via the multiprocessing primitive)."""
    global _BIAS_CALL_COUNT, _BIAS_L6_RETURNS, _BIAS_SACRUM_RETURNS
    if _BIAS_CALL_COUNT is None:
        _BIAS_CALL_COUNT     = _mp.Value("i", 0)
        _BIAS_L6_RETURNS     = _mp.Value("i", 0)
        _BIAS_SACRUM_RETURNS = _mp.Value("i", 0)


def _read_bias_counters() -> Tuple[int, int, int]:
    """Read the current values from the cross-process counters.
    Returns (calls, l6_returns, sacrum_returns) as plain ints."""
    if _BIAS_CALL_COUNT is None:
        return 0, 0, 0
    try:
        return (int(_BIAS_CALL_COUNT.value),
                int(_BIAS_L6_RETURNS.value),
                int(_BIAS_SACRUM_RETURNS.value))
    except Exception:
        return 0, 0, 0


def _global_biased_np_choice(*args, **kwargs):
    """Module-level replacement for ``np.random.choice``.

    Picklable, no closures over trainer instance state. Called from
    both parent and worker processes. Increments shared
    multiprocessing.Value counters when the bias fires.

    Pass-through fallback ensures any unrelated np.random.choice call
    (size=, p=, multi-arg, called from outside generate_train_batch)
    behaves identically to the original numpy implementation.
    """
    try:
        if (len(args) == 1
                and not kwargs
                and isinstance(args[0], (int, np.integer))):
            frame = sys._getframe(1)
            if frame.f_code.co_name == "generate_train_batch":
                eligible = frame.f_locals.get("eligible_classes_or_regions")
                if (eligible is not None
                        and hasattr(eligible, "__len__")
                        and len(eligible) == int(args[0])
                        and len(eligible) > 1):
                    rules = _BIAS_RULES_FOR_WORKERS
                    if rules:
                        # Atomic increment of shared call counter.
                        if _BIAS_CALL_COUNT is not None:
                            with _BIAS_CALL_COUNT.get_lock():
                                _BIAS_CALL_COUNT.value += 1
                        biased_idx = _select_biased_index(
                            eligible, rules, random.random)
                        if biased_idx is not None:
                            cls = eligible[biased_idx]
                            if isinstance(cls, (int, np.integer)):
                                if int(cls) == _L6_LABEL_ID:
                                    if _BIAS_L6_RETURNS is not None:
                                        with _BIAS_L6_RETURNS.get_lock():
                                            _BIAS_L6_RETURNS.value += 1
                                elif int(cls) == _SACRUM_LABEL_ID:
                                    if _BIAS_SACRUM_RETURNS is not None:
                                        with _BIAS_SACRUM_RETURNS.get_lock():
                                            _BIAS_SACRUM_RETURNS.value += 1
                            return biased_idx
    except Exception:
        # Never let the intercept break training.
        pass
    # Pass-through to original numpy implementation.
    if _BIAS_ORIGINAL_NP_CHOICE is not None:
        return _BIAS_ORIGINAL_NP_CHOICE(*args, **kwargs)
    # Should never happen — _BIAS_ORIGINAL_NP_CHOICE is set before
    # the patch is installed. If it does, fall back to whatever's
    # currently in np.random.choice (likely us, infinite recursion
    # would be bad — so re-raise).
    raise RuntimeError("bias hook fallback invoked before original captured")


def _install_global_bias_hook(rules: Dict[str, Tuple[int, str, float]]) -> bool:
    """Install the global np.random.choice patch.

    Idempotent — safe to call multiple times. Returns True if the
    install actually happened (or was already in place), False on
    error. Must be called BEFORE workers fork.
    """
    global _BIAS_RULES_FOR_WORKERS, _BIAS_INSTALLED, _BIAS_ORIGINAL_NP_CHOICE
    _BIAS_RULES_FOR_WORKERS = rules
    _ensure_bias_counters()
    if _BIAS_INSTALLED:
        return True  # already patched (e.g., second fold in same proc)
    try:
        _BIAS_ORIGINAL_NP_CHOICE = np.random.choice
        np.random.choice = _global_biased_np_choice
        _BIAS_INSTALLED = True
        return True
    except Exception:
        return False


def _reset_bias_counters_for_new_fold() -> None:
    """Reset shared counters (called per-fold to avoid carry-over when
    the same parent process trains multiple folds)."""
    if _BIAS_CALL_COUNT is None:
        return
    with _BIAS_CALL_COUNT.get_lock():
        _BIAS_CALL_COUNT.value = 0
    with _BIAS_L6_RETURNS.get_lock():
        _BIAS_L6_RETURNS.value = 0
    with _BIAS_SACRUM_RETURNS.get_lock():
        _BIAS_SACRUM_RETURNS.value = 0


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
    _val_case_to_subtype: Optional[Dict[str, str]] = None
    _per_subgroup_logging_enabled: bool = True

    _l6_patch_bias_enabled: bool = True
    _l6_patch_bias_target_frac: float = 0.6
    _lumb_case_id_set: Optional[Set[str]] = None

    # Diagnostic counters — incremented by the np.random.choice intercept.
    # Cumulative across the run (not reset per epoch).
    _bias_call_count: int = 0
    _bias_l6_returns: int = 0
    _bias_sacrum_returns: int = 0

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

    # ── Patch sampling bias (np.random.choice intercept) ───────────────

    # Subtype -> (target class id, env var name, default frac).
    # Lumb cases bias patches toward L6 (the class TS misses entirely).
    # sacr_count cases bias patches toward sacrum (the L4/sacrum
    # transition zone is where the missing-L5 morphology is visible).
    _PATCH_BIAS_RULES = {
        _SUBTYPE_LUMB:       (_L6_LABEL_ID,     "SPINESURG_L6_PATCH_BIAS_FRAC",     0.6),
        _SUBTYPE_SACR_COUNT: (_SACRUM_LABEL_ID, "SPINESURG_SACRUM_PATCH_BIAS_FRAC", 0.5),
    }
    _patch_bias_targets: Optional[Dict[str, int]] = None  # cid -> target_class
    _patch_bias_fracs:   Optional[Dict[str, float]] = None  # cid -> frac

    def _maybe_apply_l6_patch_bias(self):
        """Install per-class patch sampling bias on the dataloader.

        Implementation: monkey-patch ``np.random.choice`` only while
        ``generate_train_batch`` is executing. The intercept inspects
        the caller's frame to find ``eligible_classes_or_regions``,
        then applies ``_select_biased_index`` to decide whether to
        override the choice.

        Method name preserved (``_maybe_apply_l6_patch_bias``) for
        backward compat with v10/v11 logs; the feature is broader than
        L6 — see ``_PATCH_BIAS_RULES``.
        """
        self._phase_begin("patch_bias")
        if not _env_truthy("SPINESURG_L6_PATCH_BIAS", default=True):
            self._phase("patch_bias", "DISABLED via SPINESURG_L6_PATCH_BIAS=0")
            self._phase_end("patch_bias", ok=True, note="disabled")
            return

        if self._val_case_to_subtype is None:
            self._phase("patch_bias",
                "subtype map not yet loaded; calling _maybe_load_subtype_map")
            self._maybe_load_subtype_map()

        # Resolve env-overridden fracs at install time so the intercept
        # doesn't have to read env on every call.
        rules: Dict[str, Tuple[int, str, float]] = {}
        for sub, (target_cls, env_name, default_frac) in self._PATCH_BIAS_RULES.items():
            try:
                frac = max(0.0, min(0.95, _env_float(env_name, default_frac)))
            except Exception:
                frac = default_frac
            rules[sub] = (target_cls, env_name, frac)
            self._phase("patch_bias",
                f"rule: {sub} -> class {target_cls} @ frac={frac:.2f} "
                f"(env {env_name})")

        # Build informational per-case maps (used only for the install-
        # time log line; intercept itself is content-driven, not
        # case-id-driven).
        targets: Dict[str, int] = {}
        fracs: Dict[str, float] = {}
        per_subtype_count: Counter = Counter()
        if self._val_case_to_subtype:
            for cid, sub in self._val_case_to_subtype.items():
                if sub in rules:
                    target_cls, _, frac = rules[sub]
                    targets[cid] = target_cls
                    fracs[cid] = frac
                    per_subtype_count[sub] += 1

        self._patch_bias_targets = targets
        self._patch_bias_fracs = fracs
        self._patch_bias_rules_resolved = rules
        self._l6_patch_bias_enabled = True

        # Backward-compat alias (some downstream code may still read this)
        self._lumb_case_id_set = {
            cid for cid, t in targets.items() if t == _L6_LABEL_ID
        }

        self._phase("patch_bias",
            f"informational pool: {dict(per_subtype_count)} "
            f"({len(targets)} cases match a rule)")
        if not targets:
            self._phase("patch_bias",
                "WARN: no cases match any bias rule. Bias will install "
                "but never fire. Check subtype map loaded correctly.")

        loader = getattr(self, "dataloader_train", None)
        self._phase("patch_bias",
            f"dataloader_train type: "
            f"{type(loader).__name__ if loader is not None else None}")
        underlying = loader
        located_via = "dataloader_train"
        for attr in ("generator", "data_loader", "_data_loader"):
            if hasattr(underlying, attr):
                cand = getattr(underlying, attr)
                self._phase("patch_bias",
                    f"  attr {attr}: type={type(cand).__name__}")
                if hasattr(cand, "generate_train_batch") or hasattr(cand, "_data"):
                    underlying = cand
                    located_via = f"dataloader_train.{attr}"
                    break
        if underlying is None or not hasattr(underlying, "generate_train_batch"):
            self._phase("patch_bias",
                "ABORT: cannot locate generate_train_batch on dataloader. "
                "nnU-Net internals may have changed.")
            self._l6_patch_bias_enabled = False
            self._phase_end("patch_bias", ok=False,
                note="generate_train_batch not found")
            return

        self._phase("patch_bias",
            f"located generate_train_batch via {located_via} "
            f"({type(underlying).__name__})")

        # Run startup self-test BEFORE installing the real hook. Catches
        # frame-inspection breakage in this Python (the v15 silent-no-op
        # bug class) at startup instead of after hours of training.
        self._phase("patch_bias", "running self-test (200 trials)...")
        try:
            n_trials, n_calls, n_l6 = _selftest_patch_bias(rules, n_trials=200)
            self._phase("patch_bias",
                f"self-test results: trials={n_trials} "
                f"intercept_calls={n_calls} L6_returns={n_l6}")
            if n_calls != n_trials:
                self._phase("patch_bias",
                    f"SELF-TEST FAILED: intercept fired only "
                    f"{n_calls}/{n_trials}. Frame inspection may be "
                    f"broken; bias hook will be a no-op.")
            elif n_l6 < 30 or n_l6 > 170:
                # Expected for lumb frac=0.6 -> ~120 returns. Wide window.
                self._phase("patch_bias",
                    f"SELF-TEST WARN: L6 returns {n_l6}/{n_trials} outside "
                    f"plausible range [30,170] for frac=0.6.")
            else:
                self._phase("patch_bias", "self-test PASSED ✓")
        except Exception as exc:
            self._phase("patch_bias", f"self-test threw exception: {exc}")

        # ─────────────────────────────────────────────────────────────────
        # v17.2: install GLOBALLY at module scope, not on the loader instance.
        #
        # Background: nnU-Net v2's NonDetMultiThreadedAugmenter spawns
        # worker processes that get a copy (via fork or pickle) of the
        # data loader. Pre-v17.2 we set
        # ``underlying.generate_train_batch = patched`` on the parent's
        # instance — but the workers had their own already-constructed
        # copies, so the patch never reached them. The startup self-test
        # passed (parent process) but actual training never saw a single
        # bias intercept (visible in every prior log as "calls=0" after
        # 250 iterations × 5 epochs of generated batches).
        #
        # The fix: replace ``np.random.choice`` itself, at the np.random
        # module level. Linux fork copies the parent's memory image, so
        # workers spawned AFTER this assignment see the patched function
        # automatically. Counters live in multiprocessing.Value so worker
        # increments propagate back to the parent for diagnostic output.
        #
        # The frame-inspection guard inside _global_biased_np_choice
        # ensures this only intercepts the specific
        #   ``np.random.choice(len(eligible_classes_or_regions))``
        # call inside nnU-Net's generate_train_batch — every other
        # numpy random call in the worker passes through untouched.
        # ─────────────────────────────────────────────────────────────────

        # Reset cross-process counters for this fold.
        _reset_bias_counters_for_new_fold()

        # Snapshot rules into module-level state. Workers spawned via
        # fork inherit this; spawn-mode workers re-import this module
        # (its identity is preserved by Python's import system) and
        # then read _BIAS_RULES_FOR_WORKERS at the time _global_biased_np_choice
        # actually fires.
        if not _install_global_bias_hook(rules):
            self._phase("patch_bias",
                "ABORT: global install failed (np.random.choice readonly?)")
            self._l6_patch_bias_enabled = False
            self._phase_end("patch_bias", ok=False, note="global install failed")
            return

        # Compatibility shim — older code paths in this file read
        # self._bias_call_count etc. Wire those reads to the shared
        # counters so existing log/payload logic just works.
        # (Properties not used; we set plain ints periodically. To
        # always read fresh, code that needs the live value should
        # call _read_bias_counters() directly. The lines below also
        # match what the v17 dataclass defaults declared.)
        self._bias_call_count = 0
        self._bias_l6_returns = 0
        self._bias_sacrum_returns = 0

        # First-batch diagnostic state — set on the trainer for the
        # parent process; ``_log_first_batch_diagnostic`` runs there
        # the first time we see a batch return back.
        self._first_batch_logged = False

        self._phase("patch_bias",
            f"installed: np.random.choice wrapped GLOBALLY at module "
            f"scope (worker forks inherit the patch via copy-on-write)")
        self._phase_end("patch_bias", ok=True,
            note=f"{len(targets)} cases match rules")

    def _refresh_bias_counters_from_shared(self) -> None:
        """Pull the latest cross-process counter values into the
        instance attrs so existing read sites (log lines, W&B payload,
        summary tables) continue to work without code changes."""
        c, l6, sa = _read_bias_counters()
        self._bias_call_count   = c
        self._bias_l6_returns   = l6
        self._bias_sacrum_returns = sa

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

            log("")
            log("  PATCH-CLASS BIAS HOOK")
            bias_enabled = getattr(self, "_l6_patch_bias_enabled", False)
            log(f"    enabled: {bias_enabled}")
            rules_resolved = getattr(self, "_patch_bias_rules_resolved", None)
            if rules_resolved:
                for sub, (target_cls, env_name, frac) in rules_resolved.items():
                    target_name = (_ANATOMY_NAMES[target_cls - 1]
                                   if 1 <= target_cls <= len(_ANATOMY_NAMES)
                                   else f"label_{target_cls}")
                    log(f"    rule: {sub:<20s} -> {target_name} "
                        f"(label={target_cls}) @ frac={frac:.2f} "
                        f"[env: {env_name}]")
            else:
                log("    no rules resolved (hook may have failed install)")
            log(f"    counters: calls={self._bias_call_count} "
                f"L6_returns={self._bias_l6_returns} "
                f"sacrum_returns={self._bias_sacrum_returns}")
            log("    NOTE: counters increment on first batch — see "
                "FIRST-BATCH log below for confirmation.")

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

    def _log_first_batch_diagnostic(self, batch_result) -> None:
        """One-shot log fired the first time generate_train_batch is
        entered. Confirms (a) the patched function was reached, (b)
        what case keys are in the first batch, and (c) the bias-counter
        state immediately after the first call."""
        log = self.print_to_log_file
        log("")
        log("─" * 72)
        log("FIRST BATCH DIAGNOSTIC (one-shot)")
        log("─" * 72)
        log("  ✓ patched generate_train_batch was called "
            "(if you see this, the hook is plumbed)")
        log(f"  bias counters after batch 0: "
            f"calls={self._bias_call_count} "
            f"L6_returns={self._bias_l6_returns} "
            f"sacrum_returns={self._bias_sacrum_returns}")

        keys = None
        if isinstance(batch_result, dict):
            for k in ("keys", "identifiers", "case_ids"):
                if k in batch_result:
                    cand = batch_result[k]
                    try:
                        keys = list(cand)
                        break
                    except Exception:
                        pass
        if keys:
            log(f"  case keys in first batch (n={len(keys)}): {keys}")
            if self._val_case_to_subtype:
                subs = [self._val_case_to_subtype.get(str(k), "?") for k in keys]
                log(f"  case subtypes:                       {subs}")
            pool = getattr(self, "_lstv_case_ids", None) or set()
            if pool:
                hits = [str(k) in pool for k in keys]
                log(f"  in oversample pool:                  {hits}  "
                    f"({sum(hits)}/{len(hits)} are LSTV)")
        else:
            log("  (case keys not exposed by this nnU-Net build; "
                "skipping per-key breakdown)")

        if self._bias_call_count == 0:
            log("  NOTE: bias_call_count=0 after first batch is normal "
                "if oversample_foreground_percent < 1.0 — force_fg")
            log("        patches are sampled probabilistically. Counter")
            log("        should grow over the first epoch; check the")
            log("        per-epoch summary log.")
        log("─" * 72)
        log("")

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
                dice_per_cls = _per_case_dice_per_class(
                    pred_argmax[b].cpu().to(torch.int16),
                    gt[b].cpu().to(torch.int16))
                self._val_subgroup_dice.setdefault(subtype, []).append(dice_per_cls)
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

        # Cumulative bias-hook diagnostic counters. Refresh from the
        # cross-process multiprocessing.Value shared state so worker
        # increments propagate into this payload (v17.2).
        try:
            self._refresh_bias_counters_from_shared()
        except Exception:
            pass
        out["debug/bias_hook/total_calls"]   = float(self._bias_call_count)
        out["debug/bias_hook/L6_picks"]      = float(self._bias_l6_returns)
        out["debug/bias_hook/sacrum_picks"]  = float(self._bias_sacrum_returns)
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
            # Bias-hook diagnostic line. If total_calls is zero by epoch
            # 1, the hook never fired and the install is broken.
            calls = int(payload.get("debug/bias_hook/total_calls", 0))
            l6p   = int(payload.get("debug/bias_hook/L6_picks", 0))
            sp    = int(payload.get("debug/bias_hook/sacrum_picks", 0))
            self.print_to_log_file(
                f"  bias hook (cumulative): calls={calls}  L6_picks={l6p}  "
                f"sacrum_picks={sp}")
        except Exception as exc:
            try: self.print_to_log_file(f"per-subgroup summary failed: {exc}")
            except Exception: pass

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
            "l6_patch_bias": _env_truthy("SPINESURG_L6_PATCH_BIAS", default=True),
            "l6_patch_bias_frac": _env_float("SPINESURG_L6_PATCH_BIAS_FRAC", 0.6),
            "lstv_oversample_frac": _env_float("LSTV_OVERSAMPLE_FRAC", 0.25),
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
        super().on_train_start()
        try: _install_nnunet_warning_filter()
        except Exception: pass
        try: self._maybe_load_subtype_map()
        except Exception as exc: self.print_to_log_file(f"subtype map: {exc}")
        try: self._maybe_apply_ce_reweighting()
        except Exception as exc: self.print_to_log_file(f"CE reweight: {exc}")
        try: self._maybe_apply_l6_patch_bias()
        except Exception as exc: self.print_to_log_file(f"L6 patch bias: {exc}")
        try: self._install_log_intercepts()
        except Exception: pass
        try: self._init_wandb()
        except Exception as exc: self.print_to_log_file(f"WandB init: {exc}")

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self._reset_subgroup_dice_buffer()

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
            # Lifetime bias-hook stats
            summary["summary/bias_hook_total_calls"]  = float(self._bias_call_count)
            summary["summary/bias_hook_L6_picks"]     = float(self._bias_l6_returns)
            summary["summary/bias_hook_sacrum_picks"] = float(self._bias_sacrum_returns)
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
# LSTV oversampling mixin (queue-level, 6-way)
# =============================================================================

class _LSTVOversampleMixin:
    """Oversample all 5 LSTV variant subtypes at the dataloader queue level.

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

    def on_train_start(self):
        super().on_train_start()
        try: self._apply_lstv_sampler()
        except Exception as exc: self.print_to_log_file(f"LSTV oversample: {exc}")
        # Banner runs LAST so it captures post-sampler pool composition.
        # super().on_train_start() above ran _WandBMixin's setup
        # (subtype map, CE reweight, patch-bias hook, W&B init);
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
    """500 epochs + LSTV queue oversample + L6 patch bias + CE reweight.
    DEFAULT for the SpineSurg-CT paper run.

    v17.2: Two production-bug fixes.
         (1) Patch-bias hook installed at module level (np.random.choice
             replaced globally with frame-inspection guard) instead of
             on the trainer instance. The v15-v17.1 instance-level
             install passed startup self-test in the parent process but
             never fired in worker processes spawned by
             NonDetMultiThreadedAugmenter — visible as "calls=0" every
             epoch in every fold's log. v17.2 patches via module state
             that survives fork/spawn. Counters via multiprocessing.Value
             so worker increments propagate to the parent.
         (2) Dedicated LSTV val pass: handle the case where the loader's
             keys attribute is a method (e.g., dict.keys) rather than
             a settable list. v17.1 raised "method object is not iterable"
             and silently no-op'd. v17.2 detects this at probe time,
             logs the actual attribute name + kind in the startup
             banner, and skips the forced override (which would replace
             the method with a list and break subsequent callers).
    v17.1: Dedicated LSTV validation pass — runs forward pass on every
         LSTV case in this fold's val set every epoch, regardless of
         whether the regular val sampler caught them in its
         num_val_iterations subsample. Eliminates n=0 LSTV val epochs
         that silenced the headline metric. Adds ~30s/epoch (8 cases
         × bs 2 forward passes on H200). Configurable via
         SPINESURG_LSTV_DEDICATED_VAL{,_EVERY} env vars.
    v17: Oversample pool now includes all 5 LSTV variants (added
         ambiguous, n=4). Symmetric per-record subtype refinement at
         convert time: spine_only demotes pelvis-region subtypes
         (sacralization, semisacralization), pelvic_native demotes
         lumbar-region subtypes (lumb, sacr_count). Verbose
         startup-diagnostic banner emitted via
         _emit_startup_diagnostics. First-batch diagnostic confirms
         hook + queue plumbing on the first generate_train_batch call.
    v16: Patch-bias hook now patches np.random.choice (was patching
         stdlib random.choice — wrong target, hook was a no-op).
         Added bias_call_count / L6_picks / sacrum_picks counters.
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
