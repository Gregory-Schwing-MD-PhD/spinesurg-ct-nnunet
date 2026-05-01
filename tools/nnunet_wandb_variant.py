"""
SpineSurg-CT -- nnU-Net v2 trainers with W&B logging (v10)
tools/nnunet_wandb_variant.py

Changes from v9
===============
1. 6-way LSTV subgroup binning. Mirrors splits_5fold.json schema v6 and
   lstv_cases.json schema v3:

     normal | lumb | sacr_count | semisacralization | sacralization | ambiguous

   Previously was 4-way (normal/lumb/sacr/ambiguous) which silently
   merged semisacralization into sacralization due to the parser bug
   in mask_index.py upstream (now fixed).

2. lstv_cases.json reader updated to schema v3:
   - Reads `case_to_subtype` (was `case_ids`)
   - Reads `lstv_oversample_pool` directly (was filtering manually)
   - Validates `splits_schema` field

3. Oversampling pool: only common LSTV (lumb, sacr_count, sacralization)
   are oversampled. The rare classes (semisacralization n=3 records,
   ambiguous n=4 records) are EXCLUDED to avoid memorization. They're
   still tracked for per-subgroup dice metrics.

4. L6 patch bias still triggers ONLY on lumb cases (not on any other
   subtype). L6 only appears in lumbarization morphology.

5. Per-subgroup tracking now reports 6 subgroups in WandB and stdout.
   Headline metrics expanded to track sacralization-flavored classes
   (sacr_count, semi, sacr) separately.

Unchanged from v9:
- cudnn.benchmark, TF32
- LSTV oversample queue mixin
- W&B logging with retry
- Profiler hooks (opt-in)
- CE reweighting (L6 weight=4.0 default)

Trainer __init__ rules: every subclass must declare full nnUNetTrainer
signature explicitly (nnUNetTrainer.__init__ inspects self.__init__'s
parameters and looks them up in locals()).

Author: Gregory Schwing, MD-PhD  |  Wayne State University / DMC
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
import warnings
from collections import Counter, defaultdict
from os.path import isfile, isdir, join
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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
_IGNORE_LABEL    = 10
_VALID_CONFIGS   = ("fused", "spine_only", "pelvic_native")

# 6-way taxonomy (matches splits_5fold.json v6 + lstv_cases.json v3)
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
# Subtypes used for queue-level oversampling. Excludes:
#   - normal       (we oversample MINORITY classes)
#   - semi (n=3)   (too rare; oversampling causes memorization)
#   - ambiguous (n=4) (cross-anatomy disagreement; n=2 patients)
_OVERSAMPLE_SUBTYPES = (_SUBTYPE_LUMB, _SUBTYPE_SACR_COUNT, _SUBTYPE_SACR)

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
    """Canonicalize an LSTV subtype string from any source.

    Returns one of the 6 canonical subtypes, or 'normal' for unknown.
    """
    if not sub: return _SUBTYPE_NORMAL
    s = str(sub).strip().lower()
    if s.startswith("ambig"):       return _SUBTYPE_AMBIG
    if "lumb" in s:                  return _SUBTYPE_LUMB
    # Order matters: 'semi' before 'sacr' since 'semisacralization' has both
    if "semi" in s:                  return _SUBTYPE_SEMI
    if "sacr_count" in s or "sacrcount" in s: return _SUBTYPE_SACR_COUNT
    if "sacr" in s:                  return _SUBTYPE_SACR
    return _SUBTYPE_NORMAL


def _read_lstv_cases_json(json_path: Path
                           ) -> Optional[Tuple[Set[str], Counter, Dict[str, str], List[str]]]:
    """
    Returns (all_case_ids, subtype_counts, case_id_to_subtype_map, oversample_pool) or None.

    Schema-aware reader:
      v3 (Apr 2026): 6-way taxonomy. Reads 'case_to_subtype' dict.
                     Reads 'lstv_oversample_pool' list directly.
      v2/v1 (legacy): 4-way taxonomy. Reads 'case_ids' dict (cid -> subtype string).
                      Builds oversample pool by filtering out ambiguous.

    Subtype strings are canonicalized via _canonicalize_subtype_str().
    """
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
        # New schema: case_to_subtype is the canonical map
        cts = data.get("case_to_subtype") or {}
        if not isinstance(cts, dict): return None
        for cid, sub in cts.items():
            canon = _canonicalize_subtype_str(sub)
            case_to_subtype[cid] = canon
            all_case_ids.add(cid)
        # Use the pool from JSON if available, else compute it
        oversample_pool = data.get("lstv_oversample_pool") or []
        if not oversample_pool:
            # Compute pool: oversample-eligible subtypes only
            oversample_pool = sorted(
                cid for cid, sub in case_to_subtype.items()
                if sub in _OVERSAMPLE_SUBTYPES
            )
    else:
        # Legacy schema: case_ids is the dict (no separate pool)
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
    """Fallback path: scan HF per-record manifests directly. 6-way aware."""
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
    """Last-resort: identify lumb cases by scanning labels for L6 voxels."""
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
# Class reweighting (unchanged from v9)
# =============================================================================

def _build_ce_class_weights(num_classes: int = 11) -> Optional[torch.Tensor]:
    if not _env_truthy("SPINESURG_CE_REWEIGHT", default=True):
        return None
    weights = {
        0: 0.5,  # background
        1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0,
        6: 4.0,  # L6
        7: 1.0, 8: 1.0, 9: 1.0,
        10: 0.0,  # ignore label
    }
    name_to_id = {"background": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5,
                   "L6": 6, "sacrum": 7, "left_hip": 8, "right_hip": 9, "ignore": 10}
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

    # Per-subgroup tracking (6-way)
    _val_subgroup_dice: Dict[str, List[List[Optional[float]]]] = None
    _val_case_to_subtype: Optional[Dict[str, str]] = None
    _per_subgroup_logging_enabled: bool = True

    # L6 patch bias
    _l6_patch_bias_enabled: bool = True
    _l6_patch_bias_target_frac: float = 0.6
    _lumb_case_id_set: Optional[Set[str]] = None

    # ── Progress JSON / log intercepts (unchanged) ──────────────────────

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

    # ── CE reweighting (unchanged from v9) ─────────────────────────────────

    def _maybe_apply_ce_reweighting(self):
        if not _env_truthy("SPINESURG_CE_REWEIGHT", default=True):
            self.print_to_log_file("CE reweighting: DISABLED via SPINESURG_CE_REWEIGHT=0")
            return
        try:
            from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
            from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
        except ImportError as e:
            self.print_to_log_file(f"CE reweighting: import failed: {e}; skipping.")
            return

        weights = _build_ce_class_weights(num_classes=11)
        if weights is None:
            self.print_to_log_file("CE reweighting: weights=None; skipping.")
            return

        device = getattr(self, "device", torch.device("cuda"))
        weights = weights.to(device)
        ds_enabled = bool(getattr(self, "enable_deep_supervision", True))

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
                new_loss = DeepSupervisionWrapper(new_loss, ds_weights)
            self.loss = new_loss
            self.print_to_log_file(
                f"CE reweighting: ENABLED. weights={weights.cpu().numpy().tolist()}  "
                f"deep_supervision={ds_enabled}")
        except Exception as exc:
            self.print_to_log_file(f"CE reweighting: failed to install: {exc}; using default loss.")

    # ── L6 patch sampling bias (only on lumb cases) ─────────────────────

    def _maybe_apply_l6_patch_bias(self):
        if not _env_truthy("SPINESURG_L6_PATCH_BIAS", default=True):
            self.print_to_log_file("L6 patch bias: DISABLED via SPINESURG_L6_PATCH_BIAS=0")
            return

        try:
            target_frac = _env_float("SPINESURG_L6_PATCH_BIAS_FRAC", 0.6)
            target_frac = max(0.0, min(0.95, target_frac))
        except Exception:
            target_frac = 0.6
        self._l6_patch_bias_target_frac = target_frac

        if self._val_case_to_subtype is None:
            self._maybe_load_subtype_map()

        # ONLY lumb cases get L6 bias (L6 only exists in lumbarization).
        lumb_ids: Set[str] = set()
        if self._val_case_to_subtype:
            for cid, sub in self._val_case_to_subtype.items():
                if sub == _SUBTYPE_LUMB:
                    lumb_ids.add(cid)
        self._lumb_case_id_set = lumb_ids
        self._l6_patch_bias_enabled = True

        if not lumb_ids:
            self.print_to_log_file(
                "L6 patch bias: subtype map empty or no lumb cases; bias hook will be inactive.")
            return

        self.print_to_log_file(
            f"L6 patch bias: ENABLED. "
            f"{len(lumb_ids)} lumbarization case_ids in bias pool, "
            f"target L6-fraction-of-fg-patches={target_frac:.2f}")

        loader = getattr(self, "dataloader_train", None)
        underlying = loader
        for attr in ("generator", "data_loader", "_data_loader"):
            if hasattr(underlying, attr):
                cand = getattr(underlying, attr)
                if hasattr(cand, "generate_train_batch") or hasattr(cand, "_data"):
                    underlying = cand
                    break
        if underlying is None or not hasattr(underlying, "generate_train_batch"):
            self.print_to_log_file(
                "L6 patch bias: cannot locate dataloader.generate_train_batch; "
                "framework internals may have changed. Disabling.")
            self._l6_patch_bias_enabled = False
            return

        trainer_ref = self
        original_gen = underlying.generate_train_batch

        def patched_generate_train_batch(*args, **kwargs):
            if not trainer_ref._l6_patch_bias_enabled:
                return original_gen(*args, **kwargs)
            lumb_ids = trainer_ref._lumb_case_id_set or set()
            if not lumb_ids:
                return original_gen(*args, **kwargs)
            target_frac = trainer_ref._l6_patch_bias_target_frac
            original_choice = random.choice

            def biased_choice(seq, *a, **k):
                try:
                    if isinstance(seq, (list, tuple)) and len(seq) > 1 \
                            and _L6_LABEL_ID in seq \
                            and all(isinstance(x, (int, np.integer)) for x in seq):
                        if random.random() < target_frac:
                            return _L6_LABEL_ID
                except Exception:
                    pass
                return original_choice(seq, *a, **k)

            random.choice = biased_choice
            try:
                return original_gen(*args, **kwargs)
            finally:
                random.choice = original_choice

        underlying.generate_train_batch = patched_generate_train_batch
        self.print_to_log_file(
            "L6 patch bias: hook installed on dataloader.generate_train_batch.")

    # ── Per-subgroup setup (6-way) ───────────────────────────────────────

    def _maybe_load_subtype_map(self):
        self._per_subgroup_logging_enabled = _env_truthy(
            "SPINESURG_LSTV_PER_GROUP_DICE", default=True)
        if not self._per_subgroup_logging_enabled:
            self.print_to_log_file("LSTV per-subgroup dice: DISABLED")
            return
        if self._val_case_to_subtype is not None:
            return
        try: ds_name = self.plans_manager.dataset_name
        except Exception: ds_name = None
        case_to_sub: Dict[str, str] = {}
        if ds_name:
            for jp in _candidate_lstv_json_paths(ds_name):
                result = _read_lstv_cases_json(jp)
                if result is not None:
                    _, subtypes, case_to_sub, _ = result
                    self.print_to_log_file(
                        f"LSTV per-subgroup: loaded subtype map from {jp} "
                        f"({len(case_to_sub)} cases total)")
                    self.print_to_log_file(f"  subtype counts: {dict(subtypes)}")
                    break
        if not case_to_sub:
            for hf_dir in _candidate_hf_export_dirs():
                result = _read_lstv_records_from_manifest(hf_dir)
                if result is not None:
                    _, _, case_to_sub, _ = result
                    self.print_to_log_file(
                        f"LSTV per-subgroup: loaded subtype map from manifests at {hf_dir}")
                    break
        self._val_case_to_subtype = case_to_sub
        if not case_to_sub:
            self.print_to_log_file("LSTV per-subgroup: no subtype map; all cases -> 'normal'")

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
            keys_raw = batch.get("keys") or batch.get("identifier") or []
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

        # Headline: L6 dice on lumbarization
        lumb_cases = self._val_subgroup_dice.get(_SUBTYPE_LUMB, [])
        if lumb_cases:
            l6_idx = _FG_CLASS_IDS.index(_L6_LABEL_ID)
            l6_vals = [c[l6_idx] for c in lumb_cases if c[l6_idx] is not None]
            if l6_vals:
                out["val/headline/L6_dice_on_lumbarization"] = float(np.mean(l6_vals))
                out["val/headline/L6_n_lumbarization_cases"] = float(len(l6_vals))

        # Combined "any LSTV" composite (everything except normal)
        lstv_means: List[float] = []
        for sub in (_SUBTYPE_LUMB, _SUBTYPE_SACR_COUNT, _SUBTYPE_SEMI,
                     _SUBTYPE_SACR, _SUBTYPE_AMBIG):
            for case_dice in self._val_subgroup_dice.get(sub, []):
                vals = [v for v in case_dice if v is not None]
                if vals: lstv_means.append(float(np.mean(vals)))
        if lstv_means:
            out["val/lstv_subgroup/any_lstv/mean_fg_dice"] = float(np.mean(lstv_means))
            out["val/lstv_subgroup/any_lstv/n_cases"] = float(len(lstv_means))

        # Composite: any sacralization-flavored (sacr_count + semi + sacr)
        any_sacr_means: List[float] = []
        for sub in (_SUBTYPE_SACR_COUNT, _SUBTYPE_SEMI, _SUBTYPE_SACR):
            for case_dice in self._val_subgroup_dice.get(sub, []):
                vals = [v for v in case_dice if v is not None]
                if vals: any_sacr_means.append(float(np.mean(vals)))
        if any_sacr_means:
            out["val/lstv_subgroup/any_sacralization/mean_fg_dice"] = float(np.mean(any_sacr_means))
            out["val/lstv_subgroup/any_sacralization/n_cases"] = float(len(any_sacr_means))

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
    """Oversample lumb + sacr_count + sacralization cases at the dataloader queue level.

    EXCLUDED from oversampling (n too small):
    - normal (we oversample MINORITIES)
    - semisacralization (n=3 records, oversampling memorizes)
    - ambiguous (n=4 records, cross-anatomy disagreement; n=2 patients)

    Both rare classes are still tracked for per-subgroup metrics and
    evaluated cleanly on val.
    """
    _lstv_case_ids: Optional[Set[str]] = None
    _lstv_subtype_counts: Optional[Counter] = None
    _lstv_detection_source: Optional[str] = None

    def _identify_lstv_cases(self) -> Set[str]:
        if self._lstv_case_ids is not None:
            return self._lstv_case_ids
        try: ds_name = self.plans_manager.dataset_name
        except Exception: ds_name = None

        # Try lstv_cases.json first (preferred path; v3 schema gives us the pool directly)
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
                    n_excluded = len(case_to_sub) - len(pool_set) - subtypes.get(_SUBTYPE_NORMAL, 0)
                    self.print_to_log_file(
                        f"LSTV oversample: loaded {len(case_to_sub)} cases; "
                        f"using {len(pool_set)} for oversampling "
                        f"(excluded {n_excluded} as too rare: semi/ambig). "
                        f"Subtypes: {dict(subtypes)}")
                    return pool_set

        # Fallback: HF manifests
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

        # Last resort: legacy L6 voxel scan (lumb only)
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

    v10 changes: 6-way subgroups (semisacralization separated from
    sacralization). Oversample pool excludes semi (n=3) and ambiguous (n=4).
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
