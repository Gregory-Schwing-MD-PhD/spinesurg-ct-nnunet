"""
SpineSurg-CT -- nnU-Net v2 trainers with W&B logging (v8 FINAL)
tools/nnunet_wandb_variant.py

Changes from v7.2 (FINAL)
=========================
1. LSTV_OVERSAMPLE_FRAC default lowered from 0.50 to 0.25.
   - At 50% target frac with 19 lumbarization + 27 sacralization cases
     we were duplicating each unique LSTV case ~20x per epoch, which
     is high overfitting risk on rare classes.
   - At 25% the duplication factor drops to ~9x, still oversampling
     enough to give the model meaningful exposure to LSTV anatomy
     while leaving room for the augmentation pipeline to introduce
     novel views of each case rather than just memorizing them.
   - Override at runtime via LSTV_OVERSAMPLE_FRAC=0.50 if needed.

2. Per-LSTV-subgroup validation dice logging.
   - Splits val cases into 'normal', 'lumbarization', 'sacralization'
     based on lstv_cases.json subtype tags.
   - Computes per-class dice on each val case independently, tagged
     with its subgroup.
   - Logs to W&B as:
         val/lstv_subgroup/{subtype}/dice/{class_name}
         val/lstv_subgroup/{subtype}/n_cases
         val/lstv_subgroup/{subtype}/mean_lumbar_dice
         val/lstv_subgroup/{subtype}/mean_pelvis_dice
   - Also logs L6 dice on lumbarization-only cases as the headline
     "rare class" metric since this is the paper's key claim.
   - Existing global dice logging is unchanged; this is additive.
   - Disable with SPINESURG_LSTV_PER_GROUP_DICE=0 if computation
     overhead becomes a problem (it shouldn't; per-case dice on
     val patches is cheap).

What's on (unchanged from v7.2)
================================
1. cudnn.benchmark = True
2. set_float32_matmul_precision('high') (TF32 for fp32 matmul)
3. Profiler hooks (opt-in via SPINESURG_PROFILE=1)
4. channels_last_3d toggle (opt-in; empirically slower)
5. LSTV oversampling for both subtypes (sacralization + lumbarization)
6. W&B init with retry + offline fallback
7. NaN-safe dice aggregation
8. SPINESURG_RUN_VAL_EXPORT toggle for perform_actual_validation
9. SPINESURG_PERF=0 escape hatch to disable perf tuning

What's NOT on (unchanged from v7.2)
====================================
- torch.compile mode='reduce-overhead' / CUDA graphs:
  Deep supervision returns a list of tensors; validation reuses these
  after forward returns. CUDA graphs reuse output buffers, which gets
  overwritten by the dataloader's next prefetched batch -> RuntimeError
  during validation. nnU-Net's default compile (mode='default') works.

- Tensor.to default-non_blocking=True monkey-patch:
  PyTorch's nn.Module._apply -> convert calls t.to(device, dtype,
  non_blocking) with non_blocking as a positional arg. The patch
  detected "non_blocking not in kwargs" and added it as a kwarg,
  producing a duplicate-arg TypeError that killed every launch.

- channels_last_3d (toggle off by default):
  ~13.5% transpose overhead replaced by ~21% clone+contig+copy from
  3D instance norm not having a NHWC kernel path. Net loss.

Trainer __init__ rules
======================
Every trainer subclass that overrides __init__ MUST declare the FULL
nnUNetTrainer signature explicitly. nnUNetTrainer.__init__ inspects
self.__init__'s parameters and looks them up in its own locals().
Using *args/**kwargs makes it look up locals()['args'], raising
KeyError. Required signature:
    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):

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
    nnUNetTrainer_5epochs,
    nnUNetTrainer_100epochs,
)


_METRIC_KEYS = (
    "train_losses", "val_losses",
    "mean_fg_dice", "ema_fg_dice",
    "dice_per_class_or_region",
    "lrs",
    "epoch_start_timestamps", "epoch_end_timestamps",
)

_ANATOMY_NAMES = [
    "L1", "L2", "L3", "L4", "L5", "L6",
    "sacrum", "left_hip", "right_hip",
]

# Class IDs in the segmentation map (matches CLASS_NAMES in
# tools/convert_hf_to_nnunet.py). 0 = background, 1-6 = lumbar, 7-9 = pelvis,
# 10 = ignore.
_FG_CLASS_IDS    = list(range(1, 10))     # 9 foreground classes
_LUMBAR_IDS      = list(range(1, 7))      # L1..L6
_PELVIS_IDS      = list(range(7, 10))     # sacrum, hips
_L6_LABEL_ID     = 6
_IGNORE_LABEL    = 10
_VALID_CONFIGS   = ("fused", "spine_only", "pelvic_native")
_SUBTYPE_NORMAL  = "normal"
_SUBTYPE_LUMB    = "lumbarization"
_SUBTYPE_SACR    = "sacralization"
_KNOWN_SUBTYPES  = (_SUBTYPE_NORMAL, _SUBTYPE_LUMB, _SUBTYPE_SACR)
LSTV_CASES_SCHEMA_VERSION = 1


# -----------------------------------------------------------------------------
# Warning + log suppression
# -----------------------------------------------------------------------------

def _install_nnunet_warning_filter() -> None:
    """Suppress benign numpy + torch.compile/dynamo/inductor noise."""
    warnings.filterwarnings(
        "ignore",
        message="invalid value encountered in scalar divide",
        category=RuntimeWarning,
    )
    np.seterr(invalid="ignore", divide="warn")

    for logger_name in (
        "torch.fx.experimental.symbolic_shapes",
        "torch._dynamo",
        "torch._dynamo.symbolic_convert",
        "torch._dynamo.guards",
        "torch._dynamo.output_graph",
        "torch._inductor",
        "torch._inductor.compile_fx",
        "torch._inductor.scheduler",
        "torch._inductor.codecache",
        "torch._functorch",
    ):
        logging.getLogger(logger_name).setLevel(logging.ERROR)


# -----------------------------------------------------------------------------
# Performance tuning (no monkey-patches)
# -----------------------------------------------------------------------------

def _install_perf_tuning() -> None:
    """cudnn.benchmark + TF32 only. Both safe, both small wins."""
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass


def _restore_perf_tuning() -> None:
    """No-op (no monkey-patches to restore)."""
    pass


# -----------------------------------------------------------------------------
# Env helpers
# -----------------------------------------------------------------------------

def _env_truthy(name: str, default: bool = False) -> bool:
    val = os.environ.get(name, "").strip().lower()
    if not val:
        return default
    return val in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


# -----------------------------------------------------------------------------
# LSTV case identification
# -----------------------------------------------------------------------------

def _safe_id(token: str) -> str:
    """Mirror of convert_hf_to_nnunet.py:safe_id() — must stay in sync."""
    s = str(token)
    for bad in (".", "/", "\\", " ", ":", "(", ")"):
        s = s.replace(bad, "_")
    return s.strip("_")


def _read_lstv_cases_json(json_path: Path
                           ) -> Optional[Tuple[Set[str], Counter, Dict[str, str]]]:
    """
    Returns (case_id_set, subtype_counts, case_id_to_subtype_map) or None.

    The third element is needed by the per-subgroup dice logger so each
    case_id can be classified into 'normal' / 'lumbarization' /
    'sacralization' on the fly during validation.
    """
    if not json_path.exists():
        return None
    try:
        data = json.loads(json_path.read_text())
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if data.get("schema_version") != LSTV_CASES_SCHEMA_VERSION:
        return None
    case_ids_dict = data.get("case_ids") or {}
    if not isinstance(case_ids_dict, dict):
        return None
    case_ids = set(case_ids_dict.keys())
    subtype_counts = Counter()
    case_to_subtype: Dict[str, str] = {}
    for cid, subtype in case_ids_dict.items():
        if not isinstance(subtype, str):
            continue
        sub = subtype.strip().lower()
        if not sub:
            continue
        # Normalize subtype to the canonical set; partial matches are OK
        if "lumb" in sub:
            sub_canon = _SUBTYPE_LUMB
        elif "sacr" in sub:
            sub_canon = _SUBTYPE_SACR
        elif sub == "normal":
            sub_canon = _SUBTYPE_NORMAL
        else:
            # Unknown subtype label — log it but include it as an "lstv"
            # bucket so it's not lost. Falls into 'normal' for grouping
            # since we don't know which way to bin it.
            sub_canon = sub
        subtype_counts[sub_canon] += 1
        case_to_subtype[cid] = sub_canon
    return case_ids, subtype_counts, case_to_subtype


def _candidate_lstv_json_paths(dataset_name: str) -> List[Path]:
    cands: List[Path] = []
    pre = os.environ.get("nnUNet_preprocessed")
    if pre:
        cands.append(Path(pre) / dataset_name / "lstv_cases.json")
    raw = os.environ.get("nnUNet_raw") or os.environ.get("nnUNet_raw_data_base")
    if raw:
        cands.append(Path(raw) / dataset_name / "lstv_cases.json")
    return cands


def _candidate_hf_export_dirs() -> List[Path]:
    cands: List[Path] = []
    env = os.environ.get("HF_EXPORT_DIR", "").strip()
    if env:
        cands.append(Path(env))
    home = Path.home()
    cands.append(home / "CTSpinoPelvic1K" / "data" / "hf_export")
    cands.append(home / "spinesurg-ct-nnunet" / "data" / "hf_export")
    raw = os.environ.get("nnUNet_raw") or os.environ.get("nnUNet_raw_data_base")
    if raw:
        cands.append(Path(raw).parent.parent / "data" / "hf_export")
        cands.append(Path(raw).parent / "data" / "hf_export")
    seen: Set[str] = set()
    uniq: List[Path] = []
    for c in cands:
        s = str(c)
        if s not in seen:
            seen.add(s)
            uniq.append(c)
    return uniq


def _read_lstv_records_from_manifest(hf_export_dir: Path
                                      ) -> Optional[Tuple[Set[str], Counter, Dict[str, str]]]:
    if not hf_export_dir.is_dir():
        return None
    records: List[Dict] = []
    for p in (hf_export_dir / "manifest_train.json",
              hf_export_dir / "manifest_validation.json",
              hf_export_dir / "manifest_test.json"):
        if p.exists():
            try:
                data = json.loads(p.read_text())
                if isinstance(data, list):
                    records.extend(data)
                elif isinstance(data, dict) and "records" in data:
                    records.extend(data["records"])
            except Exception:
                continue
    if not records:
        combined = hf_export_dir / "manifest.json"
        if combined.exists():
            try:
                data = json.loads(combined.read_text())
                if isinstance(data, list):
                    records = data
                elif isinstance(data, dict) and "records" in data:
                    records = data["records"]
            except Exception:
                return None
    if not records:
        return None
    case_ids: Set[str] = set()
    subtypes: Counter = Counter()
    case_to_subtype: Dict[str, str] = {}
    for r in records:
        lstv = str(r.get("lstv_label", "")).strip().lower()
        if not lstv or lstv == "normal":
            continue
        token = r.get("token")
        config = r.get("config", "fused")
        if token is None or config not in _VALID_CONFIGS:
            continue
        cid = f"{_safe_id(token)}__{config}"
        case_ids.add(cid)
        if "lumb" in lstv:
            sub = _SUBTYPE_LUMB
        elif "sacr" in lstv:
            sub = _SUBTYPE_SACR
        else:
            sub = lstv
        subtypes[sub] += 1
        case_to_subtype[cid] = sub
    if not case_ids:
        return None
    return case_ids, subtypes, case_to_subtype


def _scan_lstv_case_ids_by_l6_voxels(labels_dir: str) -> Set[str]:
    import nibabel as nib
    out: Set[str] = set()
    if not os.path.isdir(labels_dir):
        return out
    for fn in sorted(os.listdir(labels_dir)):
        if not fn.endswith(".nii.gz"):
            continue
        try:
            arr = np.asarray(nib.load(join(labels_dir, fn)).dataobj).astype(np.int16)
            if np.any(arr == _L6_LABEL_ID):
                out.add(fn[: -len(".nii.gz")])
        except Exception:
            continue
    return out


# =============================================================================
# Per-case dice (used by the per-subgroup validation logger)
# =============================================================================

def _per_case_dice_per_class(pred_argmax: torch.Tensor,
                              gt: torch.Tensor,
                              ignore_label: int = _IGNORE_LABEL,
                              num_classes: int = 10
                              ) -> List[Optional[float]]:
    """
    Compute soft-dice for each non-background class on a single case.

    Args:
        pred_argmax: (D, H, W) int tensor of predicted class IDs
        gt:          (D, H, W) int tensor of ground truth class IDs
        ignore_label: voxels with this GT label are excluded from both
                      numerator and denominator.
        num_classes: total class count including background and ignore.

    Returns a list of length len(_FG_CLASS_IDS), with None for classes
    that have no ground truth voxels in this case (avoids polluting the
    aggregate with cases that don't contain the class).
    """
    pred_argmax = pred_argmax.flatten()
    gt          = gt.flatten()

    valid = gt != ignore_label
    if not valid.any():
        return [None] * len(_FG_CLASS_IDS)
    pred_argmax = pred_argmax[valid]
    gt          = gt[valid]

    out: List[Optional[float]] = []
    for cid in _FG_CLASS_IDS:
        gt_mask   = gt == cid
        pred_mask = pred_argmax == cid
        gt_n   = int(gt_mask.sum().item())
        if gt_n == 0:
            # GT has no voxels of this class → dice is undefined for
            # this case. Skip rather than report 0 or 1.
            out.append(None)
            continue
        pred_n = int(pred_mask.sum().item())
        inter  = int((gt_mask & pred_mask).sum().item())
        denom  = gt_n + pred_n
        if denom == 0:
            out.append(None)
        else:
            out.append(2.0 * inter / denom)
    return out


# =============================================================================
# W&B mixin (with perf-tuning, profiler, channels_last hooks)
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

    # Per-subgroup dice tracking (populated each val epoch, cleared in
    # on_epoch_end after logging).
    _val_subgroup_dice: Dict[str, List[List[Optional[float]]]] = None
    _val_case_to_subtype: Optional[Dict[str, str]] = None
    _per_subgroup_logging_enabled: bool = True

    def _progress_json_path(self) -> Optional[str]:
        out = getattr(self, "output_folder", None)
        return join(out, "progress.json") if out else None

    def _load_progress_json(self) -> dict:
        path = self._progress_json_path()
        if path is None or not isfile(path):
            return {}
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _wrap_log_method(self, obj, cache) -> bool:
        try:
            if not hasattr(obj, "log") or not callable(obj.log):
                return False
            if self._intercepted_objects is None:
                self._intercepted_objects = set()
            if id(obj) in self._intercepted_objects:
                return False
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
                        epoch_i = int(epoch)
                        seq = cache[key]
                        while len(seq) < epoch_i + 1:
                            seq.append(None)
                        seq[epoch_i] = value
                except Exception:
                    pass
                return original_log(*args, **kwargs)

            obj.log = log_intercept
            self._intercepted_objects.add(id(obj))
            return True
        except Exception:
            return False

    def _install_log_intercepts(self) -> None:
        if self._metric_cache is None:
            self._metric_cache = defaultdict(list)
        cache = self._metric_cache
        logger = getattr(self, "logger", None)
        if logger is None:
            return
        wrapped = []
        if self._wrap_log_method(logger, cache):
            wrapped.append(type(logger).__name__)
        for attr in dir(logger):
            if attr.startswith("_"):
                continue
            try:
                val = getattr(logger, attr)
            except Exception:
                continue
            if isinstance(val, (list, tuple)):
                for i, child in enumerate(val):
                    if hasattr(child, "log") and callable(getattr(child, "log", None)):
                        if self._wrap_log_method(child, cache):
                            wrapped.append(f"{attr}[{i}]:{type(child).__name__}")
            elif hasattr(val, "log") and callable(getattr(val, "log", None)):
                if val is not logger and self._wrap_log_method(val, cache):
                    wrapped.append(f"{attr}:{type(val).__name__}")
        if wrapped:
            self.print_to_log_file(f"WandB: log intercepts installed on: {wrapped}")

    def _current_logs(self) -> dict:
        disk = self._load_progress_json()
        if disk and any(k in disk for k in _METRIC_KEYS):
            return disk
        if self._metric_cache is not None and len(self._metric_cache) > 0:
            return dict(self._metric_cache)
        return {}

    def _dump_environment(self) -> None:
        try:
            logger = getattr(self, "logger", None)
            out = getattr(self, "output_folder", None)
            p = self._progress_json_path()
            self.print_to_log_file(
                f"WandB diag: logger class = "
                f"{type(logger).__name__ if logger else None}")
            self.print_to_log_file(f"WandB diag: output_folder = {out}")
            self.print_to_log_file(
                f"WandB diag: progress.json = {p}  "
                f"exists={isfile(p) if p else False}")
            self.print_to_log_file(
                f"WandB diag: anatomy mapping = "
                f"{dict(enumerate(_ANATOMY_NAMES))}")
            self.print_to_log_file(
                f"WandB diag: SPINESURG_RUN_VAL_EXPORT = "
                f"{os.environ.get('SPINESURG_RUN_VAL_EXPORT', '<unset>')}  "
                f"(effective = {_env_truthy('SPINESURG_RUN_VAL_EXPORT')})")
            self.print_to_log_file(
                f"WandB diag: nnUNet_compile = "
                f"{os.environ.get('nnUNet_compile', '<unset>')}")
            self.print_to_log_file(
                f"WandB diag: SPINESURG_LSTV_PER_GROUP_DICE = "
                f"{os.environ.get('SPINESURG_LSTV_PER_GROUP_DICE', '<unset>')}  "
                f"(effective = {self._per_subgroup_logging_enabled})")
        except Exception:
            pass

    # ---- Perf tuning -----------------------------------------------------

    def _maybe_install_perf_tuning(self) -> None:
        if os.environ.get("SPINESURG_PERF", "1").strip().lower() in ("0", "false", "off", "no"):
            self.print_to_log_file("Perf tuning: DISABLED via SPINESURG_PERF=0")
            return
        try:
            _install_perf_tuning()
            self._perf_tuning_installed = True
            self.print_to_log_file(
                "Perf tuning: ENABLED (cudnn.benchmark, tf32)")
        except Exception as exc:
            self.print_to_log_file(f"Perf tuning: install failed: {exc}")

    def _restore_perf_tuning_safe(self) -> None:
        # No-op (no monkey patches to restore)
        pass

    # ---- channels_last_3d (opt-in) ---------------------------------------

    def _maybe_apply_channels_last(self) -> None:
        if not _env_truthy("SPINESURG_CHANNELS_LAST"):
            return
        net = getattr(self, "network", None)
        if net is None:
            self.print_to_log_file(
                "channels_last_3d: self.network is None; skipping.")
            return
        try:
            net.to(memory_format=torch.channels_last_3d)
            self._channels_last_active = True
            self.print_to_log_file(
                "channels_last_3d: model converted. "
                "Per-batch input conversion in train_step / validation_step.")
            self.print_to_log_file(
                "channels_last_3d: WARNING -- empirically SLOWER on "
                "this model. Use only for experimentation.")
        except Exception as exc:
            self.print_to_log_file(
                f"channels_last_3d: conversion failed: {exc}. Disabling.")
            self._channels_last_active = False

    # ---- Profiler --------------------------------------------------------

    def _maybe_install_profiler(self) -> None:
        if not _env_truthy("SPINESURG_PROFILE"):
            return
        try:
            from torch.profiler import (
                profile, schedule, tensorboard_trace_handler, ProfilerActivity
            )
        except ImportError:
            self.print_to_log_file("Profile: torch.profiler not available; skipping.")
            return

        out_dir = join(self.output_folder, "profile")
        os.makedirs(out_dir, exist_ok=True)

        wait_steps   = _env_int("SPINESURG_PROFILE_WAIT",   5)
        warmup_steps = _env_int("SPINESURG_PROFILE_WARMUP", 3)
        active_steps = _env_int("SPINESURG_PROFILE_ACTIVE", 10)

        try:
            self._profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schedule(
                    wait=wait_steps, warmup=warmup_steps,
                    active=active_steps, repeat=1,
                ),
                on_trace_ready=tensorboard_trace_handler(out_dir),
                record_shapes=True,
                with_stack=False,
                profile_memory=False,
            )
            self._profiler.start()
            self._profiler_total_steps = wait_steps + warmup_steps + active_steps
            self._profiler_step_count = 0
            self.print_to_log_file(
                f"Profile: ENABLED "
                f"(wait={wait_steps} warmup={warmup_steps} active={active_steps}, "
                f"total {self._profiler_total_steps} steps)")
            self.print_to_log_file(f"Profile: trace dir = {out_dir}")
        except Exception as exc:
            self.print_to_log_file(f"Profile: install failed: {exc}")
            self._profiler = None

    def _stop_profiler(self) -> None:
        prof = getattr(self, "_profiler", None)
        if prof is None:
            return
        try:
            self.print_to_log_file("=" * 70)
            self.print_to_log_file("Profile summary -- top 25 ops by CUDA total time:")
            self.print_to_log_file(prof.key_averages().table(
                sort_by="cuda_time_total", row_limit=25))
            self.print_to_log_file("-" * 70)
            self.print_to_log_file("Profile summary -- top 25 ops by CPU total time:")
            self.print_to_log_file(prof.key_averages().table(
                sort_by="cpu_time_total", row_limit=25))
            self.print_to_log_file("=" * 70)
            prof.stop()
            self.print_to_log_file(
                "Profile: stopped. Training continues without overhead.")
        except Exception as exc:
            self.print_to_log_file(f"Profile: stop failed: {exc}")
        self._profiler = None

    # ---- Per-subgroup val dice setup ------------------------------------

    def _maybe_load_subtype_map(self) -> None:
        """Load lstv_cases.json into self._val_case_to_subtype.

        Returns silently if disabled, file missing, or already loaded.
        Cases not in the file are treated as 'normal' at validation time.
        """
        self._per_subgroup_logging_enabled = _env_truthy(
            "SPINESURG_LSTV_PER_GROUP_DICE", default=True)
        if not self._per_subgroup_logging_enabled:
            self.print_to_log_file(
                "LSTV per-subgroup dice: DISABLED via "
                "SPINESURG_LSTV_PER_GROUP_DICE=0")
            return
        if self._val_case_to_subtype is not None:
            return
        try:
            ds_name = self.plans_manager.dataset_name
        except Exception:
            ds_name = None
        case_to_sub: Dict[str, str] = {}
        if ds_name:
            for json_path in _candidate_lstv_json_paths(ds_name):
                result = _read_lstv_cases_json(json_path)
                if result is not None:
                    _, _, case_to_sub = result
                    self.print_to_log_file(
                        f"LSTV per-subgroup dice: loaded subtype map from "
                        f"{json_path} ({len(case_to_sub)} LSTV cases)")
                    break
        if not case_to_sub:
            for hf_dir in _candidate_hf_export_dirs():
                result = _read_lstv_records_from_manifest(hf_dir)
                if result is not None:
                    _, _, case_to_sub = result
                    self.print_to_log_file(
                        f"LSTV per-subgroup dice: loaded subtype map from "
                        f"manifests at {hf_dir} ({len(case_to_sub)} LSTV cases)")
                    break
        self._val_case_to_subtype = case_to_sub
        if not case_to_sub:
            self.print_to_log_file(
                "LSTV per-subgroup dice: no subtype map found; per-subgroup "
                "logging will run but all cases will fall into 'normal' bucket.")

    def _reset_subgroup_dice_buffer(self) -> None:
        self._val_subgroup_dice = {
            _SUBTYPE_NORMAL: [],
            _SUBTYPE_LUMB:   [],
            _SUBTYPE_SACR:   [],
        }

    def _subtype_for_case(self, case_id: str) -> str:
        """Return the canonical subtype for a case_id ('normal' if unknown)."""
        if not self._val_case_to_subtype:
            return _SUBTYPE_NORMAL
        return self._val_case_to_subtype.get(case_id, _SUBTYPE_NORMAL)

    def _record_per_case_dice(self, batch, output) -> None:
        """
        Compute per-case per-class dice for a validation batch and stash
        into self._val_subgroup_dice[<subtype>].

        Failure-tolerant: any exception just bails on this batch (we
        don't want to break training because per-subgroup dice failed).
        """
        if not self._per_subgroup_logging_enabled:
            return
        if self._val_subgroup_dice is None:
            self._reset_subgroup_dice_buffer()
        try:
            # nnU-Net's deep supervision returns a list of tensors;
            # the first element is the highest-resolution prediction.
            if isinstance(output, (list, tuple)):
                pred_logits = output[0]
            else:
                pred_logits = output
            # gt is "target" in nnU-Net; same deep supervision shape.
            gt = batch.get("target")
            if isinstance(gt, (list, tuple)):
                gt = gt[0]
            # Extract case_ids robustly. Different nnU-Net versions stash
            # them as 'keys' (list of str), 'identifier' (list of str),
            # or sometimes a numpy/torch array. Normalize to a flat
            # list of strings before any truthiness check -- don't use
            # `or` on these because a multi-element numpy/torch array
            # raises "truth value ambiguous" when bool()-checked.
            keys_raw = batch.get("keys")
            if keys_raw is None:
                keys_raw = batch.get("identifier")
            if keys_raw is None:
                keys_raw = []
            if isinstance(keys_raw, str):
                keys = [keys_raw]
            elif hasattr(keys_raw, "tolist"):
                # numpy / torch tensor
                keys = [str(k) for k in keys_raw.tolist()]
            else:
                try:
                    keys = [str(k) for k in keys_raw]
                except TypeError:
                    keys = [str(keys_raw)]
            if pred_logits is None or gt is None or len(keys) == 0:
                return

            # pred_logits: (B, C, D, H, W); gt: (B, 1, D, H, W) or (B, D, H, W)
            if pred_logits.dim() != 5:
                return
            if gt.dim() == 5 and gt.size(1) == 1:
                gt = gt[:, 0]
            elif gt.dim() != 4:
                return

            with torch.no_grad():
                pred_argmax = pred_logits.argmax(dim=1)  # (B, D, H, W)

            B = pred_argmax.size(0)
            if len(keys) != B:
                # If we can't map case_ids 1:1 to batch dims, bail.
                return

            for b in range(B):
                cid = str(keys[b])
                # nnU-Net sometimes appends "_0000" to case_ids in keys;
                # strip it for the subtype lookup since lstv_cases.json
                # stores plain case_ids.
                cid_lookup = cid[:-5] if cid.endswith("_0000") else cid
                subtype = self._subtype_for_case(cid_lookup)
                dice_per_cls = _per_case_dice_per_class(
                    pred_argmax[b].cpu().to(torch.int16),
                    gt[b].cpu().to(torch.int16),
                )
                bucket = self._val_subgroup_dice.setdefault(subtype, [])
                bucket.append(dice_per_cls)
        except Exception as exc:
            try:
                self.print_to_log_file(
                    f"per-subgroup dice: batch failed silently: {exc}")
            except Exception:
                pass

    def _aggregate_subgroup_dice(self) -> Dict[str, float]:
        """
        Aggregate the buffered per-case per-class dice into W&B-loggable
        keys. Returns a flat dict suitable for self._log_wandb().
        """
        out: Dict[str, float] = {}
        if not self._per_subgroup_logging_enabled or self._val_subgroup_dice is None:
            return out

        for subtype, cases in self._val_subgroup_dice.items():
            n_cases = len(cases)
            out[f"val/lstv_subgroup/{subtype}/n_cases"] = float(n_cases)
            if n_cases == 0:
                continue

            # Per-class mean dice (nan-mean across cases that had this class)
            per_class_means: Dict[int, float] = {}
            for ci, cls_id in enumerate(_FG_CLASS_IDS):
                vals = [c[ci] for c in cases if c[ci] is not None]
                if not vals:
                    continue
                m = float(np.mean(vals))
                per_class_means[cls_id] = m
                cname = _ANATOMY_NAMES[ci]
                out[f"val/lstv_subgroup/{subtype}/dice/{cname}"] = m
                out[f"val/lstv_subgroup/{subtype}/n_with_class/{cname}"] = float(len(vals))

            # Group-level summary metrics
            lumbar_means = [per_class_means[c] for c in _LUMBAR_IDS
                              if c in per_class_means]
            pelvis_means = [per_class_means[c] for c in _PELVIS_IDS
                              if c in per_class_means]
            fg_means = list(per_class_means.values())
            if lumbar_means:
                out[f"val/lstv_subgroup/{subtype}/mean_lumbar_dice"] = float(np.mean(lumbar_means))
            if pelvis_means:
                out[f"val/lstv_subgroup/{subtype}/mean_pelvis_dice"] = float(np.mean(pelvis_means))
            if fg_means:
                out[f"val/lstv_subgroup/{subtype}/mean_fg_dice"] = float(np.mean(fg_means))

        # Headline metric: L6 dice on lumbarization-only val cases.
        # This is the paper's key claim — TotalSegmentator can't see L6
        # because it doesn't exist in their training data; we should.
        lumb_cases = self._val_subgroup_dice.get(_SUBTYPE_LUMB, [])
        if lumb_cases:
            try:
                l6_idx = _FG_CLASS_IDS.index(_L6_LABEL_ID)
                l6_vals = [c[l6_idx] for c in lumb_cases if c[l6_idx] is not None]
                if l6_vals:
                    out["val/headline/L6_dice_on_lumbarization"] = float(np.mean(l6_vals))
                    out["val/headline/L6_n_lumbarization_cases"] = float(len(l6_vals))
            except (ValueError, IndexError):
                pass

        # Overall foreground dice on LSTV (lumb + sacr) vs normal cases —
        # useful for spotting train/val gap on rare anatomy.
        lstv_fg_means: List[float] = []
        for sub in (_SUBTYPE_LUMB, _SUBTYPE_SACR):
            cases = self._val_subgroup_dice.get(sub, [])
            for case_dice in cases:
                vals = [v for v in case_dice if v is not None]
                if vals:
                    lstv_fg_means.append(float(np.mean(vals)))
        if lstv_fg_means:
            out["val/lstv_subgroup/lstv_combined/mean_fg_dice"] = float(np.mean(lstv_fg_means))
            out["val/lstv_subgroup/lstv_combined/n_cases"] = float(len(lstv_fg_means))

        return out

    def _print_per_subgroup_summary(self,
                                      epoch: int,
                                      subgroup_payload: Dict[str, float]) -> None:
        """
        Print a compact per-epoch per-subgroup dice summary to the SLURM log.

        Format (one block per epoch):

            ─── LSTV per-subgroup val dice (epoch 12) ───
            normal         (n=94)  mean_fg=0.612  lumb=0.531  pelvis=0.778
            sacralization  (n=5)   mean_fg=0.541  lumb=0.448  pelvis=0.728
            lumbarization  (n=4)   mean_fg=0.498  lumb=0.402  pelvis=0.694
              L6 dice on lumbarization cases: 0.184  (n=4)

        Failure-tolerant: any exception bails silently. Per-subgroup
        logging should never break training.
        """
        try:
            self.print_to_log_file(
                f"--- LSTV per-subgroup val dice (epoch {epoch}) ---")
            for sub in (_SUBTYPE_NORMAL, _SUBTYPE_SACR, _SUBTYPE_LUMB):
                n_key = f"val/lstv_subgroup/{sub}/n_cases"
                fg_key = f"val/lstv_subgroup/{sub}/mean_fg_dice"
                lb_key = f"val/lstv_subgroup/{sub}/mean_lumbar_dice"
                pv_key = f"val/lstv_subgroup/{sub}/mean_pelvis_dice"
                n = subgroup_payload.get(n_key)
                if n is None or n == 0:
                    self.print_to_log_file(f"  {sub:<14s} (n=0)  no val cases this epoch")
                    continue

                def _fmt(key):
                    v = subgroup_payload.get(key)
                    return f"{v:.3f}" if v is not None else "  n/a"

                self.print_to_log_file(
                    f"  {sub:<14s} (n={int(n)})  "
                    f"mean_fg={_fmt(fg_key)}  "
                    f"lumb={_fmt(lb_key)}  pelvis={_fmt(pv_key)}")

            # Per-class dice for lumbarization (the rare class — most useful
            # to watch). Only print if we have at least one lumb case.
            lumb_n = subgroup_payload.get(f"val/lstv_subgroup/{_SUBTYPE_LUMB}/n_cases")
            if lumb_n and int(lumb_n) > 0:
                cls_parts = []
                for cls_name in _ANATOMY_NAMES:
                    k = f"val/lstv_subgroup/{_SUBTYPE_LUMB}/dice/{cls_name}"
                    v = subgroup_payload.get(k)
                    if v is not None:
                        cls_parts.append(f"{cls_name}={v:.2f}")
                    else:
                        cls_parts.append(f"{cls_name}=---")
                self.print_to_log_file(
                    f"  lumbarization per-class: [{', '.join(cls_parts)}]")

            # Per-class dice for sacralization (the other rare class).
            sacr_n = subgroup_payload.get(f"val/lstv_subgroup/{_SUBTYPE_SACR}/n_cases")
            if sacr_n and int(sacr_n) > 0:
                cls_parts = []
                for cls_name in _ANATOMY_NAMES:
                    k = f"val/lstv_subgroup/{_SUBTYPE_SACR}/dice/{cls_name}"
                    v = subgroup_payload.get(k)
                    if v is not None:
                        cls_parts.append(f"{cls_name}={v:.2f}")
                    else:
                        cls_parts.append(f"{cls_name}=---")
                self.print_to_log_file(
                    f"  sacralization per-class: [{', '.join(cls_parts)}]")

            # Headline metric: L6 dice on lumbarization cases — paper's key claim
            l6_dice = subgroup_payload.get("val/headline/L6_dice_on_lumbarization")
            l6_n    = subgroup_payload.get("val/headline/L6_n_lumbarization_cases")
            if l6_dice is not None and l6_n is not None:
                self.print_to_log_file(
                    f"  HEADLINE: L6 dice on lumbarization cases = "
                    f"{l6_dice:.3f}  (n={int(l6_n)})")
        except Exception as exc:
            try:
                self.print_to_log_file(
                    f"per-subgroup summary print failed: {exc}")
            except Exception:
                pass

    # ---- train_step / validation_step wrappers --------------------------

    def train_step(self, batch):  # type: ignore[override]
        if self._channels_last_active and isinstance(batch, dict) and 'data' in batch:
            try:
                d = batch['data']
                if isinstance(d, torch.Tensor) and d.dim() == 5:
                    batch['data'] = d.contiguous(memory_format=torch.channels_last_3d)
            except Exception:
                pass

        result = super().train_step(batch)

        prof = getattr(self, "_profiler", None)
        if prof is not None:
            try:
                prof.step()
                self._profiler_step_count += 1
                if self._profiler_step_count >= self._profiler_total_steps:
                    self._stop_profiler()
            except Exception as exc:
                try:
                    self.print_to_log_file(f"Profile: step failed: {exc}")
                except Exception:
                    pass
        return result

    def validation_step(self, batch):  # type: ignore[override]
        if self._channels_last_active and isinstance(batch, dict) and 'data' in batch:
            try:
                d = batch['data']
                if isinstance(d, torch.Tensor) and d.dim() == 5:
                    batch['data'] = d.contiguous(memory_format=torch.channels_last_3d)
            except Exception:
                pass

        # Per-subgroup dice path: we need the network output, which the
        # parent validation_step doesn't return. So we run our own
        # forward in inference mode to compute per-case dice, then call
        # the parent for its global aggregation. The double-forward
        # cost is small relative to validation IO, and only happens
        # during val (50 iter/epoch by default), not training.
        if self._per_subgroup_logging_enabled:
            try:
                net = getattr(self, "network", None)
                if net is not None and isinstance(batch, dict) and 'data' in batch:
                    data = batch['data']
                    if isinstance(data, torch.Tensor):
                        # Move to device if needed, matching what nnU-Net
                        # does internally
                        device = getattr(self, "device", None)
                        if device is not None and data.device != device:
                            data = data.to(device, non_blocking=True)
                        net.eval()
                        with torch.no_grad():
                            with torch.amp.autocast(
                                device_type=device.type if device is not None else 'cuda',
                                enabled=True,
                            ):
                                output = net(data)
                        self._record_per_case_dice(batch, output)
            except Exception as exc:
                try:
                    self.print_to_log_file(
                        f"per-subgroup dice: validation_step pre-forward "
                        f"failed: {exc}")
                except Exception:
                    pass

        return super().validation_step(batch)

    # ---- W&B init / log / finish ----------------------------------------

    def _init_wandb(self) -> None:
        if self._wandb_run is not None:
            return
        if not os.environ.get("WANDB_API_KEY") and not _env_truthy("WANDB_ALLOW_OFFLINE", True):
            self.print_to_log_file("WandB: WANDB_API_KEY not set and offline disallowed; skipping.")
            return
        try:
            import wandb
        except ImportError:
            self.print_to_log_file("WandB: wandb not installed; skipping.")
            return

        try:
            run_name = os.environ.get("WANDB_RUN_NAME") or (
                f"{self.plans_manager.dataset_name}__"
                f"{self.__class__.__name__}__"
                f"{self.plans_manager.plans_name}__f{self.fold}"
            )
        except Exception:
            run_name = os.environ.get("WANDB_RUN_NAME") or self.__class__.__name__

        config = {"trainer": self.__class__.__name__,
                  "anatomy_mapping": dict(enumerate(_ANATOMY_NAMES)),
                  "run_val_export": _env_truthy("SPINESURG_RUN_VAL_EXPORT"),
                  "profile_enabled": _env_truthy("SPINESURG_PROFILE"),
                  "channels_last_3d": _env_truthy("SPINESURG_CHANNELS_LAST"),
                  "compile_enabled": os.environ.get("nnUNet_compile", "<default>"),
                  "perf_tuning":
                      os.environ.get("SPINESURG_PERF", "1") != "0",
                  "lstv_per_subgroup_dice":
                      self._per_subgroup_logging_enabled}
        for attr, getter in [
            ("dataset",      lambda: self.plans_manager.dataset_name),
            ("plans",        lambda: self.plans_manager.plans_name),
            ("fold",         lambda: self.fold),
            ("num_epochs",   lambda: self.num_epochs),
            ("batch_size",   lambda: self.batch_size),
            ("patch_size",   lambda: list(self.configuration_manager.patch_size)),
            ("iterations_per_epoch",     lambda: self.num_iterations_per_epoch),
            ("val_iterations_per_epoch", lambda: self.num_val_iterations_per_epoch),
            ("initial_lr",   lambda: self.initial_lr),
            ("weight_decay", lambda: self.weight_decay),
            ("oversample_foreground_percent",
             lambda: self.oversample_foreground_percent),
        ]:
            try:
                config[attr] = getter()
            except Exception:
                pass

        # Pass through LSTV oversampling config so it shows up in W&B
        try:
            config["lstv_oversample_frac"] = float(
                os.environ.get("LSTV_OVERSAMPLE_FRAC", "0.25"))
        except Exception:
            pass

        max_retries = _env_int("WANDB_INIT_MAX_RETRIES", 3)
        init_timeout = _env_int("WANDB_INIT_TIMEOUT_SEC", 120)
        allow_offline = _env_truthy("WANDB_ALLOW_OFFLINE", True)

        for attempt in range(1, max_retries + 1):
            try:
                settings = wandb.Settings(init_timeout=init_timeout) if hasattr(wandb, "Settings") else None
                init_kwargs = dict(
                    project=os.environ.get("WANDB_PROJECT", "SpineSurg-CT"),
                    entity=os.environ.get("WANDB_ENTITY") or None,
                    name=run_name,
                    id=os.environ.get("WANDB_RUN_ID") or None,
                    resume="allow",
                    config=config,
                )
                if settings is not None:
                    init_kwargs["settings"] = settings
                self._wandb_run = wandb.init(**init_kwargs)
                self.print_to_log_file(
                    f"WandB: initialized (attempt {attempt}) run "
                    f"{self._wandb_run.name} ({self._wandb_run.url})")
                return
            except Exception as exc:
                self.print_to_log_file(
                    f"WandB init attempt {attempt}/{max_retries} failed: {exc}")
                if attempt < max_retries:
                    backoff = min(60, 2 ** attempt + random.uniform(0, 3))
                    self.print_to_log_file(f"  retrying in {backoff:.1f}s...")
                    time.sleep(backoff)

        if allow_offline:
            try:
                self.print_to_log_file("WandB: falling back to OFFLINE mode.")
                os.environ["WANDB_MODE"] = "offline"
                self._wandb_run = wandb.init(
                    project=os.environ.get("WANDB_PROJECT", "SpineSurg-CT"),
                    entity=os.environ.get("WANDB_ENTITY") or None,
                    name=run_name,
                    id=os.environ.get("WANDB_RUN_ID") or None,
                    resume="allow",
                    config=config,
                    mode="offline",
                )
            except Exception as exc:
                self.print_to_log_file(f"WandB offline fallback failed: {exc}")
                self._wandb_run = None
        else:
            self.print_to_log_file("WandB: offline fallback disabled; continuing without logging.")

    def _log_wandb(self, data: dict, step: Optional[int] = None) -> None:
        if self._wandb_run is None:
            return
        clean = {k: v for k, v in data.items() if v is not None}
        if not clean:
            return
        try:
            if step is not None:
                self._wandb_run.log(clean, step=step)
            else:
                self._wandb_run.log(clean)
        except Exception as exc:
            self.print_to_log_file(f"WandB log failed: {exc}")

    def _finish_wandb(self) -> None:
        if self._wandb_run is None:
            return
        try:
            self._wandb_run.finish()
        except Exception:
            pass
        self._wandb_run = None

    # ---- Lifecycle hooks -------------------------------------------------

    def on_train_start(self):  # type: ignore[override]
        try:
            self._maybe_install_perf_tuning()
        except Exception as exc:
            print(f"[perf] perf-tuning failed: {exc}", flush=True)

        super().on_train_start()

        try:
            _install_nnunet_warning_filter()
        except Exception:
            pass
        try:
            self._maybe_load_subtype_map()
        except Exception as exc:
            try:
                self.print_to_log_file(
                    f"LSTV per-subgroup dice: subtype map load failed: {exc}")
            except Exception:
                pass
        try:
            self._dump_environment()
        except Exception:
            pass
        try:
            self._install_log_intercepts()
        except Exception:
            pass
        try:
            self._maybe_apply_channels_last()
        except Exception as exc:
            try:
                self.print_to_log_file(f"channels_last_3d: failed: {exc}")
            except Exception:
                pass
        try:
            self._maybe_install_profiler()
        except Exception as exc:
            try:
                self.print_to_log_file(f"Profile: install failed: {exc}")
            except Exception:
                pass
        try:
            self._init_wandb()
        except Exception as exc:
            try:
                self.print_to_log_file(f"WandB: init failed: {exc}")
            except Exception:
                pass

    def on_train_epoch_start(self):  # type: ignore[override]
        super().on_train_epoch_start()
        # Reset per-subgroup buffer at the start of each epoch so the
        # *validation* phase that comes after training fills a clean
        # buffer.
        self._reset_subgroup_dice_buffer()

    def on_epoch_end(self):  # type: ignore[override]
        super().on_epoch_end()
        if self._wandb_run is None:
            return
        try:
            L = self._current_logs()
            epoch = self.current_epoch

            def last(key):
                seq = L.get(key, [])
                if not seq:
                    return None
                for v in reversed(seq):
                    if v is not None:
                        return v
                return None

            epoch_time = None
            starts = L.get("epoch_start_timestamps", [])
            ends   = L.get("epoch_end_timestamps", [])
            if starts and ends and len(starts) == len(ends):
                try:
                    epoch_time = float(ends[-1] - starts[-1])
                except Exception:
                    epoch_time = None

            per_class_log: Dict[str, float] = {}
            dice_per_class = last("dice_per_class_or_region")
            lumbar_vals: List[float] = []
            pelvis_vals: List[float] = []
            fg_vals: List[float] = []
            if dice_per_class is not None:
                try:
                    for i, v in enumerate(dice_per_class):
                        if i >= len(_ANATOMY_NAMES):
                            continue
                        name = _ANATOMY_NAMES[i]
                        try:
                            vf = float(v)
                        except Exception:
                            continue
                        if np.isnan(vf):
                            continue
                        per_class_log[f"val/dice/{name}"] = vf
                        fg_vals.append(vf)
                        if i < 6:
                            lumbar_vals.append(vf)
                        else:
                            pelvis_vals.append(vf)
                    if lumbar_vals:
                        per_class_log["val/dice/mean_lumbar"] = float(np.mean(lumbar_vals))
                    if pelvis_vals:
                        per_class_log["val/dice/mean_pelvis"] = float(np.mean(pelvis_vals))
                    if fg_vals:
                        per_class_log["val/dice/mean_foreground_nanmasked"] = float(np.mean(fg_vals))
                except Exception:
                    pass

            payload = {
                "epoch":             epoch,
                "train/loss":        _as_float(last("train_losses")),
                "val/loss":          _as_float(last("val_losses")),
                "val/mean_fg_dice":  _as_float(last("mean_fg_dice")),
                "val/ema_fg_dice":   _as_float(last("ema_fg_dice")),
                "train/lr":          _as_float(last("lrs")),
                "timing/epoch_sec":  epoch_time,
                **per_class_log,
            }

            # Per-subgroup dice
            try:
                subgroup_payload = self._aggregate_subgroup_dice()
                payload.update(subgroup_payload)
                # Per-epoch human-readable summary (always; this is the
                # number you actually want to watch scroll by).
                self._print_per_subgroup_summary(epoch, subgroup_payload)
                # First-epoch verbose diagnostic (keys list) for debugging.
                if epoch <= 1:
                    sg_keys = sorted(k for k in subgroup_payload if k is not None)
                    self.print_to_log_file(
                        f"WandB: epoch {epoch} per-subgroup keys ({len(sg_keys)}): "
                        f"{sg_keys[:8]}{'...' if len(sg_keys) > 8 else ''}")
            except Exception as exc:
                try:
                    self.print_to_log_file(
                        f"WandB: per-subgroup dice aggregation failed: {exc}")
                except Exception:
                    pass

            if epoch <= 1:
                self.print_to_log_file(
                    f"WandB: epoch {epoch} keys_seen = "
                    f"{sorted(L.keys()) if L else []}")
                non_none = sorted(k for k, v in payload.items() if v is not None)
                self.print_to_log_file(
                    f"WandB: epoch {epoch} payload (non-None) = {non_none}")

            self._log_wandb(payload, step=epoch)
        except Exception as exc:
            try:
                self.print_to_log_file(f"WandB: on_epoch_end failed: {exc}")
            except Exception:
                pass

    def on_train_end(self):  # type: ignore[override]
        try:
            L = self._current_logs()
            ema  = [v for v in L.get("ema_fg_dice",  []) if v is not None]
            mean = [v for v in L.get("mean_fg_dice", []) if v is not None]
            summary = {}
            if ema:
                summary["summary/best_ema_fg_dice"] = max(ema)
            if mean:
                summary["summary/best_mean_fg_dice"] = max(mean)

            dpc_history = L.get("dice_per_class_or_region", [])
            if dpc_history:
                try:
                    n_classes = max((len(row) for row in dpc_history if row), default=0)
                    for ci in range(n_classes):
                        seq = []
                        for row in dpc_history:
                            if not row or ci >= len(row) or row[ci] is None:
                                continue
                            try:
                                vf = float(row[ci])
                            except Exception:
                                continue
                            if np.isnan(vf):
                                continue
                            seq.append(vf)
                        if not seq:
                            continue
                        if ci < len(_ANATOMY_NAMES):
                            summary[f"summary/best_dice/{_ANATOMY_NAMES[ci]}"] = max(seq)
                except Exception:
                    pass

            if summary:
                self._log_wandb(summary)
        except Exception:
            pass
        super().on_train_end()
        try:
            self._finish_wandb()
        except Exception:
            pass
        try:
            self._stop_profiler()
        except Exception:
            pass
        try:
            self._restore_perf_tuning_safe()
        except Exception:
            pass

    def perform_actual_validation(self, save_probabilities: bool = False):  # type: ignore[override]
        if _env_truthy("SPINESURG_RUN_VAL_EXPORT"):
            self.print_to_log_file(
                "perform_actual_validation: ENABLED via SPINESURG_RUN_VAL_EXPORT")
            return super().perform_actual_validation(save_probabilities)
        self.print_to_log_file(
            "perform_actual_validation: SKIPPED "
            "(set SPINESURG_RUN_VAL_EXPORT=1 to enable)")
        return


def _as_float(x):
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    if np.isnan(xf):
        return None
    return xf


# =============================================================================
# LSTV oversampling mixin
# =============================================================================

class _LSTVOversampleMixin:
    """
    Oversample lstv_label != "normal" cases at the dataloader level.
    Covers BOTH lumbarization and sacralization.

    Detection precedence (first hit wins):
      a) <nnUNet_preprocessed>/<dataset>/lstv_cases.json   (preferred)
      b) <nnUNet_raw>/<dataset>/lstv_cases.json
      c) Live HF manifest scan (HF_EXPORT_DIR or auto-detect)
      d) Legacy L6-voxel scan over labelsTr (lumbarization only)

    Env vars:
        LSTV_OVERSAMPLE_FRAC  target fraction (default 0.25, capped 0.95)
                              Lowered from 0.50 in v8 -- avg_dup_factor at
                              0.50 was ~20x per unique LSTV case, very
                              high overfitting risk on rare classes.
                              0.25 gives ~9x duplication, more moderate.
        HF_EXPORT_DIR         override HF export path
        LSTV_LABELS_DIR       override labelsTr path

    MRO: _LSTVOversampleMixin -> _WandBMixin -> nnUNetTrainer
    """

    _lstv_case_ids: Optional[Set[str]] = None
    _lstv_subtype_counts: Optional[Counter] = None
    _lstv_detection_source: Optional[str] = None

    def _identify_lstv_cases(self) -> Set[str]:
        if self._lstv_case_ids is not None:
            return self._lstv_case_ids
        try:
            ds_name = self.plans_manager.dataset_name
        except Exception:
            ds_name = None
        if ds_name:
            for json_path in _candidate_lstv_json_paths(ds_name):
                self.print_to_log_file(
                    f"LSTV oversample: checking {json_path}")
                result = _read_lstv_cases_json(json_path)
                if result is not None:
                    ids, subtypes, _ = result
                    self._lstv_case_ids = ids
                    self._lstv_subtype_counts = subtypes
                    self._lstv_detection_source = f"json:{json_path}"
                    self.print_to_log_file(
                        f"LSTV oversample: loaded {len(ids)} case_ids from "
                        f"{json_path.name}. Subtypes: {dict(subtypes)}")
                    return ids
        for hf_dir in _candidate_hf_export_dirs():
            self.print_to_log_file(
                f"LSTV oversample: checking manifest path {hf_dir}")
            result = _read_lstv_records_from_manifest(hf_dir)
            if result is not None:
                ids, subtypes, _ = result
                self._lstv_case_ids = ids
                self._lstv_subtype_counts = subtypes
                self._lstv_detection_source = f"manifest:{hf_dir}"
                self.print_to_log_file(
                    f"LSTV oversample: scanned manifests at {hf_dir}, "
                    f"found {len(ids)} LSTV case_ids. "
                    f"Subtypes: {dict(subtypes)}")
                return ids
        self.print_to_log_file(
            "LSTV oversample: no lstv_cases.json or manifest found; "
            "falling back to L6-voxel scan (CAUTION: misses sacralization).")
        labels_dir = self._lstv_labels_dir_for_legacy_scan()
        if labels_dir is None:
            self.print_to_log_file(
                "LSTV oversample: cannot locate labelsTr either; disabling.")
            self._lstv_case_ids = set()
            return self._lstv_case_ids
        t0 = time.time()
        ids = _scan_lstv_case_ids_by_l6_voxels(labels_dir)
        self._lstv_case_ids = ids
        self._lstv_subtype_counts = Counter({"lumbarization_l6scan": len(ids)})
        self._lstv_detection_source = f"l6scan:{labels_dir}"
        self.print_to_log_file(
            f"LSTV oversample: legacy L6 scan found {len(ids)} cases "
            f"({time.time() - t0:.1f}s). NOTE: sacralization NOT included.")
        return ids

    def _lstv_labels_dir_for_legacy_scan(self) -> Optional[str]:
        override = os.environ.get("LSTV_LABELS_DIR", "").strip()
        if override:
            return override
        raw_root = os.environ.get("nnUNet_raw") or os.environ.get("nnUNet_raw_data_base")
        if raw_root:
            try:
                ds_name = self.plans_manager.dataset_name
                cand = join(raw_root, ds_name, "labelsTr")
                if isdir(cand):
                    return cand
            except Exception:
                pass
        return None

    def _apply_lstv_sampler(self) -> None:
        # v8: default lowered from 0.5 to 0.25.
        try:
            frac = float(os.environ.get("LSTV_OVERSAMPLE_FRAC", "0.25"))
        except ValueError:
            self.print_to_log_file(
                "LSTV oversample: invalid LSTV_OVERSAMPLE_FRAC; using 0.25")
            frac = 0.25
        frac = max(0.0, min(0.95, frac))
        if frac <= 0:
            self.print_to_log_file("LSTV oversample: frac=0, no-op.")
            return
        lstv_ids = self._identify_lstv_cases()
        if not lstv_ids:
            self.print_to_log_file("LSTV oversample: no LSTV cases identified, no-op.")
            return
        tr_dataset = getattr(self, "dataloader_train", None)
        if tr_dataset is None:
            self.print_to_log_file(
                "LSTV oversample: self.dataloader_train is None at "
                "on_train_start; patching impossible.")
            return
        keys_attr_paths = [
            ("generator", "_data", "identifiers"),
            ("generator", "_data", "keys"),
            ("_data", "identifiers"),
            ("_data", "keys"),
            ("data_loader", "_data", "identifiers"),
            ("data_loader", "_data", "keys"),
        ]
        keys_list = None
        keys_owner = None
        keys_attr = None
        traversal_log: List[str] = []
        for path in keys_attr_paths:
            obj = tr_dataset
            ok = True
            for p in path[:-1]:
                obj = getattr(obj, p, None)
                if obj is None:
                    ok = False
                    break
            if not ok:
                continue
            final_attr = path[-1]
            if hasattr(obj, final_attr):
                candidate = getattr(obj, final_attr)
                if callable(candidate):
                    try:
                        candidate = list(candidate())
                    except Exception:
                        continue
                else:
                    try:
                        candidate = list(candidate)
                    except Exception:
                        continue
                if candidate and all(isinstance(k, str) for k in candidate):
                    keys_list = candidate
                    keys_owner = obj
                    keys_attr = final_attr
                    traversal_log.append(
                        f"FOUND at {'.'.join(path)} ({len(candidate)} entries)")
                    break
                else:
                    traversal_log.append(
                        f"  {'.'.join(path)} present but empty/non-string")
            else:
                traversal_log.append(
                    f"  {'.'.join(path)} attribute missing")
        if keys_list is None:
            self.print_to_log_file(
                f"LSTV oversample: could not locate dataloader key list. "
                f"Traversal: {traversal_log}. Framework internals may have "
                f"changed; oversampling DISABLED for this run.")
            return
        original_n = len(keys_list)
        lstv_in_keys = [k for k in keys_list if k in lstv_ids]
        if not lstv_in_keys:
            self.print_to_log_file(
                "LSTV oversample: no LSTV cases present in this fold's "
                "training split; no-op.")
            return
        p = len(lstv_in_keys) / original_n
        if p >= frac:
            self.print_to_log_file(
                f"LSTV oversample: natural LSTV fraction {p:.3f} >= target "
                f"{frac:.3f}; no duplication needed.")
            return
        needed_total = int(round(
            (frac * original_n - len(lstv_in_keys)) / (1.0 - frac)))
        needed_total = max(1, needed_total)
        duplicates = [
            lstv_in_keys[i % len(lstv_in_keys)] for i in range(needed_total)
        ]
        new_keys = keys_list + duplicates
        try:
            setattr(keys_owner, keys_attr, new_keys)
            verify = list(getattr(keys_owner, keys_attr))
        except Exception as exc:
            self.print_to_log_file(
                f"LSTV oversample: setattr({keys_attr}) failed: {exc}. "
                f"Cannot patch dataloader; oversampling DISABLED.")
            return
        if len(verify) != len(new_keys):
            self.print_to_log_file(
                f"LSTV oversample: WARNING — set {len(new_keys)} keys but "
                f"dataloader reports {len(verify)}.")
        achieved = (len(lstv_in_keys) + len(duplicates)) / max(1, len(verify))
        unique_lstv = len(set(lstv_in_keys))
        avg_dup_factor = (
            (len(lstv_in_keys) + len(duplicates)) / max(1, unique_lstv)
        )
        self.print_to_log_file(
            f"LSTV oversample: source={self._lstv_detection_source}")
        self.print_to_log_file(
            f"LSTV oversample: fold {self.fold} train pool: "
            f"original_N={original_n}  LSTV={len(lstv_in_keys)} "
            f"({p*100:.1f}%);  unique_LSTV={unique_lstv};  "
            f"added {len(duplicates)} duplicates -> "
            f"new_N={len(verify)}  achieved_LSTV_frac={achieved*100:.1f}% "
            f"(target {frac*100:.0f}%);  avg_dup_factor={avg_dup_factor:.1f}x")
        if avg_dup_factor > 25:
            self.print_to_log_file(
                f"LSTV oversample: WARNING — each unique LSTV case is "
                f"duplicated {avg_dup_factor:.0f}x. High overfitting risk.")

    def on_train_start(self):  # type: ignore[override]
        super().on_train_start()
        try:
            self._apply_lstv_sampler()
        except Exception as exc:
            try:
                self.print_to_log_file(f"LSTV oversample: fatal: {exc}")
            except Exception:
                pass


# =============================================================================
# Concrete trainer classes
# =============================================================================
#
# IMPORTANT: any trainer subclass that overrides __init__ must declare
# the FULL nnUNetTrainer signature explicitly. Using *args/**kwargs
# breaks nnUNetTrainer's inspect-based init kwargs capture.
# =============================================================================


class nnUNetTrainerWandB(_WandBMixin, nnUNetTrainer):
    """Default-length nnU-Net training + W&B."""
    pass


class nnUNetTrainerWandB_5epochs(_WandBMixin, nnUNetTrainer_5epochs):
    pass


class nnUNetTrainerWandB_100epochs(_WandBMixin, nnUNetTrainer_100epochs):
    pass


class nnUNetTrainerWandB_500epochs(_WandBMixin, nnUNetTrainer):
    """500 epochs (default 250 iter/epoch)."""
    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset, device)
        self.num_epochs = 500


class nnUNetTrainerWandB_1000epochs(_WandBMixin, nnUNetTrainer):
    """1000 epochs (default 250 iter/epoch)."""
    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset, device)
        self.num_epochs = 1000


class nnUNetTrainerWandB_1000ep_500iter(_WandBMixin, nnUNetTrainer):
    """1000 epochs, 500 iter/epoch (2x default coverage per epoch)."""
    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset, device)
        self.num_epochs = 1000
        self.num_iterations_per_epoch = 500
        self.num_val_iterations_per_epoch = 50


class nnUNetTrainerWandB_500ep_LSTVOversample(
    _LSTVOversampleMixin, _WandBMixin, nnUNetTrainer
):
    """500 epochs + LSTV oversampling. DEFAULT TRAINER for the SpineSurg-CT
    paper run. 500 ep × 250 iter = 125k training steps.

    v8: LSTV_OVERSAMPLE_FRAC default lowered to 0.25 (was 0.50)."""
    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset, device)
        self.num_epochs = 500


class nnUNetTrainerWandB_1000ep_LSTVOversample(
    _LSTVOversampleMixin, _WandBMixin, nnUNetTrainer
):
    """1000 epochs + LSTV oversampling (default 250 iter/epoch).

    MRO: _LSTVOversampleMixin -> _WandBMixin -> nnUNetTrainer
    """
    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset, device)
        self.num_epochs = 1000
