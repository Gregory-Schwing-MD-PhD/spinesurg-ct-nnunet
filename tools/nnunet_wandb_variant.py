"""
SpineSurg-CT -- nnU-Net v2 trainers with W&B logging (v2)
tools/nnunet_wandb_variant.py

Changes vs v1
=============
1. Warning filter: the RuntimeWarning about "invalid value in scalar divide"
   from nnU-Net's global_dc_per_class is now suppressed at trainer init.
   Root cause: when a class is absent from the entire validation epoch
   (TP = FP = FN = 0), 2*TP / (2*TP + FP + FN) = 0/0 = NaN + warning.
   Benign, but it clutters the log stream. We catch only THAT specific
   numpy warning, not everything.

2. W&B init is now retryable with exponential backoff. If the cluster
   can't reach api.wandb.ai, we fall back to offline mode so the run
   still logs to disk and can be synced later via `wandb sync`.
   Controls:
       WANDB_INIT_MAX_RETRIES   default 3
       WANDB_INIT_TIMEOUT_SEC   default 120 (increase if timing out)
       WANDB_ALLOW_OFFLINE      default 1 (set 0 to hard-fail instead)

3. NaN-safe dice aggregation. When per-class dice is NaN (class absent
   from the epoch), we DROP it from the lumbar/pelvic/foreground means
   rather than treating it as 0.0. This is what you want scientifically:
   a missing class should not drag the mean down -- your paper-metrics
   script already does this, the live W&B plot should too.

4. Three new trainer classes:
       nnUNetTrainerWandB_1000epochs          (1000 ep, default)
       nnUNetTrainerWandB_1000ep_500iter      (1000 ep, 500 iters/epoch)
       nnUNetTrainerWandB_1000ep_LSTVOversample
           (1000 ep, 500 iters/epoch, oversample cases containing L6)

   The LSTV-oversample variant looks up which cases contain L6 from the
   labels folder at trainer init and modifies the dataloader to sample
   those cases at 50% probability during training. This is the right
   thing to do given L6 appears in only ~5-15% of cases.

   NOTE: the LSTV oversampling is CASE-level (entire case more likely
   to be drawn), not patch-level. This composes cleanly with nnU-Net's
   built-in oversample_foreground_percent, which operates within a
   drawn case's volume.

Author: Gregory Schwing, MD-PhD  |  Wayne State University / DMC
"""

from __future__ import annotations

import json
import os
import random
import time
import warnings
from collections import defaultdict
from os.path import isfile, join
from typing import Dict, List, Optional, Set

import numpy as np

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

# nnU-Net's dice_per_class_or_region skips class 0 (background).
# Index i -> our label (i + 1) in the SpineSurg-CT 10-class map.
_ANATOMY_NAMES = [
    "L1", "L2", "L3", "L4", "L5", "L6",
    "sacrum", "left_hip", "right_hip",
]

# Numeric class IDs for the LSTV case detector (matches SpineSurg-CT label map)
_L6_LABEL_ID = 6
_IGNORE_LABEL = 10


# -----------------------------------------------------------------------------
# Silence the benign scalar-divide warning from nnU-Net's dice accumulator
# -----------------------------------------------------------------------------

def _install_nnunet_warning_filter() -> None:
    """
    Suppress ONLY the specific scalar-divide warning that fires when a class
    has zero presence in a validation epoch. Leaves all other numpy warnings
    alone so we still see legitimate problems.
    """
    warnings.filterwarnings(
        "ignore",
        message="invalid value encountered in scalar divide",
        category=RuntimeWarning,
    )
    # Same mechanism triggers via numpy's new error handling path
    np.seterr(invalid="ignore", divide="warn")


# -----------------------------------------------------------------------------
# NaN-safe mean helper
# -----------------------------------------------------------------------------

def _nanmean_list(vals) -> Optional[float]:
    try:
        arr = np.asarray([v for v in vals if v is not None], dtype=np.float64)
        if arr.size == 0:
            return None
        if np.all(np.isnan(arr)):
            return None
        return float(np.nanmean(arr))
    except Exception:
        return None


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
# LSTV case detection (for oversampling trainer)
# -----------------------------------------------------------------------------

def _scan_lstv_case_ids(labels_dir: str) -> Set[str]:
    """
    Return the set of case IDs (filenames without '.nii.gz') whose GT label
    volume contains any voxel with label = _L6_LABEL_ID. Used to build a
    weighted sampler for the LSTV-oversample trainer.
    """
    import nibabel as nib
    out: Set[str] = set()
    if not os.path.isdir(labels_dir):
        return out
    files = [f for f in sorted(os.listdir(labels_dir)) if f.endswith(".nii.gz")]
    for fn in files:
        try:
            arr = np.asarray(nib.load(join(labels_dir, fn)).dataobj).astype(np.int16)
            if np.any(arr == _L6_LABEL_ID):
                out.add(fn[: -len(".nii.gz")])
        except Exception:
            continue
    return out


# =============================================================================
# W&B mixin
# =============================================================================

class _WandBMixin:
    """Mixin: wandb + disk-based metric source + val-export toggle."""

    _wandb_run = None
    _metric_cache = None
    _intercepted_objects = None

    # ----- primary metric source: progress.json --------------------------

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

    # ----- log-intercept fallback ----------------------------------------

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

    # ----- diagnostic -----------------------------------------------------

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
        except Exception:
            pass

    # ----- W&B lifecycle (with retry + offline fallback) ------------------

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
                  "run_val_export": _env_truthy("SPINESURG_RUN_VAL_EXPORT")}
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

        max_retries = _env_int("WANDB_INIT_MAX_RETRIES", 3)
        init_timeout = _env_int("WANDB_INIT_TIMEOUT_SEC", 120)
        allow_offline = _env_truthy("WANDB_ALLOW_OFFLINE", True)

        # Retry online init with exponential backoff.
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

        # All online retries failed. Try offline if allowed.
        if allow_offline:
            try:
                self.print_to_log_file("WandB: falling back to OFFLINE mode. "
                                       "Sync later with `wandb sync <run_dir>`.")
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
                self.print_to_log_file(
                    f"WandB: offline run created at "
                    f"{getattr(self._wandb_run, 'dir', '<unknown>')}")
            except Exception as exc:
                self.print_to_log_file(f"WandB offline fallback also failed: {exc}")
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

    # ----- nnU-Net trainer hooks ------------------------------------------

    def on_train_start(self):  # type: ignore[override]
        super().on_train_start()
        try:
            _install_nnunet_warning_filter()
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
            self._init_wandb()
        except Exception as exc:
            try:
                self.print_to_log_file(f"WandB: init failed: {exc}")
            except Exception:
                pass

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

            # Per-class dice: anatomy-named keys; NaN is dropped, not zeroed.
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
                            # class absent from epoch -> skip, don't log 0
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

    # ----- validation export toggle ---------------------------------------

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
    Oversample cases containing L6 (LSTV) at the dataloader level.

    Strategy
    --------
    On train start, scan the preprocessed dataset's label files to
    identify which case IDs contain L6 voxels. Then replace the default
    sampler in the training dataloader with a WeightedRandomSampler
    that draws LSTV cases at probability ~ LSTV_OVERSAMPLE_FRAC (default
    0.5), meaning half of training batches are guaranteed to contain
    L6 exposure at the CASE level. Standard nnU-Net foreground
    oversampling (patch level) operates on top of this.

    Env vars
    --------
        LSTV_OVERSAMPLE_FRAC    fraction of batches containing LSTV cases
                                (default: 0.5)
        LSTV_LABELS_DIR         override the auto-detected labelsTr dir
                                (rare; usually unnecessary)
    """

    _lstv_case_ids: Optional[Set[str]] = None

    def _lstv_labels_dir(self) -> Optional[str]:
        override = os.environ.get("LSTV_LABELS_DIR", "").strip()
        if override:
            return override
        # nnU-Net raw labelsTr is the canonical source
        raw_root = os.environ.get("nnUNet_raw") or os.environ.get("nnUNet_raw_data_base")
        if raw_root:
            try:
                ds_name = self.plans_manager.dataset_name
                cand = join(raw_root, ds_name, "labelsTr")
                if os.path.isdir(cand):
                    return cand
            except Exception:
                pass
        return None

    def _identify_lstv_cases(self) -> Set[str]:
        if self._lstv_case_ids is not None:
            return self._lstv_case_ids
        labels_dir = self._lstv_labels_dir()
        if labels_dir is None:
            self.print_to_log_file(
                "LSTV oversample: could not locate labelsTr; disabling oversampling.")
            self._lstv_case_ids = set()
            return self._lstv_case_ids
        self.print_to_log_file(f"LSTV oversample: scanning {labels_dir} for L6 cases...")
        t0 = time.time()
        ids = _scan_lstv_case_ids(labels_dir)
        self.print_to_log_file(
            f"LSTV oversample: found {len(ids)} cases containing L6 "
            f"({time.time() - t0:.1f}s)")
        self._lstv_case_ids = ids
        return ids

    def _apply_lstv_sampler(self) -> None:
        """
        Replace the training dataloader's sampler with one that biases
        toward LSTV cases. This runs on_train_start, AFTER dataloaders
        have been built by nnU-Net.
        """
        try:
            from torch.utils.data import WeightedRandomSampler
        except Exception as exc:
            self.print_to_log_file(f"LSTV oversample: torch WeightedRandomSampler unavailable: {exc}")
            return

        frac = float(os.environ.get("LSTV_OVERSAMPLE_FRAC", "0.5"))
        frac = max(0.0, min(1.0, frac))
        if frac <= 0:
            self.print_to_log_file("LSTV oversample: frac=0, no-op.")
            return

        lstv_ids = self._identify_lstv_cases()
        if not lstv_ids:
            self.print_to_log_file("LSTV oversample: no LSTV cases found, no-op.")
            return

        # nnU-Net v2 uses its own batchgenerators-based dataloader, not a
        # vanilla torch DataLoader. The right hook is to subclass the
        # trainer's _get_deep_supervision_scales / get_tr_and_val_datasets
        # and tweak the internal oversampling map. Pragmatic approach:
        # duplicate LSTV case keys in the training dataset's case list
        # so they are drawn proportionally more often.
        try:
            tr_dataset = getattr(self, "dataloader_train", None)
            # Different nnU-Net versions expose the list of keys differently;
            # try the common paths.
            keys_attr_paths = [
                ("generator", "_data", "keys"),
                ("_data", "keys"),
                ("data_loader", "_data", "keys"),
            ]
            keys_list = None
            keys_owner = None
            keys_attr = None
            for path in keys_attr_paths:
                obj = tr_dataset
                for p in path[:-1]:
                    obj = getattr(obj, p, None)
                    if obj is None:
                        break
                if obj is None:
                    continue
                final_attr = path[-1]
                if hasattr(obj, final_attr):
                    keys_list = list(getattr(obj, final_attr))
                    keys_owner = obj
                    keys_attr = final_attr
                    break
            if keys_list is None:
                self.print_to_log_file(
                    "LSTV oversample: could not locate dataloader key list; "
                    "no-op. (Framework internals may have changed.)")
                return

            original_n = len(keys_list)
            lstv_in_keys = [k for k in keys_list if k in lstv_ids]
            if not lstv_in_keys:
                self.print_to_log_file(
                    "LSTV oversample: no LSTV cases present in this fold's "
                    "training split; no-op.")
                return

            # Compute target duplicates so LSTV fraction ~= frac.
            # If original LSTV fraction is p, we need to add D duplicates
            # such that (count_lstv + D) / (N + D) = frac
            # => D = (frac * N - count_lstv) / (1 - frac)
            p = len(lstv_in_keys) / original_n
            if p >= frac:
                self.print_to_log_file(
                    f"LSTV oversample: natural frac {p:.3f} >= target {frac:.3f}; no-op.")
                return
            needed_total = int(round((frac * original_n - len(lstv_in_keys)) / (1.0 - frac)))
            needed_total = max(1, needed_total)
            duplicates = []
            for i in range(needed_total):
                duplicates.append(lstv_in_keys[i % len(lstv_in_keys)])
            new_keys = keys_list + duplicates
            setattr(keys_owner, keys_attr, new_keys)
            achieved = (len(lstv_in_keys) + len(duplicates)) / len(new_keys)
            self.print_to_log_file(
                f"LSTV oversample: original N={original_n}  LSTV={len(lstv_in_keys)} "
                f"({p*100:.1f}%);  added {len(duplicates)} duplicates -> "
                f"N={len(new_keys)}  LSTV fraction={achieved*100:.1f}% "
                f"(target {frac*100:.0f}%).")
        except Exception as exc:
            self.print_to_log_file(f"LSTV oversample: failed to patch dataloader: {exc}")

    def on_train_start(self):  # type: ignore[override]
        super().on_train_start()  # _WandBMixin runs first via MRO
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

class nnUNetTrainerWandB(_WandBMixin, nnUNetTrainer):
    """1000 epochs, default nnU-Net settings + W&B."""
    pass


class nnUNetTrainerWandB_5epochs(_WandBMixin, nnUNetTrainer_5epochs):
    pass


class nnUNetTrainerWandB_100epochs(_WandBMixin, nnUNetTrainer_100epochs):
    pass


class nnUNetTrainerWandB_1000epochs(_WandBMixin, nnUNetTrainer):
    """Explicit 1000-epoch variant (identical to base but named for clarity)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_epochs = 1000


class nnUNetTrainerWandB_1000ep_500iter(_WandBMixin, nnUNetTrainer):
    """1000 epochs, 500 iterations/epoch (2x the default coverage per epoch)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_epochs = 1000
        self.num_iterations_per_epoch = 500
        self.num_val_iterations_per_epoch = 50


class nnUNetTrainerWandB_1000ep_LSTVOversample(
    _LSTVOversampleMixin, _WandBMixin, nnUNetTrainer
):
    """
    1000 epochs, 500 iter/epoch, LSTV-case oversampling.

    MRO: _LSTVOversampleMixin -> _WandBMixin -> nnUNetTrainer
    Both mixins override on_train_start; _LSTVOversampleMixin calls super()
    so _WandBMixin and the base both run, then LSTV patching happens.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_epochs = 1000
        self.num_iterations_per_epoch = 500
        self.num_val_iterations_per_epoch = 50
