"""
SpineSurg-CT — LSTV-aware nnU-Net v2 3D DataLoader (May 2026)
tools/lstv_biased_dataloader.py

Clean class-override replacement for the np.random.choice monkey-patching
that v15-v17.x of nnunet_wandb_variant.py attempted (and never quite got
working across multiprocessing forks). This file implements the patch
class biasing as a proper subclass of nnUNetDataLoader3D.

Design
======
nnU-Net v2's `nnUNetDataLoaderBase.get_bbox(data_shape, force_fg,
class_locations, overwrite_class=None)` already supports forcing a
specific class via `overwrite_class`. We override ONLY `get_bbox` to
fill in `overwrite_class` based on subtype detection, then delegate to
the parent. No copying of `generate_train_batch`, no monkey patching,
no fork-time gymnastics.

Subtype detection from class_locations content
==============================================
The model's preprocessed `class_locations` dict tells us which classes
have foreground voxels in each case. We use that signal directly:

  Lumbarization (lumb)
    Signature: class 6 (L6) has voxels.
    Why it works: only lumb patients have an L6 vertebral body. No
    other subtype produces L6 voxels.
    Action: bias toward L6 at probability `bias_l6_prob` (default 0.60).

  Sacralization with count change (sacr_count)
    Signature: L4 has voxels AND L5 does NOT have voxels AND sacrum
    has voxels.
    Why all three conditions: L5 absence alone is not enough — pelvic-
    only views also have no L5 voxels (lumbar region is masked as
    ignore=10). Requiring L4 voxels confirms the lumbar region IS
    annotated, so L5's absence reflects real sacralization-with-count
    rather than annotation masking.
    Action: bias toward sacrum at probability `bias_sacrum_prob`
    (default 0.50).

  All other cases (normal, sacralization, semisacralization, ambiguous,
  partial-annotation views): no bias — fall through to nnU-Net's
  uniform sampling among eligible foreground classes.

Mutual exclusivity: a case cannot satisfy both signatures simultaneously
(lumb requires L6 voxels, sacr_count requires L5 absent — these are
not mutually contradictory, but L6 is rare enough that we prioritize
the lumb signature when both fire).

Configuration
=============
All bias probabilities are configurable via __init__ args (programmatic)
OR environment variables (deployed):

    SPINESURG_LSTV_BIAS_L6_PROB       (default 0.60)
    SPINESURG_LSTV_BIAS_SACRUM_PROB   (default 0.50)
    SPINESURG_LSTV_BIAS_ENABLED       (default 1; "0" disables all bias)

Setting `SPINESURG_LSTV_BIAS_ENABLED=0` is a clean kill switch — it
bypasses the bias logic entirely (no perf overhead) and the loader
behaves as a pure nnUNetDataLoader3D.

Diagnostics
===========
Each loader instance maintains per-process counters:

    self._bias_n_force_fg        : count of force_fg bbox calls handled
    self._bias_n_l6_returned     : count of L6 forces
    self._bias_n_sacrum_returned : count of sacrum forces

NOTE: Under MultiThreadedAugmenter each worker process has its own
loader instance with its own counters; the parent process's counters
never see worker-side increments. To verify bias is firing in
production, the right signal is the model's L6 dice — if it rises off
zero on lumb cases, bias is working. For tighter verification use the
unit tests in tests/test_lstv_biased_dataloader.py, which exercise
the override in-process and assert the rate exactly.

Usage in trainer
================
    from .lstv_biased_dataloader import LSTVBiasedDataLoader3D

    class _LSTVOversampleMixin:
        def get_dataloaders(self):
            import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as trainer_mod
            original = trainer_mod.nnUNetDataLoader3D
            trainer_mod.nnUNetDataLoader3D = LSTVBiasedDataLoader3D
            try:
                return super().get_dataloaders()
            finally:
                trainer_mod.nnUNetDataLoader3D = original

Author: Gregory Schwing, MD-PhD  |  Wayne State University / DMC
"""

from __future__ import annotations

import multiprocessing as _mp
import os
import numpy as np
from typing import Dict, Optional, Sequence, Union

# ── Class IDs in the SpineSurg-CT 10-class scheme ──────────────────────
# These MUST match dataset.json["labels"]. Hardcoded because the dataset
# label scheme is a contract; if it changes, this whole module is wrong.
BACKGROUND = 0
L1_LABEL = 1
L2_LABEL = 2
L3_LABEL = 3
L4_LABEL = 4
L5_LABEL = 5
L6_LABEL = 6
SACRUM_LABEL = 7
LEFT_HIP_LABEL = 8
RIGHT_HIP_LABEL = 9
IGNORE_LABEL = 10  # not a network output; only appears in seg via masking


# class_locations entries can be either a single class id (int) or a
# tuple-of-ids (region-based training). Our detection signatures look
# only at int keys; tuple keys are silently passed through to uniform
# sampling.
ClassLocationKey = Union[int, tuple]
ClassLocations = Dict[ClassLocationKey, "np.ndarray"]


# ── Pure detection functions ───────────────────────────────────────────

def _has_voxels(class_locations: ClassLocations, key: int) -> bool:
    """Return True iff class_locations has a non-empty entry for `key`.

    Handles all the ways nnU-Net might represent "no voxels":
      - key absent from dict
      - key present with value None
      - key present with empty list / empty array

    Tuple-keyed entries (region-based training) are not considered when
    `key` is an int — they live under tuple keys.
    """
    v = class_locations.get(key)
    if v is None:
        return False
    try:
        return len(v) > 0
    except TypeError:  # not a sequence — treat as no voxels
        return False


def _is_lumbarization_signature(class_locations: ClassLocations) -> bool:
    """Lumbarization has an L6 vertebral body. No other subtype produces
    L6 voxels in our label scheme, so L6 voxels => lumb."""
    return _has_voxels(class_locations, L6_LABEL)


def _is_sacralization_count_signature(class_locations: ClassLocations) -> bool:
    """Sacralization with count change (sacr_count): L5 has fused into
    the sacrum body, so the lowest free vertebra is L4 and the sacrum
    is enlarged. Detection requires L5 absent (the actual finding),
    L4 present (rules out pelvic-only views where the lumbar region
    is masked), and sacrum present."""
    return (
        _has_voxels(class_locations, L4_LABEL)
        and not _has_voxels(class_locations, L5_LABEL)
        and _has_voxels(class_locations, SACRUM_LABEL)
    )


def select_lstv_biased_class(
    class_locations: ClassLocations,
    *,
    bias_l6_prob: float = 0.60,
    bias_sacrum_prob: float = 0.50,
    rng: Optional[np.random.Generator] = None,
) -> Optional[int]:
    """Pure function: choose biased foreground class from class_locations.

    This is the core decision logic, factored out of the DataLoader for
    testability. No nnU-Net dependency.

    Returns:
        Class id (int) to force, or None if no bias should be applied
        (caller should fall back to its default sampling).

    Args:
        class_locations: nnU-Net's preprocessed dict mapping class id ->
            array of foreground voxel coordinates for that class.
        bias_l6_prob: P(force L6) when the lumbarization signature is
            detected. 0.0 disables L6 bias entirely.
        bias_sacrum_prob: P(force sacrum) when the sacr_count signature
            is detected. 0.0 disables sacrum bias entirely.
        rng: numpy random source. Defaults to the global np.random module
            (so seed via np.random.seed). Pass a Generator for tests.

    Determinism note:
        rng=None uses np.random.random() which is the unseeded global
        source; tests should pass an explicit Generator with a fixed
        seed for reproducibility.
    """
    if rng is None:
        random_uniform = np.random.random
    else:
        random_uniform = rng.random

    # Lumbarization takes priority: lumb's defining anatomy (L6) is rarer
    # and harder for the model to learn than sacr_count's, so when both
    # signatures somehow fire we want L6 to win the bias slot.
    if _is_lumbarization_signature(class_locations):
        if bias_l6_prob > 0 and random_uniform() < bias_l6_prob:
            return L6_LABEL
    elif _is_sacralization_count_signature(class_locations):
        if bias_sacrum_prob > 0 and random_uniform() < bias_sacrum_prob:
            return SACRUM_LABEL

    return None


# ── Shared cross-process counters (v19.1) ──────────────────────────────
#
# Per-instance counters on the mixin (self._bias_n_*) are useful for unit
# tests but invisible across fork. nnU-Net's MultiThreadedAugmenter
# spawns 24 worker processes that each get their own copy of the loader
# instance — increments in workers never reach the parent's loader, so
# the parent process never sees a non-zero counter even if bias is
# firing thousands of times per epoch in workers.
#
# Fix: mirror the per-instance counters in module-level
# multiprocessing.Value objects. These are created at module import
# time, before nnU-Net forks workers, so workers inherit the same
# shared-memory backing and increments are visible across processes.
#
# Usage from the trainer:
#   from lstv_biased_dataloader import (
#       read_shared_bias_counters, reset_shared_bias_counters,
#   )
#   start = read_shared_bias_counters()    # at epoch start
#   ...
#   end = read_shared_bias_counters()      # at epoch end
#   per_epoch = {k: end[k] - start[k] for k in end}
#
# Counter values are CUMULATIVE across the lifetime of the Python
# process. To get per-epoch counts, snapshot at epoch start and diff
# at epoch end. Don't reset_shared_bias_counters() between epochs;
# that would create a race condition with workers that are mid-batch
# when the reset fires. Just diff.

try:
    _SHARED_BIAS_N_FORCE_FG = _mp.Value("q", 0)        # signed 64-bit
    _SHARED_BIAS_N_L6_RETURNED = _mp.Value("q", 0)
    _SHARED_BIAS_N_SACRUM_RETURNED = _mp.Value("q", 0)
    _SHARED_BIAS_N_UNIFORM_FALLBACK = _mp.Value("q", 0)
    _SHARED_COUNTERS_AVAILABLE = True
except Exception:  # pragma: no cover — multiprocessing should always be available
    _SHARED_BIAS_N_FORCE_FG = None
    _SHARED_BIAS_N_L6_RETURNED = None
    _SHARED_BIAS_N_SACRUM_RETURNED = None
    _SHARED_BIAS_N_UNIFORM_FALLBACK = None
    _SHARED_COUNTERS_AVAILABLE = False


def _safe_increment_shared(counter) -> None:
    """Atomically increment a multiprocessing.Value. No-op on None or
    on lock errors — diagnostic logging must never crash training."""
    if counter is None:
        return
    try:
        with counter.get_lock():
            counter.value += 1
    except Exception:
        pass


def read_shared_bias_counters() -> Dict[str, int]:
    """Read all shared bias counters atomically. Returns a snapshot
    dict. Safe to call from any process.

    Returns empty dict if multiprocessing.Value setup failed at import
    time (extremely unusual; would indicate a broken Python install).
    """
    if not _SHARED_COUNTERS_AVAILABLE:
        return {}
    try:
        with _SHARED_BIAS_N_FORCE_FG.get_lock():
            n_fg = int(_SHARED_BIAS_N_FORCE_FG.value)
        with _SHARED_BIAS_N_L6_RETURNED.get_lock():
            n_l6 = int(_SHARED_BIAS_N_L6_RETURNED.value)
        with _SHARED_BIAS_N_SACRUM_RETURNED.get_lock():
            n_sac = int(_SHARED_BIAS_N_SACRUM_RETURNED.value)
        with _SHARED_BIAS_N_UNIFORM_FALLBACK.get_lock():
            n_uf = int(_SHARED_BIAS_N_UNIFORM_FALLBACK.value)
    except Exception:
        return {}
    return {
        "n_force_fg": n_fg,
        "n_l6_returned": n_l6,
        "n_sacrum_returned": n_sac,
        "n_uniform_fallback": n_uf,
    }


def reset_shared_bias_counters() -> None:
    """Reset all shared counters to 0. Should only be called BEFORE
    workers fork (e.g. at training start) — calling mid-training
    creates a race with workers that may be mid-bbox-call.

    No-op if shared counters weren't successfully initialized.
    """
    if not _SHARED_COUNTERS_AVAILABLE:
        return
    for counter in (_SHARED_BIAS_N_FORCE_FG, _SHARED_BIAS_N_L6_RETURNED,
                     _SHARED_BIAS_N_SACRUM_RETURNED,
                     _SHARED_BIAS_N_UNIFORM_FALLBACK):
        try:
            with counter.get_lock():
                counter.value = 0
        except Exception:
            pass


# ── Mixin (testable independent of nnU-Net) ────────────────────────────

class LSTVBiasMixin:
    """Mixin that injects LSTV bias into any DataLoader-shaped class.

    Composed with `nnUNetDataLoader3D` to produce the production loader
    `LSTVBiasedDataLoader3D` (defined below). Composed with a fake
    parent for tests, so we can assert the override behavior without
    standing up a full nnU-Net dataset fixture.

    The mixin overrides exactly one method: `get_bbox`. Everything else
    delegates to super().
    """

    def __init__(
        self,
        *args,
        bias_l6_prob: Optional[float] = None,
        bias_sacrum_prob: Optional[float] = None,
        bias_enabled: Optional[bool] = None,
        **kwargs,
    ):
        # Strip our kwargs before delegating to super — the parent
        # nnUNetDataLoader3D doesn't expect them.
        super().__init__(*args, **kwargs)

        # Resolve config: explicit arg > env var > default
        self.bias_l6_prob = self._resolve_float(
            bias_l6_prob, "SPINESURG_LSTV_BIAS_L6_PROB", 0.60)
        self.bias_sacrum_prob = self._resolve_float(
            bias_sacrum_prob, "SPINESURG_LSTV_BIAS_SACRUM_PROB", 0.50)
        self.bias_enabled = self._resolve_bool(
            bias_enabled, "SPINESURG_LSTV_BIAS_ENABLED", True)

        # Per-process diagnostic counters. Reset to zero on construction.
        # In MultiThreadedAugmenter, each worker has its own copy.
        self._bias_n_force_fg = 0
        self._bias_n_l6_returned = 0
        self._bias_n_sacrum_returned = 0
        self._bias_n_uniform_fallback = 0

    @staticmethod
    def _resolve_float(arg_value, env_name, default):
        if arg_value is not None:
            return float(arg_value)
        env_value = os.environ.get(env_name, "").strip()
        if env_value:
            try:
                return float(env_value)
            except ValueError:
                pass
        return float(default)

    @staticmethod
    def _resolve_bool(arg_value, env_name, default):
        if arg_value is not None:
            return bool(arg_value)
        env_value = os.environ.get(env_name, "").strip().lower()
        if env_value in ("0", "false", "no", "off"):
            return False
        if env_value in ("1", "true", "yes", "on"):
            return True
        return bool(default)

    def get_bbox(
        self,
        data_shape,
        force_fg: bool,
        class_locations: Optional[ClassLocations],
        overwrite_class: Optional[Union[int, tuple]] = None,
    ):
        """Override: inject LSTV bias before delegating to parent.

        The parent's `get_bbox` already supports `overwrite_class` to
        deterministically crop around a chosen class. We compute that
        choice here when the conditions are right, then call super.

        Bias applies ONLY when:
          - force_fg is True (foreground-biased patch)
          - class_locations is not None (case has any foreground)
          - overwrite_class is None (caller didn't already force a class)
          - bias_enabled is True

        In all other cases we pass through unchanged.

        v19.1: increments BOTH per-instance AND shared cross-process
        counters. The shared counters survive fork/spawn so the parent
        process can read worker-side increments at epoch end.
        """
        if (
            force_fg
            and class_locations is not None
            and overwrite_class is None
            and self.bias_enabled
        ):
            self._bias_n_force_fg += 1
            _safe_increment_shared(_SHARED_BIAS_N_FORCE_FG)
            chosen = select_lstv_biased_class(
                class_locations,
                bias_l6_prob=self.bias_l6_prob,
                bias_sacrum_prob=self.bias_sacrum_prob,
            )
            if chosen == L6_LABEL:
                self._bias_n_l6_returned += 1
                _safe_increment_shared(_SHARED_BIAS_N_L6_RETURNED)
                overwrite_class = L6_LABEL
            elif chosen == SACRUM_LABEL:
                self._bias_n_sacrum_returned += 1
                _safe_increment_shared(_SHARED_BIAS_N_SACRUM_RETURNED)
                overwrite_class = SACRUM_LABEL
            else:
                # No signature matched (or roll missed) — let parent
                # do uniform sampling.
                self._bias_n_uniform_fallback += 1
                _safe_increment_shared(_SHARED_BIAS_N_UNIFORM_FALLBACK)

        return super().get_bbox(data_shape, force_fg, class_locations, overwrite_class)

    def bias_diagnostic_summary(self) -> Dict[str, int]:
        """Snapshot of this loader's bias counters. For debug / logging."""
        return {
            "n_force_fg": self._bias_n_force_fg,
            "n_l6_returned": self._bias_n_l6_returned,
            "n_sacrum_returned": self._bias_n_sacrum_returned,
            "n_uniform_fallback": self._bias_n_uniform_fallback,
            "bias_enabled": self.bias_enabled,
            "bias_l6_prob": self.bias_l6_prob,
            "bias_sacrum_prob": self.bias_sacrum_prob,
        }


# ── Concrete production class ──────────────────────────────────────────

# The actual loader class used in production. Imports nnU-Net at module
# top — if nnU-Net is missing the import fails loudly (intentional;
# this module has no purpose without it). Tests that only exercise the
# mixin / pure functions don't import this file at all — they import
# from a test-only fake parent path.

try:
    from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
    _NNUNET_AVAILABLE = True
except ImportError:  # pragma: no cover
    nnUNetDataLoader3D = object  # type: ignore
    _NNUNET_AVAILABLE = False


class LSTVBiasedDataLoader3D(LSTVBiasMixin, nnUNetDataLoader3D):
    """Production nnU-Net v2 3D DataLoader with LSTV-aware patch sampling.

    MRO: LSTVBiasedDataLoader3D -> LSTVBiasMixin -> nnUNetDataLoader3D
    -> nnUNetDataLoaderBase -> ...

    The mixin's `get_bbox` runs first, applies LSTV bias by setting
    `overwrite_class`, then delegates to nnU-Net's `get_bbox` which
    does the actual cropping.
    """
    pass


__all__ = [
    "L6_LABEL",
    "L4_LABEL",
    "L5_LABEL",
    "SACRUM_LABEL",
    "select_lstv_biased_class",
    "LSTVBiasMixin",
    "LSTVBiasedDataLoader3D",
    # v19.1: cross-process counter helpers
    "read_shared_bias_counters",
    "reset_shared_bias_counters",
]
