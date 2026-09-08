"""
SpineSurg-CT -- nnU-Net v2 trainers with W&B logging (v20)
tools/nnunet_wandb_variant.py

v20 — May 2026: MERGED LAST_LUMBAR + FOCUSED CONFUSION OUTPUT.
==============================================================
Two coupled changes for Dataset803 (SpineSurgCTFullMerged):

(1) MERGED LAST_LUMBAR. The L5/L6 distinction is removed at the
    network's training target. Source NIfTIs have their L6 voxels
    remapped to label 5 (now called "last_lumbar") by
    convert_hf_to_nnunet.py before nnU-Net ever sees them. All other
    labels are shifted down by one to maintain contiguous label
    values (sacrum 7→6, left_hip 8→7, right_hip 9→8, ignore 10→9).

    This eliminates the v19.1 class-collision failure mode where
    85-95% of GT-L6 voxels were misclassified as L5 across all 5
    folds (verified in jobs 35994118-35994122). Direct multi-class
    semantic segmentation cannot solve transitional vertebra labeling
    because the disambiguating signal is non-local (vertebra count
    from cervical or sacral landmarks). We follow the same approach
    Möller 2026 (VERIDAH §2.2) uses: train against a merged
    last_lumbar / last_thoracic class, then recover individual labels
    downstream via instance post-processing or VERIDAH's constrained
    sequence predictor.

    With this change, the patch bias dataloader (v18) is functionally
    obsolete — there is no L6 class to bias toward. SET
    SPINESURG_LSTV_BIAS_ENABLED=0 in your SLURM script to bypass it.
    The case-level LSTV oversampling (queue duplication for non-normal
    subtypes) is preserved at 25% target queue fraction (matches
    Möller's 4× per-case oversampling for anomalous cases).

(2) FOCUSED CONFUSION OUTPUT. The v19 bidirectional confusion logging
    emitted hundreds of W&B keys per epoch (one per subtype × GT class
    × predicted class for every direction). The v20 confusion logger
    emits only the three clinically-meaningful headline blocks plus
    one specificity number, totaling ~14 keys per epoch instead of
    ~1000+:

      Block 1 — last_lumbar GT on lumb cases (where do they end up?)
        Want: high fraction predicted as last_lumbar (the merge
        worked). Concerning: fractions to sacrum / L4 / background
        indicate boundary failures the post-processor will inherit.
          val/headline/last_lumbar_GT_on_lumb/total_voxels
          val/headline/last_lumbar_GT_on_lumb/as_last_lumbar_frac
          val/headline/last_lumbar_GT_on_lumb/as_sacrum_frac
          val/headline/last_lumbar_GT_on_lumb/as_L4_frac
          val/headline/last_lumbar_GT_on_lumb/as_background_frac

      Block 2 — L4 GT on sacr_count cases (where do they end up?)
        Want: high fraction predicted as L4 (no class collision in
        the new direction). Concerning: any fraction predicted as
        last_lumbar means the model still has the "bottom-most lumbar
        is always last_lumbar" prior, just relocated to the L4/L5
        boundary instead of the L5/L6 boundary.
          val/headline/L4_GT_on_sacr_count/total_voxels
          val/headline/L4_GT_on_sacr_count/as_L4_frac
          val/headline/L4_GT_on_sacr_count/as_last_lumbar_frac
          val/headline/L4_GT_on_sacr_count/as_sacrum_frac
          val/headline/L4_GT_on_sacr_count/as_background_frac

      Block 3 — predicted last_lumbar on sacr_count cases (what's
      actually there in GT?). On sacr_count cases the network should
      not predict last_lumbar at all (4-lumbar anatomy means the
      last_lumbar class is absent from GT). When it does, what is it
      pointing at?
          val/headline/pred_last_lumbar_on_sacr_count/total_voxels
          val/headline/pred_last_lumbar_on_sacr_count/actually_L4_frac
          val/headline/pred_last_lumbar_on_sacr_count/actually_sacrum_frac
          val/headline/pred_last_lumbar_on_sacr_count/actually_background_frac

      Specificity headline (one scalar per epoch):
          val/headline/last_lumbar_specificity_on_sacr_count
          val/headline/last_lumbar_specificity_on_sacr_count_n_cases

    Standard per-class dice metrics (val/dice/L1, val/dice/L2, …,
    val/dice/last_lumbar, val/dice/sacrum, val/dice/left_hip,
    val/dice/right_hip, mean_lumbar, mean_pelvis, mean_foreground)
    and per-subgroup dice (val/lstv_subgroup/{sub}/dice/{class}) are
    UNCHANGED — these are still cheap and valuable.

    The new headline metric is val/headline/last_lumbar_dice_on_lumbarization
    (replaces the old L6_dice_on_lumbarization). Under the merged
    scheme this should reach the same range as the network's other
    well-segmented vertebrae (~0.85+) because the class collision is
    gone.

History
-------
v19.1 — May 2026: Cross-process bias counters (multiprocessing.Value)
        for per-epoch patch-bias activity logging.
v19   — May 2026: Bidirectional confusion tracking (GT→pred and
        pred→GT). Replaced by v20's focused 3-block headline output.
v18   — May 2026: Patch class biasing via LSTVBiasedDataLoader3D
        subclass instead of np.random.choice monkey-patching.
v17.4 — Hallucination metrics (specificity tracking on absent
        classes). v20 retains only one specificity headline
        (last_lumbar on sacr_count).
v17.1-v17.3 — Dedicated LSTV val pass.
v17   — Symmetric per-record subtype demotion; ambiguous in oversample
        pool; verbose startup banner.
v15-v16 — Multiple monkey-patch attempts (superseded by v18).
v9-v14 — 6-way subgroup binning; lstv_cases.json schema v3; CE
         weight tensor sizing; partial-annotation contract.

Partial-annotation contract
===========================
Some training cases (separate-mode spine-only and pelvic-only) have
labels with value 9 (was 10 in source NIfTIs; renumbered by the
convert script) = IGNORE in regions outside the present annotator's
domain. nnU-Net v2's DC_and_CE_loss respects ignore_label when
configured, masking those voxels out of the gradient. Required:

  - dataset.json must have "ignore": 9 in its labels dict
    (set by convert_hf_to_nnunet.py:LABEL_NAMES, May 2026)
  - export_hf.py:merge_labels writes value 10 in source NIfTIs;
    convert_hf_to_nnunet.py remaps that to 9 for nnU-Net consumption
  - This trainer's _IGNORE_LABEL = 9 must match the converted dataset

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
# Functionally obsolete in v20 (no L6 class to bias toward); set
# SPINESURG_LSTV_BIAS_ENABLED=0 to disable. The import path is kept
# so the env-controlled bypass works on systems where the dataloader
# is no longer present alongside the trainer.
LSTVBiasedDataLoader3D = None
_LSTV_LOADER_IMPORT_ERROR: Optional[str] = None
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
        from lstv_biased_dataloader import (  # type: ignore
            LSTVBiasedDataLoader3D,
            read_shared_bias_counters as _read_shared_bias_counters,
            reset_shared_bias_counters as _reset_shared_bias_counters,
        )
    except ImportError as _exc2:
        try:
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


# ── Env helpers ────────────────────────────────────────────────────────────
# Defined EARLY (before the label-scheme toggle and the anatomy constants)
# because the label scheme must be resolved at import time — see the block
# below. These are pure os.environ reads with no other dependencies.

def _env_truthy(name, default=False):
    v = os.environ.get(name, "").strip().lower()
    return (v in ("1", "true", "yes", "on")) if v else default


def _env_int(name, default):
    try: return int(os.environ.get(name, str(default)))
    except Exception: return default


def _env_float(name, default):
    try: return float(os.environ.get(name, str(default)))
    except Exception: return default


# ── Label-scheme toggle (resolved ONCE, at import time) ────────────────────
# SPINESURG_LABEL_SCHEME selects which label scheme this trainer is built
# for. It MUST be resolved here, before any `def` below, because several
# module-level helpers bind scheme-dependent constants as DEFAULT ARGUMENTS
# (e.g. ignore_label=_IGNORE_LABEL, num_classes=_DEFAULT_NUM_OUTPUT_CLASSES),
# and Python evaluates default arguments at def-time. sbatch exports the env
# var before Python starts (see slurm/spine_train_array.sh), so it is stable
# at import. Do NOT try to mutate these constants at runtime.
#
#   merged   (default) -> Dataset803, 9 output channels (bg + 8 fg);
#                         L5/L6 collapsed into a single last_lumbar class.
#   unmerged           -> Dataset802, 10 output channels (bg + 9 fg);
#                         L5 and L6 are distinct classes (NeurIPS rebuttal
#                         baseline: retrained to show L6 collapses to ~0 Dice).
def _resolve_label_scheme() -> str:
    v = os.environ.get("SPINESURG_LABEL_SCHEME", "").strip().lower()
    if v in ("unmerged", "802", "legacy", "10class", "10-class"):
        return "unmerged"
    if v in ("oneshot", "one-shot", "810", "lstv_oneshot"):
        return "oneshot"
    # Anything else (unset / "merged" / "803" / typos) -> merged (default).
    return "merged"


_SCHEME = _resolve_label_scheme()


# ── Anatomy constants (scheme-dependent; finalized at import time) ─────────
# _ANATOMY_NAMES is parallel to _FG_CLASS_IDS so that
# _ANATOMY_NAMES[cid - 1] yields the human-readable name for the
# foreground class with ID cid.
if _SCHEME in ("unmerged", "oneshot"):
    # Dataset802 — legacy 10-class scheme; L5 and L6 are distinct.
    # Foreground class IDs are 1..9 (9 classes).
    _ANATOMY_NAMES = ["L1", "L2", "L3", "L4", "L5", "L6",
                      "sacrum", "left_hip", "right_hip"]

    _FG_CLASS_IDS    = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    _LUMBAR_IDS      = [1, 2, 3, 4, 5, 6]   # L1..L6
    _PELVIS_IDS      = [7, 8, 9]            # sacrum, left_hip, right_hip

    # Semantic label-ID aliases used by the headline logic.
    _L4_LABEL_ID     = 4   # bottom lumbar on sacr_count (4-lumbar) cases
    _L5_LABEL_ID     = 5   # distinct L5
    _L6_LABEL_ID     = 6   # distinct L6 (the collapse-to-L5 rebuttal metric)
    _SACRUM_LABEL_ID = 7
    _IGNORE_LABEL    = 10

    # Network output channel count under the Dataset802 label schema:
    # background + 9 foreground classes (ignore is masked, not classified).
    _DEFAULT_NUM_OUTPUT_CLASSES = 10
elif _SCHEME == "oneshot":
    # Dataset810 — the LSTV one-shot scheme (tools/convert_hf_to_nnunet.py,
    # LABEL_NAMES_ONESHOT): thoracic levels T10..T13 and lumbar levels L1..L6 distinct,
    # ribs 1-11 pooled per side, the twelfth and thirteenth ribs and the lumbar rib per
    # side as their own classes, sacrum, hips, femur. 22 foreground classes, ignore = 23.
    _ANATOMY_NAMES = ["T10", "T11", "T12", "T13", "L1", "L2", "L3", "L4", "L5", "L6",
                      "sacrum", "rib_left", "rib_right", "rib12_left", "rib12_right",
                      "rib13_left", "rib13_right", "lumbar_rib_left", "lumbar_rib_right",
                      "left_hip", "right_hip", "femur"]
    _FG_CLASS_IDS    = list(range(1, 23))
    _LUMBAR_IDS      = [5, 6, 7, 8, 9, 10]          # L1..L6
    _PELVIS_IDS      = [11, 20, 21, 22]             # sacrum, hips, femur
    _L4_LABEL_ID     = 8
    _L5_LABEL_ID     = 9
    _L6_LABEL_ID     = 10
    _SACRUM_LABEL_ID = 11
    _IGNORE_LABEL    = 23
    _DEFAULT_NUM_OUTPUT_CLASSES = 23
else:
    # Dataset803 (default) — merged last_lumbar, contiguous IDs.
    # Foreground class IDs are 1..8 (8 classes).
    _ANATOMY_NAMES = ["L1", "L2", "L3", "L4", "last_lumbar",
                       "sacrum", "left_hip", "right_hip"]

    _FG_CLASS_IDS    = [1, 2, 3, 4, 5, 6, 7, 8]
    _LUMBAR_IDS      = [1, 2, 3, 4, 5]   # L1..last_lumbar
    _PELVIS_IDS      = [6, 7, 8]         # sacrum, left_hip, right_hip

    # Semantic label-ID aliases used by the headline logic.
    _L4_LABEL_ID          = 4   # bottom anatomical lumbar in sacr_count cases
    _LAST_LUMBAR_LABEL_ID = 5   # merged former-L5 + former-L6 (was 5 / 6)
    _SACRUM_LABEL_ID      = 6   # was 7 in unmerged scheme
    _IGNORE_LABEL         = 9   # was 10 in unmerged scheme

    # Network output channel count under the Dataset803 label schema:
    # background + 8 foreground classes (ignore is masked, not classified).
    _DEFAULT_NUM_OUTPUT_CLASSES = 9

_VALID_CONFIGS   = ("fused", "spine_only", "pelvic_native")

# 6-way LSTV taxonomy (unchanged from v17+).
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
_OVERSAMPLE_SUBTYPES = tuple(s for s in _KNOWN_SUBTYPES if s != _SUBTYPE_NORMAL)

LSTV_CASES_SCHEMA_MIN = 1
LSTV_CASES_SCHEMA_MAX = 3

# Hallucination threshold: a per-class predicted-voxel count above which
# we consider the prediction a "meaningful hallucination" rather than
# noise at a boundary. ~1 cc at typical CT spacing of ~0.8 mm³/voxel.
_HALLUC_VOXEL_THRESHOLD = 100

# W&B verbosity. When True, the emitter additionally logs the full
# per-(subgroup, class) matrix (val/lstv_subgroup/{sub}/dice/{class} and
# .../n_with_class/{class}) — ~100 low-signal keys/epoch. Default False
# emits the FOCUSED set only (headlines + per-subgroup roll-ups + nnU-Net
# per-class dice); every rebuttal metric (val/headline/*) stays always-on.
_WANDB_VERBOSE = _env_truthy("SPINESURG_WANDB_VERBOSE", default=False)


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
        pool = {cid for cid, sub in case_to_subtype.items() if sub in _OVERSAMPLE_SUBTYPES}
        # THE THORACOLUMBAR HALF OF THE POOL. A lumbar rib sits in 16 of 802 records and a
        # T13 in fewer, and most of those records are "normal" by lumbosacral subtype, so
        # a subtype-only pool never shows them to the sampler more than uniformly. Under
        # the one-shot scheme those are the classes the benchmark exists to learn, so any
        # case whose attributes flag a lumbar rib or a T13 joins the pool too.
        attrs = data.get("case_to_attrs") or {}
        if isinstance(attrs, dict):
            for cid, at in attrs.items():
                if isinstance(at, dict) and (at.get("has_lumbar_rib") or at.get("has_t13")):
                    pool.add(cid)
                    all_case_ids.add(cid)
                    case_to_subtype.setdefault(cid, _SUBTYPE_NORMAL)
        oversample_pool = sorted(pool)
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


def _scan_lstv_case_ids_by_last_lumbar_voxels(labels_dir: str) -> Set[str]:
    """Legacy fallback: identify lumb-candidate cases by scanning for
    last_lumbar voxels (label 5).

    Note that under the v20 merged-label scheme this is a much weaker
    signal than the v19 L6-voxel scan was: every spine has last_lumbar
    voxels (L5 in normals, last_lumbar=5 in lumb cases). This fallback
    therefore returns ALL cases with any lumbar annotation, which makes
    it useless for distinguishing lumb from normal. It exists only for
    backwards compatibility with extreme-fallback paths; in practice
    the lstv_cases.json or HF manifest path always succeeds and is
    used preferentially.
    """
    import nibabel as nib
    out: Set[str] = set()
    if not os.path.isdir(labels_dir): return out
    for fn in sorted(os.listdir(labels_dir)):
        if not fn.endswith(".nii.gz"): continue
        try:
            arr = np.asarray(nib.load(join(labels_dir, fn)).dataobj).astype(np.int16)
            # Scan for the bottom-most lumbar label (last_lumbar=5 under the
            # merged scheme, L6=6 under the unmerged scheme). Referenced via
            # _LUMBAR_IDS[-1] so this stays valid under both label schemes.
            if np.any(arr == _LUMBAR_IDS[-1]):
                out.add(fn[:-len(".nii.gz")])
        except Exception: continue
    return out


# ── Per-case dice / voxel-count helpers ──────────────────────────────────────

def _per_case_dice_per_class(pred_argmax: torch.Tensor, gt: torch.Tensor,
                              ignore_label: int = _IGNORE_LABEL) -> List[Optional[float]]:
    """Per-class Dice on a single case. Returns a list of length
    len(_FG_CLASS_IDS); entry i corresponds to class _FG_CLASS_IDS[i].
    None means the class is absent from this case's GT (Dice undefined).

    Voxels with gt == ignore_label are masked out of both numerator
    and denominator so partial-annotation cases don't get punished
    for unannotated regions.
    """
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

    Used for hallucination tracking — when gt_n == 0, the case is in
    the "should not predict" condition for that class; pred_n is then
    the number of voxels the model wrongly emitted.

    Voxels with gt == ignore_label are excluded from BOTH gt_n and
    pred_n, matching the standard dice computation. This means a
    pelvic-only case (which has ignore in the lumbar region) won't
    contribute spurious "last_lumbar hallucination" stats.

    Returns a list of (gt_n, pred_n) integer tuples in the same order
    as _FG_CLASS_IDS.
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


# ── Bidirectional confusion (v19 unchanged; v20 only changes the
#    aggregation/emission to focus on three headline blocks) ─────────────────

def _per_case_confusion_by_gt(pred_argmax: torch.Tensor, gt: torch.Tensor,
                               num_classes: int = _DEFAULT_NUM_OUTPUT_CLASSES,
                               ignore_label: int = _IGNORE_LABEL
                               ) -> Dict[int, "np.ndarray"]:
    """GT-direction confusion: for each GT class C present in this
    case, return a length-num_classes vector where index k = count of
    voxels with (gt==C, pred==k).

    Returns {gt_class: confusion_vector}. Classes absent from GT
    (gt_n == 0) are not in the dict. Voxels with gt == ignore_label
    are excluded entirely.
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
        pred_at_gt = np.clip(pred_at_gt, 0, num_classes - 1)
        out[cid] = np.bincount(pred_at_gt, minlength=num_classes).astype(np.int64)
    return out


def _per_case_confusion_by_pred(pred_argmax: torch.Tensor, gt: torch.Tensor,
                                 num_classes: int = _DEFAULT_NUM_OUTPUT_CLASSES,
                                 ignore_label: int = _IGNORE_LABEL
                                 ) -> Dict[int, "np.ndarray"]:
    """PRED-direction confusion: for each predicted class C with
    non-zero voxels in this case, return a length-num_classes vector
    where index k = count of voxels with (pred==C, gt==k).

    Returns {pred_class: confusion_vector}. Classes absent from PRED
    (pred_n == 0) are not in the dict. Voxels with gt == ignore_label
    are excluded entirely.
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

    Background = 0; foreground class IDs map via _ANATOMY_NAMES[cid-1].
    Unknown IDs render as 'label{cid}' (defensive against unexpected
    network output values).
    """
    if cid == 0:
        return "background"
    if 1 <= cid <= len(_ANATOMY_NAMES):
        return _ANATOMY_NAMES[cid - 1]
    return f"label{cid}"


# ── v20 confusion-headline keys (single source of truth for tests) ──────────
# A "confusion key spec" describes one entry in a headline block. The
# block aggregator looks up `(direction, anchor_class, target_class,
# subgroup)`, computes the appropriate fraction, and emits W&B keys
# named `val/headline/{block_id}/{leaf}`.
#
# We name them out here (not deep inside the aggregator) so the test
# suite can assert the full set of keys is emitted on a synthetic
# fixture, and so the print-summary code uses the SAME spec to format
# the SLURM-log block.
#
# Direction:
#   "by_gt"   — anchor on GT class; ask "where do those voxels go in
#               prediction?" (frac = pred_class_count / gt_class_total)
#   "by_pred" — anchor on PRED class; ask "what was actually there in
#               GT?" (frac = gt_class_count / pred_class_total)
#
# anchor_class: the class we hold fixed (GT side or PRED side).
# target_classes: which classes to emit fractions for. We deliberately
# emit only L4, last_lumbar, sacrum, and background so the W&B run
# stays uncluttered. Other classes (L1-L3, hips) would be near-zero
# noise on the relevant subgroups anyway.

if _SCHEME in ("unmerged", "oneshot"):
    # Unmerged (Dataset802) rebuttal block: anchor on GT-L6 over lumb cases
    # and ask where those voxels get classified. The `as_L5_frac` entry is
    # THE rebuttal metric — direct multi-class classification collapses L6
    # onto L5, so this fraction is expected to be high while val/dice/L6 is
    # ~0. Wired through the same _aggregate_block_by_gt machinery as merged.
    _HEADLINE_TARGETS_L6_GT_ON_LUMB = (
        ("L5",         _L5_LABEL_ID,     "as_L5_frac",
         "THE collapse — GT-L6 voxels predicted as L5 (unmerged failure mode)"),
        ("L4",         _L4_LABEL_ID,     "as_L4_frac",
         "GT-L6 voxels predicted as L4"),
        ("sacrum",     _SACRUM_LABEL_ID, "as_sacrum_frac",
         "GT-L6 voxels predicted as sacrum"),
        ("background", 0,                "as_background_frac",
         "GT-L6 voxels predicted as background"),
    )
else:
    _HEADLINE_TARGETS_LL_GT_ON_LUMB = (
        ("last_lumbar", _LAST_LUMBAR_LABEL_ID, "as_last_lumbar_frac",
         "good — merged class predicted correctly"),
        ("sacrum",      _SACRUM_LABEL_ID,      "as_sacrum_frac",
         "concerning — bottom lumbar predicted as sacrum"),
        ("L4",          _L4_LABEL_ID,          "as_L4_frac",
         "concerning — bottom lumbar predicted as L4 (count truncated)"),
        ("background",  0,                     "as_background_frac",
         "concerning — lumbar voxel predicted as background"),
    )

    _HEADLINE_TARGETS_L4_GT_ON_SACR_COUNT = (
        ("L4",          _L4_LABEL_ID,          "as_L4_frac",
         "good — L4 predicted correctly"),
        ("last_lumbar", _LAST_LUMBAR_LABEL_ID, "as_last_lumbar_frac",
         "concerning — model still calls bottom lumbar 'last_lumbar' even with 4-lumbar count"),
        ("sacrum",      _SACRUM_LABEL_ID,      "as_sacrum_frac",
         "concerning — L4 predicted as sacrum"),
        ("background",  0,                     "as_background_frac",
         "concerning — L4 predicted as background"),
    )

    _HEADLINE_TARGETS_PRED_LL_ON_SACR_COUNT = (
        ("L4",          _L4_LABEL_ID,          "actually_L4_frac",
         "predicted last_lumbar at L4 anatomy (the failure mode)"),
        ("sacrum",      _SACRUM_LABEL_ID,      "actually_sacrum_frac",
         "predicted last_lumbar at sacrum anatomy"),
        ("background",  0,                     "actually_background_frac",
         "predicted last_lumbar in noise"),
    )


# =============================================================================
# Class reweighting
# =============================================================================

def _build_ce_class_weights(num_classes: int) -> Optional[torch.Tensor]:
    """Build a CE weight tensor of shape (num_classes,).

    `num_classes` MUST equal the number of output channels of the
    network — typically 9 here (background + 8 foreground). The
    ignore label is NOT a class in the network output (it's masked
    via ignore_index) and therefore has no entry in this weight tensor.

    v20 CE weights, with rationale:
      bg=0.5         — downweight; large area, easy to learn
      L1-L4=1.0      — baseline
      last_lumbar=1.5 — modest boost; class is no longer rare (was
                       L5+L6 combined), but it still benefits from a
                       small upweight because mis-labeling the bottom
                       lumbar is the clinically-important error
      sacrum=2.0     — emphasized for sacr_count and Castellvi
                       morphologies where the L5/sacrum boundary is
                       the discriminative feature
      hips=1.0       — baseline

    The L6=4.0 boost from v19 is GONE — there is no L6 channel.
    """
    if not _env_truthy("SPINESURG_CE_REWEIGHT", default=True):
        return None
    if num_classes is None or num_classes <= 0:
        return None

    if _SCHEME == "oneshot":
        # One-shot (Dataset810): 22 foreground channels. Boost what is rare and what the
        # benchmark exists to learn: L6, T13, the twelfth and thirteenth ribs, the lumbar
        # ribs, and the sacrum at the junction. Everything else is 1.
        weights = {0: 0.5}
        for cid, nm in enumerate(_ANATOMY_NAMES, start=1):
            weights[cid] = {"T12": 1.5, "T13": 2.5, "L5": 1.5, "L6": 3.0, "sacrum": 1.5,
                            "rib12_left": 2.0, "rib12_right": 2.0,
                            "rib13_left": 2.5, "rib13_right": 2.5,
                            "lumbar_rib_left": 3.0, "lumbar_rib_right": 3.0}.get(nm, 1.0)
        name_to_id = {"background": 0, **{nm: cid for cid, nm in enumerate(_ANATOMY_NAMES, start=1)}}
    elif _SCHEME in ("unmerged", "oneshot"):
        # Unmerged (Dataset802) — L5 and L6 are distinct channels. Restore
        # the pre-merge rare-class boost: L6 is the rare class (present only
        # on lumbarized 6-lumbar spines) and needs a strong upweight; L5 a
        # modest one; sacrum emphasized at the lumbosacral boundary.
        # Indices 0-9: background, L1-L4, L5, L6, sacrum, left_hip,
        # right_hip. (No entry for ignore=10; ignore is masked.)
        weights = {
            0: 0.5,  # background
            1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0,  # L1-L4
            5: 1.5,  # L5
            6: 4.0,  # L6 (rare; strong boost to fight the collapse-to-L5)
            7: 2.0,  # sacrum
            8: 1.0, 9: 1.0,  # hips
        }
        name_to_id = {
            "background":  0,
            "L1":          1, "L2": 2, "L3": 3, "L4": 4,
            "L5":          5, "L6": 6,
            "sacrum":      7,
            "left_hip":    8, "right_hip": 9,
        }
    else:
        # Per-class weights keyed by the class index in the network output.
        # Indices 0-8: background, L1-L4, last_lumbar, sacrum, left_hip,
        # right_hip. (No entry for ignore=9; ignore is masked.)
        weights = {
            0: 0.5,  # background
            1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0,  # L1-L4
            5: 1.5,  # last_lumbar (merged former L5 + former L6)
            6: 2.0,  # sacrum
            7: 1.0, 8: 1.0,  # hips
        }
        name_to_id = {
            "background":  0,
            "L1":          1, "L2": 2, "L3": 3, "L4": 4,
            "last_lumbar": 5,
            "sacrum":      6,
            "left_hip":    7, "right_hip": 8,
        }
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
      3. _DEFAULT_NUM_OUTPUT_CLASSES (9) as final fallback
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
# v20 confusion aggregation — focused headline blocks
# =============================================================================

def _aggregate_block_by_gt(
    confusion_buffers: Dict[str, List[Dict[int, np.ndarray]]],
    subtype: str,
    anchor_gt_cid: int,
    target_specs: Tuple[Tuple[str, int, str, str], ...],
    block_id: str,
    out: Dict[str, float],
) -> None:
    """Aggregate one 'GT-direction' confusion block into `out`.

    For the given `subtype`, sums per-case confusion vectors anchored
    on GT class `anchor_gt_cid` across all val cases of that subtype.
    Then emits W&B keys under `val/headline/{block_id}/...`:

      - {block_id}/total_voxels
      - {block_id}/{leaf}  for each (name, target_cid, leaf, _) in
        target_specs, where the value is target_voxels / total_voxels

    If no cases exist or the anchor class has no GT voxels in any
    case, no keys are emitted (caller can detect via missing keys).
    """
    cases = confusion_buffers.get(subtype, [])
    if not cases:
        return
    summed: Optional[np.ndarray] = None
    for case_dict in cases:
        vec = case_dict.get(anchor_gt_cid)
        if vec is None:
            continue
        if summed is None:
            summed = np.zeros_like(vec, dtype=np.int64)
        summed = summed + vec
    if summed is None:
        return
    total = int(summed.sum())
    if total == 0:
        return
    base = f"val/headline/{block_id}"
    out[f"{base}/total_voxels"] = float(total)
    for name, target_cid, leaf, _rationale in target_specs:
        if 0 <= target_cid < len(summed):
            out[f"{base}/{leaf}"] = float(int(summed[target_cid]) / total)


def _aggregate_block_by_pred(
    confusion_buffers: Dict[str, List[Dict[int, np.ndarray]]],
    subtype: str,
    anchor_pred_cid: int,
    target_specs: Tuple[Tuple[str, int, str, str], ...],
    block_id: str,
    out: Dict[str, float],
) -> None:
    """Aggregate one 'PRED-direction' confusion block. Symmetric to
    _aggregate_block_by_gt but anchored on a predicted class — emits
    `actually_{name}_frac` under `val/headline/{block_id}/...`.
    """
    cases = confusion_buffers.get(subtype, [])
    if not cases:
        return
    summed: Optional[np.ndarray] = None
    for case_dict in cases:
        vec = case_dict.get(anchor_pred_cid)
        if vec is None:
            continue
        if summed is None:
            summed = np.zeros_like(vec, dtype=np.int64)
        summed = summed + vec
    if summed is None:
        return
    total = int(summed.sum())
    if total == 0:
        return
    base = f"val/headline/{block_id}"
    out[f"{base}/total_voxels"] = float(total)
    for name, target_cid, leaf, _rationale in target_specs:
        if 0 <= target_cid < len(summed):
            out[f"{base}/{leaf}"] = float(int(summed[target_cid]) / total)


# =============================================================================
# W&B mixin (6-way subgroup tracking + v20 focused confusion)
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
    _val_subgroup_voxels: Dict[str, List[List[Tuple[int, int]]]] = None
    _val_subgroup_confusion_by_gt: Dict[str, List[Dict[int, "np.ndarray"]]] = None
    _val_subgroup_confusion_by_pred: Dict[str, List[Dict[int, "np.ndarray"]]] = None
    _val_case_to_subtype: Optional[Dict[str, str]] = None
    _per_subgroup_logging_enabled: bool = True
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

    # ── Phase logging helpers ────────────────────────────────────────────

    def _phase(self, phase: str, msg: str) -> None:
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

    # ── Label-scheme / dataset consistency check ──────────────────────────

    def _verify_label_scheme_matches_dataset(self) -> None:
        """Assert the import-time _SCHEME matches the dataset.json actually
        being trained on.

        Training Dataset802 (unmerged) data with the merged constants — or
        Dataset803 (merged) data with unmerged constants — would silently
        corrupt every per-class metric (channel-index vs label-value
        mismatch) and use the wrong ignore_index in the loss. This guard
        turns that into a loud, immediate failure.

        The dataset scheme is detected from its label dict: an 'L6'
        foreground key (or 9 foreground classes) => unmerged; a
        'last_lumbar' key (or 8 foreground classes) => merged. On mismatch,
        raises RuntimeError naming both the env _SCHEME and the detected
        dataset scheme. If the label dict cannot be read, the check is
        skipped (best-effort; it never blocks training on its own error).
        """
        labels: Dict = {}
        try:
            labels = dict(self.plans_manager.dataset_json.get("labels", {}))
        except Exception:
            try:
                labels = dict(getattr(self, "dataset_json", {}).get("labels", {}))
            except Exception:
                labels = {}
        if not labels:
            try:
                self.print_to_log_file(
                    "label-scheme check: dataset.json labels unavailable; "
                    f"skipping (env _SCHEME={_SCHEME})")
            except Exception:
                pass
            return

        label_keys = {str(k) for k in labels}
        n_fg = sum(1 for k in labels
                   if str(k).lower() not in ("ignore", "background"))
        if "rib12_left" in label_keys or "lumbar_rib_left" in label_keys:
            detected = "oneshot"
        elif "L6" in label_keys or n_fg == 9:
            detected = "unmerged"
        elif "last_lumbar" in label_keys or n_fg == 8:
            detected = "merged"
        else:
            detected = None

        try:
            self.print_to_log_file(
                f"label-scheme check: env _SCHEME={_SCHEME}, "
                f"detected dataset scheme={detected} (labels={labels})")
        except Exception:
            pass

        if detected is not None and detected != _SCHEME:
            raise RuntimeError(
                f"LABEL SCHEME MISMATCH: SPINESURG_LABEL_SCHEME resolved to "
                f"'{_SCHEME}' but the dataset.json being trained indicates a "
                f"'{detected}' scheme (labels={labels}). Set "
                f"SPINESURG_LABEL_SCHEME={detected} (802 -> unmerged, "
                f"803 -> merged), or point the trainer at the matching "
                f"dataset. Aborting to avoid corrupt per-class metrics and a "
                f"wrong loss ignore_index.")

    # ── Startup diagnostics ──────────────────────────────────────────────

    def _emit_startup_diagnostics(self) -> None:
        log = self.print_to_log_file
        try:
            log("")
            log("=" * 72)
            log("SPINESURG-CT TRAINER STARTUP DIAGNOSTICS (v20)")
            log("=" * 72)

            log(f"  trainer class:    {type(self).__name__}")
            try: ds_name = self.plans_manager.dataset_name
            except Exception: ds_name = "<unknown>"
            log(f"  dataset:          {ds_name}")
            log(f"  fold:             {getattr(self, 'fold', '?')}")
            _scheme_desc = ("merged (last_lumbar = former L5+L6)"
                            if _SCHEME == "merged"
                            else "unmerged (L5 and L6 distinct)")
            log(f"  label scheme:     {_SCHEME}  [{_scheme_desc}]")
            log(f"  num_fg_classes:   {len(_FG_CLASS_IDS)}  ({_ANATOMY_NAMES})")
            log(f"  ignore_label:     {_IGNORE_LABEL}")
            log(f"  num_output_chans: {_DEFAULT_NUM_OUTPUT_CLASSES}  (bg + fg)")
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
                log("    -> no JSON found; will fall back to manifests/last_lumbar scan")

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

            # ── PATCH BIAS section (functionally obsolete in v20) ──────
            log("")
            log("  PATCH BIAS (v18 — functionally obsolete in v20)")
            bias_enabled = _env_truthy("SPINESURG_LSTV_BIAS_ENABLED", default=False)
            log(f"    enabled (env SPINESURG_LSTV_BIAS_ENABLED): {bias_enabled}")
            log(f"    NOTE: under v20 merged labels, the bias loader's "
                f"L6-detection logic is no-op (no L6 voxels in dataset).")
            log(f"    Default in v20: DISABLED. Re-enable only for "
                f"experiments on legacy unmerged datasets.")
            if bias_enabled and not _LSTV_BIASED_LOADER_AVAILABLE:
                log(f"    ✗ LSTVBiasedDataLoader3D NOT IMPORTABLE")
                log(f"      import error: {_LSTV_LOADER_IMPORT_ERROR}")
                log(f"      bias is INACTIVE — nnU-Net default uniform sampling")
            elif bias_enabled:
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

            log("")
            log("  PER-SUBGROUP VAL DICE")
            log(f"    enabled: {self._per_subgroup_logging_enabled}")
            if self._val_case_to_subtype is not None:
                log(f"    subtype map size: {len(self._val_case_to_subtype)} cases")
                map_counts = Counter(self._val_case_to_subtype.values())
                log(f"    subtype distribution: {dict(map_counts)}")
            log(f"    classes tracked: {_ANATOMY_NAMES}")
            try:
                ddv_enabled, ddv_every = self._resolve_lstv_dedicated_val_settings()
                ddv_str = f"every {ddv_every} epoch(s)" if ddv_enabled else "DISABLED"
                log(f"    dedicated LSTV val pass: {ddv_str} "
                    f"(env: SPINESURG_LSTV_DEDICATED_VAL, "
                    f"SPINESURG_LSTV_DEDICATED_VAL_EVERY)")
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
                                        "pass-through mode.")
                            else:
                                log("    LSTV val cases this fold: 0 "
                                    "(dedicated pass will be a no-op)")
                        except Exception as exc:
                            log(f"    val keys probe failed: {exc}")
                    else:
                        log("    val keys probe: no usable list found")
            except Exception as exc:
                log(f"    dedicated-val settings resolve failed: {exc}")

            ce_w = getattr(self, "_ce_class_weights", None)
            if ce_w is not None:
                log("")
                log("  CE CLASS REWEIGHTING")
                try:
                    ce_str = ", ".join(
                        f"{_ANATOMY_NAMES[i-1] if 1 <= i <= len(_ANATOMY_NAMES) else 'bg' if i==0 else f'lab{i}'}"
                        f"={float(w):.2f}"
                        for i, w in enumerate(ce_w)
                    )
                    log(f"    weights: [{ce_str}]")
                except Exception:
                    log(f"    weights: {ce_w}")

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

    # ── Dedicated LSTV validation pass ──────────────────────────────────

    _lstv_dedicated_val_enabled: bool = True

    def _resolve_lstv_dedicated_val_settings(self) -> Tuple[bool, int]:
        enabled = _env_truthy("SPINESURG_LSTV_DEDICATED_VAL", default=True)
        every = max(1, _env_int("SPINESURG_LSTV_DEDICATED_VAL_EVERY", 1))
        return enabled, every

    def _find_underlying_val_loader(self):
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

        candidates: List[Tuple[object, str]] = []
        for ka in ("indices", "list_of_keys", "_indices"):
            if hasattr(underlying, ka):
                candidates.append((underlying, ka))
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
        if not self._per_subgroup_logging_enabled: return
        enabled, every = self._resolve_lstv_dedicated_val_settings()
        if not enabled: return

        epoch = getattr(self, "current_epoch", 0)
        if (epoch % every) != 0:
            return

        if self._val_case_to_subtype is None:
            return
        if self._val_subgroup_dice is None:
            self._reset_subgroup_dice_buffer()

        result = self._find_underlying_val_loader()
        if not isinstance(result, tuple) or len(result) != 4:
            self.print_to_log_file(
                "LSTV dedicated val: unexpected probe shape; skipping.")
            return
        underlying, keys_owner, keys_attr, is_callable = result
        if underlying is None or keys_owner is None:
            self.print_to_log_file(
                "LSTV dedicated val: cannot locate val dataloader internals; "
                "skipping (regular val sampler still active).")
            return

        try:
            raw = getattr(keys_owner, keys_attr)
            if is_callable:
                full_keys = list(raw())
            else:
                full_keys = list(raw)
        except Exception as exc:
            self.print_to_log_file(f"LSTV dedicated val: keys read failed: {exc}")
            return

        lstv_subtypes = set(_OVERSAMPLE_SUBTYPES)
        lstv_in_val = [
            k for k in full_keys
            if self._val_case_to_subtype.get(k, _SUBTYPE_NORMAL) in lstv_subtypes
        ]
        if not lstv_in_val:
            return

        if is_callable:
            self.print_to_log_file(
                f"LSTV dedicated val: keys attribute "
                f"{type(keys_owner).__name__}.{keys_attr} is a method, "
                f"not a settable list; cannot constrain val sampler. "
                f"Found {len(lstv_in_val)} LSTV cases in val but skipping "
                f"forced forward passes (regular val sampler will still "
                f"catch some). Pass-through mode.")
            return

        for sub in lstv_subtypes:
            self._val_subgroup_dice[sub] = []

        net = getattr(self, "network", None)
        if net is None:
            self.print_to_log_file(
                "LSTV dedicated val: self.network is None; skipping")
            return
        device = getattr(self, "device", torch.device("cuda"))

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
                    batch["keys"] = [cid] * (data.shape[0] if data.dim() >= 1 else 1)
                    self._record_per_case_dice(batch, output)
                    n_succeeded += 1
                except Exception as exc:
                    try:
                        self.print_to_log_file(
                            f"LSTV dedicated val: case {cid} failed: {exc}")
                    except Exception: pass
        finally:
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

        n_oversample = sum(1 for s in case_to_sub.values()
                            if s in _OVERSAMPLE_SUBTYPES)
        self._phase("subtype_map",
            f"oversample-pool size: {n_oversample} cases "
            f"(eligible subtypes: {sorted(_OVERSAMPLE_SUBTYPES)})")
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
        self._val_subgroup_voxels = {sub: [] for sub in _KNOWN_SUBTYPES}
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
            num_classes = pred_logits.size(1)
            for b in range(B):
                cid = str(keys[b])
                cid_lookup = cid[:-5] if cid.endswith("_0000") else cid
                subtype = self._subtype_for_case(cid_lookup)
                pred_b = pred_argmax[b].cpu().to(torch.int16)
                gt_b   = gt[b].cpu().to(torch.int16)
                dice_per_cls = _per_case_dice_per_class(pred_b, gt_b)
                voxels_per_cls = _per_case_voxel_counts_per_class(pred_b, gt_b)
                confusion_by_gt = _per_case_confusion_by_gt(
                    pred_b, gt_b, num_classes=num_classes)
                confusion_by_pred = _per_case_confusion_by_pred(
                    pred_b, gt_b, num_classes=num_classes)
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
        """Aggregate per-class dice across val cases per subgroup.

        Emits the standard per-(subgroup, class) dice keys
        (val/lstv_subgroup/{sub}/dice/{name}) and per-subgroup roll-ups
        (mean_lumbar, mean_pelvis, mean_fg). Also emits the v20 headline
        metrics:

          val/headline/last_lumbar_dice_on_lumbarization
          val/headline/last_lumbar_n_lumbarization_cases
          val/headline/L4_dice_on_sacr_count
          val/headline/L4_n_sacr_count_cases

        Then merges in v20 hallucination + confusion payloads.
        """
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
                per_class_means[cls_id] = m  # always computed: roll-ups + headlines depend on it
                # Verbose-only: the bulky per-(subgroup, class) matrix. The
                # roll-ups below and the val/headline/* metrics are derived
                # from per_class_means / the raw buffers, so gating the raw
                # matrix keys does not affect any always-on metric.
                if _WANDB_VERBOSE:
                    out[f"val/lstv_subgroup/{subtype}/dice/{_ANATOMY_NAMES[ci]}"] = m
                    out[f"val/lstv_subgroup/{subtype}/n_with_class/{_ANATOMY_NAMES[ci]}"] = float(len(vals))
            lumbar_means = [per_class_means[c] for c in _LUMBAR_IDS if c in per_class_means]
            pelvis_means = [per_class_means[c] for c in _PELVIS_IDS if c in per_class_means]
            fg_means = list(per_class_means.values())
            if lumbar_means: out[f"val/lstv_subgroup/{subtype}/mean_lumbar_dice"] = float(np.mean(lumbar_means))
            if pelvis_means: out[f"val/lstv_subgroup/{subtype}/mean_pelvis_dice"] = float(np.mean(pelvis_means))
            if fg_means:     out[f"val/lstv_subgroup/{subtype}/mean_fg_dice"] = float(np.mean(fg_means))

        # ── headline: bottom-lumbar dice on lumbarization ────────────
        # Merged (v20): last_lumbar dice on lumb cases — replaces the v19
        # L6_dice_on_lumbarization headline; should reach ~0.85+ once the
        # class collision is resolved.
        # Unmerged (802): L6 dice on lumb cases — THE rebuttal number,
        # expected ~0 because direct classification collapses L6 onto L5.
        lumb_cases = self._val_subgroup_dice.get(_SUBTYPE_LUMB, [])
        if lumb_cases:
            if _SCHEME in ("unmerged", "oneshot"):
                ll_idx = _FG_CLASS_IDS.index(_L6_LABEL_ID)
                ll_vals = [c[ll_idx] for c in lumb_cases if c[ll_idx] is not None]
                if ll_vals:
                    out["val/headline/L6_dice_on_lumbarization"] = float(np.mean(ll_vals))
                    out["val/headline/L6_n_lumbarization_cases"] = float(len(ll_vals))
            else:
                ll_idx = _FG_CLASS_IDS.index(_LAST_LUMBAR_LABEL_ID)
                ll_vals = [c[ll_idx] for c in lumb_cases if c[ll_idx] is not None]
                if ll_vals:
                    out["val/headline/last_lumbar_dice_on_lumbarization"] = float(np.mean(ll_vals))
                    out["val/headline/last_lumbar_n_lumbarization_cases"] = float(len(ll_vals))

        # ── v20 headline: L4 dice on sacr_count ──────────────────────
        # Symmetric counterpart: on 4-lumbar cases, L4 is the bottom
        # vertebra. We want this dice to be high. A drop here paired
        # with predicted-last_lumbar voxels appearing in Block 3 is
        # the new manifestation of the "bottom-most lumbar is always
        # last_lumbar" prior.
        sc_cases = self._val_subgroup_dice.get(_SUBTYPE_SACR_COUNT, [])
        if sc_cases:
            l4_idx = _FG_CLASS_IDS.index(_L4_LABEL_ID)
            l4_vals = [c[l4_idx] for c in sc_cases if c[l4_idx] is not None]
            if l4_vals:
                out["val/headline/L4_dice_on_sacr_count"] = float(np.mean(l4_vals))
                out["val/headline/L4_n_sacr_count_cases"] = float(len(l4_vals))

        # ── any-LSTV roll-up (unchanged) ─────────────────────────────
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

        # ── v20 hallucination + confusion ────────────────────────────
        try:
            halluc_payload = self._aggregate_subgroup_hallucination()
            out.update(halluc_payload)
        except Exception as exc:
            try: self.print_to_log_file(f"hallucination aggregation: {exc}")
            except Exception: pass
        try:
            confusion_payload = self._aggregate_subgroup_confusion()
            out.update(confusion_payload)
        except Exception as exc:
            try: self.print_to_log_file(f"confusion aggregation: {exc}")
            except Exception: pass
        return out

    def _aggregate_subgroup_hallucination(self) -> Dict[str, float]:
        """v20: Single specificity headline.

        Under the merged-label scheme, last_lumbar is the only
        foreground class GUARANTEED ABSENT in any subtype's GT —
        specifically, on sacr_count cases (4 lumbar, no last_lumbar).
        Other LSTV subtypes (sacralization, semisacralization) have
        last_lumbar present in GT (they have 5 lumbar vertebrae with
        anomalous lumbosacral morphology, not a missing vertebra).

        We compute one specificity number — fraction of sacr_count
        val cases where the model emitted ≤ _HALLUC_VOXEL_THRESHOLD
        voxels of last_lumbar — and emit it under
        val/headline/last_lumbar_specificity_on_sacr_count.

        Higher is better; 1.0 means no meaningful hallucination of
        last_lumbar on any sacr_count val case.
        """
        out: Dict[str, float] = {}
        if self._val_subgroup_voxels is None:
            return out
        cases = self._val_subgroup_voxels.get(_SUBTYPE_SACR_COUNT, [])
        if not cases:
            return out
        threshold = _HALLUC_VOXEL_THRESHOLD
        # The bottom lumbar class is guaranteed absent from sacr_count GT
        # (4-lumbar anatomy). Merged: last_lumbar (5). Unmerged: L6 (6).
        if _SCHEME in ("unmerged", "oneshot"):
            absent_cid = _L6_LABEL_ID
            key = "L6_specificity_on_sacr_count"
        else:
            absent_cid = _LAST_LUMBAR_LABEL_ID
            key = "last_lumbar_specificity_on_sacr_count"
        ll_idx = _FG_CLASS_IDS.index(absent_cid)
        absent_preds = [c[ll_idx][1] for c in cases if c[ll_idx][0] == 0]
        if not absent_preds:
            return out
        n_meaningful = sum(1 for p in absent_preds if p > threshold)
        specificity = 1.0 - (n_meaningful / len(absent_preds))
        out[f"val/headline/{key}"] = specificity
        out[f"val/headline/{key}_n_cases"] = float(len(absent_preds))
        out[f"val/headline/{key}_mean_pred_voxels"] = (
            float(np.mean(absent_preds)) if absent_preds else 0.0
        )
        return out

    def _aggregate_subgroup_confusion(self) -> Dict[str, float]:
        """v20: focused 3-block confusion output.

        Replaces v19's verbose per-(subtype, class, direction) emission
        (~1000 keys) with three clinically-meaningful headline blocks
        anchored on L4 / last_lumbar transitions:

          last_lumbar_GT_on_lumb       — Block 1, GT-direction
          L4_GT_on_sacr_count          — Block 2, GT-direction
          pred_last_lumbar_on_sacr_count — Block 3, PRED-direction

        Each block emits one total-voxels key plus 3-4 fraction keys.
        """
        out: Dict[str, float] = {}
        if (self._val_subgroup_confusion_by_gt is None
                or self._val_subgroup_confusion_by_pred is None):
            return out

        if _SCHEME in ("unmerged", "oneshot"):
            # Unmerged (802) rebuttal block: GT-L6 on lumb cases, where do
            # those voxels get classified? `as_L5_frac` is THE collapse
            # evidence (GT-L6 predicted as L5).
            _aggregate_block_by_gt(
                self._val_subgroup_confusion_by_gt,
                subtype=_SUBTYPE_LUMB,
                anchor_gt_cid=_L6_LABEL_ID,
                target_specs=_HEADLINE_TARGETS_L6_GT_ON_LUMB,
                block_id="L6_GT_on_lumb",
                out=out,
            )
            return out

        # Block 1: last_lumbar GT on lumb cases
        _aggregate_block_by_gt(
            self._val_subgroup_confusion_by_gt,
            subtype=_SUBTYPE_LUMB,
            anchor_gt_cid=_LAST_LUMBAR_LABEL_ID,
            target_specs=_HEADLINE_TARGETS_LL_GT_ON_LUMB,
            block_id="last_lumbar_GT_on_lumb",
            out=out,
        )

        # Block 2: L4 GT on sacr_count cases
        _aggregate_block_by_gt(
            self._val_subgroup_confusion_by_gt,
            subtype=_SUBTYPE_SACR_COUNT,
            anchor_gt_cid=_L4_LABEL_ID,
            target_specs=_HEADLINE_TARGETS_L4_GT_ON_SACR_COUNT,
            block_id="L4_GT_on_sacr_count",
            out=out,
        )

        # Block 3: predicted last_lumbar on sacr_count cases
        _aggregate_block_by_pred(
            self._val_subgroup_confusion_by_pred,
            subtype=_SUBTYPE_SACR_COUNT,
            anchor_pred_cid=_LAST_LUMBAR_LABEL_ID,
            target_specs=_HEADLINE_TARGETS_PRED_LL_ON_SACR_COUNT,
            block_id="pred_last_lumbar_on_sacr_count",
            out=out,
        )

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
            # Per-class breakdown reads the verbose-only matrix keys; when
            # SPINESURG_WANDB_VERBOSE is off those keys are absent, so this
            # would print an all-"---" line. Guard it (still degrades
            # gracefully via .get if a single class is missing while verbose).
            if _WANDB_VERBOSE:
                for sub in (_SUBTYPE_LUMB, _SUBTYPE_SACR_COUNT, _SUBTYPE_SEMI,
                             _SUBTYPE_SACR, _SUBTYPE_AMBIG):
                    n = payload.get(f"val/lstv_subgroup/{sub}/n_cases")
                    if not n or int(n) == 0: continue
                    cls_parts = []
                    for cn in _ANATOMY_NAMES:
                        v = payload.get(f"val/lstv_subgroup/{sub}/dice/{cn}")
                        cls_parts.append(f"{cn}={v:.2f}" if v is not None else f"{cn}=---")
                    self.print_to_log_file(f"  {sub} per-class: [{', '.join(cls_parts)}]")

            if _SCHEME in ("unmerged", "oneshot"):
                # ── unmerged (802) rebuttal headline: L6 collapse ──────
                l6_dice = payload.get("val/headline/L6_dice_on_lumbarization")
                l6_n    = payload.get("val/headline/L6_n_lumbarization_cases")
                if l6_dice is not None and l6_n is not None:
                    self.print_to_log_file(
                        f"  HEADLINE: L6 dice on lumbarization cases "
                        f"= {l6_dice:.3f}  (n={int(l6_n)})  "
                        f"[expected ~0 — L6 collapses to L5 under direct "
                        f"multi-class classification]")
                l4_dice = payload.get("val/headline/L4_dice_on_sacr_count")
                l4_n    = payload.get("val/headline/L4_n_sacr_count_cases")
                if l4_dice is not None and l4_n is not None:
                    self.print_to_log_file(
                        f"  HEADLINE: L4 dice on sacr_count cases "
                        f"= {l4_dice:.3f}  (n={int(l4_n)})")
                spec = payload.get("val/headline/L6_specificity_on_sacr_count")
                spec_n = payload.get("val/headline/L6_specificity_on_sacr_count_n_cases")
                spec_mean = payload.get("val/headline/L6_specificity_on_sacr_count_mean_pred_voxels")
                if spec is not None and spec_n is not None:
                    mean_str = (f", mean_pred={spec_mean:.0f} voxels"
                                if spec_mean is not None else "")
                    self.print_to_log_file(
                        f"  HEADLINE: L6 specificity on sacr_count "
                        f"= {spec:.3f}  (n_absent={int(spec_n)}{mean_str})  "
                        f"[higher is better; 1.0 = no hallucination]")

                self.print_to_log_file(
                    "  --- unmerged confusion block (L6 collapse) ---")
                self._print_confusion_block(
                    payload,
                    block_id="L6_GT_on_lumb",
                    header="    L6 GT-voxels on lumb cases "
                           "(total={total}, where do they end up classified?):",
                    target_specs=_HEADLINE_TARGETS_L6_GT_ON_LUMB,
                    empty_msg="    L6 GT-voxels on lumb cases: "
                              "(no lumb cases this epoch)",
                )
            else:
                # ── v20 headline metrics ──────────────────────────────
                ll_dice = payload.get("val/headline/last_lumbar_dice_on_lumbarization")
                ll_n    = payload.get("val/headline/last_lumbar_n_lumbarization_cases")
                if ll_dice is not None and ll_n is not None:
                    self.print_to_log_file(
                        f"  HEADLINE: last_lumbar dice on lumbarization cases "
                        f"= {ll_dice:.3f}  (n={int(ll_n)})")
                l4_dice = payload.get("val/headline/L4_dice_on_sacr_count")
                l4_n    = payload.get("val/headline/L4_n_sacr_count_cases")
                if l4_dice is not None and l4_n is not None:
                    self.print_to_log_file(
                        f"  HEADLINE: L4 dice on sacr_count cases "
                        f"= {l4_dice:.3f}  (n={int(l4_n)})")

                # ── v20 specificity headline ──────────────────────────
                spec = payload.get("val/headline/last_lumbar_specificity_on_sacr_count")
                spec_n = payload.get("val/headline/last_lumbar_specificity_on_sacr_count_n_cases")
                spec_mean = payload.get("val/headline/last_lumbar_specificity_on_sacr_count_mean_pred_voxels")
                if spec is not None and spec_n is not None:
                    mean_str = (f", mean_pred={spec_mean:.0f} voxels"
                                if spec_mean is not None else "")
                    self.print_to_log_file(
                        f"  HEADLINE: last_lumbar specificity on sacr_count "
                        f"= {spec:.3f}  (n_absent={int(spec_n)}{mean_str})  "
                        f"[higher is better; 1.0 = no hallucination]")

                # ── v20 confusion blocks (3 focused) ──────────────────
                self.print_to_log_file(
                    "  --- v20 confusion blocks (L4 / last_lumbar focus) ---")
                self._print_confusion_block(
                    payload,
                    block_id="last_lumbar_GT_on_lumb",
                    header="    last_lumbar GT-voxels on lumb cases "
                           "(total={total}, where do they end up classified?):",
                    target_specs=_HEADLINE_TARGETS_LL_GT_ON_LUMB,
                    empty_msg="    last_lumbar GT-voxels on lumb cases: "
                              "(no lumb cases this epoch)",
                )
                self._print_confusion_block(
                    payload,
                    block_id="L4_GT_on_sacr_count",
                    header="    L4 GT-voxels on sacr_count cases "
                           "(total={total}, where do they end up classified?):",
                    target_specs=_HEADLINE_TARGETS_L4_GT_ON_SACR_COUNT,
                    empty_msg="    L4 GT-voxels on sacr_count cases: "
                              "(no sacr_count cases this epoch)",
                )
                self._print_confusion_block(
                    payload,
                    block_id="pred_last_lumbar_on_sacr_count",
                    header="    Predicted-last_lumbar voxels on sacr_count cases "
                           "(total={total}, what's actually there in GT?):",
                    target_specs=_HEADLINE_TARGETS_PRED_LL_ON_SACR_COUNT,
                    empty_msg="    Predicted-last_lumbar voxels on sacr_count cases: "
                              "(no sacr_count cases this epoch, or model "
                              "predicted no last_lumbar there — good)",
                )
        except Exception as exc:
            try: self.print_to_log_file(f"per-subgroup summary failed: {exc}")
            except Exception: pass

    def _print_confusion_block(self, payload: Dict[str, float],
                                block_id: str, header: str,
                                target_specs: Tuple, empty_msg: str) -> None:
        """Print one v20 confusion headline block. If the block has no
        total voxels (no eligible cases this epoch), print the
        empty_msg fallback line.
        """
        total = payload.get(f"val/headline/{block_id}/total_voxels")
        if total is None or int(total) == 0:
            self.print_to_log_file(empty_msg)
            return
        self.print_to_log_file(header.format(total=int(total)))
        for name, _target_cid, leaf, rationale in target_specs:
            frac = payload.get(f"val/headline/{block_id}/{leaf}")
            if frac is None:
                continue
            # Marker visually distinguishes "good" from "concerning" lines.
            marker = "✓" if "good" in rationale.lower() else " "
            self.print_to_log_file(
                f"      {marker} {name:<12s}: {frac*100:5.1f}%   ({rationale})"
            )

    # ── v19.1: cross-process bias counter reporting (kept; quiet under v20) ──

    def _collect_bias_counter_payload(self, epoch: int) -> Dict[str, float]:
        """Read shared bias counters, diff against epoch-start snapshot,
        emit per-epoch + cumulative numbers as a W&B payload dict.

        Under v20 (bias disabled), counters typically stay at 0 and
        this method is a near-no-op. Kept for parity with legacy runs
        and to support optional re-enabling on legacy datasets.
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

        n_fg_e = now.get("n_force_fg", 0) - start.get("n_force_fg", 0)
        n_l6_e = now.get("n_l6_returned", 0) - start.get("n_l6_returned", 0)
        n_sac_e = now.get("n_sacrum_returned", 0) - start.get("n_sacrum_returned", 0)
        n_uf_e = now.get("n_uniform_fallback", 0) - start.get("n_uniform_fallback", 0)

        # Only emit per-epoch keys if the bias loader actually fired.
        # On v20 (bias disabled) the values stay at 0 and we suppress
        # the keys to keep the W&B run tidy.
        if n_fg_e > 0 or n_l6_e > 0 or n_sac_e > 0 or n_uf_e > 0:
            out["patch_bias/this_epoch/n_force_fg"]         = float(n_fg_e)
            out["patch_bias/this_epoch/n_l6_returned"]      = float(n_l6_e)
            out["patch_bias/this_epoch/n_sacrum_returned"]  = float(n_sac_e)
            out["patch_bias/this_epoch/n_uniform_fallback"] = float(n_uf_e)
            if n_fg_e > 0:
                out["patch_bias/this_epoch/l6_rate"] = n_l6_e / n_fg_e
                out["patch_bias/this_epoch/sacrum_rate"] = n_sac_e / n_fg_e
                out["patch_bias/this_epoch/uniform_fallback_rate"] = n_uf_e / n_fg_e
            out["patch_bias/cumulative/n_force_fg"]         = float(now.get("n_force_fg", 0))
            out["patch_bias/cumulative/n_l6_returned"]      = float(now.get("n_l6_returned", 0))
            out["patch_bias/cumulative/n_sacrum_returned"]  = float(now.get("n_sacrum_returned", 0))
            out["patch_bias/cumulative/n_uniform_fallback"] = float(now.get("n_uniform_fallback", 0))

            try:
                self.print_to_log_file(
                    f"--- patch bias activity (epoch {epoch}) ---"
                )
                self.print_to_log_file(
                    f"  this epoch: force_fg={n_fg_e}, "
                    f"L6 forces={n_l6_e}, sacrum forces={n_sac_e}, "
                    f"uniform fallback={n_uf_e}"
                )
                if n_fg_e > 0:
                    self.print_to_log_file(
                        f"  rates: L6={(n_l6_e/n_fg_e)*100:.2f}%, "
                        f"sacrum={(n_sac_e/n_fg_e)*100:.2f}%, "
                        f"no-bias={(n_uf_e/n_fg_e)*100:.2f}%"
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
            "trainer_version": "v20",
            "label_scheme": ("merged_last_lumbar" if _SCHEME == "merged"
                             else "oneshot_22class" if _SCHEME == "oneshot"
                             else "unmerged_l5_l6"),
            "label_scheme_env": _SCHEME,
            "anatomy_mapping": dict(enumerate(_ANATOMY_NAMES, start=1)),
            "ignore_label": _IGNORE_LABEL,
            "subgroup_taxonomy": "6-way",
            "subgroups": list(_KNOWN_SUBTYPES),
            "oversample_subtypes": list(_OVERSAMPLE_SUBTYPES),
            "ce_reweight": _env_truthy("SPINESURG_CE_REWEIGHT", default=True),
            "lstv_oversample_frac": _env_float("LSTV_OVERSAMPLE_FRAC", 0.25),
            "patch_bias_enabled": _env_truthy("SPINESURG_LSTV_BIAS_ENABLED", default=False),
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

        # Fail loudly BEFORE any training work if the import-time label
        # scheme (SPINESURG_LABEL_SCHEME) does not match the dataset.json
        # being trained. NOT wrapped in try/except — a mismatch must abort.
        self._verify_label_scheme_matches_dataset()

        try: _install_nnunet_warning_filter()
        except Exception: pass
        try: self._maybe_apply_ce_reweighting()
        except Exception as exc: self.print_to_log_file(f"CE reweight: {exc}")
        try: self._install_log_intercepts()
        except Exception: pass
        try: self._init_wandb()
        except Exception as exc: self.print_to_log_file(f"WandB init: {exc}")

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self._reset_subgroup_dice_buffer()
        if _read_shared_bias_counters is not None:
            try:
                self._bias_counters_at_epoch_start = _read_shared_bias_counters()
            except Exception as exc:
                self._bias_counters_at_epoch_start = {}
                try: self.print_to_log_file(
                    f"bias counter snapshot at epoch start: {exc}")
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
                # nnU-Net's dice_per_class_or_region is a vector of
                # length num_fg_classes (excluding background). For
                # Dataset803 this has 8 entries corresponding to
                # _FG_CLASS_IDS in order.
                for i, v in enumerate(dpc):
                    if i >= len(_ANATOMY_NAMES): continue
                    try: vf = float(v)
                    except Exception: continue
                    if np.isnan(vf): continue
                    per_class_log[f"val/dice/{_ANATOMY_NAMES[i]}"] = vf
                    fg_v.append(vf)
                    # _FG_CLASS_IDS[i] determines whether this index
                    # is lumbar (1-5) or pelvis (6-8).
                    cls_id = _FG_CLASS_IDS[i] if i < len(_FG_CLASS_IDS) else None
                    if cls_id in _LUMBAR_IDS:
                        lumbar_v.append(vf)
                    elif cls_id in _PELVIS_IDS:
                        pelvis_v.append(vf)
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
                self._validate_lstv_dedicated()
            except Exception as exc:
                self.print_to_log_file(f"LSTV dedicated val: {exc}")
            try:
                sg = self._aggregate_subgroup_dice()
                payload.update(sg)
                self._print_per_subgroup_summary(epoch, sg)
            except Exception as exc:
                self.print_to_log_file(f"subgroup aggregation: {exc}")
            try:
                bias_payload = self._collect_bias_counter_payload(epoch)
                payload.update(bias_payload)
            except Exception as exc:
                self.print_to_log_file(f"bias counter logging: {exc}")
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
    """Oversample all 5 LSTV variant subtypes at the dataloader queue level.

    v20: The patch-class biasing layer (LSTVBiasedDataLoader3D) is
    bypassed by default because under merged labels there is no L6
    class to bias toward. The case-level oversampling that this mixin
    performs at queue level is unchanged and remains the primary lever
    for ensuring the model sees rare LSTV anatomy proportionally more
    often during training.

    Pool composition (5 of 6 subtypes; 'normal' is the majority):
      - lumb              (lumbarization, 6-lumbar count)
      - sacr_count        (sacralization with full L5->sacrum count change)
      - sacralization     (sacral-wing morphology, count unchanged)
      - semisacralization (unilateral sacral-wing fusion)
      - ambiguous         (uncertain Castellvi classification)
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

        self.print_to_log_file(
            "LSTV oversample: legacy fallback path inactive under v20 "
            "(last_lumbar scan returns all cases). Ensure lstv_cases.json "
            "or HF manifest is present for the dataset.")
        self._lstv_case_ids = set()
        return self._lstv_case_ids

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

    def get_dataloaders(self):
        """v20: bias substitution is OFF by default.

        Under merged labels, the LSTVBiasedDataLoader3D's L6-detection
        logic is meaningless (no L6 voxels exist in the dataset). Set
        SPINESURG_LSTV_BIAS_ENABLED=1 only if running on a legacy
        unmerged dataset; otherwise the default-off path produces
        nnU-Net's standard uniform foreground sampling.
        """
        bias_enabled = _env_truthy("SPINESURG_LSTV_BIAS_ENABLED", default=False)

        if not bias_enabled:
            return super().get_dataloaders()

        if not _LSTV_BIASED_LOADER_AVAILABLE:
            self.print_to_log_file(
                f"LSTV patch bias: LSTVBiasedDataLoader3D not importable "
                f"({_LSTV_LOADER_IMPORT_ERROR}); using nnU-Net default loader.")
            return super().get_dataloaders()

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
                "trainer module; using nnU-Net default loader.")
            return super().get_dataloaders()

        l6_p = _env_float("SPINESURG_LSTV_BIAS_L6_PROB", 0.60)
        sac_p = _env_float("SPINESURG_LSTV_BIAS_SACRUM_PROB", 0.50)
        self.print_to_log_file(
            f"LSTV patch bias: substituting nnUNetDataLoader3D -> "
            f"LSTVBiasedDataLoader3D for get_dataloaders() "
            f"(L6 prob={l6_p:.2f}, sacrum prob={sac_p:.2f})  "
            f"[NOTE: under v20 merged labels, the loader's "
            f"L6-detection is no-op; bias has no effect on Dataset803]")

        _trainer_mod.nnUNetDataLoader3D = LSTVBiasedDataLoader3D
        try:
            return super().get_dataloaders()
        finally:
            _trainer_mod.nnUNetDataLoader3D = original_3d

    def on_train_start(self):
        super().on_train_start()
        try: self._apply_lstv_sampler()
        except Exception as exc: self.print_to_log_file(f"LSTV oversample: {exc}")
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
    """500 epochs + LSTV queue oversample + CE reweight.
    DEFAULT for the SpineSurg-CT paper run.

    v20 (May 2026): Merged last_lumbar at the dataset level (Dataset803);
         class collision at the L5/L6 boundary eliminated by training
         against a single merged class. Patch class biasing
         (SPINESURG_LSTV_BIAS_ENABLED) defaults to OFF — the bias
         loader's L6 detection is meaningless under merged labels.
         Confusion W&B output reduced from ~1000 keys/epoch to ~14
         (three focused headline blocks: last_lumbar GT on lumb,
         L4 GT on sacr_count, predicted last_lumbar on sacr_count).
    v18: Patch class biasing via LSTVBiasedDataLoader3D.
    v17.4: Hallucination metrics.
    v17.1-v17.3: Dedicated LSTV val pass.
    v17: 6-way oversample pool; symmetric subtype refinement.
    v15-v16: monkey-patch attempts (superseded).
    v12: Per-subgroup dice numpy-keys fix.
    v11: CE weight tensor sized to actual network output.
    v10: 6-way subgroups.
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
