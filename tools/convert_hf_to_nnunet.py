"""
convert_hf_to_nnunet.py — Convert HF NIfTI-pair export to nnU-Net v2 raw layout.

Reads
-----
  HF_DIR/
    manifest_train.json
    manifest_validation.json
    manifest_test.json
    ct/{token}_ct.nii.gz
    labels/{token}_label.nii.gz
  splits_5fold.json    (from generate_5fold_splits.py v6+)

Writes
------
  NNUNET_RAW/Dataset{ID}_{NAME}/
    imagesTr/{caseID}_0000.nii.gz   (symlinks or copies of CT)
    labelsTr/{caseID}.nii.gz        (REMAPPED copies — see below)
    imagesTs/{caseID}_0000.nii.gz   (test set)
    labelsTs/{caseID}.nii.gz
    dataset.json                    (label scheme + ignore)
    lstv_cases.json                 (schema v3 — 6-way LSTV subtypes)
    splits_final.json               (5-fold CV splits, mirrored from input)

Modes
-----
Default: full build (link/copy + dataset.json + splits + lstv_cases).
--regen_splits_only: only re-write splits_final.json + lstv_cases.json.
    Skips the per-record link/copy loop and skips dataset.json. Use this
    when the dataset already exists on disk and you only need to refresh
    the fold assignments / subtype mapping (e.g. after re-running
    generate_5fold_splits.py).

Label scheme (May 2026 — Dataset803 / "merged" variant)
-------------------------------------------------------
  0 background
  1 L1
  2 L2
  3 L3
  4 L4
  5 last_lumbar  (was L5+L6 merged at convert time — see
                  LABEL_REMAP_AT_CONVERT)
  6 sacrum       (renumbered from 7)
  7 left_hip     (renumbered from 8)
  8 right_hip    (renumbered from 9)
  9 ignore       (renumbered from 10)

Why contiguous renumbering matters
==================================
nnU-Net v2 builds the network's output channel count from the SORTED
LIST of label values (excluding the ignore label). The trainer's
per-class dice / confusion code then compares network argmax outputs
(channel indices) against ground-truth label values, treating them
as equivalent. This equivalence holds only when label values are
CONTIGUOUS — a gap (e.g., labels 0,1,...,5,7,8,9 with no 6) breaks
the channel-index ↔ label-value mapping and corrupts every per-class
metric. So when we drop L6, we shift sacrum/hips/ignore down by one
to maintain contiguity.

Why labels are merged
=====================
Direct multi-class semantic segmentation of L5 vs L6 fails on
transitional anatomy because the disambiguating signal is non-local:
"the bottom-most lumbar vertebra" looks anatomically near-identical
whether it is the L5 of a 5-lumbar normal spine or the L6 of a
6-lumbar lumbarized spine. Voxel-wise classification cannot solve
this — see Möller et al. 2026 (VERIDAH), Section 2.2:

    "we replace the T13 and L6 labels in the reference annotation
     with T12 and L5, respectively. Thus, the existence of a T13
     and L6 is indicated by two consecutive T12 or L5 labels."

We follow the same pattern for L6 only (we do not have T13 cases in
this dataset). Individual L5/L6 labels are recovered downstream via
instance-level post-processing (connected components on the
last_lumbar mask, ordered superior-to-inferior, with count
determining whether the bottom is L5 or L6).

Why "ignore" is here
====================
Some HF export records are *partial annotations* — separate-mode
patients have one CT scan annotated only for spine (L1-L6) and a
different CT scan annotated only for pelvis (sacrum + hips). The
patched export_hf.py (May 2026) writes those label files with value
10 in voxels outside the present annotator's domain.

nnU-Net v2 honors `"ignore"` in `labels` by:
  - setting `has_ignore_label = True` on the LabelManager
  - using `ignore_label = 10` as `ignore_index` in CE loss
  - masking ignore voxels out of the Dice loss via the DC_and_CE_loss
    wrapper when `ignore_label` is configured
  - NOT allocating an output head for class 10

Without this entry, partial-annotation labels would falsely supervise
the network with "background" wherever the partial annotator did not
trace, poisoning the loss for last_lumbar / sacrum / hips on roughly
two-thirds of the training set. See export_hf.py
"PARTIAL ANNOTATION CONTRACT".

Per-record subtype refinement (May 2026, schema v3)
===================================================
Unchanged from prior version. A patient with LSTV morphology may
contribute multiple records with different `config` values (fused,
spine_only, pelvic_native). The patient-level subtype is correct only
for records whose config actually shows the relevant anatomy:

  pelvic_native  ->  demote {lumb, sacr_count} to "normal"
                     (lumbar region masked as ignore=10)
  spine_only     ->  demote {sacralization, semisacralization} to
                     "normal" (pelvis masked as ignore=10)
  fused          ->  preserve all subtypes
  ambiguous      ->  preserved on every config

lstv_cases.json schema v3 (Apr 2026)
====================================
6-way LSTV subtype taxonomy:
  normal, lumb, sacr_count, semisacralization, sacralization, ambiguous

  {
    "schema_version": 3,
    "subtypes": ["normal", "lumb", ...],
    "subtype_counts": {...},
    "case_to_subtype":   {caseID: subtype, ...},
    "case_to_token":     {caseID: patient_token, ...},
    "case_to_attrs":     {caseID: {n_lumb_max, has_l6_any, lstv_label, ...}},
    "lstv_oversample_pool": [caseID, ...]   (all non-normal cases for sampler)
  }
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("spinesurg.convert_hf_to_nnunet")

LSTV_CASES_SCHEMA = 3

SUBTYPES = (
    "normal",
    "lumb",
    "sacr_count",
    "semisacralization",
    "sacralization",
    "ambiguous",
)

# Subtypes whose defining anatomy lives in the LUMBAR vertebral region.
# pelvic_native config masks the lumbar region as ignore=10, so a
# pelvic_native record does NOT carry these subtypes' signal.
_LUMBAR_REGION_SUBTYPES = frozenset({
    "lumb",
    "sacr_count",
})

# Subtypes whose defining anatomy lives in the SACRAL/PELVIC region.
# spine_only config masks the pelvis as ignore=10, so a spine_only
# record does NOT carry these subtypes' signal.
_PELVIS_REGION_SUBTYPES = frozenset({
    "sacralization",
    "semisacralization",
})

_VALID_CONFIGS = ("fused", "spine_only", "pelvic_native")
_PELVIC_ONLY_CONFIG = "pelvic_native"
_SPINE_ONLY_CONFIG  = "spine_only"

# ── Label scheme (Dataset803 — merged last_lumbar) ─────────────────────
#
# nnU-Net v2 reads the "labels" dict to determine the network's output
# head channel count. Each foreground name -> int value pair becomes
# an output channel. The "ignore" entry is special: it is NOT a
# channel; it is used as the ignore_index in the CE loss.
#
# Label 6 is intentionally absent from LABEL_NAMES — the L6 voxels
# from source NIfTIs are remapped to label 5 (last_lumbar) at convert
# time via LABEL_REMAP_AT_CONVERT. The network has no channel for the
# old L6 because there are no label-6 voxels in the produced data.
LABEL_NAMES = {
    "background":  0,
    "L1":          1,
    "L2":          2,
    "L3":          3,
    "L4":          4,
    "last_lumbar": 5,   # merged L5+L6 (see module docstring)
    "sacrum":      6,   # renumbered from 7
    "left_hip":    7,   # renumbered from 8
    "right_hip":   8,   # renumbered from 9
    "ignore":      9,   # renumbered from 10
}

# Source-label -> trained-label remap. Applied by
# _remap_and_write_label() during the link/copy loop. Source labels
# not listed here pass through unchanged.
#
# We collapse L6 into last_lumbar AND shift everything above it down
# by one to maintain contiguous label values. See module docstring
# "Why contiguous renumbering matters".
#
# In the source NIfTIs (HF export):
#   0 bg, 1-4 L1-L4, 5 L5, 6 L6, 7 sacrum, 8 left_hip, 9 right_hip,
#   10 ignore (partial-annotation contract)
#
# In the converted NIfTIs (this dataset):
#   0 bg, 1-4 L1-L4, 5 last_lumbar (former L5+L6), 6 sacrum,
#   7 left_hip, 8 right_hip, 9 ignore
LABEL_REMAP_AT_CONVERT: Dict[int, int] = {
    6: 5,   # L6 -> last_lumbar (the merge)
    7: 6,   # sacrum: shift down
    8: 7,   # left_hip: shift down
    9: 8,   # right_hip: shift down
    10: 9,  # ignore: shift down (partial-annotation contract)
}

# No-ignore variant (fused-only ablation, May 2026).
# ==================================================
# Reviewer ask (camera-ready): a "fused-only versus all-cases" ablation,
# evaluated on the fused test split. Fused records are FULLY annotated —
# export_hf.py guarantees they carry NO ignore (10) voxels (it logs an
# error if a fused label ever contains one). So when we restrict the
# training/test pool to fused records, the ignore label becomes vestigial
# and we drop it entirely: there is no partial-annotation machinery, no
# ignore_index in the loss. This is the "don't use the ignore protocol"
# arm of the ablation.
#
# The L6 -> last_lumbar merge is KEPT (it is orthogonal to the ignore
# protocol) so the label semantics stay identical to the main Dataset803
# model and the two arms differ only in (training pool, ignore protocol).
#
# Source 10 (ignore) -> background (0) defensively. On fused data this
# never fires; it only matters if a stray ignore voxel slips through, in
# which case treating it as background is the safe, fully-supervised
# choice (there is no ignore class to route it to).
LABEL_REMAP_NO_IGNORE: Dict[int, int] = {
    6: 5,   # L6 -> last_lumbar (the merge)
    7: 6,   # sacrum: shift down
    8: 7,   # left_hip: shift down
    9: 8,   # right_hip: shift down
    10: 0,  # ignore -> background (defensive; fused carries none)
}

# 9-key scheme (8 foreground + background), no ignore class.
LABEL_NAMES_NO_IGNORE = {
    "background":  0,
    "L1":          1,
    "L2":          2,
    "L3":          3,
    "L4":          4,
    "last_lumbar": 5,
    "sacrum":      6,
    "left_hip":    7,
    "right_hip":   8,
}

# The maximum label value the source NIfTIs are expected to contain.
# Anything above this in a source label file is unexpected; the remap
# is defensive and clamps via lookup table size.
_MAX_SOURCE_LABEL = 10


def _build_remap_lut(remap: Dict[int, int],
                     max_label: int = _MAX_SOURCE_LABEL) -> np.ndarray:
    """Build a vectorized lookup table for label remapping.

    Returns an int32 array `lut` of length `max_label + 1` where
    `lut[old_label] == new_label`. Source labels not in `remap` map
    to themselves. Source labels above `max_label` are not handled
    by the LUT; the caller must clip or assert before indexing.
    """
    lut = np.arange(max_label + 1, dtype=np.int32)
    for src, dst in remap.items():
        if not (0 <= src <= max_label):
            raise ValueError(
                f"label remap source {src} outside expected range "
                f"[0, {max_label}]; widen _MAX_SOURCE_LABEL or fix remap dict")
        lut[src] = dst
    return lut


def _remap_and_write_label(src: Path, dst: Path, lut: np.ndarray) -> None:
    """Read source label NIfTI, apply the remap LUT, write to dst.

    Used when the dataset.json label scheme differs from the source
    files (e.g., merging L6 into last_lumbar to avoid class collision
    in transitional vertebra training). Preserves the source affine
    and header. Output dtype is uint8 (label IDs are small).

    The LUT must accommodate the maximum value in the source file.
    Values exceeding the LUT length are clipped to the LUT max — this
    is defensive against unexpected source labels and mirrors the
    behavior of np.clip + indexing.
    """
    import nibabel as nib

    if dst.exists() or dst.is_symlink():
        dst.unlink()
    img = nib.load(src)
    arr = np.asarray(img.get_fdata(), dtype=np.int32)
    src_max = int(arr.max()) if arr.size else 0
    if src_max >= len(lut):
        log.warning("source label %s contains value %d above LUT max %d; clipping",
                    src.name, src_max, len(lut) - 1)
        arr = np.clip(arr, 0, len(lut) - 1)
    arr_remapped = lut[arr].astype(np.uint8)
    out = nib.Nifti1Image(arr_remapped, img.affine, img.header)
    out.set_data_dtype(np.uint8)
    nib.save(out, dst)


def _refine_subtype_for_record_config(patient_subtype: str,
                                       record_config: str) -> str:
    """Adjust the patient-level subtype based on this record's
    annotation config (symmetric anatomical-region demotion).

    Unchanged from prior version. See module docstring for full
    semantics.
    """
    rc = (record_config or "").strip().lower()
    if rc == _PELVIC_ONLY_CONFIG:
        if patient_subtype in _LUMBAR_REGION_SUBTYPES:
            return "normal"
    elif rc == _SPINE_ONLY_CONFIG:
        if patient_subtype in _PELVIS_REGION_SUBTYPES:
            return "normal"
    return patient_subtype


def _load_manifest_records(hf_dir: Path) -> Dict[str, List[Dict]]:
    """Load manifests grouped by split name (train/validation/test)."""
    out: Dict[str, List[Dict]] = {}
    for split, fn in (
        ("train",      "manifest_train.json"),
        ("validation", "manifest_validation.json"),
        ("test",       "manifest_test.json"),
    ):
        p = hf_dir / fn
        if not p.exists():
            log.warning("manifest not found: %s", p)
            out[split] = []
            continue
        data = json.loads(p.read_text())
        if isinstance(data, dict):
            data = data.get("records", data.get("cases", list(data.values())))
        if isinstance(data, list):
            out[split] = [r for r in data if isinstance(r, dict)]
        else:
            out[split] = []
        log.info("loaded %d records from %s", len(out[split]), fn)
    return out


def _link_or_copy(src: Path, dst: Path, use_symlinks: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if use_symlinks:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def _case_id(token: str, record_idx: int = 0) -> str:
    """Generate canonical caseID for nnU-Net. Multiple records per patient
    get suffixed _r1, _r2 etc; first record uses bare token."""
    if record_idx == 0:
        return f"COLONOG_{token}"
    return f"COLONOG_{token}_r{record_idx}"


def _resolve_subtype_for_case(record: Dict,
                                splits_subtypes: Dict[str, str]) -> str:
    """Determine the canonical 6-way subtype for a single record.

    Steps:
      1. Resolve patient-level subtype via splits-derived map (preferred)
         or per-record manifest fields (fallback).
      2. Refine based on this record's `config` to demote lumbar-region
         subtypes to 'normal' for pelvic-only views.
    """
    tok = str(record.get("token") or record.get("patient_token") or "")
    record_config = str(record.get("config", ""))

    # Step 1: patient-level subtype
    if tok in splits_subtypes:
        patient_subtype = splits_subtypes[tok]
    else:
        label = str(record.get("lstv_label", "")).upper()
        pel   = str(record.get("lstv_pelvic", "")).upper()
        vert  = str(record.get("lstv_vertebral", "")).upper()
        n_lumb = int(record.get("n_lumbar_labels", 0) or 0)
        has_l6 = bool(record.get("has_l6", False))

        if label == "AMBIGUOUS":
            patient_subtype = "ambiguous"
        elif (vert == "LUMBARIZATION" and pel == "SACRALIZATION") or \
             (vert == "SACRALIZATION" and pel == "LUMBARIZATION"):
            patient_subtype = "ambiguous"
        elif n_lumb == 6 or has_l6 or label == "LUMBARIZATION":
            patient_subtype = "lumb"
        elif label in ("SEMI_SACRAL", "SEMI_SACRALIZATION") or \
             pel in ("SEMI_SACRAL", "SEMI_SACRALIZATION"):
            patient_subtype = "semisacralization"
        elif vert == "SACRALIZATION" and n_lumb == 4:
            patient_subtype = "sacr_count"
        elif label == "SACRALIZATION" or pel == "SACRALIZATION":
            patient_subtype = "sacralization"
        else:
            patient_subtype = "normal"

    return _refine_subtype_for_record_config(patient_subtype, record_config)


def _build_case_indices(
    manifests: Dict[str, List[Dict]],
    splits_subtypes: Dict[str, str],
    hf_dir: Path,
    images_tr: Optional[Path],
    labels_tr: Optional[Path],
    images_ts: Optional[Path],
    labels_ts: Optional[Path],
    use_symlinks: bool,
    do_link: bool,
    label_remap_lut: Optional[np.ndarray],
    include_configs: Optional[frozenset] = None,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Dict],
           List[str], List[str], int, Dict[str, int],
           Dict[str, int], int, Dict[str, int]]:
    """
    Iterate manifests, optionally write CT (link/copy) and label
    (remap-and-write) files into the nnU-Net layout, build indices.

    include_configs: when not None, only records whose `config` (lower-cased)
        is in this set are kept; all others are dropped from EVERY split
        (train/val/test). Used by the fused-only ablation (include_configs
        = {"fused"}). None means keep all records.

    Returns: (case_to_subtype, case_to_token, case_to_attrs,
              train_case_ids, test_case_ids, n_skipped,
              refinement_stats, remap_stats, n_filtered, filtered_by_config)

    refinement_stats: how many cases were demoted by config.
    remap_stats: how many label files were physically rewritten via
                 the remap LUT (vs symlinked when no remap configured).
    n_filtered: how many records were dropped by include_configs.
    filtered_by_config: per-config breakdown of the dropped records.
    """
    case_to_subtype: Dict[str, str] = {}
    case_to_token:   Dict[str, str] = {}
    case_to_attrs:   Dict[str, Dict] = {}
    train_case_ids: List[str] = []
    test_case_ids:  List[str] = []
    n_skipped = 0

    refinement_stats: Dict[str, int] = defaultdict(int)
    remap_stats: Dict[str, int] = defaultdict(int)
    n_filtered = 0
    filtered_by_config: Dict[str, int] = defaultdict(int)

    record_idx_by_token: Dict[str, int] = defaultdict(int)

    for split, recs in manifests.items():
        for rec in recs:
            tok = str(rec.get("token") or rec.get("patient_token") or "")
            if not tok:
                n_skipped += 1
                continue

            # Config filter (fused-only ablation). Drop records whose
            # `config` is not in the allow-list from every split. Done
            # BEFORE the per-token record index is consumed so the kept
            # records keep contiguous _r suffixes.
            if include_configs is not None:
                rec_config = str(rec.get("config", "")).strip().lower()
                if rec_config not in include_configs:
                    n_filtered += 1
                    filtered_by_config[rec_config or "(none)"] += 1
                    continue

            ct_rel    = rec.get("ct_file") or rec.get("ct") or ""
            label_rel = rec.get("label_file") or rec.get("label") or ""
            if not ct_rel or not label_rel:
                log.warning("missing ct/label paths token=%s split=%s; skipping",
                            tok, split)
                n_skipped += 1
                continue

            ct_src    = hf_dir / ct_rel
            label_src = hf_dir / label_rel

            if do_link:
                if not ct_src.exists() or not label_src.exists():
                    log.warning("file not found token=%s ct=%s label=%s; skipping",
                                tok, ct_src.exists(), label_src.exists())
                    n_skipped += 1
                    continue

            r_idx = record_idx_by_token[tok]
            record_idx_by_token[tok] += 1
            case_id = _case_id(tok, r_idx)

            if split == "test":
                test_case_ids.append(case_id)
                if do_link:
                    ct_dst    = images_ts / f"{case_id}_0000.nii.gz"
                    label_dst = labels_ts / f"{case_id}.nii.gz"
            else:
                train_case_ids.append(case_id)
                if do_link:
                    ct_dst    = images_tr / f"{case_id}_0000.nii.gz"
                    label_dst = labels_tr / f"{case_id}.nii.gz"

            if do_link:
                try:
                    # CT image: symlink/copy unchanged.
                    _link_or_copy(ct_src, ct_dst, use_symlinks)
                    # Label: remap-and-write if a remap LUT is
                    # configured; otherwise symlink/copy through.
                    if label_remap_lut is not None:
                        _remap_and_write_label(label_src, label_dst,
                                                label_remap_lut)
                        remap_stats["n_remapped"] += 1
                    else:
                        _link_or_copy(label_src, label_dst, use_symlinks)
                        remap_stats["n_passthrough"] += 1
                except Exception as e:
                    log.warning("link/copy/remap failed token=%s: %s", tok, e)
                    n_skipped += 1
                    if split == "test":
                        test_case_ids.pop()
                    else:
                        train_case_ids.pop()
                    continue

            patient_subtype_raw = splits_subtypes.get(tok)
            if patient_subtype_raw is None:
                subtype = _resolve_subtype_for_case(rec, splits_subtypes)
            else:
                rec_config = str(rec.get("config", ""))
                subtype = _refine_subtype_for_record_config(
                    patient_subtype_raw, rec_config)
                if subtype != patient_subtype_raw:
                    key = f"{patient_subtype_raw}->normal (config={rec_config})"
                    refinement_stats[key] += 1

            case_to_subtype[case_id] = subtype
            case_to_token[case_id]   = tok
            case_to_attrs[case_id]   = {
                "n_lumbar_labels": int(rec.get("n_lumbar_labels", 0) or 0),
                "has_l6":          bool(rec.get("has_l6", False)),
                "lstv_label":      str(rec.get("lstv_label", "")),
                "lstv_pelvic":     str(rec.get("lstv_pelvic", "")),
                "lstv_vertebral":  str(rec.get("lstv_vertebral", "")),
                "lstv_class":      int(rec.get("lstv_class", 0) or 0),
                "match_type":      str(rec.get("match_type", "")),
                "config":          str(rec.get("config", "")),
                "split":           split,
                "partial_annotation": bool(rec.get("partial_annotation", False)),
                "patient_subtype_raw": patient_subtype_raw or "(none)",
            }

    return (case_to_subtype, case_to_token, case_to_attrs,
            train_case_ids, test_case_ids, n_skipped,
            dict(refinement_stats), dict(remap_stats),
            n_filtered, dict(filtered_by_config))


def _write_splits_final(
    folds: List[Dict],
    case_to_token: Dict[str, str],
    train_case_ids: List[str],
    out_path: Path,
) -> List[Dict]:
    """Mirror splits_5fold.json folds onto nnU-Net case_ids."""
    token_to_caseids: Dict[str, List[str]] = defaultdict(list)
    train_set = set(train_case_ids)
    for case_id, tok in case_to_token.items():
        if case_id in train_set:
            token_to_caseids[tok].append(case_id)

    nnunet_splits = []
    for fold in folds:
        train_toks = fold.get("train", [])
        val_toks   = fold.get("val", [])
        train_caseids = []
        val_caseids   = []
        for t in train_toks:
            train_caseids.extend(token_to_caseids.get(t, []))
        for t in val_toks:
            val_caseids.extend(token_to_caseids.get(t, []))
        nnunet_splits.append({
            "train": sorted(train_caseids),
            "val":   sorted(val_caseids),
        })

    out_path.write_text(json.dumps(nnunet_splits, indent=2))
    log.info("Wrote splits_final.json: %d folds -> %s", len(nnunet_splits), out_path)
    for i, sf in enumerate(nnunet_splits):
        log.info("  fold %d: train=%d  val=%d", i, len(sf["train"]), len(sf["val"]))
    return nnunet_splits


def _write_lstv_cases(
    case_to_subtype: Dict[str, str],
    case_to_token:   Dict[str, str],
    case_to_attrs:   Dict[str, Dict],
    train_case_ids:  List[str],
    test_case_ids:   List[str],
    splits_schema:   int,
    out_path:        Path,
) -> Dict:
    """Write lstv_cases.json schema v3 (6-way subtypes)."""
    train_set = set(train_case_ids)
    subtype_counts = Counter(case_to_subtype.values())
    lstv_oversample_pool = sorted(
        cid for cid, st in case_to_subtype.items()
        if st != "normal" and cid in train_set
    )

    lstv_cases = {
        "schema_version":      LSTV_CASES_SCHEMA,
        "splits_schema":       splits_schema,
        "subtypes":            list(SUBTYPES),
        "subtype_counts":      dict(subtype_counts),
        "n_train_cases":       len(train_case_ids),
        "n_test_cases":        len(test_case_ids),
        "case_to_subtype":     case_to_subtype,
        "case_to_token":       case_to_token,
        "case_to_attrs":       case_to_attrs,
        "lstv_oversample_pool": lstv_oversample_pool,
    }
    out_path.write_text(json.dumps(lstv_cases, indent=2, default=str))
    log.info("Wrote lstv_cases.json (schema v%d) -> %s", LSTV_CASES_SCHEMA, out_path)
    log.info("  subtype counts:")
    for st in SUBTYPES:
        log.info("    %-20s %d", st, subtype_counts.get(st, 0))
    log.info("  oversample pool size: %d (non-normal training cases)",
             len(lstv_oversample_pool))
    return lstv_cases


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf_dir",   required=True, type=Path)
    p.add_argument("--splits",   required=True, type=Path)
    p.add_argument("--nnunet_raw", required=True, type=Path)
    p.add_argument("--dataset_id",   type=int, default=803,
                    help="Dataset ID. 803 = merged-label variant (default); "
                         "802 = unmerged baseline.")
    p.add_argument("--dataset_name", default="SpineSurgCTFullMerged",
                    help="Dataset name. SpineSurgCTFullMerged (default) for "
                         "the merged-label variant; SpineSurgCTFull for the "
                         "unmerged baseline.")
    p.add_argument("--symlinks", action="store_true",
                    help="Symlink CT images instead of copying. Note: label "
                         "files are ALWAYS physically rewritten when a label "
                         "remap is configured (LABEL_REMAP_AT_CONVERT non-empty).")
    p.add_argument("--no_remap", action="store_true",
                    help="Disable the L6->last_lumbar label remap. Use this "
                         "to reproduce the legacy unmerged 10-class scheme on "
                         "Dataset802. Default: remap is APPLIED.")
    p.add_argument("--include_configs", default="",
                    help="Comma-separated list of record `config` values to "
                         "KEEP (e.g. 'fused'). Records whose config is not "
                         "listed are dropped from ALL splits (train/val/test). "
                         "Empty (default) keeps every record. Use 'fused' for "
                         "the fused-only ablation.")
    p.add_argument("--drop_ignore_label", action="store_true",
                    help="Build the label scheme WITHOUT the ignore class and "
                         "remap any source ignore voxels (10) to background "
                         "(0). For the fused-only ablation, where records are "
                         "fully annotated and carry no ignore voxels. Keeps the "
                         "L6->last_lumbar merge. Incompatible with --no_remap.")
    p.add_argument("--regen_splits_only", action="store_true")
    args = p.parse_args()

    if args.drop_ignore_label and args.no_remap:
        log.error("--drop_ignore_label is incompatible with --no_remap "
                  "(no-ignore is defined only on the merged contiguous scheme).")
        raise SystemExit(2)

    include_configs: Optional[frozenset] = None
    if args.include_configs.strip():
        include_configs = frozenset(
            c.strip().lower() for c in args.include_configs.split(",") if c.strip()
        )
        unknown = include_configs - set(_VALID_CONFIGS)
        if unknown:
            log.warning("--include_configs lists unrecognized config(s) %s; "
                        "valid configs are %s. Records with those configs "
                        "(if any) will be kept; typos silently keep nothing.",
                        sorted(unknown), list(_VALID_CONFIGS))
        log.info("Config filter ACTIVE: keeping only records with config in %s",
                 sorted(include_configs))

    if not args.hf_dir.exists():
        log.error("HF dir not found: %s", args.hf_dir)
        raise SystemExit(1)
    if not args.splits.exists():
        log.error("splits file not found: %s", args.splits)
        raise SystemExit(1)

    splits = json.loads(args.splits.read_text())
    splits_schema = splits.get("schema_version", 0)
    if splits_schema < 6:
        log.warning("splits_5fold.json schema_version=%d (expected >= 6); "
                    "lstv_cases.json subtype assignments may be inconsistent.",
                    splits_schema)

    splits_subtypes: Dict[str, str] = splits.get("patient_subtypes", {})
    log.info("Splits: %d patients, %d with subtype assigned",
             splits.get("n_patients", 0), len(splits_subtypes))

    manifests = _load_manifest_records(args.hf_dir)
    n_total_records = sum(len(v) for v in manifests.values())
    log.info("Manifests: %d total records across train/val/test", n_total_records)

    # Resolve label remap config and build LUT.
    if args.no_remap or not LABEL_REMAP_AT_CONVERT:
        label_remap_lut = None
        active_label_names = {
            "background":  0,
            "L1":          1,
            "L2":          2,
            "L3":          3,
            "L4":          4,
            "L5":          5,
            "L6":          6,
            "sacrum":      7,
            "left_hip":    8,
            "right_hip":   9,
            "ignore":      10,
        }
        log.info("Label remap: DISABLED (legacy 10-class scheme; --no_remap)")
        log.info("  -> output dataset will have non-contiguous labels and is "
                 "intended only as a reference baseline; do NOT train against "
                 "it with the v20+ trainer (the trainer assumes contiguous "
                 "label values).")
    elif args.drop_ignore_label:
        label_remap_lut = _build_remap_lut(LABEL_REMAP_NO_IGNORE)
        active_label_names = LABEL_NAMES_NO_IGNORE
        log.info("Label remap (NO-IGNORE variant): %s", dict(LABEL_REMAP_NO_IGNORE))
        log.info("  -> 9-key scheme (8 fg + bg), NO ignore class. Source "
                 "ignore (10) -> background (0) defensively. Intended for the "
                 "fused-only ablation (fused records carry no ignore voxels).")
        if include_configs is None:
            log.warning("--drop_ignore_label set WITHOUT --include_configs: "
                        "any partial-annotation (spine_only/pelvic_native) "
                        "records present will have their ignore voxels folded "
                        "into BACKGROUND, which falsely supervises unannotated "
                        "regions. Pass --include_configs fused for the ablation.")
    else:
        label_remap_lut = _build_remap_lut(LABEL_REMAP_AT_CONVERT)
        active_label_names = LABEL_NAMES
        log.info("Label remap at convert time: %s", dict(LABEL_REMAP_AT_CONVERT))
        log.info("  -> source labels are PHYSICALLY REWRITTEN; --symlinks "
                 "applies to CT images only.")

    ds_dir = args.nnunet_raw / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"

    do_link = not args.regen_splits_only
    if do_link:
        images_tr = ds_dir / "imagesTr"
        labels_tr = ds_dir / "labelsTr"
        images_ts = ds_dir / "imagesTs"
        labels_ts = ds_dir / "labelsTs"
        for d in (images_tr, labels_tr, images_ts, labels_ts):
            d.mkdir(parents=True, exist_ok=True)
        log.info("Mode: full build (link/copy + dataset.json + splits + lstv_cases)")
        log.info("Output: %s", ds_dir)
    else:
        if not ds_dir.exists():
            log.error("--regen_splits_only requires existing dataset dir: %s", ds_dir)
            log.error("       Run without --regen_splits_only first to build the dataset.")
            raise SystemExit(1)
        images_tr = labels_tr = images_ts = labels_ts = None
        log.info("Mode: regen_splits_only (refresh splits_final.json + lstv_cases.json only)")
        log.info("Dataset dir: %s (existing, not modified)", ds_dir)

    (case_to_subtype, case_to_token, case_to_attrs,
     train_case_ids, test_case_ids, n_skipped,
     refinement_stats, remap_stats,
     n_filtered, filtered_by_config) = _build_case_indices(
        manifests, splits_subtypes, args.hf_dir,
        images_tr, labels_tr, images_ts, labels_ts,
        use_symlinks=args.symlinks, do_link=do_link,
        label_remap_lut=label_remap_lut,
        include_configs=include_configs,
    )

    if include_configs is not None:
        log.info("Config filter dropped %d record(s): %s",
                 n_filtered, filtered_by_config or "{}")
        log.info("  kept configs: %s", sorted(include_configs))

    if do_link:
        log.info("Linked/copied: train=%d  test=%d  skipped=%d",
                 len(train_case_ids), len(test_case_ids), n_skipped)
        if remap_stats:
            log.info("Label file disposition:")
            for k, n in sorted(remap_stats.items()):
                log.info("  %-20s %d", k, n)
    else:
        log.info("Indexed: train=%d  test=%d  skipped=%d  (no link/copy in regen mode)",
                 len(train_case_ids), len(test_case_ids), n_skipped)

    if refinement_stats:
        log.info("Per-record subtype refinements (anatomical-region demotions):")
        for transition, n in sorted(refinement_stats.items()):
            log.info("  %-50s %d", transition, n)
    else:
        log.info("Per-record subtype refinements: none "
                 "(no spine_only / pelvic_native records of LSTV patients, "
                 "or `config` field missing)")

    # ── dataset.json (skipped in regen mode) ───────────────────────────────
    if do_link:
        if args.drop_ignore_label:
            description = (
                "CTSpinoPelvic1K — FUSED-ONLY ablation, 9-key spine + pelvis "
                "CT (8 fg + bg) with merged last_lumbar (L5+L6) and NO ignore "
                "label. Training/test pool restricted to fully-annotated "
                "fused records; the partial-annotation ignore protocol is "
                "disabled. L6 source labels remapped to last_lumbar (5); "
                "sacrum/hips shifted down for contiguous label values. "
                "Reviewer ablation: fused-only vs all-cases, evaluated on the "
                "fused test split. Recover individual L5/L6 via instance "
                "post-processing or VERIDAH (Möller 2026)."
            )
            release = (
                "May 2026 — fused-only ablation (no ignore protocol) / "
                "schema v6 splits / v3 lstv_cases / L6 merged into "
                "last_lumbar following Möller 2026 (VERIDAH §2.2) / "
                "contiguous label IDs"
            )
        elif label_remap_lut is not None:
            description = (
                "CTSpinoPelvic1K — fused 9-class spine + pelvis CT (8 fg + bg) "
                "with merged last_lumbar (L5+L6) and ignore label (9) for "
                "partial annotations. L6 source labels remapped to "
                "last_lumbar (5) at convert time; sacrum/hips/ignore "
                "shifted down by one for contiguous label values. "
                "Recover individual L5/L6 via instance post-processing or "
                "VERIDAH (Möller 2026)."
            )
            release = (
                "May 2026 — schema v6 splits / v3 lstv_cases / "
                "partial-annotation ignore label / per-record subtype "
                "refinement / L6 merged into last_lumbar following "
                "Möller 2026 (VERIDAH §2.2) / contiguous label IDs"
            )
        else:
            description = (
                "CTSpinoPelvic1K — fused 10-class spine + pelvis CT "
                "with ignore label (10) for partial annotations "
                "(LEGACY UNMERGED VARIANT)"
            )
            release = (
                "May 2026 — schema v6 splits / v3 lstv_cases / "
                "partial-annotation ignore label / per-record subtype "
                "refinement (legacy 10-class)"
            )
        dataset_json = {
            "channel_names": {"0": "CT"},
            "labels":        active_label_names,
            "numTraining":   len(train_case_ids),
            "file_ending":   ".nii.gz",
            "name":          args.dataset_name,
            "description":   description,
            "reference":     "CTSpine1K + CTPelvic1K, COLONOG cohort",
            "release":       release,
        }
        (ds_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2))
        n_fg = sum(1 for k in active_label_names
                   if k not in ("background", "ignore"))
        has_ignore = "ignore" in active_label_names
        log.info("Wrote dataset.json with %d label keys (%d foreground + bg%s)",
                 len(active_label_names), n_fg,
                 " + ignore" if has_ignore else "; NO ignore class")

    # ── splits_final.json + lstv_cases.json (always written) ───────────────
    _write_splits_final(
        splits.get("folds", []),
        case_to_token, train_case_ids,
        ds_dir / "splits_final.json",
    )
    _write_lstv_cases(
        case_to_subtype, case_to_token, case_to_attrs,
        train_case_ids, test_case_ids, splits_schema,
        ds_dir / "lstv_cases.json",
    )

    log.info("=" * 70)
    log.info("Done. Dataset at: %s", ds_dir)
    log.info("=" * 70)


if __name__ == "__main__":
    main()
