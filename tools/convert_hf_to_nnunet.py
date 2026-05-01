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
    labelsTr/{caseID}.nii.gz        (symlinks or copies of label)
    imagesTs/{caseID}_0000.nii.gz   (test set)
    labelsTs/{caseID}.nii.gz
    dataset.json                    (10-class label scheme + ignore)
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

dataset.json label scheme (May 2026 update)
-------------------------------------------
  0 background  1 L1  2 L2  3 L3  4 L4  5 L5  6 L6  7 sacrum  8 left_hip  9 right_hip
  10 ignore (NEW)

Why "ignore" is here:
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
    - NOT allocating an output head for class 10 (network output stays
      at 10 channels: bg + 9 fg)

  Without this entry, partial-annotation labels would falsely supervise
  the network with "background" wherever the partial annotator did not
  trace, poisoning the loss for L6 / sacrum / hips on roughly two-thirds
  of the training set. See export_hf.py "PARTIAL ANNOTATION CONTRACT".

lstv_cases.json schema v3 (Apr 2026)
====================================
6-way LSTV subtype taxonomy mirroring splits_5fold.json schema v6:
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

# 10-class scheme + ignore label for partial-annotation cases.
# nnU-Net v2 treats the literal key "ignore" specially — see module
# docstring for full semantics. The value 10 must match the value
# written by export_hf.merge_labels() in partial-annotation mode and
# the trainer's _IGNORE_LABEL constant.
LABEL_NAMES = {
    "background": 0,
    "L1":         1,
    "L2":         2,
    "L3":         3,
    "L4":         4,
    "L5":         5,
    "L6":         6,
    "sacrum":     7,
    "left_hip":   8,
    "right_hip":  9,
    "ignore":     10,
}


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
    """
    Use the splits-derived subtype if available (canonical 6-way taxonomy).
    Fall back to per-record resolution if a token isn't in splits.
    """
    tok = str(record.get("token") or record.get("patient_token") or "")
    if tok in splits_subtypes:
        return splits_subtypes[tok]

    label = str(record.get("lstv_label", "")).upper()
    pel   = str(record.get("lstv_pelvic", "")).upper()
    vert  = str(record.get("lstv_vertebral", "")).upper()
    n_lumb = int(record.get("n_lumbar_labels", 0) or 0)
    has_l6 = bool(record.get("has_l6", False))

    if label == "AMBIGUOUS":
        return "ambiguous"
    if (vert == "LUMBARIZATION" and pel == "SACRALIZATION") or \
       (vert == "SACRALIZATION" and pel == "LUMBARIZATION"):
        return "ambiguous"
    if n_lumb == 6 or has_l6 or label == "LUMBARIZATION":
        return "lumb"
    if label in ("SEMI_SACRAL", "SEMI_SACRALIZATION") or \
       pel in ("SEMI_SACRAL", "SEMI_SACRALIZATION"):
        return "semisacralization"
    if vert == "SACRALIZATION" and n_lumb == 4:
        return "sacr_count"
    if label == "SACRALIZATION" or pel == "SACRALIZATION":
        return "sacralization"
    return "normal"


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
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Dict],
           List[str], List[str], int]:
    """
    Iterate manifests, optionally symlink/copy CT+label files, and build
    the case_id -> {subtype, token, attrs} indices.
    """
    case_to_subtype: Dict[str, str] = {}
    case_to_token:   Dict[str, str] = {}
    case_to_attrs:   Dict[str, Dict] = {}
    train_case_ids: List[str] = []
    test_case_ids:  List[str] = []
    n_skipped = 0

    record_idx_by_token: Dict[str, int] = defaultdict(int)

    for split, recs in manifests.items():
        for rec in recs:
            tok = str(rec.get("token") or rec.get("patient_token") or "")
            if not tok:
                n_skipped += 1
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
                    _link_or_copy(ct_src,    ct_dst,    use_symlinks)
                    _link_or_copy(label_src, label_dst, use_symlinks)
                except Exception as e:
                    log.warning("link/copy failed token=%s: %s", tok, e)
                    n_skipped += 1
                    if split == "test":
                        test_case_ids.pop()
                    else:
                        train_case_ids.pop()
                    continue

            subtype = _resolve_subtype_for_case(rec, splits_subtypes)
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
                # NEW: track partial-annotation status from manifest if present.
                "partial_annotation": bool(rec.get("partial_annotation", False)),
            }

    return (case_to_subtype, case_to_token, case_to_attrs,
            train_case_ids, test_case_ids, n_skipped)


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
    p.add_argument("--dataset_id",   type=int, default=802)
    p.add_argument("--dataset_name", default="SpineSurgCTFull")
    p.add_argument("--symlinks", action="store_true")
    p.add_argument("--regen_splits_only", action="store_true")
    args = p.parse_args()

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
     train_case_ids, test_case_ids, n_skipped) = _build_case_indices(
        manifests, splits_subtypes, args.hf_dir,
        images_tr, labels_tr, images_ts, labels_ts,
        use_symlinks=args.symlinks, do_link=do_link,
    )

    if do_link:
        log.info("Linked/copied: train=%d  test=%d  skipped=%d",
                 len(train_case_ids), len(test_case_ids), n_skipped)
    else:
        log.info("Indexed: train=%d  test=%d  skipped=%d  (no link/copy in regen mode)",
                 len(train_case_ids), len(test_case_ids), n_skipped)

    # ── dataset.json (skipped in regen mode) ───────────────────────────────
    if do_link:
        dataset_json = {
            "channel_names": {"0": "CT"},
            "labels":        LABEL_NAMES,
            "numTraining":   len(train_case_ids),
            "file_ending":   ".nii.gz",
            "name":          args.dataset_name,
            "description":   "CTSpinoPelvic1K — fused 10-class spine + pelvis CT "
                              "with ignore label (10) for partial annotations",
            "reference":     "CTSpine1K + CTPelvic1K, COLONOG cohort",
            "release":       "May 2026 — schema v6 splits / v3 lstv_cases / "
                              "partial-annotation ignore label",
        }
        (ds_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2))
        log.info("Wrote dataset.json (labels include ignore=10 for partial-annotation cases)")

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
