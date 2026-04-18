#!/usr/bin/env python3
"""
SpineSurg-CT -- HF export -> nnU-Net v2 dataset format
tools/convert_hf_to_nnunet.py

v5 change: --splits_file (token-level) is the source of truth
==============================================================
splits_final.json is now BUILT FROM an upstream, token-level 5-fold CV
splits file (scripts/generate_5fold_splits.py -> data/splits_5fold.json).
The HF export's manifest_train/val/test distinction is IGNORED for
partitioning; all records are pooled, and each record's patient token
is looked up in the splits file to decide:
    - imagesTs/labelsTs   if token is in splits.test_tokens
    - imagesTr/labelsTr   otherwise; fold assignment per splits.folds[*]

This means the splits file is the single source of truth for both
test holdout AND 5-fold CV. Rerunning the upstream splits generator
(e.g. after adding new cases to placed_manifest) and then rerunning
this script propagates cleanly through.

Flags:
    --splits_file PATH       Required for K-fold mode.
    --single_fold_splits     Legacy: ignore --splits_file; write 5
                             identical folds matching the HF train/val
                             manifest split. Useful if you want fold 0
                             to equal the HF val set exactly.
    --regen_splits_only      If dataset already built, only rewrite
                             splits_final.json from --splits_file.

Fixes carried forward from earlier versions
===========================================
1. Cross-manifest case_id collisions. Namespaced {safe_token}__{config},
   shared seen_ids across train+val manifests.
2. Partial-annotation support via ignore_label = 10 (nnU-Net v2
   contiguous-label constraint; NOT 255).

PARTIAL-ANNOTATION SUPPORT
==========================
    fused         -> label copied verbatim (all 10 classes valid)
    spine_only    -> background (0) voxels REWRITTEN to 10 (ignore)
    pelvic_native -> background (0) voxels REWRITTEN to 10 (ignore)

dataset.json declares "labels": {..., "ignore": 10}. Test set is fused
only (unless --test_all_configs).

Output layout
-------------
    nnUNet_raw/Dataset{ID}_{Name}/
        imagesTr/{case_id}_0000.nii.gz
        labelsTr/{case_id}.nii.gz
        imagesTs/{case_id}_0000.nii.gz
        labelsTs/{case_id}.nii.gz
        dataset.json
        splits_final.json

Author: Gregory Schwing, MD-PhD  |  Wayne State University / DMC
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import nibabel as nib


# =============================================================================
# Label map / constants
# =============================================================================

CLASS_NAMES: Dict[str, int] = {
    "background": 0,
    "L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5, "L6": 6,
    "sacrum": 7, "left_hip": 8, "right_hip": 9,
    "ignore": 10,
}

IGNORE_LABEL   = 10
SPINE_CLASSES  = {1, 2, 3, 4, 5, 6}
PELVIC_CLASSES = {7, 8, 9}

VALID_CONFIGS  = ("fused", "spine_only", "pelvic_native")


# =============================================================================
# Case ID utilities
# =============================================================================

def safe_id(token: str) -> str:
    s = str(token)
    for bad in (".", "/", "\\", " ", ":", "(", ")"):
        s = s.replace(bad, "_")
    s = s.strip("_")
    if not s:
        raise ValueError(f"Empty case_id after sanitizing: '{token}'")
    return s


def make_case_id(token: str, config: str, seen_ids: Set[str]) -> str:
    """Namespaced {safe_token}__{config}; disambiguate collisions with __dupN."""
    base = f"{safe_id(token)}__{config}"
    if base not in seen_ids:
        seen_ids.add(base)
        return base
    for suffix in range(1, 101):
        candidate = f"{base}__dup{suffix}"
        if candidate not in seen_ids:
            seen_ids.add(candidate)
            print(f"    NOTE: duplicate token+config, using {candidate}",
                  file=sys.stderr)
            return candidate
    raise RuntimeError(
        f"More than 100 duplicates of {base!r}. Something is very wrong "
        f"with the manifest."
    )


# =============================================================================
# File ops
# =============================================================================

def link(src: Path, dst: Path, *, allow_overwrite: bool = False) -> None:
    if dst.exists() or dst.is_symlink():
        if not allow_overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing {dst} (src was {src}).")
        dst.unlink()
    dst.symlink_to(src.resolve())


def rewrite_label(src_path: Path, dst_path: Path, config: str) -> Dict:
    """Read label, rewrite background->IGNORE for partial configs, write to dst."""
    if dst_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing {dst_path} (src was {src_path}).")

    img = nib.load(str(src_path))
    arr = np.asarray(img.get_fdata()).astype(np.int16)

    before = {
        "bg":     int((arr == 0).sum()),
        "spine":  int(np.isin(arr, list(SPINE_CLASSES)).sum()),
        "pelvic": int(np.isin(arr, list(PELVIC_CLASSES)).sum()),
    }

    if config == "fused":
        pass
    elif config in ("spine_only", "pelvic_native"):
        arr = np.where(arr == 0, IGNORE_LABEL, arr).astype(np.int16)
    else:
        raise ValueError(f"Unknown config: {config}")

    after = {
        "bg":     int((arr == 0).sum()),
        "spine":  int(np.isin(arr, list(SPINE_CLASSES)).sum()),
        "pelvic": int(np.isin(arr, list(PELVIC_CLASSES)).sum()),
        "ignore": int((arr == IGNORE_LABEL).sum()),
    }

    new_header = img.header.copy()
    new_header.set_data_dtype(np.int16)
    out_img = nib.Nifti1Image(arr, img.affine, header=new_header)
    out_img.set_data_dtype(np.int16)
    nib.save(out_img, str(dst_path))

    return {"before": before, "after": after}


# =============================================================================
# Manifest parsing
# =============================================================================

def stratify_debug(raw: List[Dict], debug_n: int) -> List[Dict]:
    if debug_n <= 0 or not raw:
        return raw
    by_cfg: Dict[str, List[Dict]] = defaultdict(list)
    for r in raw:
        by_cfg[r.get("config", "fused")].append(r)
    n_cfgs = len(by_cfg)
    per_cfg = max(1, debug_n // n_cfgs)
    picked: List[Dict] = []
    for cfg in sorted(by_cfg.keys()):
        picked.extend(by_cfg[cfg][:per_cfg])
    picked = picked[:debug_n]
    breakdown_parts = []
    for cfg in sorted(by_cfg.keys()):
        n_cfg = sum(1 for r in picked if r.get("config", "fused") == cfg)
        breakdown_parts.append(f"{cfg}={n_cfg}")
    breakdown = ", ".join(breakdown_parts)
    print(f"    [debug_n={debug_n}] kept {len(picked)} cases  ({breakdown})")
    return picked


def parse_manifest(
    manifest_path: Path,
    ct_dir: Path,
    lbl_dir: Path,
    fused_only: bool,
    seen_ids: Set[str],
    debug_n: int = 0,
) -> List[Dict]:
    if not manifest_path.exists():
        print(f"  WARN: {manifest_path} not found", file=sys.stderr)
        return []

    raw = json.loads(manifest_path.read_text())
    raw = stratify_debug(raw, debug_n)

    out: List[Dict] = []
    skipped_cfg = skipped_missing = 0

    for r in raw:
        config = r.get("config", "fused")
        if config not in VALID_CONFIGS:
            raise ValueError(f"Unknown config in manifest: {config!r}")
        if fused_only and config != "fused":
            skipped_cfg += 1
            continue

        ct  = ct_dir  / r.get("ct_file",    "")
        lbl = lbl_dir / r.get("label_file", "")
        if not ct.exists() or not lbl.exists():
            skipped_missing += 1
            continue

        token = r.get("token") or Path(r["ct_file"]).stem.replace(".nii", "")
        cid = make_case_id(token, config, seen_ids)

        out.append({
            "case_id":    cid,
            "token":      str(token),
            "ct":         ct,
            "label":      lbl,
            "config":     config,
            "lstv_label": r.get("lstv_label", ""),
            "is_pseudo":  bool(r.get("is_pseudo", False)),
        })

    print(f"  {manifest_path.name}: {len(out)} kept  "
          f"(skipped: cfg={skipped_cfg} missing={skipped_missing})")
    return out


# =============================================================================
# Splits file loading (NEW)
# =============================================================================

class SplitsLookup:
    """Token -> routing + fold assignment, from upstream splits_5fold.json."""

    def __init__(self, splits_data: Dict):
        self.data = splits_data
        self.n_folds = int(splits_data["n_folds"])
        self.test_tokens: Set[str] = set(str(t) for t in splits_data["test_tokens"])
        # token -> val fold index (0..n_folds-1), for non-test tokens only
        self.token_to_val_fold: Dict[str, int] = {}
        for i, f in enumerate(splits_data["folds"]):
            for t in f.get("val_tokens", []):
                ts = str(t)
                if ts in self.token_to_val_fold:
                    raise RuntimeError(
                        f"Token {ts!r} appears in multiple fold val sets "
                        f"(folds {self.token_to_val_fold[ts]} and {i}). "
                        f"Splits file is malformed.")
                self.token_to_val_fold[ts] = i
        # Per-token info (schema v3+). May be empty for older splits files.
        self.token_info: Dict[str, Dict] = {
            str(k): v for k, v in (splits_data.get("token_info") or {}).items()
        }

    def is_test(self, token: str) -> bool:
        return str(token) in self.test_tokens

    def val_fold_for(self, token: str) -> Optional[int]:
        """Fold index whose VAL set contains this token, or None if token
        isn't in any fold's val set (shouldn't happen for real tokens)."""
        return self.token_to_val_fold.get(str(token))

    def known_tokens(self) -> Set[str]:
        return set(self.test_tokens) | set(self.token_to_val_fold.keys())

    def match_type_for(self, token: str) -> Optional[str]:
        """match_type from placed_manifest.json, via token_info. None if
        the splits file is schema v2 (no token_info) or the token is unknown."""
        info = self.token_info.get(str(token))
        if not info:
            return None
        mt = info.get("match_type")
        return str(mt).lower() if mt else None

    def has_token_info(self) -> bool:
        return bool(self.token_info)


def load_splits_file(path: Path) -> SplitsLookup:
    if not path.exists():
        raise FileNotFoundError(
            f"--splits_file {path} not found. Run sbatch slurm/generate_splits.sh first.")
    data = json.loads(path.read_text())
    schema = data.get("schema_version", 0)
    if schema < 2:
        raise ValueError(
            f"{path} has schema_version={schema}; need >= 2. "
            f"Regenerate with scripts/generate_5fold_splits.py.")
    if "folds" not in data or not data["folds"]:
        raise ValueError(f"{path} has no folds.")
    if "test_tokens" not in data:
        raise ValueError(f"{path} missing test_tokens.")
    return SplitsLookup(data)


# =============================================================================
# splits_final.json writer
# =============================================================================

def write_splits_final_from_lookup(
        ds_dir: Path,
        train_recs: List[Dict],
        lookup: SplitsLookup) -> None:
    """
    Build nnU-Net's splits_final.json from an upstream token-level splits
    file. All train_recs are in the training pool (imagesTr); each is
    assigned to exactly one fold's val set based on its token. Records
    whose token isn't in the splits file's train/val pool are placed in
    ALL folds' train sets (never val) -- this is the safe behavior for
    pseudo cases or for tokens missing from the upstream splits.
    """
    n_folds = lookup.n_folds
    # case_ids grouped by their val-fold assignment, plus a "no fold val"
    # bucket for pseudo / unmapped cases.
    val_ids_per_fold: List[List[str]] = [[] for _ in range(n_folds)]
    no_val_ids: List[str] = []
    for r in train_recs:
        tok = r["token"]
        fi = lookup.val_fold_for(tok)
        if fi is None:
            no_val_ids.append(r["case_id"])
        else:
            val_ids_per_fold[fi].append(r["case_id"])

    if no_val_ids:
        print(f"  NOTE: {len(no_val_ids)} training case(s) have tokens "
              f"not in the splits file's trainval pool. "
              f"They will appear in every fold's TRAIN set and no fold's VAL set.")

    all_train_ids = [r["case_id"] for r in train_recs]
    all_train_set = set(all_train_ids)

    splits = []
    for i in range(n_folds):
        val_ids = sorted(set(val_ids_per_fold[i]))
        train_ids = sorted(all_train_set - set(val_ids))
        splits.append({"train": train_ids, "val": val_ids})

    # Sanity: val sets are disjoint, their union is the mapped pool
    all_val_union: Set[str] = set()
    for i, f in enumerate(splits):
        s = set(f["val"])
        if all_val_union & s:
            raise RuntimeError(
                f"Fold val sets not disjoint (fold {i} overlaps prior).")
        all_val_union |= s
    print(f"  fold val sizes: {[len(f['val']) for f in splits]}  "
          f"train sizes: {[len(f['train']) for f in splits]}")

    out = ds_dir / "splits_final.json"
    out.write_text(json.dumps(splits, indent=2))
    print(f"  wrote {out}")


def write_splits_final_single_fold(
        ds_dir: Path,
        train_ids: List[str],
        val_ids: List[str]) -> None:
    """Legacy behavior: 5 identical folds = HF train/val split."""
    fold = {"train": sorted(train_ids), "val": sorted(val_ids)}
    splits = [fold for _ in range(5)]
    out = ds_dir / "splits_final.json"
    out.write_text(json.dumps(splits, indent=2))
    print(f"  wrote {out} (single-fold replicated)")


def splits_final_looks_kfold(ds_dir: Path) -> bool:
    """
    Returns True iff splits_final.json has distinct val sets across folds
    (the K-fold shape). Used to decide auto-regen behavior.
    """
    p = ds_dir / "splits_final.json"
    if not p.exists():
        return False
    try:
        s = json.loads(p.read_text())
        if not isinstance(s, list) or len(s) < 2:
            return False
        vals = [set(f.get("val", [])) for f in s]
        union = set().union(*vals)
        total = sum(len(v) for v in vals)
        disjoint = total == len(union)
        identical = all(v == vals[0] for v in vals[1:]) and len(vals[0]) > 0
        return disjoint and not identical
    except Exception:
        return False


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert SpineSurg-CT HF export -> nnU-Net v2 dataset "
                    "with upstream-driven 5-fold CV splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--hf_export_dir",   required=True, type=Path)
    ap.add_argument("--nnunet_raw_dir",  required=True, type=Path)
    ap.add_argument("--dataset_id",      type=int, default=802)
    ap.add_argument("--dataset_name",    default="SpineSurgCTFull")

    ap.add_argument("--splits_file", type=Path, default=None,
                    help="Path to upstream splits_5fold.json. Required unless "
                         "--single_fold_splits is set.")
    ap.add_argument("--single_fold_splits", action="store_true", default=False,
                    help="Ignore --splits_file and write 5 identical folds "
                         "matching the HF train/val manifest split.")
    ap.add_argument("--regen_splits_only", action="store_true", default=False,
                    help="Skip file rebuild; only (re)write splits_final.json "
                         "from --splits_file (dataset must already exist).")

    ap.add_argument("--include_train_match_types", default=None,
                    help="ABLATION: comma-separated list of match_types "
                         "(from placed_manifest.json) to INCLUDE in the "
                         "training pool. E.g. 'fused' for fused-only "
                         "training. Test set is UNAFFECTED so metrics remain "
                         "comparable across ablations. Requires splits_file "
                         "with schema v3 (has token_info).")
    ap.add_argument("--test_match_types", default=None,
                    help="ABLATION: comma-separated list of match_types to "
                         "include in the test set (filters splits.test_tokens). "
                         "Default: all test tokens included. Use 'fused' to "
                         "restrict test to gold-standard fused cases only.")

    ap.add_argument("--train_fused_only", action="store_true", default=False,
                    help="Train only on fused cases (legacy).")
    ap.add_argument("--test_all_configs", action="store_true", default=False,
                    help="Include partial configs in test (not recommended).")

    ap.add_argument("--exclude_pseudo",  action="store_true", default=True)
    ap.add_argument("--include_pseudo",  dest="exclude_pseudo", action="store_false")

    ap.add_argument("--debug_n", type=int, default=0,
                    help="If >0, keep only this many cases PER MANIFEST.")

    args = ap.parse_args()

    if not args.single_fold_splits and args.splits_file is None:
        print("ERROR: --splits_file is required unless --single_fold_splits is set.\n"
              "       Run sbatch slurm/generate_splits.sh first.", file=sys.stderr)
        return 1

    # ----- load splits file (if in K-fold mode) -----
    lookup: Optional[SplitsLookup] = None
    if not args.single_fold_splits:
        print(f"Loading splits from {args.splits_file}")
        lookup = load_splits_file(args.splits_file)
        print(f"  n_folds={lookup.n_folds}  "
              f"test_tokens={len(lookup.test_tokens)}  "
              f"trainval_tokens={len(lookup.token_to_val_fold)}")

    # ----- paths -----
    ct_dir  = args.hf_export_dir / "ct"
    lbl_dir = args.hf_export_dir / "labels"
    if not ct_dir.is_dir() or not lbl_dir.is_dir():
        print(f"ERROR: missing {ct_dir} or {lbl_dir}", file=sys.stderr)
        return 1

    ds_dir    = args.nnunet_raw_dir / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"

    # =========================================================================
    # --regen_splits_only branch: rewrite splits_final.json and exit
    # =========================================================================
    if args.regen_splits_only:
        if not (ds_dir / "dataset.json").exists():
            print(f"ERROR: --regen_splits_only but dataset not built "
                  f"({ds_dir}/dataset.json missing).", file=sys.stderr)
            return 1
        # We need train_recs to rewrite splits. Re-parse manifests.
        print(f"\n--regen_splits_only: re-parsing manifests to rewrite splits")
        train_val_seen: Set[str] = set()
        tr = parse_manifest(
            args.hf_export_dir / "manifest_train.json",
            ct_dir, lbl_dir, args.train_fused_only, train_val_seen, args.debug_n)
        vl = parse_manifest(
            args.hf_export_dir / "manifest_validation.json",
            ct_dir, lbl_dir, args.train_fused_only, train_val_seen, args.debug_n)
        # Test set is separate in splits_final.json writing, so we don't
        # include test_recs in train_recs.
        if args.exclude_pseudo:
            tr = [r for r in tr if not r["is_pseudo"]]
            vl = [r for r in vl if not r["is_pseudo"]]
        if args.single_fold_splits:
            write_splits_final_single_fold(
                ds_dir, [r["case_id"] for r in tr], [r["case_id"] for r in vl])
        else:
            assert lookup is not None
            # In K-fold mode, the HF train/val distinction is irrelevant.
            # Pool them, filter out test tokens (those shouldn't be in
            # train+val manifests anyway, but be defensive), and use
            # splits file to partition.
            combined = [r for r in tr + vl if not lookup.is_test(r["token"])]
            removed  = len(tr) + len(vl) - len(combined)
            if removed:
                print(f"  removed {removed} records whose token is in test set")
            write_splits_final_from_lookup(ds_dir, combined, lookup)
        print("Done (splits only).")
        return 0

    # =========================================================================
    # Normal build path
    # =========================================================================
    train_val_seen: Set[str] = set()
    test_seen:      Set[str] = set()

    print("\nReading manifests (TRAIN + VAL):")
    if args.debug_n > 0:
        print(f"  [DEBUG MODE: keeping <= {args.debug_n} cases per manifest]")
    train_recs = parse_manifest(
        args.hf_export_dir / "manifest_train.json",
        ct_dir, lbl_dir, args.train_fused_only, train_val_seen, args.debug_n)
    val_recs = parse_manifest(
        args.hf_export_dir / "manifest_validation.json",
        ct_dir, lbl_dir, args.train_fused_only, train_val_seen, args.debug_n)

    print("Reading manifest (TEST):")
    test_recs = parse_manifest(
        args.hf_export_dir / "manifest_test.json",
        ct_dir, lbl_dir,
        fused_only=not args.test_all_configs,
        seen_ids=test_seen, debug_n=args.debug_n)

    if args.exclude_pseudo:
        before = len(train_recs) + len(val_recs) + len(test_recs)
        train_recs = [r for r in train_recs if not r["is_pseudo"]]
        val_recs   = [r for r in val_recs   if not r["is_pseudo"]]
        test_recs  = [r for r in test_recs  if not r["is_pseudo"]]
        after = len(train_recs) + len(val_recs) + len(test_recs)
        print(f"  excluded pseudo cases: {before - after}")

    if not train_recs and not val_recs:
        print("ERROR: no train+val records after filtering.", file=sys.stderr)
        return 1

    # ----- If in K-fold mode, RE-ROUTE records via the splits file ----------
    # HF's train/val/test partitioning is superseded: splits file decides.
    if lookup is not None:
        pool = train_recs + val_recs + test_recs
        pool_train = [r for r in pool if not lookup.is_test(r["token"])]
        pool_test  = [r for r in pool if     lookup.is_test(r["token"])]

        # ------ Ablation: filter training pool by match_type --------------
        if args.include_train_match_types:
            if not lookup.has_token_info():
                print("ERROR: --include_train_match_types requires splits file "
                      "with token_info (schema v3+). Regenerate with "
                      "scripts/generate_5fold_splits.py.", file=sys.stderr)
                return 1
            allowed = {mt.strip().lower()
                       for mt in args.include_train_match_types.split(",")
                       if mt.strip()}
            before = len(pool_train)
            kept, n_excluded, n_unknown = [], 0, 0
            excluded_counts = defaultdict(int)
            for r in pool_train:
                mt = lookup.match_type_for(r["token"])
                if mt is None:
                    n_unknown += 1
                    continue  # unknown match_type -> exclude (safer than guess)
                if mt in allowed:
                    kept.append(r)
                else:
                    n_excluded += 1
                    excluded_counts[mt] += 1
            pool_train = kept
            print(f"\n[ABLATION] --include_train_match_types={sorted(allowed)}")
            print(f"  kept    : {len(pool_train)} training records (from {before})")
            print(f"  excluded by match_type: {dict(excluded_counts)} (total {n_excluded})")
            if n_unknown:
                print(f"  excluded (unknown match_type): {n_unknown}")

        # ------ Ablation: filter test set by match_type -------------------
        if args.test_match_types:
            if not lookup.has_token_info():
                print("ERROR: --test_match_types requires splits file with "
                      "token_info (schema v3+).", file=sys.stderr)
                return 1
            allowed = {mt.strip().lower()
                       for mt in args.test_match_types.split(",")
                       if mt.strip()}
            before = len(pool_test)
            kept_test, n_excluded_test = [], 0
            excluded_counts_test = defaultdict(int)
            for r in pool_test:
                mt = lookup.match_type_for(r["token"])
                if mt is None:
                    continue
                if mt in allowed:
                    kept_test.append(r)
                else:
                    n_excluded_test += 1
                    excluded_counts_test[mt] += 1
            pool_test = kept_test
            print(f"\n[ABLATION] --test_match_types={sorted(allowed)}")
            print(f"  kept    : {len(pool_test)} test records (from {before})")
            print(f"  excluded: {dict(excluded_counts_test)} (total {n_excluded_test})")

        # Audit: which HF records have unknown tokens (not in splits file)?
        known = lookup.known_tokens()
        unknown = [r for r in pool_train if r["token"] not in known]
        if unknown:
            print(f"  WARN: {len(unknown)} train-pool records have tokens "
                  f"not in splits file. They go to imagesTr but appear in "
                  f"NO fold's val set. Examples: "
                  f"{sorted({r['token'] for r in unknown})[:5]}")
        # Reassign the canonical train/test for the rest of the script
        train_recs = pool_train
        val_recs   = []               # HF "val" no longer special
        test_recs  = pool_test
        print(f"\nRe-routed by splits file:")
        print(f"  imagesTr pool : {len(train_recs)}")
        print(f"  imagesTs pool : {len(test_recs)}")

    ds_dir    = args.nnunet_raw_dir / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    images_tr = ds_dir / "imagesTr"
    labels_tr = ds_dir / "labelsTr"
    images_ts = ds_dir / "imagesTs"
    labels_ts = ds_dir / "labelsTs"

    # ----- Idempotency: existing dataset dir -----
    if ds_dir.exists():
        has_content = any(ds_dir.iterdir())
        if args.debug_n > 0 and has_content:
            print(f"  [debug_n] wiping existing {ds_dir}")
            shutil.rmtree(ds_dir)
        elif has_content:
            marker = ds_dir / "dataset.json"
            if marker.exists():
                # Dataset already built. Check splits state.
                want_kfold = not args.single_fold_splits
                have_kfold = splits_final_looks_kfold(ds_dir)
                if want_kfold and not have_kfold:
                    print(f"  NOTE: {ds_dir} populated but splits_final.json "
                          f"is single-fold style. Rewriting splits only.")
                    write_splits_final_from_lookup(ds_dir, train_recs, lookup)
                    return 0
                elif not want_kfold and have_kfold:
                    print(f"  NOTE: {ds_dir} populated with K-fold splits "
                          f"but --single_fold_splits requested. Rewriting "
                          f"splits only.")
                    write_splits_final_single_fold(
                        ds_dir,
                        # reconstitute original train/val ids
                        [r["case_id"] for r in parse_manifest(
                            args.hf_export_dir / "manifest_train.json",
                            ct_dir, lbl_dir, args.train_fused_only,
                            set(), args.debug_n)
                         if not r["is_pseudo"] or not args.exclude_pseudo],
                        [r["case_id"] for r in parse_manifest(
                            args.hf_export_dir / "manifest_validation.json",
                            ct_dir, lbl_dir, args.train_fused_only,
                            set(), args.debug_n)
                         if not r["is_pseudo"] or not args.exclude_pseudo],
                    )
                    return 0
                else:
                    print(f"  NOTE: {ds_dir} already populated with "
                          f"{'K-fold' if have_kfold else 'single-fold'} "
                          f"splits. No-op.")
                    print(f"        Delete {ds_dir} to rebuild, or pass "
                          f"--regen_splits_only to refresh splits.")
                    return 0
            raise RuntimeError(
                f"{ds_dir} exists and is non-empty but has no dataset.json. "
                f"Refusing to mix with stale state. Delete it and rerun.")

    for d in (images_tr, labels_tr, images_ts, labels_ts):
        d.mkdir(parents=True, exist_ok=True)

    # ----- write imagesTr/labelsTr ----------------------------------------
    combined = train_recs + val_recs   # val_recs is [] in K-fold mode

    cfg_counter = {"fused": 0, "spine_only": 0, "pelvic_native": 0}
    n_rewritten = 0
    t_rewrite   = 0.0
    wall_start  = time.time()

    print(f"\nWriting {len(combined)} train/val cases to {ds_dir}")
    for i, rec in enumerate(combined, 1):
        cid    = rec["case_id"]
        config = rec["config"]
        cfg_counter[config] = cfg_counter.get(config, 0) + 1

        link(rec["ct"], images_tr / f"{cid}_0000.nii.gz")

        dst_lbl = labels_tr / f"{cid}.nii.gz"
        t0 = time.time()
        if config == "fused":
            if dst_lbl.exists():
                raise FileExistsError(f"collision at {dst_lbl}")
            shutil.copy2(rec["label"], dst_lbl)
        else:
            rewrite_label(rec["label"], dst_lbl, config)
            n_rewritten += 1
        t_rewrite += time.time() - t0

        if i % 50 == 0 or i == len(combined):
            wall = time.time() - wall_start
            print(f"  [{i:4d}/{len(combined)}]  "
                  f"fused={cfg_counter['fused']} "
                  f"spine_only={cfg_counter['spine_only']} "
                  f"pelvic_native={cfg_counter['pelvic_native']}  "
                  f"t_io={t_rewrite:.0f}s  wall={wall:.0f}s")

    print(f"\nRewrote {n_rewritten} partial-annotation labels in {t_rewrite:.0f}s")

    # ----- post-write verification ----------------------------------------
    n_img_tr = len(list(images_tr.glob("*_0000.nii.gz")))
    n_lbl_tr = len(list(labels_tr.glob("*.nii.gz")))
    if n_img_tr != len(combined) or n_lbl_tr != len(combined):
        raise RuntimeError(
            f"Disk state mismatch! combined={len(combined)} "
            f"imagesTr={n_img_tr} labelsTr={n_lbl_tr}")
    print(f"  verified: imagesTr={n_img_tr} labelsTr={n_lbl_tr}")

    # ----- write imagesTs/labelsTs ----------------------------------------
    print(f"\nWriting {len(test_recs)} test cases to {images_ts}")
    for rec in test_recs:
        cid = rec["case_id"]
        link(rec["ct"], images_ts / f"{cid}_0000.nii.gz")
        if rec["config"] == "fused":
            link(rec["label"], labels_ts / f"{cid}.nii.gz")
        else:
            rewrite_label(rec["label"], labels_ts / f"{cid}.nii.gz", rec["config"])

    n_img_ts = len(list(images_ts.glob("*_0000.nii.gz")))
    n_lbl_ts = len(list(labels_ts.glob("*.nii.gz")))
    if n_img_ts != len(test_recs) or n_lbl_ts != len(test_recs):
        raise RuntimeError(
            f"Test disk state mismatch: test_recs={len(test_recs)} "
            f"imagesTs={n_img_ts} labelsTs={n_lbl_ts}")

    # ----- dataset.json ---------------------------------------------------
    dataset_json = {
        "channel_names":  {"0": "CT"},
        "labels":         CLASS_NAMES,
        "numTraining":    len(combined),
        "file_ending":    ".nii.gz",
        "name":           args.dataset_name,
        "description":    (
            "SpineSurg-CT unified spine + pelvis 10-class segmentation with "
            "partial-annotation support and upstream token-level "
            "5-fold CV splits."
        ),
    }
    (ds_dir / "dataset.json").write_text(json.dumps(dataset_json, indent=2))

    # ----- splits_final.json ---------------------------------------------
    print(f"\nWriting splits_final.json")
    if args.single_fold_splits:
        # Fallback: use the HF manifest train/val distinction (re-parse to
        # reconstruct it if we re-routed above).
        tr_seen: Set[str] = set()
        hf_tr = parse_manifest(
            args.hf_export_dir / "manifest_train.json",
            ct_dir, lbl_dir, args.train_fused_only, tr_seen, args.debug_n)
        hf_vl = parse_manifest(
            args.hf_export_dir / "manifest_validation.json",
            ct_dir, lbl_dir, args.train_fused_only, tr_seen, args.debug_n)
        if args.exclude_pseudo:
            hf_tr = [r for r in hf_tr if not r["is_pseudo"]]
            hf_vl = [r for r in hf_vl if not r["is_pseudo"]]
        write_splits_final_single_fold(
            ds_dir, [r["case_id"] for r in hf_tr], [r["case_id"] for r in hf_vl])
    else:
        assert lookup is not None
        write_splits_final_from_lookup(ds_dir, train_recs, lookup)

    # ----- summary --------------------------------------------------------
    print()
    print("=" * 70)
    print(f"Dataset:        {ds_dir}")
    mode = "K-fold (upstream splits)" if lookup is not None else "single-fold"
    print(f"  Splits mode:  {mode}")
    print(f"  TRAIN pool:   {len(combined)} cases "
          f"(fused={cfg_counter['fused']}  "
          f"spine_only={cfg_counter['spine_only']}  "
          f"pelvic_native={cfg_counter['pelvic_native']})")
    print(f"  TEST:         {len(test_recs)} cases")
    print(f"  ignore_label: {IGNORE_LABEL}")
    if args.debug_n > 0:
        print(f"  DEBUG MODE:   debug_n={args.debug_n}")
    print("=" * 70)
    print()
    print("NEXT STEPS")
    print(f"  export nnUNet_raw={args.nnunet_raw_dir}")
    print(f"  nnUNetv2_plan_and_preprocess -d {args.dataset_id} --verify_dataset_integrity")
    print(f"  cp {ds_dir}/splits_final.json \\")
    print(f"     $nnUNet_preprocessed/Dataset{args.dataset_id:03d}_{args.dataset_name}/splits_final.json")
    print(f"  nnUNetv2_train {args.dataset_id} 3d_fullres 0 -tr nnUNetTrainer_100epochs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
