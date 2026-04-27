#!/usr/bin/env python3
"""
SpineSurg-CT -- HF export -> nnU-Net v2 dataset format
tools/convert_hf_to_nnunet.py

Fixes (cumulative)
==================
1. Cross-manifest case_id collisions. Namespaced {safe_token}__{config}.
2. Partial-annotation support via ignore_label = 10.
3. Manifest path-prefix tolerance (v6 schema): strip "ct/"/"labels/".
4. Symlinks use absolute paths AS-GIVEN (no .resolve()). When this
   script runs in a container with /data/hf_export bind-mounted from
   a host directory, .resolve() would embed the container-internal
   path into symlinks, breaking them for any reader outside the
   container. Pass a host path -> get a host-readable symlink.
5. Self-healing idempotency. When the dataset is already built, the
   script verifies a sample of symlinks resolve. If they're broken in
   the recognizable "/data/hf_export/..." pattern, it rewrites every
   broken symlink in-place to point at the host HF export directory.
   Recovers a previous run that was made before fix #4 without a
   full rebuild.
6. lstv_cases.json metadata writer (Apr 2026). After the dataset is
   built (or re-built via --regen_splits_only), writes a small JSON
   file listing every case_id whose lstv_label != "normal", along
   with the subtype. Consumed by tools/nnunet_wandb_variant.py at
   training time to oversample LSTV cases (both lumbarization AND
   sacralization, unlike the legacy L6-voxel scan which only finds
   lumbarization).

Author: Gregory Schwing, MD-PhD  |  Wayne State University / DMC
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
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

# Symlinks written by older versions used Path.resolve() inside a
# container with /data/hf_export bind-mounted, which embedded the
# container-internal path into the symlink. We recognize those broken
# targets by this prefix and rewrite them in-place.
LEGACY_CONTAINER_HF_PREFIX = "/data/hf_export/"

# Schema version for lstv_cases.json. Bump on format changes so the
# trainer-side reader can detect incompatibilities.
LSTV_CASES_SCHEMA_VERSION = 1


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
# Path helpers
# =============================================================================

def _strip_leading_segment(rel: str, segment: str) -> str:
    """Strip a leading "{segment}/" from `rel` if present."""
    if not rel:
        return rel
    prefix = f"{segment}/"
    if rel.startswith(prefix):
        return rel[len(prefix):]
    bs_prefix = f"{segment}\\"
    if rel.startswith(bs_prefix):
        return rel[len(bs_prefix):]
    return rel


# =============================================================================
# File ops
# =============================================================================

def link(src: Path, dst: Path, *, allow_overwrite: bool = False) -> None:
    """
    Create a symlink at `dst` pointing at `src`. Uses the absolute path
    of `src` AS-GIVEN -- does NOT call .resolve(). See top-of-file
    notes (fix #4) for why this matters in a containerized run.
    """
    if dst.exists() or dst.is_symlink():
        if not allow_overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing {dst} (src was {src}).")
        dst.unlink()
    src_abs = Path(src) if Path(src).is_absolute() else Path(src).absolute()
    dst.symlink_to(src_abs)


def rewrite_label(src_path: Path, dst_path: Path, config: str) -> Dict:
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
# Self-healing repair: legacy container-path symlinks -> host-path symlinks
# =============================================================================

def _sample_broken_symlinks(ds_dir: Path, n_check: int = 20) -> List[Path]:
    """Return up to n_check symlinks under ds_dir whose targets don't resolve."""
    broken: List[Path] = []
    for sub in ("imagesTr", "imagesTs", "labelsTr", "labelsTs"):
        d = ds_dir / sub
        if not d.is_dir():
            continue
        for p in d.iterdir():
            if not p.is_symlink():
                continue
            try:
                if p.resolve(strict=True).exists():
                    continue
            except (FileNotFoundError, OSError):
                pass
            broken.append(p)
            if len(broken) >= n_check:
                return broken
    return broken


def repair_broken_symlinks(ds_dir: Path,
                            hf_export_dir: Path) -> Tuple[int, int]:
    """
    Walk every symlink under ds_dir's image/label directories. Any
    symlink whose target starts with LEGACY_CONTAINER_HF_PREFIX gets
    rewritten in-place to point at the corresponding file under
    `hf_export_dir`. Returns (n_fixed, n_left_broken).
    """
    n_fixed = 0
    n_still_broken = 0
    n_unrecognized = 0
    examples_unrecognized: List[str] = []
    examples_missing: List[Tuple[str, str]] = []

    for sub in ("imagesTr", "imagesTs", "labelsTr", "labelsTs"):
        d = ds_dir / sub
        if not d.is_dir():
            continue
        for p in d.iterdir():
            if not p.is_symlink():
                continue
            try:
                if p.resolve(strict=True).exists():
                    continue
            except (FileNotFoundError, OSError):
                pass
            tgt = os.readlink(p)
            if not tgt.startswith(LEGACY_CONTAINER_HF_PREFIX):
                n_unrecognized += 1
                if len(examples_unrecognized) < 3:
                    examples_unrecognized.append(f"{p.name} -> {tgt}")
                continue
            rel = tgt[len(LEGACY_CONTAINER_HF_PREFIX):]
            new_tgt = hf_export_dir / rel
            if not new_tgt.exists():
                n_still_broken += 1
                if len(examples_missing) < 3:
                    examples_missing.append((p.name, str(new_tgt)))
                continue
            p.unlink()
            p.symlink_to(new_tgt.absolute())
            n_fixed += 1

    if n_fixed:
        print(f"  repaired {n_fixed} broken symlink(s) "
              f"(legacy container paths -> host paths under {hf_export_dir})")
    if n_unrecognized:
        print(f"  WARN: {n_unrecognized} broken symlink(s) had unrecognized "
              f"targets and were not auto-repaired.", file=sys.stderr)
        for ex in examples_unrecognized:
            print(f"    {ex}", file=sys.stderr)
    if n_still_broken:
        print(f"  WARN: {n_still_broken} legacy symlink(s) point at files "
              f"that don't exist under {hf_export_dir}. The HF export may "
              f"have changed since the dataset was built.", file=sys.stderr)
        for name, tgt in examples_missing:
            print(f"    {name} -> {tgt}", file=sys.stderr)
    return n_fixed, n_still_broken + n_unrecognized


# =============================================================================
# LSTV cases JSON writer (consumed by nnunet_wandb_variant.py at training)
# =============================================================================

def write_lstv_cases_json(ds_dir: Path,
                           records: List[Dict],
                           also_write_to: Optional[Path] = None) -> Dict:
    """
    Write lstv_cases.json into ds_dir based on the provided record list.

    Each record is a dict produced by parse_manifest() with these keys:
        case_id, token, config, lstv_label, ...

    Output JSON schema (v1):
        {
          "schema_version": 1,
          "n_cases": <int>,
          "subtype_counts": {"lumbarization": N, "sacralization": M, ...},
          "case_ids": { "<case_id>": "<lstv_label>", ... }
        }

    The trainer-side oversampling mixin reads this file to identify
    LSTV cases without rescanning the HF manifests. Covers both
    lumbarization AND sacralization (the legacy L6-voxel scan only
    found lumbarization, since sacralization cases have no L6 voxels
    in their labels).

    If `also_write_to` is given, a copy is also written there. Used to
    drop the file into the preprocessed/ directory at training time so
    the trainer can find it without env-var configuration.

    Returns the data dict that was written.
    """
    case_ids: Dict[str, str] = {}
    subtypes: Counter = Counter()
    for r in records:
        lstv = str(r.get("lstv_label", "")).strip().lower()
        if not lstv or lstv == "normal":
            continue
        cid = r.get("case_id")
        if not cid:
            continue
        case_ids[cid] = lstv
        subtypes[lstv] += 1

    data = {
        "schema_version": LSTV_CASES_SCHEMA_VERSION,
        "n_cases":        len(case_ids),
        "subtype_counts": dict(subtypes),
        "case_ids":       case_ids,
    }

    out_path = ds_dir / "lstv_cases.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2, sort_keys=True))
    print(f"  wrote {out_path}  "
          f"(n_lstv={data['n_cases']}, subtypes={data['subtype_counts']})")

    if also_write_to is not None:
        also_write_to.parent.mkdir(parents=True, exist_ok=True)
        also_write_to.write_text(json.dumps(data, indent=2, sort_keys=True))
        print(f"  wrote {also_write_to}")

    return data


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
    sample_missing: List[str] = []

    for r in raw:
        config = r.get("config", "fused")
        if config not in VALID_CONFIGS:
            raise ValueError(f"Unknown config in manifest: {config!r}")
        if fused_only and config != "fused":
            skipped_cfg += 1
            continue

        ct_rel  = _strip_leading_segment(r.get("ct_file",    ""), "ct")
        lbl_rel = _strip_leading_segment(r.get("label_file", ""), "labels")

        ct  = ct_dir  / ct_rel
        lbl = lbl_dir / lbl_rel
        if not ct.exists() or not lbl.exists():
            skipped_missing += 1
            if len(sample_missing) < 3:
                sample_missing.append(
                    f"ct={ct} (exists={ct.exists()})  "
                    f"lbl={lbl} (exists={lbl.exists()})"
                )
            continue

        token = r.get("token") or Path(ct_rel).stem.replace(".nii", "")
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
    if skipped_missing > 0 and sample_missing:
        print(f"  first {len(sample_missing)} missing example(s):", file=sys.stderr)
        for s in sample_missing:
            print(f"    {s}", file=sys.stderr)
    return out


# =============================================================================
# Splits file loading
# =============================================================================

class SplitsLookup:
    def __init__(self, splits_data: Dict):
        self.data = splits_data
        self.n_folds = int(splits_data["n_folds"])
        self.test_tokens: Set[str] = set(str(t) for t in splits_data["test_tokens"])
        self.token_to_val_fold: Dict[str, int] = {}
        for i, f in enumerate(splits_data["folds"]):
            for t in f.get("val_tokens", []):
                ts = str(t)
                if ts in self.token_to_val_fold:
                    raise RuntimeError(
                        f"Token {ts!r} appears in multiple fold val sets "
                        f"(folds {self.token_to_val_fold[ts]} and {i}).")
                self.token_to_val_fold[ts] = i
        self.token_info: Dict[str, Dict] = {
            str(k): v for k, v in (splits_data.get("token_info") or {}).items()
        }

    def is_test(self, token: str) -> bool:
        return str(token) in self.test_tokens

    def val_fold_for(self, token: str) -> Optional[int]:
        return self.token_to_val_fold.get(str(token))

    def known_tokens(self) -> Set[str]:
        return set(self.test_tokens) | set(self.token_to_val_fold.keys())

    def match_type_for(self, token: str) -> Optional[str]:
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
            f"--splits_file {path} not found.")
    data = json.loads(path.read_text())
    schema = data.get("schema_version", 0)
    if schema < 2:
        raise ValueError(
            f"{path} has schema_version={schema}; need >= 2.")
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
    n_folds = lookup.n_folds
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
              f"not in the splits file's trainval pool.")

    all_train_ids = [r["case_id"] for r in train_recs]
    all_train_set = set(all_train_ids)

    splits = []
    for i in range(n_folds):
        val_ids = sorted(set(val_ids_per_fold[i]))
        train_ids = sorted(all_train_set - set(val_ids))
        splits.append({"train": train_ids, "val": val_ids})

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
    fold = {"train": sorted(train_ids), "val": sorted(val_ids)}
    splits = [fold for _ in range(5)]
    out = ds_dir / "splits_final.json"
    out.write_text(json.dumps(splits, indent=2))
    print(f"  wrote {out} (single-fold replicated)")


def splits_final_looks_kfold(ds_dir: Path) -> bool:
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
        description="Convert SpineSurg-CT HF export -> nnU-Net v2 dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--hf_export_dir",   required=True, type=Path)
    ap.add_argument("--nnunet_raw_dir",  required=True, type=Path)
    ap.add_argument("--dataset_id",      type=int, default=802)
    ap.add_argument("--dataset_name",    default="SpineSurgCTFull")

    ap.add_argument("--splits_file", type=Path, default=None)
    ap.add_argument("--single_fold_splits", action="store_true", default=False)
    ap.add_argument("--regen_splits_only", action="store_true", default=False)

    ap.add_argument("--include_train_match_types", default=None)
    ap.add_argument("--test_match_types", default=None)

    ap.add_argument("--train_fused_only", action="store_true", default=False)
    ap.add_argument("--test_all_configs", action="store_true", default=False)

    ap.add_argument("--exclude_pseudo",  action="store_true", default=True)
    ap.add_argument("--include_pseudo",  dest="exclude_pseudo", action="store_false")

    ap.add_argument("--debug_n", type=int, default=0)

    args = ap.parse_args()

    if not args.single_fold_splits and args.splits_file is None:
        print("ERROR: --splits_file is required unless --single_fold_splits is set.",
              file=sys.stderr)
        return 1

    lookup: Optional[SplitsLookup] = None
    if not args.single_fold_splits:
        print(f"Loading splits from {args.splits_file}")
        lookup = load_splits_file(args.splits_file)
        print(f"  n_folds={lookup.n_folds}  "
              f"test_tokens={len(lookup.test_tokens)}  "
              f"trainval_tokens={len(lookup.token_to_val_fold)}")

    ct_dir  = args.hf_export_dir / "ct"
    lbl_dir = args.hf_export_dir / "labels"
    if not ct_dir.is_dir() or not lbl_dir.is_dir():
        print(f"ERROR: missing {ct_dir} or {lbl_dir}", file=sys.stderr)
        return 1

    ds_dir = args.nnunet_raw_dir / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"

    # =========================================================================
    # --regen_splits_only branch
    # =========================================================================
    if args.regen_splits_only:
        if not (ds_dir / "dataset.json").exists():
            print(f"ERROR: --regen_splits_only but dataset not built "
                  f"({ds_dir}/dataset.json missing).", file=sys.stderr)
            return 1
        print(f"\n--regen_splits_only: re-parsing manifests to rewrite splits")
        train_val_seen: Set[str] = set()
        tr = parse_manifest(
            args.hf_export_dir / "manifest_train.json",
            ct_dir, lbl_dir, args.train_fused_only, train_val_seen, args.debug_n)
        vl = parse_manifest(
            args.hf_export_dir / "manifest_validation.json",
            ct_dir, lbl_dir, args.train_fused_only, train_val_seen, args.debug_n)
        ts = parse_manifest(
            args.hf_export_dir / "manifest_test.json",
            ct_dir, lbl_dir,
            fused_only=not args.test_all_configs,
            seen_ids=set(),  # test pool is independent of train/val seen set
            debug_n=args.debug_n)
        if args.exclude_pseudo:
            tr = [r for r in tr if not r["is_pseudo"]]
            vl = [r for r in vl if not r["is_pseudo"]]
            ts = [r for r in ts if not r["is_pseudo"]]
        if args.single_fold_splits:
            write_splits_final_single_fold(
                ds_dir, [r["case_id"] for r in tr], [r["case_id"] for r in vl])
        else:
            assert lookup is not None
            combined = [r for r in tr + vl if not lookup.is_test(r["token"])]
            removed  = len(tr) + len(vl) - len(combined)
            if removed:
                print(f"  removed {removed} records whose token is in test set")
            write_splits_final_from_lookup(ds_dir, combined, lookup)

        # Refresh lstv_cases.json over the entire (train+val+test) record set
        all_recs = tr + vl + ts
        write_lstv_cases_json(ds_dir, all_recs)

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

    # K-fold re-routing
    if lookup is not None:
        pool = train_recs + val_recs + test_recs
        pool_train = [r for r in pool if not lookup.is_test(r["token"])]
        pool_test  = [r for r in pool if     lookup.is_test(r["token"])]

        if args.include_train_match_types:
            if not lookup.has_token_info():
                print("ERROR: --include_train_match_types requires schema v3+.",
                      file=sys.stderr)
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
                    continue
                if mt in allowed:
                    kept.append(r)
                else:
                    n_excluded += 1
                    excluded_counts[mt] += 1
            pool_train = kept
            print(f"\n[ABLATION] --include_train_match_types={sorted(allowed)}")
            print(f"  kept    : {len(pool_train)} training records (from {before})")
            print(f"  excluded: {dict(excluded_counts)} (total {n_excluded})")
            if n_unknown:
                print(f"  excluded (unknown match_type): {n_unknown}")

        if args.test_match_types:
            if not lookup.has_token_info():
                print("ERROR: --test_match_types requires schema v3+.",
                      file=sys.stderr)
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

        known = lookup.known_tokens()
        unknown = [r for r in pool_train if r["token"] not in known]
        if unknown:
            print(f"  WARN: {len(unknown)} train-pool records have tokens "
                  f"not in splits file. Examples: "
                  f"{sorted({r['token'] for r in unknown})[:5]}")
        train_recs = pool_train
        val_recs   = []
        test_recs  = pool_test
        print(f"\nRe-routed by splits file:")
        print(f"  imagesTr pool : {len(train_recs)}")
        print(f"  imagesTs pool : {len(test_recs)}")

    images_tr = ds_dir / "imagesTr"
    labels_tr = ds_dir / "labelsTr"
    images_ts = ds_dir / "imagesTs"
    labels_ts = ds_dir / "labelsTs"

    # =========================================================================
    # Idempotency for an existing dataset directory.
    # =========================================================================
    if ds_dir.exists():
        has_content = any(ds_dir.iterdir())
        if args.debug_n > 0 and has_content:
            print(f"  [debug_n] wiping existing {ds_dir}")
            shutil.rmtree(ds_dir)
        elif has_content:
            marker = ds_dir / "dataset.json"
            if marker.exists():
                # Self-heal symlinks if needed (applies to all states).
                broken = _sample_broken_symlinks(ds_dir, n_check=20)
                if broken:
                    print(f"  NOTE: detected {len(broken)} broken symlink(s) "
                          f"in sample; attempting in-place repair...")
                    n_fixed, n_left = repair_broken_symlinks(
                        ds_dir, args.hf_export_dir)
                    if n_fixed == 0 and n_left > 0:
                        print(f"  ERROR: {n_left} broken symlink(s) could not "
                              f"be auto-repaired. Inspect manually or delete "
                              f"{ds_dir} and rerun.", file=sys.stderr)
                        return 1
                    if n_left > 0:
                        print(f"  WARN: {n_fixed} repaired, {n_left} still "
                              f"broken; see warnings above.", file=sys.stderr)
                    else:
                        post = _sample_broken_symlinks(ds_dir, n_check=5)
                        if post:
                            print(f"  WARN: {len(post)} still-broken symlinks "
                                  f"after repair pass; investigate manually.",
                                  file=sys.stderr)
                        else:
                            print(f"  symlinks now resolve cleanly.")

                # Splits-shape idempotency.
                want_kfold = not args.single_fold_splits
                have_kfold = splits_final_looks_kfold(ds_dir)
                if want_kfold and not have_kfold:
                    print(f"  NOTE: {ds_dir} populated but splits_final.json "
                          f"is single-fold style. Rewriting splits only.")
                    write_splits_final_from_lookup(ds_dir, train_recs, lookup)
                    write_lstv_cases_json(ds_dir, train_recs + test_recs)
                    return 0
                elif not want_kfold and have_kfold:
                    print(f"  NOTE: {ds_dir} populated with K-fold splits "
                          f"but --single_fold_splits requested. Rewriting "
                          f"splits only.")
                    hf_tr = parse_manifest(
                        args.hf_export_dir / "manifest_train.json",
                        ct_dir, lbl_dir, args.train_fused_only,
                        set(), args.debug_n)
                    hf_vl = parse_manifest(
                        args.hf_export_dir / "manifest_validation.json",
                        ct_dir, lbl_dir, args.train_fused_only,
                        set(), args.debug_n)
                    if args.exclude_pseudo:
                        hf_tr = [r for r in hf_tr if not r["is_pseudo"]]
                        hf_vl = [r for r in hf_vl if not r["is_pseudo"]]
                    write_splits_final_single_fold(
                        ds_dir,
                        [r["case_id"] for r in hf_tr],
                        [r["case_id"] for r in hf_vl],
                    )
                    write_lstv_cases_json(ds_dir, hf_tr + hf_vl + test_recs)
                    return 0
                else:
                    # Even if dataset is already built, refresh lstv_cases.json
                    # to handle re-runs after metadata-only edits.
                    print(f"  NOTE: {ds_dir} already populated with "
                          f"{'K-fold' if have_kfold else 'single-fold'} "
                          f"splits. Refreshing lstv_cases.json only.")
                    if lookup is not None:
                        write_lstv_cases_json(ds_dir, train_recs + test_recs)
                    print(f"        Delete {ds_dir} to rebuild, or pass "
                          f"--regen_splits_only to refresh splits.")
                    return 0
            raise RuntimeError(
                f"{ds_dir} exists and is non-empty but has no dataset.json. "
                f"Refusing to mix with stale state. Delete it and rerun.")

    for d in (images_tr, labels_tr, images_ts, labels_ts):
        d.mkdir(parents=True, exist_ok=True)

    combined = train_recs + val_recs

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

    n_img_tr = len(list(images_tr.glob("*_0000.nii.gz")))
    n_lbl_tr = len(list(labels_tr.glob("*.nii.gz")))
    if n_img_tr != len(combined) or n_lbl_tr != len(combined):
        raise RuntimeError(
            f"Disk state mismatch! combined={len(combined)} "
            f"imagesTr={n_img_tr} labelsTr={n_lbl_tr}")
    print(f"  verified: imagesTr={n_img_tr} labelsTr={n_lbl_tr}")

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

    print(f"\nWriting splits_final.json")
    if args.single_fold_splits:
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

    # Write LSTV cases metadata for the trainer to consume.
    print(f"\nWriting lstv_cases.json")
    write_lstv_cases_json(ds_dir, train_recs + val_recs + test_recs)

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
    print(f"  cp {ds_dir}/lstv_cases.json \\")
    print(f"     $nnUNet_preprocessed/Dataset{args.dataset_id:03d}_{args.dataset_name}/lstv_cases.json")
    print(f"  nnUNetv2_train {args.dataset_id} 3d_fullres 0 -tr nnUNetTrainerWandB_500ep_LSTVOversample")
    return 0


if __name__ == "__main__":
    sys.exit(main())
