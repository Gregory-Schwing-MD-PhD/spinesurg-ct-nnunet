#!/usr/bin/env python3
"""
SpineSurg-CT -- generate 5-fold CV splits at the patient-token level
scripts/generate_5fold_splits.py

Replaces the 70/15/15 stratified logic in generate_splits.py / src/data/pipeline.py
with a proper train_pool + test_holdout split followed by stratified 5-fold CV
on the train pool. Token-level (not case-level), so patients with multiple
configs (fused + spine_only etc.) never leak across folds.

Stratification
--------------
Primary stratum is `match_type` from placed_manifest.json -- each value
(`fused`, `separate`, `spine_only`, `pelvic_only`) is kept as a DISTINCT
stratum. 'fused' (same-series spine + pelvic) is the gold standard; keeping
it separate from 'separate' (different-series spine + pelvic) guarantees
that the test set gets known representation of each class, most importantly
of the gold-standard same-series fused cases.

If LSTV labels are available from the training manifest
(--training_manifest), the stratum becomes (match_type x has_lstv),
yielding up to ~8 strata -- mirroring your upstream 6-stratum scheme with
match_type granularity preserved.

If an LSTV cell has fewer cases than n_folds, it is coalesced into its
non-LSTV parent (prevents StratifiedKFold from failing on rare cells)
while keeping the match_type distinction intact.

Output
------
    data/splits_5fold.json

Schema (version 2):
    {
      "schema_version": 2,
      "created_at": "2026-04-18T...",
      "source_manifest":      "/data/.../placed_manifest.json",
      "source_manifest_mtime": "2026-04-17T23:11:05",
      "training_manifest":     "/data/.../colonog_training_manifest.json"  or null,
      "test_fraction":         0.15,
      "n_folds":               5,
      "kfold_seed":            42,
      "strata_scheme":         "match_type_x_lstv",
      "n_tokens_total":        987,
      "n_tokens_test":         148,
      "n_tokens_trainval":     839,
      "test_tokens":           ["123", ...],
      "folds": [
        {"train_tokens": [...], "val_tokens": [...]},
        ...
      ],
      "strata_counts_total":   {"fused|no_lstv": 500, ...},
      "strata_counts_test":    {...},
      "strata_counts_per_fold": [{...}, ...]
    }

Downstream
----------
    tools/convert_hf_to_nnunet.py reads this file via --splits_file. All
    records whose patient_token is in test_tokens land in imagesTs/labelsTs.
    All other records land in imagesTr/labelsTr with fold assignment per
    the "folds" list.

Usage
-----
    python scripts/generate_5fold_splits.py \\
        --placed_manifest  data/ctpelvic1k/masks/placed/placed_manifest.json \\
        --training_manifest data/matched/colonog_training_manifest.json \\
        --out              data/splits_5fold.json \\
        --n_folds          5 \\
        --seed             42 \\
        --test_fraction    0.15

Idempotent: if --out already exists and looks like a valid 5-fold file
matching the current placed_manifest, no-op. Use --overwrite to force.

Author: Gregory Schwing, MD-PhD  |  Wayne State University / DMC
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Suppress sklearn's "least populated class in y has only N members" warning.
# Our coalesce_rare_strata logic already handles rare cells; the remaining
# warnings after coalescing are cosmetic (exactly at threshold).
warnings.filterwarnings(
    "ignore",
    message="The least populated class in y has only",
    category=UserWarning,
)

try:
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, KFold
except ImportError:
    print("ERROR: scikit-learn required. `pip install scikit-learn`", file=sys.stderr)
    sys.exit(1)


# =============================================================================
# LSTV label classification
# =============================================================================

# Values treated as "no LSTV present" (case-insensitive). Everything else
# counts as LSTV-positive for stratification.
_LSTV_NEGATIVE = {
    "", "none", "n/a", "na", "normal", "absent", "no", "false", "0",
    "negative", "normal_variant",
}


def is_lstv(label) -> bool:
    if label is None:
        return False
    s = str(label).strip().lower()
    return s not in _LSTV_NEGATIVE


# =============================================================================
# Manifest readers
# =============================================================================

def read_placed_manifest(path: Path) -> List[Dict]:
    """
    Read placed_manifest.json and return a list of per-token records with
    fields: token, match_type, and any LSTV info found in-band.
    """
    data = json.loads(path.read_text())
    cases = data.get("cases", [])
    if not cases:
        raise RuntimeError(f"No cases in {path}")
    out = []
    for c in cases:
        token = str(c.get("patient_token") or c.get("token") or "")
        if not token:
            continue
        match_type = c.get("match_type") or "unknown"
        # Opportunistic LSTV lookup: placed_manifest may or may not carry it
        lstv_label = (
            c.get("lstv_label")
            or (c.get("spine") or {}).get("lstv_label")
            or (c.get("metadata") or {}).get("lstv_label")
            or None
        )
        out.append({
            "token":      token,
            "match_type": match_type,
            "lstv_label": lstv_label,
        })
    # Dedup (placed manifest is patient-centric but be defensive)
    by_token: Dict[str, Dict] = {}
    for r in out:
        if r["token"] in by_token:
            # Keep first record's match_type; merge LSTV info
            prev = by_token[r["token"]]
            if prev.get("lstv_label") is None and r.get("lstv_label") is not None:
                prev["lstv_label"] = r["lstv_label"]
        else:
            by_token[r["token"]] = r
    return list(by_token.values())


def augment_with_training_manifest(records: List[Dict],
                                    training_manifest_path: Path) -> int:
    """
    If LSTV info wasn't in placed_manifest.json, try to pull it from
    colonog_training_manifest.json (which is downstream and usually has
    per-case lstv_label). Matches on token. Returns the number of
    records updated.
    """
    if not training_manifest_path.exists():
        return 0
    try:
        tm = json.loads(training_manifest_path.read_text())
    except Exception as exc:
        print(f"WARN: could not parse {training_manifest_path}: {exc}",
              file=sys.stderr)
        return 0
    # training manifest layouts vary; accept list-of-dicts or {"cases": [...]}
    entries = tm if isinstance(tm, list) else tm.get("cases", tm.get("records", []))
    lstv_by_token: Dict[str, str] = {}
    for e in entries:
        if not isinstance(e, dict):
            continue
        tok = str(e.get("patient_token") or e.get("token") or "")
        if not tok:
            continue
        lbl = e.get("lstv_label")
        if lbl and tok not in lstv_by_token:
            lstv_by_token[tok] = lbl
    if not lstv_by_token:
        return 0
    n_updated = 0
    for r in records:
        if r.get("lstv_label") is None and r["token"] in lstv_by_token:
            r["lstv_label"] = lstv_by_token[r["token"]]
            n_updated += 1
    return n_updated


# =============================================================================
# Stratification
# =============================================================================

def _primary_stratum(match_type: str) -> str:
    """
    Each match_type is its own stratum. 'fused' (same TCIA series, gold
    standard) is kept distinct from 'separate' (spine + pelvic placed from
    different series -- still valid, but with potential cross-series drift).
    Keeping them distinct guarantees the test set has known representation
    of each class, most importantly of the gold-standard 'fused' cases.
    """
    mt = (match_type or "unknown").lower()
    return mt


def compute_strata(records: List[Dict]) -> List[str]:
    """
    Stratum = primary_match_type + '|' + ('lstv' if LSTV-positive else 'no_lstv').
    If we have no LSTV info at all, strata degrade to primary_match_type.
    Collapses rare strata (< n_folds members) into their non-LSTV parent.
    """
    strata = []
    any_lstv_info = any(r.get("lstv_label") is not None for r in records)
    for r in records:
        prim = _primary_stratum(r["match_type"])
        if any_lstv_info:
            lstv_tag = "lstv" if is_lstv(r["lstv_label"]) else "no_lstv"
            strata.append(f"{prim}|{lstv_tag}")
        else:
            strata.append(prim)
    return strata


def coalesce_rare_strata(strata: List[str], min_count: int) -> List[str]:
    """
    If a stratum has fewer members than min_count (e.g. < n_folds, or < 1/
    test_fraction), StratifiedKFold / StratifiedShuffleSplit will fail.
    Coalesce rare strata into their non-LSTV parent ("fused|lstv" -> "fused"
    if there are too few "fused|lstv" tokens).
    """
    counts = Counter(strata)
    rare = {s for s, c in counts.items() if c < min_count}
    if not rare:
        return strata
    out = []
    for s in strata:
        if s in rare and "|" in s:
            out.append(s.split("|", 1)[0])   # drop LSTV tag
        else:
            out.append(s)
    return out


# =============================================================================
# Validation (idempotency)
# =============================================================================

def existing_splits_still_valid(out_path: Path, current_tokens: List[str],
                                  n_folds: int) -> bool:
    """
    If out_path exists and its tokens exactly match current_tokens, we can
    no-op. This is the idempotency hook.
    """
    if not out_path.exists():
        return False
    try:
        j = json.loads(out_path.read_text())
    except Exception:
        return False
    if j.get("n_folds") != n_folds:
        return False
    if j.get("schema_version", 0) < 3:
        # v2 splits files don't have token_info, which downstream ablations
        # need. Force regeneration to upgrade the schema.
        return False
    tokens_in_splits = set(j.get("test_tokens", []))
    for f in j.get("folds", []):
        tokens_in_splits.update(f.get("train_tokens", []))
        tokens_in_splits.update(f.get("val_tokens", []))
    return tokens_in_splits == set(current_tokens)


# =============================================================================
# Main
# =============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate 5-fold CV splits at the patient-token level.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--placed_manifest", required=True, type=Path,
                    help="Path to placed_manifest.json (newest source of truth).")
    ap.add_argument("--training_manifest", type=Path, default=None,
                    help="Optional: colonog_training_manifest.json. If given, "
                         "used as a fallback source of lstv_label info.")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output splits JSON path (e.g. data/splits_5fold.json).")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--test_fraction", type=float, default=0.15,
                    help="Fraction of tokens held out as test set.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true",
                    help="Regenerate even if output is already valid.")
    args = ap.parse_args()

    if not args.placed_manifest.exists():
        print(f"ERROR: {args.placed_manifest} not found", file=sys.stderr)
        return 1
    if not 0.0 < args.test_fraction < 0.5:
        print(f"ERROR: --test_fraction must be in (0, 0.5), got {args.test_fraction}",
              file=sys.stderr)
        return 1
    if args.n_folds < 2:
        print(f"ERROR: --n_folds must be >= 2, got {args.n_folds}", file=sys.stderr)
        return 1

    # ----- read -----
    print(f"Reading {args.placed_manifest}")
    records = read_placed_manifest(args.placed_manifest)
    print(f"  found {len(records)} unique tokens")

    if args.training_manifest:
        n = augment_with_training_manifest(records, args.training_manifest)
        if n > 0:
            print(f"  enriched {n} tokens with lstv_label from "
                  f"{args.training_manifest.name}")

    tokens = [r["token"] for r in records]

    # ----- idempotency -----
    if not args.overwrite and existing_splits_still_valid(
            args.out, tokens, args.n_folds):
        print(f"Existing {args.out} matches the current token set "
              f"and is valid {args.n_folds}-fold. No-op. "
              f"Use --overwrite to regenerate.")
        return 0

    # ----- strata -----
    strata_raw = compute_strata(records)
    # For StratifiedShuffleSplit (test holdout), each class needs >= 2 members
    # (one train, one test). For StratifiedKFold, >= n_folds.
    # Coalesce rare strata early to cover both.
    min_count = max(args.n_folds, int(round(1.0 / args.test_fraction)))
    strata = coalesce_rare_strata(strata_raw, min_count)

    scheme = ("match_type_x_lstv"
              if any(r.get("lstv_label") is not None for r in records)
              else "match_type_only")
    print(f"  strata scheme: {scheme}")
    print(f"  strata counts: {dict(Counter(strata))}")

    # ----- test holdout (stratified) -----
    print(f"\nHolding out {args.test_fraction * 100:.0f}% as test set...")
    try:
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=args.test_fraction, random_state=args.seed)
        trainval_idx, test_idx = next(sss.split(tokens, strata))
    except ValueError as exc:
        print(f"  WARN: stratified test split failed ({exc}); "
              f"falling back to random.", file=sys.stderr)
        from sklearn.model_selection import ShuffleSplit
        ss = ShuffleSplit(n_splits=1, test_size=args.test_fraction,
                          random_state=args.seed)
        trainval_idx, test_idx = next(ss.split(tokens))

    test_tokens  = sorted(tokens[i] for i in test_idx)
    trainval_tokens = [tokens[i] for i in trainval_idx]
    trainval_strata = [strata[i] for i in trainval_idx]
    print(f"  test:      {len(test_tokens)} tokens")
    print(f"  trainval:  {len(trainval_tokens)} tokens")

    # ----- K-fold (stratified) on train pool -----
    print(f"\nStratified {args.n_folds}-fold on trainval pool...")
    trainval_strata_safe = coalesce_rare_strata(trainval_strata, args.n_folds)
    try:
        skf = StratifiedKFold(
            n_splits=args.n_folds, shuffle=True, random_state=args.seed)
        fold_splits = list(skf.split(trainval_tokens, trainval_strata_safe))
    except ValueError as exc:
        print(f"  WARN: StratifiedKFold failed ({exc}); falling back to KFold.",
              file=sys.stderr)
        kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
        fold_splits = list(kf.split(trainval_tokens))

    folds_out = []
    for i, (tr_i, va_i) in enumerate(fold_splits):
        tr_tokens = sorted(trainval_tokens[j] for j in tr_i)
        va_tokens = sorted(trainval_tokens[j] for j in va_i)
        folds_out.append({"train_tokens": tr_tokens, "val_tokens": va_tokens})
        print(f"  fold {i}: train={len(tr_tokens)}  val={len(va_tokens)}  "
              f"val_strata={dict(Counter(trainval_strata_safe[j] for j in va_i))}")

    # Disjointness sanity check
    val_union = set()
    for f in folds_out:
        s = set(f["val_tokens"])
        if val_union & s:
            raise RuntimeError(
                "Fold val sets are not disjoint -- sklearn.KFold is supposed "
                "to guarantee this. Something is very wrong.")
        val_union |= s
    if val_union != set(trainval_tokens):
        raise RuntimeError(
            f"Fold val union ({len(val_union)}) != trainval pool "
            f"({len(trainval_tokens)}). Fold generation is broken.")

    # ----- strata counts for audit -----
    stratum_by_token = dict(zip(tokens, strata))

    def _counts(token_list):
        return dict(Counter(stratum_by_token[t] for t in token_list))

    # Per-token info for downstream filtering (e.g. ablations by match_type).
    # Keyed by token string; values carry match_type and a boolean LSTV flag.
    token_info = {
        r["token"]: {
            "match_type": r["match_type"],
            "has_lstv":   is_lstv(r.get("lstv_label")),
        }
        for r in records
    }

    doc = {
        "schema_version":      3,
        "created_at":          datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source_manifest":     str(args.placed_manifest.resolve()),
        "source_manifest_mtime":
            datetime.fromtimestamp(
                args.placed_manifest.stat().st_mtime, tz=timezone.utc
            ).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "training_manifest":   str(args.training_manifest.resolve())
            if args.training_manifest else None,
        "test_fraction":       args.test_fraction,
        "n_folds":             args.n_folds,
        "kfold_seed":          args.seed,
        "strata_scheme":       scheme,
        "n_tokens_total":      len(tokens),
        "n_tokens_test":       len(test_tokens),
        "n_tokens_trainval":   len(trainval_tokens),
        "test_tokens":         test_tokens,
        "folds":               folds_out,
        "strata_counts_total":
            _counts(tokens),
        "strata_counts_test":
            _counts(test_tokens),
        "strata_counts_per_fold_val":
            [_counts(f["val_tokens"]) for f in folds_out],
        "token_info":          token_info,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2))
    print(f"\nWrote {args.out}")
    print(f"  total tokens : {len(tokens)}")
    print(f"  test         : {len(test_tokens)}")
    print(f"  folds        : {args.n_folds}  (stratified on '{scheme}')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
