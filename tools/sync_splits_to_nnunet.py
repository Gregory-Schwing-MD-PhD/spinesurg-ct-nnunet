#!/usr/bin/env python3
"""
sync_splits_to_nnunet.py — Single source of truth for splits + LSTV cases.

Reads:
  data/hf_export/splits_5fold.json    (patient-token level, schema v4+)
  data/hf_export/manifest.json        (HF export per-record manifest)

Writes:
  nnunet/preprocessed/Dataset802_SpineSurgCTFull/splits_final.json
  nnunet/preprocessed/Dataset802_SpineSurgCTFull/lstv_cases.json

Why this script exists
======================
The training pipeline has historically read from THREE different sources of
truth, producing disagreements that silently destroyed the LSTV-conditioned
results:

  1. splits_5fold.json — patient-token-level fold assignments (broken until v4)
  2. splits_final.json — nnU-Net's case_id-level fold assignments (derived
     from #1 in some unspecified previous step)
  3. lstv_cases.json   — the trainer's LSTV oversampler + per-subgroup dice
     logger reads this; was built from yet-another source

This script is the canonical bridge. It takes splits_5fold.json (now correct
in schema v4) and produces the other two files such that ALL THREE agree.

Output schemas
==============

splits_final.json  (nnU-Net format)
-----------------------------------
nnU-Net expects a JSON list of length n_folds, where each entry is:
  {"train": [case_id1, case_id2, ...], "val": [case_id1, ...]}

case_ids on this dataset are <token:04d>__<config> for each (patient, config)
pair, where config ∈ {fused, spine_only, pelvic_native}. A patient with both
spine and pelvic separate annotations contributes TWO case_ids to the same
fold (both train or both val — never split across).

lstv_cases.json
---------------
{
  "schema_version": 1,
  "case_ids": {
    "<case_id>": "lumbarization" | "sacralization" | "normal",
    ...
  },
  "subtype_counts": {"lumbarization": N, "sacralization": M, "normal": K}
}

Read by tools/nnunet_wandb_variant.py (trainer) at training time for:
  - LSTVOversampleMixin: which case_ids to over-sample
  - per-subgroup val dice: which subgroup each val case belongs to

case_id derivation
==================
HF manifest stores each record with `ct_file = "ct/<token:04d>[_spine|_pelvic]_ct.nii.gz"`
plus a `config` column ∈ {fused, spine_only, pelvic_native}. The
convert_hf_to_nnunet step (which is not in scope here) turns each HF record
into a case_id of the form `<safe_token>__<config>`. We mirror that
deterministic transform so the case_ids generated here match what nnU-Net
sees on disk in nnunet/raw/Dataset802_SpineSurgCTFull/imagesTr.

Patient-level split discipline
===============================
A patient and ALL their case_ids land in the same fold (or all in test).
Splitting a patient across folds violates patient-level CV.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


# Mirror of tools/convert_hf_to_nnunet.py:safe_id() — must stay in sync.
def safe_id(token: str) -> str:
    s = str(token)
    for bad in (".", "/", "\\", " ", ":", "(", ")"):
        s = s.replace(bad, "_")
    return s.strip("_")


def load_splits(path: Path) -> dict:
    if not path.exists():
        print(f"ERROR: splits file not found: {path}", file=sys.stderr)
        sys.exit(1)
    d = json.loads(path.read_text())
    schema = d.get("schema_version", 0)
    if schema < 4:
        print(f"ERROR: splits file has schema v{schema}, need v4+. "
              f"Re-run generate_5fold_splits.py first.", file=sys.stderr)
        sys.exit(1)
    return d


def load_hf_manifest(path: Path) -> List[dict]:
    if not path.exists():
        print(f"ERROR: HF manifest not found: {path}", file=sys.stderr)
        sys.exit(1)
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        for k in ("records", "cases", "items"):
            if k in raw and isinstance(raw[k], list):
                return raw[k]
        if all(isinstance(v, dict) for v in raw.values()):
            return [{"token": k, **v} for k, v in raw.items()]
        raise RuntimeError(f"{path}: unrecognized HF manifest shape")
    if isinstance(raw, list):
        return raw
    raise RuntimeError(f"{path}: HF manifest root is {type(raw).__name__}")


def build_token_to_case_ids(hf_records: List[dict]) -> Dict[str, List[Tuple[str, str]]]:
    """token -> list of (case_id, lstv_subtype) tuples.

    case_id = "<safe_token>__<config>"
    config ∈ {fused, spine_only, pelvic_native}

    A single patient may contribute multiple records (one per config) when
    they have separate spine and pelvic annotations. All of them go to the
    same fold.
    """
    out: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    skipped = Counter()

    for r in hf_records:
        if not isinstance(r, dict):
            continue
        token = str(r.get("token") or r.get("patient_token") or "")
        config = r.get("config", "")
        if not token or not config:
            skipped["missing_token_or_config"] += 1
            continue
        if config not in ("fused", "spine_only", "pelvic_native"):
            skipped[f"unknown_config:{config}"] += 1
            continue

        case_id = f"{safe_id(token)}__{config}"

        # Resolve LSTV subtype. The HF manifest's `lstv_label` is already
        # resolved by export_hf.py:build_work, so just bucket it.
        lstv = str(r.get("lstv_label", "")).strip().upper()
        if "LUMB" in lstv:
            sub = "lumbarization"
        elif "SACR" in lstv:
            sub = "sacralization"
        else:
            sub = "normal"

        out[token].append((case_id, sub))

    if skipped:
        print(f"WARN: skipped HF records: {dict(skipped)}", file=sys.stderr)

    return dict(out)


def write_splits_final(splits: dict,
                        token_to_case_ids: Dict[str, List[Tuple[str, str]]],
                        out_path: Path) -> Tuple[List[dict], Set[str]]:
    """Convert patient-token splits to nnU-Net case_id splits.

    Returns (folds_out, unmapped_tokens) where folds_out is the list nnU-Net
    expects and unmapped_tokens are tokens that had no matching HF records.
    """
    folds = splits["folds"]
    test_tokens = set(splits["test_tokens"])
    unmapped: Set[str] = set()

    folds_out: List[dict] = []
    for fold_idx, fold in enumerate(folds):
        train_tokens = fold["train_tokens"]
        val_tokens = fold["val_tokens"]

        train_cases: List[str] = []
        val_cases: List[str] = []

        for t in train_tokens:
            cases = token_to_case_ids.get(t, [])
            if not cases:
                unmapped.add(t)
                continue
            train_cases.extend(cid for cid, _ in cases)

        for t in val_tokens:
            cases = token_to_case_ids.get(t, [])
            if not cases:
                unmapped.add(t)
                continue
            val_cases.extend(cid for cid, _ in cases)

        train_cases.sort()
        val_cases.sort()

        # Sanity: train and val should be disjoint at the case level (they
        # come from disjoint token sets, but worth asserting).
        overlap = set(train_cases) & set(val_cases)
        if overlap:
            raise RuntimeError(
                f"Fold {fold_idx} has train ∩ val case_id overlap "
                f"(should never happen): {sorted(overlap)[:10]}")

        folds_out.append({"train": train_cases, "val": val_cases})
        print(f"  fold {fold_idx}: train={len(train_cases)} cases  "
              f"val={len(val_cases)} cases")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(folds_out, indent=2))
    return folds_out, unmapped


def write_lstv_cases(token_to_case_ids: Dict[str, List[Tuple[str, str]]],
                      splits: dict,
                      out_path: Path) -> dict:
    """Write lstv_cases.json from the canonical splits + manifest.

    Schema:
      {
        "schema_version": 1,
        "case_ids": {<case_id>: "lumbarization" | "sacralization" | "normal", ...},
        "subtype_counts": {...}
      }

    The trainer's _LSTVOversampleMixin reads `case_ids` and uses any
    case_id whose value is NOT "normal" as an LSTV case for oversampling.
    The per-subgroup dice logger uses the same dict to bucket val cases.
    """
    case_to_sub: Dict[str, str] = {}

    # Limit to tokens actually in the splits (defensive — a stray HF record
    # with no fold assignment shouldn't pollute the trainer's case set).
    splits_tokens: Set[str] = set(splits.get("test_tokens", []))
    for f in splits.get("folds", []):
        splits_tokens.update(f.get("train_tokens", []))
        splits_tokens.update(f.get("val_tokens", []))

    for token, cases in token_to_case_ids.items():
        if token not in splits_tokens:
            continue
        for case_id, sub in cases:
            case_to_sub[case_id] = sub

    counts = Counter(case_to_sub.values())
    out = {
        "schema_version": 1,
        "source_splits_file": str(splits.get("source_manifest", "?")),
        "case_ids": case_to_sub,
        "subtype_counts": dict(counts),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Sync splits_5fold.json -> nnU-Net splits_final.json + lstv_cases.json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--splits", required=True, type=Path,
                    help="Patient-token splits file (schema v4+)")
    ap.add_argument("--hf_manifest", required=True, type=Path,
                    help="HF manifest.json with per-record (token, config, lstv_label)")
    ap.add_argument("--nnunet_preprocessed", required=True, type=Path,
                    help="nnunet/preprocessed/Dataset802_SpineSurgCTFull (writes splits_final.json + lstv_cases.json here)")
    ap.add_argument("--also_write_raw", type=Path, default=None,
                    help="Optional second location to copy lstv_cases.json (e.g. nnunet/raw/<dataset>)")
    args = ap.parse_args()

    print("=" * 70)
    print("sync_splits_to_nnunet.py")
    print(f"  splits          : {args.splits}")
    print(f"  hf manifest     : {args.hf_manifest}")
    print(f"  nnunet prep dir : {args.nnunet_preprocessed}")
    print("=" * 70)

    splits = load_splits(args.splits)
    print(f"  splits schema: v{splits['schema_version']}")
    print(f"  total tokens : {splits['n_tokens_total']}")
    print(f"  test         : {splits['n_tokens_test']}")
    print(f"  trainval     : {splits['n_tokens_trainval']}")
    print(f"  n_folds      : {splits['n_folds']}")
    print(f"  scheme       : {splits['strata_scheme']}")
    if "lstv_subtype_counts" in splits:
        print(f"  LSTV totals  : {splits['lstv_subtype_counts']['total']}")

    hf_records = load_hf_manifest(args.hf_manifest)
    print(f"  HF records   : {len(hf_records)}")

    token_to_case_ids = build_token_to_case_ids(hf_records)
    print(f"  Tokens with HF records: {len(token_to_case_ids)}")

    # ── Build splits_final.json ────────────────────────────────────────
    splits_final_path = args.nnunet_preprocessed / "splits_final.json"
    print(f"\nWriting nnU-Net splits to {splits_final_path}")
    folds_out, unmapped = write_splits_final(
        splits, token_to_case_ids, splits_final_path)

    if unmapped:
        print(f"  WARN: {len(unmapped)} tokens in splits had no matching HF records: "
              f"{sorted(unmapped)[:10]}{'...' if len(unmapped) > 10 else ''}",
              file=sys.stderr)

    # ── Build lstv_cases.json ──────────────────────────────────────────
    lstv_cases_path = args.nnunet_preprocessed / "lstv_cases.json"
    print(f"\nWriting LSTV case map to {lstv_cases_path}")
    lstv_doc = write_lstv_cases(token_to_case_ids, splits, lstv_cases_path)
    print(f"  total case_ids       : {len(lstv_doc['case_ids'])}")
    print(f"  subtype counts       : {lstv_doc['subtype_counts']}")

    if args.also_write_raw:
        secondary = args.also_write_raw / "lstv_cases.json"
        secondary.parent.mkdir(parents=True, exist_ok=True)
        secondary.write_text(json.dumps(lstv_doc, indent=2))
        print(f"  also wrote: {secondary}")

    # ── Per-fold LSTV balance summary ──────────────────────────────────
    print(f"\nPer-fold LSTV balance (from generated splits_final.json):")
    for fi, fold in enumerate(folds_out):
        tr_subs = Counter(lstv_doc["case_ids"].get(c, "?") for c in fold["train"])
        va_subs = Counter(lstv_doc["case_ids"].get(c, "?") for c in fold["val"])
        print(f"  fold {fi}  val:   {dict(va_subs)}")
        print(f"          train: {dict(tr_subs)}")

    # Test set
    test_tokens = set(splits["test_tokens"])
    test_cases = []
    for t in test_tokens:
        for cid, _ in token_to_case_ids.get(t, []):
            test_cases.append(cid)
    test_subs = Counter(lstv_doc["case_ids"].get(c, "?") for c in test_cases)
    print(f"  test           : {dict(test_subs)}  ({len(test_cases)} cases)")

    print("\n  ✓ done. Trainer's lstv_cases.json + nnU-Net splits_final.json are now consistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
