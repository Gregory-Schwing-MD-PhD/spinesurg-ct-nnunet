#!/usr/bin/env python3
"""benchmark/make_case_table.py — the one case list every system in the benchmark runs on.

One row per Dataset810 training record (all 802; the five folds partition them, so our own
model is scored on its validation fold and every competitor on all 802):

    case, token, ct, label, fold, subtype, config, has_l6, has_lumbar_rib, lumbar_rib_side,
    n_lumbar_labels, lstv_label

`ct` is the host path of the CT (imagesTr holds symlinks written inside the prep container,
where the HF export was bound at /data/hf_export; the prefix is rewritten here).
`label` is the one-shot (22 foreground + ignore) label the network was trained on, which is
the common scoring space for every system.

    python benchmark/make_case_table.py \
        --raw nnunet/raw/Dataset810_SpineSurgLSTVOneShot \
        --preprocessed nnunet/preprocessed/Dataset810_SpineSurgLSTVOneShot \
        --hf_export ~/data/CTSpinoPelvic1K --out benchmark/cases.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

CONTAINER_HF = "/data/hf_export"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--preprocessed", required=True, type=Path)
    ap.add_argument("--hf_export", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    splits = json.loads((a.preprocessed / "splits_final.json").read_text())
    folds = splits["folds"] if isinstance(splits, dict) and "folds" in splits else splits
    case_fold = {}
    for i, f in enumerate(folds):
        for c in f["val"]:
            case_fold[c] = i

    lc = json.loads((a.raw / "lstv_cases.json").read_text())
    subtype = lc.get("case_to_subtype", {})
    attrs = lc.get("case_to_attrs", {})
    tokens = lc.get("case_to_token", {})

    rows = []
    for lab in sorted((a.raw / "labelsTr").glob("*.nii.gz")):
        case = lab.name.replace(".nii.gz", "")
        img = a.raw / "imagesTr" / f"{case}_0000.nii.gz"
        target = os.readlink(img) if img.is_symlink() else str(img)
        if target.startswith(CONTAINER_HF):
            target = str(a.hf_export.expanduser()) + target[len(CONTAINER_HF):]
        at = attrs.get(case, {})
        rows.append({
            "case": case, "token": tokens.get(case, ""), "ct": target, "label": str(lab.resolve()),
            "fold": case_fold.get(case, ""), "subtype": subtype.get(case, "normal"),
            "config": at.get("config", ""), "has_l6": int(bool(at.get("has_l6"))),
            "has_lumbar_rib": int(bool(at.get("has_lumbar_rib"))),
            "lumbar_rib_side": at.get("lumbar_rib_side", ""),
            "n_lumbar_labels": at.get("n_lumbar_labels", ""), "lstv_label": at.get("lstv_label", ""),
        })
    missing = [r["case"] for r in rows if not Path(r["ct"]).exists()]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} cases -> {a.out}; folds: {sorted(set(r['fold'] for r in rows))}; "
          f"L6 {sum(r['has_l6'] for r in rows)}, lumbar rib {sum(r['has_lumbar_rib'] for r in rows)}; "
          f"CT missing on host: {len(missing)}" + (f" e.g. {missing[:3]}" if missing else ""))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
