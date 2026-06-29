"""eval_xrsp1k_seg.py — evaluate the XRSpinoPelvic1K 2D segmenter two ways:
  (1) per-class Dice of predicted vs ground-truth masks on the held-out test split, and
  (2) the spinopelvic ANGLE error when ostk.metrics2d is run through the PREDICTED masks
      instead of the GT masks (the end-to-end film -> angle error the paper reports).

Predictions and labels are nnU-Net consecutive ids; label_map.json maps them back to v4 ids
so ostk.metrics2d (which keys on structure names) can measure angles. We monkeypatch
ostk.labels.LABELS to the v4 scheme at runtime, so nothing in the installed ostk (the live
demo runs the v3 scheme) is modified on disk.

  python tools/eval_xrsp1k_seg.py \
      --pred  <nnUNet predictions dir> \
      --gt    $nnUNet_raw/Dataset910_XRSpinoPelvic1K/labelsTs \
      --label_map $nnUNet_raw/Dataset910_XRSpinoPelvic1K/label_map.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def v4_label_dict() -> dict:
    """structure name -> v4 VerSe-native id (what ostk.metrics2d needs to key on)."""
    d = {}
    verse = (["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
             + [f"T{n}" for n in range(1, 13)] + [f"L{n}" for n in range(1, 7)])
    for i, nm in enumerate(verse, start=1):
        d[nm] = i
    d["sacrum"] = 26; d["coccyx"] = 27; d["T13"] = 28; d["S1"] = 29
    d["left_hip"] = 30; d["right_hip"] = 31; d["femur_left"] = 32; d["femur_right"] = 33
    for n in range(1, 13):
        d[f"rib_left_{n}"] = 33 + n
    for n in range(1, 13):
        d[f"rib_right_{n}"] = 45 + n
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, type=Path, help="nnU-Net predictions dir (*.png)")
    ap.add_argument("--gt", required=True, type=Path, help="labelsTs dir (*.png)")
    ap.add_argument("--label_map", required=True, type=Path, help="Dataset.../label_map.json")
    a = ap.parse_args()

    nn2v4 = {int(k): v["v4_id"] for k, v in json.loads(a.label_map.read_text())["nnunet_to_v4"].items()}
    name_by_nn = {int(k): v["name"] for k, v in json.loads(a.label_map.read_text())["nnunet_to_v4"].items()}
    cases = sorted(p.stem for p in a.gt.glob("*.png") if (a.pred / p.name).exists())
    if not cases:
        raise SystemExit("no overlapping pred/gt cases found")

    # ---- (1) per-class Dice ----
    inter = {k: 0.0 for k in nn2v4}
    denom = {k: 0.0 for k in nn2v4}
    for c in cases:
        g = np.array(Image.open(a.gt / f"{c}.png"))
        p = np.array(Image.open(a.pred / f"{c}.png"))
        for k in nn2v4:
            gk, pk = g == k, p == k
            inter[k] += 2.0 * np.count_nonzero(gk & pk)
            denom[k] += np.count_nonzero(gk) + np.count_nonzero(pk)
    dice = {k: (inter[k] / denom[k]) for k in nn2v4 if denom[k] > 0}
    mean_dice = float(np.mean(list(dice.values()))) if dice else 0.0
    print(f"=== Dice ({len(cases)} test cases) ===  mean={mean_dice:.3f}")
    for k in sorted(dice, key=lambda k: name_by_nn[k]):
        print(f"  {name_by_nn[k]:14s} {dice[k]:.3f}")

    # ---- (2) angle error through predicted vs GT masks (ostk.metrics2d, v4-patched) ----
    try:
        import ostk.labels as L
        L.LABELS = v4_label_dict()                            # runtime patch (disk ostk untouched)
        from ostk.metrics2d import spinopelvic_summary_from_mask_2d
    except Exception as exc:                                  # noqa: BLE001
        print(f"\n[angles] skipped — could not import ostk ({exc}). pip install the engine to enable.")
        return 0

    inv = {nn: v4 for nn, v4 in nn2v4.items()}               # nn id -> v4 id
    def to_v4(mask):
        out = np.zeros_like(mask, dtype=np.int16)
        for nn, v4 in inv.items():
            out[mask == nn] = v4
        return out

    diffs = {k: [] for k in ("PI", "SS", "PT", "LL")}
    for c in cases:
        gp = spinopelvic_summary_from_mask_2d(to_v4(np.array(Image.open(a.gt / f"{c}.png"))), case_id=c)
        pp = spinopelvic_summary_from_mask_2d(to_v4(np.array(Image.open(a.pred / f"{c}.png"))), case_id=c)
        for k in diffs:
            if gp.get(k) is not None and pp.get(k) is not None:
                diffs[k].append(abs(gp[k] - pp[k]))
    print("\n=== 2D angle error: predicted-mask vs GT-mask (deg) ===")
    for k in ("PI", "SS", "PT", "LL"):
        v = diffs[k]
        if v:
            print(f"  {k}: MAE {np.mean(v):.2f}  median {np.median(v):.2f}  max {np.max(v):.2f}  (n={len(v)})")
        else:
            print(f"  {k}: no comparable cases")
    print("\nNote: this isolates the segmentation-induced angle error. To report error vs the "
          "3D ground truth, also compare against the engine's 3D values per case.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
