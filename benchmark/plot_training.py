#!/usr/bin/env python3
"""benchmark/plot_training.py — the training-time evidence that the rare classes are learned:
(a) L6 Dice on the six-lumbar validation cases per epoch and fold (the dedicated LSTV pass),
against the collapse fraction (GT-L6 voxels called L5); (b) per-class pseudo-Dice at the
latest epoch, typical classes beside the rare ones.

    python benchmark/plot_training.py --csv benchmark/data/training_series.csv --out figs/fig_training
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})
GROUPS = [("thoracic", ["T10", "T11", "T12"]), ("lumbar", ["L1", "L2", "L3", "L4", "L5"]),
          ("rare", ["L6", "lumbar_rib_left", "lumbar_rib_right", "rib12_left", "rib12_right"]),
          ("pelvis", ["sacrum", "left_hip", "right_hip", "femur"]), ("ribs 1-11", ["rib_left", "rib_right"])]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="stem; writes .pdf and .png")
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    folds = sorted(df.fold.unique())
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [1.1, 1]})

    ax = axes[0]
    for f in folds:
        d = df[(df.fold == f) & df.l6_dice_lumb.notna()]
        if len(d):
            s = d.l6_dice_lumb.rolling(3, min_periods=1, center=True).mean()
            ax.plot(d.epoch, s, lw=1.4, label=f"fold {f}")
    d = df[df.l6_to_l5_pct.notna()].groupby("epoch").l6_to_l5_pct.mean() / 100
    if len(d):
        ax.plot(d.index, d.values, color="k", ls="--", lw=1.2, label="GT-L6 called L5 (mean)")
    ax.set_xlabel("epoch"); ax.set_ylabel("L6 Dice, six-lumbar validation cases")
    ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=7, ncol=1, loc="upper left")
    ax.set_title("(a) L6 is learned as its own class", fontsize=9, loc="left")

    ax = axes[1]
    last = df[df["pdice_L1"].notna()].sort_values("epoch").groupby("fold").tail(1)   # latest COMPLETED epoch per fold
    # the summary line's pseudo-Dice comes from 50 random validation patches, so a class carried
    # by four validation records is rarely inside them: L6 is taken from the dedicated pass
    # instead, and the lumbar ribs are marked as not measurable from patches
    l6_last = df[df.l6_dice_lumb.notna()].sort_values("epoch").groupby("fold").tail(1).l6_dice_lumb.mean()
    labels, vals, xs, x, special = [], [], [], 0, {}
    for gname, cls in GROUPS:
        for c in cls:
            col = f"pdice_{c}"
            v = pd.to_numeric(last[col], errors="coerce") if col in last else pd.Series(dtype=float)
            val = v.mean() if v.notna().any() else np.nan
            if c == "L6":
                val = l6_last; special[x] = "dedicated pass"
            elif c.startswith("lumbar_rib") and (np.isnan(val) or val == 0):
                val = np.nan; special[x] = "not in patches"
            vals.append(val); labels.append(c.replace("_", " ")); xs.append(x); x += 1
        x += 0.6
    colors = ["0.35" if not lab.startswith(("L6", "lumbar", "rib12")) else "k" for lab in labels]
    ax.bar(xs, vals, color=colors, width=0.8)
    for xi, note in special.items():
        ax.text(xi, 0.03 if np.isnan(vals[xs.index(xi)]) else vals[xs.index(xi)] + 0.03, note, ha="center", va="bottom", fontsize=5.5)
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    ax.set_ylim(0, 1); ax.set_ylabel("Dice, mean over folds")
    ep = int(last.epoch.min()), int(last.epoch.max())
    ax.set_title(f"(b) per-class Dice, epochs {ep[0]}–{ep[1]} of 500", fontsize=9, loc="left")
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out.with_suffix(".pdf")); fig.savefig(a.out.with_suffix(".png"), dpi=200)
    print("wrote", a.out.with_suffix(".pdf"), "| folds", folds, "| epochs", ep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
