#!/usr/bin/env python3
"""benchmark/plot_training.py — the training-time evidence that the rare classes are learned:
(a) L6 Dice on the six-lumbar validation cases per epoch and fold (the dedicated LSTV pass),
against the collapse fraction (GT-L6 voxels called L5); (b) per-class Dice at the latest
completed epoch, typical classes beside the rare ones.

    python benchmark/plot_training.py --csv benchmark/data/training_series.csv --out figs/fig_training
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titlesize": 8.5, "axes.titleweight": "bold",
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7})
GROUPS = [("thoracic", ["T10", "T11", "T12"]), ("lumbar", ["L1", "L2", "L3", "L4", "L5", "L6"]),
          ("ribs", ["rib_left", "rib_right", "rib12_left", "rib12_right", "lumbar_rib_left", "lumbar_rib_right"]),
          ("pelvis", ["sacrum", "left_hip", "right_hip", "femur"])]
PRETTY = {"rib_left": "ribs 1–11 L", "rib_right": "ribs 1–11 R", "rib12_left": "rib 12 L", "rib12_right": "rib 12 R",
          "lumbar_rib_left": "lumbar rib L", "lumbar_rib_right": "lumbar rib R", "left_hip": "hip L", "right_hip": "hip R"}
FOLD_COLORS = ["#1f4e79", "#2e86ab", "#7fb3d5", "#a6a6a6"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="stem; writes .pdf and .png")
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    folds = sorted(df.fold.unique())
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.4, 3.3), gridspec_kw={"width_ratios": [1.0, 1.15], "wspace": 0.32})

    # (a) L6 on the six-lumbar validation cases, dedicated pass, 3-epoch running mean
    for k, f in enumerate(folds):
        d = df[(df.fold == f) & df.l6_dice_lumb.notna()]
        if len(d):
            ax.plot(d.epoch, d.l6_dice_lumb.rolling(3, min_periods=1, center=True).mean(),
                    lw=1.3, color=FOLD_COLORS[k % len(FOLD_COLORS)], label=f"fold {f}")
    d = df[df.l6_to_l5_pct.notna()].groupby("epoch").l6_to_l5_pct.mean() / 100
    if len(d):
        ax.plot(d.index, d.rolling(3, min_periods=1, center=True).mean().values, color="k", ls="--", lw=1.0,
                label="L6 voxels called L5 (mean)")
    ax.set_xlabel("epoch"); ax.set_ylabel("Dice on six-lumbar validation cases")
    ax.set_ylim(0, 1); ax.set_xlim(left=0)
    ax.legend(frameon=False, loc="upper left", ncol=1, handlelength=1.6)
    ax.set_title("a  L6 learned as its own class", loc="left")

    # (b) per-class Dice at the latest completed epoch of each fold, mean over folds
    last = df[df["pdice_L1"].notna()].sort_values("epoch").groupby("fold").tail(1)
    l6_last = df[df.l6_dice_lumb.notna()].sort_values("epoch").groupby("fold").tail(1).l6_dice_lumb.mean()
    xs, vals, labels, kinds, x = [], [], [], [], 0.0
    for gname, cls in GROUPS:
        for c in cls:
            col = f"pdice_{c}"
            v = pd.to_numeric(last[col], errors="coerce") if col in last else pd.Series(dtype=float)
            val = float(v.mean()) if v.notna().any() else np.nan
            kind = "patch"
            if c == "L6":
                val, kind = l6_last, "dedicated"
            elif c.startswith("lumbar_rib") and (np.isnan(val) or val == 0):
                val, kind = np.nan, "na"
            xs.append(x); vals.append(val); labels.append(PRETTY.get(c, c)); kinds.append(kind); x += 1
        x += 0.7
    for xi, v, kind in zip(xs, vals, kinds):
        if kind == "patch":
            bx.bar(xi, v, width=0.78, color="#4d4d4d")
        elif kind == "dedicated":
            bx.bar(xi, v, width=0.78, color="#1f4e79")
        else:
            bx.plot(xi, 0.02, marker="x", color="k", ms=5, mew=1.2, ls="none")
    bx.set_xticks(xs); bx.set_xticklabels(labels, rotation=60, ha="right")
    bx.set_ylim(0, 1.22); bx.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); bx.set_ylabel("Dice, mean over folds")
    ep = int(last.epoch.min()), int(last.epoch.max())
    bx.set_title(f"b  per-class Dice at epochs {ep[0]}–{ep[1]} of 500", loc="left")
    handles = [Patch(color="#4d4d4d", label="random validation patches"),
               Patch(color="#1f4e79", label="dedicated pass on whole cases"),
               Line2D([], [], marker="x", color="k", ls="none", ms=5, mew=1.2, label="not measurable from patches")]
    bx.legend(handles=handles, frameon=False, loc="upper left", handlelength=1.2, ncol=1, borderaxespad=0.2)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.3)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out.with_suffix(".pdf")); fig.savefig(a.out.with_suffix(".png"), dpi=220)
    print("wrote", a.out.with_suffix(".pdf"), "| folds", folds, "| epochs", ep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
