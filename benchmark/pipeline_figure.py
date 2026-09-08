#!/usr/bin/env python3
"""benchmark/pipeline_figure.py — the "what" figure: the one-shot arm and the decoder as a
schematic, next to the merged baseline and a competitor, so the difference is visible in
one glance: names come from local appearance, the count is inferred with a posterior.

    python benchmark/pipeline_figure.py --out figs/fig_pipeline_oneshot
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "sans-serif", "font.size": 8})


def box(ax, x, y, w, h, text, fc="w", lw=1.0, fs=7.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec="k", lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.15)


def arrow(ax, x0, y0, x1, y1, text=None):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=0.9, color="k", shrinkA=0, shrinkB=0))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.12, text, ha="center", va="bottom", fontsize=6.5)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.set_xlim(0, 13); ax.set_ylim(-0.3, 6.5); ax.axis("off")

    rows = {"comp": 5.4, "base": 3.5, "ours": 1.3}
    for key, lab in (("comp", "a competitor\n(fixed vocabulary)"), ("base", "our merged baseline\n(May 2026)"),
                     ("ours", "ours: one-shot identity\n+ sequence decoder")):
        ax.text(0.05, rows[key] + 0.45, lab, ha="left", va="center", fontsize=7.5, fontweight="bold")

    # shared input
    box(ax, 2.35, 3.5, 1.0, 0.9, "CT", fc="0.92")
    xa, xb = 3.35, 4.2

    y = rows["comp"]
    box(ax, xb, y, 2.7, 0.9, "semantic net\nC1–L5, ribs 1–12", fc="w")
    box(ax, 7.4, y, 5.3, 0.9, "L5 is the last name it has: an L6 becomes L5,\na lumbar rib becomes rib 12, the count is silent", fc="0.96", fs=7)
    arrow(ax, xa, 3.95 + 0.25, xb, y + 0.45); arrow(ax, 6.9, y + 0.45, 7.4, y + 0.45)

    y = rows["base"]
    box(ax, xb, y, 2.7, 0.9, "semantic net\nL1–L4, last lumbar, sacrum", fc="w", fs=6.6)
    box(ax, 7.4, y, 5.3, 0.9, "never shifts a level, by construction, but cannot\nsay whether the bottom body is L5, L6 or S1", fc="0.96", fs=7)
    arrow(ax, xa, 3.95, xb, y + 0.45); arrow(ax, 6.9, y + 0.45, 7.4, y + 0.45)

    y = rows["ours"]
    box(ax, xb, y - 0.2, 3.15, 1.3, "semantic net, 22 classes\nT10–T13, L1–L6, sacrum,\nribs 1–11 | 12 | 13 | lumbar rib,\nhips, femur", fc="w", lw=1.4, fs=6.2)
    box(ax, 7.75, y - 0.2, 2.35, 1.3, "instances and\ntype posteriors\nthoracic / lumbar / sacral\n+ rib contact", fc="w", fs=6.2)
    box(ax, 10.45, y - 0.2, 2.5, 1.3, "monotone decode:\nposterior over the count,\nreadings A / B / C,\ndouble shift reported", fc="w", lw=1.4, fs=6.2)
    arrow(ax, xa, 3.95 - 0.25, xb, y + 0.45); arrow(ax, 7.35, y + 0.45, 7.75, y + 0.45); arrow(ax, 10.1, y + 0.45, 10.45, y + 0.45)
    ax.text(7.55, y + 1.2, "softmax", ha="center", va="bottom", fontsize=6.3)
    ax.text(xb, 0.55, "sampler: half of every queue from the 39 variant carriers; the patch is forced onto L6 or the lumbar rib",
            fontsize=6.3, ha="left", va="top", style="italic")
    ax.text(xb, 0.2, "names come from local appearance (T12 vs L1: AUC 0.99 on shape alone); only the count is inferred, with a probability",
            fontsize=6.3, ha="left", va="top", style="italic")
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out.with_suffix(".pdf")); fig.savefig(a.out.with_suffix(".png"), dpi=200)
    print("wrote", a.out.with_suffix(".pdf"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
