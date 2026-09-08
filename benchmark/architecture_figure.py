#!/usr/bin/env python3
"""benchmark/architecture_figure.py — the network as a DL paper draws it (after Krizhevsky
et al. 2012, Fig. 2): one box per stage with its spatial extent and channel count, the
residual encoder on the left, the decoder on the right, skip connections across, deep
supervision heads, the 44-way softmax, and the sequence decoder that turns voxels into a
count with a posterior. Numbers come from the ResEnc-L plan actually trained
(nnUNetResEncUNetLPlans_100G, patch reshaped to 384 x 256 x 256).

    python benchmark/architecture_figure.py --out figs/fig_architecture
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch

plt.rcParams.update({"font.family": "sans-serif", "font.size": 7})

PATCH = (384, 256, 256)
FEATS = [32, 64, 128, 256, 320, 320, 320]
BLOCKS = [1, 3, 4, 6, 6, 6, 6]
N_OUT = 44
K = 0.45                      # cavalier depth factor
NL = chr(10)


def cuboid(ax, x, y, w, h, d, fc="w", lw=0.8, label=None):
    front = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    top = [(x, y + h), (x + w, y + h), (x + w + d * K, y + h + d * K), (x + d * K, y + h + d * K)]
    side = [(x + w, y), (x + w + d * K, y + d * K), (x + w + d * K, y + h + d * K), (x + w, y + h)]
    ax.add_patch(Polygon(side, closed=True, fc="0.75", ec="k", lw=lw))
    ax.add_patch(Polygon(top, closed=True, fc="0.88", ec="k", lw=lw))
    ax.add_patch(Polygon(front, closed=True, fc=fc, ec="k", lw=lw))
    if label:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=6.2, linespacing=1.25)


def under(ax, xc, text, row):
    """A label below the baseline, on one of two rows so neighbours never collide."""
    ax.text(xc, -0.12 - 0.5 * row, text, ha="center", va="top", fontsize=5.6, linespacing=1.1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    fig, ax = plt.subplots(figsize=(10.8, 4.5))
    ax.set_ylim(-1.75, 4.8); ax.axis("off")
    n = len(FEATS)
    spatial = [tuple(p // (2 ** i) for p in PATCH) for i in range(n)]
    dims = ["×".join(str(v) for v in sp) for sp in spatial]
    y0 = 0.0

    # input slab, labelled above so the first stage label has the ground to itself
    cuboid(ax, -0.1, y0, 0.18, 2.85, 0.35, fc="0.93")
    ax.text(0.05, 3.15, "CT patch, 1 channel" + NL + dims[0] + NL + "0.80 × 0.75 × 0.75 mm", ha="center", va="bottom", fontsize=5.8)

    # encoder
    enc, x = [], 1.2
    for i in range(n):
        h = 2.6 * (0.62 ** i) + 0.25
        d = 0.25 + 0.9 * (FEATS[i] / 320)
        w = 0.32 + 0.05 * BLOCKS[i]
        cuboid(ax, x, y0, w, h, d, label=str(FEATS[i]))
        blocks = "{} res. block{}".format(BLOCKS[i], "s" if BLOCKS[i] > 1 else "")
        under(ax, x + w / 2, dims[i] + NL + blocks, i % 2)
        enc.append((x, w, h, d))
        x += w + 0.55 + d * K
    # decoder
    dec = []
    for i in range(n - 2, -1, -1):
        h = 2.6 * (0.62 ** i) + 0.25
        d = 0.25 + 0.9 * (FEATS[i] / 320)
        w = 0.37
        cuboid(ax, x, y0, w, h, d, label=str(FEATS[i]))
        under(ax, x + w / 2, dims[i], i % 2)
        dec.append((i, x, w, h, d))
        x += w + 0.55 + d * K
    # output slab
    out_x = x
    cuboid(ax, out_x, y0, 0.18, 2.85, 0.35, fc="0.93")
    ax.text(out_x + 0.1, 3.1, "softmax" + NL + "{} channels".format(N_OUT) + NL + dims[0], ha="center", va="bottom", fontsize=5.8)

    # skip connections
    for (i, dx, dw, dh, dd) in dec:
        ex, ew, eh, ed = enc[i]
        yy = y0 + eh + ed * K + 0.08
        ax.add_patch(FancyArrowPatch((ex + ew + ed * K, yy), (dx, yy), arrowstyle="-|>", mutation_scale=6,
                                     lw=0.6, color="0.4", ls=(0, (2, 2)), connectionstyle="arc3,rad=-0.15"))
    ax.text(enc[3][0] + 0.2, 4.05, "skip connections (concatenate)", fontsize=6, color="0.35")
    # deep supervision on the four finest decoder stages
    for (i, dx, dw, dh, dd) in dec:
        if i <= 3:
            ax.annotate("", xy=(dx + dw / 2, y0 - 1.15), xytext=(dx + dw / 2, y0 - 0.02),
                        arrowprops=dict(arrowstyle="-|>", lw=0.6, color="0.4", mutation_scale=6))
    ax.text(dec[-1][1] + 0.5, y0 - 1.18, "deep supervision: class-weighted Dice + cross-entropy at four scales",
            fontsize=5.8, color="0.35", ha="left", va="top")

    # sequence decoder: the box is sized from the MEASURED text extent, with padding
    sx = out_x + 1.05
    dec_text = ("sequence decoder" + NL + NL + "instances from the disc-space cut" + NL + "type posterior per body" + NL
                + "monotone dynamic program" + NL + "count posterior, readings A / B / C")
    ax.set_xlim(-0.45, sx + 4.6)
    fig.canvas.draw()
    t = ax.text(sx, 1.3, dec_text, ha="left", va="center", fontsize=6.2, linespacing=1.25)
    bb = t.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(ax.transData.inverted())
    pad = 0.3
    bw = (bb.x1 - bb.x0) + 2 * pad
    t.remove()
    cuboid(ax, sx, 0.15, bw, 2.3, 0.3, lw=1.2, label=dec_text)
    under(ax, sx + bw / 2, "a posterior, not a verdict", 0)
    ax.add_patch(FancyArrowPatch((out_x + 0.35, 1.3), (sx, 1.3), arrowstyle="-|>", mutation_scale=8, lw=0.9, color="k"))
    ax.set_xlim(-0.45, sx + bw + 0.5)

    ax.text(1.2, 4.45, "residual encoder: 7 stages, stride 2, 3³ kernels, instance normalisation", fontsize=6.5)
    ax.text(dec[0][1], 4.45, "decoder: transposed convolution + 1 block per stage", fontsize=6.5)
    ax.text(-0.1, -1.72, "ResEnc-L (nnU-Net v2), 100 GB plan on an H200; batch 2; SGD with Nesterov momentum 0.99, polynomial LR from 1e-2, 500 epochs" + NL
            + "sampler: half of every queue from the 39 variant carriers, patch forced onto L6 or the lumbar rib",
            fontsize=6, va="bottom", linespacing=1.3)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(a.out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print("wrote", a.out.with_suffix(".pdf"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
