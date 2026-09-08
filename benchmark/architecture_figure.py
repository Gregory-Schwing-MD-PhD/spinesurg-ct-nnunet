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


def cuboid(ax, x, y, w, h, d, fc="w", ec="k", lw=0.8, label=None, sub=None):
    """A box in cavalier projection: front face (w x h) at (x, y), depth d drawn up-right."""
    k = 0.45
    front = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    top = [(x, y + h), (x + w, y + h), (x + w + d * k, y + h + d * k), (x + d * k, y + h + d * k)]
    side = [(x + w, y), (x + w + d * k, y + d * k), (x + w + d * k, y + h + d * k), (x + w, y + h)]
    for poly, shade in ((side, "0.75"), (top, "0.88"), (front, fc)):
        ax.add_patch(Polygon(poly, closed=True, fc=shade if poly is not front else fc, ec=ec, lw=lw))
    if label:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=6.5)
    if sub:
        ax.text(x + w / 2, y - 0.12, sub, ha="center", va="top", fontsize=5.8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    ax.set_xlim(-0.2, 15.2); ax.set_ylim(-1.35, 4.6); ax.axis("off")

    n = len(FEATS)
    spatial = [tuple(p // (2 ** i) for p in PATCH) for i in range(n)]
    # encoder boxes: height shrinks with spatial size (log scale), depth grows with channels
    enc_x, y0 = [], 0.0
    x = 0.9
    for i in range(n):
        h = 2.6 * (0.62 ** i) + 0.25
        d = 0.25 + 0.9 * (FEATS[i] / 320)
        w = 0.32 + 0.05 * BLOCKS[i]
        cuboid(ax, x, y0, w, h, d, label=f"{FEATS[i]}", sub=f"{spatial[i][0]}×{spatial[i][1]}×{spatial[i][2]}\n{BLOCKS[i]} res. block{'s' if BLOCKS[i] > 1 else ''}")
        enc_x.append((x, w, h, d))
        x += w + 0.55 + d * 0.45
    # decoder boxes mirror the encoder (one conv block per stage)
    dec_x = []
    for i in range(n - 2, -1, -1):
        h = 2.6 * (0.62 ** i) + 0.25
        d = 0.25 + 0.9 * (FEATS[i] / 320)
        w = 0.37
        cuboid(ax, x, y0, w, h, d, label=f"{FEATS[i]}", sub=f"{spatial[i][0]}×{spatial[i][1]}×{spatial[i][2]}")
        dec_x.append((i, x, w, h, d))
        x += w + 0.55 + d * 0.45
    # input and output slabs
    cuboid(ax, -0.1, y0, 0.18, 2.85, 0.35, fc="0.93", label="", sub="CT patch\n384×256×256\n0.80×0.75×0.75 mm")
    ax.text(0.0, 3.05, "1 ch", ha="center", fontsize=6)
    out_x = x
    cuboid(ax, out_x, y0, 0.18, 2.85, 0.35, fc="0.93", label="", sub=f"softmax, {N_OUT} ch\n384×256×256")
    # skip connections
    for (i, dx, dw, dh, dd) in dec_x:
        ex, ew, eh, ed = enc_x[i]
        yy = y0 + eh + ed * 0.45 + 0.08
        ax.add_patch(FancyArrowPatch((ex + ew + ed * 0.45, yy), (dx, yy), arrowstyle="-|>", mutation_scale=6,
                                     lw=0.6, color="0.4", ls=(0, (2, 2)), connectionstyle="arc3,rad=-0.15"))
    ax.text(enc_x[3][0] + 0.2, 3.95, "skip connections (concatenate)", fontsize=6, color="0.35")
    # deep supervision heads on the four finest decoder stages
    for (i, dx, dw, dh, dd) in dec_x:
        if i <= 3:
            ax.annotate("", xy=(dx + dw / 2, y0 - 0.55), xytext=(dx + dw / 2, y0), arrowprops=dict(arrowstyle="-|>", lw=0.6, color="0.4", mutation_scale=6))
    ax.text(dec_x[-1][1] - 1.7, y0 - 0.72, "deep supervision: class-weighted Dice + CE at four scales", fontsize=6, color="0.35", ha="left", va="top")
    # sequence decoder block
    sx = out_x + 0.75
    cuboid(ax, sx, 0.15, 1.9, 2.3, 0.3, fc="w", lw=1.2, label="sequence decoder\n\ninstances (disc-space cut)\n→ type posteriors\n→ monotone DP\n→ rib-free count,\nreadings A / B / C", sub="a posterior, not a verdict")
    ax.add_patch(FancyArrowPatch((out_x + 0.35, 1.3), (sx, 1.3), arrowstyle="-|>", mutation_scale=8, lw=0.9, color="k"))
    ax.text(0.9, 4.3, "residual encoder, 7 stages, stride 2, 3³ kernels, instance norm", fontsize=6.5)
    ax.text(dec_x[0][1], 4.3, "decoder, transposed conv + 1 block per stage", fontsize=6.5)
    ax.text(-0.1, -1.25, "ResEnc-L (nnU-Net v2), 100 GB plan on an H200; batch 2; SGD Nesterov 0.99, poly LR 1e-2, 500 epochs; "
            "sampler: half of every queue from the 39 variant carriers, patch forced onto L6 or the lumbar rib", fontsize=6, va="bottom")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out.with_suffix(".pdf"), bbox_inches="tight"); fig.savefig(a.out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print("wrote", a.out.with_suffix(".pdf"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
