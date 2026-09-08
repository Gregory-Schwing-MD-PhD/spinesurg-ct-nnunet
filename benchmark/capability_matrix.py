#!/usr/bin/env python3
"""benchmark/capability_matrix.py — what each system can say about a spine, as a matrix figure
(solid black cells) and a LaTeX table. Rows are the things a surgeon needs at the junctions;
columns are the systems. This is the "why" figure: no competitor can name an L6, a T13 and a
lumbar rib, none reports a posterior over the count, and none was trained on transitional
anatomy that is labelled as such.

    python benchmark/capability_matrix.py --out figs/fig_capabilities --tex figs/tab_capabilities.tex
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SYSTEMS = ["TotalSeg. v2", "NV-Segment-CT\n(VISTA3D)", "SPINEPS-CT\n+ VERIDAH", "RibSeg v2", "ours\n(one-shot + decoder)"]
SYSTEMS_TEX = ["TotalSeg.~v2", "NV-Segment-CT", "SPINEPS+VERIDAH", "RibSeg~v2", "\\textbf{ours}"]
ROWS = [
    ("vertebrae T10–L5 by name",            [1, 1, 1, 0, 1]),
    ("sacrum (and S1)",                          [1, 1, 1, 0, 1]),
    ("hips and femora",                          [1, 1, 0, 0, 1]),
    ("ribs, numbered per side",                  [1, 1, 0, 1, 2]),   # 2 = the labels have it (13 per side); this arm pools 1-11, the numbered-rib arm is queued
    ("twelfth rib as its own class",             [1, 1, 0, 1, 1]),
    ("a rib on L1 (lumbar rib)",                 [0, 0, 0, 0, 1]),
    ("a sixth lumbar body (L6)",                 [0, 0, 1, 0, 1]),
    ("a thirteenth thoracic body (T13)",         [0, 0, 1, 0, 1]),
    ("voxel probabilities released",             [0, 0, 0, 0, 1]),
    ("posterior over the rib-free count",        [0, 0, 0, 0, 1]),
    ("readings A / B / C for six bodies",        [0, 0, 0, 0, 1]),
    ("trained on labelled transitional anatomy", [0, 0, 1, 0, 1]),
    ("public CT test set with LSTV strata",      [0, 0, 0, 0, 1]),
]
NOTES = {"ribs, numbered per side": "ours pools ribs 1–11 by design",
         "a rib on L1 (lumbar rib)": "Möller's assignment reaches L1 but no lower"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tex", required=True, type=Path)
    a = ap.parse_args()
    M = np.array([[1 if x == 1 else 0 for x in r[1]] for r in ROWS])
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.imshow(1 - M, cmap="gray", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(SYSTEMS))); ax.set_xticklabels(SYSTEMS, fontsize=7.5)
    ax.set_yticks(range(len(ROWS))); ax.set_yticklabels([r[0] for r in ROWS], fontsize=8)
    ax.xaxis.tick_top()
    for i, r in enumerate(ROWS):                                   # hatched: in the labels, arm queued
        for j, x in enumerate(r[1]):
            if x == 2:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="////", ec="k", lw=0))
    ax.set_xticks(np.arange(-0.5, len(SYSTEMS)), minor=True); ax.set_yticks(np.arange(-0.5, len(ROWS)), minor=True)
    ax.grid(which="minor", color="w", lw=2); ax.grid(which="major", visible=False)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # outline the block that is ours alone
    ours_only = [i for i, r in enumerate(ROWS) if r[1][-1] == 1 and sum(r[1][:-1]) == 0]
    if ours_only:
        y0, y1 = min(ours_only) - 0.5, max(ours_only) + 0.5
        ax.add_patch(plt.Rectangle((len(SYSTEMS) - 1.5, y0), 1, y1 - y0, fill=False, lw=1.6, ec="k"))
        ax.text(len(SYSTEMS) - 0.38, (y0 + y1) / 2, "ours alone", rotation=-90, va="center", ha="left", fontsize=7.5, clip_on=False)
    fig.text(0.01, 0.01, "hatched: in the released labels (thirteen ribs per side); the arm that names them is queued behind the one-shot folds", fontsize=6.5, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out.with_suffix(".pdf")); fig.savefig(a.out.with_suffix(".png"), dpi=200)

    lines = ["\\begin{tabular}{l" + "c" * len(SYSTEMS) + "}", "\\toprule",
             " & " + " & ".join(SYSTEMS_TEX) + " \\\\", "\\midrule"]
    for name, v in ROWS:
        lines.append(name.replace("–", "--").replace("ö", "\\\"o") + " & " +
                     " & ".join("$\\blacksquare$" if x else "" for x in v) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    a.tex.write_text("\n".join(lines) + "\n")
    print("wrote", a.out.with_suffix(".pdf"), a.tex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
