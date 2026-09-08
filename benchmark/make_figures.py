#!/usr/bin/env python3
"""benchmark/make_figures.py — result figures across systems from $BENCH/<system>/score/
(summary.json from tools/eval_oneshot.py, decode.csv from tools/decode_sequence.py).
Runs on whatever systems exist; re-run as each one lands.

  fig_bench_dice     per-class Dice, typical (grey) and transitional (black) records, one panel per system
  fig_bench_count    rib-free count exact-match rate by transitional status, per system
  fig_bench_ribs     where the ground-truth twelfth-rib and lumbar-rib voxels went, per system
  fig_bench_rare     rare classes: emitted on the records that carry them, and where absent

    python benchmark/make_figures.py --bench ~/bench --out figs [--systems totalsegmentator ours_f2 ...]
Copy $BENCH/*/score to a local dir first if running off the grid.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})
CLASSES = ["T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5", "L6", "sacrum", "rib_left", "rib_right",
           "rib12_left", "rib12_right", "lumbar_rib_left", "lumbar_rib_right", "left_hip", "right_hip", "femur"]
PRETTY = {"totalsegmentator": "TotalSegmentator v2", "vista3d": "NV-Segment-CT", "spineps": "SPINEPS + VERIDAH",
          "gt": "ground truth (check)"}


def pretty(s: str) -> str:
    if s.startswith("ours"):
        return "ours" + (f" ({s[4:].strip('_')})" if len(s) > 4 else "")
    return PRETTY.get(s, s)


def load(bench: Path, systems):
    out = {}
    for s in systems:
        p = bench / s / "score" / "summary.json"
        if p.exists():
            d = json.loads(p.read_text())
            dec = bench / s / "score" / "decode.csv"
            if dec.exists():
                rows = [r for r in csv.DictReader(open(dec)) if r.get("count_error", "") != ""]
                d["decoder_exact"] = (sum(1 for r in rows if r["count_error"] == "0") / len(rows)) if rows else None
            out[s] = d
    return out


def fig_dice(data, out):
    n = len(data)
    fig, axes = plt.subplots(n, 1, figsize=(7.2, 1.6 * n + 0.6), sharex=True, squeeze=False)
    x = np.arange(len(CLASSES))
    for ax, (s, d) in zip(axes[:, 0], data.items()):
        typ = [d["dice_mean"]["typical"].get(c, np.nan) for c in CLASSES]
        tr = [d["dice_mean"]["transitional"].get(c, np.nan) for c in CLASSES]
        ax.bar(x - 0.2, typ, 0.4, color="0.6", label="typical records")
        ax.bar(x + 0.2, tr, 0.4, color="k", label="transitional records")
        ax.set_ylim(0, 1); ax.set_ylabel("Dice"); ax.set_title(pretty(s), loc="left", fontsize=9)
        for i, c in enumerate(CLASSES):                      # a class the system cannot name
            if np.isnan(typ[i]) and np.isnan(tr[i]):
                ax.text(i, 0.04, "×", ha="center", fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=7.5, ncol=2, loc="upper right")
    axes[-1, 0].set_xticks(x); axes[-1, 0].set_xticklabels([c.replace("_", " ") for c in CLASSES], rotation=70, ha="right", fontsize=7)
    fig.tight_layout(); fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"), dpi=200); plt.close(fig)


def fig_count(data, out):
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    names = list(data); x = np.arange(len(names))
    for k, (grp, col, off) in enumerate((("typical", "0.6", -0.27), ("transitional", "k", 0.0))):
        ax.bar(x + off, [100 * data[s]["rib_free_count"][grp]["exact"] for s in names], 0.27, color=col, label=f"{grp} records")
    dec = [100 * data[s]["decoder_exact"] if data[s].get("decoder_exact") is not None else np.nan for s in names]
    ax.bar(x + 0.27, dec, 0.27, color="w", edgecolor="k", hatch="//", label="decoder, all records")
    ax.set_xticks(x); ax.set_xticklabels([pretty(s) for s in names], fontsize=7.5)
    ax.set_ylim(0, 100); ax.set_ylabel("rib-free count exact (%)")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"), dpi=200); plt.close(fig)


def fig_ribs(data, out):
    keys = ["rib12_left", "rib12_right", "lumbar_rib_left", "lumbar_rib_right"]
    dest_order = ["rib12_left", "rib12_right", "lumbar_rib_left", "lumbar_rib_right", "rib_left", "rib_right", "background"]
    shades = {"rib12_left": "0.15", "rib12_right": "0.3", "lumbar_rib_left": "0.0", "lumbar_rib_right": "0.1",
              "rib_left": "0.6", "rib_right": "0.7", "background": "0.9"}
    n = len(data)
    fig, axes = plt.subplots(1, n, figsize=(2.2 * n + 1.2, 2.8), sharey=True, squeeze=False)
    for ax, (s, d) in zip(axes[0], data.items()):
        conf = d.get("rib_confusion_pct_of_gt_voxels", {})
        for i, k in enumerate(keys):
            dist = conf.get(k, {}); bottom = 0
            for dest in dest_order + [x for x in dist if x not in dest_order]:
                v = dist.get(dest, 0)
                if v <= 0:
                    continue
                ax.bar(i, v, bottom=bottom, color=shades.get(dest, "0.5"), edgecolor="w", lw=0.5, width=0.7)
                if v >= 12:
                    ax.text(i, bottom + v / 2, dest.replace("_", " "), ha="center", va="center", fontsize=6,
                            color="w" if float(shades.get(dest, "0.5")) < 0.5 else "k")
                bottom += v
        ax.set_xticks(range(len(keys))); ax.set_xticklabels([k.replace("_", "\n") for k in keys], fontsize=7)
        ax.set_title(pretty(s), loc="left", fontsize=9); ax.set_ylim(0, 100)
    axes[0][0].set_ylabel("% of ground-truth voxels, by predicted class")
    fig.tight_layout(); fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"), dpi=200); plt.close(fig)


def fig_rare(data, out):
    keys = ["L6", "lumbar_rib_left", "lumbar_rib_right", "rib12_left", "rib12_right"]
    names = list(data); x = np.arange(len(keys)); w = 0.8 / max(1, len(names))
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    for j, s in enumerate(names):
        e = data[s].get("rare_class_emission", {})
        hit = [100 * e[k]["emitted_on_gt_cases"] / e[k]["gt_cases"] if k in e and e[k]["gt_cases"] else np.nan for k in keys]
        ax.bar(x + (j - (len(names) - 1) / 2) * w, hit, w, label=pretty(s), color=str(0.15 + 0.7 * j / max(1, len(names) - 1)))
    ax.set_xticks(x); ax.set_xticklabels([k.replace("_", " ") for k in keys], fontsize=7.5)
    ax.set_ylim(0, 100); ax.set_ylabel("records carrying the class\nwhere it is emitted (%)")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"), dpi=200); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--systems", nargs="*", default=None)
    a = ap.parse_args()
    bench = a.bench.expanduser()
    systems = a.systems or sorted(p.name for p in bench.iterdir() if (p / "score" / "summary.json").exists())
    data = load(bench, systems)
    if not data:
        print("no scored systems under", bench); return 1
    a.out.mkdir(parents=True, exist_ok=True)
    fig_dice(data, a.out / "fig_bench_dice"); fig_count(data, a.out / "fig_bench_count")
    fig_ribs(data, a.out / "fig_bench_ribs"); fig_rare(data, a.out / "fig_bench_rare")
    print("figures for", list(data), "->", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
