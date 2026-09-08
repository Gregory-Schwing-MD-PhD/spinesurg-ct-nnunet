#!/usr/bin/env python3
"""benchmark/extract_training_series.py — per-epoch series from the nnU-Net training logs of
the one-shot folds (login node, text only): the per-class pseudo-Dice line, the dedicated
LSTV validation headline (L6 Dice on lumbarization cases), the collapse fraction (GT-L6
voxels called L5), the per-subgroup Dice and the epoch time.

    python benchmark/extract_training_series.py --results nnunet/results/Dataset810_... --out benchmark/training_series.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

NAMES = ["T10", "T11", "T12", "T13", "L1", "L2", "L3", "L4", "L5", "L6", "sacrum", "rib_left",
         "rib_right", "rib12_left", "rib12_right", "rib13_left", "rib13_right",
         "lumbar_rib_left", "lumbar_rib_right", "left_hip", "right_hip", "femur"]


def parse_fold(log_paths: list[Path], fold: int) -> list[dict]:
    rows: dict[int, dict] = {}
    epoch = None
    for lp in sorted(log_paths):
        for line in lp.read_text(errors="replace").splitlines():
            m = re.search(r": Epoch (\d+)\s*$", line)
            if m:
                epoch = int(m.group(1)); rows.setdefault(epoch, {"fold": fold, "epoch": epoch}); continue
            if epoch is None:
                continue
            r = rows[epoch]
            m = re.search(r"train_loss (-?[0-9.]+)", line)
            if m: r["train_loss"] = float(m.group(1)); continue
            m = re.search(r"val_loss (-?[0-9.]+)", line)
            if m: r["val_loss"] = float(m.group(1)); continue
            m = re.search(r"Pseudo dice \[(.*)\]", line)
            if m:
                vals = [v.strip() for v in m.group(1).split(",")]
                for n, v in zip(NAMES, vals):
                    r[f"pdice_{n}"] = "" if v == "nan" else float(v)
                continue
            m = re.search(r"Epoch time: ([0-9.]+) s", line)
            if m: r["epoch_s"] = float(m.group(1)); continue
            m = re.search(r"HEADLINE: L6 dice on lumbarization cases = ([0-9.]+)\s+\(n=(\d+)\)", line)
            if m: r["l6_dice_lumb"] = float(m.group(1)); r["l6_n"] = int(m.group(2)); continue
            m = re.search(r"^\S+ \S+:\s+L5\s+:\s+([0-9.]+)%\s+\(THE collapse", line)
            if m: r["l6_to_l5_pct"] = float(m.group(1)); continue
            m = re.search(r"^\S+ \S+:\s+(normal|lumb|sacralization|semisacralization|sacr_count|ambiguous)\s+\(n=(\d+)\)\s+mean_fg=([0-9.]+)\s+lumb=([0-9.]+)\s+pelvis=([0-9.]+)", line)
            if m:
                g = m.group(1)
                r[f"sub_{g}_n"] = int(m.group(2)); r[f"sub_{g}_fg"] = float(m.group(3))
                r[f"sub_{g}_lumb"] = float(m.group(4)); r[f"sub_{g}_pelvis"] = float(m.group(5))
    return [rows[k] for k in sorted(rows)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    rows = []
    for fd in sorted(a.results.glob("fold_*")):
        fold = int(fd.name.split("_")[1])
        logs = sorted(fd.glob("training_log_*.txt"))
        if logs:
            rows += parse_fold(logs[-1:], fold)     # the latest log is the live run
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    by_fold = {}
    for r in rows:
        by_fold[r["fold"]] = by_fold.get(r["fold"], 0) + 1
    print(json.dumps({"rows": len(rows), "epochs_by_fold": by_fold}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
