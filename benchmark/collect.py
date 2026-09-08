#!/usr/bin/env python3
"""benchmark/collect.py — one table across systems from $BENCH/<system>/score/summary.json
(tools/eval_oneshot.py output) and decode.csv (tools/decode_sequence.py).

    python benchmark/collect.py --bench ~/bench --out ~/bench/comparison.md [--systems gt totalsegmentator ...]

Columns: n cases, rib-free count exact rate (all / typical / transitional), decoder count
exact rate, mean Dice on typical and transitional records for the junction classes, and
whether the system ever emits L6, T13 and the lumbar ribs on the records that carry them.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

KEY_CLASSES = ("T12", "L1", "L4", "L5", "L6", "sacrum", "rib12_left", "rib12_right",
               "lumbar_rib_left", "lumbar_rib_right", "left_hip", "femur")


def pct(x):
    return "" if x is None else f"{100 * x:.1f}"


def load(bench: Path, system: str) -> dict | None:
    s = bench / system / "score" / "summary.json"
    if not s.exists():
        return None
    d = json.loads(s.read_text())
    dec = bench / system / "score" / "decode.csv"
    if dec.exists():
        rows = [r for r in csv.DictReader(open(dec)) if r.get("count_error", "") != ""]
        if rows:
            d["decoder_exact"] = sum(1 for r in rows if r["count_error"] == "0") / len(rows)
            d["decoder_n"] = len(rows)
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--systems", nargs="*", default=None)
    a = ap.parse_args()
    bench = a.bench.expanduser()
    systems = a.systems or sorted(p.name for p in bench.iterdir() if (p / "score" / "summary.json").exists())
    data = {s: load(bench, s) for s in systems}
    data = {s: d for s, d in data.items() if d}

    lines = ["# Competitor benchmark on CTSpinoPelvic1K (one-shot scoring space)", ""]
    lines.append("| system | n | count exact all | typical | transitional | decoder exact | " +
                 " | ".join(f"Dice {c} typ/trans" for c in KEY_CLASSES) + " |")
    lines.append("|" + "---|" * (6 + len(KEY_CLASSES)))
    for s, d in data.items():
        c = d["rib_free_count"]
        dm = d["dice_mean"]
        cells = [s, str(d["n_cases"]), pct(c["all"]["exact"]), pct(c["typical"]["exact"]),
                 pct(c["transitional"]["exact"]),
                 f"{pct(d.get('decoder_exact'))} (n={d.get('decoder_n', '')})" if "decoder_exact" in d else ""]
        for k in KEY_CLASSES:
            t, tr = dm.get("typical", {}).get(k), dm.get("transitional", {}).get(k)
            cells.append(f"{'' if t is None else f'{t:.3f}'}/{'' if tr is None else f'{tr:.3f}'}")
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Rare classes: GT cases / emitted on them / emitted where absent", ""]
    rare = sorted({k for d in data.values() for k in d.get("rare_class_emission", {})})
    lines.append("| system | " + " | ".join(rare) + " |")
    lines.append("|" + "---|" * (1 + len(rare)))
    for s, d in data.items():
        e = d.get("rare_class_emission", {})
        lines.append(f"| {s} | " + " | ".join(
            f"{e[k]['gt_cases']}/{e[k]['emitted_on_gt_cases']}/{e[k]['emitted_where_absent']}" if k in e else ""
            for k in rare) + " |")
    lines += ["", "## Where the ground-truth rib voxels went (top 3, % of GT voxels)", ""]
    for s, d in data.items():
        lines.append(f"**{s}**")
        for k, v in d.get("rib_confusion_pct_of_gt_voxels", {}).items():
            lines.append(f"- {k}: " + ", ".join(f"{pn} {p}%" for pn, p in list(v.items())[:3]))
        lines.append("")
    a.out.expanduser().parent.mkdir(parents=True, exist_ok=True)
    a.out.expanduser().write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
