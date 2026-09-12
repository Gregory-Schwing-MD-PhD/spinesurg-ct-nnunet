"""Report L6's pseudo dice by position derived from dataset.json, with nan handled.

TWO THINGS THIS GETS RIGHT THAT COUNTING BY EYE DOES NOT.

Position, not label id. nnU-Net prints an unlabelled list over the classes it PREDICTS:
background dropped, `ignore` dropped, the rest ordered by label value. L6's label id is 10
and its position is 9, and reporting position 10 would hand you the sacrum's number under
L6's name -- the same substitution that has produced three wrong-but-plausible values in
this project already.

`nan` is a value. The list is printed as Python floats and a class with no instance in the
epoch's validation batches prints `nan`. A regex over digits skips those, shortens the list,
and shifts every position after the first one. That is what made an earlier read of this
same log look like a 34-vs-43 class-count crisis; the data was fine and the parser was not.
The nans are also the check: they fall on the empty rib slots, whose positions are known
ahead of time, so if they land elsewhere the mapping is wrong.

    python l6_report.py <preprocessed_dir> <training_log.txt>
"""
from __future__ import annotations

import json
import math
import os
import re
import sys

ds, log = sys.argv[1], sys.argv[2]
labels = json.load(open(os.path.join(ds, "dataset.json")))["labels"]

fg = sorted(((v, k) for k, v in labels.items()
             if not isinstance(v, (list, tuple)) and v != 0 and k != "ignore"),
            key=lambda t: t[0])
pos = {name: i for i, (_, name) in enumerate(fg)}
name_at = {i: name for name, i in pos.items()}

EPOCH = re.compile(r": Epoch (\d+)\s*$")
DICE = re.compile(r"Pseudo dice \[(.*?)\]")
NUM = re.compile(r"nan|-?\d+(?:\.\d+)?(?:e[-+]?\d+)?", re.I)

epoch, rows = None, []
for line in open(log, errors="replace"):
    m = EPOCH.search(line)
    if m:
        epoch = int(m.group(1))
        continue
    m = DICE.search(line)
    if m:
        vals = [math.nan if t.lower() == "nan" else float(t)
                for t in NUM.findall(m.group(1))]
        rows.append((epoch, vals))

if not rows:
    print("no pseudo-dice line yet")
    sys.exit(0)

last_epoch, last = rows[-1]
if len(last) != len(fg):
    print(f"! list has {len(last)} entries, {len(fg)} classes predicted -- positions not "
          f"trustworthy, reporting nothing.")
    sys.exit(1)

nan_at = sorted(name_at[i] for i, v in enumerate(last) if isinstance(v, float) and math.isnan(v))
print(f"{len(rows)} epoch(s) logged; latest = epoch {last_epoch}")
print(f"classes reading nan (no validation instance this epoch): {nan_at}\n")

WATCH = ["L3", "L4", "L5", "L6", "sacrum", "left_hip", "right_hip"]
print(f"{'class':<11}{'id':>4}{'pos':>5}" + "".join(f"{f'ep{e}':>9}" for e, _ in rows[-6:]))
for w in WATCH:
    if w not in pos:
        continue
    cells = "".join(f"{v[pos[w]]:>9.4f}" if not math.isnan(v[pos[w]]) else f"{'nan':>9}"
                    for _, v in rows[-6:])
    print(f"{w:<11}{labels[w]:>4}{pos[w]:>5}{cells}")

l6 = last[pos["L6"]]
hist = [v[pos["L6"]] for _, v in rows if not math.isnan(v[pos["L6"]])]
best = max(hist) if hist else 0.0
print()
if best > 0.0:
    print(f"L6 has been NON-ZERO (best {best:.4f}, latest {l6:.4f}). Every one of the ten "
          f"previous runs held 0.0000 at every epoch, one to epoch 83 -- the class is now "
          f"being sampled.")
else:
    print(f"L6 still 0.0000 through epoch {last_epoch}. On a class in 2.2% of cases this is "
          f"expected early; it becomes a finding if it persists past roughly epoch 30, now "
          f"that the bias is verified to fire on label 10.")
