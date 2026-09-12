"""One train/validation split that can actually measure the rare classes.

WHY NOT FIVE FOLDS, FOR NOW. Five-fold cross-validation answers "how much does this
estimate vary with the sample", which is a sensible question when every class has hundreds
of instances. L6 has about 22 in 802. Split five ways that is four or five validation cases
per fold, and `nan` already appears in the logs -- some epochs have none at all. A Dice
computed on four cases moves 25 points when one vertebra is missed; five such numbers do
not become an error bar, they become five coin flips.

So: one split while the rare classes are the question, five folds again when they are not.
The common classes lose almost nothing -- they had hundreds of validation instances and
now have a few hundred -- and the rare ones go from unmeasurable to measurable.

WHAT THIS SPLITTER DOES THAT nnU-Net'S DOES NOT. nnU-Net splits at random with a fixed
seed. That is fine for classes that are everywhere and useless for a class in 2.7% of
records: random assignment routinely leaves a rare class entirely on one side. This one
places the rare classes FIRST, guaranteeing each a floor in validation, then fills the rest
at random to hit the target ratio.

HOW THE RARE CLASSES GET REPORTED ANYWAY. `MIN_VAL` is the fewest validation cases at
which a per-class Dice from THIS SINGLE SPLIT means anything. It is not a statement that a
class below it is unreportable -- every class here has a reporting route, and the route is
what the summary prints:

  at or above the floor   -> per-class Dice from this split, as usual.

  below it                -> POOLED OUT-OF-FOLD. Run the folds, predict each case with the
                             fold that did not train on it, and pool. Every positive case
                             then contributes exactly one prediction from a model that never
                             saw it, so a class with 22 positives is measured on n=22 rather
                             than on the 3-6 a single split could spare -- and each model
                             still trains on ~18 of them instead of 16. This is how VerSe
                             reports its rare transitional classes, and it is strictly more
                             informative than a held-out slice at this n, in both directions.

So the single split below is the run that shows the rare classes are LEARNED AT ALL (a
non-zero Dice on a handful of cases settles that in days, not weeks). The number that goes
in the paper comes from the pooled out-of-fold pass over the folds. Both are needed; neither
substitutes for the other.

    python consolidate_splits.py <preprocessed_dataset_dir> [--val-frac 0.15] [--write]

Without --write it prints the split it would make and changes nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MIN_VAL = 6          # fewest validation cases at which a per-class Dice means anything
SEED = 20260911


def scan(ds_dir: str, plans_sub: str = "nnUNetPlans_3d_fullres"):
    """case -> set of label values present, read from the preprocessed segmentations."""
    import glob
    segs = sorted(glob.glob(os.path.join(ds_dir, plans_sub, "*_seg.npy")))
    if not segs:
        raise SystemExit(f"no *_seg.npy under {os.path.join(ds_dir, plans_sub)}")
    out = {}
    for i, f in enumerate(segs, 1):
        case = os.path.basename(f).replace("_seg.npy", "")
        try:
            a = np.load(f, mmap_mode="r")
            out[case] = {int(v) for v in np.unique(np.asarray(a)) if v > 0}
        except Exception as exc:                                   # noqa: BLE001
            print(f"  ! {case}: {type(exc).__name__}", file=sys.stderr)
        if i % 100 == 0:
            print(f"  scanned {i}/{len(segs)}", flush=True)
    return out


def build(present: dict[str, set[int]], id_to_name: dict[int, str], val_frac: float):
    cases = sorted(present)
    rng = random.Random(SEED)

    # how rare is each class, ascending: the scarcest gets to choose first
    count = Counter()
    for vs in present.values():
        for v in vs:
            count[v] += 1
    order = [v for v, _ in sorted(count.items(), key=lambda kv: kv[1])]

    target_val = max(1, int(round(len(cases) * val_frac)))
    val: set[str] = set()

    # PLACE THE RARE CLASSES FIRST. Each gets min(MIN_VAL, ceil(val_frac * n)) cases in
    # validation; a class with 22 positives contributes 6, a class with 4 contributes 1.
    for v in order:
        holders = [c for c in cases if v in present[c]]
        want = min(len(holders), max(1, min(MIN_VAL, int(round(len(holders) * val_frac)) or 1)))
        have = sum(1 for c in holders if c in val)
        if have >= want:
            continue
        pool = [c for c in holders if c not in val]
        rng.shuffle(pool)
        for c in pool[: want - have]:
            if len(val) < target_val * 2:        # never let the floor blow past the target
                val.add(c)

    # fill the rest at random to reach the target ratio
    rest = [c for c in cases if c not in val]
    rng.shuffle(rest)
    for c in rest:
        if len(val) >= target_val:
            break
        val.add(c)

    train = [c for c in cases if c not in val]
    return sorted(train), sorted(val), count, order


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_dir")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    dj = json.load(open(os.path.join(a.dataset_dir, "dataset.json")))
    id_to_name = {v: k for k, v in dj["labels"].items() if v != 0}

    print(f"reading class_locations from {a.dataset_dir}")
    # nnU-Net already recorded which classes each case carries, and it is the RIGHT
    # source: class_locations is what the dataloader samples from, so a class absent
    # here cannot be reached even if voxels exist in the segmentation.
    from fast_scan import load as _load
    present = _load(a.dataset_dir)
    train, val, count, order = build(present, id_to_name, a.val_frac)

    print(f"\nsingle split: {len(train)} train / {len(val)} validation "
          f"({100.0*len(val)/(len(train)+len(val)):.1f}%)\n")
    print(f"{'class':<20}{'cases':>7}{'in val':>8}{'reported by':>14}")
    under = []
    for v in order:
        n = count[v]
        nv = sum(1 for c in val if v in present[c])
        ok = "this split" if nv >= MIN_VAL else "pooled OOF"
        if nv < MIN_VAL:
            under.append(id_to_name.get(v, str(v)))
        print(f"{id_to_name.get(v, v):<20}{n:>7}{nv:>8}{ok:>14}")

    print(f"\n{len(under)} class(es) below {MIN_VAL} validation cases in this split:")
    print("  " + (", ".join(under) if under else "none"))
    if under:
        print("  -> reported by POOLED OUT-OF-FOLD prediction over the folds, not by")
        print("     this split: every positive case gets one prediction from a model")
        print("     that never trained on it, so the measurement uses ALL of them.")
        print("     This split's job for these classes is to show they are learned.")

    if a.write:
        p = os.path.join(a.dataset_dir, "splits_final.json")
        if os.path.exists(p):
            bak = p + ".5fold.bak"
            if not os.path.exists(bak):
                os.rename(p, bak)
                print(f"\nprevious five-fold split kept at {bak}")
        json.dump([{"train": train, "val": val}], open(p, "w"), indent=1)
        print(f"wrote {p} (one fold)")
    else:
        print("\n(dry run; pass --write to install it)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
