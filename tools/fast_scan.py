"""Read which classes each case carries from nnU-Net's own class_locations.

The first version of this memory-mapped every *_seg.npy and called np.unique on it --
802 volumes at a few hundred megabytes each, and it was still going after half an hour.
nnU-Net has already done the work: preprocessing writes a per-case pickle whose
`class_locations` maps each class to the voxel coordinates sampled for foreground
patches. A key with a non-empty array IS the statement "this case contains this class",
which is the same fact, already computed, at a thousandth of the cost.

It is also the RIGHT source: class_locations is what the dataloader samples from, so a
class missing here is a class the sampler cannot reach even if voxels exist in the
segmentation.
"""
from __future__ import annotations

import glob
import json
import os
import pickle
import sys


def load(ds_dir: str, plans_sub: str = "nnUNetPlans_3d_fullres"):
    pkls = sorted(glob.glob(os.path.join(ds_dir, plans_sub, "*.pkl")))
    if not pkls:
        raise SystemExit(f"no *.pkl under {os.path.join(ds_dir, plans_sub)}")
    out = {}
    for f in pkls:
        case = os.path.basename(f)[:-4]
        try:
            with open(f, "rb") as fh:
                p = pickle.load(fh)
        except Exception:                                          # noqa: BLE001
            continue
        cl = p.get("class_locations") or {}
        keys = set()
        for k, v in cl.items():
            if v is None:
                continue
            try:
                n = len(v)
            except TypeError:
                n = 0
            if n == 0:
                continue
            # region-based training stores tuple keys; only int keys are single classes
            if isinstance(k, (int,)) or (hasattr(k, "item") and not isinstance(k, tuple)):
                keys.add(int(k))
        out[case] = keys
    return out


if __name__ == "__main__":
    ds = sys.argv[1]
    d = json.load(open(os.path.join(ds, "dataset.json")))
    inv = {v: k for k, v in d["labels"].items() if v != 0}
    present = load(ds)
    print(f"{len(present)} cases read from class_locations")
    from collections import Counter
    c = Counter()
    for vs in present.values():
        for v in vs:
            c[v] += 1
    print(f"\n{'class':<20}{'cases':>7}{'%':>8}")
    for v, n in sorted(c.items(), key=lambda kv: kv[1]):
        print(f"{inv.get(v, v):<20}{n:>7}{100.0*n/len(present):>7.1f}%")
    missing = [inv[v] for v in sorted(inv) if c.get(v, 0) == 0]
    print(f"\nclasses with NO sampleable location in any case: {missing or 'none'}")
