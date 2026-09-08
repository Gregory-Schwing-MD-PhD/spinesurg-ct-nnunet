#!/usr/bin/env python3
"""tools/reshape_patch.py — give the patch its craniocaudal reach without spending memory.

nnU-Net's planner shapes the patch after the median image (here 545 x 511 x 512 voxels at
0.80 x 0.75 x 0.75 mm), which yields 256 x 320 x 320: 20.5 cm along the column and 24 cm
in-plane. Counting needs the column: T10 to the sacrum is about 30 cm. 384 x 256 x 256 has
the SAME voxel count within 4% (25.2M vs 26.2M), so the same memory and the same batch of
two on the H200, but 30.7 cm along the column and 19.3 cm in-plane, which still holds the
spine and most of the pelvis in every patch. All axes stay divisible by 2^6 for the seven
ResEnc-L stages, so the architecture is untouched.

    python tools/reshape_patch.py --plans <preprocessed>/<Dataset>/nnUNetResEncUNetLPlans_100G.json \
        --patch 384 256 256 [--config 3d_fullres]

Writes a .orig backup once and edits in place; the fold jobs read the plans at start.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", required=True, type=Path)
    ap.add_argument("--patch", nargs=3, type=int, default=[384, 256, 256])
    ap.add_argument("--config", default="3d_fullres")
    a = ap.parse_args()
    p = json.loads(a.plans.read_text())
    c = p["configurations"][a.config]
    old = list(c["patch_size"])
    strides = c["architecture"]["arch_kwargs"]["strides"]
    total = [1, 1, 1]
    for s in strides:
        for i in range(3):
            total[i] *= int(s[i])
    for i in range(3):
        if a.patch[i] % total[i]:
            raise SystemExit(f"axis {i}: patch {a.patch[i]} not divisible by total stride {total[i]}")
    sp = c["spacing"]
    print(f"old patch {old} = {[round(o * s / 10, 1) for o, s in zip(old, sp)]} cm, {old[0]*old[1]*old[2]/1e6:.1f}M voxels")
    print(f"new patch {a.patch} = {[round(o * s / 10, 1) for o, s in zip(a.patch, sp)]} cm, {a.patch[0]*a.patch[1]*a.patch[2]/1e6:.1f}M voxels")
    if a.patch[0] * a.patch[1] * a.patch[2] > 1.05 * old[0] * old[1] * old[2]:
        raise SystemExit("new patch is more than 5% larger than the planned one; it would not fit the memory target")
    bak = a.plans.with_suffix(".json.orig")
    if not bak.exists():
        shutil.copy2(a.plans, bak)
    c["patch_size"] = list(a.patch)
    c.setdefault("_notes", []).append(f"patch reshaped from {old} to {list(a.patch)} by tools/reshape_patch.py: same voxel budget, craniocaudal reach")
    a.plans.write_text(json.dumps(p, indent=2))
    print("wrote", a.plans, "(backup", bak.name + ")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
