"""Name a vertebra by its own transverse process, not by the rib next to it.

    python tools/tp_morphology.py --labels <dir of *_label.nii.gz> --out tp.csv
    python tools/tp_morphology.py --labels ~/vf_v10 --out vf_tp.csv --levels T11,T12,L1,L2,L3

WHY THE EXISTING ATLAS CANNOT DO THIS. CTSpinoPelvic1K's per-level morphometrics separate
T12 from L1 hardly at all -- anterior height d=0.80, pedicle width d=0.54, end-plate width
d=0.14, canal AP depth d=0.00 with a hundred per cent overlap. Body, canal and pedicle
dimensions change smoothly down the column: they encode POSITION, not thoracic-versus-
lumbar IDENTITY. Asking them which of two neighbours a vertebra is, is asking the wrong
question of the right data.

WHAT DOES DEFINE A THORACIC VERTEBRA, and it is on the vertebra rather than beside it: the
transverse process is short, thick and club-shaped because it carries a costal facet. A
lumbar costal process is long, thin and blade-like and carries nothing. That difference
holds whether or not the rib itself was segmented, which is the point -- a rib is an
adjacent structure and cannot be what names the bone.

THE MEASUREMENT. The vertebra is split at the canal into body and posterior elements; the
wings reaching laterally past the pedicles are the transverse processes. For each side:

    reach_mm        how far the tip goes lateral to the body's edge
    thickness_mm    the process's own cross-section where it leaves the pedicle,
                    as sqrt(area) of its slice perpendicular to the reach
    slenderness     reach / thickness -- high for a lumbar blade, low for a thoracic club

Sides are reported separately because a unilateral lumbar rib is a real phenotype and an
average over the two would hide it.

AXES COME FROM THE AFFINE, never from an assumption about array order. CTSpinoPelvic1K
stores ('P','I','R') and VerSe arrives in five different orientations; a hard-coded axis
index is how this project has measured the wrong bone before.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

# v10 ids
LEVEL_ID = {"T10": 17, "T11": 18, "T12": 19, "T13": 28,
            "L1": 20, "L2": 21, "L3": 22, "L4": 23, "L5": 24, "L6": 25}
MIN_VOX = 2000


def axes_of(affine):
    """(lr, ap, si) array-axis indices and the sign that makes each increase L->R, P->A, I->S."""
    codes = nib.aff2axcodes(affine)
    out = {}
    for i, c in enumerate(codes):
        if c in "LR":
            out["lr"] = (i, 1 if c == "R" else -1)
        elif c in "AP":
            out["ap"] = (i, 1 if c == "A" else -1)
        else:
            out["si"] = (i, 1 if c == "S" else -1)
    return out


def tp_metrics(mask: np.ndarray, zooms, ax) -> dict:
    """Per-side transverse-process reach, thickness and slenderness, in millimetres."""
    lr_i, lr_s = ax["lr"]
    ap_i, _ = ax["ap"]
    si_i, _ = ax["si"]
    idx = np.argwhere(mask)
    if len(idx) < MIN_VOX:
        return {}

    lr = idx[:, lr_i] * lr_s * zooms[lr_i]          # millimetres, increasing to the right
    ap = idx[:, ap_i] * zooms[ap_i]
    si = idx[:, si_i] * zooms[si_i]
    mid = float(np.median(lr))

    # THE BODY'S OWN HALF-WIDTH, measured where the process cannot reach: the anterior
    # third of the vertebra is body only, and its lateral extent is the datum that "reach"
    # is measured from. Using the whole vertebra's width instead would fold the process
    # being measured into its own baseline.
    ant = ap <= np.percentile(ap, 33)
    if ant.sum() < 50:
        return {}
    body_half = float(np.percentile(np.abs(lr[ant] - mid), 97))

    out = {"body_half_mm": round(body_half, 2)}
    for side, sel in (("left", lr < mid), ("right", lr > mid)):
        d = np.abs(lr[sel] - mid)
        if not len(d):
            continue
        reach = float(np.percentile(d, 99.5)) - body_half
        # thickness where the process leaves the pedicle: the slab between the body edge
        # and halfway to the tip, as sqrt of its cross-sectional area in the AP-SI plane.
        lo, hi = body_half, body_half + max(reach, 1.0) * 0.5
        slab = sel.copy()
        slab[sel] = (d >= lo) & (d <= hi)
        n = int(slab.sum())
        if n < 20 or reach <= 0:
            out[f"{side}_reach_mm"] = round(max(reach, 0.0), 2)
            out[f"{side}_thickness_mm"] = None
            out[f"{side}_slenderness"] = None
            continue
        # area per unit length along the reach -> a mean cross-section
        span_mm = max(hi - lo, 1e-3)
        vox_mm3 = float(np.prod([zooms[i] for i in (lr_i, ap_i, si_i)]))
        area = n * vox_mm3 / span_mm
        thick = math.sqrt(max(area, 1e-6))
        out[f"{side}_reach_mm"] = round(reach, 2)
        out[f"{side}_thickness_mm"] = round(thick, 2)
        out[f"{side}_slenderness"] = round(reach / thick, 3)
        # how far the process runs in AP and SI: a club is compact, a blade is flat
        out[f"{side}_ap_spread_mm"] = round(float(np.ptp(ap[slab])), 2)
        out[f"{side}_si_spread_mm"] = round(float(np.ptp(si[slab])), 2)
    return out


def one(path: Path, levels) -> list[dict]:
    img = nib.load(str(path))
    lab = np.asanyarray(img.dataobj).astype(np.int16)
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    ax = axes_of(img.affine)
    if len(ax) != 3:
        return []
    case = path.name.split("_label")[0].replace(".nii.gz", "")
    rows = []
    for name in levels:
        cid = LEVEL_ID[name]
        m = lab == cid
        if m.sum() < MIN_VOX:
            continue
        r = tp_metrics(m, zooms, ax)
        if not r:
            continue
        r.update(case=case, level=name, n_vox=int(m.sum()))
        rows.append(r)
    return rows


def main() -> int:
    ap_ = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--labels", required=True)
    ap_.add_argument("--out", required=True)
    ap_.add_argument("--levels", default="T11,T12,L1,L2,L3")
    ap_.add_argument("--limit", type=int, default=0)
    ap_.add_argument("--workers", type=int, default=8)
    a = ap_.parse_args()

    levels = [x.strip() for x in a.levels.split(",") if x.strip() in LEVEL_ID]
    files = sorted(Path(a.labels).glob("*.nii.gz"))
    if a.limit:
        files = files[:a.limit]
    print(f"{len(files)} labels, levels {levels}", flush=True)

    from concurrent.futures import ProcessPoolExecutor, as_completed
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(one, f, levels): f for f in files}
        for i, fu in enumerate(as_completed(futs), 1):
            try:
                rows.extend(fu.result())
            except Exception as exc:                                # noqa: BLE001
                print(f"  {futs[fu].name}: {type(exc).__name__}: {exc}")
            if i % 100 == 0:
                print(f"  ... {i}/{len(files)}", flush=True)

    if not rows:
        print("nothing measured")
        return 1
    cols = ["case", "level", "n_vox", "body_half_mm"]
    for s in ("left", "right"):
        cols += [f"{s}_reach_mm", f"{s}_thickness_mm", f"{s}_slenderness",
                 f"{s}_ap_spread_mm", f"{s}_si_spread_mm"]
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.out}  ({len(rows)} vertebrae)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
