#!/usr/bin/env python3
"""benchmark/patch_context.py — what a training patch actually sees, measured on the labels.

For each candidate patch shape (voxels at the plan spacing 0.80 x 0.75 x 0.75 mm) and each
label volume, a window of that physical size is centred on the column (centroid of the
vertebra labels, at the craniocaudal middle of the labelled spine) and we record:
  fg_frac        labelled foreground voxels inside the window / window volume (label density)
  levels_whole   number of vertebral levels lying entirely inside the window
  levels_touched number of levels with any voxel inside
  spine_frac     fraction of all vertebra voxels of the case inside the window
  hip_frac       fraction of hip + femur voxels inside
  anchors        whether both the sacrum and a rib-bearing body (T12 or above) are inside
Then a render: coronal and sagittal projections of one case's labels with the windows drawn
to scale, so the trade-off between the long column and the wide pelvis is visible.

    python benchmark/patch_context.py --labels ~/data/CTSpinoPelvic1K/labels --out ~/bench/patch_context \
        --shapes 256,320,320 384,256,256 448,224,224 --limit 0 --render 0376
Labels are the released v10 scheme (VerSe ids 1-25 vertebrae, 26 sacrum, 28 T13, 29 S1,
30/31 hips, 32/33 femora, ribs 34-59, lumbar ribs 60/61, hardware 62-68).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np

SPACING = (0.80, 0.7544530034065247, 0.7544530034065247)     # plan spacing, craniocaudal first
VERT = list(range(8, 26)) + [28]                             # T1..L6 and T13 (VerSe ids)
SACRUM = [26, 29]
HIPS = [30, 31, 32, 33]
RIB_BEARING_MAX = 19                                         # T12 (VerSe 19) or above


def sup_axis(img):
    ax = nib.aff2axcodes(img.affine)
    for i, c in enumerate(ax):
        if c in ("S", "I"):
            return i, (1 if c == "S" else -1)
    raise ValueError(ax)


def analyse(lab_path: Path, shapes_mm: dict) -> dict:
    img = nib.load(str(lab_path))
    lab = np.asanyarray(img.dataobj).astype(np.int16)
    zooms = np.asarray(img.header.get_zooms()[:3], float)
    axis, _ = sup_axis(img)
    others = [i for i in range(3) if i != axis]
    vert = np.isin(lab, VERT)
    if vert.sum() < 1000:
        return {"case": lab_path.name, "error": "no vertebrae"}
    fg = lab > 0
    idx = np.argwhere(vert)
    centre = idx.mean(0)                                     # column centroid, in voxels
    # craniocaudal centre = middle of the labelled spine extent
    lo, hi = idx[:, axis].min(), idx[:, axis].max()
    centre[axis] = (lo + hi) / 2
    out = {"case": lab_path.name.replace("_label.nii.gz", ""), "spine_extent_mm": round((hi - lo + 1) * zooms[axis], 1)}
    levels_present = [v for v in VERT if (lab == v).any()]
    for name, mm in shapes_mm.items():
        half = np.array([mm[0] / zooms[axis] / 2 if i == axis else mm[1 + others.index(i)] / zooms[i] / 2 for i in range(3)])
        a = np.maximum(0, np.round(centre - half).astype(int))
        b = np.minimum(np.array(lab.shape), np.round(centre + half).astype(int))
        sl = tuple(slice(int(a[i]), int(b[i])) for i in range(3))
        win = lab[sl]
        vol = float(np.prod([b[i] - a[i] for i in range(3)]))
        whole = touched = 0
        for v in levels_present:
            m = lab == v
            inside = (win == v).sum()
            if inside > 0:
                touched += 1
                if inside == m.sum():
                    whole += 1
        sac_in = np.isin(win, SACRUM).any()
        rib_bearing_in = any((win == v).any() for v in range(8, RIB_BEARING_MAX + 1))
        out.update({
            f"{name}_fg_frac": round(float((win > 0).sum() / vol), 4),
            f"{name}_levels_whole": whole, f"{name}_levels_touched": touched,
            f"{name}_spine_frac": round(float(np.isin(win, VERT).sum() / vert.sum()), 4),
            f"{name}_hip_frac": round(float(np.isin(win, HIPS).sum() / max(1, np.isin(lab, HIPS).sum())), 4),
            f"{name}_anchors": int(bool(sac_in and rib_bearing_in)),
        })
    return out


def render(lab_path: Path, shapes_mm: dict, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    img = nib.load(str(lab_path))
    lab = np.asanyarray(img.dataobj).astype(np.int16)
    zooms = np.asarray(img.header.get_zooms()[:3], float)
    axis, sign = sup_axis(img)
    others = [i for i in range(3) if i != axis]
    vert = np.isin(lab, VERT)
    idx = np.argwhere(vert)
    centre = idx.mean(0); lo, hi = idx[:, axis].min(), idx[:, axis].max(); centre[axis] = (lo + hi) / 2
    # two projections through the label volume: along each of the two in-plane axes
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.2))
    styles = [("#1f4e79", "-"), ("#c0392b", "-"), ("#7f8c8d", "--")]
    for ax, proj_axis in zip(axes, others):
        keep = [i for i in range(3) if i != proj_axis]        # axes shown: one in-plane + the column
        mip = (lab > 0).max(axis=proj_axis).astype(float)
        vmip = vert.max(axis=proj_axis).astype(float)
        show = mip * 0.35 + vmip * 0.65
        # orient: column vertical, superior up
        if keep.index(axis) == 0:
            show = show.T; h_ax, w_ax = axis, keep[1]
        else:
            h_ax, w_ax = axis, keep[0]
        if sign < 0:
            pass
        extent = [0, show.shape[1] * zooms[w_ax], 0, show.shape[0] * zooms[h_ax]]
        ax.imshow(show, cmap="gray_r", origin="lower" if sign > 0 else "upper", extent=extent, aspect="equal", interpolation="nearest")
        cy = centre[h_ax] * zooms[h_ax]; cx = centre[w_ax] * zooms[w_ax]
        if sign < 0:
            cy = show.shape[0] * zooms[h_ax] - cy
        for (name, mm), (col, ls) in zip(shapes_mm.items(), styles):
            wmm = mm[1 + others.index(w_ax)]; hmm = mm[0]
            ax.add_patch(Rectangle((cx - wmm / 2, cy - hmm / 2), wmm, hmm, fill=False, ec=col, ls=ls, lw=1.4, label=name))
        ax.set_xlabel("mm"); ax.set_ylabel("mm")
        ax.set_title("coronal" if proj_axis != others[0] else "sagittal", loc="left", fontsize=9)
        ax.grid(False)
    axes[0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.suptitle("one training patch of each shape, centred on the column (labels, projected)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200); fig.savefig(out_png.with_suffix(".pdf"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--shapes", nargs="+", default=["256,320,320", "384,256,256", "448,224,224"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--render", default="0376")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    shapes_mm = {}
    for s in a.shapes:
        v = [int(x) for x in s.split(",")]
        shapes_mm["p" + "x".join(map(str, v))] = (v[0] * SPACING[0], v[1] * SPACING[1], v[2] * SPACING[2])
    (a.out / "shapes_mm.json").write_text(json.dumps(shapes_mm, indent=1))
    files = sorted(a.labels.glob("*_label.nii.gz"))
    if a.limit:
        files = files[: a.limit]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(analyse, files, [shapes_mm] * len(files), chunksize=4))
    rows = [r for r in rows if "error" not in r]
    keys = list(rows[0].keys())
    with open(a.out / "patch_context.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    summ = {}
    for name in shapes_mm:
        summ[name] = {k: round(float(np.mean([r[f"{name}_{k}"] for r in rows])), 3)
                      for k in ("fg_frac", "levels_whole", "levels_touched", "spine_frac", "hip_frac", "anchors")}
    summ["_n"] = len(rows); summ["_shapes_mm"] = {k: [round(x, 1) for x in v] for k, v in shapes_mm.items()}
    (a.out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    rp = a.labels / f"{a.render}_label.nii.gz"
    if rp.exists():
        render(rp, shapes_mm, a.out / f"fig_patch_context_{a.render}.png")
        print("rendered", a.render)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
