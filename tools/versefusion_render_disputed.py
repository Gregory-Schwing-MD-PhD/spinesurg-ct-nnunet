"""Render the twelve disputed vertebrae so the question can be answered by looking.

    python tools/versefusion_render_disputed.py --labels ~/vf_813/labelsTr \
        --src ~/VerSeFusion/data/unified --disputed logs/versefusion_disputed.csv \
        --out logs/disputed_renders

WHY A PICTURE AND NOT A NUMBER. Three automated tests were tried on this and none of them
can settle it:

  "Does a labelled rib touch the vertebra" returns no on every case and means nothing,
  because TotalSegmentator emits at most rib_1..rib_12 per side. On a thirteen-rib spine
  it has no label left for the last one, so absence is guaranteed whatever the truth is.

  "Is there a rib nearby" is confounded by anatomy: ribs slope caudally, so the twelfth
  rib passes lateral to the first lumbar vertebra in a normal spine too.

  "Is there unnamed bone nearby" is contaminated by the ignore class, which is every
  voxel over 150 HU that nothing named -- contrast in the aorta and bowel included. It
  reports MORE unnamed bone at L2 than at L1.

A coronal projection through the bone shows a thirteenth rib immediately, and this repo's
own standing rule is that rendering settles what reasoning from numbers gets wrong. Twelve
cases is a few minutes of looking.

WHAT IS DRAWN. A coronal maximum-intensity projection of bone-density CT, with the
disputed vertebra tinted and the levels above and below labelled, so the reader can count
down from a known level and see whether a rib leaves the tinted body.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import nibabel as nib                    # noqa: E402
import numpy as np                       # noqa: E402

LEVELS = {1: "T10", 2: "T11", 3: "T12", 4: "T13", 5: "L1", 6: "L2", 7: "L3",
          8: "L4", 9: "L5", 10: "L6", 11: "sacrum"}
DISPUTED_ID = 5                          # VerSe's L1, the vertebra Moller calls T13
BONE_HU = 200.0


def render(ct_p: Path, lab_p: Path, cid: str, out: Path) -> bool:
    ct_i, lab_i = nib.load(str(ct_p)), nib.load(str(lab_p))
    lab = np.asanyarray(lab_i.dataobj).astype(np.uint8)
    if not (lab == DISPUTED_ID).any():
        return False
    ct = ct_i.get_fdata(dtype=np.float32)
    zooms = lab_i.header.get_zooms()[:3]

    # Dataset813 is stored PIR: axis 0 posterior, axis 1 inferior, axis 2 right. A coronal
    # view is therefore a projection along axis 0, giving (inferior, right).
    idx = np.argwhere(lab == DISPUTED_ID)
    c_si = int(idx[:, 1].mean())
    half = int(round(120.0 / zooms[1]))           # 120 mm above and below
    lo, hi = max(0, c_si - half), min(lab.shape[1], c_si + half)

    bone = np.clip(ct[:, lo:hi, :], BONE_HU, 1500.0)
    mip = bone.max(axis=0)
    dis = (lab[:, lo:hi, :] == DISPUTED_ID).any(axis=0)

    aspect = zooms[1] / zooms[2]
    fig, ax = plt.subplots(figsize=(5.2, 6.4), dpi=130)
    ax.imshow(mip, cmap="gray", aspect=aspect, interpolation="bilinear")
    ov = np.zeros(dis.shape + (4,), np.float32)
    ov[dis] = (1.0, 0.35, 0.1, 0.45)
    ax.imshow(ov, aspect=aspect, interpolation="nearest")

    for cid_lab, name in LEVELS.items():
        m = (lab[:, lo:hi, :] == cid_lab).any(axis=0)
        if not m.any():
            continue
        yy, xx = np.nonzero(m)
        ax.text(xx.mean(), yy.mean(), name, color="#25d07a", fontsize=7.5,
                ha="center", va="center",
                bbox=dict(fc="black", ec="none", alpha=0.45, pad=0.8))
    ax.set_title(f"{cid}: is the tinted body a T13 (rib-bearing) or L1?\n"
                 f"coronal bone MIP; VerSe says L1, VERIDAH says T13",
                 fontsize=8.5)
    ax.set_xticks([]); ax.set_yticks([])
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{cid}_disputed.png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--src", required=True, help="only used to report missing cases")
    ap.add_argument("--disputed", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ids = [r["case"] for r in csv.DictReader(open(a.disputed))]
    done, missing = [], []
    for cid in ids:
        lab_p = Path(a.labels) / f"VERSE_{cid}.nii.gz"
        ct_p = Path(a.labels).parent / "imagesTr" / f"VERSE_{cid}_0000.nii.gz"
        if not (lab_p.exists() and ct_p.exists()):
            missing.append(cid)
            continue
        (done if render(ct_p, lab_p, cid, Path(a.out)) else missing).append(cid)
        print(f"  {cid}: {'rendered' if cid in done else 'no disputed level'}", flush=True)
    print(f"\n  rendered {len(done)} to {a.out}")
    if missing:
        print(f"  not available yet: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
