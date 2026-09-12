"""Does the disputed vertebra bear a rib? Adjudicate T13-against-L6 on the anatomy.

    python tools/versefusion_adjudicate_t13.py --labels ~/vf_813/labelsTr \
        --disputed logs/versefusion_disputed.csv

THE DISPUTE. VerSe's masks label a certain vertebra L1. Hendrik Moller's VERIDAH
corrections say that same vertebra (fid vert=20, VerSe's L1) is a T13 bearing supernumerary
ribs on both sides -- SR_l=1, SR_r=1 -- and that the column below it therefore numbers
L1..L5 rather than L1..L6. On six of the twelve cases that removes an L6; on the other six
there was no sixth in the field of view to remove. Taking data/unified as shipped adopts
VerSe's reading on all twelve; taking the audit CSV adopts Moller's.

WHY THIS IS DECIDABLE RATHER THAN A MATTER OF OPINION. A thoracic vertebra is thoracic
because it articulates with a rib. That is the discriminator this release is built on -- it
carries T13 with its own rib pair for a thoracic-type thirteenth vertebra, and separate
lumbar-rib classes for a lumbar body with a rudimentary one -- and it is what Moller cites.
So the question "is this bone a T13 or an L1" reduces to "does it have a rib", which is in
the image.

HOW IT IS MEASURED, AND WHY THE CONTROLS MATTER. Rib contact is reported for the disputed
vertebra AND for its neighbours in the same case: T11 and T12, which certainly bear ribs,
and L2, which certainly does not. Those calibrate the measurement per case, so the verdict
is "L1's rib contact looks like T12's" or "looks like L2's" rather than a threshold carried
in from somewhere else. A case whose controls do not separate is reported as unusable
rather than forced to an answer.

WHAT CAN GO WRONG, STATED. TotalSegmentator numbers ribs 1-12. On a spine with thirteen rib
pairs it must either miss one or fold it into another, so ABSENCE of rib contact at the
disputed level is weaker evidence than presence. A clean positive says Moller is right; a
negative is a reason to look at the case, not to overrule him.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

# Dataset813 ids
T11, T12, L1, L2 = 2, 3, 5, 6
RIBS = list(range(12, 38))          # every rib class, both sides
CONTACT_MM = 6.0                    # a rib head sits within a few mm of its vertebra


def rib_contact(lab: np.ndarray, zooms, cid: int, ribs: np.ndarray) -> dict:
    """How much rib sits within CONTACT_MM of this vertebra, and on which sides."""
    m = lab == cid
    out = {"present": bool(m.any()), "n_vox": int(m.sum())}
    if not m.any() or not ribs.any():
        out.update(contact_vox=0, contact_frac=0.0, sides=0)
        return out
    # CROP FIRST. A distance transform over a 512x512x743 volume, eight of them per case,
    # is what got this killed on a shared node -- and it is wasted work: the question only
    # reaches CONTACT_MM from the vertebra, so everything beyond that margin is irrelevant.
    idx = np.argwhere(m)
    pad = [int(np.ceil(CONTACT_MM / z)) + 2 for z in zooms]
    lo = np.maximum(idx.min(axis=0) - pad, 0)
    hi = np.minimum(idx.max(axis=0) + pad + 1, m.shape)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    m_c, ribs_c = m[sl], ribs[sl]
    dist = ndimage.distance_transform_edt(~m_c, sampling=zooms)
    near_c = ribs_c & (dist <= CONTACT_MM)
    near = np.zeros_like(m)
    near[sl] = near_c
    out["contact_vox"] = int(near.sum())
    out["contact_frac"] = float(near.sum()) / float(m.sum())
    # left and right, split at the centroid of this vertebra along the L-R axis.
    # Dataset813 is stored PIR, so axis 2 runs to the patient's right.
    if near.any():
        cx = float(np.argwhere(m)[:, 2].mean())
        xs = np.argwhere(near)[:, 2]
        out["sides"] = int((xs < cx).any()) + int((xs > cx).any())
    else:
        out["sides"] = 0
    return out


def adjudicate(lab_path: Path) -> dict:
    import nibabel as nib
    img = nib.load(str(lab_path))
    lab = np.asanyarray(img.dataobj).astype(np.uint8)
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    ribs = np.isin(lab, RIBS)

    rec = {"case": lab_path.stem.replace("VERSE_", ""),
           "n_rib_classes": int(len(set(np.unique(lab[ribs]).tolist()))) if ribs.any() else 0}
    for name, cid in (("T11", T11), ("T12", T12), ("L1", L1), ("L2", L2)):
        rec[name] = rib_contact(lab, zooms, cid, ribs)

    t12, l1, l2 = rec["T12"], rec["L1"], rec["L2"]
    if not (t12["present"] and l1["present"] and l2["present"]):
        rec["verdict"] = "incomplete: a control level is outside the field of view"
        return rec
    if t12["contact_frac"] <= 0 or t12["contact_frac"] <= l2["contact_frac"]:
        rec["verdict"] = "unusable: the controls do not separate"
        return rec

    # where does L1 sit between "no rib" (L2) and "rib" (T12)?
    span = t12["contact_frac"] - l2["contact_frac"]
    rec["l1_between_controls"] = float((l1["contact_frac"] - l2["contact_frac"]) / span)
    if rec["l1_between_controls"] >= 0.5 and l1["sides"] == 2:
        rec["verdict"] = "THORACIC: L1 bears ribs bilaterally -- supports Moller (T13)"
    elif rec["l1_between_controls"] >= 0.5:
        rec["verdict"] = "thoracic-leaning: rib contact on one side only"
    elif rec["l1_between_controls"] <= 0.2:
        rec["verdict"] = "LUMBAR: L1 looks like L2 -- supports VerSe (L1, so a sixth lumbar)"
    else:
        rec["verdict"] = "equivocal"
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True, help="vf_813/labelsTr")
    ap.add_argument("--disputed", default="", help="versefusion_disputed.csv")
    ap.add_argument("--cases", default="", help="comma-separated case ids instead")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    if a.cases:
        ids = [c.strip() for c in a.cases.split(",") if c.strip()]
    elif a.disputed:
        ids = [r["case"] for r in csv.DictReader(open(a.disputed))]
    else:
        print("need --disputed or --cases")
        return 2

    lab_dir = Path(a.labels)
    recs, missing = [], []
    for cid in ids:
        p = lab_dir / f"VERSE_{cid}.nii.gz"
        if not p.exists():
            missing.append(cid)
            continue
        recs.append(adjudicate(p))

    print(f"{len(recs)} adjudicated, {len(missing)} not yet converted {missing}\n")
    print(f"  {'case':<11} {'T11':>7} {'T12':>7} {'L1':>7} {'L2':>7} {'sides':>5}  verdict")
    for r in recs:
        print(f"  {r['case']:<11} "
              f"{r['T11']['contact_frac']:7.3f} {r['T12']['contact_frac']:7.3f} "
              f"{r['L1']['contact_frac']:7.3f} {r['L2']['contact_frac']:7.3f} "
              f"{r['L1']['sides']:>5}  {r['verdict']}")

    tally: dict = {}
    for r in recs:
        k = r["verdict"].split(":")[0]
        tally[k] = tally.get(k, 0) + 1
    print(f"\n  {tally}")
    print("\n  A clean THORACIC verdict supports Moller. A LUMBAR verdict is a reason to")
    print("  look at the case, not to overrule him: TotalSegmentator numbers ribs 1-12,")
    print("  so on a thirteen-rib spine it must miss or merge one, which makes absence")
    print("  weaker evidence than presence.")
    if a.out:
        Path(a.out).write_text(json.dumps(recs, indent=1))
        print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
