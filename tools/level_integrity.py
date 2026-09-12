"""Is each vertebra ONE class, and is every class ONE vertebra?

Per-class Dice answers neither question. A run can hold Dice 0.87 on every lumbar level and
still be unusable, because Dice is computed per class over a whole volume and cannot see
that the voxels came from the wrong bodies. The three failures that matter here are all
invisible to it:

  COLLAPSE    every L6 voxel is called L5. L5's Dice barely moves -- it gains voxels it
              should not have, from a class that contributes a rounding error to the
              denominator -- and L6's Dice is 0. Seen as "L6 is hard", it is actually "L6
              does not exist as far as the network is concerned".

  SMEAR       one vertebral body is split down the middle, its top half called L5 and its
              bottom half L6. BOTH classes score about 0.5. Nothing is missing and nothing
              is extra; the boundary is simply in the wrong place. Two mediocre Dice values
              look like a model that needs more epochs, and it is not: the decoder will
              read that body as two vertebrae and return a column one level too long.

  MERGE       two neighbouring bodies both called L5. The decoder has a stop-gap for this
              (decode_sequence.split_tall_component erodes the component until it severs at
              the facets), so it is often recoverable -- but only if it is MEASURED, because
              a recovered case and a broken one look identical in Dice.

The count is the thing the dataset exists to get right, so it is reported directly: how many
distinct lumbar names the network used against how many lumbar bodies the ground truth has.
Five names on six bodies is the headline failure of this project and it should not have to
be inferred from anything.

WHAT A PATCH CAN AND CANNOT TELL YOU. During training these run on 3-D patches, which cut
bodies at the patch edge. That costs less than it looks: clipping leaves a PARTIAL body, not
a two-named one, and a smear needs two predicted names inside one ground-truth body, so the
patch edge cannot manufacture one. Slivers are dropped by a voxel floor. The count is the
part that is patch-relative, and it is defined that way rather than guarded: both sides are
counted over the bodies present in the array, so on a patch it reads "names used for the
bodies here" and on a whole volume "names used for the column". The same functions run
offline on whole volumes, which is what `main` does.

    python tools/level_integrity.py --pred PRED.nii.gz --gt GT.nii.gz \
        --dataset_json nnunet/raw/Dataset813_.../dataset.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# A body must give up this much of itself to a second name before that counts as a smear
# rather than a ragged edge. Boundary voxels are a few percent of a body; a genuine smear
# takes a third or more.
SMEAR_FRAC = 0.20
# ... and the modal name must hold at least this much for the body to count as intact.
PURE_FRAC = 0.80
# A predicted name claims a body when it takes this much of it. Deliberately higher than
# SMEAR_FRAC: "this name covers two bodies" is a stronger claim than "this body leaks".
CLAIM_FRAC = 0.30
# Ignore specks. A lumbar body is ~10^5 voxels at 1 mm; 200 is noise by any measure.
MIN_VOX = 200


def _as_int(a) -> np.ndarray:
    return np.asarray(a).astype(np.int32, copy=False)


def collapse_profile(pred, gt, cid: int, names: Optional[Dict[int, str]] = None,
                     top: int = 4) -> dict:
    """Where do the ground-truth voxels of class `cid` actually end up?

    The full distribution over predicted names, not a hand-picked list of suspects: a fixed
    list answers only the question you already thought to ask, and reports 0.0% for the
    class that is in fact eating the bone.
    """
    pred, gt = _as_int(pred), _as_int(gt)
    m = gt == cid
    n = int(m.sum())
    out = {"cid": cid, "name": (names or {}).get(cid, str(cid)), "n_gt_vox": n}
    if n == 0:
        out["present"] = False
        return out
    vals, counts = np.unique(pred[m], return_counts=True)
    order = np.argsort(-counts)
    frac = [(int(vals[i]), float(counts[i]) / n) for i in order[:top]]
    out["present"] = True
    out["to"] = [{"cid": v, "name": (names or {}).get(v, str(v)), "frac": f} for v, f in frac]
    out["self_frac"] = next((f for v, f in frac if v == cid), 0.0)
    # the largest recipient that is neither the class itself nor background
    steal = [(v, f) for v, f in frac if v not in (cid, 0)]
    out["top_other"] = ({"cid": steal[0][0], "name": (names or {}).get(steal[0][0], str(steal[0][0])),
                         "frac": steal[0][1]} if steal else None)
    out["collapsed"] = bool(out["self_frac"] < 0.10 and steal and steal[0][1] >= 0.30)
    return out


def _usable_labels(gt, ids: Sequence[int]) -> List[int]:
    """GT labels with enough voxels here to say anything about.

    No boundary test. A body clipped by the patch edge is a PARTIAL body, not a two-named
    one, and a smear needs two predicted names inside one ground-truth body -- which the
    patch edge cannot produce. The voxel floor is what removes slivers too small to judge.
    """
    gt = _as_int(gt)
    return [c for c in ids if int((gt == c).sum()) >= MIN_VOX]


def body_purity(pred, gt, ids: Sequence[int],
                names: Optional[Dict[int, str]] = None) -> dict:
    """Is each ground-truth body ONE predicted name?

    Reports, per body, the modal predicted name and the share it holds. A body whose modal
    share is below PURE_FRAC while a second name holds SMEAR_FRAC is SMEARED: the network
    put a level boundary through the middle of a vertebra.
    """
    pred, gt = _as_int(pred), _as_int(gt)
    use = _usable_labels(gt, ids)
    bodies, smeared = [], []
    for cid in use:
        m = gt == cid
        n = int(m.sum())
        vals, counts = np.unique(pred[m], return_counts=True)
        order = np.argsort(-counts)
        modal, modal_f = int(vals[order[0]]), float(counts[order[0]]) / n
        second = (int(vals[order[1]]), float(counts[order[1]]) / n) if len(order) > 1 else (None, 0.0)
        rec = {"cid": cid, "name": (names or {}).get(cid, str(cid)), "n_vox": n,
               "modal": modal, "modal_name": (names or {}).get(modal, str(modal)),
               "purity": modal_f,
               "second": second[0], "second_frac": second[1],
               "second_name": (names or {}).get(second[0], str(second[0])),
               "n_names_over_smear": int((counts / n >= SMEAR_FRAC).sum())}
        rec["smeared"] = bool(modal_f < PURE_FRAC and second[1] >= SMEAR_FRAC)
        rec["misnamed"] = bool(modal != cid and modal != 0)
        bodies.append(rec)
        if rec["smeared"]:
            smeared.append(rec)
    return {"n_bodies": len(use), "bodies": bodies, "smeared": smeared,
            "n_smeared": len(smeared),
            "mean_purity": float(np.mean([b["purity"] for b in bodies])) if bodies else float("nan")}


def class_spread(pred, gt, ids: Sequence[int],
                 names: Optional[Dict[int, str]] = None) -> dict:
    """Is each predicted name ONE body?

    A name that takes CLAIM_FRAC of two different ground-truth bodies has merged them. The
    decoder can sometimes sever this (split_tall_component), but only a measurement says
    whether it had to.
    """
    pred, gt = _as_int(pred), _as_int(gt)
    use = _usable_labels(gt, ids)
    claims: Dict[int, List[int]] = {}
    for cid in use:
        m = gt == cid
        n = int(m.sum())
        vals, counts = np.unique(pred[m], return_counts=True)
        for v, c in zip(vals, counts):
            if v == 0 or int(v) not in ids:
                continue
            if c / n >= CLAIM_FRAC:
                claims.setdefault(int(v), []).append(cid)
    merged = [{"pred": v, "pred_name": (names or {}).get(v, str(v)),
               "covers": sorted(gs),
               "covers_names": [(names or {}).get(g, str(g)) for g in sorted(gs)]}
              for v, gs in claims.items() if len(gs) > 1]
    return {"merged": merged, "n_merged": len(merged)}


def count_integrity(pred, gt, lumbar_ids: Sequence[int],
                    names: Optional[Dict[int, str]] = None) -> dict:
    """How many distinct lumbar names did the network use, against how many bodies exist?

    This is the number the dataset exists to get right. Five names on six bodies is the
    failure this project is about, and it is stated rather than inferred.
    """
    pred, gt = _as_int(pred), _as_int(gt)
    gt_used = [c for c in lumbar_ids if (gt == c).sum() >= MIN_VOX]
    pr_used = [c for c in lumbar_ids if (pred == c).sum() >= MIN_VOX]
    return {"n_gt_lumbar": len(gt_used), "n_pred_lumbar": len(pr_used),
            "deficit": len(gt_used) - len(pr_used),
            "gt_names": [(names or {}).get(c, str(c)) for c in gt_used],
            "pred_names": [(names or {}).get(c, str(c)) for c in pr_used],
            "missing": [(names or {}).get(c, str(c)) for c in gt_used if c not in pr_used]}


def report(pred, gt, lumbar_ids: Sequence[int], focus_cid: int,
           names: Optional[Dict[int, str]] = None) -> dict:
    """Everything above for one case."""
    return {
        "collapse": collapse_profile(pred, gt, focus_cid, names),
        "purity": body_purity(pred, gt, lumbar_ids, names),
        "spread": class_spread(pred, gt, lumbar_ids, names),
        "count": count_integrity(pred, gt, lumbar_ids, names),
    }


def format_report(r: dict, title: str = "LEVEL INTEGRITY") -> List[str]:
    """Say what happened in words, so it cannot be missed in a log."""
    L = [f"--- {title} ---"]
    c = r["collapse"]
    if not c.get("present"):
        L.append(f"  {c['name']}: absent from ground truth here")
    else:
        L.append(f"  {c['name']}: {100*c['self_frac']:5.1f}% of its {c['n_gt_vox']} voxels "
                 f"kept the name {c['name']}")
        if c["top_other"]:
            t = c["top_other"]
            L.append(f"      largest recipient: {t['name']} takes {100*t['frac']:.1f}%"
                     + ("   <-- COLLAPSE" if c["collapsed"] else ""))
    p = r["purity"]
    L.append(f"  bodies whole: {p['n_bodies'] - p['n_smeared']} of {p['n_bodies']}"
             f"   mean purity {p['mean_purity']:.3f}")
    for b in p["smeared"]:
        L.append(f"      SMEARED {b['name']}: {100*b['purity']:.0f}% {b['modal_name']} / "
                 f"{100*b['second_frac']:.0f}% {b['second_name']}   <-- one body, two names")
    s = r["spread"]
    for m in s["merged"]:
        L.append(f"      MERGED  {m['pred_name']} covers {', '.join(m['covers_names'])}"
                 f"   <-- one name, {len(m['covers'])} bodies")
    n = r["count"]
    flag = "   <-- COUNT WRONG" if n["deficit"] else ""
    L.append(f"  lumbar names used {n['n_pred_lumbar']} for {n['n_gt_lumbar']} bodies{flag}")
    if n["missing"]:
        L.append(f"      never predicted: {', '.join(n['missing'])}")
    return L


def aggregate(reports: Sequence[dict]) -> Dict[str, float]:
    """Scalars for W&B. Rates, not counts, so they read the same at any cohort size."""
    if not reports:
        return {}
    coll = [r["collapse"] for r in reports if r["collapse"].get("present")]
    pur = [r["purity"] for r in reports]
    spr = [r["spread"] for r in reports]
    cnt = [r["count"] for r in reports]
    n_int = sum(p["n_bodies"] for p in pur)
    out = {
        "level_integrity/n_cases": float(len(reports)),
        "level_integrity/smeared_body_rate": (sum(p["n_smeared"] for p in pur) / n_int) if n_int else float("nan"),
        "level_integrity/mean_body_purity": float(np.nanmean([p["mean_purity"] for p in pur])),
        "level_integrity/merged_name_rate": float(np.mean([s["n_merged"] for s in spr])),
        "level_integrity/lumbar_count_deficit": float(np.mean([c["deficit"] for c in cnt])),
        "level_integrity/count_wrong_rate": float(np.mean([1.0 if c["deficit"] else 0.0 for c in cnt])),
    }
    if coll:
        out["level_integrity/focus_self_frac"] = float(np.mean([c["self_frac"] for c in coll]))
        out["level_integrity/focus_collapse_rate"] = float(np.mean([1.0 if c["collapsed"] else 0.0 for c in coll]))
        # which class is eating it, as a rate per recipient
        tally: Dict[str, int] = {}
        for c in coll:
            if c["top_other"]:
                tally[c["top_other"]["name"]] = tally.get(c["top_other"]["name"], 0) + 1
        for k, v in tally.items():
            out[f"level_integrity/focus_taken_by/{k}"] = v / len(coll)
    return out


def names_from_dataset_json(path) -> Tuple[Dict[int, str], List[int], Optional[int]]:
    labels = json.loads(Path(path).read_text())["labels"]
    by_id, lumbar, l6 = {}, [], None
    for name, v in labels.items():
        if name in ("background", "ignore") or isinstance(v, (list, tuple)):
            continue
        by_id[int(v)] = name
        if name in ("L1", "L2", "L3", "L4", "L5", "L6"):
            lumbar.append(int(v))
        if name == "L6":
            l6 = int(v)
    return by_id, sorted(lumbar), l6


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="predicted segmentation (NIfTI)")
    ap.add_argument("--gt", required=True, help="ground truth (NIfTI)")
    ap.add_argument("--dataset_json", required=True)
    a = ap.parse_args()

    import nibabel as nib
    names, lumbar, l6 = names_from_dataset_json(a.dataset_json)
    if l6 is None:
        print("this dataset has no L6 class", file=sys.stderr)
        return 2
    pred = np.asanyarray(nib.load(a.pred).dataobj)
    gt = np.asanyarray(nib.load(a.gt).dataobj)
    if pred.shape != gt.shape:
        print(f"shape mismatch {pred.shape} vs {gt.shape}", file=sys.stderr)
        return 2
    r = report(pred, gt, lumbar, l6, names)
    print("\n".join(format_report(r, f"LEVEL INTEGRITY  {Path(a.pred).name}")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
