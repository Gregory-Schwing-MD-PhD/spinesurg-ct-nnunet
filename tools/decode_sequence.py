#!/usr/bin/env python3
"""tools/decode_sequence.py — from voxel probabilities to a posterior over the readings.

A semantic network labels voxels; it does not know that a column has to read thoracic,
thoracic, lumbar, lumbar, sacrum from top to bottom. This decoder adds that, and nothing
else, on top of the one-shot arm's output:

  1. vertebra instances: connected components of the union of all vertebra classes;
  2. per instance, the mean softmax over its voxels, folded into three TYPE probabilities
     (thoracic = T10..T13, lumbar = L1..L6, sacral = sacrum) and the most likely name;
  3. rib evidence per instance: which rib classes touch it within REACH_MM;
  4. the monotone sequence: choose the boundary k so that instances above it are thoracic
     and below it lumbar; the posterior over k is the softmax of the summed log type
     probabilities, so the rib-free count comes with a probability, not a verdict;
  5. the three readings whenever the MAP rib-free count is six:
       B (a rib-less T12 counted as lumbar)  = posterior mass on k+1,
       C (a lumbarized S1)                   = sacral share of the lowest instance's type mass,
       A (a true L6)                         = the remainder,
     and the double shift is B*C, reported rather than excluded.

    python tools/decode_sequence.py \
        --predictions_dir nnunet/results/Dataset810_.../fold_0/validation \
        --dataset_json nnunet/raw/Dataset810_SpineSurgLSTVOneShot/dataset.json \
        --labels_dir nnunet/raw/Dataset810_SpineSurgLSTVOneShot/labelsTr \
        --out results/oneshot_fold0/decoded.csv

Needs the .npz probability files nnU-Net writes with --npz next to each prediction. Volumes
are read as stored; the superior axis is taken from the affine, never assumed.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

MIN_VOX = 300
MIN_INSTANCE_MM3 = 2000.0   # smallest thing that can be a vertebra piece (2 cm^3)
REACH_MM = 4.0
THORACIC = ("T10", "T11", "T12", "T13")
LUMBAR = ("L1", "L2", "L3", "L4", "L5", "L6")
RIBS = ("rib_left", "rib_right", "rib12_left", "rib12_right", "rib13_left", "rib13_right",
        "lumbar_rib_left", "lumbar_rib_right")


def names_from(dataset_json: Path) -> dict[str, int]:
    d = json.loads(dataset_json.read_text())
    return {k: int(v) for k, v in d["labels"].items() if not isinstance(v, list)}


def sup_axis(img):
    ax = nib.aff2axcodes(img.affine)
    for i, c in enumerate(ax):
        if c in ("S", "I"):
            return i, (1 if c == "S" else -1)
    raise ValueError(f"no superior axis in {ax}")


def load_probs(npz_path: Path):
    z = np.load(str(npz_path))
    key = "probabilities" if "probabilities" in z else z.files[0]
    return z[key]                       # (C, ...) in nnU-Net's internal (transposed) order


TALL_RATIO = 1.3       # height along the superior axis, in typical single-vertebra extents
TALL_VOL = 1.6         # or volume, in median single-vertebra volumes
SEED_GAP = 0.4         # stacked seeds must be this many body heights apart (tilted spines
                       # shorten the projected gap, so it is well under one body)


def _height_vox(m: np.ndarray, axis: int) -> tuple[int, int]:
    idx = np.nonzero(m.any(axis=tuple(i for i in range(3) if i != axis)))[0]
    return int(idx.min()), int(idx.max())


def split_tall_component(m: np.ndarray, axis: int, zooms, body_mm: float, med_vol: float) -> list[np.ndarray]:
    """Split one connected component that holds more than one body. Two neighbouring
    vertebrae with the same name touch only at the facet joints, a few millimetres thick,
    so eroding the component by 2-8 mm severs them while each body survives; every voxel is
    then given to the nearest surviving seed. A component is a candidate when it is taller
    than TALL_RATIO typical extents of this case or bigger than TALL_VOL median volumes
    (extents along the scanner axis shrink on tilted spines; volume does not). Guards: the
    seeds that count are at least a tenth of the largest, and they must be STACKED (z-centroids
    at least SEED_GAP bodies apart); a body and its posterior elements overlap in z and are
    kept whole. On ground truth with L6 renamed L5 this recovers the sixth body on 0376, 1053
    and the instrumented 0068 (benchmark/debug_split.py)."""
    lo, hi = _height_vox(m, axis)
    height_mm = (hi - lo + 1) * zooms[axis]
    vol = int(m.sum())
    tall = (body_mm > 0 and height_mm > TALL_RATIO * body_mm) or (med_vol > 0 and vol > TALL_VOL * med_vol)
    if not tall:
        return [m]
    vox = float(min(zooms))
    for r_mm in (2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
        it = max(1, int(round(r_mm / vox)))
        er = ndimage.binary_erosion(m, structure=np.ones((3, 3, 3)), iterations=it)
        cc, n = ndimage.label(er, structure=np.ones((3, 3, 3)))
        if n < 2:
            continue
        sizes = ndimage.sum(er, cc, range(1, n + 1))
        floor = max(MIN_VOX, 0.1 * float(sizes.max()))
        seeds = [k + 1 for k, sz in enumerate(sizes) if sz >= floor]
        if len(seeds) < 2:
            continue
        zc = sorted(float(np.argwhere(cc == k)[:, axis].mean()) * zooms[axis] for k in seeds)
        if min(b - a for a, b in zip(zc, zc[1:])) < SEED_GAP * body_mm:
            continue                                   # pieces overlap in z: not two bodies
        seed_map = np.where(np.isin(cc, seeds), cc, 0)
        _, idx = ndimage.distance_transform_edt(seed_map == 0, return_indices=True)
        owner = seed_map[tuple(idx)] * m
        pieces = [owner == k for k in seeds]
        return [p for p in pieces if p.sum() >= MIN_VOX // 3] or [m]
    return [m]


def decode_case(pred_path: Path, npz_path: Path | None, names: dict[str, int], gt_path: Path | None):
    img = nib.load(str(pred_path))
    seg = np.asanyarray(img.dataobj).astype(np.int16)
    zooms = np.asarray(img.header.get_zooms()[:3], float)
    axis, sign = sup_axis(img)
    n_classes = max(names.values()) + 1
    if npz_path is None:
        # one-hot: a hard-label system (a competitor, or ground truth). The mean softmax over
        # an instance of name class v is then exactly e_v, so no (C, X, Y, Z) array is built.
        probs = None
    else:
        probs = load_probs(npz_path)
        # nnU-Net stores probabilities in its own axis order (transposed); align to the NIfTI array
        if probs.shape[1:] != seg.shape:
            if probs.shape[1:][::-1] == seg.shape:
                probs = np.transpose(probs, (0, 3, 2, 1))
            else:
                raise ValueError(f"probability shape {probs.shape} does not match {seg.shape}")
    vert_ids = [names[n] for n in THORACIC + LUMBAR if n in names]
    thor_ids = [names[n] for n in THORACIC if n in names]
    lum_ids = [names[n] for n in LUMBAR if n in names]
    sac_id = names.get("sacrum")
    rib_ids = {n: i for n, i in names.items() if n in RIBS or n.startswith(("rib_left_", "rib_right_"))}

    # INSTANCES PER NAME CLASS, not per union: adjacent vertebrae touch at the facets, so the
    # union of vertebra classes is one connected blob for the whole column (the self-test on
    # ground truth returned a single instance per record). The network's per-name output
    # separates neighbours by construction; two neighbours given the SAME name stay apart
    # because their centroids are a body height apart, while fragments of one vertebra
    # (a detached spinous process) share a name and a centroid and are merged.
    # pass 1: raw connected components per name class, and the case's own typical body height
    raw = {}                                          # vid -> [mask, ...]
    for vid in vert_ids:
        m_all = seg == vid
        if m_all.sum() < MIN_VOX:
            continue
        cc, n = ndimage.label(m_all, structure=np.ones((3, 3, 3)))
        raw[vid] = [cc == lab for lab in range(1, n + 1) if (cc == lab).sum() >= MIN_VOX // 3]
    whole = [m for ms in raw.values() for m in ms if m.sum() >= MIN_VOX]      # whole bodies only
    heights = [(_height_vox(m, axis)[1] - _height_vox(m, axis)[0] + 1) * zooms[axis] for m in whole]
    body_mm = float(np.median(heights)) if len(heights) >= 3 else 35.0
    med_vol = float(np.median([int(m.sum()) for m in whole])) if len(whole) >= 3 else 0.0

    comps = []                                        # (name_id, mask, z_mm)
    for vid, masks in raw.items():
        parts = []
        for m in masks:
            # TWO NEIGHBOURS GIVEN THE SAME NAME TOUCH AT THE FACETS AND FORM ONE COMPONENT
            # (an L6 called L5 next to the true L5, or a competitor with no L6 class). The
            # sequence would then be one body short, so a component much taller than this
            # case's typical body is split at the waists of its craniocaudal area profile.
            pieces = split_tall_component(m, axis, zooms, body_mm, med_vol)
            for piece in pieces:
                idx = np.argwhere(piece)
                parts.append([piece, float(idx[:, axis].mean() * zooms[axis] * sign), len(pieces) > 1])
        parts.sort(key=lambda p: -p[1])
        # fragments of ONE vertebra (a body cut off from its posterior elements by hardware,
        # a detached spinous process) share a name and sit within half a body of each other;
        # two bodies with the same name are a whole body apart. Height overlap is NOT the
        # test: a vertebra's articular processes overlap the next body in height.
        # Pieces that came out of a split are two bodies by construction and are never
        # re-merged (on a tilted spine their projected centroids can sit closer than half a
        # body: 1053 in the self-test).
        merged = []                                    # [mask, z_mm, from_split]
        for m, z, split in parts:
            hit = None if split else next((g for g in merged if not g[2] and abs(g[1] - z) < 0.5 * body_mm), None)
            if hit is not None:
                hit[0] |= m
                idx = np.argwhere(hit[0])
                hit[1] = float(idx[:, axis].mean() * zooms[axis] * sign)
            else:
                merged.append([m, z, split])
        comps += [(vid, m, z) for m, z, _ in merged]
    # an instance must be a piece of bone, not a speck: at least MIN_INSTANCE_MM3, and at
    # least a fifth of this case's median instance (label noise at another level otherwise
    # becomes a vertebra)
    vox_mm3 = float(np.prod(zooms))
    vols = [int(m.sum()) * vox_mm3 for _, m, _ in comps]
    floor_mm3 = max(MIN_INSTANCE_MM3, 0.2 * float(np.median(vols))) if vols else MIN_INSTANCE_MM3
    inst = []
    for vid, m, z_mm in comps:
        nv = int(m.sum())
        if nv < MIN_VOX or nv * vox_mm3 < floor_mm3:
            continue
        if probs is None:                                          # one-hot: e_vid
            mean_p = np.zeros(n_classes, dtype=np.float32); mean_p[vid] = 1.0
        else:
            mean_p = probs[:, m].astype(np.float32).mean(1)      # (C,)
        p_thor = float(mean_p[thor_ids].sum()); p_lum = float(mean_p[lum_ids].sum())
        p_sac = float(mean_p[sac_id]) if sac_id is not None else 0.0
        tot = max(p_thor + p_lum + p_sac, 1e-6)
        name_id = int(np.argmax([mean_p[i] if i in vert_ids else -1 for i in range(len(mean_p))]))
        name = next(k for k, v in names.items() if v == name_id)
        # rib contact within REACH_MM
        reach = tuple(int(np.ceil(REACH_MM / zz)) for zz in zooms)
        dil = ndimage.binary_dilation(m, structure=np.ones((3, 3, 3)), iterations=max(reach))
        touching = {rn: int((seg[dil] == rid).sum()) for rn, rid in rib_ids.items()}
        touching = {k: v for k, v in touching.items() if v >= 50}
        inst.append({"z_mm": z_mm, "voxels": nv, "p_thor": p_thor / tot, "p_lum": p_lum / tot,
                     "p_sac": p_sac / tot, "name": name, "ribs": touching})
    inst.sort(key=lambda r: -r["z_mm"])                 # superior first
    N = len(inst)
    if N == 0:
        return {"case": pred_path.name.replace(".nii.gz", ""), "error": "no vertebra instance"}

    # monotone decode: k thoracic on top, N-k lumbar below (sacrum is not an instance here)
    lt = np.log(np.clip([r["p_thor"] for r in inst], 1e-6, 1))
    ll = np.log(np.clip([r["p_lum"] for r in inst], 1e-6, 1))
    scores = np.array([lt[:k].sum() + ll[k:].sum() for k in range(N + 1)])
    post = np.exp(scores - scores.max()); post /= post.sum()
    k_map = int(np.argmax(post))
    rib_free_post = {int(N - k): float(post[k]) for k in range(N + 1)}
    rib_free_map = N - k_map

    low = inst[-1]
    p_c = low["p_sac"] / max(low["p_sac"] + low["p_lum"], 1e-6)     # lowest body sacral-type share
    p_b = float(post[k_map + 1]) if k_map + 1 <= N else 0.0          # one more body thoracic on top
    readings = None
    if rib_free_map == 6:
        p_a = max(0.0, 1.0 - p_b - p_c + p_b * p_c)
        readings = {"A_true_L6": round(p_a, 3), "B_ribless_T12": round(p_b, 3),
                    "C_lumbarized_S1": round(p_c, 3), "double_shift": round(p_b * p_c, 3)}

    out = {"case": pred_path.name.replace(".nii.gz", ""), "n_instances": N, "k_thoracic_map": k_map,
           "rib_free_map": rib_free_map, "rib_free_prob": round(float(post[k_map]), 3),
           "rib_free_posterior": json.dumps({str(k): round(v, 3) for k, v in rib_free_post.items() if v > 0.001}),
           "lowest_p_sac": round(low["p_sac"], 3), "top_ribfree_p_thor": round(inst[k_map]["p_thor"], 3) if k_map < N else "",
           "readings_if_six": json.dumps(readings) if readings else "",
           "instances": json.dumps([{"z": round(r["z_mm"], 1), "name": r["name"], "thor": round(r["p_thor"], 3),
                                     "lum": round(r["p_lum"], 3), "sac": round(r["p_sac"], 3), "ribs": r["ribs"]}
                                    for r in inst])}
    if gt_path is not None and gt_path.exists():
        gt = np.asanyarray(nib.load(str(gt_path)).dataobj).astype(np.int16)
        gt_lum = sum(1 for i in lum_ids if (gt == i).sum() >= MIN_VOX)
        out["gt_rib_free"] = gt_lum
        out["count_error"] = rib_free_map - gt_lum
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions_dir", required=True, type=Path)
    ap.add_argument("--dataset_json", required=True, type=Path)
    ap.add_argument("--labels_dir", type=Path, default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--onehot", action="store_true",
                    help="no softmax on disk (a competitor's hard labels): build one-hot "
                         "probabilities from the label map, so the decoder reads its names as "
                         "certainties and the posterior collapses to that system's own count")
    a = ap.parse_args()
    names = names_from(a.dataset_json)
    preds = sorted(a.predictions_dir.glob("*.nii.gz"))
    if a.limit:
        preds = preds[:a.limit]
    rows = []
    for p in preds:
        npz = p.with_suffix("").with_suffix(".npz")
        if not npz.exists() and not a.onehot:
            print("no npz for", p.name); continue
        gt = (a.labels_dir / p.name) if a.labels_dir else None
        try:
            rows.append(decode_case(p, None if a.onehot else npz, names, gt))
        except Exception as e:  # noqa: BLE001
            rows.append({"case": p.name.replace(".nii.gz", ""), "error": repr(e)})
        print(rows[-1].get("case"), rows[-1].get("rib_free_map", rows[-1].get("error")), rows[-1].get("gt_rib_free", ""), flush=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "case", k))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    ok = [r for r in rows if "count_error" in r]
    if ok:
        exact = sum(1 for r in ok if r["count_error"] == 0)
        print(f"rib-free count exact in {exact}/{len(ok)}; mean |error| {np.mean([abs(r['count_error']) for r in ok]):.2f}")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
