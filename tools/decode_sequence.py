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


def decode_case(pred_path: Path, npz_path: Path, names: dict[str, int], gt_path: Path | None):
    img = nib.load(str(pred_path))
    seg = np.asanyarray(img.dataobj).astype(np.int16)
    zooms = np.asarray(img.header.get_zooms()[:3], float)
    axis, sign = sup_axis(img)
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
    rib_ids = {n: names[n] for n in RIBS if n in names}

    vert = np.isin(seg, vert_ids)
    cc, n = ndimage.label(vert, structure=np.ones((3, 3, 3)))
    inst = []
    for lab in range(1, n + 1):
        m = cc == lab
        nv = int(m.sum())
        if nv < MIN_VOX:
            continue
        idx = np.argwhere(m)
        z_mm = float(idx[:, axis].mean() * zooms[axis] * sign)
        mean_p = probs[:, m].astype(np.float32).mean(1)          # (C,)
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
    a = ap.parse_args()
    names = names_from(a.dataset_json)
    preds = sorted(a.predictions_dir.glob("*.nii.gz"))
    if a.limit:
        preds = preds[:a.limit]
    rows = []
    for p in preds:
        npz = p.with_suffix("").with_suffix(".npz")
        if not npz.exists():
            print("no npz for", p.name); continue
        gt = (a.labels_dir / p.name) if a.labels_dir else None
        try:
            rows.append(decode_case(p, npz, names, gt))
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
