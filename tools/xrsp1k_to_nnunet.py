"""xrsp1k_to_nnunet.py — convert XRSpinoPelvic1K (DRR + dense mask) into an nnU-Net v2 2D
raw dataset for training the open spinopelvic-radiograph segmenter.

Input  : the XRSpinoPelvic1K folder (per-case  <view>_drr.png  +  <view>_mask.png).
Output : $nnUNet_raw/Dataset<ID>_<NAME>/ with
           imagesTr/<case>_0000.png  labelsTr/<case>.png   (train + internal CV)
           imagesTs/<case>_0000.png  labelsTs/<case>.png   (held-out test, for our eval)
           dataset.json   label_map.json   test_cases.json

The v4 VerSe-native ids (sparse, 8..57) are remapped to CONSECUTIVE nnU-Net ids (0=bg);
label_map.json records the nnU-Net<->v4 mapping so eval can convert predictions back to v4
ids and measure angles with ostk.metrics2d.

  python tools/xrsp1k_to_nnunet.py --xrsp1k data/xrsp1k --out $nnUNet_raw \
      --dataset_id 910 --name XRSpinoPelvic1K --view lateral --test_frac 0.15
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def v4_names() -> dict:
    """v4 VerSe-native id -> structure name (bone subset), mirrors CTSpinoPelvic1K
    label_scheme.label_dict()."""
    names = {}
    verse = (["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
             + [f"T{n}" for n in range(1, 13)] + [f"L{n}" for n in range(1, 7)])
    for i, nm in enumerate(verse, start=1):                  # 1..25
        names[i] = nm
    names[26] = "sacrum"; names[27] = "coccyx"; names[28] = "T13"; names[29] = "S1"
    names[30] = "left_hip"; names[31] = "right_hip"
    names[32] = "femur_left"; names[33] = "femur_right"
    for n in range(1, 13):
        names[33 + n] = f"rib_left_{n}"                      # 34..45
    for n in range(1, 13):
        names[45 + n] = f"rib_right_{n}"                     # 46..57
    return names


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xrsp1k", required=True, type=Path, help="generated XRSpinoPelvic1K folder")
    ap.add_argument("--out", required=True, type=Path, help="nnUNet_raw root")
    ap.add_argument("--dataset_id", type=int, default=910)
    ap.add_argument("--name", default="XRSpinoPelvic1K")
    ap.add_argument("--view", default="lateral", help="which projection to train on")
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    cases = sorted(p.parent.name for p in a.xrsp1k.glob(f"*/{a.view}_mask.png"))
    if not cases:
        raise SystemExit(f"no cases with {a.view}_mask.png under {a.xrsp1k}")

    NAMES = v4_names()
    masks, present = {}, set()
    for c in cases:
        m = np.array(Image.open(a.xrsp1k / c / f"{a.view}_mask.png"))
        masks[c] = m
        present.update(int(x) for x in np.unique(m) if x != 0)
    present = sorted(present)

    remap = {0: 0}
    labels_json = {"background": 0}
    for new_id, v4id in enumerate(present, start=1):
        remap[v4id] = new_id
        labels_json[NAMES.get(v4id, f"id{v4id}")] = new_id
    label_map = {"nnunet_to_v4": {str(remap[v]): {"v4_id": v, "name": NAMES.get(v, f"id{v}")}
                                  for v in present}}

    ds = a.out / f"Dataset{a.dataset_id:03d}_{a.name}"
    for sub in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        (ds / sub).mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(a.seed)                      # deterministic split
    idx = np.arange(len(cases)); rng.shuffle(idx)
    n_test = int(round(a.test_frac * len(cases)))
    test = {cases[i] for i in idx[:n_test]}

    lut = np.zeros(int(max(present)) + 1, dtype=np.uint8)    # v4 id -> consecutive, vectorised
    for v, n in remap.items():
        lut[v] = n

    n_tr = 0
    for c in cases:
        split = "Ts" if c in test else "Tr"
        Image.open(a.xrsp1k / c / f"{a.view}_drr.png").convert("L").save(
            ds / f"images{split}" / f"{c}_0000.png")
        Image.fromarray(lut[masks[c]]).save(ds / f"labels{split}" / f"{c}.png")
        n_tr += split == "Tr"

    json.dump({
        "channel_names": {"0": "grayscale"},
        "labels": labels_json,
        "numTraining": n_tr,
        "file_ending": ".png",
        "overwrite_image_reader_writer": "NaturalImage2DIO",
        "description": (f"XRSpinoPelvic1K {a.view} DRR -> dense spinopelvic mask; v4 ids "
                        "remapped consecutively (see label_map.json)"),
    }, open(ds / "dataset.json", "w"), indent=2)
    json.dump(label_map, open(ds / "label_map.json", "w"), indent=2)
    json.dump(sorted(test), open(ds / "test_cases.json", "w"), indent=2)
    print(f"wrote {ds}\n  {n_tr} train+val, {len(test)} test, {len(present)} foreground classes")
    print("  NOTE: split is case-level; for patient-disjoint, group by patient via the "
          "CTSpinoPelvic1K manifest token before splitting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
