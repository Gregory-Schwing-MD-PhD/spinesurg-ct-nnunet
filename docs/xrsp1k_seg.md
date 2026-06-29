# Training the XRSpinoPelvic1K 2D segmenter (open spinopelvic-radiograph model)

A 2D nnU-Net that segments the spinopelvic skeleton (thoracic/lumbar vertebrae, sacrum,
femoral heads, ribs) on radiographs, trained on XRSpinoPelvic1K. Its predicted masks feed
`ostk.metrics2d` to recover PI/SS/PT/LL with no manual landmarks — the paper's segmentation
baseline.

> **Not "first."** TotalSegmentator 2D (`tsxr`) is an open whole-skeleton DRR-trained
> radiograph segmenter. Ours is spine-specific and angle-driving; positioned as a baseline,
> not a first. Keep that framing in any writeup.

## 0. Prereqs
- nnU-Net v2 (`pip install nnunetv2`), and the standard env vars `nnUNet_raw`,
  `nnUNet_preprocessed`, `nnUNet_results`.
- The generated XRSpinoPelvic1K folder (from `XRaySpinoPelvic1K/scripts/ship_xrsp1k.sh`).
- OpenSpineToolkit importable (`pip install -e` the ostk repo) for the angle eval.

## 1. Convert to an nnU-Net 2D dataset
```bash
python tools/xrsp1k_to_nnunet.py \
  --xrsp1k /path/to/xrsp1k --out "$nnUNet_raw" \
  --dataset_id 910 --name XRSpinoPelvic1K --view lateral --test_frac 0.15
```
Writes `Dataset910_XRSpinoPelvic1K/` (imagesTr/labelsTr + imagesTs/labelsTs), `dataset.json`,
`label_map.json` (nnU-Net↔v4 ids), `test_cases.json`. (Split is case-level; for a
patient-disjoint split, group by the CTSpinoPelvic1K manifest token first.)

## 2. Preprocess + train (single fold for the baseline)
```bash
nnUNetv2_plan_and_preprocess -d 910 -c 2d --verify_dataset_integrity
nnUNetv2_train 910 2d 0          # fold 0 only — the quick baseline
# camera-ready: add folds 1..4 (and/or more epochs) for full 5-fold cross-validation
```

## 3. Predict the held-out test split
```bash
nnUNetv2_predict \
  -i "$nnUNet_raw/Dataset910_XRSpinoPelvic1K/imagesTs" \
  -o /path/to/preds -d 910 -c 2d -f 0
```

## 4. Evaluate (Dice + end-to-end angle error)
```bash
python tools/eval_xrsp1k_seg.py \
  --pred /path/to/preds \
  --gt "$nnUNet_raw/Dataset910_XRSpinoPelvic1K/labelsTs" \
  --label_map "$nnUNet_raw/Dataset910_XRSpinoPelvic1K/label_map.json"
```
Prints per-class Dice and the PI/SS/PT/LL MAE when the angles are measured through the
**predicted** masks vs the GT masks (the eval monkeypatches `ostk.labels` to the v4 scheme at
runtime, so the installed ostk — which the live demo runs on the v3 scheme — is untouched).

Paste mean Dice and the angle MAEs into the paper's "Segmentation baseline" result and the
`\TODO`s in Table/contribution 4.
