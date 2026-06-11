# Fused-only (no-ignore) ablation — reviewer "fused-only vs all-cases"

Reviewer ask (camera-ready, see `_rebuttal_r2.txt` item 1): a **fused-only vs
all-cases** ablation, evaluated on the **fused test split**, to show what the
partial-annotation *ignore protocol* buys over simply training on the
fully-annotated fused cases.

This ablation trains **one fold (fold 0)** of an nnU-Net that:

- sees **only `config == "fused"` records** (every `spine_only` /
  `pelvic_native` partial record is dropped from train, val, AND test), and
- uses **no ignore label** — fused records are fully annotated and carry zero
  ignore voxels by construction (`export_hf.py` enforces this), so the
  9-key scheme `{bg, L1-L4, last_lumbar, sacrum, left_hip, right_hip}` has no
  `ignore` class and no ignore-masked loss.

The **L6→last_lumbar merge is kept**, so the label semantics are identical to
the main `Dataset803` model. The two arms therefore differ in exactly two
coupled things — training pool (fused-only) and the ignore protocol (off) —
which is precisely the variable the reviewer asked us to isolate.

Everything reuses the main pipeline; only `DATASET_ID`/`DATASET_NAME` and two
convert flags change. The main 803 run is never touched (different dataset dir).

## How it's wired

- `tools/convert_hf_to_nnunet.py` — new flags:
  - `--include_configs fused` → keep only fused records in every split.
  - `--drop_ignore_label` → 9-key no-ignore scheme; source ignore(10)→bg(0)
    defensively (never fires on fused data). Incompatible with `--no_remap`.
- `slurm/spine_prep.sh` — passes those through via `INCLUDE_CONFIGS` /
  `DROP_IGNORE_LABEL`, and its `dataset.json` sanity check accepts the
  no-ignore scheme.
- `slurm/spine_train_array.sh` — unchanged; already `DATASET_ID`/`DATASET_NAME`
  parameterized. Train one fold with `--array=0`.
- `slurm/spine_eval_single.sh` — now `DATASET_ID`/`DATASET_NAME`/`N_CLASSES`
  parameterized (`N_CLASSES=9` for the no-ignore scheme).

Dataset ID **804 / `SpineSurgCTFusedOnly`** is used here (fresh; the stale
`slurm/spine_ablation.sh` referenced removed convert flags and is not used).

## Run it (3 sbatch jobs, on the cluster)

```bash
# 1) Build + preprocess the fused-only no-ignore dataset (CPU, ~1h)
INCLUDE_CONFIGS=fused DROP_IGNORE_LABEL=1 \
  DATASET_ID=804 DATASET_NAME=SpineSurgCTFusedOnly \
  sbatch slurm/spine_prep.sh

# 2) Train fold 0 only (H200, ~30-40h; requeue-safe)
DATASET_ID=804 DATASET_NAME=SpineSurgCTFusedOnly \
  TRAINER=nnUNetTrainerWandB_500ep_LSTVOversample \
  sbatch --array=0 slurm/spine_train_array.sh

# 3) Predict + eval on the fused test split (imagesTs is already fused-only)
FOLD=0 DATASET_ID=804 DATASET_NAME=SpineSurgCTFusedOnly \
  TRAINER=nnUNetTrainerWandB_500ep_LSTVOversample N_CLASSES=9 \
  sbatch slurm/spine_eval_single.sh
```

Use the same `TRAINER` as the main reported 5-fold run so the two arms are
apples-to-apples (the array default is `nnUNetTrainerWandB_500ep_LSTVOversample`).

### Why the trainer needs no changes

`nnunet_wandb_variant.py` always installs `DC_and_CE_loss(ignore_label=9)` in
`_maybe_apply_ce_reweighting`. On the fused-only dataset **no voxel ever equals
9** (labels are 0–8), so the ignore mask is a no-op and the loss reduces exactly
to plain full-supervision training — i.e. the ignore protocol is inert, which is
precisely "not using it". Both `Dataset803` and `Dataset804` produce the **same
9 output channels** (bg + 8 fg; the ignore label is masked, never a channel), so
`_detect_num_output_classes` and the CE weight tensor stay consistent. The
trainer is therefore left untouched (changing it would risk the main 803 run).

## The other arm (all-cases model on the fused test split)

For the comparison you also need the existing **all-cases `Dataset803`** model
scored on the **same fused test cases**. `Dataset804/imagesTs` is exactly those
fused test cases, and `Dataset804/labelsTs` is identical GT (803 also merges L6;
fused cases have no ignore voxels), so point the 803 model at the 804 test dir:

```bash
# predict the 803 fold-0 model on the fused test images, eval vs fused GT
singularity exec --nv \
  --bind "$PWD:/workspace" --bind "$PWD/nnunet:/nnunet_nfs" \
  --bind "$PWD/tools/nnunet_wandb_variant.py:/opt/conda/lib/python3.11/site-packages/nnunetv2/training/nnUNetTrainer/variants/nnunet_wandb_variant.py" \
  containers/spinesurg-ct.sif \
  nnUNetv2_predict \
    -i /nnunet_nfs/raw/Dataset804_SpineSurgCTFusedOnly/imagesTs \
    -o /nnunet_nfs/predictions/ALLCASES803_on_fusedtest_f0 \
    -d 803 -c 3d_fullres -f 0 \
    -tr nnUNetTrainerWandB_500ep_LSTVOversample \
    -p  nnUNetResEncUNetPlans_100G
# then eval_paper_metrics.py --predictions_dir <that> \
#   --labels_dir /nnunet_nfs/raw/Dataset804_SpineSurgCTFusedOnly/labelsTs
```

Report fused-only (step 3) vs all-cases-on-fused-test side by side.
