# SpineSurg-CT nnU-Net pipeline (v5)

**Splits are now generated upstream at the patient-token level.** The
nnU-Net convert step just loads them.

## Pipeline overview

```
data/ctpelvic1k/masks/placed/placed_manifest.json    (upstream, from place_fused_masks.py)
                        │
                        ▼
[slurm/generate_splits.sh]
scripts/generate_5fold_splits.py
                        │
                        ▼
data/splits_5fold.json    ← single source of truth for test holdout + 5-fold CV
                        │
                        ▼              HF export manifests (from export_hf.py)
                        │                      │
                        ▼                      ▼
[slurm/spine_prep.sh]          tools/convert_hf_to_nnunet.py --splits_file ...
                        │
                        ▼
nnunet/raw/Dataset802_SpineSurgCTFull/
    imagesTr/  labelsTr/   ← train pool (non-test tokens)
    imagesTs/  labelsTs/   ← test tokens (held out)
    splits_final.json      ← 5 folds, partitioning the train pool
                        │
                        ▼
[slurm/spine_train_fold.sh  or  slurm/spine_train_array.sh]
                        │
                        ▼
[slurm/spine_eval_single.sh  or  slurm/spine_eval_ensemble.sh]
```

## File list

```
scripts/
  generate_5fold_splits.py        token-level 5-fold CV generator

tools/
  nnunet_wandb_variant.py         W&B + LSTV oversampling + warning filters
  convert_hf_to_nnunet.py         loads splits_5fold.json; supports ablation filters
  compare_ablations.py            aggregates ablation metrics into one table

slurm/
  generate_splits.sh              reads newest placed_manifest.json
  spine_prep.sh                   convert + preprocess (requires splits_5fold.json)
  spine_train_fold.sh             single fold, 48h, resumable
  spine_train_array.sh            5-fold CV array
  spine_eval_single.sh            single fold eval
  spine_eval_ensemble.sh          5-fold ensemble eval
  spine_ablation.sh               6-ablation array (see Ablation section)
```

## Submission order

```bash
# (1) Upstream: place + build + export. UNCHANGED from your existing flow.
sbatch slurm/prepare_training.sh
sbatch slurm/export_hf.sh

# (2) Generate 5-fold CV splits from the newest placed_manifest.
sbatch slurm/generate_splits.sh

# (3) Convert + preprocess (reads splits_5fold.json).
sbatch slurm/spine_prep.sh

# (4) Train. Pick one of:
FOLD=0 sbatch slurm/spine_train_fold.sh          # just fold 0, single run
sbatch slurm/spine_train_array.sh                # all 5 folds, array %2

# (5) Evaluate.
FOLD=0 sbatch slurm/spine_eval_single.sh         # if single fold
sbatch slurm/spine_eval_ensemble.sh              # 5-fold ensemble
```

## Stratification scheme

`generate_5fold_splits.py` stratifies on:
- **match_type** (from placed_manifest.json): `fused`, `separate`,
  `spine_only`, `pelvic_only` — **each kept as its own stratum**. `fused`
  (same-series spine + pelvic) is the gold standard; keeping it distinct
  from `separate` (cross-series placement) guarantees the test set gets a
  known fraction of true gold-standard cases for reporting.
- **LSTV presence** (from placed_manifest.json if present, else from
  colonog_training_manifest.json): binary `lstv` / `no_lstv`

Crossed → up to 8 strata. Rare cells (< n_folds members) auto-coalesce
into their non-LSTV parent to prevent StratifiedKFold failures, while
the match_type distinction is preserved.

If LSTV info is nowhere to be found, the generator falls back to
match_type-only stratification and logs a warning.

## What the splits file looks like

```json
{
  "schema_version": 2,
  "created_at": "2026-04-18T...Z",
  "source_manifest": "/data/.../placed_manifest.json",
  "source_manifest_mtime": "2026-04-17T...Z",
  "training_manifest": "/data/.../colonog_training_manifest.json",
  "test_fraction": 0.15,
  "n_folds": 5,
  "kfold_seed": 42,
  "strata_scheme": "match_type_x_lstv",
  "n_tokens_total": 987,
  "n_tokens_test": 148,
  "n_tokens_trainval": 839,
  "test_tokens": ["123", "456", ...],
  "folds": [
    {"train_tokens": [...], "val_tokens": [...]},
    ...
  ],
  "strata_counts_total":    {"fused|no_lstv": 520, "fused|lstv": 61, ...},
  "strata_counts_test":     {...},
  "strata_counts_per_fold_val": [{...}, ...]
}
```

## Idempotency

Every stage is safe to rerun.

| Stage | No-op condition | Partial rerun |
|---|---|---|
| `generate_splits.sh` | `splits_5fold.json` exists and token set matches current `placed_manifest.json` | `OVERWRITE=1` to force |
| `spine_prep.sh` convert | `dataset.json` exists AND splits match requested mode | `--regen_splits_only` refreshes just splits |
| `spine_prep.sh` preprocess | `.preprocessed_complete_<plans>` marker + npz files | — |
| `spine_train_fold.sh` / array task | `checkpoint_final.pth` exists | resume via `checkpoint_latest.pth` + `--c` |
| `spine_ablation.sh` task | All stages (convert / preprocess / train / predict) independently idempotent | same as above, per stage |
| `spine_eval_*.sh` | all test predictions exist | `--continue_prediction` skips done |

## Ablation studies

`slurm/spine_ablation.sh` is a SLURM array driver that runs 6 one-fold
ablations end-to-end and leaves comparable outputs in
`nnunet/predictions/ABL_<tag>_.../paper_metrics/`. All ablations share
the same master test set, so numbers are directly comparable.

```bash
# Run all 6 ablations (%2 = 2 folds concurrent, ~5 days wall clock)
sbatch slurm/spine_ablation.sh

# Run just one ablation (e.g. fused_only = index 0)
sbatch --array=0 slurm/spine_ablation.sh

# After they finish, aggregate into one comparison table
singularity exec --bind $PWD:/workspace --pwd /workspace \
    containers/spinesurg-ct.sif \
    python tools/compare_ablations.py \
        --results_root nnunet/predictions
# -> nnunet/predictions/ablation_comparison.{md,tex,csv}
```

### Ablation matrix (built-in)

| # | Tag | Changes vs main | Shows |
|---|---|---|---|
| 0 | `fused_only` | Train on fused only | Value of separate + partial-annotation data |
| 1 | `no_partial` | Train on fused + separate only | Value of spine_only + pelvic_only partials |
| 2 | `no_lstv_oversample` | Drop LSTV oversampling | Value of LSTV case oversampling |
| 3 | `short_100ep` | 100 epochs instead of 1000 | Training length sanity |
| 4 | `resenc_l` | ResEncL (bigger model) | Model capacity scaling |
| 5 | `testfusedonly` | Train fused+separate, test fused only | Cross-series contamination effect |

Data-filter ablations (0, 1, 5) build their own dataset IDs (803, 804, 805).
Trainer-swap ablations (2, 3, 4) reuse Dataset802 with different
trainer/planner classes. Each ablation gets its own `results/<dataset>/<trainer>__<plans>/fold_0/`
directory and its own prediction dir, so nothing collides.

### How the filtering works

`generate_5fold_splits.py` now writes a `token_info` dict (token →
`match_type`, `has_lstv`) in `splits_5fold.json` (schema v3). At convert
time, `--include_train_match_types=fused` filters the training pool to
fused cases only, while leaving the test set untouched. `--test_match_types`
similarly restricts the test set (used by ablation 5).

The test set's `match_type` composition is fixed by `generate_splits.sh`,
so ablations evaluating on "all test cases" see identical test-set
compositions across runs. Ablation 5 evaluates on a strict subset of that.

### Refreshing splits after new cases land

If `placed_manifest.json` gains new cases:

```bash
OVERWRITE=1 sbatch slurm/generate_splits.sh     # regenerate splits_5fold.json
sbatch slurm/spine_prep.sh                      # auto-detects stale splits,
                                                # rewrites splits_final.json
                                                # without touching imagesTr
```

The convert script notices that `splits_final.json` in the dataset dir
doesn't match `splits_5fold.json` and rewrites just the splits. Trained
models are **not** invalidated, but note that if new train cases appeared
you'll want to retrain to use them.

## Switching between K-fold and single-fold modes

Default = K-fold (from `splits_5fold.json`).

Legacy single-fold mode (fold 0 = HF train/val split, folds 1–4 identical):

```bash
SINGLE_FOLD_SPLITS=1 sbatch slurm/spine_prep.sh
```

This skips `--splits_file`, lets the convert script build the legacy
5-identical-folds `splits_final.json` directly from the HF manifests.
Useful if you specifically want to train one model whose val set exactly
equals your HF val set.

## Env var reference

| Variable | Default | Used by | Purpose |
|---|---|---|---|
| `OVERWRITE` | 0 | generate_splits.sh | Regenerate `splits_5fold.json` |
| `N_FOLDS` | 5 | generate_splits.sh | CV fold count |
| `KFOLD_SEED` | 42 | generate_splits.sh | sklearn seed |
| `TEST_FRACTION` | 0.15 | generate_splits.sh | Test holdout fraction |
| `SINGLE_FOLD_SPLITS` | 0 | spine_prep.sh | Use legacy 5-identical-folds |
| `FOLD` | 0 | spine_train_fold.sh, spine_eval_single.sh | Which fold |
| `TRAINER` | `nnUNetTrainerWandB_1000ep_LSTVOversample` | train scripts | Swap trainer class |
| `LSTV_OVERSAMPLE_FRAC` | 0.5 | train scripts | LSTV case oversampling |
| `WANDB_INIT_MAX_RETRIES` | 5 | train scripts | Online wandb retry count |
| `WANDB_INIT_TIMEOUT_SEC` | 180 | train scripts | Per-retry timeout |
| `WANDB_ALLOW_OFFLINE` | 1 | train scripts | Offline fallback |

## Why token-level splits

If the same patient `42` produces multiple case_ids downstream (e.g.
`42__fused` from the main HF split, plus `42__spine_only` from a
partial-annotation pass), those should never appear in different folds
— that's data leakage through patient identity. `splits_5fold.json`
stores **tokens**, and `convert_hf_to_nnunet.py` routes every derived
case_id to the same fold as its parent token. Pseudo / augmented cases
with tokens not in the splits file go to training only (never val), which
is the correct behavior for synthetic augmentation.
