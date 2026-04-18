# Ablations

The `slurm/spine_ablation.sh` array submits six ablations that collectively
defend the main modeling choices:

| # | name                   | dataset | plans                          | trainer                               | what it tests |
|---|------------------------|---------|--------------------------------|---------------------------------------|---------------|
| 0 | `fused_only`           | 803     | nnUNetResEncUNetPlans_100G     | WandB_1000ep_LSTVOversample           | Effect of training on only fused (spine+pelvis) cases |
| 1 | `no_partial`           | 804     | nnUNetResEncUNetPlans_100G     | WandB_1000ep_LSTVOversample           | Effect of dropping partial-match cases |
| 2 | `no_lstv_oversample`   | 802     | nnUNetResEncUNetPlans_100G     | WandB_1000ep                          | Effect of LSTV (L6) case oversampling |
| 3 | `short_100ep`          | 802     | nnUNetResEncUNetPlans_100G     | WandB_100ep_LSTVOversample            | Sensitivity to training length |
| 4 | `resenc_l`             | 802     | nnUNetResEncUNetLPlans_100G    | WandB_1000ep_LSTVOversample           | Bigger encoder (ResEncM -> ResEncL) |
| 5 | `testfusedonly`        | 805     | nnUNetResEncUNetPlans_100G     | WandB_1000ep_LSTVOversample           | Evaluation on fused-only test set |

## Cost model

Preprocessed state is cached per `(dataset_id, plans_name)`. The main
preprocess (`Dataset802 × ResEncUNetPlans_100G`) is reused for ablations
2, 3, 5 (same dataset + plans). Ablations 0, 1, 4 each trigger their own
preprocess run:

- `fused_only` (803): ~1/4 the cases, so preprocess is faster (~15-30 min)
- `no_partial` (804): ~80% of cases, preprocess ~1-2 h
- `resenc_l` (Dataset802 + different plans): full preprocess, ~1.5-2.5 h

## Running

```bash
make ablate                       # submit all 6 as an array
# or a subset:
sbatch --array=0,2,4 slurm/spine_ablation.sh
```

Each ablation writes metrics to its own results directory. Aggregate:

```bash
make compare-ablations            # produces md/tex/csv tables
```

## Expected signal

- If `fused_only` (0) matches or beats the main model on the fused-only test
  subset: more data doesn't help if most of it is partial. Unlikely.
- If `no_partial` (1) matches the main: partial-match cases aren't adding
  signal. Decide whether to keep them for generalization vs. drop for cleanliness.
- If `no_lstv_oversample` (2) has clearly worse L6 Dice: oversampling
  justified. Report the delta prominently.
- If `short_100ep` (3) matches the main: 1000 epochs is overkill; cut budget
  next time. Otherwise: report training curves.
- If `resenc_l` (4) beats the main by a clinically meaningful margin:
  worth the compute; otherwise, M is the right size.
- `testfusedonly` (5) is the headline number for the fused subset.

## LaTeX table for paper

```bash
python tools/compare_ablations.py --format latex > tables/ablations.tex
```

Drops into your MLHC paper template at `\input{tables/ablations.tex}`.
