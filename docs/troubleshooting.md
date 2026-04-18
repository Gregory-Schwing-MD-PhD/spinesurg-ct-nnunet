# Troubleshooting

## Preprocessing

### "ENOSPC: No space left on device" on scratch

The background rsync+cleanup daemon usually prevents this, but if your
scratch is very small:

```bash
DURABLE_RSYNC_EVERY_SEC=300 sbatch slurm/spine_prep.sh   # cleanup every 5 min
```

Or run with fewer workers so the per-cycle buildup is smaller:

```bash
N_PREPROCESS_THREADS=24 sbatch slurm/spine_prep.sh
```

### "NFS cache HIT" but I want to re-preprocess from scratch

The cache check keys on the completion marker:

```bash
rm nnunet/preprocessed/Dataset802_SpineSurgCTFull/.preprocessed_complete_*
sbatch slurm/spine_prep.sh
```

### Preprocess runs fast (minutes) but then training fails to find data

Check that `splits_final.json` made it into the preprocessed directory:

```bash
ls nnunet/preprocessed/Dataset802_SpineSurgCTFull/splits_final.json
```

If missing, the Stage 7 copy step failed. Rerun `spine_prep.sh` — the NFS
cache check will short-circuit the heavy parts; only the splits copy + marker
touch remain.

### Daemon log shows `moved=0` for a long time

That's normal early on — it takes >2 min for the first batch of finished
`.npz` files to age enough to be safely moved (the daemon uses a 2-min
safety buffer on `-mmin`). Check again after 15-20 min of preprocessing.

## Training

### W&B logging fails

The trainer variants have an offline fallback:

```bash
export WANDB_MODE=offline
```

Syncs happen when training ends. If you're hitting API rate limits, stagger
fold start times:

```bash
sbatch --array=0-4%1 slurm/spine_train_array.sh   # 1 at a time
```

### CUDA OOM on a fold

Your preprocessing used a patch size larger than what fits in your actual
GPU. Options:

1. Rerun `plan_sizes.sh` + `visualize_patch_sizes.py`, pick a smaller
   target, re-preprocess:

   ```bash
   rm nnunet/preprocessed/Dataset802_*/.preprocessed_complete_*
   GPU_MEMORY_TARGET_GB=80 sbatch slurm/spine_prep.sh
   ```

2. Reduce batch size in the trainer (edit
   `tools/nnunet_wandb_variant.py`); avoids re-preprocessing.

## Container

### `singularity build` fails on a shared cluster

Most clusters don't allow local builds. Use `--remote` (Sylabs cloud) or
build on a workstation and copy the `.sif`:

```bash
REMOTE=1 bash containers/build_container.sh
# or on your laptop:
bash containers/build_container.sh
scp containers/spinesurg-ct.sif warrior:spinesurg-ct-nnunet/containers/
```

### "FATAL: binary in container (/usr/bin/python) is not accessible"

The container wasn't built with `--fakeroot` on a permission-permissive
system. Rebuild:

```bash
USE_FAKEROOT=1 bash containers/build_container.sh
```

## Data

### `download_dataset.py` asks for credentials but I'm on HPC

Pre-authenticate on a login node:

```bash
huggingface-cli login        # enter token; stores in ~/.cache/huggingface
```

Then the download script uses the cached token on compute nodes.

### Symlinked dataset works locally but fails in SLURM jobs

Compute nodes may not have the same filesystem visibility. Either:
- Put the dataset on a filesystem that's mounted on ALL nodes, or
- Resolve the symlink during `convert` (the pipeline does this automatically
  when staging raw to scratch — `rsync -L`).

## Splits

### Stratification fails with "class has too few samples"

Some strata are too small (e.g. `pelvic_only` with <5 LSTV cases).
`generate_5fold_splits.py` will fall back to merging small strata. Check
the log output for warnings. If concerned, inspect the resulting
`data/splits_5fold.json` schema v3 `token_info` counts.
