# spinesurg-ct-nnunet

nnU-Net v2 training pipeline for multi-class spine + pelvis CT segmentation.
Targets 10 classes: L1-L6 (including lumbosacral transitional vertebrae, L6 = 6),
sacrum, left hip, right hip, plus an ignore label.

**Dataset:** [CTSpinoPelvic1K](https://huggingface.co/datasets/gschwing/CTSpinoPelvic1K) — kept as a separate repo to decouple dataset distribution from pipeline code.

## Key features

- **Local-scratch-optimized preprocessing** — raw NIfTIs staged to local
  scratch before 48-worker preprocess; background daemon rsyncs + cleans up
  scratch every 10 min; final rsync persists to NFS. Typical wall time
  ~1.5-2.5h vs 6h direct-to-NFS for a ~1000-case dataset.
- **Patch-size visualization** — pick a GPU memory target by overlaying
  candidate patch footprints on a real CT.
- **Stratified 5-fold CV** with token-level splits (no patient leakage) on
  match_type × LSTV strata. Schema v3 manifest with per-token metadata.
- **6-ablation framework** with automatic metrics aggregation
  (fused-only, no partial, no LSTV oversampling, short 100ep, ResEncL,
  test-fused-only).
- **W&B-integrated nnU-Net trainer** with LSTV oversampling, NaN-safe
  Dice aggregation, offline-fallback logging.

## Quickstart

```bash
# 0. Clone this repo
git clone https://github.com/<you>/spinesurg-ct-nnunet.git
cd spinesurg-ct-nnunet

# 1. Get the dataset (see data/README.md for options)
make download-data           # from HuggingFace
#   OR: ln -s /path/to/CTSpinoPelvic1K data/hf_export

# 2. Build the Singularity container
make build-container

# 3. Generate 5-fold splits (token-level, stratified)
make splits

# 4. Pick a GPU memory target
make plans                                # plan_experiment @ 60/80/100/120/140 GB
make viz CT=data/hf_export/ct/CASE_001.nii.gz
# -> inspect figures/patch_viz.png, pick GPU_MEMORY_TARGET_GB

# 5. Preprocess (local-scratch, resumable, reused across folds + ablations)
make preprocess                           # default GPU_MEMORY_TARGET_GB=100

# 6. Train
make train-array                          # 5-fold CV (array job)
make train-fold FOLD=0                    # single fold

# 7. Eval
make eval-ensemble                        # 5-fold ensemble on test set

# 8. Ablations
make ablate                               # array of 6 ablations
python tools/compare_ablations.py         # aggregate metrics
```

## Repo layout

```
spinesurg-ct-nnunet/
├── scripts/               One-off scripts (dataset download, splits generation)
├── tools/                 Reusable Python utilities (bind-mounted into container)
├── slurm/                 SLURM batch scripts (the pipeline stages)
├── containers/            Singularity definition + build helpers
├── data/                  Dataset lives here (gitignored; see data/README.md)
├── docs/                  Detailed pipeline documentation
├── logs/                  SLURM logs (gitignored)
├── nnunet/                nnU-Net raw/preprocessed/results (gitignored)
├── figures/               Generated figures (patch viz, etc.)
└── Makefile               Shortcut commands
```

See [**docs/pipeline.md**](docs/pipeline.md) for the full pipeline walkthrough,
ablation details, troubleshooting, and design notes on preprocessing
performance.

## Requirements

- **Cluster:** SLURM with at least one H200, A100-80GB, or equivalent GPU node
- **Container:** Singularity / Apptainer (built from `containers/spinesurg-ct.def`)
- **Host tooling:** rsync, Python 3.10+ (for tools and `download_dataset.py`)

Detailed Python deps for host tooling: [`requirements.txt`](requirements.txt).

## Citation

See [CITATION.cff](CITATION.cff).

## License

[Apache-2.0](LICENSE)
