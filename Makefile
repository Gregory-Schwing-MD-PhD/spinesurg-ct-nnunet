# spinesurg-ct-nnunet  --  common commands
#
# Run `make help` for a list of targets.

SHELL := /bin/bash

# Defaults (override via env: `GPU_MEMORY_TARGET_GB=80 make preprocess`)
DATASET_ID            ?= 802
DATASET_NAME          ?= SpineSurgCTFull
GPU_MEMORY_TARGET_GB  ?= 100
PLANNER               ?= nnUNetPlannerResEncM
FOLD                  ?= 0
CT                    ?=
HF_REPO               ?= gschwing/CTSpinoPelvic1K

DS_DIR_NAME = Dataset$(shell printf '%03d' $(DATASET_ID))_$(DATASET_NAME)
PLANS_GLOB  = nnunet/preprocessed/$(DS_DIR_NAME)/nnUNetResEncUNetPlans_*G.json

.PHONY: help download-data build-container splits plans viz \
        preprocess train-fold train-array eval-fold eval-ensemble \
        ablate compare-ablations clean-logs clean-figures status \
        check-syntax

help:  ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?##/ \
	  {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# =============================================================================
# Dataset
# =============================================================================
download-data:  ## Download CTSpinoPelvic1K from HuggingFace -> data/hf_export/
	python scripts/download_dataset.py --repo_id $(HF_REPO) --dest data/hf_export

# =============================================================================
# Container
# =============================================================================
build-container:  ## Build the Singularity container -> containers/spinesurg-ct.sif
	bash containers/build_container.sh

# =============================================================================
# Splits
# =============================================================================
splits:  ## Generate 5-fold splits from data/hf_export/ (token-level, stratified)
	sbatch slurm/generate_splits.sh

# =============================================================================
# Planning and patch-size visualization
# =============================================================================
plans:  ## Plan at multiple GPU memory targets (fast, no preprocessing)
	sbatch slurm/plan_sizes.sh

viz:  ## Visualize patch sizes on a CT (usage: make viz CT=path/to/ct.nii.gz)
	@if [ -z "$(CT)" ]; then \
	  echo "usage: make viz CT=<path/to/ct.nii.gz>"; \
	  echo "       (use a file from data/hf_export/ct/)"; \
	  exit 1; \
	fi
	@mkdir -p figures
	python tools/visualize_patch_sizes.py \
	    --ct    $(CT) \
	    $(foreach p,$(wildcard $(PLANS_GLOB)),--plans $(p)) \
	    --out   figures/patch_viz.png
	@echo "  -> figures/patch_viz.png"

# =============================================================================
# Preprocess
# =============================================================================
preprocess:  ## Convert + preprocess to local scratch, rsync to NFS
	GPU_MEMORY_TARGET_GB=$(GPU_MEMORY_TARGET_GB) PLANNER=$(PLANNER) \
	  DATASET_ID=$(DATASET_ID) DATASET_NAME=$(DATASET_NAME) \
	  sbatch slurm/spine_prep.sh

# =============================================================================
# Train
# =============================================================================
train-fold:  ## Train a single fold (usage: make train-fold FOLD=0)
	FOLD=$(FOLD) sbatch slurm/spine_train_fold.sh

train-array:  ## Submit 5-fold CV array (2 folds at a time)
	sbatch slurm/spine_train_array.sh

# =============================================================================
# Eval
# =============================================================================
eval-fold:  ## Eval a single fold (usage: make eval-fold FOLD=0)
	FOLD=$(FOLD) sbatch slurm/spine_eval_single.sh

eval-ensemble:  ## Eval the 5-fold ensemble on the held-out test set
	sbatch slurm/spine_eval_ensemble.sh

# =============================================================================
# Ablations
# =============================================================================
ablate:  ## Submit 6-ablation array (fused_only, no_partial, etc.)
	sbatch slurm/spine_ablation.sh

compare-ablations:  ## Aggregate ablation metrics into md/tex/csv tables
	python tools/compare_ablations.py

# =============================================================================
# Housekeeping
# =============================================================================
status:  ## Show SLURM queue for this user
	squeue -u $$USER -o "%.18i %.9P %.20j %.2t %.10M %.6D %R"

clean-logs:  ## Delete log files older than 30 days
	find logs -type f -mtime +30 -print -delete || true

clean-figures:  ## Delete all generated figures (keeps .gitkeep)
	find figures -type f ! -name '.gitkeep' -print -delete || true

check-syntax:  ## Syntax-check all .py and .sh files
	@echo "Python:"
	@for f in scripts/*.py tools/*.py; do \
	  python -c "import ast; ast.parse(open('$$f').read())" && echo "  OK $$f"; \
	done
	@echo "Shell:"
	@for f in slurm/*.sh containers/*.sh; do \
	  bash -n "$$f" && echo "  OK $$f"; \
	done
