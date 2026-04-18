#!/bin/bash
#SBATCH --job-name=spine_kfold
#SBATCH -q gpu
#SBATCH --array=0-4%2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=250G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/spine_kfold_f%a_%j.out
#SBATCH --error=logs/spine_kfold_f%a_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=go2432@wayne.edu
# =============================================================================
# SpineSurg-CT  --  Stage B (5-fold CV training array)
# =============================================================================
# Prereqs:
#   - slurm/generate_splits.sh has produced data/splits_5fold.json
#   - slurm/spine_prep.sh has converted + preprocessed, and splits_final.json
#     in preprocessed/ already reflects the 5-fold CV (built from splits_5fold.json
#     by convert_hf_to_nnunet.py).
#
# No more splits regeneration happens here. Each array task just trains
# one fold (or skips if already done, or resumes if --c applicable).
#
# Array throttle %2 runs max 2 folds at a time. Change to %1 (serial),
# %4 (whole msa node), or remove entirely.
# =============================================================================

set -euo pipefail

FOLD=${SLURM_ARRAY_TASK_ID:-0}

export SPINESURG_RUN_VAL_EXPORT="${SPINESURG_RUN_VAL_EXPORT:-0}"
export WANDB_INIT_MAX_RETRIES="${WANDB_INIT_MAX_RETRIES:-5}"
export WANDB_INIT_TIMEOUT_SEC="${WANDB_INIT_TIMEOUT_SEC:-180}"
export WANDB_ALLOW_OFFLINE="${WANDB_ALLOW_OFFLINE:-1}"
export LSTV_OVERSAMPLE_FRAC="${LSTV_OVERSAMPLE_FRAC:-0.5}"

DATASET_ID=802
DATASET_NAME="SpineSurgCTFull"
CONFIG="3d_fullres"
TRAINER="${TRAINER:-nnUNetTrainerWandB_1000ep_LSTVOversample}"
PLANNER="nnUNetPlannerResEncM"
GPU_MEMORY_TARGET_GB=100
PLANS="nnUNetResEncUNetPlans_${GPU_MEMORY_TARGET_GB}G"
NNUNET_EXPORT_POOL=12
NNUNET_DA_WORKERS=12

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-${HOME}/SpineSurg-CT}"
NNUNET_NFS="${PROJECT_ROOT}/nnunet"
CONTAINER="${PROJECT_ROOT}/containers/spinesurg-ct.sif"
WANDB_TRAINER_HOST="${PROJECT_ROOT}/tools/nnunet_wandb_variant.py"
WANDB_TRAINER_CONTAINER="/opt/conda/lib/python3.11/site-packages/nnunetv2/training/nnUNetTrainer/variants/nnunet_wandb_variant.py"
DS_DIR_NAME="Dataset$(printf '%03d' ${DATASET_ID})_${DATASET_NAME}"

NFS_PREP_DS="${NNUNET_NFS}/preprocessed/${DS_DIR_NAME}"
NFS_RAW_DS="${NNUNET_NFS}/raw/${DS_DIR_NAME}"

# ----- Preflight ------------------------------------------------------------
if [[ ! -f "${NFS_PREP_DS}/dataset_fingerprint.json" ]]; then
    echo "ERROR: preprocessed dataset missing; run sbatch slurm/spine_prep.sh first." >&2
    exit 1
fi
if [[ ! -f "${NFS_PREP_DS}/splits_final.json" ]]; then
    echo "ERROR: splits_final.json missing from ${NFS_PREP_DS}." >&2
    echo "       Re-run slurm/spine_prep.sh (it copies splits into preprocessed)." >&2
    exit 1
fi

# ----- Idempotency: skip if fold already trained ----------------------------
RESULT_DIR="${NNUNET_NFS}/results/${DS_DIR_NAME}/${TRAINER}__${PLANS}__${CONFIG}/fold_${FOLD}"
FINAL_CKPT="${RESULT_DIR}/checkpoint_final.pth"
LATEST_CKPT="${RESULT_DIR}/checkpoint_latest.pth"

if [[ -f "${FINAL_CKPT}" ]]; then
    echo "[fold ${FOLD}] checkpoint_final.pth exists; skipping."
    exit 0
fi

RESUME_FLAG=""
if [[ -f "${LATEST_CKPT}" ]]; then
    RESUME_FLAG="--c"
    echo "[fold ${FOLD}] resuming from checkpoint_latest.pth"
fi

# ----- Local scratch --------------------------------------------------------
if [[ -n "${SLURM_TMPDIR:-}" && -d "${SLURM_TMPDIR}" ]]; then
    LOCAL_SCRATCH="${SLURM_TMPDIR}"
elif [[ -d "/scratch/${USER}" ]]; then
    LOCAL_SCRATCH="/scratch/${USER}/job_${SLURM_JOB_ID}_f${FOLD}"; mkdir -p "${LOCAL_SCRATCH}"
else
    LOCAL_SCRATCH="/tmp/${USER}_${SLURM_JOB_ID}_f${FOLD}"; mkdir -p "${LOCAL_SCRATCH}"
fi
LOCAL_NNUNET="${LOCAL_SCRATCH}/nnunet"
LOCAL_PREP_DS="${LOCAL_NNUNET}/preprocessed/${DS_DIR_NAME}"
LOCAL_RAW_DS="${LOCAL_NNUNET}/raw/${DS_DIR_NAME}"
mkdir -p "${LOCAL_PREP_DS}" "${LOCAL_RAW_DS}"

echo "[fold ${FOLD}] staging preprocessed + raw to ${LOCAL_SCRATCH}"
rsync -aW --no-compress "${NFS_PREP_DS}/" "${LOCAL_PREP_DS}/"
rsync -aW --no-compress "${NFS_RAW_DS}/"  "${LOCAL_RAW_DS}/"

# ----- W&B token ------------------------------------------------------------
WANDB_API_KEY=""
[[ -f "${HOME}/.wandb/token" ]] && WANDB_API_KEY="$(cat ${HOME}/.wandb/token)"

# ----- Environment ----------------------------------------------------------
export SINGULARITYENV_TMPDIR="/workspace/tmp"
export SINGULARITYENV_nnUNet_raw="/nnunet_local/raw"
export SINGULARITYENV_nnUNet_preprocessed="/nnunet_local/preprocessed"
export SINGULARITYENV_nnUNet_results="/nnunet_nfs/results"
export SINGULARITYENV_nnUNet_def_n_proc="${NNUNET_EXPORT_POOL}"
export SINGULARITYENV_nnUNet_n_proc_DA="${NNUNET_DA_WORKERS}"
export SINGULARITYENV_SPINESURG_RUN_VAL_EXPORT="${SPINESURG_RUN_VAL_EXPORT}"
export SINGULARITYENV_WANDB_API_KEY="${WANDB_API_KEY}"
export SINGULARITYENV_WANDB_PROJECT="SpineSurg-CT"
export SINGULARITYENV_WANDB_RUN_NAME="${DS_DIR_NAME}_${TRAINER}_${PLANS}_f${FOLD}_arr${SLURM_ARRAY_JOB_ID}"
export SINGULARITYENV_WANDB_RUN_ID="${DS_DIR_NAME}_${TRAINER}_${PLANS}_f${FOLD}"
export SINGULARITYENV_WANDB_INIT_MAX_RETRIES="${WANDB_INIT_MAX_RETRIES}"
export SINGULARITYENV_WANDB_INIT_TIMEOUT_SEC="${WANDB_INIT_TIMEOUT_SEC}"
export SINGULARITYENV_WANDB_ALLOW_OFFLINE="${WANDB_ALLOW_OFFLINE}"
export SINGULARITYENV_LSTV_OVERSAMPLE_FRAC="${LSTV_OVERSAMPLE_FRAC}"
export OMP_NUM_THREADS=4

export SINGULARITY_TMPDIR="/tmp/${USER}_job_${SLURM_JOB_ID}_f${FOLD}_singularity"
export XDG_RUNTIME_DIR="${SINGULARITY_TMPDIR}/runtime"
mkdir -p "${SINGULARITY_TMPDIR}" "${XDG_RUNTIME_DIR}"
trap "rm -rf ${SINGULARITY_TMPDIR}" EXIT

TRAIN_BINDS=(
    --nv
    --bind "${PROJECT_ROOT}:/workspace"
    --bind "${LOCAL_NNUNET}:/nnunet_local"
    --bind "${NNUNET_NFS}:/nnunet_nfs"
    --bind "${HOME}/.wandb:${HOME}/.wandb"
    --bind "${WANDB_TRAINER_HOST}:${WANDB_TRAINER_CONTAINER}"
    --pwd  /workspace
)

echo "================================================================"
echo " 5-fold CV  |  fold ${FOLD}  |  array ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo " Trainer   : ${TRAINER}"
echo " LSTV frac : ${LSTV_OVERSAMPLE_FRAC}"
echo " Status    : $([[ -n ${RESUME_FLAG} ]] && echo 'resuming' || echo 'fresh start')"
echo " Started   : $(date)"
echo "================================================================"

singularity exec "${TRAIN_BINDS[@]}" "${CONTAINER}" \
    nnUNetv2_train ${DATASET_ID} ${CONFIG} ${FOLD} \
        -tr ${TRAINER} \
        -p  ${PLANS} \
        --npz \
        ${RESUME_FLAG}

if [[ -f "${FINAL_CKPT}" ]]; then
    echo "[fold ${FOLD}] DONE  $(date)"
else
    echo "WARN: training exited cleanly but checkpoint_final.pth missing; resubmit to resume." >&2
    exit 2
fi
