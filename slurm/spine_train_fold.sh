#!/bin/bash
#SBATCH --job-name=spine_train
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=250G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/spine_train_f%x_%j.out
#SBATCH --error=logs/spine_train_f%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=go2432@wayne.edu
# =============================================================================
# SpineSurg-CT  --  Stage B (single fold)
# =============================================================================
# Trains ONE fold. Fully idempotent:
#   - if checkpoint_final.pth exists -> skip (does nothing, exits 0)
#   - elif checkpoint_latest.pth exists -> resume with --c
#   - else -> fresh training
#
# Usage:
#     FOLD=0 sbatch slurm/spine_train_fold.sh
#     FOLD=1 sbatch slurm/spine_train_fold.sh   (after 5-fold splits are in place)
#     ...
#
# If a job times out mid-training, just resubmit the same command. It will
# pick up from the last checkpoint_latest.pth.
#
# For 5-fold CV, either:
#   (a) Submit 5 independent sbatch jobs with FOLD=0..4, or
#   (b) Use spine_train_array.sh (SLURM array, auto-parallel across nodes)
# =============================================================================

set -euo pipefail

FOLD="${FOLD:-0}"
if ! [[ "${FOLD}" =~ ^[0-4]$ ]]; then
    echo "ERROR: FOLD must be 0..4, got '${FOLD}'" >&2
    exit 1
fi

# ----- TOGGLES ---------------------------------------------------------------
export SPINESURG_RUN_VAL_EXPORT="${SPINESURG_RUN_VAL_EXPORT:-0}"
export WANDB_INIT_MAX_RETRIES="${WANDB_INIT_MAX_RETRIES:-5}"
export WANDB_INIT_TIMEOUT_SEC="${WANDB_INIT_TIMEOUT_SEC:-180}"
export WANDB_ALLOW_OFFLINE="${WANDB_ALLOW_OFFLINE:-1}"
export LSTV_OVERSAMPLE_FRAC="${LSTV_OVERSAMPLE_FRAC:-0.5}"
# -----------------------------------------------------------------------------

# ----- Config (match Stage A) -----------------------------------------------
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

# ----- Preflight -------------------------------------------------------------
NFS_PREP_DS="${NNUNET_NFS}/preprocessed/${DS_DIR_NAME}"
NFS_RAW_DS="${NNUNET_NFS}/raw/${DS_DIR_NAME}"
if [[ ! -f "${NFS_PREP_DS}/dataset_fingerprint.json" ]]; then
    echo "ERROR: preprocessed dataset missing; run sbatch slurm/spine_prep.sh first." >&2
    exit 1
fi
if [[ ! -f "${NFS_PREP_DS}/splits_final.json" ]]; then
    echo "ERROR: splits_final.json missing from ${NFS_PREP_DS}. Re-run prep." >&2
    exit 1
fi

# ----- Result paths + idempotency check -------------------------------------
RESULT_DIR="${NNUNET_NFS}/results/${DS_DIR_NAME}/${TRAINER}__${PLANS}__${CONFIG}/fold_${FOLD}"
FINAL_CKPT="${RESULT_DIR}/checkpoint_final.pth"
LATEST_CKPT="${RESULT_DIR}/checkpoint_latest.pth"

if [[ -f "${FINAL_CKPT}" ]]; then
    echo "================================================================"
    echo " Fold ${FOLD} already complete (checkpoint_final.pth exists)."
    echo " No-op. Delete ${FINAL_CKPT} if you want to retrain."
    echo "================================================================"
    exit 0
fi

RESUME_FLAG=""
RESUME_NOTE="fresh start"
if [[ -f "${LATEST_CKPT}" ]]; then
    RESUME_FLAG="--c"
    RESUME_NOTE="resuming from checkpoint_latest.pth"
fi

# ----- W&B token -------------------------------------------------------------
WANDB_API_KEY=""
[[ -f "${HOME}/.wandb/token" ]] && WANDB_API_KEY="$(cat ${HOME}/.wandb/token)"

# ----- Local scratch --------------------------------------------------------
if [[ -n "${SLURM_TMPDIR:-}" && -d "${SLURM_TMPDIR}" ]]; then
    LOCAL_SCRATCH="${SLURM_TMPDIR}"
elif [[ -d "/scratch/${USER}" ]]; then
    LOCAL_SCRATCH="/scratch/${USER}/job_${SLURM_JOB_ID}_f${FOLD}"; mkdir -p "${LOCAL_SCRATCH}"
else
    LOCAL_SCRATCH="/tmp/${USER}_${SLURM_JOB_ID}_f${FOLD}"; mkdir -p "${LOCAL_SCRATCH}"
fi
LOCAL_NNUNET="${LOCAL_SCRATCH}/nnunet"
LOCAL_PREP="${LOCAL_NNUNET}/preprocessed"
LOCAL_RAW="${LOCAL_NNUNET}/raw"
mkdir -p "${LOCAL_PREP}" "${LOCAL_RAW}"
LOCAL_PREP_DS="${LOCAL_PREP}/${DS_DIR_NAME}"
LOCAL_RAW_DS="${LOCAL_RAW}/${DS_DIR_NAME}"

echo "[fold ${FOLD}] staging preprocessed -> ${LOCAL_PREP_DS}"
mkdir -p "${LOCAL_PREP_DS}"
rsync -aW --no-compress "${NFS_PREP_DS}/" "${LOCAL_PREP_DS}/"

echo "[fold ${FOLD}] staging raw -> ${LOCAL_RAW_DS} (needed for LSTV scanner)"
mkdir -p "${LOCAL_RAW_DS}"
rsync -aW --no-compress "${NFS_RAW_DS}/" "${LOCAL_RAW_DS}/"

# ----- Singularity env ------------------------------------------------------
export SINGULARITYENV_TMPDIR="/workspace/tmp"
export SINGULARITYENV_nnUNet_raw="/nnunet_local/raw"
export SINGULARITYENV_nnUNet_preprocessed="/nnunet_local/preprocessed"
export SINGULARITYENV_nnUNet_results="/nnunet_nfs/results"
export SINGULARITYENV_nnUNet_def_n_proc="${NNUNET_EXPORT_POOL}"
export SINGULARITYENV_nnUNet_n_proc_DA="${NNUNET_DA_WORKERS}"
export SINGULARITYENV_SPINESURG_RUN_VAL_EXPORT="${SPINESURG_RUN_VAL_EXPORT}"
export SINGULARITYENV_WANDB_API_KEY="${WANDB_API_KEY}"
export SINGULARITYENV_WANDB_PROJECT="SpineSurg-CT"
export SINGULARITYENV_WANDB_RUN_NAME="${DS_DIR_NAME}_${TRAINER}_${PLANS}_f${FOLD}_j${SLURM_JOB_ID}"
export SINGULARITYENV_WANDB_INIT_MAX_RETRIES="${WANDB_INIT_MAX_RETRIES}"
export SINGULARITYENV_WANDB_INIT_TIMEOUT_SEC="${WANDB_INIT_TIMEOUT_SEC}"
export SINGULARITYENV_WANDB_ALLOW_OFFLINE="${WANDB_ALLOW_OFFLINE}"
export SINGULARITYENV_LSTV_OVERSAMPLE_FRAC="${LSTV_OVERSAMPLE_FRAC}"
export SINGULARITYENV_WANDB_RUN_ID="${DS_DIR_NAME}_${TRAINER}_${PLANS}_f${FOLD}"
export OMP_NUM_THREADS=4

export SINGULARITY_TMPDIR="/tmp/${USER}_job_${SLURM_JOB_ID}_f${FOLD}_singularity"
export XDG_RUNTIME_DIR="${SINGULARITY_TMPDIR}/runtime"
mkdir -p "${SINGULARITY_TMPDIR}" "${XDG_RUNTIME_DIR}"
trap "rm -rf ${SINGULARITY_TMPDIR}" EXIT

SING_BINDS=(
    --nv
    --bind "${PROJECT_ROOT}:/workspace"
    --bind "${LOCAL_NNUNET}:/nnunet_local"
    --bind "${NNUNET_NFS}:/nnunet_nfs"
    --bind "${HOME}/.wandb:${HOME}/.wandb"
    --bind "${WANDB_TRAINER_HOST}:${WANDB_TRAINER_CONTAINER}"
    --pwd  /workspace
)

echo "================================================================"
echo " SpineSurg-CT  train fold ${FOLD}"
echo " Job        : ${SLURM_JOB_ID} @ $(hostname)"
echo " Trainer    : ${TRAINER}"
echo " Plans      : ${PLANS}"
echo " LSTV frac  : ${LSTV_OVERSAMPLE_FRAC}"
echo " W&B retry  : ${WANDB_INIT_MAX_RETRIES} @ ${WANDB_INIT_TIMEOUT_SEC}s  offline=${WANDB_ALLOW_OFFLINE}"
echo " Status     : ${RESUME_NOTE}"
echo " Started    : $(date)"
echo "================================================================"

# ----- Train ----------------------------------------------------------------
singularity exec "${SING_BINDS[@]}" "${CONTAINER}" \
    nnUNetv2_train ${DATASET_ID} ${CONFIG} ${FOLD} \
        -tr ${TRAINER} \
        -p  ${PLANS} \
        --npz \
        ${RESUME_FLAG}

# If we got here, training completed (otherwise nnUNetv2_train would have
# exited non-zero and set -e would have killed us). Final checkpoint should
# now exist on NFS (nnUNet_results is bound to NFS).
if [[ -f "${FINAL_CKPT}" ]]; then
    echo ""
    echo "[fold ${FOLD}] DONE  checkpoint_final.pth written"
    echo "  ${FINAL_CKPT}"
else
    echo ""
    echo "WARN: training exited cleanly but checkpoint_final.pth missing at" >&2
    echo "  ${FINAL_CKPT}" >&2
    echo "Resubmit this job to continue from checkpoint_latest.pth." >&2
    exit 2
fi

echo "  finished: $(date)"
