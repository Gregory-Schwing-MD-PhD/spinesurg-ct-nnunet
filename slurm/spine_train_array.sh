#!/bin/bash
#SBATCH --job-name=spine_kfold
#SBATCH -q gpu
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=250G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=5-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --open-mode=append
#SBATCH --output=logs/spine_kfold_f%a_%A.out
#SBATCH --error=logs/spine_kfold_f%a_%A.err
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=go2432@wayne.edu
# =============================================================================
# SpineSurg-CT  --  Stage B (5-fold CV training array, FULL PARALLEL)
# =============================================================================
#
# Configuration
# -------------
# 1. All 5 folds run concurrently on their own H200 (--array=0-4, no throttle).
# 2. Default trainer: nnUNetTrainerWandB_1000ep_LSTVOversample
#    (1000 epochs, 500 iter/epoch, LSTV oversampling for both subtypes).
# 3. --requeue + --signal=B:USR1@180: SLURM auto-resubmits the same array
#    task on walltime hit or node failure, and the script catches USR1
#    180s before walltime to flush a fresh checkpoint cleanly so the
#    requeue resumes from current state instead of one ~5 epochs old.
#    NOTE: this user's gpu queue is non-preempting, so --requeue is pure
#    upside (no risk of being bumped back to the queue mid-training).
# 4. --time=5-00:00:00 (5 days). 1000 epochs * 500 iter/epoch on H200
#    runs ~80h with LSTV oversample I/O overhead; 5 days has margin.
# 5. --open-mode=append: log lines accumulate across requeues. Filename
#    uses array job ID (%A) so all requeues of fold N append to the
#    same log file.
#
# Resume guarantee
# ----------------
# Three independent layers protect against losing progress:
#
#   a) nnU-Net writes checkpoint_latest.pth to nnUNet_results (bound
#      directly to NFS) at the end of every epoch. No local staging
#      of results — they hit durable storage immediately.
#   b) Fold-level idempotency: if checkpoint_final.pth exists, this
#      script exits 0 immediately. If checkpoint_latest.pth exists,
#      training resumes with --c. Both apply on every (re)launch.
#   c) SLURM --requeue: walltime / node failure -> SLURM resubmits the
#      same array task. The script re-runs from the top, hits the
#      latest checkpoint via (b), continues.
#
# How to (re)launch
# -----------------
# Fresh array of 5 folds:
#     sbatch slurm/spine_train_array.sh
#
# Manually resume one specific fold (e.g., crashed and didn't auto-requeue):
#     sbatch --array=2 slurm/spine_train_array.sh
#
# Force-restart a fold from scratch:
#     rm nnunet/results/Dataset802_SpineSurgCTFull/<TRAINER>__<PLANS>__3d_fullres/fold_2/checkpoint_*.pth
#     sbatch --array=2 slurm/spine_train_array.sh
#
# Monitor:
#     squeue -u $USER -o "%.10i %.30j %.10T %.20R %.10M"
#     tail -f logs/spine_kfold_f0_<ARRAY_JOB_ID>.out
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
    CKPT_AGE=$(( $(date +%s) - $(stat -c %Y "${LATEST_CKPT}" 2>/dev/null || stat -f %m "${LATEST_CKPT}") ))
    CKPT_SIZE=$(stat -c %s "${LATEST_CKPT}" 2>/dev/null || stat -f %z "${LATEST_CKPT}")
    echo "[fold ${FOLD}] resuming from checkpoint_latest.pth"
    echo "[fold ${FOLD}]   ckpt age:  ${CKPT_AGE}s ago"
    echo "[fold ${FOLD}]   ckpt size: ${CKPT_SIZE} bytes"
fi

# ----- USR1 trap for graceful shutdown on walltime ---------------------------
# SBATCH --signal=B:USR1@180 sends USR1 to the BATCH script 180s before
# walltime expires. We forward it to the singularity container's training
# process so nnU-Net can checkpoint-and-exit before the hard kill. On the
# subsequent --requeue, training resumes from this fresh checkpoint.
TRAIN_PID=""
on_usr1() {
    echo "[fold ${FOLD}] received SIGUSR1 (180s before walltime); forwarding to trainer for graceful shutdown" >&2
    if [[ -n "${TRAIN_PID}" ]] && kill -0 "${TRAIN_PID}" 2>/dev/null; then
        kill -USR1 "${TRAIN_PID}" 2>/dev/null || true
        # Give nnU-Net up to 150s to flush; SLURM hard-kills at 180s anyway.
        wait "${TRAIN_PID}" 2>/dev/null || true
    fi
    echo "[fold ${FOLD}] graceful shutdown complete; SLURM will requeue" >&2
}
trap on_usr1 USR1

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
STAGE_START=$(date +%s)
rsync -aW --no-compress "${NFS_PREP_DS}/" "${LOCAL_PREP_DS}/"
rsync -aW --no-compress "${NFS_RAW_DS}/"  "${LOCAL_RAW_DS}/"
STAGE_TIME=$(($(date +%s) - STAGE_START))
echo "[fold ${FOLD}] staging done in ${STAGE_TIME}s"

# ----- W&B token ------------------------------------------------------------
WANDB_API_KEY=""
[[ -f "${HOME}/.wandb/token" ]] && WANDB_API_KEY="$(cat ${HOME}/.wandb/token)"

# ----- Environment ----------------------------------------------------------
export SINGULARITYENV_TMPDIR="/workspace/tmp"
export SINGULARITYENV_nnUNet_raw="/nnunet_local/raw"
export SINGULARITYENV_nnUNet_preprocessed="/nnunet_local/preprocessed"
# nnUNet_results binds to NFS so checkpoints persist across requeues.
export SINGULARITYENV_nnUNet_results="/nnunet_nfs/results"
export SINGULARITYENV_nnUNet_def_n_proc="${NNUNET_EXPORT_POOL}"
export SINGULARITYENV_nnUNet_n_proc_DA="${NNUNET_DA_WORKERS}"
export SINGULARITYENV_SPINESURG_RUN_VAL_EXPORT="${SPINESURG_RUN_VAL_EXPORT}"
export SINGULARITYENV_WANDB_API_KEY="${WANDB_API_KEY}"
export SINGULARITYENV_WANDB_PROJECT="SpineSurg-CT"
# Stable W&B run ID across requeues so logs stitch together cleanly:
export SINGULARITYENV_WANDB_RUN_ID="${DS_DIR_NAME}_${TRAINER}_${PLANS}_f${FOLD}"
export SINGULARITYENV_WANDB_RUN_NAME="${DS_DIR_NAME}_${TRAINER}_${PLANS}_f${FOLD}"
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
echo " 5-fold CV (parallel)  |  fold ${FOLD}"
echo "   array job     : ${SLURM_ARRAY_JOB_ID:-?}"
echo "   array task    : ${SLURM_ARRAY_TASK_ID:-?}"
echo "   slurm job     : ${SLURM_JOB_ID:-?}"
echo "   restart_count : ${SLURM_RESTART_COUNT:-0}"
echo "   node          : $(hostname)"
echo "   gpu           : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "   trainer       : ${TRAINER}"
echo "   plans         : ${PLANS}"
echo "   LSTV frac     : ${LSTV_OVERSAMPLE_FRAC}"
echo "   status        : $([[ -n ${RESUME_FLAG} ]] && echo 'RESUMING' || echo 'fresh start')"
echo "   started       : $(date)"
echo "================================================================"

# ----- Launch training in background so the USR1 trap can fire --------------
singularity exec "${TRAIN_BINDS[@]}" "${CONTAINER}" \
    nnUNetv2_train ${DATASET_ID} ${CONFIG} ${FOLD} \
        -tr ${TRAINER} \
        -p  ${PLANS} \
        --npz \
        ${RESUME_FLAG} &
TRAIN_PID=$!
echo "[fold ${FOLD}] training pid=${TRAIN_PID}"

# `wait` is interrupted by signals; the USR1 trap runs above this line.
wait "${TRAIN_PID}"
TRAIN_EXIT=$?

if [[ -f "${FINAL_CKPT}" ]]; then
    echo "[fold ${FOLD}] DONE  $(date)  (exit=${TRAIN_EXIT})"
    exit 0
fi

# Training did not produce checkpoint_final.pth. Two scenarios:
#   a) Walltime hit / node failure: SLURM will auto-requeue this array
#      task. Exit non-zero so SLURM sees the job hasn't finished.
#   b) Actual training crash (NaN loss, OOM, etc.): mail-on-FAIL fires,
#      user inspects logs and resubmits manually.
# Exit codes 130 (SIGINT) and 143 (SIGTERM) are common preemption signals.
if [[ -n "${SLURM_RESTART_COUNT:-}" || "${TRAIN_EXIT}" -eq 130 || "${TRAIN_EXIT}" -eq 143 ]]; then
    echo "[fold ${FOLD}] training did not complete (exit=${TRAIN_EXIT}); SLURM will requeue."
    exit 1
fi
echo "WARN: training exited (code=${TRAIN_EXIT}) but checkpoint_final.pth missing." >&2
echo "      To resume from checkpoint_latest.pth:" >&2
echo "        sbatch --array=${FOLD} slurm/spine_train_array.sh" >&2
echo "      To force-restart fold ${FOLD} from scratch:" >&2
echo "        rm ${RESULT_DIR}/checkpoint_*.pth" >&2
echo "        sbatch --array=${FOLD} slurm/spine_train_array.sh" >&2
exit 2
