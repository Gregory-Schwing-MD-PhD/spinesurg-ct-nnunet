#!/bin/bash
#SBATCH --job-name=spine_train
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=250G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@1200
#SBATCH --output=logs/spine_train_f%x_%j.out
#SBATCH --error=logs/spine_train_f%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=go2432@wayne.edu
# SELF-REQUEUE. 500 epochs of ResEnc-L run past the 48 h a job asks for. Twenty minutes
# before the limit SLURM sends USR1 to this shell; the trap resubmits the same fold with the
# same environment, and the new job resumes from checkpoint_latest.pth (the --c path below).
# A fold that finishes writes checkpoint_final.pth and the next submission is a no-op.
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

# The singularity that works on the GPU nodes is the 3.8.6 in the nextflow conda env (the
# one spine_prep.sh uses). The user-local singularity-ce 4.0.2 that is first in PATH fails
# to extract the image there ("fork/exec /usr/bin/singularity: no such file"), which is
# what killed five folds on 2026-09-07 right after staging.
export PATH="${HOME}/mambaforge/envs/nextflow/bin:${PATH}"
unset LD_LIBRARY_PATH PYTHONPATH
which singularity

FOLD="${FOLD:-0}"
if ! [[ "${FOLD}" =~ ^[0-4]$ ]]; then
    echo "ERROR: FOLD must be 0..4, got '${FOLD}'" >&2
    exit 1
fi

# ----- TOGGLES ---------------------------------------------------------------
export SPINESURG_RUN_VAL_EXPORT="${SPINESURG_RUN_VAL_EXPORT:-1}"   # write validation predictions; the evaluation needs them
export WANDB_INIT_MAX_RETRIES="${WANDB_INIT_MAX_RETRIES:-5}"
export WANDB_INIT_TIMEOUT_SEC="${WANDB_INIT_TIMEOUT_SEC:-180}"
export WANDB_ALLOW_OFFLINE="${WANDB_ALLOW_OFFLINE:-1}"
export LSTV_OVERSAMPLE_FRAC="${LSTV_OVERSAMPLE_FRAC:-0.5}"
# -----------------------------------------------------------------------------

# ----- Config (match Stage A) -----------------------------------------------
# Every one of these is overridable from the environment so that the LSTV benchmark arms
# (Dataset810/811/812, ResEnc-L) run through the same script as the 802/803 baselines.
DATASET_ID="${DATASET_ID:-802}"
DATASET_NAME="${DATASET_NAME:-SpineSurgCTFull}"
CONFIG="${CONFIG:-3d_fullres}"
TRAINER="${TRAINER:-nnUNetTrainerWandB_1000ep_LSTVOversample}"
PLANNER="${PLANNER:-nnUNetPlannerResEncM}"
GPU_MEMORY_TARGET_GB="${GPU_MEMORY_TARGET_GB:-100}"
case "${PLANNER}" in
    nnUNetPlannerResEncM)  PLANS="nnUNetResEncUNetPlans_${GPU_MEMORY_TARGET_GB}G"   ;;
    nnUNetPlannerResEncL)  PLANS="nnUNetResEncUNetLPlans_${GPU_MEMORY_TARGET_GB}G"  ;;
    nnUNetPlannerResEncXL) PLANS="nnUNetResEncUNetXLPlans_${GPU_MEMORY_TARGET_GB}G" ;;
    ExperimentPlanner)     PLANS="nnUNetPlans"                                     ;;
    *) echo "ERROR: unknown planner ${PLANNER}" >&2; exit 1 ;;
esac
# a sibling plans file (patch-shape ablation, slurm/make_plan_variants.sh): results land in their own dir
PLANS="${PLANS_OVERRIDE:-${PLANS}}"
# msa nodes: 128 cores for 4 GPUs, so 32 CPUs per fold; the H200 sat idle behind the
# augmentation pipeline at 12 workers (epoch 19 min, GPU 0% when sampled)
NNUNET_EXPORT_POOL="${NNUNET_EXPORT_POOL:-12}"
NNUNET_DA_WORKERS="${NNUNET_DA_WORKERS:-20}"

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

# ----- Where the data is read from --------------------------------------------
# STAGE_LOCAL=1 copies the preprocessed and raw trees to node-local scratch first (the old
# behaviour). Default is 0: read straight from NFS. Dataset810 is 222 GB packed and about a
# terabyte once nnU-Net unpacks it; two folds on one GPU node filled /tmp and both died at
# container start (2026-09-07). With the .npy unpacked once by slurm/unpack_dataset.sh, the
# trainer memory-maps patches from NFS and copies nothing.
STAGE_LOCAL="${STAGE_LOCAL:-0}"; STAGE_LOCAL_REQUESTED="${STAGE_LOCAL}"
if [[ "${STAGE_LOCAL}" == "auto" ]]; then
    # stage only if this node can hold one unpacked copy (~1.1 TB) with margin; a second
    # staged fold on the same node would fill /tmp and kill both
    free_gb=$(( $(df -k --output=avail /tmp | tail -1) / 1024 / 1024 ))
    if [[ ${free_gb} -ge 1300 ]]; then STAGE_LOCAL=1; else STAGE_LOCAL=0; fi
    echo "[fold ${FOLD}] STAGE_LOCAL=auto -> ${STAGE_LOCAL} (/tmp free ${free_gb} GB)"
fi
if [[ "${STAGE_LOCAL}" == "1" ]]; then
    if [[ -n "${SLURM_TMPDIR:-}" && -d "${SLURM_TMPDIR}" ]]; then
        LOCAL_SCRATCH="${SLURM_TMPDIR}"
    elif [[ -d "/scratch/${USER}" ]]; then
        LOCAL_SCRATCH="/scratch/${USER}/job_${SLURM_JOB_ID}_f${FOLD}"; mkdir -p "${LOCAL_SCRATCH}"
    else
        LOCAL_SCRATCH="/tmp/${USER}_${SLURM_JOB_ID}_f${FOLD}"; mkdir -p "${LOCAL_SCRATCH}"
    fi
    trap 'rm -rf "${LOCAL_SCRATCH}" "${SINGULARITY_TMPDIR:-/nonexistent}"' EXIT
    LOCAL_NNUNET="${LOCAL_SCRATCH}/nnunet"
    LOCAL_PREP="${LOCAL_NNUNET}/preprocessed"
    LOCAL_RAW="${LOCAL_NNUNET}/raw"
    mkdir -p "${LOCAL_PREP}" "${LOCAL_RAW}"
    LOCAL_PREP_DS="${LOCAL_PREP}/${DS_DIR_NAME}"
    LOCAL_RAW_DS="${LOCAL_RAW}/${DS_DIR_NAME}"
    echo "[fold ${FOLD}] staging preprocessed (packed .npz only; unpacked locally by the trainer) -> ${LOCAL_PREP_DS}"
    mkdir -p "${LOCAL_PREP_DS}"
    # the .npy unpacked on NFS by slurm/unpack_dataset.sh are ~1 TB; copying the 222 GB of
    # .npz and letting nnU-Net unpack on the node's 1.7 TB /tmp fits ONE fold per node
    rsync -aW --no-compress --exclude '*.npy' "${NFS_PREP_DS}/" "${LOCAL_PREP_DS}/"
    df -h "${LOCAL_SCRATCH}" | tail -1
    echo "[fold ${FOLD}] staging raw -> ${LOCAL_RAW_DS} (needed for LSTV scanner)"
    mkdir -p "${LOCAL_RAW_DS}"
    rsync -aW --no-compress "${NFS_RAW_DS}/" "${LOCAL_RAW_DS}/"
else
    LOCAL_NNUNET="${NNUNET_NFS}"
    echo "[fold ${FOLD}] reading preprocessed and raw straight from ${NNUNET_NFS} (STAGE_LOCAL=0)"
    if ! ls "${NFS_PREP_DS}"/nnUNetPlans_3d_fullres/*.npy >/dev/null 2>&1; then
        echo "WARN: no .npy in ${NFS_PREP_DS}; run slurm/unpack_dataset.sh first so folds do not race to unpack" >&2
    fi
fi

# ----- Singularity env ------------------------------------------------------
# Python multiprocessing puts its listener sockets under TMPDIR; on NFS (/workspace/tmp)
# they vanished mid-run and every augmentation worker died (all four Dataset810 folds,
# 2026-09-08). Node-local, per job.
JOB_TMP="/tmp/${USER}_job_${SLURM_JOB_ID}_f${FOLD}_tmp"; mkdir -p "${JOB_TMP}"
export SINGULARITYENV_TMPDIR="${JOB_TMP}"
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
export SINGULARITYENV_SPINESURG_LABEL_SCHEME="${SPINESURG_LABEL_SCHEME:-merged}"
# the biased dataloader is not baked into the image: bound in above, and importable from
# /workspace/tools as a second route
export SINGULARITYENV_PYTHONPATH="/workspace/tools"
export SINGULARITYENV_WANDB_RUN_ID="${DS_DIR_NAME}_${TRAINER}_${PLANS}_f${FOLD}"
export OMP_NUM_THREADS=4

export SINGULARITY_TMPDIR="/tmp/${USER}_job_${SLURM_JOB_ID}_f${FOLD}_singularity"
export XDG_RUNTIME_DIR="${SINGULARITY_TMPDIR}/runtime"
mkdir -p "${SINGULARITY_TMPDIR}" "${XDG_RUNTIME_DIR}"
trap "rm -rf ${SINGULARITY_TMPDIR} ${JOB_TMP}" EXIT

SING_BINDS=(
    --nv
    --bind "${PROJECT_ROOT}:/workspace"
    --bind "${LOCAL_NNUNET}:/nnunet_local"
    --bind "${NNUNET_NFS}:/nnunet_nfs"
    --bind "${HOME}/.wandb:${HOME}/.wandb"
    --bind "${WANDB_TRAINER_HOST}:${WANDB_TRAINER_CONTAINER}"
    --bind "${PROJECT_ROOT}/tools/lstv_biased_dataloader.py:${WANDB_TRAINER_CONTAINER%/*}/lstv_biased_dataloader.py"
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
requeue_self() {
    echo "[fold ${FOLD}] wall-time limit approaching at $(date); resubmitting to resume from checkpoint_latest.pth"
    cd "${PROJECT_ROOT}"
    sbatch --export=ALL,FOLD="${FOLD}",DATASET_ID="${DATASET_ID}",DATASET_NAME="${DATASET_NAME}",PLANNER="${PLANNER}",TRAINER="${TRAINER}",LSTV_OVERSAMPLE_FRAC="${LSTV_OVERSAMPLE_FRAC}",SPINESURG_RUN_VAL_EXPORT="${SPINESURG_RUN_VAL_EXPORT}",SPINESURG_LABEL_SCHEME="${SPINESURG_LABEL_SCHEME:-merged}",STAGE_LOCAL="${STAGE_LOCAL:-0}" \
        slurm/spine_train_fold.sh
    # let the trainer reach its next checkpoint write (every 50 epochs) rather than killing it now
    wait "${TRAIN_PID}" || true
    exit 0
}
trap requeue_self USR1

singularity exec "${SING_BINDS[@]}" "${CONTAINER}" \
    nnUNetv2_train ${DATASET_ID} ${CONFIG} ${FOLD} \
        -tr ${TRAINER} \
        -p  ${PLANS} \
        --npz \
        ${RESUME_FLAG} &
TRAIN_PID=$!
wait "${TRAIN_PID}"; TRAIN_RC=$?

# A crash that is not ours (worker death, NFS hiccup, node event) must not cost the run:
# resubmit to resume from checkpoint_latest.pth, up to RETRY_MAX times.
RETRY="${RETRY:-0}"; RETRY_MAX="${RETRY_MAX:-6}"
if [[ ${TRAIN_RC} -ne 0 && ! -f "${FINAL_CKPT}" ]]; then
    if [[ ${RETRY} -lt ${RETRY_MAX} ]]; then
        echo "[fold ${FOLD}] training exited ${TRAIN_RC} at $(date); resubmitting (retry $((RETRY+1))/${RETRY_MAX}) to resume"
        cd "${PROJECT_ROOT}"
        sbatch --export=ALL,FOLD="${FOLD}",DATASET_ID="${DATASET_ID}",DATASET_NAME="${DATASET_NAME}",PLANNER="${PLANNER}",TRAINER="${TRAINER}",LSTV_OVERSAMPLE_FRAC="${LSTV_OVERSAMPLE_FRAC}",SPINESURG_RUN_VAL_EXPORT="${SPINESURG_RUN_VAL_EXPORT}",SPINESURG_LABEL_SCHEME="${SPINESURG_LABEL_SCHEME:-merged}",STAGE_LOCAL="${STAGE_LOCAL_REQUESTED:-0}",PLANS_OVERRIDE="${PLANS_OVERRIDE:-}",RETRY=$((RETRY+1))             slurm/spine_train_fold.sh
        exit ${TRAIN_RC}
    fi
    echo "[fold ${FOLD}] training exited ${TRAIN_RC} and retries are exhausted" >&2
    exit ${TRAIN_RC}
fi

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
