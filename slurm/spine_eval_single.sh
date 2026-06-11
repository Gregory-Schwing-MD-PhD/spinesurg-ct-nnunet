#!/bin/bash
#SBATCH --job-name=spine_eval1
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=250G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=6:00:00
#SBATCH --output=logs/spine_eval1_%j.out
#SBATCH --error=logs/spine_eval1_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=go2432@wayne.edu
# =============================================================================
# SpineSurg-CT  --  Stage C (single fold eval)
# =============================================================================
# Predict the test set using ONE fold, then run both eval scripts.
# Idempotent: --continue_prediction skips test cases already predicted;
# eval scripts always rewrite (cheap).
#
# Usage:
#     FOLD=0 sbatch slurm/spine_eval_single.sh
#
# Fused-only ablation (predict + eval the fold-0 model on the fused test
# split). N_CLASSES=9 because the no-ignore scheme has values 0..8:
#     FOLD=0 DATASET_ID=804 DATASET_NAME=SpineSurgCTFusedOnly \
#       TRAINER=nnUNetTrainerWandB_500ep_LSTVOversample N_CLASSES=9 \
#       sbatch slurm/spine_eval_single.sh
# =============================================================================

set -euo pipefail

FOLD="${FOLD:-0}"

DATASET_ID="${DATASET_ID:-802}"
DATASET_NAME="${DATASET_NAME:-SpineSurgCTFull}"
CONFIG="${CONFIG:-3d_fullres}"
TRAINER="${TRAINER:-nnUNetTrainerWandB_1000ep_LSTVOversample}"
GPU_MEMORY_TARGET_GB="${GPU_MEMORY_TARGET_GB:-100}"
PLANS="${PLANS:-nnUNetResEncUNetPlans_${GPU_MEMORY_TARGET_GB}G}"
N_CLASSES="${N_CLASSES:-10}"

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-${HOME}/SpineSurg-CT}"
NNUNET_NFS="${PROJECT_ROOT}/nnunet"
CONTAINER="${PROJECT_ROOT}/containers/spinesurg-ct.sif"
WANDB_TRAINER_HOST="${PROJECT_ROOT}/tools/nnunet_wandb_variant.py"
WANDB_TRAINER_CONTAINER="/opt/conda/lib/python3.11/site-packages/nnunetv2/training/nnUNetTrainer/variants/nnunet_wandb_variant.py"

DS_DIR_NAME="Dataset$(printf '%03d' ${DATASET_ID})_${DATASET_NAME}"

FINAL_CKPT="${NNUNET_NFS}/results/${DS_DIR_NAME}/${TRAINER}__${PLANS}__${CONFIG}/fold_${FOLD}/checkpoint_final.pth"
if [[ ! -f "${FINAL_CKPT}" ]]; then
    echo "ERROR: fold ${FOLD} not trained (missing ${FINAL_CKPT})" >&2
    exit 1
fi

export SINGULARITYENV_nnUNet_raw="/nnunet_nfs/raw"
export SINGULARITYENV_nnUNet_preprocessed="/nnunet_nfs/preprocessed"
export SINGULARITYENV_nnUNet_results="/nnunet_nfs/results"
export SINGULARITY_TMPDIR="/tmp/${USER}_job_${SLURM_JOB_ID}_singularity"
export XDG_RUNTIME_DIR="${SINGULARITY_TMPDIR}/runtime"
mkdir -p "${SINGULARITY_TMPDIR}" "${XDG_RUNTIME_DIR}"
trap "rm -rf ${SINGULARITY_TMPDIR}" EXIT

SING_BINDS=(
    --nv
    --bind "${PROJECT_ROOT}:/workspace"
    --bind "${NNUNET_NFS}:/nnunet_nfs"
    --bind "${WANDB_TRAINER_HOST}:${WANDB_TRAINER_CONTAINER}"
    --pwd  /workspace
)

PRED_DIR_NFS="${NNUNET_NFS}/predictions/${DS_DIR_NAME}_${CONFIG}_${TRAINER}_${PLANS}_f${FOLD}"
PRED_DIR_CONT="/nnunet_nfs/predictions/${DS_DIR_NAME}_${CONFIG}_${TRAINER}_${PLANS}_f${FOLD}"
mkdir -p "${PRED_DIR_NFS}"

N_EXISTING=$({ find "${PRED_DIR_NFS}" -maxdepth 1 -name "*.nii.gz" 2>/dev/null || true; } | wc -l)
N_TEST=$({ find "${NNUNET_NFS}/raw/${DS_DIR_NAME}/imagesTs" -maxdepth 1 -name "*_0000.nii.gz" 2>/dev/null || true; } | wc -l)
echo "Test cases: ${N_TEST}   existing predictions: ${N_EXISTING}"

if [[ "${N_EXISTING}" -ge "${N_TEST}" && "${N_TEST}" -gt 0 ]]; then
    echo "All test predictions exist; skipping predict."
else
    CONTINUE_FLAG=""
    [[ "${N_EXISTING}" -gt 0 ]] && CONTINUE_FLAG="--continue_prediction"
    singularity exec "${SING_BINDS[@]}" "${CONTAINER}" \
        nnUNetv2_predict \
            -i  "/nnunet_nfs/raw/${DS_DIR_NAME}/imagesTs" \
            -o  "${PRED_DIR_CONT}" \
            -d  ${DATASET_ID} \
            -c  ${CONFIG} \
            -f  ${FOLD} \
            -tr ${TRAINER} \
            -p  ${PLANS} \
            ${CONTINUE_FLAG}
fi

echo ""; echo "----- basic eval -----"
singularity exec "${SING_BINDS[@]}" "${CONTAINER}" \
    python tools/eval_nnunet_predictions.py \
        --predictions_dir "${PRED_DIR_CONT}" \
        --labels_dir      "/nnunet_nfs/raw/${DS_DIR_NAME}/labelsTs" \
        --output_json     "${PRED_DIR_CONT}/metrics.json" \
        --output_md       "${PRED_DIR_CONT}/metrics.md" \
        --n_classes       ${N_CLASSES}

echo ""; echo "----- paper metrics -----"
singularity exec "${SING_BINDS[@]}" "${CONTAINER}" \
    python tools/eval_paper_metrics.py \
        --predictions_dir "${PRED_DIR_CONT}" \
        --labels_dir      "/nnunet_nfs/raw/${DS_DIR_NAME}/labelsTs" \
        --output_dir      "${PRED_DIR_CONT}/paper_metrics"

echo ""
echo "================================================================"
echo " SINGLE-FOLD EVAL COMPLETE  (fold ${FOLD})"
echo "   predictions : ${PRED_DIR_NFS}"
echo "   basic       : ${PRED_DIR_NFS}/metrics.md"
echo "   paper       : ${PRED_DIR_NFS}/paper_metrics/metrics_paper.md"
echo "================================================================"
