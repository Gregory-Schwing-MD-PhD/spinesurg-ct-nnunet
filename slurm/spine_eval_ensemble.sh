#!/bin/bash
#SBATCH --job-name=spine_eval5
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=250G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=8:00:00
#SBATCH --output=logs/spine_eval5_%j.out
#SBATCH --error=logs/spine_eval5_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=go2432@wayne.edu
# =============================================================================
# SpineSurg-CT  --  Stage C (5-fold ensemble eval)
# =============================================================================
# Runs after all 5 folds are trained. Ensemble predicts the test set using
# all 5 checkpoints (averaged softmax), then runs both eval scripts.
#
# Idempotent: --continue_prediction skips cases already done.
# Verifies all 5 folds exist before starting (hard-fails otherwise).
# =============================================================================

set -euo pipefail

DATASET_ID=802
DATASET_NAME="SpineSurgCTFull"
CONFIG="3d_fullres"
TRAINER="${TRAINER:-nnUNetTrainerWandB_1000ep_LSTVOversample}"
GPU_MEMORY_TARGET_GB=100
PLANS="nnUNetResEncUNetPlans_${GPU_MEMORY_TARGET_GB}G"

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-${HOME}/SpineSurg-CT}"
NNUNET_NFS="${PROJECT_ROOT}/nnunet"
CONTAINER="${PROJECT_ROOT}/containers/spinesurg-ct.sif"
WANDB_TRAINER_HOST="${PROJECT_ROOT}/tools/nnunet_wandb_variant.py"
WANDB_TRAINER_CONTAINER="/opt/conda/lib/python3.11/site-packages/nnunetv2/training/nnUNetTrainer/variants/nnunet_wandb_variant.py"

DS_DIR_NAME="Dataset$(printf '%03d' ${DATASET_ID})_${DATASET_NAME}"

# ----- Verify all 5 folds trained -------------------------------------------
for F in 0 1 2 3 4; do
    CKPT="${NNUNET_NFS}/results/${DS_DIR_NAME}/${TRAINER}__${PLANS}__${CONFIG}/fold_${F}/checkpoint_final.pth"
    if [[ ! -f "${CKPT}" ]]; then
        echo "ERROR: fold ${F} checkpoint missing: ${CKPT}" >&2
        echo "       Train all 5 folds before running ensemble eval." >&2
        exit 1
    fi
done
echo "All 5 fold checkpoints present."

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

PRED_DIR_NFS="${NNUNET_NFS}/predictions/${DS_DIR_NAME}_${CONFIG}_${TRAINER}_${PLANS}_ENSEMBLE"
PRED_DIR_CONT="/nnunet_nfs/predictions/${DS_DIR_NAME}_${CONFIG}_${TRAINER}_${PLANS}_ENSEMBLE"
mkdir -p "${PRED_DIR_NFS}"

N_EXISTING=$({ find "${PRED_DIR_NFS}" -maxdepth 1 -name "*.nii.gz" 2>/dev/null || true; } | wc -l)
N_TEST=$({ find "${NNUNET_NFS}/raw/${DS_DIR_NAME}/imagesTs" -maxdepth 1 -name "*_0000.nii.gz" 2>/dev/null || true; } | wc -l)
echo "Test cases: ${N_TEST}   existing ensemble predictions: ${N_EXISTING}"

if [[ "${N_EXISTING}" -ge "${N_TEST}" && "${N_TEST}" -gt 0 ]]; then
    echo "All ensemble predictions exist; skipping predict."
else
    CONTINUE_FLAG=""
    [[ "${N_EXISTING}" -gt 0 ]] && CONTINUE_FLAG="--continue_prediction"
    singularity exec "${SING_BINDS[@]}" "${CONTAINER}" \
        nnUNetv2_predict \
            -i  "/nnunet_nfs/raw/${DS_DIR_NAME}/imagesTs" \
            -o  "${PRED_DIR_CONT}" \
            -d  ${DATASET_ID} \
            -c  ${CONFIG} \
            -f  0 1 2 3 4 \
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
        --n_classes       10

echo ""; echo "----- paper metrics -----"
singularity exec "${SING_BINDS[@]}" "${CONTAINER}" \
    python tools/eval_paper_metrics.py \
        --predictions_dir "${PRED_DIR_CONT}" \
        --labels_dir      "/nnunet_nfs/raw/${DS_DIR_NAME}/labelsTs" \
        --output_dir      "${PRED_DIR_CONT}/paper_metrics"

echo ""
echo "================================================================"
echo " 5-FOLD ENSEMBLE EVAL COMPLETE"
echo "   predictions : ${PRED_DIR_NFS}"
echo "   basic       : ${PRED_DIR_NFS}/metrics.md"
echo "   paper       : ${PRED_DIR_NFS}/paper_metrics/metrics_paper.md"
echo "================================================================"
