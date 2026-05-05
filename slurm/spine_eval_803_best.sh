#!/bin/bash
#SBATCH --job-name=spine_eval_803
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=logs/spine_eval_803_%j.out
#SBATCH --error=logs/spine_eval_803_%j.err
# =============================================================================
# Stage 2 / 2: EVAL ONLY — Dataset803 / 500ep / checkpoint_best
# =============================================================================
# CPU-only. Reads predictions written by spine_predict_803_best.sh and
# computes per-case dice + 6-way LSTV subgroup aggregates.
# Env block mirrors slurm/spine_prep.sh exactly.
#
# Re-run-safe: idempotent, overwrites previous eval/ output. Predictions
# themselves are never touched.
#
# Override checkpoint flavor (must match the predict job that wrote
# the predictions):
#   sbatch --export=ALL,CHECKPOINT=checkpoint_latest.pth slurm/spine_eval_803_best.sh
# =============================================================================

set -euo pipefail

# -- Singularity runtime dirs (matches slurm/spine_prep.sh) ------------------
export SINGULARITY_TMPDIR="/tmp/${USER}_job_${SLURM_JOB_ID}"
export XDG_RUNTIME_DIR="${SINGULARITY_TMPDIR}/runtime"
export NXF_SINGULARITY_CACHEDIR="${HOME}/singularity_cache"
mkdir -p "${SINGULARITY_TMPDIR}" "${XDG_RUNTIME_DIR}" "${NXF_SINGULARITY_CACHEDIR}"
trap 'rm -rf "${SINGULARITY_TMPDIR}"' EXIT
export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
unset JAVA_HOME; which singularity
export NXF_SINGULARITY_HOME_MOUNT=true
unset LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE

# -- Config ------------------------------------------------------------------
DATASET_ID=803
DATASET_NAME="SpineSurgCTFullMerged"
CONFIG="3d_fullres"
TRAINER="${TRAINER:-nnUNetTrainerWandB_500ep_LSTVOversample}"
GPU_MEMORY_TARGET_GB=100
PLANS="nnUNetResEncUNetPlans_${GPU_MEMORY_TARGET_GB}G"
CHECKPOINT="${CHECKPOINT:-checkpoint_best.pth}"
WORKERS="${WORKERS:-20}"

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-${HOME}/spinesurg-ct-nnunet}"
NNUNET_NFS="${PROJECT_ROOT}/nnunet"
CONTAINER="${PROJECT_ROOT}/containers/spinesurg-ct.sif"
EVAL_SCRIPT_HOST="${PROJECT_ROOT}/tools/eval_ensemble_test.py"

DS_DIR_NAME="Dataset$(printf '%03d' ${DATASET_ID})_${DATASET_NAME}"

mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/tmp"

if [[ ! -f "${EVAL_SCRIPT_HOST}" ]]; then
    echo "ERROR: ${EVAL_SCRIPT_HOST} not found." >&2
    exit 1
fi

CKPT_TAG=$(echo "${CHECKPOINT}" | sed -e 's/checkpoint_//' -e 's/\.pth$//')
PRED_DIR_NFS="${NNUNET_NFS}/predictions/${DS_DIR_NAME}_${CONFIG}_${TRAINER}_${PLANS}_ENSEMBLE_${CKPT_TAG}"
PRED_DIR_CONT="/nnunet_nfs/predictions/${DS_DIR_NAME}_${CONFIG}_${TRAINER}_${PLANS}_ENSEMBLE_${CKPT_TAG}"

if [[ ! -d "${PRED_DIR_NFS}" ]]; then
    echo "ERROR: predictions directory not found: ${PRED_DIR_NFS}" >&2
    echo "       Run spine_predict_803_best.sh first." >&2
    exit 1
fi

echo "================================================================"
echo " SPINESURG-CT EVAL  (Dataset${DATASET_ID} / ${CONFIG})"
echo "================================================================"
echo "  trainer       : ${TRAINER}"
echo "  checkpoint    : ${CHECKPOINT}"
echo "  workers       : ${WORKERS}"
echo "  predictions   : ${PRED_DIR_NFS}"
echo "  job id        : ${SLURM_JOB_ID:-<not in slurm>}"
echo "  start time    : $(date -Iseconds)"
echo "================================================================"

N_PRED=$({ find "${PRED_DIR_NFS}" -maxdepth 1 -name "*.nii.gz" 2>/dev/null || true; } | wc -l)
N_TEST=$({ find "${NNUNET_NFS}/raw/${DS_DIR_NAME}/labelsTs" -maxdepth 1 -name "*.nii.gz" 2>/dev/null || true; } | wc -l)
echo "Predictions found: ${N_PRED}   labelsTs: ${N_TEST}"
if [[ "${N_PRED}" -lt "${N_TEST}" ]]; then
    echo "WARN: only ${N_PRED}/${N_TEST} predictions present. Eval will run on partial set." >&2
fi

LSTV_JSON_HOST="${PROJECT_ROOT}/nnunet/raw/${DS_DIR_NAME}/lstv_cases.json"
if [[ ! -f "${LSTV_JSON_HOST}" ]]; then
    echo "WARN: lstv_cases.json not found at ${LSTV_JSON_HOST}" >&2
    echo "      Subgroup analysis will fall back to all-Normal." >&2
fi

# -- Singularity env mirroring spine_prep.sh ---------------------------------
export SINGULARITYENV_TMPDIR="/workspace/tmp"
export SINGULARITYENV_nnUNet_raw="/nnunet_nfs/raw"
export SINGULARITYENV_nnUNet_preprocessed="/nnunet_nfs/preprocessed"
export SINGULARITYENV_nnUNet_results="/nnunet_nfs/results"

SING_BINDS=(
    --bind "${PROJECT_ROOT}:/workspace"
    --bind "${NNUNET_NFS}:/nnunet_nfs"
    --pwd  /workspace
)

EVAL_OUT_DIR_NFS="${PRED_DIR_NFS}/eval"
EVAL_OUT_DIR_CONT="${PRED_DIR_CONT}/eval"
mkdir -p "${EVAL_OUT_DIR_NFS}"

echo ""
echo "Launching eval at $(date -Iseconds) ..."
echo "----------------------------------------------------------------"
stdbuf -oL -eL singularity exec "${SING_BINDS[@]}" "${CONTAINER}" \
    python -u /workspace/tools/eval_ensemble_test.py \
        --predictions_dir "${PRED_DIR_CONT}" \
        --labels_dir      "/nnunet_nfs/raw/${DS_DIR_NAME}/labelsTs" \
        --lstv_cases_json "/nnunet_nfs/raw/${DS_DIR_NAME}/lstv_cases.json" \
        --output_dir      "${EVAL_OUT_DIR_CONT}" \
        --workers         ${WORKERS} \
        --verbose

echo ""
echo "================================================================"
echo " EVAL COMPLETE  (${CHECKPOINT})  $(date -Iseconds)"
echo "================================================================"
echo "   per-case CSV  : ${EVAL_OUT_DIR_NFS}/per_case_dice.csv"
echo "   subgroup CSV  : ${EVAL_OUT_DIR_NFS}/subgroup_summary.csv"
echo "   subgroup MD   : ${EVAL_OUT_DIR_NFS}/subgroup_summary.md"
echo "   headline JSON : ${EVAL_OUT_DIR_NFS}/headline.json"
echo "================================================================"
echo ""
echo "Quick view:"
echo "   cat ${EVAL_OUT_DIR_NFS}/subgroup_summary.md"
echo "   jq . ${EVAL_OUT_DIR_NFS}/headline.json"
