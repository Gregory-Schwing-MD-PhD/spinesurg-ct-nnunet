#!/bin/bash
#SBATCH --job-name=spine_prep
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=8:00:00
#SBATCH --output=logs/spine_prep_%j.out
#SBATCH --error=logs/spine_prep_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=go2432@wayne.edu
# =============================================================================
# SpineSurg-CT  --  Stage A: convert + preprocess (SIMPLE, NFS-only)
# =============================================================================
# Convert writes symlinks like
#     imagesTr/X__fused_0000.nii.gz -> /data/hf_export/ct/X.nii.gz
# That target is a CONTAINER path; it resolves inside the container via the
# /data/hf_export bind mount but looks "broken" from the host. Therefore:
# do NOT run any host-side dangling-symlink cleanup. Earlier versions did,
# and they wiped the whole dataset on every "resume" run.
#
# Threading: 12 workers × 4 BLAS threads = 48 threads, fits RLIMIT_NPROC.
# =============================================================================

set -euo pipefail

# ── Singularity runtime dirs ─────────────────────────────────────────────────
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

# ── Config ───────────────────────────────────────────────────────────────────
DATASET_ID="${DATASET_ID:-802}"
DATASET_NAME="${DATASET_NAME:-SpineSurgCTFull}"
CONFIG="${CONFIG:-3d_fullres}"
PLANNER="${PLANNER:-nnUNetPlannerResEncM}"
GPU_MEMORY_TARGET_GB="${GPU_MEMORY_TARGET_GB:-100}"
N_PREPROCESS_THREADS="${N_PREPROCESS_THREADS:-12}"
BLAS_THREADS="${BLAS_THREADS:-4}"
SINGLE_FOLD_SPLITS="${SINGLE_FOLD_SPLITS:-0}"
INCLUDE_TRAIN_MATCH_TYPES="${INCLUDE_TRAIN_MATCH_TYPES:-}"
TEST_MATCH_TYPES="${TEST_MATCH_TYPES:-}"

case "${PLANNER}" in
    nnUNetPlannerResEncM)  PLANS="nnUNetResEncUNetPlans_${GPU_MEMORY_TARGET_GB}G"   ;;
    nnUNetPlannerResEncL)  PLANS="nnUNetResEncUNetLPlans_${GPU_MEMORY_TARGET_GB}G"  ;;
    nnUNetPlannerResEncXL) PLANS="nnUNetResEncUNetXLPlans_${GPU_MEMORY_TARGET_GB}G" ;;
    ExperimentPlanner)     PLANS="nnUNetPlans"                                     ;;
    *) echo "ERROR: unknown planner ${PLANNER}" >&2; exit 1 ;;
esac

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-${HOME}/spinesurg-ct-nnunet}"
HF_EXPORT_NFS="${PROJECT_ROOT}/data/hf_export"
SPLITS_FILE_HOST="${PROJECT_ROOT}/data/splits_5fold.json"
NNUNET_NFS="${PROJECT_ROOT}/nnunet"
CONTAINER="${PROJECT_ROOT}/containers/spinesurg-ct.sif"
WANDB_TRAINER_HOST="${PROJECT_ROOT}/tools/nnunet_wandb_variant.py"
WANDB_TRAINER_CONTAINER="/opt/conda/lib/python3.11/site-packages/nnunetv2/training/nnUNetTrainer/variants/nnunet_wandb_variant.py"

DS_DIR_NAME="Dataset$(printf '%03d' ${DATASET_ID})_${DATASET_NAME}"
NFS_RAW_DS="${NNUNET_NFS}/raw/${DS_DIR_NAME}"
NFS_PREP_DS="${NNUNET_NFS}/preprocessed/${DS_DIR_NAME}"
COMPLETE_MARKER="${NFS_PREP_DS}/.preprocessed_complete_${PLANS}"

mkdir -p "${NNUNET_NFS}/raw" "${NNUNET_NFS}/preprocessed" "${NNUNET_NFS}/results" \
         "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/tmp"

# ── Preflight ────────────────────────────────────────────────────────────────
[[ ! -f "${CONTAINER}"          ]] && { echo "ERROR: container not found: ${CONTAINER}" >&2; exit 1; }
[[ ! -d "${HF_EXPORT_NFS}/ct"   ]] && { echo "ERROR: ${HF_EXPORT_NFS}/ct missing" >&2; exit 1; }
[[ ! -f "${WANDB_TRAINER_HOST}" ]] && { echo "ERROR: ${WANDB_TRAINER_HOST} missing" >&2; exit 1; }
[[ ! -f "${PROJECT_ROOT}/tools/convert_hf_to_nnunet.py" ]] && {
    echo "ERROR: tools/convert_hf_to_nnunet.py missing" >&2; exit 1; }

if [[ "${SINGLE_FOLD_SPLITS}" != "1" ]]; then
    [[ ! -f "${SPLITS_FILE_HOST}" ]] && {
        echo "ERROR: ${SPLITS_FILE_HOST} missing." >&2; exit 1; }
    SP_MTIME=$(stat -c '%y' "${SPLITS_FILE_HOST}" 2>/dev/null || stat -f '%Sm' "${SPLITS_FILE_HOST}")
    echo "  splits_5fold.json mtime: ${SP_MTIME}"
fi

# ── Step 0: NFS cache check (fast exit if already complete) ──────────────────
if [[ -f "${COMPLETE_MARKER}" ]] \
   && [[ -d "${NFS_PREP_DS}/${CONFIG}" ]] \
   && [[ -f "${NFS_PREP_DS}/dataset_fingerprint.json" ]] \
   && [[ -f "${NFS_PREP_DS}/${PLANS}.json" ]] \
   && [[ -f "${NFS_RAW_DS}/dataset.json" ]]; then
    N_NPZ=$({ find "${NFS_PREP_DS}/${CONFIG}" -maxdepth 1 -name "*.npz" 2>/dev/null || true; } | wc -l)
    SIZE=$(du -sh "${NFS_PREP_DS}" 2>/dev/null | awk '{print $1}')
    echo "================================================================"
    echo " NFS cache HIT: ${NFS_PREP_DS}"
    echo "   ${N_NPZ} .npz files, ${SIZE} total"
    echo "   plans  : ${PLANS}"
    echo "================================================================"
    [[ -f "${NFS_RAW_DS}/splits_final.json" ]] && cp -f "${NFS_RAW_DS}/splits_final.json" "${NFS_PREP_DS}/"
    [[ -f "${NFS_RAW_DS}/lstv_cases.json"   ]] && cp -f "${NFS_RAW_DS}/lstv_cases.json"   "${NFS_PREP_DS}/"
    echo " NEXT: sbatch slurm/spine_train_array.sh"
    exit 0
fi

# >>> NO host-side dangling-symlink cleanup. <<<
# Convert writes container-path symlinks (/data/hf_export/...). They look
# "broken" from the host but resolve inside the container via the bind
# mount. Cleaning them up was the reason resumes kept failing.

echo "================================================================"
echo " Stage A: convert + preprocess (SIMPLE, NFS-only)"
echo "   Dataset      : ${DS_DIR_NAME}"
echo "   Plans        : ${PLANS} (target ${GPU_MEMORY_TARGET_GB} GB)"
echo "   NFS dest     : ${NFS_PREP_DS}"
echo "   Splits mode  : $([[ ${SINGLE_FOLD_SPLITS} == 1 ]] && echo 'single-fold' || echo '5-fold CV')"
echo "   Workers      : ${N_PREPROCESS_THREADS} × ${BLAS_THREADS} BLAS threads = $((N_PREPROCESS_THREADS*BLAS_THREADS)) total"
echo "   Started      : $(date)"
echo "================================================================"

# ── Singularity setup ────────────────────────────────────────────────────────
export SINGULARITYENV_TMPDIR="/workspace/tmp"
export SINGULARITYENV_nnUNet_raw="/nnunet_nfs/raw"
export SINGULARITYENV_nnUNet_preprocessed="/nnunet_nfs/preprocessed"
export SINGULARITYENV_nnUNet_results="/nnunet_nfs/results"

# Cap BLAS threads inside each worker to avoid blowing past RLIMIT_NPROC.
export SINGULARITYENV_OMP_NUM_THREADS=${BLAS_THREADS}
export SINGULARITYENV_OPENBLAS_NUM_THREADS=${BLAS_THREADS}
export SINGULARITYENV_MKL_NUM_THREADS=${BLAS_THREADS}
export SINGULARITYENV_NUMEXPR_NUM_THREADS=${BLAS_THREADS}
export SINGULARITYENV_VECLIB_MAXIMUM_THREADS=${BLAS_THREADS}

SING_BINDS=(
    --bind "${PROJECT_ROOT}:/workspace"
    --bind "${HF_EXPORT_NFS}:/data/hf_export"
    --bind "${NNUNET_NFS}:/nnunet_nfs"
    --bind "${WANDB_TRAINER_HOST}:${WANDB_TRAINER_CONTAINER}"
    --pwd  /workspace
)
if [[ "${SINGLE_FOLD_SPLITS}" != "1" ]]; then
    SING_BINDS+=( --bind "${SPLITS_FILE_HOST}:/workspace/splits_5fold.json:ro" )
fi

# ── Step 1: convert HF export -> nnU-Net raw (on NFS) ────────────────────────
echo ""; echo "----- Step 1: convert HF export -> nnU-Net raw -----"
if [[ -f "${NFS_RAW_DS}/dataset.json" ]]; then
    echo "  dataset.json present; skipping convert."
else
    CONVERT_ARGS=(
        --hf_export_dir  /data/hf_export
        --nnunet_raw_dir /nnunet_nfs/raw
        --dataset_id     ${DATASET_ID}
        --dataset_name   ${DATASET_NAME}
    )
    if [[ "${SINGLE_FOLD_SPLITS}" == "1" ]]; then
        CONVERT_ARGS+=( --single_fold_splits )
    else
        CONVERT_ARGS+=( --splits_file /workspace/splits_5fold.json )
    fi
    [[ -n "${INCLUDE_TRAIN_MATCH_TYPES}" ]] && \
        CONVERT_ARGS+=( --include_train_match_types "${INCLUDE_TRAIN_MATCH_TYPES}" )
    [[ -n "${TEST_MATCH_TYPES}" ]] && \
        CONVERT_ARGS+=( --test_match_types "${TEST_MATCH_TYPES}" )
    singularity exec "${SING_BINDS[@]}" "${CONTAINER}" \
        python tools/convert_hf_to_nnunet.py "${CONVERT_ARGS[@]}"
fi

# ── Step 2: plan + preprocess (reads + writes on NFS) ────────────────────────
echo ""; echo "----- Step 2: plan + preprocess -----"
rm -f "${COMPLETE_MARKER}"

PREP_START=$(date +%s)
singularity exec "${SING_BINDS[@]}" "${CONTAINER}" \
    nnUNetv2_plan_and_preprocess \
        -d    ${DATASET_ID} \
        -pl   ${PLANNER} \
        -gpu_memory_target ${GPU_MEMORY_TARGET_GB} \
        -overwrite_plans_name ${PLANS} \
        -c    ${CONFIG} \
        -np   ${N_PREPROCESS_THREADS} \
        --verify_dataset_integrity
PREP_TIME=$(($(date +%s) - PREP_START))
echo "  preprocess done in ${PREP_TIME}s"

# ── Step 3: copy splits + LSTV metadata into preprocessed/ ───────────────────
echo ""; echo "----- Step 3: copy metadata into preprocessed/ -----"
[[ -f "${NFS_RAW_DS}/splits_final.json" ]] && cp -f "${NFS_RAW_DS}/splits_final.json" "${NFS_PREP_DS}/" && echo "  copied splits_final.json"
[[ -f "${NFS_RAW_DS}/lstv_cases.json"   ]] && cp -f "${NFS_RAW_DS}/lstv_cases.json"   "${NFS_PREP_DS}/" && echo "  copied lstv_cases.json"

# ── Sanity check ─────────────────────────────────────────────────────────────
N_NPZ_NFS=$({ find "${NFS_PREP_DS}/${CONFIG}" -maxdepth 1 -name "*.npz" 2>/dev/null || true; } | wc -l)
NFS_SIZE=$(du -sh "${NFS_PREP_DS}" 2>/dev/null | awk '{print $1}')
[[ "${N_NPZ_NFS}" -eq 0 ]] && { echo "ERROR: zero .npz files after preprocess." >&2; exit 2; }

touch "${COMPLETE_MARKER}"

echo ""
echo "================================================================"
echo " Stage A COMPLETE   $(date)"
echo "   preprocess time : ${PREP_TIME}s"
echo "   .npz files      : ${N_NPZ_NFS}"
echo "   preprocessed    : ${NFS_PREP_DS}  (${NFS_SIZE})"
echo ""
echo " NEXT:  sbatch slurm/spine_train_array.sh"
echo "================================================================"
