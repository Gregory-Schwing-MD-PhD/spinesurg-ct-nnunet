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
# Threading: 12 workers x 4 BLAS threads = 48 threads, fits RLIMIT_NPROC.
#
# v2 fix (2026-04-27)
# -------------------
# The preprocessed-data directory under preprocessed/<dataset>/ is named
# after the plans' data_identifier (e.g. "nnUNetPlans_3d_fullres"),
# NOT after ${CONFIG} (e.g. "3d_fullres"). resolve_prep_data_dir() now
# reads data_identifier from the plans JSON (with a glob fallback).
#
# v3 (2026-04-29) — flexible HF_EXPORT_DIR + SPLITS_FILE overrides
# ----------------------------------------------------------------
# Earlier this script hardcoded HF export at
#   ${PROJECT_ROOT}/data/hf_export
# and splits at
#   ${PROJECT_ROOT}/data/splits_5fold.json
# That broke when the export lives under ~/CTSpinoPelvic1K/data/hf_export
# (the natural Stage 3 output) and the splits ride along inside that
# export at hf_export/splits_5fold.json (where export_dataset.sh writes
# them). Two new env vars make this configurable:
#
#   HF_EXPORT_DIR   path to hf_export/ (CT, labels, manifest.json, ...)
#                   default tries:
#                     1. ${PROJECT_ROOT}/data/hf_export   (legacy)
#                     2. ${HOME}/CTSpinoPelvic1K/data/hf_export
#                   first hit wins.
#
#   SPLITS_FILE     path to splits_5fold.json
#                   default tries:
#                     1. ${HF_EXPORT_DIR}/splits_5fold.json  (preferred,
#                        co-located with the export it describes)
#                     2. ${PROJECT_ROOT}/data/splits_5fold.json (legacy)
#                   first hit wins.
#
# Both can be overridden explicitly to point anywhere on NFS:
#     HF_EXPORT_DIR=/some/path/hf_export sbatch slurm/spine_prep.sh
# =============================================================================

set -euo pipefail

# -- Singularity runtime dirs ------------------------------------------------
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

# -- HF export discovery -----------------------------------------------------
# Default search order: legacy in-project location first (so existing setups
# keep working), then the natural CTSpinoPelvic1K location. Either can be
# overridden with HF_EXPORT_DIR=...
if [[ -z "${HF_EXPORT_DIR:-}" ]]; then
    for cand in \
        "${PROJECT_ROOT}/data/hf_export" \
        "${HOME}/CTSpinoPelvic1K/data/hf_export"; do
        if [[ -d "${cand}/ct" ]]; then
            HF_EXPORT_DIR="${cand}"
            break
        fi
    done
fi
HF_EXPORT_DIR="${HF_EXPORT_DIR:-${PROJECT_ROOT}/data/hf_export}"
HF_EXPORT_NFS="${HF_EXPORT_DIR}"

# -- Splits file discovery ---------------------------------------------------
# Prefer the splits file co-located with the HF export it describes.
# Earlier setups put it in ${PROJECT_ROOT}/data/ — fall back to that for
# legacy compatibility.
if [[ -z "${SPLITS_FILE:-}" ]]; then
    for cand in \
        "${HF_EXPORT_NFS}/splits_5fold.json" \
        "${PROJECT_ROOT}/data/splits_5fold.json"; do
        if [[ -f "${cand}" ]]; then
            SPLITS_FILE="${cand}"
            break
        fi
    done
fi
SPLITS_FILE_HOST="${SPLITS_FILE:-${HF_EXPORT_NFS}/splits_5fold.json}"

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

# -- Resolver: find the preprocessed-data subdirectory -----------------------
# nnU-Net writes preprocessed npz/pkl files into
#   <preprocessed>/<dataset>/<data_identifier>/
# where <data_identifier> comes from plans.configurations.<config>.data_identifier.
# That string varies with planner + plans-name (e.g. "nnUNetPlans_3d_fullres",
# "nnUNetResEncUNetPlans_100G_3d_fullres", etc.), so we resolve it at
# runtime instead of assuming it equals ${CONFIG}.
#
# Strategy:
#   1. Read data_identifier from <plans>.json via python (preferred).
#   2. Fall back to a glob: any direct child of preprocessed/<dataset>/
#      whose name contains ${CONFIG} and that holds at least one .npz.
# Echoes the resolved absolute path on stdout, or empty string on failure.
resolve_prep_data_dir() {
    local plans_json="${NFS_PREP_DS}/${PLANS}.json"
    local cfg="${CONFIG}"
    local resolved=""

    if [[ -f "${plans_json}" ]] && command -v python >/dev/null 2>&1; then
        resolved=$(python - "${plans_json}" "${cfg}" <<'PY' 2>/dev/null || true
import json, sys
try:
    with open(sys.argv[1]) as f:
        p = json.load(f)
    di = p["configurations"][sys.argv[2]]["data_identifier"]
    print(di)
except Exception:
    pass
PY
)
    fi

    if [[ -n "${resolved}" ]] && [[ -d "${NFS_PREP_DS}/${resolved}" ]]; then
        echo "${NFS_PREP_DS}/${resolved}"
        return 0
    fi

    # Fallback: glob for a child dir containing ${CONFIG} in its name with npz files.
    if [[ -d "${NFS_PREP_DS}" ]]; then
        local cand
        while IFS= read -r cand; do
            if compgen -G "${cand}/*.npz" > /dev/null 2>&1; then
                echo "${cand}"
                return 0
            fi
        done < <(find "${NFS_PREP_DS}" -mindepth 1 -maxdepth 1 -type d -name "*${cfg}*" 2>/dev/null)
    fi

    echo ""
    return 1
}

# -- Preflight ---------------------------------------------------------------
[[ ! -f "${CONTAINER}"          ]] && { echo "ERROR: container not found: ${CONTAINER}" >&2; exit 1; }
[[ ! -d "${HF_EXPORT_NFS}/ct"   ]] && {
    echo "ERROR: ${HF_EXPORT_NFS}/ct missing" >&2
    echo "  Override with HF_EXPORT_DIR=/path/to/hf_export sbatch slurm/spine_prep.sh" >&2
    exit 1
}
[[ ! -f "${WANDB_TRAINER_HOST}" ]] && { echo "ERROR: ${WANDB_TRAINER_HOST} missing" >&2; exit 1; }
[[ ! -f "${PROJECT_ROOT}/tools/convert_hf_to_nnunet.py" ]] && {
    echo "ERROR: tools/convert_hf_to_nnunet.py missing" >&2; exit 1; }

if [[ "${SINGLE_FOLD_SPLITS}" != "1" ]]; then
    [[ ! -f "${SPLITS_FILE_HOST}" ]] && {
        echo "ERROR: splits file not found at ${SPLITS_FILE_HOST}" >&2
        echo "  Searched: ${HF_EXPORT_NFS}/splits_5fold.json" >&2
        echo "            ${PROJECT_ROOT}/data/splits_5fold.json" >&2
        echo "  Override with SPLITS_FILE=/path/to/splits.json or run with SINGLE_FOLD_SPLITS=1" >&2
        exit 1
    }
    SP_MTIME=$(stat -c '%y' "${SPLITS_FILE_HOST}" 2>/dev/null || stat -f '%Sm' "${SPLITS_FILE_HOST}")
    echo "  splits_5fold.json mtime: ${SP_MTIME}"
fi

# -- Step 0: NFS cache check (fast exit if already complete) -----------------
if [[ -f "${COMPLETE_MARKER}" ]] \
   && [[ -f "${NFS_PREP_DS}/dataset_fingerprint.json" ]] \
   && [[ -f "${NFS_PREP_DS}/${PLANS}.json" ]] \
   && [[ -f "${NFS_RAW_DS}/dataset.json" ]]; then
    PREP_DATA_DIR_CACHE=$(resolve_prep_data_dir || true)
    if [[ -n "${PREP_DATA_DIR_CACHE}" && -d "${PREP_DATA_DIR_CACHE}" ]]; then
        N_NPZ=$({ find "${PREP_DATA_DIR_CACHE}" -maxdepth 1 -name "*.npz" 2>/dev/null || true; } | wc -l)
        if [[ "${N_NPZ}" -gt 0 ]]; then
            SIZE=$(du -sh "${NFS_PREP_DS}" 2>/dev/null | awk '{print $1}')
            echo "================================================================"
            echo " NFS cache HIT: ${NFS_PREP_DS}"
            echo "   data dir : ${PREP_DATA_DIR_CACHE}"
            echo "   ${N_NPZ} .npz files, ${SIZE} total"
            echo "   plans    : ${PLANS}"
            echo "================================================================"
            [[ -f "${NFS_RAW_DS}/splits_final.json" ]] && cp -f "${NFS_RAW_DS}/splits_final.json" "${NFS_PREP_DS}/"
            [[ -f "${NFS_RAW_DS}/lstv_cases.json"   ]] && cp -f "${NFS_RAW_DS}/lstv_cases.json"   "${NFS_PREP_DS}/"
            echo " NEXT: sbatch slurm/spine_train_array.sh"
            exit 0
        fi
    fi
    echo "  cache check: marker present but data dir/.npz files missing; re-running."
fi

# >>> NO host-side dangling-symlink cleanup. <<<
# Convert writes container-path symlinks (/data/hf_export/...). They look
# "broken" from the host but resolve inside the container via the bind
# mount. Cleaning them up was the reason resumes kept failing.

echo "================================================================"
echo " Stage A: convert + preprocess (SIMPLE, NFS-only)"
echo "   Dataset      : ${DS_DIR_NAME}"
echo "   Plans        : ${PLANS} (target ${GPU_MEMORY_TARGET_GB} GB)"
echo "   HF export    : ${HF_EXPORT_NFS}"
echo "   Splits file  : ${SPLITS_FILE_HOST}"
echo "   NFS dest     : ${NFS_PREP_DS}"
echo "   Splits mode  : $([[ ${SINGLE_FOLD_SPLITS} == 1 ]] && echo 'single-fold' || echo '5-fold CV')"
echo "   Workers      : ${N_PREPROCESS_THREADS} x ${BLAS_THREADS} BLAS threads = $((N_PREPROCESS_THREADS*BLAS_THREADS)) total"
echo "   Started      : $(date)"
echo "================================================================"

# -- Singularity setup -------------------------------------------------------
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

# -- Step 1: convert HF export -> nnU-Net raw (on NFS) -----------------------
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

# -- Step 2: plan + preprocess (reads + writes on NFS) -----------------------
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

# -- Step 3: copy splits + LSTV metadata into preprocessed/ ------------------
echo ""; echo "----- Step 3: copy metadata into preprocessed/ -----"
[[ -f "${NFS_RAW_DS}/splits_final.json" ]] && cp -f "${NFS_RAW_DS}/splits_final.json" "${NFS_PREP_DS}/" && echo "  copied splits_final.json"
[[ -f "${NFS_RAW_DS}/lstv_cases.json"   ]] && cp -f "${NFS_RAW_DS}/lstv_cases.json"   "${NFS_PREP_DS}/" && echo "  copied lstv_cases.json"

# -- Sanity check ------------------------------------------------------------
PREP_DATA_DIR=$(resolve_prep_data_dir || true)
if [[ -z "${PREP_DATA_DIR}" || ! -d "${PREP_DATA_DIR}" ]]; then
    echo "ERROR: could not locate preprocessed-data dir under ${NFS_PREP_DS}" >&2
    echo "  contents of ${NFS_PREP_DS}:" >&2
    ls -la "${NFS_PREP_DS}" >&2 || true
    exit 2
fi
N_NPZ_NFS=$({ find "${PREP_DATA_DIR}" -maxdepth 1 -name "*.npz" 2>/dev/null || true; } | wc -l)
NFS_SIZE=$(du -sh "${NFS_PREP_DS}" 2>/dev/null | awk '{print $1}')
if [[ "${N_NPZ_NFS}" -eq 0 ]]; then
    echo "ERROR: zero .npz files after preprocess in ${PREP_DATA_DIR}." >&2
    exit 2
fi

touch "${COMPLETE_MARKER}"

echo ""
echo "================================================================"
echo " Stage A COMPLETE   $(date)"
echo "   preprocess time : ${PREP_TIME}s"
echo "   data dir        : ${PREP_DATA_DIR}"
echo "   .npz files      : ${N_NPZ_NFS}"
echo "   preprocessed    : ${NFS_PREP_DS}  (${NFS_SIZE})"
echo ""
echo " NEXT:  sbatch slurm/spine_train_array.sh"
echo "================================================================"
