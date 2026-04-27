#!/bin/bash
#SBATCH --job-name=spine_convert
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/spine_convert_%j.out
#SBATCH --error=logs/spine_convert_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=go2432@wayne.edu
# =============================================================================
# SpineSurg-CT — convert HF export to nnU-Net v2 raw dataset (only)
# =============================================================================
# Calls tools/convert_hf_to_nnunet.py inside the spinesurg-ct.sif container
# to produce nnunet/raw/Dataset<id>_<name>/ from data/hf_export. No
# preprocessing, no training. This is the script to run when:
#
#   - The HF export was rebuilt (e.g. filenames changed) and the existing
#     symlinks under nnunet/raw/ now point at non-existent files.
#   - You want to bring in a new splits_5fold.json without re-preprocessing.
#   - You want a clean rebuild (WIPE=1) for OCD reasons.
#
# Why a separate script vs. spine_prep.sh
# ---------------------------------------
# spine_prep.sh runs convert + preprocess in one job because preprocessing
# wants the 48-CPU node + GPU it's already allocated. Convert by itself is
# a tiny disk-I/O job (writes ~1100 symlinks + ~800 NIfTI labels with
# ignore_label rewriting). 8 CPUs, 32 GB, no GPU.
#
# Default paths
# -------------
# Assumes the CTSpinoPelvic1K HF export lives at:
#   $HOME/CTSpinoPelvic1K/data/hf_export
# and produces output at:
#   $HOME/spinesurg-ct-nnunet/nnunet/raw/Dataset802_SpineSurgCTFull
# Override via env vars; see the "Config" block below.
#
# Usage
# -----
# Submit from the spinesurg-ct-nnunet repo root:
#
#   cd ~/spinesurg-ct-nnunet
#   sbatch slurm/spine_convert.sh
#
# Override paths or settings:
#
#   HF_EXPORT_NFS=$HOME/elsewhere/hf_export sbatch slurm/spine_convert.sh
#   WIPE=1 sbatch slurm/spine_convert.sh                  # rebuild from scratch
#   SINGLE_FOLD_SPLITS=1 sbatch slurm/spine_convert.sh    # legacy single-fold splits
#
# Sanity checks before submission
# -------------------------------
# Make sure the splits file actually exists:
#   ls $HOME/CTSpinoPelvic1K/data/hf_export/splits_5fold.json
# If not, point SPLITS_FILE_HOST at the right file or generate it first.
# =============================================================================

set -euo pipefail

# ── Config ───────────────────────────────────────────────────────────────────
DATASET_ID="${DATASET_ID:-802}"
DATASET_NAME="${DATASET_NAME:-SpineSurgCTFull}"
SINGLE_FOLD_SPLITS="${SINGLE_FOLD_SPLITS:-0}"
WIPE="${WIPE:-0}"   # set to 1 to rm -rf the existing dataset before rebuild

# Path layout. PROJECT_ROOT is the spinesurg-ct-nnunet repo (where this
# script and tools/convert_hf_to_nnunet.py live). HF_EXPORT_NFS points
# at the dataset directory in the CTSpinoPelvic1K sibling repo by
# default. Override either with env vars.
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-${HOME}/spinesurg-ct-nnunet}"
HF_EXPORT_NFS="${HF_EXPORT_NFS:-${HOME}/CTSpinoPelvic1K/data/hf_export}"
NNUNET_NFS="${NNUNET_NFS:-${PROJECT_ROOT}/nnunet}"
SPLITS_FILE_HOST="${SPLITS_FILE_HOST:-${HF_EXPORT_NFS}/splits_5fold.json}"
CONTAINER="${CONTAINER:-${PROJECT_ROOT}/containers/spinesurg-ct.sif}"

DS_DIR_NAME="Dataset$(printf '%03d' ${DATASET_ID})_${DATASET_NAME}"
NFS_RAW_DS="${NNUNET_NFS}/raw/${DS_DIR_NAME}"

# Filter pass-throughs (rare for normal convert)
INCLUDE_TRAIN_MATCH_TYPES="${INCLUDE_TRAIN_MATCH_TYPES:-}"
TEST_MATCH_TYPES="${TEST_MATCH_TYPES:-}"

# Optional ablation switches — uncommon but pass-through anyway
TRAIN_FUSED_ONLY_FLAG=""
[[ "${TRAIN_FUSED_ONLY:-0}" == "1" ]] && TRAIN_FUSED_ONLY_FLAG="--train_fused_only"
TEST_ALL_CONFIGS_FLAG=""
[[ "${TEST_ALL_CONFIGS:-0}" == "1" ]] && TEST_ALL_CONFIGS_FLAG="--test_all_configs"

# ── Singularity runtime dirs ─────────────────────────────────────────────────
export SINGULARITY_TMPDIR="/tmp/${USER}_job_${SLURM_JOB_ID:-$$}"
export XDG_RUNTIME_DIR="${SINGULARITY_TMPDIR}/runtime"
export NXF_SINGULARITY_CACHEDIR="${HOME}/singularity_cache"
mkdir -p "${SINGULARITY_TMPDIR}" "${XDG_RUNTIME_DIR}" "${NXF_SINGULARITY_CACHEDIR}"
trap 'rm -rf "${SINGULARITY_TMPDIR}"' EXIT

export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
unset JAVA_HOME; which singularity
export NXF_SINGULARITY_HOME_MOUNT=true
unset LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE

mkdir -p "${NNUNET_NFS}/raw" "${PROJECT_ROOT}/logs"

# ── Preflight ────────────────────────────────────────────────────────────────
[[ ! -f "${CONTAINER}" ]] && { echo "ERROR: container not found: ${CONTAINER}" >&2; exit 1; }
[[ ! -d "${HF_EXPORT_NFS}/ct" ]] && { echo "ERROR: ${HF_EXPORT_NFS}/ct missing" >&2; exit 1; }
[[ ! -f "${PROJECT_ROOT}/tools/convert_hf_to_nnunet.py" ]] && {
    echo "ERROR: ${PROJECT_ROOT}/tools/convert_hf_to_nnunet.py missing" >&2; exit 1; }

if [[ "${SINGLE_FOLD_SPLITS}" != "1" ]]; then
    [[ ! -f "${SPLITS_FILE_HOST}" ]] && {
        echo "ERROR: splits file not found: ${SPLITS_FILE_HOST}" >&2
        echo "       set SPLITS_FILE_HOST or pass SINGLE_FOLD_SPLITS=1" >&2
        exit 1; }
fi

# ── Optional wipe ────────────────────────────────────────────────────────────
if [[ "${WIPE}" == "1" && -d "${NFS_RAW_DS}" ]]; then
    echo "  WIPE=1 -> rm -rf ${NFS_RAW_DS}"
    rm -rf "${NFS_RAW_DS}"
fi

# ── Banner ───────────────────────────────────────────────────────────────────
echo "================================================================"
echo " spine_convert : HF export -> nnU-Net raw"
echo "   Job             : ${SLURM_JOB_ID:-local}"
echo "   Node            : $(hostname)"
echo "   PROJECT_ROOT    : ${PROJECT_ROOT}"
echo "   HF_EXPORT_NFS   : ${HF_EXPORT_NFS}"
echo "   NFS_RAW_DS      : ${NFS_RAW_DS}"
echo "   CONTAINER       : ${CONTAINER}"
echo "   Splits mode     : $([[ ${SINGLE_FOLD_SPLITS} == 1 ]] && echo 'single-fold' || echo "5-fold (${SPLITS_FILE_HOST})")"
echo "   WIPE            : ${WIPE}"
echo "   Started         : $(date)"
echo "================================================================"

# ── Singularity binds ────────────────────────────────────────────────────────
SING_BINDS=(
    --bind "${PROJECT_ROOT}:/workspace"
    --bind "${HF_EXPORT_NFS}:/data/hf_export"
    --bind "${NNUNET_NFS}:/nnunet_nfs"
    --pwd  /workspace
)
if [[ "${SINGLE_FOLD_SPLITS}" != "1" ]]; then
    SING_BINDS+=( --bind "${SPLITS_FILE_HOST}:/workspace/splits_5fold.json:ro" )
fi

# ── Convert ──────────────────────────────────────────────────────────────────
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
[[ -n "${TRAIN_FUSED_ONLY_FLAG}" ]] && CONVERT_ARGS+=( ${TRAIN_FUSED_ONLY_FLAG} )
[[ -n "${TEST_ALL_CONFIGS_FLAG}" ]] && CONVERT_ARGS+=( ${TEST_ALL_CONFIGS_FLAG} )

t0=$(date +%s)
singularity exec "${SING_BINDS[@]}" "${CONTAINER}" \
    python tools/convert_hf_to_nnunet.py "${CONVERT_ARGS[@]}"
elapsed=$(($(date +%s) - t0))

# ── Verify ───────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " Convert complete in ${elapsed}s"
echo "================================================================"

if [[ -d "${NFS_RAW_DS}" ]]; then
    n_imtr=$(ls "${NFS_RAW_DS}/imagesTr/" 2>/dev/null | wc -l)
    n_lbtr=$(ls "${NFS_RAW_DS}/labelsTr/" 2>/dev/null | wc -l)
    n_imts=$(ls "${NFS_RAW_DS}/imagesTs/" 2>/dev/null | wc -l)
    n_lbts=$(ls "${NFS_RAW_DS}/labelsTs/" 2>/dev/null | wc -l)
    n_broken=$(find "${NFS_RAW_DS}/imagesTr" "${NFS_RAW_DS}/imagesTs" \
                 -type l ! -exec test -e {} \; -print 2>/dev/null | wc -l)
    echo "  imagesTr : ${n_imtr}"
    echo "  labelsTr : ${n_lbtr}"
    echo "  imagesTs : ${n_imts}"
    echo "  labelsTs : ${n_lbts}"
    echo "  broken symlinks : ${n_broken}"
    if [[ "${n_broken}" -gt 0 ]]; then
        echo "  WARN: ${n_broken} symlinks point at nonexistent targets." >&2
        echo "        Showing first 3:" >&2
        find "${NFS_RAW_DS}/imagesTr" "${NFS_RAW_DS}/imagesTs" \
            -type l ! -exec test -e {} \; -print 2>/dev/null | head -3 >&2
    fi
    echo ""
    echo "  Sample symlinks (verify they point at the new filenames):"
    ls -la "${NFS_RAW_DS}/imagesTr/" 2>/dev/null | head -5
fi

echo ""
echo " NEXT:"
echo "   sbatch slurm/spine_prep.sh    # preprocess (Stage A continues)"
echo " Or for a smoke-test single fold:"
echo "   FOLD=0 sbatch slurm/spine_train_fold.sh"
echo "================================================================"
