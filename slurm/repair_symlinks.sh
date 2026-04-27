#!/bin/bash
#SBATCH --job-name=spine_repair_symlinks
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/repair_symlinks_%j.out
#SBATCH --error=logs/repair_symlinks_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=go2432@wayne.edu
# =============================================================================
# SpineSurg-CT -- repair broken symlinks in nnunet/raw/Dataset.../
# =============================================================================
# What this does, and why it exists
# ---------------------------------
# An earlier version of tools/convert_hf_to_nnunet.py used Path.resolve()
# to compute symlink targets, which -- when running inside a container
# with /data/hf_export bind-mounted from the host -- embedded the
# container-internal path ("/data/hf_export/...") into every symlink.
# Those symlinks are unreadable from the host, which breaks downstream
# tools: rsync, plan_experiment's fingerprint extraction, the patch
# visualizer, everything.
#
# The patched convert script (v2+) fixes this two ways:
#   1. New symlinks use absolute-as-given paths (no .resolve()).
#   2. When the dataset already exists, the idempotency branch samples
#      a few symlinks, and if they're broken in the recognizable
#      legacy pattern, rewrites them all in-place to point at the host
#      HF export directory -- taking seconds instead of the 90+ minute
#      full rebuild that would otherwise be required.
#
# This wrapper invokes ONLY the convert step with the mirrored bind
# needed for self-heal to work. It does NOT:
#   - rsync to local scratch
#   - run plan_and_preprocess
#   - touch any of the preprocessing state
#
# Use this before plan_sizes.sh / visualize_patch_sizes.py to get
# host-readable symlinks without committing to a GPU_MEMORY_TARGET_GB.
#
# Safe to rerun. Once symlinks are clean, reruns are a fast no-op.
#
# USAGE
# -----
#   sbatch slurm/repair_symlinks.sh
# =============================================================================

set -euo pipefail

# ── Config (mirrors spine_prep.sh so paths stay consistent) ────────────────
DATASET_ID="${DATASET_ID:-802}"
DATASET_NAME="${DATASET_NAME:-SpineSurgCTFull}"
SINGLE_FOLD_SPLITS="${SINGLE_FOLD_SPLITS:-0}"

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
HF_EXPORT_NFS="${PROJECT_ROOT}/data/hf_export"
SPLITS_FILE_HOST="${PROJECT_ROOT}/data/splits_5fold.json"
NNUNET_NFS="${PROJECT_ROOT}/nnunet"
CONTAINER="${PROJECT_ROOT}/containers/spinesurg-ct.sif"

DS_DIR_NAME="Dataset$(printf '%03d' ${DATASET_ID})_${DATASET_NAME}"
NFS_RAW_DS="${NNUNET_NFS}/raw/${DS_DIR_NAME}"

# Resolve HF_EXPORT_NFS to a canonical path so the mirrored bind works
# even if the user symlinked data/hf_export to elsewhere on NFS.
HF_EXPORT_REAL=$(readlink -f "${HF_EXPORT_NFS}")

mkdir -p "${PROJECT_ROOT}/logs"

# ── Singularity runtime ────────────────────────────────────────────────────
export SINGULARITY_TMPDIR="/tmp/${USER}_job_${SLURM_JOB_ID:-$$}"
export XDG_RUNTIME_DIR="${SINGULARITY_TMPDIR}/runtime"
export NXF_SINGULARITY_CACHEDIR="${HOME}/singularity_cache"
mkdir -p "${SINGULARITY_TMPDIR}" "${XDG_RUNTIME_DIR}" "${NXF_SINGULARITY_CACHEDIR}"
trap 'rm -rf "${SINGULARITY_TMPDIR}"' EXIT

export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
unset JAVA_HOME LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE
export NXF_SINGULARITY_HOME_MOUNT=true

# ── Preflight ──────────────────────────────────────────────────────────────
[[ ! -f "${CONTAINER}"                                ]] && { echo "ERROR: container not found: ${CONTAINER}" >&2; exit 1; }
[[ ! -d "${HF_EXPORT_REAL}/ct"                        ]] && { echo "ERROR: ${HF_EXPORT_REAL}/ct missing" >&2; exit 1; }
[[ ! -f "${PROJECT_ROOT}/tools/convert_hf_to_nnunet.py" ]] && { echo "ERROR: tools/convert_hf_to_nnunet.py missing" >&2; exit 1; }
[[ ! -d "${NFS_RAW_DS}"                               ]] && {
    echo "ERROR: ${NFS_RAW_DS} does not exist." >&2
    echo "       Nothing to repair -- run spine_prep.sh first to build the dataset." >&2
    exit 1
}
[[ ! -f "${NFS_RAW_DS}/dataset.json"                  ]] && {
    echo "ERROR: ${NFS_RAW_DS}/dataset.json missing." >&2
    echo "       Dataset is half-built; delete ${NFS_RAW_DS} and run spine_prep.sh." >&2
    exit 1
}

if [[ "${SINGLE_FOLD_SPLITS}" != "1" ]]; then
    [[ ! -f "${SPLITS_FILE_HOST}" ]] && {
        echo "ERROR: ${SPLITS_FILE_HOST} missing." >&2; exit 1; }
fi

# ── Pre-check: are symlinks actually broken? ──────────────────────────────
# If they already resolve, this job is a no-op and we can exit fast.
echo "================================================================"
echo " SpineSurg-CT  symlink repair (convert-only, no preprocess)"
echo "   Dataset      : ${DS_DIR_NAME}"
echo "   HF export    : ${HF_EXPORT_REAL}"
echo "   Raw dataset  : ${NFS_RAW_DS}"
echo "================================================================"

echo ""
echo "----- Pre-check: sampling 5 symlinks to test -----"
N_BROKEN=0
N_SAMPLED=0
for p in "${NFS_RAW_DS}/imagesTr/"*_0000.nii.gz; do
    [[ -e "${p}" ]] || continue                   # glob no-match: empty dir
    N_SAMPLED=$((N_SAMPLED + 1))
    if [[ ! -e "${p}" || ! -r "${p}" ]] 2>/dev/null; then : ; fi
    # -L follows the symlink; file is readable only if target resolves
    if ! [[ -L "${p}" && -e "$(readlink -f "${p}" 2>/dev/null)" ]]; then
        N_BROKEN=$((N_BROKEN + 1))
        if [[ "${N_BROKEN}" -le 2 ]]; then
            echo "    broken: ${p##*/} -> $(readlink "${p}")"
        fi
    fi
    [[ "${N_SAMPLED}" -ge 5 ]] && break
done
echo "  sampled=${N_SAMPLED}  broken=${N_BROKEN}"

if [[ "${N_BROKEN}" -eq 0 && "${N_SAMPLED}" -gt 0 ]]; then
    echo ""
    echo "  All sampled symlinks resolve -- nothing to repair."
    echo "  (If you still suspect issues, run `ls -lL ${NFS_RAW_DS}/imagesTr | head` manually.)"
    exit 0
fi

# ── Bind configuration ─────────────────────────────────────────────────────
# Key difference from spine_prep.sh:
#   - Bind HF_EXPORT_REAL at its HOST PATH inside the container, not at
#     /data/hf_export. This way --hf_export_dir in the convert invocation
#     is a host path, so the self-heal writes host-readable symlinks.
#   - No /nnunet_local bind; no preprocessing happens here.
#   - No --nv flag; no GPU needed.
SING_BINDS=(
    --bind "${PROJECT_ROOT}:/workspace"
    --bind "${HF_EXPORT_REAL}:${HF_EXPORT_REAL}"   # mirrored: host path = container path
    --bind "${NNUNET_NFS}:/nnunet_nfs"
    --pwd  /workspace
)
if [[ "${SINGLE_FOLD_SPLITS}" != "1" ]]; then
    SING_BINDS+=( --bind "${SPLITS_FILE_HOST}:/workspace/splits_5fold.json:ro" )
fi

CONVERT_ARGS=(
    --hf_export_dir  "${HF_EXPORT_REAL}"           # host path -> self-heal writes host paths
    --nnunet_raw_dir /nnunet_nfs/raw
    --dataset_id     ${DATASET_ID}
    --dataset_name   ${DATASET_NAME}
)
if [[ "${SINGLE_FOLD_SPLITS}" == "1" ]]; then
    CONVERT_ARGS+=( --single_fold_splits )
else
    CONVERT_ARGS+=( --splits_file /workspace/splits_5fold.json )
fi

# ── Run convert (self-heal branch) ─────────────────────────────────────────
echo ""
echo "----- Invoking convert_hf_to_nnunet.py (self-heal path) -----"
echo "  Since dataset.json already exists, the script will detect broken"
echo "  symlinks and rewrite them to point at ${HF_EXPORT_REAL}."
echo ""

singularity exec "${SING_BINDS[@]}" "${CONTAINER}" \
    python tools/convert_hf_to_nnunet.py "${CONVERT_ARGS[@]}"
EXIT_CODE=$?

# ── Post-check ─────────────────────────────────────────────────────────────
echo ""
echo "----- Post-check: re-sampling 5 symlinks -----"
N_BROKEN_AFTER=0
N_SAMPLED_AFTER=0
for p in "${NFS_RAW_DS}/imagesTr/"*_0000.nii.gz; do
    [[ -e "${p}" ]] || continue
    N_SAMPLED_AFTER=$((N_SAMPLED_AFTER + 1))
    if ! [[ -L "${p}" && -e "$(readlink -f "${p}" 2>/dev/null)" ]]; then
        N_BROKEN_AFTER=$((N_BROKEN_AFTER + 1))
    fi
    [[ "${N_SAMPLED_AFTER}" -ge 5 ]] && break
done
echo "  sampled=${N_SAMPLED_AFTER}  still_broken=${N_BROKEN_AFTER}"

echo "================================================================"
if [[ "${EXIT_CODE}" -eq 0 && "${N_BROKEN_AFTER}" -eq 0 ]]; then
    echo " Symlink repair COMPLETE"
    echo ""
    echo " NEXT:"
    echo "   1. sbatch slurm/plan_sizes.sh        (plan at 60/80/100/120/140 GB)"
    echo "   2. python tools/visualize_patch_sizes.py ...   (pick a target)"
    echo "   3. GPU_MEMORY_TARGET_GB=<your_pick> sbatch slurm/spine_prep.sh"
else
    echo " Symlink repair FAILED (convert exit=${EXIT_CODE}, still_broken=${N_BROKEN_AFTER})"
    echo " Inspect ${NFS_RAW_DS}/imagesTr/ manually."
fi
echo "================================================================"
exit ${EXIT_CODE}
