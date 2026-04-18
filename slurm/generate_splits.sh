#!/bin/bash
#SBATCH --job-name=spinesurg_splits
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/generate_splits_%j.out
#SBATCH --error=logs/generate_splits_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=go2432@wayne.edu
# =============================================================================
# generate_splits.sh -- Generate 5-fold CV splits (token level)
# =============================================================================
# Reads the NEWEST placed_manifest.json (the freshest source of truth for
# case inventory, match_type, and -- if available -- LSTV labels), and
# writes data/splits_5fold.json containing:
#   - a stratified test holdout (default 15%)
#   - 5 stratified K-fold CV partitions of the remaining pool
# stratified on (match_type x lstv_label) when LSTV info is available,
# else on match_type alone.
#
# Splits are at the PATIENT-TOKEN level, not case level, so a patient
# with multiple configs (fused + spine_only, etc.) always ends up in the
# same fold.
#
# Must run AFTER prepare_training.sh (needs placed_manifest.json).
# Must run BEFORE spine_prep.sh (nnU-Net convert requires splits file).
#
# Safe to rerun: idempotent. No-ops if the existing splits_5fold.json
# matches the current placed_manifest token set. Use OVERWRITE=1 to force.
#
# USAGE
# -----
#   # First run (or after placed_manifest was regenerated)
#   sbatch slurm/generate_splits.sh
#
#   # Force regenerate (changes seed, fold count, strata, etc.)
#   OVERWRITE=1 sbatch slurm/generate_splits.sh
#   N_FOLDS=3  KFOLD_SEED=7  TEST_FRACTION=0.2  OVERWRITE=1 \
#       sbatch slurm/generate_splits.sh
# =============================================================================

set -euo pipefail

# ----- Tunables via env -----------------------------------------------------
N_FOLDS="${N_FOLDS:-5}"
KFOLD_SEED="${KFOLD_SEED:-42}"
TEST_FRACTION="${TEST_FRACTION:-0.15}"
OVERWRITE="${OVERWRITE:-0}"

# ----- Paths ----------------------------------------------------------------
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
DATA_DIR="${PROJECT_ROOT}/data"
CONTAINER="${PROJECT_ROOT}/containers/spinesurg-ct.sif"

# NEWEST source of truth: placed_manifest.json produced by place_fused_masks.py
PLACED_MANIFEST="${DATA_DIR}/ctpelvic1k/masks/placed/placed_manifest.json"
# Optional LSTV enrichment source
TRAINING_MANIFEST="${DATA_DIR}/matched/colonog_training_manifest.json"
# Output
SPLITS_JSON="${DATA_DIR}/splits_5fold.json"

mkdir -p logs

# ----- Preflight ------------------------------------------------------------
[[ ! -f "${CONTAINER}" ]] && {
    echo "ERROR: container not found: ${CONTAINER}" >&2; exit 1; }
[[ ! -f "${PLACED_MANIFEST}" ]] && {
    echo "ERROR: placed_manifest.json not found: ${PLACED_MANIFEST}" >&2
    echo "       Run prepare_training.sh first." >&2
    exit 1; }

# Report placed_manifest mtime so we know we're using the newest
PM_MTIME=$(stat -c '%y' "${PLACED_MANIFEST}" 2>/dev/null || stat -f '%Sm' "${PLACED_MANIFEST}")
echo "  placed_manifest.json mtime: ${PM_MTIME}"

if [[ -f "${TRAINING_MANIFEST}" ]]; then
    TM_ARG="--training_manifest /data/matched/colonog_training_manifest.json"
    TM_MTIME=$(stat -c '%y' "${TRAINING_MANIFEST}" 2>/dev/null || stat -f '%Sm' "${TRAINING_MANIFEST}")
    echo "  training_manifest.json mtime: ${TM_MTIME}  (used for LSTV enrichment)"
else
    TM_ARG=""
    echo "  training_manifest.json not found (continuing without LSTV enrichment)"
fi

# ----- Singularity ---------------------------------------------------------
export SINGULARITY_TMPDIR="/tmp/${USER}_job_${SLURM_JOB_ID:-$$}"
export XDG_RUNTIME_DIR="${SINGULARITY_TMPDIR}/runtime"
export NXF_SINGULARITY_CACHEDIR="${HOME}/singularity_cache"
mkdir -p "${SINGULARITY_TMPDIR}" "${XDG_RUNTIME_DIR}" "${NXF_SINGULARITY_CACHEDIR}"
trap 'rm -rf "${SINGULARITY_TMPDIR}"' EXIT

export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
unset JAVA_HOME LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE
export NXF_SINGULARITY_HOME_MOUNT=true

PPATH="/workspace/scripts:/workspace/src:/workspace"
OVERWRITE_ARG=""; [[ "${OVERWRITE}" == "1" ]] && OVERWRITE_ARG="--overwrite"

echo "======================================================================"
echo " SpineSurg-CT  generate 5-fold CV splits"
echo " Job ID        : ${SLURM_JOB_ID:-local}"
echo " Node          : $(hostname)"
echo " Placed mfst   : ${PLACED_MANIFEST}"
echo " Training mfst : ${TRAINING_MANIFEST}  (if present)"
echo " Output        : ${SPLITS_JSON}"
echo " N_FOLDS       : ${N_FOLDS}"
echo " KFOLD_SEED    : ${KFOLD_SEED}"
echo " TEST_FRACTION : ${TEST_FRACTION}"
echo " OVERWRITE     : ${OVERWRITE}"
echo "======================================================================"

singularity exec \
    --env PYTHONPATH="${PPATH}" \
    --bind "${PROJECT_ROOT}:/workspace" \
    --bind "${DATA_DIR}:/data" \
    --pwd /workspace \
    "${CONTAINER}" \
    python scripts/generate_5fold_splits.py \
        --placed_manifest   /data/ctpelvic1k/masks/placed/placed_manifest.json \
        ${TM_ARG} \
        --out               /data/splits_5fold.json \
        --n_folds           "${N_FOLDS}" \
        --seed              "${KFOLD_SEED}" \
        --test_fraction     "${TEST_FRACTION}" \
        ${OVERWRITE_ARG}

EXIT_CODE=$?

echo "======================================================================"
if [[ ${EXIT_CODE} -eq 0 && -f "${SPLITS_JSON}" ]]; then
    echo " Splits COMPLETE"
    python3 - "${SPLITS_JSON}" << 'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"  source      : {d.get('source_manifest')}")
print(f"  schema      : v{d.get('schema_version')}   scheme: {d.get('strata_scheme')}")
print(f"  tokens      : total={d['n_tokens_total']}  test={d['n_tokens_test']}  trainval={d['n_tokens_trainval']}")
print(f"  n_folds     : {d['n_folds']}")
print(f"  strata (total) : {d.get('strata_counts_total', {})}")
print(f"  strata (test)  : {d.get('strata_counts_test', {})}")
for i, s in enumerate(d.get('strata_counts_per_fold_val', [])):
    print(f"  fold {i} val  : {s}")
PYEOF
    echo ""
    echo " Next: sbatch slurm/spine_prep.sh"
else
    echo " Splits FAILED (exit ${EXIT_CODE})"
fi
echo "======================================================================"
exit ${EXIT_CODE}
