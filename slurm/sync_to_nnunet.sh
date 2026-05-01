#!/usr/bin/env bash
#SBATCH --job-name=sync_to_nnunet
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/sync_to_nnunet_%j.out
#SBATCH --error=logs/sync_to_nnunet_%j.err
#SBATCH --mail-type=END,FAIL

# =============================================================================
# sync_to_nnunet.sh — Pull splits_5fold.json from CTSpinoPelvic1K into the
# nnU-Net trainer's expected paths.
#
# Mirrors the singularity setup pattern from slurm/spine_prep.sh:
#   - container-path binds (/workspace, /data/hf_export, /nnunet_nfs)
#   - SINGULARITYENV nnUNet_raw / preprocessed / results
#   - same TMPDIR + conda env hygiene
#
# Workflow:
#   Step 1 (already done by user separately):
#     sbatch ~/CTSpinoPelvic1K/slurm/splits_and_push.sh
#       Reads:  data/placed/placed_manifest_orientation_fixed.json
#       Writes: data/hf_export/splits_5fold.json (schema v5)
#
#   Step 2 (this script):
#     convert_hf_to_nnunet.py --regen_splits_only
#       Reads (inside container):
#         /data/hf_export/splits_5fold.json
#         /data/hf_export/manifest_{train,validation,test}.json
#       Writes (inside container, lands on NFS):
#         /nnunet_nfs/raw/Dataset802_SpineSurgCTFull/splits_final.json
#         /nnunet_nfs/raw/Dataset802_SpineSurgCTFull/lstv_cases.json
#     cp splits_final.json + lstv_cases.json -> nnunet/preprocessed/...
#
# Does NOT touch:
#   - imagesTr/labelsTr (--regen_splits_only does not rebuild the dataset)
#   - any training checkpoints
#   - the HuggingFace repo
#
# Assumes:
#   - splits_and_push.sh has already produced an up-to-date
#     <CTSPINO_ROOT>/data/hf_export/splits_5fold.json
#   - the HF manifests are co-located in that hf_export/ dir
#   - nnunet/raw/Dataset802_SpineSurgCTFull/dataset.json exists (built by
#     spine_prep.sh)
#
# Usage:
#   sbatch slurm/sync_to_nnunet.sh
#
# Dry run:
#   DRY_RUN=1 sbatch slurm/sync_to_nnunet.sh
#
# Override the splits source (defaults to ${HF_EXPORT_DIR}/splits_5fold.json
# then ${PROJECT_ROOT}/data/splits_5fold.json):
#   SPLITS_FILE=/some/other/splits.json sbatch slurm/sync_to_nnunet.sh
# =============================================================================

set -euo pipefail

# ── Singularity runtime dirs (mirrors spine_prep.sh) ────────────────────────
export SINGULARITY_TMPDIR="/tmp/${USER}_sync_${SLURM_JOB_ID:-$$}"
export XDG_RUNTIME_DIR="${SINGULARITY_TMPDIR}/runtime"
export NXF_SINGULARITY_CACHEDIR="${HOME}/singularity_cache"
mkdir -p "${SINGULARITY_TMPDIR}" "${XDG_RUNTIME_DIR}" "${NXF_SINGULARITY_CACHEDIR}"
trap 'rm -rf "${SINGULARITY_TMPDIR}" 2>/dev/null || true' EXIT

export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
unset JAVA_HOME LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE
export NXF_SINGULARITY_HOME_MOUNT=true

# ── Path resolution (mirrors spine_prep.sh logic) ────────────────────────────
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-${HOME}/spinesurg-ct-nnunet}"
NNUNET_ROOT="${NNUNET_ROOT:-${PROJECT_ROOT}}"
CTSPINO_ROOT="${CTSPINO_ROOT:-${HOME}/CTSpinoPelvic1K}"

DATASET_ID="${DATASET_ID:-802}"
DATASET_NAME="${DATASET_NAME:-SpineSurgCTFull}"
DS_DIR_NAME="Dataset$(printf '%03d' ${DATASET_ID})_${DATASET_NAME}"

# HF export discovery: legacy in-project location first, then ~/CTSpinoPelvic1K
if [[ -z "${HF_EXPORT_DIR:-}" ]]; then
    for cand in \
        "${PROJECT_ROOT}/data/hf_export" \
        "${CTSPINO_ROOT}/data/hf_export"; do
        if [[ -d "${cand}/ct" ]]; then
            HF_EXPORT_DIR="${cand}"
            break
        fi
    done
fi
HF_EXPORT_DIR="${HF_EXPORT_DIR:-${CTSPINO_ROOT}/data/hf_export}"

# Splits file: prefer co-located in hf_export/, fall back to legacy
if [[ -z "${SPLITS_FILE:-}" ]]; then
    for cand in \
        "${HF_EXPORT_DIR}/splits_5fold.json" \
        "${PROJECT_ROOT}/data/splits_5fold.json"; do
        if [[ -f "${cand}" ]]; then
            SPLITS_FILE="${cand}"
            break
        fi
    done
fi
SPLITS_FILE="${SPLITS_FILE:-${HF_EXPORT_DIR}/splits_5fold.json}"

NNUNET_NFS="${NNUNET_ROOT}/nnunet"
CONVERT_PY="${CONVERT_PY:-${NNUNET_ROOT}/tools/convert_hf_to_nnunet.py}"
CONTAINER="${CONTAINER:-${NNUNET_ROOT}/containers/spinesurg-ct.sif}"

NFS_RAW_DS="${NNUNET_NFS}/raw/${DS_DIR_NAME}"
NFS_PREP_DS="${NNUNET_NFS}/preprocessed/${DS_DIR_NAME}"

DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${NNUNET_ROOT}/logs" "${NNUNET_ROOT}/tmp"

echo "======================================================================"
echo " sync_to_nnunet — propagate splits_5fold.json -> nnU-Net trainer dirs"
echo "   Job ID            : ${SLURM_JOB_ID:-local}"
echo "   Node              : $(hostname)"
echo "   Started           : $(date)"
echo ""
echo "   nnU-Net root      : ${NNUNET_ROOT}"
echo "   CTSpinoPelvic1K   : ${CTSPINO_ROOT}"
echo "   HF export dir     : ${HF_EXPORT_DIR}"
echo "   splits source     : ${SPLITS_FILE}"
echo "   nnU-Net raw ds    : ${NFS_RAW_DS}"
echo "   nnU-Net prep ds   : ${NFS_PREP_DS}"
echo "   convert.py        : ${CONVERT_PY}"
echo "   container         : ${CONTAINER}"
echo "   DRY_RUN           : ${DRY_RUN}"
echo "======================================================================"

# ── Pre-flight ───────────────────────────────────────────────────────────────
PRE_FAIL=0
[[ ! -f "${CONTAINER}"   ]] && { echo "ERROR: container not found: ${CONTAINER}" >&2; PRE_FAIL=1; }
[[ ! -f "${CONVERT_PY}"  ]] && { echo "ERROR: convert.py not found: ${CONVERT_PY}" >&2; PRE_FAIL=1; }
[[ ! -f "${SPLITS_FILE}" ]] && {
    echo "ERROR: splits file not found: ${SPLITS_FILE}" >&2
    echo "  Searched: ${HF_EXPORT_DIR}/splits_5fold.json" >&2
    echo "            ${PROJECT_ROOT}/data/splits_5fold.json" >&2
    echo "  Override: SPLITS_FILE=/path/to/splits.json sbatch $0" >&2
    PRE_FAIL=1
}
for d in "${HF_EXPORT_DIR}/ct" "${HF_EXPORT_DIR}/labels" "${NFS_RAW_DS}" "${NFS_PREP_DS}"; do
    [[ ! -d "${d}" ]] && { echo "ERROR: required dir missing: ${d}" >&2; PRE_FAIL=1; }
done
for f in manifest_train.json manifest_validation.json manifest_test.json; do
    [[ ! -f "${HF_EXPORT_DIR}/${f}" ]] && {
        echo "ERROR: required HF manifest missing: ${HF_EXPORT_DIR}/${f}" >&2; PRE_FAIL=1; }
done
[[ ! -f "${NFS_RAW_DS}/dataset.json" ]] && {
    echo "ERROR: ${NFS_RAW_DS}/dataset.json missing" >&2
    echo "  Run spine_prep.sh first to build the dataset before syncing splits." >&2
    PRE_FAIL=1
}
[[ ${PRE_FAIL} -eq 1 ]] && exit 1

# Quick read of the splits file to surface schema + LSTV totals up-front
echo ""
python3 - "${SPLITS_FILE}" << 'PYEOF'
import json, sys
s = json.load(open(sys.argv[1]))
schema = s.get("schema_version", 0)
print(f"  splits_5fold.json schema: v{schema}")
if schema < 4:
    print(f"  WARN: schema v{schema} predates the LSTV-resolution fix.")
    print(f"        Re-run splits_and_push.sh with the v5 generator first.")
elif schema < 5:
    print(f"  NOTE: schema v{schema}; v5 adds 'ambiguous' as a 4th LSTV subtype.")
print(f"  total tokens : {s.get('n_tokens_total','?')}")
print(f"  test tokens  : {s.get('n_tokens_test','?')}")
print(f"  n_folds      : {s.get('n_folds','?')}")
print(f"  scheme       : {s.get('strata_scheme','?')}")
sub = s.get("lstv_subtype_counts", {}).get("total", {})
if sub:
    print(f"  LSTV totals  : {sub}")
n_amb = s.get("n_ambiguous", 0)
if n_amb:
    print(f"  ambiguous    : {n_amb} (Castellvi II/III boundary cases)")
PYEOF

# ── Singularity environment (matches spine_prep.sh) ─────────────────────────
export SINGULARITYENV_TMPDIR="/workspace/tmp"
export SINGULARITYENV_nnUNet_raw="/nnunet_nfs/raw"
export SINGULARITYENV_nnUNet_preprocessed="/nnunet_nfs/preprocessed"
export SINGULARITYENV_nnUNet_results="/nnunet_nfs/results"

# Cap BLAS threads (sync only reads/writes JSON; this is just hygiene)
export SINGULARITYENV_OMP_NUM_THREADS=2
export SINGULARITYENV_OPENBLAS_NUM_THREADS=2
export SINGULARITYENV_MKL_NUM_THREADS=2
export SINGULARITYENV_NUMEXPR_NUM_THREADS=2
export SINGULARITYENV_VECLIB_MAXIMUM_THREADS=2

SING_BINDS=(
    --bind "${NNUNET_ROOT}:/workspace"
    --bind "${HF_EXPORT_DIR}:/data/hf_export"
    --bind "${NNUNET_NFS}:/nnunet_nfs"
    --bind "${SPLITS_FILE}:/workspace/splits_5fold.json:ro"
    --pwd  /workspace
)

# ── Backup pre-existing splits files ────────────────────────────────────────
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="${NNUNET_ROOT}/.splits_backup_${TIMESTAMP}"
mkdir -p "${BACKUP_DIR}"

backup_if_exists() {
    local src="$1"
    if [[ -f "${src}" ]]; then
        local stem
        stem="$(echo ${src} | sed 's|/|__|g')"
        cp -p "${src}" "${BACKUP_DIR}/${stem}"
        echo "  backed up: ${src}"
    fi
}

echo ""
echo "Backing up pre-fix nnU-Net splits files to ${BACKUP_DIR} ..."
backup_if_exists "${NFS_RAW_DS}/splits_final.json"
backup_if_exists "${NFS_RAW_DS}/lstv_cases.json"
backup_if_exists "${NFS_PREP_DS}/splits_final.json"
backup_if_exists "${NFS_PREP_DS}/lstv_cases.json"

# ── Step 1: convert_hf_to_nnunet.py --regen_splits_only ─────────────────────
echo ""
echo "======================================================================"
echo " Step 1: convert_hf_to_nnunet.py --regen_splits_only"
echo "======================================================================"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "  DRY_RUN: would run convert_hf_to_nnunet.py --regen_splits_only"
    echo "    (paths inside container:"
    echo "       --hf_export_dir  /data/hf_export"
    echo "       --nnunet_raw_dir /nnunet_nfs/raw"
    echo "       --splits_file    /workspace/splits_5fold.json )"
else
    singularity exec "${SING_BINDS[@]}" "${CONTAINER}" \
        python3 tools/convert_hf_to_nnunet.py \
            --hf_export_dir  /data/hf_export \
            --nnunet_raw_dir /nnunet_nfs/raw \
            --dataset_id     "${DATASET_ID}" \
            --dataset_name   "${DATASET_NAME}" \
            --splits_file    /workspace/splits_5fold.json \
            --regen_splits_only
fi

# ── Step 2: copy from raw/ to preprocessed/ ─────────────────────────────────
echo ""
echo "======================================================================"
echo " Step 2: copy rebuilt splits into preprocessed dir"
echo "======================================================================"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "  DRY_RUN: would cp ${NFS_RAW_DS}/{splits_final,lstv_cases}.json -> ${NFS_PREP_DS}/"
else
    for f in splits_final.json lstv_cases.json; do
        if [[ -f "${NFS_RAW_DS}/${f}" ]]; then
            cp -f "${NFS_RAW_DS}/${f}" "${NFS_PREP_DS}/${f}"
            echo "  copied ${f} into preprocessed dir"
        else
            echo "  WARN: ${NFS_RAW_DS}/${f} not produced by Step 1." >&2
        fi
    done
fi

# ── Step 3: verify ──────────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo " Step 3: verify"
echo "======================================================================"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "  DRY_RUN: skipping verification."
    exit 0
fi

python3 - "${SPLITS_FILE}" "${NFS_PREP_DS}/splits_final.json" "${NFS_PREP_DS}/lstv_cases.json" << 'PYEOF'
import json, sys
from collections import Counter

splits_path, nnunet_splits_path, lstv_cases_path = sys.argv[1:4]

s = json.load(open(splits_path))
print(f"\n  splits_5fold.json (patient-token):")
print(f"    schema      : v{s['schema_version']}")
print(f"    total       : {s['n_tokens_total']}")
print(f"    test        : {s['n_tokens_test']}")
print(f"    n_folds     : {s['n_folds']}")
print(f"    scheme      : {s['strata_scheme']}")
sub_total = s.get('lstv_subtype_counts', {}).get('total', {})
print(f"    LSTV totals : {sub_total}")

nf = json.load(open(nnunet_splits_path))
print(f"\n  splits_final.json (nnU-Net case_id):")
print(f"    n_folds: {len(nf)}")
for i, fold in enumerate(nf):
    print(f"    fold {i}: train={len(fold['train'])} cases  val={len(fold['val'])} cases")

lc = json.load(open(lstv_cases_path))
print(f"\n  lstv_cases.json:")
print(f"    schema_version  : {lc['schema_version']}")
print(f"    n_cases         : {lc['n_cases']}")
print(f"    subtype_counts  : {lc['subtype_counts']}")

all_nnunet_cases = set()
for fold in nf:
    all_nnunet_cases.update(fold['train'])
    all_nnunet_cases.update(fold['val'])
print(f"\n  total unique case_ids in splits_final.json: {len(all_nnunet_cases)}")

print(f"\n  per-fold LSTV balance (val side, from lstv_cases.json):")
for i, fold in enumerate(nf):
    val = Counter(lc['case_ids'].get(c, 'normal') for c in fold['val'])
    print(f"    fold {i}: {dict(val)}")
PYEOF

echo ""
echo "======================================================================"
echo " sync_to_nnunet complete at $(date)"
echo "======================================================================"
echo ""
echo " Backup of pre-fix files: ${BACKUP_DIR}"
echo ""
echo " NEXT STEPS:"
echo ""
echo "   1. Confirm per-fold LSTV balance above looks reasonable"
echo "      (each fold val should have ~3 lumb and ~3-4 sacr; folds 0-3 may"
echo "      have 1 ambiguous each)."
echo ""
echo "   2. Kill in-flight training jobs:"
echo "        scancel -u \$USER --jobname=spine_kfold"
echo ""
echo "   3. New splits assign different patients to different folds, so"
echo "      resuming current ckpts contaminates val. Delete them:"
echo "        rm ${NNUNET_NFS}/results/${DS_DIR_NAME}/*/fold_*/checkpoint_*.pth"
echo "        rm -rf ~/.cache/spinesurg/*"
echo ""
echo "   4. Resubmit:"
echo "        sbatch slurm/spine_train_array.sh"
echo "======================================================================"
