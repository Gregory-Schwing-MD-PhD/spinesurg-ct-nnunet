#!/usr/bin/env bash
#SBATCH --job-name=convert_hf_to_nnunet
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/convert_hf_to_nnunet_%j.out
#SBATCH --error=logs/convert_hf_to_nnunet_%j.err
#SBATCH --mail-type=END,FAIL

# =============================================================================
# convert_hf_to_nnunet.sh — Run the new convert_hf_to_nnunet.py to produce
# nnU-Net dataset layout + lstv_cases.json (schema v3, 6-way LSTV) +
# splits_final.json (mirrored from splits_5fold.json schema v6).
#
# Replaces the old sync_to_nnunet.sh which used `--regen_splits_only` and
# the v2 lstv_cases.json schema. Both the CLI and the JSON schema changed
# with the Apr 2026 parser-bug fix.
#
# Reads:
#   ~/CTSpinoPelvic1K/data/hf_export/manifest_{train,validation,test}.json
#   ~/CTSpinoPelvic1K/data/hf_export/splits_5fold.json   (schema v6)
#   ~/CTSpinoPelvic1K/data/hf_export/ct/                 (CT NIfTIs)
#   ~/CTSpinoPelvic1K/data/hf_export/labels/             (label NIfTIs)
#
# Writes:
#   ${NNUNET_ROOT}/nnunet/raw/Dataset802_SpineSurgCTFull/
#     imagesTr/{caseID}_0000.nii.gz   (symlinks)
#     labelsTr/{caseID}.nii.gz        (symlinks)
#     imagesTs/{caseID}_0000.nii.gz   (symlinks; test set)
#     labelsTs/{caseID}.nii.gz        (symlinks; test set)
#     dataset.json
#     splits_final.json               (5-fold CV, nnU-Net format)
#     lstv_cases.json                 (schema v3, 6-way subtypes)
#
# Also copies splits_final.json + lstv_cases.json into the preprocessed
# directory so the trainer picks them up.
#
# Usage:
#   sbatch slurm/convert_hf_to_nnunet.sh
# =============================================================================

set -euo pipefail

# ── Singularity runtime dirs (mirrors sync_to_nnunet.sh) ────────────────────
export SINGULARITY_TMPDIR="/tmp/${USER}_convert_${SLURM_JOB_ID:-$$}"
export XDG_RUNTIME_DIR="${SINGULARITY_TMPDIR}/runtime"
mkdir -p "${SINGULARITY_TMPDIR}" "${XDG_RUNTIME_DIR}"
trap 'rm -rf "${SINGULARITY_TMPDIR}" 2>/dev/null || true' EXIT

export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
unset JAVA_HOME LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE

# ── Path resolution ─────────────────────────────────────────────────────────
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-${HOME}/spinesurg-ct-nnunet}"
NNUNET_ROOT="${NNUNET_ROOT:-${PROJECT_ROOT}}"
CTSPINO_ROOT="${CTSPINO_ROOT:-${HOME}/CTSpinoPelvic1K}"

DATASET_ID="${DATASET_ID:-802}"
DATASET_NAME="${DATASET_NAME:-SpineSurgCTFull}"
DS_DIR_NAME="Dataset$(printf '%03d' ${DATASET_ID})_${DATASET_NAME}"

HF_EXPORT_DIR="${HF_EXPORT_DIR:-${CTSPINO_ROOT}/data/hf_export}"
SPLITS_FILE="${SPLITS_FILE:-${HF_EXPORT_DIR}/splits_5fold.json}"

NNUNET_NFS="${NNUNET_ROOT}/nnunet"
CONVERT_PY="${CONVERT_PY:-${NNUNET_ROOT}/tools/convert_hf_to_nnunet.py}"
CONTAINER="${CONTAINER:-${NNUNET_ROOT}/containers/spinesurg-ct.sif}"

NFS_RAW_DS="${NNUNET_NFS}/raw/${DS_DIR_NAME}"
NFS_PREP_DS="${NNUNET_NFS}/preprocessed/${DS_DIR_NAME}"

mkdir -p "${NNUNET_ROOT}/logs"

echo "======================================================================"
echo " convert_hf_to_nnunet — produce nnU-Net dataset + v3 lstv_cases.json"
echo "   Job ID            : ${SLURM_JOB_ID:-local}"
echo "   Started           : $(date)"
echo "   nnU-Net root      : ${NNUNET_ROOT}"
echo "   HF export dir     : ${HF_EXPORT_DIR}"
echo "   splits source     : ${SPLITS_FILE}"
echo "   nnU-Net raw ds    : ${NFS_RAW_DS}"
echo "   nnU-Net prep ds   : ${NFS_PREP_DS}"
echo "   convert.py        : ${CONVERT_PY}"
echo "   container         : ${CONTAINER}"
echo "======================================================================"

# ── Pre-flight ───────────────────────────────────────────────────────────────
PRE_FAIL=0
[[ ! -f "${CONTAINER}"   ]] && { echo "ERROR: container not found: ${CONTAINER}" >&2; PRE_FAIL=1; }
[[ ! -f "${CONVERT_PY}"  ]] && { echo "ERROR: convert.py not found: ${CONVERT_PY}" >&2; PRE_FAIL=1; }
[[ ! -f "${SPLITS_FILE}" ]] && { echo "ERROR: splits_5fold.json not found: ${SPLITS_FILE}" >&2; PRE_FAIL=1; }
[[ ! -d "${HF_EXPORT_DIR}/ct"     ]] && { echo "ERROR: ${HF_EXPORT_DIR}/ct missing" >&2; PRE_FAIL=1; }
[[ ! -d "${HF_EXPORT_DIR}/labels" ]] && { echo "ERROR: ${HF_EXPORT_DIR}/labels missing" >&2; PRE_FAIL=1; }
for f in manifest_train.json manifest_validation.json manifest_test.json; do
    [[ ! -f "${HF_EXPORT_DIR}/${f}" ]] && {
        echo "ERROR: ${HF_EXPORT_DIR}/${f} missing" >&2; PRE_FAIL=1; }
done
[[ ${PRE_FAIL} -eq 1 ]] && exit 1

# Quick splits schema check
python3 - "${SPLITS_FILE}" << 'PYEOF'
import json, sys
s = json.load(open(sys.argv[1]))
schema = s.get("schema_version", 0)
print(f"  splits_5fold.json schema: v{schema}")
if schema < 6:
    print(f"  ERROR: schema v{schema} < 6. The 6-way subtype taxonomy ")
    print(f"         requires schema v6+. Re-run generate_5fold_splits.py.")
    sys.exit(1)
print(f"  n_patients     : {s.get('n_patients','?')}")
print(f"  n_folds        : {s.get('n_folds','?')}")
print(f"  Subtype counts:")
for st, n in s.get('subtype_counts', {}).items():
    print(f"    {st:<22} {n}")
PYEOF

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
echo "Backing up pre-fix splits files to ${BACKUP_DIR} ..."
backup_if_exists "${NFS_RAW_DS}/splits_final.json"
backup_if_exists "${NFS_RAW_DS}/lstv_cases.json"
backup_if_exists "${NFS_PREP_DS}/splits_final.json"
backup_if_exists "${NFS_PREP_DS}/lstv_cases.json"

# ── Singularity environment ─────────────────────────────────────────────────
export SINGULARITYENV_TMPDIR="/workspace/tmp"
export SINGULARITYENV_nnUNet_raw="/nnunet_nfs/raw"
export SINGULARITYENV_nnUNet_preprocessed="/nnunet_nfs/preprocessed"
export SINGULARITYENV_nnUNet_results="/nnunet_nfs/results"
export SINGULARITYENV_OMP_NUM_THREADS=2
export SINGULARITYENV_OPENBLAS_NUM_THREADS=2
export SINGULARITYENV_MKL_NUM_THREADS=2

SING_BINDS=(
    --bind "${NNUNET_ROOT}:/workspace"
    --bind "${HF_EXPORT_DIR}:/data/hf_export"
    --bind "${NNUNET_NFS}:/nnunet_nfs"
    --bind "${SPLITS_FILE}:/workspace/splits_5fold.json:ro"
    --pwd  /workspace
)

# ── Step 1: Run new convert_hf_to_nnunet.py ─────────────────────────────────
echo ""
echo "======================================================================"
echo " Step 1: convert_hf_to_nnunet.py (new CLI)"
echo "======================================================================"

singularity exec "${SING_BINDS[@]}" "${CONTAINER}" \
    python3 tools/convert_hf_to_nnunet.py \
        --hf_dir       /data/hf_export \
        --splits       /workspace/splits_5fold.json \
        --nnunet_raw   /nnunet_nfs/raw \
        --dataset_id   "${DATASET_ID}" \
        --dataset_name "${DATASET_NAME}" \
        --symlinks

# ── Step 2: copy from raw/ to preprocessed/ ─────────────────────────────────
echo ""
echo "======================================================================"
echo " Step 2: copy splits_final.json + lstv_cases.json to preprocessed/"
echo "======================================================================"

for f in splits_final.json lstv_cases.json; do
    if [[ -f "${NFS_RAW_DS}/${f}" ]]; then
        cp -f "${NFS_RAW_DS}/${f}" "${NFS_PREP_DS}/${f}"
        echo "  copied ${f}"
    else
        echo "  WARN: ${NFS_RAW_DS}/${f} not produced by Step 1." >&2
    fi
done

# ── Step 3: verify ──────────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo " Step 3: verify"
echo "======================================================================"

python3 - "${NFS_PREP_DS}/splits_final.json" "${NFS_PREP_DS}/lstv_cases.json" << 'PYEOF'
import json, sys
from collections import Counter

splits_path, lstv_cases_path = sys.argv[1:3]

nf = json.load(open(splits_path))
print(f"  splits_final.json:")
print(f"    n_folds: {len(nf)}")
for i, fold in enumerate(nf):
    print(f"    fold {i}: train={len(fold['train'])} cases  val={len(fold['val'])} cases")

lc = json.load(open(lstv_cases_path))
print(f"\n  lstv_cases.json:")
print(f"    schema_version       : {lc.get('schema_version', '?')}")
print(f"    splits_schema        : {lc.get('splits_schema', '?')}")
print(f"    n_train_cases        : {lc.get('n_train_cases', '?')}")
print(f"    n_test_cases         : {lc.get('n_test_cases', '?')}")
print(f"    oversample pool size : {len(lc.get('lstv_oversample_pool', []))}")
print(f"    subtype_counts:")
for st, n in lc.get('subtype_counts', {}).items():
    print(f"      {st:<22} {n}")

# Per-fold LSTV balance
case_to_sub = lc.get('case_to_subtype', {})
print(f"\n  per-fold LSTV balance (val side):")
for i, fold in enumerate(nf):
    val = Counter(case_to_sub.get(c, 'normal') for c in fold['val'])
    print(f"    fold {i}: {dict(val)}")
PYEOF

echo ""
echo "======================================================================"
echo " convert_hf_to_nnunet complete at $(date)"
echo "======================================================================"
echo ""
echo " Backup: ${BACKUP_DIR}"
echo ""
echo " NEXT STEPS:"
echo "   1. Confirm per-fold LSTV balance above looks reasonable."
echo "   2. Kill in-flight training jobs:"
echo "        scancel -u \$USER --jobname=spine_kfold"
echo "   3. Delete stale checkpoints (new splits assign different patients):"
echo "        rm ${NNUNET_NFS}/results/${DS_DIR_NAME}/*/fold_*/checkpoint_*.pth"
echo "        rm -f ${NNUNET_NFS}/preprocessed/${DS_DIR_NAME}/nnUNetPlans_3d_fullres/.preunpack_ok"
echo "        rm -f ${NNUNET_NFS}/preprocessed/${DS_DIR_NAME}/nnUNetPlans_3d_fullres/.scrub_ok"
echo "        rm -rf ~/.cache/spinesurg/*"
echo "   4. Resubmit:"
echo "        sbatch slurm/spine_train_array.sh"
echo "======================================================================"
