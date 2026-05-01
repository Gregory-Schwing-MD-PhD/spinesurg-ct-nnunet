#!/usr/bin/env bash
#SBATCH --job-name=fix_splits
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/fix_splits_%j.out
#SBATCH --error=logs/fix_splits_%j.err
#SBATCH --mail-type=END,FAIL

# =============================================================================
# fix_splits.sh — Regenerate splits with correct LSTV resolution.
#
# Does NOT rerun the upstream placement / export pipeline.
#
# Pipeline:
#   Step 1. CTSpinoPelvic1K/scripts/generate_5fold_splits.py (v4)
#           Reads:  data/placed/placed_manifest_orientation_fixed.json
#           Writes: data/hf_export/splits_5fold.json (schema v4)
#
#   Step 2. spinesurg-ct-nnunet/tools/convert_hf_to_nnunet.py --regen_splits_only
#           Reads:  the v4 splits_5fold.json + manifest_{train,val,test}.json
#           Writes: nnunet/raw/Dataset802_SpineSurgCTFull/splits_final.json
#                   nnunet/raw/Dataset802_SpineSurgCTFull/lstv_cases.json
#
#   Step 3. cp the rebuilt files into nnunet/preprocessed/...
#           (this is the location the trainer actually reads)
#
#   Step 4. verify
#
# Why this works without rebuilding the dataset:
#   convert_hf_to_nnunet.py's --regen_splits_only branch only rewrites
#   splits_final.json + lstv_cases.json based on the existing
#   imagesTr/labelsTr layout. It does not touch the symlinks or label
#   .nii.gz files. So the dataset on disk is preserved.
#
# Prerequisites (already on disk; this job does NOT regenerate them):
#   - data/placed/placed_manifest_orientation_fixed.json
#   - data/hf_export/manifest_{train,validation,test}.json
#   - nnunet/raw/Dataset802_SpineSurgCTFull/  (built by convert_hf_to_nnunet.py)
#
# Usage:
#   sbatch slurm/fix_splits.sh
#
# Dry run:
#   DRY_RUN=1 sbatch slurm/fix_splits.sh
# =============================================================================

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${PROJECT_ROOT}"

if [[ -f configs/default.env ]]; then
    source configs/default.env
fi

# ── Path resolution ──────────────────────────────────────────────────────────

CTSPINO_ROOT="${CTSPINO_ROOT:-${HOME}/CTSpinoPelvic1K}"
NNUNET_ROOT="${NNUNET_ROOT:-${HOME}/spinesurg-ct-nnunet}"

PLACED_MANIFEST="${PLACED_MANIFEST:-${CTSPINO_ROOT}/data/placed/placed_manifest_orientation_fixed.json}"
HF_EXPORT_DIR="${HF_EXPORT_DIR:-${CTSPINO_ROOT}/data/hf_export}"
SPLITS_OUT="${SPLITS_OUT:-${HF_EXPORT_DIR}/splits_5fold.json}"

NNUNET_RAW_BASE="${NNUNET_RAW_BASE:-${NNUNET_ROOT}/nnunet/raw}"
NNUNET_PREP="${NNUNET_PREP:-${NNUNET_ROOT}/nnunet/preprocessed/Dataset802_SpineSurgCTFull}"
DATASET_ID="${DATASET_ID:-802}"
DATASET_NAME="${DATASET_NAME:-SpineSurgCTFull}"
NNUNET_RAW_DS="${NNUNET_RAW_BASE}/Dataset$(printf '%03d' ${DATASET_ID})_${DATASET_NAME}"

GEN_SPLITS_PY="${GEN_SPLITS_PY:-${CTSPINO_ROOT}/scripts/generate_5fold_splits.py}"
CONVERT_PY="${CONVERT_PY:-${NNUNET_ROOT}/tools/convert_hf_to_nnunet.py}"

# Optional: container for convert_hf_to_nnunet (it needs nibabel).
# If empty or missing, runs natively.
CONTAINER="${CONTAINER:-${NNUNET_ROOT}/containers/spinesurg-ct.sif}"

DRY_RUN="${DRY_RUN:-0}"

mkdir -p "${CTSPINO_ROOT}/logs"

echo "======================================================================"
echo " fix_splits — regenerate splits with correct LSTV resolution"
echo "   Job ID                 : ${SLURM_JOB_ID:-local}"
echo "   Node                   : $(hostname)"
echo "   Started                : $(date)"
echo ""
echo "   placed manifest        : ${PLACED_MANIFEST}"
echo "   HF export dir          : ${HF_EXPORT_DIR}"
echo "   splits output          : ${SPLITS_OUT}"
echo "   nnU-Net raw dataset    : ${NNUNET_RAW_DS}"
echo "   nnU-Net preprocessed   : ${NNUNET_PREP}"
echo "   gen_splits.py          : ${GEN_SPLITS_PY}"
echo "   convert.py             : ${CONVERT_PY}"
if [[ -n "${CONTAINER}" && -f "${CONTAINER}" ]]; then
    echo "   container              : ${CONTAINER}"
else
    echo "   container              : <native python3>"
fi
echo "   DRY_RUN                : ${DRY_RUN}"
echo "======================================================================"

# ── Pre-flight ───────────────────────────────────────────────────────────────
PRE_FAIL=0
for f in "${PLACED_MANIFEST}" "${GEN_SPLITS_PY}" "${CONVERT_PY}"; do
    if [[ ! -f "${f}" ]]; then
        echo "ERROR: required file missing: ${f}" >&2
        PRE_FAIL=1
    fi
done
for d in "${HF_EXPORT_DIR}" "${NNUNET_RAW_DS}" "${NNUNET_PREP}"; do
    if [[ ! -d "${d}" ]]; then
        echo "ERROR: required directory missing: ${d}" >&2
        PRE_FAIL=1
    fi
done
for f in manifest_train.json manifest_validation.json manifest_test.json; do
    if [[ ! -f "${HF_EXPORT_DIR}/${f}" ]]; then
        echo "ERROR: required HF manifest missing: ${HF_EXPORT_DIR}/${f}" >&2
        PRE_FAIL=1
    fi
done
[[ ${PRE_FAIL} -eq 1 ]] && exit 1

# ── Run helper (container-aware) ────────────────────────────────────────────
_run_python() {
    local script="$1"
    shift
    if [[ -n "${CONTAINER}" && -f "${CONTAINER}" ]]; then
        singularity exec \
            --bind "${CTSPINO_ROOT}:${CTSPINO_ROOT}" \
            --bind "${NNUNET_ROOT}:${NNUNET_ROOT}" \
            "${CONTAINER}" \
            python3 "${script}" "$@"
    else
        python3 "${script}" "$@"
    fi
}

# ── Backup existing splits files ────────────────────────────────────────────
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP_DIR="${CTSPINO_ROOT}/data/hf_export/.splits_backup_${TIMESTAMP}"
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
echo "Backing up existing splits files to ${BACKUP_DIR} ..."
backup_if_exists "${SPLITS_OUT}"
backup_if_exists "${NNUNET_RAW_DS}/splits_final.json"
backup_if_exists "${NNUNET_RAW_DS}/lstv_cases.json"
backup_if_exists "${NNUNET_PREP}/splits_final.json"
backup_if_exists "${NNUNET_PREP}/lstv_cases.json"

# ── Step 1: regenerate patient-token splits (v4) ────────────────────────────
echo ""
echo "======================================================================"
echo " Step 1: generate_5fold_splits.py (schema v4)"
echo "======================================================================"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "  DRY_RUN: would run generate_5fold_splits.py ..."
else
    _run_python "${GEN_SPLITS_PY}" \
        --placed_manifest "${PLACED_MANIFEST}" \
        --out             "${SPLITS_OUT}" \
        --n_folds         5 \
        --test_fraction   0.15 \
        --seed            42 \
        --overwrite
fi

# ── Step 2: convert_hf_to_nnunet.py --regen_splits_only ─────────────────────
echo ""
echo "======================================================================"
echo " Step 2: convert_hf_to_nnunet.py --regen_splits_only"
echo "======================================================================"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "  DRY_RUN: would run convert_hf_to_nnunet.py --regen_splits_only ..."
else
    _run_python "${CONVERT_PY}" \
        --hf_export_dir  "${HF_EXPORT_DIR}" \
        --nnunet_raw_dir "${NNUNET_RAW_BASE}" \
        --dataset_id     "${DATASET_ID}" \
        --dataset_name   "${DATASET_NAME}" \
        --splits_file    "${SPLITS_OUT}" \
        --regen_splits_only
fi

# ── Step 3: copy rebuilt files into nnunet/preprocessed/... ─────────────────
# convert_hf_to_nnunet.py writes into nnunet/raw/.../ but the trainer reads
# from nnunet/preprocessed/.../  — that is the canonical training-time
# location. The convert script's own "NEXT STEPS" output instructs the
# user to do this cp manually. We do it here so the splits are consistent
# in both places.
echo ""
echo "======================================================================"
echo " Step 3: copy rebuilt splits into preprocessed dir"
echo "======================================================================"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "  DRY_RUN: would cp ${NNUNET_RAW_DS}/{splits_final,lstv_cases}.json -> ${NNUNET_PREP}/"
else
    for f in splits_final.json lstv_cases.json; do
        if [[ -f "${NNUNET_RAW_DS}/${f}" ]]; then
            cp -p "${NNUNET_RAW_DS}/${f}" "${NNUNET_PREP}/${f}"
            echo "  copied ${f} into preprocessed dir"
        else
            echo "  WARN: ${NNUNET_RAW_DS}/${f} not produced by Step 2." >&2
        fi
    done
fi

# ── Step 4: verify ──────────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo " Step 4: verify outputs"
echo "======================================================================"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "  DRY_RUN: skipping verification."
    exit 0
fi

# Inline verifier — uses native python3 since it only needs json/Counter.
python3 - "${SPLITS_OUT}" "${NNUNET_PREP}/splits_final.json" "${NNUNET_PREP}/lstv_cases.json" << 'PYEOF'
import json, sys
from collections import Counter

splits_path, nnunet_splits_path, lstv_cases_path = sys.argv[1:4]

s = json.load(open(splits_path))
print(f"\n  splits_5fold.json (patient-token level):")
print(f"    schema      : v{s['schema_version']}")
print(f"    total       : {s['n_tokens_total']}")
print(f"    test        : {s['n_tokens_test']}")
print(f"    n_folds     : {s['n_folds']}")
print(f"    scheme      : {s['strata_scheme']}")
sub_total = s.get('lstv_subtype_counts', {}).get('total', {})
print(f"    LSTV totals : {sub_total}")
print(f"    per-fold val LSTV:")
for i, c in enumerate(s.get('lstv_subtype_counts', {}).get('folds_val', [])):
    print(f"      fold {i}: {c}")

nf = json.load(open(nnunet_splits_path))
print(f"\n  splits_final.json (nnU-Net case_id level):")
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

print(f"\n  per-fold LSTV balance (from lstv_cases.json + splits_final.json):")
for i, fold in enumerate(nf):
    val_subs = Counter()
    for cid in fold['val']:
        sub = lc['case_ids'].get(cid, 'normal')
        val_subs[sub] += 1
    train_subs = Counter()
    for cid in fold['train']:
        sub = lc['case_ids'].get(cid, 'normal')
        train_subs[sub] += 1
    print(f"    fold {i}  val:   {dict(val_subs)}")
    print(f"            train: {dict(train_subs)}")
PYEOF

echo ""
echo "======================================================================"
echo " fix_splits complete at $(date)"
echo "======================================================================"
echo ""
echo " NEXT STEPS:"
echo ""
echo "   1. Inspect the per-fold LSTV balance printed above."
echo "      Each fold's val should have ~3 lumb and ~4 sacr."
echo ""
echo "   2. Kill in-flight training jobs:"
echo "        scancel -u \$USER --jobname=spine_kfold"
echo ""
echo "   3. The new splits assign different patients to different folds, so"
echo "      resuming from current ckpts contaminates val. Nuke them:"
echo "        rm ${NNUNET_ROOT}/nnunet/results/Dataset${DATASET_ID}_${DATASET_NAME}/*/fold_*/checkpoint_*.pth"
echo "        rm -rf ~/.cache/spinesurg/*"
echo ""
echo "   4. Resubmit training:"
echo "        sbatch slurm/spine_train_array.sh"
echo ""
echo " BACKUP of pre-fix files: ${BACKUP_DIR}"
echo "======================================================================"
