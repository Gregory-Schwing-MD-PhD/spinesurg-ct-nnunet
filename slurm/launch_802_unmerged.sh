#!/bin/bash
# =============================================================================
# launch_802_unmerged.sh  —  one-shot launcher for the UNMERGED baseline
# =============================================================================
# Submit-wrapper (run on the LOGIN NODE, not via sbatch) that fires the whole
# Dataset802 (10-class, L5/L6 SEPARATE) NeurIPS-rebuttal run in one go:
#
#   1. spine_prep.sh         (convert 802 with --no_remap + preprocess)
#   2. spine_train_array.sh  (5-fold CV, folds 0-4) — chained to prep via a
#                             SLURM  --dependency=afterok  so it only starts if
#                             prep succeeds.
#
# The trainer's SPINESURG_LABEL_SCHEME auto-resolves to `unmerged` from
# DATASET_ID=802 (see spine_train_array.sh), so val/dice/L6 and the
# val/headline/L6_GT_on_lumb/as_L5_frac collapse metric get logged.
#
# Usage
# -----
#   bash slurm/launch_802_unmerged.sh
#
#   # if the SUBMITTED source isn't at the default data/hf_export, point prep
#   # at it (e.g. the v2 export):
#   HF_EXPORT_DIR=$HOME/CTSpinoPelvic1K/data/hf_export_v2 \
#     bash slurm/launch_802_unmerged.sh
#
#   # 802 already preprocessed? skip prep, just launch the array:
#   SKIP_PREP=1 bash slurm/launch_802_unmerged.sh
#
#   # single fold instead of all five (e.g. smoke test):
#   TRAIN_ARRAY=0 bash slurm/launch_802_unmerged.sh
#
# Any spine_train_array.sh knob (TRAINER, LSTV_OVERSAMPLE_FRAC, PLANS, ...)
# set in the environment is forwarded via --export=ALL.
# =============================================================================
set -euo pipefail

# Resolve repo root from this script's location (slurm/ -> repo root) and cd
# there so sbatch sets SLURM_SUBMIT_DIR = repo root (both scripts derive
# PROJECT_ROOT from it).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# -- Unmerged 802 config (override any of these on the command line) ---------
export DATASET_ID="${DATASET_ID:-802}"
export DATASET_NAME="${DATASET_NAME:-SpineSurgCTFull}"
export LEGACY_NO_REMAP="${LEGACY_NO_REMAP:-1}"                       # prep: keep L5/L6 separate
export SPINESURG_LABEL_SCHEME="${SPINESURG_LABEL_SCHEME:-unmerged}"  # trainer constants/metrics
export SPINESURG_LSTV_BIAS_ENABLED="${SPINESURG_LSTV_BIAS_ENABLED:-1}"  # A': L6 oversampling on
# HF_EXPORT_DIR is passed through if set (else spine_prep.sh auto-discovers).

SKIP_PREP="${SKIP_PREP:-0}"
TRAIN_ARRAY="${TRAIN_ARRAY:-0-4}"   # all five folds by default

echo "================================================================"
echo " launch_802_unmerged  —  Dataset802 (L5/L6 SEPARATE, 10-class)"
echo "   repo            : ${REPO_ROOT}"
echo "   DATASET_ID      : ${DATASET_ID}   DATASET_NAME: ${DATASET_NAME}"
echo "   trainer scheme  : SPINESURG_LABEL_SCHEME=${SPINESURG_LABEL_SCHEME}"
echo "   prep remap      : LEGACY_NO_REMAP=${LEGACY_NO_REMAP} (--no_remap => L6 kept)"
echo "   L6 oversample   : SPINESURG_LSTV_BIAS_ENABLED=${SPINESURG_LSTV_BIAS_ENABLED}"
echo "   HF_EXPORT_DIR   : ${HF_EXPORT_DIR:-<spine_prep.sh default discovery>}"
echo "   train folds     : ${TRAIN_ARRAY}"
echo "   skip prep       : ${SKIP_PREP}"
echo "================================================================"

# Guardrail: warn (don't block) if the default source dir looks absent AND no
# override was given — this is the usual "dataset not staged" foot-gun.
if [[ "${SKIP_PREP}" != "1" && -z "${HF_EXPORT_DIR:-}" ]]; then
    for cand in "${REPO_ROOT}/data/hf_export" "${HOME}/CTSpinoPelvic1K/data/hf_export"; do
        [[ -d "${cand}/ct" ]] && { FOUND_SRC="${cand}"; break; }
    done
    if [[ -z "${FOUND_SRC:-}" ]]; then
        echo "WARN: no hf_export/ct found at the default locations and HF_EXPORT_DIR is unset." >&2
        echo "      prep will likely fail. Point it at the SUBMITTED source, e.g.:" >&2
        echo "        HF_EXPORT_DIR=\$HOME/CTSpinoPelvic1K/data/hf_export_v2 bash slurm/launch_802_unmerged.sh" >&2
        echo "      Continuing anyway (spine_prep.sh will do the authoritative check)." >&2
    else
        echo "  source found   : ${FOUND_SRC}"
    fi
fi

if [[ "${SKIP_PREP}" == "1" ]]; then
    TRAIN_JOB=$(sbatch --parsable --export=ALL \
        --array="${TRAIN_ARRAY}" slurm/spine_train_array.sh)
    echo ""
    echo "  submitted train array : ${TRAIN_JOB}  (folds ${TRAIN_ARRAY})"
else
    PREP_JOB=$(sbatch --parsable --export=ALL -J prep_802 slurm/spine_prep.sh)
    echo ""
    echo "  submitted prep        : ${PREP_JOB}"
    TRAIN_JOB=$(sbatch --parsable --export=ALL \
        --dependency=afterok:"${PREP_JOB}" \
        --array="${TRAIN_ARRAY}" slurm/spine_train_array.sh)
    echo "  submitted train array : ${TRAIN_JOB}  (folds ${TRAIN_ARRAY}, starts after prep ${PREP_JOB} OK)"
fi

echo ""
echo "  monitor :  squeue --me"
echo "  logs    :  logs/spine_prep_*.out   logs/spine_kfold_f*_*.out"
echo "  cancel  :  scancel ${PREP_JOB:-}${PREP_JOB:+ }${TRAIN_JOB}"
echo ""
echo "  watch for (unmerged evidence):"
echo "    val/dice/L6                         -> should stay ~0"
echo "    val/lstv_subgroup/lumb/dice/L6      -> ~0 on lumbarization"
echo "    val/headline/L6_GT_on_lumb/as_L5_frac -> high (GT-L6 predicted as L5 = the collapse)"
echo "================================================================"
