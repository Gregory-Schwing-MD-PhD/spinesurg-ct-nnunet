#!/bin/bash
#SBATCH --job-name=spine_eval_803_full
#SBATCH -q primary
#SBATCH --exclude=msa1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=logs/spine_eval_803_full_%j.out
#SBATCH --error=logs/spine_eval_803_full_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=go2432@wayne.edu
# =============================================================================
# Stage 2 / 2: FULL EVAL (PARALLEL) — Dataset803 / 500ep / checkpoint_best
# =============================================================================
# CPU-only. PARALLEL version — runs 6 evals concurrently (5 folds + 1
# ensemble) inside a single SLURM job.
#
# CPU budget: 48 cpus-per-task / 6 parallel evals = 8 workers each.
# Memory:     128G / 6 = ~21G per eval.
# Wall time:  ~2-3 min (vs ~25 min sequential, ~5 min at 4 workers/eval).
#
# If SLURM doesn't have a 48-cpu node available quickly, override:
#   sbatch --cpus-per-task=24 --export=ALL,WORKERS_PER_EVAL=4 ...
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
DATASET_ID=803
DATASET_NAME="SpineSurgCTFullMerged"
CONFIG="3d_fullres"
TRAINER="${TRAINER:-nnUNetTrainerWandB_500ep_LSTVOversample}"
GPU_MEMORY_TARGET_GB=100
PLANS="nnUNetResEncUNetPlans_${GPU_MEMORY_TARGET_GB}G"
CHECKPOINT="${CHECKPOINT:-checkpoint_best.pth}"

WORKERS_PER_EVAL="${WORKERS_PER_EVAL:-8}"
SKIP_PERFOLD="${SKIP_PERFOLD:-0}"

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-${HOME}/spinesurg-ct-nnunet}"
NNUNET_NFS="${PROJECT_ROOT}/nnunet"
CONTAINER="${PROJECT_ROOT}/containers/spinesurg-ct.sif"
EVAL_SCRIPT_HOST="${PROJECT_ROOT}/tools/eval_full.py"
AGG_SCRIPT_HOST="${PROJECT_ROOT}/tools/aggregate_folds.py"

DS_DIR_NAME="Dataset$(printf '%03d' ${DATASET_ID})_${DATASET_NAME}"
CKPT_TAG=$(echo "${CHECKPOINT}" | sed -e 's/checkpoint_//' -e 's/\.pth$//')

# -- Paths -------------------------------------------------------------------
PRED_PERFOLD_NFS="${NNUNET_NFS}/predictions/${DS_DIR_NAME}_${CONFIG}_${TRAINER}_${PLANS}_PERFOLD_${CKPT_TAG}"
PRED_ENSEMBLE_NFS="${NNUNET_NFS}/predictions/${DS_DIR_NAME}_${CONFIG}_${TRAINER}_${PLANS}_ENSEMBLE_${CKPT_TAG}"
LABELS_NFS="${NNUNET_NFS}/raw/${DS_DIR_NAME}/labelsTs"
LSTV_JSON_HOST="${NNUNET_NFS}/raw/${DS_DIR_NAME}/lstv_cases.json"

EVAL_BASE_NFS="${PRED_ENSEMBLE_NFS}/eval_full"
mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/tmp" "${EVAL_BASE_NFS}"

PRED_PERFOLD_CONT="/nnunet_nfs/predictions/${DS_DIR_NAME}_${CONFIG}_${TRAINER}_${PLANS}_PERFOLD_${CKPT_TAG}"
PRED_ENSEMBLE_CONT="/nnunet_nfs/predictions/${DS_DIR_NAME}_${CONFIG}_${TRAINER}_${PLANS}_ENSEMBLE_${CKPT_TAG}"
LABELS_CONT="/nnunet_nfs/raw/${DS_DIR_NAME}/labelsTs"
LSTV_JSON_CONT="/nnunet_nfs/raw/${DS_DIR_NAME}/lstv_cases.json"
EVAL_BASE_CONT="${PRED_ENSEMBLE_CONT}/eval_full"

# -- Preflight ---------------------------------------------------------------
[[ ! -f "${CONTAINER}"        ]] && { echo "ERROR: container not found: ${CONTAINER}" >&2; exit 1; }
[[ ! -f "${EVAL_SCRIPT_HOST}" ]] && { echo "ERROR: ${EVAL_SCRIPT_HOST} not found." >&2; exit 1; }
[[ ! -f "${AGG_SCRIPT_HOST}"  ]] && { echo "ERROR: ${AGG_SCRIPT_HOST} not found." >&2; exit 1; }
[[ ! -d "${LABELS_NFS}"       ]] && { echo "ERROR: labelsTs not found: ${LABELS_NFS}" >&2; exit 1; }
[[ ! -f "${LSTV_JSON_HOST}"   ]] && {
    echo "WARN: lstv_cases.json not found at ${LSTV_JSON_HOST}" >&2
}

echo "================================================================"
echo " SPINESURG-CT FULL EVAL (PARALLEL)  Dataset${DATASET_ID} / ${CHECKPOINT}"
echo "================================================================"
echo "  trainer            : ${TRAINER}"
echo "  workers per eval   : ${WORKERS_PER_EVAL}  (6 parallel evals × ${WORKERS_PER_EVAL} = $((WORKERS_PER_EVAL * 6)) cpus)"
echo "  per-fold pred dir  : ${PRED_PERFOLD_NFS}"
echo "  ensemble pred dir  : ${PRED_ENSEMBLE_NFS}"
echo "  labels dir         : ${LABELS_NFS}"
echo "  lstv cases         : ${LSTV_JSON_HOST}"
echo "  eval base out      : ${EVAL_BASE_NFS}"
echo "  skip per-fold      : ${SKIP_PERFOLD}"
echo "  job id             : ${SLURM_JOB_ID:-<not in slurm>}"
echo "  start time         : $(date -Iseconds)"
echo "================================================================"

# -- Sanity counts -----------------------------------------------------------
N_LABELS=$({ find "${LABELS_NFS}" -maxdepth 1 -name "*.nii.gz" 2>/dev/null || true; } | wc -l)
echo ""
echo "labelsTs count: ${N_LABELS}"
if [[ -d "${PRED_ENSEMBLE_NFS}" ]]; then
    N_ENS=$({ find "${PRED_ENSEMBLE_NFS}" -maxdepth 1 -name "*.nii.gz" 2>/dev/null || true; } | wc -l)
    echo "ensemble predictions: ${N_ENS}"
else
    echo "ensemble predictions: MISSING"
fi
for F in 0 1 2 3 4; do
    PD="${PRED_PERFOLD_NFS}/fold_${F}"
    if [[ -d "${PD}" ]]; then
        NF=$({ find "${PD}" -maxdepth 1 -name "*.nii.gz" 2>/dev/null || true; } | wc -l)
        echo "fold ${F} predictions: ${NF}"
    else
        echo "fold ${F} predictions: MISSING"
    fi
done

# -- Singularity binds -------------------------------------------------------
export SINGULARITYENV_TMPDIR="/workspace/tmp"
export SINGULARITYENV_nnUNet_raw="/nnunet_nfs/raw"
export SINGULARITYENV_nnUNet_preprocessed="/nnunet_nfs/preprocessed"
export SINGULARITYENV_nnUNet_results="/nnunet_nfs/results"

SING_BINDS=(
    --bind "${PROJECT_ROOT}:/workspace"
    --bind "${NNUNET_NFS}:/nnunet_nfs"
    --pwd  /workspace
)

# -- Background eval launcher ------------------------------------------------
run_eval_bg () {
    local TASK_LABEL="$1"
    local PRED_CONT="$2"
    local OUT_HOST="$3"
    local OUT_CONT="$4"

    mkdir -p "${OUT_HOST}"
    rm -f "${OUT_HOST}/.exit_code" "${OUT_HOST}/.started" "${OUT_HOST}/.done"
    touch "${OUT_HOST}/.started"

    {
        echo "================================================================"
        echo " EVAL ${TASK_LABEL}  $(date -Iseconds)"
        echo "   predictions : ${PRED_CONT}"
        echo "   output      : ${OUT_CONT}"
        echo "   workers     : ${WORKERS_PER_EVAL}"
        echo "================================================================"
        set +e
        stdbuf -oL -eL singularity exec "${SING_BINDS[@]}" "${CONTAINER}" \
            python -u /workspace/tools/eval_full.py \
                --predictions_dir "${PRED_CONT}" \
                --labels_dir      "${LABELS_CONT}" \
                --lstv_cases_json "${LSTV_JSON_CONT}" \
                --output_dir      "${OUT_CONT}" \
                --workers         ${WORKERS_PER_EVAL}
        local RC=$?
        echo "${RC}" > "${OUT_HOST}/.exit_code"
        touch "${OUT_HOST}/.done"
        echo ""
        echo "[$(date -Iseconds)] EVAL ${TASK_LABEL} exited with code ${RC}"
    } > "${OUT_HOST}/eval.log" 2>&1 &

    local PID=$!
    EVAL_PIDS+=("${PID}")
    EVAL_LABELS+=("${TASK_LABEL}")
    EVAL_OUT_HOSTS+=("${OUT_HOST}")
    echo "  ${TASK_LABEL}: PID ${PID}, log ${OUT_HOST}/eval.log"
}

# -- Build the list of evals ------------------------------------------------
EVAL_PIDS=()
EVAL_LABELS=()
EVAL_OUT_HOSTS=()
FOLD_OUT_DIRS_HOST=()
FOLD_OUT_DIRS_CONT=()

echo ""
echo "----------------------------------------------------------------"
echo " Launching parallel evals..."
echo "----------------------------------------------------------------"

if [[ "${SKIP_PERFOLD}" != "1" ]]; then
    for F in 0 1 2 3 4; do
        PD_HOST="${PRED_PERFOLD_NFS}/fold_${F}"
        if [[ ! -d "${PD_HOST}" ]]; then
            echo "  WARN: ${PD_HOST} missing; skipping fold ${F} eval."
            continue
        fi
        PD_CONT="${PRED_PERFOLD_CONT}/fold_${F}"
        OD_HOST="${EVAL_BASE_NFS}/fold_${F}"
        OD_CONT="${EVAL_BASE_CONT}/fold_${F}"
        FOLD_OUT_DIRS_HOST+=("${OD_HOST}")
        FOLD_OUT_DIRS_CONT+=("${OD_CONT}")
        run_eval_bg "fold_${F}" "${PD_CONT}" "${OD_HOST}" "${OD_CONT}"
    done
else
    echo "  SKIP_PERFOLD=1; not launching per-fold evals."
fi

ENS_OUT_HOST=""
ENS_OUT_CONT=""
if [[ -d "${PRED_ENSEMBLE_NFS}" ]]; then
    ENS_OUT_HOST="${EVAL_BASE_NFS}/ensemble"
    ENS_OUT_CONT="${EVAL_BASE_CONT}/ensemble"
    run_eval_bg "ensemble" "${PRED_ENSEMBLE_CONT}" "${ENS_OUT_HOST}" "${ENS_OUT_CONT}"
fi

N_EVALS=${#EVAL_PIDS[@]}
if [[ ${N_EVALS} -eq 0 ]]; then
    echo "ERROR: no evals to run." >&2
    exit 1
fi

echo ""
echo "Launched ${N_EVALS} evals in parallel."
echo ""

# -- Background progress monitor (30s polling) ------------------------------
(
    while true; do
        sleep 30
        N_DONE=0
        for OD in "${EVAL_OUT_HOSTS[@]}"; do
            [[ -f "${OD}/.done" ]] && N_DONE=$((N_DONE + 1))
        done
        if [[ ${N_DONE} -ge ${N_EVALS} ]]; then break; fi
        echo "[$(date +%H:%M:%S)] progress: ${N_DONE}/${N_EVALS} evals done"
    done
) &
MONITOR_PID=$!

# -- Wait for all eval PIDs -------------------------------------------------
echo "Waiting for all evals to finish (monitor PID ${MONITOR_PID})..."
for PID in "${EVAL_PIDS[@]}"; do
    wait "${PID}" || true
done
kill "${MONITOR_PID}" 2>/dev/null || true
wait "${MONITOR_PID}" 2>/dev/null || true

# -- Per-eval results summary -----------------------------------------------
echo ""
echo "================================================================"
echo " PER-EVAL RESULTS"
echo "================================================================"
ALL_OK=1
for i in "${!EVAL_LABELS[@]}"; do
    LABEL="${EVAL_LABELS[$i]}"
    OD="${EVAL_OUT_HOSTS[$i]}"
    RC=$(cat "${OD}/.exit_code" 2>/dev/null || echo "MISSING")
    if [[ "${RC}" == "0" ]]; then
        echo "  OK    ${LABEL}  (log: ${OD}/eval.log)"
    else
        echo "  FAIL  ${LABEL}  exit=${RC}  (log: ${OD}/eval.log)"
        ALL_OK=0
    fi
done
echo "================================================================"

if [[ ${ALL_OK} -eq 0 ]]; then
    echo ""
    echo "FAILED EVAL TAILS:"
    for i in "${!EVAL_LABELS[@]}"; do
        OD="${EVAL_OUT_HOSTS[$i]}"
        RC=$(cat "${OD}/.exit_code" 2>/dev/null || echo "MISSING")
        if [[ "${RC}" != "0" ]]; then
            echo ""
            echo "----- ${EVAL_LABELS[$i]} (exit=${RC}) -----"
            tail -n 20 "${OD}/eval.log" 2>/dev/null || echo "(log not readable)"
        fi
    done
fi

# -- Cross-fold aggregation -------------------------------------------------
XF_OUT_HOST="${EVAL_BASE_NFS}/cross_fold"
XF_OUT_CONT="${EVAL_BASE_CONT}/cross_fold"

VALID_FOLD_OUT_HOST=()
VALID_FOLD_OUT_CONT=()
for i in "${!FOLD_OUT_DIRS_HOST[@]}"; do
    if [[ -f "${FOLD_OUT_DIRS_HOST[$i]}/headline.json" ]]; then
        VALID_FOLD_OUT_HOST+=("${FOLD_OUT_DIRS_HOST[$i]}")
        VALID_FOLD_OUT_CONT+=("${FOLD_OUT_DIRS_CONT[$i]}")
    fi
done

if [[ ${#VALID_FOLD_OUT_CONT[@]} -ge 2 ]]; then
    echo ""
    echo "----------------------------------------------------------------"
    echo " CROSS-FOLD AGGREGATION  $(date -Iseconds)"
    echo "   n folds  : ${#VALID_FOLD_OUT_CONT[@]}  (out of ${#FOLD_OUT_DIRS_CONT[@]} attempted)"
    echo "   output   : ${XF_OUT_HOST}"
    echo "----------------------------------------------------------------"
    ENS_FLAG=()
    if [[ -n "${ENS_OUT_HOST}" && -f "${ENS_OUT_HOST}/headline.json" ]]; then
        ENS_FLAG=(--ensemble_dir "${ENS_OUT_CONT}")
    fi
    stdbuf -oL -eL singularity exec "${SING_BINDS[@]}" "${CONTAINER}" \
        python -u /workspace/tools/aggregate_folds.py \
            --fold_dirs "${VALID_FOLD_OUT_CONT[@]}" \
            "${ENS_FLAG[@]}" \
            --output_dir "${XF_OUT_CONT}"
else
    echo ""
    echo "Skipping cross-fold aggregation — only ${#VALID_FOLD_OUT_CONT[@]} fold(s) produced valid output."
fi

# -- Final summary -----------------------------------------------------------
echo ""
echo "================================================================"
echo " FULL EVAL COMPLETE  ${CHECKPOINT}  $(date -Iseconds)"
echo "================================================================"
echo "  Per-fold outputs:"
for D in "${FOLD_OUT_DIRS_HOST[@]}"; do
    if [[ -f "${D}/subgroup_summary.md" ]]; then
        echo "    ${D}/subgroup_summary.md"
    else
        echo "    ${D}/  (no subgroup_summary.md — eval may have failed)"
    fi
done
if [[ -n "${ENS_OUT_HOST}" ]]; then
    echo "  Ensemble output:"
    if [[ -f "${ENS_OUT_HOST}/subgroup_summary.md" ]]; then
        echo "    ${ENS_OUT_HOST}/subgroup_summary.md"
    else
        echo "    ${ENS_OUT_HOST}/  (no subgroup_summary.md — eval may have failed)"
    fi
fi
if [[ -d "${XF_OUT_HOST}" ]]; then
    echo "  Cross-fold (PAPER-READY):"
    echo "    ${XF_OUT_HOST}/cross_fold_summary.md"
    echo "    ${XF_OUT_HOST}/cross_fold_summary.csv"
    echo "    ${XF_OUT_HOST}/cross_fold_headline.json"
    echo "    ${XF_OUT_HOST}/cross_fold_confusion.json"
fi
echo "================================================================"
echo ""
echo "Quick view:"
if [[ -f "${XF_OUT_HOST}/cross_fold_summary.md" ]]; then
    echo "   cat ${XF_OUT_HOST}/cross_fold_summary.md"
fi
if [[ -n "${ENS_OUT_HOST}" && -f "${ENS_OUT_HOST}/subgroup_summary.md" ]]; then
    echo "   cat ${ENS_OUT_HOST}/subgroup_summary.md"
fi

if [[ ${ALL_OK} -eq 0 ]]; then
    echo ""
    echo "WARN: at least one eval failed. See per-eval logs above." >&2
    exit 1
fi

exit 0
