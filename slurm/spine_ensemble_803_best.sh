#!/bin/bash
#SBATCH --job-name=spine_ensemble_803
#SBATCH -q primary
#SBATCH --exclude=msa1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=logs/spine_ensemble_803_%j.out
#SBATCH --error=logs/spine_ensemble_803_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=go2432@wayne.edu
# =============================================================================
# Stage 1.5 / 2: ENSEMBLE COMBINE — Dataset803 / 500ep / checkpoint_best
# =============================================================================
# CPU-only. Combines per-fold softmax predictions (fold_0/...fold_4/) into
# a single ensembled prediction set via nnUNetv2_ensemble.
#
# Reads:  predictions/${BASE}_PERFOLD_best/fold_{0..4}/*.npz (+ .pkl)
# Writes: predictions/${BASE}_ENSEMBLE_best/*.nii.gz
#
# Logging
# -------
# A background watcher polls the output dir every 30s and prints:
#   * .nii.gz count vs. expected total
#   * delta from last poll (cases completed in last 30s)
#   * average and instantaneous rate
#   * ETA based on current rate
# nnUNetv2_ensemble's own tqdm progress passes through stdbuf -oL -eL.
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

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-${HOME}/spinesurg-ct-nnunet}"
NNUNET_NFS="${PROJECT_ROOT}/nnunet"
CONTAINER="${PROJECT_ROOT}/containers/spinesurg-ct.sif"

DS_DIR_NAME="Dataset$(printf '%03d' ${DATASET_ID})_${DATASET_NAME}"
CKPT_TAG=$(echo "${CHECKPOINT}" | sed -e 's/checkpoint_//' -e 's/\.pth$//')

PRED_PERFOLD_NFS="${NNUNET_NFS}/predictions/${DS_DIR_NAME}_${CONFIG}_${TRAINER}_${PLANS}_PERFOLD_${CKPT_TAG}"
PRED_ENSEMBLE_NFS="${NNUNET_NFS}/predictions/${DS_DIR_NAME}_${CONFIG}_${TRAINER}_${PLANS}_ENSEMBLE_${CKPT_TAG}"

PRED_PERFOLD_CONT="/nnunet_nfs/predictions/${DS_DIR_NAME}_${CONFIG}_${TRAINER}_${PLANS}_PERFOLD_${CKPT_TAG}"
PRED_ENSEMBLE_CONT="/nnunet_nfs/predictions/${DS_DIR_NAME}_${CONFIG}_${TRAINER}_${PLANS}_ENSEMBLE_${CKPT_TAG}"

mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/tmp" "${PRED_ENSEMBLE_NFS}"

# -- Preflight: each fold dir must have predictions --------------------------
echo "================================================================"
echo " SPINESURG-CT ENSEMBLE COMBINE  Dataset${DATASET_ID} / ${CHECKPOINT}"
echo "================================================================"
echo "  trainer           : ${TRAINER}"
echo "  per-fold input    : ${PRED_PERFOLD_NFS}"
echo "  ensemble output   : ${PRED_ENSEMBLE_NFS}"
echo "  job id            : ${SLURM_JOB_ID:-<not in slurm>}"
echo "  node              : ${SLURMD_NODENAME:-$(hostname)}"
echo "  start time        : $(date -Iseconds)"
echo "================================================================"

ALL_OK=1
N_EXPECTED=0
for F in 0 1 2 3 4; do
    PD="${PRED_PERFOLD_NFS}/fold_${F}"
    if [[ ! -d "${PD}" ]]; then
        echo "ERROR: fold ${F} dir missing: ${PD}" >&2
        ALL_OK=0
        continue
    fi
    N_NII=$({ find "${PD}" -maxdepth 1 -name "*.nii.gz" 2>/dev/null || true; } | wc -l)
    N_NPZ=$({ find "${PD}" -maxdepth 1 -name "*.npz"   2>/dev/null || true; } | wc -l)
    N_PKL=$({ find "${PD}" -maxdepth 1 -name "*.pkl"   2>/dev/null || true; } | wc -l)
    echo "  fold ${F}: ${N_NII} .nii.gz, ${N_NPZ} .npz, ${N_PKL} .pkl"
    if [[ ${N_NPZ} -eq 0 ]]; then
        echo "  ERROR: fold ${F} has 0 .npz files (ensemble needs softmax probabilities)" >&2
        ALL_OK=0
    fi
    # Expected output count = max .npz across folds (they should all match)
    if [[ ${N_NPZ} -gt ${N_EXPECTED} ]]; then
        N_EXPECTED=${N_NPZ}
    fi
done

if [[ ${ALL_OK} -eq 0 ]]; then
    echo ""
    echo "ERROR: preflight failed — cannot run ensemble." >&2
    exit 1
fi

echo ""
echo "Expected ensemble output: ${N_EXPECTED} cases"

# -- Disk space sanity check ------------------------------------------------
AVAIL_GB=$(df -BG "${PRED_ENSEMBLE_NFS}" 2>/dev/null | awk 'NR==2 {gsub("G","",$4); print $4}')
if [[ -n "${AVAIL_GB}" ]]; then
    echo "Available disk on output volume: ${AVAIL_GB} GB"
    if [[ ${AVAIL_GB} -lt 5 ]]; then
        echo "WARN: <5GB free. Ensemble outputs ~5MB × ${N_EXPECTED} cases = ~$((N_EXPECTED * 5))MB."
        echo "      If quota-bound, free space NOW or this will hang silently." >&2
    fi
fi

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

# -- Background progress watcher --------------------------------------------
WATCH_DONE_FLAG="${PRED_ENSEMBLE_NFS}/.watcher_should_exit"
rm -f "${WATCH_DONE_FLAG}"
(
    T_START=$(date +%s)
    LAST_N=0
    LAST_T=${T_START}
    while [[ ! -f "${WATCH_DONE_FLAG}" ]]; do
        sleep 30
        NOW=$(date +%s)
        N_CUR=$({ find "${PRED_ENSEMBLE_NFS}" -maxdepth 1 -name "*.nii.gz" 2>/dev/null || true; } | wc -l)
        ELAPSED=$((NOW - T_START))
        DELTA_N=$((N_CUR - LAST_N))
        DELTA_T=$((NOW - LAST_T))
        if [[ ${ELAPSED} -gt 0 && ${N_CUR} -gt 0 ]]; then
            RATE_AVG=$(awk -v n=${N_CUR} -v t=${ELAPSED} 'BEGIN { printf "%.2f", (60.0 * n / t) }')
            REMAIN=$((N_EXPECTED - N_CUR))
            ETA_SEC=$(awk -v r=${RATE_AVG} -v rem=${REMAIN} 'BEGIN { if (r > 0) printf "%.0f", (60.0 * rem / r); else print "?" }')
            ETA_HM=$(awk -v s=${ETA_SEC} 'BEGIN { h=int(s/3600); m=int((s%3600)/60); printf "%dh%02dm", h, m }')
        else
            RATE_AVG="—"; ETA_HM="—"
        fi
        if [[ ${DELTA_T} -gt 0 ]]; then
            RATE_INST=$(awk -v dn=${DELTA_N} -v dt=${DELTA_T} 'BEGIN { printf "%.2f", (60.0 * dn / dt) }')
        else
            RATE_INST="—"
        fi
        printf "[ensemble watcher %s] %d/%d done (Δ%d in %ds) | rate %s/min inst, %s/min avg | ETA %s\n" \
            "$(date +%H:%M:%S)" "${N_CUR}" "${N_EXPECTED}" "${DELTA_N}" "${DELTA_T}" \
            "${RATE_INST}" "${RATE_AVG}" "${ETA_HM}"
        LAST_N=${N_CUR}; LAST_T=${NOW}
        if [[ ${N_CUR} -ge ${N_EXPECTED} ]]; then break; fi
    done
) &
WATCHER_PID=$!
trap 'rm -rf "${SINGULARITY_TMPDIR}"; touch "${WATCH_DONE_FLAG}" 2>/dev/null; kill ${WATCHER_PID} 2>/dev/null || true' EXIT

# -- Run ensemble combine ---------------------------------------------------
echo ""
echo "Launching nnUNetv2_ensemble at $(date -Iseconds) ..."
echo "  watcher PID: ${WATCHER_PID} (polls every 30s)"
echo "----------------------------------------------------------------"
set +e
stdbuf -oL -eL singularity exec "${SING_BINDS[@]}" "${CONTAINER}" \
    nnUNetv2_ensemble \
        -i "${PRED_PERFOLD_CONT}/fold_0" \
           "${PRED_PERFOLD_CONT}/fold_1" \
           "${PRED_PERFOLD_CONT}/fold_2" \
           "${PRED_PERFOLD_CONT}/fold_3" \
           "${PRED_PERFOLD_CONT}/fold_4" \
        -o "${PRED_ENSEMBLE_CONT}"
ENSEMBLE_EXIT=$?
set -e

# -- Stop watcher -----------------------------------------------------------
touch "${WATCH_DONE_FLAG}"
sleep 2
kill ${WATCHER_PID} 2>/dev/null || true
rm -f "${WATCH_DONE_FLAG}"

# -- Validate output --------------------------------------------------------
N_OUT=$({ find "${PRED_ENSEMBLE_NFS}" -maxdepth 1 -name "*.nii.gz" 2>/dev/null || true; } | wc -l)

echo ""
echo "================================================================"
echo " ENSEMBLE COMPLETE  ${CHECKPOINT}  $(date -Iseconds)"
echo "================================================================"
echo "  ensemble exit code : ${ENSEMBLE_EXIT}"
echo "  cases done         : ${N_OUT} / ${N_EXPECTED}"
echo "  output dir         : ${PRED_ENSEMBLE_NFS}"
echo "================================================================"

if [[ ${N_OUT} -eq 0 ]]; then
    echo "ERROR: ensemble produced 0 outputs." >&2
    exit 1
fi
if [[ ${N_OUT} -lt ${N_EXPECTED} ]]; then
    echo "WARN: ${N_OUT} of ${N_EXPECTED} cases produced. Some cases may be missing." >&2
    echo "      Most likely cause: case set differs across fold dirs (ensemble" >&2
    echo "      only combines cases present in ALL 5 folds)." >&2
fi

echo ""
echo "Next: run eval"
echo "   sbatch slurm/spine_eval_803_full.sh"
echo ""
echo "Quick view:"
echo "   ls ${PRED_ENSEMBLE_NFS} | head -5"

if [[ ${ENSEMBLE_EXIT} -ne 0 && ${N_OUT} -lt ${N_EXPECTED} ]]; then
    exit ${ENSEMBLE_EXIT}
fi
exit 0
