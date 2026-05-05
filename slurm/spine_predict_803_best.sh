#!/bin/bash
#SBATCH --job-name=spine_pred_803
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=250G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/spine_pred_803_%j.out
#SBATCH --error=logs/spine_pred_803_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=go2432@wayne.edu
# =============================================================================
# Stage 1 / 2: PREDICT ONLY — Dataset803 / 500ep / checkpoint_best
# =============================================================================
# GPU-bound. Writes ensemble predictions for the 174 test cases.
# Env block + binds mirror slurm/spine_prep.sh.
#
# CRITICAL: imagesTs/*_0000.nii.gz are SYMLINKS to /data/hf_export/ct/*
# (container path created by convert_hf_to_nnunet.py --symlinks). The
# /data/hf_export bind MUST be present or those symlinks dangle inside
# the container and nnUNetv2_predict reports "0 cases in the source
# folder". Earlier version of this script omitted that bind and caused
# job 36001641 to do nothing. Don't repeat.
#
#   * -npp 8 / -nps 6 — keep H200 fed
#   * stdbuf line-buffered stdout/stderr
#   * background ETA watcher every 60s
#   * --continue_prediction — re-submit if walltime hits, resumes
#
# Override at submit time:
#   sbatch --export=ALL,CHECKPOINT=checkpoint_latest.pth slurm/spine_predict_803_best.sh
# =============================================================================

set -euo pipefail

# -- Singularity runtime dirs (matches slurm/spine_prep.sh) ------------------
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

NPP="${NPP:-8}"   # preprocessing workers
NPS="${NPS:-6}"   # export workers

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-${HOME}/spinesurg-ct-nnunet}"
NNUNET_NFS="${PROJECT_ROOT}/nnunet"
CONTAINER="${PROJECT_ROOT}/containers/spinesurg-ct.sif"
WANDB_TRAINER_HOST="${PROJECT_ROOT}/tools/nnunet_wandb_variant.py"
WANDB_TRAINER_CONTAINER="/opt/conda/lib/python3.11/site-packages/nnunetv2/training/nnUNetTrainer/variants/nnunet_wandb_variant.py"

# -- HF export discovery (matches slurm/spine_prep.sh) -----------------------
# Required because imagesTs/*_0000.nii.gz are symlinks pointing at
# /data/hf_export/ct/* (container-internal path). Without this bind,
# the symlinks dangle inside the container.
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

DS_DIR_NAME="Dataset$(printf '%03d' ${DATASET_ID})_${DATASET_NAME}"
RUN_DIR="${NNUNET_NFS}/results/${DS_DIR_NAME}/${TRAINER}__${PLANS}__${CONFIG}"

mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/tmp"

# ----- Config banner --------------------------------------------------------
echo "================================================================"
echo " SPINESURG-CT PREDICT  (Dataset${DATASET_ID} / ${CONFIG})"
echo "================================================================"
echo "  trainer       : ${TRAINER}"
echo "  plans         : ${PLANS}"
echo "  checkpoint    : ${CHECKPOINT}"
echo "  preproc workers (-npp): ${NPP}"
echo "  export workers (-nps) : ${NPS}"
echo "  project root  : ${PROJECT_ROOT}"
echo "  HF export     : ${HF_EXPORT_NFS}"
echo "  job id        : ${SLURM_JOB_ID:-<not in slurm>}"
echo "  node          : ${SLURMD_NODENAME:-$(hostname)}"
echo "  start time    : $(date -Iseconds)"
echo "================================================================"

# ----- Preflight ------------------------------------------------------------
[[ ! -f "${CONTAINER}"        ]] && { echo "ERROR: container not found: ${CONTAINER}" >&2; exit 1; }
[[ ! -d "${HF_EXPORT_NFS}/ct" ]] && {
    echo "ERROR: ${HF_EXPORT_NFS}/ct missing — symlinked imagesTs CTs will not resolve." >&2
    echo "  Override with HF_EXPORT_DIR=/path/to/hf_export sbatch ..." >&2
    exit 1
}

# ----- Verify each fold has the requested checkpoint ------------------------
echo ""
echo "Verifying checkpoints under ${RUN_DIR} ..."
MISSING=()
for F in 0 1 2 3 4; do
    FOLD_DIR="${RUN_DIR}/fold_${F}"
    if [[ ! -d "${FOLD_DIR}" ]]; then
        echo "  fold ${F}: NO FOLD DIRECTORY"
        MISSING+=(${F}); continue
    fi
    AVAIL=()
    for CKPT in checkpoint_final.pth checkpoint_best.pth checkpoint_latest.pth; do
        [[ -f "${FOLD_DIR}/${CKPT}" ]] && AVAIL+=(${CKPT})
    done
    if [[ -f "${FOLD_DIR}/${CHECKPOINT}" ]]; then
        SIZE=$(du -h "${FOLD_DIR}/${CHECKPOINT}" | cut -f1)
        MTIME=$(stat -c '%y' "${FOLD_DIR}/${CHECKPOINT}" | cut -d. -f1)
        echo "  fold ${F}: ${CHECKPOINT}  (${SIZE}, ${MTIME})  [available: ${AVAIL[*]:-none}]"
    else
        echo "  fold ${F}: requested ${CHECKPOINT} MISSING  [available: ${AVAIL[*]:-none}]"
        MISSING+=(${F})
    fi
done
if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "ERROR: folds ${MISSING[*]} are missing ${CHECKPOINT}." >&2
    echo "       Re-submit with: sbatch --export=ALL,CHECKPOINT=checkpoint_latest.pth $0" >&2
    exit 1
fi
echo "  All 5 folds have ${CHECKPOINT}."

# ----- GPU sanity -----------------------------------------------------------
echo ""
echo "GPU visible to job:"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,driver_version --format=csv | sed 's/^/  /'

# -- Singularity env mirroring spine_prep.sh ---------------------------------
export SINGULARITYENV_TMPDIR="/workspace/tmp"
export SINGULARITYENV_nnUNet_raw="/nnunet_nfs/raw"
export SINGULARITYENV_nnUNet_preprocessed="/nnunet_nfs/preprocessed"
export SINGULARITYENV_nnUNet_results="/nnunet_nfs/results"

SING_BINDS=(
    --nv
    --bind "${PROJECT_ROOT}:/workspace"
    --bind "${HF_EXPORT_NFS}:/data/hf_export"
    --bind "${NNUNET_NFS}:/nnunet_nfs"
    --bind "${WANDB_TRAINER_HOST}:${WANDB_TRAINER_CONTAINER}"
    --pwd  /workspace
)

CKPT_TAG=$(echo "${CHECKPOINT}" | sed -e 's/checkpoint_//' -e 's/\.pth$//')
PRED_DIR_NFS="${NNUNET_NFS}/predictions/${DS_DIR_NAME}_${CONFIG}_${TRAINER}_${PLANS}_ENSEMBLE_${CKPT_TAG}"
PRED_DIR_CONT="/nnunet_nfs/predictions/${DS_DIR_NAME}_${CONFIG}_${TRAINER}_${PLANS}_ENSEMBLE_${CKPT_TAG}"
mkdir -p "${PRED_DIR_NFS}"

N_EXISTING=$({ find "${PRED_DIR_NFS}" -maxdepth 1 -name "*.nii.gz" 2>/dev/null || true; } | wc -l)
N_TEST=$({ find "${NNUNET_NFS}/raw/${DS_DIR_NAME}/imagesTs" -maxdepth 1 -name "*_0000.nii.gz" 2>/dev/null || true; } | wc -l)
echo ""
echo "Test cases (host find): ${N_TEST}   existing predictions: ${N_EXISTING}"
echo "Output dir: ${PRED_DIR_NFS}"

# Probe symlink resolution INSIDE the container before launching predict.
# If imagesTs CTs don't resolve through the bind, fail now with a clear
# error rather than have nnUNetv2_predict silently report "0 cases".
echo ""
echo "Probing symlink resolution inside container ..."
PROBE_RESULT=$(singularity exec "${SING_BINDS[@]}" "${CONTAINER}" \
    bash -c "ls /nnunet_nfs/raw/${DS_DIR_NAME}/imagesTs/*_0000.nii.gz 2>/dev/null \
              | head -1 \
              | xargs -I{} test -f {} && echo OK || echo FAIL")
if [[ "${PROBE_RESULT}" != "OK" ]]; then
    echo "ERROR: imagesTs CT symlinks do not resolve inside the container." >&2
    echo "       This means /data/hf_export bind is missing or HF_EXPORT_DIR" >&2
    echo "       points at the wrong location. Sample symlink target:" >&2
    HOST_SAMPLE=$(find "${NNUNET_NFS}/raw/${DS_DIR_NAME}/imagesTs" -maxdepth 1 -name "*_0000.nii.gz" 2>/dev/null | head -1)
    if [[ -n "${HOST_SAMPLE}" ]]; then
        echo "       ${HOST_SAMPLE}" >&2
        echo "       -> $(readlink ${HOST_SAMPLE})" >&2
    fi
    exit 2
fi
echo "  symlinks resolve inside container: OK"

if [[ "${N_EXISTING}" -ge "${N_TEST}" && "${N_TEST}" -gt 0 ]]; then
    echo "All ensemble predictions already exist; nothing to do."
    exit 0
fi

CONTINUE_FLAG=""
[[ "${N_EXISTING}" -gt 0 ]] && CONTINUE_FLAG="--continue_prediction"

# ----- Background ETA watcher ----------------------------------------------
WATCH_DONE_FLAG="${PRED_DIR_NFS}/.watcher_should_exit"
rm -f "${WATCH_DONE_FLAG}"
(
    T_START=$(date +%s)
    LAST_N=${N_EXISTING}
    LAST_T=${T_START}
    while [[ ! -f "${WATCH_DONE_FLAG}" ]]; do
        sleep 60
        NOW=$(date +%s)
        N_CUR=$({ find "${PRED_DIR_NFS}" -maxdepth 1 -name "*.nii.gz" 2>/dev/null || true; } | wc -l)
        ELAPSED=$((NOW - T_START))
        DELTA_N=$((N_CUR - LAST_N))
        DELTA_T=$((NOW - LAST_T))
        TOTAL_DONE=$((N_CUR - N_EXISTING))
        if [[ ${ELAPSED} -gt 0 && ${TOTAL_DONE} -gt 0 ]]; then
            RATE_AVG=$(awk -v n=${TOTAL_DONE} -v t=${ELAPSED} 'BEGIN { printf "%.2f", (60.0 * n / t) }')
            REMAIN=$((N_TEST - N_CUR))
            ETA_SEC=$(awk -v r=${RATE_AVG} -v rem=${REMAIN} 'BEGIN { if (r > 0) printf "%.0f", (60.0 * rem / r); else print "?" }')
            ETA_HM=$(awk -v s=${ETA_SEC} 'BEGIN { h=int(s/3600); m=int((s%3600)/60); printf "%dh%02dm", h, m }')
        else
            RATE_AVG="—"
            ETA_HM="—"
        fi
        if [[ ${DELTA_T} -gt 0 ]]; then
            RATE_INST=$(awk -v dn=${DELTA_N} -v dt=${DELTA_T} 'BEGIN { printf "%.2f", (60.0 * dn / dt) }')
        else
            RATE_INST="—"
        fi
        printf "[watcher %s] %d/%d done (Δ%d in %ds) | rate %s/min inst, %s/min avg | ETA %s\n" \
            "$(date +%H:%M:%S)" "${N_CUR}" "${N_TEST}" "${DELTA_N}" "${DELTA_T}" \
            "${RATE_INST}" "${RATE_AVG}" "${ETA_HM}"
        LAST_N=${N_CUR}
        LAST_T=${NOW}
    done
) &
WATCHER_PID=$!
trap 'rm -rf "${SINGULARITY_TMPDIR}"; touch "${WATCH_DONE_FLAG}" 2>/dev/null; kill ${WATCHER_PID} 2>/dev/null || true' EXIT

# ----- Run predict ----------------------------------------------------------
echo ""
echo "Launching nnUNetv2_predict at $(date -Iseconds) ..."
echo "----------------------------------------------------------------"
stdbuf -oL -eL singularity exec "${SING_BINDS[@]}" "${CONTAINER}" \
    nnUNetv2_predict \
        -i  "/nnunet_nfs/raw/${DS_DIR_NAME}/imagesTs" \
        -o  "${PRED_DIR_CONT}" \
        -d  ${DATASET_ID} \
        -c  ${CONFIG} \
        -f  0 1 2 3 4 \
        -tr ${TRAINER} \
        -p  ${PLANS} \
        -chk ${CHECKPOINT} \
        -npp ${NPP} \
        -nps ${NPS} \
        --verbose \
        ${CONTINUE_FLAG}

# ----- Stop watcher and summarize ------------------------------------------
touch "${WATCH_DONE_FLAG}"
sleep 2
kill ${WATCHER_PID} 2>/dev/null || true
rm -f "${WATCH_DONE_FLAG}"

N_FINAL=$({ find "${PRED_DIR_NFS}" -maxdepth 1 -name "*.nii.gz" 2>/dev/null || true; } | wc -l)
echo ""
echo "================================================================"
echo " PREDICT COMPLETE  (${CHECKPOINT})  $(date -Iseconds)"
echo "================================================================"
echo "   predictions written : ${N_FINAL} / ${N_TEST}"
echo "   output directory    : ${PRED_DIR_NFS}"
echo ""
if [[ ${N_FINAL} -lt ${N_TEST} ]]; then
    echo " WARN: only ${N_FINAL} of ${N_TEST} predictions completed."
    echo "       Re-submit this script — --continue_prediction will resume."
else
    echo " Next: run eval"
    echo "   sbatch slurm/spine_eval_803_best.sh"
fi
echo "================================================================"
