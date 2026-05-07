#!/bin/bash
#SBATCH --job-name=spine_pred_803
#SBATCH -q gpu
#SBATCH --array=0-19
#SBATCH --exclude=msa1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=16:00:00
#SBATCH --output=logs/spine_pred_803_t%a_%A.out
#SBATCH --error=logs/spine_pred_803_t%a_%A.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=go2432@wayne.edu
# =============================================================================
# Stage 1 / 2: PREDICT — 2D shard (fold × case-shard)  Dataset803 / 500ep / best
# =============================================================================
#
# 20-task array: task_id = fold * 4 + shard   (FOLDS=5, SHARDS_PER_FOLD=4)
#
#   task  0 -> fold 0 shard 0      task 10 -> fold 2 shard 2
#   task  1 -> fold 0 shard 1      task 11 -> fold 2 shard 3
#   task  2 -> fold 0 shard 2      task 12 -> fold 3 shard 0
#   task  3 -> fold 0 shard 3      task 13 -> fold 3 shard 1
#   task  4 -> fold 1 shard 0      task 14 -> fold 3 shard 2
#   task  5 -> fold 1 shard 1      task 15 -> fold 3 shard 3
#   task  6 -> fold 1 shard 2      task 16 -> fold 4 shard 0
#   task  7 -> fold 1 shard 3      task 17 -> fold 4 shard 1
#   task  8 -> fold 2 shard 0      task 18 -> fold 4 shard 2
#   task  9 -> fold 2 shard 1      task 19 -> fold 4 shard 3
#
# Each shard processes cases [shard::4] of imagesTs/ sorted alphabetically,
# matching nnU-Net v2's internal -num_parts / -part_id stride. With 174
# test cases and 4 shards, sizes are 44/44/43/43.
#
# All 4 shards of a fold write to the SAME fold_${FOLD}/ dir (different
# cases, no collision on the data files). Metadata files (plans.json,
# dataset.json, predict_from_raw_data_args.json) are identical across
# shards; we stagger startup by SHARD*3 seconds to make the concurrent
# writes effectively sequential and avoid the (benign but ugly) race.
#
# Why 2D sharding and NOT multiple predictors per GPU
# ----------------------------------------------------
# At this patch size (256x320x320) inference is GPU-compute-bound, not
# memory-bound. A second predictor on the same H200 contends for SMs
# and yields ~1.2x throughput, not 2x, while adding IPC and file-write
# races. Case-level sharding across dedicated GPUs is strictly better.
# The "intra-job parallelism" that actually matters is preprocessing
# and export pipelining, which -npp/-nps already give us — while the
# H200 chews on case N's sliding-window inference, NPP=8 CPU workers
# preprocess case N+1..N+8 and NPS=6 export workers write case
# N-1..N-6. The GPU is never waiting.
#
# Submission
# ----------
#   sbatch slurm/spine_predict_803_best_array.sh
#
# Chain ensemble (auto-runs after all 20 tasks finish):
#   ARR=$(sbatch --parsable slurm/spine_predict_803_best_array.sh)
#   sbatch --dependency=afterok:${ARR} slurm/spine_ensemble_803_best.sh
#
# Re-run a single shard (e.g., fold 2 shard 1 = task 9):
#   sbatch --array=9 slurm/spine_predict_803_best_array.sh
#
# Re-run all shards of one fold (fold 3 = tasks 12-15):
#   sbatch --array=12-15 slurm/spine_predict_803_best_array.sh
#
# Override checkpoint:
#   sbatch --export=ALL,CHECKPOINT=checkpoint_latest.pth \
#       slurm/spine_predict_803_best_array.sh
#
# What this preserves from the original single-job predict
# ---------------------------------------------------------
#   * /data/hf_export bind so imagesTs symlinks resolve in-container
#   * /dev/shm tmpdir for nnU-Net multiprocessing IPC (NFS-socket race
#     fix; per-task /dev/shm path so 4 tasks per node don't collide)
#   * --continue_prediction always-on (idempotent re-runs)
#   * Filesystem-based completion validation (cosmetic exit codes
#     from the multiprocessing shutdown race are ignored when all
#     expected outputs are present and non-empty)
#   * stdbuf line-buffered logs + 60s ETA watcher
#
# What changed
# ------------
#   * --array=0-4  ->  --array=0-19  (2D unrolled to 1D)
#   * -f 0 1 2 3 4  ->  -f ${FOLD} -num_parts 4 -part_id ${SHARD}
#   * --save_probabilities (required for downstream ensemble)
#   * Per-shard expected case list computed up front; completion
#     validated by per-case existence check, not raw count
#   * cpus 24 -> 16, mem 250G -> 128G  (so 4 tasks fit per node)
#   * walltime 36h -> 4h (one shard ~ 44 cases)
# =============================================================================

set -euo pipefail

# -- Decode 2D shard from 1D array task id -----------------------------------
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
ARRAY_JOB=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}
NUM_FOLDS=5
NUM_SHARDS=4
FOLD=$(( TASK_ID / NUM_SHARDS ))
SHARD=$(( TASK_ID % NUM_SHARDS ))
SHARD_TAG="f${FOLD}s${SHARD}"

if (( FOLD < 0 || FOLD >= NUM_FOLDS )); then
    echo "ERROR: invalid task ${TASK_ID} -> fold ${FOLD} (must be 0..$((NUM_FOLDS-1)))" >&2
    exit 1
fi

# -- Stagger startup so concurrent shards don't race on metadata writes ------
# All 4 shards of a fold write plans.json/dataset.json/predict_from_raw_data_args.json
# to the same fold_${F}/ dir on first call. Content is identical across shards,
# so a race is benign but messy. SHARD*3s offset makes the writes sequential.
if [[ ${SHARD} -gt 0 ]]; then
    echo "[${SHARD_TAG}] startup stagger: sleep $((SHARD * 3))s"
    sleep $((SHARD * 3))
fi

# -- Singularity runtime dirs (per-task uniqueness) --------------------------
export SINGULARITY_TMPDIR="/tmp/${USER}_job_${SLURM_JOB_ID}_${SHARD_TAG}"
export XDG_RUNTIME_DIR="${SINGULARITY_TMPDIR}/runtime"
export NXF_SINGULARITY_CACHEDIR="${HOME}/singularity_cache"
mkdir -p "${SINGULARITY_TMPDIR}" "${XDG_RUNTIME_DIR}" "${NXF_SINGULARITY_CACHEDIR}"
export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
unset JAVA_HOME; which singularity
export NXF_SINGULARITY_HOME_MOUNT=true
unset LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE

# -- Node-local tmpfs for nnU-Net multiprocessing IPC (per-task) -------------
# 4 tasks may land on the same node; each gets its own /dev/shm subdir.
NODE_LOCAL_TMP="/dev/shm/${USER}_pymp_${SLURM_JOB_ID}_${SHARD_TAG}"
mkdir -p "${NODE_LOCAL_TMP}"

# -- Config ------------------------------------------------------------------
DATASET_ID=803
DATASET_NAME="SpineSurgCTFullMerged"
CONFIG="3d_fullres"
TRAINER="${TRAINER:-nnUNetTrainerWandB_500ep_LSTVOversample}"
GPU_MEMORY_TARGET_GB=100
PLANS="nnUNetResEncUNetPlans_${GPU_MEMORY_TARGET_GB}G"
CHECKPOINT="${CHECKPOINT:-checkpoint_best.pth}"

NPP="${NPP:-8}"   # preprocessing workers (CPU; pipelines with GPU)
NPS="${NPS:-6}"   # export workers (CPU; pipelines with GPU)
# Budget: NPP + NPS + main + OMP buffer = 8+6+1+1 = 16, fits cpus-per-task

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-${HOME}/spinesurg-ct-nnunet}"
NNUNET_NFS="${PROJECT_ROOT}/nnunet"
CONTAINER="${PROJECT_ROOT}/containers/spinesurg-ct.sif"
WANDB_TRAINER_HOST="${PROJECT_ROOT}/tools/nnunet_wandb_variant.py"
WANDB_TRAINER_CONTAINER="/opt/conda/lib/python3.11/site-packages/nnunetv2/training/nnUNetTrainer/variants/nnunet_wandb_variant.py"

# -- HF export discovery (matches slurm/spine_prep.sh) -----------------------
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
IMAGES_TS_NFS="${NNUNET_NFS}/raw/${DS_DIR_NAME}/imagesTs"

mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/tmp"

# -- Per-fold output dir (shared across 4 shards of this fold) --------------
CKPT_TAG=$(echo "${CHECKPOINT}" | sed -e 's/checkpoint_//' -e 's/\.pth$//')
PRED_BASE_NFS="${NNUNET_NFS}/predictions/${DS_DIR_NAME}_${CONFIG}_${TRAINER}_${PLANS}_PERFOLD_${CKPT_TAG}"
PRED_BASE_CONT="/nnunet_nfs/predictions/${DS_DIR_NAME}_${CONFIG}_${TRAINER}_${PLANS}_PERFOLD_${CKPT_TAG}"
PRED_DIR_NFS="${PRED_BASE_NFS}/fold_${FOLD}"
PRED_DIR_CONT="${PRED_BASE_CONT}/fold_${FOLD}"
mkdir -p "${PRED_DIR_NFS}"

# -- Compute THIS shard's expected cases (matches nnU-Net's [part::n] stride) ---
# nnU-Net v2's predict slices the sorted case list as cases[part_id::num_parts].
# We replicate that with awk so we know exactly which cases belong to us.
ALL_CASE_BASES=$(
    find "${IMAGES_TS_NFS}" -maxdepth 1 -name '*_0000.nii.gz' -printf '%f\n' 2>/dev/null \
        | sort \
        | sed 's/_0000\.nii\.gz$//'
)
N_TEST=$(echo "${ALL_CASE_BASES}" | grep -c .)
SHARD_CASE_BASES=$(echo "${ALL_CASE_BASES}" | awk -v s=${SHARD} -v n=${NUM_SHARDS} '(NR-1) % n == s')
N_EXPECTED=$(echo "${SHARD_CASE_BASES}" | grep -c .)

# ----- Config banner --------------------------------------------------------
echo "================================================================"
echo " SPINESURG-CT PREDICT (sharded)"
echo "   task ${TASK_ID}  ->  fold ${FOLD}, shard ${SHARD}/${NUM_SHARDS}"
echo "================================================================"
echo "  trainer       : ${TRAINER}"
echo "  plans         : ${PLANS}"
echo "  checkpoint    : ${CHECKPOINT}"
echo "  dataset       : ${DS_DIR_NAME}"
echo "  test cases    : ${N_TEST} total, ${N_EXPECTED} on this shard"
echo "  preproc workers (-npp): ${NPP}"
echo "  export workers (-nps) : ${NPS}"
echo "  project root  : ${PROJECT_ROOT}"
echo "  HF export     : ${HF_EXPORT_NFS}"
echo "  pymp tmpdir   : ${NODE_LOCAL_TMP} (per-task /dev/shm)"
echo "  output dir    : ${PRED_DIR_NFS}"
echo "  array job     : ${ARRAY_JOB}"
echo "  slurm job     : ${SLURM_JOB_ID:-<not in slurm>}"
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
[[ ${N_TEST} -eq 0 ]] && {
    echo "ERROR: no test cases found in ${IMAGES_TS_NFS}" >&2
    exit 1
}
[[ ${N_EXPECTED} -eq 0 ]] && {
    echo "ERROR: shard ${SHARD} of ${NUM_SHARDS} has 0 expected cases" >&2
    echo "       (N_TEST=${N_TEST}; misconfigured striping?)" >&2
    exit 1
}

# ----- Verify THIS fold's checkpoint ----------------------------------------
echo ""
FOLD_DIR="${RUN_DIR}/fold_${FOLD}"
if [[ ! -d "${FOLD_DIR}" ]]; then
    echo "ERROR: fold ${FOLD} directory missing: ${FOLD_DIR}" >&2
    exit 1
fi
AVAIL=()
for CKPT in checkpoint_final.pth checkpoint_best.pth checkpoint_latest.pth; do
    [[ -f "${FOLD_DIR}/${CKPT}" ]] && AVAIL+=(${CKPT})
done
if [[ ! -f "${FOLD_DIR}/${CHECKPOINT}" ]]; then
    echo "ERROR: fold ${FOLD} missing requested ${CHECKPOINT}" >&2
    echo "       available: ${AVAIL[*]:-none}" >&2
    exit 1
fi
SIZE=$(du -h "${FOLD_DIR}/${CHECKPOINT}" | cut -f1)
MTIME=$(stat -c '%y' "${FOLD_DIR}/${CHECKPOINT}" | cut -d. -f1)
echo "  fold ${FOLD}: ${CHECKPOINT}  (${SIZE}, ${MTIME})  [available: ${AVAIL[*]}]"

# ----- GPU sanity -----------------------------------------------------------
echo ""
echo "GPU visible to job:"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,driver_version --format=csv | sed 's/^/  /'

# -- Singularity env / binds -------------------------------------------------
export SINGULARITYENV_TMPDIR="/pymp_tmp"
export SINGULARITYENV_nnUNet_raw="/nnunet_nfs/raw"
export SINGULARITYENV_nnUNet_preprocessed="/nnunet_nfs/preprocessed"
export SINGULARITYENV_nnUNet_results="/nnunet_nfs/results"

SING_BINDS=(
    --nv
    --bind "${PROJECT_ROOT}:/workspace"
    --bind "${HF_EXPORT_NFS}:/data/hf_export"
    --bind "${NNUNET_NFS}:/nnunet_nfs"
    --bind "${NODE_LOCAL_TMP}:/pymp_tmp"
    --bind "${WANDB_TRAINER_HOST}:${WANDB_TRAINER_CONTAINER}"
    --pwd  /workspace
)

# ----- Per-shard idempotency check ------------------------------------------
# Count THIS shard's cases that already have BOTH .nii.gz AND .npz present.
count_shard_done() {
    local n_done=0
    while IFS= read -r CASE; do
        [[ -z "$CASE" ]] && continue
        if [[ -s "${PRED_DIR_NFS}/${CASE}.nii.gz" && -s "${PRED_DIR_NFS}/${CASE}.npz" ]]; then
            n_done=$((n_done + 1))
        fi
    done <<< "${SHARD_CASE_BASES}"
    echo ${n_done}
}
N_SHARD_DONE_BEFORE=$(count_shard_done)
N_FOLD_NII=$({ find "${PRED_DIR_NFS}" -maxdepth 1 -name "*.nii.gz" 2>/dev/null || true; } | wc -l)
N_FOLD_NPZ=$({ find "${PRED_DIR_NFS}" -maxdepth 1 -name "*.npz"   2>/dev/null || true; } | wc -l)
echo ""
echo "Pre-run state of fold_${FOLD}/:"
echo "  fold dir total: ${N_FOLD_NII} .nii.gz, ${N_FOLD_NPZ} .npz (across all 4 shards)"
echo "  this shard done: ${N_SHARD_DONE_BEFORE} / ${N_EXPECTED}"

if [[ ${N_SHARD_DONE_BEFORE} -ge ${N_EXPECTED} ]]; then
    echo "Shard ${SHARD_TAG} already complete (.nii.gz + .npz for all expected cases)."
    exit 0
fi

# Probe symlink resolution INSIDE the container before launching predict.
echo ""
echo "Probing symlink resolution inside container ..."
PROBE_RESULT=$(singularity exec "${SING_BINDS[@]}" "${CONTAINER}" \
    bash -c "ls /nnunet_nfs/raw/${DS_DIR_NAME}/imagesTs/*_0000.nii.gz 2>/dev/null \
              | head -1 \
              | xargs -I{} test -f {} && echo OK || echo FAIL")
if [[ "${PROBE_RESULT}" != "OK" ]]; then
    echo "ERROR: imagesTs CT symlinks do not resolve inside the container." >&2
    echo "       /data/hf_export bind missing or HF_EXPORT_DIR wrong." >&2
    HOST_SAMPLE=$(find "${IMAGES_TS_NFS}" -maxdepth 1 -name "*_0000.nii.gz" 2>/dev/null | head -1)
    if [[ -n "${HOST_SAMPLE}" ]]; then
        echo "       Sample: ${HOST_SAMPLE}" >&2
        echo "       -> $(readlink ${HOST_SAMPLE})" >&2
    fi
    exit 2
fi
echo "  symlinks resolve inside container: OK"

# Always pass --continue_prediction. nnU-Net skips per-case work that's
# already on disk, so re-submission of any task is idempotent.
CONTINUE_FLAG="--continue_prediction"

# ----- Background ETA watcher ----------------------------------------------
# Tracks this shard's progress against N_EXPECTED. Also reports the fold's
# total across all 4 shards as a secondary signal.
WATCH_DONE_FLAG="${PRED_DIR_NFS}/.watcher_${SHARD_TAG}_should_exit"
rm -f "${WATCH_DONE_FLAG}"
(
    T_START=$(date +%s)
    LAST_N=${N_SHARD_DONE_BEFORE}
    LAST_T=${T_START}
    while [[ ! -f "${WATCH_DONE_FLAG}" ]]; do
        sleep 60
        NOW=$(date +%s)
        N_CUR=$(count_shard_done)
        N_FOLD_NPZ_CUR=$({ find "${PRED_DIR_NFS}" -maxdepth 1 -name "*.npz" 2>/dev/null || true; } | wc -l)
        ELAPSED=$((NOW - T_START))
        DELTA_N=$((N_CUR - LAST_N))
        DELTA_T=$((NOW - LAST_T))
        TOTAL_DONE=$((N_CUR - N_SHARD_DONE_BEFORE))
        if [[ ${ELAPSED} -gt 0 && ${TOTAL_DONE} -gt 0 ]]; then
            RATE_AVG=$(awk -v n=${TOTAL_DONE} -v t=${ELAPSED} 'BEGIN { printf "%.2f", (60.0 * n / t) }')
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
        printf "[watcher ${SHARD_TAG} %s] shard %d/%d (Δ%d in %ds) | rate %s/min inst, %s/min avg | ETA %s | fold-npz %d/%d\n" \
            "$(date +%H:%M:%S)" "${N_CUR}" "${N_EXPECTED}" "${DELTA_N}" "${DELTA_T}" \
            "${RATE_INST}" "${RATE_AVG}" "${ETA_HM}" "${N_FOLD_NPZ_CUR}" "${N_TEST}"
        LAST_N=${N_CUR}; LAST_T=${NOW}
    done
) &
WATCHER_PID=$!
trap 'rm -rf "${SINGULARITY_TMPDIR}" "${NODE_LOCAL_TMP}"; touch "${WATCH_DONE_FLAG}" 2>/dev/null; kill ${WATCHER_PID} 2>/dev/null || true' EXIT

# ----- Run predict (this fold, this case shard) -----------------------------
echo ""
echo "Launching nnUNetv2_predict for ${SHARD_TAG} at $(date -Iseconds) ..."
echo "  -f ${FOLD}  -num_parts ${NUM_SHARDS}  -part_id ${SHARD}"
echo "----------------------------------------------------------------"
set +e
stdbuf -oL -eL singularity exec "${SING_BINDS[@]}" "${CONTAINER}" \
    nnUNetv2_predict \
        -i  "/nnunet_nfs/raw/${DS_DIR_NAME}/imagesTs" \
        -o  "${PRED_DIR_CONT}" \
        -d  ${DATASET_ID} \
        -c  ${CONFIG} \
        -f  ${FOLD} \
        -tr ${TRAINER} \
        -p  ${PLANS} \
        -chk ${CHECKPOINT} \
        -npp ${NPP} \
        -nps ${NPS} \
        -num_parts ${NUM_SHARDS} \
        -part_id   ${SHARD} \
        --save_probabilities \
        --verbose \
        ${CONTINUE_FLAG}
NNUNET_EXIT=$?
set -e

# ----- Stop watcher and validate by per-case existence ---------------------
touch "${WATCH_DONE_FLAG}"
sleep 2
kill ${WATCHER_PID} 2>/dev/null || true
rm -f "${WATCH_DONE_FLAG}"

sleep 5
N_SHARD_DONE_AFTER=$(count_shard_done)

# Per-case missing list for this shard (helps diagnose partial completion)
MISSING_CASES=()
while IFS= read -r CASE; do
    [[ -z "$CASE" ]] && continue
    if [[ ! -s "${PRED_DIR_NFS}/${CASE}.nii.gz" || ! -s "${PRED_DIR_NFS}/${CASE}.npz" ]]; then
        MISSING_CASES+=("${CASE}")
    fi
done <<< "${SHARD_CASE_BASES}"

# Check for tiny / zero-byte files specifically among this shard's cases
N_BAD=0
while IFS= read -r CASE; do
    [[ -z "$CASE" ]] && continue
    for EXT in nii.gz npz; do
        F="${PRED_DIR_NFS}/${CASE}.${EXT}"
        if [[ -e "$F" && ! -s "$F" ]]; then
            N_BAD=$((N_BAD + 1))
            echo " WARN: ${F} is zero-byte" >&2
        elif [[ -e "$F" && $(stat -c %s "$F") -lt 1024 ]]; then
            N_BAD=$((N_BAD + 1))
            echo " WARN: ${F} is < 1KB" >&2
        fi
    done
done <<< "${SHARD_CASE_BASES}"

echo ""
echo "================================================================"
echo " SHARD ${SHARD_TAG} COMPLETE  (${CHECKPOINT})  $(date -Iseconds)"
echo "================================================================"
echo "   nnUNet exit code  : ${NNUNET_EXIT}"
echo "   shard cases done  : ${N_SHARD_DONE_AFTER} / ${N_EXPECTED}"
echo "   suspicious files  : ${N_BAD} (zero-byte / <1KB)"
echo "   output directory  : ${PRED_DIR_NFS}"
if [[ ${#MISSING_CASES[@]} -gt 0 && ${#MISSING_CASES[@]} -le 10 ]]; then
    echo "   missing cases     : ${MISSING_CASES[*]}"
elif [[ ${#MISSING_CASES[@]} -gt 10 ]]; then
    echo "   missing cases     : ${#MISSING_CASES[@]} (truncated; first 10: ${MISSING_CASES[@]:0:10})"
fi
echo ""

if [[ ${N_SHARD_DONE_AFTER} -ge ${N_EXPECTED} && ${N_BAD} -eq 0 ]]; then
    if [[ ${NNUNET_EXIT} -ne 0 ]]; then
        echo " NOTE: nnUNetv2_predict exited ${NNUNET_EXIT}, but all"
        echo "       ${N_SHARD_DONE_AFTER}/${N_EXPECTED} shard cases are present"
        echo "       (.nii.gz + .npz, all >=1KB). Treating ${SHARD_TAG} as successful."
        echo "       (Multiprocessing-shutdown race in nnU-Net v2 — cosmetic.)"
    fi
    echo ""
    echo " Shard ${SHARD_TAG} done."
    echo " When all 20 tasks (5 folds × 4 shards) finish, run:"
    echo "   sbatch slurm/spine_ensemble_803_best.sh"
    echo "================================================================"
    exit 0
elif [[ ${N_SHARD_DONE_AFTER} -lt ${N_EXPECTED} ]]; then
    echo " WARN: ${N_SHARD_DONE_AFTER} of ${N_EXPECTED} shard cases complete."
    echo "       Re-submit this shard:  sbatch --array=${TASK_ID} $0"
    echo "       --continue_prediction will resume."
    echo "================================================================"
    exit ${NNUNET_EXIT}
else
    echo " ERROR: counts present but ${N_BAD} bad files. Inspect above." >&2
    echo "================================================================"
    exit 1
fi
