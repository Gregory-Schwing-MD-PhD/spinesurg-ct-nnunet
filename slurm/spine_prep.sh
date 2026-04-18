#!/bin/bash
#SBATCH --job-name=spine_prep
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=500G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=6:00:00
#SBATCH --output=logs/spine_prep_%j.out
#SBATCH --error=logs/spine_prep_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=go2432@wayne.edu
# =============================================================================
# SpineSurg-CT  --  Stage A: convert + preprocess (local-scratch optimized)
# =============================================================================
# Why this is fast (and why the old version was 6 hours)
# ------------------------------------------------------
# Three things happen in parallel during preprocess that will all hit NFS
# hard unless you intercept them:
#   1. 48 workers READING raw CT NIfTIs from NFS -> read contention
#   2. 48 workers WRITING .npz/.pkl to NFS       -> write contention
#   3. total preprocessed output (~50 GB) piling up on scratch -> fill risk
#
# This script fixes all three:
#   1. Raw NIfTIs are staged NFS -> local scratch ONCE (sequential read,
#      symlinks resolved). 48 workers then read from local.
#   2. Preprocess writes go to local scratch. No NFS write contention.
#   3. A background daemon every DURABLE_RSYNC_EVERY_SEC (default 600 s)
#      moves .npz/.pkl files older than 2 min from scratch to NFS using
#      rsync --remove-source-files. Scratch use stays bounded.
#
# Durability
# ----------
# After each daemon tick, NFS holds a growing fraction of finished cases.
# If SLURM kills the job mid-preprocess, the next run stages those back
# from NFS to local and nnU-Net resumes (skips .npz that already exist).
# On successful completion, a final rsync + marker atomically finish.
#
# Persistence / reuse
# -------------------
# Preprocessed state on NFS is per (dataset_id x plans_name). It is shared
# across ALL 5 folds and across every ablation that uses the same combo.
# You pay the preprocessing cost exactly once per (dataset, plans) pair.
#
# Idempotency
# -----------
# Step 0 checks NFS cache; if the marker + plans + .npz are all present,
# the whole script short-circuits in seconds. Marker is removed only
# right before preprocess runs and rewritten only after final rsync, so
# partial states stay detectable (marker absent -> partial).
#
# Prereqs
# -------
# 1. data/splits_5fold.json              (from slurm/generate_splits.sh)
# 2. data/hf_export/                     (from slurm/export_hf.sh)
# 3. (recommended) pre-chosen GPU_MEMORY_TARGET_GB via
#      slurm/plan_sizes.sh + tools/visualize_patch_sizes.py
# =============================================================================

set -euo pipefail

# ----- Config ---------------------------------------------------------------
DATASET_ID="${DATASET_ID:-802}"
DATASET_NAME="${DATASET_NAME:-SpineSurgCTFull}"
CONFIG="${CONFIG:-3d_fullres}"
PLANNER="${PLANNER:-nnUNetPlannerResEncM}"
GPU_MEMORY_TARGET_GB="${GPU_MEMORY_TARGET_GB:-100}"
N_PREPROCESS_THREADS="${N_PREPROCESS_THREADS:-48}"
SINGLE_FOLD_SPLITS="${SINGLE_FOLD_SPLITS:-0}"
# How often the background daemon rsyncs+cleans. Smaller = lower peak
# scratch use + more durability, but more NFS load. 10 min is a good
# balance for ~1000 cases.
DURABLE_RSYNC_EVERY_SEC="${DURABLE_RSYNC_EVERY_SEC:-600}"
# Files must be at least this old (seconds) before the daemon moves them.
# Protects in-flight writes. Minimum safe value ~60 s.
DAEMON_MIN_AGE_MIN="${DAEMON_MIN_AGE_MIN:-2}"
# Ablation filter pass-throughs (rare for normal prep)
INCLUDE_TRAIN_MATCH_TYPES="${INCLUDE_TRAIN_MATCH_TYPES:-}"
TEST_MATCH_TYPES="${TEST_MATCH_TYPES:-}"

case "${PLANNER}" in
    nnUNetPlannerResEncM)  PLANS="nnUNetResEncUNetPlans_${GPU_MEMORY_TARGET_GB}G"   ;;
    nnUNetPlannerResEncL)  PLANS="nnUNetResEncUNetLPlans_${GPU_MEMORY_TARGET_GB}G"  ;;
    nnUNetPlannerResEncXL) PLANS="nnUNetResEncUNetXLPlans_${GPU_MEMORY_TARGET_GB}G" ;;
    ExperimentPlanner)     PLANS="nnUNetPlans"                                     ;;
    *) echo "ERROR: unknown planner ${PLANNER}" >&2; exit 1 ;;
esac

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-${HOME}/SpineSurg-CT}"
HF_EXPORT_NFS="${PROJECT_ROOT}/data/hf_export"
SPLITS_FILE_HOST="${PROJECT_ROOT}/data/splits_5fold.json"
NNUNET_NFS="${PROJECT_ROOT}/nnunet"
CONTAINER="${PROJECT_ROOT}/containers/spinesurg-ct.sif"
WANDB_TRAINER_HOST="${PROJECT_ROOT}/tools/nnunet_wandb_variant.py"
WANDB_TRAINER_CONTAINER="/opt/conda/lib/python3.11/site-packages/nnunetv2/training/nnUNetTrainer/variants/nnunet_wandb_variant.py"

DS_DIR_NAME="Dataset$(printf '%03d' ${DATASET_ID})_${DATASET_NAME}"
NFS_RAW_DS="${NNUNET_NFS}/raw/${DS_DIR_NAME}"
NFS_PREP_DS="${NNUNET_NFS}/preprocessed/${DS_DIR_NAME}"
COMPLETE_MARKER="${NFS_PREP_DS}/.preprocessed_complete_${PLANS}"

mkdir -p "${NNUNET_NFS}/raw" "${NNUNET_NFS}/preprocessed" "${NNUNET_NFS}/results" \
         "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/tmp"

# ============================================================================
# Preflight
# ============================================================================
[[ ! -f "${CONTAINER}"          ]] && { echo "ERROR: container not found: ${CONTAINER}" >&2; exit 1; }
[[ ! -d "${HF_EXPORT_NFS}/ct"   ]] && { echo "ERROR: ${HF_EXPORT_NFS}/ct missing" >&2; exit 1; }
[[ ! -f "${WANDB_TRAINER_HOST}" ]] && { echo "ERROR: ${WANDB_TRAINER_HOST} missing" >&2; exit 1; }
[[ ! -f "${PROJECT_ROOT}/tools/convert_hf_to_nnunet.py" ]] && {
    echo "ERROR: tools/convert_hf_to_nnunet.py missing" >&2; exit 1; }

if [[ "${SINGLE_FOLD_SPLITS}" != "1" ]]; then
    [[ ! -f "${SPLITS_FILE_HOST}" ]] && {
        echo "ERROR: ${SPLITS_FILE_HOST} missing. Run slurm/generate_splits.sh first." >&2
        exit 1; }
    SP_MTIME=$(stat -c '%y' "${SPLITS_FILE_HOST}" 2>/dev/null || stat -f '%Sm' "${SPLITS_FILE_HOST}")
    echo "  splits_5fold.json mtime: ${SP_MTIME}"
fi

# ============================================================================
# Step 0: NFS cache check -- exit in seconds if already complete
# ============================================================================
if [[ -f "${COMPLETE_MARKER}" ]] \
   && [[ -d "${NFS_PREP_DS}/${CONFIG}" ]] \
   && [[ -f "${NFS_PREP_DS}/dataset_fingerprint.json" ]] \
   && [[ -f "${NFS_PREP_DS}/${PLANS}.json" ]] \
   && [[ -f "${NFS_RAW_DS}/dataset.json" ]]; then
    N_NPZ=$({ find "${NFS_PREP_DS}/${CONFIG}" -maxdepth 1 -name "*.npz" 2>/dev/null || true; } | wc -l)
    SIZE=$(du -sh "${NFS_PREP_DS}" 2>/dev/null | awk '{print $1}')
    echo "================================================================"
    echo " NFS cache HIT: ${NFS_PREP_DS}"
    echo "   ${N_NPZ} .npz files, ${SIZE} total"
    echo "   plans  : ${PLANS}"
    echo "   marker : $(stat -c '%y' "${COMPLETE_MARKER}" 2>/dev/null || stat -f '%Sm' "${COMPLETE_MARKER}")"
    echo "================================================================"
    if [[ -f "${NFS_RAW_DS}/splits_final.json" ]]; then
        cp -f "${NFS_RAW_DS}/splits_final.json" "${NFS_PREP_DS}/splits_final.json"
        echo "  refreshed splits_final.json"
    fi
    echo ""
    echo " NEXT: FOLD=0 sbatch slurm/spine_train_fold.sh"
    exit 0
fi

# ============================================================================
# Local scratch setup
# ============================================================================
if [[ -n "${SLURM_TMPDIR:-}" && -d "${SLURM_TMPDIR}" ]]; then
    LOCAL_SCRATCH="${SLURM_TMPDIR}"
elif [[ -d "/scratch/${USER}" ]]; then
    LOCAL_SCRATCH="/scratch/${USER}/job_${SLURM_JOB_ID}"; mkdir -p "${LOCAL_SCRATCH}"
else
    LOCAL_SCRATCH="/tmp/${USER}_${SLURM_JOB_ID}"; mkdir -p "${LOCAL_SCRATCH}"
fi

LOCAL_NNUNET="${LOCAL_SCRATCH}/nnunet"
LOCAL_RAW_DS="${LOCAL_NNUNET}/raw/${DS_DIR_NAME}"
LOCAL_PREP_DS="${LOCAL_NNUNET}/preprocessed/${DS_DIR_NAME}"
mkdir -p "${LOCAL_RAW_DS}" "${LOCAL_PREP_DS}"

SCRATCH_AVAIL_GB=$(df -BG "${LOCAL_SCRATCH}" 2>/dev/null | awk 'NR==2 {gsub("G",""); print $4+0}' || echo 0)

echo "================================================================"
echo " Stage A: convert + preprocess (local-scratch optimized)"
echo "   Dataset         : ${DS_DIR_NAME}"
echo "   Plans           : ${PLANS} (target ${GPU_MEMORY_TARGET_GB} GB)"
echo "   Local scratch   : ${LOCAL_SCRATCH}  (${SCRATCH_AVAIL_GB} GB free)"
echo "   NFS dest        : ${NFS_PREP_DS}"
echo "   Splits mode     : $([[ ${SINGLE_FOLD_SPLITS} == 1 ]] && echo 'single-fold (legacy)' || echo '5-fold CV')"
echo "   Preprocess thr  : ${N_PREPROCESS_THREADS}"
echo "   Daemon interval : ${DURABLE_RSYNC_EVERY_SEC}s  (rsync+cleanup)"
echo "   Daemon min age  : ${DAEMON_MIN_AGE_MIN} min   (safety for in-flight)"
echo "   Started         : $(date)"
echo "================================================================"

# Rough needs: ~50 GB raw + peak ~30 GB transient preprocessed = ~80 GB.
# Warn but do not exit; user may have a different-sized dataset.
if [[ "${SCRATCH_AVAIL_GB}" -lt 120 ]]; then
    echo "  WARN: ${SCRATCH_AVAIL_GB} GB scratch free; this is tight for a 1000-case dataset." >&2
    echo "        The rsync+cleanup daemon will manage this, but if you see" >&2
    echo "        ENOSPC errors, reduce DURABLE_RSYNC_EVERY_SEC to 300." >&2
fi

# ============================================================================
# Singularity setup
# ============================================================================
export SINGULARITYENV_TMPDIR="/workspace/tmp"
# nnU-Net reads raw + writes preprocessed both on LOCAL scratch during preprocess.
# The convert step uses a CLI argument so it is not affected by these env vars.
export SINGULARITYENV_nnUNet_raw="/nnunet_local/raw"
export SINGULARITYENV_nnUNet_preprocessed="/nnunet_local/preprocessed"
export SINGULARITYENV_nnUNet_results="/nnunet_nfs/results"
export SINGULARITY_TMPDIR="/tmp/${USER}_job_${SLURM_JOB_ID}_singularity"
export XDG_RUNTIME_DIR="${SINGULARITY_TMPDIR}/runtime"
mkdir -p "${SINGULARITY_TMPDIR}" "${XDG_RUNTIME_DIR}"

# Cleanup trap: stop daemon, remove singularity tmp
RSYNC_DAEMON_PID=""
cleanup() {
    if [[ -n "${RSYNC_DAEMON_PID:-}" ]] && kill -0 "${RSYNC_DAEMON_PID}" 2>/dev/null; then
        echo "  [trap] stopping rsync+cleanup daemon (pid=${RSYNC_DAEMON_PID})"
        kill "${RSYNC_DAEMON_PID}" 2>/dev/null || true
        wait "${RSYNC_DAEMON_PID}" 2>/dev/null || true
    fi
    rm -rf "${SINGULARITY_TMPDIR}" || true
}
trap cleanup EXIT

SING_BINDS=(
    --nv
    --bind "${PROJECT_ROOT}:/workspace"
    --bind "${HF_EXPORT_NFS}:/data/hf_export"
    --bind "${NNUNET_NFS}:/nnunet_nfs"
    --bind "${LOCAL_NNUNET}:/nnunet_local"
    --bind "${WANDB_TRAINER_HOST}:${WANDB_TRAINER_CONTAINER}"
    --pwd  /workspace
)
if [[ "${SINGLE_FOLD_SPLITS}" != "1" ]]; then
    SING_BINDS+=( --bind "${SPLITS_FILE_HOST}:/workspace/splits_5fold.json:ro" )
fi

# ============================================================================
# Step 1: convert (idempotent, writes to NFS -- tiny symlink-based layout)
# ============================================================================
echo ""; echo "----- Step 1: convert HF export -> nnU-Net raw (on NFS) -----"
if [[ -f "${NFS_RAW_DS}/dataset.json" ]]; then
    echo "  dataset.json present; skipping convert."
else
    CONVERT_ARGS=(
        --hf_export_dir  /data/hf_export
        --nnunet_raw_dir /nnunet_nfs/raw
        --dataset_id     ${DATASET_ID}
        --dataset_name   ${DATASET_NAME}
    )
    if [[ "${SINGLE_FOLD_SPLITS}" == "1" ]]; then
        CONVERT_ARGS+=( --single_fold_splits )
    else
        CONVERT_ARGS+=( --splits_file /workspace/splits_5fold.json )
    fi
    [[ -n "${INCLUDE_TRAIN_MATCH_TYPES}" ]] && \
        CONVERT_ARGS+=( --include_train_match_types "${INCLUDE_TRAIN_MATCH_TYPES}" )
    [[ -n "${TEST_MATCH_TYPES}" ]] && \
        CONVERT_ARGS+=( --test_match_types "${TEST_MATCH_TYPES}" )
    singularity exec "${SING_BINDS[@]}" "${CONTAINER}" \
        python tools/convert_hf_to_nnunet.py "${CONVERT_ARGS[@]}"
fi

# ============================================================================
# Step 2: stage raw NFS -> local scratch (-L resolves symlinks)
# ============================================================================
# One big sequential NFS read. Eliminates read contention from 48 workers.
# -L is critical: the convert script writes symlinks into the HF export.
# Without -L, every worker would still chase those symlinks back to NFS.
echo ""; echo "----- Step 2: stage raw NFS -> local scratch -----"
RAW_STAGE_START=$(date +%s)
rsync -aWL --no-compress --human-readable \
    "${NFS_RAW_DS}/" "${LOCAL_RAW_DS}/"
RAW_STAGE_TIME=$(($(date +%s) - RAW_STAGE_START))
RAW_STAGE_SIZE=$(du -sh "${LOCAL_RAW_DS}" 2>/dev/null | awk '{print $1}')
echo "  staged ${RAW_STAGE_SIZE} of raw data in ${RAW_STAGE_TIME}s"

# ============================================================================
# Step 3: resume from partial NFS preprocessed state (if any)
# ============================================================================
# If a previous job wrote some .npz files to NFS before dying, pull them
# back to local so nnU-Net's skip-if-exists logic picks up where we left off.
if [[ -d "${NFS_PREP_DS}" ]] && [[ -n "$(ls -A "${NFS_PREP_DS}" 2>/dev/null || true)" ]]; then
    N_PARTIAL=$({ find "${NFS_PREP_DS}" -name "*.npz" 2>/dev/null || true; } | wc -l)
    if [[ "${N_PARTIAL}" -gt 0 ]]; then
        echo ""; echo "----- Step 3: resume from partial NFS state -----"
        echo "  found ${N_PARTIAL} .npz on NFS; staging to local for resume..."
        RESUME_START=$(date +%s)
        rsync -aW --no-compress "${NFS_PREP_DS}/" "${LOCAL_PREP_DS}/"
        echo "  resumed in $(($(date +%s) - RESUME_START))s"
    fi
fi

# ============================================================================
# Step 4: start background rsync+cleanup daemon
# ============================================================================
# Loop: every DURABLE_RSYNC_EVERY_SEC seconds,
#   1. Build list of .npz/.pkl older than DAEMON_MIN_AGE_MIN minutes.
#   2. rsync --files-from=LIST --remove-source-files  local -> NFS.
#      This atomically transfers then deletes from scratch.
#   3. rsync metadata files (plans, fingerprint, etc.) but do NOT remove them
#      from local (they're small and nnU-Net may reread them).
DAEMON_LOG="${PROJECT_ROOT}/logs/rsync_daemon_${SLURM_JOB_ID}.log"
echo ""; echo "----- Step 4: start rsync+cleanup daemon -----"
echo "  log: ${DAEMON_LOG}"

(
    set +e   # do not exit daemon on transient failures
    while sleep "${DURABLE_RSYNC_EVERY_SEC}"; do
        TS=$(date '+%Y-%m-%d %H:%M:%S')
        MOVE_LIST="/tmp/${USER}_move_${SLURM_JOB_ID}_$$.txt"

        # Relative-path list of files safe to move
        (cd "${LOCAL_PREP_DS}" 2>/dev/null && \
         find . \( -name "*.npz" -o -name "*.pkl" \) -type f -mmin "+${DAEMON_MIN_AGE_MIN}" \
        ) > "${MOVE_LIST}" 2>/dev/null

        N_MOVE=$(wc -l < "${MOVE_LIST}" 2>/dev/null | tr -d ' ')
        N_MOVE="${N_MOVE:-0}"

        if [[ "${N_MOVE}" -gt 0 ]]; then
            # Atomic move: rsync then remove from source on successful transfer
            rsync -aW --no-compress \
                  --files-from="${MOVE_LIST}" \
                  --remove-source-files \
                  "${LOCAL_PREP_DS}/" "${NFS_PREP_DS}/" \
                  >>"${DAEMON_LOG}" 2>&1

            # Prune empty directories left behind on local
            find "${LOCAL_PREP_DS}" -type d -empty -not -path "${LOCAL_PREP_DS}" -delete 2>/dev/null
        fi

        # Sync metadata (small; keep on local too)
        rsync -auW --no-compress \
              --exclude='*.npz' --exclude='*.pkl' \
              "${LOCAL_PREP_DS}/" "${NFS_PREP_DS}/" \
              >>"${DAEMON_LOG}" 2>&1

        N_LOCAL=$({ find "${LOCAL_PREP_DS}" -name "*.npz" 2>/dev/null || true; } | wc -l)
        N_NFS=$({ find "${NFS_PREP_DS}" -name "*.npz" 2>/dev/null || true; } | wc -l)
        SCRATCH_FREE=$(df -BG "${LOCAL_SCRATCH}" 2>/dev/null | awk 'NR==2 {gsub("G",""); print $4+0}' || echo "?")
        echo "${TS}  moved=${N_MOVE}  local_npz=${N_LOCAL}  nfs_npz=${N_NFS}  scratch_free=${SCRATCH_FREE}GB" >>"${DAEMON_LOG}"

        rm -f "${MOVE_LIST}"
    done
) &
RSYNC_DAEMON_PID=$!
echo "  daemon pid: ${RSYNC_DAEMON_PID}"

# ============================================================================
# Step 5: preprocess (reads local raw, writes local preprocessed)
# ============================================================================
echo ""; echo "----- Step 5: plan + preprocess (local I/O) -----"
rm -f "${COMPLETE_MARKER}"   # invalidate marker; rewritten after final rsync

PREP_START=$(date +%s)
singularity exec "${SING_BINDS[@]}" "${CONTAINER}" \
    nnUNetv2_plan_and_preprocess \
        -d    ${DATASET_ID} \
        -pl   ${PLANNER} \
        -gpu_memory_target ${GPU_MEMORY_TARGET_GB} \
        -overwrite_plans_name ${PLANS} \
        -c    ${CONFIG} \
        -np   ${N_PREPROCESS_THREADS} \
        --verify_dataset_integrity
PREP_TIME=$(($(date +%s) - PREP_START))
echo "  preprocess done in ${PREP_TIME}s"

# ============================================================================
# Step 6: stop daemon, final rsync + cleanup
# ============================================================================
echo ""; echo "----- Step 6: stop daemon + final rsync -----"
if [[ -n "${RSYNC_DAEMON_PID:-}" ]] && kill -0 "${RSYNC_DAEMON_PID}" 2>/dev/null; then
    kill "${RSYNC_DAEMON_PID}" 2>/dev/null || true
    wait "${RSYNC_DAEMON_PID}" 2>/dev/null || true
    RSYNC_DAEMON_PID=""
fi

# Final sweep: rsync whatever the daemon didn't get (too-recent files,
# plans/fingerprint, etc.). --remove-source-files optional here since
# the job is about to release scratch anyway.
RSYNC_START=$(date +%s)
rsync -aW --no-compress --human-readable \
    "${LOCAL_PREP_DS}/" "${NFS_PREP_DS}/"
RSYNC_TIME=$(($(date +%s) - RSYNC_START))

N_NPZ_NFS=$({ find "${NFS_PREP_DS}/${CONFIG}" -maxdepth 1 -name "*.npz" 2>/dev/null || true; } | wc -l)
NFS_SIZE=$(du -sh "${NFS_PREP_DS}" 2>/dev/null | awk '{print $1}')
echo "  final rsync: ${N_NPZ_NFS} .npz on NFS (${NFS_SIZE}) in ${RSYNC_TIME}s"

if [[ "${N_NPZ_NFS}" -eq 0 ]]; then
    echo "ERROR: zero .npz files on NFS after preprocess; something failed." >&2
    exit 2
fi

# ============================================================================
# Step 7: copy splits_final.json into preprocessed/ (nnU-Net reads it there)
# ============================================================================
if [[ -f "${NFS_RAW_DS}/splits_final.json" ]]; then
    cp -f "${NFS_RAW_DS}/splits_final.json" "${NFS_PREP_DS}/splits_final.json"
    echo "  copied splits_final.json into preprocessed/"
fi

# ============================================================================
# Step 8: mark complete (LAST step so partial states stay detectable)
# ============================================================================
touch "${COMPLETE_MARKER}"
echo "  wrote marker: ${COMPLETE_MARKER}"

TOTAL_TIME=$((RAW_STAGE_TIME + PREP_TIME + RSYNC_TIME))
echo ""
echo "================================================================"
echo " Stage A COMPLETE   $(date)"
echo "   raw staging   : ${RAW_STAGE_TIME}s   (${RAW_STAGE_SIZE})"
echo "   preprocess    : ${PREP_TIME}s"
echo "   final rsync   : ${RSYNC_TIME}s"
echo "   total         : ${TOTAL_TIME}s"
echo "   preprocessed  : ${NFS_PREP_DS}  (${NFS_SIZE})"
echo ""
echo " This state is reused across ALL folds and all ablations that"
echo " share the same (dataset_id, plans_name) combination."
echo ""
echo " NEXT:"
echo "   Single fold  :  FOLD=0 sbatch slurm/spine_train_fold.sh"
echo "   5-fold CV    :  sbatch slurm/spine_train_array.sh"
echo "   Ablations    :  sbatch slurm/spine_ablation.sh"
echo "================================================================"
