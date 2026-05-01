#!/bin/bash
#SBATCH --job-name=spine_kfold
#SBATCH -q gpu
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=5-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --open-mode=append
#SBATCH --output=logs/spine_kfold_f%a_%A.out
#SBATCH --error=logs/spine_kfold_f%a_%A.err
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=go2432@wayne.edu
# =============================================================================
# SpineSurg-CT  --  Stage B (5-fold CV, shared /tmp local-NVMe cache)
# =============================================================================
#
# Apr 2026 — local-NVMe staging via shared /tmp cache
# ===================================================
# Problem (observed on prior NFS-direct runs):
#   Per-epoch time spiked from ~5 min to ~20 min during 5-fold parallel
#   runs. Cause: 5 folds × 24 DA workers = 120 processes hitting the
#   same NFS server with random reads of mmap'd .npy files.
#
# Why earlier mitigations didn't work:
#   - Per-fold staging to local /tmp: copying 437 GB at 21 MB/s NFS rate
#     takes 6+ hours per fold start.
#   - /dev/shm caching: H200 nodes have only 752 GB RAM and tmpfs default
#     of 50% = 378 GB, smaller than the 830 GB cache. Doesn't fit.
#
# What works:
#   /tmp on H200 nodes is local NVMe (1.7 TB / 1.6 TB free, /dev/mapper
#   XFS). Reads are ~3 GB/s (~150x NFS), writes don't count against
#   --mem (cgroups account RAM, not arbitrary disk paths).
#
#   Multiple folds may land on the same node (5 folds, 4 nodes → at
#   least one node hosts 2). Two separate caches × 830 GB = 1.66 TB,
#   exceeding 1.6 TB available /tmp. Therefore the cache is SHARED
#   across folds on the same node.
#
# Design:
#   Cache root: /tmp/spinesurg_${USER}_${PLANS}/preprocessed/<dataset>/
#     - Per-user (no cross-user collisions)
#     - Per-plans (changes if dataset is re-preprocessed)
#     - Shared across folds and jobs on the same node
#     - Persists until node reboot or eviction
#
#   Setup (under flock):
#     1. Check if .preunpack_ok marker exists in cache
#     2. If yes: skip — first fold to land already built it
#     3. If no:
#        a. mkdir cache structure
#        b. symlink .npz, .pkl from NFS source (read once during unpack)
#        c. symlink dataset metadata: splits_final.json, lstv_cases.json,
#           dataset.json, dataset_fingerprint.json, plans JSON,
#           gt_segmentations/
#        d. Run unpack_dataset() into the cache; .npy files are real
#           local-NVMe files. Progress is polled every 30s and printed.
#        e. Touch .preunpack_ok marker
#     4. Release flock
#
#   At training time:
#     - Bind /tmp/spinesurg_${USER}_${PLANS} → /tmp_prep in container
#     - Set SINGULARITYENV_nnUNet_preprocessed=/tmp_prep/preprocessed
#     - nnU-Net reads everything from local NVMe
#
#   Cleanup at job end:
#     Best-effort eviction. Only delete the cache if no other folds
#     from this user are still using it on this node. Otherwise leave
#     it for them to use.
#
# Override to fall back to NFS-direct (for debugging or if /tmp full):
#   SPINESURG_USE_TMP_CACHE=0 sbatch slurm/spine_train_array.sh
#
# Force cache rebuild (e.g., after dataset changes):
#   SPINESURG_FORCE_DATAPREP=1 sbatch slurm/spine_train_array.sh
#
# Quick reference
# ---------------
#   sbatch slurm/spine_train_array.sh                    # all 5 folds
#   sbatch --array=0 slurm/spine_train_array.sh          # single fold
#   TRAINER=nnUNetTrainerWandB_1000ep_LSTVOversample sbatch slurm/spine_train_array.sh
#   LSTV_OVERSAMPLE_FRAC=0.5 sbatch slurm/spine_train_array.sh
#   SPINESURG_PROFILE=1 sbatch --array=0 slurm/spine_train_array.sh
# =============================================================================

set -euo pipefail

FOLD=${SLURM_ARRAY_TASK_ID:-0}
ARRAY_JOB=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}

# -- Singularity / Nextflow runtime dirs (matches spine_prep.sh) -------------
export SINGULARITY_TMPDIR="/tmp/${USER}_job_${SLURM_JOB_ID}_f${FOLD}"
export XDG_RUNTIME_DIR="${SINGULARITY_TMPDIR}/runtime"
export NXF_SINGULARITY_CACHEDIR="${HOME}/singularity_cache"
mkdir -p "${SINGULARITY_TMPDIR}" "${XDG_RUNTIME_DIR}" "${NXF_SINGULARITY_CACHEDIR}"
export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
unset JAVA_HOME; which singularity
export NXF_SINGULARITY_HOME_MOUNT=true
unset LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE

# -- Per-fold persistent compile cache (small; lives in $HOME) ---------------
PERSISTENT_CACHE_ROOT="${SPINESURG_CACHE_ROOT:-${HOME}/.cache/spinesurg}"
PERSISTENT_CACHE="${PERSISTENT_CACHE_ROOT}/${ARRAY_JOB}_f${FOLD}"
if ! mkdir -p "${PERSISTENT_CACHE}" 2>/dev/null; then
    PERSISTENT_CACHE="/tmp/${USER}_spinesurg_${ARRAY_JOB}_f${FOLD}"
    mkdir -p "${PERSISTENT_CACHE}"
fi

EPHEMERAL_TMP="/tmp/${USER}_${SLURM_JOB_ID}_f${FOLD}_tmp"
mkdir -p "${EPHEMERAL_TMP}"

# -- Tunables ----------------------------------------------------------------
export SPINESURG_RUN_VAL_EXPORT="${SPINESURG_RUN_VAL_EXPORT:-0}"
export WANDB_INIT_MAX_RETRIES="${WANDB_INIT_MAX_RETRIES:-5}"
export WANDB_INIT_TIMEOUT_SEC="${WANDB_INIT_TIMEOUT_SEC:-180}"
export WANDB_ALLOW_OFFLINE="${WANDB_ALLOW_OFFLINE:-1}"
export LSTV_OVERSAMPLE_FRAC="${LSTV_OVERSAMPLE_FRAC:-0.25}"

WANDB_RUN_TAG="${WANDB_RUN_TAG:-${ARRAY_JOB}}"
export WANDB_RUN_TAG

export SPINESURG_PERF="${SPINESURG_PERF:-1}"
export SPINESURG_CHANNELS_LAST="${SPINESURG_CHANNELS_LAST:-0}"
export SPINESURG_FORCE_DATAPREP="${SPINESURG_FORCE_DATAPREP:-0}"
export SPINESURG_PROFILE="${SPINESURG_PROFILE:-0}"
export SPINESURG_PROFILE_WAIT="${SPINESURG_PROFILE_WAIT:-5}"
export SPINESURG_PROFILE_WARMUP="${SPINESURG_PROFILE_WARMUP:-3}"
export SPINESURG_PROFILE_ACTIVE="${SPINESURG_PROFILE_ACTIVE:-10}"

SPINESURG_USE_TMP_CACHE="${SPINESURG_USE_TMP_CACHE:-1}"

# -- Training config ---------------------------------------------------------
DATASET_ID=802
DATASET_NAME="SpineSurgCTFull"
CONFIG="3d_fullres"
TRAINER="${TRAINER:-nnUNetTrainerWandB_500ep_LSTVOversample}"
PLANNER="nnUNetPlannerResEncM"
GPU_MEMORY_TARGET_GB=100
PLANS="nnUNetResEncUNetPlans_${GPU_MEMORY_TARGET_GB}G"

PREUNPACK_WORKERS=24
NNUNET_EXPORT_POOL=12
NNUNET_DA_WORKERS=24

# -- Paths -------------------------------------------------------------------
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-${HOME}/SpineSurg-CT}"
NNUNET_NFS="${PROJECT_ROOT}/nnunet"
CONTAINER="${PROJECT_ROOT}/containers/spinesurg-ct.sif"
WANDB_TRAINER_HOST="${PROJECT_ROOT}/tools/nnunet_wandb_variant.py"
WANDB_TRAINER_CONTAINER="/opt/conda/lib/python3.11/site-packages/nnunetv2/training/nnUNetTrainer/variants/nnunet_wandb_variant.py"
DS_DIR_NAME="Dataset$(printf '%03d' ${DATASET_ID})_${DATASET_NAME}"

NFS_PREP_DS="${NNUNET_NFS}/preprocessed/${DS_DIR_NAME}"
NFS_RAW_DS="${NNUNET_NFS}/raw/${DS_DIR_NAME}"

# Local-NVMe shared cache root (per-user, per-plans, shared across folds)
SPINESURG_CACHE_ROOT_TMP="${SPINESURG_CACHE_ROOT_TMP:-/tmp/spinesurg_${USER}_${PLANS}}"
TMP_CACHE_PREP_ROOT="${SPINESURG_CACHE_ROOT_TMP}/preprocessed"
TMP_CACHE_PREP_DS="${TMP_CACHE_PREP_ROOT}/${DS_DIR_NAME}"
TMP_CACHE_LOCK="/tmp/spinesurg_${USER}_${PLANS}.lock"
TMP_CACHE_MARKER="${SPINESURG_CACHE_ROOT_TMP}/.preunpack_ok"

# ----- Preflight ------------------------------------------------------------
[[ ! -f "${CONTAINER}" ]] && { echo "ERROR: container not found: ${CONTAINER}" >&2; exit 1; }
if [[ ! -f "${NFS_PREP_DS}/dataset_fingerprint.json" ]]; then
    echo "ERROR: preprocessed dataset missing; run sbatch slurm/spine_prep.sh first." >&2
    exit 1
fi
if [[ ! -f "${NFS_PREP_DS}/splits_final.json" ]]; then
    echo "ERROR: splits_final.json missing from ${NFS_PREP_DS}." >&2
    exit 1
fi

# ----- Idempotency: skip if fold already trained ----------------------------
RESULT_DIR="${NNUNET_NFS}/results/${DS_DIR_NAME}/${TRAINER}__${PLANS}__${CONFIG}/fold_${FOLD}"
FINAL_CKPT="${RESULT_DIR}/checkpoint_final.pth"
LATEST_CKPT="${RESULT_DIR}/checkpoint_latest.pth"

if [[ -f "${FINAL_CKPT}" ]]; then
    echo "[fold ${FOLD}] checkpoint_final.pth exists; skipping."
    exit 0
fi

RESUME_FLAG=""
if [[ -f "${LATEST_CKPT}" ]]; then
    RESUME_FLAG="--c"
    CKPT_AGE=$(( $(date +%s) - $(stat -c %Y "${LATEST_CKPT}" 2>/dev/null || stat -f %m "${LATEST_CKPT}") ))
    CKPT_SIZE=$(stat -c %s "${LATEST_CKPT}" 2>/dev/null || stat -f %z "${LATEST_CKPT}")
    echo "[fold ${FOLD}] resuming from checkpoint_latest.pth (age ${CKPT_AGE}s, size ${CKPT_SIZE}B)"
fi

# ----- Resolve nnU-Net data_identifier --------------------------------------
resolve_data_identifier() {
    local plans_json="${NFS_PREP_DS}/${PLANS}.json"
    if [[ -f "${plans_json}" ]]; then
        python - "${plans_json}" "${CONFIG}" 2>/dev/null <<'PY'
import json, sys
try:
    with open(sys.argv[1]) as f:
        p = json.load(f)
    print(p["configurations"][sys.argv[2]]["data_identifier"])
except Exception:
    pass
PY
    fi
}

DATA_IDENTIFIER="$(resolve_data_identifier || true)"
if [[ -z "${DATA_IDENTIFIER}" ]]; then
    DATA_IDENTIFIER=$(basename "$(find "${NFS_PREP_DS}" -mindepth 1 -maxdepth 1 \
        -type d -name "*${CONFIG}*" 2>/dev/null | head -1)")
fi
NFS_PREP_DATA_DIR="${NFS_PREP_DS}/${DATA_IDENTIFIER}"
TMP_PREP_DATA_DIR="${TMP_CACHE_PREP_DS}/${DATA_IDENTIFIER}"
echo "[fold ${FOLD}] data_identifier = ${DATA_IDENTIFIER}"

# =============================================================================
# /tmp shared-cache setup with progress polling
# =============================================================================

setup_tmp_cache() {
    # Returns 0 on success (cache ready at TMP_CACHE_PREP_DS).
    # Returns 1 on failure (caller should fall back to NFS-direct).
    local lock_t0=$(date +%s)

    # ── Acquire lock; wait up to 60 minutes for another fold to finish unpack
    exec 200>"${TMP_CACHE_LOCK}"
    if ! flock -w 3600 -x 200; then
        echo "[fold ${FOLD}] /tmp cache: failed to acquire lock within 60 min; falling back" >&2
        return 1
    fi
    local lock_acquired_at=$(date +%s)
    local wait_s=$((lock_acquired_at - lock_t0))
    if [[ ${wait_s} -gt 5 ]]; then
        echo "[fold ${FOLD}] /tmp cache: waited ${wait_s}s for lock"
    fi

    # ── Force rebuild?
    if [[ "${SPINESURG_FORCE_DATAPREP}" == "1" ]]; then
        echo "[fold ${FOLD}] /tmp cache: SPINESURG_FORCE_DATAPREP=1 — wiping ${SPINESURG_CACHE_ROOT_TMP}"
        rm -rf "${SPINESURG_CACHE_ROOT_TMP}"
    fi

    # ── Marker present and source unchanged?
    if [[ -f "${TMP_CACHE_MARKER}" ]]; then
        local newer_npz
        newer_npz=$(find "${NFS_PREP_DATA_DIR}" -maxdepth 1 -name '*.npz' \
            -newer "${TMP_CACHE_MARKER}" 2>/dev/null | head -1)
        if [[ -z "${newer_npz}" ]]; then
            local age=$(( $(date +%s) - $(stat -c %Y "${TMP_CACHE_MARKER}" 2>/dev/null || echo 0) ))
            local size=$(du -sh "${SPINESURG_CACHE_ROOT_TMP}" 2>/dev/null | awk '{print $1}')
            local n_npy=$(find "${TMP_PREP_DATA_DIR}" -maxdepth 1 -name '*.npy' 2>/dev/null | wc -l)
            echo "[fold ${FOLD}] /tmp cache: HIT (marker ${age}s old, ${n_npy} .npy, ${size} total)"
            flock -u 200
            return 0
        else
            echo "[fold ${FOLD}] /tmp cache: marker stale (NFS source has newer .npz); rebuilding"
            rm -rf "${SPINESURG_CACHE_ROOT_TMP}"
        fi
    fi

    echo "[fold ${FOLD}] /tmp cache: building at ${SPINESURG_CACHE_ROOT_TMP}"

    # ── Disk space check
    local tmp_free_gb
    tmp_free_gb=$(df --output=avail -BG /tmp 2>/dev/null | tail -1 | tr -d 'G ')
    if [[ -n "${tmp_free_gb}" && ${tmp_free_gb} -lt 1000 ]]; then
        echo "[fold ${FOLD}] /tmp cache: WARNING /tmp has only ${tmp_free_gb}G free; need ~830G"
        if [[ ${tmp_free_gb} -lt 600 ]]; then
            echo "[fold ${FOLD}] /tmp cache: too little space; aborting cache setup" >&2
            flock -u 200
            return 1
        fi
    fi

    local build_t0=$(date +%s)

    # ── Build cache directory structure
    mkdir -p "${TMP_PREP_DATA_DIR}"

    # ── Symlink dataset-level metadata files (small, read once)
    local meta
    for meta in dataset.json dataset_fingerprint.json splits_final.json \
                 lstv_cases.json "${PLANS}.json" "nnUNetPlans.json"; do
        if [[ -f "${NFS_PREP_DS}/${meta}" ]]; then
            ln -sfn "${NFS_PREP_DS}/${meta}" "${TMP_CACHE_PREP_DS}/${meta}"
        fi
    done
    # Symlink gt_segmentations directory if present
    if [[ -d "${NFS_PREP_DS}/gt_segmentations" ]]; then
        ln -sfn "${NFS_PREP_DS}/gt_segmentations" "${TMP_CACHE_PREP_DS}/gt_segmentations"
    fi

    # ── Symlink .npz and .pkl files into cache (read once during unpack)
    local n_link=0
    local f bn
    for f in "${NFS_PREP_DATA_DIR}"/*.npz "${NFS_PREP_DATA_DIR}"/*.pkl; do
        [[ -e "${f}" ]] || continue
        bn=$(basename "${f}")
        ln -sfn "${f}" "${TMP_PREP_DATA_DIR}/${bn}"
        n_link=$((n_link + 1))
    done
    echo "[fold ${FOLD}] /tmp cache: symlinked ${n_link} .npz/.pkl files"

    # ── Run unpack into cache (writes real .npy files to local NVMe)
    # Background process with 30s progress polling.
    local n_npz_total
    n_npz_total=$(find "${TMP_PREP_DATA_DIR}" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)
    echo "[fold ${FOLD}] /tmp cache: unpacking ${n_npz_total} cases with ${PREUNPACK_WORKERS} workers"
    echo "[fold ${FOLD}] /tmp cache: progress will be polled every 30s..."

    # Run unpack in background, log to a file so we can see exit status
    local unpack_log="${EPHEMERAL_TMP}/unpack_f${FOLD}.log"
    (
        singularity exec --bind "${TMP_PREP_DATA_DIR}:/cache:rw" "${CONTAINER}" \
            python -u -c "
from nnunetv2.training.dataloading.utils import unpack_dataset
unpack_dataset('/cache', unpack_segmentation=True,
               overwrite_existing=False, num_processes=${PREUNPACK_WORKERS})
print('[unpack] done', flush=True)
" >"${unpack_log}" 2>&1
    ) &
    local unpack_pid=$!

    # Polling loop — print progress every 30s while unpack runs
    local last_n_npy=0
    local last_check=${build_t0}
    local poll_count=0
    while kill -0 ${unpack_pid} 2>/dev/null; do
        sleep 30
        local now=$(date +%s)
        local n_npy_now
        n_npy_now=$(find "${TMP_PREP_DATA_DIR}" -maxdepth 1 -name '*.npy' 2>/dev/null | wc -l)
        local cases_done=$((n_npy_now / 2))
        local pct=0
        [[ ${n_npz_total} -gt 0 ]] && pct=$((cases_done * 100 / n_npz_total))
        local interval=$((now - last_check))
        local rate_per_min=0
        [[ ${interval} -gt 0 ]] && rate_per_min=$(( (n_npy_now - last_n_npy) * 60 / interval ))
        local files_remaining=$((n_npz_total * 2 - n_npy_now))
        local eta_min=0
        [[ ${rate_per_min} -gt 0 ]] && eta_min=$((files_remaining / rate_per_min))
        local elapsed=$((now - build_t0))
        echo "[fold ${FOLD}] /tmp cache:   progress: ${cases_done}/${n_npz_total} cases (${pct}%)  rate=${rate_per_min}/min  eta=${eta_min}min  elapsed=${elapsed}s"
        last_n_npy=${n_npy_now}
        last_check=${now}
        poll_count=$((poll_count + 1))
        # Hard ceiling: 90 minutes
        if [[ ${poll_count} -gt 180 ]]; then
            echo "[fold ${FOLD}] /tmp cache: unpack exceeded 90 min; killing" >&2
            kill -TERM ${unpack_pid} 2>/dev/null || true
            sleep 5
            kill -KILL ${unpack_pid} 2>/dev/null || true
            break
        fi
    done

    wait ${unpack_pid}
    local unpack_rc=$?
    if [[ ${unpack_rc} -ne 0 ]]; then
        echo "[fold ${FOLD}] /tmp cache: unpack FAILED (rc=${unpack_rc})" >&2
        echo "[fold ${FOLD}] /tmp cache: tail of unpack log:" >&2
        tail -30 "${unpack_log}" >&2 || true
        flock -u 200
        return 1
    fi

    # ── Validate cache: count .npy files (image + seg = 2 per case usually)
    local n_npy
    n_npy=$(find "${TMP_PREP_DATA_DIR}" -maxdepth 1 -name '*.npy' 2>/dev/null | wc -l)
    local n_npz
    n_npz=$(find "${TMP_PREP_DATA_DIR}" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)
    if [[ ${n_npy} -lt $((n_npz * 2 - 5)) ]]; then
        # Allow small slack
        echo "[fold ${FOLD}] /tmp cache: only ${n_npy} .npy files for ${n_npz} .npz; suspicious" >&2
        flock -u 200
        return 1
    fi

    # ── Write marker (atomic) and report
    touch "${TMP_CACHE_MARKER}"
    local build_time=$(($(date +%s) - build_t0))
    local size=$(du -sh "${SPINESURG_CACHE_ROOT_TMP}" 2>/dev/null | awk '{print $1}')
    echo "[fold ${FOLD}] /tmp cache: BUILT in ${build_time}s (${n_npy} .npy, ${size} total)"

    flock -u 200
    return 0
}

# Best-effort cache cleanup at job end. Only delete if we are the last
# spine_kfold job from this user on this node.
cleanup_tmp_cache_if_last() {
    if [[ "${SPINESURG_USE_TMP_CACHE}" != "1" ]]; then
        return 0
    fi
    if [[ ! -d "${SPINESURG_CACHE_ROOT_TMP}" ]]; then
        return 0
    fi
    local node_name=$(hostname -s)
    local other_jobs
    other_jobs=$(squeue --me --noheader -h \
        --format="%i %N %j" 2>/dev/null \
        | awk -v n="${node_name}" -v me="${SLURM_JOB_ID}" \
            '$2==n && $1!=me && $3 ~ /spine_kfold/ {print $1}' | wc -l)
    if [[ ${other_jobs} -gt 0 ]]; then
        echo "[fold ${FOLD}] /tmp cache: ${other_jobs} other spine_kfold job(s) on this node; preserving cache"
        return 0
    fi
    local size=$(du -sh "${SPINESURG_CACHE_ROOT_TMP}" 2>/dev/null | awk '{print $1}')
    echo "[fold ${FOLD}] /tmp cache: last fold on node, evicting (was ${size})"
    rm -rf "${SPINESURG_CACHE_ROOT_TMP}"
}

# Cleanup trap
cleanup_on_exit() {
    rm -rf "${SINGULARITY_TMPDIR}" "${EPHEMERAL_TMP}" 2>/dev/null || true
    cleanup_tmp_cache_if_last
}
trap cleanup_on_exit EXIT

# ── Set up the cache (or fall back to NFS-direct) ───────────────────────────
USE_CACHE=0
DATA_SOURCE_LABEL="NFS direct"

if [[ "${SPINESURG_USE_TMP_CACHE}" == "1" ]]; then
    if setup_tmp_cache; then
        USE_CACHE=1
        DATA_SOURCE_LABEL="local NVMe (${TMP_CACHE_PREP_DS})"
    else
        echo "[fold ${FOLD}] /tmp cache setup failed; using NFS-direct"
        # Fall back to legacy NFS preunpack
        SCRUB_MARKER="${NFS_PREP_DATA_DIR}/.scrub_ok"
        PREUNPACK_MARKER="${NFS_PREP_DATA_DIR}/.preunpack_ok"
        if [[ ! -f "${PREUNPACK_MARKER}" ]] || [[ "${SPINESURG_FORCE_DATAPREP}" == "1" ]]; then
            echo "[fold ${FOLD}] NFS preunpack starting (${PREUNPACK_WORKERS} workers)"
            singularity exec --bind "${NFS_PREP_DATA_DIR}:/data:rw" "${CONTAINER}" \
                python -c "
from nnunetv2.training.dataloading.utils import unpack_dataset
unpack_dataset('/data', unpack_segmentation=True,
               overwrite_existing=False, num_processes=${PREUNPACK_WORKERS})
" && touch "${PREUNPACK_MARKER}" 2>/dev/null || true
        fi
    fi
else
    echo "[fold ${FOLD}] /tmp cache disabled by env var; using NFS-direct"
fi

# ── SLURM signal handling for graceful shutdown ─────────────────────────────
TRAIN_PID=""
on_usr1() {
    echo "[fold ${FOLD}] received SIGUSR1; forwarding for graceful shutdown" >&2
    if [[ -n "${TRAIN_PID}" ]] && kill -0 "${TRAIN_PID}" 2>/dev/null; then
        kill -USR1 "${TRAIN_PID}" 2>/dev/null || true
        wait "${TRAIN_PID}" 2>/dev/null || true
    fi
}
trap on_usr1 USR1

# ── W&B token ────────────────────────────────────────────────────────────────
WANDB_API_KEY=""
[[ -f "${HOME}/.wandb/token" ]] && WANDB_API_KEY="$(cat ${HOME}/.wandb/token)"

# ── Container env ────────────────────────────────────────────────────────────
export SINGULARITYENV_TMPDIR="/container_tmp"
export SINGULARITYENV_TORCHINDUCTOR_CACHE_DIR="/container_cache/torchinductor"
export SINGULARITYENV_TRITON_CACHE_DIR="/container_cache/triton"
export SINGULARITYENV_TORCHINDUCTOR_FX_GRAPH_CACHE="1"
export SINGULARITYENV_MALLOC_TRIM_THRESHOLD_=100000

# Critical: nnUNet_preprocessed points at chosen source (cache or NFS)
if [[ "${USE_CACHE}" == "1" ]]; then
    export SINGULARITYENV_nnUNet_raw="/nnunet_nfs/raw"
    export SINGULARITYENV_nnUNet_preprocessed="/tmp_prep/preprocessed"
    export SINGULARITYENV_nnUNet_results="/nnunet_nfs/results"
else
    export SINGULARITYENV_nnUNet_raw="/nnunet_nfs/raw"
    export SINGULARITYENV_nnUNet_preprocessed="/nnunet_nfs/preprocessed"
    export SINGULARITYENV_nnUNet_results="/nnunet_nfs/results"
fi

export SINGULARITYENV_nnUNet_def_n_proc="${NNUNET_EXPORT_POOL}"
export SINGULARITYENV_nnUNet_n_proc_DA="${NNUNET_DA_WORKERS}"
export SINGULARITYENV_SPINESURG_RUN_VAL_EXPORT="${SPINESURG_RUN_VAL_EXPORT}"
export SINGULARITYENV_WANDB_API_KEY="${WANDB_API_KEY}"
export SINGULARITYENV_WANDB_PROJECT="SpineSurg-CT"
export SINGULARITYENV_WANDB_RUN_ID="${DS_DIR_NAME}_${TRAINER}_${PLANS}_${WANDB_RUN_TAG}_f${FOLD}"
export SINGULARITYENV_WANDB_RUN_NAME="${DS_DIR_NAME}_${TRAINER}_${PLANS}_${WANDB_RUN_TAG}_f${FOLD}"
export SINGULARITYENV_WANDB_INIT_MAX_RETRIES="${WANDB_INIT_MAX_RETRIES}"
export SINGULARITYENV_WANDB_INIT_TIMEOUT_SEC="${WANDB_INIT_TIMEOUT_SEC}"
export SINGULARITYENV_WANDB_ALLOW_OFFLINE="${WANDB_ALLOW_OFFLINE}"
export SINGULARITYENV_LSTV_OVERSAMPLE_FRAC="${LSTV_OVERSAMPLE_FRAC}"
export SINGULARITYENV_SPINESURG_PERF="${SPINESURG_PERF}"
export SINGULARITYENV_SPINESURG_CHANNELS_LAST="${SPINESURG_CHANNELS_LAST}"
export SINGULARITYENV_SPINESURG_PROFILE="${SPINESURG_PROFILE}"
export SINGULARITYENV_SPINESURG_PROFILE_WAIT="${SPINESURG_PROFILE_WAIT}"
export SINGULARITYENV_SPINESURG_PROFILE_WARMUP="${SPINESURG_PROFILE_WARMUP}"
export SINGULARITYENV_SPINESURG_PROFILE_ACTIVE="${SPINESURG_PROFILE_ACTIVE}"

export OMP_NUM_THREADS=4

# Build container binds. Add /tmp_prep bind only when cache is in use.
TRAIN_BINDS=(
    --nv
    --bind "/dev/shm:/dev/shm"
    --bind "${EPHEMERAL_TMP}:/container_tmp:rw"
    --bind "${PERSISTENT_CACHE}:/container_cache:rw"
    --bind "${PROJECT_ROOT}:/workspace"
    --bind "${NNUNET_NFS}:/nnunet_nfs"
    --bind "${HOME}/.wandb:${HOME}/.wandb"
    --bind "${WANDB_TRAINER_HOST}:${WANDB_TRAINER_CONTAINER}"
    --pwd  /workspace
)
if [[ "${USE_CACHE}" == "1" ]]; then
    TRAIN_BINDS+=( --bind "${SPINESURG_CACHE_ROOT_TMP}:/tmp_prep:ro" )
fi

CACHE_SIZE=$(du -sh "${PERSISTENT_CACHE}" 2>/dev/null | awk '{print $1}')
CACHE_FILES=$(find "${PERSISTENT_CACHE}" -type f 2>/dev/null | wc -l)
TMP_CACHE_SIZE=""
if [[ "${USE_CACHE}" == "1" ]]; then
    TMP_CACHE_SIZE=$(du -sh "${SPINESURG_CACHE_ROOT_TMP}" 2>/dev/null | awk '{print $1}')
fi
TMP_FREE=$(df -h /tmp 2>/dev/null | awk 'NR==2 {print $4}')
PROFILE_STATE=$([[ "${SPINESURG_PROFILE}" == "1" ]] \
    && echo "ENABLED (wait=${SPINESURG_PROFILE_WAIT} warmup=${SPINESURG_PROFILE_WARMUP} active=${SPINESURG_PROFILE_ACTIVE})" \
    || echo "disabled")
PERF_STATE=$([[ "${SPINESURG_PERF}" == "0" ]] && echo "disabled" || echo "ENABLED")
CHLAST_STATE=$([[ "${SPINESURG_CHANNELS_LAST}" == "1" ]] && echo "ENABLED" || echo "disabled")

echo "================================================================"
echo " 5-fold CV (/tmp shared cache)  |  fold ${FOLD}"
echo "   array job     : ${SLURM_ARRAY_JOB_ID:-?}"
echo "   array task    : ${SLURM_ARRAY_TASK_ID:-?}"
echo "   slurm job     : ${SLURM_JOB_ID:-?}"
echo "   restart_count : ${SLURM_RESTART_COUNT:-0}"
echo "   node          : $(hostname)"
echo "   gpu           : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "   cpus alloc    : ${SLURM_CPUS_PER_TASK:-?}"
echo "   trainer       : ${TRAINER}"
echo "   plans         : ${PLANS}"
echo "   LSTV frac     : ${LSTV_OVERSAMPLE_FRAC}"
echo "   wandb run id  : ${SINGULARITYENV_WANDB_RUN_ID}"
echo "   wandb tag     : ${WANDB_RUN_TAG}"
echo "   data source   : ${DATA_SOURCE_LABEL}"
[[ -n "${TMP_CACHE_SIZE}" ]] && \
echo "   /tmp cache    : ${SPINESURG_CACHE_ROOT_TMP} (${TMP_CACHE_SIZE}; /tmp free=${TMP_FREE})"
echo "   workers       : preunpack=${PREUNPACK_WORKERS}  da=${NNUNET_DA_WORKERS}  export=${NNUNET_EXPORT_POOL}"
echo "   compile cache : ${PERSISTENT_CACHE} (${CACHE_FILES} files, ${CACHE_SIZE:-0})"
echo "   perf tuning   : ${PERF_STATE}"
echo "   channels_last : ${CHLAST_STATE}"
echo "   profiler      : ${PROFILE_STATE}"
echo "   /dev/shm      : host-bound ($(df -h /dev/shm 2>/dev/null | awk 'NR==2 {print $2 " total, " $4 " avail"}'))"
echo "   status        : $([[ -n ${RESUME_FLAG} ]] && echo 'RESUMING' || echo 'fresh start')"
echo "   started       : $(date)"
echo "================================================================"

singularity exec "${TRAIN_BINDS[@]}" "${CONTAINER}" \
    nnUNetv2_train ${DATASET_ID} ${CONFIG} ${FOLD} \
        -tr ${TRAINER} \
        -p  ${PLANS} \
        --npz \
        ${RESUME_FLAG} &
TRAIN_PID=$!
echo "[fold ${FOLD}] training pid=${TRAIN_PID}"

wait "${TRAIN_PID}"
TRAIN_EXIT=$?

if [[ -f "${FINAL_CKPT}" ]]; then
    echo "[fold ${FOLD}] DONE  $(date)  (exit=${TRAIN_EXIT})"
    exit 0
fi

if [[ -n "${SLURM_RESTART_COUNT:-}" || "${TRAIN_EXIT}" -eq 130 || "${TRAIN_EXIT}" -eq 143 ]]; then
    echo "[fold ${FOLD}] training did not complete (exit=${TRAIN_EXIT}); SLURM will requeue."
    exit 1
fi
echo "WARN: training exited (code=${TRAIN_EXIT}) but checkpoint_final.pth missing." >&2
echo "      To resume from checkpoint_latest.pth:" >&2
echo "        sbatch --array=${FOLD} slurm/spine_train_array.sh" >&2
echo "      To force-restart fold ${FOLD} from scratch:" >&2
echo "        rm ${RESULT_DIR}/checkpoint_*.pth" >&2
echo "        rm -rf ${PERSISTENT_CACHE_ROOT}/*_f${FOLD}" >&2
echo "        sbatch --array=${FOLD} slurm/spine_train_array.sh" >&2
exit 2
