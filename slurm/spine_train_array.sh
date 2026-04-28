#!/bin/bash
#SBATCH --job-name=spine_kfold
#SBATCH -q gpu
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=250G
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
# SpineSurg-CT  --  Stage B (5-fold CV training, LOCAL /tmp staging, v8)
# =============================================================================
#
# v8 BREAKTHROUGH: local-disk staging
# ====================================
# Investigation revealed each compute node has a 1.7 TB local XFS
# filesystem mounted at /tmp:
#     /dev/mapper/root_vg-lv_tmp   xfs   1.7T   339G   1.4T  20%   /tmp
#
# This is local NVMe, not network. Random patch reads from a 256x320x320
# 3D volume are 5-10x faster from local XFS than from NFS, and survive
# the eviction patterns that hurt page cache once 5 folds compete.
#
# v8 stages the preprocessed data dir to /tmp once per node and binds
# it into the container. If multiple folds land on the same node, they
# share the staged copy (mutex via flock to avoid race-staging).
#
# Disk math:
#   80 GB preprocessed dataset
#   5 folds * 80 GB = 400 GB worst case
#   Available: 1.4 TB
#   Per-fold overhead is dominated by per-job tmpdir (~few GB), trivial
#
# Staging cost:
#   First fold on a node: rsync ~80 GB once from NFS to /tmp.
#   At ~1 GB/s effective rsync throughput, ~90 seconds.
#   Subsequent folds on same node: wait for staging, then 0 cost.
#   Eliminates: ~250-500 ms/iter dataloader gap from NFS.
#   Over 500 epochs * 250 iter = 125k iters -> saves 30-60k seconds (8-17 hrs).
#
# Cleanup:
#   Staged data persists in /tmp across requeues (good -- no re-stage
#   on resume). Optionally cleared at job end via STAGED_CLEANUP=1.
#   Otherwise lives until node reboot or admin cleanup.
#
# To disable staging and revert to NFS-direct:
#   SPINESURG_STAGE=0 sbatch slurm/spine_train_array.sh
#
# =============================================================================
#
# Default trainer: nnUNetTrainerWandB_500ep_LSTVOversample
#   500 epochs * 250 iter/epoch = 125k training steps per fold.
#
# Quick reference
# ---------------
# Standard production launch (all 5 folds, staged):
#     sbatch slurm/spine_train_array.sh
#
# Disable staging (fall back to NFS):
#     SPINESURG_STAGE=0 sbatch slurm/spine_train_array.sh
#
# Profile (capture step CPU+CUDA, ~3min profile window):
#     SPINESURG_PROFILE=1 sbatch --array=0 slurm/spine_train_array.sh
#
# Disable v6 perf tuning (rollback path):
#     SPINESURG_PERF=0 sbatch slurm/spine_train_array.sh
#
# Force re-run dataprep markers (e.g., dataset changed):
#     SPINESURG_FORCE_DATAPREP=1 sbatch slurm/spine_train_array.sh
#
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

# -- Per-fold persistent compile cache (still on home; small) ----------------
PERSISTENT_CACHE_ROOT="${SPINESURG_CACHE_ROOT:-${HOME}/.cache/spinesurg}"
PERSISTENT_CACHE="${PERSISTENT_CACHE_ROOT}/${ARRAY_JOB}_f${FOLD}"
if ! mkdir -p "${PERSISTENT_CACHE}" 2>/dev/null; then
    PERSISTENT_CACHE="/tmp/${USER}_spinesurg_${ARRAY_JOB}_f${FOLD}"
    mkdir -p "${PERSISTENT_CACHE}"
fi

EPHEMERAL_TMP="/tmp/${USER}_${SLURM_JOB_ID}_f${FOLD}_tmp"
mkdir -p "${EPHEMERAL_TMP}"
trap 'rm -rf "${SINGULARITY_TMPDIR}" "${EPHEMERAL_TMP}"' EXIT

# -- Tunables ----------------------------------------------------------------
export SPINESURG_RUN_VAL_EXPORT="${SPINESURG_RUN_VAL_EXPORT:-0}"
export WANDB_INIT_MAX_RETRIES="${WANDB_INIT_MAX_RETRIES:-5}"
export WANDB_INIT_TIMEOUT_SEC="${WANDB_INIT_TIMEOUT_SEC:-180}"
export WANDB_ALLOW_OFFLINE="${WANDB_ALLOW_OFFLINE:-1}"
export LSTV_OVERSAMPLE_FRAC="${LSTV_OVERSAMPLE_FRAC:-0.5}"

# v8: local /tmp staging (default ON)
export SPINESURG_STAGE="${SPINESURG_STAGE:-1}"

# Whether to remove staged dir at job end. Default OFF -- staging is
# expensive and resumes benefit from existing staged data.
export STAGED_CLEANUP="${STAGED_CLEANUP:-0}"

# Other toggles
export SPINESURG_PERF="${SPINESURG_PERF:-1}"
export SPINESURG_CHANNELS_LAST="${SPINESURG_CHANNELS_LAST:-0}"
export SPINESURG_FORCE_DATAPREP="${SPINESURG_FORCE_DATAPREP:-0}"
export SPINESURG_PROFILE="${SPINESURG_PROFILE:-0}"
export SPINESURG_PROFILE_WAIT="${SPINESURG_PROFILE_WAIT:-5}"
export SPINESURG_PROFILE_WARMUP="${SPINESURG_PROFILE_WARMUP:-3}"
export SPINESURG_PROFILE_ACTIVE="${SPINESURG_PROFILE_ACTIVE:-10}"

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

# ----- Idempotency ----------------------------------------------------------
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

TRAIN_PID=""
on_usr1() {
    echo "[fold ${FOLD}] received SIGUSR1; forwarding for graceful shutdown" >&2
    if [[ -n "${TRAIN_PID}" ]] && kill -0 "${TRAIN_PID}" 2>/dev/null; then
        kill -USR1 "${TRAIN_PID}" 2>/dev/null || true
        wait "${TRAIN_PID}" 2>/dev/null || true
    fi
}
trap on_usr1 USR1

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
echo "[fold ${FOLD}] data_identifier = ${DATA_IDENTIFIER}"

# Marker files for skip-if-done logic (live with the data, so they
# move with it when staged)
SCRUB_MARKER_NAME=".scrub_ok"
PREUNPACK_MARKER_NAME=".preunpack_ok"

# =============================================================================
# v8: LOCAL /tmp STAGING
# =============================================================================
#
# Staging directory layout (on each compute node):
#
#   /tmp/${USER}_spinesurg_stage/
#       ${DS_DIR_NAME}/
#           ${DATA_IDENTIFIER}/   <- the data dir, copied from NFS
#               *.npz, *.npy, *.pkl, etc.
#           lstv_cases.json       <- copied for trainer consumption
#           splits_final.json     <- copied for trainer consumption
#           dataset.json
#           dataset_fingerprint.json
#           ${PLANS}.json
#       .stage_lock               <- flock target for cross-fold mutex
#       .stage_complete           <- written when staging finishes
#
# All folds on the same node share the staging directory. The first
# fold to start does the actual rsync; others wait on the lock.
# =============================================================================

STAGE_ROOT="/tmp/${USER}_spinesurg_stage"
STAGE_DS_ROOT="${STAGE_ROOT}/${DS_DIR_NAME}"
STAGE_DATA_DIR="${STAGE_DS_ROOT}/${DATA_IDENTIFIER}"
STAGE_LOCK="${STAGE_ROOT}/.stage_lock"
STAGE_COMPLETE="${STAGE_ROOT}/.stage_complete"

# Effective preprocessed dataset path (NFS by default; flipped to
# stage path after successful staging)
PREP_DS_DIR_FOR_TRAINING="${NFS_PREP_DS}"
PREP_DATA_DIR_FOR_TRAINING="${NFS_PREP_DATA_DIR}"

stage_to_local() {
    if [[ "${SPINESURG_STAGE}" != "1" ]]; then
        echo "[fold ${FOLD}] staging: DISABLED via SPINESURG_STAGE=0; using NFS direct"
        return 0
    fi

    mkdir -p "${STAGE_ROOT}"
    chmod 700 "${STAGE_ROOT}" 2>/dev/null || true

    # Acquire exclusive lock for staging. The first fold rsyncs;
    # subsequent folds wait, then see the marker and skip.
    # Lock timeout: 10 min (well above ~90s rsync time).
    echo "[fold ${FOLD}] staging: acquiring lock at ${STAGE_LOCK}"
    local t0_lock=$(date +%s)
    exec {STAGE_LOCK_FD}>"${STAGE_LOCK}"
    if ! flock -w 600 "${STAGE_LOCK_FD}"; then
        echo "[fold ${FOLD}] staging: failed to acquire lock within 600s; falling back to NFS" >&2
        eval "exec ${STAGE_LOCK_FD}>&-" 2>/dev/null
        return 1
    fi
    echo "[fold ${FOLD}] staging: lock acquired after $(($(date +%s) - t0_lock))s"

    # Check if another fold already staged. We use a dataset+plans-aware
    # marker so a stale marker from a different dataset won't fool us.
    local expected_marker="${STAGE_DS_ROOT}/.stage_complete_${DATA_IDENTIFIER}"
    if [[ -f "${expected_marker}" ]]; then
        # Verify the data actually exists too (paranoia)
        local n_npz_staged
        n_npz_staged=$(find "${STAGE_DATA_DIR}" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)
        local n_npz_nfs
        n_npz_nfs=$(find "${NFS_PREP_DATA_DIR}" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)
        if [[ "${n_npz_staged}" -eq "${n_npz_nfs}" && "${n_npz_staged}" -gt 0 ]]; then
            local age
            age=$(( $(date +%s) - $(stat -c %Y "${expected_marker}" 2>/dev/null) ))
            echo "[fold ${FOLD}] staging: REUSING existing stage (${n_npz_staged} .npz files, ${age}s old)"
            PREP_DS_DIR_FOR_TRAINING="${STAGE_DS_ROOT}"
            PREP_DATA_DIR_FOR_TRAINING="${STAGE_DATA_DIR}"
            eval "exec ${STAGE_LOCK_FD}>&-" 2>/dev/null
            return 0
        else
            echo "[fold ${FOLD}] staging: marker present but file count mismatch "
            echo "    staged=${n_npz_staged} nfs=${n_npz_nfs}; will re-stage"
            rm -f "${expected_marker}"
        fi
    fi

    # Pre-flight: do we have enough disk space?
    local nfs_size_gb
    nfs_size_gb=$(du -sBG --max-depth=0 "${NFS_PREP_DATA_DIR}" 2>/dev/null | awk '{print $1}' | tr -d 'G')
    local tmp_avail_gb
    tmp_avail_gb=$(df -BG /tmp | awk 'NR==2 {print $4}' | tr -d 'G')
    echo "[fold ${FOLD}] staging: NFS dataset ~${nfs_size_gb}G, /tmp avail ~${tmp_avail_gb}G"
    if [[ -n "${nfs_size_gb}" && -n "${tmp_avail_gb}" && "${tmp_avail_gb}" -lt $(( nfs_size_gb + 50 )) ]]; then
        echo "[fold ${FOLD}] staging: insufficient /tmp space (${tmp_avail_gb}G < ${nfs_size_gb}G + 50G margin); falling back to NFS" >&2
        eval "exec ${STAGE_LOCK_FD}>&-" 2>/dev/null
        return 1
    fi

    # Stage the data dir
    mkdir -p "${STAGE_DS_ROOT}"
    echo "[fold ${FOLD}] staging: rsync ${NFS_PREP_DATA_DIR} -> ${STAGE_DATA_DIR}"
    local t0_rsync=$(date +%s)
    if ! rsync -a --info=progress2 --no-i-r \
            "${NFS_PREP_DATA_DIR}/" "${STAGE_DATA_DIR}/"; then
        echo "[fold ${FOLD}] staging: data dir rsync FAILED; falling back to NFS" >&2
        eval "exec ${STAGE_LOCK_FD}>&-" 2>/dev/null
        return 1
    fi
    echo "[fold ${FOLD}] staging: data dir done in $(($(date +%s) - t0_rsync))s"

    # Stage sibling files (json, splits, etc.) needed by nnU-Net + trainer
    echo "[fold ${FOLD}] staging: copying dataset metadata files"
    for f in dataset.json dataset_fingerprint.json \
             "${PLANS}.json" splits_final.json lstv_cases.json; do
        if [[ -f "${NFS_PREP_DS}/${f}" ]]; then
            cp -p "${NFS_PREP_DS}/${f}" "${STAGE_DS_ROOT}/${f}"
        fi
    done

    # Mark complete and flip to staged paths
    touch "${expected_marker}"
    PREP_DS_DIR_FOR_TRAINING="${STAGE_DS_ROOT}"
    PREP_DATA_DIR_FOR_TRAINING="${STAGE_DATA_DIR}"
    echo "[fold ${FOLD}] staging: COMPLETE -- training will read from ${STAGE_DATA_DIR}"

    eval "exec ${STAGE_LOCK_FD}>&-" 2>/dev/null
    return 0
}

# Run staging (or fall back to NFS-direct on failure)
if ! stage_to_local; then
    echo "[fold ${FOLD}] staging: fell back to NFS direct mode"
    PREP_DS_DIR_FOR_TRAINING="${NFS_PREP_DS}"
    PREP_DATA_DIR_FOR_TRAINING="${NFS_PREP_DATA_DIR}"
fi

SCRUB_MARKER="${PREP_DATA_DIR_FOR_TRAINING}/${SCRUB_MARKER_NAME}"
PREUNPACK_MARKER="${PREP_DATA_DIR_FOR_TRAINING}/${PREUNPACK_MARKER_NAME}"

# ----- Skip-if-done helper --------------------------------------------------
should_skip_dataprep() {
    local marker="$1"
    local label="$2"
    if [[ "${SPINESURG_FORCE_DATAPREP}" == "1" ]]; then
        echo "[fold ${FOLD}] ${label}: SPINESURG_FORCE_DATAPREP=1, will run"
        return 1
    fi
    if [[ ! -f "${marker}" ]]; then
        return 1
    fi
    local newer
    newer=$(find "${PREP_DATA_DIR_FOR_TRAINING}" -maxdepth 1 -name '*.npz' \
        -newer "${marker}" 2>/dev/null | head -1)
    if [[ -n "${newer}" ]]; then
        echo "[fold ${FOLD}] ${label}: marker stale (data dir has newer files), will re-run"
        return 1
    fi
    local age
    age=$(( $(date +%s) - $(stat -c %Y "${marker}" 2>/dev/null || stat -f %m "${marker}") ))
    echo "[fold ${FOLD}] ${label}: SKIP (marker present, ${age}s old)"
    return 0
}

# ----- Scrub corrupt .npy cache (now operates on the EFFECTIVE path) --------
scrub_npy_cache() {
    local target_dir="$1"
    [[ ! -d "${target_dir}" ]] && return 0
    if should_skip_dataprep "${SCRUB_MARKER}" "scrub"; then
        return 0
    fi
    local n_npy
    n_npy=$(find "${target_dir}" -maxdepth 1 -name '*.npy' 2>/dev/null | wc -l)
    if [[ "${n_npy}" -eq 0 ]]; then
        echo "[fold ${FOLD}] scrub: no .npy files, marking done"
        touch "${SCRUB_MARKER}" 2>/dev/null || true
        return 0
    fi
    echo "[fold ${FOLD}] scrub: validating ${n_npy} .npy file(s)"
    local t0=$(date +%s)
    if singularity exec --bind "${target_dir}:/scrub:rw" "${CONTAINER}" \
        python -c "
import numpy as np
from pathlib import Path
bad = total = 0
for f in sorted(Path('/scrub').glob('*.npy')):
    total += 1
    try:
        a = np.load(f, mmap_mode='r')
        _ = a.shape
    except Exception as e:
        try:
            f.unlink()
            print(f'  removed: {f.name} ({type(e).__name__})', flush=True)
            bad += 1
        except FileNotFoundError:
            pass
print(f'[scrub] scanned {total}, removed {bad}', flush=True)
"; then
        touch "${SCRUB_MARKER}" 2>/dev/null || true
        echo "[fold ${FOLD}] scrub done in $(($(date +%s) - t0))s (marker written)"
    else
        echo "[fold ${FOLD}] scrub FAILED after $(($(date +%s) - t0))s; not marking" >&2
    fi
}

# ----- Pre-unpack -----------------------------------------------------------
preunpack_dataset() {
    local target_dir="$1"
    local n_workers="$2"
    [[ ! -d "${target_dir}" ]] && return 0
    if should_skip_dataprep "${PREUNPACK_MARKER}" "preunpack"; then
        return 0
    fi
    local n_npz
    n_npz=$(find "${target_dir}" -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l)
    [[ "${n_npz}" -eq 0 ]] && return 0
    local n_npy_existing
    n_npy_existing=$(find "${target_dir}" -maxdepth 1 -name '*.npy' 2>/dev/null | wc -l)
    echo "[fold ${FOLD}] preunpack: ${n_npz} .npz, ${n_npy_existing} .npy already present, ${n_workers} workers"
    local t0=$(date +%s)
    if singularity exec --bind "${target_dir}:/data:rw" "${CONTAINER}" \
        python -c "
from nnunetv2.training.dataloading.utils import unpack_dataset
unpack_dataset('/data', unpack_segmentation=True,
               overwrite_existing=False, num_processes=${n_workers})
print('[preunpack] done', flush=True)
"; then
        touch "${PREUNPACK_MARKER}" 2>/dev/null || true
        echo "[fold ${FOLD}] preunpack done in $(($(date +%s) - t0))s (marker written)"
    else
        echo "[fold ${FOLD}] preunpack FAILED after $(($(date +%s) - t0))s; nnU-Net will retry inline" >&2
    fi
}

scrub_npy_cache    "${PREP_DATA_DIR_FOR_TRAINING}"
preunpack_dataset  "${PREP_DATA_DIR_FOR_TRAINING}" "${PREUNPACK_WORKERS}"

# ----- W&B token ------------------------------------------------------------
WANDB_API_KEY=""
[[ -f "${HOME}/.wandb/token" ]] && WANDB_API_KEY="$(cat ${HOME}/.wandb/token)"

# ----- Container env --------------------------------------------------------
export SINGULARITYENV_TMPDIR="/container_tmp"
export SINGULARITYENV_TORCHINDUCTOR_CACHE_DIR="/container_cache/torchinductor"
export SINGULARITYENV_TRITON_CACHE_DIR="/container_cache/triton"
export SINGULARITYENV_TORCHINDUCTOR_FX_GRAPH_CACHE="1"
export SINGULARITYENV_MALLOC_TRIM_THRESHOLD_=100000

# v8: nnU-Net path env now points at the effective (staged or NFS) location.
# We bind the parent of PREP_DS_DIR_FOR_TRAINING -- preserving the
# Dataset802_SpineSurgCTFull directory name so nnU-Net's internal
# path computation works identically to the NFS case.
EFFECTIVE_PREPROCESSED_PARENT="$(dirname "${PREP_DS_DIR_FOR_TRAINING}")"
export SINGULARITYENV_nnUNet_raw="/nnunet_raw"
export SINGULARITYENV_nnUNet_preprocessed="/nnunet_preprocessed"
export SINGULARITYENV_nnUNet_results="/nnunet_results"

export SINGULARITYENV_nnUNet_def_n_proc="${NNUNET_EXPORT_POOL}"
export SINGULARITYENV_nnUNet_n_proc_DA="${NNUNET_DA_WORKERS}"
export SINGULARITYENV_SPINESURG_RUN_VAL_EXPORT="${SPINESURG_RUN_VAL_EXPORT}"
export SINGULARITYENV_WANDB_API_KEY="${WANDB_API_KEY}"
export SINGULARITYENV_WANDB_PROJECT="SpineSurg-CT"
export SINGULARITYENV_WANDB_RUN_ID="${DS_DIR_NAME}_${TRAINER}_${PLANS}_f${FOLD}"
export SINGULARITYENV_WANDB_RUN_NAME="${DS_DIR_NAME}_${TRAINER}_${PLANS}_f${FOLD}"
export SINGULARITYENV_WANDB_INIT_MAX_RETRIES="${WANDB_INIT_MAX_RETRIES}"
export SINGULARITYENV_WANDB_INIT_TIMEOUT_SEC="${WANDB_INIT_TIMEOUT_SEC}"
export SINGULARITYENV_WANDB_ALLOW_OFFLINE="${WANDB_ALLOW_OFFLINE}"
export SINGULARITYENV_LSTV_OVERSAMPLE_FRAC="${LSTV_OVERSAMPLE_FRAC}"
export SINGULARITYENV_SPINESURG_PERF="${SPINESURG_PERF}"
export SINGULARITYENV_SPINESURG_CHANNELS_LAST="${SPINESURG_CHANNELS_LAST}"
export SINGULARITYENV_SPINESURG_PROFILE="${SPINESURG_PROFILE}"
export SINGULARITYENV_SPINESURG_PROFILE_WAIT="${SPINESURG_PROFILE_WAIT}"
export SINGULARITYENV_SPINESURG_PROFILE_WARMUP="${SPINESURG_PROFILE_WARMUP}"
export SINGULARITYENV_SPINESURG_PROFILE_WARMUP="${SPINESURG_PROFILE_WARMUP}"
export SINGULARITYENV_SPINESURG_PROFILE_ACTIVE="${SPINESURG_PROFILE_ACTIVE}"

export OMP_NUM_THREADS=4

# v8: bind preprocessed from the EFFECTIVE path (staged or NFS).
# Raw and results are still NFS-only (raw is read rarely, results
# need to be persistent across requeues / nodes).
TRAIN_BINDS=(
    --nv
    --bind "/dev/shm:/dev/shm"
    --bind "${EPHEMERAL_TMP}:/container_tmp:rw"
    --bind "${PERSISTENT_CACHE}:/container_cache:rw"
    --bind "${PROJECT_ROOT}:/workspace"
    --bind "${NNUNET_NFS}/raw:/nnunet_raw:ro"
    --bind "${EFFECTIVE_PREPROCESSED_PARENT}:/nnunet_preprocessed:rw"
    --bind "${NNUNET_NFS}/results:/nnunet_results:rw"
    --bind "${HOME}/.wandb:${HOME}/.wandb"
    --bind "${WANDB_TRAINER_HOST}:${WANDB_TRAINER_CONTAINER}"
    --pwd  /workspace
)

CACHE_SIZE=$(du -sh "${PERSISTENT_CACHE}" 2>/dev/null | awk '{print $1}')
CACHE_FILES=$(find "${PERSISTENT_CACHE}" -type f 2>/dev/null | wc -l)
PROFILE_STATE=$([[ "${SPINESURG_PROFILE}" == "1" ]] \
    && echo "ENABLED (wait=${SPINESURG_PROFILE_WAIT} warmup=${SPINESURG_PROFILE_WARMUP} active=${SPINESURG_PROFILE_ACTIVE})" \
    || echo "disabled")
PERF_STATE=$([[ "${SPINESURG_PERF}" == "0" ]] && echo "disabled" || echo "ENABLED")
CHLAST_STATE=$([[ "${SPINESURG_CHANNELS_LAST}" == "1" ]] && echo "ENABLED" || echo "disabled")
STAGE_STATE=$([[ "${PREP_DATA_DIR_FOR_TRAINING}" == "${STAGE_DATA_DIR}" ]] && echo "LOCAL /tmp" || echo "NFS direct")

echo "================================================================"
echo " 5-fold CV (v8 LOCAL STAGING)  |  fold ${FOLD}"
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
echo "   data source   : ${STAGE_STATE}"
echo "                   ${PREP_DATA_DIR_FOR_TRAINING}"
echo "   workers       : preunpack=${PREUNPACK_WORKERS}  da=${NNUNET_DA_WORKERS}  export=${NNUNET_EXPORT_POOL}"
echo "   compile cache : ${PERSISTENT_CACHE} (${CACHE_FILES} files, ${CACHE_SIZE:-0})"
echo "   perf tuning   : ${PERF_STATE}"
echo "   channels_last : ${CHLAST_STATE}"
echo "   profiler      : ${PROFILE_STATE}"
echo "   /tmp          : $(df -h /tmp 2>/dev/null | awk 'NR==2 {print $2 " total, " $4 " avail"}')"
echo "   /dev/shm      : $(df -h /dev/shm 2>/dev/null | awk 'NR==2 {print $2 " total, " $4 " avail"}')"
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

# Optional cleanup of staged dir at job end
if [[ "${STAGED_CLEANUP}" == "1" && "${PREP_DATA_DIR_FOR_TRAINING}" == "${STAGE_DATA_DIR}" ]]; then
    echo "[fold ${FOLD}] staging: STAGED_CLEANUP=1, removing ${STAGE_DS_ROOT}"
    rm -rf "${STAGE_DS_ROOT}" 2>/dev/null || true
fi

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
