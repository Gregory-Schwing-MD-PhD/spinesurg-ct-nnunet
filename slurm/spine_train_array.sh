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
# SpineSurg-CT  --  Stage B (5-fold CV training, NFS-direct, FINAL)
# =============================================================================
#
# Decision summary
# ----------------
# After extensive profiling, this is the production config:
#
#   - NFS-direct training (no /tmp staging). Investigated, rejected:
#     * Staging at observed NFS rate (~21 MB/s) takes 12+ hours.
#     * .npz files are ~500 MB each (874 cases ~ 437 GB), too big for
#       compressed-mode (--use_compressed) reads to win over .npy mmap.
#     * Page-cache warming after epoch 0 closes most of the gap anyway.
#
#   - 24 DA workers (matches cpus-per-task)
#   - cudnn.benchmark + TF32 (small free wins)
#   - Default torch.compile (mode='default'; reduce-overhead breaks val)
#   - LSTV oversampling 25% target (lowered from 50% in v8 to reduce
#     per-case duplication factor from ~20x to ~9x; lower overfitting risk)
#   - Per-fold persistent compile cache (survives requeues)
#   - SKIP-IF-DONE markers for scrub + preunpack
#
# Quick reference
# ---------------
# Standard production launch (all 5 folds):
#     sbatch slurm/spine_train_array.sh
#
# Single fold:
#     sbatch --array=0 slurm/spine_train_array.sh
#
# Override trainer:
#     TRAINER=nnUNetTrainerWandB_1000ep_LSTVOversample sbatch slurm/spine_train_array.sh
#
# Override LSTV oversample fraction:
#     LSTV_OVERSAMPLE_FRAC=0.5 sbatch slurm/spine_train_array.sh
#
# Override W&B run-ID tag (e.g. when iterating after a wipe):
#     WANDB_RUN_TAG=v8_strat25 sbatch slurm/spine_train_array.sh
#     -> run IDs become Dataset802_..._v8_strat25_f0 ... _f4
#     If unset, defaults to the SLURM array-job ID (always unique per
#     submission), so colliding with deleted-but-zombie cloud runs
#     doesn't happen unless the user pins a tag they've used before.
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
# Smoke test
# ----------
# To test the pipeline end-to-end with ~30 min runtime:
#   1. Backup splits:
#        cp nnunet/preprocessed/Dataset802_SpineSurgCTFull/splits_final.json{,.full}
#   2. Truncate to 20 train / 8 val per fold:
#        python3 -c "
#        import json
#        p = 'nnunet/preprocessed/Dataset802_SpineSurgCTFull/splits_final.json'
#        s = json.load(open(p+'.full'))
#        for f in s: f['train']=f['train'][:20]; f['val']=f['val'][:8]
#        json.dump(s, open(p,'w'), indent=2)
#        "
#   3. Run: TRAINER=nnUNetTrainerWandB_5epochs sbatch --array=0 slurm/spine_train_array.sh
#   4. RESTORE before production run:
#        mv nnunet/preprocessed/Dataset802_SpineSurgCTFull/splits_final.json{.full,}
#
# Expected timings (post-cache-warm)
# ----------------------------------
#   Cold epoch 0:        ~550s   (cache cold + compile)
#   Warm epoch 1+:       ~280-400s (single fold, no contention)
#   With 5-fold contention: 1.3-1.8x slower -> ~400-700s
#   Per fold:            ~3-4 days
#   All 5 folds parallel: ~3-4 days (NFS contention partly offset by
#                                     staggered per-node page caches)
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

# -- Per-fold persistent compile cache (small; lives in $HOME) ---------------
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
export LSTV_OVERSAMPLE_FRAC="${LSTV_OVERSAMPLE_FRAC:-0.25}"

# W&B run ID/name tag.
#
# Why a tag exists at all: WANDB_RUN_ID is the cloud-side primary key for a
# run. A deterministic ID (dataset__trainer__plans__fold) means every rerun
# RESUMES the same run -- which seems great until you delete a run on the
# cloud, after which the next launch's `wandb.init(id=..., resume="allow")`
# can hang for several minutes hitting cloud-side cleanup state. It also
# means that if the previous run wrote progress.json up to step N, the new
# run rejects step writes <N as "non-monotonic" until the new training
# catches up.
#
# Solution: append a tag to the run ID. Default tag = SLURM_ARRAY_JOB_ID,
# which is always unique per submission and shared across all 5 folds in
# the same array (so the 5 runs are visually grouped in the W&B UI).
#
# To use a human-friendly tag (e.g. when re-running after wiping the W&B
# project), pass WANDB_RUN_TAG explicitly:
#     WANDB_RUN_TAG=v8_strat25 sbatch slurm/spine_train_array.sh
WANDB_RUN_TAG="${WANDB_RUN_TAG:-${ARRAY_JOB}}"
export WANDB_RUN_TAG

# Perf tuning toggle (cudnn.benchmark + TF32 only)
export SPINESURG_PERF="${SPINESURG_PERF:-1}"

# Channels-last toggle (off by default; empirically slower on this model)
export SPINESURG_CHANNELS_LAST="${SPINESURG_CHANNELS_LAST:-0}"

# Force re-run of dataprep steps even if markers exist
export SPINESURG_FORCE_DATAPREP="${SPINESURG_FORCE_DATAPREP:-0}"

# Profiler toggles (opt-in)
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

# Marker files for skip-if-done logic
SCRUB_MARKER="${NFS_PREP_DATA_DIR}/.scrub_ok"
PREUNPACK_MARKER="${NFS_PREP_DATA_DIR}/.preunpack_ok"

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
    newer=$(find "${NFS_PREP_DATA_DIR}" -maxdepth 1 -name '*.npz' \
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

# ----- Scrub corrupt .npy cache ---------------------------------------------
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

scrub_npy_cache    "${NFS_PREP_DATA_DIR}"
preunpack_dataset  "${NFS_PREP_DATA_DIR}" "${PREUNPACK_WORKERS}"

# ----- W&B token ------------------------------------------------------------
WANDB_API_KEY=""
[[ -f "${HOME}/.wandb/token" ]] && WANDB_API_KEY="$(cat ${HOME}/.wandb/token)"

# ----- Container env --------------------------------------------------------
export SINGULARITYENV_TMPDIR="/container_tmp"
export SINGULARITYENV_TORCHINDUCTOR_CACHE_DIR="/container_cache/torchinductor"
export SINGULARITYENV_TRITON_CACHE_DIR="/container_cache/triton"
export SINGULARITYENV_TORCHINDUCTOR_FX_GRAPH_CACHE="1"
export SINGULARITYENV_MALLOC_TRIM_THRESHOLD_=100000

export SINGULARITYENV_nnUNet_raw="/nnunet_nfs/raw"
export SINGULARITYENV_nnUNet_preprocessed="/nnunet_nfs/preprocessed"
export SINGULARITYENV_nnUNet_results="/nnunet_nfs/results"
export SINGULARITYENV_nnUNet_def_n_proc="${NNUNET_EXPORT_POOL}"
export SINGULARITYENV_nnUNet_n_proc_DA="${NNUNET_DA_WORKERS}"
export SINGULARITYENV_SPINESURG_RUN_VAL_EXPORT="${SPINESURG_RUN_VAL_EXPORT}"
export SINGULARITYENV_WANDB_API_KEY="${WANDB_API_KEY}"
export SINGULARITYENV_WANDB_PROJECT="SpineSurg-CT"
# W&B run ID/name now incorporate WANDB_RUN_TAG so reruns don't collide
# with deleted/zombie cloud runs. See the WANDB_RUN_TAG comment block
# above for the full rationale.
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

CACHE_SIZE=$(du -sh "${PERSISTENT_CACHE}" 2>/dev/null | awk '{print $1}')
CACHE_FILES=$(find "${PERSISTENT_CACHE}" -type f 2>/dev/null | wc -l)
PROFILE_STATE=$([[ "${SPINESURG_PROFILE}" == "1" ]] \
    && echo "ENABLED (wait=${SPINESURG_PROFILE_WAIT} warmup=${SPINESURG_PROFILE_WARMUP} active=${SPINESURG_PROFILE_ACTIVE})" \
    || echo "disabled")
PERF_STATE=$([[ "${SPINESURG_PERF}" == "0" ]] && echo "disabled" || echo "ENABLED")
CHLAST_STATE=$([[ "${SPINESURG_CHANNELS_LAST}" == "1" ]] && echo "ENABLED" || echo "disabled")

echo "================================================================"
echo " 5-fold CV (NFS-direct, FINAL)  |  fold ${FOLD}"
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
echo "   data source   : NFS direct (${NFS_PREP_DATA_DIR})"
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
