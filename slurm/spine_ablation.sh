#!/bin/bash
#SBATCH --job-name=spine_ablation
#SBATCH -q gpu
#SBATCH --array=0-5%2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=250G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/spine_abl_%a_%j.out
#SBATCH --error=logs/spine_abl_%a_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=go2432@wayne.edu
# =============================================================================
# SpineSurg-CT  --  Ablation studies (fold 0 only)
# =============================================================================
# Each array task runs ONE self-contained ablation:
#   convert  ->  preprocess  ->  train (fold 0)  ->  predict test  ->  eval
#
# All ablations use the SAME master splits_5fold.json (so the test set is
# identical across ablations and results are directly comparable). Training
# pool composition and/or trainer/planner vary per ablation.
#
# Fully idempotent. Resubmitting with identical params no-ops completed stages
# and resumes training via --c if a prior job timed out.
#
# After the array finishes, summarize:
#   python tools/compare_ablations.py --results_root nnunet/predictions
#
# Prereqs: slurm/generate_splits.sh produced data/splits_5fold.json with
# schema v3 (has token_info). slurm/spine_prep.sh may have already produced
# Dataset802 for ablations that reuse it.
#
# Array throttle %2 runs at most 2 ablations concurrently. Change to %1 for
# serial, or remove entirely. With 6 ablations x ~30-40h each, %2 gives you
# full results in about 5 days of wall clock.
# =============================================================================

set -euo pipefail

ABL=${SLURM_ARRAY_TASK_ID:-0}
FOLD=0

# ----- Toggles --------------------------------------------------------------
export SPINESURG_RUN_VAL_EXPORT="${SPINESURG_RUN_VAL_EXPORT:-0}"
export WANDB_INIT_MAX_RETRIES="${WANDB_INIT_MAX_RETRIES:-5}"
export WANDB_INIT_TIMEOUT_SEC="${WANDB_INIT_TIMEOUT_SEC:-180}"
export WANDB_ALLOW_OFFLINE="${WANDB_ALLOW_OFFLINE:-1}"

# ----- Common paths ---------------------------------------------------------
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-${HOME}/SpineSurg-CT}"
HF_EXPORT_NFS="${PROJECT_ROOT}/data/hf_export"
SPLITS_FILE_HOST="${PROJECT_ROOT}/data/splits_5fold.json"
NNUNET_NFS="${PROJECT_ROOT}/nnunet"
CONTAINER="${PROJECT_ROOT}/containers/spinesurg-ct.sif"
WANDB_TRAINER_HOST="${PROJECT_ROOT}/tools/nnunet_wandb_variant.py"
WANDB_TRAINER_CONTAINER="/opt/conda/lib/python3.11/site-packages/nnunetv2/training/nnUNetTrainer/variants/nnunet_wandb_variant.py"

CONFIG="3d_fullres"
NNUNET_EXPORT_POOL=12
NNUNET_DA_WORKERS=12
N_PREP_THREADS=24

# =============================================================================
# Ablation matrix
# =============================================================================
# Defaults (overridden per-ablation below):
DATASET_ID=802
DATASET_NAME="SpineSurgCTFull"
TRAINER="nnUNetTrainerWandB_1000ep_LSTVOversample"
PLANNER="nnUNetPlannerResEncM"
GPU_MEMORY_TARGET_GB=100
INCLUDE_TRAIN_MATCH_TYPES=""
TEST_MATCH_TYPES=""
export LSTV_OVERSAMPLE_FRAC="${LSTV_OVERSAMPLE_FRAC:-0.5}"

case ${ABL} in
    0)  # fused_only: train on fused only
        TAG="fused_only"
        DATASET_ID=803
        DATASET_NAME="SpineSurgCTFusedOnly"
        INCLUDE_TRAIN_MATCH_TYPES="fused"
        ;;
    1)  # no_partial: train on fused + separate (drop spine_only + pelvic_only)
        TAG="no_partial"
        DATASET_ID=804
        DATASET_NAME="SpineSurgCTNoPartial"
        INCLUDE_TRAIN_MATCH_TYPES="fused,separate"
        ;;
    2)  # no_lstv_oversample: same data as main, but no LSTV oversampling
        TAG="no_lstv_oversample"
        DATASET_ID=802           # reuse main dataset
        DATASET_NAME="SpineSurgCTFull"
        TRAINER="nnUNetTrainerWandB_1000ep_500iter"
        export LSTV_OVERSAMPLE_FRAC=0.0
        ;;
    3)  # short_100ep sanity: same everything but just 100 epochs
        TAG="short_100ep"
        DATASET_ID=802
        DATASET_NAME="SpineSurgCTFull"
        TRAINER="nnUNetTrainerWandB_100epochs"
        ;;
    4)  # resenc_l: bigger encoder, same data
        TAG="resenc_l"
        DATASET_ID=802
        DATASET_NAME="SpineSurgCTFull"
        TRAINER="nnUNetTrainerWandB_1000ep_LSTVOversample"
        PLANNER="nnUNetPlannerResEncL"
        ;;
    5)  # separate_in_training_test_fused_only:
        # Train fused+separate, test ONLY on fused cases.
        # Isolates cross-series contamination effect at inference.
        TAG="testfusedonly"
        DATASET_ID=805
        DATASET_NAME="SpineSurgCTTestFusedOnly"
        INCLUDE_TRAIN_MATCH_TYPES="fused,separate"
        TEST_MATCH_TYPES="fused"
        ;;
    *)
        echo "ERROR: Unknown ABL=${ABL}" >&2
        exit 1
        ;;
esac

# Compute plans name from planner
case "${PLANNER}" in
    nnUNetPlannerResEncM)  PLANS="nnUNetResEncUNetPlans_${GPU_MEMORY_TARGET_GB}G"  ;;
    nnUNetPlannerResEncL)  PLANS="nnUNetResEncUNetLPlans_${GPU_MEMORY_TARGET_GB}G" ;;
    nnUNetPlannerResEncXL) PLANS="nnUNetResEncUNetXLPlans_${GPU_MEMORY_TARGET_GB}G";;
    ExperimentPlanner)     PLANS="nnUNetPlans"                                    ;;
    *)                     echo "Unknown planner ${PLANNER}"; exit 1              ;;
esac

DS_DIR_NAME="Dataset$(printf '%03d' ${DATASET_ID})_${DATASET_NAME}"

# ----- Preflight ------------------------------------------------------------
[[ ! -f "${CONTAINER}"         ]] && { echo "ERROR: container missing" >&2; exit 1; }
[[ ! -f "${SPLITS_FILE_HOST}"  ]] && {
    echo "ERROR: ${SPLITS_FILE_HOST} missing. Run slurm/generate_splits.sh first." >&2
    exit 1; }
[[ ! -d "${HF_EXPORT_NFS}/ct"  ]] && { echo "ERROR: HF export missing" >&2; exit 1; }

# Verify splits file has token_info (schema v3+)
SCHEMA=$(python3 -c "import json; print(json.load(open('${SPLITS_FILE_HOST}')).get('schema_version', 0))" 2>/dev/null || echo 0)
if [[ "${SCHEMA}" -lt 3 ]] && { [[ -n "${INCLUDE_TRAIN_MATCH_TYPES}" ]] || [[ -n "${TEST_MATCH_TYPES}" ]]; }; then
    echo "ERROR: splits file is schema v${SCHEMA}; ablations need v3+ (token_info)." >&2
    echo "       Regenerate: OVERWRITE=1 sbatch slurm/generate_splits.sh" >&2
    exit 1
fi

WANDB_API_KEY=""
[[ -f "${HOME}/.wandb/token" ]] && WANDB_API_KEY="$(cat ${HOME}/.wandb/token)"

export SINGULARITY_TMPDIR="/tmp/${USER}_job_${SLURM_JOB_ID}_abl${ABL}_singularity"
export XDG_RUNTIME_DIR="${SINGULARITY_TMPDIR}/runtime"
mkdir -p "${SINGULARITY_TMPDIR}" "${XDG_RUNTIME_DIR}"
trap "rm -rf ${SINGULARITY_TMPDIR}" EXIT

echo "================================================================"
echo " Ablation ${ABL}: ${TAG}"
echo "   Dataset        : ${DS_DIR_NAME}"
echo "   Trainer        : ${TRAINER}"
echo "   Planner        : ${PLANNER}"
echo "   Plans          : ${PLANS}"
echo "   Train filter   : ${INCLUDE_TRAIN_MATCH_TYPES:-<all>}"
echo "   Test filter    : ${TEST_MATCH_TYPES:-<all>}"
echo "   LSTV frac      : ${LSTV_OVERSAMPLE_FRAC}"
echo "   Started        : $(date)"
echo "================================================================"

# =============================================================================
# Stage 1: Convert (idempotent)
# =============================================================================
NFS_RAW_DS="${NNUNET_NFS}/raw/${DS_DIR_NAME}"

CONVERT_BINDS=(
    --nv
    --bind "${PROJECT_ROOT}:/workspace"
    --bind "${HF_EXPORT_NFS}:/data/hf_export"
    --bind "${NNUNET_NFS}:/nnunet_nfs"
    --bind "${SPLITS_FILE_HOST}:/workspace/splits_5fold.json:ro"
    --bind "${WANDB_TRAINER_HOST}:${WANDB_TRAINER_CONTAINER}"
    --pwd  /workspace
)

if [[ -f "${NFS_RAW_DS}/dataset.json" ]]; then
    echo "[${TAG}] dataset.json present; skipping convert."
else
    echo "[${TAG}] running convert..."
    CONVERT_ARGS=(
        --hf_export_dir  /data/hf_export
        --nnunet_raw_dir /nnunet_nfs/raw
        --dataset_id     ${DATASET_ID}
        --dataset_name   ${DATASET_NAME}
        --splits_file    /workspace/splits_5fold.json
    )
    [[ -n "${INCLUDE_TRAIN_MATCH_TYPES}" ]] && \
        CONVERT_ARGS+=( --include_train_match_types "${INCLUDE_TRAIN_MATCH_TYPES}" )
    [[ -n "${TEST_MATCH_TYPES}" ]] && \
        CONVERT_ARGS+=( --test_match_types "${TEST_MATCH_TYPES}" )

    singularity exec "${CONVERT_BINDS[@]}" "${CONTAINER}" \
        python tools/convert_hf_to_nnunet.py "${CONVERT_ARGS[@]}"
fi

# =============================================================================
# Stage 2: Preprocess (idempotent per dataset+plans combo)
# =============================================================================
export SINGULARITYENV_nnUNet_raw="/nnunet_nfs/raw"
export SINGULARITYENV_nnUNet_preprocessed="/nnunet_nfs/preprocessed"
export SINGULARITYENV_nnUNet_results="/nnunet_nfs/results"

NFS_PREP_DS="${NNUNET_NFS}/preprocessed/${DS_DIR_NAME}"
PREP_MARKER="${NFS_PREP_DS}/.preprocessed_complete_${PLANS}"

if [[ -f "${PREP_MARKER}" ]] \
   && [[ -d "${NFS_PREP_DS}/${CONFIG}" ]] \
   && [[ -f "${NFS_PREP_DS}/${PLANS}.json" ]]; then
    echo "[${TAG}] preprocess cached; skipping."
else
    echo "[${TAG}] running preprocess..."
    mkdir -p "${NFS_PREP_DS}"; rm -f "${PREP_MARKER}"
    singularity exec "${CONVERT_BINDS[@]}" "${CONTAINER}" \
        nnUNetv2_plan_and_preprocess \
            -d ${DATASET_ID} \
            -pl ${PLANNER} \
            -gpu_memory_target ${GPU_MEMORY_TARGET_GB} \
            -overwrite_plans_name ${PLANS} \
            -c ${CONFIG} \
            -np ${N_PREP_THREADS} \
            --verify_dataset_integrity
    touch "${PREP_MARKER}"
fi

# Copy splits_final.json into preprocessed dir if not there
if [[ -f "${NFS_RAW_DS}/splits_final.json" ]] && [[ ! -f "${NFS_PREP_DS}/splits_final.json" ]]; then
    cp -f "${NFS_RAW_DS}/splits_final.json" "${NFS_PREP_DS}/splits_final.json"
fi

# =============================================================================
# Stage 3: Train (idempotent: skip / resume / fresh)
# =============================================================================
RESULT_DIR="${NNUNET_NFS}/results/${DS_DIR_NAME}/${TRAINER}__${PLANS}__${CONFIG}/fold_${FOLD}"
FINAL_CKPT="${RESULT_DIR}/checkpoint_final.pth"
LATEST_CKPT="${RESULT_DIR}/checkpoint_latest.pth"

RESUME_FLAG=""
if [[ -f "${FINAL_CKPT}" ]]; then
    echo "[${TAG}] training already complete; skipping."
else
    [[ -f "${LATEST_CKPT}" ]] && { RESUME_FLAG="--c"; echo "[${TAG}] resuming from latest checkpoint"; }

    # Stage to local scratch for training throughput
    if [[ -n "${SLURM_TMPDIR:-}" && -d "${SLURM_TMPDIR}" ]]; then
        LOCAL_SCRATCH="${SLURM_TMPDIR}"
    else
        LOCAL_SCRATCH="/tmp/${USER}_${SLURM_JOB_ID}_abl${ABL}"; mkdir -p "${LOCAL_SCRATCH}"
    fi
    LOCAL_NNUNET="${LOCAL_SCRATCH}/nnunet"
    LOCAL_PREP_DS="${LOCAL_NNUNET}/preprocessed/${DS_DIR_NAME}"
    LOCAL_RAW_DS="${LOCAL_NNUNET}/raw/${DS_DIR_NAME}"
    mkdir -p "${LOCAL_PREP_DS}" "${LOCAL_RAW_DS}"
    echo "[${TAG}] staging preprocessed + raw to local scratch..."
    rsync -aW --no-compress "${NFS_PREP_DS}/" "${LOCAL_PREP_DS}/"
    rsync -aW --no-compress "${NFS_RAW_DS}/"  "${LOCAL_RAW_DS}/"

    export SINGULARITYENV_TMPDIR="/workspace/tmp"
    export SINGULARITYENV_nnUNet_raw="/nnunet_local/raw"
    export SINGULARITYENV_nnUNet_preprocessed="/nnunet_local/preprocessed"
    export SINGULARITYENV_nnUNet_results="/nnunet_nfs/results"
    export SINGULARITYENV_nnUNet_def_n_proc="${NNUNET_EXPORT_POOL}"
    export SINGULARITYENV_nnUNet_n_proc_DA="${NNUNET_DA_WORKERS}"
    export SINGULARITYENV_SPINESURG_RUN_VAL_EXPORT="${SPINESURG_RUN_VAL_EXPORT}"
    export SINGULARITYENV_WANDB_API_KEY="${WANDB_API_KEY}"
    export SINGULARITYENV_WANDB_PROJECT="SpineSurg-CT"
    export SINGULARITYENV_WANDB_RUN_NAME="ABL_${TAG}_${DS_DIR_NAME}_${TRAINER}_${PLANS}_f${FOLD}_j${SLURM_JOB_ID}"
    export SINGULARITYENV_WANDB_RUN_ID="ABL_${TAG}_${DS_DIR_NAME}_${TRAINER}_${PLANS}_f${FOLD}"
    export SINGULARITYENV_WANDB_INIT_MAX_RETRIES="${WANDB_INIT_MAX_RETRIES}"
    export SINGULARITYENV_WANDB_INIT_TIMEOUT_SEC="${WANDB_INIT_TIMEOUT_SEC}"
    export SINGULARITYENV_WANDB_ALLOW_OFFLINE="${WANDB_ALLOW_OFFLINE}"
    export SINGULARITYENV_LSTV_OVERSAMPLE_FRAC="${LSTV_OVERSAMPLE_FRAC}"
    export OMP_NUM_THREADS=4

    TRAIN_BINDS=(
        --nv
        --bind "${PROJECT_ROOT}:/workspace"
        --bind "${LOCAL_NNUNET}:/nnunet_local"
        --bind "${NNUNET_NFS}:/nnunet_nfs"
        --bind "${HOME}/.wandb:${HOME}/.wandb"
        --bind "${WANDB_TRAINER_HOST}:${WANDB_TRAINER_CONTAINER}"
        --pwd  /workspace
    )

    singularity exec "${TRAIN_BINDS[@]}" "${CONTAINER}" \
        nnUNetv2_train ${DATASET_ID} ${CONFIG} ${FOLD} \
            -tr ${TRAINER} \
            -p  ${PLANS} \
            --npz \
            ${RESUME_FLAG}
fi

# =============================================================================
# Stage 4: Predict test set (idempotent via --continue_prediction)
# =============================================================================
PRED_DIR_NFS="${NNUNET_NFS}/predictions/ABL_${TAG}_${DS_DIR_NAME}_${TRAINER}_${PLANS}_f${FOLD}"
PRED_DIR_CONT="/nnunet_nfs/predictions/ABL_${TAG}_${DS_DIR_NAME}_${TRAINER}_${PLANS}_f${FOLD}"
mkdir -p "${PRED_DIR_NFS}"

N_EXISTING=$({ find "${PRED_DIR_NFS}" -maxdepth 1 -name "*.nii.gz" 2>/dev/null || true; } | wc -l)
N_TEST=$({ find "${NFS_RAW_DS}/imagesTs" -maxdepth 1 -name "*_0000.nii.gz" 2>/dev/null || true; } | wc -l)

if [[ "${N_EXISTING}" -ge "${N_TEST}" && "${N_TEST}" -gt 0 ]]; then
    echo "[${TAG}] all test predictions exist; skipping predict."
else
    CONTINUE_FLAG=""
    [[ "${N_EXISTING}" -gt 0 ]] && CONTINUE_FLAG="--continue_prediction"
    singularity exec "${CONVERT_BINDS[@]}" "${CONTAINER}" \
        nnUNetv2_predict \
            -i  "/nnunet_nfs/raw/${DS_DIR_NAME}/imagesTs" \
            -o  "${PRED_DIR_CONT}" \
            -d  ${DATASET_ID} \
            -c  ${CONFIG} \
            -f  ${FOLD} \
            -tr ${TRAINER} \
            -p  ${PLANS} \
            ${CONTINUE_FLAG}
fi

# =============================================================================
# Stage 5: Eval (always runs, cheap)
# =============================================================================
echo "[${TAG}] running paper metrics..."
singularity exec "${CONVERT_BINDS[@]}" "${CONTAINER}" \
    python tools/eval_paper_metrics.py \
        --predictions_dir "${PRED_DIR_CONT}" \
        --labels_dir      "/nnunet_nfs/raw/${DS_DIR_NAME}/labelsTs" \
        --output_dir      "${PRED_DIR_CONT}/paper_metrics"

# Tag the output dir with ablation metadata for later aggregation
python3 - <<PYEOF
import json
meta = {
    "ablation_tag": "${TAG}",
    "ablation_index": ${ABL},
    "dataset_id": ${DATASET_ID},
    "dataset_name": "${DATASET_NAME}",
    "trainer": "${TRAINER}",
    "planner": "${PLANNER}",
    "plans": "${PLANS}",
    "fold": ${FOLD},
    "include_train_match_types": "${INCLUDE_TRAIN_MATCH_TYPES}",
    "test_match_types": "${TEST_MATCH_TYPES}",
    "lstv_oversample_frac": ${LSTV_OVERSAMPLE_FRAC},
}
with open("${PRED_DIR_NFS}/paper_metrics/ablation_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
PYEOF

echo ""
echo "================================================================"
echo " Ablation ${ABL} (${TAG}) COMPLETE"
echo "   metrics : ${PRED_DIR_NFS}/paper_metrics/metrics_paper.md"
echo "   Done    : $(date)"
echo "================================================================"
