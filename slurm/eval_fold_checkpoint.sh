#!/bin/bash
#SBATCH --job-name=spine_evalck
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/spine_evalck_f%j.out
#SBATCH --error=logs/spine_evalck_f%j.err
#
# Evaluate a fold MID-TRAINING from a checkpoint (default checkpoint_best.pth), so a
# preliminary result exists days before checkpoint_final: predict the fold's validation
# cases with probabilities, then run tools/eval_oneshot.py (per-class Dice, rib-free
# count by transitional status, rib confusion, rare-class emission) and
# tools/decode_sequence.py (posterior over the rib-free count and the A/B/C readings).
#
#   FOLD=0 sbatch slurm/eval_fold_checkpoint.sh
#   FOLD=0 CHECKPOINT=checkpoint_latest.pth sbatch slurm/eval_fold_checkpoint.sh
#
# Reads straight from NFS (the .npy were unpacked once; prediction reads raw images only).
# Output: <fold result dir>/eval_<checkpoint>_<date>/{images,pred,metrics,decode.csv}
set -uo pipefail

export PATH="${HOME}/mambaforge/envs/nextflow/bin:${PATH}"   # singularity 3.8.6 works on the compute nodes

FOLD="${FOLD:?set FOLD}"
CHECKPOINT="${CHECKPOINT:-checkpoint_best.pth}"
DATASET_ID="${DATASET_ID:-810}"
DATASET_NAME="${DATASET_NAME:-SpineSurgLSTVOneShot}"
CONFIG="${CONFIG:-3d_fullres}"
TRAINER="${TRAINER:-nnUNetTrainerWandB_500ep_LSTVOversample}"
PLANNER="${PLANNER:-nnUNetPlannerResEncL}"
GPU_MEMORY_TARGET_GB="${GPU_MEMORY_TARGET_GB:-100}"
export SPINESURG_LABEL_SCHEME="${SPINESURG_LABEL_SCHEME:-oneshot}"
case "${PLANNER}" in
    nnUNetPlannerResEncM)  PLANS="nnUNetResEncUNetPlans_${GPU_MEMORY_TARGET_GB}G"   ;;
    nnUNetPlannerResEncL)  PLANS="nnUNetResEncUNetLPlans_${GPU_MEMORY_TARGET_GB}G"  ;;
    nnUNetPlannerResEncXL) PLANS="nnUNetResEncUNetXLPlans_${GPU_MEMORY_TARGET_GB}G" ;;
    ExperimentPlanner)     PLANS="nnUNetPlans"                                     ;;
    *) echo "ERROR: unknown planner ${PLANNER}" >&2; exit 1 ;;
esac

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-${HOME}/spinesurg-ct-nnunet}"
NNUNET_NFS="${PROJECT_ROOT}/nnunet"
CONTAINER="${PROJECT_ROOT}/containers/spinesurg-ct.sif"
WANDB_TRAINER_HOST="${PROJECT_ROOT}/tools/nnunet_wandb_variant.py"
WANDB_TRAINER_CONTAINER="/opt/conda/lib/python3.11/site-packages/nnunetv2/training/nnUNetTrainer/variants/nnunet_wandb_variant.py"
DS_DIR_NAME="Dataset$(printf '%03d' ${DATASET_ID})_${DATASET_NAME}"
RAW_DS="${NNUNET_NFS}/raw/${DS_DIR_NAME}"
PREP_DS="${NNUNET_NFS}/preprocessed/${DS_DIR_NAME}"
RESULT_DIR="${NNUNET_NFS}/results/${DS_DIR_NAME}/${TRAINER}__${PLANS}__${CONFIG}/fold_${FOLD}"
CKPT="${RESULT_DIR}/${CHECKPOINT}"
[[ -f "${CKPT}" ]] || { echo "ERROR: no ${CKPT}" >&2; exit 1; }

STAMP="$(date +%Y%m%d)"
EVAL_DIR="${RESULT_DIR}/eval_${CHECKPOINT%.pth}_${STAMP}"
IMG_DIR="${EVAL_DIR}/images"
PRED_DIR="${EVAL_DIR}/pred"
mkdir -p "${IMG_DIR}" "${PRED_DIR}" "${EVAL_DIR}/metrics"
cp "${CKPT}" "${EVAL_DIR}/${CHECKPOINT}"      # freeze the weights this evaluation used

# ----- the fold's validation cases, as symlinks nnU-Net can predict from ---------------
python3 - "${PREP_DS}/splits_final.json" "${FOLD}" "${RAW_DS}/imagesTr" "${IMG_DIR}" <<'EOF'
import json, os, sys
splits, fold, src, dst = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
d = json.load(open(splits))
folds = d["folds"] if isinstance(d, dict) and "folds" in d else d
val = folds[fold]["val"]
n = 0
for c in val:
    s = os.path.join(src, f"{c}_0000.nii.gz"); t = os.path.join(dst, f"{c}_0000.nii.gz")
    if not os.path.exists(s): print("missing", s); continue
    if not os.path.lexists(t): os.symlink(s, t)
    n += 1
print(f"fold {fold}: {n} validation cases linked into {dst}")
EOF

# ----- Singularity env (same routes as spine_train_fold.sh, NFS everywhere) -----------
export SINGULARITYENV_TMPDIR="/workspace/tmp"
export SINGULARITYENV_nnUNet_raw="/nnunet_nfs/raw"
export SINGULARITYENV_nnUNet_preprocessed="/nnunet_nfs/preprocessed"
export SINGULARITYENV_nnUNet_results="/nnunet_nfs/results"
export SINGULARITYENV_SPINESURG_LABEL_SCHEME="${SPINESURG_LABEL_SCHEME}"
export SINGULARITYENV_PYTHONPATH="/workspace/tools"
export SINGULARITYENV_WANDB_MODE="disabled"
export SINGULARITYENV_LSTV_OVERSAMPLE_FRAC="0.5"
export SINGULARITY_TMPDIR="/tmp/${USER}_job_${SLURM_JOB_ID}_evalck_singularity"
export XDG_RUNTIME_DIR="${SINGULARITY_TMPDIR}/runtime"
mkdir -p "${SINGULARITY_TMPDIR}" "${XDG_RUNTIME_DIR}" "${PROJECT_ROOT}/tmp"
trap "rm -rf ${SINGULARITY_TMPDIR}" EXIT

SING_BINDS=(
    --nv
    --bind "${PROJECT_ROOT}:/workspace"
    --bind "${NNUNET_NFS}:/nnunet_nfs"
    --bind "${WANDB_TRAINER_HOST}:${WANDB_TRAINER_CONTAINER}"
    --bind "${PROJECT_ROOT}/tools/lstv_biased_dataloader.py:${WANDB_TRAINER_CONTAINER%/*}/lstv_biased_dataloader.py"
    --pwd  /workspace
)
c_eval="/nnunet_nfs/results/${DS_DIR_NAME}/${TRAINER}__${PLANS}__${CONFIG}/fold_${FOLD}/$(basename "${EVAL_DIR}")"

echo "================================================================"
echo " eval fold ${FOLD} from ${CHECKPOINT}  ($(grep -c . <<<"$(ls ${IMG_DIR})") cases)"
echo " Job ${SLURM_JOB_ID} @ $(hostname)  $(date)"
echo " -> ${EVAL_DIR}"
echo "================================================================"

singularity exec "${SING_BINDS[@]}" "${CONTAINER}" \
    nnUNetv2_predict -i "${c_eval}/images" -o "${c_eval}/pred" \
        -d ${DATASET_ID} -c ${CONFIG} -f ${FOLD} -tr ${TRAINER} -p ${PLANS} \
        -chk "${CHECKPOINT}" --save_probabilities -npp 4 -nps 4
rc=$?; echo "nnUNetv2_predict exit ${rc}"; [[ ${rc} -eq 0 ]] || exit ${rc}

echo "=============== eval_oneshot ==============="
singularity exec "${SING_BINDS[@]}" "${CONTAINER}" \
    python /workspace/tools/eval_oneshot.py \
        --predictions_dir "${c_eval}/pred" \
        --labels_dir "/nnunet_nfs/raw/${DS_DIR_NAME}/labelsTr" \
        --dataset_json "/nnunet_nfs/raw/${DS_DIR_NAME}/dataset.json" \
        --lstv_cases_json "/nnunet_nfs/raw/${DS_DIR_NAME}/lstv_cases.json" \
        --output_dir "${c_eval}/metrics" --workers 12
echo "=============== decode_sequence ==============="
singularity exec "${SING_BINDS[@]}" "${CONTAINER}" \
    python /workspace/tools/decode_sequence.py \
        --predictions_dir "${c_eval}/pred" \
        --dataset_json "/nnunet_nfs/raw/${DS_DIR_NAME}/dataset.json" \
        --labels_dir "/nnunet_nfs/raw/${DS_DIR_NAME}/labelsTr" \
        --out "${c_eval}/decode.csv"
echo "=============== DONE $(date) ==============="
cat "${EVAL_DIR}/metrics/report.txt" 2>/dev/null | head -60
