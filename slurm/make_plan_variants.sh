#!/bin/bash
#SBATCH --job-name=plan_variants
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/plan_variants_%j.out
#SBATCH --error=logs/plan_variants_%j.err
#
# The patch-shape ablation. After reshape_plans.sh has turned the main plan into 384x256x256,
# write two sibling plans files from the planner's original (256x320x320, kept as .orig):
#   <PLANS>_p256.json   the planner's own shape (control)
#   <PLANS>_p448.json   448x224x224, longer still (35.8 cm along the column, 16.9 cm in-plane)
# and submit fold 0 of the dataset on each, so three shapes train on the same fold, same
# sampler, same everything else. Compare the dedicated LSTV pass and the per-class Dice over
# the first 100 epochs, then keep the winner.
#
#   sbatch --dependency=afterok:<reshape>:<unpack> --export=ALL,DATASET_ID=813,DATASET_NAME=SpineSurgLSTVFullRibs slurm/make_plan_variants.sh
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-${HOME}/spinesurg-ct-nnunet}"
DATASET_ID="${DATASET_ID:-813}"; DATASET_NAME="${DATASET_NAME:-SpineSurgLSTVFullRibs}"
PLANS="${PLANS:-nnUNetResEncUNetLPlans_100G}"
DS="Dataset$(printf '%03d' ${DATASET_ID})_${DATASET_NAME}"
P="nnunet/preprocessed/${DS}"
[[ -f "${P}/${PLANS}.json.orig" ]] || { echo "ERROR: ${P}/${PLANS}.json.orig missing (reshape_plans.sh not run?)" >&2; exit 1; }
# VARIANTS: space-separated "name:d0,d1,d2" (voxels, column axis first); default = the
# planner's shape as control and a longer column. p512 = 512x192x192 (41 cm x 14.5 cm, 18.9M
# voxels) is the extreme column: the hips are cut at the acetabula, which is the price it pays.
VARIANTS="${VARIANTS:-p256:256,320,320 p448:448,224,224}"
COMMON="DATASET_ID=${DATASET_ID},DATASET_NAME=${DATASET_NAME},PLANNER=nnUNetPlannerResEncL,GPU_MEMORY_TARGET_GB=100,TRAINER=nnUNetTrainerWandB_500ep_LSTVOversample,LSTV_OVERSAMPLE_FRAC=0.5,SPINESURG_LABEL_SCHEME=fullribs,STAGE_LOCAL=0"
for spec in ${VARIANTS}; do
    v="${spec%%:*}"; dims="${spec##*:}"
    python3 tools/reshape_patch.py --plans "${P}/${PLANS}.json.orig" --patch ${dims//,/ } --out "${P}/${PLANS}_${v}.json" || exit 1
    j=$(sbatch --parsable --export=ALL,FOLD=0,${COMMON},PLANS_OVERRIDE=${PLANS}_${v} slurm/spine_train_fold.sh)
    echo "ablation fold 0 ${v} (${dims}): job ${j}"
done
