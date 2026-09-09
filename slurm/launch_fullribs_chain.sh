#!/bin/bash
# Submit the numbered-rib arm as one dependency chain behind a running prep job:
#   reshape plans -> unpack -> five folds (STAGE_LOCAL=auto) -> patch-shape ablation (fold 0)
# Runs on the login node in seconds; every step is a SLURM job with afterok dependencies.
#
#   PREP_JOB=<id> bash slurm/launch_fullribs_chain.sh
set -uo pipefail
cd "${HOME}/spinesurg-ct-nnunet"
PREP_JOB="${PREP_JOB:?set PREP_JOB}"
DS_ID=813; DS_NAME=SpineSurgLSTVFullRibs
COMMON="DATASET_ID=${DS_ID},DATASET_NAME=${DS_NAME},PLANNER=nnUNetPlannerResEncL,GPU_MEMORY_TARGET_GB=100,TRAINER=nnUNetTrainerWandB_500ep_LSTVOversample,LSTV_OVERSAMPLE_FRAC=0.5,SPINESURG_LABEL_SCHEME=fullribs"

R=$(sbatch --parsable --dependency=afterok:${PREP_JOB} --export=ALL,DATASET_ID=${DS_ID},DATASET_NAME=${DS_NAME} slurm/reshape_plans.sh)
echo "reshape ${R} (after prep ${PREP_JOB})"
U=$(sbatch --parsable --dependency=afterok:${PREP_JOB} --export=ALL,DATASET_ID=${DS_ID},DATASET_NAME=${DS_NAME} slurm/unpack_dataset.sh)
echo "unpack ${U} (after prep)"
for f in 0 1 2 3 4; do
    j=$(sbatch --parsable --dependency=afterok:${U}:${R} --export=ALL,FOLD=${f},${COMMON},STAGE_LOCAL=auto slurm/spine_train_fold.sh)
    echo "fold ${f} job ${j} (STAGE_LOCAL=auto, after unpack+reshape)"
done
V=$(sbatch --parsable --dependency=afterok:${U}:${R} --export=ALL,DATASET_ID=${DS_ID},DATASET_NAME=${DS_NAME},VARIANTS="p256:256,320,320 p448:448,224,224 p512:512,192,192" slurm/make_plan_variants.sh)
echo "patch ablation submitter ${V} (after unpack+reshape)"
squeue -u "${USER}" -o "%.12i %.14j %.8T %.10M %R"
