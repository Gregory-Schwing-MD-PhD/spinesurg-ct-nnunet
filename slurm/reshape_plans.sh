#!/bin/bash
#SBATCH --job-name=reshape_plans
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/reshape_plans_%j.out
#SBATCH --error=logs/reshape_plans_%j.err
# After prep: reshape the ResEnc-L patch to 384 x 256 x 256 (tools/reshape_patch.py).
#   sbatch --dependency=afterok:<prep> --export=ALL,DATASET_ID=813,DATASET_NAME=SpineSurgLSTVFullRibs slurm/reshape_plans.sh
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-${HOME}/spinesurg-ct-nnunet}"
DATASET_ID="${DATASET_ID:-813}"; DATASET_NAME="${DATASET_NAME:-SpineSurgLSTVFullRibs}"
PLANS="${PLANS:-nnUNetResEncUNetLPlans_100G}"
DS="Dataset$(printf '%03d' ${DATASET_ID})_${DATASET_NAME}"
python3 tools/reshape_patch.py --plans "nnunet/preprocessed/${DS}/${PLANS}.json" --patch ${PATCH:-384 256 256}
