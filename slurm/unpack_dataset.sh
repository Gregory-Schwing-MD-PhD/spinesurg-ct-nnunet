#!/usr/bin/env bash
#SBATCH --job-name=unpack_ds
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/unpack_ds_%j.out
#SBATCH --error=logs/unpack_ds_%j.err
# =============================================================================
# unpack_dataset.sh — unpack a preprocessed dataset's .npz to .npy ONCE, on the shared
# filesystem, so that every fold trains straight from NFS without staging.
#
#   DATASET_ID=810 DATASET_NAME=SpineSurgLSTVOneShot sbatch slurm/unpack_dataset.sh
#
# WHY. nnU-Net unpacks .npz to .npy at the start of training, next to the .npz. Five folds
# each staging 222 GB to a node's /tmp and then unpacking to ~1 TB there filled the GPU
# nodes' local disks and killed four folds at once (2026-09-07). Unpacked once on NFS, the
# folds read patches through memory maps and never copy anything; the race between five
# trainers unpacking the same files concurrently is avoided because the .npy already exist.
# =============================================================================
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
DATASET_ID="${DATASET_ID:-810}"
DATASET_NAME="${DATASET_NAME:-SpineSurgLSTVOneShot}"
DS="Dataset$(printf '%03d' "$DATASET_ID")_${DATASET_NAME}"
DATA_DIR="nnunet/preprocessed/${DS}/nnUNetPlans_3d_fullres"
[ -d "$DATA_DIR" ] || { echo "no $DATA_DIR"; exit 1; }
export SINGULARITY_TMPDIR="/tmp/${USER}_job_${SLURM_JOB_ID}"
mkdir -p "$SINGULARITY_TMPDIR"
trap 'rm -rf "$SINGULARITY_TMPDIR"' EXIT
echo "unpacking $DATA_DIR with ${SLURM_CPUS_PER_TASK} processes at $(date)"
singularity exec --bind "$PWD:/workspace" --pwd /workspace containers/spinesurg-ct.sif \
    python -c "
from nnunetv2.training.dataloading.utils import unpack_dataset
unpack_dataset('/workspace/$DATA_DIR', unpack_segmentation=True, overwrite_existing=False, num_processes=${SLURM_CPUS_PER_TASK})
print('done')
"
echo "npz $(ls $DATA_DIR/*.npz | wc -l)  npy $(ls $DATA_DIR/*.npy | wc -l)"
du -sh --apparent-size "$DATA_DIR"
echo "finished $(date)"
