#!/bin/bash
#SBATCH --job-name=vf_fov
#SBATCH --partition=reqp
#SBATCH --qos=requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/vf_fov_%j.out
#SBATCH --error=logs/vf_fov_%j.err
#
# How much image lies BELOW the lowest labelled vertebra, per case.
#
#   sbatch -D ~/spinesurg-ct-nnunet slurm/versefusion_fov.sh
#
# This sizes the annotation job. VerSe labels no sacrum anywhere in the corpus, but "not
# labelled" is not "not in the scan": the pelvis may be in the field of view and simply
# unannotated, which is what annotating fixes. A case with 100 mm below its last vertebra
# has a sacrum and probably hips to label; one with 5 mm has nothing there, and putting it
# in front of a student is wasted effort.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/spinesurg-ct-nnunet}"

SRC="${SRC:-$HOME/VerSeFusion/data/unified}"
OUT="${OUT:-logs/vf_fov_${SLURM_JOB_ID}.csv}"
PY="$HOME/mambaforge/envs/spineps/bin/python"

echo "src ${SRC} -> ${OUT}   started $(date)"
"${PY}" -u tools/versefusion_fov.py "${SRC}" "${OUT}"
echo "finished $(date)"
