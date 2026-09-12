#!/bin/bash
#SBATCH --job-name=vf_adj
#SBATCH --partition=reqp
#SBATCH --qos=requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/vf_adj_%j.out
#SBATCH --error=logs/vf_adj_%j.err
#
# Does the disputed vertebra bear a rib? T13 against L6, decided on the anatomy.
#
#   sbatch -D ~/spinesurg-ct-nnunet slurm/versefusion_adjudicate.sh
#
# A BATCH JOB, because the first two attempts at this kind of work were run on the login
# node and both were OOM-killed. Distance transforms over 512x512x743 volumes do not belong
# on a shared head node even when they fit.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/spinesurg-ct-nnunet}"

LABELS="${LABELS:-$HOME/vf_813/labelsTr}"
DISPUTED="${DISPUTED:-logs/versefusion_disputed.csv}"
OUT="${OUT:-logs/t13_adjudication_${SLURM_JOB_ID}.json}"
PY="$HOME/mambaforge/envs/spineps/bin/python"

echo "labels ${LABELS}  ($(ls "${LABELS}" 2>/dev/null | wc -l) converted)"
echo "started $(date)"
"${PY}" -u tools/versefusion_adjudicate_t13.py \
    --labels "${LABELS}" --disputed "${DISPUTED}" --out "${OUT}"
echo "finished $(date)"
