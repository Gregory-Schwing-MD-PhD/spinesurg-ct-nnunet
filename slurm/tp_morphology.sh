#!/bin/bash
#SBATCH --job-name=tp_morph
#SBATCH --partition=reqp
#SBATCH --qos=requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/tp_morph_%j.out
#SBATCH --error=logs/tp_morph_%j.err
#
# Transverse-process morphology at the thoracolumbar junction, on both corpora.
#
#   sbatch -D ~/spinesurg-ct-nnunet slurm/tp_morphology.sh
#
# CTSpinoPelvic1K first, because the measurement has to be shown to separate T12 from L1
# on labels where the answer is known BEFORE it is allowed to say anything about the
# disputed VerSe cases. If it cannot tell a known T12 from a known L1, its verdict on an
# unknown one is worthless.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/spinesurg-ct-nnunet}"

CTSP_LABELS="${CTSP_LABELS:-$HOME/data/CTSpinoPelvic1K/labels}"
VF_LABELS="${VF_LABELS:-$HOME/vf_v10}"
LIMIT="${LIMIT:-400}"
PY="$HOME/mambaforge/envs/spineps/bin/python"
mkdir -p logs

echo "=== CTSpinoPelvic1K, where the levels are known"
"${PY}" -u tools/tp_morphology.py --labels "${CTSP_LABELS}" \
    --out logs/tp_ctsp1k.csv --limit "${LIMIT}" --workers 16

echo
echo "=== VerSeFusion, the disputed cases"
"${PY}" -u tools/tp_morphology.py --labels "${VF_LABELS}" \
    --out logs/tp_versefusion.csv --workers 8

echo "finished $(date)"
