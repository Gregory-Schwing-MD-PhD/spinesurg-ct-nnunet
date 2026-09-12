#!/bin/bash
#SBATCH --job-name=vf_render
#SBATCH --partition=reqp
#SBATCH --qos=requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/vf_render_%j.out
#SBATCH --error=logs/vf_render_%j.err
#
# Render the twelve disputed vertebrae, so T13-against-L6 can be settled by looking.
#
#   sbatch -D ~/spinesurg-ct-nnunet slurm/versefusion_render_disputed.sh
#
# Written as a file and committed, not piped through a shell heredoc. Three of these
# scripts were lost to heredoc mangling in one session: a quoted heredoc still let $HOME
# expand on the SUBMITTING machine, producing a job that cd-ed to a Windows path.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/spinesurg-ct-nnunet}"

LABELS="${LABELS:-$HOME/vf_813/labelsTr}"
SRC="${SRC:-$HOME/VerSeFusion/data/unified}"
DISPUTED="${DISPUTED:-logs/versefusion_disputed.csv}"
OUT="${OUT:-logs/disputed_renders}"
PY="$HOME/mambaforge/envs/spineps/bin/python"

echo "labels ${LABELS} ($(ls "${LABELS}" 2>/dev/null | wc -l) converted)  started $(date)"
"${PY}" -u tools/versefusion_render_disputed.py \
    --labels "${LABELS}" --src "${SRC}" --disputed "${DISPUTED}" --out "${OUT}"
echo "finished $(date)"
