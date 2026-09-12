#!/bin/bash
#SBATCH --job-name=vf_tlm
#SBATCH --partition=reqp
#SBATCH --qos=requeue
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/vf_tlm_%j.out
#SBATCH --error=logs/vf_tlm_%j.err
#
# Classify the disputed thoracolumbar junctions with the manuscript's own morphometrics.
#
#   CASES=verse509,verse512,... sbatch -D ~/spinesurg-ct-nnunet slurm/versefusion_tl_morphometrics.sh
#   sbatch -D ~/spinesurg-ct-nnunet slurm/versefusion_tl_morphometrics.sh      # all of them
#
# Two steps. Bridge the Dataset813 ids to the v10 release ids, then run
# CTSpinoPelvic1K's extract_transition_morphometrics.py unchanged -- the code the paper's
# numbers come from, whose _nearest_vert IS the rib-articulation test and whose
# rib12_11_ratio is the validated stump criterion.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/spinesurg-ct-nnunet}"

SRC="${SRC:-$HOME/vf_813/labelsTr}"
V10="${V10:-$HOME/vf_v10}"
OUT="${OUT:-$HOME/vf_morphometrics}"
CTSP="${CTSP:-$HOME/CTSpinoPelvic1K}"
PY="$HOME/mambaforge/envs/spineps/bin/python"

echo "=== bridging ids  ($(ls "${SRC}" 2>/dev/null | wc -l) available)"
ARGS=(--src "${SRC}" --dst "${V10}")
[ -n "${CASES:-}" ] && ARGS+=(--cases "${CASES}")
"${PY}" -u tools/ds813_to_v10.py "${ARGS[@]}"

echo
echo "=== measuring the thoracolumbar junction"
mkdir -p "${OUT}"
cd "${CTSP}"
"${PY}" -u scripts/extract_transition_morphometrics.py \
    --labels "${V10}" --out "${OUT}" --workers 12
echo "finished $(date)"
ls -la "${OUT}" | head
