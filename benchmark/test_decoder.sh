#!/bin/bash
#SBATCH --job-name=dec_test
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/dec_test_%j.out
#SBATCH --error=logs/dec_test_%j.err
# Decoder self-tests on ground truth (tests/test_decode_sequence_on_labels.py), in the
# training container. Run A: one-hot ground truth must reproduce its own rib-free count.
# Run B: L6 renamed L5 (the collapse / a competitor without L6) must STILL count six.
set -uo pipefail
export PATH="${HOME}/mambaforge/envs/nextflow/bin:${PATH}"
unset PYTHONPATH LD_LIBRARY_PATH
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-${HOME}/spinesurg-ct-nnunet}"
LABELS="${LABELS:-${HOME}/data/CTSpinoPelvic1K/labels}"
CASES="${CASES:-0007 0376 1053 0068}"
OUT="${OUT:-/tmp/${USER}_dec_test_${SLURM_JOB_ID}}"
mkdir -p "${OUT}" "${PROJECT_ROOT}/logs"
trap 'rm -rf "${OUT}"' EXIT
run() { singularity exec --bind "${PROJECT_ROOT}:/workspace" --bind "${HOME}:${HOME}" --bind /tmp:/tmp --pwd /workspace \
        "${PROJECT_ROOT}/containers/spinesurg-ct.sif" python /workspace/tests/test_decode_sequence_on_labels.py \
        --labels "${LABELS}" --cases ${CASES} "$@"; }
echo "=== A: one-hot ground truth"
run --out "${OUT}/a" --onehot; ra=$?
echo "=== B: L6 renamed L5, one-hot"
run --out "${OUT}/b" --onehot --collapse; rb=$?
echo "A exit ${ra}, B exit ${rb}"
[[ ${ra} -eq 0 && ${rb} -eq 0 ]]
