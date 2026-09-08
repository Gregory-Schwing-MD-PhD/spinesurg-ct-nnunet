#!/bin/bash
#SBATCH --job-name=dbg_split
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/dbg_split_%j.out
#SBATCH --error=logs/dbg_split_%j.err
set -uo pipefail
export PATH="${HOME}/mambaforge/envs/nextflow/bin:${PATH}"
unset PYTHONPATH LD_LIBRARY_PATH
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-${HOME}/spinesurg-ct-nnunet}"
for c in ${CASES:-0376 1053 0068}; do
    echo "=============== ${c}"
    singularity exec --bind "${PROJECT_ROOT}:/workspace" --bind "${HOME}:${HOME}" --pwd /workspace \
        "${PROJECT_ROOT}/containers/spinesurg-ct.sif" python /workspace/benchmark/debug_split.py \
        --labels "${HOME}/data/CTSpinoPelvic1K/labels" --case "${c}"
done
