#!/bin/bash
#SBATCH --job-name=patch_ctx
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/patch_ctx_%j.out
#SBATCH --error=logs/patch_ctx_%j.err
# Label-based measurement of what each patch shape sees (benchmark/patch_context.py), all 802
# released label volumes, plus the projection render. CPU only, in the training container.
set -uo pipefail
export PATH="${HOME}/mambaforge/envs/nextflow/bin:${PATH}"
unset PYTHONPATH LD_LIBRARY_PATH
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-${HOME}/spinesurg-ct-nnunet}"
OUT="${OUT:-${HOME}/bench/patch_context}"
mkdir -p "${OUT}" "${PROJECT_ROOT}/logs"
singularity exec --bind "${PROJECT_ROOT}:/workspace" --bind "${HOME}:${HOME}" --pwd /workspace \
    "${PROJECT_ROOT}/containers/spinesurg-ct.sif" python /workspace/benchmark/patch_context.py \
    --labels "${HOME}/data/CTSpinoPelvic1K/labels" --out "${OUT}" \
    --shapes ${SHAPES:-256,320,320 384,256,256 448,224,224} --limit "${LIMIT:-0}" --render "${RENDER:-0376}" \
    --workers "${SLURM_CPUS_PER_TASK:-8}"
