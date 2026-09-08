#!/bin/bash
#SBATCH --job-name=bench_spribs
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/bench_spribs_%A_%a.out
#SBATCH --error=logs/bench_spribs_%A_%a.err
#
# Möller's rib stack (binary rib nnU-Net + rib-segmentation assignment/length/stump features)
# on top of the SPINEPS masks from run_spineps.sh. Same shards as run_spineps.sh so each
# case's masks exist. Outputs to $BENCH/spineps/ribs/.
#
#   sbatch benchmark/run_spineps_ribs.sh
#   LIMIT=2 sbatch --array=0 --time=00:50:00 benchmark/run_spineps_ribs.sh   # smoke
set -uo pipefail
source "${HOME}/mambaforge/etc/profile.d/conda.sh"; conda activate spineps

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-${HOME}/spinesurg-ct-nnunet}"
BENCH="${BENCH:-${HOME}/bench}"
CASES="${CASES:-${PROJECT_ROOT}/benchmark/cases.csv}"
RIB_MODEL="${RIB_MODEL:-${HOME}/CTSpinoPelvic1K/models/moller_ribseg/ribseg_model_weights}"
RIB_FOLDS="${RIB_FOLDS:-0}"
RIBSEG_REPO="${RIBSEG_REPO:-${HOME}/rib-segmentation}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
N_SHARDS="${N_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}"
LIMIT="${LIMIT:-0}"
OUT="${BENCH}/spineps/ribs"
WORK="/tmp/${USER}_${SLURM_JOB_ID}_spribs"
mkdir -p "${OUT}" "${WORK}" "${PROJECT_ROOT}/logs"
trap 'rm -rf "${WORK}"' EXIT
[[ -f "${RIB_MODEL}/plans.json" ]] || { echo "ERROR: rib model missing at ${RIB_MODEL}" >&2; exit 1; }
[[ -f "${RIBSEG_REPO}/run.py" ]] || { echo "ERROR: rib-segmentation clone missing at ${RIBSEG_REPO}" >&2; exit 1; }

echo "bench_spribs shard ${SHARD}/${N_SHARDS} on $(hostname) $(date)"
python "${PROJECT_ROOT}/benchmark/run_spineps_ribs.py" --cases "${CASES}" --spineps_dir "${BENCH}/spineps/native" \
    --out_dir "${OUT}" --work "${WORK}" --rib_model "${RIB_MODEL}" --rib_folds "${RIB_FOLDS}" \
    --ribseg_repo "${RIBSEG_REPO}" --shard "${SHARD}" --n_shards "${N_SHARDS}" --limit "${LIMIT}"
rc=$?
echo "exit ${rc} $(date); $(ls ${OUT}/*_ribmask.nii.gz 2>/dev/null | wc -l) rib masks so far"
exit ${rc}
