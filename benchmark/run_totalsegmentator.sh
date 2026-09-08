#!/bin/bash
#SBATCH --job-name=bench_ts
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/bench_ts_%A_%a.out
#SBATCH --error=logs/bench_ts_%A_%a.err
#
# TotalSegmentator v2 over the benchmark case table, sharded across an array. Native output
# (TS ids, CT grid) to $BENCH/totalsegmentator/native; resumable (existing files skipped).
#
#   sbatch benchmark/run_totalsegmentator.sh                       # all 802, 4 shards
#   N_SHARDS=4 sbatch --array=2 benchmark/run_totalsegmentator.sh  # redo one shard
#   LIMIT=2 FAST=1 sbatch --array=0 --time=00:30:00 benchmark/run_totalsegmentator.sh  # smoke
#
# Container and weights are the ones the dataset build used (CTSpinoPelvic1K repo):
#   ~/CTSpinoPelvic1K/containers/ctspinopelvic1k-ts.sif, weights ~/totalseg_weights,
#   config ~/.totalseg. Compute nodes have no internet, so weights must already be there.
set -uo pipefail
export PATH="${HOME}/mambaforge/envs/nextflow/bin:${PATH}"
unset PYTHONPATH LD_LIBRARY_PATH

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-${HOME}/spinesurg-ct-nnunet}"
BENCH="${BENCH:-${HOME}/bench}"
CASES="${CASES:-${PROJECT_ROOT}/benchmark/cases.csv}"
SIF="${SIF:-${HOME}/CTSpinoPelvic1K/containers/ctspinopelvic1k-ts.sif}"
TS_WEIGHTS="${TS_WEIGHTS:-${HOME}/totalseg_weights}"
TS_CONFIG="${TS_CONFIG:-${HOME}/.totalseg}"
HF_EXPORT_DIR="${HF_EXPORT_DIR:-${HOME}/data/CTSpinoPelvic1K}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
N_SHARDS="${N_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}"
LIMIT="${LIMIT:-0}"
FAST="${FAST:-0}"
OUT="${BENCH}/totalsegmentator/native"
mkdir -p "${OUT}" "${PROJECT_ROOT}/logs"

for f in "${SIF}" "${CASES}" "${TS_WEIGHTS}/Dataset292_TotalSegmentator_part2_vertebrae_1532subj"; do
    [[ -e "${f}" ]] || { echo "ERROR: missing ${f}" >&2; exit 1; }
done

export SINGULARITY_TMPDIR="/tmp/${USER}_${SLURM_JOB_ID}_bench_ts"
export XDG_RUNTIME_DIR="${SINGULARITY_TMPDIR}/runtime"
mkdir -p "${SINGULARITY_TMPDIR}" "${XDG_RUNTIME_DIR}"
trap 'rm -rf "${SINGULARITY_TMPDIR}"' EXIT

export SINGULARITYENV_TOTALSEG_WEIGHTS_PATH="${TS_WEIGHTS}"
export SINGULARITYENV_TOTALSEG_HOME_DIR="${TS_CONFIG}"
export SINGULARITYENV_HOME="${TS_CONFIG}"
export SINGULARITYENV_OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export SINGULARITYENV_NUMEXPR_MAX_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "bench_ts shard ${SHARD}/${N_SHARDS} on $(hostname) $(date); fast=${FAST} limit=${LIMIT}"
FLAGS=(--cases "${CASES}" --out_dir "${OUT}" --shard "${SHARD}" --n_shards "${N_SHARDS}" --limit "${LIMIT}")
[[ "${FAST}" == "1" ]] && FLAGS+=(--fast)

singularity exec --nv \
    --bind "${PROJECT_ROOT}:/workspace" \
    --bind "${BENCH}:${BENCH}" \
    --bind "${HF_EXPORT_DIR}:${HF_EXPORT_DIR}" \
    --bind "${TS_WEIGHTS}:${TS_WEIGHTS}" \
    --bind "${TS_CONFIG}:${TS_CONFIG}" \
    --pwd /workspace \
    "${SIF}" python /workspace/benchmark/run_ts_cases.py "${FLAGS[@]}"
rc=$?
echo "exit ${rc} $(date); $(ls ${OUT}/*.nii.gz 2>/dev/null | wc -l) native predictions so far"
exit ${rc}
