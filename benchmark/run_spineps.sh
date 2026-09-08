#!/bin/bash
#SBATCH --job-name=bench_spineps
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/bench_spineps_%A_%a.out
#SBATCH --error=logs/bench_spineps_%A_%a.err
#
# SPINEPS CT (semantic + instance, v2.0.0 weights) with VERIDAH CT labeling (v1.4.0
# ct_labeling) over the benchmark case table. conda env `spineps` from install_envs.sh,
# models from fetch_weights.sh in ~/spineps_models. Native vertebra instance masks (VerSe
# ids) -> $BENCH/spineps/native/<case>_seg-vert_msk.nii.gz; resumable.
#
#   sbatch benchmark/run_spineps.sh
#   LIMIT=2 sbatch --array=0 --time=00:40:00 benchmark/run_spineps.sh   # smoke
set -uo pipefail
source "${HOME}/mambaforge/etc/profile.d/conda.sh"; conda activate spineps
export SPINEPS_SEGMENTOR_MODELS="${SPINEPS_SEGMENTOR_MODELS:-${HOME}/spineps_models}"

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-${HOME}/spinesurg-ct-nnunet}"
BENCH="${BENCH:-${HOME}/bench}"
CASES="${CASES:-${PROJECT_ROOT}/benchmark/cases.csv}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
N_SHARDS="${N_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}"
LIMIT="${LIMIT:-0}"
EXTRA="${EXTRA:-}"
NATIVE="${BENCH}/spineps/native"
WORK="/tmp/${USER}_${SLURM_JOB_ID}_spineps"
mkdir -p "${NATIVE}" "${WORK}" "${PROJECT_ROOT}/logs"
trap 'rm -rf "${WORK}"' EXIT
[[ -d "${SPINEPS_SEGMENTOR_MODELS}" ]] || { echo "ERROR: no models at ${SPINEPS_SEGMENTOR_MODELS}" >&2; exit 1; }

# the release zips unpack one level deep (ct/CT_semantic, CT_instance/CT_instance,
# ct_labeling/ct_labeling); fetch_weights.sh flattens them, and the models are passed as
# absolute folders (spineps accepts a path wherever it accepts a model name)
MODEL_SEMANTIC="${MODEL_SEMANTIC:-${SPINEPS_SEGMENTOR_MODELS}/CT_semantic}"
MODEL_INSTANCE="${MODEL_INSTANCE:-${SPINEPS_SEGMENTOR_MODELS}/CT_instance}"
MODEL_LABELING="${MODEL_LABELING:-${SPINEPS_SEGMENTOR_MODELS}/ct_labeling}"
for m in "${MODEL_SEMANTIC}" "${MODEL_INSTANCE}" "${MODEL_LABELING}"; do
    [[ -f "${m}/inference_config.json" ]] || { echo "ERROR: no inference_config.json in ${m}" >&2; exit 1; }
done
echo "bench_spineps shard ${SHARD}/${N_SHARDS} on $(hostname) $(date); models: ${MODEL_SEMANTIC} ${MODEL_INSTANCE} ${MODEL_LABELING}"
python "${PROJECT_ROOT}/benchmark/run_spineps_cases.py" --cases "${CASES}" --out_dir "${NATIVE}" --work "${WORK}" \
    --shard "${SHARD}" --n_shards "${N_SHARDS}" --limit "${LIMIT}" --extra "${EXTRA}" \
    --model_semantic "${MODEL_SEMANTIC}" --model_instance "${MODEL_INSTANCE}" --model_labeling "${MODEL_LABELING}"
rc=$?
echo "exit ${rc} $(date); $(ls ${NATIVE}/*_seg-vert_msk.nii.gz 2>/dev/null | wc -l) native predictions so far"
exit ${rc}
