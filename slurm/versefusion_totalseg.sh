#!/bin/bash
#SBATCH --job-name=vf_ts
#SBATCH --partition=gmsap,gvohp
#SBATCH -q gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --exclude=msa1
#SBATCH --output=logs/vf_ts_%A_%a.out
#SBATCH --error=logs/vf_ts_%A_%a.err
#
# Fill in the bones VerSe does not label, with TotalSegmentator.
#
#   sbatch --array=0-7 -D ~/spinesurg-ct-nnunet slurm/versefusion_totalseg.sh
#   DST=~/vf_813 sbatch --array=0-7 ...
#
# SHARDED AND RESUMABLE, like slurm/v3_totalseg.sh in CTSpinoPelvic1K, which is the job
# that built this corpus's own ribs. Each array task takes every n-th case, so a task that
# hits the wall or is preempted is resubmitted on its own without re-running the rest; the
# --resume flag skips any case whose label already exists.
#
# --exclude=msa1 DELIBERATELY. The Dataset813 training run lives on msa1 and this job must
# not compete with it for the GPU.
#
# N_SHARDS_OVERRIDE pins the ORIGINAL shard count so a partial resubmit
# (--array=3,5) keeps the SAME case split. Without it SLURM_ARRAY_TASK_COUNT would be the
# size of the resubmitted subset and silently re-shard, which is how cases get skipped.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/spinesurg-ct-nnunet}"
PROJECT_ROOT="$(pwd)"

SRC="${SRC:-$HOME/VerSeFusion/data/unified}"
DST="${DST:-$HOME/vf_813}"
SIF="${SIF:-$HOME/CTSpinoPelvic1K/containers/ctspinopelvic1k-ts.sif}"
TOTALSEG_WEIGHTS="${TOTALSEG_WEIGHTS:-$HOME/totalseg_weights}"
TOTALSEG_CONFIG_DIR="${TOTALSEG_CONFIG_DIR:-$HOME/.totalseg}"
SING="${SING:-$HOME/mambaforge/envs/nextflow/bin/singularity}"

SHARD="${SLURM_ARRAY_TASK_ID:-0}"
N_SHARDS="${N_SHARDS_OVERRIDE:-${SLURM_ARRAY_TASK_COUNT:-1}}"

[[ -f "${SIF}" ]] || { echo "ERROR: no TS container at ${SIF}" >&2; exit 1; }
[[ -d "${SRC}" ]] || { echo "ERROR: no source at ${SRC}" >&2; exit 1; }
mkdir -p "${DST}" logs "${TOTALSEG_WEIGHTS}" "${TOTALSEG_CONFIG_DIR}"

BINDS="${PROJECT_ROOT}:/workspace,${SRC}:${SRC},${DST}:${DST}"
BINDS+=",${TOTALSEG_WEIGHTS}:${TOTALSEG_WEIGHTS},${TOTALSEG_CONFIG_DIR}:${TOTALSEG_CONFIG_DIR}"
# HOME is pointed at the TS config dir because TotalSegmentator writes its config under
# $HOME and the real home is on NFS.
CENV="TOTALSEG_WEIGHTS_PATH=${TOTALSEG_WEIGHTS},TOTALSEG_HOME_DIR=${TOTALSEG_CONFIG_DIR}"
CENV+=",HOME=${TOTALSEG_CONFIG_DIR},PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
CENV+=",PYTHONUNBUFFERED=1"

echo "================================================================"
echo " VerSeFusion: TotalSegmentator for the bones VerSe omits"
echo " shard   : ${SHARD} of ${N_SHARDS}"
echo " src/dst : ${SRC} -> ${DST}"
echo " node    : $(hostname)   started $(date)"
echo "================================================================"

# CASE=verse504 runs one case and nothing else, for looking at the output before
# committing a GPU array to 374 of them.
ARGS=(--src "${SRC}" --dst "${DST}")
if [ -n "${CASE:-}" ]; then
    ARGS+=(--case "${CASE}")
else
    ARGS+=(--shard "${SHARD}" --n_shards "${N_SHARDS}" --resume)
fi

stdbuf -oL -eL "${SING}" exec --nv --env "${CENV}" --bind "${BINDS}" --pwd /workspace \
    "${SIF}" python -u /workspace/tools/versefusion_totalseg.py "${ARGS[@]}"
RC=$?
echo "shard ${SHARD} finished $(date)  rc=${RC}"
exit ${RC}
