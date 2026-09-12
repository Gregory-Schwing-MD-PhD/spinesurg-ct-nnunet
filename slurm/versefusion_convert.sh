#!/bin/bash
#SBATCH --job-name=vf_convert
#SBATCH --partition=reqp
#SBATCH --qos=requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=logs/vf_convert_%j.out
#SBATCH --error=logs/vf_convert_%j.err
#
# Map VerSeFusion into the Dataset813 label space.
#
#   sbatch -D ~/spinesurg-ct-nnunet slurm/versefusion_convert.sh            # dry run
#   DRY=0 DST=~/vf_813 sbatch -D ~/spinesurg-ct-nnunet slurm/versefusion_convert.sh
#
# A BATCH JOB AND NOT THE LOGIN NODE. The first attempt ran twelve workers on the login
# node and a worker was OOM-killed at case 50: these volumes reach 960x960x110, and the
# corpus is stored float64. Eight workers with 96 GB is comfortable; it is also the right
# place to run it on a shared cluster.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$HOME/spinesurg-ct-nnunet}"

PY="$HOME/mambaforge/envs/spineps/bin/python"
SRC="${SRC:-$HOME/VerSeFusion/data/unified}"
DRY="${DRY:-1}"
DST="${DST:-$HOME/vf_813}"
REPORT="${REPORT:-$PWD/logs/vf_convert_${SLURM_JOB_ID}.json}"

echo "================================================================"
echo " VerSeFusion -> Dataset813 label space"
echo " src    : ${SRC}"
echo " dry    : ${DRY}"
[ "${DRY}" = "0" ] && echo " dst    : ${DST}"
echo " report : ${REPORT}"
echo " started: $(date)"
echo "================================================================"

ARGS=(--src "${SRC}" --workers 8 --report "${REPORT}")
if [ "${DRY}" = "1" ]; then ARGS+=(--dry); else ARGS+=(--dst "${DST}"); fi

"${PY}" -u tools/convert_versefusion.py "${ARGS[@]}"
RC=$?
echo "finished: $(date)  rc=${RC}"
exit ${RC}
