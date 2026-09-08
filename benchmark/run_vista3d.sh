#!/bin/bash
#SBATCH --job-name=bench_vista
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/bench_vista_%A_%a.out
#SBATCH --error=logs/bench_vista_%A_%a.err
#
# NV-Segment-CT (the released VISTA3D-CT lineage, 127 classes) over the benchmark case table
# through its MONAI bundle batch inference. conda env `vista3d` and the clone at
# ~/NV-Segment-CTMR come from benchmark/install_envs.sh; the checkpoint from
# benchmark/fetch_weights.sh (Hugging Face cache) since compute nodes have no internet.
# Native output (label_dict ids, CT grid) -> $BENCH/vista3d/native/<case>.nii.gz; resumable.
#
#   sbatch benchmark/run_vista3d.sh
#   LIMIT=2 sbatch --array=0 --time=00:40:00 benchmark/run_vista3d.sh   # smoke
set -uo pipefail
source "${HOME}/mambaforge/etc/profile.d/conda.sh"; conda activate vista3d
export HF_HOME="${HOME}/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-${HOME}/spinesurg-ct-nnunet}"
BENCH="${BENCH:-${HOME}/bench}"
CASES="${CASES:-${PROJECT_ROOT}/benchmark/cases.csv}"
BUNDLE="${BUNDLE:-${HOME}/NV-Segment-CTMR/NV-Segment-CT}"
SHARD="${SLURM_ARRAY_TASK_ID:-0}"
N_SHARDS="${N_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}"
LIMIT="${LIMIT:-0}"
NATIVE="${BENCH}/vista3d/native"
WORK="${BENCH}/vista3d/work/shard${SHARD}"
mkdir -p "${NATIVE}" "${WORK}/in" "${WORK}/out" "${PROJECT_ROOT}/logs"
[[ -f "${BUNDLE}/configs/inference.json" ]] || { echo "ERROR: bundle missing at ${BUNDLE}" >&2; exit 1; }
cp -n "${BUNDLE}/configs/label_dict.json" "${BENCH}/vista3d/label_dict.json" 2>/dev/null || true

# this shard's cases, as symlinks named by case id, skipping those already done
python3 - "${CASES}" "${SHARD}" "${N_SHARDS}" "${LIMIT}" "${NATIVE}" "${WORK}/in" <<'EOF'
import csv, os, sys
cases, shard, n, limit, native, indir = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]
rows = [r for i, r in enumerate(csv.DictReader(open(cases))) if i % n == shard]
if limit: rows = rows[:limit]
todo = 0
for r in rows:
    if os.path.exists(os.path.join(native, r["case"] + ".nii.gz")): continue
    t = os.path.join(indir, r["case"] + ".nii.gz")
    if not os.path.lexists(t): os.symlink(r["ct"], t)
    todo += 1
print(f"shard {shard}/{n}: {len(rows)} cases, {todo} to do")
EOF

cd "${BUNDLE}"
echo "bench_vista shard ${SHARD} on $(hostname) $(date)"
python -m monai.bundle run --config_file "['configs/inference.json', 'configs/batch_inference.json']" \
    --input_dir="${WORK}/in" --output_dir="${WORK}/out"
rc=$?
echo "bundle exit ${rc} $(date)"

# collect: {output_dir}/{case}/{case}_<postfix>.nii.gz -> native/<case>.nii.gz
n=0
for d in "${WORK}/out"/*/; do
    c=$(basename "${d}")
    f=$(ls "${d}"/${c}_*.nii.gz 2>/dev/null | head -1)
    [[ -n "${f}" ]] && { mv -f "${f}" "${NATIVE}/${c}.nii.gz"; rm -f "${WORK}/in/${c}.nii.gz"; n=$((n+1)); }
done
echo "collected ${n}; $(ls ${NATIVE}/*.nii.gz 2>/dev/null | wc -l) native predictions total"
exit ${rc}
