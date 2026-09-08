#!/bin/bash
#SBATCH --job-name=bench_score
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/bench_score_%j.out
#SBATCH --error=logs/bench_score_%j.err
#
# Score one system in the one-shot space: map its native predictions (benchmark/to_oneshot.py),
# then run the two scorers our own folds use, tools/eval_oneshot.py and
# tools/decode_sequence.py --onehot. CPU only. Runs inside containers/spinesurg-ct.sif.
#
#   SYSTEM=totalsegmentator sbatch benchmark/score.sh
#   SYSTEM=vista3d sbatch benchmark/score.sh
#   SYSTEM=spineps PATTERN='{case}_seg-vert_msk.nii.gz' sbatch benchmark/score.sh
#   SYSTEM=ours_fold2 PRED_DIR=<fold_2/eval_.../pred> NAME_MAP=identity sbatch benchmark/score.sh
#
# Output: $BENCH/<SYSTEM>/oneshot/*.nii.gz, $BENCH/<SYSTEM>/score/{report.txt,summary.json,
# per_case.csv,decode.csv}
set -uo pipefail
export PATH="${HOME}/mambaforge/envs/nextflow/bin:${PATH}"
unset PYTHONPATH LD_LIBRARY_PATH

SYSTEM="${SYSTEM:?set SYSTEM}"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-${HOME}/spinesurg-ct-nnunet}"
BENCH="${BENCH:-${HOME}/bench}"
CASES="${CASES:-${PROJECT_ROOT}/benchmark/cases.csv}"
SIF="${SIF:-${PROJECT_ROOT}/containers/spinesurg-ct.sif}"
DS="${PROJECT_ROOT}/nnunet/raw/Dataset810_SpineSurgLSTVOneShot"
PRED_DIR="${PRED_DIR:-${BENCH}/${SYSTEM}/native}"
PATTERN="${PATTERN:-}"; [[ -n "${PATTERN}" ]] || PATTERN='{case}.nii.gz'   # braces inside ${:-} mis-parse
NAME_MAP="${NAME_MAP:-${PROJECT_ROOT}/benchmark/mappings/${SYSTEM}.json}"
CLASS_MAP="${CLASS_MAP:-}"
[[ "${SYSTEM}" == "totalsegmentator" && -z "${CLASS_MAP}" ]] && CLASS_MAP="${BENCH}/mappings/totalsegmentator_class_map.json"
ONESHOT="${BENCH}/${SYSTEM}/oneshot"
SCORE="${BENCH}/${SYSTEM}/score"
mkdir -p "${ONESHOT}" "${SCORE}" "${PROJECT_ROOT}/logs"

export SINGULARITY_TMPDIR="/tmp/${USER}_${SLURM_JOB_ID}_bench_score"
mkdir -p "${SINGULARITY_TMPDIR}"; trap 'rm -rf "${SINGULARITY_TMPDIR}"' EXIT
run() { singularity exec --bind "${PROJECT_ROOT}:/workspace" --bind "${HOME}:${HOME}" --pwd /workspace "${SIF}" "$@"; }

echo "score ${SYSTEM}: preds ${PRED_DIR} (${PATTERN}) $(date)"
if [[ "${NAME_MAP}" == "identity" ]]; then
    # already in the one-shot space (our own network): score in place
    ONESHOT="${PRED_DIR}"
elif [[ "${SYSTEM}" == spineps* ]]; then
    # vertebra instances + Möller's rib assignment composed into one volume per case
    run python /workspace/benchmark/spineps_to_oneshot.py --cases "${CASES}" --native "${PRED_DIR}" \
        --ribs "${RIBS_DIR:-${BENCH}/spineps/ribs}" --dataset_json "${DS}/dataset.json" --out_dir "${ONESHOT}" || exit 1
else
    CM=(); [[ -n "${CLASS_MAP}" ]] && CM=(--class_map "${CLASS_MAP}")
    run python /workspace/benchmark/to_oneshot.py --cases "${CASES}" --pred_dir "${PRED_DIR}" \
        --out_dir "${ONESHOT}" --dataset_json "${DS}/dataset.json" --name_map "${NAME_MAP}" \
        --pattern "${PATTERN}" "${CM[@]}" || exit 1
fi
run python /workspace/tools/eval_oneshot.py --predictions_dir "${ONESHOT}" --labels_dir "${DS}/labelsTr" \
    --dataset_json "${DS}/dataset.json" --lstv_cases_json "${DS}/lstv_cases.json" \
    --output_dir "${SCORE}" --workers "${SLURM_CPUS_PER_TASK:-8}" || exit 1
# hard labels (every competitor, and ground truth) decode one-hot; our folds have softmax .npz
ONEHOT=(--onehot); [[ "${PROBS:-0}" == "1" ]] && ONEHOT=()
run python /workspace/tools/decode_sequence.py --predictions_dir "${ONESHOT}" --dataset_json "${DS}/dataset.json" \
    --labels_dir "${DS}/labelsTr" --out "${SCORE}/decode.csv" "${ONEHOT[@]}" || exit 1
echo "=============== ${SYSTEM} DONE $(date) ==============="
head -60 "${SCORE}/report.txt"
