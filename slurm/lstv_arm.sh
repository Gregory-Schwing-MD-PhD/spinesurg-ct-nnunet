#!/bin/bash
#SBATCH --job-name=lstv_arm
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/lstv_arm_%j.out
#SBATCH --error=logs/lstv_arm_%j.out
# =============================================================================
# Submit one arm of the LSTV benchmark: prep, then five training folds chained
# behind it so a failed conversion never leaves five GPU jobs training on a
# half-written dataset.
#
# This is a SUBMITTER, not the work. It runs on a normal queue in under a
# minute and its only job is to get the dependency graph right, because the
# expensive mistake here is not a wrong hyperparameter -- it is five H200 jobs
# starting on a dataset that was never finished.
#
#   ARM=oneshot     sbatch slurm/lstv_arm.sh
#   ARM=rib_regions sbatch slurm/lstv_arm.sh
#   ARM=countfree   sbatch slurm/lstv_arm.sh
#
# Optional:
#   FOLDS="0 1 2 3 4"   which folds to train (default all five)
#   PLANNER=...         nnU-Net planner (default ResEnc-L)
#   DRY=1               print the submissions without making them
# =============================================================================
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

ARM="${ARM:-}"
FOLDS="${FOLDS:-0 1 2 3 4}"
PLANNER="${PLANNER:-nnUNetPlannerResEncL}"
DRY="${DRY:-0}"

# Each arm gets its own dataset id. Two arms sharing one would overwrite each
# other's preprocessed data, and that failure shows up as an unexplained change
# in accuracy rather than as an error -- the worst kind.
case "${ARM}" in
    oneshot)     DS_ID=810; DS_NAME=SpineSurgLSTVOneShot   ;;
    rib_regions) DS_ID=811; DS_NAME=SpineSurgLSTVRibRegions ;;
    countfree)   DS_ID=812; DS_NAME=SpineSurgLSTVCountFree  ;;
    *)
        echo "ERROR: set ARM to oneshot, rib_regions or countfree." >&2
        echo "  oneshot      21 classes; T10-T12 and L1-L6 distinct, twelfth rib kept" >&2
        echo "               apart from the lumbar rib. The only target on which the" >&2
        echo "               hypoplastic-versus-lumbar question is measurable." >&2
        echo "  rib_regions  count-free plus rib-bearing as a nested nnU-Net region." >&2
        echo "  countfree    no vertebral identity; counting happens downstream." >&2
        exit 2 ;;
esac

echo "================================================================"
echo " LSTV benchmark arm: ${ARM}"
echo "   dataset : Dataset${DS_ID}_${DS_NAME}"
echo "   planner : ${PLANNER}"
echo "   folds   : ${FOLDS}"
echo "================================================================"

submit() {
    if [[ "${DRY}" == "1" ]]; then
        # to stderr: stdout is captured by the caller and must carry ONLY the job id
        echo "  DRY: $*" >&2
        echo "dry_$RANDOM"
        return
    fi
    local out
    out=$("$@") || { echo "SUBMIT FAILED: $*" >&2; exit 1; }
    echo "${out##* }"
}

# -- Stage A: convert + preprocess -------------------------------------------
# spine_prep.sh runs tools/preflight_lstv.py first and exits non-zero if any
# structural check fails, so a bad label scheme stops here rather than at the
# end of a training run.
PREP_ID=$(submit sbatch --parsable \
    --export=ALL,SCHEME="${ARM}",DATASET_ID="${DS_ID}",DATASET_NAME="${DS_NAME}",PLANNER="${PLANNER}" \
    slurm/spine_prep.sh)
echo "  prep job: ${PREP_ID}"

# -- Stage B: five folds, each waiting on the prep ---------------------------
# afterok, not afterany: if the conversion fails, nothing trains. Five H200
# jobs starting on a half-written dataset is the expensive version of this
# mistake and it is entirely preventable by one word.
for f in ${FOLDS}; do
    TID=$(submit sbatch --parsable --dependency=afterok:"${PREP_ID}" \
        --export=ALL,FOLD="${f}",DATASET_ID="${DS_ID}",DATASET_NAME="${DS_NAME}" \
        slurm/spine_train_fold.sh)
    echo "  fold ${f}: ${TID}  (waits on ${PREP_ID})"
done

echo
echo "Submitted. Watch with:  squeue -u \$USER -o '%.10i %.14j %.8T %.10M %R'"
echo
echo "WHAT TO CHECK FIRST when the folds finish, before believing anything:"
echo "  1. rib-free count accuracy split by typical vs transitional anatomy."
echo "     A model 97% accurate overall and 0% on transitional cases is the null"
echo "     result wearing a good number, and only the split shows it."
echo "  2. the twelfth-rib versus lumbar-rib confusion matrix, read BY CLASS."
echo "     Among ribs 50 mm or shorter this corpus holds 152 twelfth ribs against"
echo "     10 lumbar ribs, so always saying 'twelfth' scores 94% on short ribs."
echo "  3. whether the rare classes appeared in training at all. nnU-Net's 33%"
echo "     foreground oversampling picks a class within an ALREADY-SELECTED case,"
echo "     and case selection is uniform -- so a class in 16 of 802 cases is seen"
echo "     a fraction of a percent of the time. See docs/LSTV_BENCHMARK.md."
