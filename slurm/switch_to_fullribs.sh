#!/bin/bash
# Retire the one-shot arm (Dataset810) in favour of the numbered-rib arm (Dataset813) without
# handing a GPU slot to another user in between. Runs on the LOGIN node under nohup:
#
#   PREP_JOB=<fullribs prep job id> nohup bash slurm/switch_to_fullribs.sh > logs/switch_to_fullribs.log 2>&1 &
#
#   1. unpack job (npz -> npy on NFS) after the prep completes
#   2. five folds after the unpack (fold 4 stages to node-local /tmp: the staging A/B test)
#   3. once the unpack has completed, i.e. the folds are eligible: cancel the one-shot folds,
#      their staged fold 4 and the queued checkpoint evaluation; archive their training logs
#      (the training-curve figure was drawn from them); delete Dataset810 results and
#      preprocessed (945 GB). Dataset810 RAW stays: it is the benchmark's scoring space.
set -uo pipefail
cd "${HOME}/spinesurg-ct-nnunet"
PREP_JOB="${PREP_JOB:?set PREP_JOB}"
ONESHOT_JOBS="${ONESHOT_JOBS:-40163538 40163539 40163540 40163541 40164146 40164341}"
DS_ID=813; DS_NAME=SpineSurgLSTVFullRibs
COMMON="DATASET_ID=${DS_ID},DATASET_NAME=${DS_NAME},PLANNER=nnUNetPlannerResEncL,GPU_MEMORY_TARGET_GB=100,TRAINER=nnUNetTrainerWandB_500ep_LSTVOversample,LSTV_OVERSAMPLE_FRAC=0.5,SPINESURG_LABEL_SCHEME=fullribs"

state() { sacct -j "$1" -o State -n 2>/dev/null | head -1 | tr -d ' '; }

echo "$(date) waiting on prep ${PREP_JOB}"
UNPACK=$(sbatch --parsable --dependency=afterok:${PREP_JOB} --export=ALL,DATASET_ID=${DS_ID},DATASET_NAME=${DS_NAME} slurm/unpack_dataset.sh)
echo "$(date) unpack job ${UNPACK} (after prep)"
FOLDS=()
for f in 0 1 2 3 4; do
    sl=0; [[ ${f} -eq 4 ]] && sl=1
    j=$(sbatch --parsable --dependency=afterok:${UNPACK} --export=ALL,FOLD=${f},${COMMON},STAGE_LOCAL=${sl} slurm/spine_train_fold.sh)
    FOLDS+=("${j}"); echo "$(date) fold ${f} job ${j} (STAGE_LOCAL=${sl}, after unpack)"
done

# wait for the unpack: from then on the new folds are eligible for the freed GPUs
while :; do
    st=$(state "${UNPACK}")
    case "${st}" in
        COMPLETED) break ;;
        FAILED|CANCELLED*|TIMEOUT|NODE_FAIL)
            echo "$(date) unpack ${UNPACK} ended ${st}; prep $(state ${PREP_JOB}). NOT touching the one-shot folds."; exit 1 ;;
    esac
    sleep 300
done
echo "$(date) unpack ${UNPACK} COMPLETED; prep $(state ${PREP_JOB}); folds ${FOLDS[*]} now eligible"

mkdir -p "${HOME}/archive"
tar -czf "${HOME}/archive/oneshot_Dataset810_training_logs_$(date +%Y%m%d).tgz" \
    nnunet/results/Dataset810_SpineSurgLSTVOneShot/*/fold_*/training_log_*.txt \
    nnunet/results/Dataset810_SpineSurgLSTVOneShot/*/fold_*/progress.png 2>/dev/null
echo "$(date) archived one-shot logs: $(ls -la ${HOME}/archive/oneshot_Dataset810_training_logs_*.tgz | tail -1)"

scancel ${ONESHOT_JOBS}
echo "$(date) cancelled one-shot jobs ${ONESHOT_JOBS}"
sleep 90
rm -rf nnunet/results/Dataset810_SpineSurgLSTVOneShot nnunet/preprocessed/Dataset810_SpineSurgLSTVOneShot
echo "$(date) deleted Dataset810 results and preprocessed (raw kept for the benchmark)"
squeue -u "${USER}" -o "%.12i %.14j %.8T %.10M %R"
echo "$(date) done"
