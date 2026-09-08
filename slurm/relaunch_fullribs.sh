#!/bin/bash
# Login node. Throw away the first numbered-rib prep (no disc_space class) and relaunch the
# whole chain with the disc-space scheme: kill the running switch script, cancel its prep /
# unpack / folds, delete the partial Dataset813 output, resubmit prep, restart the switch.
#
#   bash slurm/relaunch_fullribs.sh
set -uo pipefail
cd "${HOME}/spinesurg-ct-nnunet"
pkill -f "bash slurm/switch_to_fullribs.sh" && echo "killed switch script" || echo "no switch script running"
scancel 40164963 40165184 40165185 40165186 40165187 40165188 40165189 2>/dev/null
echo "cancelled first numbered-rib chain"
sleep 20
rm -rf nnunet/raw/Dataset813_SpineSurgLSTVFullRibs nnunet/preprocessed/Dataset813_SpineSurgLSTVFullRibs nnunet/results/Dataset813_SpineSurgLSTVFullRibs
echo "deleted partial Dataset813"
PREP=$(sbatch --parsable --export=ALL,SCHEME=fullribs,DATASET_ID=813,DATASET_NAME=SpineSurgLSTVFullRibs,PLANNER=nnUNetPlannerResEncL,HF_EXPORT_DIR=${HOME}/data/CTSpinoPelvic1K slurm/spine_prep.sh)
echo "prep ${PREP}"
PREP_JOB=${PREP} nohup bash slurm/switch_to_fullribs.sh > logs/switch_to_fullribs.log 2>&1 < /dev/null &
sleep 5
cat logs/switch_to_fullribs.log
squeue -u "${USER}" -o "%.12i %.14j %.8T %.10M %R"
