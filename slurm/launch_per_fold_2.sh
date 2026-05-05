#!/usr/bin/env bash
# Submit 5 folds with deterministic node assignment, chained off the
# already-running spine_prep job 35995965.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PREP_JOBID=35995965

NODES=(msa1 msa2 msa3 msa4 msa1)
for f in 0 1 2 3 4; do
    JID=$(sbatch --parsable \
                  --nodelist="${NODES[$f]}" \
                  --array=$f \
                  --dependency=afterok:${PREP_JOBID} \
                  --kill-on-invalid-dep=yes \
                  slurm/spine_train_array.sh)
    echo "  fold ${f}: submitted to ${NODES[$f]}, jobid=${JID} (waiting on ${PREP_JOBID})"
done

echo ""
echo "All 5 folds queued. They will start when prep job ${PREP_JOBID} finishes."
echo "Watch: squeue --me"
