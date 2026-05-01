#!/usr/bin/env bash
# Submit 5 folds with deterministic node assignment.
# msa1 hosts folds 0 AND 4; msa2-4 each host one fold.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
NODES=(msa1 msa2 msa3 msa4 msa1)
for f in 0 1 2 3 4; do
    sbatch --nodelist=${NODES[$f]} --array=$f slurm/spine_train_array.sh
done
