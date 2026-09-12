#!/bin/bash
#SBATCH --job-name=consolidate
#SBATCH --partition=reqp
#SBATCH --qos=requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/wsu/home/go/go24/go2432/spinesurg-ct-nnunet/logs/consolidate_%j.out
#SBATCH --error=/wsu/home/go/go24/go2432/spinesurg-ct-nnunet/logs/consolidate_%j.err
set -u
/wsu/home/go/go24/go2432/mambaforge/envs/spinedet/bin/python   /wsu/home/go/go24/go2432/spinesurg-ct-nnunet/tools/consolidate_splits.py   /wsu/home/go/go24/go2432/spinesurg-ct-nnunet/nnunet/preprocessed/Dataset813_SpineSurgLSTVFullRibs   --val-frac 0.15
