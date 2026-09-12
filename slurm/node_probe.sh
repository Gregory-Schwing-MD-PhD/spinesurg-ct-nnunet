#!/usr/bin/env bash
#SBATCH --job-name=node_probe
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=00:05:00
#SBATCH --output=logs/node_probe_%N_%j.out
#SBATCH --error=logs/node_probe_%N_%j.err
echo "node $(hostname)  SLURM_TMPDIR=${SLURM_TMPDIR:-unset}"
df -h /tmp /scratch /local /localscratch /dev/shm 2>/dev/null
ls -ld /scratch /scratch/$USER /local 2>/dev/null
du -sh /tmp/go2432_* 2>/dev/null | head
