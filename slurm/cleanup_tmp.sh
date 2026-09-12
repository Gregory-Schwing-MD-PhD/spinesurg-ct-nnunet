#!/usr/bin/env bash
#SBATCH --job-name=tmp_clean
#SBATCH -q gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --time=00:15:00
#SBATCH --output=logs/tmp_clean_%N_%j.out
#SBATCH --error=logs/tmp_clean_%N_%j.err
# Remove the staging folders that failed fold jobs left in this node's /tmp (their trap only
# removed the singularity tmpdir), and report the node-local disk afterwards.
echo "node $(hostname) $(date)"
df -h /tmp | tail -1
du -sh /tmp/go2432_* 2>/dev/null
rm -rf /tmp/go2432_*
df -h /tmp | tail -1
