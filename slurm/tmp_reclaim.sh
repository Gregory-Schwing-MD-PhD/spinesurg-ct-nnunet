#!/bin/bash
#SBATCH --job-name=tmp_reclaim
#SBATCH --partition=reqp
#SBATCH --qos=requeue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=/wsu/home/go/go24/go2432/spinesurg-ct-nnunet/logs/tmp_reclaim_%N_%j.out
#SBATCH --error=/wsu/home/go/go24/go2432/spinesurg-ct-nnunet/logs/tmp_reclaim_%N_%j.err
#
# Reclaim node-local /tmp left behind by fold jobs that died before their EXIT trap ran.
# Deliberately NO --gres: the stock cleanup_tmp.sh asks for an H200 to delete files, which
# holds a training GPU idle for the length of the delete.
#
# ONLY removes directories whose job id is no longer in the queue. `rm -rf /tmp/go2432_*`
# would also take out a live job's staging directory, and on this cluster that is how two
# 18-hour runs were lost once already.
set -u
echo "node $(hostname)  $(date)"
df -h /tmp | tail -1

live=$(squeue -u "$USER" -h -o %i | tr '\n' ' ')
echo "live job ids: ${live:-<none>}"

shopt -s nullglob
for d in /tmp/go2432_* /tmp/"$USER"_*; do
    [[ -d "$d" ]] || continue
    jid=$(basename "$d" | grep -oE '[0-9]{6,}' | head -1)
    if [[ -n "$jid" ]] && grep -qw "$jid" <<<"$live"; then
        echo "  KEEP  $d (job $jid still queued/running)"
        continue
    fi
    sz=$(du -sh "$d" 2>/dev/null | cut -f1)
    echo "  RM    $d  ($sz)"
    rm -rf "$d"
done

echo "after:"
df -h /tmp | tail -1
