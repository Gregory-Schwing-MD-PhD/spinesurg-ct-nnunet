#!/usr/bin/env bash
# =============================================================================
# spinesurg-ct-nnunet — HPC Singularity/Apptainer Pull Script
# containers/build_container.sh
#
# Run this ON THE HPC GRID (warrior or equivalent).
# Pulls the Docker Hub image and converts it to a Singularity .sif container.
#
# Prerequisites on HPC:
#   - Apptainer or Singularity available (module load apptainer|singularity)
#   - Internet access from the login/compute node (or proxy configured)
#   - Sufficient disk space: ~10 GB
#
# Usage:
#   bash containers/build_container.sh
#
# Environment overrides:
#   BUILDER=singularity|apptainer   (default: auto-detect)
#   DOCKERHUB_USER=<user>           (default: gregoryschwingmdphd)
#   IMAGE=<name>                    (default: spinesurg-ct)
#   TAG=<tag>                       (default: latest)
#   OUTPUT=<path>                   (default: containers/spinesurg-ct.sif)
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DOCKERHUB_USER="${DOCKERHUB_USER:-gregoryschwingmdphd}"
IMAGE="${IMAGE:-spinesurg-ct}"
TAG="${TAG:-latest}"
DOCKER_URI="docker://${DOCKERHUB_USER}/${IMAGE}:${TAG}"
OUTPUT="${OUTPUT:-${HERE}/spinesurg-ct.sif}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "[ERROR] $*" >&2; exit 1; }

# ── Pick builder ──────────────────────────────────────────────────────────────
if [[ -n "${BUILDER:-}" ]]; then
    :
elif command -v apptainer >/dev/null 2>&1; then
    BUILDER="apptainer"
elif command -v singularity >/dev/null 2>&1; then
    BUILDER="singularity"
else
    if module load apptainer 2>/dev/null && command -v apptainer >/dev/null 2>&1; then
        BUILDER="apptainer"
    elif module load singularity 2>/dev/null && command -v singularity >/dev/null 2>&1; then
        BUILDER="singularity"
    else
        die "neither apptainer nor singularity found. Try: module load apptainer"
    fi
fi

mkdir -p "$(dirname "${OUTPUT}")"

echo "============================================================"
log "=== spinesurg-ct-nnunet HPC Singularity Pull ==="
log "Builder        : ${BUILDER}"
log "Docker Hub user: ${DOCKERHUB_USER}"
log "Source         : ${DOCKER_URI}"
log "Output         : ${OUTPUT}"
echo "============================================================"

# ── Pull training image ───────────────────────────────────────────────────────
log "Pulling image …"
"${BUILDER}" pull --force "${OUTPUT}" "${DOCKER_URI}"
log "  ✓  ${OUTPUT}  ($(du -sh "${OUTPUT}" | cut -f1))"

# ── Sanity test ───────────────────────────────────────────────────────────────
echo ""
log "Sanity test:"
"${BUILDER}" exec --nv "${OUTPUT}" python -c "
import torch, monai, nnunetv2, wandb
from importlib.metadata import version
print(f'nnU-Net  {version(\"nnunetv2\")}')
print(f'torch    {torch.__version__}  /  CUDA {torch.version.cuda}')
print(f'MONAI    {monai.__version__}')
print(f'wandb    {wandb.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"

# ── Usage reminders ───────────────────────────────────────────────────────────
echo ""
echo "  Training job:"
echo "    sbatch slurm/nnet_baseline.sh"
echo ""
echo "  Manual exec (example):"
echo "    ${BUILDER} exec --nv \\"
echo "      --bind \$(pwd):/workspace \\"
echo "      --bind \$(pwd)/data/hf_export:/data/hf_export \\"
echo "      --pwd /workspace \\"
echo "      ${OUTPUT} \\"
echo "      python tools/convert_hf_to_nnunet.py --help"
echo ""
log "Pull complete."
