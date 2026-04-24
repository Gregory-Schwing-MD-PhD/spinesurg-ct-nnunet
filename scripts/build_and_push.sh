#!/usr/bin/env bash
# =============================================================================
# spinesurg-ct-nnunet — Docker Build & Push Script
# scripts/build_and_push.sh
#
# Run this on your LOCAL WORKSTATION.
# Builds the training image from docker/Dockerfile and pushes it to Docker Hub.
# The Warrior-side pull is handled by containers/build_container.sh.
#
# Prerequisites:
#   - Docker Desktop (or Docker Engine) running
#   - Logged in to Docker Hub:  docker login
#
# Usage:
#   chmod +x scripts/build_and_push.sh
#   ./scripts/build_and_push.sh
#
#   # Override user or tag:
#   DOCKERHUB_USER=myuser TAG=v2.4.0 ./scripts/build_and_push.sh
#
#   # Build without pushing (smoke test):
#   PUSH=0 ./scripts/build_and_push.sh
# =============================================================================

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
DOCKERHUB_USER="${DOCKERHUB_USER:-gregoryschwingmdphd}"
IMAGE_NAME="${IMAGE_NAME:-spinesurg-ct}"
TAG="${TAG:-latest}"
PUSH="${PUSH:-1}"

IMAGE="${DOCKERHUB_USER}/${IMAGE_NAME}:${TAG}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "[ERROR] $*" >&2; exit 1; }

# ── Validate ──────────────────────────────────────────────────────────────────
command -v docker >/dev/null 2>&1 || die "Docker not found. Install Docker Desktop."
[[ -f docker/Dockerfile ]]        || die "Run from the repository root (docker/Dockerfile not found)."

log "=== spinesurg-ct-nnunet Docker Build & Push ==="
log "Docker Hub user : ${DOCKERHUB_USER}"
log "Image           : ${IMAGE}"
log "Push            : ${PUSH}"

# ── Build ─────────────────────────────────────────────────────────────────────
log "Step 1/2 — Building image: ${IMAGE}"
docker build \
    --file docker/Dockerfile \
    --tag  "${IMAGE}" \
    --progress=plain \
    .

# ── Push ──────────────────────────────────────────────────────────────────────
if [[ "${PUSH}" == "1" ]]; then
    log "Step 2/2 — Pushing image to Docker Hub …"
    docker push "${IMAGE}"
    log "  ✓  ${IMAGE}"
else
    log "Step 2/2 — Skipped (PUSH=0)."
fi

# ── Next steps ────────────────────────────────────────────────────────────────
echo ""
echo "  ┌──────────────────────────────────────────────────────────────────┐"
echo "  │  Now on Warrior, run:                                            │"
echo "  │    bash containers/build_container.sh                            │"
echo "  │                                                                   │"
echo "  │  Or directly:                                                     │"
echo "  │    singularity pull containers/spinesurg-ct.sif \\                │"
echo "  │        docker://${IMAGE}         │"
echo "  └──────────────────────────────────────────────────────────────────┘"
log "Done."
