#!/bin/bash
# Build the spinesurg-ct Singularity/Apptainer container.
#
# Usage:
#   bash containers/build_container.sh
#
# Environment overrides:
#   BUILDER=singularity|apptainer   (default: auto-detect)
#   OUTPUT=<path>                   (default: containers/spinesurg-ct.sif)
#   USE_FAKEROOT=1                  (default: 1 if not root, else 0)
#   REMOTE=1                        (build via Sylabs cloud; requires remote auth)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEF_FILE="${HERE}/spinesurg-ct.def"
OUTPUT="${OUTPUT:-${HERE}/spinesurg-ct.sif}"

[[ ! -f "${DEF_FILE}" ]] && { echo "ERROR: ${DEF_FILE} missing" >&2; exit 1; }

# Pick builder
if [[ -n "${BUILDER:-}" ]]; then
    :
elif command -v apptainer >/dev/null 2>&1; then
    BUILDER="apptainer"
elif command -v singularity >/dev/null 2>&1; then
    BUILDER="singularity"
else
    echo "ERROR: neither apptainer nor singularity found on PATH" >&2
    echo "  On Wayne State Warrior cluster, try:  module load apptainer" >&2
    exit 1
fi

# Decide fakeroot default
if [[ -z "${USE_FAKEROOT:-}" ]]; then
    if [[ "$(id -u)" == "0" ]]; then
        USE_FAKEROOT=0
    else
        USE_FAKEROOT=1
    fi
fi

BUILD_ARGS=()
[[ "${USE_FAKEROOT}" == "1" ]] && BUILD_ARGS+=( --fakeroot )
[[ "${REMOTE:-0}"   == "1" ]] && BUILD_ARGS+=( --remote )
BUILD_ARGS+=( --force )

echo "============================================================"
echo " Building spinesurg-ct container"
echo "   Builder : ${BUILDER}"
echo "   Def     : ${DEF_FILE}"
echo "   Output  : ${OUTPUT}"
echo "   Fakeroot: ${USE_FAKEROOT}"
echo "   Remote  : ${REMOTE:-0}"
echo "============================================================"

"${BUILDER}" build "${BUILD_ARGS[@]}" "${OUTPUT}" "${DEF_FILE}"

echo ""
echo "Built: ${OUTPUT}"
ls -lh "${OUTPUT}"

echo ""
echo "Sanity test:"
"${BUILDER}" exec --nv "${OUTPUT}" python -c \
    "import nnunetv2, torch; print(f'nnU-Net {nnunetv2.__version__} / Torch {torch.__version__} / CUDA {torch.version.cuda}')"
