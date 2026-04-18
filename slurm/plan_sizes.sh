#!/bin/bash
#SBATCH --job-name=spine_plan_sizes
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/plan_sizes_%j.out
#SBATCH --error=logs/plan_sizes_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=go2432@wayne.edu
# =============================================================================
# SpineSurg-CT  --  plan (no preprocess) for multiple GPU memory targets
# =============================================================================
# Produces one plans.json per target in nnunet/preprocessed/<dataset>/.
# This is FAST (~5-10 min total for 5 targets) because it only runs:
#   1. extract_fingerprint (once, cached)
#   2. plan_experiment (per target, just generates the plans JSON)
# NO preprocessing happens -- that's what spine_prep.sh does later.
#
# After running this:
#   - Read the per-target patch sizes in the summary table below
#   - Optionally visualize with tools/visualize_patch_sizes.py
#   - Pick your GPU_MEMORY_TARGET_GB, then run sbatch slurm/spine_prep.sh
#
# Idempotent: skips plan generation if the plans JSON already exists.
# Override with OVERWRITE=1.
#
# USAGE
# -----
#   sbatch slurm/plan_sizes.sh
#   TARGETS="60 80 100 120 140" PLANNER=nnUNetPlannerResEncM sbatch slurm/plan_sizes.sh
#   PLANNER=nnUNetPlannerResEncL sbatch slurm/plan_sizes.sh     # bigger encoder
#   OVERWRITE=1 sbatch slurm/plan_sizes.sh
# =============================================================================

set -euo pipefail

DATASET_ID="${DATASET_ID:-802}"
DATASET_NAME="${DATASET_NAME:-SpineSurgCTFull}"
PLANNER="${PLANNER:-nnUNetPlannerResEncM}"
TARGETS="${TARGETS:-60 80 100 120 140}"
CONFIG="${CONFIG:-3d_fullres}"
OVERWRITE="${OVERWRITE:-0}"

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
NNUNET_NFS="${PROJECT_ROOT}/nnunet"
CONTAINER="${PROJECT_ROOT}/containers/spinesurg-ct.sif"
DS_DIR_NAME="Dataset$(printf '%03d' ${DATASET_ID})_${DATASET_NAME}"
NFS_PREP_DS="${NNUNET_NFS}/preprocessed/${DS_DIR_NAME}"

mkdir -p logs "${NFS_PREP_DS}"

# ----- Preflight -----------------------------------------------------------
[[ ! -f "${CONTAINER}" ]] && { echo "ERROR: container missing" >&2; exit 1; }
if [[ ! -d "${NNUNET_NFS}/raw/${DS_DIR_NAME}" ]]; then
    echo "ERROR: raw dataset ${NNUNET_NFS}/raw/${DS_DIR_NAME} missing." >&2
    echo "       Run slurm/spine_prep.sh (it calls convert first)" >&2
    echo "       OR run convert_hf_to_nnunet.py manually to build the raw layout." >&2
    exit 1
fi

# Compute plans-name prefix from planner
case "${PLANNER}" in
    nnUNetPlannerResEncM)  PLANS_PREFIX="nnUNetResEncUNetPlans"   ;;
    nnUNetPlannerResEncL)  PLANS_PREFIX="nnUNetResEncUNetLPlans"  ;;
    nnUNetPlannerResEncXL) PLANS_PREFIX="nnUNetResEncUNetXLPlans" ;;
    ExperimentPlanner)     PLANS_PREFIX="nnUNetPlans"             ;;
    *) echo "ERROR: unknown planner ${PLANNER}" >&2; exit 1 ;;
esac

# ----- Singularity ----------------------------------------------------------
export SINGULARITYENV_nnUNet_raw="/nnunet_nfs/raw"
export SINGULARITYENV_nnUNet_preprocessed="/nnunet_nfs/preprocessed"
export SINGULARITYENV_nnUNet_results="/nnunet_nfs/results"
export SINGULARITY_TMPDIR="/tmp/${USER}_plan_sizes_${SLURM_JOB_ID:-$$}"
export XDG_RUNTIME_DIR="${SINGULARITY_TMPDIR}/runtime"
mkdir -p "${SINGULARITY_TMPDIR}" "${XDG_RUNTIME_DIR}"
trap "rm -rf ${SINGULARITY_TMPDIR}" EXIT

SING_BINDS=(
    --bind "${PROJECT_ROOT}:/workspace"
    --bind "${NNUNET_NFS}:/nnunet_nfs"
    --pwd  /workspace
)

echo "================================================================"
echo " plan_sizes"
echo "   Dataset : ${DS_DIR_NAME}"
echo "   Planner : ${PLANNER}"
echo "   Targets : ${TARGETS} (GB)"
echo "   Prefix  : ${PLANS_PREFIX}"
echo "================================================================"

# ----- Stage 1: fingerprint (once, cached) ---------------------------------
FP_FILE="${NFS_PREP_DS}/dataset_fingerprint.json"
if [[ -f "${FP_FILE}" && "${OVERWRITE}" != "1" ]]; then
    echo "[fingerprint] cached"
else
    echo "[fingerprint] extracting..."
    singularity exec "${SING_BINDS[@]}" "${CONTAINER}" \
        nnUNetv2_extract_fingerprint -d ${DATASET_ID} --verify_dataset_integrity
fi

# ----- Stage 2: plan per target --------------------------------------------
echo ""
echo "Planning for each GPU memory target (no preprocess):"
for T in ${TARGETS}; do
    PLANS_NAME="${PLANS_PREFIX}_${T}G"
    PLANS_FILE="${NFS_PREP_DS}/${PLANS_NAME}.json"
    if [[ -f "${PLANS_FILE}" && "${OVERWRITE}" != "1" ]]; then
        echo "  [${T}G] cached (${PLANS_NAME}.json)"
        continue
    fi
    echo "  [${T}G] planning -> ${PLANS_NAME}..."
    singularity exec "${SING_BINDS[@]}" "${CONTAINER}" \
        nnUNetv2_plan_experiment \
            -d ${DATASET_ID} \
            -pl ${PLANNER} \
            -gpu_memory_target ${T} \
            -overwrite_plans_name ${PLANS_NAME}
done

# ----- Summary table -------------------------------------------------------
echo ""
echo "================================================================"
echo " Summary: patch sizes per target (${CONFIG})"
echo "================================================================"
printf "  %-8s %-40s %-30s %s\n" "TARGET" "PATCH (voxels)" "PATCH (mm)" "SPACING (mm)"
for T in ${TARGETS}; do
    PLANS_NAME="${PLANS_PREFIX}_${T}G"
    PLANS_FILE="${NFS_PREP_DS}/${PLANS_NAME}.json"
    if [[ ! -f "${PLANS_FILE}" ]]; then
        printf "  %-8s %s\n" "${T}G" "<missing>"
        continue
    fi
    python3 - "${PLANS_FILE}" "${CONFIG}" "${T}" <<'PYEOF'
import json, sys
plans = json.load(open(sys.argv[1]))
cfg_key = sys.argv[2]
T = sys.argv[3]
cfg = plans.get("configurations", {}).get(cfg_key, {})
patch = cfg.get("patch_size") or [0,0,0]
sp = cfg.get("spacing") or plans.get("original_median_spacing_after_transp") or [0,0,0]
patch_mm = [patch[i] * sp[i] for i in range(3)] if sp and all(sp) else [0,0,0]
patch_str = f"[{patch[0]},{patch[1]},{patch[2]}]"
patch_mm_str = f"{patch_mm[0]:.0f}x{patch_mm[1]:.0f}x{patch_mm[2]:.0f}"
sp_str = f"{sp[0]:.2f}x{sp[1]:.2f}x{sp[2]:.2f}" if sp and all(sp) else "?"
print(f"  {T+'G':<8s} {patch_str:<40s} {patch_mm_str:<30s} {sp_str}")
PYEOF
done

echo ""
echo "NEXT"
echo "  1. Eyeball the patch sizes above; aim for coverage of your anatomy of"
echo "     interest (lumbar region is ~200-300 mm SI; full lumbopelvis ~400 mm)."
echo ""
echo "  2. Visualize on one of your CTs for intuition:"
echo "     python tools/visualize_patch_sizes.py \\"
echo "         --ct      data/hf_export/ct/<your case>.nii.gz \\"
for T in ${TARGETS}; do
    echo "         --plans   nnunet/preprocessed/${DS_DIR_NAME}/${PLANS_PREFIX}_${T}G.json \\"
done
echo "         --out     patch_size_comparison.png"
echo ""
echo "  3. Pick a target, set GPU_MEMORY_TARGET_GB in slurm/spine_prep.sh,"
echo "     then sbatch slurm/spine_prep.sh (preprocess only the chosen plans)."
