#!/bin/bash
#SBATCH --job-name=spine_plan_viz
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/plan_viz_%j.out
#SBATCH --error=logs/plan_viz_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=go2432@wayne.edu
# =============================================================================
# SpineSurg-CT  --  plan + visualize patch sizes in one job
# =============================================================================
# Combines slurm/plan_sizes.sh + tools/visualize_patch_sizes.py:
#   1. extract_fingerprint (once, cached)
#   2. plan_experiment per GPU memory target -> one plans.json each
#   3. Summary table of patch sizes (voxels, mm, spacing)
#   4. Render an overlay figure showing every candidate patch on a real
#      CT, so you can pick a memory target by eyeballing anatomy coverage
#      rather than reading numbers.
#
# All boxes drawn with low-alpha edges + transparent fill so they
# overlap legibly on one panel per axis. Smallest patch on top so it's
# never hidden by larger boxes.
#
# After this job finishes:
#   scp warrior:~/spinesurg-ct-nnunet/figures/patch_size_comparison.png .
#   open patch_size_comparison.png
#
# Then pick a target and run:
#   GPU_MEMORY_TARGET_GB=<picked> sbatch slurm/spine_prep.sh
#
# Idempotent: skips plan generation if plans JSON already exists. Always
# regenerates the figure (cheap). Override plans with OVERWRITE=1.
#
# USAGE
# -----
#   sbatch slurm/plan_and_visualize.sh
#   CT_TOKEN=189 sbatch slurm/plan_and_visualize.sh
#   TARGETS="60 80 100 120 140" sbatch slurm/plan_and_visualize.sh
#   PLANNER=nnUNetPlannerResEncL OVERWRITE=1 sbatch slurm/plan_and_visualize.sh
# =============================================================================

set -uo pipefail

# ── Singularity runtime ────────────────────────────────────────────────────
export SINGULARITY_TMPDIR="/tmp/${USER}_job_${SLURM_JOB_ID:-$$}"
export XDG_RUNTIME_DIR="${SINGULARITY_TMPDIR}/runtime"
export NXF_SINGULARITY_CACHEDIR="${HOME}/singularity_cache"
mkdir -p "${SINGULARITY_TMPDIR}" "${XDG_RUNTIME_DIR}" "${NXF_SINGULARITY_CACHEDIR}"
trap 'rm -rf "${SINGULARITY_TMPDIR}"' EXIT
export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
unset JAVA_HOME LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE
export NXF_SINGULARITY_HOME_MOUNT=true

# ── Config ─────────────────────────────────────────────────────────────────
DATASET_ID="${DATASET_ID:-802}"
DATASET_NAME="${DATASET_NAME:-SpineSurgCTFull}"
PLANNER="${PLANNER:-nnUNetPlannerResEncM}"
TARGETS="${TARGETS:-60 80 100 120 140}"
CONFIG="${CONFIG:-3d_fullres}"
OVERWRITE="${OVERWRITE:-0}"

# Which CT to render the patches over.
#   - CT_PATH:  explicit absolute or repo-relative path (wins if set)
#   - CT_TOKEN: just the token (e.g. "189"); script picks a file automatically
#   - default:  pick the first reasonable CT in data/hf_export/ct/
CT_PATH="${CT_PATH:-}"
CT_TOKEN="${CT_TOKEN:-}"

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
HF_EXPORT_NFS="${PROJECT_ROOT}/data/hf_export"
HF_EXPORT_REAL=$(readlink -f "${HF_EXPORT_NFS}")
NNUNET_NFS="${PROJECT_ROOT}/nnunet"
CONTAINER="${PROJECT_ROOT}/containers/spinesurg-ct.sif"
DS_DIR_NAME="Dataset$(printf '%03d' ${DATASET_ID})_${DATASET_NAME}"
NFS_PREP_DS="${NNUNET_NFS}/preprocessed/${DS_DIR_NAME}"
FIGURES_DIR="${PROJECT_ROOT}/figures"
OUT_PNG="${FIGURES_DIR}/patch_size_comparison.png"

mkdir -p logs "${NFS_PREP_DS}" "${FIGURES_DIR}"

# ── Preflight ──────────────────────────────────────────────────────────────
[[ ! -f "${CONTAINER}" ]] && { echo "ERROR: container missing" >&2; exit 1; }
if [[ ! -d "${NNUNET_NFS}/raw/${DS_DIR_NAME}" ]]; then
    echo "ERROR: raw dataset ${NNUNET_NFS}/raw/${DS_DIR_NAME} missing." >&2
    echo "       Run slurm/spine_prep.sh (or slurm/repair_symlinks.sh) first." >&2
    exit 1
fi
[[ ! -d "${HF_EXPORT_REAL}/ct" ]] && { echo "ERROR: ${HF_EXPORT_REAL}/ct missing" >&2; exit 1; }

# Compute plans-name prefix from planner
case "${PLANNER}" in
    nnUNetPlannerResEncM)  PLANS_PREFIX="nnUNetResEncUNetPlans"   ;;
    nnUNetPlannerResEncL)  PLANS_PREFIX="nnUNetResEncUNetLPlans"  ;;
    nnUNetPlannerResEncXL) PLANS_PREFIX="nnUNetResEncUNetXLPlans" ;;
    ExperimentPlanner)     PLANS_PREFIX="nnUNetPlans"             ;;
    *) echo "ERROR: unknown planner ${PLANNER}" >&2; exit 1 ;;
esac

# ── Resolve CT path ────────────────────────────────────────────────────────
# Picks the first CT we can find given the user-supplied hint, or scans the
# directory if nothing was given. Bail out early if nothing matches so we
# don't waste 10 min planning before a typo blows up the viz step.
if [[ -n "${CT_PATH}" ]]; then
    if [[ ! -f "${CT_PATH}" && -f "${PROJECT_ROOT}/${CT_PATH}" ]]; then
        CT_PATH="${PROJECT_ROOT}/${CT_PATH}"
    fi
elif [[ -n "${CT_TOKEN}" ]]; then
    # Try a few common patterns (the dataset uses <token>_unknown_ct.nii.gz)
    for pat in "${CT_TOKEN}_unknown_ct.nii.gz" "${CT_TOKEN}.nii.gz" "${CT_TOKEN}_*.nii.gz"; do
        candidate=$(ls "${HF_EXPORT_REAL}/ct/"${pat} 2>/dev/null | head -n 1 || true)
        [[ -n "${candidate}" ]] && { CT_PATH="${candidate}"; break; }
    done
else
    # No hint -- pick the first numerically-named CT we can find. Skips
    # CTC- prefixed files because those are colonography-only and may have
    # a tighter FOV.
    CT_PATH=$(ls "${HF_EXPORT_REAL}/ct/" 2>/dev/null \
              | grep -E '^[0-9]+_.*\.nii\.gz$' \
              | sort -n | head -n 1 \
              | awk -v d="${HF_EXPORT_REAL}/ct" '{print d "/" $0}')
fi
if [[ -z "${CT_PATH}" || ! -f "${CT_PATH}" ]]; then
    echo "ERROR: could not resolve a CT to render. Tried:" >&2
    echo "   CT_PATH=${CT_PATH:-<unset>}" >&2
    echo "   CT_TOKEN=${CT_TOKEN:-<unset>}" >&2
    echo "   Set CT_PATH=/abs/path/to/case.nii.gz or CT_TOKEN=<token>." >&2
    exit 1
fi

# ── Singularity binds (used for both planning and viz) ─────────────────────
export SINGULARITYENV_nnUNet_raw="/nnunet_nfs/raw"
export SINGULARITYENV_nnUNet_preprocessed="/nnunet_nfs/preprocessed"
export SINGULARITYENV_nnUNet_results="/nnunet_nfs/results"

SING_BINDS=(
    --bind "${PROJECT_ROOT}:/workspace"
    --bind "${HF_EXPORT_REAL}:${HF_EXPORT_REAL}"   # mirrored bind so abs paths work
    --bind "${NNUNET_NFS}:/nnunet_nfs"
    --pwd  /workspace
)

echo "================================================================"
echo " plan + visualize"
echo "   Dataset : ${DS_DIR_NAME}"
echo "   Planner : ${PLANNER}"
echo "   Targets : ${TARGETS} (GB)"
echo "   Prefix  : ${PLANS_PREFIX}"
echo "   CT      : ${CT_PATH}"
echo "   Out PNG : ${OUT_PNG}"
echo "================================================================"

# ── Stage 1: fingerprint (once, cached) ───────────────────────────────────
FP_FILE="${NFS_PREP_DS}/dataset_fingerprint.json"
if [[ -f "${FP_FILE}" && "${OVERWRITE}" != "1" ]]; then
    echo "[fingerprint] cached"
else
    echo "[fingerprint] extracting..."
    singularity exec "${SING_BINDS[@]}" "${CONTAINER}" \
        nnUNetv2_extract_fingerprint -d ${DATASET_ID} --verify_dataset_integrity
fi

# ── Stage 2: plan per target ───────────────────────────────────────────────
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

# ── Stage 3: summary table ─────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " Summary: patch sizes per target (${CONFIG})"
echo "================================================================"
printf "  %-8s %-40s %-30s %s\n" "TARGET" "PATCH (voxels)" "PATCH (mm)" "SPACING (mm)"
PLANS_FILES=()
for T in ${TARGETS}; do
    PLANS_NAME="${PLANS_PREFIX}_${T}G"
    PLANS_FILE="${NFS_PREP_DS}/${PLANS_NAME}.json"
    if [[ ! -f "${PLANS_FILE}" ]]; then
        printf "  %-8s %s\n" "${T}G" "<missing>"
        continue
    fi
    PLANS_FILES+=( "${PLANS_FILE}" )
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

if [[ "${#PLANS_FILES[@]}" -eq 0 ]]; then
    echo "ERROR: no plans files were generated; cannot build figure." >&2
    exit 2
fi

# ── Stage 4: render the overlay figure ─────────────────────────────────────
# We invoke the same tools/visualize_patch_sizes.py that the README points
# to, but pass every plans file in one call so all candidate boxes land on
# the same axes. The script already picks distinct colors per box and
# leaves the fill transparent, so overlapping rectangles read cleanly.
echo ""
echo "----- Rendering patch overlay figure -----"
echo "  CT  : ${CT_PATH}"
echo "  PNG : ${OUT_PNG}"

VIZ_ARGS=( --ct "${CT_PATH}" --out "${OUT_PNG}" )
for PF in "${PLANS_FILES[@]}"; do
    VIZ_ARGS+=( --plans "${PF}" )
done

singularity exec "${SING_BINDS[@]}" "${CONTAINER}" \
    python tools/visualize_patch_sizes.py "${VIZ_ARGS[@]}"

echo ""
echo "================================================================"
echo " plan + visualize COMPLETE"
echo ""
echo " To view:"
echo "   scp ${USER}@warrior:${OUT_PNG} ."
echo "   open patch_size_comparison.png    # macOS"
echo "   xdg-open patch_size_comparison.png  # linux"
echo ""
echo " Decision criteria (sagittal panel = the spine running vertically):"
echo "   * Box ends above sacrum            -> too small; model can't see L5/S1"
echo "   * Box covers L1 to sacral promontory -> covers diagnostic region"
echo "   * Box covers L1 to ischial tuberosities -> ideal for Castellvi context"
echo ""
echo " Then commit a target:"
echo "   GPU_MEMORY_TARGET_GB=<picked> sbatch slurm/spine_prep.sh"
echo "================================================================"
