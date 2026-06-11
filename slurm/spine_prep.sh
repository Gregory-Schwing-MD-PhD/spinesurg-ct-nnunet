#!/bin/bash
#SBATCH --job-name=spine_prep
#SBATCH -q primary
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=8:00:00
#SBATCH --output=logs/spine_prep_%j.out
#SBATCH --error=logs/spine_prep_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=go2432@wayne.edu
# =============================================================================
# SpineSurg-CT  --  Stage A: convert + preprocess (SIMPLE, NFS-only)
# =============================================================================
# Convert writes symlinks like
#     imagesTr/X__fused_0000.nii.gz -> /data/hf_export/ct/X.nii.gz
# for CT IMAGES. That target is a CONTAINER path; it resolves inside the
# container via the /data/hf_export bind mount but looks "broken" from the
# host. Therefore: do NOT run any host-side dangling-symlink cleanup.
# Earlier versions did, and they wiped the whole dataset on every "resume"
# run.
#
# LABEL files (May 2026 v5+): under Dataset803 with the merged-label
# scheme, label NIfTIs are PHYSICALLY REWRITTEN by the convert script via
# a vectorized LUT (LABEL_REMAP_AT_CONVERT = {6:5, 7:6, 8:7, 9:8, 10:9}).
# This collapses L6 into last_lumbar and shifts sacrum/hips/ignore down
# by one for contiguous label IDs. Symlinking labels would be wrong
# under Dataset803 — the source NIfTIs still contain L6 voxels which the
# v20 trainer is not configured to predict. The convert script handles
# this correctly: --symlinks applies to CT only when a remap is active.
#
# Threading: 12 workers x 4 BLAS threads = 48 threads, fits RLIMIT_NPROC.
#
# v2 fix (2026-04-27)
# -------------------
# The preprocessed-data directory under preprocessed/<dataset>/ is named
# after the plans' data_identifier (e.g. "nnUNetPlans_3d_fullres"),
# NOT after ${CONFIG} (e.g. "3d_fullres"). resolve_prep_data_dir() now
# reads data_identifier from the plans JSON (with a glob fallback).
#
# v3 (2026-04-29) — flexible HF_EXPORT_DIR + SPLITS_FILE overrides
# ----------------------------------------------------------------
# HF_EXPORT_DIR + SPLITS_FILE env-var auto-discovery (see comments below).
#
# v4 (2026-05-01) — convert_hf_to_nnunet.py CLI updated
# ------------------------------------------------------
# CLI renamed: --hf_export_dir -> --hf_dir, --nnunet_raw_dir -> --nnunet_raw,
# --splits_file -> --splits.
#
# v5 (2026-05-03) — Dataset803 merged-label defaults
# ---------------------------------------------------
# Defaults flipped from Dataset802 (10-class unmerged) to Dataset803
# (9-class merged last_lumbar):
#
#   DATASET_ID:    802 -> 803
#   DATASET_NAME:  SpineSurgCTFull -> SpineSurgCTFullMerged
#
# To run the legacy 10-class baseline (e.g. for a paper baseline row),
# override on submit:
#   DATASET_ID=802 DATASET_NAME=SpineSurgCTFull \
#     sbatch slurm/spine_prep.sh
# AND pass --no_remap to convert_hf_to_nnunet.py (see notes below). The
# convert script accepts either; the trainer assumes contiguous labels
# and is currently incompatible with the unmerged scheme.
#
# Dataset.json sanity check (v5): now validates the ENTIRE label dict
# against the expected scheme for each dataset ID. For Dataset803 this
# means {bg:0, L1-L4:1-4, last_lumbar:5, sacrum:6, hips:7-8, ignore:9}.
# Catches the failure modes of (a) running v5 against a stale
# Dataset802 build, (b) running --no_remap accidentally.
# =============================================================================

set -euo pipefail

# -- Singularity runtime dirs ------------------------------------------------
export SINGULARITY_TMPDIR="/tmp/${USER}_job_${SLURM_JOB_ID}"
export XDG_RUNTIME_DIR="${SINGULARITY_TMPDIR}/runtime"
export NXF_SINGULARITY_CACHEDIR="${HOME}/singularity_cache"
mkdir -p "${SINGULARITY_TMPDIR}" "${XDG_RUNTIME_DIR}" "${NXF_SINGULARITY_CACHEDIR}"
trap 'rm -rf "${SINGULARITY_TMPDIR}"' EXIT
export CONDA_PREFIX="${HOME}/mambaforge/envs/nextflow"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
unset JAVA_HOME; which singularity
export NXF_SINGULARITY_HOME_MOUNT=true
unset LD_LIBRARY_PATH PYTHONPATH R_LIBS R_LIBS_USER R_LIBS_SITE

# -- Config ------------------------------------------------------------------
# v5 defaults: Dataset803 (merged last_lumbar). Override DATASET_ID=802 +
# DATASET_NAME=SpineSurgCTFull + LEGACY_NO_REMAP=1 to run the unmerged
# baseline.
DATASET_ID="${DATASET_ID:-803}"
DATASET_NAME="${DATASET_NAME:-SpineSurgCTFullMerged}"
LEGACY_NO_REMAP="${LEGACY_NO_REMAP:-0}"
CONFIG="${CONFIG:-3d_fullres}"
PLANNER="${PLANNER:-nnUNetPlannerResEncM}"
GPU_MEMORY_TARGET_GB="${GPU_MEMORY_TARGET_GB:-100}"
N_PREPROCESS_THREADS="${N_PREPROCESS_THREADS:-12}"
BLAS_THREADS="${BLAS_THREADS:-4}"
SINGLE_FOLD_SPLITS="${SINGLE_FOLD_SPLITS:-0}"
INCLUDE_TRAIN_MATCH_TYPES="${INCLUDE_TRAIN_MATCH_TYPES:-}"
TEST_MATCH_TYPES="${TEST_MATCH_TYPES:-}"

# Fused-only ablation (May 2026) — reviewer "fused-only vs all-cases" ask.
#   INCLUDE_CONFIGS=fused     keep only fused records (drop spine_only +
#                             pelvic_native partial-annotation records).
#   DROP_IGNORE_LABEL=1       build the 9-key no-ignore scheme (8 fg + bg);
#                             fused records carry no ignore voxels, so the
#                             ignore protocol is disabled entirely.
# Example (build the ablation dataset):
#   INCLUDE_CONFIGS=fused DROP_IGNORE_LABEL=1 \
#     DATASET_ID=804 DATASET_NAME=SpineSurgCTFusedOnly \
#     sbatch slurm/spine_prep.sh
INCLUDE_CONFIGS="${INCLUDE_CONFIGS:-}"
DROP_IGNORE_LABEL="${DROP_IGNORE_LABEL:-0}"

# Hard error on env vars that map to convert features that no longer
# exist (May 2026 CLI). Better to fail loudly than silently train on
# an unfiltered dataset.
if [[ "${SINGLE_FOLD_SPLITS}" == "1" ]]; then
    echo "ERROR: SINGLE_FOLD_SPLITS=1 is no longer supported by convert_hf_to_nnunet.py." >&2
    echo "       The new CLI requires --splits to point at a real splits_5fold.json." >&2
    exit 1
fi
if [[ -n "${INCLUDE_TRAIN_MATCH_TYPES}" || -n "${TEST_MATCH_TYPES}" ]]; then
    echo "ERROR: INCLUDE_TRAIN_MATCH_TYPES / TEST_MATCH_TYPES are no longer supported." >&2
    echo "       The convert script now uses ALL records from manifest_train.json + " >&2
    echo "       manifest_validation.json as training cases, and manifest_test.json " >&2
    echo "       as test cases. For the fused-only ablation use INCLUDE_CONFIGS=fused" >&2
    echo "       (config-level filter), not the removed match-type filters." >&2
    exit 1
fi

# Fused-only ablation guards (mirror convert_hf_to_nnunet.py).
if [[ "${DROP_IGNORE_LABEL}" == "1" && "${LEGACY_NO_REMAP}" == "1" ]]; then
    echo "ERROR: DROP_IGNORE_LABEL=1 is incompatible with LEGACY_NO_REMAP=1." >&2
    echo "       The no-ignore scheme is defined only on the merged contiguous" >&2
    echo "       label scheme; drop LEGACY_NO_REMAP for the ablation." >&2
    exit 1
fi
if [[ "${DROP_IGNORE_LABEL}" == "1" && -z "${INCLUDE_CONFIGS}" ]]; then
    echo "WARN: DROP_IGNORE_LABEL=1 without INCLUDE_CONFIGS — partial-annotation" >&2
    echo "      records (spine_only/pelvic_native) would have their ignore voxels" >&2
    echo "      folded into BACKGROUND, falsely supervising unannotated regions." >&2
    echo "      For the reviewer ablation pass INCLUDE_CONFIGS=fused." >&2
fi

# v5: warn loudly when Dataset802 is requested without --no_remap, since
# that combination produces a 9-class dataset under the 802 ID — almost
# certainly an operator mistake.
if [[ "${DATASET_ID}" == "802" && "${LEGACY_NO_REMAP}" != "1" ]]; then
    echo "WARN: DATASET_ID=802 requested but LEGACY_NO_REMAP=0." >&2
    echo "      Dataset802 conventionally means the legacy 10-class scheme." >&2
    echo "      Set LEGACY_NO_REMAP=1 to pass --no_remap to the convert script," >&2
    echo "      or use DATASET_ID=803 for the merged-label scheme." >&2
fi

case "${PLANNER}" in
    nnUNetPlannerResEncM)  PLANS="nnUNetResEncUNetPlans_${GPU_MEMORY_TARGET_GB}G"   ;;
    nnUNetPlannerResEncL)  PLANS="nnUNetResEncUNetLPlans_${GPU_MEMORY_TARGET_GB}G"  ;;
    nnUNetPlannerResEncXL) PLANS="nnUNetResEncUNetXLPlans_${GPU_MEMORY_TARGET_GB}G" ;;
    ExperimentPlanner)     PLANS="nnUNetPlans"                                     ;;
    *) echo "ERROR: unknown planner ${PLANNER}" >&2; exit 1 ;;
esac

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-${HOME}/spinesurg-ct-nnunet}"

# -- HF export discovery -----------------------------------------------------
if [[ -z "${HF_EXPORT_DIR:-}" ]]; then
    for cand in \
        "${PROJECT_ROOT}/data/hf_export" \
        "${HOME}/CTSpinoPelvic1K/data/hf_export"; do
        if [[ -d "${cand}/ct" ]]; then
            HF_EXPORT_DIR="${cand}"
            break
        fi
    done
fi
HF_EXPORT_DIR="${HF_EXPORT_DIR:-${PROJECT_ROOT}/data/hf_export}"
HF_EXPORT_NFS="${HF_EXPORT_DIR}"

# -- Splits file discovery ---------------------------------------------------
if [[ -z "${SPLITS_FILE:-}" ]]; then
    for cand in \
        "${HF_EXPORT_NFS}/splits_5fold.json" \
        "${PROJECT_ROOT}/data/splits_5fold.json"; do
        if [[ -f "${cand}" ]]; then
            SPLITS_FILE="${cand}"
            break
        fi
    done
fi
SPLITS_FILE_HOST="${SPLITS_FILE:-${HF_EXPORT_NFS}/splits_5fold.json}"

NNUNET_NFS="${PROJECT_ROOT}/nnunet"
CONTAINER="${PROJECT_ROOT}/containers/spinesurg-ct.sif"
WANDB_TRAINER_HOST="${PROJECT_ROOT}/tools/nnunet_wandb_variant.py"
WANDB_TRAINER_CONTAINER="/opt/conda/lib/python3.11/site-packages/nnunetv2/training/nnUNetTrainer/variants/nnunet_wandb_variant.py"

DS_DIR_NAME="Dataset$(printf '%03d' ${DATASET_ID})_${DATASET_NAME}"
NFS_RAW_DS="${NNUNET_NFS}/raw/${DS_DIR_NAME}"
NFS_PREP_DS="${NNUNET_NFS}/preprocessed/${DS_DIR_NAME}"
COMPLETE_MARKER="${NFS_PREP_DS}/.preprocessed_complete_${PLANS}"

mkdir -p "${NNUNET_NFS}/raw" "${NNUNET_NFS}/preprocessed" "${NNUNET_NFS}/results" \
         "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/tmp"

# -- Resolver: find the preprocessed-data subdirectory -----------------------
resolve_prep_data_dir() {
    local plans_json="${NFS_PREP_DS}/${PLANS}.json"
    local cfg="${CONFIG}"
    local resolved=""

    if [[ -f "${plans_json}" ]] && command -v python >/dev/null 2>&1; then
        resolved=$(python - "${plans_json}" "${cfg}" <<'PY' 2>/dev/null || true
import json, sys
try:
    with open(sys.argv[1]) as f:
        p = json.load(f)
    di = p["configurations"][sys.argv[2]]["data_identifier"]
    print(di)
except Exception:
    pass
PY
)
    fi

    if [[ -n "${resolved}" ]] && [[ -d "${NFS_PREP_DS}/${resolved}" ]]; then
        echo "${NFS_PREP_DS}/${resolved}"
        return 0
    fi

    if [[ -d "${NFS_PREP_DS}" ]]; then
        local cand
        while IFS= read -r cand; do
            if compgen -G "${cand}/*.npz" > /dev/null 2>&1; then
                echo "${cand}"
                return 0
            fi
        done < <(find "${NFS_PREP_DS}" -mindepth 1 -maxdepth 1 -type d -name "*${cfg}*" 2>/dev/null)
    fi

    echo ""
    return 1
}

# -- Preflight ---------------------------------------------------------------
[[ ! -f "${CONTAINER}"          ]] && { echo "ERROR: container not found: ${CONTAINER}" >&2; exit 1; }
[[ ! -d "${HF_EXPORT_NFS}/ct"   ]] && {
    echo "ERROR: ${HF_EXPORT_NFS}/ct missing" >&2
    echo "  Override with HF_EXPORT_DIR=/path/to/hf_export sbatch slurm/spine_prep.sh" >&2
    exit 1
}
[[ ! -f "${WANDB_TRAINER_HOST}" ]] && { echo "ERROR: ${WANDB_TRAINER_HOST} missing" >&2; exit 1; }
[[ ! -f "${PROJECT_ROOT}/tools/convert_hf_to_nnunet.py" ]] && {
    echo "ERROR: tools/convert_hf_to_nnunet.py missing" >&2; exit 1; }

[[ ! -f "${SPLITS_FILE_HOST}" ]] && {
    echo "ERROR: splits file not found at ${SPLITS_FILE_HOST}" >&2
    echo "  Searched: ${HF_EXPORT_NFS}/splits_5fold.json" >&2
    echo "            ${PROJECT_ROOT}/data/splits_5fold.json" >&2
    echo "  Override with SPLITS_FILE=/path/to/splits.json" >&2
    exit 1
}
SP_MTIME=$(stat -c '%y' "${SPLITS_FILE_HOST}" 2>/dev/null || stat -f '%Sm' "${SPLITS_FILE_HOST}")
echo "  splits_5fold.json mtime: ${SP_MTIME}"

# -- Step 0: NFS cache check (fast exit if already complete) -----------------
if [[ -f "${COMPLETE_MARKER}" ]] \
   && [[ -f "${NFS_PREP_DS}/dataset_fingerprint.json" ]] \
   && [[ -f "${NFS_PREP_DS}/${PLANS}.json" ]] \
   && [[ -f "${NFS_RAW_DS}/dataset.json" ]]; then
    PREP_DATA_DIR_CACHE=$(resolve_prep_data_dir || true)
    if [[ -n "${PREP_DATA_DIR_CACHE}" && -d "${PREP_DATA_DIR_CACHE}" ]]; then
        N_NPZ=$({ find "${PREP_DATA_DIR_CACHE}" -maxdepth 1 -name "*.npz" 2>/dev/null || true; } | wc -l)
        if [[ "${N_NPZ}" -gt 0 ]]; then
            SIZE=$(du -sh "${NFS_PREP_DS}" 2>/dev/null | awk '{print $1}')
            echo "================================================================"
            echo " NFS cache HIT: ${NFS_PREP_DS}"
            echo "   data dir : ${PREP_DATA_DIR_CACHE}"
            echo "   ${N_NPZ} .npz files, ${SIZE} total"
            echo "   plans    : ${PLANS}"
            echo "================================================================"
            [[ -f "${NFS_RAW_DS}/splits_final.json" ]] && cp -f "${NFS_RAW_DS}/splits_final.json" "${NFS_PREP_DS}/"
            [[ -f "${NFS_RAW_DS}/lstv_cases.json"   ]] && cp -f "${NFS_RAW_DS}/lstv_cases.json"   "${NFS_PREP_DS}/"
            echo " NEXT: sbatch slurm/spine_train_array.sh"
            exit 0
        fi
    fi
    echo "  cache check: marker present but data dir/.npz files missing; re-running."
fi

# >>> NO host-side dangling-symlink cleanup. <<<
# CT images are container-path symlinks (/data/hf_export/...). They look
# "broken" from the host but resolve inside the container via the bind
# mount. Label files are real (physically rewritten under merged-label
# remap) and don't need any cleanup either.

# Convert script flags (v5):
#   --no_remap is passed when running the legacy 802 baseline. Without
#   it, the convert script applies LABEL_REMAP_AT_CONVERT = {6:5, 7:6,
#   8:7, 9:8, 10:9} to every label file, producing the merged 9-class
#   contiguous scheme.
CONVERT_REMAP_FLAG=""
if [[ "${LEGACY_NO_REMAP}" == "1" ]]; then
    CONVERT_REMAP_FLAG="--no_remap"
fi

# Fused-only ablation flags (empty unless requested via env).
CONVERT_ABLATION_FLAGS=""
if [[ -n "${INCLUDE_CONFIGS}" ]]; then
    CONVERT_ABLATION_FLAGS="${CONVERT_ABLATION_FLAGS} --include_configs ${INCLUDE_CONFIGS}"
fi
if [[ "${DROP_IGNORE_LABEL}" == "1" ]]; then
    CONVERT_ABLATION_FLAGS="${CONVERT_ABLATION_FLAGS} --drop_ignore_label"
fi

echo "================================================================"
echo " Stage A: convert + preprocess (SIMPLE, NFS-only)"
echo "   Dataset      : ${DS_DIR_NAME}"
if [[ "${LEGACY_NO_REMAP}" == "1" ]]; then
    echo "   Label scheme : LEGACY 10-class (--no_remap; L5/L6 separate)"
elif [[ "${DROP_IGNORE_LABEL}" == "1" ]]; then
    echo "   Label scheme : FUSED-ONLY no-ignore 9-key (8 fg + bg; NO ignore class)"
else
    echo "   Label scheme : v20 merged 9-class (last_lumbar = former L5+L6)"
fi
[[ -n "${INCLUDE_CONFIGS}" ]] && echo "   Config filter: keep only [${INCLUDE_CONFIGS}]"
echo "   Plans        : ${PLANS} (target ${GPU_MEMORY_TARGET_GB} GB)"
echo "   HF export    : ${HF_EXPORT_NFS}"
echo "   Splits file  : ${SPLITS_FILE_HOST}"
echo "   NFS dest     : ${NFS_PREP_DS}"
echo "   Workers      : ${N_PREPROCESS_THREADS} x ${BLAS_THREADS} BLAS threads = $((N_PREPROCESS_THREADS*BLAS_THREADS)) total"
echo "   Started      : $(date)"
echo "================================================================"

# -- Singularity setup -------------------------------------------------------
export SINGULARITYENV_TMPDIR="/workspace/tmp"
export SINGULARITYENV_nnUNet_raw="/nnunet_nfs/raw"
export SINGULARITYENV_nnUNet_preprocessed="/nnunet_nfs/preprocessed"
export SINGULARITYENV_nnUNet_results="/nnunet_nfs/results"

export SINGULARITYENV_OMP_NUM_THREADS=${BLAS_THREADS}
export SINGULARITYENV_OPENBLAS_NUM_THREADS=${BLAS_THREADS}
export SINGULARITYENV_MKL_NUM_THREADS=${BLAS_THREADS}
export SINGULARITYENV_NUMEXPR_NUM_THREADS=${BLAS_THREADS}
export SINGULARITYENV_VECLIB_MAXIMUM_THREADS=${BLAS_THREADS}

SING_BINDS=(
    --bind "${PROJECT_ROOT}:/workspace"
    --bind "${HF_EXPORT_NFS}:/data/hf_export"
    --bind "${NNUNET_NFS}:/nnunet_nfs"
    --bind "${WANDB_TRAINER_HOST}:${WANDB_TRAINER_CONTAINER}"
    --bind "${SPLITS_FILE_HOST}:/workspace/splits_5fold.json:ro"
    --pwd  /workspace
)

# -- Step 1: convert HF export -> nnU-Net raw (on NFS) -----------------------
echo ""; echo "----- Step 1: convert HF export -> nnU-Net raw -----"
if [[ -f "${NFS_RAW_DS}/dataset.json" ]]; then
    echo "  dataset.json present; skipping convert."
else
    singularity exec "${SING_BINDS[@]}" "${CONTAINER}" \
        python tools/convert_hf_to_nnunet.py \
            --hf_dir     /data/hf_export \
            --splits     /workspace/splits_5fold.json \
            --nnunet_raw /nnunet_nfs/raw \
            --dataset_id   ${DATASET_ID} \
            --dataset_name ${DATASET_NAME} \
            --symlinks ${CONVERT_REMAP_FLAG} ${CONVERT_ABLATION_FLAGS}
fi

# -- Step 2: plan + preprocess (reads + writes on NFS) -----------------------
echo ""; echo "----- Step 2: plan + preprocess -----"
rm -f "${COMPLETE_MARKER}"

PREP_START=$(date +%s)
singularity exec "${SING_BINDS[@]}" "${CONTAINER}" \
    nnUNetv2_plan_and_preprocess \
        -d    ${DATASET_ID} \
        -pl   ${PLANNER} \
        -gpu_memory_target ${GPU_MEMORY_TARGET_GB} \
        -overwrite_plans_name ${PLANS} \
        -c    ${CONFIG} \
        -np   ${N_PREPROCESS_THREADS} \
        --verify_dataset_integrity
PREP_TIME=$(($(date +%s) - PREP_START))
echo "  preprocess done in ${PREP_TIME}s"

# -- Step 3: copy splits + LSTV metadata into preprocessed/ ------------------
echo ""; echo "----- Step 3: copy metadata into preprocessed/ -----"
[[ -f "${NFS_RAW_DS}/splits_final.json" ]] && cp -f "${NFS_RAW_DS}/splits_final.json" "${NFS_PREP_DS}/" && echo "  copied splits_final.json"
[[ -f "${NFS_RAW_DS}/lstv_cases.json"   ]] && cp -f "${NFS_RAW_DS}/lstv_cases.json"   "${NFS_PREP_DS}/" && echo "  copied lstv_cases.json"

# -- Sanity check ------------------------------------------------------------
PREP_DATA_DIR=$(resolve_prep_data_dir || true)
if [[ -z "${PREP_DATA_DIR}" || ! -d "${PREP_DATA_DIR}" ]]; then
    echo "ERROR: could not locate preprocessed-data dir under ${NFS_PREP_DS}" >&2
    echo "  contents of ${NFS_PREP_DS}:" >&2
    ls -la "${NFS_PREP_DS}" >&2 || true
    exit 2
fi
N_NPZ_NFS=$({ find "${PREP_DATA_DIR}" -maxdepth 1 -name "*.npz" 2>/dev/null || true; } | wc -l)
NFS_SIZE=$(du -sh "${NFS_PREP_DS}" 2>/dev/null | awk '{print $1}')
if [[ "${N_NPZ_NFS}" -eq 0 ]]; then
    echo "ERROR: zero .npz files after preprocess in ${PREP_DATA_DIR}." >&2
    exit 2
fi

# v5 dataset.json sanity check: validate the FULL label dict against
# the expected scheme for the DATASET_ID. Catches:
#   - Stale Dataset802 build still living under the 803 dir name
#   - Forgot --no_remap when rebuilding 802 baseline
#   - export_hf.py changes that changed the source label values
DS_JSON_OK=$(python - "${NFS_RAW_DS}/dataset.json" "${DATASET_ID}" "${LEGACY_NO_REMAP}" "${DROP_IGNORE_LABEL}" <<'PY' 2>/dev/null || echo "fail"
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    ds_id = int(sys.argv[2])
    legacy_no_remap = sys.argv[3] == "1"
    drop_ignore = len(sys.argv) > 4 and sys.argv[4] == "1"
    labels = d.get("labels", {})

    # Fused-only no-ignore ablation: expect the 9-key scheme with NO
    # ignore class. Validated separately because the default check below
    # requires an "ignore" key (which this scheme intentionally lacks).
    if drop_ignore:
        expected = {
            "background":  0,
            "L1":          1,
            "L2":          2,
            "L3":          3,
            "L4":          4,
            "last_lumbar": 5,
            "sacrum":      6,
            "left_hip":    7,
            "right_hip":   8,
        }
        norm = {k: int(v) for k, v in labels.items()}
        if "ignore" in norm:
            print(f"unexpected-ignore-key (no-ignore ablation): labels={norm}")
        elif norm == expected:
            print("ok (fused-only no-ignore 9-key)")
        else:
            mism = []
            for k in sorted(set(norm) | set(expected)):
                if norm.get(k) != expected.get(k):
                    mism.append(f"{k}: got={norm.get(k)} expected={expected.get(k)}")
            print(f"label-mismatch (no-ignore): {'; '.join(mism)}")
        sys.exit(0)

    if "ignore" not in labels:
        print(f"missing-ignore-key: labels={labels}")
        sys.exit(0)

    # Decide which scheme to expect
    use_merged = (ds_id == 803) or (not legacy_no_remap and ds_id != 802)

    if use_merged:
        expected = {
            "background":  0,
            "L1":          1,
            "L2":          2,
            "L3":          3,
            "L4":          4,
            "last_lumbar": 5,
            "sacrum":      6,
            "left_hip":    7,
            "right_hip":   8,
            "ignore":      9,
        }
        scheme = "v20 merged"
    else:
        expected = {
            "background":  0,
            "L1":          1,
            "L2":          2,
            "L3":          3,
            "L4":          4,
            "L5":          5,
            "L6":          6,
            "sacrum":      7,
            "left_hip":    8,
            "right_hip":   9,
            "ignore":      10,
        }
        scheme = "legacy 10-class"
    # nnU-Net v2 may serialize int values as strings; normalize.
    norm = {k: int(v) for k, v in labels.items()}
    if norm == expected:
        print(f"ok ({scheme})")
    else:
        # Be helpful: report the first mismatch
        mism = []
        for k in sorted(set(norm) | set(expected)):
            if norm.get(k) != expected.get(k):
                mism.append(f"{k}: got={norm.get(k)} expected={expected.get(k)}")
        print(f"label-mismatch ({scheme}): {'; '.join(mism)}")
except Exception as e:
    print(f"fail: {e}")
PY
)
if [[ "${DS_JSON_OK}" != ok* ]]; then
    echo "ERROR: dataset.json sanity check FAILED: ${DS_JSON_OK}" >&2
    if [[ "${DROP_IGNORE_LABEL}" == "1" ]]; then
        echo "       Expected fused-only no-ignore scheme: bg=0, L1-L4=1-4," >&2
        echo "       last_lumbar=5, sacrum=6, left_hip=7, right_hip=8 (NO ignore)." >&2
        echo "       Ensure --include_configs fused --drop_ignore_label reached" >&2
        echo "       convert_hf_to_nnunet.py (INCLUDE_CONFIGS=fused DROP_IGNORE_LABEL=1)." >&2
    elif [[ "${DATASET_ID}" == "803" ]]; then
        echo "       Expected v20 merged scheme: bg=0, L1-L4=1-4, last_lumbar=5," >&2
        echo "       sacrum=6, left_hip=7, right_hip=8, ignore=9." >&2
        echo "       Re-run convert_hf_to_nnunet.py against the patched export." >&2
    else
        echo "       Expected legacy 10-class scheme: bg=0, L1-L6=1-6, sacrum=7," >&2
        echo "       left_hip=8, right_hip=9, ignore=10." >&2
        echo "       Set LEGACY_NO_REMAP=1 if rebuilding the 802 baseline." >&2
    fi
    exit 3
fi
if [[ "${DROP_IGNORE_LABEL}" == "1" ]]; then
    echo "  dataset.json: ${DS_JSON_OK} (fused-only; ignore protocol disabled)"
else
    echo "  dataset.json: ${DS_JSON_OK} (partial-annotation training will work)"
fi

touch "${COMPLETE_MARKER}"

echo ""
echo "================================================================"
echo " Stage A COMPLETE   $(date)"
echo "   preprocess time : ${PREP_TIME}s"
echo "   data dir        : ${PREP_DATA_DIR}"
echo "   .npz files      : ${N_NPZ_NFS}"
echo "   preprocessed    : ${NFS_PREP_DS}  (${NFS_SIZE})"
echo ""
if [[ "${DROP_IGNORE_LABEL}" == "1" || -n "${INCLUDE_CONFIGS}" ]]; then
    echo " NEXT (train fold 0 of the ablation):"
    echo "   DATASET_ID=${DATASET_ID} DATASET_NAME=${DATASET_NAME} \\"
    echo "     sbatch --array=0 slurm/spine_train_array.sh"
else
    echo " NEXT:  sbatch slurm/spine_train_array.sh"
fi
echo "================================================================"
