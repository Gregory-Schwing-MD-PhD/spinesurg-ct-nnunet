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
# LABEL files (May 2026 v5+): the HF export is now VerSe-native, so label
# NIfTIs are ALWAYS PHYSICALLY REWRITTEN by the convert script via a
# vectorized LUT that remaps VerSe ids -> the training scheme. Under
# Dataset803 (merged) it maps VerSe L1-L4 (20-23) -> 1-4, L5+L6 (24/25)
# -> last_lumbar (5), sacrum (26) -> 6, hips (30/31) -> 7/8, ignore (255)
# -> 9, and drops EVERY non-training VerSe id (thoracic/cervical, coccyx,
# T13, S1, femurs, ribs, soft-tissue) to background. Symlinking labels
# would be wrong — the source carries dozens of non-training VerSe ids.
# The convert script handles this: --symlinks applies to CT images only.
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
# The dataset (CT, labels, manifests, splits) lives in the SEPARATE
# CTSpinoPelvic1K repo — see .gitignore ("Dataset (separate repo; do NOT
# commit)"). That copy is CANONICAL. A project-local data/hf_export under
# spinesurg-ct-nnunet is a shadow: used only if the canonical is absent, and
# warned about loudly so a stale local copy can never silently win.
# Override either with HF_EXPORT_DIR=/path (export) or SPLITS_FILE=/path.
CTSPINO_ROOT="${CTSPINO_ROOT:-${HOME}/CTSpinoPelvic1K}"
CTSPINO_HF_EXPORT="${CTSPINO_ROOT}/data/hf_export"
LOCAL_HF_EXPORT="${PROJECT_ROOT}/data/hf_export"

if [[ -z "${HF_EXPORT_DIR:-}" ]]; then
    if [[ -d "${CTSPINO_HF_EXPORT}/ct" ]]; then
        HF_EXPORT_DIR="${CTSPINO_HF_EXPORT}"
        if [[ -d "${LOCAL_HF_EXPORT}/ct" ]]; then
            echo "WARN: a project-local hf_export shadow exists at" >&2
            echo "        ${LOCAL_HF_EXPORT}" >&2
            echo "      but the CANONICAL CTSpinoPelvic1K export is being used:" >&2
            echo "        ${CTSPINO_HF_EXPORT}" >&2
            echo "      To deliberately use the local one: HF_EXPORT_DIR=${LOCAL_HF_EXPORT}" >&2
        fi
    elif [[ -d "${LOCAL_HF_EXPORT}/ct" ]]; then
        HF_EXPORT_DIR="${LOCAL_HF_EXPORT}"
        echo "WARN: canonical CTSpinoPelvic1K export not found at" >&2
        echo "        ${CTSPINO_HF_EXPORT}" >&2
        echo "      falling back to project-local shadow ${LOCAL_HF_EXPORT}." >&2
    fi
fi
HF_EXPORT_DIR="${HF_EXPORT_DIR:-${CTSPINO_HF_EXPORT}}"
HF_EXPORT_NFS="${HF_EXPORT_DIR}"

# -- Splits file discovery ---------------------------------------------------
# Splits MUST come from the SAME export that provides the CT/labels/manifests
# (the splitter derived them from that manifest), so the default tracks
# HF_EXPORT_NFS. A project-local data/splits_5fold.json is a shadow, used
# only if the export's own splits is missing — and warned about.
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

if [[ "${SPLITS_FILE_HOST}" == "${PROJECT_ROOT}/data/splits_5fold.json" \
      && -f "${CTSPINO_HF_EXPORT}/splits_5fold.json" ]]; then
    echo "WARN: using project-local splits shadow" >&2
    echo "        ${SPLITS_FILE_HOST}" >&2
    echo "      while canonical ${CTSPINO_HF_EXPORT}/splits_5fold.json exists." >&2
    echo "      Pin SPLITS_FILE=... to be explicit." >&2
fi

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
echo "  splits source : ${SPLITS_FILE_HOST}"
echo "  splits mtime  : ${SP_MTIME}"

# Schema guard: the 6-way LSTV taxonomy needs schema v6+. This also catches
# stray pre-schema junk files (e.g. an old splits_5fold.json with no
# schema_version) being picked up by accident.
SP_SCHEMA=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('schema_version',0))" "${SPLITS_FILE_HOST}" 2>/dev/null || echo 0)
echo "  splits schema : v${SP_SCHEMA}"
if [[ "${SP_SCHEMA}" -lt 6 ]]; then
    echo "ERROR: splits file ${SPLITS_FILE_HOST} is schema v${SP_SCHEMA} (< 6)." >&2
    echo "       The 6-way LSTV subtype taxonomy requires v6+. Regenerate with" >&2
    echo "       scripts/generate_5fold_splits.py against the current HF export," >&2
    echo "       or point SPLITS_FILE at the canonical CTSpinoPelvic1K v6 file." >&2
    exit 1
fi

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
#   --no_remap selects the UNMERGED remap: VerSe-native -> unmerged
#   10-class (Dataset802), keeping L5 and L6 distinct. Without it, the
#   convert script applies the MERGED remap: VerSe-native -> merged
#   9-class (Dataset803), collapsing VerSe L5+L6 into last_lumbar. Both
#   remaps drop every non-training VerSe id to background.
CONVERT_REMAP_FLAG=""
if [[ "${LEGACY_NO_REMAP}" == "1" ]]; then
    CONVERT_REMAP_FLAG="--no_remap"
fi

# ---------------------------------------------------------------------------
# LSTV BENCHMARK SCHEMES (2026-08). SCHEME= selects one of three targets built
# for the transitional-anatomy comparison. They are mutually exclusive with
# each other and with LEGACY_NO_REMAP / DROP_IGNORE_LABEL, and the convert
# script rejects the combination rather than silently preferring one.
#
#   SCHEME=oneshot      21 classes: T10-T12 and L1-L6 distinct, ribs 1-11
#                       pooled, the TWELFTH rib kept apart from the lumbar
#                       rib. This is the ONLY target on which the
#                       hypoplastic-twelfth versus lumbar-rib question is
#                       measurable -- the older schemes have neither thoracic
#                       nor rib classes, so a model can be neither right nor
#                       wrong about it.
#   SCHEME=rib_regions  count-free, with the vertebra class split into
#                       rib-bearing and not, as nested nnU-Net REGIONS. Lets
#                       the network be certain a thing is a vertebra while
#                       uncertain whether it bears a rib.
#   SCHEME=countfree    no vertebral identity at all; counting happens
#                       downstream in code. The comparator arm.
#
# Suggested dataset ids, kept apart so the three can coexist:
#   oneshot 810, rib_regions 811, countfree 812
SCHEME="${SCHEME:-}"
CONVERT_SCHEME_FLAG=""
case "${SCHEME}" in
    "")            ;;
    oneshot)       CONVERT_SCHEME_FLAG="--oneshot" ;;
    rib_regions)   CONVERT_SCHEME_FLAG="--rib_regions" ;;
    countfree)     CONVERT_SCHEME_FLAG="--countfree" ;;
    *) echo "ERROR: SCHEME must be oneshot, rib_regions or countfree (got '${SCHEME}')" >&2
       exit 1 ;;
esac
if [[ -n "${CONVERT_SCHEME_FLAG}" ]]; then
    if [[ "${LEGACY_NO_REMAP}" == "1" || "${DROP_IGNORE_LABEL}" == "1" ]]; then
        echo "ERROR: SCHEME=${SCHEME} defines its own label scheme; it cannot be combined" >&2
        echo "       with LEGACY_NO_REMAP or DROP_IGNORE_LABEL." >&2
        exit 1
    fi
    # A benchmark whose arms share a dataset id will overwrite each other's
    # preprocessed data, and the failure looks like an unexplained accuracy change
    # rather than an error. Refuse the default id outright.
    if [[ "${DATASET_ID}" == "803" ]]; then
        echo "ERROR: SCHEME=${SCHEME} was submitted with the DEFAULT DATASET_ID=803." >&2
        echo "       Give each arm its own id or they will overwrite one another:" >&2
        echo "         oneshot 810, rib_regions 811, countfree 812" >&2
        exit 1
    fi
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
if [[ -n "${SCHEME}" ]]; then
    echo "   LSTV SCHEME  : ${SCHEME} (${CONVERT_SCHEME_FLAG})"
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

# -- Step 0: preflight -------------------------------------------------------
# Every check here is one that fails SILENTLY otherwise: a misplaced ignore
# label, non-contiguous class values, a regions_class_order that paints the
# encompassing region over the one nested inside it, or a validation fold
# holding none of the rare class the benchmark exists to measure. nnU-Net
# trains happily through all of those and reports a number that looks fine.
#
# It runs only for the LSTV schemes, because only they introduce regions and
# the new class layouts. Label-volume checks are skipped when the directory is
# not visible from inside the container; the scheme checks need no data and are
# the ones that matter most.
if [[ -n "${CONVERT_SCHEME_FLAG}" ]]; then
    echo ""; echo "----- Step 0: preflight -----"
    singularity exec "${SING_BINDS[@]}" "${CONTAINER}" \
        python tools/preflight_lstv.py \
            --labels "${PREFLIGHT_LABELS:-}" \
            --splits /workspace/splits_5fold.json \
            --lstv-cases "${PREFLIGHT_LSTV_CASES:-}" || {
        echo "PREFLIGHT FAILED -- nothing downstream is worth running." >&2
        exit 1; }
fi

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
            --symlinks ${CONVERT_REMAP_FLAG} ${CONVERT_SCHEME_FLAG} ${CONVERT_ABLATION_FLAGS}
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
