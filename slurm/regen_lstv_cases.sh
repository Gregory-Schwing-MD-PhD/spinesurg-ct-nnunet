#!/usr/bin/env bash
# regen_lstv_cases.sh — rewrite splits_final.json + lstv_cases.json for an existing
# nnU-Net raw dataset with the CURRENT converter, and copy them into preprocessed/.
# JSON-only, seconds of work, so it runs on the login node.
#
#   HF_EXPORT_DIR=~/data/CTSpinoPelvic1K SCHEME=oneshot DATASET_ID=810 \
#   DATASET_NAME=SpineSurgLSTVOneShot bash slurm/regen_lstv_cases.sh
set -euo pipefail
cd "$(dirname "$0")/.."
HF_EXPORT_DIR="${HF_EXPORT_DIR:-$HOME/data/CTSpinoPelvic1K}"
SPLITS_FILE="${SPLITS_FILE:-$HF_EXPORT_DIR/splits_5fold.json}"
DATASET_ID="${DATASET_ID:-810}"
DATASET_NAME="${DATASET_NAME:-SpineSurgLSTVOneShot}"
case "${SCHEME:-oneshot}" in
  oneshot) FLAG=--oneshot ;; rib_regions) FLAG=--rib_regions ;; countfree) FLAG=--countfree ;;
  *) echo "SCHEME must be oneshot, rib_regions or countfree"; exit 2 ;;
esac
DS="Dataset$(printf '%03d' "$DATASET_ID")_${DATASET_NAME}"
module load singularity 2>/dev/null || true
singularity exec --bind "$PWD:/workspace" --bind "$HF_EXPORT_DIR:/data/hf_export" \
    --bind "$PWD/nnunet:/nnunet_nfs" --bind "$SPLITS_FILE:/workspace/splits_5fold.json" \
    --pwd /workspace containers/spinesurg-ct.sif \
    python tools/convert_hf_to_nnunet.py --hf_dir /data/hf_export --splits /workspace/splits_5fold.json \
        --nnunet_raw /nnunet_nfs/raw --dataset_id "$DATASET_ID" --dataset_name "$DATASET_NAME" \
        --symlinks $FLAG --regen_splits_only
for f in splits_final.json lstv_cases.json; do
  [ -f "nnunet/raw/$DS/$f" ] && [ -d "nnunet/preprocessed/$DS" ] && cp -f "nnunet/raw/$DS/$f" "nnunet/preprocessed/$DS/" && echo "copied $f"
done
python3 - "nnunet/raw/$DS/lstv_cases.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
attrs = d.get("case_to_attrs", {})
print("cases:", len(attrs), " with lumbar rib:", sum(1 for a in attrs.values() if a.get("has_lumbar_rib")),
      " subtypes:", {k: v for k, v in sorted(__import__('collections').Counter(d.get('case_to_subtype', {}).values()).items())})
PY
