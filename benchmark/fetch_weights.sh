#!/bin/bash
# Runs on the grid LOGIN node (internet). Pretrained weights for the competitors, into $HOME,
# where the compute nodes (no internet) can read them.
#
#   nohup bash benchmark/fetch_weights.sh > logs/fetch_weights.log 2>&1 &
set -uo pipefail
cd "${HOME}"

# ---- SPINEPS CT models (semantic + instance from v2.0.0, VERIDAH CT labeling from v1.4.0) ----
# spineps looks in $SPINEPS_SEGMENTOR_MODELS for one folder per model.
M="${HOME}/spineps_models"; mkdir -p "${M}"
for u in \
  https://github.com/Hendrik-code/spineps/releases/download/v2.0.0/ct.zip \
  https://github.com/Hendrik-code/spineps/releases/download/v2.0.0/CT_instance.zip \
  https://github.com/Hendrik-code/spineps/releases/download/v1.4.0/ct_labeling.zip ; do
    z="${M}/$(basename "${u}")"
    if [[ ! -f "${z}" ]]; then
        echo "=== ${u} $(date)"; wget -q -O "${z}" "${u}" || { echo "download failed ${u}"; rm -f "${z}"; continue; }
    fi
    d="${M}/$(basename "${z}" .zip)"
    [[ -d "${d}" ]] || { mkdir -p "${d}"; unzip -q -o "${z}" -d "${d}"; }
done
find "${M}" -maxdepth 3 -type d | head -30
du -sh "${M}"

# ---- NV-Segment-CT (VISTA3D lineage) checkpoint from Hugging Face --------------------------
source "${HOME}/mambaforge/etc/profile.d/conda.sh"
if conda env list | grep -qE '^vista3d '; then
    conda activate vista3d
    pip install --quiet "huggingface_hub[cli]"
    export HF_HOME="${HOME}/.cache/huggingface"
    python - <<'EOF'
from huggingface_hub import snapshot_download
for repo in ("nvidia/NV-Segment-CT",):
    try:
        p = snapshot_download(repo_id=repo)
        print("downloaded", repo, "->", p)
    except Exception as e:
        print("HF download failed for", repo, ":", repr(e)[:300])
EOF
    conda deactivate
else
    echo "vista3d env not there yet; run install_envs.sh first, then re-run for the NV-Segment-CT checkpoint"
fi
echo "=== done $(date)"
