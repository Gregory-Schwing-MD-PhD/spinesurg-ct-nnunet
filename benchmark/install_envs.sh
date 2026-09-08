#!/bin/bash
# Runs on the grid LOGIN node (the only place with internet): conda envs for the competitors
# that have no container here. Weights are fetched separately (benchmark/fetch_weights.sh)
# into $HOME so the compute nodes, which have no internet, find them.
#
#   nohup bash benchmark/install_envs.sh > logs/install_envs.log 2>&1 &
set -uo pipefail
source "${HOME}/mambaforge/etc/profile.d/conda.sh"
cd "${HOME}"

if ! conda env list | grep -qE '^spineps '; then
    echo "=== spineps env $(date)"
    mamba create -y -n spineps python=3.11 >/dev/null
    conda activate spineps
    pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu124
    pip install --quiet spineps
    python -c "import spineps, torch; print('spineps', getattr(spineps, '__version__', '?'), 'torch', torch.__version__, torch.version.cuda)"
    conda deactivate
else
    echo "spineps env exists"
fi

if ! conda env list | grep -qE '^vista3d '; then
    echo "=== vista3d env $(date)"
    mamba create -y -n vista3d python=3.11 >/dev/null
    conda activate vista3d
    [[ -d "${HOME}/NV-Segment-CTMR" ]] || git clone --depth 1 https://github.com/NVIDIA-Medtech/NV-Segment-CTMR.git "${HOME}/NV-Segment-CTMR"
    pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu124
    pip install --quiet -r "${HOME}/NV-Segment-CTMR/NV-Segment-CTMR/requirements.txt" 2>/dev/null || pip install --quiet -r "${HOME}/NV-Segment-CTMR/requirements.txt"
    python -c "import monai, torch; print('monai', monai.__version__, 'torch', torch.__version__, torch.version.cuda)"
    conda deactivate
else
    echo "vista3d env exists"
fi

if [[ ! -d "${HOME}/CTSpinoPelvic1K/third_party/RibSeg" ]]; then
    echo "=== RibSeg clone $(date)"
    mkdir -p "${HOME}/CTSpinoPelvic1K/third_party"
    git clone --depth 1 https://github.com/HINTLab/RibSeg.git "${HOME}/CTSpinoPelvic1K/third_party/RibSeg"
fi
echo "=== done $(date)"
