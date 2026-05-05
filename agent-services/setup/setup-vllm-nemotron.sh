#!/usr/bin/env bash
set -euo pipefail

# vLLM venv for NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 on RTX PRO 6000 (SM120, 96GB)
# Uses pinned vLLM build from NVIDIA's quick start guide + custom super_v3 reasoning parser
#
# Reference: https://huggingface.co/unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4#quick-start-guide
# Cookbook:  https://github.com/NVIDIA-NeMo/Nemotron/blob/main/usage-cookbook/Nemotron-3-Super/vllm_cookbook.ipynb
#
# After running this, start the server with:
#   bash bin/start-nemotron-super-120b.sh

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VLLM_VENV="$PROJECT_DIR/.venvs/vllm-nemotron"
PARSER_DIR="$PROJECT_DIR/llm"

echo "=== Nemotron Super 120B vLLM Setup ==="

# Verify CUDA toolkit
if [[ ! -x /opt/cuda/bin/nvcc ]]; then
    echo "ERROR: CUDA toolkit not found at /opt/cuda"
    exit 1
fi

NVCC_VERSION=$(/opt/cuda/bin/nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
echo "System CUDA: $NVCC_VERSION"

# Handle existing venv
if [[ -d "$VLLM_VENV" ]]; then
    echo "venv already exists at $VLLM_VENV"
    read -rp "Recreate from scratch? [y/N] " yn
    if [[ "$yn" =~ ^[Yy]$ ]]; then
        echo "Removing existing venv..."
        rm -rf "$VLLM_VENV"
    else
        echo "Upgrading in-place..."
    fi
fi

if [[ ! -d "$VLLM_VENV" ]]; then
    echo "Creating venv at $VLLM_VENV (Python 3.12)..."
    uv venv "$VLLM_VENV" --python 3.12
fi

PIP="$VLLM_VENV/bin/python -m pip"

# Bootstrap pip
uv pip install pip --python "$VLLM_VENV/bin/python"

# Install PyTorch with CUDA 13.0 support (Blackwell sm120)
echo ""
echo "=== Installing PyTorch 2.10.0+cu130 ==="
$PIP install \
    torch==2.10.0+cu130 \
    torchaudio==2.10.0+cu130 \
    torchvision==0.25.0+cu130 \
    --extra-index-url https://download.pytorch.org/whl/cu130

# Install pinned vLLM build from NVIDIA's Nemotron quick start guide (cu130 variant)
echo ""
echo "=== Installing vLLM (NVIDIA pinned build, cu130) ==="
$PIP install -U vllm \
    --extra-index-url https://wheels.vllm.ai/097eb544e9a22810c9b7a59e586b61627b308362/cu130

# Install ninja (for JIT compilation)
$PIP install ninja fastsafetensors 2>/dev/null || $PIP install ninja

# Download the custom super_v3 reasoning parser
echo ""
echo "=== Downloading super_v3 reasoning parser ==="
PARSER_PATH="$PARSER_DIR/super_v3_reasoning_parser.py"
if [[ ! -f "$PARSER_PATH" ]]; then
    "$VLLM_VENV/bin/python" -c "
from huggingface_hub import hf_hub_download
import shutil
path = hf_hub_download(
    repo_id='nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4',
    filename='super_v3_reasoning_parser.py'
)
shutil.copy(path, '$PARSER_PATH')
print(f'Downloaded to $PARSER_PATH')
"
else
    echo "Parser already exists at $PARSER_PATH"
fi

# Verify installation
echo ""
echo "=== Verifying Installation ==="
"$VLLM_VENV/bin/python" -c "
import torch
print(f'PyTorch:      {torch.__version__}')
print(f'CUDA runtime: {torch.version.cuda}')
arch = torch.cuda.get_arch_list()
print(f'Arch list:    {arch}')
if 'sm_120' in arch:
    print('Blackwell sm120 support: YES')
else:
    print('WARNING: sm120 not in arch list')
import vllm
print(f'vLLM:         {vllm.__version__}')
# Check for the op that was missing before
has_op = hasattr(torch.ops._C, 'per_token_group_fp8_quant')
print(f'per_token_group_fp8_quant op: {\"YES\" if has_op else \"NO (may still work via vllm _C)\"}')
"

echo ""
echo "=== Setup Complete ==="
echo "venv: $VLLM_VENV"
echo "Reasoning parser: $PARSER_PATH"
echo ""
echo "To start Nemotron Super:"
echo "  bash bin/start-nemotron-super-120b.sh"
