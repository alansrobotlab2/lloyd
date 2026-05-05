#!/usr/bin/env bash
set -euo pipefail

# One-time setup for vLLM venv with CUDA 13.0 (Blackwell sm120 support)
# Used by Qwen3.5-122B-A10B NVFP4 on GPU 1 (RTX PRO 6000 Blackwell)
#
# Requires: CUDA toolkit at /opt/cuda (>= 13.0), uv
#
# After running this, start the server with:
#   supervisorctl -c supervisor/supervisord.conf start agent-llm-122b

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VLLM_VENV="$PROJECT_DIR/.venvs/vllm"
MODEL_DIR="$PROJECT_DIR/llm/models/Sehyo-Qwen3.5-122B-A10B-NVFP4"

echo "=== vLLM Setup (CUDA 13.0 + Blackwell) ==="

# Verify CUDA toolkit
if [[ ! -x /opt/cuda/bin/nvcc ]]; then
    echo "ERROR: CUDA toolkit not found at /opt/cuda"
    echo "  Install: sudo pacman -S cuda"
    exit 1
fi

NVCC_VERSION=$(/opt/cuda/bin/nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
echo "System CUDA: $NVCC_VERSION"

# Create venv
if [[ -d "$VLLM_VENV" ]]; then
    echo "vLLM venv already exists at $VLLM_VENV"
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

# Install PyTorch with CUDA 13.0 support
# cu130 is required for Blackwell (sm120) — enables compute_120f arch family target
# which unlocks fast TMA WS grouped GEMM CUTLASS tactics (vs broken 120a fallbacks)
echo "Installing PyTorch 2.10.0+cu130..."
$PIP install \
    torch==2.10.0+cu130 \
    torchaudio==2.10.0+cu130 \
    torchvision==0.25.0+cu130 \
    --extra-index-url https://download.pytorch.org/whl/cu130

# Install matching cuda-python
echo "Installing cuda-python 13.0..."
$PIP install cuda-python==13.0.3

# Install vLLM
echo "Installing vLLM 0.17.1..."
$PIP install vllm==0.17.1

# Upgrade FlashInfer to 0.6.6 (fixes GDC barriers, better sm120 CUTLASS support)
# vLLM 0.17.1 pins 0.6.4 but 0.6.6 is API-compatible
echo "Installing FlashInfer 0.6.6..."
$PIP install flashinfer-python==0.6.6

# Install ninja (required for FlashInfer JIT compilation of CUTLASS kernels)
echo "Installing ninja..."
$PIP install ninja

# Verify installation
echo ""
echo "=== Verifying Installation ==="
"$VLLM_VENV/bin/python" -c "
import torch
print(f'PyTorch:      {torch.__version__}')
print(f'CUDA runtime: {torch.version.cuda}')
print(f'CUDA compiled:{torch._C._cuda_getCompiledVersion()}')
print(f'Arch list:    {torch.cuda.get_arch_list()}')
import vllm
print(f'vLLM:         {vllm.__version__}')
import flashinfer
print(f'FlashInfer:   {flashinfer.__version__}')
# Verify sm120 is in arch list
arch = torch.cuda.get_arch_list()
if 'sm_120' in arch:
    print('Blackwell sm120 support: YES')
else:
    print('WARNING: sm120 not in arch list — CUTLASS kernels may not work correctly')
print()
print('IMPORTANT: start-llm-122b.sh sets FLASHINFER_CUDA_ARCH_LIST=12.0f')
print('This forces FlashInfer to JIT-compile with compute_120f (arch family target)')
print('which enables fast TMA WS CUTLASS tactics on Blackwell.')
"

# Clear FlashInfer JIT cache (force recompile with new CUDA)
FLASHINFER_CACHE="$HOME/.cache/flashinfer"
if [[ -d "$FLASHINFER_CACHE" ]]; then
    echo ""
    echo "Clearing FlashInfer JIT cache at $FLASHINFER_CACHE..."
    rm -rf "$FLASHINFER_CACHE"
    echo "Cache cleared. Kernels will be recompiled on first run."
fi

# Clear vLLM torch compile cache
VLLM_CACHE="$HOME/.cache/vllm/torch_compile_cache"
if [[ -d "$VLLM_CACHE" ]]; then
    echo "Clearing vLLM torch compile cache..."
    rm -rf "$VLLM_CACHE"
fi

# Check for model
echo ""
if [[ -f "$MODEL_DIR/config.json" ]]; then
    echo "Model found: $MODEL_DIR"
else
    echo "WARNING: Model not found at $MODEL_DIR"
    echo "  Download the Sehyo-Qwen3.5-122B-A10B-NVFP4 model before starting the server."
fi

echo ""
echo "=== Setup Complete ==="
echo "vLLM venv: $VLLM_VENV"
echo ""
echo "To start the server:"
echo "  supervisorctl -c supervisor/supervisord.conf start agent-llm-122b"
echo ""
echo "NOTE: First startup will be slow — FlashInfer JIT-compiles CUTLASS"
echo "kernels for your GPU. Subsequent starts use the cached kernels."
