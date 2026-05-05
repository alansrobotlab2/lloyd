#!/usr/bin/env bash
set -euo pipefail

# vLLM 0.19 + TurboQuant venv
#
# Uses stable vLLM 0.19.0 from PyPI (first release with NVFP4 CUTLASS MoE
# support and Blackwell NaN fixes) plus the turboquant-vllm PyPI package
# for KV cache quantization.
#
# After running this, start the server with:
#   ./scripts/run-vllm-19-turboquant.sh

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VLLM_VENV="$PROJECT_DIR/.venvs/vllm-19-turboquant"
PATCH_DIR="$PROJECT_DIR/setup/patches"

echo "=== vLLM 0.19 + TurboQuant Setup ==="

# Verify CUDA toolkit
if [[ ! -x /opt/cuda/bin/nvcc ]]; then
    echo "ERROR: CUDA toolkit not found at /opt/cuda"
    exit 1
fi

NVCC_VERSION=$(/opt/cuda/bin/nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
echo "System CUDA: $NVCC_VERSION"

# Handle existing venv
if [[ -d "$VLLM_VENV" ]]; then
    echo "Venv already exists at $VLLM_VENV"
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

# PyTorch 2.10.0+cu130 (Blackwell sm120 support)
echo ""
echo "=== Installing PyTorch 2.10.0+cu130 ==="
$PIP install \
    torch==2.10.0+cu130 \
    torchaudio==2.10.0+cu130 \
    torchvision==0.25.0+cu130 \
    --extra-index-url https://download.pytorch.org/whl/cu130

# cuda-python
echo ""
echo "=== Installing cuda-python 13.0 ==="
$PIP install cuda-python==13.0.3

# vLLM 0.19.0 stable
# flashinfer-cubin is a new explicit dep in 0.19 (precompiled CUDA kernels)
echo ""
echo "=== Installing vLLM 0.19.0 ==="
$PIP install "vllm==0.19.0" \
    "flashinfer-python==0.6.6" \
    "flashinfer-cubin==0.6.6" \
    --extra-index-url https://pypi.org/simple

# turboquant-vllm — KV cache quantization (now compatible: requires vllm>=0.19)
echo ""
echo "=== Installing turboquant-vllm ==="
$PIP install turboquant-vllm

# Extras
$PIP install ninja fastsafetensors 2>/dev/null || $PIP install ninja

# Apply patches (dry-run first — they may not apply cleanly to 0.19)
echo ""
echo "=== Applying patches ==="

VLLM_SITE=$("$VLLM_VENV/bin/python" -c "import vllm; import os; print(os.path.dirname(vllm.__file__))")
echo "vLLM package at: $VLLM_SITE"

for PATCH in aot_cache_fix fastsafetensors_natural_sort; do
    PFILE="$PATCH_DIR/${PATCH}.patch"
    if [[ -f "$PFILE" ]]; then
        echo "Applying ${PATCH}.patch..."
        cd "$VLLM_SITE/.."
        if patch -p1 --forward --dry-run < "$PFILE" 2>/dev/null; then
            patch -p1 --forward < "$PFILE"
            echo "  Applied."
        else
            echo "  Skipped (already applied or doesn't match 0.19 — OK)."
        fi
        cd "$PROJECT_DIR"
    fi
done

# Verify
echo ""
echo "=== Verifying Installation ==="
"$VLLM_VENV/bin/python" -c "
import torch
print(f'PyTorch:        {torch.__version__}')
print(f'CUDA runtime:   {torch.version.cuda}')
arch = torch.cuda.get_arch_list()
print(f'Arch list:      {arch}')
print(f'sm_120 support: {\"YES\" if \"sm_120\" in arch else \"NO\"}')
import vllm
print(f'vLLM:           {vllm.__version__}')
import flashinfer
print(f'FlashInfer:     {flashinfer.__version__}')
import turboquant_vllm
print(f'turboquant-vllm:{turboquant_vllm.__version__}')
"

echo ""
echo "=== Setup Complete ==="
echo "Venv: $VLLM_VENV"
echo ""
echo "To start: ./scripts/run-vllm-19-turboquant.sh"
