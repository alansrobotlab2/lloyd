#!/usr/bin/env bash
set -euo pipefail

# Experimental vLLM venv for MTP 3 testing on RTX PRO 6000 (SM120, 96GB)
# Based on SPARK repo (github.com/bjk110/SPARK_Qwen3.5-122B-A10B-NVFP4)
# adapted for SM120 instead of SM121
#
# Key differences from production setup-vllm.sh:
#   - vLLM nightly (latest cu130) instead of 0.17.1 stable
#   - Applies aot_cache_fix + fastsafetensors patches from SPARK
#   - Clears torch compile + flashinfer caches for clean start
#
# After running this, start the server with:
#   supervisorctl -c supervisor/supervisord.conf start agent-llm-122b-experimental

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VLLM_VENV="$PROJECT_DIR/.venvs/vllm-experimental"
PATCH_DIR="$PROJECT_DIR/setup/patches"

echo "=== Experimental vLLM Setup (Nightly + SPARK Patches) ==="

# Verify CUDA toolkit
if [[ ! -x /opt/cuda/bin/nvcc ]]; then
    echo "ERROR: CUDA toolkit not found at /opt/cuda"
    exit 1
fi

NVCC_VERSION=$(/opt/cuda/bin/nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
echo "System CUDA: $NVCC_VERSION"

# Handle existing venv
if [[ -L "$VLLM_VENV" ]]; then
    echo "Removing existing symlink at $VLLM_VENV"
    rm "$VLLM_VENV"
elif [[ -d "$VLLM_VENV" ]]; then
    echo "Experimental venv already exists at $VLLM_VENV"
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

UV="uv"
PIP="$VLLM_VENV/bin/python -m pip"

# Bootstrap pip in the venv
$UV pip install pip --python "$VLLM_VENV/bin/python"

# Install PyTorch with CUDA 13.0 support (Blackwell sm120)
echo ""
echo "=== Installing PyTorch 2.11.0+cu130 ==="
$PIP install \
    torch==2.11.0+cu130 \
    torchaudio==2.11.0+cu130 \
    torchvision==0.26.0+cu130 \
    --extra-index-url https://download.pytorch.org/whl/cu130

# Install cuda-python
echo ""
echo "=== Installing cuda-python 13.0 ==="
$PIP install cuda-python==13.0.3

# Install vLLM nightly (cu130)
# Using nightly for latest MTP/spec-decode fixes
#
# IMPORTANT: --pre is REQUIRED. The cu130 wheel index only ships dev nightlies
# (e.g. 0.19.1rc1.dev86+g70406eb1d.cu130). Without --pre, pip falls back to the
# stable PyPI vllm wheel, which is built against CUDA 12 and crashes at import
# with "libcudart.so.12: cannot open shared object file".
echo ""
echo "=== Installing vLLM nightly (cu130) ==="
$PIP install vllm --pre \
    --index-url https://wheels.vllm.ai/nightly/cu130 \
    --extra-index-url https://pypi.org/simple

# Install FlashInfer 0.6.9 (ships PR #3080 — B12x CuTe-DSL micro-kernel for
# SM120 NVFP4 MoE; previously the missing piece for Qwen3.6-35B-A3B perf).
# vLLM nightly metadata pins to ==0.6.8.post1 but 0.6.9 is API-compatible
# in practice; ignore the pip dependency-resolver warning.
echo ""
echo "=== Installing FlashInfer 0.6.9 ==="
$PIP install flashinfer-python==0.6.9 flashinfer-cubin==0.6.9 \
    || $PIP install flashinfer-python flashinfer-cubin

# Install ninja (required for FlashInfer JIT compilation)
$PIP install ninja

# Install fastsafetensors if available
$PIP install fastsafetensors 2>/dev/null || echo "fastsafetensors not available, skipping"

# Apply patches
#
# Status as of vLLM 0.20.2rc1.dev6 (2026-05-03):
#   - fastsafetensors_natural_sort.patch:  UPSTREAMED. _natural_sort_key now
#     ships in vllm/model_executor/model_loader/weight_utils.py. No-op.
#   - aot_cache_fix.patch:  OBSOLETED by caching.py refactor in 0.20.x.
#     The original GraphPickler "Unexpected raw Node" error has not been
#     observed since the refactor + torch 2.11. Kept on disk for reference.
#     If MTP + torch.compile starts failing again, port the
#     _strip_unpicklable_node_meta logic forward into the new
#     serialize_compile_artifacts path.
# Both patches are dry-run-checked and only applied if they still match cleanly.
echo ""
echo "=== Checking SPARK patches ==="

VLLM_SITE=$("$VLLM_VENV/bin/python" -c "import vllm; import os; print(os.path.dirname(vllm.__file__))")
echo "vLLM package at: $VLLM_SITE"

for patch_file in "$PATCH_DIR/aot_cache_fix.patch" "$PATCH_DIR/fastsafetensors_natural_sort.patch"; do
    [[ -f "$patch_file" ]] || continue
    echo "Checking $(basename "$patch_file")..."
    cd "$VLLM_SITE/.."
    if patch -p1 --forward --dry-run < "$patch_file" >/dev/null 2>&1; then
        patch -p1 --forward < "$patch_file"
        echo "  Applied."
    else
        echo "  Skipped (upstreamed, obsolete, or already applied)."
    fi
    cd "$PROJECT_DIR"
done

# Verify installation
echo ""
echo "=== Verifying Installation ==="
"$VLLM_VENV/bin/python" -c "
import torch
print(f'PyTorch:      {torch.__version__}')
print(f'CUDA runtime: {torch.version.cuda}')
print(f'Arch list:    {torch.cuda.get_arch_list()}')
import vllm
print(f'vLLM:         {vllm.__version__}')
import flashinfer
print(f'FlashInfer:   {flashinfer.__version__}')
arch = torch.cuda.get_arch_list()
if 'sm_120' in arch:
    print('Blackwell sm120 support: YES')
else:
    print('WARNING: sm120 not in arch list')
"

# Clear caches for clean start
echo ""
echo "=== Clearing caches ==="

FLASHINFER_CACHE="$HOME/.cache/flashinfer"
VLLM_CACHE="$HOME/.cache/vllm/torch_compile_cache"

# Only clear experimental-specific caches if they exist
# Don't nuke the production caches
echo "Note: torch compile cache is shared. If the nightly version"
echo "is incompatible with cached graphs, clear manually with:"
echo "  rm -rf $VLLM_CACHE"
echo ""
echo "FlashInfer JIT cache (also shared):"
echo "  rm -rf $FLASHINFER_CACHE"

echo ""
echo "=== Setup Complete ==="
echo "Experimental venv: $VLLM_VENV"
VLLM_VER=$("$VLLM_VENV/bin/python" -c "import vllm; print(vllm.__version__)")
echo "vLLM version: $VLLM_VER"
echo ""
echo "To start the experimental server:"
echo "  supervisorctl -c supervisor/supervisord.conf stop agent-llm-122b"
echo "  supervisorctl -c supervisor/supervisord.conf start agent-llm-122b-experimental"
echo ""
echo "To revert to production:"
echo "  supervisorctl -c supervisor/supervisord.conf stop agent-llm-122b-experimental"
echo "  supervisorctl -c supervisor/supervisord.conf start agent-llm-122b"
