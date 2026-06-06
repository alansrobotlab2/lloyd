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
VLLM_VENV="$HOME/lloyd/.venvs/vllm-experimental"
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
# IMPORTANT: --pre is REQUIRED, but is NO LONGER SUFFICIENT on its own.
# The cu130 wheel index only ships dev nightlies (e.g.
# 0.22.1rc1.dev221+gb593396c7). PyPI now also ships FINAL releases (0.22.0,
# 0.22.1, ...) built against CUDA 12. An unpinned `pip install vllm --pre`
# with PyPI as an extra index resolves the PyPI cu12 final wheel — a final
# release outranks a .devN pre-release of the same version — and that wheel
# crashes at import with "libcudart.so.12: cannot open shared object file".
#
# Fix: read the single version the cu130 nightly index actually publishes and
# pin to it, so PyPI can only ever supply transitive deps, never vllm itself.
echo ""
echo "=== Resolving latest vLLM cu130 nightly ==="
VLLM_NIGHTLY=$("$VLLM_VENV/bin/python" -m pip index versions vllm --pre \
    --index-url https://wheels.vllm.ai/nightly/cu130 2>/dev/null \
    | grep -oP '^vllm \(\K[^)]+')
if [[ -z "$VLLM_NIGHTLY" ]]; then
    echo "ERROR: could not resolve latest cu130 nightly version from wheels.vllm.ai"
    exit 1
fi
echo "Latest cu130 nightly: $VLLM_NIGHTLY"
echo ""
echo "=== Installing vLLM nightly (cu130) — pinned to $VLLM_NIGHTLY ==="
$PIP install "vllm==$VLLM_NIGHTLY" --pre \
    --index-url https://wheels.vllm.ai/nightly/cu130 \
    --extra-index-url https://pypi.org/simple

# FlashInfer is now a HARD dependency of the cu130 nightly and is resolved to
# the correct version automatically (0.6.12 as of 2026-06-05, which ships the
# SM120 NVFP4 MoE CuTe-DSL kernels). Do NOT pin an older version here: the
# previous explicit `flashinfer-python==0.6.9` step ran AFTER the vllm install
# and would now DOWNGRADE the 0.6.12 that vLLM just pulled in. Just verify it
# imports; only (re)install if somehow absent.
echo ""
echo "=== Verifying FlashInfer (pulled in by vLLM) ==="
if "$VLLM_VENV/bin/python" -c "import flashinfer; print('FlashInfer', flashinfer.__version__)"; then
    :
else
    echo "FlashInfer missing — installing latest"
    $PIP install flashinfer-python flashinfer-cubin
fi

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
