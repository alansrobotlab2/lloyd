#!/usr/bin/env bash
set -euo pipefail

# One-time setup for llama.cpp + the secondary slot's model.
# Clones and builds llama.cpp with CUDA support, then downloads the GGUF.
#
# This is what backs `agent-services/bin/start-secondary.sh MODEL=qwen36`,
# the :8091 secondary. Qwen3.6-35B-A3B is served by llama.cpp rather than
# vLLM because a GGUF Q3 is the only build of it that fits a 24 GB 3090 at
# the full 262144 window: unsloth's NVFP4 needs SM100+ (the 3090 is SM86)
# and vLLM's GGUF path does not cover this hybrid linear-attention MoE.
#
# CMAKE_CUDA_ARCHITECTURES covers both cards that could host it — 86 for
# the two 3090s (the secondary's actual home) and 120 for the RTX PRO 6000.
# Dropping to just 86 roughly halves the build time if you never intend to
# run llama.cpp on the Blackwell.
#
# Requires: cmake, CUDA toolkit at /opt/cuda
#
# After running this, start the server with:
#   MODEL=qwen36 bash bin/start-secondary.sh

# Verify cmake is available
if ! command -v cmake &>/dev/null; then
    echo "ERROR: cmake not found. Install it first:"
    echo "  sudo pacman -S cmake    # Arch"
    echo "  sudo apt install cmake  # Debian/Ubuntu"
    exit 1
fi

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_DIR="$PROJECT_DIR/llm/llama.cpp"
MODELS_ROOT="$PROJECT_DIR/llm/models"
MODEL_REPO="unsloth/Qwen3.6-35B-A3B-GGUF"
# UD-Q3_K_XL (16.85 GB). Q4_K_S is 19.5 GiB of weights and does not leave
# room for the 5.0 GiB f16 KV cache the full window needs. Do NOT substitute
# UD-IQ4_NL: that quant of this model is incoherent — HTML/XML/tool_call
# fragments instead of prose — on SM89 and SM120 alike. ggml-org/llama.cpp#21495
MODEL_NAME="Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf"
MODEL_DIR="$MODELS_ROOT/unsloth-Qwen3.6-35B-A3B-GGUF"

echo "=== LLM Server Setup (llama.cpp + Qwen3.6-35B-A3B) ==="

# Clone llama.cpp
if [ -d "$LLAMA_DIR" ]; then
    echo "llama.cpp already cloned, pulling latest..."
    git -C "$LLAMA_DIR" pull
else
    echo "Cloning llama.cpp..."
    git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
fi

# Build with CUDA
echo "Building llama.cpp with CUDA support..."
cd "$LLAMA_DIR"
export LD_LIBRARY_PATH="/opt/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cmake -B build \
    -DGGML_CUDA=ON \
    -DGGML_NATIVE=ON \
    -DCMAKE_CUDA_ARCHITECTURES='86;120' \
    -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc \
    -DCMAKE_EXE_LINKER_FLAGS="-L/opt/cuda/lib64 -Wl,-rpath,/opt/cuda/lib64" \
    -DCMAKE_SHARED_LINKER_FLAGS="-L/opt/cuda/lib64 -Wl,-rpath,/opt/cuda/lib64"
cmake --build build --target llama-server -j"$(nproc)"

echo "llama-server built at $LLAMA_DIR/build/bin/llama-server"

# Models live *in* the tree, not in $HOME. This used to symlink
# llm/models -> $HOME/models, which on a fresh rebuild would silently
# redirect MODEL_DIR out of the tree while every start-*.sh kept resolving
# $PROJECT_DIR/llm/models. On this box it is already a real directory
# holding ~200 GB, so the symlink was a no-op here and only ever bit the
# rebuild path SETUP.md documents.
mkdir -p "$MODELS_ROOT"

# Download model if not present
if [ ! -f "$MODEL_DIR/$MODEL_NAME" ]; then
    echo "Downloading $MODEL_REPO ($MODEL_NAME, ~17GB)..."
    mkdir -p "$MODEL_DIR"
    "$HOME/lloyd/.venvs/lloyd/bin/hf" download "$MODEL_REPO" "$MODEL_NAME" \
        --local-dir "$MODEL_DIR"
else
    echo "Model already present: $MODEL_DIR/$MODEL_NAME"
fi

echo ""
echo "Done. LLM server setup complete."
echo "  llama-server: $LLAMA_DIR/build/bin/llama-server"
echo "  Model: $MODEL_DIR/$MODEL_NAME"
echo "Run: MODEL=qwen36 bash bin/start-secondary.sh"
