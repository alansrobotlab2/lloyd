#!/usr/bin/env bash
set -euo pipefail

# One-time setup for the local LLM server (llama.cpp + Qwen3.5 model).
# Clones and builds llama.cpp with CUDA support, then symlinks model directory.
#
# Requires: cmake, CUDA toolkit at /opt/cuda
#
# After running this, start the server with:
#   bash bin/start-llm.sh

# Verify cmake is available
if ! command -v cmake &>/dev/null; then
    echo "ERROR: cmake not found. Install it first:"
    echo "  sudo pacman -S cmake    # Arch"
    echo "  sudo apt install cmake  # Debian/Ubuntu"
    exit 1
fi

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_DIR="$PROJECT_DIR/llm/llama.cpp"
MODELS_LINK="$PROJECT_DIR/llm/models"
MODEL_REPO="bartowski/Qwen3.5-35B-A3B-GGUF"
MODEL_NAME="Qwen3.5-35B-A3B-Q4_K_M.gguf"
MODEL_DIR="$HOME/models/Qwen3.5-35B-A3B-Q4"

echo "=== LLM Server Setup (llama.cpp + Qwen3.5) ==="

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

# Symlink models directory
if [ ! -e "$MODELS_LINK" ]; then
    echo "Symlinking models directory..."
    ln -sf "$HOME/models" "$MODELS_LINK"
fi

# Download model if not present
if [ ! -f "$MODEL_DIR/$MODEL_NAME" ]; then
    echo "Downloading $MODEL_REPO ($MODEL_NAME, ~24GB)..."
    mkdir -p "$MODEL_DIR"
    huggingface-cli download "$MODEL_REPO" "$MODEL_NAME" --local-dir "$MODEL_DIR"
else
    echo "Model already present: $MODEL_DIR/$MODEL_NAME"
fi

echo ""
echo "Done. LLM server setup complete."
echo "  llama-server: $LLAMA_DIR/build/bin/llama-server"
echo "  Model: $MODEL_DIR/$MODEL_NAME"
echo "Run: bash bin/start-llm.sh"
