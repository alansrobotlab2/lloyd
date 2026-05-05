#!/usr/bin/env bash
set -euo pipefail

# Setup for MiniMax-M2.7 (Unsloth IQ3_XXS GGUF quantization)
# Downloads model (~80GB) and ensures llama.cpp is built.
#
# After running this, start the server with:
#   bash bin/start-minimax-m2.7.sh

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_DIR="$PROJECT_DIR/llm/llama.cpp"
MODELS_LINK="$PROJECT_DIR/llm/models"
MODEL_REPO="unsloth/MiniMax-M2.7-GGUF"
MODEL_DIR="$HOME/models/MiniMax-M2.7-UD-IQ3_XXS"
HF_CLI="$PROJECT_DIR/.venvs/vllm-experimental/bin/huggingface-cli"

echo "=== MiniMax-M2.7 Setup (Unsloth IQ3_XXS GGUF) ==="

# Ensure llama.cpp is built
if [[ ! -x "$LLAMA_DIR/build/bin/llama-server" ]]; then
    echo "llama-server not found. Building llama.cpp..."
    if [[ ! -d "$LLAMA_DIR" ]]; then
        git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
    else
        git -C "$LLAMA_DIR" pull
    fi
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
else
    echo "llama-server already built: $LLAMA_DIR/build/bin/llama-server"
fi

# Ensure models symlink exists
if [[ ! -e "$MODELS_LINK" ]]; then
    ln -sf "$HOME/models" "$MODELS_LINK"
fi

# Download model
if ls "$MODEL_DIR"/UD-IQ3_XXS/MiniMax-M2.7-UD-IQ3_XXS-*.gguf &>/dev/null; then
    echo "Model already present: $MODEL_DIR"
else
    echo "Downloading $MODEL_REPO IQ3_XXS (~80GB, 3 files)..."
    mkdir -p "$MODEL_DIR"
    "$HF_CLI" download "$MODEL_REPO" \
        --include "UD-IQ3_XXS/*" \
        --local-dir "$MODEL_DIR"
fi

echo ""
echo "Done. MiniMax-M2.7 setup complete."
echo "  llama-server: $LLAMA_DIR/build/bin/llama-server"
echo "  Model: $MODEL_DIR/"
echo "Run: bash bin/start-minimax-m2.7.sh"
