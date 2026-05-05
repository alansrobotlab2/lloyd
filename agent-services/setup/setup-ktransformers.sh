#!/usr/bin/env bash
set -euo pipefail

# One-time setup for ktransformers venv with unsloth Qwen3.5-122B-A10B-UD-Q4_K_XL
# Uses gpu2 (RTX 3090, index 2) + CPU offloading for the model
#
# After running this, start inference with:
#   bash bin/start-llm-122b.sh

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
KT_VENV="$PROJECT_DIR/venvs/ktransformers"

echo "=== ktransformers Setup ==="

echo "Creating ktransformers venv at $KT_VENV (Python 3.12)..."
uv venv "$KT_VENV" --python 3.12

# Install PyTorch cu124 (matches driver 590.x, Arch Linux)
echo "Installing PyTorch cu124..."
uv pip install --python "$KT_VENV/bin/python" \
    torch==2.7.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Install ktransformers + unsloth dependencies
echo "Installing ktransformers and dependencies..."
uv pip install --python "$KT_VENV/bin/python" \
    ktransformers \
    unsloth \
    bitsandbytes \
    accelerate \
    peft \
    sentencepiece \
    protobuf==5.29.3

# Install additional deps for quantized models
echo "Installing extra dependencies..."
uv pip install --python "$KT_VENV/bin/python" \
    huggingface-hub \
    safetensors \
    torchao

# Download model config files (tokenizer, config.json — needed by ktransformers)
MODEL_DIR="$PROJECT_DIR/llm/models/Qwen3.5-122B-A10B-UD-Q4_K_XL"
mkdir -p "$MODEL_DIR"

echo "Downloading model config files..."
"$KT_VENV/bin/python" -m huggingface_hub.commands.huggingface_cli download \
    unsloth/Qwen3.5-122B-A10B-GGUF \
    --include "UD-Q4_K_XL/*.gguf" "config.json" "tokenizer*" "special_tokens_map.json" "generation_config.json" \
    --local-dir "$MODEL_DIR"

# Move GGUF files from subfolder to model dir
if [[ -d "$MODEL_DIR/UD-Q4_K_XL" ]]; then
    mv "$MODEL_DIR/UD-Q4_K_XL/"*.gguf "$MODEL_DIR/"
    rmdir "$MODEL_DIR/UD-Q4_K_XL" 2>/dev/null || true
fi

echo ""
echo "WARNING: GGUF download is ~214GB (3 shards). This will take a while."
echo "Model files will be at: $MODEL_DIR"

echo ""
echo "=== Setup Complete ==="
echo "ktransformers venv ready at: $KT_VENV"
echo ""
echo "To start inference:"
echo "  bash bin/start-llm-122b.sh"
echo ""
echo "To enable as a systemd service:"
echo "  systemctl --user enable --now lloyd-ktransformers.service"
echo ""