#!/usr/bin/env bash
set -euo pipefail

# One-time setup for CosyVoice3 TTS venv.
# Creates venvs/cosyvoice/ (Python 3.10) with all dependencies.
# Separate venv because CosyVoice pins numpy<=1.26.4 (conflicts with main project).
#
# After running this, start the server with:
#   bash bin/start-cosyvoice-tts.sh

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CV_VENV="$PROJECT_DIR/venvs/cosyvoice"
CV_REPO="$CV_VENV/CosyVoice"
MODEL_DIR="$CV_REPO/pretrained_models/Fun-CosyVoice3-0.5B"

echo "=== CosyVoice3 Setup ==="

echo "Creating CosyVoice venv at $CV_VENV (Python 3.10)..."
uv venv "$CV_VENV" --python 3.10

echo "Cloning CosyVoice repo (with Matcha-TTS submodule)..."
if [ -d "$CV_REPO" ]; then
    echo "  Repo already exists, pulling latest..."
    git -C "$CV_REPO" pull
    git -C "$CV_REPO" submodule update --init --recursive
else
    git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git "$CV_REPO"
fi

echo "Installing PyTorch (cu121 — compatible with CUDA 12.x)..."
uv pip install --python "$CV_VENV/bin/python" \
    torch==2.3.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121

# setuptools <70 needed for openai-whisper build (pkg_resources removed in v82)
echo "Installing setuptools and pip..."
uv pip install --python "$CV_VENV/bin/python" "setuptools<70" pip

echo "Installing CosyVoice requirements..."
# Filter out: tensorrt (not needed, fails on Arch), openai-whisper (needs
# special build handling), deepspeed (training-only). Use unsafe-best-match
# so protobuf resolves from PyPI even though onnxruntime extra index is listed.
grep -v -i -E 'tensorrt|openai-whisper|deepspeed' "$CV_REPO/requirements.txt" | \
    uv pip install --python "$CV_VENV/bin/python" -r /dev/stdin \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    --index-strategy unsafe-best-match

# openai-whisper needs pkg_resources at build time; build without isolation
# so it picks up the setuptools we just installed
echo "Installing openai-whisper (no build isolation)..."
"$CV_VENV/bin/python" -m pip install --no-build-isolation openai-whisper==20231117

echo "Installing FastAPI + uvicorn for TTS server..."
uv pip install --python "$CV_VENV/bin/python" fastapi uvicorn

echo "Installing TensorRT for flow decoder acceleration..."
"$CV_VENV/bin/python" -m pip install tensorrt

echo "Installing vLLM 0.9.0 for LLM acceleration (~4x faster token generation)..."
# vLLM upgrades torch and other deps; pin numpy back after
"$CV_VENV/bin/python" -m pip install vllm==0.9.0
"$CV_VENV/bin/python" -m pip install numpy==1.26.4
# onnxruntime-gpu must match the CUDA libs from the upgraded torch
"$CV_VENV/bin/python" -m pip install "onnxruntime-gpu>=1.20"

echo "Downloading Fun-CosyVoice3-0.5B-2512 model (~10GB)..."
mkdir -p "$CV_REPO/pretrained_models"
"$CV_VENV/bin/python" -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
    local_dir='$MODEL_DIR',
)
print('Model downloaded to $MODEL_DIR')
"

echo "Activating RL variant (symlink llm.rl.pt -> llm.pt)..."
if [ -f "$MODEL_DIR/llm.rl.pt" ]; then
    if [ -f "$MODEL_DIR/llm.pt" ] && [ ! -L "$MODEL_DIR/llm.pt" ]; then
        mv "$MODEL_DIR/llm.pt" "$MODEL_DIR/llm.base.pt"
        echo "  Backed up original llm.pt -> llm.base.pt"
    fi
    ln -sf llm.rl.pt "$MODEL_DIR/llm.pt"
    echo "  Symlinked llm.rl.pt -> llm.pt (RL variant active)"
else
    echo "  WARNING: llm.rl.pt not found, using base llm.pt"
fi

echo ""
echo "Done. CosyVoice3 venv ready at $CV_VENV"
echo "Model at $MODEL_DIR"
echo "Run: bash bin/start-cosyvoice-tts.sh"
