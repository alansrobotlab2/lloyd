#!/usr/bin/env bash
set -euo pipefail

# Starts the Index TTS WebUI (Gradio) on port 8094 (GPU 1)
#
# Zero-shot voice cloning TTS with emotion control (IndexTTS2)
# Model: IndexTeam/IndexTTS-2 (~3.5GB checkpoints)
# Features: voice cloning, emotion via reference audio / vector / text description
# VRAM: ~4GB FP16, ~6-8GB FP32
#
# Download models first:
#   source .venvs/index-tts/bin/activate
#   pip install "huggingface-hub[cli,hf_xet]"
#   huggingface-cli download IndexTeam/IndexTTS-2 --local-dir services/tts/index-tts/checkpoints

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$PROJECT_DIR/.venvs/index-tts"
INDEX_TTS_DIR="$PROJECT_DIR/services/tts/index-tts"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Index TTS venv not found at $VENV"
  echo "Create it with: cd $INDEX_TTS_DIR && uv sync --all-extras"
  exit 1
fi

# Check that model checkpoints exist
if [[ ! -f "$INDEX_TTS_DIR/checkpoints/gpt.pth" ]]; then
  echo "Model checkpoints not found in $INDEX_TTS_DIR/checkpoints/"
  echo "Download with:"
  echo "  huggingface-cli download IndexTeam/IndexTTS-2 --local-dir $INDEX_TTS_DIR/checkpoints"
  exit 1
fi

source "$VENV/bin/activate"

HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}"

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export HF_TOKEN="$HF_TOKEN"

cd "$INDEX_TTS_DIR"

python webui.py \
    --host 127.0.0.1 \
    --port 8094 \
    --fp16 \
    --model_dir ./checkpoints
