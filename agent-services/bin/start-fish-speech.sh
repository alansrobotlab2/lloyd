#!/usr/bin/env bash
set -euo pipefail

# Starts the Fish Speech API server on port 8098 (GPU 1)
#
# High-quality TTS with voice cloning via reference audio
# Model: fishaudio/fish-speech (auto-downloads on first run)
# VRAM: ~4-6GB
#
# API: POST /v1/tts with streaming WAV output

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$PROJECT_DIR/.venvs/fish-speech"
FISH_SPEECH_DIR="$PROJECT_DIR/services/tts/fish-speech"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Fish Speech venv not found at $VENV"
  echo "Create it with: cd $FISH_SPEECH_DIR && VIRTUAL_ENV=$VENV uv sync --python 3.12 --extra cu126"
  exit 1
fi

# source activate replaced with direct python path

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

cd "$FISH_SPEECH_DIR"

"$VENV/bin/python" -m tools.api_server \
    --listen 127.0.0.1:8098 \
    --half
