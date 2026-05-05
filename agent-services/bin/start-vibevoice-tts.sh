#!/usr/bin/env bash
set -euo pipefail

# Starts the VibeVoice TTS Gradio server on port 8095 (GPU 1)
# Model: microsoft/VibeVoice-1.5B (~3GB VRAM with bf16, downloads on first run)
# Built-in voices: Alice, Carter, Frank, Mary, Maya, Samuel, Anchen, Bowen, Xinran
# Features: multi-speaker (up to 4), long-form conversational, streaming, voice cloning via reference audio
# Uses flash_attention_2 for best quality (falls back to SDPA without)

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$PROJECT_DIR/.venvs/vibevoice"

MODEL="microsoft/VibeVoice-1.5B"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "VibeVoice venv not found at $VENV"
  exit 1
fi

source "$VENV/bin/activate"

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

# Gradio respects these env vars; server_port is commented out in the demo script
export GRADIO_SERVER_PORT=8095
export GRADIO_SERVER_NAME=127.0.0.1

python "$PROJECT_DIR/services/tts/vibevoice/demo/gradio_demo.py" \
    --model_path "$MODEL" \
    --inference_steps 10
