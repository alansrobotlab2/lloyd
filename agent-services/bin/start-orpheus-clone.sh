#!/usr/bin/env bash
set -euo pipefail

# Starts the Orpheus TTS Clone server on port 8097 (GPU 1)
# Zero-shot voice cloning using SNAC-encoded reference audio

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$PROJECT_DIR/.venvs/orpheus"
MODEL="unsloth/orpheus-3b-0.1-ft"
REFERENCE_DIR="$PROJECT_DIR/references/cullen"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Orpheus venv not found. Run: bash setup/setup-orpheus.sh"
  exit 1
fi

source "$VENV/bin/activate"

# Ensure snac + librosa are installed
pip show snac &>/dev/null || pip install snac
pip show librosa &>/dev/null || pip install librosa

HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}"

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export HF_TOKEN="$HF_TOKEN"

python "$PROJECT_DIR/services/tts/orpheus_clone_server.py" \
    --host 127.0.0.1 \
    --port 8097 \
    --model "$MODEL" \
    --reference-dir "$REFERENCE_DIR"
