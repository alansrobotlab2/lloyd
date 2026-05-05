#!/usr/bin/env bash
set -euo pipefail

# Starts the Orpheus TTS server on port 8090 (GPU 1)
# Run setup first: bash setup/setup-orpheus.sh
#
# Voices: tara, leah, jess, leo, dan, mia, zac, zoe
# Expression tags:  <laugh> <chuckle> <sigh> <cough> <sniffle> <groan> <yawn> <gasp>
# Emotion tags:     <happy> <sad> <angry> <excited> <fearful> <surprised> <disgusted> <neutral>

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$PROJECT_DIR/.venvs/orpheus"

# Model downloaded from HuggingFace on first run (~6.5GB), cached to ~/.cache/huggingface/
# Using unsloth mirror — canopylabs/orpheus-3b-0.1-ft is gated
MODEL="unsloth/orpheus-3b-0.1-ft"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Orpheus venv not found. Run: bash setup/setup-orpheus.sh"
  exit 1
fi

source "$VENV/bin/activate"

HF_TOKEN="${HF_TOKEN:-$(cat "$HOME/.cache/huggingface/token" 2>/dev/null || true)}"

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export HF_TOKEN="$HF_TOKEN"

python "$PROJECT_DIR/services/tts/orpheus_server.py" \
    --host 127.0.0.1 \
    --port 8090 \
    --model "$MODEL" \
    --voice dan
