#!/usr/bin/env bash
set -euo pipefail

# Starts CosyVoice3 TTS server on port 8090 (GPU 1)
# Run setup first: bash setup/setup-cosyvoice.sh
#
# Alternative to Orpheus — both use port 8090, run one at a time.

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CV_VENV="$PROJECT_DIR/.venvs/cosyvoice"
CV_REPO="$CV_VENV/CosyVoice"
MODEL_DIR="$CV_REPO/pretrained_models/Fun-CosyVoice3-0.5B"

# Reference voice
PROMPT_WAV="$PROJECT_DIR/services/tts/references/ed/ed_002.wav"
PROMPT_TEXT="$(cat "$PROJECT_DIR/services/tts/references/ed/ed_002.lab")"

if [[ ! -x "$CV_VENV/bin/python" ]]; then
  echo "CosyVoice venv not found. Run: bash setup/setup-cosyvoice.sh"
  exit 1
fi

source "$CV_VENV/bin/activate"
export PYTHONPATH="$CV_REPO:$CV_REPO/third_party/Matcha-TTS${PYTHONPATH:+:$PYTHONPATH}"

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
python "$PROJECT_DIR/services/tts/cosyvoice_server.py" \
    --host 127.0.0.1 \
    --port 8090 \
    --model-dir "$MODEL_DIR" \
    --prompt-wav "$PROMPT_WAV" \
    --prompt-text "$PROMPT_TEXT" \
    --fp16 \
    --trt \
    --vllm \
    --cfg-rate 0.3 \
    --flow-steps 6 \
    --token-hop 13

# fast
# --cfg-rate 0.0 (re-enable some guidance)
# --flow-steps 6 (more diffusion steps)
# --token-hop 13 (longer first chunk, better quality)

# defaults
# --cfg-rate 0.3 (re-enable some guidance)
# --flow-steps 8 (more diffusion steps)
# --token-hop 25 (longer first chunk, better quality)
