#!/usr/bin/env bash
set -euo pipefail

# Starts the Qwen3-TTS API server on port 8090 (GPU 1)
#
# OpenAI-compatible TTS with streaming PCM output
# Model: Qwen/Qwen3-TTS-12Hz-1.7B-Base (local copy)
# VRAM: ~4-6GB estimated
#
# API: POST /v1/audio/speech

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$HOME/lloyd/.venvs/qwen3-tts"
QWEN3_TTS_DIR="$PROJECT_DIR/services/tts/qwen3-tts"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "Qwen3-TTS venv not found at $VENV"
  echo "Create it with: uv venv $VENV --python 3.12"
  exit 1
fi

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="/run/host/usr/lib:/opt/cuda/lib64:${LD_LIBRARY_PATH:-}"
export TTS_BACKEND=optimized
export TTS_CONFIG="$QWEN3_TTS_DIR/config.yaml"
export PORT=8090

cd "$QWEN3_TTS_DIR"

# Wait for port to be free (don't kill — supervisor handles process lifecycle)
for i in $(seq 1 30); do
  ss -tlnp 2>/dev/null | grep -q ":${PORT} " || break
  echo "Waiting for port $PORT to be free... (${i}/30)"
  [[ $i -eq 30 ]] && { echo "ERROR: port $PORT still in use after 30s"; exit 1; }
  sleep 1
done

exec "$VENV/bin/python" -m uvicorn api.main:app \
    --host 0.0.0.0 \
    --port 8090
