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
export LD_LIBRARY_PATH="/usr/lib:/opt/cuda/lib64:${LD_LIBRARY_PATH:-}"
export TTS_BACKEND=optimized
export TTS_CONFIG="$QWEN3_TTS_DIR/config.yaml"
export PORT=8090

# Eager loading: load and compile the model at BOOT, not on the first thing
# anyone says. The backend's own default is lazy — api/main.py resolves
# `TTS_LAZY_LOAD = _env_bool("TTS_LAZY_LOAD", True)` — and this model's
# `compile_mode: max-autotune` makes the first synthesis request pay the
# inductor autotune. Measured 2026-09-19: the stack restarted at 19:54, the
# first voice turn arrived at 20:04:42 and audio came out at 20:08:48 — 4 min
# 6 s, of which "Warmup 1/3 streaming" alone was 2 min 59 s.
#
# The cost of that first request is not only latency. TTSStreamer._http in
# agent-services/livekit_worker.py is one serial client with `read=120.0`, so an
# utterance that waits longer than 120 s is DISCARDED, not merely delayed: that
# incident lost "One moment." and "Honestly?" exactly 120 s apart, and the answer
# to a 20:04:39 question was first heard at 20:08:56, starting mid-sentence.
#
# Eager loading moves the wait to where nobody is talking: the warmup runs
# inside uvicorn's lifespan, so :8090 does not answer /health for ~4 min after a
# restart. TTS_WARMUP_ON_START (set in conf.d/agent-tts.conf) rides along but
# does nothing for this Base model — eager init is the fix, not the warmup.
#
# `${VAR:-default}`, never a bare export: supervisor passes its `environment=`
# line into this script's environment before bash runs it, so a hard `export
# TTS_LAZY_LOAD=false` would silently clobber an operator who set it back to lazy
# there. The default lives here rather than only in the conf so that a person
# running this script gets the engine production boots — until #1446 the knob
# existed only in conf.d/agent-tts.conf, and a hand-run of this file reproduced
# the 2026-09-19 loss exactly.
export TTS_LAZY_LOAD="${TTS_LAZY_LOAD:-false}"

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
