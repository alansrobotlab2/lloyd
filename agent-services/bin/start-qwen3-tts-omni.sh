#!/usr/bin/env bash
set -euo pipefail

# Qwen3-TTS 1.7B Base served by vLLM-Omni on :8093 — the TTS server voice mode
# uses since 2026-10-09 (architecture/voice.md "The TTS server").
#
# Same model, same cloned voice, same OpenAI-shaped `/v1/audio/speech`, a
# different engine: CUDA graphs on both stages, async codec chunking, and a
# shared-memory hand-off between the talker and the codec decoder. Measured on
# this box against the vendored server on GPU 0 (`scripts/voice/tts_bakeoff.py`,
# ten sentences, pool paused): first byte p50 0.09 s against 0.26 s, RTF 0.19
# against 0.49, the same speaker similarity (0.91) and WER within noise.
#
# GPU 3 (one of the two 3090s added 2026-10-07), pinned by UUID so the index
# trap (memory: /dev/nvidiaN != nvidia-smi index) cannot bite; conf.d/
# agent-tts-omni.conf repeats the pin. The venv is `.venvs/vllm-omni`
# (SETUP.md): vllm-omni 0.30.0 with the matching vllm 0.30.0, plus `ninja`,
# which inductor needs on PATH — the first boot died in the talker's compile
# with `FileNotFoundError: 'ninja'` because `$VENV/bin` was not on it.
#
# Boot takes ~2.5 min (two engines, CUDA-graph capture) and `/health` answers
# only at the end, so a request during the window gets a connection refusal,
# not the 120 s hang the old server's lazy load produced. The cloned voice is a
# persisted upload (`scripts/voice/register_tts_voice.py`), stored under
# `SPEAKER_SAMPLES_DIR` — runtime data, so it lives in ~/lloyd-data and is
# re-registered from the profile directory if lost; the profile is the asset.
#
# `${VAR:-default}` throughout: supervisor's `environment=` line arrives before
# bash runs, so a hard export here would clobber an operator's override.

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$HOME/lloyd/.venvs/vllm-omni"
MODEL="${TTS_OMNI_MODEL:-$PROJECT_DIR/services/tts/qwen3-tts/models/Qwen3-TTS-12Hz-1.7B-Base}"
# Our copy of upstream's qwen3_tts.yaml with the codec chunk ramp on (see it).
DEPLOY="${TTS_OMNI_DEPLOY:-$PROJECT_DIR/conf/qwen3-tts-omni.deploy.yaml}"

if [[ ! -x "$VENV/bin/vllm" ]]; then
  echo "vllm-omni venv not found at $VENV — SETUP.md 'vllm-omni'" >&2
  exit 1
fi
if [[ ! -f "$DEPLOY" ]]; then
  echo "deploy config not found: $DEPLOY" >&2
  exit 1
fi
if [[ ! -d "$MODEL" ]]; then
  echo "model directory not found: $MODEL (SETUP.md Part 8)" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-GPU-0bd41021-2916-fb5d-c100-5bdc5a9c41c0}"
export LD_LIBRARY_PATH="/usr/lib:/opt/cuda/lib64:${LD_LIBRARY_PATH:-}"
# ninja (inductor) and nvcc live here; the venv is not activated.
export PATH="$VENV/bin:/opt/cuda/bin:$PATH"
export SPEAKER_SAMPLES_DIR="${SPEAKER_SAMPLES_DIR:-/home/alansrobotlab/lloyd-data/tts/speakers}"
mkdir -p "$SPEAKER_SAMPLES_DIR"

cd "$PROJECT_DIR"
exec "$VENV/bin/vllm" serve "$MODEL" \
  --deploy-config "$DEPLOY" \
  --omni \
  --host "${TTS_OMNI_HOST:-127.0.0.1}" \
  --port "${TTS_OMNI_PORT:-8093}" \
  --served-model-name qwen3-tts \
  --trust-remote-code
