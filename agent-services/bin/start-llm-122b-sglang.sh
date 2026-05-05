#!/usr/bin/env bash
set -euo pipefail

# Starts Qwen3.5-122B-A10B via SGLang on GPU 0
# AWQ 4-bit quantization + MTP speculative decoding (NEXTN)
# OpenAI-compatible API on port 8096

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

SGLANG_VENV="$PROJECT_DIR/.venvs/sglang"
MODEL_DIR="$PROJECT_DIR/llm/models/Qwen3.5-122B-A10B-AWQ-4bit"

if [[ ! -x "$SGLANG_VENV/bin/python" ]]; then
  echo "SGLang venv not found at $SGLANG_VENV"
  exit 1
fi

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "Model not found at $MODEL_DIR"
  exit 1
fi

export PATH="$SGLANG_VENV/bin:/opt/cuda/bin:/run/host/usr/bin:/usr/bin:/usr/sbin:$PATH"
export LD_LIBRARY_PATH="/run/host/usr/lib:/opt/cuda/targets/x86_64-linux/lib:/opt/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_HOME="/opt/cuda"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export SGLANG_ENABLE_SPEC_V2=1

exec "$SGLANG_VENV/bin/python" -m sglang.launch_server \
  --model-path "$MODEL_DIR" \
  --served-model-name Qwen3.5-122B-A10B \
  --port 8096 \
  --host 127.0.0.1 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --context-length 262144 \
  --max-running-requests 8 \
  --chunked-prefill-size 8192 \
  --mem-fraction-static 0.88 \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend triton \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --mamba-scheduler-strategy extra_buffer \
  --enable-cudagraph-gc \
  --log-level info
