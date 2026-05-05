#!/usr/bin/env bash
set -euo pipefail

# Qwen3.5-4B general-purpose LLM on GPU 1, port 8099
# 32K context, deterministic (temp 0.0), no sampling randomness

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

LLAMA_SERVER="$PROJECT_DIR/llm/llama.cpp/build/bin/llama-server"
MODEL="$PROJECT_DIR/llm/models/gguf/Qwen3.5-4B-UD-Q4_K_XL.gguf"

if [[ ! -x "$LLAMA_SERVER" ]]; then
  echo "llama-server not found at $LLAMA_SERVER"
  echo "Run: bash setup/setup-llm.sh"
  exit 1
fi

if [[ ! -f "$MODEL" ]]; then
  echo "Model not found at $MODEL"
  echo "Run: bash setup/setup-llm.sh"
  exit 1
fi

export LD_LIBRARY_PATH="/run/host/usr/lib:/opt/cuda/lib64:$PROJECT_DIR/llm/llama.cpp/build/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
"$LLAMA_SERVER" \
  --model "$MODEL" \
  --alias Qwen3.5-4B \
  --port 8099 \
  --host 127.0.0.1 \
  --gpu-layers -1 \
  --ctx-size 32768 \
  --batch-size 512 \
  --flash-attn on \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --temp 0.0
