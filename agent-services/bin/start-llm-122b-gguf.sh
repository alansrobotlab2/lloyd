#!/usr/bin/env bash
set -euo pipefail

# Starts Qwen3.5-122B-A10B via llama.cpp on GPU 0
# Unsloth IQ4_NL GGUF quantization + ngram speculative decoding
# OpenAI-compatible API on port 8096

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

LLAMA_SERVER="$PROJECT_DIR/llm/llama.cpp/build/bin/llama-server"
MODEL="$PROJECT_DIR/llm/models/Qwen3.5-122B-A10B-UD-IQ4-NL/Qwen3.5-122B-A10B-UD-IQ4_NL-00001-of-00003.gguf"

if [[ ! -x "$LLAMA_SERVER" ]]; then
  echo "llama-server not found at $LLAMA_SERVER"
  echo "Run: bash setup/setup-llm.sh"
  exit 1
fi

if [[ ! -f "$MODEL" ]]; then
  echo "Model not found at $MODEL"
  exit 1
fi

export LD_LIBRARY_PATH="/usr/lib:/opt/cuda/lib64:/opt/cuda/targets/x86_64-linux/lib:$PROJECT_DIR/llm/llama.cpp/build/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
"$LLAMA_SERVER" \
  --model "$MODEL" \
  --alias Qwen3.5-122B-A10B \
  --port 8096 \
  --host 127.0.0.1 \
  --gpu-layers -1 \
  --ctx-size 262144 \
  --batch-size 2048 \
  --ubatch-size 512 \
  --flash-attn on \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --chat-template qwen3 \
  --chat-template-kwargs '{"enable_thinking": true}' \
  --temp 1.0 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.0 \
  --presence-penalty 1.5 \
  --repeat-penalty 1.0 \
  --threads 4 \
  --parallel 4 \
  --draft-max 3 \
  --spec-type ngram-mod
