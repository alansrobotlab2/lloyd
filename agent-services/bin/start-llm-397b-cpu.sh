#!/usr/bin/env bash
set -euo pipefail

# Starts Qwen3.5-397B-A17B via llama-server on CPU only
# IQ3_S quantization, 256K context, OpenAI-compatible API on port 8097
#
# ~150GB model (IQ3_S, 4 shards) — runs entirely in RAM, no GPU
# Thinking enabled server-side; non-thinking sampling per Unsloth guide

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

LLAMA_SERVER="$PROJECT_DIR/llm/llama.cpp/build/bin/llama-server"
MODEL_DIR="$PROJECT_DIR/llm/models"
MODEL="$MODEL_DIR/Qwen3.5-397B-A17B-UD-IQ3_S/Qwen3.5-397B-A17B-UD-IQ3_S-00001-of-00004.gguf"

if [[ ! -x "$LLAMA_SERVER" ]]; then
  echo "llama-server not found at $LLAMA_SERVER"
  echo "Run: bash setup/setup-llm.sh"
  exit 1
fi

if [[ ! -f "$MODEL" ]]; then
  echo "Model not found at $MODEL"
  exit 1
fi

export LD_LIBRARY_PATH="/usr/lib:/opt/cuda/lib64:$PROJECT_DIR/llm/llama.cpp/build/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

CUDA_VISIBLE_DEVICES="" \
"$LLAMA_SERVER" \
  --model "$MODEL" \
  --alias Qwen3.5-397B-A17B \
  --port 8097 \
  --host 127.0.0.1 \
  --gpu-layers 0 \
  --ctx-size 262144 \
  --batch-size 512 \
  --threads 32 \
  --mlock \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --chat-template-kwargs '{"enable_thinking": true}' \
  --temp 0.7 \
  --top-p 0.8 \
  --top-k 20 \
  --min-p 0.0 \
  --presence-penalty 1.5 \
  --repeat-penalty 1.0
