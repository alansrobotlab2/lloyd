#!/usr/bin/env bash
set -euo pipefail

# Starts Qwen3.5-397B-A17B via llama-server split across GPU 0 + GPU 1
# 256K context, auto GPU layer offload, OpenAI-compatible API on port 8096
#
# ~107GB model (UD-IQ1_M, 4 shards)
# GPU0 = RTX 5090 (32GB), GPU1 = Blackwell (96GB, voice/TTS/qmd on GPU1) — tensor split 25/75
# Thinking enabled server-side; non-thinking sampling per Unsloth guide

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

LLAMA_SERVER="$PROJECT_DIR/llm/llama.cpp/build/bin/llama-server"
MODEL_DIR="$PROJECT_DIR/llm/models"
MODEL="$MODEL_DIR/Qwen3.5-397B-A17B-UD-IQ2_XXS/Qwen3.5-397B-A17B-UD-IQ2_XXS-00001-of-00004.gguf"

if [[ ! -x "$LLAMA_SERVER" ]]; then
  echo "llama-server not found at $LLAMA_SERVER"
  echo "Run: bash setup/setup-llm.sh"
  exit 1
fi

if [[ ! -f "$MODEL" ]]; then
  echo "Model not found at $MODEL"
  exit 1
fi

export LD_LIBRARY_PATH="/opt/cuda/lib64:$PROJECT_DIR/llm/llama.cpp/build/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 \
"$LLAMA_SERVER" \
  --model "$MODEL" \
  --alias Qwen3.5-397B-A17B \
  --port 8096 \
  --host 127.0.0.1 \
  --gpu-layers auto \
  --tensor-split 18,82 \
  --ctx-size 262144 \
  --batch-size 512 \
  --flash-attn on \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --chat-template-kwargs '{"enable_thinking": true}' \
  --temp 0.7 \
  --top-p 0.8 \
  --top-k 20 \
  --min-p 0.0 \
  --presence-penalty 1.5 \
  --repeat-penalty 1.0
