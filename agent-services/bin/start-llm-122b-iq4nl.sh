#!/usr/bin/env bash
set -euo pipefail

# Starts Qwen3.5-122B-A10B UD-IQ4_NL via llama-server
# 256K context, OpenAI-compatible API on port 8096
#
# ~62GB model (3 shards) — fits on single GPU1 (Blackwell 96GB)
# GPU assignment controlled by CUDA_VISIBLE_DEVICES env var:
#   Single GPU:  CUDA_VISIBLE_DEVICES=1
#   Dual GPU:    CUDA_VISIBLE_DEVICES=0,1 with --tensor-split
# Thinking enabled server-side; non-thinking sampling per Unsloth guide

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

LLAMA_SERVER="$PROJECT_DIR/llm/llama.cpp/build/bin/llama-server"
MODEL_DIR="$PROJECT_DIR/llm/models"
MODEL="$MODEL_DIR/Qwen3.5-122B-A10B-UD-IQ4-NL/Qwen3.5-122B-A10B-UD-IQ4_NL-00001-of-00003.gguf"

if [[ ! -x "$LLAMA_SERVER" ]]; then
  echo "llama-server not found at $LLAMA_SERVER"
  exit 1
fi

if [[ ! -f "$MODEL" ]]; then
  echo "Model not found at $MODEL"
  exit 1
fi

export LD_LIBRARY_PATH="/opt/cuda/lib64:$PROJECT_DIR/llm/llama.cpp/build/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# CUDA_VISIBLE_DEVICES and --tensor-split passed via env or defaults below
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
TENSOR_SPLIT="${TENSOR_SPLIT:-}"

SPLIT_ARGS=()
[[ -n "$TENSOR_SPLIT" ]] && SPLIT_ARGS+=(--tensor-split "$TENSOR_SPLIT")

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
"$LLAMA_SERVER" \
  --model "$MODEL" \
  --alias Qwen3.5-122B-A10B \
  --port 8096 \
  --host 127.0.0.1 \
  --gpu-layers -1 \
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
  --repeat-penalty 1.0 \
  "${SPLIT_ARGS[@]}"
