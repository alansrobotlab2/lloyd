#!/usr/bin/env bash
set -euo pipefail

# Starts MiniMax-M2.7 via llama.cpp on GPU 1 (RTX PRO 6000, 96GB)
# Unsloth IQ3_S GGUF quantization (~84GB) + ngram speculative decoding
# 229B total params / 10B active (256 experts, 8 active per token)
# OpenAI-compatible API on port 8098

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Use Unsloth pre-compiled b8746 (CUDA 13.0) to avoid CUDA 13.2 gibberish bug
# See: https://huggingface.co/unsloth/MiniMax-M2.7-GGUF/discussions/2
LLAMA_SERVER="$PROJECT_DIR/llm/llama-b8746-unsloth/llama-server"
MODEL="$PROJECT_DIR/llm/models/MiniMax-M2.7-UD-IQ3_S/UD-IQ3_S/MiniMax-M2.7-UD-IQ3_S-00001-of-00003.gguf"

if [[ ! -x "$LLAMA_SERVER" ]]; then
  echo "llama-server not found at $LLAMA_SERVER"
  echo "Run: bash setup/setup-minimax-m2.7.sh"
  exit 1
fi

if [[ ! -f "$MODEL" ]]; then
  echo "Model not found at $MODEL"
  echo "Run: bash setup/setup-minimax-m2.7.sh"
  exit 1
fi

export LD_LIBRARY_PATH="$PROJECT_DIR/llm/llama-b8746-unsloth:/run/host/usr/lib:/opt/cuda/lib64:/opt/cuda/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
"$LLAMA_SERVER" \
  --model "$MODEL" \
  --alias MiniMax-M2.7 \
  --port 8098 \
  --host 127.0.0.1 \
  --gpu-layers -1 \
  --ctx-size 196608 \
  --batch-size 8192 \
  --ubatch-size 2048 \
  --flash-attn on \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --temp 1.0 \
  --top-p 0.95 \
  --top-k 40 \
  --min-p 0.01 \
  --repeat-penalty 1.05 \
  --threads 4 \
  --parallel 1 \
  --spec-type ngram-mod \
  --draft-min 3 \
  --draft-max 16
