#!/usr/bin/env bash
set -euo pipefail

# Qwen3.5-35B-A3B UD-IQ3_S + Qwen3.5-0.8B-UD-Q4_K_XL draft
# Speculative decoding experiment — GPU 2 (RTX 3090, 24GB)
# Draft model runs on same GPU; 0.8B is ~560MB so negligible VRAM cost

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

LLAMA_SERVER="$PROJECT_DIR/llm/llama.cpp/build/bin/llama-server"
MODEL="$PROJECT_DIR/llm/models/Qwen3.5-35B-A3B/Qwen3.5-35B-A3B-UD-IQ3_S.gguf"
DRAFT_MODEL="$PROJECT_DIR/llm/models/gguf/Qwen3.5-0.8B-UD-Q4_K_XL.gguf"

export LD_LIBRARY_PATH="/run/host/usr/lib:/opt/cuda/lib64:$PROJECT_DIR/llm/llama.cpp/build/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
"$LLAMA_SERVER" \
  --model "$MODEL" \
  --alias Qwen3.5-35B-A3B \
  --model-draft "$DRAFT_MODEL" \
  --port 8091 \
  --host 127.0.0.1 \
  --gpu-layers -1 \
  --gpu-layers-draft -1 \
  --ctx-size 262144 \
  --batch-size 512 \
  --flash-attn on \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --draft 16 \
  --draft-p-min 0.6 \
  --chat-template-kwargs '{"enable_thinking": true}' \
  --reasoning-format deepseek \
  --temp 0.7 \
  --top-p 0.8 \
  --top-k 20 \
  --min-p 0.0 \
  --presence-penalty 1.5 \
  --repeat-penalty 1.0
