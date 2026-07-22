#!/usr/bin/env bash
set -euo pipefail

# Starts Qwen3.6-35B-A3B via llama-server on GPU 0 (RTX 5090, 32GB)
# UD-Q3_K_XL quant (~17GB), 256K context, all layers GPU-offloaded, port 8091
#
# Hybrid DeltaNet+Attention arch: only 10/40 layers use KV cache -> small KV footprint
# Thinking enabled server-side; per-request control via OpenClaw thinking levels
# Non-thinking sampling: temp 0.7, top_p 0.8, top_k 20 (per Unsloth guide)
# NOTE: Qwen3.5-35B-A3B-UD-IQ4_NL (Unsloth Dynamic IQ4_NL) produces incoherent
# output — HTML/XML/tool_call fragments instead of natural language. NOT a
# hardware issue: reproduced on 4090 (SM89) and our Blackwell (SM120) both,
# with llama.cpp b8668+ and b8808. See ggml-org/llama.cpp#21495. The broken
# quantization is specific to this model + UD-IQ4_NL; UD-Q3_K_XL and standard
# Q4_K_S of the same model are coherent. Prefer Q4_K_S (better quality) or
# UD-Q3_K_XL (smaller VRAM footprint).

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

LLAMA_SERVER="$PROJECT_DIR/llm/llama.cpp/build/bin/llama-server"
MODEL_DIR="$PROJECT_DIR/llm/models"
MODEL="$MODEL_DIR/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf"

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

export LD_LIBRARY_PATH="/usr/lib:/opt/cuda/lib64:$PROJECT_DIR/llm/llama.cpp/build/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
"$LLAMA_SERVER" \
  --model "$MODEL" \
  --alias Qwen3.6-35B-A3B,secondary \
  --port 8091 \
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
  --repeat-penalty 1.0
