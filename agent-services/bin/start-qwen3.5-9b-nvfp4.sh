#!/usr/bin/env bash
set -euo pipefail

# Starts AxionML/Qwen3.5-9B-NVFP4 via vLLM.
# OpenAI-compatible API on port 8091 (secondary LLM slot, GPU 0 / RTX 5090).
#
# Model: Qwen3_5ForConditionalGeneration. Hybrid linear_attention + full_attention
# (every 4th layer is full-attn — same family as Qwen3.6-35B-A3B-NVFP4 but dense
# and ~9B params). hidden_size=4096, intermediate_size=12288, head_dim=256.
# Multimodal vision tower present (visual* excluded from quant); disabled at
# server level via --limit-mm-per-prompt for text-only hot path.
# MTP head present (mtp.layers.0 excluded from quant) → speculative decode on.
# kv_cache_quant_algo not declared in checkpoint, so KV stays at FP8 by config.
#
# Stack: vllm-experimental venv. qwen3_xml/qwen3 parsers (same as Qwen3.6-35B).
#
# Usage:
#   ./start-qwen3.5-9b-nvfp4.sh                 # GPU 0, port 8091
#   PORT=8100 GPU=1 ./start-qwen3.5-9b-nvfp4.sh # override
#
# Untuned baseline — flags mirror start-35b-nvfp4.sh minus 35B-specific tunes.
# Bench MTP K and presence_penalty before declaring config final.

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

VLLM_VENV="$HOME/lloyd/.venvs/vllm-experimental"
MODEL_DIR="$PROJECT_DIR/llm/models/AxionML-Qwen3.5-9B-NVFP4"
PORT="${PORT:-8091}"
GPU="${GPU:-0}"

if [[ ! -x "$VLLM_VENV/bin/python" ]]; then
  echo "vLLM experimental venv not found at $VLLM_VENV"
  echo "Run: bash setup/setup-vllm-experimental.sh"
  exit 1
fi

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "Model not found at $MODEL_DIR"
  echo "Download: hf download AxionML/Qwen3.5-9B-NVFP4 --local-dir $MODEL_DIR"
  exit 1
fi

export PATH="$VLLM_VENV/bin:/opt/cuda/bin:/usr/bin:/usr/sbin:$PATH"
export LD_LIBRARY_PATH="/run/host/usr/lib:/opt/cuda/targets/x86_64-linux/lib:/opt/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_HOME="/opt/cuda"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"
export VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=1073741824
export VLLM_ENABLE_CUDAGRAPH_GC=1
export VLLM_USE_FLASHINFER_SAMPLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# nvcc 13.2 can't build flashinfer kernels against system gcc-16 libstdc++
# (__builtin_is_virtual_base_of unknown). Pin to gcc-15. Drop when CUDA
# bumps past 13.2 with gcc 16 support.
export NVCC_CCBIN=/usr/bin/g++-15

echo "==> AxionML/Qwen3.5-9B-NVFP4"
echo "    Venv:   $VLLM_VENV"
echo "    Model:  $MODEL_DIR"
echo "    Port:   $PORT"
echo "    GPU:    $GPU"
echo ""

exec "$VLLM_VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name Qwen3.5-9B-NVFP4 secondary \
  --port "$PORT" \
  --host 127.0.0.1 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --max-num-seqs 8 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.85 \
  --scheduling-policy priority \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend FLASHINFER \
  --enable-prefix-caching \
  --no-enable-log-requests \
  --enable-flashinfer-autotune \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --no-async-scheduling \
  --performance-mode interactivity \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 4}' \
  --override-generation-config '{"presence_penalty": 1.5}' \
  --limit-mm-per-prompt '{"image": 0, "video": 0, "audio": 0}'
