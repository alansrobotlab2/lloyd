#!/usr/bin/env bash
set -euo pipefail

# Starts cosmicproc/gemma-4-E4B-it-NVFP4 via vLLM.
# OpenAI-compatible API on port 8091 (secondary LLM slot, GPU 0 / RTX 5090).
#
# Model: Gemma4ForConditionalGeneration, DENSE 4B variant (enable_moe_block=false).
# 42 layers, hidden_size=2560, intermediate_size=10240, head_dim=256, global_head_dim=512,
# hybrid sliding/full attention, num_kv_shared_layers=18 (KV-shared across layer
# pairs — small KV footprint for the size). Multimodal: image, video, AND audio
# towers all present; disabled at the server level via --limit-mm-per-prompt to
# keep the hot path text-only. Drop the limit if you want vision/audio input.
#
# Stack: vllm-experimental venv. tool-call-parser=gemma4 to match the
# checkpoint's chat template. Dense — no --moe-backend flag needed.
#
# Usage:
#   ./start-gemma-4-e4b-nvfp4.sh                 # GPU 0, port 8091
#   PORT=8100 GPU=1 ./start-gemma-4-e4b-nvfp4.sh # override
#
# Untuned baseline. ~5GB at NVFP4 weights — fits on any 8+GB card with room
# for full context. Lloyd's smallest tool-capable model.

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

VLLM_VENV="$HOME/lloyd/.venvs/vllm-experimental"
MODEL_DIR="$PROJECT_DIR/llm/models/cosmicproc-gemma-4-E4B-it-NVFP4"
PORT="${PORT:-8091}"
GPU="${GPU:-0}"

if [[ ! -x "$VLLM_VENV/bin/python" ]]; then
  echo "vLLM experimental venv not found at $VLLM_VENV"
  echo "Run: bash setup/setup-vllm-experimental.sh"
  exit 1
fi

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "Model not found at $MODEL_DIR"
  echo "Download: hf download cosmicproc/gemma-4-E4B-it-NVFP4 --local-dir $MODEL_DIR"
  exit 1
fi

export PATH="$VLLM_VENV/bin:/opt/cuda/bin:/usr/bin:/usr/sbin:$PATH"
export LD_LIBRARY_PATH="/usr/lib:/opt/cuda/targets/x86_64-linux/lib:/opt/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_HOME="/opt/cuda"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"
export VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=1073741824
export VLLM_ENABLE_CUDAGRAPH_GC=1
export VLLM_USE_FLASHINFER_SAMPLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Pin nvcc's host compiler to gcc-15. The system gcc is 16.1.1, whose
# libstdc++ uses __builtin_is_virtual_base_of — unknown to nvcc 13.2. gcc-15
# ships its own libstdc++ at /usr/lib/gcc/.../15.2.1/include/c++/ which nvcc
# happily compiles against. Drop when CUDA bumps past 13.2 with GCC 16
# support (or install via paru -S gcc15 if missing).
export NVCC_CCBIN=/usr/bin/g++-15
# Marlin NVFP4 GEMM backend: same safe choice as the 26B Gemma variant.
# Drop this and rebench if CUTLASS proves stable on this dense checkpoint.
export VLLM_NVFP4_GEMM_BACKEND=marlin

echo "==> cosmicproc/gemma-4-E4B-it-NVFP4"
echo "    Venv:   $VLLM_VENV"
echo "    Model:  $MODEL_DIR"
echo "    Port:   $PORT"
echo "    GPU:    $GPU"
echo ""

exec "$VLLM_VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name gemma-4-E4B-it-NVFP4 secondary \
  --port "$PORT" \
  --host 127.0.0.1 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --max-model-len 131072 \
  --max-num-seqs 8 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.55 \
  --scheduling-policy priority \
  --quantization modelopt \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend TRITON_ATTN \
  --enable-prefix-caching \
  --no-enable-log-requests \
  --enable-flashinfer-autotune \
  --performance-mode interactivity \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --limit-mm-per-prompt '{"image": 0, "video": 0, "audio": 0}'
