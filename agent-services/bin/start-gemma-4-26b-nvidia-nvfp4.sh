#!/usr/bin/env bash
set -euo pipefail

# Starts nvidia/Gemma-4-26B-A4B-NVFP4 (NVIDIA's official NVFP4 quant) via vLLM.
# OpenAI-compatible API on port 8091 (secondary LLM slot, GPU 0 / RTX 5090).
#
# Counterpart to start-gemma-4-26b-nvfp4.sh, which uses the bg-digitalservices
# checkpoint with a manual gemma4_patched.py loader workaround. This script
# uses the upstream NVIDIA quant, which should not need the patched loader.
#
# Model: Gemma4ForConditionalGeneration (MoE, 128 experts × top-8). text_config
# head_dim=256, global_head_dim=512, hybrid sliding/full attention. Multimodal
# (image/audio/video) heads present but disabled via --limit-mm-per-prompt to
# keep the hot path text-only.
#
# Stack: vllm-experimental venv. tool-call-parser=gemma4 to match the
# checkpoint's chat template.
#
# Usage:
#   ./start-gemma-4-26b-nvidia-nvfp4.sh                 # GPU 0, port 8091
#   PORT=8100 GPU=1 ./start-gemma-4-26b-nvidia-nvfp4.sh # override
#
# This is an untuned baseline. Validate with a smoke test before benching.

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

VLLM_VENV="$HOME/lloyd/.venvs/vllm-experimental"
MODEL_DIR="$PROJECT_DIR/llm/models/nvidia-Gemma-4-26B-A4B-NVFP4"
PORT="${PORT:-8091}"
GPU="${GPU:-0}"

if [[ ! -x "$VLLM_VENV/bin/python" ]]; then
  echo "vLLM experimental venv not found at $VLLM_VENV"
  echo "Run: bash setup/setup-vllm-experimental.sh"
  exit 1
fi

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "Model not found at $MODEL_DIR"
  echo "Download: hf download nvidia/Gemma-4-26B-A4B-NVFP4 --local-dir $MODEL_DIR"
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
# nvcc 13.2 can't build flashinfer kernels against system gcc-16 libstdc++
# (__builtin_is_virtual_base_of unknown). Pin to gcc-15. Drop when CUDA
# bumps past 13.2 with gcc 16 support.
export NVCC_CCBIN=/usr/bin/g++-15
# Marlin NVFP4 GEMM backend is the safe default on SM120 for this family
# (matches the bg-digitalservices script's choice). Drop this and rebench
# if NVIDIA's quant is happy with CUTLASS on your GPU.
export VLLM_NVFP4_GEMM_BACKEND=marlin

echo "==> nvidia/Gemma-4-26B-A4B-NVFP4"
echo "    Venv:   $VLLM_VENV"
echo "    Model:  $MODEL_DIR"
echo "    Port:   $PORT"
echo "    GPU:    $GPU"
echo ""

exec "$VLLM_VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name gemma-4-26B-A4B-NVFP4 secondary \
  --port "$PORT" \
  --host 127.0.0.1 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --max-model-len 131072 \
  --max-num-seqs 4 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.85 \
  --scheduling-policy priority \
  --quantization modelopt \
  --moe-backend marlin \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend TRITON_ATTN \
  --enable-prefix-caching \
  --no-enable-log-requests \
  --enable-flashinfer-autotune \
  --performance-mode interactivity \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --limit-mm-per-prompt '{"image": 0, "video": 0, "audio": 0}'
