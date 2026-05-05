#!/usr/bin/env bash
set -euo pipefail

# Starts NVIDIA-Nemotron-3-Super-120B-A12B via vLLM on GPU 0
# Mixed NVFP4/FP8 quantization (ModelOpt), KV cache FP8, Triton attention
# OpenAI-compatible API on port 8096 (replaces 122B slot, served as Qwen3.5-122B-A10B for compat)
#
# Reference: https://huggingface.co/unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4#quick-start-guide
# Setup:     bash setup/setup-vllm-nemotron.sh
#
# PERFORMANCE NOTES (2026-04-01):
#   - Achieved ~71 tok/s decode without autotuner vs 118 tok/s on Qwen 122B w/ MTP3
#   - --no-enable-flashinfer-autotune required to avoid crash:
#       flashinfer fp8_gemm_sm100 autotuner calls cuBLASLt which fails to init
#       (CUDA 13 system libs, but vllm wheel links libcublas.so.12 from Ollama)
#   - MTP (num_nextn_predict_layers: 1 in config) does not improve throughput meaningfully
#   - TRITON_ATTN required; FLASHINFER attention backend not supported on this arch
#   - Root cause of speed gap is likely the disabled autotuner missing optimal NVFP4 kernels
#   - TODO: investigate proper cuBLASLt 12 init or a fully native CUDA 13 build of flashinfer

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

VLLM_VENV="$PROJECT_DIR/.venvs/vllm-122b"
MODEL_DIR="$PROJECT_DIR/llm/models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
REASONING_PARSER="$PROJECT_DIR/llm/super_v3_reasoning_parser.py"

if [[ ! -x "$VLLM_VENV/bin/python" ]]; then
  echo "vLLM 122b venv not found at $VLLM_VENV"
  exit 1
fi

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "Model not found at $MODEL_DIR"
  exit 1
fi

if [[ ! -f "$REASONING_PARSER" ]]; then
  echo "Reasoning parser not found at $REASONING_PARSER"
  echo "Run: bash setup/setup-vllm-nemotron.sh"
  exit 1
fi

export PATH="$VLLM_VENV/bin:/opt/cuda/bin:/usr/bin:/usr/sbin:$PATH"
export LD_LIBRARY_PATH="/run/host/usr/lib:/usr/local/lib/ollama/cuda_v12:/opt/cuda/targets/x86_64-linux/lib:/opt/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_HOME="/opt/cuda"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=1073741824
export VLLM_ENABLE_CUDAGRAPH_GC=1
export VLLM_USE_FLASHINFER_SAMPLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec "$VLLM_VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name Qwen3.5-122B-A10B \
  --port 8096 \
  --host 127.0.0.1 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --dtype auto \
  --kv-cache-dtype fp8 \
  --attention-backend TRITON_ATTN \
  --async-scheduling \
  --max-model-len 262144 \
  --max-num-seqs 512 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.93 \
  --scheduling-policy priority \
  --enable-prefix-caching \
  --no-enable-log-requests \
  --no-enable-flashinfer-autotune \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser-plugin "$REASONING_PARSER" \
  --reasoning-parser super_v3 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 1}'
