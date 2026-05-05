#!/usr/bin/env bash
set -euo pipefail

# Starts MiniMax-M2.7-REAP-NVFP4 via vLLM on the RTX PRO 6000 Blackwell.
# OpenAI-compatible API on port 8096 (reuses 122B port — one of them runs at a time).
#
# Model: MiniMaxM2ForCausalLM, REAPed from 256 → 154 experts (8 active/token),
# 62 layers, hidden 3072, quantized to NVFP4 (modelopt) with FP8 KV cache.
# On-disk size ~79 GiB, produced by ~/Projects/Model-Optimizer.
#
# Stack: vllm-experimental venv (vLLM 0.19+ cu130 nightly, FlashInfer 0.6.7,
# PyTorch 2.10+cu130, sm_120). Uses the same NVFP4-on-SM120 config as the
# Qwen3.5-122B-A10B-NVFP4 script — see start-llm-122b.sh for the tuning notes
# that also apply here (FLASHINFER_CUTLASS MoE backend, async scheduling,
# autotune re-enabled, gpu-memory-utilization 0.95).
#
# This is the initial bringup — settings are conservative copies of the 122B
# config. Tune max-num-batched-tokens, MTP depth, and max-model-len once it's
# serving.

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

VLLM_VENV="$HOME/lloyd/.venvs/vllm-experimental"
MODEL_DIR="/home/alansrobotlab/Projects/Model-Optimizer/workspaces/minimax-m2.7-reap-nvfp4/output"

if [[ ! -x "$VLLM_VENV/bin/python" ]]; then
  echo "vLLM experimental venv not found at $VLLM_VENV"
  exit 1
fi

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "Model not found at $MODEL_DIR"
  exit 1
fi

export PATH="$VLLM_VENV/bin:/opt/cuda/bin:/usr/bin:/usr/sbin:$PATH"
export LD_LIBRARY_PATH="/run/host/usr/lib:/opt/cuda/targets/x86_64-linux/lib:/opt/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_HOME="/opt/cuda"
# PCI bus order: index 1 = RTX PRO 6000 Blackwell (same as 122B).
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=1073741824
export VLLM_ENABLE_CUDAGRAPH_GC=1
export VLLM_USE_FLASHINFER_SAMPLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec "$VLLM_VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name MiniMax-M2.7-REAP-NVFP4 \
  --port 8096 \
  --host 127.0.0.1 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --max-model-len 65536 \
  --max-num-seqs 4 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.95 \
  --scheduling-policy priority \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend FLASHINFER \
  --enable-prefix-caching \
  --no-enable-log-requests \
  --enable-flashinfer-autotune \
  --async-scheduling \
  --enable-auto-tool-choice \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2 \
  --performance-mode interactivity
  # NOTE: MTP speculative decoding is NOT enabled — vLLM 0.19.1 does not
  # register a minimax_m2 MTP draft model type (only deepseek/qwen3_5/glm4_moe/
  # etc. are in MTPModelTypes as of this build). The model's num_mtp_modules=3
  # weights are loaded but unused. Revisit when upstream adds it.
