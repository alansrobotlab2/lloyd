#!/usr/bin/env bash
set -euo pipefail

# Starts Qwen/Qwen3.6-35B-A3B-FP8 via vLLM on GPU 1 (RTX PRO 6000 Blackwell, 96GB).
# OpenAI-compatible API on port 8096 (same slot as NVFP4 — one of them runs at a time).
#
# Purpose: comparison build vs RedHatAI-Qwen3.6-35B-A3B-NVFP4. The NVFP4 variant
# wedges the engine under concurrency on Blackwell per vLLM issue #40350
# (NVFP4 × MTP × async × FlashInfer deadlock). FP8 is the reporter's confirmed
# workaround — same model, different quantization path, no wedge.
#
# Model: Qwen3.6-35B-A3B, FP8 (E4M3) block-scaled quantization, block size 128.
# Weights ~18-20 GiB at FP8 (vs ~9 GiB at NVFP4); leaves ~76 GiB for KV cache.
# Thinking enabled; per-request control via enable_thinking in the chat template.
#
# Stack: vllm-experimental venv (vLLM 0.19+ nightly, FlashInfer 0.6.8, PyTorch 2.10+cu130, sm_120).
#
# Tuning notes:
#   - MTP: enabled here to match NVFP4 baseline for apples-to-apples throughput
#     comparison. If this wedges like the NVFP4 build, remove --speculative-config
#     (FP8 hang would disprove the NVFP4-specific theory in #40350).
#   - Attention backend: FLASHINFER — FP8 is a well-supported path here, unlike
#     NVFP4's SM120 kernel gaps.
#   - kv-cache-dtype fp8_e4m3: unchanged.

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

VLLM_VENV="$HOME/lloyd/.venvs/vllm-experimental"
MODEL_DIR="$PROJECT_DIR/llm/models/Qwen-Qwen3.6-35B-A3B-FP8"

if [[ ! -x "$VLLM_VENV/bin/python" ]]; then
  echo "vLLM experimental venv not found at $VLLM_VENV"
  echo "Run: bash setup/setup-vllm-experimental.sh"
  exit 1
fi

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "Model not found at $MODEL_DIR"
  echo "Download: snapshot_download('Qwen/Qwen3.6-35B-A3B-FP8', local_dir=$MODEL_DIR)"
  exit 1
fi

export PATH="$VLLM_VENV/bin:/opt/cuda/bin:/usr/bin:/usr/sbin:$PATH"
export LD_LIBRARY_PATH="/run/host/usr/lib:/opt/cuda/targets/x86_64-linux/lib:/opt/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_HOME="/opt/cuda"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=1073741824
export VLLM_ENABLE_CUDAGRAPH_GC=1
export VLLM_USE_FLASHINFER_SAMPLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec "$VLLM_VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name Qwen3.6-35B-A3B-fp8 primary \
  --port 8096 \
  --host 127.0.0.1 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --max-num-seqs 8 \
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
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --performance-mode interactivity \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 4}' \
  --override-generation-config '{"presence_penalty": 1.5}'
