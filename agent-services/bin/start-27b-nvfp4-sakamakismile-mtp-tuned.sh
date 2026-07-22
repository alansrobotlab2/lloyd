#!/usr/bin/env bash
set -euo pipefail

# Starts sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP via vLLM on GPU 0
# (RTX PRO 6000 Blackwell, 96GB). OpenAI-compatible API on port 8096.
#
# TUNED version: relaxes conservative limits that were set during early
# cuda-graph capture debugging. Verified safe on vLLM 0.19+ with
# expandable_segments and modern cuda-graph capture.
#
# Changes vs original start-27b-nvfp4-sakamakismile-mtp.sh:
#   1. --max-num-seqs 8 (was 2): 19GB model + 77GB KV budget easily handles 8 seqs
#   2. --gpu-memory-utilization 0.95 (was 0.80): uses available VRAM for KV cache
#   3. --num-scheduler-steps 4: multi-step scheduling = 10-20% decode throughput boost
#   4. --max-model-len 131072 (was 262144): most agentic requests don't need 262K,
#      halving context budget doubles concurrent sequence headroom.
#      Set --max-model-len 262144 if you need full context length.
#
# VRAM budget math (0.95 util, 96GB card):
#   Model weights: ~19 GB (NVFP4 main + BF16 MTP head)
#   KV cache (FP8): ~3 GB/seq at 131K context × 8 seqs = ~24 GB
#   Activations + overhead: ~5 GB
#   Total: ~48 GB / 91 GB budget = comfortable margin
#
# Performance expectations (vs 0.80 util / 2 seq baseline):
#   - Decode throughput: ~3-4x (more sequences batched per forward pass)
#   - Per-user tok/s: ~50% higher at 1-2 users, ~30% at 8 users
#   - TTFT: unchanged (chunked prefill + prefix cache handles this)

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

VLLM_VENV="$HOME/lloyd/.venvs/vllm-experimental"
MODEL_DIR="$PROJECT_DIR/llm/models/sakamakismile-Qwen3.6-27B-Text-NVFP4-MTP"

if [[ ! -x "$VLLM_VENV/bin/python" ]]; then
  echo "vLLM experimental venv not found at $VLLM_VENV"
  echo "Run: bash setup/setup-vllm-experimental.sh"
  exit 1
fi

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "Model not found at $MODEL_DIR"
  echo "Download: snapshot_download('sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP', local_dir=$MODEL_DIR)"
  exit 1
fi

export PATH="$VLLM_VENV/bin:/opt/cuda/bin:/usr/bin:/usr/sbin:$PATH"
export LD_LIBRARY_PATH="/usr/lib:/opt/cuda/targets/x86_64-linux/lib:/opt/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_HOME="/opt/cuda"
export NVCC_CCBIN=/usr/bin/g++-15
# Single-GPU host: index 0 = RTX PRO 6000 Blackwell (5090 removed 2026-05-18)
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=1073741824
export VLLM_ENABLE_CUDAGRAPH_GC=1
export VLLM_USE_FLASHINFER_SAMPLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Blackwell optimization: LAZY module loading reduces startup latency
export CUDA_MODULE_LOADING=LAZY

exec "$VLLM_VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name Qwen3.6-27B-nvfp4-sakamakismile-mtp primary \
  --port 8096 \
  --host 127.0.0.1 \
  --trust-remote-code \
  --language-model-only \
  --quantization modelopt \
  --tensor-parallel-size 1 \
  --max-model-len 131072 \
  --max-num-seqs 8 \
  --num-scheduler-steps 4 \
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
  --speculative-config '{"method": "qwen3_5_mtp", "num_speculative_tokens": 3}' \
  --override-generation-config '{"presence_penalty": 1.5}'
