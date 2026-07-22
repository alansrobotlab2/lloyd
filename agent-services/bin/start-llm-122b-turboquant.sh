#!/usr/bin/env bash
set -euo pipefail

# Starts Qwen3.5-122B-A10B via vLLM (vllm-turboquant build) with TurboQuant
# KV cache quantization.  Identical to start-llm-122b.sh except:
#   - vllm-turboquant venv (vLLM patched with TurboQuant round-trip)
#   - --kv-cache-dtype turboquant_4bit
#
# To revert: update agent-llm-122b.conf to point back to start-llm-122b.sh

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

VLLM_VENV="$PROJECT_DIR/.venvs/vllm-turboquant"
MODEL_DIR="$PROJECT_DIR/llm/models/Sehyo-Qwen3.5-122B-A10B-NVFP4"
TQ_BITS="${TQ_BITS:-4}"
KV_DTYPE="turboquant_${TQ_BITS}bit"

if [[ ! -x "$VLLM_VENV/bin/python" ]]; then
  echo "vllm-turboquant venv not found at $VLLM_VENV"
  exit 1
fi

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "Model not found at $MODEL_DIR"
  exit 1
fi

echo "==> Starting 122B with TurboQuant (${KV_DTYPE})"

export PATH="$VLLM_VENV/bin:/opt/cuda/bin:/usr/bin:/usr/sbin:$PATH"
export LD_LIBRARY_PATH="/usr/lib:/opt/cuda/targets/x86_64-linux/lib:/opt/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_HOME="/opt/cuda"
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
  --max-model-len 262144 \
  --max-num-seqs 4 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.96 \
  --scheduling-policy priority \
  --kv-cache-dtype "$KV_DTYPE" \
  --attention-backend FLASHINFER \
  --enable-prefix-caching \
  --no-enable-log-requests \
  --no-enable-flashinfer-autotune \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --performance-mode interactivity \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'
