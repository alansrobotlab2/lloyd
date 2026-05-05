#!/usr/bin/env bash
set -euo pipefail

# Starts Mistral-Small-4-119B-2603 via vLLM on GPU 0
# NVFP4 quantization + Eagle speculative decoding
# Best practices: https://huggingface.co/unsloth/Mistral-Small-4-119B-2603-GGUF
# Official NVFP4 model card: https://huggingface.co/mistralai/Mistral-Small-4-119B-2603-NVFP4
# OpenAI-compatible API on port 8096 (replaces 122B slot, served as Qwen3.5-122B-A10B for OpenClaw compat)
#
# Downloads required before first run:
#   huggingface-cli download mistralai/Mistral-Small-4-119B-2603-NVFP4 \
#     --local-dir llm/models/Mistral-Small-4-119B-2603-NVFP4
#   huggingface-cli download mistralai/Mistral-Small-4-119B-2603-eagle \
#     --local-dir llm/models/Mistral-Small-4-119B-2603-eagle

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

VLLM_VENV="$PROJECT_DIR/.venvs/vllm-mistral2"
MODEL_DIR="$PROJECT_DIR/llm/models/Mistral-Small-4-119B-2603-NVFP4"
EAGLE_DIR="$PROJECT_DIR/llm/models/Mistral-Small-4-119B-2603-eagle"

if [[ ! -x "$VLLM_VENV/bin/python" ]]; then
  echo "vLLM experimental venv not found at $VLLM_VENV"
  exit 1
fi

if [[ ! -f "$MODEL_DIR/params.json" ]]; then
  echo "Model not found at $MODEL_DIR"
  echo "Run: hf download mistralai/Mistral-Small-4-119B-2603-NVFP4 --local-dir $MODEL_DIR"
  exit 1
fi

if [[ ! -f "$EAGLE_DIR/params.json" ]]; then
  echo "Eagle draft model not found at $EAGLE_DIR"
  echo "Run: hf download mistralai/Mistral-Small-4-119B-2603-eagle --local-dir $EAGLE_DIR"
  exit 1
fi

export PATH="$VLLM_VENV/bin:/opt/cuda/bin:/usr/bin:/usr/sbin:$PATH"
export LD_LIBRARY_PATH="/run/host/usr/lib:/opt/cuda/targets/x86_64-linux/lib:/opt/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
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
  --tokenizer-mode mistral \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --max-num-seqs 2 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.93 \
  --scheduling-policy priority \
  --no-enable-flashinfer-autotune \
  --enable-prefix-caching \
  --no-enable-log-requests \
  --enable-auto-tool-choice \
  --tool-call-parser mistral \
  --reasoning-parser mistral \
  --speculative-config "{\"method\": \"eagle\", \"model\": \"$EAGLE_DIR\", \"num_speculative_tokens\": 3}"
