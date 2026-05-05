#!/usr/bin/env bash
set -euo pipefail

# Starts unsloth/Qwen3.6-27B-NVFP4 via vLLM on GPU 1 (RTX PRO 6000 Blackwell, 96GB).
# OpenAI-compatible API on port 8096 (same primary slot as 35B-nvfp4 / 35B-fp8 /
# 27B-nvfp4 mmangkad — only one runs at a time).
#
# Sibling of start-27b-nvfp4.sh (mmangkad re-quant). Same architecture
# (Qwen3_5ForConditionalGeneration, model_type=qwen3_5): dense 27B, hybrid
# DeltaNet + full-attention (64 layers, full_attention_interval=4 → 16 full /
# 48 linear), MTP head present, NVFP4 (ModelOpt) on linear ops, vision tower
# left in BF16 (per quant_config ignore list). Unsloth's variant ships the
# same single-shard model.safetensors (~25.5 GiB) plus their nvfp4_recipe.json
# / recipe.yaml.
#
# Multimodal: vision+video stack (image_token_id=248056, video_token_id=248057)
# with MRoPE. Served here as text-only via --limit-mm-per-prompt. Drop those
# flags if image/video input is needed.
#
# Stack: vllm-experimental venv (vLLM 0.19+ nightly, FlashInfer 0.6.8,
# PyTorch 2.10+cu130). Same SM120 path as the 35B-NVFP4 / 27B-NVFP4-mmangkad.
#
# Tuning inherited from start-27b-nvfp4.sh / start-35b-nvfp4.sh:
#   - --max-num-batched-tokens 8192: DeltaNet block-size constraint requires >4288.
#   - --speculative-config MTP num_speculative_tokens=4: 35B sweep winner; re-sweep
#     here once the unsloth re-quant is stable, dense 27B may prefer a different depth.
#   - --tool-call-parser qwen3_xml: empirical last-known-good under MTP. qwen3_coder
#     wedges the engine on the stop-token path with MTP enabled.
#   - --override-generation-config presence_penalty=1.5: Unsloth anti-repetition
#     default for the Qwen3.5 thinking family.
#   - --performance-mode interactivity: +10-15% single-user decode, sub-second TTFT.
#
# Weights ~25.5 GiB on disk (single model.safetensors). On the 96 GiB card at
# gpu-util 0.95 that leaves ~60+ GiB for KV cache — plenty for 262K context
# since only 16/64 layers are full-attention.

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

VLLM_VENV="$HOME/lloyd/.venvs/vllm-experimental"
MODEL_DIR="$PROJECT_DIR/llm/models/unsloth-Qwen3.6-27B-NVFP4"

if [[ ! -x "$VLLM_VENV/bin/python" ]]; then
  echo "vLLM experimental venv not found at $VLLM_VENV"
  echo "Run: bash setup/setup-vllm-experimental.sh"
  exit 1
fi

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "Model not found at $MODEL_DIR"
  echo "Download: snapshot_download('unsloth/Qwen3.6-27B-NVFP4', local_dir=$MODEL_DIR)"
  exit 1
fi

export PATH="$VLLM_VENV/bin:/opt/cuda/bin:/usr/bin:/usr/sbin:$PATH"
export LD_LIBRARY_PATH="/run/host/usr/lib:/opt/cuda/targets/x86_64-linux/lib:/opt/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_HOME="/opt/cuda"
# PCI bus order: index 0 = RTX 5090 (secondary, port 8091),
#                index 1 = RTX PRO 6000 Blackwell (this server, port 8096).
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=1073741824
export VLLM_ENABLE_CUDAGRAPH_GC=1
export VLLM_USE_FLASHINFER_SAMPLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec "$VLLM_VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name Qwen3.6-27B-nvfp4-unsloth primary \
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
  --override-generation-config '{"presence_penalty": 1.5}' \
  --limit-mm-per-prompt '{"image": 0, "video": 0, "audio": 0}'
