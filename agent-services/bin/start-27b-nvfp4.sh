#!/usr/bin/env bash
set -euo pipefail

# Starts mmangkad/Qwen3.6-27B-NVFP4 via vLLM on GPU 1 (RTX PRO 6000 Blackwell, 96GB).
# OpenAI-compatible API on port 8096 (same primary slot as 35B-nvfp4 / 35B-fp8 — one at a time).
#
# Model: despite the "3.6" name, config.json reports architectures=Qwen3_5ForConditionalGeneration
# and model_type=qwen3_5. Dense 27B (no MoE), but same hybrid DeltaNet + full-attention
# architecture as the 35B-A3B: 64 layers with full_attention_interval=4 (16 full-attn
# layers, 48 linear_attention/DeltaNet). MTP head present (mtp_num_hidden_layers=1) —
# speculative decoding enabled. NVFP4 (ModelOpt) quantization on linear ops, FP8 KV,
# vision tower left unquantized.
#
# Multimodal: vision+video stack (image_token_id=248056, video_token_id=248057) with MRoPE.
# Served here as text-only via --limit-mm-per-prompt to keep the hot path identical to
# the 35B primary. Remove those flags if image/video input is needed.
#
# Stack: vllm-experimental venv (vLLM 0.19+ nightly, FlashInfer 0.6.8, PyTorch 2.10+cu130).
# Model card recommends SGLang on B300; launched here under the same vLLM SM120 path
# as 35B-NVFP4, since the architecture is the same family.
#
# Inherited tuning from start-35b-nvfp4.sh (see that file for sweep data):
#   - --max-num-batched-tokens 8192: DeltaNet block-size constraint requires >4288
#     (same as 35B-A3B / 122B-A10B).
#   - --speculative-config MTP num_speculative_tokens=4: 35B sweep winner; re-sweep
#     here once stable since dense 27B may prefer a different depth.
#   - --tool-call-parser qwen3_xml: empirical last-known-good on the 35B under MTP.
#     qwen3_coder matches the chat_template on paper but wedges the engine in a race
#     with MTP on the stop-token path (grammar_matcher.cc:497 loop). Keep xml while
#     MTP is on; swap to qwen3_coder only if MTP is disabled.
#   - --override-generation-config presence_penalty=1.5: Unsloth anti-repetition
#     default for the Qwen3.5 thinking family. Adjust if language mixing shows up.
#   - --performance-mode interactivity: +10-15% single-user decode, sub-second TTFT.
#
# Weights ~31 GiB on disk (single model.safetensors). The NVFP4 core is ~13 GiB, but
# the DeltaNet linear_attn weights and the vision tower stay in BF16 (per the quant_config
# ignore list), which dominates the footprint. On the 96 GiB card at gpu-util 0.95 that
# still leaves ~55 GiB for KV cache — plenty for 262K context since only 16/64 layers
# are full-attention (full_attention_interval=4).

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

VLLM_VENV="$HOME/lloyd/.venvs/vllm-experimental"
MODEL_DIR="$PROJECT_DIR/llm/models/mmangkad-Qwen3.6-27B-NVFP4"

if [[ ! -x "$VLLM_VENV/bin/python" ]]; then
  echo "vLLM experimental venv not found at $VLLM_VENV"
  echo "Run: bash setup/setup-vllm-experimental.sh"
  exit 1
fi

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "Model not found at $MODEL_DIR"
  echo "Download: snapshot_download('mmangkad/Qwen3.6-27B-NVFP4', local_dir=$MODEL_DIR)"
  exit 1
fi

export PATH="$VLLM_VENV/bin:/opt/cuda/bin:/usr/bin:/usr/sbin:$PATH"
export LD_LIBRARY_PATH="/usr/lib:/opt/cuda/targets/x86_64-linux/lib:/opt/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
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
  --served-model-name Qwen3.6-27B-nvfp4 primary \
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
