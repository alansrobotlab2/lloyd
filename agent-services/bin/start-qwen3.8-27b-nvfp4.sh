#!/usr/bin/env bash
set -euo pipefail

# Starts unsloth/Qwen3.8-27B-NVFP4 via vLLM on the RTX PRO 6000 Blackwell (96GB).
# OpenAI-compatible API on port 8096 — the shared primary slot (Laguna S 2.1,
# 35B-nvfp4, 27B-nvfp4-*, ...). Only one of those runs at a time.
#
# Setup / download: bash setup/setup-qwen3.8-27b-nvfp4.sh
#
# MODEL
#   Qwen3.8-27B, dense, native vision-language, 64 layers in the Qwen3.5 hybrid
#   layout — 16 x (3 x GatedDeltaNet+FFN -> 1 x GatedAttention+FFN), so only
#   16 layers hold a real KV cache. 262,144 native context. Gated attention is
#   24 Q heads / 4 KV heads at head_dim 256 with 0.25 partial rotary; DeltaNet
#   is 48 V heads / 16 QK heads at head_dim 128.
#
# QUANTIZATION — compressed-tensors, NOT ModelOpt
#   quant_method=compressed-tensors, format=mixed-precision:
#     FP8   : self_attn q/k/v/o, linear_attn in_proj_*/out_proj, lm_head,
#             and the last 8 layers' (56-63) mlp gate/up/down
#     NVFP4 : every other mlp gate/up/down
#     BF16  : the entire vision tower (quant_config ignore list)
#   Do NOT pass --quantization modelopt (as the sakamakismile script does) —
#   vLLM autodetects compressed-tensors from config.json and honours the
#   per-config-group `format` keys. Forcing modelopt misreads the checkpoint.
#
#   The model also ships a static per-tensor FP8 kv_cache_scheme (16 k_scale +
#   16 v_scale, one pair per full-attention layer), so --kv-cache-dtype fp8_e4m3
#   below runs on calibrated scales rather than dynamic ones.
#
# MTP / SPECULATIVE DECODE — verified present, unlike the 3.6 re-quant
#   The Qwen3.6-27B unsloth quant declared text_config.mtp_num_hidden_layers=1
#   but shipped zero mtp.* tensors, so --speculative-config wedged it at load
#   (see start-27b-nvfp4-unsloth.sh). This build ships model_mtp.safetensors:
#   15 BF16 tensors (mtp.fc, mtp.layers.0.*, mtp.norm, mtp.pre_fc_norm_*),
#   ~810 MiB. vLLM registers it as Qwen3_5MTP, which implements
#   SupportsMultiModal, so MTP and the vision tower coexist.
#
#   If the engine still fails on the speculative path: drop the
#   --speculative-config line. Everything else runs unchanged, just slower.
#
# SAMPLING — deliberately NOT inheriting presence_penalty 1.5
#   The 3.5/3.6 scripts pass --override-generation-config presence_penalty=1.5.
#   That is Qwen's *non-thinking* recommendation. For Qwen3.8 in thinking mode
#   (the default here) Qwen recommends presence_penalty=0.0, temperature=1.0,
#   top_p=0.95, top_k=20 — which is exactly what the shipped
#   generation_config.json already sets, so no override is needed. Only add
#   presence_penalty back (0-2) if endless repetition shows up in practice.
#
# THINKING CONTROL
#   The chat template defaults to enable_thinking=true with
#   reasoning_effort='xhigh' (also 'medium' and 'low'; 'high' is aliased to
#   'xhigh'), and keeps prior turns' reasoning via preserve_thinking. xhigh on
#   every agentic tool turn is expensive — to shift the server-side default add:
#     --default-chat-template-kwargs '{"reasoning_effort": "medium"}'
#   Per-request kwargs still win over that. The template also honours the
#   `developer` role, which Unsloth patched in for agentic harnesses.
#
# MULTIMODAL
#   Vision + video are supported (image_token_id=248056, video_token_id=248057,
#   MRoPE with mrope_section [11,11,10]). Served text-only here via
#   --limit-mm-per-prompt, matching the other Qwen scripts; drop those flags to
#   accept image/video input. To skip loading the vision tower entirely and
#   reclaim its BF16 weights, use --language-model-only instead.
#
# TOOL CALLING
#   --tool-call-parser qwen3_xml is the last-known-good for this family;
#   qwen3_coder wedges the engine on the stop-token path. Do not change it
#   without checking the harness's XML tool-call recovery path.
#
# VRAM
#   Weights ~21.8 GiB (main) + ~0.8 GiB (MTP) + BF16 vision tower. At
#   gpu-memory-utilization 0.95 on the 96 GiB card that leaves ~65 GiB for KV
#   cache — ample at 262K context given only 16 of 64 layers are full-attention.

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Dedicated bleeding-edge venv — this model's alone, so it can track vLLM nightly
# without re-qualifying Laguna S 2.1 or the Qwen3.5/3.6 fleet.
#   build:    bash setup/setup-vllm-qwen3.8.sh
#   versions: setup/vllm-qwen3.8.versions.txt
# Fall back without editing this file:
#   VLLM_VENV=~/lloyd/.venvs/vllm-laguna bash bin/start-qwen3.8-27b-nvfp4.sh
VLLM_VENV="${VLLM_VENV:-$HOME/lloyd/.venvs/vllm-qwen3.8}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/llm/models/unsloth-Qwen3.8-27B-NVFP4}"

if [[ ! -x "$VLLM_VENV/bin/python" ]]; then
  echo "vLLM venv not found at $VLLM_VENV"
  echo "Build it: bash $PROJECT_DIR/setup/setup-vllm-qwen3.8.sh"
  exit 1
fi

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "Model not found at $MODEL_DIR"
  echo "Download: bash $PROJECT_DIR/setup/setup-qwen3.8-27b-nvfp4.sh"
  exit 1
fi

# Guard the MTP head: if a future re-download drops model_mtp.safetensors, fall
# back to plain decode instead of wedging the engine on a missing draft model.
#
# method is "mtp", not "qwen3_5_mtp": as of 0.26.1 the family-specific names are
# deprecated and auto-mapped to the generic one, which logs a WARNING on every
# boot. All three vllm venvs here (0.23.1, 0.25.1, 0.26.1) accept "mtp" directly,
# so this is safe across the VLLM_VENV fallbacks above.
#
# Expect a boot WARNING that num_speculative_tokens > 1 re-runs the same MTP layer
# and "may result in lower acceptance rate" — this head is a single layer
# (mtp_num_hidden_layers=1). Measured acceptance at 3 tokens is ~70% (67/96
# accepted, per-position 84/69/…%), so the tradeoff is worth it. If acceptance
# drops on real agentic traffic, lower to 2 and re-check /metrics:
#   curl -s localhost:8096/metrics | grep vllm:spec_decode
SPEC_ARGS=()
if [[ -f "$MODEL_DIR/model_mtp.safetensors" ]]; then
  SPEC_ARGS=(--speculative-config '{"method": "mtp", "num_speculative_tokens": 3}')
else
  echo "WARNING: model_mtp.safetensors missing — starting WITHOUT speculative decode"
fi

export PATH="$VLLM_VENV/bin:/opt/cuda/bin:/usr/bin:/usr/sbin:$PATH"
export LD_LIBRARY_PATH="/usr/lib:/opt/cuda/targets/x86_64-linux/lib:/opt/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_HOME="/opt/cuda"
# nvcc 13.2 cannot build against the gcc-16 libstdc++; FlashInfer JIT needs g++-15.
export NVCC_CCBIN=/usr/bin/g++-15
# 3 GPUs on PCI bus order: 0 = RTX 3090 (sm_86), 1 = RTX PRO 6000 Blackwell
# (sm_120a, this server), 2 = RTX 3090. CUDA_DEVICE_ORDER is mandatory here —
# without it the runtime reorders by capability and this lands on a 3090.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=1073741824
export VLLM_ENABLE_CUDAGRAPH_GC=1
export VLLM_USE_FLASHINFER_SAMPLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MODULE_LOADING=LAZY

exec "$VLLM_VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name Qwen3.8-27B-nvfp4-unsloth primary \
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
  "${SPEC_ARGS[@]}" \
  --limit-mm-per-prompt '{"image": 0, "video": 0, "audio": 0}'
