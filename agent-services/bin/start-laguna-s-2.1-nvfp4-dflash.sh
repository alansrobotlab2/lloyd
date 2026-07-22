#!/usr/bin/env bash
set -euo pipefail

# Starts poolside/Laguna-S-2.1-NVFP4 via vLLM on GPU 1
# (RTX PRO 6000 Blackwell, 96GB). OpenAI-compatible API on port 8096
# (same primary slot as the 27B/35B nvfp4 builds — only one at a time).
#
# Model: Laguna S 2.1 — 117.6B MoE, 8.5B active/token, 256 routed experts +
# 1 shared, 48 layers (12 global + 36 SWA window-512, per-head softplus
# gating), 262K context (native-1M checkpoint, quant calibrated at 1M).
# NVFP4 weights ~72 GiB on disk; KV-cache FP8 scheme ships in the
# checkpoint's quantization_config (compressed-tensors nvfp4-pack-quantized).
#
# MTP: NONE. Verified 2026-07-21 against model.safetensors.index.json —
# 145,153 tensors, zero mtp.*/nextn/draft/eagle entries, and no MTP fields
# in config.json (base model likewise). Speculative decoding instead uses
# poolside's separate quantization-matched draft model
# Laguna-S-2.1-DFlash-NVFP4 (~2.2 GiB) with method=dflash. Model card
# reports 2.9-3.1 accepted tokens/step; decode without it is
# memory-bandwidth-bound on every engine poolside tried.
#
# Smoke test 2026-07-21 (this box, this script): ~104 tok/s decode on code
# generation, tool calls parse (poolside_v1), thinking on by default and
# returned in the API `reasoning` field (vLLM >=0.23 renamed
# reasoning_content -> reasoning). DFlash measured mean ~1.46 accepted
# draft tokens/step (+1 bonus ≈ 2.5/step); acceptance is front-loaded
# (pos0 ~62%, near-zero past pos 6) — if draft overhead ever matters,
# try num_speculative_tokens 7 (the BF16 card's rec) and re-bench.
#
# Stack: vllm-laguna venv (vLLM 0.25.1 + cu130 torch). The model REQUIRES
# vLLM >= 0.25.0 — the vllm-experimental venv (0.23.1rc1 nightly, live
# primary for the Qwen builds) has no laguna arch, no dflash method, and no
# poolside_v1 parsers. Do not point this script at vllm-experimental.
#
# Tuning (per the NVFP4 model card recipe, adapted from GB10 to SM120):
#   - No --quantization flag: auto-detected from quantization_config.
#     Do NOT pass --kv-cache-dtype either — the fp8 kv_cache_scheme comes
#     from the checkpoint.
#   - No --trust-remote-code: vLLM 0.25 has native laguna support.
#   - --speculative-config method=dflash, num_speculative_tokens=15:
#     per the NVFP4 card serve recipe (main-card BF16 recipe uses 7; the
#     NVFP4 card explicitly uses 15 with the NVFP4 draft).
#   - --max-num-seqs 8: card says <=32 is REQUIRED (DFlash crashes vLLM at
#     the default 256); 8 is plenty for the single-user primary slot and
#     keeps cuda-graph capture memory down (lesson from the sakamakismile
#     OOM-at-capture history).
#   - --gpu-memory-utilization 0.90 (2026-07-21): 0.85 failed at boot —
#     engine measured 8.27 GiB free for KV vs 9.6 GiB required for one
#     262144-token request (max estimable len 223344). 0.90 leaves ~13 GiB
#     KV, which boots and covers 262K.
#   - --override-generation-config temperature=0.7 top_p=0.95: card says
#     keep this — clients that send no sampling params degrade badly on the
#     NVFP4 quant otherwise (generation_config adds eval-certified top_k 20).
#     Do NOT add min_p: vLLM rejects min_p/logit_bias under speculative
#     decoding → 400 on every sampled request.
#   - --tool-call-parser poolside_v1 --reasoning-parser poolside_v1:
#     canonical (TRT-LLM names differ; vLLM uses poolside_v1 for both).
#   - --default-chat-template-kwargs enable_thinking=true: native
#     interleaved reasoning on by default; disable per-request via
#     chat_template_kwargs.
#   - Backend flags deliberately absent: auto-selection picks
#     FlashInferCutlass for the NVFP4 path (card warns --linear-backend
#     flashinfer_b12x is broken-slower on 0.25.1). CUTE_DSL_ARCH is only
#     needed on GB10 (sm_121a); SM120 auto-detects.
#
# 1M context: checkpoint ships at 256K (recommended). To go to 1M, edit the
# model's config.json per the card (rope factor 128.0, attention_factor
# 1.4852030263919618, max_position_embeddings 1048576) and raise
# --max-model-len; expect some quality degradation.

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

VLLM_VENV="$HOME/lloyd/.venvs/vllm-laguna"
MODEL_DIR="$PROJECT_DIR/llm/models/poolside-Laguna-S-2.1-NVFP4"
DRAFT_DIR="$PROJECT_DIR/llm/models/poolside-Laguna-S-2.1-DFlash-NVFP4"

if [[ ! -x "$VLLM_VENV/bin/python" ]]; then
  echo "vLLM laguna venv not found at $VLLM_VENV"
  echo "Run: uv venv $VLLM_VENV -p 3.12 && uv pip install -p $VLLM_VENV/bin/python vllm==0.25.1 --torch-backend=cu130"
  exit 1
fi

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "Model not found at $MODEL_DIR"
  echo "Download: snapshot_download('poolside/Laguna-S-2.1-NVFP4', local_dir=$MODEL_DIR)"
  exit 1
fi

if [[ ! -f "$DRAFT_DIR/config.json" ]]; then
  echo "DFlash draft model not found at $DRAFT_DIR"
  echo "Download: snapshot_download('poolside/Laguna-S-2.1-DFlash-NVFP4', local_dir=$DRAFT_DIR)"
  exit 1
fi

export PATH="$VLLM_VENV/bin:/opt/cuda/bin:/usr/bin:/usr/sbin:$PATH"
export LD_LIBRARY_PATH="/usr/lib:/opt/cuda/targets/x86_64-linux/lib:/opt/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_HOME="/opt/cuda"
export NVCC_CCBIN=/usr/bin/g++-15
# Multi-GPU host: PCI_BUS_ID order is index 0 = RTX 3090 (24GB),
# index 1 = RTX PRO 6000 Blackwell (96GB), index 2 = RTX 3090 (24GB).
# Primary pins to the 6000 at index 1.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec "$VLLM_VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name Laguna-S-2.1-NVFP4 primary \
  --port 8096 \
  --host 127.0.0.1 \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.90 \
  --scheduling-policy priority \
  --enable-prefix-caching \
  --no-enable-log-requests \
  --async-scheduling \
  --enable-auto-tool-choice \
  --tool-call-parser poolside_v1 \
  --reasoning-parser poolside_v1 \
  --default-chat-template-kwargs '{"enable_thinking": true}' \
  --speculative-config "{\"model\": \"$DRAFT_DIR\", \"num_speculative_tokens\": 15, \"method\": \"dflash\"}" \
  --override-generation-config '{"temperature": 0.7, "top_p": 0.95}'
