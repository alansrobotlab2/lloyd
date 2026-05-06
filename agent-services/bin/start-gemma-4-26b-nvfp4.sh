#!/usr/bin/env bash
set -euo pipefail

# Starts bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4 via vLLM on GPU 0 (RTX 5090, 32GB).
# OpenAI-compatible API on port 8091 (replaces llama.cpp 35B slot — only one at a time).
# Served-model-name aliases: gemma-4-26B-A4B-it-NVFP4, secondary
#
# Model: Gemma-4 26B-A4B (128 experts × top-8, moe_intermediate_size=704), NVFP4 quant.
# Weights ~13 GiB at FP4 on a 32 GiB card; full 256K context fits comfortably because
# the architecture alternates sliding-window-1024 (most layers) with full-attention
# (every 6th layer), so the long-context KV footprint is tiny.
#
# Multimodal support (vision/audio/video) is present in the model but disabled at the
# server level via --limit-mm-per-prompt to keep the hot path text-only and maximize
# decode throughput. Remove those flags if you want to accept image/video/audio input.
#
# Stack: vllm-experimental venv (vLLM 0.19+ nightly, transformers 5.5.4, FlashInfer 0.6.8,
# PyTorch 2.10+cu130). vLLM 0.19.1 ships a Gemma-4 weight loader that doesn't correctly
# map NVFP4 MoE expert scale keys (vllm-project/vllm#38912), so we use the bg-digitalservices
# checkpoint which ships a `gemma4_patched.py` that we copy over vllm/model_executor/models/gemma4.py.
# The patch is installed manually — see setup notes below.
#
# Attention: Gemma-4 head_dim=176 (2816/16) is not a power of 2, so FLASHINFER attention
# is unsupported. Use TRITON_ATTN (same workaround as Nemotron-3-Super on SM120).
#
# MoE backend: Marlin (required by the patched loader). Nice side benefit: it bypasses
# the CUTLASS TMA WS SMEM-overflow ceiling that limits Qwen3.6-35B-A3B-NVFP4 on SM120.
# VLLM_NVFP4_GEMM_BACKEND=marlin must also be set (handled below).
#
# Setup:
#   1. bash setup/setup-vllm-experimental.sh
#   2. venv's python -m pip install -U transformers  (needed for Gemma-4 recognition)
#   3. snapshot_download('bg-digitalservices/Gemma-4-26B-A4B-it-NVFP4', local_dir=<MODEL_DIR>)
#   4. cp <MODEL_DIR>/gemma4_patched.py <VENV>/lib/python3.12/site-packages/vllm/model_executor/models/gemma4.py
#      (backup first: cp gemma4.py gemma4.py.bak)
#
# Tuning history (2026-04-18, sustained 2000-tok generation, mean of 3 prose prompts):
#   Baseline (current config)                                  184.0 t/s ±0.7  ★
#   + ngram spec decode (num_spec=3, prompt_lookup_max=4)      121.6 t/s ±3.4  (-34%, wrong fit for prose)
#   + max-num-seqs=1                                           182.4 t/s ±0.7  (≈ baseline, no meaningful change)
#   - chunked-prefill, - prefix-caching                        182.8 t/s ±0.7  (≈ baseline, no meaningful change)
#
# Performance analysis:
#   RTX 5090 memory bandwidth ~1.79 TB/s. With 4B active params at FP4 = 2 GB/tok read,
#   theoretical ceiling is ~895 t/s. Observed 184 t/s = 21% of theoretical, which is
#   better than typical MoE (15-20%). Marlin MoE kernels are doing well on SM120.
#   Further gains would require the B12x micro kernel (flashinfer#3080) once merged.

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

VLLM_VENV="$HOME/lloyd/.venvs/vllm-experimental"
MODEL_DIR="$PROJECT_DIR/llm/models/bg-digitalservices-Gemma-4-26B-A4B-it-NVFP4"

if [[ ! -x "$VLLM_VENV/bin/python" ]]; then
  echo "vLLM experimental venv not found at $VLLM_VENV"
  echo "Run: bash setup/setup-vllm-experimental.sh"
  exit 1
fi

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "Model not found at $MODEL_DIR"
  echo "Download: snapshot_download('RedHatAI/gemma-4-26B-A4B-it-NVFP4', local_dir=$MODEL_DIR)"
  exit 1
fi

export PATH="$VLLM_VENV/bin:/opt/cuda/bin:/usr/bin:/usr/sbin:$PATH"
export LD_LIBRARY_PATH="/run/host/usr/lib:/opt/cuda/targets/x86_64-linux/lib:/opt/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_HOME="/opt/cuda"
# PCI bus order: index 0 = RTX 5090 (this server, port 8091),
#                index 1 = RTX PRO 6000 Blackwell (primary vLLM, port 8096).
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=1073741824
export VLLM_ENABLE_CUDAGRAPH_GC=1
export VLLM_USE_FLASHINFER_SAMPLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Marlin backend for non-MoE NVFP4 GEMMs on SM120 (bypasses the CUTLASS TMA WS tile
# problems we hit on Qwen3.6-35B-A3B-NVFP4). Required by the bg-digitalservices
# gemma4_patched.py workaround — must be consistent with --moe-backend marlin below.
export VLLM_NVFP4_GEMM_BACKEND=marlin

exec "$VLLM_VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name gemma-4-26B-A4B-it-NVFP4 secondary \
  --port 8091 \
  --host 127.0.0.1 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --max-model-len 131072 \
  --max-num-seqs 4 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.65 \
  --scheduling-policy priority \
  --quantization modelopt \
  --moe-backend marlin \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend TRITON_ATTN \
  --enable-prefix-caching \
  --no-enable-log-requests \
  --enable-flashinfer-autotune \
  --async-scheduling \
  --performance-mode interactivity \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --limit-mm-per-prompt '{"image": 0, "video": 0, "audio": 0}'
