#!/usr/bin/env bash
set -euo pipefail

# Starts sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP via vLLM on GPU 0
# (RTX PRO 6000 Blackwell, 96GB). OpenAI-compatible API on port 8096
# (same primary slot as the other 27B/35B nvfp4 builds — only one at a time).
#
# Why this re-quant over unsloth/mmangkad:
#   - Qwen3_5ForConditionalGeneration.from_pretrained drops the MTP head during
#     loading, so naive modelopt re-quants (unsloth, mmangkad, sakamakismile's
#     base Qwen3.6-27B-NVFP4) ship ZERO mtp.* tensors despite leaving
#     text_config.mtp_num_hidden_layers=1 in config.json.
#   - This Text-NVFP4-MTP build is from the lna-lab/GGUF-to-NVFP4-SM120 pipeline,
#     which grafts the 15 MTP tensors (mtp.fc, mtp.layers.0.*, mtp.norm,
#     mtp.pre_fc_norm_*) back in BF16 after modelopt export. Verified locally:
#     2354 total tensors, 15 mtp.* present, 64 main layers (0-63).
#
# Architecture: dense 27B, Qwen3_5ForConditionalGeneration / model_type=qwen3_5,
# hybrid DeltaNet + full-attention (64 layers, full_attention_interval=4 → 16
# full / 48 linear), NVFP4 (ModelOpt) on linear ops, MTP head BF16.
#
# Multimodal: stripped. Vision tower removed entirely (0 model.visual tensors,
# no image_token_id / video_token_id / vision_config in config.json, no
# processor_config.json). Don't pass --limit-mm-per-prompt; the model is
# registered text-only.
#
# Stack: vllm-experimental venv (vLLM 0.19+ nightly, FlashInfer 0.6.8,
# PyTorch 2.10+cu130). Same SM120 path as the other Qwen3.6 NVFP4 builds.
#
# Tuning:
#   - --language-model-only: REQUIRED. Architecture is Qwen3_5ForConditionalGeneration
#     (multimodal). Without this flag vLLM tries to instantiate the image processor
#     and fails with `OSError: Can't load image processor` since this text-only
#     re-quant ships no preprocessor_config.json.
#   - --quantization modelopt: explicit per the model card; auto-detect may pick
#     compressed-tensors fallback and lose the SM120 fast path.
#   - --max-num-seqs 2: load-bearing per the model card. --max-num-seqs >=4 under
#     kv-cache fp8 + speculative n=3 + max-model-len 262144 silently OOMs during
#     cuda-graph capture on vLLM 0.19.1rc1.
#   - --max-num-batched-tokens 8192: DeltaNet block-size constraint requires >4288.
#   - --speculative-config method=qwen3_5_mtp, num_speculative_tokens=3: canonical
#     per the model card. Single MTP layer applied recursively 3x per draft pass;
#     model card reports per-position acceptance ~87%/72%/61%, mean accepted
#     length ~3-4, ~1.9x decode multiplier. n>3 diverges the drafter and
#     acceptance drops. vLLM emits a deprecation warning for qwen3_5_mtp and
#     normalizes to "mtp" internally — harmless.
#   - --tool-call-parser qwen3_xml: empirical last-known-good for the Qwen3.5/3.6
#     family. qwen3_coder wedges the engine on the stop-token path under MTP.
#   - --override-generation-config presence_penalty=1.5: Unsloth anti-repetition
#     default for the Qwen3.5 thinking family; reusing here since this is the
#     same base model.
#   - --performance-mode interactivity: +10-15% single-user decode, sub-second TTFT.
#
# Weights ~19 GiB on disk (single model.safetensors: ~18 GiB NVFP4 main +
# ~850 MiB BF16 MTP/conv1d/lm_head). On the 96 GiB card at gpu-util 0.95 that
# leaves ample headroom for KV cache across 262K context since only 16/64
# layers are full-attention.

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
export LD_LIBRARY_PATH="/run/host/usr/lib:/opt/cuda/targets/x86_64-linux/lib:/opt/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_HOME="/opt/cuda"
export NVCC_CCBIN=/usr/bin/g++-15
# Single-GPU host: index 0 = RTX PRO 6000 Blackwell (this server, port 8096).
# 5090 removed 2026-05-18.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=1073741824
export VLLM_ENABLE_CUDAGRAPH_GC=1
export VLLM_USE_FLASHINFER_SAMPLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec "$VLLM_VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name Qwen3.6-27B-nvfp4-sakamakismile-mtp primary \
  --port 8096 \
  --host 127.0.0.1 \
  --trust-remote-code \
  --language-model-only \
  --quantization modelopt \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --max-num-seqs 2 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.80 \
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
