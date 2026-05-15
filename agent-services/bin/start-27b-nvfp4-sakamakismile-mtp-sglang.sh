#!/usr/bin/env bash
set -euo pipefail

# Starts sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP via SGLang on GPU 1
# (RTX PRO 6000 Blackwell, 96GB). OpenAI-compatible API on port 8096
# (same slot as the vLLM version — run one at a time).
#
# Stack: sglang 0.5.11, torch 2.11.0+cu130, flashinfer-python 0.6.8.post1 (JIT).
# To rebuild the venv: bash agent-services/setup/setup-sglang.sh
#
# Architecture: dense 27B, Qwen3_5ForConditionalGeneration, hybrid DeltaNet +
# full-attention (64 layers), NVFP4 (ModelOpt), MTP head BF16.
#
# Key flag notes:
#   --quantization modelopt_fp4: NVFP4 / ModelOpt FP4 (more specific than
#     "modelopt"; enables SM120 FP4 GEMM path).
#   --fp4-gemm-backend auto: lets SGLang pick the fastest FP4 GEMM path for
#     SM 12.0 (typically flashinfer_trtllm on Blackwell).
#   --linear-attn-backend flashinfer: DeltaNet layers route through FlashInfer
#     on SM 12.0; triton fallback also works but is slower.
#   --mamba-scheduler-strategy extra_buffer: load-bearing for hybrid models
#     (DeltaNet shares this scheduler path with Mamba/SSM).
#   --speculative-algorithm NEXTN: uses the built-in MTP tensors (mtp.* in
#     safetensors) for draft. Same 3-step / 4-draft-token config as vLLM.
#     SGLANG_ENABLE_SPEC_V2=1 enables the faster v2 speculative path.
#   --tool-call-parser qwen3_coder: SGLang's Qwen3 XML/coder parser.
#     SGLang does not have "qwen3_xml"; qwen3_coder is the functional equivalent.
#   --max-running-requests 2: prevents OOM during cuda-graph capture under
#     KV-cache FP8 + speculative + 262K context (same constraint as vLLM
#     --max-num-seqs 2).
#   --chunked-prefill-size 8192: satisfies DeltaNet block-size constraint (>4288).

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

SGLANG_VENV="$HOME/lloyd/.venvs/sglang"
MODEL_DIR="$PROJECT_DIR/llm/models/sakamakismile-Qwen3.6-27B-Text-NVFP4-MTP"

if [[ ! -x "$SGLANG_VENV/bin/python" ]]; then
  echo "SGLang venv not found at $SGLANG_VENV"
  echo "Run: bash agent-services/setup/setup-sglang.sh"
  exit 1
fi

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "Model not found at $MODEL_DIR"
  exit 1
fi

export PATH="$SGLANG_VENV/bin:/opt/cuda/bin:/run/host/usr/bin:/usr/bin:/usr/sbin:$PATH"
export LD_LIBRARY_PATH="/run/host/usr/lib:/opt/cuda/targets/x86_64-linux/lib:/opt/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_HOME="/opt/cuda"
# PCI bus order: index 0 = RTX 5090 (secondary), index 1 = RTX PRO 6000 Blackwell.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export SGLANG_ENABLE_SPEC_V2=1

exec "$SGLANG_VENV/bin/python" -m sglang.launch_server \
  --model-path "$MODEL_DIR" \
  --served-model-name primary \
  --port 8096 \
  --host 127.0.0.1 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --context-length 262144 \
  --max-running-requests 2 \
  --chunked-prefill-size 8192 \
  --mem-fraction-static 0.90 \
  --kv-cache-dtype fp8_e4m3 \
  --quantization modelopt_fp4 \
  --fp4-gemm-backend auto \
  --attention-backend flashinfer \
  --linear-attn-backend flashinfer \
  --mamba-ssm-dtype bfloat16 \
  --mamba-scheduler-strategy extra_buffer \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --enable-cudagraph-gc \
  --log-level info
