#!/usr/bin/env bash
# Launch Qwen3.5-122B-A10B-NVFP4 via SGLang on RTX PRO 6000 Blackwell (SM120)
#
# Run this INSIDE the lloyd distrobox container, e.g.:
#   distrobox enter lloyd
#   ~/lloyd/scripts/run-sglang.sh
#
# Or via supervisord (already runs inside lloyd).
#
# Usage:
#   ./scripts/run-sglang.sh            # TP=1, 65K context (single GPU)
#   ./scripts/run-sglang.sh --tp2      # TP=2, 262K context (two GPUs)
#   PORT=8096 ./scripts/run-sglang.sh  # Custom port (default: 8096)

set -euo pipefail

VENV_DIR="$(cd "$(dirname "$0")/.." && pwd)/.venvs/sglang"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "ERROR: venv not found at $VENV_DIR"
  echo "Run ./scripts/setup-sglang.sh first."
  exit 1
fi

MODEL="${MODEL:-$HOME/agent-services/llm/models/Sehyo-Qwen3.5-122B-A10B-NVFP4}"
PORT="${PORT:-8096}"
TP_SIZE=1
CONTEXT_LENGTH=65536

if [[ "${1:-}" == "--tp2" ]]; then
  TP_SIZE=2
  CONTEXT_LENGTH=262144
  echo "==> TP=2 mode: 262K context, two GPUs"
else
  echo "==> TP=1 mode: 65K context, single GPU"
fi

echo "==> Model:   $MODEL"
echo "==> Port:    $PORT"
echo "==> Context: $CONTEXT_LENGTH"
echo ""

# CUDA env — mirrors the working start-llm-122b.sh pattern
export CUDA_HOME="/opt/cuda"
export CUDA_VISIBLE_DEVICES=0
export PATH="$VENV_DIR/bin:/opt/cuda/bin:/run/host/usr/bin:/usr/bin:/usr/sbin:$PATH"

export LD_LIBRARY_PATH="/run/host/usr/lib:/opt/cuda/targets/x86_64-linux/lib:/opt/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# --- SM120 / Blackwell required env vars ---
# export SGLANG_ENABLE_SPEC_V2=True     # Only needed with --speculative-algo NEXTN
export SGLANG_ENABLE_JIT_DEEPGEMM=0     # Prevents NaN outputs (wrong scale format on SM120)
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=PHB
export OMP_NUM_THREADS=8
export SAFETENSORS_FAST_GPU=1

exec "$VENV_DIR/bin/python" -m sglang.launch_server \
  --model-path "$MODEL" \
  --tp-size "$TP_SIZE" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --trust-remote-code \
  --mem-fraction-static 0.85 \
  --quantization modelopt_fp4 \
  --attention-backend triton \
  --moe-runner-backend flashinfer_cutlass \
  --fp4-gemm-backend flashinfer_cudnn \
  --context-length "$CONTEXT_LENGTH" \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  # --speculative-algo NEXTN \
  # --speculative-num-steps 3 \
  # --speculative-eagle-topk 1 \
  # --speculative-num-draft-tokens 4 \
  --cuda-graph-max-bs 4 \
  --max-running-requests 4 \
  --chunked-prefill-size 4096 \
  --sleep-on-idle
