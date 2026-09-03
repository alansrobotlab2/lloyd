#!/usr/bin/env bash
# Secondary LLM slot — port 8091, on one of the idle RTX 3090s.
#
# Why this exists: the box has three GPUs. The RTX PRO 6000 (GPU 1) runs the
# primary and is the contended resource; both 3090s sat at 0-2% utilisation
# while cheap autonomy tasks queued behind the primary's two worker slots.
# config.yaml already defines `models.secondary` at :8091 and autonomy.run_task
# already routes a task there via `model: secondary` in its frontmatter — only
# the server was missing.
#
# Usage:
#   ./start-secondary.sh                 # default model, GPU 2, port 8091
#   MODEL=lfm2 ./start-secondary.sh      # LFM2.5-2.6B
#   MODEL=qwen35 ./start-secondary.sh    # Qwen3.5-4B
#   GPU=0 PORT=8092 ./start-secondary.sh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VLLM_VENV="$HOME/lloyd/.venvs/vllm-qwen38-flash-next"
MODEL="${MODEL:-qwen35}"
PORT="${PORT:-8091}"
# PCI_BUS_ID order: 0 and 2 are the RTX 3090s, 1 is the RTX PRO 6000 (primary).
GPU="${GPU:-2}"
MAX_LEN="${MAX_LEN:-65536}"
MAX_SEQS="${MAX_SEQS:-4}"

case "$MODEL" in
  lfm2)
    MODEL_DIR="$PROJECT_DIR/llm/models/LiquidAI-LFM2.5-2.6B"
    SERVED="LFM2.5-2.6B"
    # LFM2.5 emits pythonic calls between <|tool_call_start|>/<|tool_call_end|>
    # and always opens with a <think> block. There is no `lfm2` REASONING
    # parser in vLLM 0.28 — deepseek_r1 is the generic <think>...</think>
    # parser and matches this chat template exactly.
    PARSER_ARGS=(--tool-call-parser lfm2 --reasoning-parser deepseek_r1)
    ;;
  qwen35)
    MODEL_DIR="$PROJECT_DIR/llm/models/Qwen-Qwen3.5-4B"
    SERVED="Qwen3.5-4B"
    # Same family conventions as the primary, so the harness needs no changes.
    PARSER_ARGS=(--tool-call-parser qwen3_xml --reasoning-parser qwen3)
    ;;
  *)
    echo "Unknown MODEL='$MODEL' (expected: lfm2 | qwen35)" >&2
    exit 2
    ;;
esac

if [[ ! -x "$VLLM_VENV/bin/python" ]]; then
  echo "vLLM venv not found at $VLLM_VENV" >&2
  exit 1
fi
if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "Model not found at $MODEL_DIR" >&2
  echo "Download it first, e.g.:" >&2
  echo "  hf download LiquidAI/LFM2.5-2.6B --local-dir $MODEL_DIR" >&2
  exit 1
fi

export PATH="$VLLM_VENV/bin:/opt/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/lib:/opt/cuda/targets/x86_64-linux/lib:/opt/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_HOME="/opt/cuda"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "==> secondary: $SERVED on GPU $GPU, port $PORT"
exec "$VLLM_VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name "$SERVED" secondary \
  --port "$PORT" \
  --host 127.0.0.1 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --max-model-len "$MAX_LEN" \
  --max-num-seqs "$MAX_SEQS" \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --no-enable-log-requests \
  --scheduling-policy priority \
  --enable-auto-tool-choice \
  "${PARSER_ARGS[@]}"
