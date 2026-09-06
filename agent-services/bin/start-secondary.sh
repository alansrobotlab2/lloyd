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
# Two runtimes live here, because the model decides the runtime:
#
#   * qwen35 / lfm2  -> vLLM. Small dense models with NVFP4/BF16 weights.
#   * qwen36         -> llama.cpp. Qwen3.6-35B-A3B only exists at a size that
#                       fits a 24 GB card as a GGUF Q3, and vLLM cannot serve
#                       it: unsloth's NVFP4 builds need SM100+ (the 3090 is
#                       SM86) and vLLM's GGUF path does not cover this hybrid
#                       linear-attention MoE. So the 35B is llama-server.
#
# Usage:
#   ./start-secondary.sh                 # default model, GPU 2, port 8091
#   MODEL=qwen36 ./start-secondary.sh    # Qwen3.6-35B-A3B UD-Q3_K_XL (default)
#   MODEL=qwen35 ./start-secondary.sh    # Qwen3.5-4B  (vLLM)
#   MODEL=lfm2 ./start-secondary.sh      # LFM2.5-2.6B (vLLM)
#   GPU=0 PORT=8092 ./start-secondary.sh
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VLLM_VENV="$HOME/lloyd/.venvs/vllm-qwen38-flash-next"
LLAMA_SERVER="$PROJECT_DIR/llm/llama.cpp/build/bin/llama-server"
MODEL="${MODEL:-qwen36}"
PORT="${PORT:-8091}"
# PCI_BUS_ID order: 0 and 2 are the RTX 3090s, 1 is the RTX PRO 6000 (primary).
GPU="${GPU:-2}"

export PATH="$VLLM_VENV/bin:/opt/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/lib:/opt/cuda/targets/x86_64-linux/lib:/opt/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_HOME="/opt/cuda"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------------------------------------------------------------- llama.cpp --
if [[ "$MODEL" == "qwen36" ]]; then
  MODEL_FILE="${MODEL_FILE:-$PROJECT_DIR/llm/models/unsloth-Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf}"
  MAX_LEN="${MAX_LEN:-262144}"

  # Why Q3 and why the whole window fits in 24 GB:
  #
  # Qwen3.6-35B-A3B is a hybrid — `full_attention_interval: 4`, so only 10 of
  # its 40 layers carry a KV cache and the other 30 are linear-attention with
  # constant-size state. At head_dim 256 x 2 KV heads x 2 (K+V) x 10 layers
  # that is 20 KiB/token, so the full 262144 window costs 5.0 GiB at f16. The
  # same window on a dense 40-layer model would want 20 GiB.
  #
  #   UD-Q3_K_XL weights (16.85 GB)          15.7 GiB
  #   KV @ f16, full 262144                   5.0 GiB
  #   compute + graph buffers                ~1.0 GiB
  #                                          --------
  #                                          ~21.7 GiB of 24
  #
  # That headroom is why lloyd-agent-worker moved off GPU 2 to GPU 0 on
  # 2026-09-06 — its 3.3 GiB would not fit alongside. GPU 2 is single-tenant
  # now; putting anything back on it will OOM this.
  #
  # Q4 does not fit: UD-Q4_K_S is 19.5 GiB of weights, so 25.5 GiB total.
  # Dropping to q8_0 KV would buy back 2.4 GiB and still leave Q4 short, so
  # there is no reason to quantize the cache — f16 it is.
  #
  # NOTE: UD-IQ4_NL of this model is *incoherent* — HTML/XML/tool_call
  # fragments instead of prose, on both SM89 and SM120, llama.cpp b8668+ and
  # b8808. See ggml-org/llama.cpp#21495. UD-Q3_K_XL is not affected.
  #
  # --parallel 1 is deliberate: llama.cpp divides --ctx-size across slots, so
  # any n>1 would cut the per-request window to 262144/n. The full window was
  # the point, so the secondary serializes requests. The old vLLM secondary
  # ran --max-num-seqs 4; if concurrency matters more than window, drop
  # MAX_LEN and raise PARALLEL together.
  PARALLEL="${PARALLEL:-1}"

  # Qwen3.6 thinking-mode-general sampling, per the Qwen3.6 model card. Their
  # own agentic evals (SWE-Bench, Terminal-Bench 2.0) ran temp=1.0/top_p=0.95,
  # which is what the secondary does all day, so this is the profile to match.
  # presence_penalty 1.5 is the card's recommendation against repetition loops
  # in quantized builds; it applies to structural tokens too, so if tool calls
  # start coming back malformed, set PRESENCE_PENALTY=0.0 (the card's own
  # "precise coding" setting) before touching anything else.
  PRESENCE_PENALTY="${PRESENCE_PENALTY:-1.5}"

  if [[ ! -x "$LLAMA_SERVER" ]]; then
    echo "llama-server not found at $LLAMA_SERVER" >&2
    echo "Build it first:  bash $PROJECT_DIR/setup/setup-llm.sh" >&2
    exit 1
  fi
  if [[ ! -f "$MODEL_FILE" ]]; then
    echo "Model not found at $MODEL_FILE" >&2
    echo "Download it first:" >&2
    echo "  hf download unsloth/Qwen3.6-35B-A3B-GGUF Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf \\" >&2
    echo "     --local-dir $PROJECT_DIR/llm/models/unsloth-Qwen3.6-35B-A3B-GGUF" >&2
    exit 1
  fi

  echo "==> secondary: Qwen3.6-35B-A3B UD-Q3_K_XL on GPU $GPU, port $PORT, ctx $MAX_LEN"
  exec "$LLAMA_SERVER" \
    --model "$MODEL_FILE" \
    --alias secondary \
    --port "$PORT" \
    --host 127.0.0.1 \
    --n-gpu-layers 999 \
    --ctx-size "$MAX_LEN" \
    --parallel "$PARALLEL" \
    --flash-attn on \
    --cache-type-k f16 \
    --cache-type-v f16 \
    --jinja \
    --reasoning on \
    --reasoning-format deepseek \
    --metrics \
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.0 \
    --presence-penalty "$PRESENCE_PENALTY" \
    --repeat-penalty 1.0
fi

# --------------------------------------------------------------------- vLLM --
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
    echo "Unknown MODEL='$MODEL' (expected: qwen36 | qwen35 | lfm2)" >&2
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
  --enable-prompt-tokens-details \
  --no-enable-log-requests \
  --scheduling-policy priority \
  --enable-auto-tool-choice \
  "${PARSER_ARGS[@]}"
