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
#
# Both branches ask the card how much is free before they exec anything (#1316),
# through agent-services/bin/gpu-mem.sh — the same helper start-djev.sh reads, so
# the two guards cannot disagree about one card. GPU 2 is single-tenant, which is
# the whole reason the check exists: without it, "both engines wanted the card"
# arrives as five `alloc_tensor_range: failed to allocate CUDA0 buffer of size
# 16294177280` in agent-llm-secondary.err (that is what it looked like on
# 2026-09-20) instead of one line saying which flag to flip.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# Every path and figure a start depends on is overridable, the two engine
# binaries included. They were not before #1316, and that is what made the VRAM
# guard below untestable: proving "the guard refused without booting anything"
# needs a fake engine whose launch a test can watch for, and with `LLAMA_SERVER`
# hardcoded the only way to reach the guard at all was to build the real llama.cpp
# and let it try. `VLLM_VENV` names `.venvs/vllm-qwen38-flash-next`, which is not
# a directory on this box as of 2026-09-23 (`vllm-flash-next-main` is), so the
# vLLM branch could not reach its exec either way without the override.
VLLM_VENV="${VLLM_VENV:-$HOME/lloyd/.venvs/vllm-qwen38-flash-next}"
LLAMA_SERVER="${LLAMA_SERVER:-$PROJECT_DIR/llm/llama.cpp/build/bin/llama-server}"
MODEL="${MODEL:-qwen36}"
PORT="${PORT:-8091}"
# PCI_BUS_ID order: 0 and 2 are the RTX 3090s, 1 is the RTX PRO 6000 (primary).
GPU="${GPU:-2}"
# What the vLLM branch hands --gpu-memory-utilization, and therefore what the
# preflight budgets for that branch. One variable feeding both, because the defect
# that made start-djev.sh approve nine OOM boots (#1355) was a preflight costing
# 2.6 GiB less than the GPU_UTIL it handed the engine.
GPU_UTIL="${GPU_UTIL:-0.90}"
# Compute buffers, CUDA graphs and the driver context — the `~1.0 GiB` line of the
# budget table in the qwen36 branch. Added to the llama.cpp need only: the vLLM
# need is a fraction of the whole card, and the tenth of the card that fraction
# leaves unreserved is what covers these there.
OVERHEAD_MIB="${OVERHEAD_MIB:-1024}"

export PATH="$VLLM_VENV/bin:/opt/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/lib:/opt/cuda/targets/x86_64-linux/lib:/opt/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_HOME="/opt/cuda"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Per-model defaults, resolved up here rather than inside the branches below,
# because the VRAM preflight has to cost the boot before it decides whether to
# let it through — and it cannot cost a model it has not named yet.
if [[ "$MODEL" == "qwen36" ]]; then
  MODEL_FILE="${MODEL_FILE:-$PROJECT_DIR/llm/models/unsloth-Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf}"
  MAX_LEN="${MAX_LEN:-262144}"
else
  MAX_LEN="${MAX_LEN:-65536}"
fi

# ------------------------------------------------ GPU 2 VRAM preflight (#1316) --
# The reciprocal of the guard start-djev.sh has had since 2026-09-20. GPU 2 holds
# one engine at a time — config.yaml's `djev:` comment calls it "an either/or, not
# a pair of flags" — and until now only the djev side asked the card first. The
# secondary exec'd llama-server and let the allocation fail inside
# llama_model_load, which is how agent-llm-secondary.err came to hold five
# `alloc_tensor_range: failed to allocate CUDA0 buffer of size 16294177280` lines
# on 2026-09-20, each followed by `exiting due to model loading error`, with
# supervisord's autorestart queueing the next identical crash. A crash loop leaves
# an .err file and hopes someone reads it; a refusal leaves a sentence naming the
# flag to flip.
#
# It sits above the -x/-f checks further down on purpose. A start that is both
# mispathed and over-budgeted should read as the GPU problem, because the GPU
# problem is the one the .err never mentioned.
#
# shellcheck source=./gpu-mem.sh
source "$PROJECT_DIR/bin/gpu-mem.sh"

FREE_MIB="" TOTAL_MIB="" GPU_NAME=""
if ! gpu_mem_read FREE_MIB TOTAL_MIB GPU_NAME "$GPU"; then
  # The asymmetry with start-djev.sh, which refuses on the same failure, is
  # deliberate and is #1316 clause 5: this slot is optional, so a dead driver or
  # an nvidia-smi that answers `N/A` must not become an engine outage. Booting
  # blind is what this script did before any guard existed — the fallback is the
  # old behaviour, not a new risk.
  echo "start-secondary: WARNING: cannot read free VRAM on GPU $GPU (nvidia-smi missing, failing, or non-numeric) — starting WITHOUT the preflight." >&2
else
  # What this boot will ask the card for, measured rather than asserted.
  #
  #   llama.cpp (MODEL=qwen36): the GGUF's own bytes + the KV the budget table in
  #     the branch below computes (20 KiB/token x MAX_LEN) + OVERHEAD_MIB. NOT the
  #     16294177280 B llama.cpp failed on: that is one CUDA0 tensor range, not the
  #     boot's demand. An unstat-able GGUF counts as zero here and is reported by
  #     the -f check below — blaming the card for a missing file would be the same
  #     misreading this guard exists to end.
  #   vLLM (MODEL=qwen35 | lfm2): what the exec at the bottom will be told to
  #     reserve, GPU_UTIL x the whole card, read from the same variable that feeds
  #     --gpu-memory-utilization. Budgeting less than the flag hands the engine is
  #     #1355 on this very card, where start-djev.sh's preflight cost 2.6 GiB
  #     under its own GPU_UTIL and approved nine OOM boots.
  if [[ "$MODEL" == "qwen36" ]]; then
    WEIGHTS_MIB=0
    if MODEL_BYTES=$(stat -c '%s' "$MODEL_FILE" 2>/dev/null); then
      WEIGHTS_MIB=$(( (MODEL_BYTES + 1048575) / 1048576 ))
    fi
    KV_MIB=$(( 20 * MAX_LEN / 1024 ))
    NEED_MIB=$(( WEIGHTS_MIB + KV_MIB + OVERHEAD_MIB ))
    NEED_WHY="weights ${WEIGHTS_MIB} + KV ${KV_MIB} + overhead ${OVERHEAD_MIB}"
  else
    # Ceiling of GPU_UTIL x the whole card, and nothing more: `--gpu-memory-utilization u`
    # targets u x total as the engine's footprint, so a free reading of exactly that is a
    # boot that fits, and the comparison below is strict. Rounding up by a flat +1 (the
    # first cut here) is a whole MiB of conservatism on an exact product and makes
    # GPU_UTIL=1.0 impossible on an empty card, which is a false refusal wearing a guard's
    # clothes. A non-dyadic util like 0.90 lands a hair above its true product in IEEE
    # doubles and so ceils one higher than exact arithmetic would — conservative, and the
    # direction a guard is allowed to err.
    NEED_MIB=$(awk -v u="$GPU_UTIL" -v t="$TOTAL_MIB" \
      'BEGIN { v = u * t; c = int(v); if (c < v) c++; printf "%d", c }')
    NEED_WHY="vLLM --gpu-memory-utilization $GPU_UTIL of ${TOTAL_MIB}"
  fi

  echo "    card           $GPU  ${GPU_NAME}  (${TOTAL_MIB} MiB total, ${FREE_MIB} MiB free)"
  echo "    needs          ${NEED_MIB} MiB (${NEED_WHY})"

  if (( FREE_MIB < NEED_MIB )); then
    echo >&2
    echo "start-secondary: refusing to start: GPU $GPU has ${FREE_MIB} MiB free, this needs ${NEED_MIB}." >&2
    if GPU_HOLDERS=$(gpu_mem_holders "$GPU") && [[ -n "$GPU_HOLDERS" ]]; then
      echo "what is on the card:" >&2
      printf '%s\n' "$GPU_HOLDERS" | sed 's/^/  /' >&2
      echo >&2
    fi
    # Named even when the listing above came back empty: the operator needs the
    # fix, not the symptom, and a card this full has a tenant whether nvidia-smi
    # can name it or not.
    echo "GPU $GPU is an either/or: this slot (agent-llm-secondary) and agent-djev" >&2
    echo "share one 24 GiB card and only one of them fits. The only vLLM program" >&2
    echo "supervised onto this card is djev, so a holder reading VLLM::EngineCore is" >&2
    echo "almost certainly it." >&2
    echo "To run the secondary instead, set secondary_enabled: true and djev.enabled: false" >&2
    echo "in config.yaml and restart the backend — server.py's _sync_llm_slots is what starts" >&2
    echo "and stops the two — and stop the other tenant first:" >&2
    echo "  supervisorctl -c $PROJECT_DIR/supervisor/supervisord.conf stop agent-djev" >&2
    exit 2
  fi
fi

# ---------------------------------------------------------------- llama.cpp --
if [[ "$MODEL" == "qwen36" ]]; then
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
# MAX_LEN already resolved above, with the preflight's copy of the default.
MAX_SEQS="${MAX_SEQS:-4}"

case "$MODEL" in
  lfm2)
    MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/llm/models/LiquidAI-LFM2.5-2.6B}"
    SERVED="LFM2.5-2.6B"
    # LFM2.5 emits pythonic calls between <|tool_call_start|>/<|tool_call_end|>
    # and always opens with a <think> block. There is no `lfm2` REASONING
    # parser in vLLM 0.28 — deepseek_r1 is the generic <think>...</think>
    # parser and matches this chat template exactly.
    PARSER_ARGS=(--tool-call-parser lfm2 --reasoning-parser deepseek_r1)
    ;;
  qwen35)
    MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/llm/models/Qwen-Qwen3.5-4B}"
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
  --gpu-memory-utilization "$GPU_UTIL" \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --no-enable-log-requests \
  --scheduling-policy priority \
  --enable-auto-tool-choice \
  "${PARSER_ARGS[@]}"
