#!/usr/bin/env bash
# djev — DiffusionGemma 26B-A4B (NVFP4) structured reads on GPU 2's RTX 3090.
#
#   :8010  vLLM, OpenAI API + /metrics   (what Mission Control scrapes)
#   :8011  structured decisions, POST /v1/systemone + /health
#
# This is upstream's entrypoint.sh (github.com/mmastrac/djev-spark) adapted from
# a DGX Spark to a discrete 24 GiB card. Build the venv first with
# setup/setup-djev.sh, which documents why an SM86 card can serve an NVFP4
# checkpoint at all.
#
# THE FIVE SETTINGS THAT ARE NOT UPSTREAM'S, AND WHY
#
#   GPU_UTIL   0.40 -> 0.97.  Upstream's 0.40 is 48 GiB of a Spark's 121 GiB of
#       unified memory. Read as a fraction of a 24 GiB card it is 9.6 GiB,
#       which will not even hold the 17.6 GiB of weights. This is the one
#       setting that silently means something different on this hardware.
#       0.97 rather than something rounder because vLLM's budget is
#       `total * util`, and weights (17.5) + KV (3.0) leave whatever is left
#       for PEAK ACTIVATIONS — of which the sampler transient is 1.25 GiB on
#       its own. At 0.93 that remainder is 1.4 GiB and the profiling step has
#       no room; at 0.97 it is 2.3 GiB. GPU 2 drives no display, so the top
#       of the card is genuinely available.
#
#   MAX_SEQS   32 -> 1.  The start-up profiling step runs the sampler at the
#       full batch and materialises about ten fp32 copies of
#       [MAX_SEQS x CANVAS, vocab]. With vocab 262144 that is exactly
#       `MAX_SEQS * CANVAS * 10` MiB — 40 GiB at upstream's 32x128, and
#       1.25 GiB at 1x128. Concurrency is what a 24 GiB card cannot afford
#       here; context is not. Raising MAX_SEQS costs 1.25 GiB per unit at
#       CANVAS=128, so 2 is affordable and 4 is not.
#
#   MAX_MODEL_LEN  4096 -> 131072.  128k, not the model's full 262144, and
#       this is the one place the card genuinely says no.
#
#       The architecture is unusually cheap per token: only 5 of the 30 layers
#       are full-attention, at 2 KV heads x 512 head dim x 2 (K+V) x 2 bytes =
#       20 KiB/token, and the other 25 are sliding-window 1024 and cost a fixed
#       ~200 MiB whatever the context. With block padding vLLM measures ~24
#       KiB/token, so the full 262144 window is 5.98 GiB of KV. On top of 17.53
#       GiB of weights that is 23.5 GiB before a single activation, against
#       23.6 GiB of card. A dense 30-layer model at this width would want
#       60 GiB, so the architecture is doing its job — it is simply doing it on
#       a card that is 17.5 GiB full before it starts.
#
#       What would have bought the full window is the FP8 KV cache the
#       checkpoint asks for, at 12 KiB/token. Ampere has no FP8, so that door
#       is shut by hardware rather than by configuration — see KV_CACHE_DTYPE.
#
#       Measured ceiling on this box: vLLM sizes the pool itself and reports
#       "estimated maximum model length is 154976" at GPU_UTIL 0.97, CANVAS
#       128, MAX_SEQS 1. 131072 is the clean power of two under it and leaves
#       real headroom; raising it toward 154976 trades that headroom for
#       context nothing has asked for yet. Upstream's own long-document
#       profile is 128k, so this is its profile, not a reduced one.
#
#   KV_CACHE_DTYPE  (unset) -> bfloat16.  The checkpoint asks for an FP8 KV
#       cache — `hf_quant_config.json` carries `"kv_cache_quant_algo": "FP8"` —
#       and vLLM honours it, so leaving this alone is not neutral. Native FP8
#       (fp8e4nv) is SM89+, and this card is SM86, so the boot dies in
#       Attention.__init__ with "FP8 KV cache is not supported by the Triton
#       attention backend ... requires SM89+". It costs nothing that was ever
#       available: every KV figure in this file is already bf16 at 2 bytes,
#       because FP8 was never on the table here. The Spark this came from is
#       SM121 and does get the half-size cache.
#
#   The memory guard  MemAvailable -> nvidia-smi.  Upstream checks host RAM
#       because a Spark's memory IS its VRAM. Here the only number that can
#       refuse a boot is free VRAM on the target card.
#
# GPU 2 IS SINGLE-TENANT. It holds this or the Qwen3.6 secondary, never both.
# The preflight refuses rather than racing it onto the card.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LLOYD_DIR="$(cd "$PROJECT_DIR/.." && pwd)"

VENV="${DJEV_VENV:-$LLOYD_DIR/.venvs/vllm-djev}"
PY="$VENV/bin/python"
DJEV_SRC="${DJEV_SRC:-$PROJECT_DIR/llm/djev-spark}"
MODEL="${DJEV_MODEL_DIR:-$PROJECT_DIR/llm/models/nvidia-diffusiongemma-26B-A4B-it-NVFP4}"
SERVED_NAME="${SERVED_NAME:-djev}"

GPU="${GPU:-2}"
PORT="${PORT:-8010}"
STRUCTURED_PORT="${STRUCTURED_PORT:-8011}"
TLS_PORT="${TLS_PORT:-0}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_SEQS="${MAX_SEQS:-1}"
CANVAS="${CANVAS:-128}"
GPU_UTIL="${GPU_UTIL:-0.97}"
# EMPTY by default, unlike upstream, and that is the point: with no
# --kv-cache-memory vLLM sizes the KV pool as `total*GPU_UTIL - weights - peak
# activations` and hands the whole remainder to KV. Upstream must pin it
# because on a Spark the pool competes with the host; on a card of its own the
# remainder IS the answer, and pinning it just means guessing low. Set a
# number of GiB here only to deliberately cap it.
KV_CACHE_GB="${KV_CACHE_GB:-}"
ATTN="${ATTN:-TRITON_ATTN}"
# bfloat16, not auto: auto reads the checkpoint's FP8 request, which this card
# cannot serve. See the header.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-bfloat16}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
EXTRA_ARGS="${EXTRA_ARGS:---async-scheduling}"
# CUDA graphs and the driver context, and ONLY those. It is deliberately not
# an activations budget: the sampler transient below is the dominant
# activation and is already counted, so adding a second activations figure
# here would double-count it and refuse boots that fit.
OVERHEAD_MIB="${OVERHEAD_MIB:-768}"
# A floor for the guard when the pool is unpinned. ~24 KiB/token of bf16 KV
# (5 full-attention layers x 2 KV heads x 512 head dim x 2 for K+V x 2 bytes,
# plus the sliding layers' fixed 1024-token windows), so 32k tokens is 768 MiB.
MIN_KV_MIB="${MIN_KV_MIB:-768}"
WAIT_SECS="${WAIT_SECS:-1800}"
TEST_PAGE="${TEST_PAGE:-}"
API_KEY="${API_KEY:-}"

export PATH="$VENV/bin:/opt/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/lib:/opt/cuda/targets/x86_64-linux/lib:/opt/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_HOME="/opt/cuda"
# Mandatory. Without it the runtime reorders devices by capability and
# CUDA_VISIBLE_DEVICES=2 lands on a different card than nvidia-smi's index 2.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# vLLM's internal ZMQ port. The primary on :8096 leaves this at the default, so
# a second engine on one box must move or the two collide at start-up.
export VLLM_PORT="${VLLM_PORT:-5300}"
export VLLM_USE_V2_MODEL_RUNNER=1
export TOKENIZERS_PARALLELISM=false

die() { echo "start-djev: $*" >&2; exit 2; }

# ── Preflight ─────────────────────────────────────────────────────────
[[ -x "$PY" ]] || die "no venv at $VENV — run setup/setup-djev.sh"
[[ -f "$MODEL/config.json" ]] || die "no checkpoint at $MODEL — run setup/setup-djev.sh"
[[ -f "$DJEV_SRC/server/structured_server.py" ]] || \
    die "no structured server at $DJEV_SRC — run setup/setup-djev.sh"
compgen -G "$VENV/.djev-overlay-*" >/dev/null || \
    die "the structured-reads overlay is not applied — run setup/setup-djev.sh"

# The weights, as they actually are on disk, rather than a number in a comment
# that a re-quantized checkpoint would silently invalidate.
WEIGHTS_MIB=$(du -sm --apparent-size "$MODEL"/*.safetensors 2>/dev/null | awk '{s+=$1} END {print s+0}')
(( WEIGHTS_MIB > 0 )) || die "no safetensors under $MODEL"
# [MAX_SEQS x CANVAS, 262144] fp32, about ten live copies during profiling.
# 262144 * 4 * 10 / 1048576 == 10 exactly, which is where the 10 comes from.
TRANSIENT_MIB=$(( MAX_SEQS * CANVAS * 10 ))
# With an unpinned pool there is no KV figure to add: whatever is left after
# weights and activations becomes KV, and a pool of zero is the engine
# refusing to start rather than a silent OOM. So the guard asks only whether
# the weights and the start-up transient fit, plus one full sequence's worth
# of KV as a floor — below that there is no point booting.
KV_MIB=$(( ${KV_CACHE_GB:-0} * 1024 ))
(( KV_MIB > 0 )) || KV_MIB=$MIN_KV_MIB
NEED_MIB=$(( WEIGHTS_MIB + KV_MIB + TRANSIENT_MIB + OVERHEAD_MIB ))
FREE_MIB=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
TOTAL_MIB=$(nvidia-smi --id="$GPU" --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' ')
GPU_NAME=$(nvidia-smi --id="$GPU" --query-gpu=name --format=csv,noheader)

cat <<EOF
==> djev: DiffusionGemma 26B-A4B NVFP4
    gpu            $GPU  $GPU_NAME  (${FREE_MIB} MiB free of ${TOTAL_MIB})
    context        $MAX_MODEL_LEN    canvas $CANVAS    max seqs $MAX_SEQS
    kv dtype       $KV_CACHE_DTYPE    pool ${KV_CACHE_GB:-auto (the remainder)}
    ports          $PORT vllm, $STRUCTURED_PORT structured
    budget         weights ${WEIGHTS_MIB} + KV ${KV_MIB} + transient ${TRANSIENT_MIB}
                   + overhead ${OVERHEAD_MIB} = ${NEED_MIB} MiB
EOF

if (( FREE_MIB < NEED_MIB )); then
    echo >&2
    echo "refusing to start: GPU $GPU has ${FREE_MIB} MiB free, this needs ${NEED_MIB}." >&2
    if nvidia-smi --id="$GPU" --query-compute-apps=pid,used_memory,process_name \
         --format=csv,noheader 2>/dev/null | grep -q .; then
        echo "what is on the card:" >&2
        nvidia-smi --id="$GPU" --query-compute-apps=pid,used_memory,process_name \
            --format=csv,noheader | sed 's/^/  /' >&2
        echo >&2
        echo "If that is llama.cpp, the Qwen3.6 secondary still holds GPU $GPU." >&2
        echo "They are an either/or: set secondary_enabled: false in config.yaml and" >&2
        echo "  supervisorctl -c agent-services/supervisor/supervisord.conf stop agent-llm-secondary" >&2
    fi
    exit 2
fi

# ── vLLM ──────────────────────────────────────────────────────────────
# --diffusion-config is the overlay's, not stock vLLM's: it fixes the served
# canvas width, which bounds the answer template and any thought block.
# --override-generation-config drops the checkpoint's max_new_tokens so a read
# is bounded by the canvas rather than by a generation cap that means nothing
# for a diffusion read.
# shellcheck disable=SC2086  # EXTRA_ARGS is a flag list
"$VENV/bin/vllm" serve "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --trust-remote-code \
    --host 127.0.0.1 \
    --port "$PORT" \
    --max-num-seqs "$MAX_SEQS" \
    --max-model-len "$MAX_MODEL_LEN" \
    --attention-backend "$ATTN" \
    --kv-cache-dtype "$KV_CACHE_DTYPE" \
    --gpu-memory-utilization "$GPU_UTIL" \
    ${KV_CACHE_GB:+--kv-cache-memory $(( ${KV_CACHE_GB:-0} * 1073741824 ))} \
    ${MAX_NUM_BATCHED_TOKENS:+--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"} \
    --max-logprobs 32 \
    --enable-prefix-caching \
    --diffusion-config "{\"canvas_length\": ${CANVAS}}" \
    --override-generation-config '{"max_new_tokens": null}' \
    $EXTRA_ARGS &
VLLM_PID=$!

healthy() { curl -sf --max-time 2 "http://127.0.0.1:$1/health" >/dev/null 2>&1; }

for (( i = 0; i < WAIT_SECS / 5; i++ )); do
    healthy "$PORT" && break
    kill -0 "$VLLM_PID" 2>/dev/null || { echo "vllm exited during start-up" >&2; exit 1; }
    sleep 5
done
healthy "$PORT" || { echo "vllm not healthy after ${WAIT_SECS}s" >&2; kill "$VLLM_PID" 2>/dev/null; exit 1; }
echo "==> vllm ready on :$PORT"

# ── Structured server ─────────────────────────────────────────────────
# Restarted whenever it exits, so its code can be reloaded without touching
# vLLM — which is the expensive half to restart. Only vLLM's exit ends this
# script. It is stdlib + transformers, so it runs from the same venv.
serve_structured() {
    while kill -0 "$VLLM_PID" 2>/dev/null; do
        # A reload signals the server, and set -e would take this loop with it.
        TEST_PAGE="$TEST_PAGE" API_KEY="$API_KEY" \
        "$PY" "$DJEV_SRC/server/structured_server.py" \
            --upstream "http://127.0.0.1:$PORT" \
            --model "$SERVED_NAME" \
            --tokenizer "$MODEL" \
            --canvas "$CANVAS" \
            --host 127.0.0.1 \
            --port "$STRUCTURED_PORT" \
            --tls-port "$TLS_PORT" \
            --cert-dir "$HOME/.cache/djev" || true
        kill -0 "$VLLM_PID" 2>/dev/null || break
        echo "structured server exited; restarting" >&2
        sleep 1
    done
}
serve_structured &
SERVER_LOOP=$!

cleanup() {
    kill "$VLLM_PID" "$SERVER_LOOP" 2>/dev/null || true
    pkill -f "structured_server.py --upstream http://127.0.0.1:$PORT" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup TERM INT

wait "$VLLM_PID"
echo "vllm exited; stopping" >&2
cleanup
exit 1
