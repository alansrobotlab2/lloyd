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
#
# DETERMINISM: WHAT IS SHIPPED, AND WHAT EACH VARIANT MEASURED  (#1357)
#
#   djev does not answer a byte-identical request twice. The argmax is stable,
#   but the label logprobs a rank score is built from move by nats between
#   identical requests, with the prefix cache measured constant — which is why
#   two promotions that touched no retrieval code were rolled back for the
#   difference, and why the eval replays djev per request (LLOYD_DJEV_REPLAY,
#   app/djev.py) instead of trusting a paired comparison. Ruled out with the
#   engine's own counters and by flag: the prefix cache, CUDA graphs, compile,
#   async scheduling, the seed canvas and sampling. What is left is the kernels
#   — the Marlin NvFp4 MoE path (the boot logs it: "Weight-only FP4 compression
#   will be used leveraging the Marlin kernel") and TRITON_ATTN.
#
#   Two levers below reach those kernels, so a variant boots by env override
#   with no edit to this file:
#
#       MOE_BACKEND=triton BATCH_INVARIANT=1 ./start-djev.sh
#
#   and one probe measures it, printing each request's prefix_cache hits/queries
#   delta so a cache effect cannot be mistaken for a kernel effect:
#
#       .venvs/lloyd/bin/python -m scripts.djev_determinism_probe --variant-label "my variant"
#
#   EVERY VARIANT BOOTED ON THIS BOX, one row each, with the probe's max |Δ
#   label logprob| in nats over 5 byte-identical runs and the recall p50 in ms on
#   the 81-query pinned eval (eval/vault_recall_queries.yaml), worst of the two
#   pinned trials in eval/baselines/automod-noise-*-20260921-*.json. The variant
#   column is `MOE_BACKEND / BATCH_INVARIANT`, and an empty MOE_BACKEND is
#   written `auto` because that is what vLLM resolves it to. `cold` is a fresh
#   cache_salt per request, so every run reports hits+0. `warm` repeats one
#   prompt: run 1 is the request that POPULATES the cache and also reports
#   hits+0, and from run 2 on every run reports queries+4800 hits+4768 — the
#   reusable prefix is whole 32-token blocks up to the boundary before the
#   position being generated, floor((4800-1)/32)*32 = 4768. So the warm regime
#   holds the cache constant across the runs it compares, which is what the
#   printed per-request delta is there to prove; read run 1 as the fill, not as
#   a miss that makes the pair incomparable. Both regimes must
#   print 0.0000: a config that passes one and not the other has hidden a regime,
#   not fixed the kernels. Measured at the probe's default shape, prompt_tokens
#   4800, 2026-09-22.
#
#   variant                             cold nats  warm nats  recall p50 ms
#   auto (Marlin FP4 MoE) / 0              5.0156     8.2064       510.5
#
#   That is the only row anyone has measured. The variants still unbooted —
#   MOE_BACKEND=triton, batched_triton, triton_unfused, marlin; BATCH_INVARIANT=1
#   (which needs MAX_MODEL_LEN under ~100k, since 131072 OOMs the KV pool, so
#   that trial also costs production context); and the two the live boot config
#   points at that no one has touched (enable_flashinfer_autotune,
#   fuse_act_quant) — are owed by backlog #1361, because each boot is a ~2 min
#   window in which the recall falls back to qmd's cross-encoder
#   (app/qmd_health.py counts djev_fallbacks) and GPU 2 is single-tenant, and a
#   self-modification round may not restart an engine at all. The two
#   #1357 triage runs saw the same signature at a shorter prompt (2176 tokens,
#   warm 2.91 / cold 11.07 nats), so the sign is not an artefact of this shape;
#   the rows are this probe's numbers.
#
#   So the defaults below are the INCUMBENT, not a winner of a sweep. When a row
#   prints 0.0000 in both regimes, flip both this line and the defaults;
#   tests/test_start_djev_flags.py refuses a shipped default that has no row, so
#   an unmeasured boot cannot become production by editing one of the two.
#
# shipped defaults: MOE_BACKEND="" BATCH_INVARIANT="0"

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
# ── Determinism levers (#1357) ────────────────────────────────────────
# Both reach the kernels the label logprobs are made of, and both are EMPTY/OFF
# by default so the production boot is byte-identical to the one measured in the
# header. They exist so a variant can be booted without editing this file:
#
#   MOE_BACKEND        vLLM's --moe-backend, forwarded verbatim when non-empty.
#       Empty means vLLM's own "auto", which on this card picks Marlin for the
#       weight-only NvFp4 weights (vllm/config/kernel.py MoEBackend; the boot
#       logs the choice). Candidate spellings for the bisect: triton,
#       batched_triton, triton_unfused, marlin. vLLM's argparse rejects anything
#       it does not know, so a typo fails at boot rather than silently booting
#       the incumbent.
#   BATCH_INVARIANT    exported as VLLM_BATCH_INVARIANT, vLLM's batch-invariant
#       mode: deterministic reductions (single-split attention, the deterministic
#       scaled_mm/layernorm paths) instead of the fastest ones. 0 or 1 only —
#       vLLM parses it with int(), so an empty or "true" value raises at import.
#       Needs MAX_MODEL_LEN under ~100k; at 131072 the KV pool does not fit.
MOE_BACKEND="${MOE_BACKEND:-}"
BATCH_INVARIANT="${BATCH_INVARIANT:-0}"
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

# Both levers are refused loudly rather than passed through, because a boot that
# silently fell back to the incumbent would produce a header row that reports a
# variant's numbers for a boot that was not that variant — which is the exact
# failure this file exists to avoid measuring.
case "$BATCH_INVARIANT" in
    0|1) ;;
    *) die "BATCH_INVARIANT must be 0 or 1 (vLLM parses VLLM_BATCH_INVARIANT with int()), got '$BATCH_INVARIANT'" ;;
esac
if [[ -n "$MOE_BACKEND" && ! "$MOE_BACKEND" =~ ^[a-z0-9_]+$ ]]; then
    die "MOE_BACKEND must be one vLLM kernel name (triton, batched_triton, triton_unfused, marlin, ...), got '$MOE_BACKEND'"
fi
export VLLM_BATCH_INVARIANT="$BATCH_INVARIANT"

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
# One reading, shared with start-secondary.sh (#1316). The three lines this
# replaced were this script's own `nvidia-smi ... | tr -d ' '` copies of the same
# question the other GPU 2 tenant asks, and the file carried no comment saying
# which way round the two disagree when the card cannot be read — so the answer
# here, stated rather than inherited, is: djev does not boot. It is the live recall
# ranker (app/djev.py); an unreadable card is not evidence of a free one, and
# putting a chosen kernel variant onto a card it cannot see is the unattended
# restart an attended window owns (#1363). `set -euo pipefail` already took the
# script down on this failure — with no message at all — so staying down is the
# incumbent behaviour; what the guard adds is the sentence.
#
# shellcheck source=./gpu-mem.sh
source "$PROJECT_DIR/bin/gpu-mem.sh"
FREE_MIB="" TOTAL_MIB="" GPU_NAME=""
gpu_mem_read FREE_MIB TOTAL_MIB GPU_NAME "$GPU" || die "no readable GPU $GPU: nvidia-smi is missing, failing, or answering non-numerically; not booting into a card it cannot measure"

cat <<EOF
==> djev: DiffusionGemma 26B-A4B NVFP4
    gpu            $GPU  $GPU_NAME  (${FREE_MIB} MiB free of ${TOTAL_MIB})
    context        $MAX_MODEL_LEN    canvas $CANVAS    max seqs $MAX_SEQS
    kernels        moe ${MOE_BACKEND:-auto}    batch-invariant $VLLM_BATCH_INVARIANT
    kv dtype       $KV_CACHE_DTYPE    pool ${KV_CACHE_GB:-auto (the remainder)}
    ports          $PORT vllm, $STRUCTURED_PORT structured
    budget         weights ${WEIGHTS_MIB} + KV ${KV_MIB} + transient ${TRANSIENT_MIB}
                   + overhead ${OVERHEAD_MIB} = ${NEED_MIB} MiB
EOF

if (( FREE_MIB < NEED_MIB )); then
    echo >&2
    echo "refusing to start: GPU $GPU has ${FREE_MIB} MiB free, this needs ${NEED_MIB}." >&2
    GPU_HOLDERS=$(gpu_mem_holders "$GPU")
    if [[ -n "$GPU_HOLDERS" ]]; then
        echo "what is on the card:" >&2
        printf '%s\n' "$GPU_HOLDERS" | sed 's/^/  /' >&2
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
    ${MOE_BACKEND:+--moe-backend "$MOE_BACKEND"} \
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

# ── Warmup ────────────────────────────────────────────────────────────
# One dummy read as soon as the structured server answers, to absorb the
# Triton JIT of `_fill_logprob_token_ids_kernel` and `_topk_log_softmax_kernel`.
# That compile costs ~1 s and happens on the FIRST structured read after boot
# and never again — measured as exactly two occurrences in the whole of
# agent-djev.log's history, both on a first read, with zero more across a
# ~200-call benchmark over many shapes. The engine log asks for this itself
# ("consider extending warmup to cover this shape/config").
#
# Detached and entirely failure-tolerant: this must not delay or fail the
# boot, and a warmup that did would be worse than the 1 s it saves. Without
# it the first real caller after every restart pays the spike — and since
# 2026-09-20 the first caller is usually a shadow row, so the spike would
# land in the latency distribution the seams are being judged on.
(
    for _ in $(seq 1 60); do
        healthy "$STRUCTURED_PORT" && break
        sleep 1
    done
    curl -sf --max-time 30 -X POST "http://127.0.0.1:$STRUCTURED_PORT/v1/systemone" \
        -H 'content-type: application/json' \
        ${API_KEY:+-H "authorization: Bearer $API_KEY"} \
        -d '{"model":"'"$SERVED_NAME"'","state":"warmup","questions":{"ok":{"type":"noul","instructions":"Is this a warmup?"}}}' \
        >/dev/null 2>&1 \
        && echo "==> djev warmed (JIT absorbed)" \
        || echo "==> djev warmup skipped (structured server not ready)" >&2
) &

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
