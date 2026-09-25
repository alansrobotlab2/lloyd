#!/usr/bin/env bash
# decider — typed decisions on GPU 2's RTX 3090, one of two checkpoints:
#
#   VARIANT=35b  decider-35b-a3b v1, NVFP4, weight-only Marlin   (the default)
#   VARIANT=4b   decider-4b v2.1, bf16, exactly as published
#   VARIANT=4b-v2  decider-4b v2, bf16 (v2.1's predecessor; JevBench #1)
#
#   :8012  POST /v1/systemone, GET /health /stats /v1/models   (one process)
#
# Same server, same venv, same port: only one runs at a time, on one card.
#
# upstream's decider.serve_vllm (github.com/Mapika/decider, decider-ai 1.5.0)
# with vLLM in-process. Build the venv and fetch the checkpoint first with
# setup/setup-decider.sh, whose header says why an NVFP4 checkpoint runs on an
# SM86 card at all (weight-only Marlin, the path djev's experts already use).
#
# NOT A DROP-IN FOR djev YET. It answers the same request shape djev's :8011
# does, and deliberately on a different port: app/djev.py reads `diagnostics`
# (label_mass, argmax_is_label, timing, chunks) that decider does not send, and
# pointing the backend here is an integration change, not a port swap.
#
# GPU 2 IS SINGLE-TENANT: this, djev or the Qwen3.6 secondary. The preflight
# refuses rather than racing another tenant onto the card.
#
# THE SETTINGS THAT ARE NOT UPSTREAM'S, AND WHY
#
#   Upstream's defaults are sized for a B300 (180 GB). On 24 GiB, for the 35B:
#
#       card as CUDA sees it                     ~23.7 GiB
#       x GPU_UTIL 0.97                          ~23.0 GiB   (vLLM's budget)
#       NVFP4 weights, resident as Marlin         19.6 GiB
#       left for activations, graphs and cache    ~3.4 GiB
#
#   GPU_UTIL  0.90 -> 0.97.  At 0.90 the remainder is ~1.7 GiB, and the
#       profiling step plus CUDA graphs take most of that. 0.97 is djev's number
#       on the same card, for the same reason: GPU 2 drives no display.
#
#   MAX_NUM_SEQS  128 -> 16.  What costs memory on this model is concurrency,
#       not context. 30 of the 40 layers are Gated DeltaNet, each carrying a
#       fixed recurrent state per SEQUENCE (32 heads x 128 x 128, plus its conv
#       state) — ~30 MiB a sequence over the 30 layers — and vLLM also reserves
#       a CUDA graph per batch size up to this. The full-attention layers are
#       cheap: 10 layers x 2 KV heads x 256 dim x 2 (K+V) x 2 bytes = 20 KiB a
#       token, so a whole 40k-token row is 0.8 GiB. 16 rows at once is what one
#       recall's worth of candidates can use; more queue in vLLM's scheduler
#       rather than failing.
#
#   MAX_BATCHED_TOKENS  16384 -> 4096.  The prefill chunk sets the activation
#       peak of the profiling run; at 16384 the MoE's intermediate buffers alone
#       would eat the remainder. 4096 is the primary's chunk for the same kind of
#       reason (CLAUDE.md, "Primary throughput"). A long row is prefilled in
#       chunks; prefix caching keeps shared states from being prefilled twice.
#
#   MAX_MODEL_LEN  40960 (upstream's).  The state is cut at 32768 tokens and a
#       question block adds up to 4096; djev's largest production prompt so far
#       was under 5,000 tokens.
#
#   VLLM_PORT  5310.  vLLM's internal ZMQ port; the primary keeps the default
#       and djev takes 5300, so all three can be named without colliding.
#
#   THE 4B'S DEFAULTS.  7.8 GiB of bf16 weights leaves ~14 GiB, so the two
#       concurrency knobs go back up: 64 seqs (its DeltaNet state is 24 layers,
#       smaller per sequence than the 35B's 30) and 8192-token chunks. The
#       cache is 8 full-attention layers x 4 KV heads x 256 x 2 x 2 bytes =
#       32 KiB a token, ~0.3M tokens in what is left. A whole recall's rows
#       (32 candidates x 4 levels, see architecture/djev.md §14) fit in two
#       scheduler waves instead of eight.
#
# UNMEASURED ON THIS CARD. None of the above has booted here yet. If the
# profiling step OOMs, lower MAX_NUM_SEQS first (8), then MAX_MODEL_LEN
# (32768); if it boots with KV to spare, the log's "GPU KV cache size" line is
# the number to raise MAX_NUM_SEQS against.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LLOYD_DIR="$(cd "$PROJECT_DIR/.." && pwd)"

VENV="${DECIDER_VENV:-$LLOYD_DIR/.venvs/vllm-decider}"
PY="$VENV/bin/python"
VARIANT="${VARIANT:-35b}"
case "$VARIANT" in
    35b) DEF_DIR=Mapika-decider-35b-a3b-nvfp4 DEF_SEQS=16 DEF_CHUNK=4096 DEF_LABEL="decider-35b-a3b NVFP4 (weight-only Marlin on SM86)" ;;
    4b)  DEF_DIR=Mapika-decider-4b-v2.1       DEF_SEQS=64 DEF_CHUNK=8192 DEF_LABEL="decider-4b v2.1 bf16" ;;
    4b-v2) DEF_DIR=Mapika-decider-4b-v2       DEF_SEQS=64 DEF_CHUNK=8192 DEF_LABEL="decider-4b v2 bf16" ;;
    *)   echo "start-decider: VARIANT must be 35b, 4b or 4b-v2, got '$VARIANT'" >&2; exit 2 ;;
esac
MODEL="${DECIDER_MODEL_DIR:-$PROJECT_DIR/llm/models/$DEF_DIR}"

GPU="${GPU:-2}"
PORT="${PORT:-8012}"
GPU_UTIL="${GPU_UTIL:-0.97}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-$DEF_SEQS}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-$DEF_CHUNK}"
MAX_STATE_TOKENS="${MAX_STATE_TOKENS:-32768}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
# What must be free on the card beyond the weights before a boot is worth
# trying: the ~0.5 GiB CUDA context, a minimal graph set, and one full-length
# row of cache. Below this the profiling step fails anyway, only slower.
HEADROOM_MIB="${HEADROOM_MIB:-2048}"

export PATH="$VENV/bin:/opt/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/lib:/opt/cuda/targets/x86_64-linux/lib:/opt/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_HOME="/opt/cuda"
# Mandatory. Without it the runtime reorders devices by capability and
# CUDA_VISIBLE_DEVICES=2 lands on a different card than nvidia-smi's index 2.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_PORT="${VLLM_PORT:-5310}"
export TOKENIZERS_PARALLELISM=false
# decider.serve_vllm reads everything from the environment.
export DECIDER_MODEL="$MODEL"
export DECIDER_MAX_STATE_TOKENS="$MAX_STATE_TOKENS"
export DECIDER_VLLM_MAX_MODEL_LEN="$MAX_MODEL_LEN"
export DECIDER_VLLM_GPU_MEMORY_UTILIZATION="$GPU_UTIL"
export DECIDER_VLLM_MAX_NUM_SEQS="$MAX_NUM_SEQS"
export DECIDER_VLLM_MAX_BATCHED_TOKENS="$MAX_BATCHED_TOKENS"
export DECIDER_VLLM_ENFORCE_EAGER="$ENFORCE_EAGER"
# The layout and the temperature come from the checkpoint's decider_config.json
# (plain layout; 35b T 1.08, 4b a temperature per answer type — setting
# DECIDER_TEMPERATURE would switch that map off). Neither is set here: a model read in a layout it was
# not trained on gives wrong probabilities (docs/SERVING.md §3).
unset DECIDER_LAYOUT DECIDER_TEMPERATURE
# `python -m` / uvicorn put the working directory first on sys.path, and a
# checkout with its own decider/ would shadow the installed package.
export PYTHONSAFEPATH=1
cd /

die() { echo "start-decider: $*" >&2; exit 2; }

# ── Preflight ─────────────────────────────────────────────────────────
[[ -x "$PY" ]] || die "no venv at $VENV — run setup/setup-decider.sh"
"$PY" -c "import decider.serve_vllm" 2>/dev/null || \
    die "decider.serve_vllm does not import from $VENV — run setup/setup-decider.sh"
[[ -f "$MODEL/config.json" && -f "$MODEL/decider_config.json" ]] || \
    die "no checkpoint at $MODEL — run setup/setup-decider.sh"

WEIGHTS_MIB=$(du -sm --apparent-size "$MODEL"/*.safetensors 2>/dev/null | awk '{s+=$1} END {print s+0}')
(( WEIGHTS_MIB > 0 )) || die "no safetensors under $MODEL"
NEED_MIB=$(( WEIGHTS_MIB + HEADROOM_MIB ))

# One reading of the card, shared with the other GPU 2 tenants (#1316). An
# unreadable card is not evidence of a free one: do not boot.
# shellcheck source=./gpu-mem.sh
source "$PROJECT_DIR/bin/gpu-mem.sh"
FREE_MIB="" TOTAL_MIB="" GPU_NAME=""
gpu_mem_read FREE_MIB TOTAL_MIB GPU_NAME "$GPU" || \
    die "no readable GPU $GPU: nvidia-smi is missing, failing, or answering non-numerically"

cat <<EOF
==> decider: $DEF_LABEL  (VARIANT=$VARIANT)
    gpu            $GPU  $GPU_NAME  (${FREE_MIB} MiB free of ${TOTAL_MIB})
    context        $MAX_MODEL_LEN (state cap $MAX_STATE_TOKENS)    max seqs $MAX_NUM_SEQS
    prefill chunk  $MAX_BATCHED_TOKENS    gpu util $GPU_UTIL    eager $ENFORCE_EAGER
    port           $PORT
    budget         weights ${WEIGHTS_MIB} + headroom ${HEADROOM_MIB} = ${NEED_MIB} MiB
EOF

if (( FREE_MIB < NEED_MIB )); then
    echo >&2
    echo "refusing to start: GPU $GPU has ${FREE_MIB} MiB free, this needs ${NEED_MIB}." >&2
    GPU_HOLDERS=$(gpu_mem_holders "$GPU")
    if [[ -n "$GPU_HOLDERS" ]]; then
        echo "what is on the card:" >&2
        printf '%s\n' "$GPU_HOLDERS" | sed 's/^/  /' >&2
        echo >&2
        echo "GPU $GPU holds one tenant. If that is djev, stop it first (djev.enabled: false" >&2
        echo "in config.yaml and restart the backend, or supervisorctl stop agent-djev)." >&2
    fi
    exit 2
fi

# ── Serve ─────────────────────────────────────────────────────────────
# exec, so supervisord's signal reaches uvicorn directly and its lifespan
# shuts the engine down. The port answers only after start-up finishes: the
# weights, the profiling step, the graphs and serve_vllm's own warm-up (one row
# per option-count width, then a shared-prefix batch), so /health failing to
# connect during a boot is normal.
exec "$VENV/bin/uvicorn" decider.serve_vllm:app \
    --host 127.0.0.1 --port "$PORT" --log-level info
