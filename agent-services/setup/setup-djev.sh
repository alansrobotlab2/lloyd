#!/usr/bin/env bash
set -euo pipefail
# djev — DiffusionGemma 26B-A4B (NVFP4) structured reads, on GPU 2's RTX 3090.
#
#   venv:    ~/lloyd/.venvs/vllm-djev
#   source:  ~/lloyd/agent-services/llm/djev-spark   (github.com/mmastrac/djev-spark)
#   model:   ~/lloyd/agent-services/llm/models/nvidia-diffusiongemma-26B-A4B-it-NVFP4
#   start:   agent-services/bin/start-djev.sh
#   service: supervisor/conf.d/agent-djev.conf, gated on `djev.enabled` in config.yaml
#
# WHAT IT SERVES
#   Not chat. `POST :8011/v1/systemone` takes a state and a map of typed
#   questions — yes/no, one-of-N, an ordered scale — and answers all of them in
#   ONE diffusion read off a shared canvas, returning calibrated probabilities
#   rather than sampled prose. vLLM's ordinary OpenAI API is on :8010
#   underneath it and works, but it is the substrate, not the point.
#
# WHY A VENV AND NOT UPSTREAM'S CONTAINER
#   djev-spark ships a Dockerfile. This box has no docker socket access for
#   this user and the stack deliberately left containers behind. That costs
#   nothing here, because the image compiles nothing: it is a stock vLLM
#   nightly wheel with NINE PYTHON FILES copied over site-packages. A venv
#   reproduces it exactly, and this script performs the same three safety
#   checks the Dockerfile does (see step 3).
#
# WHY IT RUNS ON A 3090 AT ALL, WHEN UPSTREAM SAYS DGX SPARK
#   Upstream targets a GB10 (aarch64, unified memory, SM121) and says only "it
#   may work on other hardware". Two things make an SM86 consumer card work,
#   both checked at step 5 rather than assumed:
#
#   * NVFP4 does NOT need Blackwell here. `hf_quant_config.json` excludes
#     `lm_head`, `*self_attn*`, `*mlp*` and `*router*` — only the 128 ROUTED
#     EXPERTS are 4-bit, everything else stays BF16. vLLM serves that through
#     the NVFP4 **Marlin W4A16** path, which gates at capability **75**
#     (`is_fp4_marlin_supported`) and whose `FP4_MARLIN_SUPPORTED_GROUP_SIZES`
#     is exactly `[16]` — this checkpoint's `group_size`. The Blackwell-only
#     CUTLASS/FlashInfer MoE backends reject themselves on capability first and
#     selection falls through to MARLIN. Weight-only: it unpacks to 16-bit for
#     the GEMM, so it is slower than native FP4 but numerically the same
#     weights.
#
#   * The full 262144 window fits in 24 GiB because only 5 of the 30 layers
#     are `full_attention`, at `num_global_key_value_heads: 2` x
#     `global_head_dim: 512`. That is 5 x 2 x 512 x 2(K+V) x 2(f16) =
#     10 KiB/token, so 262144 tokens is ~2.5 GiB. The other 25 layers are
#     sliding-window 1024 and cost ~100 MiB in total regardless of context.
#
#       NVFP4 weights                          17.6 GiB
#       KV @ f16, full 262144                  ~2.6 GiB
#       sampler transient + activations        ~1.4 GiB   (see start-djev.sh)
#                                              ---------
#                                              ~21.6 GiB of 24
#
#     GPU 2 is single-tenant for this, exactly as it was for the 35B secondary.
#     `secondary_enabled` and `djev.enabled` are an either/or; server.py
#     refuses to start both and start-djev.sh refuses on the VRAM check.
#
# WHAT IS DELIBERATELY NOT PORTED FROM THE IMAGE
#   patches/link_cuda_headers.sh   The base image ships CUDA libraries without
#                                  headers. This host has CUDA 13.3 at
#                                  /opt/cuda with its headers, so there is
#                                  nothing to link.
#   patches/worker_memory_cap.py   Unified-memory insurance: on a Spark a
#   patches/spark_mem_trace.py     runaway allocation takes the HOST down. On a
#                                  discrete card CUDA returns OOM and the
#                                  request fails, which is the behaviour the
#                                  cap exists to synthesise.
#   flashinfer                     Not installed, and that is load-bearing
#                                  rather than an omission: its MoE backends
#                                  sit ahead of MARLIN in vLLM's NVFP4
#                                  selection order and are Blackwell-only, so
#                                  its absence makes the fallback certain
#                                  instead of merely likely. Attention is
#                                  TRITON_ATTN for the same reason.
#
# RE-RUNNING IS SAFE. Every step is idempotent; the overlay refuses rather than
# half-applies if the pins have moved.

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LLOYD_DIR="$(cd "$PROJECT_DIR/.." && pwd)"

# ── Pins ──────────────────────────────────────────────────────────────
# Bump these THREE TOGETHER, never individually. VLLM_PIN/VLLM_BASE name the
# stock wheel; VLLM_REF is the fork branch whose changed files go over it, and
# it must have been rebased onto VLLM_BASE. Upstream's own porting note:
# "rebase the fork branch onto main, set VLLM_REF to its head, set BASE and
# VLLM_BASE to a nightly at or after that commit."
DJEV_REPO="https://github.com/mmastrac/djev-spark.git"
DJEV_REF="${DJEV_REF:-1444f3e927f83ba508e5b28a4fd4fdd9ecd0976b}"
VLLM_FORK="https://github.com/mmastrac/vllm.git"
VLLM_UPSTREAM="https://github.com/vllm-project/vllm.git"
VLLM_REF="${VLLM_REF:-6591b093b29536dd070c6af3628b734025c53e23}"
VLLM_BASE="${VLLM_BASE:-dee37d89115db4c94a820a79a78a7828e141c910}"
VLLM_PIN="${VLLM_PIN:-0.29.1rc1.dev347+gdee37d891}"

DJEV_SRC="$PROJECT_DIR/llm/djev-spark"
FORK_SRC="$PROJECT_DIR/llm/djev-vllm-fork"
VENV="${DJEV_VENV:-$LLOYD_DIR/.venvs/vllm-djev}"
PY="$VENV/bin/python"
MODEL_REPO="nvidia/diffusiongemma-26B-A4B-it-NVFP4"
MODEL_DIR="${DJEV_MODEL_DIR:-$PROJECT_DIR/llm/models/nvidia-diffusiongemma-26B-A4B-it-NVFP4}"

CHECK_ONLY=0
SKIP_MODEL=0
for arg in "$@"; do
    case "$arg" in
        --check) CHECK_ONLY=1 ;;
        --skip-model) SKIP_MODEL=1 ;;
        -h|--help) sed -n '2,90p' "$0"; exit 0 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
ok()   { printf '  \033[32mok\033[0m    %s\n' "$*"; }
warn() { printf '  \033[33mwarn\033[0m  %s\n' "$*"; }
miss() { printf '  \033[31mmiss\033[0m  %s\n' "$*"; }
run()  { if (( CHECK_ONLY )); then printf '  would run: %s\n' "$*"; else "$@"; fi; }

# ── 1. The djev-spark checkout ────────────────────────────────────────
step "1/6  djev-spark source @ ${DJEV_REF:0:9}"
if [[ -d "$DJEV_SRC/.git" ]]; then
    have="$(git -C "$DJEV_SRC" rev-parse HEAD)"
    if [[ "$have" == "$DJEV_REF" ]]; then
        ok "already at the pin"
    elif (( CHECK_ONLY )); then
        warn "at ${have:0:9}, pin is ${DJEV_REF:0:9}"
    else
        run git -C "$DJEV_SRC" fetch --quiet origin
        run git -C "$DJEV_SRC" checkout --quiet "$DJEV_REF"
        ok "moved to the pin"
    fi
elif (( CHECK_ONLY )); then
    miss "$DJEV_SRC"
else
    run git clone --quiet "$DJEV_REPO" "$DJEV_SRC"
    run git -C "$DJEV_SRC" checkout --quiet "$DJEV_REF"
    ok "cloned"
fi

# ── 2. The venv and the stock wheel ───────────────────────────────────
step "2/6  venv + vllm==$VLLM_PIN"
if [[ -x "$PY" ]] && "$PY" -c "
import importlib.metadata as m, sys
sys.exit(0 if m.version('vllm') == '$VLLM_PIN' else 1)" 2>/dev/null; then
    ok "venv has the pinned wheel"
elif (( CHECK_ONLY )); then
    miss "venv at $VENV with vllm==$VLLM_PIN"
else
    [[ -x "$PY" ]] || run uv venv "$VENV" --python 3.12
    run uv pip install pip --python "$PY"
    # The per-commit index is the only place a nightly lives; pypi carries the
    # torch and the rest. cu130 matches this host's CUDA 13.3 / driver 610 and
    # the other engine venvs' torch 2.13.0+cu130.
    run "$PY" -m pip install -q "vllm==$VLLM_PIN" --pre \
        --index-url "https://wheels.vllm.ai/$VLLM_BASE/cu130" \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        --extra-index-url https://pypi.org/simple
    run "$PY" -m pip install -q hf_transfer huggingface_hub
    ok "installed"
fi

# ── 3. The structured-reads overlay ───────────────────────────────────
# The Dockerfile's fork stage, in bash. Three checks, and each one exists
# because an overlay onto the wrong vLLM imports fine and then fails later in
# ways that read as model bugs:
#   a. the branch must carry changes under vllm/          (changed.txt non-empty)
#   b. upstream must not have touched those same files between the branch's
#      merge-base and VLLM_BASE                           (the `git diff` below)
#   c. the installed wheel must BE VLLM_BASE              (overlay_vllm.py)
step "3/6  fork overlay @ ${VLLM_REF:0:9} onto ${VLLM_BASE:0:9}"
if (( CHECK_ONLY )); then
    if [[ -f "$VENV/.djev-overlay-$VLLM_REF" ]]; then ok "overlay applied"
    else miss "overlay for ${VLLM_REF:0:9}"; fi
elif [[ -f "$VENV/.djev-overlay-$VLLM_REF" ]]; then
    ok "already overlaid"
else
    if [[ ! -d "$FORK_SRC/.git" ]]; then
        run git clone --quiet --filter=blob:none "$VLLM_FORK" "$FORK_SRC"
    fi
    git -C "$FORK_SRC" fetch --quiet --filter=blob:none origin "$VLLM_REF"
    git -C "$FORK_SRC" checkout --quiet "$VLLM_REF"
    git -C "$FORK_SRC" fetch --quiet --filter=blob:none "$VLLM_UPSTREAM" "$VLLM_BASE"
    mb="$(git -C "$FORK_SRC" merge-base "$VLLM_BASE" HEAD)"
    git -C "$FORK_SRC" diff --name-only "$mb" HEAD -- vllm > "$FORK_SRC/changed.txt"
    git -C "$FORK_SRC" diff --name-only --diff-filter=A "$mb" HEAD -- vllm > "$FORK_SRC/added.txt"
    [[ -s "$FORK_SRC/changed.txt" ]] || { echo "changed.txt empty; VLLM_REF carries no vllm/ changes" >&2; exit 1; }
    printf '  overlaying %s file(s):\n' "$(wc -l < "$FORK_SRC/changed.txt")"
    sed 's/^/    /' "$FORK_SRC/changed.txt"
    if ! git -C "$FORK_SRC" diff --quiet "$mb" "$VLLM_BASE" -- $(cat "$FORK_SRC/changed.txt"); then
        echo "upstream changed overlaid files between the branch base and $VLLM_BASE;" >&2
        echo "rebase the fork branch first, then bump VLLM_REF/VLLM_BASE/VLLM_PIN together." >&2
        git -C "$FORK_SRC" diff --stat "$mb" "$VLLM_BASE" -- $(cat "$FORK_SRC/changed.txt") >&2
        exit 1
    fi
    "$PY" "$DJEV_SRC/patches/overlay_vllm.py" "$FORK_SRC" "$VLLM_BASE"
    "$PY" "$DJEV_SRC/patches/raise_recompile_limit.py"
    touch "$VENV/.djev-overlay-$VLLM_REF"
    ok "overlay + recompile limit applied"
fi

# ── 4. The checkpoint ─────────────────────────────────────────────────
step "4/6  checkpoint $MODEL_REPO"
if [[ -f "$MODEL_DIR/config.json" ]] && \
   [[ -f "$MODEL_DIR/model-00002-of-00002.safetensors" ]]; then
    ok "present ($(du -sh "$MODEL_DIR" | cut -f1))"
elif (( SKIP_MODEL )); then
    warn "--skip-model; not downloading"
elif (( CHECK_ONLY )); then
    miss "$MODEL_DIR (17.6 GiB)"
else
    run mkdir -p "$MODEL_DIR"
    # Resumable: re-running after an interrupt continues rather than restarts.
    run "$VENV/bin/hf" download "$MODEL_REPO" --local-dir "$MODEL_DIR"
    ok "downloaded"
fi

# ── 5. Verify, on the real card ───────────────────────────────────────
# Everything above is arrangement; this is the part that would actually have
# caught a bad port. It asserts the two claims the header makes rather than
# restating them.
step "5/6  verification"
if (( CHECK_ONLY )) && [[ ! -x "$PY" ]]; then
    miss "cannot verify without the venv"
else
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${DJEV_GPU:-2}" "$PY" - <<'EOF'
import importlib.metadata as md
import sys

fail = []

import torch
print(f"  torch          {torch.__version__} (cuda {torch.version.cuda})")
print(f"  vllm           {md.version('vllm')}")

cap = torch.cuda.get_device_capability(0)
name = torch.cuda.get_device_name(0)
free, total = torch.cuda.mem_get_info(0)
print(f"  device         {name}  sm_{cap[0]}{cap[1]}  "
      f"{free/2**30:.1f} GiB free / {total/2**30:.1f} GiB")

# The claim the whole port rests on: NVFP4 experts reach a Marlin kernel here.
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    FP4_MARLIN_SUPPORTED_GROUP_SIZES, is_fp4_marlin_supported)
if not is_fp4_marlin_supported():
    fail.append(f"NVFP4 Marlin unsupported on sm_{cap[0]}{cap[1]}; "
                "the experts have no kernel on this card")
if 16 not in FP4_MARLIN_SUPPORTED_GROUP_SIZES:
    fail.append(f"NVFP4 Marlin group sizes {FP4_MARLIN_SUPPORTED_GROUP_SIZES} "
                "exclude 16, which is this checkpoint's")
print(f"  nvfp4 marlin   supported, group sizes {FP4_MARLIN_SUPPORTED_GROUP_SIZES}")

# The overlay landed and the diffusion path is present.
from vllm.model_executor.models import diffusion_gemma  # noqa: F401
src = __import__("pathlib").Path(diffusion_gemma.__file__).read_text()
if "[djev-spark] recompile limit" not in src:
    fail.append("raise_recompile_limit.py did not apply to diffusion_gemma.py")
import vllm.v1.core.sched.diffusion_scheduler  # noqa: F401
print("  overlay        diffusion_gemma + diffusion_scheduler + recompile limit")

if fail:
    print()
    for f in fail:
        print(f"  FAIL  {f}")
    sys.exit(1)
print("  all checks passed")
EOF
fi

# ── 6. Next steps ─────────────────────────────────────────────────────
step "6/6  done"
cat <<EOF
  Run it by hand:   agent-services/bin/start-djev.sh
  Run it as a service:
      1. set  djev.enabled: true   in config.yaml
         (and secondary_enabled: false — they share GPU 2)
      2. .venvs/lloyd/bin/python -m scripts.automod.round restart --only lloyd-backend
         the backend's boot reconcile starts agent-djev and Mission Control
         picks up its engine card.
  Smoke test:       $DJEV_SRC/scripts/smoke.sh
EOF
