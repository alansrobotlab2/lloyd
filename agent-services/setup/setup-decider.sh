#!/usr/bin/env bash
set -euo pipefail
# decider — one-pass typed decisions, on GPU 2's RTX 3090. Two variants,
# one venv, one launcher (VARIANT=35b|4b):
#
#   venv:    ~/lloyd/.venvs/vllm-decider
#   models:  ~/lloyd/agent-services/llm/models/Mapika-decider-35b-a3b-nvfp4   (19.6 GiB)
#            ~/lloyd/agent-services/llm/models/Mapika-decider-4b-v2.1          ( 7.8 GiB)
#   start:   agent-services/bin/start-decider.sh
#   service: supervisor/conf.d/agent-decider.conf (autostart=false)
#   source:  github.com/Mapika/decider (decider-ai),
#            huggingface.co/Mapika/decider-35b-a3b-nvfp4, huggingface.co/Mapika/decider-4b
#
# THE TWO VARIANTS
#   35b  decider-35b-a3b v1, NVFP4 (Qwen3.5-35B-A3B-Base, 3B active). The
#        repo's own tables put it ahead on accuracy and calibration; on this
#        card it runs a quantized path nobody measured (below).
#   4b   decider-4b v2.1, bf16 (Qwen3.5-4B-Base, dense). Behind the 35B on the
#        repo's tables, reported ahead on independent leaderboards, and it runs
#        here EXACTLY as published: bf16, no quantization, a native Ampere
#        path, 7.8 GiB with room for concurrency. Its per-type temperatures
#        (choice 1.110, noul 1.560, score 1.287) need decider-ai >= 1.4.0,
#        which the pin below satisfies.
#   4b-v2 decider-4b v2, bf16: v2.1's predecessor, same architecture, one
#        global temperature (1.935). #1 on JevBench v1.4.2 (sealed 34.7%).
#   All are served by the same upstream server from the same venv.
#
# WHAT IT IS (both variants)
#   A candidate replacement for djev (DiffusionGemma 26B-A4B). Same wire
#   format — POST /v1/systemone with TypeSafe's state + typed questions — but
#   a different readout: an autoregressive Qwen3.5 fine-tune whose prompt ends
#   at the answer slot "Answer: (", read as the softmax of the option-letter
#   logits at one position, at the checkpoint's fitted temperature (35b: 1.08;
#   4b: one per answer type). No generation: one prefill per row, max_tokens=1.
#
#   Published numbers (the repo's docs/RESULTS.md, bf16 on a B300): the 35B
#   above decider-4b v2.1 on every accuracy table and on calibration; the NVFP4
#   build 1.0-1.5 points under bf16 through vLLM. Both within three JevBench
#   public hard-tier items of djev (0.676 / 0.649 / 0.676). Which of the three
#   is best HERE is what the recall eval has to say — nothing below assumes it.
#
# THE SERVER IS UPSTREAM'S, UNMODIFIED
#   decider.serve_vllm (decider-ai 1.5.0) runs vLLM in-process (AsyncLLM) and
#   serves /v1/systemone, /v1/models, /health and /stats from one uvicorn
#   port. Upstream's install note, followed exactly: vLLM 0.29.0 needs numpy 2
#   and pins torch 2.13, decider-ai's own requirements pin numpy<2, so the
#   package goes in with --no-deps on top of a vLLM environment. Pinned to the
#   git commit rather than the PyPI version so the pin names the code read.
#
# WHY AN NVFP4 CHECKPOINT RUNS ON AN SM86 CARD (35b only)
#   The checkpoint is ModelOpt NVFP4, W4A4, made for Blackwell. vLLM serves
#   ModelOpt NVFP4 below SM100 through the NVFP4 Marlin W4A16 path
#   (`is_fp4_marlin_supported`, capability >= 75, group size 16 = this
#   checkpoint's): weight-only, activations stay bf16. That is the path djev's
#   NVFP4 experts already run on this very card. Two consequences:
#     * numerically this is NOT the build the model card measured (that was
#       native FP4 activations on a B300); weight-only is, if anything, closer
#       to bf16, but it is unmeasured, which is why the eval comes first.
#     * speed is Marlin's, not native FP4's.
#   The KV cache is not quantized (`kv_cache_quant_algo: null`), so the FP8-KV
#   wall djev hit on this card (SM89+ only) does not exist here.
#   Step 5 checks the Marlin claim on the real card rather than trusting this.
#
# WHY THE 35B FITS ON 24 GiB (and how tightly; the 4B leaves ~14 GiB)
#       NVFP4 weights (5 safetensors)              19.6 GiB
#       card as CUDA sees it                      ~23.7 GiB   x 0.97 = ~23.0
#       left for activations + graphs + cache      ~3.4 GiB
#   The cache is cheap: 10 of 40 layers are full attention, 2 KV heads x 256
#   head dim x 2 (K+V) x 2 bytes = 20 KiB/token; the other 30 are Gated
#   DeltaNet with a fixed ~30 MiB of recurrent state PER SEQUENCE. So context
#   is cheap and concurrency is not — start-decider.sh's header has the budget.
#   GPU 2 is single-tenant: this OR djev OR the Qwen3.6 secondary.
#
# RE-RUNNING IS SAFE. Every step is idempotent.
#
#   --check          report what is missing, change nothing
#   --skip-model     everything but the downloads
#   --variant V      35b, 4b, 4b-v2 or all (default all); repeatable

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LLOYD_DIR="${LLOYD_DIR:-$(cd "$PROJECT_DIR/.." && pwd)}"
# Models live under the production tree's agent-services, whichever checkout
# this script runs from, so a worktree does not download a second 19.6 GiB.
MODELS_DIR="${MODELS_DIR:-$LLOYD_DIR/agent-services/llm/models}"

# ── Pins ──────────────────────────────────────────────────────────────
# vLLM 0.29.0 is the version decider.serve_vllm was written and measured
# against (docs/SERVING.md §8). DECIDER_REF is decider-ai 1.5.0. MODEL_REV is
# the Hub commit of each model repo on 2026-09-25 — decider-4b's `main` is
# whichever version was released last, so the revision is what makes the
# directory name's "v2.1" true.
VLLM_PIN="${VLLM_PIN:-0.29.0}"
DECIDER_REPO="https://github.com/Mapika/decider.git"
DECIDER_REF="${DECIDER_REF:-a5120cce45b9ff70964fac54ea6e8c1ac5b08c7f}"
# variant -> repo, revision, directory, the file whose presence means "done"
# 4b-v2 is the same repo at the last commit before v2.1 replaced its weights
# (49564dd, a card edit on top of 7ab294c "decider-4b v2"): JevBench v1.4.2's
# #1 entry is v2, not v2.1, so both are kept.
declare -A V_REPO=( [35b]="Mapika/decider-35b-a3b-nvfp4"            [4b]="Mapika/decider-4b"                        [4b-v2]="Mapika/decider-4b" )
declare -A V_REV=(  [35b]="798555c06e419c4638c9ebd06c78ed8b5e92c868" [4b]="eb5fbdfc9448473ec25e399882912863afbdb70e" [4b-v2]="49564ddcfccafb6db563eb757c1d41e6c78dcb56" )
declare -A V_DIR=(  [35b]="Mapika-decider-35b-a3b-nvfp4"            [4b]="Mapika-decider-4b-v2.1"                   [4b-v2]="Mapika-decider-4b-v2" )
declare -A V_LAST=( [35b]="model-00005-of-00005.safetensors"        [4b]="model.safetensors"                        [4b-v2]="model.safetensors" )
declare -A V_SIZE=( [35b]="19.6 GiB"                                [4b]="7.8 GiB"                                  [4b-v2]="7.8 GiB" )

VENV="${DECIDER_VENV:-$LLOYD_DIR/.venvs/vllm-decider}"
PY="$VENV/bin/python"

CHECK_ONLY=0
SKIP_MODEL=0
VARIANTS=()
while (( $# )); do
    case "$1" in
        --check) CHECK_ONLY=1 ;;
        --skip-model) SKIP_MODEL=1 ;;
        --variant) shift; VARIANTS+=("${1:-}") ;;
        --variant=*) VARIANTS+=("${1#*=}") ;;
        -h|--help) sed -n '2,80p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done
(( ${#VARIANTS[@]} )) || VARIANTS=(all)
if [[ " ${VARIANTS[*]} " == *" all "* ]]; then VARIANTS=(35b 4b 4b-v2); fi
for v in "${VARIANTS[@]}"; do
    [[ -n "${V_REPO[$v]:-}" ]] || { echo "unknown variant '$v' (35b, 4b, 4b-v2, all)" >&2; exit 2; }
done
model_dir() { echo "$MODELS_DIR/${V_DIR[$1]}"; }

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
ok()   { printf '  \033[32mok\033[0m    %s\n' "$*"; }
warn() { printf '  \033[33mwarn\033[0m  %s\n' "$*"; }
miss() { printf '  \033[31mmiss\033[0m  %s\n' "$*"; }
run()  { if (( CHECK_ONLY )); then printf '  would run: %s\n' "$*"; else "$@"; fi; }

# ── 1. The venv and vLLM ──────────────────────────────────────────────
step "1/5  venv + vllm==$VLLM_PIN"
if [[ -x "$PY" ]] && "$PY" -c "
import importlib.metadata as m, sys
sys.exit(0 if m.version('vllm') == '$VLLM_PIN' else 1)" 2>/dev/null; then
    ok "venv has vllm $VLLM_PIN"
elif (( CHECK_ONLY )); then
    miss "venv at $VENV with vllm==$VLLM_PIN"
else
    [[ -x "$PY" ]] || run uv venv "$VENV" --python 3.12 --seed
    # cu130 torch, matching this host's CUDA 13.3 / driver 610 and every other
    # engine venv's torch 2.13.0+cu130. The torch index goes first so pip does
    # not settle for a different CUDA build off PyPI.
    run "$PY" -m pip install -q "vllm==$VLLM_PIN" \
        --extra-index-url https://download.pytorch.org/whl/cu130
    ok "installed"
fi

# ── 2. The server's own requirements ──────────────────────────────────
step "2/5  fastapi, uvicorn, jinja2, huggingface_hub"
if [[ -x "$PY" ]] && "$PY" -c "import fastapi, uvicorn, jinja2, huggingface_hub" 2>/dev/null \
   && [[ -x "$VENV/bin/hf" ]]; then
    ok "present"
elif (( CHECK_ONLY )); then
    miss "server requirements"
else
    run "$PY" -m pip install -q fastapi "uvicorn[standard]" jinja2 huggingface_hub
    ok "installed"
fi

# ── 3. decider-ai, --no-deps ──────────────────────────────────────────
step "3/5  decider-ai @ ${DECIDER_REF:0:9} (--no-deps)"
have_ref=""
if [[ -x "$PY" ]]; then
    have_ref="$("$PY" - <<'EOF' 2>/dev/null || true
import importlib.metadata as m, json
d = m.distribution("decider-ai")
u = json.loads(d.read_text("direct_url.json") or "{}")
print(u.get("vcs_info", {}).get("commit_id", ""))
EOF
)"
fi
if [[ "$have_ref" == "$DECIDER_REF" ]]; then
    ok "installed at the pin"
elif (( CHECK_ONLY )); then
    miss "decider-ai at ${DECIDER_REF:0:9} (have: ${have_ref:-none})"
else
    run "$PY" -m pip install -q --no-deps --force-reinstall \
        "decider-ai @ git+$DECIDER_REPO@$DECIDER_REF"
    ok "installed"
fi

# ── 4. The checkpoints ────────────────────────────────────────────────
for v in "${VARIANTS[@]}"; do
    MODEL_DIR="$(model_dir "$v")"
    step "4/5  checkpoint ${V_REPO[$v]} @ ${V_REV[$v]:0:9}  ($v)"
    if [[ -f "$MODEL_DIR/config.json" && -f "$MODEL_DIR/decider_config.json" ]] && \
       [[ -f "$MODEL_DIR/${V_LAST[$v]}" ]] && \
       ! compgen -G "$MODEL_DIR/.cache/huggingface/download/*.incomplete" >/dev/null; then
        ok "present ($(du -sh "$MODEL_DIR" | cut -f1))"
    elif (( SKIP_MODEL )); then
        warn "--skip-model; not downloading"
    elif (( CHECK_ONLY )); then
        miss "$MODEL_DIR (${V_SIZE[$v]})"
    else
        run mkdir -p "$MODEL_DIR"
        # Resumable: re-running after an interrupt continues rather than restarts.
        HF_XET_HIGH_PERFORMANCE=1 run "$VENV/bin/hf" download "${V_REPO[$v]}" \
            --revision "${V_REV[$v]}" --local-dir "$MODEL_DIR"
        ok "downloaded"
    fi
done

# ── 5. Verify, against the real card ──────────────────────────────────
# The claims the header makes, asserted rather than restated. Run from /
# so no checkout's decider/ shadows the installed one (docs/SERVING.md §3.1).
#
# NO CUDA CONTEXT. The card is read through nvidia-smi and vLLM's NVML-backed
# platform, never torch.cuda: GPU 2 is normally full with djev at 0.97, and the
# first cut of this step died with "CUDA error: out of memory" creating a
# context on it — a failed verification of a correct install.
step "5/5  verification"
if [[ ! -x "$PY" ]]; then
    miss "cannot verify without the venv"
else
    GPU_ROW="$(nvidia-smi --id="${DECIDER_GPU:-2}" \
        --query-gpu=name,compute_cap,memory.total,memory.free --format=csv,noheader,nounits)"
    (cd / && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${DECIDER_GPU:-2}" \
        GPU_ROW="$GPU_ROW" MODEL_DIRS="$(for v in "${VARIANTS[@]}"; do model_dir "$v"; done)" "$PY" - <<'EOF'
import importlib.metadata as md
import json, os, pathlib, sys

fail = []

import torch
print(f"  torch          {torch.__version__} (cuda {torch.version.cuda})")
print(f"  vllm           {md.version('vllm')}")
print(f"  decider-ai     {md.version('decider-ai')}")

name, cc, total_mib, free_mib = [x.strip() for x in os.environ["GPU_ROW"].split(",")]
total = int(total_mib) * 2**20
print(f"  device         {name}  sm_{cc.replace('.', '')}  "
      f"{int(free_mib)/1024:.1f} GiB free / {int(total_mib)/1024:.1f} GiB (nvidia-smi)")

# The claim the whole port rests on: NVFP4 reaches a Marlin kernel here.
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    FP4_MARLIN_SUPPORTED_GROUP_SIZES, is_fp4_marlin_supported)
if not is_fp4_marlin_supported():
    fail.append(f"NVFP4 Marlin unsupported on sm_{cc}")
if 16 not in FP4_MARLIN_SUPPORTED_GROUP_SIZES:
    fail.append(f"NVFP4 Marlin group sizes {FP4_MARLIN_SUPPORTED_GROUP_SIZES} exclude 16")
print(f"  nvfp4 marlin   supported={is_fp4_marlin_supported()}, "
      f"group sizes {FP4_MARLIN_SUPPORTED_GROUP_SIZES}")

from vllm.model_executor.models.registry import ModelRegistry

# The two server-side hooks decider.serve_vllm needs from vLLM.
import inspect
from vllm import SamplingParams
if "logprob_token_ids" not in inspect.signature(SamplingParams).parameters:
    fail.append("SamplingParams has no logprob_token_ids")
import decider.serve_vllm as SV  # noqa: F401  (imports fastapi, the prompt builders, systemone)
import decider.vllm_worker as W
print(f"  server         decider.serve_vllm imports; logprob id cap {W.logprob_ids_cap()}")

# Each requested checkpoint that is here. The architecture must be one vLLM
# registers; the budget is weights against this card's share. The CUDA total
# is ~0.3 GiB under nvidia-smi's, so this errs toward refusing.
archs = set(ModelRegistry.get_supported_archs())
for d in os.environ["MODEL_DIRS"].split():
    mdir = pathlib.Path(d)
    if not (mdir / "config.json").exists():
        print(f"  checkpoint     {mdir.name}: not downloaded yet")
        continue
    cfg = json.loads((mdir / "decider_config.json").read_text())
    arch = json.loads((mdir / "config.json").read_text())["architectures"][0]
    qpath = mdir / "hf_quant_config.json"
    q = json.loads(qpath.read_text())["quantization"] if qpath.exists() else None
    shards = sorted(mdir.glob("*.safetensors"))
    gib = sum(p.stat().st_size for p in shards) / 2**30
    quant = f"{q['quant_algo']} g{q['group_size']} kv={q.get('kv_cache_quant_algo')}" if q else "bf16"
    temps = cfg.get("temperature_by_type") or cfg.get("temperature")
    print(f"  checkpoint     {cfg.get('version')}  {arch} (registered: {arch in archs})  {quant}  T={temps}  "
          f"{len(shards)} shard(s) {gib:.2f} GiB")
    if arch not in archs:
        fail.append(f"{mdir.name}: vLLM does not register {arch}")
    if q and q.get("kv_cache_quant_algo"):
        fail.append(f"{mdir.name}: asks for a quantized KV cache; SM86 has no FP8")
    if gib > total / 2**30 * 0.97 - 1.5:
        fail.append(f"{mdir.name}: {gib:.1f} GiB of weights leaves under 1.5 GiB on this card")

if fail:
    print()
    for f in fail:
        print(f"  FAIL  {f}")
    sys.exit(1)
print("  all checks passed")
EOF
    )
fi

cat <<EOF

  Run it by hand (djev must be stopped first; they share GPU 2):
      VARIANT=35b agent-services/bin/start-decider.sh     # or VARIANT=4b
  Smoke test once /health answers:
      curl -s localhost:8012/v1/systemone -H 'content-type: application/json' \\
        -d '{"state":"My card was charged twice.","questions":{"dept":{"type":"choice","instructions":"Which department?","criteria":{"billing":null,"technical support":null,"sales":null}}}}'
EOF
