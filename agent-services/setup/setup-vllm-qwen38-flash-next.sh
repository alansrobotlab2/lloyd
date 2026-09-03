#!/usr/bin/env bash
set -euo pipefail

# Dedicated vLLM venv for Inferact/Qwen3.8-Flash-Next-NVFP4 with PLE CPU offload.
#
#   venv:  ~/lloyd/.venvs/vllm-qwen38-flash-next
#   start: bin/start-qwen38-flash-next.sh
#   model: setup/setup-qwen38-flash-next.sh
#
# WHY THIS IS NOT JUST `pip install vllm --pre`
#   Qwen3.8-Flash-Next itself IS on vLLM main (PR #53896, merged 2026-08-31) and
#   IS in the public nightly — vllm/models/qwen4_exp/** ships in the wheel.
#   What is NOT on main is the thing that makes it fit on one 96 GiB card:
#   VLLM_PLE_CPU_OFFLOAD, which keeps the 51B N-gram table in host RAM. That
#   lives in open PR #53899 (peakcrosser7/vllm @ release/qwen38next_offload).
#   Verified against the 2026-09-03 cu130 nightly: `grep PLE_CPU_OFFLOAD` over
#   the whole wheel returns nothing. Without offload this checkpoint wants
#   ~170 GiB of VRAM.
#
#   The vLLM recipe says "vLLM 0.29.0+, PyPI installation is not supported, use
#   the vllm/vllm-openai:qwen38-flash-next image". That is only half true, and
#   the useful half is: you need code that is not in a release. But PR #53899 is
#   19 files and ALL of them are pure Python — no CUDA, no C++, nothing that
#   needs compiling. So instead of a 1-2 hour source build or a Docker daemon
#   this host does not run, we:
#
#     1. install the prebuilt per-commit wheel for the PR's OWN base commit
#        (45aed9b0c), so the compiled kernels match the branch's Python exactly;
#     2. overlay the branch's 14 vllm/ files on top;
#     3. overlay davidtai/vllm PR #10 (4 files) for the TP=1 rendezvous fix;
#     4. patch the sm_120 PDL hang.
#
#   Pinning to the PR's base commit rather than today's nightly is the point of
#   step 1: ple_layer.py is a +587/-51 modification against THAT tree, and main
#   has moved since (the PR reports mergeable_state=dirty). Overlaying it onto a
#   newer nightly would mix two versions of the same file's neighbours.
#
# WHEN THIS BECOMES UNNECESSARY
#   When #53899 merges, this collapses to a normal nightly install and the
#   overlay steps can be deleted. Check first:
#     curl -s https://api.github.com/repos/vllm-project/vllm/pulls/53899 | grep '"merged"'

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VLLM_VENV="${VLLM_VENV:-$HOME/lloyd/.venvs/vllm-qwen38-flash-next}"

# The commit on vllm main that the offload branch was actually cut from — i.e.
# the PARENT of the branch's first commit (d5002fb009bc^). The per-commit wheel
# index is keyed by full SHA and is what makes the overlay safe.
#
# DO NOT use the GitHub API's pulls/53899 `base.sha` here. That field is the base
# *branch head* at PR creation (45aed9b0c, dev188), not the merge-base, and main
# had already moved past the branch point. Pinning to it produced a wheel whose
# InputBatch lacks `max_seq_len_np`, while the branch's overlaid model_runner.py
# passes it — the engine loaded 75.1 GiB, captured CUDA graphs, allocated KV, and
# only then died in warmup with:
#   TypeError: InputBatch.__init__() got an unexpected keyword argument 'max_seq_len_np'
# Verified 2026-09-03. If you ever re-pin, get the parent commit, not base.sha:
#   curl -s https://api.github.com/repos/peakcrosser7/vllm/commits/<first-branch-commit> \
#     | python3 -c "import json,sys;print(json.load(sys.stdin)['parents'][0]['sha'])"
BASE_SHA="c5d840ff6a50f544fca8524e8caffda9f63b7728"
VLLM_PIN="0.28.1rc1.dev157+gc5d840ff6"
# release/qwen38next_offload @ "clean up nvtx code"
OFFLOAD_SHA="ffc445f8b2e9"
# davidtai/vllm fix/ple-cpu-offload-tp1-startup-deadlock
RENDEZVOUS_SHA="600a9fd411b0"

echo "=== vLLM venv for Qwen3.8-Flash-Next (NVFP4 + PLE CPU offload) ==="

if [[ ! -x /opt/cuda/bin/nvcc ]]; then
    echo "ERROR: CUDA toolkit not found at /opt/cuda"; exit 1
fi
echo "System CUDA: $(/opt/cuda/bin/nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')"

if [[ -d "$VLLM_VENV" && "${RECREATE:-}" == "1" ]]; then
    echo "RECREATE=1 — removing $VLLM_VENV"; rm -rf "$VLLM_VENV"
fi
if [[ ! -d "$VLLM_VENV" ]]; then
    echo "Creating venv at $VLLM_VENV (Python 3.12)..."
    uv venv "$VLLM_VENV" --python 3.12
fi
PY="$VLLM_VENV/bin/python"
uv pip install pip --python "$PY" >/dev/null

echo ""
echo "=== 1/4 Installing vLLM $VLLM_PIN (per-commit wheel, base of PR #53899) ==="
"$PY" -m pip install "vllm==$VLLM_PIN" --pre \
    --index-url "https://wheels.vllm.ai/$BASE_SHA/cu130" \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --extra-index-url https://pypi.org/simple
"$PY" -m pip install -q ninja hf_transfer huggingface_hub

SP="$("$PY" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')"
echo "site-packages: $SP"

fetch_overlay() {  # repo sha "file file file"
    local repo="$1" sha="$2" files="$3" f
    for f in $files; do
        mkdir -p "$SP/$(dirname "$f")"
        curl -sfL "https://raw.githubusercontent.com/$repo/$sha/$f" -o "$SP/$f" \
            || { echo "  FAIL $f"; return 1; }
        echo "  ok  $f"
    done
}

echo ""
echo "=== 2/4 Overlaying PR #53899 (PLE offload) from peakcrosser7/vllm@$OFFLOAD_SHA ==="
fetch_overlay peakcrosser7/vllm "$OFFLOAD_SHA" "
vllm/compilation/passes/utility/fix_functionalization.py
vllm/config/parallel.py
vllm/envs.py
vllm/model_executor/layers/ple_offload_layer.py
vllm/model_executor/model_loader/weight_utils.py
vllm/models/qwen4_exp/nvidia/ple_layer.py
vllm/v1/executor/multiproc_executor.py
vllm/v1/executor/uniproc_executor.py
vllm/v1/ple_offload/__init__.py
vllm/v1/ple_offload/connector.py
vllm/v1/ple_offload/protocol.py
vllm/v1/ple_offload/worker.py
vllm/v1/worker/gpu/model_runner.py
vllm/v1/worker/gpu_worker.py"

echo ""
echo "=== 3/4 Overlaying davidtai PR #10 (TP=1 startup rendezvous deadlock) ==="
# Supersedes three files from step 2 — davidtai's branch is based on #53899, so
# these already contain the offload code plus the bounded ACK barrier. Order
# matters: this must run after step 2, never before.
fetch_overlay davidtai/vllm "$RENDEZVOUS_SHA" "
vllm/v1/engine/utils.py
vllm/v1/ple_offload/connector.py
vllm/v1/ple_offload/protocol.py
vllm/v1/ple_offload/worker.py"

echo ""
echo "=== 4/4 Patching sm_120 PDL hang in the QSA metadata kernel ==="
# current_platform.is_arch_support_pdl() is `major >= 9` (vllm/platforms/cuda.py),
# so sm_120 (major 12) reports True. The QSA metadata kernel then launches with
# programmatic dependent launch and the dependent kernel never fires, hanging any
# prompt over ~8k tokens. Reported by xexex7 on vLLM issue #53960 with the same
# suggestion: PDL should be gated on 9.0/10.0, not `major >= 9`.
# Scope this to the one kernel that hangs rather than disabling PDL globally.
QSA="$SP/vllm/models/qwen4_exp/common/qsa_cache.py"
if [[ ! -f "$QSA" ]]; then
    echo "  ERROR: $QSA missing"; exit 1
fi
if grep -q "LLOYD_PDL_SM120_PATCH" "$QSA"; then
    echo "  already patched"
else
    "$PY" - "$QSA" <<'PYEOF'
import re, sys
p = sys.argv[1]
src = open(p).read()
old = """def _metadata_launch_pdl() -> bool:
    return current_platform.is_arch_support_pdl()"""
new = """def _metadata_launch_pdl() -> bool:
    # LLOYD_PDL_SM120_PATCH — is_arch_support_pdl() is `major >= 9`, which is
    # True on sm_120 (major 12), but the dependent kernel never fires there and
    # any prompt over ~8k tokens hangs forever. Gate on the architectures PDL
    # was actually validated on (Hopper 9.x, Blackwell datacenter 10.x).
    if not current_platform.is_arch_support_pdl():
        return False
    try:
        major = current_platform.get_device_capability().major
    except Exception:
        return False
    return major in (9, 10)"""
if old not in src:
    sys.exit("  ERROR: _metadata_launch_pdl() not in the expected form; patch by hand")
open(p, "w").write(src.replace(old, new, 1))
print("  patched _metadata_launch_pdl()")
PYEOF
fi

echo ""
echo "=== Verifying ==="
"$PY" - <<'PYEOF'
import sys
import importlib.metadata as md
fail = []

import torch
print(f"PyTorch:      {torch.__version__}  (cuda {torch.version.cuda})")
if "+cu13" not in torch.__version__:
    fail.append(f"torch {torch.__version__} is not a cu13 build")
if not any(a.startswith("sm_120") for a in torch.cuda.get_arch_list()):
    fail.append("sm_120 missing from torch arch list — cannot drive the RTX PRO 6000")

import vllm
print(f"vLLM:         {vllm.__version__}")

import vllm.envs as envs
if not hasattr(envs, "VLLM_PLE_CPU_OFFLOAD"):
    fail.append("VLLM_PLE_CPU_OFFLOAD absent — the PLE offload overlay did not apply")
else:
    print("PLE offload:  VLLM_PLE_CPU_OFFLOAD present")

try:
    import vllm.v1.ple_offload.worker  # noqa: F401
    import vllm.v1.ple_offload.connector  # noqa: F401
    print("ple_offload:  worker + connector import cleanly")
except Exception as e:
    fail.append(f"vllm.v1.ple_offload failed to import: {e}")

from vllm.v1.ple_offload import protocol
if not hasattr(protocol, "barrier_timeout_s"):
    fail.append("rendezvous ACK barrier missing — davidtai PR #10 overlay did not apply")
else:
    print("rendezvous:   ACK barrier present")

# Wheel/overlay ABI skew check. The overlaid model_runner.py is written against
# the branch's base commit; if the wheel is pinned to a DIFFERENT main commit,
# the two disagree about internal signatures and you find out ~8 minutes into a
# boot, after 75 GiB has loaded. Caught exactly this on 2026-09-03 when the pin
# was taken from the PR's base.sha instead of the branch's true parent:
#   TypeError: InputBatch.__init__() got an unexpected keyword argument 'max_seq_len_np'
import inspect as _i
from vllm.v1.worker.gpu.input_batch import InputBatch
_sig = _i.signature(InputBatch.__init__).parameters
if "max_seq_len_np" not in _sig:
    fail.append(
        "wheel/overlay skew: InputBatch.__init__ has no 'max_seq_len_np' but the "
        "overlaid model_runner.py passes it — BASE_SHA is not the branch's parent "
        "commit. Re-derive it from d5002fb009bc^, not the PR's base.sha."
    )
else:
    print("ABI check:    InputBatch matches the overlaid model_runner")

import inspect
from vllm.models.qwen4_exp.common import qsa_cache
if "LLOYD_PDL_SM120_PATCH" not in inspect.getsource(qsa_cache._metadata_launch_pdl):
    fail.append("sm_120 PDL patch missing — long prompts will hang")
else:
    print(f"sm_120 PDL:   patched (_metadata_launch_pdl() -> {qsa_cache._metadata_launch_pdl()})")

from vllm.model_executor.models.registry import _MULTIMODAL_MODELS as MM
print("qwen4_exp registered:", "Qwen4ExpForConditionalGeneration" in str(MM))

for p in ("transformers", "flashinfer-python"):
    try: print(f"  {p}: {md.version(p)}")
    except md.PackageNotFoundError: print(f"  {p}: MISSING")

if fail:
    print("\nFAILED:")
    for f in fail: print(f"  - {f}")
    sys.exit(1)
print("\nOK — venv ready. Start: bash bin/start-qwen38-flash-next.sh")
PYEOF
