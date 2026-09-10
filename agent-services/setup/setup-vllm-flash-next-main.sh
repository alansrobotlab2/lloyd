#!/usr/bin/env bash
set -euo pipefail
# vLLM *main* venv for Inferact/Qwen3.8-Flash-Next-NVFP4, with FP8 KV cache.
#
#   venv:  ~/lloyd/.venvs/vllm-flash-next-main
#   start: VLLM_VENV=~/lloyd/.venvs/vllm-flash-next-main bin/start-qwen38-flash-next.sh
#   model: setup/setup-qwen38-flash-next.sh (same checkpoint as the offload venv)
#
# WHY A SECOND VENV FOR THE SAME MODEL
#   setup-vllm-qwen38-flash-next.sh builds against an 2026-08-31 base commit
#   plus the PLE CPU-offload branch (PR #53899), because at the time nothing
#   on main could keep the 95 GiB N-gram table in host RAM. On 2026-09-09
#   vLLM merged #54371 — UVA PLE offload — and paused #53899 in its favour, so
#   main now serves this checkpoint on one 96 GiB card with a plain wheel: no
#   offload worker process, no CUDA-IPC handshake, no ptrace_scope
#   requirement, and none of the three deadlocks the old start script
#   documents. Main also carries the QSA kernel rewrites (#54513, #54873,
#   #54915) and the FP8 indexer cache (#54890).
#
#   Measured 2026-09-10 on this box (bare engine, same flags, 11.5 GiB KV):
#     BF16 on main: decode 254 tok/s single-stream (production 137),
#                   prefill 9.3k tok/s, pool 398,175 tokens (identical).
#     FP8 on main:  pool 692,263 tokens (x1.74), needle 12/12 to 239k,
#                   logprob deviation inside BF16's run-to-run noise,
#                   prefill 9.5-9.7k tok/s; a 239k prompt prefills in 26 s
#                   where BF16 (main AND production) takes 128-149 s.
#   See ~/.claude memory `fp8-kv-trial-2026-09-10` for the full table.
#
# WHAT IS NOT A PLAIN WHEEL, STILL
#   1. FP8 main KV cache on the QSA path is open PR #55557 (three pure-Python
#      files). Overlaid below from the PR head, after proving the wheel's copies
#      are byte-identical to the PR's merge-base copies — otherwise the overlay
#      is a guess. Self-retiring: skipped when the wheel already carries the
#      change (its ops/qsa.py has the IS_FP8 kernel path).
#   2. The sm_120 PDL hang in the QSA metadata kernel (vLLM issue #53960).
#      is_arch_support_pdl() is still `major >= 9` on main.
#   The GDN FlashInfer prefill gate for SM12x (#55715) IS on main now, so the
#   old bin/flash-next-gdn-sm12x-patch.py is not applied here.
#
# HOST RAM
#   The UVA path pins the 95.37 GiB BF16 table (cudaHostAlloc) — it cannot be
#   swapped, unlike the old worker's pageable copy. 251 GiB here, fine. It is
#   released within ~30 s of SIGTERM.
#
# FIRST BOOT IS SLOW
#   A cold torch.compile / Triton cache costs ~8 minutes with no log lines
#   between "Model loading took" and the KV cache line; measured 775 s to
#   health on first boot, 265 s on the second. supervisord's startsecs for
#   agent-llm-primary must cover the cold case (see the .conf).
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VLLM_VENV="${VLLM_VENV:-$HOME/lloyd/.venvs/vllm-flash-next-main}"

# The main commit the cu130 nightly of 2026-09-10 was built from. Per-commit
# wheel index, keyed by full SHA — the `nightly` index rolls daily and would
# not have this build next week.
BASE_SHA="6ee5bb0a0b3e32dd1a6d9fddb61b50905ccdd6e0"
VLLM_PIN="0.28.1rc1.dev661+g6ee5bb0a0"
# PR #55557 head (semerandre/vllm, s1/qsa-kv-fp8-pr) and the main commit it
# last merged, i.e. the second parent of that head.
FP8_PR_REPO="semerandre/vllm"
FP8_PR_SHA="b7e3231af0276d6dee3363cf26f2da2aed8e195f"
FP8_PR_BASE="b28c3e1568bfae930f61d4b24940e47528c85d4a"
FP8_FILES="vllm/models/qwen4_exp/nvidia/qsa.py vllm/models/qwen4_exp/nvidia/ops/qsa.py"

echo "=== vLLM main venv for Qwen3.8-Flash-Next (UVA PLE offload + FP8 KV) ==="
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
echo "=== 1/3 Installing vLLM $VLLM_PIN (per-commit wheel) ==="
"$PY" -m pip install "vllm==$VLLM_PIN" --pre \
    --index-url "https://wheels.vllm.ai/$BASE_SHA/cu130" \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --extra-index-url https://pypi.org/simple
"$PY" -m pip install -q ninja hf_transfer huggingface_hub
SP="$("$PY" -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')"
echo "site-packages: $SP"

echo ""
echo "=== 2/3 Patching sm_120 PDL hang in the QSA metadata kernel ==="
QSA="$SP/vllm/models/qwen4_exp/common/qsa_cache.py"
if grep -q "LLOYD_PDL_SM120_PATCH" "$QSA"; then
    echo "  already patched"
else
    "$PY" - "$QSA" <<'PYEOF'
import sys
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
echo "=== 3/3 FP8 main KV cache on the QSA path (PR #55557) ==="
if grep -q "IS_FP8" "$SP/vllm/models/qwen4_exp/nvidia/ops/qsa.py"; then
    echo "  wheel already carries the fp8 QSA path — overlay skipped (PR #55557 merged?)"
else
    TMP="$(mktemp -d)"
    for f in $FP8_FILES; do
        curl -sfL "https://raw.githubusercontent.com/vllm-project/vllm/$FP8_PR_BASE/$f" -o "$TMP/base.py" \
            || { echo "  FAIL fetching merge-base $f"; exit 1; }
        if ! cmp -s "$TMP/base.py" "$SP/$f"; then
            echo "  ERROR: the wheel's $f differs from the PR's merge-base copy."
            echo "         BASE_SHA moved past what the PR was written against; re-derive"
            echo "         FP8_PR_BASE from the PR head's second parent, or drop the overlay."
            diff "$TMP/base.py" "$SP/$f" | head -20
            exit 1
        fi
        curl -sfL "https://raw.githubusercontent.com/$FP8_PR_REPO/$FP8_PR_SHA/$f" -o "$SP/$f" \
            || { echo "  FAIL fetching $f"; exit 1; }
        echo "  ok  $f (wheel == merge-base, overlaid from $FP8_PR_REPO@${FP8_PR_SHA:0:12})"
    done
    rm -rf "$TMP"
fi

echo ""
echo "=== Verifying ==="
"$PY" - <<'PYEOF'
import inspect, sys
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
    fail.append("VLLM_PLE_CPU_OFFLOAD absent — this wheel predates UVA PLE offload (#54371)")
try:
    from vllm.config import EngramConfig  # noqa: F401
    print("PLE offload:  EngramConfig present (UVA path, #54371)")
except Exception as e:
    fail.append(f"EngramConfig missing: {e}")
from vllm.models.qwen4_exp.common import qsa_cache
if "LLOYD_PDL_SM120_PATCH" not in inspect.getsource(qsa_cache._metadata_launch_pdl):
    fail.append("sm_120 PDL patch missing — long prompts will hang")
else:
    print(f"sm_120 PDL:   patched (_metadata_launch_pdl() -> {qsa_cache._metadata_launch_pdl()})")
import vllm.models.qwen4_exp.nvidia.model  # noqa: F401  package import order (circular import otherwise)
from vllm.models.qwen4_exp.nvidia import qsa
dts = qsa.Qwen4ExpQSAFlashAttentionBackend.supported_kv_cache_dtypes
if "fp8" not in dts:
    fail.append(f"QSA backend does not advertise fp8 KV: {dts}")
else:
    print(f"FP8 KV:       QSA backend advertises {dts}")
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as _gdn
if "is_device_capability_family(120)" not in inspect.getsource(_gdn._resolve_gdn_prefill_backend):
    fail.append("GDN SM12x gate missing — GDN_PREFILL_BACKEND=flashinfer would fall back to Triton")
else:
    print("sm_12x GDN:   native (#55715 on main)")
for p in ("transformers", "flashinfer-python", "triton"):
    try: print(f"  {p}: {md.version(p)}")
    except md.PackageNotFoundError: print(f"  {p}: MISSING")
if fail:
    print("\nFAILED:")
    for f in fail: print(f"  - {f}")
    sys.exit(1)
print("\nOK — venv ready. Start: VLLM_VENV=$VLLM_VENV bash bin/start-qwen38-flash-next.sh")
PYEOF

MANIFEST="$PROJECT_DIR/setup/vllm-flash-next-main.versions.txt"
{
    echo "# vllm-flash-next-main venv — resolved $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "# rebuild: bash setup/setup-vllm-flash-next-main.sh"
    echo "# base wheel: vllm==$VLLM_PIN (per-commit, main@${BASE_SHA:0:12})"
    echo "# overlay:    $FP8_PR_REPO@${FP8_PR_SHA:0:12} (PR #55557, fp8_e4m3 main KV on the QSA path)"
    echo "# patch:      _metadata_launch_pdl() -> False on sm_120 (vLLM issue #53960)"
    "$PY" -m pip freeze 2>/dev/null | grep -vE "^(pip|setuptools|wheel)=="
} > "$MANIFEST"
echo "wrote $MANIFEST"
