#!/usr/bin/env bash
set -euo pipefail
# vLLM *main* venv for Inferact/Qwen3.8-Flash-Next-NVFP4, with FP8 KV cache.
#
#   venv:  ~/lloyd/.venvs/vllm-flash-next-main-1012 (dated: a rebuild goes to a
#          new path and the previous build stays as the revert target)
#   start: VLLM_VENV=~/lloyd/.venvs/vllm-flash-next-main-1012 bin/start-qwen38-flash-next.sh
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
#   Exactly one thing, since 2026-09-17: the sm_120 PDL hang in the QSA
#   metadata kernel. is_arch_support_pdl() is `major >= 9`, so it returns True
#   on sm_120 (major 12), the QSA metadata kernel launches with PDL, and the
#   dependent kernel waits forever on any prompt over ~8k tokens. Patched into
#   the venv at step 2 below.
#
#   THAT BUG IS NOT FILED UPSTREAM, and this comment used to cite vLLM issue
#   #53960 for it, which is a different bug — the PLE offload uniproc deadlock
#   on GB10/sm_121, fixed by 95dc96d1d012 and irrelevant to the UVA build. A
#   search of the tracker returns exactly one hit for `_metadata_launch_pdl`
#   and it is a comment on that same #53960. So this patch has no upstream to
#   retire it: it must be re-proved on every rebuild, which step 2 does by
#   refusing to proceed if the function is not in the shape it expects.
#
#   The FP8 KV overlay is GONE. PR #55557 (fp8_e4m3 main KV cache on the QSA
#   path), carried here from a contributor's fork since 2026-09-10, merged to
#   main on 2026-09-16 as dff1bde84dd6 — which is what BASE_SHA now pins, so
#   the wheel carries it natively and step 3 asserts that rather than patching.
#   The merged version also brings per-dtype sm_120 tuning tables for the QSA
#   kernel (_select_sm120_config), which the fork's copy did not have.
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
VLLM_VENV="${VLLM_VENV:-$HOME/lloyd/.venvs/vllm-flash-next-main-1012}"

# Per-commit wheel index, keyed by full SHA — the `nightly` index rolls daily
# and would not have this build next week.
#
# dff1bde84dd6 is the merge of PR #55557, i.e. the exact commit that upstreamed
# the FP8 KV overlay this venv used to carry. It is chosen as the MINIMUM
# commit that gets everything this box wanted and nothing else:
#   #55557  fp8_e4m3 main KV cache on the QSA path   (was our overlay)
#   #55309  fuse PLE residual + QSA output gate      (merged 09-14; 1.44x on
#           the PLE outer-residual kernel at bs=1, 1.10x on the QSA gate)
# Everything after it on main was, at the time of pinning, DeepSeek-V4.1 and
# GLM-5.3 work plus #57273 (a QSA tuning table for sm_90 that dispatches
# before the sm_120 branch and cannot reach this card) — churn with no
# identified benefit here.
#
# Revert target, the build this slot served 2026-09-10 → 09-17:
#   BASE_SHA=6ee5bb0a0b3e32dd1a6d9fddb61b50905ccdd6e0
#   VLLM_PIN=0.28.1rc1.dev661+g6ee5bb0a0
#   plus the FP8 overlay from semerandre/vllm@b7e3231af027 (PR #55557 head),
#   which the version of this script at git 4a3ac77 still applies.
#
# The version string is the CI's, not semver: recent per-commit wheels read
# 0.2.x/0.3.x.devN because the build clones shallow and finds no release tag.
# It is consistent across neighbouring commits, so it is cosmetic — but it
# must be quoted EXACTLY here or pip cannot resolve it.
# REBUILT 2026-10-12 at a98247ab4db6 (main, 2026-10-10, the last commit touching
# vllm/models/qwen4_exp at the time; inside v0.31.1rc1). What it adds over
# dff1bde for this card, all on the Qwen4Exp path:
#   #57097  fused QK-norm/RoPE/gate + KV write into the QSA prepare launch
#   #59533  QSA QKVG and indexer QK projections merged into one GEMM
#   #58957  fused HC down projection + SiLU; #60027 HC up on the skinny GEMM
#   #58114  PLE metadata overhead; #57105 QSA indexer workspace fragmentation
#   #58489  BUGFIX pinned PLE prefetch ids dangling in piecewise CUDA graphs
#           (cpu_offload=True, exactly this deployment)
#   #58961  BUGFIX profiling KV cache held alive by QSA key views
#   #59528  BUGFIX QSA ring wrote dummy rows into the shared null block
#   #59853  M-RoPE cos/sin cache 4x smaller (images are enabled here)
# Breaking changes crossed on the way, handled in this script and the launcher:
#   #57937  VLLM_PLE_CPU_OFFLOAD removed; --engram-config is the only spelling
#   #57387  transformers floor 5.16.1 (qwen4_exp config is upstream now)
#   `python -m vllm.entrypoints.openai.api_server` deprecated -> `vllm serve`
# FlashInfer moves 0.6.18.post1 -> 0.7.0.post1 with the wheel (its pin).
#
# Previous pin, the build this slot served 2026-09-17 -> 2026-10-12, kept at
# .venvs/vllm-flash-next-main as the revert target:
#   BASE_SHA=dff1bde84dd6e34a49c150116d0f212507280910
#   VLLM_PIN=0.2.1.dev19+gdff1bde84
BASE_SHA="a98247ab4db686ee03c66d5feb3c761e52a2f8ab"
VLLM_PIN="0.31.1rc1.dev260+ga98247ab4"

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

# Prebuilt FlashInfer kernels for the version the wheel pinned. vLLM's own
# requirements/cuda.txt lists flashinfer-cubin beside flashinfer-python (the
# cubin package left PyPI at 0.6.14; both live on flashinfer.ai/whl), and the
# per-arch JIT cache skips the ~20-minute, ~45 GiB `cicc` storm a fresh venv
# otherwise pays on first boot — the storm behind the 09-17 oomd kill
# (architecture/vllm.md §8). Without them nothing breaks: the launcher's
# CUDA_HOME/PATH exports still let FlashInfer JIT every kernel it needs.
FI_VER="$("$PY" -c 'import importlib.metadata as m; print(m.version("flashinfer-python"))')"
echo "=== 1b/3 FlashInfer prebuilt cubins + JIT cache ($FI_VER, cu130) ==="
"$PY" -m pip install -q "flashinfer-cubin==$FI_VER" --index-url https://flashinfer.ai/whl/ \
    || echo "  WARN: no flashinfer-cubin $FI_VER on flashinfer.ai — JIT at first boot instead"
"$PY" -m pip install -q "flashinfer-jit-cache==$FI_VER+cu130" --index-url https://flashinfer.ai/whl/cu130 \
    || echo "  WARN: no flashinfer-jit-cache $FI_VER+cu130 — JIT at first boot instead"
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
    # any prompt over ~8k tokens hangs forever. Not filed upstream, so nothing
    # will retire this patch for us. Gate on the architectures PDL was
    # actually validated on (Hopper 9.x, Blackwell datacenter 10.x).
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
echo "=== 3/3 FP8 main KV cache on the QSA path (PR #55557, merged) ==="
# A precondition now, not a patch. BASE_SHA is at or after the merge, so a
# wheel without the kernel path means the pin is wrong — and the failure that
# would follow is the expensive kind: the engine boots, the QSA backend
# refuses fp8, the launcher's own guard fires four minutes in, and the slot
# falls back to a BF16 pool 1.74x smaller than the one production asserts.
# Fail here instead, in 10 ms.
if grep -q "IS_FP8" "$SP/vllm/models/qwen4_exp/nvidia/ops/qsa.py"; then
    echo "  ok  wheel carries the fp8 QSA kernel path (IS_FP8)"
else
    echo "  ERROR: this wheel has no IS_FP8 path in vllm/models/qwen4_exp/nvidia/ops/qsa.py."
    echo "         BASE_SHA ($BASE_SHA) predates the merge of PR #55557"
    echo "         (dff1bde84dd6, 2026-09-16). Move the pin forward, or restore the"
    echo "         overlay from the version of this script at git 4a3ac77."
    exit 1
fi
if grep -q "_select_sm120_config" "$SP/vllm/models/qwen4_exp/nvidia/ops/qsa.py"; then
    echo "  ok  wheel carries the sm_120 QSA tuning tables (_select_sm120_config)"
else
    echo "  WARN: no _select_sm120_config in this wheel — the QSA kernel will run"
    echo "        generic launch parameters on this card. Not fatal."
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
# VLLM_PLE_CPU_OFFLOAD is gone since #57937 (2026-09-21); the UVA offload is
# requested through EngramConfig only, which the next check proves exists.
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
print("\nOK — venv ready. Start: VLLM_VENV=<this venv> bash bin/start-qwen38-flash-next.sh")
PYEOF

# Named for the venv, not hardcoded: this script honours VLLM_VENV, and a
# candidate build at a second path must not overwrite the manifest describing
# the venv the primary slot is currently serving from. Identical path for the
# default venv name.
MANIFEST="$PROJECT_DIR/setup/$(basename "$VLLM_VENV").versions.txt"
{
    echo "# vllm-flash-next-main venv — resolved $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "# rebuild: bash setup/setup-vllm-flash-next-main.sh"
    echo "# base wheel: vllm==$VLLM_PIN (per-commit, main@${BASE_SHA:0:12})"
    echo "# overlay:    none (PR #55557 merged upstream as dff1bde84dd6, 2026-09-16)"
    echo "# patch:      _metadata_launch_pdl() -> False on sm_120 (not filed upstream)"
    "$PY" -m pip freeze 2>/dev/null | grep -vE "^(pip|setuptools|wheel)=="
} > "$MANIFEST"
echo "wrote $MANIFEST"
