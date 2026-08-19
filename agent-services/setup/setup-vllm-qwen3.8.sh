#!/usr/bin/env bash
set -euo pipefail

# Dedicated bleeding-edge vLLM venv for unsloth/Qwen3.8-27B-NVFP4.
#
#   venv:  ~/lloyd/.venvs/vllm-qwen3.8
#   start: bin/start-qwen3.8-27b-nvfp4.sh
#   model: setup/setup-qwen3.8-27b-nvfp4.sh
#
# WHY A SEPARATE VENV
#   The Qwen3.8 start script previously borrowed vllm-laguna (0.25.1) because it
#   happened to ship transformers 5.14.1. That coupled two unrelated models to
#   one dependency set: any upgrade for Qwen3.8 would silently re-qualify Laguna
#   S 2.1, and vice versa. This venv is Qwen3.8's alone, so it can track vLLM
#   nightly aggressively without putting the Laguna primary at risk. Blowing it
#   away and re-running this script cannot affect any other model.
#
#   Existing venvs and what they serve, for reference:
#     vllm-experimental  0.23.1rc1.dev1218  Qwen3.5/3.6 family, 35B, 122B
#     vllm-laguna        0.25.1             Laguna S 2.1 + DFlash draft
#     vllm-qwen3.8       (this one)         Qwen3.8-27B-NVFP4
#
# RESOLUTION STRATEGY — pin vLLM, let vLLM pick everything else
#   Unlike setup-vllm-experimental.sh, this script does NOT pre-install a pinned
#   torch. Pinning torch first and installing vLLM second means pip must satisfy
#   the nightly's torch requirement against an already-installed version, which
#   either downgrades vLLM or silently pulls a mismatched torch. Installing vLLM
#   first and letting it drag in its own torch/flashinfer/transformers is what
#   "bleeding edge" actually needs. The verification block below then proves the
#   result is a working CUDA 13 / SM120 stack instead of assuming it.
#
#   The version pin on vllm itself is still mandatory — see the long comment at
#   the resolve step. That trap is unchanged from setup-vllm-experimental.sh.
#
# FLASHINFER PAIRING
#   flashinfer-python and flashinfer-cubin must be the SAME version. cubin ships
#   precompiled kernels; python is the dispatch layer. As of 2026-08-18 PyPI has
#   flashinfer-python 0.6.17 but flashinfer-cubin only 0.6.13, and the 0.6.14
#   cubin installed in vllm-experimental is no longer available from PyPI, the
#   vLLM nightly index, or flashinfer.ai/whl. So do NOT chase the newest
#   flashinfer-python here — take whatever pair vLLM resolves. A mismatch is not
#   fatal (flashinfer falls back to JIT-compiling kernels, which needs g++-15 per
#   NVCC_CCBIN and makes the first boot slow) but it is worth knowing about, so
#   the check below reports it loudly rather than failing.

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VLLM_VENV="${VLLM_VENV:-$HOME/lloyd/.venvs/vllm-qwen3.8}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/llm/models/unsloth-Qwen3.8-27B-NVFP4}"
LOCKFILE="$PROJECT_DIR/setup/vllm-qwen3.8.versions.txt"

echo "=== Bleeding-edge vLLM venv for Qwen3.8-27B-NVFP4 ==="

if [[ ! -x /opt/cuda/bin/nvcc ]]; then
    echo "ERROR: CUDA toolkit not found at /opt/cuda"
    exit 1
fi
echo "System CUDA: $(/opt/cuda/bin/nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')"

# FlashInfer JIT needs gcc-15: nvcc 13.x cannot parse the gcc-16 libstdc++.
if [[ ! -x /usr/bin/g++-15 ]]; then
    echo "WARNING: /usr/bin/g++-15 missing — FlashInfer JIT will fail against gcc-16 libstdc++"
    echo "         install gcc15 (the start script exports NVCC_CCBIN=/usr/bin/g++-15)"
fi

if [[ -L "$VLLM_VENV" ]]; then
    echo "Removing existing symlink at $VLLM_VENV"
    rm "$VLLM_VENV"
elif [[ -d "$VLLM_VENV" ]]; then
    echo "Venv already exists at $VLLM_VENV"
    # Non-interactive (CI, background run) must not block or die on EOF from read,
    # which `set -e` would turn into a silent abort. Default to upgrading in place;
    # force a rebuild with RECREATE=1.
    if [[ "${RECREATE:-}" == "1" ]]; then
        yn="y"
    elif [[ -t 0 ]]; then
        read -rp "Recreate from scratch? [y/N] " yn
    else
        yn="n"
        echo "(non-interactive — upgrading in place; RECREATE=1 to rebuild)"
    fi
    if [[ "$yn" =~ ^[Yy]$ ]]; then
        rm -rf "$VLLM_VENV"
    else
        echo "Upgrading in-place..."
    fi
fi

if [[ ! -d "$VLLM_VENV" ]]; then
    echo "Creating venv at $VLLM_VENV (Python 3.12)..."
    uv venv "$VLLM_VENV" --python 3.12
fi

PY="$VLLM_VENV/bin/python"
PIP="$PY -m pip"

uv pip install pip --python "$PY"

# Resolve the latest cu130 nightly and PIN it.
#
# --pre is REQUIRED but NOT SUFFICIENT. The cu130 wheel index publishes only dev
# nightlies (0.26.1rc1.devN+g<sha>). PyPI also publishes FINAL releases built
# against CUDA 12, and a final release outranks a .devN pre-release of the same
# version — so an unpinned `pip install vllm --pre` with PyPI reachable resolves
# the PyPI cu12 wheel, which dies at import with
# "libcudart.so.12: cannot open shared object file".
#
# Reading the one version the cu130 index actually publishes and pinning to it
# means PyPI can only ever supply transitive deps, never vllm itself.
echo ""
echo "=== Resolving latest vLLM cu130 nightly ==="
VLLM_NIGHTLY=$($PY -m pip index versions vllm --pre \
    --index-url https://wheels.vllm.ai/nightly/cu130 2>/dev/null \
    | grep -oP '^vllm \(\K[^)]+')
if [[ -z "$VLLM_NIGHTLY" ]]; then
    echo "ERROR: could not resolve latest cu130 nightly from wheels.vllm.ai"
    exit 1
fi
echo "Latest cu130 nightly: $VLLM_NIGHTLY"

echo ""
echo "=== Installing vLLM $VLLM_NIGHTLY (torch/flashinfer/transformers resolved by vLLM) ==="
$PIP install "vllm==$VLLM_NIGHTLY" --pre \
    --index-url https://wheels.vllm.ai/nightly/cu130 \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --extra-index-url https://pypi.org/simple

# ninja is required for FlashInfer JIT; fastsafetensors speeds up the 23 GiB load.
echo ""
echo "=== Installing build/load helpers ==="
$PIP install ninja
$PIP install fastsafetensors 2>/dev/null || echo "fastsafetensors unavailable, skipping"

echo ""
echo "=== Verifying stack ==="
$PY - <<'EOF'
import importlib.metadata as md
import sys

fail, warn = [], []

import torch
print(f"PyTorch:      {torch.__version__}")
print(f"CUDA runtime: {torch.version.cuda}")
arch = torch.cuda.get_arch_list()
print(f"Arch list:    {' '.join(arch)}")

# The cu12-from-PyPI trap: catch it here rather than at first model load.
if "+cu13" not in torch.__version__:
    fail.append(f"torch {torch.__version__} is not a cu13 build — PyPI cu12 wheel "
                "leaked in; the index pinning above did not hold")
if not any(a.startswith("sm_120") for a in arch):
    fail.append("sm_120 missing from torch arch list — this build cannot drive "
                "the RTX PRO 6000 Blackwell")

import vllm
print(f"vLLM:         {vllm.__version__}")

for pkg in ("transformers", "compressed-tensors"):
    print(f"{pkg + ':':14}{md.version(pkg)}")

# flashinfer-python and flashinfer-cubin must agree; see header.
try:
    import flashinfer
    print(f"FlashInfer:   {flashinfer.__version__}")
    try:
        cubin = md.version("flashinfer-cubin")
        print(f"  cubin:      {cubin}")
        if cubin != md.version("flashinfer-python"):
            warn.append(f"flashinfer-python {md.version('flashinfer-python')} != "
                        f"flashinfer-cubin {cubin} — kernels will JIT-compile on "
                        "first boot (slow, needs NVCC_CCBIN=g++-15)")
    except md.PackageNotFoundError:
        warn.append("flashinfer-cubin absent — kernels will JIT-compile on first boot")
except ImportError:
    fail.append("flashinfer did not install — the start script passes "
                "--attention-backend FLASHINFER and will not boot")

if fail:
    print("\nFAILED:")
    for f in fail:
        print(f"  - {f}")
    sys.exit(1)
for w in warn:
    print(f"\nNOTE: {w}")
EOF

# Model-specific smoke test. Checks the four things the start script depends on,
# without loading 23 GiB of weights onto the GPU.
echo ""
echo "=== Qwen3.8 compatibility smoke test ==="
$PY - "$MODEL_DIR" <<'EOF'
import os, sys

model_dir = sys.argv[1]
if not os.path.isfile(os.path.join(model_dir, "config.json")):
    print(f"SKIPPED: model not downloaded at {model_dir}")
    print("Run: bash setup/setup-qwen3.8-27b-nvfp4.sh")
    sys.exit(0)

fail = []

# 1. The config parses under this venv's transformers.
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained(model_dir)
print(f"config:            {type(cfg).__name__} "
      f"(model_type={cfg.model_type}, {cfg.text_config.num_hidden_layers} layers)")

# 2. The MTP architecture the speculative path rewrites to is registered, and it
#    supports multimodal — this model keeps a BF16 vision tower alongside MTP.
from vllm.model_executor.models.registry import ModelRegistry
if "Qwen3_5MTP" in ModelRegistry.get_supported_archs():
    from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MTP
    # Use vLLM's own predicate, not issubclass(): SupportsMultiModal is a runtime
    # Protocol with non-method members, and issubclass() raises TypeError on those.
    from vllm.model_executor.models.interfaces import supports_multimodal
    mm = supports_multimodal(Qwen3_5MTP)
    print(f"Qwen3_5MTP:        registered (SupportsMultiModal={mm})")
    if not mm:
        fail.append("Qwen3_5MTP no longer SupportsMultiModal — MTP and the vision "
                    "tower may not coexist; drop --speculative-config or add "
                    "--language-model-only")
else:
    fail.append("Qwen3_5MTP not in the model registry — --speculative-config "
                "qwen3_5_mtp will fail")

# 3. Both parsers the start script names must resolve.
from vllm.reasoning import ReasoningParserManager
try:
    ReasoningParserManager.get_reasoning_parser("qwen3")
    print("reasoning parser:  qwen3 OK")
except Exception as e:
    fail.append(f"reasoning parser 'qwen3' unavailable: {e}")

# ToolParserManager has moved between releases (vllm.entrypoints.openai.tool_parsers
# -> vllm.tool_parsers as of 0.25.1). Try the known homes; a rename is a reason to
# verify at boot, not to fail the whole setup.
import importlib
ToolParserManager = None
for mod in ("vllm.tool_parsers", "vllm.entrypoints.openai.tool_parsers"):
    try:
        ToolParserManager = importlib.import_module(mod).ToolParserManager
        break
    except (ImportError, AttributeError):
        continue
if ToolParserManager is None:
    print("tool parser:       could not introspect (module moved) — verify at boot")
else:
    try:
        ToolParserManager.get_tool_parser("qwen3_xml")
        print("tool parser:       qwen3_xml OK")
    except Exception as e:
        fail.append(f"tool parser 'qwen3_xml' unavailable: {e} — do NOT substitute "
                    "qwen3_coder, it wedges this family on the stop-token path")

# 4. The mixed-precision quant must be read as compressed-tensors with per-group
#    formats honoured. Forcing --quantization modelopt here would misread it.
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
qc = cfg.quantization_config if hasattr(cfg, "quantization_config") else {}
qc = qc if isinstance(qc, dict) else qc.to_dict()
groups = {k: v.get("format") for k, v in qc.get("config_groups", {}).items()}
print(f"quant:             {qc.get('quant_method')} format={qc.get('format')} "
      f"groups={groups}")
# from_config is the same public entry point vLLM uses at load time, so this
# exercises the real parse rather than a private helper that gets renamed.
ct = CompressedTensorsConfig.from_config(qc)
formats = {v.get("format") for v in ct.target_scheme_map.values() if v.get("format")}
print(f"per-group formats: {sorted(formats)} across {len(ct.target_scheme_map)} targets")
if "nvfp4-pack-quantized" not in formats:
    fail.append("NVFP4 group format not picked up from config_groups — this vLLM "
                "does not honour per-group formats for mixed-precision")

if fail:
    print("\nFAILED:")
    for f in fail:
        print(f"  - {f}")
    sys.exit(1)
print("\nSmoke test passed.")
EOF

# Record what actually got installed, so a regression can be bisected against a
# known-good set and the venv rebuilt to match.
{
    echo "# vllm-qwen3.8 venv — resolved $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "# rebuild: bash setup/setup-vllm-qwen3.8.sh"
    "$PY" -m pip freeze
} > "$LOCKFILE"

echo ""
echo "=== Setup Complete ==="
echo "venv:     $VLLM_VENV"
echo "vLLM:     $("$PY" -c 'import vllm; print(vllm.__version__)' 2>/dev/null | tail -1)"
echo "versions: $LOCKFILE"
echo ""
echo "The start script defaults to this venv:"
echo "  bash $PROJECT_DIR/bin/start-qwen3.8-27b-nvfp4.sh"
echo ""
echo "To fall back to another venv without editing the script:"
echo "  VLLM_VENV=~/lloyd/.venvs/vllm-laguna bash $PROJECT_DIR/bin/start-qwen3.8-27b-nvfp4.sh"
echo ""
echo "Caches are shared across venvs. If a nightly upgrade leaves stale graphs:"
echo "  rm -rf ~/.cache/vllm/torch_compile_cache ~/.cache/flashinfer"
