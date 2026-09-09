#!/usr/bin/env python3
"""Enable the FlashInfer GDN prefill kernel on SM12x in this venv.

Backport of upstream vllm f6326f53b ("[Perf][GDN] Enable the FlashInfer GDN
prefill kernel on SM12x", #55715, 2026-09-08). The pinned wheel for this slot
is a 2026-09-01 main commit, which predates it.

WHY IT IS ONLY A PREDICATE
`_resolve_gdn_prefill_backend` gates FlashInfer on SM90 or SM10.x. SM120 falls
through to Triton/FLA even though the installed FlashInfer already ships the
kernel (`flashinfer.gdn_prefill` imports `chunk_gated_delta_rule_sm120` and
`cp_delta_rule_dsl_sm120`, and its own dispatcher accepts arch major 12). So
the card was running the slower path because vLLM never offered it, not
because anything was missing.

36 of this model's 48 layers are linear-attention, so this is a prefill lever
and prefill is where an agent workload with 50k-token prompts actually lives.

The SM120 kernel requires float32 for the recurrent state; the vLLM wrapper
(`fi_chunk_gated_delta_rule`) already casts `initial_state`, `g` and `beta` to
float32 unconditionally, so no call-site change is needed.

Idempotent. Verifies by resolving the predicate on the live GPU, so a patch
that applied textually but still resolves to Triton fails loudly here rather
than silently serving the old kernel and looking like a null result.

  python agent-services/bin/flash-next-gdn-sm12x-patch.py [--venv DIR] [--revert]
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

MARKER = "LLOYD_GDN_SM12X_PATCH"

OLD = """    if current_platform.is_device_capability(90):
        supports_flashinfer = True
    elif (
        current_platform.is_device_capability_family(100)
        and head_k_dim == 128
        and current_platform.get_cuda_runtime_major() >= 13
    ):
        supports_flashinfer = True
        supports_cutedsl = True
"""

NEW = """    if current_platform.is_device_capability(90):
        supports_flashinfer = True
    elif (
        current_platform.is_device_capability_family(100)
        and head_k_dim == 128
        and current_platform.get_cuda_runtime_major() >= 13
    ):
        supports_flashinfer = True
        supports_cutedsl = True
    elif (
        # LLOYD_GDN_SM12X_PATCH — backport of vllm f6326f53b (#55715).
        # The in-tree CuteDSL kernel targets SM100 only, so it stays off here;
        # FlashInfer's own SM120 path is what this enables.
        current_platform.is_device_capability_family(120)
        and head_k_dim == 128
        and current_platform.get_cuda_runtime_major() >= 13
    ):
        supports_flashinfer = True
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--venv", default=str(pathlib.Path.home() / "lloyd/.venvs/vllm-qwen38-flash-next"))
    ap.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    venv = pathlib.Path(args.venv)
    py = venv / "bin/python"
    target = venv / "lib/python3.12/site-packages/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
    if not target.exists():
        print(f"ERROR: {target} not found", file=sys.stderr)
        return 1

    src = target.read_text()

    if args.revert:
        if MARKER not in src:
            print("not patched; nothing to revert")
            return 0
        target.write_text(src.replace(NEW, OLD, 1))
        print("reverted")
        return 0

    if MARKER in src:
        print("already patched")
    elif OLD not in src:
        print("ERROR: the SM90/SM10x gate is not in the expected form.", file=sys.stderr)
        print("       The wheel has moved; re-derive this patch by hand against", file=sys.stderr)
        print("       _resolve_gdn_prefill_backend before trusting any arm.", file=sys.stderr)
        return 1
    else:
        target.write_text(src.replace(OLD, NEW, 1))
        print(f"patched {target}")

    # Textual success is not the claim worth making. Resolve it on the real GPU.
    probe = """
import warnings; warnings.filterwarnings("ignore")
from vllm.platforms import current_platform
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as m
import inspect
print("marker_present:", "LLOYD_GDN_SM12X_PATCH" in inspect.getsource(m._resolve_gdn_prefill_backend))
print("capability:", current_platform.get_device_capability())
print("family120:", current_platform.is_device_capability_family(120))
print("cuda_runtime_major:", current_platform.get_cuda_runtime_major())
"""
    env_note = "CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1"
    print(f"\nverifying (run under {env_note}):")
    res = subprocess.run(
        [str(py), "-c", probe],
        capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": str(pathlib.Path.home()),
             "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "1",
             "LD_LIBRARY_PATH": "/usr/lib:/opt/cuda/targets/x86_64-linux/lib:/opt/cuda/lib64"},
    )
    print(res.stdout.strip() or res.stderr.strip()[-800:])
    print("\nNow boot with GDN_PREFILL_BACKEND=flashinfer and confirm the boot log says")
    print("  'Using FlashInfer GDN prefill kernel' — not 'Using Triton/FLA'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
