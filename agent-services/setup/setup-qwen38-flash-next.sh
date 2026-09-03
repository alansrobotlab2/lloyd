#!/usr/bin/env bash
set -euo pipefail

# Downloads Inferact/Qwen3.8-Flash-Next-NVFP4 (~170.3 GiB).
#
# After running this:
#   bash setup/setup-vllm-qwen38-flash-next.sh   # the venv (offload overlay)
#   bash bin/start-qwen38-flash-next.sh          # serve on :8096
#
# WHY THIS CHECKPOINT AND NOT A SMALLER ONE
#   Three NVFP4/FP8 builds of Qwen3.8-Flash-Next exist. On a single 96 GiB card
#   with the N-gram (PLE) table offloaded to host RAM:
#
#     Qwen/...-FP8        172.8 GiB, FP8 PLE 47.7  -> ~125 GiB on GPU. Does not fit.
#     RadixArk/...-NVFP4  126.0 GiB, FP8 PLE 47.7  -> ~76.6 GiB on GPU. Fits, but the
#                         FP8 PLE inside a ModelOpt NVFP4 checkpoint hits vLLM
#                         issue #54765 ("no module or parameter named
#                         'ngram_embedding.weight_scale'") and needs an
#                         out-of-tree load patch. Smaller download, more patches.
#     Inferact/...-NVFP4  170.3 GiB, BF16 PLE 95.4 -> ~74.1 GiB on GPU. Chosen.
#
#   The BF16 PLE is the whole reason: it carries no weight_scale tensor, so
#   #54765 cannot fire. It costs 95.37 GiB of host RAM instead of 47.7, which is
#   irrelevant on a 251 GiB box. Every reporter who needed a swapfile for this
#   was on a 121 GiB unified-memory DGX Spark / GX10.
#
# LAYOUT (34 files)
#   model-00001-of-00004.safetensors   95.37 GiB  the BF16 N-gram table
#                                                 (20,000,000 x 2560 x 2 bytes)
#   model-0000{2,3}-of-00004           ~9.3 GiB   dense / attention / embeddings
#   nvfp4_experts-000NN-of-00016       3.96 GiB   x16, the 512-expert MoE
#   nvfp4_experts_mtp.safetensors       1.49 GiB  MTP draft head (k=3 decode)
#
#   That 95.37 GiB shard is the file that goes to host RAM at runtime; the other
#   ~74 GiB is what actually lands on the GPU.

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_REPO="Inferact/Qwen3.8-Flash-Next-NVFP4"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/llm/models/Inferact-Qwen3.8-Flash-Next-NVFP4}"
REQUIRED_GIB=185

# Works from any venv with an `hf` CLI, so the 170 GiB download can run in
# parallel with the vLLM venv build.
if [[ -n "${VLLM_VENV:-}" ]]; then
    VENV_CANDIDATES=("$VLLM_VENV")
else
    VENV_CANDIDATES=(
        "$HOME/lloyd/.venvs/vllm-qwen38-flash-next"
        "$HOME/lloyd/.venvs/vllm-qwen3.8"
        "$HOME/lloyd/.venvs/vllm-experimental"
    )
fi
VENV=""
for c in "${VENV_CANDIDATES[@]}"; do
    [[ -x "$c/bin/hf" ]] && { VENV="$c"; break; }
done
if [[ -z "$VENV" ]]; then
    echo "ERROR: no venv with an hf CLI found. Tried:"; printf '  %s\n' "${VENV_CANDIDATES[@]}"
    exit 1
fi

echo "=== Qwen3.8-Flash-Next-NVFP4 (Inferact, NVFP4 main + BF16 PLE) ==="
echo "venv:      $VENV"
echo "model dir: $MODEL_DIR"

mkdir -p "$(dirname "$MODEL_DIR")"
AVAIL_GIB=$(df -BG --output=avail "$(dirname "$MODEL_DIR")" | tail -1 | tr -dc '0-9')
if (( AVAIL_GIB < REQUIRED_GIB )); then
    echo "ERROR: need ~${REQUIRED_GIB} GiB free, have ${AVAIL_GIB} GiB"; exit 1
fi
echo "disk:      ${AVAIL_GIB} GiB free (need ~${REQUIRED_GIB})"

# hf download is resumable — re-running after an interrupt is safe and is the
# supported way to fill gaps.
export HF_HUB_ENABLE_HF_TRANSFER=1
"$VENV/bin/hf" download "$MODEL_REPO" --local-dir "$MODEL_DIR"

echo ""
echo "=== Verifying ==="
"$VENV/bin/python" - "$MODEL_DIR" <<'PYEOF'
import json, os, sys
d = sys.argv[1]
fail = []

cfg_path = os.path.join(d, "config.json")
if not os.path.isfile(cfg_path):
    sys.exit("FAILED: config.json missing")
cfg = json.load(open(cfg_path))
t = cfg.get("text_config", cfg)

arch = cfg.get("architectures", [])
print("architectures:", arch)
if "Qwen4ExpForConditionalGeneration" not in arch:
    fail.append(f"unexpected architectures {arch}")

q = cfg.get("quantization_config", {})
print("quant:", q.get("quant_method"), q.get("quant_algo"))

print("ngram_vocab_size_base:", t.get("ngram_vocab_size_base"),
      "| ple_embed_dim:", t.get("ple_embed_dim"),
      "| ple_layer_ids:", t.get("ple_layer_ids"))
print("max_position_embeddings:", t.get("max_position_embeddings"))

# The PLE shard must be BF16 — an FP8 one would carry weight_scale and trip
# vLLM issue #54765 at load. Size is the cheap proxy: 20e6 x 2560 x 2 bytes.
ple = os.path.join(d, "model-00001-of-00004.safetensors")
if os.path.isfile(ple):
    gib = os.path.getsize(ple) / 2**30
    print(f"PLE shard: {gib:.2f} GiB")
    if gib < 90:
        fail.append(f"PLE shard is {gib:.1f} GiB, expected ~95.4 — is this an FP8-PLE build?")
else:
    fail.append("model-00001-of-00004.safetensors (the PLE table) missing")

mtp = os.path.join(d, "nvfp4_experts_mtp.safetensors")
print("MTP head:", "present" if os.path.isfile(mtp) else "MISSING (decode will be non-speculative)")

experts = [f for f in os.listdir(d) if f.startswith("nvfp4_experts-")]
print(f"expert shards: {len(experts)}/16")
if len(experts) != 16:
    fail.append(f"expected 16 expert shards, found {len(experts)}")

total = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)
            if f.endswith(".safetensors"))
print(f"total safetensors: {total/2**30:.1f} GiB (expect ~170.3)")

if fail:
    print("\nFAILED:")
    for f in fail: print("  -", f)
    sys.exit(1)
print("\nOK — model ready. Next: bash setup/setup-vllm-qwen38-flash-next.sh")
PYEOF
