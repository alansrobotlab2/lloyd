#!/usr/bin/env bash
set -euo pipefail

# Setup for unsloth/Qwen3.8-27B-NVFP4 (Unsloth Dynamic V3.0 mixed-precision quant).
# Downloads the model (~23.4 GiB) and verifies the MTP head actually shipped.
#
# After running this, start the server with:
#   bash bin/start-qwen3.8-27b-nvfp4.sh
#
# WHY THE MTP VERIFICATION MATTERS
#   The previous unsloth re-quant (Qwen3.6-27B-NVFP4) advertised
#   text_config.mtp_num_hidden_layers=1 in config.json but shipped ZERO MTP
#   tensors — passing --speculative-config against it wedged vLLM at load.
#   Never trust the config field; check the safetensors index. This build DOES
#   ship the head: model_mtp.safetensors (~810 MiB, 15 BF16 tensors).
#
# QUANTIZATION
#   compressed-tensors "mixed-precision" (NOT ModelOpt — do not pass
#   --quantization modelopt):
#     - group_0 FP8  : self_attn q/k/v/o, linear_attn in_proj_*/out_proj,
#                      lm_head, and layers 56-63 mlp gate/up/down
#     - group_1 NVFP4: all other mlp gate/up/down
#     - ignore       : the whole vision tower, left in BF16
#   Also ships a static per-tensor FP8 kv_cache_scheme (16 k_scale + 16 v_scale,
#   one pair per full-attention layer), so --kv-cache-dtype fp8_e4m3 uses
#   calibrated scales rather than on-the-fly ones.
#
#   Unsloth's README warns this quant is vLLM-only — SGLang cannot load the
#   FP8 lm_head.

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_REPO="unsloth/Qwen3.8-27B-NVFP4"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/llm/models/unsloth-Qwen3.8-27B-NVFP4}"

# This script only needs an `hf` CLI and the stdlib, so it works from any venv —
# it deliberately does NOT require the dedicated vllm-qwen3.8 venv, so the 23 GiB
# download can run before (or in parallel with) setup-vllm-qwen3.8.sh. Prefer the
# model's own venv when it exists, otherwise fall back to whatever is around.
if [[ -n "${VLLM_VENV:-}" ]]; then
    VENV_CANDIDATES=("$VLLM_VENV")
else
    VENV_CANDIDATES=(
        "$HOME/lloyd/.venvs/vllm-qwen3.8"
        "$HOME/lloyd/.venvs/vllm-laguna"
        "$HOME/lloyd/.venvs/vllm-experimental"
    )
fi
VLLM_VENV=""
for candidate in "${VENV_CANDIDATES[@]}"; do
    if [[ -x "$candidate/bin/python" ]]; then
        VLLM_VENV="$candidate"
        break
    fi
done
if [[ -z "$VLLM_VENV" ]]; then
    echo "ERROR: no usable venv found. Tried:"
    printf '  %s\n' "${VENV_CANDIDATES[@]}"
    echo "Build one: bash $(dirname "$0")/setup-vllm-qwen3.8.sh"
    exit 1
fi
HF_CLI="$VLLM_VENV/bin/hf"

REQUIRED_GIB=26

echo "=== Qwen3.8-27B-NVFP4 Setup (unsloth, mixed-precision NVFP4 + FP8) ==="

if [[ ! -x "$HF_CLI" ]]; then
    echo "ERROR: hf CLI not found at $HF_CLI"
    echo "Install with: $VLLM_VENV/bin/python -m pip install -U 'huggingface_hub[cli]'"
    exit 1
fi

VLLM_VER=$("$VLLM_VENV/bin/python" -c "import vllm; print(vllm.__version__)" 2>/dev/null | tail -1)
echo "vLLM venv:  $VLLM_VENV ($VLLM_VER)"
echo "Model dir:  $MODEL_DIR"

# Disk check — the download needs ~23.4 GiB plus slack for in-flight blobs.
mkdir -p "$(dirname "$MODEL_DIR")"
AVAIL_GIB=$(df -BG --output=avail "$(dirname "$MODEL_DIR")" | tail -1 | tr -dc '0-9')
if (( AVAIL_GIB < REQUIRED_GIB )); then
    echo "ERROR: need ~${REQUIRED_GIB}G free, only ${AVAIL_GIB}G available on $(dirname "$MODEL_DIR")"
    exit 1
fi
echo "Disk free:  ${AVAIL_GIB}G (need ~${REQUIRED_GIB}G)"

# Download. hf download is resumable — re-running after an interrupt is safe and
# skips complete files.
if [[ -f "$MODEL_DIR/config.json" && -f "$MODEL_DIR/model_mtp.safetensors" ]]; then
    echo ""
    echo "Model already present, re-running download to fill any gaps..."
else
    echo ""
    echo "Downloading $MODEL_REPO (~23.4 GiB)..."
fi
"$HF_CLI" download "$MODEL_REPO" --local-dir "$MODEL_DIR"

# Verification. Checks the three things that have actually bitten us before:
# missing shards, a phantom MTP head, and an unexpected quant_method.
echo ""
echo "=== Verifying download ==="
"$VLLM_VENV/bin/python" - "$MODEL_DIR" <<'EOF'
import json, os, sys

model_dir = sys.argv[1]
fail = []

cfg = json.load(open(os.path.join(model_dir, "config.json")))
qc = cfg.get("quantization_config", {})
text = cfg.get("text_config", {})

print(f"arch:          {cfg['architectures'][0]}  (model_type={cfg['model_type']})")
print(f"quant_method:  {qc.get('quant_method')}  format={qc.get('format')}")
print(f"kv_cache:      {qc.get('kv_cache_scheme', {}).get('type')}"
      f"{qc.get('kv_cache_scheme', {}).get('num_bits', '')}"
      f" {qc.get('kv_cache_scheme', {}).get('strategy', '')}")
print(f"layers:        {text.get('num_hidden_layers')}"
      f"  (full-attention every {text.get('full_attention_interval')})")
print(f"max_pos_emb:   {text.get('max_position_embeddings')}")

if qc.get("quant_method") != "compressed-tensors":
    fail.append(f"expected compressed-tensors, got {qc.get('quant_method')!r} "
                "— the start script's quantization assumptions no longer hold")

# Every shard the index references must exist at its full size.
index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
weight_map = index["weight_map"]
shards = sorted(set(weight_map.values()))
for shard in shards:
    path = os.path.join(model_dir, shard)
    if not os.path.exists(path):
        fail.append(f"missing shard {shard}")
print(f"shards:        {len(shards)} ({', '.join(shards)})")

on_disk = sum(os.path.getsize(p) for s in shards
              if os.path.exists(p := os.path.join(model_dir, s)))
expected = index.get("metadata", {}).get("total_size")
if expected and abs(on_disk - expected) > 1024:
    fail.append(f"size mismatch: {on_disk} on disk vs {expected} in index")
print(f"weights:       {on_disk / 2**30:.1f} GiB")

# The MTP check. config says the head exists; prove it from the tensors.
declared = text.get("mtp_num_hidden_layers", 0)
mtp_tensors = [k for k in weight_map if k.startswith("mtp.")]
print(f"MTP declared:  mtp_num_hidden_layers={declared}")
print(f"MTP tensors:   {len(mtp_tensors)}")
if declared and not mtp_tensors:
    fail.append("config declares an MTP head but NO mtp.* tensors shipped — "
                "do NOT pass --speculative-config (this is the Qwen3.6 trap)")
elif mtp_tensors:
    print("MTP head:      PRESENT — --speculative-config qwen3_5_mtp is safe")

if fail:
    print("\nFAILED:")
    for f in fail:
        print(f"  - {f}")
    sys.exit(1)
print("\nAll checks passed.")
EOF

echo ""
echo "=== Setup Complete ==="
echo "Model: $MODEL_DIR"
echo ""
echo "Start it (port 8096, the primary slot — stop whatever holds it first):"
echo "  bash $PROJECT_DIR/bin/start-qwen3.8-27b-nvfp4.sh"
echo ""
echo "Or promote it to the supervised primary by pointing"
echo "  supervisor/conf.d/agent-llm-primary.conf"
echo "at that script, then:"
echo "  supervisorctl -c $PROJECT_DIR/supervisor/supervisord.conf reread"
echo "  supervisorctl -c $PROJECT_DIR/supervisor/supervisord.conf update"
echo "  supervisorctl -c $PROJECT_DIR/supervisor/supervisord.conf restart agent-llm-primary"
