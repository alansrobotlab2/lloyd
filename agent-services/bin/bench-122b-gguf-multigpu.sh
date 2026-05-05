#!/usr/bin/env bash
set -euo pipefail

# Benchmark: Qwen3.5-122B-A10B GGUF single GPU vs dual GPU (PP layer split)
#
# GPU 0: RTX 5090 (32GB)
# GPU 1: RTX PRO 6000 Blackwell (96GB)
# Interconnect: PHB (PCIe via Host Bridge, no NVLink)
# Model: IQ4_NL UD GGUF split, ~58GB

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BENCH="$PROJECT_DIR/llm/llama.cpp/build/bin/llama-bench"
MODEL="$PROJECT_DIR/llm/models/Qwen3.5-122B-A10B-UD-IQ4-NL/Qwen3.5-122B-A10B-UD-IQ4_NL-00001-of-00003.gguf"
RESULTS_DIR="$PROJECT_DIR/benchmarks"

export LD_LIBRARY_PATH="/run/host/usr/lib:/opt/cuda/lib64:$PROJECT_DIR/llm/llama.cpp/build/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=== Qwen3.5-122B-A10B GGUF Multi-GPU Benchmark ==="
echo "Timestamp: $(date -Iseconds)"
echo "Model: $MODEL (~58GB)"
echo ""

# Common args
COMMON_ARGS=(
  -m "$MODEL"
  -ngl 99
  -fa 1
  -ctk q4_0
  -ctv q4_0
  -r 3
  -p 128,512,2048
  -n 128,512
  -o md
  --progress
)

# ─── Test 1: Single GPU — RTX PRO 6000 (GPU 1) ───
echo "━━━ Test 1: Single GPU — RTX PRO 6000 (GPU 1, 96GB) ━━━"
echo ""

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  "$BENCH" "${COMMON_ARGS[@]}" 2>&1 | tee "$RESULTS_DIR/122b_single_gpu1_${TIMESTAMP}.txt"

echo ""
echo "━━━ Test 2: Dual GPU — Even Split (50/50, layer mode) ━━━"
echo ""

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,0 \
  "$BENCH" "${COMMON_ARGS[@]}" \
  -sm layer \
  -ts 0.5/0.5 2>&1 | tee "$RESULTS_DIR/122b_dual_even_${TIMESTAMP}.txt"

echo ""
echo "━━━ Test 3: Dual GPU — Weighted Split (70/30 toward 6000) ━━━"
echo ""

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,0 \
  "$BENCH" "${COMMON_ARGS[@]}" \
  -sm layer \
  -ts 0.7/0.3 2>&1 | tee "$RESULTS_DIR/122b_dual_70_30_${TIMESTAMP}.txt"

echo ""
echo "━━━ All tests complete ━━━"
echo ""
ls -la "$RESULTS_DIR"/*122b*${TIMESTAMP}*
