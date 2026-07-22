#!/usr/bin/env bash
set -euo pipefail

# Benchmark: Qwen3.5-35B-A3B single GPU vs dual GPU (layer split + row split)
#
# GPU 0: RTX PRO 6000 Blackwell (96GB) — reserved for 122B
# GPU 1: RTX 3090 (24GB)
# GPU 2: RTX 3090 (24GB)
# Interconnect: PHB (PCIe via Host Bridge, no NVLink)
# Model: IQ4_NL GGUF, ~16.6GB (fits on single 3090)

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BENCH="$PROJECT_DIR/llm/llama.cpp/build/bin/llama-bench"
MODEL="$PROJECT_DIR/llm/models/gguf/Qwen3.5-35B-A3B-UD-IQ4_NL.gguf"
RESULTS_DIR="$PROJECT_DIR/benchmarks"

export LD_LIBRARY_PATH="/usr/lib:/opt/cuda/lib64:$PROJECT_DIR/llm/llama.cpp/build/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTFILE="$RESULTS_DIR/bench_35b_multigpu_${TIMESTAMP}.md"

echo "=== Qwen3.5-35B-A3B Multi-GPU Benchmark ==="
echo "Timestamp: $(date -Iseconds)"
echo "Model: $MODEL ($(du -h "$MODEL" | cut -f1))"
echo "Results: $OUTFILE"
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

# ─── Test 1: Single GPU — RTX 3090 (GPU 1) ───
echo "━━━ Test 1: Single GPU — RTX 3090 (GPU 1) ━━━"
echo ""

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
  "$BENCH" "${COMMON_ARGS[@]}" 2>&1 | tee "$RESULTS_DIR/single_gpu1_${TIMESTAMP}.txt"

echo ""
echo "━━━ Test 2: Dual GPU — Layer Split (50/50, pipeline-style) ━━━"
echo ""

# -sm layer: layers distributed across GPUs, communication only at layer boundaries
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2 \
  "$BENCH" "${COMMON_ARGS[@]}" \
  -sm layer \
  -ts 0.5/0.5 2>&1 | tee "$RESULTS_DIR/dual_gpu_layer_${TIMESTAMP}.txt"

echo ""
echo "━━━ Test 3: Dual GPU — Row Split (50/50, tensor-style) ━━━"
echo ""

# -sm row: weight matrix rows split across GPUs, requires allreduce after each layer
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2 \
  "$BENCH" "${COMMON_ARGS[@]}" \
  -sm row \
  -ts 0.5/0.5 2>&1 | tee "$RESULTS_DIR/dual_gpu_row_${TIMESTAMP}.txt"

echo ""
echo "━━━ All tests complete ━━━"
echo ""
echo "Results saved to $RESULTS_DIR/"
ls -la "$RESULTS_DIR"/*${TIMESTAMP}*
