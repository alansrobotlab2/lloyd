#!/usr/bin/env bash
set -euo pipefail

# Benchmark: Qwen3.5-122B-A10B single GPU vs dual GPU
#
# GPU 0: RTX 5090 (32GB)
# GPU 1: RTX PRO 6000 Blackwell (96GB)
# Interconnect: PHB (PCIe via Host Bridge, no NVLink)
# Model: NVFP4 safetensors, ~76GB

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VLLM_PY="$PROJECT_DIR/.venvs/vllm/bin/python"
MODEL_DIR="$PROJECT_DIR/llm/models/Sehyo-Qwen3.5-122B-A10B-NVFP4"
RESULTS_DIR="$PROJECT_DIR/benchmarks"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

export LD_LIBRARY_PATH="/run/host/usr/lib:/opt/cuda/targets/x86_64-linux/lib:/opt/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_HOME="/opt/cuda"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=1073741824

mkdir -p "$RESULTS_DIR"

# Common vLLM engine args
COMMON_ENGINE=(
  --model "$MODEL_DIR"
  --trust-remote-code
  --enforce-eager
  --max-num-seqs 1
  --kv-cache-dtype fp8_e4m3
  --attention-backend FLASHINFER
)

# Benchmark args: test short and long prompts
BENCH_ARGS=(
  --batch-size 1
  --num-iters-warmup 2
  --num-iters 5
)

run_bench() {
  local label="$1"
  local outfile="$RESULTS_DIR/bench_122b_${label}_${TIMESTAMP}.json"
  shift
  echo ""
  echo "━━━ $label ━━━"
  echo ""

  for input_len in 128 512 2048; do
    for output_len in 128; do
      echo "  → input=$input_len output=$output_len"
      "$VLLM_PY" -m vllm.benchmarks.latency \
        "${COMMON_ENGINE[@]}" \
        "$@" \
        "${BENCH_ARGS[@]}" \
        --input-len "$input_len" \
        --output-len "$output_len" \
        --output-json "$outfile" \
        2>&1 | grep -E "Avg latency|latency|throughput|t/s|percentil|iter" | head -10
    done
  done

  echo "  Results: $outfile"
}

echo "=== Qwen3.5-122B-A10B Multi-GPU Benchmark ==="
echo "Timestamp: $(date -Iseconds)"
echo "Model: $MODEL_DIR ($(du -sh "$MODEL_DIR" | cut -f1))"
echo ""

# ─── Test 1: Single GPU — RTX PRO 6000 (GPU 1) ───
CUDA_VISIBLE_DEVICES=1 \
  run_bench "single_gpu1" \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90

# ─── Test 2: Dual GPU — Expert Parallelism ───
CUDA_VISIBLE_DEVICES=0,1 \
  run_bench "dual_ep" \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.92

# ─── Test 3: Dual GPU — Pipeline Parallelism ───
CUDA_VISIBLE_DEVICES=0,1 \
  run_bench "dual_pp" \
  --pipeline-parallel-size 2 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.92

echo ""
echo "━━━ All tests complete ━━━"
ls -la "$RESULTS_DIR"/*122b*${TIMESTAMP}* 2>/dev/null
