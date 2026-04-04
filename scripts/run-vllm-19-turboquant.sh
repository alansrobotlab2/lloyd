#!/usr/bin/env bash
# Launch Qwen3.5-122B-A10B-NVFP4 with vLLM 0.19 + TurboQuant KV cache quantization
#
# Uses vLLM 0.19.0 (NVFP4 CUTLASS MoE support, Blackwell NaN fixes) plus
# turboquant-vllm for KV cache compression.
#
# Runs on port 8098 to avoid conflicting with other instances.
#
# Usage:
#   ./scripts/run-vllm-19-turboquant.sh            # port 8098
#   PORT=8096 ./scripts/run-vllm-19-turboquant.sh  # replace production instance
#
# Test quality after starting:
#   python scripts/niah_test.py --url http://127.0.0.1:8098 --tag vllm19_turboquant

set -euo pipefail

VENV="/home/alansrobotlab/agent-services/.venvs/vllm-19-turboquant"
MODEL="/home/alansrobotlab/agent-services/llm/models/Sehyo-Qwen3.5-122B-A10B-NVFP4"
PORT="${PORT:-8098}"

echo "==> vLLM + TurboQuant (CUSTOM backend)"
echo "    Venv:     $VENV"
echo "    Model:    $MODEL"
echo "    Port:     $PORT"
echo ""

exec "$VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name Qwen3.5-122B-A10B \
  --port "$PORT" \
  --host 127.0.0.1 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --max-num-seqs 4 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.93 \
  --scheduling-policy priority \
  --attention-backend CUSTOM \
  --disable-hybrid-kv-cache-manager \
  --enable-prefix-caching \
  --no-enable-log-requests \
  --no-enable-flashinfer-autotune \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --performance-mode interactivity \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'
