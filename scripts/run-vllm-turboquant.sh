#!/usr/bin/env bash
# Launch Qwen3.5-122B-A10B-NVFP4 with TurboQuant KV cache quantization
#
# Identical to the production vllm-experimental startup EXCEPT:
#   - Uses vllm-turboquant venv (vLLM with TQ patch)
#   - Sets --kv-cache-dtype turboquant_4bit (or 3bit/2bit via $TQ_BITS)
#   - Runs on port 8097 to avoid conflicting with the production instance
#
# Usage:
#   ./scripts/run-vllm-turboquant.sh            # 4-bit TQ, port 8097
#   TQ_BITS=3 ./scripts/run-vllm-turboquant.sh  # 3-bit TQ
#   PORT=8096 ./scripts/run-vllm-turboquant.sh  # replace production instance
#
# Test quality after starting:
#   python scripts/niah_test.py --url http://127.0.0.1:8097 --tag turboquant_4bit

set -euo pipefail

VENV="/home/alansrobotlab/agent-services/.venvs/vllm-turboquant"
MODEL="/home/alansrobotlab/agent-services/llm/models/Sehyo-Qwen3.5-122B-A10B-NVFP4"
PORT="${PORT:-8097}"
TQ_BITS="${TQ_BITS:-4}"
KV_DTYPE="turboquant_${TQ_BITS}bit"

echo "==> TurboQuant vLLM"
echo "    Venv:     $VENV"
echo "    Model:    $MODEL"
echo "    Port:     $PORT"
echo "    KV dtype: $KV_DTYPE"
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
  --kv-cache-dtype "$KV_DTYPE" \
  --attention-backend FLASHINFER \
  --enable-prefix-caching \
  --no-enable-log-requests \
  --no-enable-flashinfer-autotune \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --performance-mode interactivity \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'
