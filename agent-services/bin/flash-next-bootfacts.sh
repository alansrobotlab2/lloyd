#!/usr/bin/env bash
# Pull the facts that decide an A/B arm out of one engine boot's log.
#
# These are the lines that say what the engine ACTUALLY did, as opposed to what
# the command line asked for: which NVFP4 MoE kernel the oracle picked, which
# GDN prefill kernel resolved, how much KV the profiler ended up with, and what
# the memory profiler says is still on the table. A config change that silently
# fell back to the previous kernel looks identical in the throughput numbers to
# one that did nothing, so read this before believing an arm.
set -uo pipefail
LOG="${1:-/home/alansrobotlab/lloyd/agent-services/logs/agent-llm-primary.log}"

echo "=== boot facts: $LOG ==="
grep -a -E \
  "non-default args|NvFp4 MoE backend|GDN prefill kernel|GDN decode kernel|\
Using .*attention backend|GPU KV cache size|Maximum concurrency|Available KV cache memory|\
Free memory on device|Model loading took|init engine|autotune|top-p & top-k|\
Chunked prefill is enabled|speculator|Fused multi-step|kv_cache_memory" "$LOG" \
  | grep -a -v -E "PleOffloadWorker|Received request|Avg prompt" \
  | tail -20 \
  | sed -E 's/^\(([A-Za-z]+) pid=[0-9]+\) //' \
  | cut -c1-400

echo
echo "=== errors/warnings this boot ==="
grep -a -E "ERROR|Traceback|does not support|Falling back|falling back|not supported|CUDA out of memory" "$LOG" \
  | grep -a -v -E "Received request" | tail -12 | cut -c1-300
