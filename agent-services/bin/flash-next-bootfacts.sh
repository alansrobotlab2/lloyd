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

# === regression asserts, THIS boot only ===
# Everything above is for reading; this is for failing. The 2026-09-10 cutover
# to FP8 KV (a 692,263-token pool where BF16 held 398,175, and 3200-token
# pages) is what fixed the 09-09 stall, so a boot that quietly lands back on
# BF16 — a reverted conf, a dropped KV_CACHE_DTYPE, the old venv — serves
# perfectly well while undoing it. 2026-09-06 is the precedent: an automod
# rollback reverted the secondary's launcher under the same alias and port.
#
# Scoped to the last boot: the launcher prints "A/B config:" exactly once per
# invocation, right before exec (the rule flash-next-run-arm.sh slices on).
# For a deliberate arm: EXPECT_KV_DTYPE= (empty) skips the dtype check and
# EXPECT_KV_POOL_MIN=0 skips the size check.
# Exit status: 0 ok, 1 regression, 2 the boot has not logged its KV pool yet.
EXPECT_KV_DTYPE="${EXPECT_KV_DTYPE-fp8}"
EXPECT_KV_POOL_MIN="${EXPECT_KV_POOL_MIN-600000}"

start=$(grep -an '^A/B config:' "$LOG" | tail -1 | cut -d: -f1)
[[ -z "$start" ]] && start=$(grep -an 'non-default args' "$LOG" | tail -1 | cut -d: -f1)
start=${start:-1}
dtype=$(tail -n "+$start" "$LOG" | grep -a -m1 'Initializing a V1 LLM engine' \
  | grep -o -E 'kv_cache_dtype=[A-Za-z0-9_]+' | head -1 | cut -d= -f2)
pool=$(tail -n "+$start" "$LOG" | grep -a -o -E 'GPU KV cache size: [0-9,]+ tokens' \
  | tail -1 | grep -o -E '[0-9,]+' | tr -d ,)

echo
echo "=== regression asserts (boot from log line $start) ==="
if [[ -z "$pool" ]]; then
  echo "INCOMPLETE: this boot has not logged 'GPU KV cache size' yet"
  exit 2
fi
rc=0
if [[ -n "$EXPECT_KV_DTYPE" ]]; then
  if [[ "$dtype" == "$EXPECT_KV_DTYPE"* ]]; then
    echo "ok    kv_cache_dtype=$dtype"
  else
    echo "FAIL  kv_cache_dtype=${dtype:-<not logged>}, expected $EXPECT_KV_DTYPE" \
         "— check KV_CACHE_DTYPE/VLLM_VENV in supervisor/conf.d/agent-llm-primary.conf"
    rc=1
  fi
fi
if [[ "$EXPECT_KV_POOL_MIN" -gt 0 ]]; then
  if [[ "$pool" -ge "$EXPECT_KV_POOL_MIN" ]]; then
    echo "ok    GPU KV cache size $pool tokens (>= $EXPECT_KV_POOL_MIN)"
  else
    echo "FAIL  GPU KV cache size $pool tokens, expected >= $EXPECT_KV_POOL_MIN" \
         "(FP8 production: 692263; BF16: 398175)"
    rc=1
  fi
fi
exit $rc
