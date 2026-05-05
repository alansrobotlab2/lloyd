#!/usr/bin/env bash
set -euo pipefail

# Starts Qwen3.5-122B-A10B-NVFP4 via vLLM on the RTX PRO 6000 Blackwell.
# OpenAI-compatible API on port 8096.
#
# Stack: vllm-experimental venv (vLLM 0.20.2rc1.dev6 cu130 nightly,
# FlashInfer 0.6.9, PyTorch 2.11+cu130, sm_120). Initial tuning came from the
# SPARK repo (github.com/bjk110/SPARK_Qwen3.5-122B-A10B-NVFP4); current numbers
# were collected on vLLM 0.19.1 + FlashInfer 0.6.7 — re-bench after this stack
# bump to confirm decode rates / MTP optimum still hold.
#
# Tuning history (most recent first):
#   2026-05-05  Stability fix after supervisord crash-loop (37 EngineCore
#               restarts in ~12h, autorestart=true).  Two changes:
#                 (a) --gpu-memory-utilization 0.95 → 0.92.  Every crash had
#                     identical signature: GDN linear-attention's chunk_fwd_o
#                     attempts ~116-128 MiB allocation, GPU has <120 MiB free
#                     (PyTorch sitting at ~91.4 GiB / 94.97 GiB).  vLLM 0.20.x
#                     allocates more per-step activation buffer than 0.19.x at
#                     the same util target — the old 0.93 baseline on 0.19.1
#                     was stable, but 0.20.x at 0.95 fits at idle and OOMs
#                     under load.  Going one notch below the old baseline
#                     (0.92) restores margin; costs ~3 GiB KV cache, small
#                     relative to the ~12 GiB total at bt=8192.
#                 (b) num_speculative_tokens 5 → 4.  Stderr was looping on
#                     "matcher terminated after stop token, trying to accept
#                     token id 198" (\n) — the same MTP × tool-parser zombie-
#                     token race documented in start-35b-nvfp4.sh's 2026-04-19
#                     entry.  Higher MTP K = more drafted tokens past EOS =
#                     more zombies queued to the grammar matcher.  35B
#                     converged on K=4 on this same stack; 122B's
#                     2026-04-07 sweep had K=4 and K=5 within stddev (169.2
#                     ± 22.2 vs 169.4 ± 22.3 t/s) so the speed cost is
#                     negligible.  Race exists at any K≥1 but volume drops.
#               Re-bench after stability soak to confirm decode rate.
#   2026-05-04  Flipped --async-scheduling → --no-async-scheduling for an
#               experiment run on the bumped stack.  35B sibling on the same
#               venv hit vllm-project/vllm#40610 (async proposer race in
#               synchronize_input_prep) when async-scheduling + MTP were both
#               on; same speculative-mtp path is active here, so applying the
#               same workaround pre-emptively until #40610 lands upstream.
#               Cost on 35B was ~-13% sys throughput at n=3, ~-1% at n=8.
#               Re-enable once the upstream fix ships.
#   2026-04-07  MTP depth swept 1..7 on the new stack with a corrected bench
#               (the original sweep was undercounting tokens by ~mean_accept_len
#               because it counted SSE chunk arrivals instead of completion
#               tokens — MTP packs multiple accepted tokens into one SSE delta).
#               Decode rates at max_tokens=200 (single-shot, mixed prompts):
#                 mtp=1 → 107.0 t/s, ITL 9.4 ms, accept 0.93
#                 mtp=2 → 134.6 t/s, ITL 7.5 ms, accept 0.88
#                 mtp=3 → 160.0 t/s, ITL 6.3 ms, accept 0.82  (old SPARK default)
#                 mtp=4 → 171.6 t/s, ITL 5.9 ms, accept 0.76
#                 mtp=5 → 187.6 t/s, ITL 5.4 ms, accept 0.74  ★ winner
#                 mtp=6 → 194.0 t/s peak but ±7% noisy across runs
#                 mtp=7 → 184.6 t/s, regression
#               Re-validated at max_tokens=1000, runs=2 (12 measurements each):
#                 mtp=4 → 169.2 ± 22.2 t/s
#                 mtp=5 → 169.4 ± 22.3 t/s  ★ still winner (parsimony)
#                 mtp=6 → 167.7 ± 20.6 t/s
#               At long generation MTP=4/5/6 are statistically tied because
#               variance is dominated by content-dependent EOS (some prompts
#               finish in ~400 tok, others run to the cap, swinging per-prompt
#               decode rate). MTP=5 is kept as it dominates short-form and
#               ties long-form.
#               Net vs old MTP=3 SPARK default at 200-tok benches:
#                 +17% decode tok/s, −14% ITL.
#               Compared to vLLM 0.18.1 server-reported peaks (~135-170 t/s):
#                 +10-40% depending on prompt mix.
#   2026-04-07  --async-scheduling ENABLED. Decouples the engine scheduler
#               from the worker forward; tiny ITL improvement (-1%) but
#               collapses cold first-request TTFT from ~970ms → ~270ms because
#               the scheduler no longer blocks on spec-decode warmup.
#   2026-04-07  --moe-backend LEFT AT AUTO (selects FLASHINFER_CUTLASS).
#               Tried forcing flashinfer_cutedsl: crashes at startup with
#               "kernel does not support current device cuda" — the CuteDSL
#               NVFP4 MoE path doesn't support SM 12.0 yet. Auto-select
#               correctly falls through TRTLLM/CUTEDSL/CUTEDSL_BATCHED (none
#               supported on SM 12.0 for this model) and lands on
#               FLASHINFER_CUTLASS, which is already optimal for this device.
#   2026-04-07  Migrated to vllm-experimental venv on vLLM 0.19.1 nightly +
#               FlashInfer 0.6.7. vLLM 0.19 alone freed ~3.7 GiB of working
#               memory vs 0.18.1 at the same gpu-memory-utilization.
#   2026-04-07  --enable-flashinfer-autotune RE-ENABLED. The original SPARK
#               workaround (--no-enable-flashinfer-autotune) was for an
#               flashinfer 0.6.x crash on SM 12.0 that has since been fixed.
#               On 0.19+0.6.7 it cuts cold-prompt TTFT by ~40-54% on small/
#               medium prompts (155t: 596→273ms; 8.8K: 1158→686ms) with no
#               decode regression and no instability. Some MoE tactics still
#               get skipped at startup with "GPU lacks shared memory" warnings
#               — those are H100/H200-targeted kernels and the autotuner
#               correctly falls back to ones that fit SM 12.0's smem budget.
#   2026-04-07  max-num-batched-tokens: 16384 → 8192 (with MTP=5).
#               Long-form bench (1000-token generations, 2 runs each):
#                 bt=16384, KV  10.35 GiB → decode 169.4±22 t/s
#                 bt= 8192, KV  11.88 GiB → decode 176.1±26 t/s  ★ winner
#                 bt= 4096               → CRASH (hits Mamba floor)
#               Smaller batch frees +1.53 GiB of prefill activation workspace
#               for KV cache, no decode regression. Cold TTFT on the largest
#               (39K-token) prompt rises ~3% — acceptable trade.
#               HARD MINIMUM: do not set below 4288. Qwen3.5's hybrid linear-
#               attention layers enforce
#                 "block_size (4288) must be <= max_num_batched_tokens"
#               at engine init time (Mamba cache align mode). bt=4096 hits
#               this and crash-loops at startup. The 8192 / 4288 = 1.91x
#               headroom is comfortable.
#               Earlier in the day we also tried bt=32768: that cost 4.2 GiB
#               of activation workspace for negligible benefit at seq=1-2.
#   2026-04-07  gpu-memory-utilization: 0.93 → 0.95. Pure KV cache win on the
#               96GB Pro 6000 (KV available at bt=16384: 8.45 → 10.35 GiB,
#               ~2.16x the old 0.18.1 baseline; combined with bt=8192 below,
#               final KV available is 11.88 GiB ~= 2.48x old baseline).
#   (older)     max-num-batched-tokens: 8192 → 16384; max-num-seqs: 8 → 4;
#               --no-enable-log-requests; swap-space: 8 → 0.

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

VLLM_VENV="$HOME/lloyd/.venvs/vllm-experimental"
MODEL_DIR="$PROJECT_DIR/llm/models/Sehyo-Qwen3.5-122B-A10B-NVFP4"

if [[ ! -x "$VLLM_VENV/bin/python" ]]; then
  echo "vLLM experimental venv not found at $VLLM_VENV"
  exit 1
fi

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "Model not found at $MODEL_DIR"
  exit 1
fi

export PATH="$VLLM_VENV/bin:/opt/cuda/bin:/usr/bin:/usr/sbin:$PATH"
export LD_LIBRARY_PATH="/run/host/usr/lib:/opt/cuda/targets/x86_64-linux/lib:/opt/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_HOME="/opt/cuda"
# PCI bus order: index 0 = RTX 5090 (35B), index 1 = RTX PRO 6000 Blackwell (122B).
# Matches the convention in start-llm-35b.sh; without explicit CUDA_DEVICE_ORDER
# the default FASTEST_FIRST happens to put the Pro 6000 at index 0, but that's
# nondeterministic across driver/CUDA versions.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=1073741824
export VLLM_ENABLE_CUDAGRAPH_GC=1
export VLLM_USE_FLASHINFER_SAMPLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec "$VLLM_VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name Qwen3.5-122B-A10B primary \
  --port 8096 \
  --host 127.0.0.1 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --max-num-seqs 4 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.92 \
  --scheduling-policy priority \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend FLASHINFER \
  --enable-prefix-caching \
  --no-enable-log-requests \
  --enable-flashinfer-autotune \
  --no-async-scheduling \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --performance-mode interactivity \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 4}'
