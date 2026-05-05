#!/usr/bin/env bash
set -euo pipefail

# Starts RedHatAI/Qwen3.6-35B-A3B-NVFP4 via vLLM on GPU 1 (RTX PRO 6000 Blackwell, 96GB).
# OpenAI-compatible API on port 8096 (replaces 122B slot — one of them runs at a time).
#
# Model: Qwen3.6-35B-A3B, NVFP4 (ModelOpt) quantization.  Hybrid DeltaNet+Attention
# architecture: only 10/40 layers use KV cache — small KV footprint relative to 35B size.
# Weights ~24 GiB on disk. The NVFP4 MoE experts are the only large block actually
# quantized (~18 GiB packed U8 + F8_E4M3 scales); the rest is BF16 — lm_head/embed
# (~2 GiB), linear_attn DeltaNet (~2 GiB), MTP head (~1.7 GiB), vision tower
# (~0.9 GiB), plus self_attn U8 on the 10 full-attn layers (~0.14 GiB).
# Leaves ~70 GiB for KV cache on the 96GB card.
# Thinking enabled; per-request control via enable_thinking in the chat template.
#
# Stack: vllm-experimental venv (vLLM 0.20.2 nightly, FlashInfer 0.6.9, PyTorch
# 2.11+cu130, sm_120).  MoE backend left at AUTO — auto-selects FLASHINFER_CUTLASS
# for SM 12.0 in 0.6.8 era; with 0.6.9 the B12x CuTe-DSL path is now available
# for small expert-GEMM shapes (see 2026-05-03 entry below).
#
# Setup: bash setup/setup-vllm-experimental.sh
# Download: snapshot_download('RedHatAI/Qwen3.6-35B-A3B-NVFP4', local_dir=<MODEL_DIR>)
#
# Tuning history (most recent first):
#   2026-05-03  Re-disabled --async-scheduling via --no-async-scheduling.
#               During post-MTP-sweep cleanup, EngineCore crashed with
#                 torch.AcceleratorError: CUDA error: an illegal memory access
#                   was encountered
#                 File "vllm/v1/worker/gpu_model_runner.py", line 3536, in
#                   synchronize_input_prep
#                     self.prepare_inputs_event.record()
#               Exact site documented in vllm-project/vllm#40610 ("[SpecDecode]
#               Fix async proposer synchronization"), still open upstream against
#               this nightly.  Supervisor auto-restarted; would have wedged Lloyd
#               otherwise.  vLLM's auto-enable considers 'mtp' compatible
#               (it's listed in EagleModelTypes), so the default activates async
#               for MTP — but the proposer race still bites on this build.
#               Hardcoding --no-async-scheduling until #40610 lands.
#               Cost from 2026-04-23 measurements: ~-13% sys throughput at n=3,
#               ~-1% at n=8.  The 2026-05-03 stack-bump win (n=8: 1181 → 1974)
#               was largely from kernel/scheduler improvements, not async per se,
#               so most of that gain stays.  Re-bench after restart to confirm
#               where this lands.
#   2026-05-03  MTP sweep on bumped stack with Lloyd-shaped prompts (code review,
#               refactor, debugging, technical doc, algorithm explanation — gives
#               realistic 50-65% MTP acceptance vs trivial counting prompts at 99%
#               or creative writing at 35%).  600W, warm, 1000-tok decode, n=1,
#               min_tokens=1000 forced, 2 trials each:
#                 K=2 → 162.9 t/s ±3.4   accept 77.7%  emit 2.55  p0:86 p1:69
#                 K=3 → 175.0 t/s ±3.4   accept 61.5%  emit 2.85  p0:81 p1:61 p2:43
#                 K=4 → 188.6 t/s ±11.7  accept 53.5%  emit 3.14  p0:80 p1:61 p2:43 p3:31  ★ winner
#                 K=5 → 184.5 t/s ±9.4   accept 47.5%  emit 3.38  p0:81 p1:60 p2:43 p3:32 p4:21
#                 K=6 → 188.0 t/s ±29.1  accept 40.9%  emit 3.45  p0:81 p1:57 p2:41 p3:30 p4:21 p5:15
#               K=4 wins on tightest stddev; K=6 ties on mean but doubles variance.
#               Steps/sec roughly flat 54-64 sps across K — the throughput climb
#               with K stops at K=4 because per-position acceptance decay (p3 at
#               31%, p4 at 21%) eats the extra draft work.  Confirms 2026-04-18
#               choice; keeping num_speculative_tokens=4.
#               Regression check (same engine, K=4, original counting prompt,
#               apples-to-apples vs the 276 t/s n=1 from the concurrency sweep
#               below): 290.2 t/s ±0.5 / 99.4% accept / 4.98 emit/step. No
#               throughput regression vs that baseline (in fact +5%).
#               Real-world n=1 ceiling for this stack lands near 188 t/s for
#               Lloyd-style workloads — close to the 122B's ~180 t/s on similar
#               mix.  The 35B's "3x active params" advantage gets eaten by the
#               small-expert-GEMM kernel inefficiency in "Known ceiling" above —
#               single-user latency win is small; the real wins are concurrency
#               scaling (1974 t/s @ n=8 vs ~1200 for 122B) and KV footprint.
#   2026-05-03  Concurrency re-measure post-bump (600W, warm, 1000-tok decode,
#               MTP=4, qwen3_xml, --performance-mode interactivity, 2 runs each):
#                 1 → 276 t/s sys ±4    (per-user 276)  TTFT 0.12s
#                 3 → 780 t/s sys ±9    (per-user 261)  TTFT 0.20s
#                 6 → 1459 t/s sys ±88  (per-user 244)  TTFT 0.22s
#                 8 → 1974 t/s sys ±33  (per-user 247)  TTFT 0.23s   ★ common case
#               7.15x scaling at n=8 (was 4.85x in 2026-04-18 / 4.72x in 2026-04-23
#               post-async-drop). Single-user +14% (243→276), n=8 system +67%
#               (1181→1974). TTFT roughly halved (0.5s → 0.2s) across all
#               concurrency levels.  Likely contributions: 0.20.x async-scheduling
#               back on by default (and stable on this run), FlashInfer 0.6.9
#               kernel improvements even on CUTLASS auto-pick, vLLM 0.20.x
#               scheduler/CUDA-graph fastpaths, PyTorch 2.11 FP4 PTX intrinsics.
#               No errors, no wedge across two back-to-back sweeps — but vllm#40610
#               (async proposer race) is still open upstream, so non-deterministic
#               wedges remain possible under sustained load.
#   2026-05-03  Stack bump: vLLM 0.19.2rc1.dev122 → 0.20.2rc1.dev6, FlashInfer
#               0.6.8.post1 → 0.6.9, PyTorch 2.11.0 (already current).
#               FlashInfer 0.6.9 ships PR #3080 (B12x CuTe-DSL micro-kernel for
#               SM120 NVFP4 MoE) — directly addresses the small expert-GEMM
#               bottleneck (hidden=2048, moe_intermediate=512) that was the
#               documented "Known ceiling".  However the auto-picker still
#               selected FLASHINFER_CUTLASS for this model's shapes — to test
#               whether FLASHINFER_CUTEDSL or FLASHINFER_CUTEDSL_BATCHED beats
#               CUTLASS, force-select via the moe-backend override and
#               benchmark separately.
#               Async-scheduling: 0.20.x re-enables it BY DEFAULT.  Engine log
#               says "Asynchronous scheduling is enabled" without --async-scheduling
#               in argv.  vllm#40610 and #40036 (the spec-decode races that
#               drove the rollback on 2026-04-23) are still open, but this
#               nightly's behavior is empirically clean in the bench above.
#               Patch status: natural-sort patch upstreamed; aot_cache_fix
#               obsoleted by caching.py refactor (kept on disk in case MTP +
#               torch.compile regresses).
#   2026-04-23  Power envelope sweep at n=8 (450W–600W in 50W steps, nvidia-smi -pl).
#               Workload is memory-bandwidth / expert-routing bound on this MoE,
#               not compute-bound, so most of the GPU's 600W cap is headroom:
#                 600W → 1181 t/s sys   (per-user 220)   1.97 tok/W
#                 550W → 1144 t/s sys   (-3%)             2.08 tok/W
#                 500W → 1120 t/s sys   (-5%)             2.24 tok/W  ★ sweet spot
#                 450W → 1009 t/s sys   (-15%)            2.24 tok/W
#               Clear knee between 500W and 450W: -10% in one step vs -2% across
#               500/550. Efficiency plateaus at 500W — going below buys no extra
#               tok/W. TTFT flat ~0.5s across all caps. 500W saves 100W (17% of
#               envelope) for 5% throughput loss at max concurrency.
#   2026-04-23  Concurrency re-measure after --async-scheduling removal (600W,
#               warm, same methodology as 2026-04-18 run):
#                 1 → 250 t/s sys       (per-user 248)
#                 3 → 419 t/s sys       (per-user 198)
#                 6 → 806 t/s sys       (per-user 212)
#                 8 → 1181 t/s sys      (per-user 220)  ★ common case
#               4.72x scaling at 8-way (was 4.85x with async-scheduling). Cost
#               of dropping async: ~-13% sys throughput at batch=3, ~-1% at
#               batch=8. Single-user unchanged. TTFT flat ~0.5s.
#   2026-04-23  DROPPED --async-scheduling. Engine wedged at 100% GPU util / cool
#               temp / 8 running + 18 waiting, inference hung while /v1/models
#               still responded. Matches vllm-project/vllm#40610 "[SpecDecode]
#               Fix async proposer synchronization" (open, 2026-04-22): async
#               scheduling records prepare_inputs_event before the speculative
#               proposer's GPU work finishes, so the next batch enters
#               execute_model() and mutates persistent block-table metadata
#               while the previous proposer is still reading it. Non-
#               deterministic race — exactly our zombie-slot wedge. Also see
#               #40036 (spec decode draft rejection after scheduler rewind can
#               drive num_computed_tokens negative). Both open, not in our
#               0.19.2rc1.dev122 build. Cost: ~5-15% throughput / slight TPOT
#               bump. Re-enable once #40610 merges and we bump vLLM.
#   2026-04-19  ROLLBACK: qwen3_coder → qwen3_xml. Despite qwen3_coder matching
#               the model's chat_template.jinja format on paper, in practice it
#               wedged the engine: workers hit num_requests_running=5 with
#               generation_tokens_total frozen, and stderr looped on
#               grammar_matcher.cc:497 "matcher terminated after stop token,
#               trying to accept token id 198" (\n). The parser × MTP speculator
#               interaction races on the stop token. qwen3_xml is the empirical
#               last-known-good even though it doesn't match the on-paper format.
#   2026-04-19  tool-call-parser: qwen3_xml → qwen3_coder. The model's
#               chat_template.jinja emits <tool_call><function=...><parameter=...>
#               </parameter></function></tool_call>, which is exactly what
#               qwen3_coder parses; qwen3_xml was mismatched and silently failed
#               to detect calls in some edge cases.
#   2026-04-19  Added --override-generation-config '{"presence_penalty": 1.5}'.
#               The model's generation_config.json already sets temp=1.0/top_k=20/
#               top_p=0.95 (Unsloth thinking-mode defaults), but presence_penalty
#               was 0. Lloyd session 20260419_123516 showed a repetition loop
#               where the model restated the same "Found it..." paragraph ~15
#               times and reran identical shell calls 5x in a row. 1.5 is the
#               Unsloth anti-repetition recommendation; 0-2 is the safe range
#               (higher may trigger language mixing).
#   2026-04-18  MTP sweep under --performance-mode interactivity shifted optimum from
#               MTP=3 to MTP=4. Benchmark (1000-tok, 2 runs each, warm short_512):
#                 MTP=2 → 197 t/s ±17  accept 59%
#                 MTP=3 → 210 t/s ±16  accept 53%
#                 MTP=4 → 227 t/s ±9.5 accept 44%  ★ winner (tightest stddev)
#                 MTP=5 → 217 t/s ±19  accept 39%
#                 MTP=6 → 216 t/s ±25  accept 33%
#               Acceptance rate drops with depth but mean_accept_len rises; net
#               throughput peaks at MTP=4 because the extra draft step just offsets
#               the acceptance decline.
#   2026-04-18  --performance-mode interactivity ADDED. +10-15% single-user decode
#               (warm short_512: 207 → 237 t/s) and dramatic TTFT improvement
#               (5.6s → 0.44s cold, after first CUDA-graph compile). Also bumps
#               MTP acceptance (0.488 → 0.538) because the scheduler interrupts
#               draft sequences less.
#   2026-04-18  --max-num-seqs tested 1 vs 8. Setting 1 gave 221 t/s (more stable,
#               ±13) vs 8 at 224 t/s (±35). Kept 8 — 8 doesn't hurt single-user and
#               handles concurrency cleanly up to the limit.
#   2026-04-18  Concurrency sweep 1..8 (max-num-seqs=8, interactivity mode):
#                 1 user → 243 t/s sys      (per-user 243)
#                 3      → 516 t/s sys       (per-user 226)  ★ common case
#                 6      → 908 t/s sys       (per-user 223)
#                 8      → 1192 t/s sys      (per-user 211)
#               4.85x system-throughput scaling at 8-way vs 1-user, with only 13%
#               per-user degradation — excellent MoE batching efficiency.
#               TTFT stays ~0.5s across all concurrency levels up to max-num-seqs.
#               7 users regresses vs 6 (878 vs 908 sys) — scheduler prefers even
#               multiples of 2 when max-num-seqs=8.
#
# Known ceiling:
#   SM120 NVFP4 MoE on this model is bottlenecked by small expert GEMM shapes
#   (hidden=2048, moe_intermediate=512 per expert — 4x smaller area than 122B's
#   3072/1024). CUTLASS TMA WS kernels with K=128 tiles overflow the 99KB SMEM
#   budget; K=64 tiles require a FlashInfer+CUTLASS patch that never landed
#   (flashinfer-ai/flashinfer#2786 closed, not merged). The B12x CuTe-DSL
#   micro-kernel in flashinfer-ai/flashinfer#3080 was the awaited fix.
#
#   Tested 2026-05-03 with FlashInfer 0.6.9 + vLLM 0.20.2rc1.dev6:
#   forcing --moe-backend flashinfer_cutedsl crashes engine init with
#   "NvFp4 MoE backend 'FLASHINFER_CUTEDSL' does not support the deployment
#   configuration since kernel does not support current device cuda".
#   Auto-picker LISTS cutedsl among potential backends but actual device gating
#   in the kernel rejects SM120 — PR #3080 merged the kernel skeleton but the
#   B12x SM120 support gate is incomplete in 0.6.9.  Auto correctly avoids it
#   and selects FLASHINFER_CUTLASS.  Re-test after FlashInfer 0.7.x or whichever
#   release closes the SM120 support gate for B12x.
#
# Tuning constraints:
#   - max-num-batched-tokens: minimum > 4288 due to DeltaNet block-size constraint
#     (same as Qwen3.5-122B-A10B; bt=4096 crashes at engine init). 8192 is the
#     baseline — sweep down to 4352 if KV headroom matters more than prefill BW.
#   - kv-cache-dtype fp8_e4m3: standard on this stack, negligible quality impact.

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

VLLM_VENV="$PROJECT_DIR/.venvs/vllm-experimental"
MODEL_DIR="$PROJECT_DIR/llm/models/RedHatAI-Qwen3.6-35B-A3B-NVFP4"

if [[ ! -x "$VLLM_VENV/bin/python" ]]; then
  echo "vLLM experimental venv not found at $VLLM_VENV"
  echo "Run: bash setup/setup-vllm-experimental.sh"
  exit 1
fi

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "Model not found at $MODEL_DIR"
  echo "Download: snapshot_download('RedHatAI/Qwen3.6-35B-A3B-NVFP4', local_dir=$MODEL_DIR)"
  exit 1
fi

export PATH="$VLLM_VENV/bin:/opt/cuda/bin:/usr/bin:/usr/sbin:$PATH"
export LD_LIBRARY_PATH="/run/host/usr/lib:/opt/cuda/targets/x86_64-linux/lib:/opt/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_HOME="/opt/cuda"
# PCI bus order: index 0 = RTX 5090 (35B llama.cpp, port 8091),
#                index 1 = RTX PRO 6000 Blackwell (this server, port 8092).
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=1073741824
export VLLM_ENABLE_CUDAGRAPH_GC=1
export VLLM_USE_FLASHINFER_SAMPLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec "$VLLM_VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name Qwen3.6-35B-A3B-nvfp4 primary \
  --port 8096 \
  --host 127.0.0.1 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --max-num-seqs 8 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.95 \
  --scheduling-policy priority \
  --kv-cache-dtype fp8_e4m3 \
  --attention-backend FLASHINFER \
  --enable-prefix-caching \
  --no-enable-log-requests \
  --enable-flashinfer-autotune \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --no-async-scheduling \
  --performance-mode interactivity \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 4}' \
  --override-generation-config '{"presence_penalty": 1.5}'
