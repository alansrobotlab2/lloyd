#!/usr/bin/env bash
set -euo pipefail

# Starts Inferact/Qwen3.8-Flash-Next-NVFP4 via vLLM on the RTX PRO 6000 Blackwell
# (96 GiB), with the 51B N-gram (PLE) table offloaded to host RAM.
# OpenAI-compatible API on port 8096 — the shared primary slot, same as
# start-qwen3.8-27b-nvfp4.sh. Only one of those runs at a time.
#
# Setup / download: bash setup/setup-qwen38-flash-next.sh
# Venv:             bash setup/setup-vllm-qwen38-flash-next.sh
#
# MODEL
#   Qwen3.8-Flash-Next — the Qwen4 architecture preview. 125B main + 51B N-gram
#   embedding + 4B MTP = 180B total, 6B activated per token. 48 layers, hidden
#   2560, 512 experts (10 routed + 1 shared). Hybrid attention: Gated DeltaNet
#   paired with Qwen Sparse Attention (QSA), which selects micro-blocks rather
#   than individual tokens. 262,144 native context (1M via YaRN).
#
#   The N-gram table ("PLE") is the whole reason this fits: 20,000,000 trigram
#   rows x 2560 dims, injected at layer 2 (ple_layer_ids=[2]). It is a pure
#   lookup — no matmul — so it can live in host RAM and be prefetched
#   asynchronously while the GPU does real work.
#
# WHY THIS CHECKPOINT (Inferact, NVFP4 main + BF16 PLE)
#   Weight arithmetic for a single 96 GiB card, with the PLE offloaded:
#     Qwen/...-FP8            172.8 GiB total, FP8 PLE ~47.7  -> ~125 GiB on GPU. NO FIT.
#     RadixArk/...-NVFP4      126.0 GiB total, FP8 PLE  47.7  -> ~76.6 GiB on GPU. Fits,
#                             but the FP8-quantized PLE inside a ModelOpt NVFP4
#                             checkpoint trips vLLM issue #54765 at load:
#                               "no module or parameter named
#                                'ngram_embedding.weight_scale'"
#                             _get_ple_embedding_quant_method() only selects the
#                             FP8 PLE path when the *top-level* quant config is
#                             Fp8Config; for a modelopt checkpoint it returns
#                             None and the scale tensor has nowhere to land.
#                             Needs an out-of-tree load patch.
#     Inferact/...-NVFP4      170.3 GiB total, BF16 PLE 95.37 -> ~74.1 GiB on GPU.
#                             THIS ONE. A BF16 PLE has no weight_scale tensor, so
#                             #54765 cannot fire. The 95 GiB it costs in host RAM
#                             is free on a 251 GiB box — the DGX Spark / GX10
#                             reporters needed swapfiles only because they had
#                             121 GiB of *unified* memory.
#
#   GPU-resident, measured from the checkpoint (170.23 total - 95.37 PLE):
#     16 x 3.96  nvfp4_experts        63.4 GiB   the 512-expert MoE
#     4.64+4.65+0.76 model-0000{2,3,4} 10.0 GiB  dense / attn / embeddings / visual
#     1.49       nvfp4_experts_mtp     1.5 GiB   MTP draft head
#                                     --------
#                                      74.9 GiB
#   At --gpu-memory-utilization 0.9345 (89.7 GiB of the 96 GiB card, less ~1.1
#   GiB the desktop already holds) that leaves roughly 12-13 GiB for KV cache,
#   which goes a long way here: only a minority of the 48 layers hold a real KV
#   cache (the rest carry Gated DeltaNet recurrent state) and QSA is sparse.
#   The odd-looking 0.9345 is vLLM's own number. CUDA-graph memory profiling
#   (on by default since 0.21) charges its estimate against the fraction, so the
#   boot log says 0.93 is "equivalent to --gpu-memory-utilization 0.9255 without
#   CUDA graph memory profiling. To maintain the same effective KV cache size as
#   before, increase --gpu-memory-utilization to 0.9345." Taking that advice is
#   worth ~14k KV tokens. Do not round it back to 0.93.
#
#   If you need more KV: this checkpoint ships a vision tower (333 model.visual.*
#   tensors in shards 3-4, ~1-2 GiB). We serve text-only, so
#   --language-model-only skips loading it entirely and hands that back. It is
#   not the default here only because --limit-mm-per-prompt is what the rest of
#   this fleet uses; swap it in if KV is tight.
#
# VLLM BUILD — NOT stock nightly
#   PLE CPU offload is NOT on vLLM main. It lives in open PR #53899
#   (peakcrosser7/vllm @ release/qwen38next_offload). The venv is the pinned
#   per-commit wheel for that PR's base commit (45aed9b0c) with the branch's
#   Python files overlaid, plus two fixes. See setup-vllm-qwen38-flash-next.sh.
#   Stock `pip install vllm` has the qwen4_exp model but NO offload, and without
#   offload this checkpoint needs ~170 GiB of VRAM.
#
# TWO VENVS SERVE THIS SLOT (since 2026-09-10)
#   worker  ~/lloyd/.venvs/vllm-qwen38-flash-next   setup-vllm-qwen38-flash-next.sh
#           The build described above: 08-31 base + PR #53899's offload worker.
#   uva     ~/lloyd/.venvs/vllm-flash-next-main     setup-vllm-flash-next-main.sh
#           vLLM main as of 2026-09-10: UVA PLE offload (#54371, merged 09-09 —
#           the GPU reads the pinned host table directly, so no worker process,
#           no ptrace requirement, none of the three deadlocks below), the
#           rewritten QSA kernels, and PR #55557's FP8 main KV cache overlaid.
#   VLLM_VENV picks one. PLE_IMPL is detected from the venv's tree and decides
#   the executor flag, the ptrace preflight and the offload spelling, so a swap
#   cannot be half-applied. Measured on this card, same flags, 2026-09-10: main
#   decodes 254 tok/s single-stream in BF16 against 137 on the worker build;
#   the memory note fp8-kv-trial-2026-09-10 has the whole table.
#
# THREE KNOWN DEADLOCKS, AND WHY EACH FLAG BELOW EXISTS
#   1. uniproc gap (vLLM issue #53960). vLLM picks the uniproc executor at TP=1,
#      but spawn_ple_offload()/wait_ple_offload_ready() were only called from
#      multiproc_executor. The offload worker was never spawned and the GPU side
#      waited forever on a peer that did not exist. Fixed by 95dc96d1d012, which
#      IS in the branch — but we still pass --distributed-executor-backend mp
#      on the worker build because that is the configuration everyone who got
#      this serving actually ran. Drop it only if you want to re-test the
#      uniproc path. The uva build has no offload worker and runs uniproc.
#   2. async-scheduling shared-event race. PleOffloadConnector allocated ONE
#      _input_ready_event for the whole connector, assuming one request in
#      flight; async scheduling breaks that. Fixed by 4e8b849b8d97 (in branch).
#      ENABLED below since 2026-09-06, after the config had served for three
#      days without a wedge. If the engine ever hangs with requests running but
#      no tokens emitted, drop --async-scheduling first — that is this race.
#   3. TP=1 startup rendezvous race. The registration handoff can be lost, after
#      which the first warmup forward enqueues an untimed cuStreamWaitValue32
#      that nothing ever signals. Patched from davidtai/vllm PR #10 (into the
#      #53899 branch) — adds a bounded ACK barrier. A hung boot never reaches
#      the "init engine ... took Xs" log line; that is how you tell this apart
#      from a merely slow cold boot (expect 5-15 min, it reads 170 GiB).
#
# SM120 / PDL — patched in the venv, not here
#   current_platform.is_arch_support_pdl() is `major >= 9`, so it returns True on
#   sm_120 (major 12). The QSA metadata kernel then launches with PDL and the
#   dependent kernel waits forever on prompts over ~8k tokens. The venv setup
#   forces _metadata_launch_pdl() to False in
#   vllm/models/qwen4_exp/common/qsa_cache.py. Without that patch this serves
#   short prompts fine and then hangs on the first real one.
#
# HOST-SIDE GOTCHAS
#   - Host RAM: the BF16 PLE table is 95.37 GiB resident, pageable. Expect RSS
#     on the ple-offload worker around that. 251 GiB total here, so no swapfile
#     is needed (unlike the 121 GiB unified-memory boxes in the issue thread).
#   - REQUIRED: kernel.yama.ptrace_scope must be 0. Confirmed empirically on
#     2026-09-03 — this is NOT Docker-only, it bites on bare metal too.
#     The GPU worker hands the offload process a CUDA IPC tensor handle, and
#     torch rebuilds it with pidfd_getfd, which needs PTRACE_MODE_ATTACH:
#       accept_registrations -> pickle.loads -> rebuild_cuda_tensor
#         -> _new_shared_cuda -> RuntimeError: pidfd_getfd: Operation not permitted
#     The tracer is the CHILD (PleOffloadWorker) and the tracee is its PARENT
#     (VLLM::Worker). Yama scope 1 permits tracing descendants only, and a
#     parent is not a descendant of its child — so scope 1 always fails here.
#     Set persistently via /etc/sysctl.d/99-ptrace.conf:
#       kernel.yama.ptrace_scope = 0
#     Check before blaming anything else:
#       sysctl kernel.yama.ptrace_scope   # must print 0
#     Symptom if unset: model loads fine (75.1 GiB), "PleOffload: registered"
#     prints, then the worker dies and the API server never binds :8096.
#     The narrower alternative, if you ever need scope 1 back, is to patch
#     prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY) into the GPU worker before it
#     spawns the offload process.
#   - ulimit -l is 8192 KB. CUDA pinned staging buffers normally do not count
#     against RLIMIT_MEMLOCK, but if the worker dies allocating staging memory,
#     raise it.
#
# TOOL CALLING
#   --tool-call-parser qwen3_xml, matching both the vLLM recipe and every other
#   model in this fleet. The upstream GitHub README says qwen3_coder; do not
#   follow it — qwen3_coder wedges the engine on the stop-token path for this
#   family, and the harness's XML tool-call recovery expects qwen3_xml.

# BEFORE YOU START THIS: free the card.
#   agent-llm-primary (start-qwen3.8-27b-nvfp4.sh) runs at
#   --gpu-memory-utilization 0.95 and holds ~95.6 GiB of the Blackwell. Both
#   models want port 8096 and the same GPU, so exactly one runs at a time:
#     SUP=~/.local/share/uv/tools/supervisor/bin/supervisorctl
#     $SUP -c ~/lloyd/agent-services/supervisor/supervisord.conf stop agent-llm-primary
#   Note also that the desktop session keeps ~1.1 GiB on this card (remmina and
#   the livekit worker each hold ~552 MiB). --gpu-memory-utilization is a
#   fraction of TOTAL card memory, not of what is free, so that 1.1 GiB comes
#   out of our headroom — which is part of why 0.93 rather than 0.95 below.
#
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

VLLM_VENV="${VLLM_VENV:-$HOME/lloyd/.venvs/vllm-qwen38-flash-next}"
MODEL_DIR="${MODEL_DIR:-$PROJECT_DIR/llm/models/Inferact-Qwen3.8-Flash-Next-NVFP4}"

# Context length. Deliberately NOT 262144 on first boot.
#   Budget at --gpu-memory-utilization 0.9345 on the 96 GiB card:
#     89.7 GiB total - 74.1 weights - ~2 activation - ~0.3 cudagraph = ~13 GiB KV.
#   How many tokens that buys depends on how many of the 48 layers hold a real KV
#   cache (the rest are Gated DeltaNet recurrent state) and on QSA's block
#   sparsity, which this fleet has not measured yet. If max-model-len exceeds what
#   the KV cache can hold for ONE request, vLLM refuses to start with
#   "max seq len is larger than the maximum number of tokens that can be stored".
#   MEASURED 2026-09-03 on this box at 0.93: "GPU KV cache size: 314,572
#   tokens, Maximum concurrency for 262,144 tokens per request: 1.20x", so the
#   full 262144 native context fits with ~52k tokens to spare. Set to 262144,
#   which also matches config.yaml's context_length for the primary slot.
#   Concurrency is the tradeoff the boot log reports next to it; re-read that
#   line after any change here rather than trusting this comment.
#   Drop back with MAX_MODEL_LEN=131072 if concurrency matters more than reach.
#   To go past 262144 you need YaRN, and it must go in a config.json, NOT in
#   --hf-overrides: vLLM forwards only *callable* overrides to the MTP draft's
#   config (SpeculativeConfig.compose_draft_hf_overrides drops dicts), so a
#   dict leaves the drafter on plain RoPE with a 262k horizon while the target
#   runs past it, and nothing in the log says so. bin/flash-next-yarn-model.py
#   writes a shadow checkpoint (weights symlinked, config.json rewritten):
#     bin/flash-next-yarn-model.py --factor 2.0
#     MODEL_DIR=.../Inferact-Qwen3.8-Flash-Next-NVFP4-yarn2 MAX_MODEL_LEN=524288 bash bin/start-qwen38-flash-next.sh
#   The KV pool bounds it — vLLM refuses a length one request cannot hold:
#   ~398k tokens in BF16, ~692k with KV_CACHE_DTYPE=fp8 at the 11.5 GiB below,
#   so 1M is not reachable on one card and factor 2.0 (524,288) needs fp8.
#   Static YaRN also taxes short prompts (model card), so it stays an opt-in arm.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
# The yarn shadow config already derives 262144*factor, so this is belt and
# braces for a hand-edited one; vLLM refuses a longer max-model-len otherwise.
if (( MAX_MODEL_LEN > 262144 )); then
  export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
fi

if [[ ! -x "$VLLM_VENV/bin/python" ]]; then
  echo "vLLM venv not found at $VLLM_VENV"
  echo "Build it: bash $PROJECT_DIR/setup/setup-vllm-qwen38-flash-next.sh   (worker build)"
  echo "      or: bash $PROJECT_DIR/setup/setup-vllm-flash-next-main.sh     (vLLM main, uva)"
  exit 1
fi

# Which PLE offload implementation the venv carries. Read off the tree rather
# than a flag so the venv and the flags below cannot disagree: the worker build
# ships vllm/v1/ple_offload/, main's UVA build does not.
if compgen -G "$VLLM_VENV/lib/python*/site-packages/vllm/v1/ple_offload/worker.py" >/dev/null; then
  PLE_IMPL=worker
else
  PLE_IMPL=uva
fi

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "Model not found at $MODEL_DIR"
  echo "Download: bash $PROJECT_DIR/setup/setup-qwen38-flash-next.sh"
  exit 1
fi

# Refuse to start unpatched: without the offload code this silently needs ~170
# GiB of VRAM and dies at load, which is a confusing way to find out.
if ! "$VLLM_VENV/bin/python" -c "import vllm.envs as e; raise SystemExit(0 if hasattr(e,'VLLM_PLE_CPU_OFFLOAD') else 1)" 2>/dev/null; then
  echo "ERROR: this venv has no VLLM_PLE_CPU_OFFLOAD — the PLE offload overlay is missing."
  echo "Rebuild: bash $PROJECT_DIR/setup/setup-vllm-qwen38-flash-next.sh"
  exit 1
fi

# Hard preflight (worker build only): without ptrace_scope=0 the CUDA IPC
# handshake below fails and you lose ~4 minutes loading 170 GiB before finding
# out. Fail in 10ms instead. The uva build has no second process to trace.
PTRACE_SCOPE="$(cat /proc/sys/kernel/yama/ptrace_scope 2>/dev/null || echo 0)"
if [[ "$PLE_IMPL" == "worker" && "$PTRACE_SCOPE" != "0" ]]; then
  echo "ERROR: kernel.yama.ptrace_scope is $PTRACE_SCOPE, must be 0."
  echo "  The PLE offload worker rebuilds a CUDA IPC tensor from its parent via"
  echo "  pidfd_getfd; Yama only allows tracing descendants, and the parent is not"
  echo "  a descendant of the child, so this always fails at scope 1."
  echo "  Fix (persistent):"
  echo "    echo 'kernel.yama.ptrace_scope = 0' | sudo tee /etc/sysctl.d/99-ptrace.conf"
  echo "    sudo sysctl --system"
  exit 1
fi

# MTP draft head. Present in this checkpoint as nvfp4_experts_mtp.safetensors
# (1.49 GiB, mtp_num_hidden_layers=1). Guarded the same way as the 27B script:
# a future re-download that drops it should degrade to plain decode, not wedge
# the engine on a missing draft model.
#
# MTP_ENABLED=0 turns speculative decode off without editing this file. It is an
# A/B knob only -- leave it at 1. Measured 2026-09-06, both arms on a freshly
# booted engine, batch-1 decode, identical 60,004-token reuse probe:
#
#                      decode        KV pool        concurrency   prefix reuse
#   MTP on (default)   181.9 tok/s   330,159 tok    1.26x         96.0%
#   MTP off             78.2 tok/s   465,046 tok    1.77x         99.3%
#
# MTP is worth 2.33x on decode; it costs 135k tokens of KV (the draft head's
# 2.12 GiB). Keep it on. Mean acceptance length 3.047 of 4 over 7,511 samples.
#
# THE SCARY BOOT WARNING IS BENIGN. kv_cache_utils.py::_warn_if_unannotated_eagle_mamba
# fires here -- "prefix-cache reuse across requests will be disabled" -- because vLLM
# cannot identify the qwen4_exp draft group: rule 1's marker (non_causal_multi_token_decode)
# exists only on MLA attention and QSA is not MLA, and rule 2 is gated on
# model_type == "deepseek_v4". So the coordinator conservatively flags every group,
# Mamba included, and applies the EAGLE last-block drop. Measured cost of that drop:
# 1,984 tokens per request (96.0% reuse vs 99.3%), i.e. 3.3% -- NOT the whole cache.
# Not worth patching the venv to fix.
#
# Also measured, and the reason an obvious test misleads: cross-request reuse needs
# TWO warm-up passes before it engages (passes 1-2 cache nothing, pass 3+ hits ~96%
# and runs ~18x faster). A 2-pass A/B shows 0% on BOTH arms and proves nothing.
# Mechanism behind the warm-up not yet identified.
#
# Reproduce with agent-services/bin/bench-prefix-reuse.py (decode x3 then a 5-pass
# reuse probe reading usage.prompt_tokens_details.cached_tokens, which is
# per-request and so immune to the agent's own traffic). Do NOT use
# vllm:prefix_cache_hits_total as a reuse rate: it sums across all KV groups
# (673,179 queries for one 60k prompt) and sat at ~68% while real cross-request
# reuse was zero.
MTP_ENABLED="${MTP_ENABLED:-1}"
# Draft depth. 3 is the measured default (see the table above). An A/B knob:
# position-2 acceptance is 51%, so k=4 may still pay at batch 1 while k=2 may
# win at batch 8, where the wasted verify slots cost more than the hit rate.
MTP_TOKENS="${MTP_TOKENS:-3}"
SPEC_ARGS=()
if [[ "$MTP_ENABLED" != "1" ]]; then
  echo "NOTE: MTP_ENABLED=$MTP_ENABLED — starting WITHOUT speculative decode"
elif [[ -f "$MODEL_DIR/nvfp4_experts_mtp.safetensors" ]]; then
  SPEC_ARGS=(--speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": $MTP_TOKENS}")
else
  echo "WARNING: nvfp4_experts_mtp.safetensors missing — starting WITHOUT speculative decode"
fi

# ── one-shot A/B env ──────────────────────────────────────────────────────
# bin/flash-next-run-arm.sh writes this file, and it is CONSUMED: sourced once
# and deleted in the same breath, so it can change exactly one boot. That is
# the whole design. supervisord passes a fixed `environment=` and offers no
# per-restart override, so an A/B under supervision needs a file — and a file
# that persisted would be a config that silently outlives the experiment,
# which is the failure this slot can least afford. If the sweep dies between
# the source and the delete, the next boot is production config.
# ARM_ENV overrides the path, and exists for one caller: a DRY_RUN from a test
# must not consume an arm that flash-next-run-arm.sh has staged for the real
# boot (tests/test_flash_next_launcher.py points it at a scratch file).
ARM_ENV="${ARM_ENV:-$PROJECT_DIR/logs/flash-next-arm.env}"
if [[ -f "$ARM_ENV" ]]; then
  echo "consuming one-shot arm env: $ARM_ENV"
  cat "$ARM_ENV"
  # shellcheck disable=SC1090
  source "$ARM_ENV"
  rm -f "$ARM_ENV"
fi

# ── A/B knobs ─────────────────────────────────────────────────────────────
# Every default below reproduces what this slot served before the 2026-09-08
# sweep, so an unset environment is the old config exactly. They exist so an
# arm is ONE env var rather than an edit to this file: this file is tracked,
# and a dirty tree is what scripts/automod/gate.py and promote.py both refuse
# to run against — editing it per arm would switch the self-modification loop
# off for the length of the sweep.
#
#   bash bin/start-qwen38-flash-next.sh                  # today's config
#   MOE_BACKEND=flashinfer_b12x bash bin/start-...        # one arm
#
# Measure with bin/bench-flash-next.py (one arm per boot, pool paused) and
# ALWAYS read bin/flash-next-bootfacts.sh afterwards. A backend that was
# rejected and silently fell back to the previous kernel is indistinguishable
# from one that made no difference if you only look at throughput.

MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9345}"

# Exact KV budget in bytes, which beats tuning the fraction: the fraction is a
# share of TOTAL card memory, so anything else resident on the card (the
# desktop holds ~1.1 GiB when remmina and the livekit worker are up) silently
# comes out of KV. Empty = size KV from the fraction, as before.
# The boot log prints the two numbers worth knowing here — "Replace
# gpu_memory_utilization config with --kv-cache-memory=N" for the current
# budget, and a second N to fully use the card. Do not take the second one:
# it assumes the card is otherwise empty.
# 13.5 GiB, measured 2026-09-08. The fraction was yielding 9.53 GiB (330,159
# KV tokens, 1.26x at 262k); this gives 466,191 tokens and 1.78x for no
# measurable cost to decode or prefill. Derived from the profiler's own
# accounting on this card: 94.46 GiB free at startup, 77.35 weights + 1.91
# peak activation + 0.24 cudagraph = 79.50, so ~14.9 is physically available
# and 13.5 leaves ~1.4 GiB for the desktop (remmina and the livekit worker
# take ~1.1 GiB between them when they are on this card).
# Do NOT take the boot log's larger "fully utilize gpu memory" suggestion:
# it assumes the card is otherwise empty, and it is not.
# NOTE: setting this SKIPS memory profiling entirely, so anything that
# allocates later is unaccounted for. --enable-flashinfer-autotune plus this
# put the card at 96874 of 97887 MiB.
#
# THE WHOLE CARD IS RESERVED FOR vLLM — nothing else may allocate on GPU 1,
# and on 2026-09-08 the only holders were VLLM::Worker and the PLE offload
# worker. An older comment in this tree budgeted ~1.1 GiB here for remmina and
# the livekit worker; that is stale. But do NOT conclude from that that the
# rest of the card is free for KV.
#
# THE HEADROOM IS FOR vLLM ITSELF, AND THIS IS THE EXPENSIVE LESSON OF THE DAY.
# Setting this flag SKIPS memory profiling, so nothing reserves the transient
# activations a forward pass needs. At 13.5 GiB the card sat at 96,876 of
# 97,887 MiB and served happily for 50 minutes — then a long prefill reached
# ple_layer.py::_short_conv_dilated_prefill_, asked for 444 MiB against 362 MiB
# free, and took the engine down with a plain CUDA OOM. Nothing was wrong with
# the config at boot; it was wrong on the first prefill big enough to need its
# scratch buffer, which is a workload this box sees constantly (27% of prompts
# are over 50k tokens).
#
# 11.5 GiB is the MEASURED safe point, not a guess: booted at this value the
# card reads 94,232 of 97,887 MiB, leaving 3,057 MiB — comfortably above both
# the profiler's own 1.91 GiB peak-activation estimate and the 444 MiB
# allocation that failed. It still yields ~414k KV tokens and 1.58x at 262k,
# against 330,159 and 1.26x before this work.
#
# If you want more KV, do NOT just raise this number. Either drop
# max_num_batched_tokens (the transient scales with the prefill chunk) or go
# back to --gpu-memory-utilization, whose profiling pass exists to account for
# exactly this and which is what was quietly bypassed here.
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-12348030976}"

# KV cache dtype. Empty = the engine default (BF16). 'fp8' stores the 12 QSA
# layers' K/V as e4m3 with unit scales (the checkpoint ships no calibrated
# k/v scales) and needs the uva venv, where PR #55557 is overlaid; the worker
# venv's QSA backend hard-rejects anything but BF16, so the guard below fails
# in 10 ms rather than after a 4-minute load.
#
# MEASURED 2026-09-10, uva venv, same 11.5 GiB, MTP on (memory note
# fp8-kv-trial-2026-09-10 has the full table):
#   pool        398,175 -> 692,263 tokens (x1.74; 1.52x -> 2.64x at 262k)
#   quality     needle 12/12 through 239k; per-position logprob deviation on
#               78k tokens of this repo's Python sits on BF16's own
#               run-to-run noise (sigma 0.35 vs 0.34 nats)
#   prefill     9.3k -> 9.6k tok/s; and a 239k prompt 128 s -> 26 s, because
#               the >200k QSA-indexer cliff (see lloyd-f3's admission-stall
#               work) tracks the attention page count, which fp8 halves by
#               doubling the page to 3200 tokens
#   per step    unchanged: 18.1-18.7 ms at batch 1 on every prompt below,
#               both dtypes (the QSA kernel times identically standalone
#               and is ~0.07 ms of a step; the step is MoE and dense GEMMs)
#   MTP         the drafter reads the same e4m3 cache and mispredicts more,
#               and THAT is the whole decode cost. BF16 -> FP8 single-stream
#               tok/s (acceptance) on 512-token greedy completions:
#                 code                 182 -> 176  (0.77 -> 0.72)
#                 JSON tool-call text  211 -> 213  (0.95 -> 0.96)
#                 prose                170 -> 141  (0.69 -> 0.52)
#                 thinking             180 -> 153  (0.79 -> 0.62)
#                 32k-context summary  143 -> 140  (0.55 -> 0.49)
#               i.e. 0-17% by text type, ~7% over the suite. The A/B
#               harness's "count upward" prompt is the pathological case
#               (3.67 -> 1.75 tokens/step, 254 -> 112 tok/s) and must not
#               be read as the engine's decode speed.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"

# Skip the vision tower entirely. Measured from the checkpoint's safetensors
# headers: 333 model.visual.* tensors, 0.84 GiB, which at this config's 30.3
# KiB/token is ~29k KV tokens. We serve text-only, so it is pure waste — the
# older --limit-mm-per-prompt still LOADS the tower and merely refuses to be
# handed images. Set 0 to go back to that.
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-1}"

# NVFP4 MoE kernel. Empty = the oracle's own order, which lands on
# FLASHINFER_CUTLASS here. 'flashinfer_b12x' is the SM120-native FP4 path;
# the oracle excludes it from AUTO-selection only pending an upstream CUTLASS
# SM121 guard, so it has to be asked for by name. The experts are essentially
# the whole decode cost on this model, so this is the widest single lever.
#
# DO NOT SET THIS TO flashinfer_b12x, and the reason is not only that it
# crashes. Tried 2026-09-08: the oracle selects it ("Using 'FLASHINFER_B12X'
# NvFp4 MoE backend"), then the engine dies in determine_available_memory with
# `CUDA error: an illegal memory access was encountered` before serving a
# token. supervisord restarts on defaults, so the failure reads as a null
# result unless you check — hence the boot guard in bin/flash-next-run-arm.sh.
#
# THE POINT IS THAT THERE IS NOTHING TO WIN HERE. b12x is the upgrade path for
# NVFP4 checkpoints that resolve as **W4A16** and would otherwise fall back to
# MARLIN. Upstream's own auto-selection proposal (vllm#47577) positions it
# *after* FLASHINFER_CUTLASS precisely so that "W4A4 checkpoints keep their
# current selection". This checkpoint is W4A4 — every expert carries an
# `input_scale` (75,264 of them) and quantization_config says
# with_input_scale: true, per-tensor — so it already selects the backend that
# upstream would keep it on. b12x is a sidegrade at best for this model, and
# the measured outcome is a dead engine.
#
# Known-broken upstream on this exact hardware, all open as of 2026-09-08:
#   vllm#50189  Xid 31 MMU fault, illegal write, flashinfer_b12x on SM120
#               RTX PRO 6000 TP=1 under chunked prefill — our symptom exactly
#   vllm#49476  b12x workspace allocated lazily inside profile_run, which is
#               the phase our boot died in
#   vllm#47365  empty/garbage output under TP or PP on SM120
# b12x's own README says it is "not intended to be used in production ...
# For mission-critical use cases please use FlashInfer, CUTLASS or TRTLLM."
# Revisit only if a checkpoint without activation scales ever lands here.
MOE_BACKEND="${MOE_BACKEND:-}"

# GDN prefill kernel for the 36 linear-attention layers. Empty = auto, which
# resolves to Triton/FLA on this card. 'flashinfer' needs the SM12x gate that
# upstream added on 2026-09-08 (vllm f6326f53b); without that patch in the
# venv, asking for it logs a fallback and serves Triton anyway — which is
# precisely the silent-fallback case bootfacts.sh exists to catch.
#
# MEASURED 2026-09-08, and this is the single biggest win of that sweep:
#   decode  120.2 -> 138.3 tok/s (+15%), and dead flat across runs
#   prefill 8840 -> 9290 tok/s at 35k, 8235 -> 8569 at 107k
#   MTP acceptance 30% -> 42%
# The decode gain is not a paradox even though this names the PREFILL kernel:
# with MTP k=3 every verify step is a 4-token forward, which takes the
# multi-token path, so the prefill kernel sits squarely on the decode path
# whenever speculative decoding is on. The acceptance jump is most of the
# gain, and it is a numerics difference — FlashInfer's SM120 path carries the
# recurrent state in float32.
# REQUIRES the venv patch: bin/flash-next-gdn-sm12x-patch.py.
GDN_PREFILL_BACKEND="${GDN_PREFILL_BACKEND:-flashinfer}"

# FlashInfer autotune at warmup. Off since this slot's first boot with no
# recorded reason; it tunes the very CUTLASS grouped-GEMM configs this model
# runs on, and the 27B on this same card kept a populated autotune cache.
# Costs a slower boot, caches to ~/.cache/vllm/flashinfer_autotune_cache.
#
# MEASURED 2026-09-08: leave it OFF, and the reason is variance rather than a
# mean. It gives the best prefill of any arm (9481/8853) and the best batch
# aggregate (conc-8 419 vs 380), but single-stream decode becomes a lottery:
# 65/87/82, then 67/74/133, then 92/109/259 tok/s across three measurements,
# where every non-autotune arm held within 0.5% (138.6/138.3/138.2). Not
# memory pressure — it persisted with 3 GiB of card free. Autotune runs at
# 8192 tokens and a batch-1 decode is a different shape entirely, so what it
# picks for the big shape can be wrong for the small one. An interactive
# agent wants a predictable 138 over a mean of maybe-140.
FLASHINFER_AUTOTUNE="${FLASHINFER_AUTOTUNE:-0}"

# Chunked-prefill budget, in tokens per engine step. Empty = vLLM's default
# (8192 here); production sets 4096 in agent-llm-primary.conf. A cold long
# prompt admitted beside a decoding stream costs that stream one step per
# chunk. Measured 2026-09-10, a cold 200k prompt beside a 119k decode
# (bench-admission-stall.py cold), neighbour step p50 / prefill time:
#   8192  608 ms / 20.2 s    and the prefill references ~2.5x its KV
#   4096  333 ms / 21.6 s    no overshoot
#   2048  195 ms / 25.7 s
# architecture/vllm-throughput-mitigation.md §3.3 has the rest.
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"

# Escape hatch for one-off arms. Word-split deliberately.
EXTRA_ARGS="${EXTRA_ARGS:-}"

AB_ARGS=()
if [[ -n "$KV_CACHE_MEMORY_BYTES" ]]; then
  AB_ARGS+=(--kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES")
fi
if [[ "$LANGUAGE_MODEL_ONLY" == "1" ]]; then
  # Mutually exclusive with --limit-mm-per-prompt in practice: there is no
  # multimodal input to limit once the tower is not loaded.
  AB_ARGS+=(--language-model-only)
else
  AB_ARGS+=(--limit-mm-per-prompt '{"image": 0, "video": 0, "audio": 0}')
fi
[[ -n "$MOE_BACKEND" ]] && AB_ARGS+=(--moe-backend "$MOE_BACKEND")
[[ -n "$MAX_NUM_BATCHED_TOKENS" ]] && AB_ARGS+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
[[ -n "$GDN_PREFILL_BACKEND" ]] && AB_ARGS+=(--gdn-prefill-backend "$GDN_PREFILL_BACKEND")
if [[ "$FLASHINFER_AUTOTUNE" == "1" ]]; then
  AB_ARGS+=(--enable-flashinfer-autotune)
else
  AB_ARGS+=(--no-enable-flashinfer-autotune)
fi
if [[ -n "$KV_CACHE_DTYPE" ]]; then
  if [[ "$KV_CACHE_DTYPE" == fp8* ]] && ! grep -qs "IS_FP8" "$VLLM_VENV"/lib/python*/site-packages/vllm/models/qwen4_exp/nvidia/ops/qsa.py; then
    echo "ERROR: KV_CACHE_DTYPE=$KV_CACHE_DTYPE but this venv's QSA kernel has no fp8 path (PR #55557)."
    echo "  Use the uva venv: VLLM_VENV=\$HOME/lloyd/.venvs/vllm-flash-next-main (setup-vllm-flash-next-main.sh)."
    exit 1
  fi
  AB_ARGS+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
fi
if [[ "$PLE_IMPL" == "uva" ]]; then
  # main's spelling; VLLM_PLE_CPU_OFFLOAD=1 still works there but logs "legacy".
  AB_ARGS+=(--engram-config '{"cpu_offload": true}')
else
  # See deadlock 1 above: the offload worker is only spawned by the mp executor.
  AB_ARGS+=(--distributed-executor-backend mp)
fi
# shellcheck disable=SC2206  # intentional word split
[[ -n "$EXTRA_ARGS" ]] && AB_ARGS+=($EXTRA_ARGS)

echo "A/B config: venv=$VLLM_VENV ple=$PLE_IMPL kv_dtype=${KV_CACHE_DTYPE:-bf16}" \
     "max_num_seqs=$MAX_NUM_SEQS gpu_mem_util=$GPU_MEMORY_UTILIZATION" \
     "kv_bytes=${KV_CACHE_MEMORY_BYTES:-<fraction>} lm_only=$LANGUAGE_MODEL_ONLY" \
     "moe=${MOE_BACKEND:-<auto>} gdn=${GDN_PREFILL_BACKEND:-<auto>}" \
     "autotune=$FLASHINFER_AUTOTUNE mtp=$MTP_ENABLED/k=$MTP_TOKENS max_model_len=$MAX_MODEL_LEN" \
     "batched=${MAX_NUM_BATCHED_TOKENS:-<default>}"

export PATH="$VLLM_VENV/bin:/opt/cuda/bin:/usr/bin:/usr/sbin:$PATH"
export LD_LIBRARY_PATH="/usr/lib:/opt/cuda/targets/x86_64-linux/lib:/opt/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_HOME="/opt/cuda"
export NVCC_CCBIN=/usr/bin/g++-15
# 3 GPUs on PCI bus order: 0 = RTX 3090, 1 = RTX PRO 6000 Blackwell, 2 = RTX 3090.
# CUDA_DEVICE_ORDER is mandatory — without it the runtime reorders by capability
# and this lands on a 3090, which cannot run NVFP4 at all.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

# The flag this whole script exists for: keep the 95.37 GiB BF16 N-gram table in
# host RAM and prefetch rows asynchronously. On the uva build the same request
# travels as --engram-config (above); the env var there only earns a
# deprecation warning.
if [[ "$PLE_IMPL" == "worker" ]]; then
  export VLLM_PLE_CPU_OFFLOAD=1
  # Bounds the startup registration rendezvous (davidtai's ACK barrier reads
  # this knob). Cold boot reads 170 GiB from disk, so give it room.
  export VLLM_PLE_OFFLOAD_READY_TIMEOUT=1800
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MODULE_LOADING=LAZY
export VLLM_ENABLE_CUDAGRAPH_GC=1

CMD=("$VLLM_VENV/bin/python" -m vllm.entrypoints.openai.api_server
  --model "$MODEL_DIR"
  --served-model-name Qwen3.8-Flash-Next-nvfp4 primary
  --port "${PORT:-8096}"
  --host 127.0.0.1
  --trust-remote-code
  --tensor-parallel-size 1
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --enable-prefix-caching
  --enable-prompt-tokens-details
  --no-enable-log-requests
  --scheduling-policy priority
  --async-scheduling
  --enable-auto-tool-choice
  --tool-call-parser qwen3_xml
  --reasoning-parser qwen3
  "${SPEC_ARGS[@]}"
  "${AB_ARGS[@]}")

# DRY_RUN=1 prints the assembled command and the env it would run under, so
# the venv detection and knob plumbing above can be checked without a
# 4-minute boot. PORT is for side-by-side trials on a bare port (8097 was
# the 2026-09-10 FP8 trial); production stays on 8096.
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN: PLE_IMPL=$PLE_IMPL VLLM_PLE_CPU_OFFLOAD=${VLLM_PLE_CPU_OFFLOAD:-<unset>} VLLM_ALLOW_LONG_MAX_MODEL_LEN=${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-<unset>}"
  printf '%q ' "${CMD[@]}"; echo
  exit 0
fi
exec "${CMD[@]}"
