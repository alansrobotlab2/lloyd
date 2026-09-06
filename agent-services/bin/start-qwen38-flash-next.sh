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
# THREE KNOWN DEADLOCKS, AND WHY EACH FLAG BELOW EXISTS
#   1. uniproc gap (vLLM issue #53960). vLLM picks the uniproc executor at TP=1,
#      but spawn_ple_offload()/wait_ple_offload_ready() were only called from
#      multiproc_executor. The offload worker was never spawned and the GPU side
#      waited forever on a peer that did not exist. Fixed by 95dc96d1d012, which
#      IS in the branch — but we still pass --distributed-executor-backend mp
#      because that is the configuration everyone who got this serving actually
#      ran. Drop it only if you want to re-test the uniproc path.
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
#   To go past 262144 you also need YaRN, which is a config override, not a flag:
#     --hf-overrides '{"rope_parameters":{"rope_type":"yarn","factor":4.0,
#                      "original_max_position_embeddings":262144}}'
#     plus VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"

if [[ ! -x "$VLLM_VENV/bin/python" ]]; then
  echo "vLLM venv not found at $VLLM_VENV"
  echo "Build it: bash $PROJECT_DIR/setup/setup-vllm-qwen38-flash-next.sh"
  exit 1
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

# Hard preflight: without ptrace_scope=0 the CUDA IPC handshake below fails and
# you lose ~4 minutes loading 170 GiB before finding out. Fail in 10ms instead.
PTRACE_SCOPE="$(cat /proc/sys/kernel/yama/ptrace_scope 2>/dev/null || echo 0)"
if [[ "$PTRACE_SCOPE" != "0" ]]; then
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
SPEC_ARGS=()
if [[ -f "$MODEL_DIR/nvfp4_experts_mtp.safetensors" ]]; then
  SPEC_ARGS=(--speculative-config '{"method": "mtp", "num_speculative_tokens": 3}')
else
  echo "WARNING: nvfp4_experts_mtp.safetensors missing — starting WITHOUT speculative decode"
fi

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
# host RAM and prefetch rows asynchronously.
export VLLM_PLE_CPU_OFFLOAD=1
# Bounds the startup registration rendezvous (davidtai's ACK barrier reads this
# knob). Cold boot reads 170 GiB from disk, so give it room.
export VLLM_PLE_OFFLOAD_READY_TIMEOUT=1800

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_MODULE_LOADING=LAZY
export VLLM_ENABLE_CUDAGRAPH_GC=1

exec "$VLLM_VENV/bin/python" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" \
  --served-model-name Qwen3.8-Flash-Next-nvfp4 primary \
  --port 8096 \
  --host 127.0.0.1 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --distributed-executor-backend mp \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.9345 \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --no-enable-flashinfer-autotune \
  --no-enable-log-requests \
  --scheduling-policy priority \
  --async-scheduling \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  "${SPEC_ARGS[@]}" \
  --limit-mm-per-prompt '{"image": 0, "video": 0, "audio": 0}'
