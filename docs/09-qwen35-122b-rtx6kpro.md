# Running Qwen3.5-122B-A10B (NVFP4 + MTP) on RTX PRO 6000 Blackwell

> **Hardware:** Single NVIDIA RTX PRO 6000 Blackwell (SM120, 96 GB GDDR7)  
> **Goal:** Maximum generation speed via NVFP4 + NEXTN MTP speculative decoding

---

## Quick Summary

The 122B-A10B NVFP4 weights are ~61 GB, so **the model fits on a single 96 GB card** (with reduced
context), or TP=2 across two cards for full 262K context. The fastest path is **SGLang + NVFP4 +
NEXTN MTP**. Do not use llama.cpp or vLLM if you want to squeeze the most out of SM120.

---

## Recommended Checkpoint

**`Sehyo/Qwen3.5-122B-A10B-NVFP4`** — includes multimodal support and MTP weights (added 2026-03-02).

> ⚠️ `txn545/Qwen3.5-122B-A10B-NVFP4` does **not** include `mtp.*` weights — they are stripped
> during quantization. Avoid it unless you manually merge BF16 MTP weights from the base model.

### KV Cache Tradeoff (Sehyo vs. NVIDIA ModelOpt)

| Property | NVIDIA ModelOpt | Sehyo (llm-compressor) |
|---|---|---|
| KV cache scheme | Calibrated FP8 (`k_scale`/`v_scale` tensors) | `null` — defaults to BF16 at runtime |
| VRAM for KV cache | 1× | 2× |
| MTP weights | ✅ (nvidia/397B checkpoint) | ✅ (Sehyo/122B as of 2026-03-02) |
| Multimodal | ✅ | ✅ |

Because Sehyo defaults to BF16 KV cache on a single 96 GB card, **plan for 65K–128K context
rather than the full 262K**. Use TP=2 (two GPUs) to unlock the full 262K window.

---

## Dependencies

```bash
# SGLang — use the latest main or the Blackwell-patched branch
pip install sglang[all]
pip install "transformers>=5.2.0" accelerate

# cuDNN — required for the safe FP4 GEMM backend (avoids race condition)
pip install nvidia-cudnn-cu13==9.19.1.2
```

---

## Launch Command

### Single GPU (TP=1, ~65K context)

```bash
export SGLANG_ENABLE_SPEC_V2=True        # MANDATORY — see gotchas below
export SGLANG_DISABLE_DEEP_GEMM=1        # Prevents NaN outputs on SM120
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=PHB
export OMP_NUM_THREADS=8
export SAFETENSORS_FAST_GPU=1

python -m sglang.launch_server \
  --model-path Sehyo/Qwen3.5-122B-A10B-NVFP4 \
  --tp-size 1 \
  --host 0.0.0.0 --port 8000 \
  --trust-remote-code \
  --mem-fraction-static 0.85 \
  --quantization modelopt_fp4 \
  --attention-backend triton \
  --moe-runner-backend flashinfer_cutlass \
  --fp4-gemm-backend flashinfer_cudnn \
  --context-length 65536 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --speculative-algo NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --cuda-graph-max-bs 4 \
  --max-running-requests 4 \
  --chunked-prefill-size 4096 \
  --sleep-on-idle
```

### Two GPUs (TP=2, full 262K context)

Change one flag and bump context:

```bash
  --tp-size 2 \
  --context-length 262144 \
```

Everything else stays the same.

---

## Critical Gotchas for SM120 (Blackwell)

### 1. `SGLANG_ENABLE_SPEC_V2=True` is non-negotiable

Without it, SGLang silently converts NEXTN to EAGLE and tries to load the full model a **second
time** as a draft model — instant OOM on a 96 GB card (61 GB × 2 = 122 GB).

### 2. `SGLANG_DISABLE_DEEP_GEMM=1` prevents NaN outputs

On SM120, the DeepGemm scale format detection incorrectly assumes `ue8m0` scales. NVFP4 uses
`float8_e4m3fn` scales, causing NaN output and garbage generation. Always set this.

### 3. Use `--fp4-gemm-backend flashinfer_cudnn`, not the default

The default FP4 GEMM backend (`flashinfer_cutlass`) has a **race condition bug** that silently
corrupts memory. The cuDNN backend is both safer and marginally faster. Requires
`nvidia-cudnn-cu13==9.19.1.2`.

### 4. Keep `--speculative-num-steps` at 3 or fewer

`num_speculative_tokens > 3` causes illegal memory access errors, especially at long context.
The sweet spot is `--speculative-num-steps 3 --speculative-num-draft-tokens 4`.

### 5. `--moe-runner-backend flashinfer_cutlass` is correct for SM120

On SM120, `flashinfer_cutlass` wraps CUTLASS and is the recommended backend for MTP speculative
decoding. The `deep_gemm` option silently falls back to CUTLASS anyway (DeepGemm is not supported
on SM120), so `flashinfer_cutlass` is explicit and reliable.

### 6. FP8 KV cache behavior

The Sehyo checkpoint has no calibrated FP8 KV scales, so it defaults to BF16 KV cache. If you
force `--kv-cache-dtype fp8_e4m3` with this checkpoint, you may get uncalibrated scales (scale=1.0)
which can degrade quality. Leave KV cache dtype unset and let it default to BF16.

---

## Backend Reference

| SGLang Flag | Best Value (SM120) | Notes |
|---|---|---|
| `--attention-backend` | `triton` | Required for Blackwell compatibility |
| `--moe-runner-backend` | `flashinfer_cutlass` | Fastest for MTP on SM120 |
| `--fp4-gemm-backend` | `flashinfer_cudnn` | Avoids race condition in cutlass path |
| `--speculative-algo` | `NEXTN` | Uses built-in MTP head; no separate draft model |
| `--speculative-num-steps` | `3` | Max stable value; >3 causes illegal memory access |
| `--speculative-num-draft-tokens` | `4` | Pairs with num-steps=3 |

---

## Alternative: Q4 GGUF via llama.cpp

Simpler setup, but **no NEXTN MTP** and **no native SM120 NVFP4 tensor cores** — noticeably slower.

```bash
# Download
export LLAMA_CACHE="unsloth/Qwen3.5-122B-A10B-GGUF"

# Serve
./llama.cpp/llama-server \
  -hf unsloth/Qwen3.5-122B-A10B-GGUF:UD-Q4_K_XL \
  --ctx-size 65536 \
  -ngl 100 \
  --port 8080 \
  --host 0.0.0.0
```

Use this only if you want the simplest possible setup and don't need peak throughput.

---

## Expected Performance

| Config | GPUs | Approx tok/s |
|---|---|---|
| Qwen3.5-397B NVFP4 + MTP=3 | 4× RTX PRO 6000 | ~130 |
| Qwen3.5-397B NVFP4 + MTP=3 | 8× RTX PRO 6000 | ~350 |
| **Qwen3.5-122B NVFP4 + MTP=3** | **1–2× RTX PRO 6000** | **expect faster than 397B/4GPU** |

The 122B MoE activates only 10B parameters per forward pass, so single-GPU decode is fast. The
primary bottleneck is GDDR7 bandwidth for weight streaming, not compute. Community benchmarks for
the 122B on a single PRO 6000 are still being established — report back if you run it!

---

## Useful Resources

- [voipmonitor/rtx6kpro](https://github.com/voipmonitor/rtx6kpro) — community wiki synthesized from ~5,000 Discord messages on running large LLMs on RTX PRO 6000 Blackwell
- [Sehyo/Qwen3.5-122B-A10B-NVFP4](https://huggingface.co/Sehyo/Qwen3.5-122B-A10B-NVFP4) — recommended checkpoint
- [SGLang GitHub](https://github.com/sgl-project/sglang) — use latest main or Blackwell-patched branch
- [Qwen3.5 HuggingFace](https://huggingface.co/Qwen/Qwen3.5-122B-A10B) — official model card
