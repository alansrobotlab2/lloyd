---
type: medium-research
tags: [vlm, edge-ai, on-device-inference, qwen, llava, mobile-ai, quantization]
date: 2026-07-10
last_verified: 2026-07-10
domain: ai
sources:
  - url: "https://deepwiki.com/QwenLM/Qwen2.5-VL/7.1-performance-optimization"
    title: "Qwen2.5-VL Performance Optimization (DeepWiki)"
    accessed: "2026-07-10"
  - url: "https://huggingface.co/docs/transformers/model_doc/llava_onevision"
    title: "LLaVA-OneVision Model Documentation — HuggingFace"
    accessed: "2026-07-10"
  - url: "https://www.liquid.ai/blog/lfm2-5-230m"
    title: "LFM2.5-230M: Built to Run Anywhere — Liquid AI Blog"
    accessed: "2026-07-10"
  - url: "https://www.ertas.ai/blog/edge-ai-local-inference-2026"
    title: "Edge AI in 2026: Why 80% of Inference Is Moving Local — Ertas AI"
    accessed: "2026-07-10"
  - url: "https://stable-learn.com/en/ivy-vl-launch/"
    title: "Ivy-VL Launch: 3B Parameters Dominates Edge Visual AI — Stable Learn"
    accessed: "2026-07-10"
  - url: "https://arxiv.org/html/2503.21782v2"
    title: "Mobile-VideoGPT: Fast and Accurate Model for Mobile Video Understanding — arXiv"
    accessed: "2026-07-10"
  - url: "https://developersvoice.com/blog/mobile/mobile_ai_architecture_guide_2025/"
    title: "On-Device AI for Mobile: Tiny LLMs, Vision Models — DevelopersVoice"
    accessed: "2026-07-10"
---

# Real-time VLM Inference on Edge Devices: Qwen2.5-VL and LLaVA-OneVision

Deploying vision-language models on mobile and edge hardware is a major near-term frontier. Qwen2.5-VL and LLaVA-OneVision represent the two most-cited families for on-device VLM workloads, spanning from 0.5B parameter variants designed specifically for edge to 72B models for server-side baselines. Quantization (AWQ, GPTQ, INT4/INT8) is the critical enabler — the 3B variant of Qwen2.5-VL fits at ~1.44 GB in INT4, making it viable for mid-range mobile NPUs.

## Key Findings

- **Qwen2.5-VL** comes in 3B, 7B, and 72B sizes, with AWQ and GPTQ INT4 quantized versions targeting edge deployment. The 3B model requires only ~1.44 GB theoretical VRAM at INT4, well within mobile SoC memory budgets [DeepWiki].
- **LLaVA-OneVision** is available in 0.5B, 7B, and 72B sizes. The 0.5B variant (~893M parameters) is explicitly designed for mobile/edge, supporting single-image, multi-image, and video inputs via a native-resolution design that avoids fixed-resolution padding [HuggingFace].
- **Ivy-VL** (3B, built on LLaVA-OneVision + Qwen2.5-3B-Instruct + SigLIP vision encoder) demonstrates a practical edge-deployable VLM stacking approach, specifically optimized for performance/efficiency trade-offs on constrained hardware [Stable Learn].
- **Mobile-VideoGPT-0.5B** achieves 45.9 tok/s on mobile, compared to ~22.7 tok/s for LLaVA-OneVision-0.5B — roughly 2× faster while maintaining comparable accuracy on video understanding tasks [Mobile-VideoGPT arXiv].
- **LFM2.5-230M** (Liquid AI) sets a practical benchmark: 213 tok/s decode on Samsung Galaxy S25 Ultra, 42 tok/s on Raspberry Pi 5. While not a VLM per se, it establishes decode-speed expectations for sub-billion models on current mobile NPU/CPU combos [Liquid AI Blog].
- **Quantization is mandatory** for edge VLM deployment: AWQ provides the best accuracy/size trade-off, while GPTQ slightly favors accuracy at larger model size. Both AWQ and GPTQ INT4 variants of Qwen2.5-VL are published and tested on HuggingFace [DeepWiki].

## Details

### Model Size vs. Hardware Budget

| Precision | Qwen2.5-VL 3B | Qwen2.5-VL 7B | Qwen2.5-VL 72B |
|-----------|--------------|--------------|----------------|
| FP32 | 11.5 GB | 26.34 GB | 266.21 GB |
| BF16 | 5.75 GB | 13.17 GB | 133.11 GB |
| INT8 | 2.87 GB | 6.59 GB | 66.5 GB |
| INT4 | 1.44 GB | 3.29 GB | 33.28 GB |

*Source: DeepWiki Qwen2.5-VL performance optimization doc*

The 3B INT4 model at 1.44 GB is the primary target for mobile SoCs (Snapdragon 8 Elite, Apple A17/M-series, Tensor G4). The 7B INT4 at 3.29 GB pushes into high-end tablet/desktop NPU territory. The 72B family remains cloud-only for practical purposes.

### Key Architectural Design Choices

**Qwen2.5-VL:**
- Uses dynamic resolution processing with configurable `min_pixels`/`max_pixels` parameters
- Supports Flash Attention 2 for multi-image and video workloads
- YaRN (Yet another RoPE extension) enables context lengths beyond 32K tokens
- MRoPE is more economical with position IDs, allowing direct `max_position_embeddings` increases for long video

**LLaVA-OneVision:**
- Native-resolution training avoids the fixed-resolution padding bottleneck
- Supports single-image, multi-image, and video inputs in a unified interface
- Built on Qwen2 language backbone for the 7B and 72B variants
- 0.5B variant has ~893M parameters, targeting on-device inference

### Inference Optimization Stack

The practical optimization pipeline for edge VLM inference:
1. **Quantization** — AWQ or GPTQ to INT4/INT8 (mandatory for sub-4B mobile models)
2. **Flash Attention 2** — reduces memory for multi-image/video processing
3. **Dynamic resolution** — control input pixel budget to manage token count
4. **vLLM** — for production serving; also supports local inference
5. **llama.cpp / GGUF** — cross-platform CPU/NPU inference for Android/iOS deployment

TFLite remains the most widely deployed inference engine for Android-based vision models [DevelopersVoice].

### Ecosystem Context

Edge AI hardware is growing rapidly — the edge AI hardware market is projected to grow from $26B in 2025 to $59B by 2030 [Ertas AI]. By 2026, an estimated 80% of AI inference is expected to occur locally on devices rather than in cloud data centers. Key hardware targets:
- **Qualcomm Snapdragon 8 Elite** — Hexagon Tensor Processor, primary Android VLM accelerator
- **Apple A17 Pro / M-series** — Neural Engine with MLX framework support
- **Google Tensor G4** — on-device AI for Pixel devices
- **NVIDIA Jetson Orin** — edge robotics and autonomous platforms

## Related (vault entities)

- Edge VLM Inference
- VLM Edge Deployment
- Qwen2.5-VL
- LLaVA-OneVision 7B
- SmolVLM
- Mobile-VideoGPT

## Open Questions

- How do Qwen2.5-VL 3B AWQ and LLaVA-OneVision 0.5B compare on standardized vision benchmarks (MMLU, OK-VQA, ScienceQA) when quantized to INT4? No direct cross-model benchmark found.
- What are the thermal constraints for sustained VLM inference on mobile NPUs? (The arXiv paper "Phase Matters: Characterizing Heterogeneous Vision-Language..." on Snapdragon 8 Elite addresses this specifically — worth reviewing.)
- Can 7B VLMs run acceptably on flagship mobile devices with aggressive INT4 quantization + NPU offloading, or is 3B the practical ceiling?
- How do dynamic resolution strategies affect accuracy when aggressively constraining token budgets for real-time mobile inference?
- Is there an Android-specific inference stack (beyond llama.cpp) optimized for VLM workloads on Snapdragon Hexagon?

## Confidence

0.75: Memory requirements and quantization options for Qwen2.5-VL are well-documented via the DeepWiki mirror of the official docs. LLaVA-OneVision sizes and architecture are confirmed via HuggingFace model docs and GitHub. Ivy-VL composition is confirmed via HuggingFace and Stable Learn. Mobile-VideoGPT speed claims come from the arXiv paper. LFM2.5-230M benchmarks are from the Liquid AI blog. The main gap is a direct head-to-head comparison of Qwen2.5-VL vs LLaVA-OneVision on mobile hardware — no single source benchmarks them on the same device.