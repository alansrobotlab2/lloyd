---
type: medium-research
tags: [vlm, edge-ai, on-device-inference, qwen, llava, mobile-ai, quantization, edge-deployment]
date: 2026-07-10
last_updated: 2026-07-10
domain: ai
sources:
  - url: "https://deepwiki.com/QwenLM/Qwen2.5-VL/7.1-performance-optimization"
    title: "Qwen2.5-VL Performance Optimization (DeepWiki)"
    accessed: "2026-06-30"
  - url: "https://huggingface.co/docs/transformers/model_doc/llava_onevision"
    title: "LLaVA-OneVision Model Documentation — HuggingFace"
    accessed: "2026-06-30"
  - url: "https://arxiv.org/html/2503.21782"
    title: "Mobile-VideoGPT: Fast and Accurate Model for Mobile Video Understanding — arXiv"
    accessed: "2026-06-30"
  - url: "https://ollama.com/library/qwen2.5vl"
    title: "Qwen2.5-VL — Ollama Library"
    accessed: "2026-06-30"
  - url: "https://huggingface.co/collections/Qwen/qwen25-vl"
    title: "Qwen2.5-VL — HuggingFace Collection"
    accessed: "2026-06-30"
  - url: "https://www.runlocalai.co/models/llava-onevision-7b"
    title: "LLaVA-OneVision 7B — Local Inference Guide | RunLocalAI"
    accessed: "2026-06-30"
  - url: "https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    title: "LLaVA-OneVision 0.5B — HuggingFace Model Card"
    accessed: "2026-06-30"
  - url: "https://amshaker.github.io/Mobile-VideoGPT/"
    title: "Mobile-VideoGPT Project Page"
    accessed: "2026-06-30"
  - url: "https://www.liquid.ai/blog/lfm2-5-230m"
    title: "LFM2.5-230M: Built to Run Anywhere — Liquid AI Blog"
    accessed: "2026-06-30"
  - url: "https://stable-learn.com/en/ivy-vl-launch/"
    title: "Ivy-VL Launch: 3B Parameters Dominates Edge Visual AI — Stable Learn"
    accessed: "2026-06-30"
  - url: "https://machinelearning.apple.com/research/fast-vision-language-models"
    title: "FastVLM: Efficient Vision Encoding for Vision Language Models — Apple ML Research"
    accessed: "2026-06-30"
  - url: "https://github.com/apple/ml-fastvlm"
    title: "FastVLM Official Repository — GitHub"
    accessed: "2026-06-30"
  - url: "https://arxiv.org/abs/2504.06298"
    title: "Ternarization of Vision Language Models for Edge Devices — arXiv"
    accessed: "2026-06-30"
  - url: "https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-2"
    title: "LLaVA-OneVision-2 — Fully Open LMMs"
    accessed: "2026-06-30"
  - url: "https://www.ultralytics.com/blog/fastvlm-apple-introduces-its-new-vision-language-model"
    title: "Apple Releases FastVLM — Ultralytics Blog"
    accessed: "2026-06-30"
---

# Real-time VLM Inference on Edge Devices: Qwen2.5-VL & LLaVA-OneVision

## Summary

Deploying vision-language models on mobile and edge hardware has reached a practical inflection point with Qwen2.5-VL and LLaVA-OneVision as the two dominant open-weight families. Qwen2.5-VL offers 3B, 7B, and 72B variants — the 3B model at INT4 quantization (~1.44 GB) fits comfortably on flagship mobile SoCs, while the 7B variant targets high-end tablets and desktop NPUs. LLaVA-OneVision provides 0.5B, 7B, and 72B sizes; its 0.5B variant (~893M parameters) is purpose-built for on-device inference with native-resolution training that avoids fixed-resolution padding. Both families support quantization (AWQ, GPTQ, INT4/INT8) and cross-platform inference via llama.cpp/GGUF, vLLM, and TFLite. Current benchmarks show Mobile-VideoGPT-0.5B achieving ~45.9 tok/s on mobile hardware versus ~22.7 tok/s for LLaVA-OneVision-0.5B — roughly 2× speedup while maintaining comparable accuracy.

## Key Facts

- **Qwen2.5-VL 3B at INT4 = 1.44 GB**: The primary target for mobile SoCs (Snapdragon 8 Elite, Apple A17/M-series, Tensor G4). Fits within typical 16 GB device RAM with headroom for vision encoder and KV cache.
- **Qwen2.5-VL 3B outperforms Qwen2-VL 7B**: The smaller model beats the previous generation's larger model on several vision tasks, making it a practical edge candidate.
- **LLaVA-OneVision 0.5B (~893M params)**: Supports single-image, multi-image, and video inputs in a unified interface. Native-resolution training avoids the fixed-resolution padding overhead that plagues many VLMs.
- **LLaVA-OneVision-0.5B speed on mobile**: ~22.7 tok/s on edge devices — usable for near-real-time image understanding but too slow for continuous video analysis without architectural optimization.
- **Mobile-VideoGPT-0.5B baseline**: 45.9 tok/s on mobile, 6-point average improvement over LLaVA-OneVision-0.5B across six video benchmarks. Represents the current frontier for sub-billion mobile VLMs.
- **Quantization is mandatory**: AWQ provides the best accuracy/size trade-off for edge. GPTQ slightly favors accuracy at larger model size. Both are published for Qwen2.5-VL.
- **LLaVA-OneVision-1.5 series**: The 1.5 generation extends the family with 4B-Base and 8B-Instruct variants, trained on native-resolution images using FP8 mixed-precision, MoE, and long-sequence parallelism via NVIDIA's Megatron-LM framework. The 8B-Instruct variant is positioned for consumer-grade edge deployment.
- **FastVLM (Apple, CVPR 2025)**: Introduces FastViTHD encoder delivering up to 85× faster time-to-first-token than LLaVA-OneVision-0.5B on iPhone 16 Pro, and 7.9× faster than Cambrian-1-8B while matching accuracy. Open-sourced with MLX demo for macOS/iOS. Represents a paradigm shift: optimizing the vision encoder independently to slash TTF, since vision encoding dominates latency in on-device VLM inference.
- **Ternarization for edge VLMs (Crulis et al., 2025)**: A new extreme quantization approach compressing pre-trained VLMs to ternary weights {-1, 0, 1} via k-means initialization + 2-epoch fine-tuning. Provides faster token generation than INT4 while retaining more information than binarization. Custom TFLite operators enable edge execution. Openly released conversion code.
- **GGUF / llama.cpp VLM support**: Qwen2.5-VL-7B GGUF quantizations (Q4_K_M through Q2_K) enable cross-platform CPU/NPU inference. Community GGUF builds by Mungert support llama.cpp's visual model pipeline, though vision requires a forked llama.cpp with mmproj support.
- **Hardware targets**: Qualcomm Snapdragon 8 Elite (Hexagon Tensor Processor), Apple A17 Pro / M-series (Neural Engine + MLX), Google Tensor G4 (Pixel), NVIDIA Jetson Orin (edge robotics).
- **Market context**: Edge AI hardware projected to grow from $26B (2025) to $59B (2030). ~80% of AI inference expected to run locally by 2026.

### Model Size vs. Precision Budget

| Precision | Qwen2.5-VL 3B | Qwen2.5-VL 7B | Qwen2.5-VL 72B |
|-----------|--------------|--------------|----------------|
| FP32 | 11.5 GB | 26.34 GB | 266.21 GB |
| BF16 | 5.75 GB | 13.17 GB | 133.11 GB |
| INT8 | 2.87 GB | 6.59 GB | 66.5 GB |
| INT4 | 1.44 GB | 3.29 GB | 33.28 GB |

*Source: DeepWiki Qwen2.5-VL performance optimization doc*

### Architectural Design Choices

**Qwen2.5-VL:**
- Dynamic resolution processing with configurable `min_pixels`/`max_pixels` parameters
- Flash Attention 2 for multi-image and video workloads
- YaRN (Yet another RoPE extension) enables context lengths beyond 32K tokens
- MRoPE is more economical with position IDs, allowing direct `max_position_embeddings` increases for long video

**LLaVA-OneVision:**
- Native-resolution training avoids fixed-resolution padding bottleneck
- Unified interface for single-image, multi-image, and video inputs
- Built on Qwen2 language backbone (7B and 72B variants)
- SigLIP vision encoder across all sizes

### Inference Optimization Stack

The practical pipeline for edge VLM deployment:
1. **Quantization** — AWQ or GPTQ to INT4/INT8 (mandatory for sub-4B mobile models). Ternarization emerging as extreme quantization alternative.
2. **Flash Attention 2** — reduces memory for multi-image/video processing
3. **Dynamic resolution** — control input pixel budget to manage token count and latency
4. **Fast vision encoding** — FastVLM/FastViTHD decouples vision encoder optimization from language model, slashing TTF independently
5. **vLLM** — production serving with local inference support
6. **llama.cpp / GGUF** — cross-platform CPU/NPU inference for Android/iOS deployment. Qwen2.5-VL requires forked llama.cpp with mmproj support for vision.
7. **TFLite** — most widely deployed inference engine for Android vision models; ternarization operators custom-built for TFLite

### FastVLM: The Vision Encoder Bottleneck

**FastVLM** (Apple, CVPR 2025) identifies and attacks the vision encoding bottleneck that dominates on-device VLM latency. Key insights:
- **Vision encoding is the TTF bottleneck**: In on-device VLMs, the vision encoder (typically a ViT) processes the image before any language token generation can begin. This serial dependency means TTF is dominated by vision encoding time.
- **FastViTHD encoder**: A redesigned vision encoder that achieves 85× faster TTF than LLaVA-OneVision-0.5B on iPhone 16 Pro. The approach optimizes the vision encoder independently rather than co-designing with the language model.
- **Architectural implication**: This suggests a design principle for edge VLMs — decouple vision encoder optimization from language model choice, enabling faster iteration on the TTF-critical component.
- **Open-source**: Full implementation at apple/ml-fastvlm with MLX demo for macOS/iOS.

### LLaVA-OneVision-1.5 Series

The 1.5 generation introduces architectural advances for edge-capable VLMs:
- **4B-Base and 8B-Instruct variants**: Smaller footprint variants targetting edge deployment
- **FP8 mixed-precision training**: Reduces memory during training, enables smaller deployment sizes
- **Mixture-of-Experts (MoE)**: Selective activation reduces per-token compute
- **Long-sequence parallelism**: Built on Megatron-LM framework for efficient training
- **Task transfer**: Strong capability transfer from image to video understanding without video-specific fine-tuning

## Related (vault entities)

- Edge VLM Inference
- VLM Edge Deployment
- Qwen2.5-VL-7B
- LLaVA-OneVision-1.5
- LLaVA-OneVision 7B
- Mobile-VideoGPT
- SmolVLM
- Ivy-VL

## Open Questions

- **Direct cross-model benchmarks**: No single source benchmarks Qwen2.5-VL 3B AWQ vs LLaVA-OneVision 0.5B on the same mobile hardware. A head-to-head comparison on standardized vision benchmarks (MMLU, OK-VQA, ScienceQA) at INT4 is needed.
- **Thermal constraints**: Sustained VLM inference on mobile NPUs generates heat — how does thermal throttling affect real-world throughput over extended sessions? The arXiv paper "Phase Matters: Characterizing Heterogeneous Vision-Language..." on Snapdragon 8 Elite addresses this but hasn't been fully integrated.
- **7B on mobile ceiling**: Can 7B VLMs run acceptably on flagship mobile devices with aggressive INT4 quantization + NPU offloading, or is 3B the practical ceiling? The 7B INT4 at 3.29 GB is feasible on 16 GB+ devices but KV cache overhead may push it beyond.
- **Dynamic resolution impact**: How much accuracy is lost when aggressively constraining token budgets via dynamic resolution for real-time mobile inference?
- **Android-specific stack**: Beyond llama.cpp, is there an Android-specific inference stack optimized for VLM workloads on Snapdragon Hexagon NPU?
- **Multi-modal context**: How do video workloads (frame sequences) compare to single-image in terms of sustained throughput and memory pressure on mobile?

## Sources

1. Qwen2.5-VL Performance Optimization — DeepWiki mirror of official docs (model sizes, quantization options, memory budgets)
2. LLaVA-OneVision Model Documentation — HuggingFace Transformers docs (architecture, sizes, capabilities)
3. LLaVA-OneVision Paper — arXiv:2408.03326 "LLaVA-OneVision: Easy Visual Task Transfer" (unified multimodal design, task transfer capabilities)
4. Mobile-VideoGPT — arXiv:2503.21782 "Fast and Accurate Model for Mobile Video Understanding" (speed benchmarks, video understanding results)
5. Qwen2.5-VL — HuggingFace Collection (model family overview, 3B as edge AI solution)
6. Qwen2.5-VL — Ollama Library (availability, edge AI positioning)
7. LFM2.5-230M — Liquid AI Blog (decode-speed reference for sub-billion models on mobile)
8. Ivy-VL — Stable Learn / HuggingFace (compositional edge VLM approach)
9. LLava-OneVision 0.5B — HuggingFace model card (parameter count: 893.7M, vision-language capabilities)

## Confidence

0.80: Qwen2.5-VL memory budgets and quantization options are well-documented via official docs. LLaVA-OneVision architecture and sizes are confirmed via HuggingFace and arXiv. Mobile-VideoGPT speed claims are backed by peer-reviewed benchmarks. The 3B INT4 sizing is the most certain data point. The main uncertainty is the absence of direct head-to-head benchmarks between Qwen2.5-VL 3B and LLaVA-OneVision 0.5B on identical mobile hardware — the speed comparison comes from different papers/devices. Thermal and sustained-throughput data remain sparse.