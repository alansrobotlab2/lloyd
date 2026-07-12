---
type: medium-research
tags: [monocular-depth-estimation, zero-shot, metric-depth, autonomous-navigation, metric3dv2, depth-anything, depthpro, anydepth, unidepthv2, depthanycamera]
date: 2025-07-20
last_verified: 2026-07-07
domain: robotics
sources:
  - url: "https://arxiv.org/abs/2404.15506"
    title: "Metric3Dv2: A Versatile Monocular Geometric Foundation Model"
    accessed: "2025-07-26"
  - url: "https://github.com/YvanYin/Metric3D"
    title: "YvanYin/Metric3D — GitHub"
    accessed: "2025-07-26"
  - url: "https://depth-anything.github.io/"
    title: "Depth Anything — Project Page"
    accessed: "2025-07-26"
  - url: "https://huggingface.co/apple/DepthPro-hf"
    title: "Apple DepthPro — Hugging Face"
    accessed: "2025-07-26"
  - url: "https://www.emergentmind.com/topics/monocular-metric-depth-estimator"
    title: "Monocular Metric Depth Estimator — Emergent Mind"
    accessed: "2025-07-26"
  - url: "https://www.alphaxiv.org/overview/2404.15506v4"
    title: "Metric3Dv2 — alphaXiv Overview"
    accessed: "2025-07-26"
  - url: "https://www.emergentmind.com/topics/zero-shot-monocular-depth-estimation-33c0cde0-f3ac-4a40-aec5-a5e16a359b79"
    title: "Zero-Shot Monocular Depth Estimation — Emergent Mind"
    accessed: "2025-07-26"
  - url: "https://openaccess.thecvf.com/content/CVPR2025/papers/Guo_Depth_Any_Camera_Zero-Shot_Metric_Depth_Estimation_from_Any_Camera_CVPR_2025_paper.pdf"
    title: "Depth Any Camera (CVPR 2025)"
    accessed: "2025-07-26"
  - url: "https://arxiv.org/abs/2601.02760"
    title: "AnyDepth: Depth Estimation Made Easy"
    accessed: "2026-07-07"
  - url: "https://arxiv.org/abs/2502.20110"
    title: "UniDepthV2: Universal Monocular Metric Depth Estimation Made Simpler"
    accessed: "2026-07-07"
  - url: "https://people.ee.ethz.ch/~csakarid/UniDepthV2/UniDepthV2_universal_monocular_metric_depth_estimation_made_simpler-Piccinelli+Sakaridis+Yang+Segu+Li+Abbeloos+Van_Gool-arXiv_2025.pdf"
    title: "UniDepthV2 — ETH Zurich"
    accessed: "2026-07-07"
---

# Zero-Shot Monocular Depth Estimation for Autonomous Navigation

Monocular zero-shot depth estimation enables metrically accurate 3D scene reconstruction from single-camera RGB images without dataset-specific fine-tuning. The field has converged on large-scale multi-camera training (16M+ images, thousands of camera models) with canonical camera-space transformations to resolve metric ambiguity. **Note: "ZeroShotDepth" is not a specific model — it's a general task/category name.** The key models (Metric3Dv2, Depth Anything V2, DepthPro, AnyDepth) all operate in this zero-shot paradigm. These models support real-time inference suitable for autonomous navigation, SLAM, and mobile robotics pipelines that require metric depth without LiDAR or stereo hardware.

## Key Findings

- **Metric3Dv2** (Hu et al., 2024, TPAMI) is a geometric foundation model jointly estimating zero-shot metric depth and surface normals from single images. Trained on 16M+ images across thousands of camera models. [1]
- **Canonical camera-space transformation (CSTM)** is the core innovation — explicitly resolves metric ambiguity from varying camera intrinsics by normalizing to a canonical focal length (fc = 1000px). Can be plugged into existing monocular models. [1]
- **Joint depth-normal optimization** uses ConvGRU-based recurrent refinement with explicit depth-normal consistency loss (L_d-n), enabling surface normal estimation to learn beyond scarce normal labels by distilling knowledge from abundant depth data. [1]
- **ZoeDepth** (Bhat et al., 2023, ICCV) is the foundational zero-shot metric depth model that fuses scale-invariant pre-training with domain-specific metric fine-tuning. Outperformed prior methods on cross-dataset generalization. [2]
- **Depth Anything V2** (Li et al., 2024) uses a DINOv2 ViT encoder + Dense Prediction Transformer decoder, trained on 595M synthetic samples via teacher-student framework. Achieves strong zero-shot relative depth; metric variants beat ZoeDepth on several benchmarks. Small variant (24.8M params) enables real-time edge deployment. [3]
- **Apple DepthPro** (2024) is a 504M parameter zero-shot metric depth foundation model producing high-resolution, sharp depth maps. Fits on 6GiB VRAM laptops, designed for Apple Silicon deployment. [4]
- **AnyDepth** (Ren et al., 2026, arXiv:2601.02760) — "Depth Estimation Made Easy" — proposes a lightweight data-centric framework using DINOv3 encoder + Simple Depth Transformer (SDT) decoder. Achieves 85–89% parameter reduction vs. DPT while maintaining accuracy via quality-based sample filtering and single-path feature fusion. [6]
- **UniDepthV2** (Piccinelli et al., 2025, arXiv:2502.20110) — Universal monocular metric depth estimation with edge-guided loss for sharper depth predictions and better-localized depth discontinuities. Handles perspective, fisheye, and panoramic images. [7]
- Real-time inference (30+ FPS at standard resolutions) is achievable on modern GPUs for navigation-grade pipelines. ViT-based models require GPU acceleration; lighter MiDaS-style models can run on embedded CPUs. [5]

## Details

### Metric3Dv2 Architecture & Training
Metric3Dv2 introduces two novel modules: (1) a **canonical camera-space transformation** that normalizes input geometry across camera models, removing the need for per-camera calibration at inference time, and (2) a **joint depth-normal optimization** module that lets a depth estimator bootstrap surface normal predictions from metric depth priors. Training uses 16M+ images from thousands of camera models with heterogeneous annotations. The paper is accepted to IEEE TPAMI (DOI: 10.1109/TPAMI.2024.3444912). The model generalizes zero-shot to in-the-wild images with unseen camera settings and enables single-image metrology. [1]

### Model Landscape Comparison
The current zero-shot monocular depth landscape consists of several competing approaches:

| Model | Metric Depth | Surface Normals | Zero-Shot | Architecture | Params |
|-------|-------------|-----------------|-----------|--------------|--------|
| Metric3Dv2 | Yes | Yes | Yes | ViT + joint depth-normal | Unknown |
| ZoeDepth | Yes | No | Yes | ViT + DPT | Unknown |
| Depth Anything V2 | Yes (variant) | No | Yes | DINOv2 + DPT | 24.8M (small) |
| DepthPro | Yes | No | Yes | ViT + DPT | 504M |
| UniDepthV2 | Yes | No | Yes | Universal multi-domain | Unknown |
| AnyDepth | Yes | No | Yes | DINOv3 + SDT | ~10% of DPT |

The field has shifted from affine-invariant (relative) depth toward metric depth with real-world scale recovery, which is essential for autonomous navigation where distance accuracy matters for collision avoidance and path planning. [2][5]

### AnyDepth — Lightweight Data-Centric Approach
AnyDepth (Ren et al., 2026) represents a shift toward efficiency: rather than scaling model size and data quantity, it focuses on data quality and architectural simplicity. Key innovations:
- **Simple Depth Transformer (SDT)**: Replaces DPT's multi-branch cross-scale alignment with single-path fusion, reducing parameters by 85–89% while matching or exceeding DPT accuracy
- **DINOv3 encoder**: Uses the latest self-supervised visual encoder for high-quality dense features
- **Quality-based filtering**: Removes harmful/noisy training samples, reducing dataset size while improving quality
- **Progressive DySample upsampling**: Avoids single-stage ×16 upsample errors; decomposes into two ×4 stages with local refinement
- Achieves comparable accuracy to DPT on 5 benchmarks with dramatically lower FLOPs and latency [6]

### UniDepthV2 — Universal Metric Depth
UniDepthV2 (Piccinelli et al., 2025) addresses the universal camera challenge head-on:
- Novel edge-guided loss produces sharper depth predictions with better-localized depth discontinuities
- Handles perspective, fisheye, and panoramic images without explicit per-camera calibration
- Designed for flexible deployment across diverse camera settings where explicit calibration is impractical [7]

### Autonomous Navigation Relevance
Monocular metric depth is increasingly used as a lightweight sensor fusion component in autonomous systems:
- Replaces or augments stereo/LiDAR in cost-constrained platforms (mobile robots, drones, low-cost autonomy)
- Enables real-time obstacle detection and distance estimation from single cameras
- Pairs with VLM perception pipelines (e.g., real-time VLM-perception-fused pipelines) for semantic-aware navigation
- Key limitation: metric accuracy degrades significantly for unseen camera models (fisheye, 360°) unless explicitly trained on them — Depth Any Camera (CVPR 2025) addresses this gap

### Real-Time Performance Considerations
- ViT-based models (DepthPro, Metric3Dv2) require GPU acceleration for real-time inference
- MiDaS-style lighter models can run on embedded CPUs for mobile deployment
- Latency depends heavily on input resolution — standard 256×256 or 512×512 achieves 30+ FPS on modern GPUs
- AnyDepth's SDT decoder achieves significantly lower latency than DPT-based approaches across all resolutions, particularly at high-res inputs
- Production pipelines typically use distilled/quantized variants for edge deployment

## Open Questions

- How do zero-shot metric depth models perform on non-perspective cameras (fisheye, catadioptric, 360°) in real autonomous driving scenarios? (Depth Any Camera, CVPR 2025, addresses this partially)
- What is the real-world metric error (RMSE) of these models vs. stereo/LiDAR on unseen urban environments at driving-relevant distances (50–200m)?
- Can zero-shot depth models be reliably used for collision avoidance in safety-critical systems, or is the error distribution too long-tailed?
- How do on-device (Apple Silicon, Jetson, mobile GPU) deployments compare to server GPU performance for real-time pipelines?
- Will diffusion-based approaches (SharpDepth, PatchRefiner) eventually replace foundation models for sharpened metric depth at acceptable latency?

## Sources

1. [Hu et al., Metric3Dv2 (arXiv:2404.15506)](https://arxiv.org/abs/2404.15506) — Core model description, architecture, training data, TPAMI acceptance
2. [Bhat et al., ZoeDepth (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/papers/Guizilini_Towards_Zero-Shot_Scale-Aware_Monocular_Depth_Estimation_ICCV_2023_paper.pdf) — Foundational zero-shot metric depth approach
3. [Depth Anything Project](https://depth-anything.github.io/) — Foundation model for robust monocular depth estimation
4. [Apple DepthPro (Hugging Face)](https://huggingface.co/apple/DepthPro-hf) — Apple's zero-shot metric depth foundation model
5. [Emergent Mind — Monocular Metric Depth Estimator](https://www.emergentmind.com/topics/monocular-metric-depth-estimator) — Landscape overview and performance trends
6. [Ren et al., AnyDepth (arXiv:2601.02760)](https://arxiv.org/abs/2601.02760) — Lightweight data-centric framework, SDT decoder, DINOv3 encoder
7. [Piccinelli et al., UniDepthV2 (arXiv:2502.20110)](https://arxiv.org/abs/2502.20110) — Universal metric depth with edge-guided loss
8. [Emergent Mind — Metric3Dv2](https://www.emergentmind.com/topics/metric3dv2) — Metric3Dv2 overview

## Confidence
0.85: Core facts about Metric3Dv2 drawn directly from the arXiv abstract and metadata (TPAMI acceptance, training scale, module descriptions). Model landscape comparison is synthesized from multiple source descriptions. Open questions reflect documented gaps (fisheye performance, long-range accuracy) identified in CVPR 2025 follow-up work. Confidence reduced slightly because "ZeroShotDepth" is not a single model name but rather a general category — the specific models (ZoeDepth, Depth Anything V2, DepthPro) are well-documented but direct benchmark comparisons require cross-referencing individual papers.