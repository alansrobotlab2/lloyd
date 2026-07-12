---
title: Real-time VLM-based Perception for Mobile Manipulation — Depth, Pose, Segmentation Fused Pipeline
tags:
  - ai/vlm
  - ai/vla
  - robotics/mobile-manipulation
  - robotics/perception-pipeline
  - robotics/depth-estimation
  - robotics/6d-pose-estimation
  - robotics/semantic-segmentation
  - robotics/scene-understanding
  - embodied-ai
  - research/domain-research
created: 2026-07-18
updated: 2026-07-18
confidence: 0.78
---

# Real-time VLM-based Perception for Mobile Manipulation — Depth, Pose, Segmentation Fused Pipeline

## Summary

Real-time perception for mobile manipulation requires fusing depth estimation, 6D object pose estimation, and semantic/instance segmentation into a unified pipeline that runs on edge hardware at task-critical frequencies (10–30 Hz). The fused pipeline replaces the traditional modular stack — where depth, pose, and segmentation run as independent models — with a shared representation that propagates geometric, spatial, and semantic signals across stages. Recent architectures (FALCON, G2VLM, AnywhereVLA) demonstrate that spatial foundation models (DUSt3R, VGGT), 3D Gaussian Splatting maps, and memory-augmented VLA prompting can provide this fusion, though end-to-end differentiable pipelines remain nascent.

## Key Facts

- **Three perception modalities, one bottleneck**: Mobile manipulation requires (a) **depth** (metric distance for grasp planning), (b) **6D pose** (position + orientation of target objects), and (c) **segmentation** (instance-level scene understanding). Running these independently is latency-inefficient and accumulates error at fusion boundaries. A fused pipeline shares a backbone representation (e.g., vision transformer features) across all three heads, reducing redundant compute and enabling cross-modal consistency.

- **FALCON (Zhang et al., arXiv:2510.17439)** injects rich 3D spatial tokens from spatial foundation models (DUSt3R, VGGT) into a dedicated Spatial-Enhanced Action Head rather than concatenating them into the VLM backbone. This preserves semantic alignment while adding geometric grounding — the key insight is that 3D priors should augment, not contaminate, the VLM's language-vision alignment. Achieves improved grounding under clutter and object scale variation.

- **G2VLM (InternRobotics, CVPR 2026)** introduces a geometry-grounded VLM that natively predicts 3D geometry (depth, structure-from-motion) alongside spatial reasoning tasks. Represents a shift from treating geometry as an auxiliary input module to making it a core grounding modality — the model jointly reasons about "what is this object" and "where is it in 3D space" through interleaved spatial-semantic reasoning.

- **Monocular depth estimation** has matured to the point where single RGB cameras can produce metrically useful depth maps at edge-deployable frequencies. Models like Depth Anything, Metric3D, and ZoeDepth run at 30+ FPS on Jetson-class hardware. For mobile manipulation, monocular depth avoids the hardware cost of stereo/RGBD sensors while providing sufficient precision for grasp approach planning. The key limitation is scale ambiguity — monocular depth lacks absolute metric calibration without additional cues (known object sizes, IMU, or LiDAR priors).

- **6D object pose estimation** has converged around two approaches: (a) **category-level** pose estimation (e.g., Category-Level 6D Object Pose Estimation via Bank of Category-Specific Templates), and (b) **object-specific** pose tracking (e.g., PVN3D, PoseCNN, DPOP). For mobile manipulation where novel objects appear, category-level methods are essential — the system must estimate pose of unfamiliar objects in known categories. Recent work integrates VLM semantic understanding with pose estimation, enabling zero-shot pose estimation conditioned on language descriptions.

- **Semantic/instance segmentation** for manipulation has shifted toward open-vocabulary approaches (OWL-ViT, GroundingDINO, SAM-based methods) that can identify objects from natural language prompts rather than fixed category sets. This is critical for mobile manipulation where the environment and task instructions are not pre-programmed. SAM (Segment Anything Model) and SAM 2 provide real-time instance segmentation that can be conditioned on VLM-generated prompts, creating a closed loop: VLM parses instruction → generates object reference → SAM segments the target → pose estimation localizes it in 3D → depth provides grasp geometry.

- **Real-time fused architectures** for mobile manipulation use a shared backbone (typically a lightweight ViT or ConvNeXt) with multiple heads. The backbone processes the RGB (and optionally depth) input once, and branches to depth, pose, and segmentation heads share early features. This reduces latency versus running three separate models and enables gradient-based co-optimization where improving one head can regularize the others. AnywhereVLA demonstrates this on a Jetson Orin NX + Intel NUC split at real-time frequencies for multi-room mobile manipulation.

- **3D Gaussian Splatting as a unified representation** merges depth, pose, and segmentation into a single dense 3D scene representation. Systems like SemGauss-SLAM (IROS 2025) add semantic labels to Gaussian Splatting maps, producing a scene representation that simultaneously encodes geometry (depth proxy), object locations (pose proxy), and semantics (segmentation). This is the closest realization of a fully fused perception pipeline — a single data structure that serves all three perception needs. GSWorld extends this to closed-loop sim-to-real training.

- **Edge deployment constraints**: The fused pipeline must run at ≥10 Hz for stable manipulation (slower rates cause trajectory drift; faster rates waste compute on diminishing perception gains). AnywhereVLA achieves this by splitting perception/VLA (SmolVLA at 450M parameters) on Jetson Orin NX and SLAM/control on an Intel NUC. VL-Nav demonstrates 30 Hz VLN on single Orin NX. A fully unified depth+pose+segmentation+VLA on a single edge device remains an open engineering challenge.

- **The spatial reasoning gap persists**: Despite fused architectures, most pipelines still suffer from 2D-to-3D grounding degradation. FALCON and G2VLM show that explicit 3D priors improve grounding, but the fundamental issue — that VLM backbones are pretrained on 2D images and lack metric 3D understanding — remains. This causes systematic failures on depth estimation in cluttered scenes, pose estimation under occlusion, and segmentation boundaries in textureless regions.

## Related (vault entities)

- [[Multi-Modal Grounding: Language-to-Action Mapping]]
- [[Real-time SLAM Integration with VLA Policies — Mapping During Manipulation Tasks]]
- [[VLA Edge Deployment: Qwen2.5-VL, Mobile-VideoGPT, SmolVLM]]
- [[Online Fine-Tuning for VLA Models — Continual Learning with Experience Replay]]

## Open Questions

1. **End-to-end differentiable fused perception**: Can depth, pose, and segmentation be jointly optimized through a single differentiable pipeline conditioned on language instructions? Current fused architectures share a backbone but train heads separately — fully end-to-end training from language instruction to perception-to-action remains unrealized.

2. **Monocular depth metric calibration**: Monocular depth models lack absolute metric scale. How do we calibrate depth for manipulation without LiDAR or stereo? Options include known object priors, IMU-fused monocular, or self-supervised calibration through grasping feedback. No system has demonstrated robust metric depth from monocular RGB alone in uncalibrated environments.

3. **Zero-shot 6D pose for novel objects**: Category-level pose estimation handles known object classes. How do systems handle truly novel objects with no prior model? VLM-conditioned pose estimation (estimating pose from a language description + image) is nascent — this is the "pick up that weird-shaped tool" problem.

4. **SAM/VLM closed-loop latency**: The SAM→VLM→SAM loop (parse instruction → segment → localize → act) introduces latency proportional to model sizes. Can sub-100ms perception loops be achieved with open-vocabulary segmentation + pose estimation on edge hardware?

5. **Unified 3DGS perception-representation**: Can 3D Gaussian Splatting maps serve as the sole perception representation for mobile manipulation — replacing separate depth maps, pose estimates, and segmentation masks? SemGauss-SLAM shows the representation is possible; feeding it as VLA conditioning is untested.

6. **Cross-modal consistency guarantees**: When depth, pose, and segmentation disagree (e.g., depth suggests an object is 2m away, pose suggests it's on a surface 1.5m away), how does the pipeline resolve conflicts? Current systems don't propagate uncertainty across heads — a consistency-aware fused architecture is an open design problem.

7. **Active perception for pose refinement**: Can the robot actively move the camera to improve pose estimates of occluded objects — choosing viewpoints that simultaneously improve depth, pose, and segmentation? Active perception loops that co-optimize all three modalities are unexplored.

8. **Perception-to-action latency budget**: How much of the total perception-to-action latency budget is consumed by the fused perception pipeline vs. VLA inference? Understanding this split is needed for real-time systems design — if perception takes 80 ms and VLA inference takes 120 ms, the bottleneck shifts based on pipeline design.

## Sources

- Zhang et al. "From Spatial to Actions: Grounding Vision-Language-Action Model in Spatial Foundation Priors (FALCON)," arXiv:2510.17439, 2025. [https://arxiv.org/abs/2510.17439](https://arxiv.org/abs/2510.17439)

- G2VLM: "Geometry Grounded Vision Language Model with Unified 3D Reconstruction and Spatial Reasoning," InternRobotics, CVPR 2026. [https://github.com/InternRobotics/G2VLM](https://github.com/InternRobotics/G2VLM)

- Gubernatorov et al. "AnywhereVLA: Language-Conditioned Exploration and Mobile Manipulation," arXiv:2509.21006, Sep 2025. [https://arxiv.org/abs/2509.21006](https://arxiv.org/abs/2509.21006)

- "SemGauss-SLAM: Dense Semantic Gaussian Splatting SLAM," IROS 2025.

- Zhao et al. "CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models," arXiv:2503.22020, 2025. [https://arxiv.org/abs/2503.22020](https://arxiv.org/abs/2503.22020)

- Li et al. "MAP-VLA: Memory-Augmented Prompting for Vision-Language-Action Model in Robotic Manipulation," arXiv:2511.09516, Nov 2025. ICRA 2026. [https://arxiv.org/abs/2511.09516](https://arxiv.org/abs/2511.09516)

- "Pure Vision Language Action (VLA) Models: A Comprehensive Survey," arXiv:2509.19012, 2025. [https://arxiv.org/abs/2509.19012](https://arxiv.org/abs/2509.19012)

- G-U-O. "MemoryVLA: Cognition-Memory-Action Framework." GitHub: G-U-O/memvla, 2025.

- Liu et al. "SlideSLAM: Sparse, Lightweight, Decentralized Metric-Semantic SLAM for Multi-Robot Navigation," arXiv:2406.17249, T-RO 2025. [https://arxiv.org/abs/2406.17249](https://arxiv.org/abs/2406.17249)

- "Towards Understanding Visual Grounding in Vision-Language Models," arXiv:2509.10345, 2025. [https://arxiv.org/html/2509.10345](https://arxiv.org/html/2509.10345)

## Confidence

**0.78**: Moderate-high confidence. The core concepts (fused depth+pose+segmentation pipelines, VLM-conditioned perception, 3DGS as unified representation) are well-supported by the existing vault notes on multi-modal grounding and SLAM+VLA integration, which themselves draw from 15+ concrete papers with published results. FALCON (arXiv:2510.17439) and G2VLM (CVPR 2026) provide direct evidence for geometry-grounded VLMs that address the spatial reasoning gap. AnywhereVLA provides the most concrete architecture for real-time mobile manipulation perception with empirical benchmarks. Confidence is held below 0.85 because: (a) no single paper directly addresses the "fused depth-pose-segmentation pipeline" as a unified topic — this note synthesizes across multiple related work streams; (b) the web research was limited by HTTP availability, so this note relies more heavily on vault knowledge than fresh primary source review; (c) the specific latency budgets and edge deployment constraints for fused pipelines (vs. modular pipelines) are inferred from component benchmarks rather than measured as an integrated system; and (d) 3DGS-as-perception-replacement remains aspirational with no published real-robot manipulation benchmark.