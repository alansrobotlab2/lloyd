---
type: quick-research
tags:
  - ai/vlm
  - ai/vla
  - robotics/mobile-manipulation
  - robotics/perception-pipeline
  - robotics/depth-estimation
  - robotics/6d-pose-estimation
  - robotics/semantic-segmentation
  - embodied-ai
  - research/domain-research
date: 2026-09-10
last_verified: 2026-09-10
sources:
  - url: "https://arxiv.org/abs/2503.05850"
    title: "FALCON: From Spatial to Action"
  - url: "https://arxiv.org/abs/2602.19710"
    title: "Pose-VLA: Universal Pose Pretraining"
  - url: "https://arxiv.org/abs/2605.21414"
    title: "PointACT: Multi-Scale Point-Action Interaction"
  - url: "https://arxiv.org/abs/2602.23024"
    title: "InCoM: Intent-Driven Perception"
  - url: "https://arxiv.org/abs/2606.31329"
    title: "3D HAMSTER: 3D-Aware Hierarchical Planning"
  - url: "https://arxiv.org/abs/2602.05233"
    title: "MobileManiBench"
---

# Real-Time VLM-Based Perception for Mobile Manipulation — Depth, Pose, Segmentation Fused Pipeline

## Summary

Real-time VLM-based perception for mobile manipulation fuses monocular depth estimation, 6D object pose estimation, and semantic/instance segmentation into a unified pipeline feeding Vision-Language-Action (VLA) models. Rather than running depth, pose, and segmentation as independent modules, modern approaches co-design these capabilities so the VLM backbone shares feature representations across tasks — enabling joint reasoning over geometry, semantics, and action. The fused pipeline must operate at 20–50 Hz for closed-loop visuomotor control, with monocular depth (Depth Anything V2) providing metric geometry from single-camera inputs. No single system has demonstrated the complete pipeline at target frequencies on edge hardware, but architectures like FALCON (spatial token injection), PointACT (3D point-cloud fusion), and G^3VLA (geometric inductive bias) provide converging design patterns.

## Key Facts

- **Fused over modular**: FALCON injects spatial tokens into a dedicated action head rather than contaminating the VLM backbone. PointACT uses 3D point clouds as the unified geometric primitive carrying depth, pose, and spatial structure in one representation.
- **Monocular depth backbone**: Depth Anything V2 (Small) produces dense depth maps at 50 FPS on RTX 3090, 18 FPS on Jetson Orin NX with INT8. Depth-augmented training improves grasp success by 12–18% in clutter.
- **6D pose for zero-shot manipulation**: Pose-VLA pre-trains on 1.4M images with 6.5M 3D annotations using discrete SE(3) Pose Tokens as universal geometric interface. Achieves 79.5% average success on RoboTwin 2.0 with only 100 demos per task.
- **Semantic segmentation as grounding**: Open-vocabulary segmentation (Grounding DINO + SAM) generates instance masks from language prompts, guiding depth and pose to focus on task-relevant objects.
- **Spatial chain-of-thought (Perceptio)**: Generates explicit 2D segmentation and 3D depth tokens as intermediate reasoning before text output, using frozen SAM2 encoder + VQ-VAE depth codebook.
- **Intent-driven perception (InCoM)**: Infers latent motion intent to dynamically reweight multi-scale perceptual features, allocating compute to depth/pose/segmentation based on manipulation stage.
- **3D-aware planning (3D HAMSTER)**: Eliminates 2D-to-3D grounding ambiguity by operating natively in 3D space for hierarchical VLM planning and control.
- **Benchmarking infrastructure**: MobileManiBench provides Isaac Sim-based trajectories with multi-modal annotations (language, multi-view RGB-depth-segmentation, synchronized states) for evaluating fused perception pipelines.

## Related (vault entities)

- [[Vision-Language-Action Models]]
- [[Depth Estimation]]
- [[6D Object Pose Estimation]]
- [[Semantic Segmentation Robotics]]
- [[Mobile Manipulation]]
- [[Multi-Modal Grounding: Language-to-Action Mapping]]
- [[VLA Edge Deployment: Qwen2.5-VL, Mobile-VideoGPT, SmolVLM]]
- [[Real-time SLAM Integration with VLA Policies]]
- [[World Action Models]]
- [[3D Gaussian Splatting for Perception]]

## Open Questions

- **End-to-end differentiable fused perception**: Can depth, pose, and segmentation be jointly optimized through a single pipeline conditioned on language instructions? Current systems share backbones but train heads separately.
- **Monocular depth metric calibration**: How to calibrate monocular depth for manipulation without LiDAR/stereo — known object priors, IMU fusion, or self-supervised calibration through grasping feedback?
- **Real-time edge deployment**: Can the full fused pipeline close at ≥20 Hz on consumer-grade edge hardware (single Jetson Orin) without cloud inference? AnywhereVLA's Orin+NUC split is the closest deployed architecture.
- **Cross-modal consistency**: When depth, pose, and segmentation disagree, how does the pipeline resolve conflicts? No system propagates uncertainty across heads.
- **3DGS-as-perception-replacement**: Can 3D Gaussian Splatting maps serve as the sole perception representation, replacing separate depth, pose, and segmentation modules entirely?
- **Intent-driven allocation generality**: InCoM shows stage-adaptive attention works, but does it generalize across task types and robot morphologies?
- **Perception-to-action latency budget**: How much of total latency is consumed by the fused perception pipeline vs. VLA inference? This split determines where optimization effort should go.
- **Benchmark coverage**: MobileManiBench has rich annotations but limited coverage of real-world edge cases (lighting, novel objects, dynamic scenes).

## Sources

- FALCON: Spatial token injection for VLA action heads [arxiv.org/abs/2503.05850]
- Pose-VLA: Universal SE(3) pose pretraining [arxiv.org/abs/2602.19710]
- PointACT: Multi-scale point-action interaction, RSS 2026 [arxiv.org/abs/2605.21414]
- InCoM: Intent-driven perceptual attention for mobile manipulation [arxiv.org/abs/2602.23024]
- 3D HAMSTER: 3D-aware hierarchical VLM planning [arxiv.org/abs/2606.31329]
- MobileManiBench: Isaac Sim-based mobile manipulation benchmark [arxiv.org/abs/2602.05233]
- Full consolidated note with 33 sources: `knowledge/robotics/vlm-perception-mobile-manipulation-fused-pipeline.md`

## Confidence

0.85: Core architectural claims are well-supported by 33+ published papers with concrete benchmarks and reproducible results. New material from InCoM, 3D HAMSTER, and MobileManiBench strengthens the field's coverage. Confidence held below 0.90 because no single system demonstrates the complete fused depth-pose-segmentation pipeline at target frequencies (20–50 Hz) on edge hardware end-to-end for mobile manipulation, and evaluation protocols remain fragmented across the robotics/VLM communities.