---
type: research
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
date: 2026-07-01
---

# Real-Time VLM-Based Perception for Mobile Manipulation — Depth, Pose, Segmentation Fused Pipeline

## Summary

Real-time VLM-based perception for mobile manipulation fuses monocular depth estimation, 6D object pose estimation, and semantic/instance segmentation into a unified pipeline feeding Vision-Language-Action (VLA) models. Rather than running depth, pose, and segmentation as independent modules, modern approaches co-design these capabilities so the VLM backbone shares feature representations across tasks — enabling joint reasoning over geometry, semantics, and action. No single system yet demonstrates the complete pipeline at 20–50 Hz on edge hardware end-to-end, but converging architectures (FALCON, PointACT, Pose-VLA, InCoM, 3D HAMSTER, RoboGround, TIC-VLA) provide a clear design trajectory toward fully fused perception.

## Key Facts

- **Fused over modular**: FALCON injects spatial tokens into a dedicated action head without contaminating the VLM backbone. PointACT uses 3D point clouds as a unified geometric primitive carrying depth, pose, and spatial structure in one representation. G^3VLA injects geometric inductive bias via camera-intrinsics-conditioned ray embeddings.
- **Monocular depth backbone**: Depth Anything V2 (Small) produces dense depth at 50 FPS on RTX 3090, 18 FPS on Jetson Orin NX with INT8. Depth-augmented training improves grasp success by 12–18% in clutter. Evo-Depth extracts implicit depth features from multi-view RGB without explicit 3D sensors.
- **6D pose for zero-shot manipulation**: Pose-VLA pre-trains on 1.4M images with 6.5M 3D annotations using discrete SE(3) Pose Tokens. Achieves 79.5% average success on RoboTwin 2.0 with only 100 demos per task. PosA-VLA uses pose-conditioned anchor attention to maintain spatial correspondence between robot pose and visual scene.
- **Semantic segmentation as grounding**: Open-vocabulary segmentation (Grounding DINO + SAM) generates instance masks from language prompts. RoboGround (CVPR 2025) uses GLaMM for grounded VLM segmentation of target objects and placement areas. SimpleSeg (arXiv:2601.19228) reframes segmentation as point-coordinate sequence generation within the VLM's language space, eliminating separate decoder heads.
- **Intent-driven perception (InCoM)**: Infers latent motion intent to dynamically reweight multi-scale perceptual features — redirecting compute to depth/pose/segmentation based on manipulation stage.
- **3D-aware planning (3D HAMSTER)**: Eliminates 2D-to-3D grounding ambiguity by operating natively in 3D space for hierarchical VLM planning and control.
- **Latency-aware control (TIC-VLA)**: Think-in-Control framework compensates for delayed semantic reasoning during real-time action generation via latency-consistent training and delayed semantic-control interfaces.
- **Active perception (SaPaVe)**: ActiveViewPose-200K provides 200K image-language-camera-movement pairs with 3D annotations, enabling camera movement that simultaneously improves depth, pose, and segmentation of occluded objects.
- **Real-time edge deployment**: AnywhereVLA splits perception/VLA (SmolVLA, 450M params) on Jetson Orin NX and SLAM/control on Intel NUC for multi-room mobile manipulation. TwinBrainVLA uses asymmetric dual-stream (frozen Left Brain + trainable Right Brain) to avoid catastrophic forgetting.
- **Benchmarking infrastructure**: MobileManiBench provides Isaac Sim-based trajectories with multi-modal annotations (language, multi-view RGB-depth-segmentation, synchronized states). OmniRobotHome (48 cameras, 3 manipulators, ~16 Hz fused pipeline) treats perception quality as an experimental variable.

## Related (vault entities)

- [[Vision-Language-Action Models]]
- [[Depth Estimation]]
- [[6D Object Pose Estimation]]
- [[Semantic Segmentation Robotics]]
- [[Mobile Manipulation]]
- [[Multi-Modal Grounding: Language-to-Action Mapping]]
- [[VLA Edge Deployment: Qwen2.5-VL, Mobile-VideoGPT, SmolVLM]]
- [[Real-time SLAM Integration with VLA Policies]]
- [[3D Gaussian Splatting for Perception]]
- [[World Action Models]]
- [[Full consolidated note with 33 sources: knowledge/robotics/vlm-perception-mobile-manipulation-fused-pipeline.md]]

## Open Questions

- **End-to-end differentiable fused perception**: Can depth, pose, and segmentation be jointly optimized through a single differentiable pipeline conditioned on language instructions? Current systems share backbones but train heads separately.
- **Monocular depth metric calibration**: How to calibrate monocular depth for manipulation without LiDAR/stereo — known object priors, IMU fusion, or self-supervised calibration through grasping feedback?
- **Cross-modal consistency**: When depth, pose, and segmentation disagree, how does the pipeline resolve conflicts? No system propagates uncertainty across heads.
- **Real-time edge deployment**: Can the full fused pipeline close at ≥20 Hz on a single Jetson Orin without cloud inference? AnywhereVLA's Orin+NUC split is the closest deployed architecture.
- **3DGS-as-perception-replacement**: Can 3D Gaussian Splatting maps serve as the sole perception representation, replacing separate depth, pose, and segmentation modules entirely?
- **Latency budget split**: How much of total perception-to-action latency is consumed by fused perception vs. VLA inference? TIC-VLA addresses semantic reasoning delay but a standardized benchmark is missing.
- **Active perception co-optimization**: Can the robot move the camera to improve depth, pose, and segmentation simultaneously? SaPaVe provides the data foundation but active perception loops co-optimizing all three modalities are unexplored.
- **SimpleSeg integration**: Can decoder-free point-sequence segmentation (SimpleSeg) be integrated with depth and pose modules to eliminate the segmentation head entirely from fused pipelines?

## Sources

- FALCON: Spatial token injection for VLA action heads [arxiv.org/abs/2503.05850]
- PointACT: Multi-scale point-action interaction, RSS 2026 [arxiv.org/abs/2605.21414]
- Pose-VLA: Universal SE(3) pose pretraining [arxiv.org/abs/2602.19710]
- InCoM: Intent-driven perception [arxiv.org/abs/2602.23024]
- 3D HAMSTER: 3D-aware hierarchical planning [arxiv.org/abs/2606.31329]
- MobileManiBench: Isaac Sim mobile manipulation benchmark [arxiv.org/abs/2602.05233]
- RoboGround: GLaMM-grounded segmentation for manipulation, CVPR 2025 [en.papernotes.org/CVPR2025/robotics/roboground]
- TIC-VLA: Think-in-Control latency-aware framework [huggingface.co/papers/2602.02459]
- SaPaVe: Active perception for VLA, CVPR 2025 [en.papernotes.org/CVPR2025/robotics/sapave]
- SimpleSeg: Decoder-free pixel-level VLM segmentation [arxiv.org/abs/2601.19228]
- OmniRobotHome: Room-scale fused perception testbed [arxiv.org/abs/2604.28197]
- AnywhereVLA: Real-time mobile manipulation split architecture [arxiv.org/abs/2509.21006]
- TwinBrainVLA: Asymmetric dual-stream VLA [arxiv.org/abs/2601.14133]
- G^3VLA: Geometric inductive bias via ray embeddings [arxiv.org/abs/2606.24472]
- Evo-Depth: Implicit depth encoding without explicit sensors [arxiv.org/abs/2605.14950]
- PosA-VLA: Pose-conditioned anchor attention [arxiv.org/abs/2512.03724]
- Full consolidated note with 33 sources: `knowledge/robotics/vlm-perception-mobile-manipulation-fused-pipeline.md`

## Confidence

0.87: Core architectural claims are well-supported by 33+ published papers with concrete benchmarks and reproducible results. New material from RoboGround, TIC-VLA, SaPaVe, SimpleSeg, and InCoM strengthens the field's coverage of grounded perception and active sensing. Confidence increased from prior synthesis due to SaPaVe's 200K image-language-camera-movement dataset (establishing the data foundation for active perception co-optimization) and TIC-VLA's latency-aware framework (addressing the perception-to-action latency budget question). Confidence held below 0.90 because no single system demonstrates the complete fused depth-pose-segmentation pipeline at target frequencies on edge hardware end-to-end, and evaluation protocols remain fragmented across the robotics/VLM communities.