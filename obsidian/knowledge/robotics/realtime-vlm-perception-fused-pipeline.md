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
date: 2026-09-10
last_verified: 2026-09-10
sources:
  - url: "https://arxiv.org/abs/2503.05850"
    title: "FALCON: From Spatial to Action"
  - url: "https://arxiv.org/abs/2605.21414"
    title: "PointACT: Multi-Scale Point-Action Interaction"
  - url: "https://arxiv.org/abs/2602.19710"
    title: "Pose-VLA: Universal Pose Pretraining"
  - url: "https://arxiv.org/abs/2602.23024"
    title: "InCoM: Intent-Driven Perception"
  - url: "https://arxiv.org/abs/2606.31329"
    title: "3D HAMSTER: 3D-Aware Hierarchical Planning"
  - url: "https://arxiv.org/abs/2602.05233"
    title: "MobileManiBench"
  - url: "https://arxiv.org/abs/2509.21006"
    title: "AnywhereVLA"
  - url: "https://arxiv.org/abs/2606.24472"
    title: "G^3VLA: Geometric Inductive Bias"
  - url: "https://arxiv.org/abs/2605.14950"
    title: "Evo-Depth: Implicit Depth Encoding"
  - url: "https://arxiv.org/abs/2604.28197"
    title: "OmniRobotHome"
  - url: "https://arxiv.org/abs/2601.14133"
    title: "TwinBrainVLA"
  - url: "https://arxiv.org/abs/2602.05233"
    title: "MobileManiBench"
  - url: "https://arxiv.org/abs/2512.03724"
    title: "PosA-VLA: Pose-Conditioned Anchor Attention"
  - url: "https://arxiv.org/abs/2509.19012"
    title: "Pure VLA Models: Comprehensive Survey"
source_count: 33
confidence: 0.85
---

# Real-Time VLM-Based Perception for Mobile Manipulation — Depth, Pose, Segmentation Fused Pipeline

## Summary

Real-time VLM-based perception for mobile manipulation fuses monocular depth estimation, 6D object pose estimation, and semantic/instance segmentation into a unified pipeline feeding Vision-Language-Action (VLA) models. Rather than running depth, pose, and segmentation as independent modules, modern architectures co-design these capabilities so the VLM backbone shares feature representations across tasks — enabling joint reasoning over geometry, semantics, and action. Key design patterns include spatial token injection into dedicated action heads (FALCON), 3D point-cloud fusion as unified geometric primitive (PointACT), and geometric inductive bias via camera-intrinsics-conditioned ray embeddings (G^3VLA). No single system yet demonstrates the complete fused pipeline at 20–50 Hz on edge hardware end-to-end, but the converging architecture space is rapidly maturing toward deployable systems.

## Key Facts

- **Fused architecture beats modular stacking**: FALCON injects spatial tokens into a dedicated action head rather than contaminating the VLM backbone, preserving pre-trained semantic representations. PointACT uses 3D point clouds as a unified geometric primitive carrying depth, pose, and spatial structure in one representation. G^3VLA injects geometric inductive bias via intrinsic-conditioned ray embeddings and bidirectional cross-view fusion.

- **Monocular depth as geometry backbone**: Depth Anything V2 (Small) produces dense depth maps at 50 FPS on RTX 3090, 18 FPS on Jetson Orin NX with INT8 quantization. Depth-augmented training improves grasp success by 12–18% in clutter. Evo-Depth extracts implicit depth features from multi-view RGB without explicit 3D sensors.

- **6D pose for zero-shot manipulation**: Pose-VLA pre-trains on 1.4M images with 6.5M 3D annotations using discrete SE(3) Pose Tokens as universal geometric interface. Achieves 79.5% average success on RoboTwin 2.0 with only 100 demos per task. PosA-VLA uses pose-conditioned anchor attention to maintain spatial correspondence between robot pose and visual scene.

- **Semantic segmentation as grounding layer**: Open-vocabulary segmentation (Grounding DINO + SAM) generates instance masks from language prompts, guiding depth and pose to task-relevant objects. SimpleSeg eliminates separate decoder heads by reframing segmentation as point-coordinate sequence generation within the VLM's language space.

- **Intent-driven perception allocation**: InCoM infers latent motion intent to dynamically reweight multi-scale perceptual features, redirecting compute to depth/pose/segmentation based on manipulation stage — addressing efficiency gaps where all modalities run at fixed bandwidth regardless of task phase.

- **3D-aware planning eliminates 2D-to-3D ambiguity**: 3D HAMSTER operates natively in 3D space for hierarchical VLM planning and control, reducing occlusion and viewpoint failures that plague 2D image-space fused perception pipelines.

- **Real-time edge deployment architectures**: AnywhereVLA splits perception/VLA (SmolVLA, 450M params) on Jetson Orin NX and SLAM/control on Intel NUC, achieving multi-room mobile manipulation at real-time frequencies. TwinBrainVLA uses asymmetric dual-stream (frozen Left Brain + trainable Right Brain via AsyMoT) to avoid catastrophic forgetting.

- **Benchmarking infrastructure**: MobileManiBench provides Isaac Sim-based trajectories with multi-modal annotations (language, multi-view RGB-depth-segmentation, synchronized states). OmniRobotHome's 48-camera testbed runs a ~16 Hz fused stereo manipulation pipeline and treats perception quality as an experimental variable.

- **Spatial chain-of-thought (Perceptio)**: Generates explicit 2D segmentation and 3D depth tokens as intermediate reasoning before text output, using frozen SAM2 encoder + VQ-VAE depth codebook. Ablation shows depth tokens are critical for spatial reasoning — removing them collapses HardBLINK accuracy by 25.8%.

- **World Action Models unification**: WAM survey (arXiv:2606.20781) shows depth, pose, and segmentation can be projected into semantic, depth, and flow latent-action codebooks before action decoding — unifying all three perception modalities under a single world-model framework where rendered future states serve as common planning currency.

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
- [[Self-Improving Autonomous Agents]]

## Open Questions

- **End-to-end differentiable fused perception**: Can depth, pose, and segmentation be jointly optimized through a single differentiable pipeline conditioned on language instructions? Current systems share backbones but train heads separately.

- **Monocular depth metric calibration**: How to calibrate monocular depth for manipulation without LiDAR/stereo — known object priors, IMU fusion, or self-supervised calibration through grasping feedback? No system has demonstrated robust metric depth from monocular RGB alone in uncalibrated environments.

- **Cross-modal consistency**: When depth, pose, and segmentation disagree (e.g., depth suggests an object is 2m away, pose suggests 1.5m), how does the pipeline resolve conflicts? No system propagates uncertainty across heads.

- **Real-time edge deployment at target frequency**: Can the full fused pipeline close at ≥20 Hz on a single Jetson Orin without cloud inference? AnywhereVLA's Orin+NUC split is the closest deployed architecture.

- **3DGS-as-perception-replacement**: Can 3D Gaussian Splatting maps serve as the sole perception representation, replacing separate depth, pose, and segmentation modules entirely? SemGauss-SLAM shows the representation is possible; feeding it as VLA conditioning remains untested on real robots.

- **Perception-to-action latency budget**: How much of total latency is consumed by fused perception vs. VLA inference? This split determines where optimization effort should go, but no standardized benchmark exists.

- **Active perception co-optimization**: Can the robot move the camera to simultaneously improve depth, pose, and segmentation? SaPaVe provides 200K image-language-camera-movement pairs but active loops co-optimizing all three modalities remain unexplored.

- **Benchmark standardization**: No unified benchmark measures perception pipeline latency → manipulation success rate for mobile manipulation, making cross-system evaluation difficult.

## Sources

1. FALCON: Spatial token injection for VLA action heads [arxiv.org/abs/2503.05850]
2. PointACT: Multi-scale point-action interaction, RSS 2026 [arxiv.org/abs/2605.21414]
3. Pose-VLA: Universal SE(3) pose pretraining [arxiv.org/abs/2602.19710]
4. InCoM: Intent-driven perception for mobile manipulation [arxiv.org/abs/2602.23024]
5. 3D HAMSTER: 3D-aware hierarchical planning [arxiv.org/abs/2606.31329]
6. MobileManiBench: Isaac Sim mobile manipulation benchmark [arxiv.org/abs/2602.05233]
7. AnywhereVLA: Real-time mobile manipulation split architecture [arxiv.org/abs/2509.21006]
8. G^3VLA: Geometric inductive bias via ray embeddings [arxiv.org/abs/2606.24472]
9. Evo-Depth: Implicit depth encoding without explicit sensors [arxiv.org/abs/2605.14950]
10. OmniRobotHome: Room-scale fused perception testbed [arxiv.org/abs/2604.28197]
11. TwinBrainVLA: Asymmetric dual-stream VLA [arxiv.org/abs/2601.14133]
12. PosA-VLA: Pose-conditioned anchor attention [arxiv.org/abs/2512.03724]
13. SimpleSeg: Decoder-free VLM segmentation [arxiv.org/abs/2601.19228]
14. Pure VLA Models survey [arxiv.org/abs/2509.19012]
15. CogVLA: Cognition-aligned VLA, NeurIPS 2025 [github.com/iLearn-Lab/NeurIPS25-CogVLA]
16. G2VLM: Geometry-grounded VLM, CVPR 2026 [github.com/InternRobotics/G2VLM]
17. SemGauss-SLAM: Dense semantic Gaussian Splatting SLAM, IROS 2025
18. MolmoAct2: Open-source VLA, 87.1% real-world success [allenai.org/blog/molmoact2]
19. WAM survey: World Action Models [arxiv.org/abs/2606.20781]
20. Phys2Real: VLM physical priors for sim-to-real [arxiv.org/abs/2510.11689]
21. LA4VLA: Vision-agnostic language-action pretraining [arxiv.org/abs/2606.27295]
22. PAMAE: Phase-aware Mixture-of-Experts [arxiv.org/abs/2606.27144]
23. EchoVLA: Declarative memory for long-horizon manipulation [arxiv.org/abs/2511.18112]
24. MAP-VLA: Memory-augmented VLA prompting [arxiv.org/abs/2511.09516]
25. SpatialVLM: 2B synthetic spatial VQA examples, CVPR 2024 [Google DeepMind]
26. HoloAgent-0: Unified navigation + perception framework [arxiv.org/abs/2606.23565]
27. RoboStream: Spatio-temporal reasoning with memory [arxiv.org/abs/2603.12939]
28. OK-Robot: Zero-shot mobile manipulation [Meta]
29. IROS: Dual-process real-time VLM architecture [arxiv.org/abs/2601.21506]
30. RoboGround: GLaMM-grounded segmentation [CVPR 2025]
31. TIC-VLA: Think-in-Control latency framework
32. SaPaVe: Active perception for VLA [CVPR 2025]
33. Knowledge Insulating VLA models [arxiv.org/abs/2505.23705]

## Confidence

**0.85**: Core architectural claims (FALCON token injection, OpenVLA fusion, Depth Anything V2 performance, Pose-VLA universal pose pretraining) are well-supported by 33+ published papers with concrete benchmarks and reproducible results. The field spans spatial token injection, geometric inductive bias, asymmetric dual-stream VLAs, declarative memory, intent-driven perception, and 3D-aware planning. Confidence held below 0.90 because: (a) no single system demonstrates the complete fused depth-pose-segmentation pipeline at target frequencies (20–50 Hz) on edge hardware end-to-end for mobile manipulation; (b) evaluation protocols remain fragmented across the robotics/VLM communities with no standardized perception-to-manipulation-latency benchmark; (c) 3DGS-as-VLA-conditioning and end-to-end differentiable fused perception remain aspirational rather than demonstrated; (d) the specific latency budget split between perception pipeline and VLA inference is unmeasured as an integrated system.