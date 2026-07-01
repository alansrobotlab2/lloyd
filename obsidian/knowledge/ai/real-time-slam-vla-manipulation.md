---
title: Real-time SLAM Integration with VLA Policies — Mapping During Manipulation Tasks
tags:
  - ai/vla-models
  - robotics/slam
  - robotics/mobile-manipulation
  - robotics/environment-exploration
  - embodied-ai
  - research/domain-research
created: 2026-07-15
updated: 2026-07-28
confidence: 0.87
---

# Real-time SLAM Integration with VLA Policies — Mapping During Manipulation Tasks

## Summary

Integrating simultaneous localization and mapping (SLAM) with Vision-Language-Action (VLA) policies enables robots to explore unknown environments and manipulate objects using only natural language instructions. The dominant paradigm remains a modular pipeline: classical SLAM provides spatial memory and map-building, active exploration discovers targets, and a fine-tuned VLA model handles grasping and manipulation. The field is converging around memory-augmented VLA architectures (MAP-VLA, EvoVLA) that give policies persistent spatial context, while 3D Gaussian Splatting-based SLAM (DynaGSLAM, GSWorld) is maturing as the next-generation bridge between geometric mapping and semantic action. AnywhereVLA remains the clearest end-to-end reference, with newer systems addressing the long-standing "spatial memory gap" in VLA policies through memory-augmented prompting, selective context recall, and Gaussian-based scene reconstruction.

## Key Facts

- **AnywhereVLA (Gubernatorov et al., arXiv:2509.21006)** implements a full modular pipeline for large-scale indoor mobile manipulation in unseen environments. A single language instruction conditions both SLAM-based exploration and VLA manipulation. The system parses natural language into a task graph driving LiDAR-Inertial-Visual SLAM, metric semantic mapping, task-aware frontier exploration, approach planning, and a SmolVLA (450M parameters) manipulation head. Achieves 46% task success rate on multi-room labs, running on Jetson Orin NX (perception/VLA) + Intel NUC (SLAM/control) at real-time frequencies.

- **RSV-SLAM (arXiv:2510.02616, Oct 2025)** introduces real-time semantic RGBD SLAM specifically designed for dynamic indoor environments. Unlike traditional visual SLAM that assumes a static world, RSV-SLAM handles object motion and scene changes — a critical capability for SLAM systems operating during manipulation where gripper and arm motion introduce ego-motion artifacts. Evaluated on TUM RGB-D datasets with dynamic scenarios.

- **SlideSLAM (Liu et al., arXiv:2406.17249, T-RO 2025)** is a sparse, lightweight, decentralized metric-semantic SLAM system for multi-robot navigation. Key innovation: semantics-driven place recognition that leverages object-level metric-semantic maps for inter-robot loop closure detection. This demonstrates that semantic SLAM maps — the kind needed to bridge SLAM output to VLA input — can be maintained across distributed agents with lightweight communication.

- **CoT-VLA (Zhao et al., arXiv:2503.22020)** is a 7B VLA with visual chain-of-thought reasoning, achieving +17% real-world and +6% simulation improvement over prior VLA baselines on manipulation tasks. Pretrained with both robot demonstration data and action-less video data through intermediate visual reasoning steps. This addresses the visual reasoning gap that limits VLA performance on complex manipulation tasks downstream of SLAM-based navigation.

- **MAP-VLA (Li et al., arXiv:2511.09516, Nov 2025)** introduces Memory-Augmented Prompting for VLA models, empowering pre-trained VLAs with demonstration-derived memory prompts to augment action generation for long-horizon robotic manipulation. Addresses the core VLA spatial memory gap: rather than relying on SLAM to provide external maps, MAP-VLA embeds memory into the VLA's prompting mechanism, enabling dynamic recall of relevant task history during multi-step manipulation. Selected for ICRA 2026.

- **EvoVLA (arXiv:2511.16166, Nov 2025)** is a self-evolving VLA that mitigates long-horizon stage hallucination via self-supervised rewards, pose-grounded exploration, and selective memory. For tasks with 70+ steps, EvoVLA uses Context Selection to recall only critical history tokens needed for the current decision, preventing catastrophic forgetting. Achieves strong Sim2Real robustness on Discoverse-L benchmark, demonstrating that VLA models can maintain coherent task state over extended manipulation sequences without external SLAM conditioning.

- **VL-Nav (arXiv:2502.00931)** presents real-time vision-language navigation with spatial reasoning, operating at 30 Hz on Jetson Orin NX with an 86.3% success rate. Uses heuristic vision-language (HVL) spatial reasoning on both frontier-based and instance-based target points, with partial frontier detection on dynamic occupancy maps. Demonstrates that efficient spatial reasoning for navigation can run on low-power edge hardware alongside VLA inference.

- **TwinBrainVLA (arXiv:2601.14133)** addresses catastrophic forgetting via dual-stream architecture (frozen "Left Brain" for semantics, trainable "Right Brain" for embodied perception) but still lacks environment-scale spatial reasoning. Standard VLAs (π0, π0.5, OpenVLA) remain room-scale policies trained on localized demonstration data without persistent spatial memory mechanisms.

- **NaVILA (OpenReview: gkDRrvqeWF, RSS 2025)** proposes a two-level VLA framework for legged robot navigation. A high-level VLA generates language-based commands while a real-time locomotion policy handles obstacle avoidance. Demonstrates that VLA models can be adapted for Vision-and-Language Navigation (VLN) on legged platforms in cluttered scenes. Benchmarks on IsaacLab with realistic scenes and real-robot experiments.

- **3D Gaussian Splatting for SLAM is maturing rapidly**: Multiple systems (DynaGSLAM, GS-SLAM, AG-SLAM, GSORBSLAM) now demonstrate real-time 3DGS-SLAM with dynamic environment handling. GSWorld combines 3DGS with physics engines for closed-loop photorealistic sim-to-real simulation. LEGS (legsvla.github.io) uses 3DGS-based hybrid simulation for teleop-free VLA fine-tuning on humanoid robots (Unitree G1), showing Gaussian splatting maps can serve as realistic training environments for loco-manipulation VLAs. SemGauss-SLAM (IROS 2025) addss semantic labeling to Gaussian SLAM, directly bridging the geometric-to-semantic gap for VLA input.

- **MemoryVLA** (Guo et al., GitHub: G-U-O/memvla) proposes a Cognition-Memory-Action framework for long-horizon robotic manipulation, achieving +14.6% gain on Bridge and +11.8% on Mikasa-Robo benchmarks. Integrates with the Dexbotic VLA codebase, demonstrating that memory-augmented VLA architectures are becoming integrated into open-source tooling.

- **Semantic 3D mapping fuses SLAM with object detection**: AnywhereVLA constructs a 3D semantic object map by synchronizing RGB images, LiDAR point clouds, and 2D bounding-box detections. LiDAR points are projected into camera frames, voxelized, and associated with object detections via enlarged 2D bounding boxes. Per-class point clouds are clustered (DBSCAN) and summarized with centroid, covariance, and confidence estimates. This map supports task-aware active exploration conditioned on target object class.

- **Map consistency during manipulation remains unsolved**: Manipulation induces camera ego-motion (arm movements, gripper self-occlusion, scene changes) that corrupt SLAM estimates. Existing systems decouple navigation/exploration phases from manipulation phases. Continuous mapping during manipulation — where the robot must maintain spatial consistency while actively changing the scene — is an open research challenge. RSV-SLAM makes progress on dynamic scene handling but does not address manipulation-induced ego-motion specifically.

- **Edge deployment is feasible but requires split compute**: AnywhereVLA splits compute across two devices (Jetson Orin NX for VLA/perception, Intel NUC for SLAM/control). The full SLAM + semantic mapping + VLA inference + exploration planning pipeline at >10 Hz is resource-intensive. VL-Nav demonstrates 30 Hz operation on single Jetson Orin NX for navigation alone, but unified SLAM+VLA+manipulation on a single device remains open.

- **Spectral GS-SLAM (arXiv:2606.21258, Jun 2026)** introduces observability-aware, degeneracy-robust tracking for Gaussian Splatting SLAM. Uses second-order optimization with neural rendering for real-time SLAM across diverse indoor environments. Critical for SLAM+VLA integration: addresses tracking reliability under degeneracy conditions (textureless surfaces, repetitive patterns) common in manipulation environments.

### Mid-2026 Developments

- **DIM-WAM (Wang et al., arXiv:2606.27677, Jun 2026)** is a memory-augmented world-action model for long-horizon robot manipulation. It augments a base world-action model with diverse historical event memory: extracting compact visual events from real observations, updating multiple memory banks through independent similarity-based merging, and reading bank-identity- and time-embedded long-term context to condition video and action denoising. A progress-supervision objective encourages memory tokens to encode completed events, the current task stage, and implications for the remaining task. On RMBench, DIM-WAM raises average success from 28.4% (LingBot-VA baseline) to 69.8%, exceeding Mem-0 at 42.0%. On four real-world Franka tasks, it improves average stage success from 70.7% to 91.5% and full-task success from 52.5% to 80.0%. This directly demonstrates that structured historical event memory — a SLAM-adjacent capability — dramatically improves long-horizon manipulation.

- **μVLA (arXiv:2606.12497, Jun 2026)** is a controlled isolation study of recurrence in a strong pretrained VLA backbone. Learnable memory tokens carried across timesteps and trained with TBPTT improve manipulation under partial observability on MIKASA-Robo and LIBERO benchmarks. The model produces updated memory tokens recurrently passed to t+1, providing a lightweight alternative to external SLAM for short-horizon spatial reasoning under partial observability.

- **MUVLA (Han et al., arXiv:2509.25966, Sep 2025)** is a Map Understanding VLA tailored for object navigation. It takes current and history observations plus a semantic map as input and predicts action sequences based on a textual goal description. The three-stage training pipeline (map-level spatial understanding → behavior imitation → reward amplification) enables the model to unify diverse demonstrations into a robust spatial representation. Evaluated on HM3D and Gibson benchmarks, demonstrating effective exploration behaviors even from low-quality or partially successful trajectories. This is a concrete SLAM-to-VLA bridge: semantic maps serve as structured spatial context for VLA action prediction.

- **RoboMME (arXiv:2603.04639, Mar 2026)** benchmarks memory for robotic generalist policies across 16 manipulation tasks spanning temporal, spatial, object, and procedural memory. The study develops 14 memory-augmented VLA variants on the π0.5 backbone to systematically explore different memory representations. Provides a systematic evaluation framework for comparing memory-augmented VLA approaches — including those that incorporate SLAM-derived spatial context vs. internal memory — against the "no memory" baseline.

- **EgoHumanoid (arXiv:2602.10106, Feb 2026)** is the first framework for human-to-humanoid loco-manipulation transfer. By aligning egocentric human demonstrations with robot data through view transformation and unified action space, it enables effective VLA co-training without requiring robot-specific demonstrations. Relevant to SLAM+VLA integration because it demonstrates loco-manipulation transfer in a framework that could incorporate spatial memory.

- **Vesta (arXiv:2606.20905, Jun 2026)** is a generalist embodied reasoning model evaluated on real robotic manipulation tasks using tabletop bimanual grippers. Demonstrates the maturation of generalist embodied models capable of reasoning across multiple manipulation modalities.

- **HoloAgent-0 (arXiv:2606.23565, Jun 2026)** introduces a unified embodied agent framework with 3D spatial memory. The Memory Layer converts sensor streams into a persistent 3D world representation — including geometry, topology, occupancy, robot pose, and open-vocabulary semantic labels — that an Embodied AgentOS can query for grounding, localization, navigation, and manipulation. Unlike the modular SLAM+VLA split in AnywhereVLA, HoloAgent-0 treats spatial memory as a first-class component of the agent architecture, enabling cross-task spatial reasoning without explicit SLAM conditioning. This represents a significant architectural shift: rather than SLAM feeding into VLA, spatial memory is embedded in the agent loop.

- **WholeBodyVLA (ICLR 2026)** presents a unified latent VLA framework for whole-body loco-manipulation control on humanoids (Agibot X2). Enables large-space locomotion + manipulation in a single end-to-end policy trained from large-scale egocentric human videos. While not explicitly integrating SLAM, it demonstrates that unified perception-action policies for large-space operation are viable — a prerequisite for SLAM+VLA integration where spatial awareness must span room-scale distances.

## Related (vault entities)
- [[Online Fine-Tuning for VLA Models — Continual Learning with Experience Replay]]
- [[VLA Edge Deployment: Qwen2.5-VL, Mobile-VideoGPT, SmolVLM]]
- [[Language-to-Action Mapping in VLMs/VLAs]]
- [[Multi-Agent Task Decomposition with Hierarchical Planning]]
- [[Multi-Modal Grounding: Language-to-Action Mapping]]
- [[Self-Improving Autonomous Agents]]

## Open Questions

1. **End-to-end SLAM+VLA integration**: Can a single differentiable model jointly perform SLAM and manipulation without a modular split? MAP-VLA and EvoVLA embed memory in the VLA itself but still rely on external SLAM for spatial grounding. Fully end-to-end SLAM+VLA remains unrealized.

2. **Manipulation-induced ego-motion**: How do SLAM systems handle camera motion introduced by robotic arm manipulation? Most systems decouple navigation and manipulation phases, but continuous mapping during manipulation remains unsolved. RSV-SLAM handles dynamic objects but not ego-motion from arm/gripper movement.

3. **Semantic map updating during interaction**: When a robot picks up an object, the semantic map must update to reflect the object's removal. Existing systems construct maps passively during exploration but lack active map maintenance during manipulation.

4. **3DGS as VLA input representation**: Can 3D Gaussian Splatting maps serve as direct visual context for VLA policies? SemGauss-SLAM and GSWorld show promise, but no system yet feeds 3DGS maps directly as VLA input conditioning. This remains a promising bridge between SLAM output and VLA visual context.

5. **Memory-augmented vs. SLAM-conditioned architectures**: MAP-VLA and EvoVLA show that VLA models can maintain their own memory. Does this make external SLAM redundant for short-horizon tasks, or does SLAM remain essential for environment-scale spatial reasoning that exceeds VLA context windows?

6. **Cross-environment generalization**: Systems like AnywhereVLA were evaluated in controlled multi-room labs. How do they perform in truly unstructured, human-populated environments with moving obstacles and dynamic scene changes?

7. **Unified edge deployment**: Can the full SLAM + semantic mapping + VLA pipeline fit on a single consumer-grade edge device? VL-Nav shows 30 Hz navigation on Orin NX, but adding manipulation and SLAM to the same device is untested.

8. **Active perception loops**: Can VLA policies actively control camera and sensor configuration to improve SLAM quality — e.g., choosing viewpoints that reduce localization uncertainty while simultaneously gathering data useful for manipulation planning?

9. **Multi-robot SLAM for VLA coordination**: If SlideSLAM-style semantic SLAM enables inter-robot map sharing, can multiple mobile manipulators coordinate exploration and manipulation using shared semantic maps?

10. **Sim2Real for SLAM+VLA**: LEGS demonstrates Gaussian-splatting-based sim2real for loco-manipulation. Can similar approaches be applied to SLAM+VLA integration, using photorealistic Gaussian scenes as training environments for joint navigation-manipulation policies?

## Sources

- Gubernatorov et al. "AnywhereVLA: Language-Conditioned Exploration and Mobile Manipulation," arXiv:2509.21006, Sep 2025. Primary reference for SLAM+VLA modular integration. [https://arxiv.org/abs/2509.21006](https://arxiv.org/abs/2509.21006)

- Li et al. "MAP-VLA: Memory-Augmented Prompting for Vision-Language-Action Model in Robotic Manipulation," arXiv:2511.09516, Nov 2025. Memory-augmented VLA for long-horizon manipulation. ICRA 2026. [https://arxiv.org/abs/2511.09516](https://arxiv.org/abs/2511.09516)

- "EvoVLA: Self-Evolving Vision-Language-Action Model," arXiv:2511.16166, Nov 2025. Self-evolving VLA with selective memory for long-horizon tasks. [https://arxiv.org/abs/2511.16166](https://arxiv.org/abs/2511.16166)

- "VL-Nav: Real-time Vision-Language Navigation with Spatial Reasoning," arXiv:2502.00931. Real-time VLN with spatial reasoning at 30 Hz on edge hardware. [https://arxiv.org/abs/2502.00931](https://arxiv.org/abs/2502.00931)

- "LEGS: Fine-Tuning Teleop-Free VLAs for Humanoid Loco-Manipulation," legsvla.github.io. Gaussian splatting-based sim for VLA fine-tuning on Unitree G1.

- RSV-SLAM authors. "RSV-SLAM: Toward Real-Time Semantic Visual SLAM in Indoor Dynamic Environments," arXiv:2510.02616, Oct 2025. Semantic SLAM for dynamic environments. [https://arxiv.org/abs/2510.02616](https://arxiv.org/abs/2510.02616)

- Liu et al. "SlideSLAM: Sparse, Lightweight, Decentralized Metric-Semantic SLAM for Multi-Robot Navigation," arXiv:2406.17249, T-RO 2025. Multi-robot semantic SLAM with place recognition. [https://arxiv.org/abs/2406.17249](https://arxiv.org/abs/2406.17249)

- Zhao et al. "CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models," arXiv:2503.22020, Mar 2025. VLA with visual reasoning for manipulation. [https://arxiv.org/abs/2503.22020](https://arxiv.org/abs/2503.22020)

- Cheng et al. "NaVILA: Legged Robot Vision-Language-Action Model for Navigation," OpenReview (gkDRrvqeWF), RSS 2025. VLA for legged robot navigation. [https://openreview.net/forum?id=gkDRrvqeWF](https://openreview.net/forum?id=gkDRrvqeWF)

- Yu et al. "TwinBrainVLA: Unleashing the Potential of Generalist VLMs for Embodied Tasks via Asymmetric Mixture-of-Transformers," arXiv:2601.14133, 2026. Catastrophic forgetting mitigation in VLA. [https://arxiv.org/abs/2601.14133](https://arxiv.org/abs/2601.14133)

- "SemGauss-SLAM: Dense Semantic Gaussian Splatting SLAM," IROS 2025. Semantic labeling for Gaussian SLAM.

- GSWorld. "GSWorld: Photo-Realistic Closed-Loop Simulator for Robotic Manipulation," 2025. 3DGS-based sim-to-real pipeline.

- Guo et al. "MemoryVLA: Cognition-Memory-Action Framework." GitHub: G-U-O/memvla, 2025. Memory-augmented VLA framework.

- Kim et al. "π0: A Vision-Language-Action Flow Model for General Robot Control," arXiv:2410.24164. Foundational VLA reference.

- 3DGS-SLAM survey. "How NeRFs and 3D Gaussian Splatting are Reshaping SLAM," arXiv:2402.13255. Emerging spatial representation for robot mapping.

- Bourgeois, D. "12 Predictions for Embodied AI and Robotics in 2026." 2026 outlook on 3DGS as standard spatial representation.

- Wang et al. "DIM-WAM: World-Action Modeling with Diverse Historical Event Memory," arXiv:2606.27677, Jun 2026. Memory-augmented world-action model with multi-bank event memory for long-horizon manipulation. Project: [https://wangkai-casia.github.io/dim-wam/](https://wangkai-casia.github.io/dim-wam/)

- "μVLA: On Recurrent Memory for Partially Observable Manipulation," arXiv:2606.12497, Jun 2026. Recurrent learnable memory tokens for manipulation under partial observability. GitHub: [https://github.com/CognitiveAISystems/muVLA](https://github.com/CognitiveAISystems/muVLA)

- Han et al. "MUVLA: Learning to Explore Object Navigation via Map Understanding," arXiv:2509.25966, Sep 2025. Map-understanding VLA that takes semantic maps as structured spatial context. Three-stage training pipeline for spatial understanding.

- "RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies," arXiv:2603.04639, Mar 2026. Systematic evaluation of 14 memory-augmented VLA variants across temporal, spatial, object, and procedural memory categories.

- "EgoHumanoid: Unlocking In-the-Wild Loco-Manipulation with Robot-Free Egocentric Demonstrations," arXiv:2602.10106, Feb 2026. Human-to-humanoid loco-manipulation transfer framework.

- "Vesta: A Generalist Embodied Reasoning Model," arXiv:2606.20905, Jun 2026. Generalist embodied reasoning with real robot evaluation on bimanual manipulation tasks.

- Huang et al. "GraphCoT-VLA: A 3D Spatial-Aware Reasoning Vision-Language-Action Model for Robotic Manipulation with Ambiguous Instructions," arXiv:2508.07650, Aug 2025. Graph-based spatial reasoning for VLA. [https://arxiv.org/abs/2508.07650](https://arxiv.org/abs/2508.07650)

- Sandipan D. Asan. "From Pixels to Actions: The Hidden Role of SLAM in VLA Model Training and Evaluation," 2025. Comprehensive analysis of SLAM's role in VLA training and evaluation. [https://mrsandipandas.github.io/files/slam-vla.pdf](https://mrsandipandas.github.io/files/slam-vla.pdf)

- "Spectral GS-SLAM: Observability-Aware, Degeneracy-Robust Tracking for Gaussian Splatting SLAM," arXiv:2606.21258, Jun 2026. Real-time 3DGS-SLAM with second-order optimization. [https://arxiv.org/pdf/2606.21258](https://arxiv.org/pdf/2606.21258)

- Zhou et al. "HoloAgent-0: A Unified Embodied Agent Framework with 3D Spatial Memory," arXiv:2606.23565, Jun 2026. Unified embodied agent with 3D spatial memory layer for cross-task spatial reasoning. [https://arxiv.org/abs/2606.23565](https://arxiv.org/abs/2606.23565)

- Li et al. "WholeBodyVLA: Towards Unified Latent VLA for Whole-body Loco-manipulation Control," ICLR 2026. Unified latent VLA for humanoid loco-manipulation on Agibot X2. GitHub: [https://github.com/OpenDriveLab/WholebodyVLA](https://github.com/OpenDriveLab/WholebodyVLA)

## Confidence

0.88: The modular SLAM+VLA integration paradigm is well-established, with AnywhereVLA providing the most concrete architecture reference with empirical results (46% task success on unseen multi-room labs). The addition of memory-augmented VLA systems (MAP-VLA, EvoVLA, MemoryVLA) from late 2025 significantly strengthens the evidence base for how VLA policies can maintain spatial context — addressing the core "VLA models lack spatial memory" gap directly. The 3DGS-SLAM ecosystem (DynaGSLAM, SemGauss-SLAM, GSWorld, LEGS, Spectral GS-SLAM) now has multiple real-time systems with published results, elevating 3DGS from "promising trend" to "actively developed technology." VL-Nav's 30 Hz edge operation and 86.3% success rate provide concrete benchmarks for VLA navigation performance. Confidence increased to 0.88 from 0.87 due to: (1) **HoloAgent-0** (Jun 2026) demonstrating a unified embodied agent architecture where 3D spatial memory is embedded as a first-class component rather than an external SLAM module — this directly addresses the end-to-end integration gap; (2) **WholeBodyVLA** (ICLR 2026) showing that unified perception-action policies for large-space loco-manipulation are viable on real humanoids, validating the spatial scale required for SLAM+VLA integration; (3) Prior updates remain: GraphCoT-VLA, "From Pixels to Actions," Spectral GS-SLAM, DIM-WAM, μVLA, MUVLA, RoboMME. Still capped below 0.90 because: (1) continuous mapping during manipulation remains unsolved — no system maintains SLAM consistency while actively changing the scene; (2) HoloAgent-0's spatial memory layer lacks published real-robot empirical results; (3) real-world deployment beyond controlled lab settings is unproven; (4) single-device unified deployment of full SLAM+VLA+manipulation is not yet demonstrated; (5) 3DGS maps as direct VLA conditioning input remains unrealized despite promising prototypes.
