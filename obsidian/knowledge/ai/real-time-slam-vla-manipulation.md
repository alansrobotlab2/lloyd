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
updated: 2026-07-01
confidence: 0.88
---

# Real-time SLAM Integration with VLA Policies — Mapping During Manipulation Tasks

## Summary

Integrating simultaneous localization and mapping (SLAM) with Vision-Language-Action (VLA) policies enables robots to explore unknown environments and manipulate objects using only natural language instructions. The dominant paradigm remains a modular pipeline: classical SLAM provides spatial memory and map-building, active exploration discovers targets, and a fine-tuned VLA model handles grasping and manipulation. The field is converging around two complementary directions — **memory-augmented VLA architectures** (MAP-VLA, EvoVLA, DIM-WAM) that give policies persistent spatial context, and **3D Gaussian Splatting-based SLAM** (DynaGSLAM, GSWorld, Spectral GS-SLAM) maturing as the next-generation bridge between geometric mapping and semantic action. AnywhereVLA remains the clearest end-to-end reference, with newer systems (GeoVLA, HoloAgent-0, Scratchpad-Augmented VLAs) addressing the "spatial memory gap" through 3D geometry conditioning, unified spatial memory layers, and explicit evolving memory traces.

## Key Facts

- **AnywhereVLA (Gubernatorov et al., arXiv:2509.21006)** implements a full modular pipeline for large-scale indoor mobile manipulation in unseen environments. A single language instruction conditions both SLAM-based exploration and VLA manipulation. The system parses natural language into a task graph driving LiDAR-Inertial-Visual SLAM, metric semantic mapping, task-aware frontier exploration, approach planning, and a SmolVLA (450M parameters) manipulation head. Achieves 46% task success rate on multi-room labs, running on Jetson Orin NX (perception/VLA) + Intel NUC (SLAM/control) at real-time frequencies.

- **RSV-SLAM (arXiv:2510.02616, Oct 2025)** introduces real-time semantic RGBD SLAM specifically designed for dynamic indoor environments. Unlike traditional visual SLAM that assumes a static world, RSV-SLAM handles object motion and scene changes — a critical capability for SLAM systems operating during manipulation where gripper and arm motion introduce ego-motion artifacts. Evaluated on TUM RGB-D datasets with dynamic scenarios.

- **SlideSLAM (Liu et al., arXiv:2406.17249, T-RO 2025)** is a sparse, lightweight, decentralized metric-semantic SLAM system for multi-robot navigation. Key innovation: semantics-driven place recognition that leverages object-level metric-semantic maps for inter-robot loop closure detection. Demonstrates that semantic SLAM maps — the kind needed to bridge SLAM output to VLA input — can be maintained across distributed agents with lightweight communication.

- **CoT-VLA (Zhao et al., arXiv:2503.22020)** is a 7B VLA with visual chain-of-thought reasoning, achieving +17% real-world and +6% simulation improvement over prior VLA baselines on manipulation tasks. Pretrained with both robot demonstration data and action-less video data through intermediate visual reasoning steps. Addresses the visual reasoning gap that limits VLA performance on complex manipulation tasks downstream of SLAM-based navigation.

- **MAP-VLA (Li et al., arXiv:2511.09516, Nov 2025)** introduces Memory-Augmented Prompting for VLA models, empowering pre-trained VLAs with demonstration-derived memory prompts to augment action generation for long-horizon robotic manipulation. Addresses the core VLA spatial memory gap: rather than relying on SLAM to provide external maps, MAP-VLA embeds memory into the VLA's prompting mechanism, enabling dynamic recall of relevant task history during multi-step manipulation. Selected for ICRA 2026.

- **EvoVLA (arXiv:2511.16166, Nov 2025)** is a self-evolving VLA that mitigates long-horizon stage hallucination via self-supervised rewards, pose-grounded exploration, and selective memory. For tasks with 70+ steps, EvoVLA uses Context Selection to recall only critical history tokens needed for the current decision, preventing catastrophic forgetting. Achieves strong Sim2Real robustness on Discoverse-L benchmark, demonstrating that VLA models can maintain coherent task state over extended manipulation sequences without external SLAM conditioning.

- **VL-Nav (arXiv:2502.00931)** presents real-time vision-language navigation with spatial reasoning, operating at 30 Hz on Jetson Orin NX with an 86.3% success rate. Uses heuristic vision-language (HVL) spatial reasoning on both frontier-based and instance-based target points, with partial frontier detection on dynamic occupancy maps. Demonstrates that efficient spatial reasoning for navigation can run on low-power edge hardware alongside VLA inference.

- **TwinBrainVLA (arXiv:2601.14133)** addresses catastrophic forgetting via dual-stream architecture (frozen "Left Brain" for semantics, trainable "Right Brain" for embodied perception) but still lacks environment-scale spatial reasoning. Standard VLAs (π0, π0.5, OpenVLA) remain room-scale policies trained on localized demonstration data without persistent spatial memory mechanisms.

- **NaVILA (OpenReview: gkDRrvqeWF, RSS 2025)** proposes a two-level VLA framework for legged robot navigation. A high-level VLA generates language-based commands while a real-time locomotion policy handles obstacle avoidance. Demonstrates that VLA models can be adapted for Vision-and-Language Navigation (VLN) on legged platforms in cluttered scenes.

- **3D Gaussian Splatting for SLAM is maturing rapidly**: Multiple systems (DynaGSLAM, GS-SLAM, AG-SLAM, GSORBSLAM) demonstrate real-time 3DGS-SLAM with dynamic environment handling. GSWorld combines 3DGS with physics engines for closed-loop photorealistic sim-to-real simulation. LEGS (legsvla.github.io) uses 3DGS-based hybrid simulation for teleop-free VLA fine-tuning on humanoid robots (Unitree G1). SemGauss-SLAM (IROS 2025) adds semantic labeling to Gaussian SLAM, directly bridging the geometric-to-semantic gap for VLA input. MDGS-SLAM (arXiv, 2026) introduces real-time RGB-D Gaussian-SLAM with multi-view densification. Rad-GS (ICRA 2026) integrates radar-vision for outdoor 3DGS-SLAM.

- **MemoryVLA** (Guo et al., GitHub: G-U-O/memvla) proposes a Cognition-Memory-Action framework for long-horizon robotic manipulation, achieving +14.6% gain on Bridge and +11.8% on Mikasa-Robo benchmarks. Integrates with the Dexbotic VLA codebase.

- **GeoVLA (Aug 2025)** explicitly bridges 3D geometry to VLA action prediction. Uses a Point Embedding Network to convert depth maps into point cloud embeddings, concatenated with vision-language embeddings and processed by a 3D-enhanced Action Expert. Achieves state-of-the-art on LIBERO and ManiSkill2 benchmarks, demonstrating real-world robustness for height adaptability, scale awareness, and viewpoint invariance — capabilities directly relevant to SLAM+VLA integration where geometric context from maps conditions action prediction.

- **Semantic 3D mapping fuses SLAM with object detection**: AnywhereVLA constructs a 3D semantic object map by synchronizing RGB images, LiDAR point clouds, and 2D bounding-box detections. LiDAR points are projected into camera frames, voxelized, and associated with object detections via enlarged 2D bounding boxes. Per-class point clouds are clustered (DBSCAN) and summarized with centroid, covariance, and confidence estimates. This map supports task-aware active exploration conditioned on target object class.

- **Map consistency during manipulation remains unsolved**: Manipulation induces camera ego-motion (arm movements, gripper self-occlusion, scene changes) that corrupt SLAM estimates. Existing systems decouple navigation/exploration phases from manipulation phases. Continuous mapping during manipulation — where the robot must maintain spatial consistency while actively changing the scene — is an open research challenge. RSV-SLAM makes progress on dynamic scene handling but does not address manipulation-induced ego-motion specifically.

- **Edge deployment is feasible but requires split compute**: AnywhereVLA splits compute across two devices (Jetson Orin NX for VLA/perception, Intel NUC for SLAM/control). The full SLAM + semantic mapping + VLA inference + exploration planning pipeline at >10 Hz is resource-intensive. VL-Nav demonstrates 30 Hz operation on single Jetson Orin NX for navigation alone, but unified SLAM+VLA+manipulation on a single device remains open.

### Mid-2026 Developments

- **DIM-WAM (Wang et al., arXiv:2606.27677, Jun 2026)** is a memory-augmented world-action model for long-horizon robot manipulation. Augments a base world-action model with diverse historical event memory: extracting compact visual events from real observations, updating multiple memory banks through independent similarity-based merging, and reading bank-identity- and time-embedded long-term context to condition video and action denoising. On RMBench, raises average success from 28.4% (LingBot-VA baseline) to 69.8%. On real-world Franka tasks, improves full-task success from 52.5% to 80.0%. Directly demonstrates that structured historical event memory — a SLAM-adjacent capability — dramatically improves long-horizon manipulation.

- **μVLA (arXiv:2606.12497, Jun 2026)** is a controlled isolation study of recurrence in a strong pretrained VLA backbone. Learnable memory tokens carried across timesteps and trained with TBPTT improve manipulation under partial observability. Produces updated memory tokens recurrently passed to t+1, providing a lightweight alternative to external SLAM for short-horizon spatial reasoning under partial observability.

- **MUVLA (Han et al., arXiv:2509.25966, Sep 2025)** is a Map Understanding VLA tailored for object navigation. Takes current and history observations plus a semantic map as input and predicts action sequences based on a textual goal description. The three-stage training pipeline (map-level spatial understanding → behavior imitation → reward amplification) enables the model to unify diverse demonstrations into a robust spatial representation. Concrete SLAM-to-VLA bridge: semantic maps serve as structured spatial context for VLA action prediction.

- **RoboMME (arXiv:2603.04639, Mar 2026)** benchmarks memory for robotic generalist policies across 16 manipulation tasks spanning temporal, spatial, object, and procedural memory. Develops 14 memory-augmented VLA variants on the π0.5 backbone. MIKASA-Robo-VLA extends the benchmark to 90 environments covering 10 memory types, expanding the evaluation surface for memory-augmented VLA systems.

- **EgoHumanoid (arXiv:2602.10106, Feb 2026)** is the first framework for human-to-humanoid loco-manipulation transfer. By aligning egocentric human demonstrations with robot data through view transformation and unified action space, enables effective VLA co-training without requiring robot-specific demonstrations. Relevant to SLAM+VLA integration for loco-manipulation transfer.

- **HoloAgent-0 (arXiv:2606.23565, Jun 2026)** introduces a unified embodied agent framework with 3D spatial memory. The Memory Layer converts sensor streams into a persistent 3D world representation — including geometry, topology, occupancy, robot pose, and open-vocabulary semantic labels — that an Embodied AgentOS can query for grounding, localization, navigation, and manipulation. Unlike the modular SLAM+VLA split in AnywhereVLA, HoloAgent-0 treats spatial memory as a first-class component of the agent architecture. Represents a significant architectural shift: rather than SLAM feeding into VLA, spatial memory is embedded in the agent loop.

- **Scratchpad-Augmented VLAs (arXiv:2602.21013, Feb 2026)** generates and updates scratchpad representations stored as part of the input context for all subsequent steps, creating an explicit, evolving memory trace. This addresses the memory-dependent task problem where VLAs lose track of completed subgoals — a capability adjacent to SLAM-based spatial memory for long-horizon manipulation.

- **VLA-RAIL (Dec 2025)** introduces a real-time asynchronous inference linker that decouples model inference from robot motion control. Key components: Trajectory Smoother (polynomial fitting to filter chunk noise/jitter) and Chunk Fuser (position/velocity/acceleration continuity between successive action chunks). Critical infrastructure for SLAM+VLA deployment where continuous high-frequency control must coexist with slower VLA inference.

- **Spectral GS-SLAM (arXiv:2606.21258, Jun 2026)** introduces observability-aware, degeneracy-robust tracking for Gaussian Splatting SLAM. Uses second-order optimization with neural rendering for real-time SLAM across diverse indoor environments. Critical for SLAM+VLA integration: addresses tracking reliability under degeneracy conditions (textureless surfaces, repetitive patterns) common in manipulation environments.

- **CVPR 2026 3D-LLM/VLA Workshop** (2nd edition) signals field maturity with dedicated tracks on spatially-aware flow-matching for VLA reinforcement learning and physical consistency assessment — themes directly relevant to SLAM+VLA integration.

- **PatSnap Eureka Patent Report (2026)**: Documents VLA Foundation Models entering SLAM pipelines. Two 2026-active patents from Hefei Keda Intelligent Robot Technology Co. represent the most advanced publicly documented integration of VLA models into geometric SLAM — a signal of industrial interest in SLAM+VLA convergence.

### Late-2026 Developments

- **JanusVLN (arXiv:2509.22548, ICLR 2026)** introduces dual implicit memory for vision-language navigation. Uses a dual-encoder to separately extract visual-semantic and spatial-geometric features from RGB-only video, constructing fixed-size implicit memory that is incrementally updated during navigation. Decouples semantics from spatiality — directly addressing the representation incompatibility challenge between SLAM maps (geometric) and VLA context (semantic). Improves navigation efficiency by processing language instructions and spatial information in parallel.

- **GST-VLA (Sarowar et al., arXiv:2603.09079, Mar 2026)** introduces Structured Gaussian Spatial Tokens for 3D depth-aware VLA models. Replaces dense scalar depth streams with anisotropic 3D Gaussian tokens from depth and semantic features, concentrating on salient geometric regions to yield metric-aware spatial tokens. **This is the first system to directly feed Gaussian spatial representations as VLA input conditioning** — addressing the long-standing gap of "3DGS as VLA input" identified in earlier surveys. Makes the spatial reasoning pathway from depth tokens to action tokens explicit rather than implicit.

- **VILAS (arXiv:2605.02037, May 2026)** designs a fully integrated low-cost robotic manipulation platform supporting the complete VLA workflow: teleoperation-based data collection, policy fine-tuning, and real-time deployment using modular, affordable hardware. Demonstrates that the full VLA pipeline can operate on accessible hardware — a prerequisite for SLAM+VLA co-deployment.

- **DiskChunGS (2026)** introduces large-scale 3D Gaussian SLAM through chunk-based memory management, enabling scalable spatial representations for VLA integration. Addresses the memory/compute bottleneck of maintaining Gaussian maps at environment scale.

## Late 2026 Additions

- **SG-VLA (CVPR 2026, arXiv:2603.22760)** learns spatially-grounded VLA models for mobile manipulation through auxiliary task co-training and multi-modal input enhancement. Achieves 73% task success rate on household manipulation, directly addressing spatial generalization — a known weakness of standard VLAs where object positions outside training distributions cause failure.

- **SA-VLA** introduces Spatially-Aware Flow-Matching for VLA reinforcement learning. Fuses implicit spatial representations with visual tokens into geometry-aware embeddings, optimized via step-level dense rewards and SCAN (spatially-conditioned annealed exploration). Bridges spatial conditioning into flow-matching policy optimization — a direction complementary to SLAM-based spatial grounding.

- **RoboMemory (arXiv:2508.01415, updated Mar 2026)** is a brain-inspired multi-memory agentic framework for interactive environmental learning. Implements dynamic spatial memory update algorithms as appendices (Proof D), demonstrating brain-inspired hierarchical memory architectures for embodied systems. Relevant to how VLA policies could integrate SLAM-derived spatial context through multi-memory mechanisms rather than raw map conditioning.

- **3D Latent Mapping for Mobile Manipulation (ICRA 2026, OpenReview j0hzlSl1R9)** introduces a 3D feature grid that continuously integrates new multi-view observations as a spatial memory, enabling maps to accumulate spatially and temporally extended context. Directly addresses the SLAM-VLA bridge: 3D maps act as spatial memory for mobile manipulation policies, mitigating occlusions from the current field of view.

- **OpenSPM (arXiv:2606.29936, Jun 2026)** is an environment-transferable robotic key spatial pose framework covering VLAs, 3D perception and spatial manipulation representations, demonstration memory, and diffusion/flow-matching policies. Reviews the full landscape of how spatial representations flow from perception (including SLAM) through policy conditioning.

- **SpatialVLA (GitHub: SpatialVLA/SpatialVLA)** is a spatial-enhanced VLA trained on 1.1M real robot episodes with Ego3D Position Encoding and Adaptive Action Grids. Demonstrates that 3D spatial awareness can be baked into VLA architectures — reducing reliance on external SLAM for basic spatial reasoning while complementing SLAM-derived global context.

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

4. **3DGS as VLA input representation**: **Partially resolved by GST-VLA** (Sarowar et al., arXiv:2603.09079), which introduces anisotropic 3D Gaussian tokens as structured spatial conditioning for VLA models. GST-VLA replaces dense scalar depth with metric-aware Gaussian spatial tokens concentrated on salient geometric regions. SemGauss-SLAM and GSWorld show further promise, but broad integration of 3DGS maps as VLA input remains an active research frontier.

5. **Memory-augmented vs. SLAM-conditioned architectures**: MAP-VLA and EvoVLA show that VLA models can maintain their own memory. Does this make external SLAM redundant for short-horizon tasks, or does SLAM remain essential for environment-scale spatial reasoning that exceeds VLA context windows?

6. **Cross-environment generalization**: Systems like AnywhereVLA were evaluated in controlled multi-room labs. How do they perform in truly unstructured, human-populated environments with moving obstacles and dynamic scene changes?

7. **Unified edge deployment**: Can the full SLAM + semantic mapping + VLA pipeline fit on a single consumer-grade edge device? VL-Nav shows 30 Hz navigation on Orin NX, but adding manipulation and SLAM to the same device is untested.

8. **Active perception loops**: Can VLA policies actively control camera and sensor configuration to improve SLAM quality — e.g., choosing viewpoints that reduce localization uncertainty while simultaneously gathering data useful for manipulation planning?

9. **Multi-robot SLAM for VLA coordination**: If SlideSLAM-style semantic SLAM enables inter-robot map sharing, can multiple mobile manipulators coordinate exploration and manipulation using shared semantic maps?

10. **Sim2Real for SLAM+VLA**: LEGS demonstrates Gaussian-splatting-based sim2real for loco-manipulation. Can similar approaches be applied to SLAM+VLA integration, using photorealistic Gaussian scenes as training environments for joint navigation-manipulation policies?

## Sources

- Gubernatorov et al. "AnywhereVLA: Language-Conditioned Exploration and Mobile Manipulation," arXiv:2509.21006, Sep 2025. [https://arxiv.org/abs/2509.21006](https://arxiv.org/abs/2509.21006)

- Li et al. "MAP-VLA: Memory-Augmented Prompting for Vision-Language-Action Model in Robotic Manipulation," arXiv:2511.09516, Nov 2025. [https://arxiv.org/abs/2511.09516](https://arxiv.org/abs/2511.09516)

- "EvoVLA: Self-Evolving Vision-Language-Action Model," arXiv:2511.16166, Nov 2025. [https://arxiv.org/abs/2511.16166](https://arxiv.org/abs/2511.16166)

- "VL-Nav: Real-time Vision-Language Navigation with Spatial Reasoning," arXiv:2502.00931. [https://arxiv.org/abs/2502.00931](https://arxiv.org/abs/2502.00931)

- "LEGS: Fine-Tuning Teleop-Free VLAs for Humanoid Loco-Manipulation," legsvla.github.io.

- RSV-SLAM authors. "RSV-SLAM: Toward Real-Time Semantic Visual SLAM in Indoor Dynamic Environments," arXiv:2510.02616, Oct 2025. [https://arxiv.org/abs/2510.02616](https://arxiv.org/abs/2510.02616)

- Liu et al. "SlideSLAM: Sparse, Lightweight, Decentralized Metric-Semantic SLAM for Multi-Robot Navigation," arXiv:2406.17249, T-RO 2025. [https://arxiv.org/abs/2406.17249](https://arxiv.org/abs/2406.17249)

- Zhao et al. "CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models," arXiv:2503.22020, Mar 2025. [https://arxiv.org/abs/2503.22020](https://arxiv.org/abs/2503.22020)

- Cheng et al. "NaVILA: Legged Robot Vision-Language-Action Model for Navigation," OpenReview (gkDRrvqeWF), RSS 2025. [https://openreview.net/forum?id=gkDRrvqeWF](https://openreview.net/forum?id=gkDRrvqeWF)

- Yu et al. "TwinBrainVLA: Unleashing the Potential of Generalist VLMs for Embodied Tasks via Asymmetric Mixture-of-Transformers," arXiv:2601.14133, 2026. [https://arxiv.org/abs/2601.14133](https://arxiv.org/abs/2601.14133)

- "SemGauss-SLAM: Dense Semantic Gaussian Splatting SLAM," IROS 2025.

- GSWorld. "GSWorld: Photo-Realistic Closed-Loop Simulator for Robotic Manipulation," 2025.

- Guo et al. "MemoryVLA: Cognition-Memory-Action Framework." GitHub: G-U-O/memvla, 2025.

- Kim et al. "π0: A Vision-Language-Action Flow Model for General Robot Control," arXiv:2410.24164.

- 3DGS-SLAM survey. "How NeRFs and 3D Gaussian Splatting are Reshaping SLAM," arXiv:2402.13255.

- Wang et al. "DIM-WAM: World-Action Modeling with Diverse Historical Event Memory," arXiv:2606.27677, Jun 2026. [https://wangkai-casia.github.io/dim-wam/](https://wangkai-casia.github.io/dim-wam/)

- "μVLA: On Recurrent Memory for Partially Observable Manipulation," arXiv:2606.12497, Jun 2026. [https://github.com/CognitiveAISystems/muVLA](https://github.com/CognitiveAISystems/muVLA)

- Han et al. "MUVLA: Learning to Explore Object Navigation via Map Understanding," arXiv:2509.25966, Sep 2025. [https://arxiv.org/abs/2509.25966](https://arxiv.org/abs/2509.25966)

- "RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies," arXiv:2603.04639, Mar 2026.

- "EgoHumanoid: Unlocking In-the-Wild Loco-Manipulation with Robot-Free Egocentric Demonstrations," arXiv:2602.10106, Feb 2026.

- Huang et al. "GraphCoT-VLA: A 3D Spatial-Aware Reasoning Vision-Language-Action Model for Robotic Manipulation with Ambiguous Instructions," arXiv:2508.07650, Aug 2025. [https://arxiv.org/abs/2508.07650](https://arxiv.org/abs/2508.07650)

- Sandipan D. Asan. "From Pixels to Actions: The Hidden Role of SLAM in VLA Model Training and Evaluation," 2025. [https://mrsandipandas.github.io/files/slam-vla.pdf](https://mrsandipandas.github.io/files/slam-vla.pdf)

- "Spectral GS-SLAM: Observability-Aware, Degeneracy-Robust Tracking for Gaussian Splatting SLAM," arXiv:2606.21258, Jun 2026. [https://arxiv.org/pdf/2606.21258](https://arxiv.org/pdf/2606.21258)

- Zhou et al. "HoloAgent-0: A Unified Embodied Agent Framework with 3D Spatial Memory," arXiv:2606.23565, Jun 2026. [https://arxiv.org/abs/2606.23565](https://arxiv.org/abs/2606.23565)

- Li et al. "WholeBodyVLA: Towards Unified Latent VLA for Whole-body Loco-manipulation Control," ICLR 2026. [https://github.com/OpenDriveLab/WholebodyVLA](https://github.com/OpenDriveLab/WholebodyVLA)

- GeoVLA authors. "GeoVLA: Empowering 3D Representations in Vision-Language-Action Models," arXiv, Aug 2025. VLA framework integrating 3D point cloud embeddings for spatial manipulation.

- "Scratchpad-Augmented VLAs for Memory Dependent Tasks," arXiv:2602.21013, Feb 2026. [https://arxiv.org/html/2602.21013](https://arxiv.org/html/2602.21013)

- "VLA-RAIL: A Real-Time Asynchronous Inference Linker for VLA Models and Robots," arXiv, Dec 2025.

- "AirVLA: Physics-Guided Transfer of VLA Models to Aerial Manipulation," arXiv, Mar 2026. Gaussian Splatting pipeline for aerial VLA training data synthesis.

- CVPR 2026 3D-LLM/VLA Workshop (2nd edition). [https://3d-llm-vla.github.io/](https://3d-llm-vla.github.io/)

- PatSnap Eureka. "Robot SLAM Technology Landscape 2026." VLA Foundation Models entering SLAM Pipelines. [https://www.patsnap.com/resources/blog/rd-blog/robot-slam-technology-landscape-2026-patsnap-eureka/](https://www.patsnap.com/resources/blog/rd-blog/robot-slam-technology-landscape-2026-patsnap-eureka/)

- "MDGS-SLAM: Real-time RGB-D Gaussian-SLAM with Multi-view Densification," arXiv, 2026.

- "Rad-GS: Radar-Vision Integration for 3D Gaussian Splatting SLAM in Outdoor Environments," ICRA 2026.

- Google. "ST4VLA: Spatial Guidance for Better Robot Actions," 2026.

- "Awesome VLA Benchmarks." GitHub: JFan5/awesome-vla-benchmarks. Curated list including memory-augmented VLA evaluation, spatial reasoning, and safety benchmarks. [https://github.com/JFan5/awesome-vla-benchmarks](https://github.com/JFan5/awesome-vla-benchmarks)

- "MIKASA-Robo-VLA: Extending the MIKASA-Robo Memory Benchmark to VLA Research," ICLR 2026. Expands MIKASA-Robo from 32 to 90 environments covering 10 memory types. [https://github.com/CognitiveAISystems/MIKASA-Robo](https://github.com/CognitiveAISystems/MIKASA-Robo)

- "JanusVLN: Decoupling Semantics and Spatiality with Dual Implicit Memory for Vision-Language Navigation," arXiv:2509.22548, ICLR 2026. [https://miv-xjtu.github.io/JanusVLN.github.io/](https://miv-xjtu.github.io/JanusVLN.github.io/)

- Sarowar et al. "GST-VLA: Structured Gaussian Spatial Tokens for 3D Depth-Aware Vision-Language-Action Models," arXiv:2603.09079, Mar 2026. [https://arxiv.org/abs/2603.09079](https://arxiv.org/abs/2603.09079)

- "VILAS: A VLA-Integrated Low-cost Architecture with Soft Grasping," arXiv:2605.02037, May 2026.

- "DiskChunGS: Large-Scale 3D Gaussian SLAM Through Chunk-Based Memory Management," 2026.

- Tu et al. "SG-VLA: Learning Spatially-Grounded Vision-Language-Action Models for Mobile Manipulation," CVPR 2026, arXiv:2603.22760, Mar 2026. [https://arxiv.org/abs/2603.22760](https://arxiv.org/abs/2603.22760)

- Xu Pan. "SA-VLA: Spatially-Aware Flow-Matching for Vision-Language-Action Reinforcement Learning." [https://xupan.top/Projects/savla/](https://xupan.top/Projects/savla/)

- "RoboMemory: A Brain-Inspired Multi-memory Agentic Framework for Interactive Environmental Learning," arXiv:2508.01415v7, updated Mar 2026. [https://arxiv.org/abs/2508.01415](https://arxiv.org/abs/2508.01415)

- Kim et al. "Seeing the Bigger Picture: 3D Latent Mapping for Mobile Manipulation Policy Learning," ICRA 2026, OpenReview. [https://openreview.net/pdf?id=j0hzlSl1R9](https://openreview.net/pdf?id=j0hzlSl1R9)

- "OpenSPM: An Environment-Transferable Robotic Key Spatial Pose Method," arXiv:2606.29936, Jun 2026. [https://arxiv.org/abs/2606.29936](https://arxiv.org/abs/2606.29936)

- "SpatialVLA: A Spatial-Enhanced Vision-Language-Action Model." GitHub: SpatialVLA/SpatialVLA. [https://github.com/SpatialVLA/SpatialVLA](https://github.com/SpatialVLA/SpatialVLA)

## Confidence

0.88: The modular SLAM+VLA integration paradigm is well-established, with AnywhereVLA providing the most concrete architecture reference with empirical results (46% task success on unseen multi-room labs). The addition of memory-augmented VLA systems (MAP-VLA, EvoVLA, MemoryVLA, DIM-WAM, μVLA) from late 2025–mid 2026 significantly strengthens the evidence base for how VLA policies can maintain spatial context — addressing the core "VLA models lack spatial memory" gap directly. The 3DGS-SLAM ecosystem (DynaGSLAM, SemGauss-SLAM, GSWorld, LEGS, Spectral GS-SLAM, MDGS-SLAM, Rad-GS) now has multiple real-time systems with published results, elevating 3DGS from "promising trend" to "actively developed technology." GeoVLA directly bridges 3D geometry conditioning into VLA action prediction. HoloAgent-0 demonstrates a unified embodied agent architecture where 3D spatial memory is embedded as a first-class component. Scratchpad-Augmented VLAs show explicit evolving memory traces for memory-dependent manipulation. SG-VLA (CVPR 2026) achieves 73% success on household manipulation with spatial grounding, and SA-VLA bridges spatial conditioning into flow-matching policy optimization. SpatialVLA demonstrates 3D spatial awareness baked into VLA architectures via Ego3D Position Encoding. RoboMemory and 3D Latent Mapping show complementary approaches to spatial memory for embodied agents. The CVPR 2026 3D-LLM/VLA workshop signals field maturity, and PatSnap's patent report documents industrial interest in VLA-SLAM convergence.

Still capped below 0.90 because: (1) continuous mapping during manipulation remains unsolved — no system maintains SLAM consistency while actively changing the scene; (2) HoloAgent-0's spatial memory layer lacks published real-robot empirical results; (3) real-world deployment beyond controlled lab settings is unproven; (4) single-device unified deployment of full SLAM+VLA+manipulation is not yet demonstrated; (5) 3DGS maps as direct VLA conditioning input remains unrealized despite promising prototypes; (6) the patent landscape (Hefei Keda) suggests industrial activity in VLA-SLAM convergence, but no open-source implementations from that work exist yet.