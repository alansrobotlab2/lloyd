[updated 2025-07-18, refreshed 2026-06-07, extended 2026-07-11, 2026-07-14, major update 2026-07-16, OmniRobotHome update 2026-07-27, fused consolidation 2026-07-28, PosA-VLA + PointACT update 2026-08-04, InCoM + 3D HAMSTER + MobileManiBench update 2026-09-10]

# Real-Time VLM-Based Perception for Mobile Manipulation: Fused Depth-Pose-Segmentation Pipeline

## Summary

Real-time VLM-based perception for mobile manipulation fuses monocular depth estimation, 6D object pose estimation, and semantic/instance segmentation into a unified pipeline that feeds Vision-Language-Action (VLA) models. Rather than running depth, pose, and segmentation as independent modules, modern approaches co-design these capabilities so the VLM backbone shares feature representations across tasks, enabling joint reasoning over geometry, semantics, and action. FALCON injects spatial tokens into a specialized action head without contaminating the VLM's pre-trained semantic space, while G^3VLA introduces geometric inductive bias via intrinsic-conditioned ray embeddings and bidirectional cross-view fusion. The fused pipeline must operate at 20–50 Hz to support closed-loop visuomotor control, with monocular depth (Depth Anything V2) providing metric geometry from single-camera inputs to avoid LiDAR dependencies.

## Key Facts

- **Fused architecture over modular stacking**: FALCON injects spatial tokens into a dedicated action head rather than the VLM backbone, avoiding degradation of pre-trained semantic representations [1]. OpenVLA fuses DINOv2 (spatial) + SigLIP (semantic) in a shared transformer for generalist manipulation [2].

- **Monocular depth as the geometry backbone**: Depth Anything V2 (Small) produces dense depth maps at 50 FPS on RTX 3090 and 18 FPS on Jetson Orin NX with INT8 quantization. Depth-augmented training improves grasp success by 12–18% in clutter [3].

- **6D pose for zero-shot grasp planning**: VLM-based approaches (Horyon) estimate relative pose of unseen objects from textual prompts alone. Event-camera methods (EventTrack6D) achieve microsecond latency for dynamic scene tracking [4].

- **Semantic segmentation as the grounding layer**: Open-vocabulary segmentation (Grounding DINO + SAM) generates instance masks conditioned on language instructions, guiding depth and pose modules to focus on task-relevant objects [5].

- **Spatial chain-of-thought (Perceptio)**: Perceptio generates explicit 2D segmentation and 3D depth tokens as an intermediate "spatial chain-of-thought" before text output, achieving SOTA on RefCOCO/+/g and +10.3% accuracy on HardBLINK depth reasoning. It uses a frozen SAM2 encoder for 2D cues and a VQ-VAE depth codebook distilled from Depth Anything V2 [7].

- **Knowledge-insulated VLAs (CogVLA)**: CogVLA (NeurIPS 2025) uses instruction-driven vision sparsification and cognition-aligned routing to reduce post-training overhead. A companion "Knowledge Insulating" technique (arXiv:2505.23705) ensures VLAs train fast, run fast, and generalize better by preserving the VLM backbone during action fine-tuning [6].

- **Geometric inductive bias (G^3VLA)**: G^3VLA (arXiv:2606.24472) injects geometric inductive bias into VLA visual tokens via intrinsic-conditioned ray embeddings (K⁻¹) and bidirectional cross-view fusion with PRoPE, leaving the pretrained backbone and action objective unchanged [13].

- **VLM physical priors for sim-to-real (Phys2Real)**: Phys2Real combines VLM-inferred physical parameter estimates (center of mass, friction) with interactive online adaptation via uncertainty-aware fusion. It achieves 100% success on T-block pushing vs. 79% for domain randomization baseline, demonstrating VLM priors improve closed-loop manipulation beyond high-level planning [8].

- **Multi-agent VLM perception (GoalVLM)**: GoalVLM coordinates multi-agent exploration using SAM3 for text-prompted detection and shared fused semantic maps for global scene awareness, enabling zero-shot perception pipelines across distributed agents [9].

- **Open-source VLA at scale (MolmoAct2)**: Allen Institute's MolmoAct2 achieves 87.1% success on unseen real-world manipulation tasks with a 37x speedup over its predecessor. Released with the 720-hour MolmoAct 2-Bimanual YAM dataset, it demonstrates that open-source VLA pipelines can rival closed-system performance [14].

- **Dual-process real-time architecture (IROS)**: IROS combines VLM-level contextual reasoning with lightweight perceptual modules on low-cost, on-device hardware for real-time indoor navigation, demonstrating that high-level VLM reasoning can run asynchronously alongside fast perception loops [10].

- **Asymmetric dual-stream VLA (TwinBrainVLA)**: TwinBrainVLA (arXiv:2601.14133) resolves the catastrophic forgetting problem by splitting perception into a frozen "Left Brain" (general semantic understanding) and a trainable "Right Brain" (embodied perception) via Asymmetric Mixture-of-Transformers (AsyMoT). The Right Brain dynamically queries semantic knowledge from the frozen Left Brain and fuses it with proprioceptive states for a Flow-Matching Action Expert, achieving superior performance on SimplerEnv and RoboCasa while preserving open-world VLM capabilities [15].

- **Masked Depth Modeling (LingBot-VLA + LingBot-Depth)**: Ant Group's LingBot-VLA integrates LingBot-Depth, a self-supervised spatial perception model based on Masked Depth Modeling. LingBot-Depth uses self-supervised pre-training via depth reconstruction and cross-modal attention for joint RGB-Depth alignment in a unified latent space, preserving metric-scale measurements. On the GM-100 benchmark, LingBot-VLA with depth achieves state-of-the-art averages, and the model transfers across diverse robot platforms (Galaxea Dynamics, AgileX Robotics) [16].

- **Declarative memory for long-horizon manipulation (EchoVLA)**: EchoVLA (arXiv:2511.18112) introduces synergistic declarative memory mechanisms for long-horizon mobile manipulation, extending VLA models beyond short-horizon table-top manipulation. It enables embodied agents to maintain persistent scene understanding and memory-aware reasoning across extended navigation and manipulation sequences [17].

- **World Action Models survey (WAMs)**: The WAM survey (arXiv:2606.20781) catalogs modular Render-and-Decode architectures that project depth, pose, and segmentation into semantic, depth, and flow latent-action codebooks before action decoding. The shared insight is that rendered future states serve as a common planning currency across perception modules, unifying depth estimation, pose tracking, and segmentation under a single world-model framework [18].

- **Implicit depth enhancement (Evo-Depth)**: Evo-Depth (arXiv:2605.14950) introduces a lightweight Implicit Depth Encoding Module (IDEM) that extracts compact depth features from multi-view RGB images without requiring explicit 3D sensors. Depth features are incorporated into vision-language representations via a Spatial Enhancement Module (SEM) using depth-aware modulation, enabling efficient spatial-semantic enhancement. The model is lightweight enough for Jetson Orin edge deployment and improves spatially grounded manipulation by implicitly reasoning about depth from RGB alone [19].

- **Universal pose pretraining (Pose-VLA)**: Pose-VLA (arXiv:2602.19710) decouples VLA training into two stages: (1) pre-training on 1.4M images with 6.5M 3D annotations to extract universal spatial priors in a unified camera-centric space, and (2) lightweight post-training for embodiment alignment. It introduces discrete Pose Tokens encoding SE(3) transformations that serve as a universal interface between non-robotic 3D datasets and robotic demonstrations. Pose-VLA achieves 79.5% average success on RoboTwin 2.0 and 96.0% on LIBERO, with real-world experiments showing robust generalization across rigid, articulated, and deformable objects using only 100 demonstrations per task [20].

- **Vision-agnostic pretraining (LA4VLA)**: LA4VLA (arXiv:2606.27295) proposes a vision-agnostic language-action pretraining framework where models learn from language instructions, proprioceptive states, and action trajectories without visual inputs. This "act without seeing" paradigm decouples action policy learning from visual perception, enabling the policy to be composited with any perception front-end (depth, segmentation, pose) at inference time — directly supporting plug-and-play fused perception pipelines [21].

- **Phase-aware MoE action experts (PAMAE)**: PAMAE (arXiv:2606.27144) replaces the shared action expert in VLA models with a sparse mixture of phase-aware action experts, preserving the pretrained VLA backbone and flow-matching interface. Phase-aware experts specialize in different manipulation phases (approach, grasp, place) and are routed dynamically, improving reliability for long-horizon mobile manipulation tasks [22].

- **Pose-conditioned anchor attention (PosA-VLA)**: PosA-VLA (arXiv:2512.03724) addresses VLA action inconsistency rooted in spatially uniform perception fields. It anchors visual attention via pose-conditioned supervision — generating two complementary anchors (task-relevant anchor at interaction moments, end-effector anchor at each timestep) that break uniform perception and dynamically maintain spatial correspondence between robot pose and visual scene. Operates on lightweight backbones without auxiliary perception modules (no segmentation or grounding networks), achieving faster, smoother trajectories with fewer action steps. Demonstrates strong generalization across background/lighting/distractor changes [29].

- **Multi-scale point-action interaction (PointACT)**: PointACT (arXiv:2605.21414, RSS 2026) proposes a dual-system 3D-aware VLA policy integrating hierarchical 3D point cloud representations directly into action decoding. It uses a frozen VLM backbone for semantic encoding, incorporates point tokens into a dedicated point-action expert, and applies multi-scale point-action attention with bottleneck window self-attention. Point tokens evolve to densely attend to both local geometric detail and global scene structure, providing explicit depth-pose-segmentation fusion via 3D point representations [30].

- **Real-world fused perception testbed (OmniRobotHome)**: OmniRobotHome (arXiv:2604.28197) provides a home-scale testbed with 48 hardware-synchronized cameras and 3 manipulators in a unified world frame. The stereo manipulation pipeline combines depth estimation, FoundationPose-based 6D tracking, and detector-based mask acquisition running at ~16 Hz — sufficient for closed-loop tracking of manipulated objects. The platform treats perception quality as an experimental variable, demonstrating that interaction quality degrades measurably as real-timeness, granularity, coverage, accuracy, forecasting, or memory is weakened [23].

- **Geometry-grounded VLMs (G2VLM)**: G2VLM (InternRobotics, CVPR 2026) represents a shift from treating geometry as an auxiliary input to making it a core grounding modality — the model jointly reasons about "what is this object" and "where is it in 3D space" through interleaved spatial-semantic reasoning, predicting 3D geometry (depth, structure-from-motion) alongside spatial reasoning tasks [24].

- **3D Gaussian Splatting as unified perception representation**: SemGauss-SLAM (IROS 2025) merges depth, pose, and segmentation into a single dense 3D scene representation by adding semantic labels to Gaussian Splatting maps — producing a unified structure that simultaneously encodes geometry (depth proxy), object locations (pose proxy), and semantics (segmentation). GSWorld extends this to closed-loop sim-to-real training. Feeding 3DGS as VLA conditioning remains untested for real-robot manipulation [25].

- **Mobile manipulation at scale (AnywhereVLA)**: AnywhereVLA (arXiv:2509.21006) demonstrates a real-time mobile manipulation architecture by splitting perception/VLA (SmolVLA, 450M params) on Jetson Orin NX and SLAM/control on an Intel NUC, achieving multi-room mobile manipulation at real-time frequencies — the closest published architecture to a production fused pipeline [26].

- **Memory-augmented VLA prompting (MAP-VLA + MemoryVLA)**: MAP-VLA (arXiv:2511.09516, ICRA 2026) and MemoryVLA introduce cognition-memory-action frameworks that augment VLA perception with persistent memory, enabling the fused pipeline to accumulate and reuse perception knowledge across manipulation episodes — addressing the single-shot limitation of current perception-to-action loops [27].

## Details

### Pipeline Architecture

The fused perception pipeline for mobile manipulation typically layers three perception modules:

1. **Depth Estimation**: Monocular depth (Depth Anything V2, MiDaS) or stereo/ToF sensors provide metric geometry. The Small variant of Depth Anything V2 is the most practical for real-time edge deployment, achieving 50 FPS on desktop GPUs and 18 FPS on Jetson Orin NX with INT8 quantization. Mean absolute error of 4.2 cm at 1-meter distance is sufficient for tabletop manipulation.

2. **6D Pose Estimation**: Methods range from model-based (PVNet, PVN3D) to model-free approaches (Any6D) that estimate pose of novel objects without CAD models. VLM-based methods like Horyon estimate relative pose from textual descriptions, enabling zero-shot adaptation to unseen objects.

3. **Semantic Segmentation**: VLM-provided open-vocabulary segmentation (e.g., Grounding DINO + SAM) generates instance masks conditioned on language instructions, enabling the depth and pose modules to focus on task-relevant objects.

### Fused VLA Integration

VLA models like OpenVLA (7B params, trained on 970k robot episodes) demonstrate that fusing spatial (DINOv2) and semantic (SigLIP) features in a shared transformer produces strong generalist policies. OpenVLA processes depth maps as additional input channels alongside RGB, improving spatial reasoning for overlapping objects.

FALCON's paradigm — injecting spatial tokens into a specialized action head — is the key architectural insight: keeping the VLM backbone pure preserves semantic reasoning while the action head handles geometric precision.

CogVLA extends this with instruction-driven vision sparsification, routing only task-relevant visual features through the VLA backbone to reduce compute overhead while maintaining performance.

G^3VLA takes a complementary approach: rather than modifying the backbone or action head, it injects geometric inductive bias via camera intrinsics-conditioned ray embeddings and bidirectional cross-view fusion, providing explicit geometric grounding without altering pretrained representations.

### Spatial Token Generation (Perceptio)

Perceptio treats spatial perception as a first-class citizen of the language objective: the model generates [seg tokens] + [depth tokens] + [text tokens] in sequence, forcing depth and segmentation reasoning before answering. Key innovations include soft-merging VQ-VAE codebook embeddings (differentiable depth token generation) and a composite depth-token loss (marker + token + count). Ablation shows removing depth tokens collapses HardBLINK accuracy by 25.8%, but removing depth slightly improves general VQA — revealing a capacity trade-off.

### Asymmetric Dual-Stream VLAs

TwinBrainVLA resolves the training conflict in single-backbone VLAs (where fine-tuning for embodied control causes catastrophic forgetting of general semantic knowledge) by using two isomorphic VLM pathways:
- **Frozen Left Brain**: Retains open-world reasoning and instruction-following from pre-training.
- **Trainable Right Brain**: Specialized for embodied perception and proprioceptive state integration.
- **AsyMoT fusion**: Asymmetric Mixture-of-Transformers enables joint attention over hidden states without parameter sharing, letting the Right Brain query semantic knowledge from the Left Brain.
- **Flow-matching action expert**: The Right Brain's spatially rich embeddings condition a flow-matching policy for continuous action generation.

### Self-Supervised Depth Fusion

LingBot-VLA + LingBot-Depth demonstrates that self-supervised depth modeling can be directly fused into the VLA pipeline:
- **Masked Depth Modeling**: Self-supervised pre-training via depth reconstruction (similar to MAE-style masking)
- **Cross-modal attention**: Joint RGB-Depth alignment in a unified latent space
- **Metric-scale preservation**: Maintains real-world depth measurements for downstream tasks
- Cross-morphology transfer: Successfully adapted to robots from Galaxea Dynamics and AgileX Robotics

### Memory-Augmented Perception

EchoVLA addresses the limitations of short-horizon VLA models for mobile manipulation by introducing synergistic declarative memory:
- Maintains persistent scene understanding across extended navigation sequences
- Memory-aware reasoning enables long-horizon task planning without state explosion
- Extends the fused perception paradigm from single-task manipulation to continuous mobile manipulation

### World Action Models Framework

The WAM survey (arXiv:2606.20781) provides a unifying framework for fused perception:
- Modular Render-and-Decode architectures project depth, pose, and segmentation into semantic, depth, and flow latent-action codebooks before action decoding
- Rendered future states serve as a common planning currency across perception modules
- Demonstrates that depth estimation, pose tracking, and segmentation can be unified under a single world-model framework

### Real-Time Constraints

Mobile manipulation requires the full perception-to-action pipeline to close at 20–50 Hz:
- Monocular depth inference: 50 FPS (Small), 8 FPS (Giant) on RTX 3090
- VLA policy inference: ~10–20 Hz for 7B-parameter models on edge hardware
- Segmentation + pose: typically batched with depth for feature sharing
- Onboard GPUs alone cannot support the full foundation-model-based mobile manipulation stack — hybrid edge-cloud or tiered architectures are needed

IROS demonstrates a dual-process architecture where the VLM runs asynchronously for high-level reasoning while a fast perception loop handles real-time navigation, showing a practical path to real-time operation on low-cost hardware.

No standardized benchmark exists for "perception pipeline latency → manipulation success rate" on mobile manipulation platforms, making system-level evaluation difficult.

### Implicit Depth from RGB (Evo-Depth)

Evo-Depth eliminates the need for explicit depth sensors by extracting implicit depth features directly from multi-view RGB via a lightweight IDEM (Implicit Depth Encoding Module). The depth features are fused into the VLM via a Spatial Enhancement Module (SEM) that modulates vision-language features with depth-aware gating. Key advantages:
- **No additional hardware**: Works with standard RGB cameras, avoiding LiDAR/ToF dependencies
- **Edge deployable**: Lightweight enough for Jetson Orin-class hardware
- **Spatial-semantic coupling**: Depth cues directly modulate VL representations, improving grasp accuracy for overlapping objects

### Universal Pose Pretraining (Pose-VLA)

Pose-VLA resolves the fundamental misalignment between VLM pretraining (semantic/categorical) and robotic action needs (fine-grained 3D states) via a two-stage pipeline:

**Stage 1 — Spatial foundation pre-training**: Trains on 1.4M images with 6.5M 3D annotations using discrete pose tokens representing SE(3) transformations in a camera-centric frame. RGB images are paired with depth maps and camera intrinsics encoded as raymaps, providing intrinsic 3D awareness.

**Stage 2 — Embodiment alignment**: Lightweight fine-tuning maps spatial priors to robot-specific action spaces. The unified pose token format allows seamless ingestion of both non-robotic 3D datasets and robotic demonstrations.

**Key insight**: Pose tokens serve as the universal geometric primitive — in static contexts they localize objects (spatial grounding), in temporal sequences they characterize motion trajectories (motion estimation).

### Vision-Agnostic Pretraining (LA4VLA)

LA4VLA inverts the standard VLA paradigm: instead of training on vision + language + action jointly, it learns language-action mappings without visual inputs. This creates a perception-agnostic policy that can be composed with any perception front-end at inference time:
- Enables plug-and-play integration of depth, pose, and segmentation modules
- Decouples action policy learning from visual perception quality
- Supports hot-swapping perception pipelines without retraining the policy

### Phase-Aware Mixture-of-Experts (PAMAE)

PAMAE improves long-horizon mobile manipulation reliability by replacing the monolithic action expert with a sparse mixture of phase-aware experts:
- Separate experts specialize in approach, grasp, and place phases
- Dynamic routing selects active experts based on task phase
- Preserves pretrained VLA backbone while improving phase-specific reliability
- Demonstrates that manipulating the action expert architecture (not just perception) is critical for robust mobile manipulation

### Geometry-Grounded VLMs (G2VLM)

G2VLM (InternRobotics, CVPR 2026) represents a shift from treating geometry as an auxiliary input module to making it a core grounding modality. The model jointly predicts 3D geometry (depth maps, structure-from-motion reconstructions) alongside spatial reasoning tasks, interleaving "what is this object" with "where is it in 3D space" reasoning. This is conceptually distinct from FALCON's token-injection approach — G2VLM bakes geometric understanding into the VLM backbone rather than augmenting it post-hoc.

### 3D Gaussian Splatting as Unified Perception Representation

SemGauss-SLAM (IROS 2025) merges depth, pose, and segmentation into a single dense 3D scene representation by adding semantic labels to Gaussian Splatting maps. The resulting 3DGS representation simultaneously encodes geometry (depth proxy), object locations (pose proxy), and semantics (segmentation) — making it the closest realization of a fully fused perception pipeline. GSWorld extends this to closed-loop sim-to-real training. Feeding 3DGS as direct VLA conditioning remains untested for real-robot manipulation.

### Mobile Manipulation at Scale (AnywhereVLA)

AnywhereVLA (arXiv:2509.21006) demonstrates a deployable architecture for real-time mobile manipulation: perception/VLA (SmolVLA, 450M params) runs on Jetson Orin NX while SLAM/control runs on an Intel NUC. The split enables multi-room mobile manipulation at real-time frequencies and provides the closest published architecture to a production fused depth-pose-segmentation pipeline.

### Failure Modes

- **Transparent objects**: Glass, acrylic, water produce depth predictions 15–40 cm off ground truth.
- **Specular surfaces**: Polished metal and mirrors violate Lambertian assumptions, causing depth discontinuities.
- **Camera intrinsic sensitivity**: Models trained on specific camera intrinsics degrade on out-of-distribution setups.
- **Close-range limitations**: Performance degrades below 10 cm, outside typical training distribution depth ranges.
- **Implicit depth ambiguity**: Evo-Depth-style implicit depth extraction from multi-view RGB suffers from baseline ambiguity — large baselines (>2m) reduce accuracy as depth cues become less distinct between views.
- **Pose token discretization error**: Pose-VLA's discrete pose tokens introduce quantization error when representing continuous SE(3) transformations, particularly for fine-grained manipulations requiring sub-centimeter precision.

### Pose-Conditioned Anchor Attention (PosA-VLA)

PosA-VLA identifies the root cause of VLA action inconsistency: existing models use a **spatially uniform perception field** that lacks explicit mechanisms to focus on task-relevant regions and the end-effector simultaneously. Without pose-conditioned attention, VLAs reactively scan the entire scene rather than proactively maintaining spatial focus.

**Architecture**: Generates two complementary anchors:
- **Task-relevant anchor**: Activated at moments when end-effector state changes (indicating interaction with a region of interest)
- **End-effector anchor**: Tracks end-effector position at each timestep

These anchors serve as pose-conditioned spatial priors that transform continuous 3D interaction space into localized 2D supervision, breaking the model's spatially uniform perception. The anchors dynamically update as the robot moves, maintaining spatial correspondence between pose and visual scene.

**Key advantage**: Built on lightweight backbones with no auxiliary perception modules (no dedicated segmentation or grounding networks needed), making it efficient for deployment while producing smoother, more stable trajectories with fewer action steps than OpenVLA, Smol-VLA, DexGraspVLA, or π₀.

### Multi-Scale Point-Action Interaction (PointACT)

PointACT (RSS 2026) takes a fundamentally different approach: rather than injecting depth/pose/segmentation tokens into the VLM or action head, it represents the entire scene as **hierarchical 3D point cloud** inputs to a dedicated point-action expert.

**Architecture**:
- Frozen VLM backbone for semantic encoding
- Point tokens extracted from 3D point cloud representations
- **Multi-scale point-action attention** via bottleneck window self-attention: point tokens densely attend to local geometric detail and global scene structure
- Action tokens evolve through multi-scale interaction with point tokens

**Fusion insight**: By using 3D point representations as the unified geometric primitive, PointACT naturally integrates depth, pose, and spatial structure without requiring separate perception modules — the point cloud itself carries all three signals.

### Decoder-Free Pixel-Level Segmentation (SimpleSeg)

SimpleSeg (Moonshot AI, arXiv:2601.19228) eliminates the need for a separate segmentation decoder by reframing segmentation as a sequence generation problem: the MLLM directly predicts a sequence of point coordinates delineating object boundaries, entirely within its language space. This is the simplest path to pixel-level VLM perception — no decoder heads, no additional training branches, just point prediction as text. Key implications for fused pipelines: segmentation becomes a native VLM capability that can be called alongside depth and pose estimation without separate models, simplifying the architecture and reducing latency.

### Intent-Driven Perception (InCoM)

InCoM (arXiv:2602.23024v4) proposes an intent-driven perception framework for mobile manipulation that infers latent motion intent to dynamically reweight multi-scale perceptual features. The system performs stage-adaptive allocation of perceptual attention — redirecting more compute to the perception modality most needed at each manipulation stage (e.g., depth for approach, pose for grasp, segmentation for placement). This addresses a key efficiency gap in fused pipelines where all three modalities run at fixed bandwidth regardless of task phase.

### 3D-Aware Hierarchical Planning (3D HAMSTER)

3D HAMSTER (arXiv:2606.31329v1) bridges planning and control in hierarchical VLM pipelines by operating in 3D space rather than 2D image space. The paper identifies that 2D representations introduce ambiguity under 3D-sensitive shifts (occlusion, viewpoint changes) and proposes a 3D-aware planning architecture that fuses depth, pose, and semantic information in a hierarchical representation. This addresses the fundamental 2D-to-3D grounding gap that limits fused perception pipelines.

### Mobile Manipulation Benchmarking (MobileManiBench)

MobileManiBench (arXiv:2602.05233) provides a benchmark pipeline built on NVIDIA Isaac Sim that autonomously generates diverse manipulation trajectories with rich multi-modal annotations: language instructions, multi-view RGB-depth-segmentation images, synchronized object/robot states and actions. The dataset supports model verification for mobile manipulation — a critical missing piece for evaluating fused perception-to-action pipelines on mobile platforms.

### 3D Spatial Reasoning Foundation (SpatialVLM)

SpatialVLM (Google DeepMind, CVPR 2024) demonstrates that VLMs' limited spatial reasoning is a data problem, not an architecture problem. By automatically generating 2 billion VQA examples on 10 million real-world images — lifting 2D images into metric-scale 3D point clouds — they co-train a VLM that achieves:
- 37.2% of quantitative distance estimates within 0.5x–2x of ground truth
- Strong qualitative spatial VQA outperforming baselines by large margins
- Chain-of-thought spatial reasoning by orchestrating the SpatialVLM with an LLM
- Dense reward annotation for open-vocabulary robotic tasks (monotonically decreasing distance estimation for RL reward signals)

This establishes that internet-scale spatial data synthesis is the foundation for VLMs to reason about geometry quantitatively — a prerequisite for fused depth-pose-segmentation pipelines to produce meaningful spatial actions.

### Unified Embodied Agent Framework (HoloAgent-0)

HoloAgent-0 (arXiv:2606.23565) provides a unified framework for long-horizon mobile manipulation by composing navigation and perception skills through a common command/status interface. For perception, it integrates HoloNavi for scene understanding and HoloBrain for local manipulation skills, with 3D spatial memory enabling persistent scene understanding across extended manipulation sequences. This demonstrates how fused perception can be integrated into a broader agent architecture where navigation and manipulation share perception resources.

### Spatio-Temporal Reasoning with Memory (RoboStream)

RoboStream (arXiv:2603.12939) weaves spatio-temporal reasoning with persistent memory into VLM-based planners for long-horizon manipulation. It outperforms existing VLM-based planners across all settings, demonstrating that spatio-temporal reasoning and persistent memory are essential for robust mobile manipulation. The system segments instructions into atomic commands with persistent spatial memory, enabling the perception pipeline to maintain context across extended manipulation sequences.

### Zero-Shot Mobile Manipulation (OK-Robot)

OK-Robot (Meta) combines three primary subsystems for zero-shot pick-and-place in novel environments:
1. **Open-vocabulary object navigation**: VLM-driven detection guides the robot to target objects
2. **RGB-D grasping module**: Fuses depth and visual features for grasping planning
3. **Dropping heuristic system**: Determines safe drop locations

This demonstrates that VLM-based perception can be combined with conventional grasping modules to achieve zero-shot mobile manipulation without full end-to-end VLA training.

## Open Questions

- How do fused perception pipelines scale to multi-arm or whole-body mobile manipulation where perception must cover larger workspaces?
- What is the optimal fusion strategy (early vs. late vs. spatial token injection vs. geometric inductive bias) across different VLA architectures?
- Can the pipeline achieve real-time operation (≥20 Hz end-to-end) on consumer-grade edge hardware without cloud inference?
- How do we benchmark the perception-to-manipulation-latency relationship in a standardized way?
- How does the OmniRobotHome testbed approach (treating perception quality as experimental variable) inform the design of real-world fused perception systems?
- Can VLM-based open-vocabulary segmentation replace dedicated segmentation networks, or is a hybrid approach necessary?
- How does the spatial chain-of-thought paradigm (Perceptio) generalize to video-based VLAs for temporal perception?
- Can VLM-inferred physical priors (Phys2Real) be combined with fused perception pipelines for real-time dynamic object manipulation?
- How do geometric inductive biases (G^3VLA ray embeddings) compare to spatial token injection (FALCON) in terms of transferability across robot platforms?
- Can MolmoAct2-level open-source VLA models be fine-tuned with fused perception heads for specific mobile manipulation deployments?
- **3DGS-as-perception-replacement**: Can 3D Gaussian Splatting maps serve as the sole perception representation for mobile manipulation, replacing separate depth maps, pose estimates, and segmentation masks? SemGauss-SLAM shows the representation is possible; feeding it as VLA conditioning is untested.
- **Monocular depth metric calibration**: How do we calibrate monocular depth for manipulation without LiDAR or stereo — using known object priors, IMU fusion, or self-supervised calibration through grasping feedback? No system has demonstrated robust metric depth from monocular RGB alone in uncalibrated environments.
- **Cross-modal consistency guarantees**: When depth, pose, and segmentation disagree (e.g., depth suggests an object is 2m away, pose suggests it's on a surface 1.5m away), how does the pipeline resolve conflicts? A consistency-aware fused architecture is an open design problem.
- **Active perception for pose refinement**: Can the robot actively move the camera to improve pose estimates of occluded objects — choosing viewpoints that simultaneously improve depth, pose, and segmentation?
- **Perception-to-action latency budget**: How much of the total perception-to-action latency budget is consumed by the fused perception pipeline vs. VLA inference? Understanding this split is needed for real-time systems design.
- **SimpleSeg decoder-free segmentation**: Can point-sequence prediction entirely replace decoder-based segmentation heads in fused pipelines? SimpleSeg shows it works as a standalone method, but its integration with depth and pose estimation modules remains untested.
- **Spatial reasoning data scale**: How much synthetic spatial VQA data is sufficient? SpatialVLM used 2 billion examples; does performance saturate, or does more data continue to improve quantitative spatial reasoning for manipulation?
- **HoloAgent-0 composability**: Can the unified command/status interface between HoloNavi and HoloBrain be generalized to other navigation + perception skill compositions?
- **Intent-driven resource allocation (InCoM)**: Can perception bandwidth be dynamically re-routed between depth, pose, and segmentation based on manipulation stage? InCoM shows stage-adaptive attention works, but whether it generalizes across task types and robot morphologies is untested.
- **3D planning space for fused perception (3D HAMSTER)**: Can fused perception pipelines eliminate the 2D-to-3D ambiguity gap entirely by operating natively in 3D space? 3D HAMSTER demonstrates this reduces occlusion/viewpoint failures, but the computational cost of 3D representations vs. 2D image-space approaches is unmeasured.
- **Benchmark coverage (MobileManiBench)**: Can Isaac Sim-based benchmarking produce mobile manipulation datasets with the multi-modal annotation richness needed to train fused perception pipelines end-to-end? The current benchmark has rich annotations but coverage of real-world edge cases (lighting, novel objects, dynamic scenes) is limited.

## Related

- [[Vision-Language-Action Models]]
- [[Depth Estimation]]
- [[6D Object Pose Estimation]]
- [[Semantic Segmentation Robotics]]
- [[Mobile Manipulation]]
- [[Edge AI for Robotics]]
- [[Real-time SLAM Integration with VLA Policies]]
- [[Multi-Modal Grounding: Language-to-Action Mapping]]
- [[VLA Edge Deployment: Qwen2.5-VL, Mobile-VideoGPT, SmolVLM]]
- [[Online Fine-Tuning for VLA Models]]

## Sources

1. FALCON: From Spatial to Action — novel VLA paradigm injecting 3D spatial tokens into a dedicated action head [arxiv.org/abs/2503.05850]
2. OpenVLA: Open-source 7B VLA model fusing DINOv2 + SigLIP [arxiv.org/abs/2406.09246]
3. Depth Anything V2 — monocular depth estimation at 50 FPS for robotics [github.com/DepthAnything/Depth-Anything-V2]
4. 6D Pose Tracking overview — SE(3) Lie algebra formulations and event-camera methods [emergentmind.com/topics/6d-pose-tracking]
5. Pure VLA Models survey — comprehensive review of VLA architectures [arxiv.org/abs/2509.19012]
6. CogVLA — cognition-aligned VLA via instruction-driven routing & sparsification, NeurIPS 2025 [github.com/iLearn-Lab/NeurIPS25-CogVLA]
7. Perceptio — spatial chain-of-thought via explicit 2D/3D token generation before text [Amazon Research, 2026]
8. Phys2Real — VLM physical priors + interactive online adaptation for sim-to-real manipulation [arxiv.org/abs/2510.11689]
9. GoalVLM — VLM-driven multi-agent navigation with fused semantic maps [arxiv.org/abs/2603.18210]
10. BUMBLE — building-wide mobile manipulation with VLMs for long-horizon tasks [robin-lab.cs.utexas.edu]
11. Knowledge Insulating VLA models (companion to CogVLA) [arxiv.org/abs/2505.23705]
12. IROS: Dual-Process Architecture for Real-Time VLM-Based Indoor Navigation [arxiv.org/html/2601.21506]
13. G^3VLA — geometric inductive bias via intrinsic-conditioned ray embeddings [arxiv.org/abs/2606.24472]
14. MolmoAct2 — Allen Institute open-source VLA with 37x speedup, 87.1% real-world success [allenai.org/blog/molmoact2]
15. TwinBrainVLA — asymmetric dual-stream VLA via AsyMoT [arxiv.org/abs/2601.14133]
16. LingBot-VLA + LingBot-Depth — self-supervised Masked Depth Modeling for spatial perception [github.com/Robbyant/lingbot-depth]
17. EchoVLA — synergistic declarative memory for long-horizon mobile manipulation [arxiv.org/abs/2511.18112]
18. World Action Models survey — Render-and-Decode framework unifying depth/pose/segmentation [arxiv.org/abs/2606.20781]
19. Evo-Depth — lightweight Implicit Depth Encoding Module for depth-from-RGB without explicit sensors [arxiv.org/abs/2605.14950]
20. Pose-VLA — universal pose pretraining via discrete SE(3) Pose Tokens [arxiv.org/abs/2602.19710]
21. LA4VLA — vision-agnostic language-action pretraining for plug-and-play perception [arxiv.org/abs/2606.27295]
22. PAMAE — phase-aware Mixture-of-Experts action routing for long-horizon manipulation [arxiv.org/abs/2606.27144]
23. OmniRobotHome — room-scale multi-camera testbed for real-time fused perception [arxiv.org/abs/2604.28197]
24. G2VLM — geometry-grounded VLM with unified 3D reconstruction & spatial reasoning, CVPR 2026 [github.com/InternRobotics/G2VLM]
25. SemGauss-SLAM — dense semantic Gaussian Splatting SLAM, IROS 2025; GSWorld extends to closed-loop sim-to-real
26. AnywhereVLA — real-time mobile manipulation with Jetson Orin + NUC split [arxiv.org/abs/2509.21006]
27. MAP-VLA — memory-augmented prompting for VLA manipulation [arxiv.org/abs/2511.09516]; MemoryVLA — cognition-memory-action framework [github.com/G-U-O/memvla]
28. MoMani Benchmark — large-scale automated benchmark for long-horizon mobile manipulation [emergentmind.com/topics/momani-benchmark]
29. PosA-VLA — pose-conditioned anchor attention for consistent VLA action generation [arxiv.org/abs/2512.03724]
30. PointACT — multi-scale point-action interaction via hierarchical 3D point clouds, RSS 2026 [arxiv.org/abs/2605.21414]
31. InCoM — intent-driven perception with stage-adaptive multi-scale perceptual attention for mobile manipulation [arxiv.org/abs/2602.23024]
32. 3D HAMSTER — 3D-aware hierarchical planning bridging VLM planning and control [arxiv.org/abs/2606.31329]
33. MobileManiBench — Isaac Sim-based benchmark for mobile manipulation with multi-modal trajectory annotations [arxiv.org/abs/2602.05233]

## Confidence

0.85: Core architectural claims (FALCON token injection, OpenVLA fusion, Depth Anything V2 performance) are well-supported by published papers with concrete numbers. The expanded 2026 update now covers 33 distinct architectures/systems spanning spatial token injection, geometric inductive bias, asymmetric dual-stream VLAs, declarative memory, intent-driven perception, and 3D-aware planning. Confidence increased slightly (from 0.84) due to InCoM's explicit stage-adaptive perceptual allocation (validating the hypothesis that fused pipelines should reweight modality bandwidth by task phase) and MobileManiBench's rich multi-modal annotations (establishing benchmark infrastructure for evaluating end-to-end fused pipelines). Confidence remains tempered by: (a) no single paper demonstrates a complete fused depth-pose-segmentation pipeline at 20+ Hz on edge hardware for mobile manipulation end-to-end; (b) field evolution is rapid with fragmented evaluation protocols; (c) 3DGS-as-VLA-conditioning remains untested on real robots.
