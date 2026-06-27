[updated 2025-07-18, refreshed 2026-06-07, extended 2026-07-11, 2026-07-14]

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

### Failure Modes

- **Transparent objects**: Glass, acrylic, water produce depth predictions 15–40 cm off ground truth.
- **Specular surfaces**: Polished metal and mirrors violate Lambertian assumptions, causing depth discontinuities.
- **Camera intrinsic sensitivity**: Models trained on specific camera intrinsics degrade on out-of-distribution setups.
- **Close-range limitations**: Performance degrades below 10 cm, outside typical training distribution depth ranges.

## Open Questions

- How do fused perception pipelines scale to multi-arm or whole-body mobile manipulation where perception must cover larger workspaces?
- What is the optimal fusion strategy (early vs. late vs. spatial token injection vs. geometric inductive bias) across different VLA architectures?
- Can the pipeline achieve real-time operation (≥20 Hz end-to-end) on consumer-grade edge hardware without cloud inference?
- How do we benchmark the perception-to-manipulation-latency relationship in a standardized way?
- What is the cost/benefit tradeoff between monocular depth, stereo depth, and event-based sensors for mobile manipulation perception?
- Can VLM-based open-vocabulary segmentation replace dedicated segmentation networks, or is a hybrid approach necessary?
- How does the spatial chain-of-thought paradigm (Perceptio) generalize to video-based VLAs for temporal perception?
- Can VLM-inferred physical priors (Phys2Real) be combined with fused perception pipelines for real-time dynamic object manipulation?
- How do geometric inductive biases (G^3VLA ray embeddings) compare to spatial token injection (FALCON) in terms of transferability across robot platforms?
- Can MolmoAct2-level open-source VLA models be fine-tuned with fused perception heads for specific mobile manipulation deployments?

## Related

- [[Vision-Language-Action Models]]
- [[Depth Estimation]]
- [[6D Object Pose Estimation]]
- [[Semantic Segmentation Robotics]]
- [[Mobile Manipulation]]
- [[Edge AI for Robotics]]

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

## Confidence

0.84: Core architectural claims (FALCON token injection, OpenVLA fusion, Depth Anything V2 performance) are well-supported by published papers with concrete numbers. Previous extensions (Perceptio, CogVLA, Phys2Real, G^3VLA, MolmoAct2, IROS) bring in 2025–2026 developments with new evidence. The July 2026 update adds four major architectures: TwinBrainVLA (asymmetric dual-stream, resolving catastrophic forgetting via AsyMoT), LingBot-VLA/LingBot-Depth (self-supervised masked depth modeling with cross-modal RGB-Depth alignment), EchoVLA (declarative memory for long-horizon mobile manipulation), and the World Action Models survey (unifying depth/pose/segmentation via Render-and-Decode latent codebooks). Confidence increased from 0.82 due to TwinBrainVLA's explicit decoupling of semantic vs embodied perception streams (validating the knowledge insulation hypothesis from CogVLA) and LingBot-Depth's metric-scale preservation (addressing close-range depth limitations). The WAM survey provides a formal unifying framework across perception modules. Confidence remains tempered by the field's rapid evolution and fragmented evaluation protocols — no single paper demonstrates a complete fused depth-pose-segmentation pipeline operating at 20+ Hz on edge hardware for mobile manipulation end-to-end.
