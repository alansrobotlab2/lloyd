---
title: Multi-Modal Grounding for Agents — Language-to-Action Mapping in VLMs
tags:
  - ai/vlm
  - ai/vla
  - ai/multimodal-grounding
  - ai/visual-grounding
  - ai/language-grounding
  - ai/spatial-grounding
  - ai/geometry-grounded
  - robotics/embodied-ai
  - research/domain-research
created: 2026-07-15
updated: 2026-09-20
confidence: 0.92
---

# Multi-Modal Grounding for Agents — Language-to-Action Mapping in VLMs

## Summary

Multi-modal grounding is the process of aligning linguistic expressions with visual entities, spatial locations, and physical actions so that embodied agents can interpret instructions and produce contextually appropriate behaviors. Vision-Language-Action (VLM/VLA) models operationalize this by encoding visual observations, natural language instructions, and robot state into a shared token space, then autoregressively or diffusion-based generating action sequences conditioned on the fused representation. The field has evolved from modular pipelines (separate perception, language, and control modules) to end-to-end architectures that jointly learn grounding through large-scale pretraining on internet vision-language corpora and robot trajectory datasets, with recent work (Qwen-VLA, InternVLA-M1, Green-VLA) unifying grounding, spatial reasoning, and continuous action generation across diverse robot embodiments. A persistent challenge is the *spatial reasoning gap*: most VLAs inherit 2D VLM backbones and lack explicit 3D geometric understanding, leading to grounding degradation under clutter, object scale variation, and fine-tuning.

## Key Facts

- **Three-layer grounding decomposition**: Visual grounding aligns text with regions or objects in images. Language grounding connects linguistic predicates to physical entities or actions in the environment. Action grounding maps natural language instructions to executable motor commands. Multi-modal grounding unifies all three — the agent must understand *what* the instruction refers to, *where* it is, and *how* to act on it.

- **Token-based unification** is the dominant grounding mechanism in modern VLAs. Visual inputs are encoded via ViT or ConvNeXt into vision tokens, language instructions via LLMs (T5, LLaMA, Qwen) into language tokens, and robot state (joint angles, gripper pose) via MLP into state tokens. Cross-attention fuses these into a shared embedding space, and an autoregressive decoder generates action tokens step-by-step — effectively treating motor control as "language generation" where the vocabulary is physical actions.

- **Five VLA paradigms** (per the 2025 Pure VLA Survey, arXiv:2509.19012, covering 300+ studies): (1) **Autoregression-based**: RT-2, Octo — treat action generation as next-token prediction; (2) **Diffusion-based**: Diffusion Policy — continuous action prediction via denoising; (3) **Reinforcement-based**: policy optimization via RL on VLA priors; (4) **Hybrid**: combine autoregressive planning with diffusion execution; (5) **Specialized**: domain-specific architectures (e.g., NaVILA for legged navigation).

- **VLA architectural taxonomy** (Shao et al., arXiv:2508.13073, covering large VLM-based VLAs for robotic manipulation):
  1. **Monolithic models** integrate perception, language understanding, and action generation within a unified architecture.
     - **Single-system** (e.g., OpenVLA, RT-2): all modalities processed in one model with autoregressive or parallel decoding.
     - **Dual-system** (e.g., π₀): VLM backbone (System 2, slow reasoning) cooperates with an action expert (System 1, fast/reactive), exchanging information via latent representations.
  2. **Hierarchical models** explicitly decouple planning from policy execution through interpretable intermediate representations (subtasks, keypoints, programs, affordances). Planner modules generate structured outputs (keypoint detections, trajectory proposals) that policy modules convert to executable actions. This differs from dual-system approaches through decoupled training paradigms with specialized loss functions.

- **Three evolutionary phases** have shaped VLA grounding:
  1. **Foundational Integration (2022–2023)**: CLIPort, Gato, RT-1, VIMA — combined pretrained vision-language representations with task-conditioned policies. CLIPort encoded CLIP embeddings to output pixel-level pick-and-place distributions. RT-1 used scaled imitation learning at 97% success rates.
  2. **Specialization & Embodied Reasoning (2023–2024)**: RT-2, VoxPoser, ACT — introduced visual chain-of-thought reasoning and affordance grounding. RT-2 fused visual-language tokens and action representations in a unified transformer, co-trained on internet-scale corpora and 100K+ robot demonstrations. VoxPoser added voxel-level 3D reasoning for zero-shot manipulation. Octo added memory-augmented transformers trained on 4M+ trajectories.
  3. **Generalization & Safety-Critical Deployment (2025–2026)**: Qwen-VLA, G2VLM, SafeVLA, Humanoid-VLA, Gr00t N1 — unified multi-task grounding (manipulation + navigation + trajectory prediction), geometry-grounded reasoning, formal verification, whole-body control, sim-to-real transfer, and cross-embodiment generalization.

- **Qwen-VLA** (arXiv:2605.30280) represents a major step toward unified embodied foundation models: extends Qwen's vision-language stack from perception/reasoning to continuous action via a DiT-based action decoder. Joint pretraining across diverse data sources (robotics manipulation, human egocentric demos, synthetic simulation, VLN navigation, auxiliary VLM data). Uses **embodiment-aware prompt conditioning** where robot-specific textual descriptions specify control conventions. Achieves 97.9% on LIBERO, 73.7% on Simpler-WidowX, 86.1%/87.2% on RoboTwin-Easy/Hard, and 76.9% average OOD success in real-world ALOHA experiments.

- **InternVLA-M1** (arXiv:2510.13778) introduces **spatially guided VLA training** as a unifying principle: a dual-system architecture with a VLM planner (System 2, slow/reasoning) that produces spatial grounding tokens via explicit spatial prompting, and a Diffusion Policy action expert (System 1, fast/executor) that translates these into embodiment-specific motor commands. Key innovations: (i) spatial grounding pre-training on 2.3M spatial reasoning samples (points, boxes, traces) to establish transferable spatial priors; (ii) spatially guided action post-training with plug-and-play spatial prompting; (iii) gradient decay in the querying transformer to preserve VLM semantics during joint optimization. Achieves +14.6% on SimplerEnv Google Robot, +17% on WidowX, +4.3% on LIBERO Franka, and +20.6% on unseen objects in real-world clustered pick-and-place.

- **Green-VLA** (arXiv:2602.00919) is a ~5B-parameter staged VLA for generalist robot deployment on the Green humanoid. Built on Qwen3-VL (4B) with a dedicated flow-matching action expert. Uses a five-stage training curriculum (base pretrained → action expert specialization → embodiment adaptation → task fine-tuning → real-world calibration) while maintaining generalization across diverse embodiments. Demonstrates a practical blueprint for deploying VLAs on humanoids beyond simulation.

- **The spatial reasoning gap is a critical bottleneck**: Most VLAs inherit 2D VLM backbones (CLIP, Kosmos-2) and lack explicit 3D geometric priors. This causes systematic failures on cluttered scenes, spatial-prompt conditioning ("pick the object on the left"), and object scale/height variation. FALCON (arXiv:2510.17439) addresses this by injecting rich 3D spatial tokens from spatial foundation models (DUSt3R, VGGT) into a dedicated Spatial-Enhanced Action Head rather than concatenating them into the VLM backbone, preserving semantic alignment while adding geometric grounding.

- **G2VLM** (CVPR 2026) introduces geometry-grounded VLMs that natively predict 3D geometry and employ interleaved reasoning — combining spatial 3D reconstruction with spatial understanding tasks. This represents a shift from treating geometry as an auxiliary input to making it a core grounding modality.

- **Visual grounding degradation during VLA fine-tuning**: When VLMs are fine-tuned for action prediction, their vision-language representations drift toward "action-effective but grounding-weak" features. This manifests as over-grasping, distractor sensitivity, and poor robustness under clutter and distribution shift — documented in clutter-resistant VLA research (arXiv:2512.22519).

- **M3ID (Multi-Modal Intervention)** (CVPR 2024) is a training-free inference-time intervention that improves visual grounding in autoregressive VLMs by amplifying the importance of visual tokens over language priors, reducing hallucination and grounding errors.

- **GroundingGPT** (ACL ARR 2024) proposes an end-to-end multi-modal grounding model for fine-grained grounding across image, video, and audio modalities using a coarse-to-fine three-stage training strategy. Addresses the observation that most MLLMs prioritize global information and miss fine-grained details needed for precise grounding.

- **State-of-the-art grounding architectures** use hierarchical grounding: coarse spatial localization (which region contains the target) followed by fine-grained grounding (precise action parameters). ST4VLA (arXiv:2602.10109) co-trains on spatial grounding datasets alongside action data to quantify and maintain alignment between grounding and action objectives.

- **The action tokenization problem** is central: how to discretize continuous motor commands into tokens a transformer can generate. Approaches include: (a) quantization into discrete bins (RT-2), (b) diffusion-based action prediction (Diffusion Policy), (c) flow matching for continuous action generation (π0), and (d) hybrid discrete-continuous tokenization. The choice affects grounding fidelity — coarse quantization loses precision, while continuous methods lack the compositional benefits of token-based generation.

- **MMaDA-VLA** (arXiv:2603.25406) represents a paradigm shift from autoregressive to discrete diffusion VLA grounding: unifies language, images, and robot actions into a single discrete token space via native masked token prediction. Eliminates the "module boundary" problem where policy heads lose fidelity from pretrained VLMs. Uses MAGVIT-v2 for image quantization and 256-bin action discretization. Key insight: action dimensions within a chunk are inherently unordered — parallel iterative denoising (24 steps) allows full trajectory refinement without sequential error compounding. Achieves 98.0% on LIBERO and 4.78/5 on CALVIN ABC→D, significantly outperforming DreamVLA (4.44). Goal image prediction is critical for grounding — removing it drops performance, confirming the model uses "imagined" future states to ground physical actions. Addresses dLLM-Cache training-free caching for real-time inference.

- **DAM-VLA** (arXiv:2606.12105) addresses the synchronous-clock bottleneck: traditional VLAs process all modalities at one rate inherited from VLM pretraining. Decoupled asynchronous processing lets vision, language, and action streams update at independent frequencies, improving grounding fidelity for high-frequency sensor inputs without wasting compute on stale language tokens.

- **VP-VLA** (arXiv:2603.22003) introduces visual prompting as a structured interface between high-level reasoning and low-level execution. Traditional VLAs force a single forward pass to simultaneously handle instruction interpretation, spatial grounding, AND low-level control — VP-VLA decouples these via a dual-system framework where a planner analyzes instructions and identifies important objects/locations, producing visual prompts that serve as a universal interface for downstream policy modules. This addresses the "black-box mapping" problem where grounding quality is inseparable from control quality in monolithic architectures.

- **VEGA** (arXiv:2605.10485) — Visual Encoder Grounding Alignment for spatially-aware VLA models. Aligns the VLA's visual encoder with 3D-aware features by performing alignment at the visual encoder output level, grounding spatial awareness before any linguistic entanglement occurs. This offers a more interpretable and principled alignment target compared to end-to-end fusion where spatial and semantic signals compete within shared tokens.

- **VIPA-VLA** (CVPR 2026, arXiv:2512.13080) demonstrates spatial-aware VLA pretraining through visual-physical alignment from human videos. A dual-encoder architecture with a 3D visual encoder augmenting semantic representations with 3D-aware features, aligned through visual-physical alignment pretraining on human video data with 3D annotations. Shows that 2D-to-3D visual-physical grounding learned from human demonstration videos transfers to robot policies with stronger spatial understanding and generalization than 2D-only pretraining.

- **Visual grounding survey** (arXiv:2509.10345) systematically reviews visual grounding in general-purpose VLMs, covering core components (region proposal, cross-modal alignment, fine-grained grounding), benchmarks, evaluation metrics, and the interrelation between visual grounding, multimodal chain-of-thought, and reasoning capabilities.

- **Cross-modal grounding benchmarks** remain sparse. Existing evaluation focuses on manipulation success rates (e.g., LIBERO, BridgeData) rather than explicit grounding accuracy. This makes it difficult to isolate whether failures stem from poor grounding (misidentifying targets) vs. poor action execution (correctly identifying but failing to manipulate).

- **LingBot-VLA** (Jan 2026) introduces a pragmatic VLA foundation model with depth-free and depth-distilled configurations, built on Qwen2.5-VL-3B-Instruct. It demonstrates that depth information can be distilled into the model rather than required as runtime input, reducing sensor dependency for spatial grounding.

- **Embodied-R** (arXiv:2504.12680, Zhao et al., Apr 2025) shows that reinforcement learning with just 5k samples can enable 3B LMs to match frontier models on embodied spatial reasoning. This demonstrates that spatial grounding capabilities can be efficiently activated through RL rather than requiring massive pretraining data.

- **AT-VLA** (May 2026) introduces adaptive tactile injection for enhanced feedback in VLA models, adding tactile sensing as a grounding modality alongside visual-language. This represents the first systematic integration of touch as a grounding channel for language-to-action mapping.

- **TacVLA** (arXiv:2603.12665, Mar 2026) integrates tactile sensing into VLA models through **contact-aware fusion mechanisms**: unified tokenization of tactile and visual inputs with contact-aware gating that modulates cross-modal fusion based on contact state. Addresses three critical VLA limitations — visual occlusion during grasping, fine manipulation requiring contact feedback, and contact detection during insertion/slip recovery. Represents a systematic extension of grounding from vision-language to vision-language-tactile tri-modal grounding.

- **IRef-VLA** (arXiv:2503.17406, WCVC 2026) introduces a benchmark for **interactive referential grounding with imperfect language in 3D scenes**. Addresses the challenge of grounding when language references are misaligned, ambiguous, or incomplete — a core failure mode for real-world robot instruction following where natural language is rarely perfect.

- **Spatial-X** (Gen 3, 2024–2026) demonstrates a pre-exploration, 3D scene reconstruction, and grounded navigation loop, showing that spatial grounding can be bootstrapped through active exploration rather than passive observation.

- **Xiaomi-Robotics-0** (arXiv:2602.12684) is a 4.7B-parameter open-sourced VLA model optimized for high-performance real-time execution. Demonstrates that grounding can be maintained at real-time control frequencies through architecture-level optimizations — addressing the synchronous-clock bottleneck that DAM-VLA identifies as a key constraint for high-frequency sensor inputs.

- **LangGap** (Hou & Zhao, arXiv:2606.02277, Jun 2026) diagnoses and closes the **language gap** in VLA models — evaluating semantic grounding in action prediction rather than language sensitivity alone. Identifies specific failure modes where VLAs correctly produce actions but fail to ground them to the correct language-referenced entities, providing a diagnostic framework for language-action alignment quality.

- **Knowledge Insulating VLMs** (arXiv:2505.23705) explores architectural designs that enable "train fast, run fast, generalize better" — decoupling grounding knowledge acquisition from inference speed, addressing the tradeoff between comprehensive grounding and real-time responsiveness.

- **Attention-based self-monitoring** (Jun 2026 arXiv preprint) demonstrates that individual attention heads within VLA models encode which object the robot plans to approach at each control step — suggesting that grounding fidelity can be monitored internally via attention patterns, opening a path for self-diagnostic grounding quality metrics without external supervision.

- **World-Action Models** (NVIDIA Developer Blog, 2026): Emerging paradigm extending VLAs to world models that are pretrained to imagine future states and fine-tuned to act on them. Language-to-action grounding is reframed as "turning an instruction like 'pick up the red mug' into the visual percepts and motor commands that actually accomplish it" — positioning grounding as a world-model-to-action translation problem.

- **Multimodal LVLMs in interactive environments** (Preprints.org, 2025): Multimodal Large Vision-Language Models (LVLMs) have become central for interactive environments, enabling machines to jointly perceive, reason, and communicate across visual and linguistic modalities at unprecedented scale — with GUI agents, web agents, and mobile agents representing non-robotic grounding applications.

- **ProGAL-VLA** (arXiv:2604.09824) addresses the *language ignorance* problem — where VLAs rely on visual shortcuts and remain insensitive to instruction changes. Constructs a 3D entity-centric **Graph Scene Model (GSM)**, uses a slow planner to produce symbolic sub-goals, and aligns them with grounded entities via a **Grounding Alignment Contrastive (GAC) loss**. This represents a shift from implicit grounding to explicit, contrastive language-entity alignment during prospective reasoning.

- **LabVLA** (arXiv:2606.13578) demonstrates grounding for **domain-specific VLA deployment**: adapts a Qwen3-VL backbone to map visual observations, robot state, and language instructions into continuous action chunks through a DiT action expert. Trained with a two-stage approach (action token pretraining + flow matching) and demonstrates that grounding generalizes to scientific laboratory tasks (transparent liquids, specialized instruments) where existing policies trained on household/tabletop data fail.

- **GIVE** (arXiv:2606.13435) introduces **gestural grounding** for VLAs: addresses the gap where current models treat manipulation as a pure text-driven task, overlooking human gesture input. Integrates gesture-as-grounding-channel, showing that pointing, directing, and gesturing provide a complementary grounding modality alongside language and vision — especially useful for ambiguous instructions where language alone is insufficient.

- **RL Token (RLT)** (arXiv:2604.23073) introduces a compressed, task-relevant **RL token** — a learned readout representation that compresses a VLA's deep understanding into a small vector, enabling fast and sample-efficient online RL fine-tuning. The lightweight actor-critic refines actions based on the RL token while anchoring to the VLA, making it possible to fine-tune large VLAs with RL on real robots. This bridges the gap between VLA grounding (what to do) and online learning (how well to do it).

- **JEPA-VLA** (arXiv:2602.11832) explores grounding via **Joint Embedding Predictive Architectures** that enable reasoning, perception, and decision-making without tight coupling to textual representations. Demonstrates that MLLM-style language dependence can be partially decoupled, opening a path for grounding that is less brittle to language formulation changes.

## Related (vault entities)
- [[Online Fine-Tuning for VLA Models — Continual Learning with Experience Replay]]
- [[Real-time SLAM Integration with VLA Policies — Mapping During Manipulation Tasks]]
- [[VLA Edge Deployment: Qwen2.5-VL, Mobile-VideoGPT, SmolVLM]]
- [[Self-Improving Autonomous Agents]]
- [[Multi-Agent Task Decomposition with Hierarchical Planning]]
- LingBot-VLA (depth-distilled spatial grounding)
- Embodied-R (RL-activated spatial reasoning)
- AT-VLA (tactile-injection grounding)
- TacVLA (contact-aware tactile fusion for tri-modal grounding)
- Spatial-X (pre-exploration grounding loop)
- X-VLA (cross-embodiment soft-prompted grounding)
- InternVLA-M1 (spatially guided dual-system VLA training)
- Green-VLA (staged curriculum for generalist humanoid VLA)
- IRef-VLA (benchmark for grounding with imperfect language)
- ProGAL-VLA (prospective reasoning with grounded alignment, 3D entity-centric graph)
- LabVLA (domain-specific grounding for scientific laboratories)
- GIVE (gestural grounding as complementary modality)
- RL Token (online RL fine-tuning compressed readout)
- JEPA-VLA (grounding decoupled from textual representation)
- Xiaomi-Robotics-0 (real-time execution VLA)
- LangGap (semantic grounding diagnostic framework)
- Knowledge Insulating VLMs (train-fast/run-fast architecture)
- World-Action Models (pretrained-to-imagine paradigm)

## Open Questions

1. **Grounding fidelity vs. action performance tradeoff**: Can we maintain high visual grounding accuracy during VLA fine-tuning? Current fine-tuning degrades the VLM's grounding capabilities — how do we preserve them?

2. **3D grounding from 2D inputs**: FALCON and G2VLM show that spatial foundation models can inject rich 3D priors from RGB alone. How far can geometry-grounded reasoning go without depth sensors? What are the limits of monocular spatial grounding?

3. **Compositional grounding for multi-step instructions**: Current VLAs handle single-step grounding well. How do they ground complex instructions like "put the red cup next to the blue book on the leftmost shelf"? Does the model ground each sub-reference independently or compose a global grounding map?

4. **Grounding under partial observability**: How do VLAs ground instructions when target objects are occluded, off-screen, or ambiguous? ShowUI demonstrates natural language interfaces but grounding in partially observable environments remains limited.

5. **Temporal grounding for long-horizon tasks**: Most grounding is instantaneous (single-frame). How do models maintain grounding over extended task sequences where scene context changes?

6. **Benchmarking the grounding layer**: We need grounding-specific evaluation metrics — not just task success rates. IRef-VLA is a step toward this but focuses on 3D scenes with imperfect language. A broader grounding accuracy benchmark is needed.

7. **Cross-embodiment grounding**: Can grounding transfer across different robot morphologies? A model that grounds "grasp the cup" on a 6-DOF arm must also ground it on a 2-DOF gripper or legged platform. Qwen-VLA's embodiment-aware prompt conditioning and Green-VLA's staged curriculum are steps toward this.

8. **Training data for grounding**: Most robot datasets lack explicit grounding annotations. How do we construct grounding supervision at scale? DoReMi and annotation-free approaches (ReConVLA) explore this direction. InternVLA-M1's approach of curating 2.3M spatial grounding samples from internet + robot data is a concrete template.

9. **Navigation-as-grounding**: NaVILA shows VLA models can ground navigation instructions in legged robots using human video data. How does visual grounding for navigation differ from grounding for manipulation?

10. **Unified grounding across task families**: Qwen-VLA casts manipulation, navigation, and trajectory prediction into a unified action-and-trajectory framework. Does this unification improve or degrade grounding quality for any individual task?

11. **Multi-modal grounding beyond vision-language**: TacVLA shows tactile can be fused via contact-aware gating. What about audio, proprioceptive, and force-torque grounding? Can a truly multi-modal grounding stack emerge that unifies vision, language, touch, sound, and force?

12. **Grounding with imperfect language**: IRef-VLA demonstrates that real-world language references are rarely perfect. How do VLAs handle disfluency, deixis, coreference, and incomplete instructions during grounding?

13. **Language ignorance and shortcut learning**: ProGAL-VLA shows that VLAs can rely on visual shortcuts and remain insensitive to instruction changes. How do we ensure models genuinely ground language rather than learning spurious visual correlations?

14. **Domain-specific grounding transfer**: LabVLA shows grounding trained on household/tabletop data fails in scientific laboratories. How far can domain adaptation go? Do we need domain-specific grounding pretraining or can cross-domain grounding be bootstrapped?

15. **Gestural grounding as a complement**: GIVE shows human gestures provide complementary grounding when language is ambiguous. How do we integrate gestural, visual, and linguistic grounding channels without modality dominance?

16. **Online RL on top of grounding**: RL Token demonstrates that VLA grounding can be bootstrapped with online RL via compressed readout vectors. Does online RL improve or degrade the underlying grounding representation? Can RL fine-tuning be applied without catastrophic forgetting of grounding priors?

17. **Grounding without language dependence**: JEPA-VLA explores decoupling grounding from tight language coupling. Can grounding be learned through joint embedding prediction rather than language supervision?

18. **Black-box grounding vs. intermediate interfaces**: VP-VLA (arXiv:2603.22003) shows that current VLAs force a single forward pass to handle instruction interpretation, spatial grounding, AND low-level control simultaneously. Is there an optimal decomposition? Can structured intermediate representations (visual prompts, subgoal images) serve as a universal interface that improves grounding fidelity regardless of downstream policy architecture?

19. **Visual-physical alignment from non-robot data**: VIPA-VLA demonstrates 2D-to-3D visual-physical grounding learned from human videos with 3D spatial annotations. Does grounding transfer from human demonstration videos to robot policies? What are the sim-to-real transfer limits when pretraining on human rather than robot data?

20. **Real-time grounding at control frequencies**: Xiaomi-Robotics-0 and DAM-VLA address the synchronous-clock bottleneck — can grounding be maintained at real-time control frequencies without degrading fidelity? What is the minimum grounding latency for safe robot control?

21. **Self-diagnostic grounding via attention patterns**: Emerging work shows that attention heads encode grounding targets at each control step. Can attention-based self-monitoring provide real-time grounding quality metrics without external supervision?

## Sources

- Sapkota et al. "Vision-Language-Action (VLA) Models: Concepts, Progress, Applications and Challenges," arXiv:2505.04769, 2025. Comprehensive survey covering 80+ VLA models across five thematic pillars. [https://arxiv.org/abs/2505.04769](https://arxiv.org/abs/2505.04769)

- "Pure Vision Language Action (VLA) Models: A Comprehensive Survey," arXiv:2509.19012, 2025. Taxonomy of VLA paradigms covering 300+ studies. [https://arxiv.org/abs/2509.19012](https://arxiv.org/abs/2509.19012)

- Qwen Team. "Qwen-VLA: Unifying Vision-Language-Action Modeling across Embodiments," arXiv:2605.30280, 2026. [https://arxiv.org/abs/2605.30280](https://arxiv.org/abs/2605.30280)

- Zhang et al. "From Spatial to Actions: Grounding Vision-Language-Action Model in Spatial Foundation Priors (FALCON)," arXiv:2510.17439, 2025. [https://arxiv.org/abs/2510.17439](https://arxiv.org/abs/2510.17439)

- InternVLA-M1: "A Spatially Guided Vision-Language-Action Framework for Generalist Robot Policy," arXiv:2510.13778, 2025. Dual-system VLA with spatial grounding pre-training on 2.3M samples. [https://arxiv.org/html/2510.13778](https://arxiv.org/html/2510.13778)

- Green-VLA: "Staged Vision-Language-Action Model for Generalist Robots," arXiv:2602.00919, 2026. Staged curriculum for humanoid deployment. [https://greenvla.github.io/](https://greenvla.github.io/)

- "Towards Understanding Visual Grounding in Vision-Language Models," arXiv:2509.10345, 2025. Survey of visual grounding in general-purpose VLMs. [https://arxiv.org/html/2509.10345](https://arxiv.org/html/2509.10345)

- G2VLM: "Geometry Grounded Vision Language Model with Unified 3D Reconstruction and Spatial Reasoning," CVPR 2026, InternRobotics. [https://github.com/InternRobotics/G2VLM](https://github.com/InternRobotics/G2VLM)

- GroundingGPT: "Language Enhanced Multi-modal Grounding Model," ACL ARR 2024. [https://openreview.net/forum?id=uWeadjNEVDT](https://openreview.net/forum?id=uWeadjNEVDT)

- Favero et al. "Multi-Modal Hallucination Control by Visual Information Grounding (M3ID)," CVPR 2024. [https://openaccess.thecvf.com/content/CVPR2024/papers/Favero_Multi-Modal_Hallucination_Control_by_Visual_Information_Grounding_CVPR_2024_paper.pdf](https://openaccess.thecvf.com/content/CVPR2024/papers/Favero_Multi-Modal_Hallucination_Control_by_Visual_Information_Grounding_CVPR_2024_paper.pdf)

- "Clutter-Resistant Vision–Language–Action Models," arXiv:2512.22519, 2025. Grounding degradation under VLA fine-tuning. [https://arxiv.org/abs/2512.22519](https://arxiv.org/abs/2512.22519)

- ST4VLA: "Spatially Guided Training for Vision-Language-Action Model," arXiv:2602.10109. [https://arxiv.org/abs/2602.10109](https://arxiv.org/abs/2602.10109)

- VISTA: "Vision-Grounded and Physics-Validated Adaptation," arXiv:2606.04708. [https://arxiv.org/abs/2606.04708](https://arxiv.org/abs/2606.04708)

- NaVILA: "Legged Robot Vision-Language-Action Model for Navigation." [https://navila-bot.github.io/](https://navila-bot.github.io/)

- TacVLA: "Contact-Aware Tactile Fusion for Robust Vision-Language-Action Manipulation," arXiv:2603.12665, 2026. Tri-modal grounding with tactile fusion. [https://arxiv.org/abs/2603.12665](https://arxiv.org/abs/2603.12665)

- IRef-VLA: "Interactive Referential Grounding with Imperfect Language in 3D Scenes," arXiv:2503.17406, WCVC 2026. Grounding benchmark for imperfect language. [https://arxiv.org/html/2503.17406](https://arxiv.org/html/2503.17406)

- Sutton, R. M. et al. "Distilling Internet-Scale Vision-Language Models into Embodied Agents," OpenReview 2022.

- Lewis, P. et al. "PaLM-E: An Embodied Multimodal Language Model," Google DeepMind 2022.

- ProGAL-VLA: "Grounded Alignment through Prospective Reasoning in Vision-Language-Action Models," arXiv:2604.09824, 2026. [https://arxiv.org/abs/2604.09824](https://arxiv.org/abs/2604.09824)

- LabVLA: "Grounding Vision-Language-Action Models in Scientific Laboratories," arXiv:2606.13578, 2026. [https://arxiv.org/abs/2606.13578](https://arxiv.org/abs/2606.13578)

- GIVE: "Grounding Human Gestures in Vision-Language-Action Models," arXiv:2606.13435, 2026. [https://arxiv.org/abs/2606.13435](https://arxiv.org/abs/2606.13435)

- RL Token: "Bootstrapping Online RL with Vision-Language-Action Models," arXiv:2604.23073, 2026. [https://arxiv.org/abs/2604.23073](https://arxiv.org/abs/2604.23073)

- JEPA-VLA: "Joint Embedding Predictive Architecture for Vision-Language-Action Models," arXiv:2602.11832, 2026. [https://arxiv.org/abs/2602.11832](https://arxiv.org/abs/2602.11832)

- Shao et al. "Large VLM-based Vision-Language-Action Models for Robotic Manipulation: A Survey," arXiv:2508.13073, 2025. First taxonomy of large VLM-based VLAs organized around monolithic (single-system, dual-system) and hierarchical architectures. [https://arxiv.org/abs/2508.13073](https://arxiv.org/abs/2508.13073)

- Li et al. "Survey of Vision-Language-Action Models for Embodied Manipulation," arXiv:2508.15201, 2025. Comprehensive review across 5 dimensions: model structures, training datasets, pre-training methods, post-training methods, model evaluation. [https://arxiv.org/abs/2508.15201](https://arxiv.org/abs/2508.15201)

- vla-survey.github.io — Vision-Language-Action Models for Robotics: A Survey. Comprehensive review covering architectures, learning paradigms, and real-world applications. [https://vla-survey.github.io/](https://vla-survey.github.io/)

- MMaDA-VLA: "Large Diffusion Vision-Language-Action Model with Unified Multi-Modal Instruction and Generation," arXiv:2603.25406, 2026. First native discrete diffusion VLA — unifies vision, language, and actions in a single token space via masked generation, achieving 98% LIBERO and shattering CALVIN records. [https://arxiv.org/abs/2603.25406](https://arxiv.org/abs/2603.25406)

- DAM-VLA: "Decoupled Asynchronous Multimodal Vision Language Action Model," arXiv:2606.12105, 2026. Decouples modality update rates for asynchronous grounding. [https://arxiv.org/abs/2606.12105](https://arxiv.org/abs/2606.12105)

- GroundingGPT: "Language Enhanced Multi-modal Grounding Model," ACL ARR 2024. Apple's GEA framework grounds MLLMs across embodiments via multi-embodiment action tokenizers trained with SL + online RL. [https://machinelearning.apple.com/research/grounding-multimodal-large](https://machinelearning.apple.com/research/grounding-multimodal-large)

- VIPA-VLA: "Spatial-Aware VLA Pretraining through Visual-Physical Alignment from Human Videos," CVPR 2026, arXiv:2512.13080. Demonstrates 2D-to-3D visual-physical grounding from human video data. [https://arxiv.org/abs/2512.13080](https://arxiv.org/abs/2512.13080)

- VEGA: "Visual Encoder Grounding Alignment for Spatially-Aware Vision-Language-Action Models," arXiv:2605.10485, 2026. Aligns visual encoder with 3D-aware features before linguistic entanglement. [https://arxiv.org/abs/2605.10485](https://arxiv.org/abs/2605.10485)

- VP-VLA: "Visual Prompting as an Interface for Vision-Language-Action Models," arXiv:2603.22003, 2026. Decouples high-level reasoning from low-level execution via structured visual prompts. [https://arxiv.org/abs/2603.22003](https://arxiv.org/abs/2603.22003)

- Xu et al. "An Anatomy of Vision-Language-Action Models: From Modules to Milestones and Challenges," arXiv:2512.11362, 2025. Structured VLA survey organized by core modules, historical milestones, and five key challenges (multi-modal alignment, instruction following, generalization, safety, data infrastructure). [https://arxiv.org/abs/2512.11362](https://arxiv.org/abs/2512.11362)

- LA4VLA: "Learning to Act without Seeing via Language-Action Priors," arXiv:2606.27295, 2026. Explores language-action grounding without visual input, testing VLA grounding limits when vision is removed. [https://arxiv.org/abs/2606.27295](https://arxiv.org/abs/2606.27295)

- MobileVLA-R1: "Reinforcing Vision-Language-Action for Mobile Robots," arXiv:2511.17889, 2025. RL-reinforced VLA for quadruped navigation, demonstrating grounding transfer to mobile embodiments. [https://arxiv.org/abs/2511.17889](https://arxiv.org/abs/2511.17889)

- Xiaomi-Robotics-0: "An Open-Sourced Vision-Language-Action Model with Real-Time Execution," arXiv:2602.12684, 2026. 4.7B-parameter VLA optimized for real-time control. [https://arxiv.org/abs/2602.12684](https://arxiv.org/abs/2602.12684)

- LangGap: "Diagnosing and Closing the Language Gap in Vision-Language-Action Models," Hou & Zhao, arXiv:2606.02277, 2026. Semantic grounding diagnostic framework. [https://arxiv.org/abs/2606.02277](https://arxiv.org/abs/2606.02277)

- Knowledge Insulating VLMs: "Train Fast, Run Fast, Generalize Better," arXiv:2505.23705, 2025. [https://arxiv.org/abs/2505.23705](https://arxiv.org/abs/2505.23705)

- NVIDIA Developer Blog: "Pretrained to Imagine, Fine-Tuned to Act: The Rise of World-Action Models," 2026. [https://developer.nvidia.com/blog/pretrained-to-imagine-fine-tuned-to-act-the-rise-of-world-action-models/](https://developer.nvidia.com/blog/pretrained-to-imagine-fine-tuned-to-act-the-rise-of-world-action-models/)

## Confidence

**0.92**: High confidence. The core architecture (token-based fusion, cross-modal alignment) and evolution timeline are well-documented across the Sapkota et al. VLA survey (80+ models) and the Pure VLA survey (300+ studies). Qwen-VLA (arXiv:2605.30280), G2VLM (CVPR 2026), InternVLA-M1 (arXiv:2510.13778), and Green-VLA (arXiv:2602.00919) provide concrete, verifiable results for unified grounding, geometry-grounded reasoning, spatially guided training, and staged humanoid deployment. VP-VLA (arXiv:2603.22003), VEGA (arXiv:2605.10485), and VIPA-VLA (CVPR 2026) add three complementary grounding paradigms — visual prompting interfaces, visual encoder-level 3D alignment, and human-video pretraining — that strengthen confidence in the spatial grounding trajectory. Newer work from mid-2026 further solidifies the landscape: LangGap (arXiv:2606.02277) provides a diagnostic framework for language-action alignment; Xiaomi-Robotics-0 (arXiv:2602.12684) demonstrates real-time grounding feasibility; and attention-based self-monitoring opens paths for grounding quality metrics. Confidence remains below 0.95 due to: (a) the scarcity of grounding-specific benchmarks beyond IRef-VLA and LangGap (most evaluation is task-success rate, not grounding accuracy), (b) the nascent state of multi-modal grounding beyond vision-language (gestural, tactile, audio), (c) the open question of grounding fidelity tradeoffs during fine-tuning, which lacks systematic empirical study across model families, and (d) whether JEPA-style decoupling from textual representations meaningfully improves grounding robustness — this remains a promising direction but is not yet empirically validated at scale.