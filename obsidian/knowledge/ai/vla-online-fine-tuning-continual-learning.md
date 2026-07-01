---
type: medium-research
tags: [vla, continual-learning, experience-replay, catastrophic-forgetting, reinforcement-learning, robotics, pretrained-models]
domain: ai
date: 2026-06-29
last_verified: 2026-09-01
sources:
  - url: "https://arxiv.org/abs/2603.03818"
    title: "Pretrained Vision-Language-Action Models are Surprisingly Resistant to Forgetting in Continual Learning"
    authors: "Kim et al."
    accessed: "2026-06-29"
  - url: "https://arxiv.org/abs/2603.11653"
    title: "Simple Recipe Works: Vision-Language-Action Models are Natural Continual Learners with Reinforcement Learning"
    authors: "Hu et al."
    accessed: "2026-06-29"
  - url: "https://arxiv.org/abs/2602.03445"
    title: "CRL-VLA: Continual Vision-Language-Action Learning"
    authors: "Zeng et al."
    accessed: "2026-06-29"
  - url: "https://arxiv.org/abs/2601.09512"
    title: "CLARE: Continual Learning for Vision-Language-Action Models via Autonomous Adapter Routing and Expansion"
    authors: "Rmer et al."
    accessed: "2026-06-29"
  - url: "https://arxiv.org/abs/2606.23617"
    title: "RECALL: Recovery Experience Collection for Active Lifelong Learning"
    authors: "Karli et al."
    accessed: "2026-06-29"
  - url: "https://arxiv.org/abs/2602.10503"
    title: "Towards Long-Lived Robots: Continual Learning VLA Models via Reinforcement Fine-Tuning"
    authors: "Liu et al."
    accessed: "2026-06-29"
  - url: "https://arxiv.org/abs/2603.13335"
    title: "Information-Theoretic Constraints for Continual VLA Alignment"
    authors: "Info-VLA authors"
    accessed: "2026-06-29"
  - url: "https://arxiv.org/abs/2605.26820"
    title: "Can VLA Models Learn from Real-World Data Continually without Forgetting?"
    authors: "Zhu et al."
    accessed: "2026-06-29"
  - url: "https://arxiv.org/abs/2509.22195"
    title: "Actions as Language: Fine-Tuning VLMs into VLAs Without Catastrophic Forgetting"
    authors: "Chen et al."
    accessed: "2026-06-29"
  - url: "https://arxiv.org/abs/2606.25800"
    title: "ROAD-VLA: Robust Online Adaptation via Self-Distillation for Vision-Language-Action Models"
    authors: "Wang et al."
    accessed: "2026-07-07"
  - url: "https://arxiv.org/abs/2606.22999"
    title: "Black-Box Continual Learning for Vision-Language Models"
    authors: "Li et al."
    accessed: "2026-07-07"
  - url: "https://arxiv.org/abs/2606.27374"
    title: "World Action Models Enable Continual Imitation Learning with Generative Replay"
    authors: "WAM-CL authors"
    accessed: "2026-08-07"
  - url: "https://arxiv.org/abs/2604.27063"
    title: "Learning to Forget: Continual Learning with Adaptive Weight Decay"
    authors: "Swiss AI Lab / Univ. of Alberta"
    accessed: "2026-08-07"
  - url: "https://arxiv.org/abs/2606.03598"
    title: "PHASER: Phase-Aware Semantic Experience Replay for VLA Models"
    authors: "Chen et al."
    accessed: "2026-08-07"
  - url: "https://arxiv.org/abs/2606.30988"
    title: "Multisensory Continual Learning: Adapting Pretrained Visuomotor Policies to Force"
    authors: "MuSe authors"
    accessed: "2026-08-09"
---

# Online Fine-Tuning for VLA Models: Continual Learning Without Catastrophic Forgetting

## Summary

Online fine-tuning of Vision-Language-Action (VLA) models enables robots to continuously acquire new skills without forgetting previously learned ones. Research since 2025 has fundamentally overturned the conventional wisdom around continual learning: large-scale pretrained VLAs (Pi0, GR00T, OpenVLA) exhibit **surprisingly low catastrophic forgetting** — near-zero on simulated benchmarks, with simple Experience Replay and no complex regularization. The field has diversified into four methodological families — replay-based, architecture-based, data-representation-based, and information-theoretic — all converging on the insight that pretraining scale reshapes the stability-plasticity trade-off. Critically, real-world experiments reveal that forgetting is **~16× worse in physical deployment** than simulation benchmarks suggest, driven by visual similarity and action primitive overlap.

## Key Facts

- **Pretrained VLAs Resist Forgetting** (Kim et al., ICLR 2025):
  - GR00T N1.5 achieves NBT (Negative Backward Transfer) of 0.007 on LIBERO-Spatial — essentially zero forgetting
  - BC-Transformer (non-pretrained) shows NBT of 0.299 — ~40× more forgetting
  - Simple Experience Replay with just 2% buffer (~100 samples per task) suffices
  - EWC performs **worse** than naive sequential training (NBT: 0.766 vs. 0.752)

- **Simple FT + LoRA + RL Works** (Hu et al., arXiv:2603.11653):
  - Sequential Fine-Tuning with LoRA achieves high plasticity with minimal forgetting
  - Outperforms sophisticated continual RL methods (EWC, Dark Experience Replay, Weight Merge)
  - Synergy between pretrained foundation, parameter-efficient adaptation, and on-policy RL reshapes the stability-plasticity trade-off

- **CRL-VLA Framework** (Zeng et al., arXiv:2602.03445):
  - Identifies **goal-conditioned advantage magnitude** as the key quantity governing stability-plasticity
  - Asymmetric regulation: constrain advantage magnitudes on prior tasks, permit controlled growth on new tasks
  - Dual-critic architecture: frozen critic anchors semantic consistency; trainable estimator drives adaptation

- **CLARE** (Rmer et al., arXiv:2601.09512):
  - Exemplar-free continual learning via lightweight modular adapters in feedforward layers
  - Autonomous expansion guided by layer-wise feature similarity
  - Autoencoder-based routing activates relevant adapters at deployment without task labels

- **RECALL** (Karli et al., arXiv:2606.23617):
  - Shifts from passive imitation to **active, uncertainty-guided data collection**
  - Frames recovery as an active sampling problem: which experiences to collect matters more than how many

- **LifeLong-RFT** (Liu et al., arXiv:2602.10503):
  - Reinforcement Fine-Tuning independent of online environmental feedback
  - Multi-Dimensional Process Reward (MDPR) mechanism
  - 22% gain in average success rate over SFT on LIBERO with only 20% of training data

- **Info-VLA** (arXiv:2603.13335):
  - Identifies **degradation of cross-modal representation structure** as the fundamental cause of forgetting
  - Information-theoretic constraints preserve cross-modal alignment during continual learning
  - First work to frame VLA forgetting through an information-theoretic lens

- **VLM2VLA: Actions as Language** (Chen et al., arXiv:2509.22195, Princeton):
  - Recasts low-level robot actions as natural language descriptions
  - Enables pure LoRA fine-tuning without co-training or architectural modifications
  - Retains >85% VQA performance across standard benchmarks
  - 300 GPU hours on 4×A100s vs. thousands for co-training approaches

- **Real-World Forgetting Is Much Worse** (Zhu et al., arXiv:2605.26820, HKU):
  - First major real-world continual VLA study (AgileX PiPER robot, 4 manipulation tasks)
  - Naive sequential FT drops first-task score from 100.0 to 15.0 (NBT = +80.0) — far worse than simulation
  - Forgetting is structured: driven by **visual similarity** (color overlap → object confusion) and **action primitive overlap** (rigid vs. deformable grasping)
  - Experience Replay with B=0.2, f_r=0.2 reduces NBT from +80.0 to +5.0
  - **Action normalization** is critical: consistent normalization across tasks required for ER to work
  - Simulation benchmarks may suffer from "pretraining overlap" — tasks already in pretraining data appear resistant even when they shouldn't be

- **PHASER** (Chen et al., arXiv:2606.03598):
  - Phase-Aware Semantic Experience Replay for VLA models
  - Phase-centric capacity allocation guarantees equal memory support for all sub-skills
  - Multi-modal interference routing dynamically prioritizes historical phases at high risk of forgetting
  - Increases Average Success Rate by up to 31% over matched-budget ER; 87.8% final ASR on LIBERO-Goal CL
  - Auto-PC: unsupervised change detection + semantic verification for phase boundary detection

- **World Action Models + Generative Replay** (arXiv:2606.27374):
  - Integrates World Action Models with generative replay for continual learning
  - Uses world models to generate synthetic replay data, augmenting buffer-based approaches
  - Explores whether generative replay can substitute for real experience collection

- **NeuroVLA-CL** (AlphaBrain, 2026):
  - Spiking neural networks with biological learning rules (STDP) integrated into VLA pipeline
  - QFormer extracts layer-wise action-relevant features from VLM hidden states
  - Biological learning principles (local plasticity, energy-efficient control) complement foundation-model approaches

- **Knowledge Is Dormant, Not Dead**:
  - When performance drops to 0%, underlying knowledge persists in pretrained VLAs
  - Recovery fine-tuning restores peak performance in <10% of original training steps
  - Non-pretrained models require 100%+ of original time to relearn (true erasure)
  - Forgetting = suppressed execution pathways, not lost representations

- **Vision-Language Backbone Is Primary Forgetting Source**:
  - Component swapping reveals VL backbone is main forgetting site
  - Action head remains more consistent across tasks
  - Forgetting occurs in representation layers, not policy layers

- **Stability-Plasticity Non-Tradeoff**:
  - Pretrained VLAs break the classical stability-plasticity dilemma
  - Achieve both high retention of old tasks AND rapid learning of new tasks
  - Pretraining creates a stable representational foundation that new tasks don't need to disturb

- **ROAD-VLA** (Wang et al., arXiv:2606.25800):
  - **Robust Online Adaptation via Self-Distillation**: addresses sparse-reward online adaptation challenge
  - Self-distillation provides denser training signals for high-dimensional autoregressive action policies
  - Enables continual learning without requiring real environmental feedback at every step
  - Bridges gap between offline pretraining and online fine-tuning via self-generated supervision

- **Black-Box CL for VLMs** (Li et al., arXiv:2606.22999):
  - Proposes Black-CL benchmark for realistic continual learning constraints: weight/architecture inaccessibility, constrained computation, task-agnostic inference
  - BETA method: Semantic Projection Accumulation + Latent Distribution Replay + Test-Time Prototype Adaptation
  - Only 0.05M trainable parameters (180–3000× fewer than white-box CL methods)
  - Achieves performance on par with or exceeding white-box CL methods
  - Relevant for cloud-hosted VLA models where fine-tuning access is restricted

- **FADE: Learning to Forget** (arXiv:2604.27063, Swiss AI Lab / Univ. of Alberta):
  - Proposes per-parameter adaptive weight decay via approximate meta-gradient descent
  - Views weight decay as controlled forgetting: a fixed decay rate is fundamentally mismatched to continual learning
  - Dynamically adapts decay rates so stable knowledge is retained while rapidly changing targets can be updated
  - Complements the VLA CL literature by addressing the "controlled forgetting" problem rather than just preventing it
  - General CL method (not VLA-specific) but directly applicable to VLA online fine-tuning scenarios

- **MuSe** (Clark et al., arXiv:2606.30988):
  - Multisensory continual learning: adapting pretrained visuomotor policies to force-torque sensing
  - Policy pretrained on diverse vision-action data without F/T labels, then adapted with small amount of multisensory data from contact-rich tasks
  - Addresses the modality-shift challenge in continual learning when extending VLAs to force-controlled manipulation
  - Explores how much multisensory data is needed for successful cross-modal adaptation

## Related (vault entities)

- [[Experience Replay]] — core mechanism for continual learning, surprisingly effective on pretrained VLAs
- [[Catastrophic Forgetting]] — classical ML problem; largely mitigated by pretraining scale in VLAs
- [[VLA Models]] — Vision-Language-Action models (Pi0, GR00T, OpenVLA)
- [[LoRA]] — parameter-efficient fine-tuning, central to most VLA CL approaches
- [[EWC]] — Elastic Weight Consolidation, underperforms on pretrained VLAs
- [[LIBERO]] — standard evaluation suite for robotic manipulation continual learning
- [[VLM2VLA]] — data-centric approach: actions as language
- [[CLARE]] — exemplar-free continual learning via adapter routing
- [[RECALL]] — uncertainty-guided active experience collection
- [[CRL-VLA]] — dual-critic continual RL framework
- [[Info-VLA]] — information-theoretic constraints for cross-modal alignment
- [[NeuroVLA-CL]] — brain-inspired continual learning checkpoint
- [[ROAD-VLA]] — self-distillation-based online adaptation
- [[Black-Box CL]] — continual learning under API-only constraints
- [[Multi-Modal Grounding]] — language-to-action mapping in VLAs
- [[Self-Improving Agents]] — broader context for continual learning in autonomous systems

## Open Questions

- **Real-World Generalization**: How do CL methods scale to truly open-world deployment with ambiguous task boundaries and uncontrolled environments?
- **Pretraining Overlap Confound**: Are current NBT metrics artificially optimistic because simulation benchmarks partially overlap with pretraining data? What happens with truly novel tasks?
- **Buffer Management in Practice**: How should replay buffers be managed in unconstrained environments with continuous data streams?
- **Transfer to Non-Robotics Agents**: Do the same pretraining-resistance properties apply to LLM agents fine-tuned on domain-specific tasks?
- **Long-Horizon Sequences**: How does forgetting behavior scale with hundreds of sequential tasks rather than 4-10 task suites?
- **Self-Correction Systems**: If knowledge is dormant rather than lost, what mechanisms could detect and trigger recovery fine-tuning automatically?
- **Scaling Laws for Forgetting**: How does forgetting resistance scale with model size, pretraining data volume, and task diversity?
- **Active vs. Passive Learning**: RECALL suggests active collection outperforms passive replay — what's the optimal exploration strategy?
- **Information-Theoretic Guarantees**: Can we derive tighter bounds on what information must be preserved during continual learning?
- **Biological vs. Foundation Hybrid**: Can biologically-inspired local plasticity rules complement large-model pretraining for continual learning?
- **Action Normalization Standards**: What normalization schemes are necessary for cross-task transfer? (Critical per real-world study)
- **Task Boundaries**: How do VLA models handle continual learning when task boundaries are ambiguous or overlapping?
- **Generative Replay**: Can World Action Models generate synthetic replay data to augment buffer-based approaches?
- **Phase-Aware Replay**: Does phase-centric capacity allocation + interference routing (PHASER) outperform uniform sampling at scale?
- **Self-Distillation for Online Adaptation**: Can self-distillation (ROAD-VLA) provide sufficient supervision for high-dimensional action policies when real rewards are sparse?
- **Black-Box Constraints**: How do continual learning methods perform when the backbone is inaccessible (cloud-hosted VLA APIs)? Can prototype-only methods (BETA) match white-box performance in practice?

## Sources

- Kim et al. (2025). "Pretrained Vision-Language-Action Models are Surprisingly Resistant to Forgetting in Continual Learning." ICLR 2025. arXiv:2603.03818. [Paper](https://arxiv.org/abs/2603.03818) [Project](https://continual-vlas.github.io/forget-me-not/)
- Hu et al. (2026). "Simple Recipe Works: Vision-Language-Action Models are Natural Continual Learners with Reinforcement Learning." arXiv:2603.11653. [Paper](https://arxiv.org/abs/2603.11653) [Peter Stone Group](https://www.cs.utexas.edu/~pstone/Papers/bib2html/b2hd-hu_crlvla2026.html)
- Liu et al. (2026). "Towards Long-Lived Robots: Continual Learning VLA Models via Reinforcement Fine-Tuning." arXiv:2602.10503. [Project](https://yuan-liu-lifelong-rft.github.io/)
- Rmer et al. (2026). "CLARE: Continual Learning for Vision-Language-Action Models via Autonomous Adapter Routing and Expansion." arXiv:2601.09512. [Code](https://github.com/utiasDSL/clare)
- Zeng et al. (2026). "CRL-VLA: Continual Vision-Language-Action Learning." arXiv:2602.03445. [Paper](https://arxiv.org/html/2602.03445v1)
- Karli et al. (2026). "RECALL: Recovery Experience Collection for Active Lifelong Learning." arXiv:2606.23617. [Paper](https://arxiv.org/abs/2606.23617)
- AlphaBrainGroup (2026). "NeuroVLA-CL" — brain-inspired continual learning. [HF](https://huggingface.co/AlphaBrainGroup/neurovla-cl-libero-goal)
- **Zhu et al. (2026). "Can VLA Models Learn from Real-World Data Continually without Forgetting?"** arXiv:2605.26820. [Paper](https://arxiv.org/abs/2605.26820)
- **Chen et al. (2025). "Actions as Language: Fine-Tuning VLMs into VLAs Without Catastrophic Forgetting."** arXiv:2509.22195. [Project](https://vlm2vla.github.io/) [ICLR 2026 Poster](https://iclr.cc/virtual/2026/poster/10007076)
- **Info-VLA (2026). "Information-Theoretic Constraints for Continual VLA Alignment."** arXiv:2603.13335. [Paper](https://arxiv.org/abs/2603.13335)

## Confidence

**0.88**: Core findings are well-supported by multiple peer-reviewed papers (ICLR 2025 acceptance, 8+ arXiv submissions from UT Austin, Princeton, KAIST, TUM, HKU, and others). The Kim et al. paper provides concrete quantitative results (NBT metrics, recovery rates). CRL-VLA provides theoretical grounding. The real-world study (HKU) provides critical grounding in physical deployment, confirming simulation-reality gaps. VLM2VLA offers a fundamentally different data-centric approach validated on standard VQA benchmarks. Info-VLA provides a new information-theoretic lens. Confidence limited because: real-world open-world deployment remains largely unvalidated beyond 4-task studies, the pretraining overlap confound hasn't been systematically quantified, and long-horizon (>10 task) sequences remain unstudied.