---
title: Online Fine-Tuning for VLA Models — Continual Learning with Experience Replay
tags:
  - ai/vla-models
  - ai/continual-learning
  - ai/experience-replay
  - robotics/robot-learning
  - ml/online-fine-tuning
  - research/domain-research
created: 2026-07-14
updated: 2026-07-28
confidence: 0.85
---

# Online Fine-Tuning for VLA Models — Continual Learning with Experience Replay

## Summary

Vision-Language-Action (VLA) models present unique opportunities and challenges for online fine-tuning and continual learning in embodied AI. Unlike conventional language models, pretrained VLAs demonstrate surprising natural resistance to catastrophic forgetting during sequential task learning, enabling simpler continual learning strategies than previously assumed. Experience replay — storing and interleaving past task data — emerges as a surprisingly effective technique for pretrained VLAs, often achieving near-zero forgetting with minimal buffer sizes. Beyond basic replay, parameter-efficient approaches like adapter-based routing (CLARE) and continual reinforcement learning frameworks (CRL-VLA) provide structured solutions for long-term deployment of robotic agents in open-world environments where robots must continuously adapt to novel tasks, objects, and environments without retraining from scratch.

## Key Facts

- **Pretrained VLAs resist forgetting naturally**: Liu et al. (arXiv:2603.03818) showed that pretrained VLA models (e.g., Pi0, GR00T) retain latent knowledge from prior tasks even when surface-level performance degrades during sequential fine-tuning, allowing rapid skill recovery via minimal re-fine-tuning. Non-pretrained small policy models (e.g., BC-Transformer) fail much faster under identical conditions.

- **Experience replay works surprisingly well for VLAs**: Simple experience replay (ER) with modest buffer sizes can achieve near-zero forgetting in pretrained VLA models. The replay buffer stores task-specific demonstration data and interleaves it during new-task fine-tuning. Key implementation factors include buffer size, sampling strategy (uniform vs. prioritized), and replay frequency — see Zhu et al. (arXiv:2605.26820).

- **CLARE — exemplar-free continual learning via adapter routing**: Römer et al. (arXiv:2601.09512) propose CLARE (Continual Learning via Adapter Routing and Expansion), which inserts lightweight adapters into VLA modules and autonomously expands them based on layer-wise feature similarity. An autoencoder-based routing mechanism dynamically activates relevant adapters at deployment without task labels. Evaluated on LIBERO and 5 real-world tasks, CLARE outperforms even exemplar-based methods.

- **CRL-VLA — continual reinforcement learning with theoretical bounds**: Zeng et al. (arXiv:2602.03445) frame continual VLA post-training as Continual Reinforcement Learning (CRL), deriving unified performance bounds linking stability-plasticity to goal-conditioned advantage magnitude scaled by policy divergence. Their asymmetric regulation approach constrains advantages on prior tasks while enabling controlled growth on new tasks via a dual-critic architecture (frozen critic + trainable estimator).

- **Simple sequential fine-tuning challenges dogma**: Research shows that simple Sequential Fine-Tuning (Seq. FT) works surprisingly well for CRL in large pretrained VLA models, contradicting the assumption that complex regularization or replay mechanisms are always necessary — see the "Challenging Dogma" analysis by thilak15 on dev.to.

- **Real-world continual VLA learning is underexplored**: Zhu et al. (arXiv:2605.26820) provide the first empirical study of real-world continual VLA learning with a dataset of 4 sequential manipulation tasks (rigid-object pick-and-place, contact-rich pressing, deformable-object folding). They find that VLA models suffer significant catastrophic forgetting from heterogeneous real-world demonstrations, highlighting the gap between simulated and real-world continual learning.

- **RECALL — active uncertainty-based data collection**: RECALL (Recovery Experience Collection for Active Lifelong Learning, Karli & Fitzgerald, arXiv:2606.23617) proposes an active continual learning paradigm where demonstrations are collected from high-uncertainty states rather than passively. The central hypothesis is that uncertainty-targeted collection yields more informative (and efficient) training data, improving data efficiency in continual VLA fine-tuning by focusing replay on states where the model is least confident.

- **PHASER — phase-aware and semantic experience replay**: PHASER (Phase-Aware and Semantic Experience Replay) uses semantic clustering and phase-aware replay scheduling to organize experience buffers by task similarity and episode phase. Evaluated across three VLA backbones on LIBERO continual learning suites, PHASER increases Average Success Rate (ASR) by up to 31% over matched-budget standard ER, achieving 87.8% final ASR on the LIBERO-Goal CL setting. Key design: during online fine-tuning, the base policy is frozen while residual perturbations are learned in the latent space, preserving learned features during adaptation.

- **lifelong-RFT — replay-fine-tuning for long-lived robots**: Yuan Liu et al.'s lifelong-RFT framework achieves a 22% gain in average success rate over standard SFT on LIBERO continual learning benchmarks while adapting to new tasks using only 20% of the training data. The method provides a post-training paradigm where experience replay is integrated with replay-fine-tuning cycles for efficient continual adaptation.

- **Stellar VLA — self-evolving skill knowledge space**: Stellar VLA jointly learns task-centric representations and a self-evolving knowledge space without additional network parameters. Two variants: T-Stellar (task-level) and TS-Stellar (hierarchical task-skill). Requires only ~10MB replay storage per task at 1% replay ratio, making it suitable for deployment-constrained robots with limited on-device storage.

- **Fleet-scale online learning**: AgileX's Learning-while-Deploying (LWD) framework demonstrates fleet-scale RL where multiple robots collect trajectories into a shared online replay buffer, enabling population-level continual learning across distributed deployments.

- **RankQ — offline-to-online RL via self-supervised action ranking**: RankQ (arXiv:2605.11151, Horizon Robotics) replaces uniform pessimistic penalties (CQL/Cal-QL) with structured self-supervised action ranking. It generates synthetic negative examples (noisy, random, permuted actions) and enforces a quality hierarchy via pairwise ranking loss: successful actions > slightly noisy > very noisy > random. This shapes a smoother Q-value landscape, enabling 42.7% higher simulation success and 41.6 percentage-point real-world improvement on cube stacking over baseline VLA policies during online fine-tuning.

- **BORA — bridging offline RL and online residual adaptation**: BORA (arXiv:2605.30226) bridges offline reinforcement learning and online residual adaptation for real-world dexterous VLA models. Instead of full fine-tuning, BORA learns a residual policy on top of a frozen offline-pretrained model, combining the stability of offline pretraining with the adaptability of online RL — a structural parallel to PHASER's frozen-base + residual-latent approach.

- **VLA-OPD — on-policy distillation for online fine-tuning**: VLA-OPD (arXiv:2603.26666) introduces an On-Policy Distillation framework bridging offline Supervised Fine-Tuning (SFT) and online Reinforcement Learning. Instead of relying on sparse environmental rewards, VLA-OPD uses an expert teacher to provide dense, token-level supervision on the student's self-generated trajectories, enabling active error correction on policy-induced states while preserving the offline-pretrained knowledge base — a direct continual learning mechanism for transitioning from offline SFT to online RL.

- **TwinBrainVLA — asymmetric mixture-of-transformers**: TwinBrainVLA (arXiv:2601.14133) addresses catastrophic forgetting during VLA fine-tuning using a dual-pathway architecture: a frozen generalist VLM ("Left Brain") and a trainable specialist ("Right Brain") coordinated via Asymmetric Mixture-of-Transformers (AsyMoT). The Right Brain dynamically queries intact semantic knowledge from the Left Brain while learning task-specific skills, achieving substantial gains on complex manipulation benchmarks (SimplerEnv, RoboCasa) by explicitly retaining general VLM capabilities that are typically lost during standard robotic fine-tuning.

## Method Categories

### Experience Replay (ER) Approaches
| Method | Key Mechanism | Strengths | Limitations |
|--------|---------------|-----------|------------|
| **Standard ER** | Store + uniformly sample past task data | Simple, effective for pretrained VLAs | Buffer management, storage cost |
| **RECALL** | Uncertainty-based active collection | Data-efficient, targets model gaps | Requires uncertainty estimation |
| **PHASER** | Replay-augmented fine-tuning with episodic rehearsal | Maintains multi-task performance | Computationally heavier |
| **CRL-VLA** | Dual-critic with advantage regulation | Theoretical guarantees, RL-native | Complex to implement |

### Parameter-Efficient / Replay-Free Approaches
| Method | Key Mechanism | Strengths | Limitations |
|--------|---------------|-----------|------------|
| **CLARE** | Adapter expansion + autoencoder routing | Exemplar-free, no task labels needed | Requires adapter insertion, routing overhead |
| **Simple Seq. FT** | Direct sequential fine-tuning | Minimal overhead, surprisingly effective | Works best with large pretrained models |

### Offline-to-Online RL Approaches
| Method | Key Mechanism | Strengths | Limitations |
|--------|---------------|-----------|------------|
| **RankQ** | Self-supervised action ranking replaces uniform pessimism | 42.7% sim gain, 41.6% real-world improvement over baseline VLA | Requires offline success/failure split, synthetic negative generation |
| **BORA** | Frozen offline-pretrained model + online residual adaptation | Bridges offline RL safety with online adaptability | Requires offline pretraining phase, residual learning overhead |
| **VLA-OPD** | Expert teacher provides dense token-level supervision on student's self-generated trajectories | Enables active error correction without forgetting offline knowledge | Requires expert teacher model, computational overhead |

### Key Design Factors for Real-World Success (Zhu et al., 2026)
- **Buffer size**: Even small buffers (100-500 samples/task) suffice for pretrained VLAs
- **Sampling strategy**: Prioritized sampling by uncertainty or difficulty improves retention
- **Replay frequency**: Interleaving rate between new and old data strongly impacts stability
- **Data heterogeneity**: Heterogeneous real-world data causes more forgetting than homogeneous simulation data

## Related

- [[VLA models]] — Vision-Language-Action model architecture and training
- [[CLARE]] — Continual Learning via Adapter Routing and Expansion
- [[RECALL]] — Recovery Experience Collection for Active Lifelong Learning
- [[PHASER]] — Replay-Augmented Experience-Guided Fine-Tuning
- [[VLA catastrophic forgetting]] — Catastrophic forgetting in continual learning for VLA models
- [[Stability-plasticity tradeoff]] — Stability-plasticity tradeoff in continual learning
- [[Embodied Agents]] — Embodied agents as a research area

## Open Questions

1. **Optimal replay buffer sizing**: What is the minimum sufficient replay buffer for real-world VLA continual learning, and how does it scale with the number of sequential tasks?

2. **Online vs. offline data quality**: How does the quality and distribution of real-world online demonstrations compare to curated offline datasets for continual learning retention?

3. **Scaling laws for continual VLA learning**: How do model size, training compute, and replay buffer size interact to determine forgetting curves for VLA models?

4. **Adaptation without forgetting**: Can pretrained VLAs learn novel tasks from self-generated experience (online RL) without any external demonstrations, purely through intrinsic motivation and replay?

5. **Multi-robot continual learning**: How should distributed robot fleets share replay buffers and model updates to enable population-level continual learning while respecting heterogeneous deployment conditions?

6. **Task similarity vs. novelty**: How does the semantic similarity between sequential tasks affect forgetting severity and optimal replay strategies?

7. **Long-horizon forgetting**: Most studies evaluate 4-10 sequential tasks. What happens over 50+ task transitions, and do forgetting errors accumulate or saturate?

8. **Hardware-specific adaptation**: Can a single VLA model continually adapt to different robot embodiments (e.g., different arms, grippers, bases) through online fine-tuning, and what architectural changes are needed?

9. **Uncertainty-guided data collection efficiency**: How much can RECALL-style uncertainty-targeted collection reduce the total data needed for continual learning, and does the benefit compound over long task sequences?

10. **Latent-space residual learning**: PHASER-freezes the base policy and learns residual perturbations in latent space. How broadly applicable is this frozen-base + residual-latent approach across different VLA architectures and deployment conditions?

## Sources

- **Liu et al. (2026)** — "Pretrained Vision-Language-Action Models are Surprisingly Resistant to Forgetting in Continual Learning" — arXiv:2603.03818 — [https://arxiv.org/abs/2603.03818](https://arxiv.org/abs/2603.03818) — Project site: [continual-vlas.github.io](https://continual-vlas.github.io/forget-me-not/)

- **Römer et al. (2026)** — "CLARE: Continual Learning for Vision-Language-Action Models via Autonomous Adapter Routing and Expansion" — arXiv:2601.09512 — [https://arxiv.org/abs/2601.09512](https://arxiv.org/abs/2601.09512) — Code: [tum-lsy.github.io/clare](https://tum-lsy.github.io/clare/)

- **Zhu et al. (2026)** — "Can VLA Models Learn from Real-World Data Continually without Forgetting?" — arXiv:2605.26820 — [https://arxiv.org/abs/2605.26820](https://arxiv.org/abs/2605.26820)

- **Zeng et al. (2026)** — "CRL-VLA: Continual Vision-Language-Action Learning" — arXiv:2602.03445 — [https://arxiv.org/abs/2602.03445](https://arxiv.org/abs/2602.03445)

- **OpenVLA-OFT** — "Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Performance" — [openvla-oft.github.io](https://openvla-oft.github.io/)

- **AgileX LWD** — "Learning while Deploying: Fleet-Scale Reinforcement Learning" — [finch.agibot.com/research/lwd](https://finch.agibot.com/research/lwd)

- **thilak15 (2026)** — "Challenging Dogma: Simple Fine-Tuning Enables Continual Learning in VLA Models" — dev.to analysis — [dev.to/thilak15](https://dev.to/thilak15/challenging-dogma-simple-fine-tuning-enables-continual-learning-in-vla-models-1mjj)

- **BemiAgent (2026)** — "Why Pretrained VLAs Almost Never Forget: Continual Learning with Experience Replay" — [bemiagent.com](https://bemiagent.com/agents/vla-continual-learning-agent)

- **Karli & Fitzgerald (2026)** — "RECALL: Recovery Experience Collection for Active Lifelong Learning in Vision-Language-Action Models" — arXiv:2606.23617 — [https://arxiv.org/abs/2606.23617](https://arxiv.org/abs/2606.23617)

- **PHASER authors (2026)** — "PHASER: Phase-Aware and Semantic Experience Replay for Vision-Language-Action Models" — arXiv:2606.03598 — Reinterprets VLA continual learning through Semi-Markov Decision Process (SMDP) lens to expose structural mismatch between trajectory data and uniform frame-level replay. Evaluated on LIBERO-Goal CL, 87.8% final ASR, +31% over matched-budget ER.

- **Stellar VLA (2025)** — "Continually Evolving Skill Knowledge in Vision Language Action Model" — arXiv:2511.18085 — Jointly learns task-centric representations and a self-evolving knowledge space with two variants (T-Stellar for task-level, TS-Stellar for hierarchical task-skill modeling). Requires only ~10MB replay storage per task at 1% replay ratio on LIBERO-scale data.

- **Yuan Liu et al. (2026)** — "Towards Long-Lived Robots: Continual Learning VLA Models via Replay Fine-Tuning" — lifelong-RFT — [yuan-liu-lifelong-rft.github.io](https://yuan-liu-lifelong-rft.github.io/) — 22% ASR gain over SFT on LIBERO CL using 20% training data

- **RankQ (2026)** — "Offline-to-Online Reinforcement Learning via Self-Supervised Action Ranking" — arXiv:2605.11151 — [https://arxiv.org/abs/2605.11151](https://arxiv.org/abs/2605.11151) — Horizon Robotics

- **BORA (2026)** — "Bridging Offline Reinforcement Learning and Online Residual Adaptation for Real-World Dexterous VLA Models" — arXiv:2605.30226 — [https://arxiv.org/abs/2605.30226](https://arxiv.org/abs/2605.30226) — Chen et al.

- **VLA-OPD (2026)** — "VLA-OPD: Bridging Offline SFT and Online RL for Vision-Language-Action Models via On-Policy Distillation" — arXiv:2603.26666 — [https://arxiv.org/abs/2603.26666](https://arxiv.org/abs/2603.26666) — Yu et al.

- **TwinBrainVLA (2026)** — "TwinBrainVLA: Unleashing the Potential of Generalist VLMs for Embodied Tasks via Asymmetric Mixture-of-Transformers" — arXiv:2601.14133 — [https://arxiv.org/abs/2601.14133](https://arxiv.org/abs/2601.14133) — Yu et al. — Code: [github.com/ZGC-EmbodyAI/TwinBrainVLA](https://github.com/ZGC-EmbodyAI/TwinBrainVLA)

## Confidence: 0.85

**Justification**: High confidence (0.85) because the topic is well-covered by recent 2026 arXiv publications with clear abstracts and methodology descriptions. Multiple independent research groups (Liu et al., Römer et al., Zhu et al., Zeng et al.) corroborate the core findings: pretrained VLAs resist forgetting, experience replay is effective, and parameter-efficient methods (CLARE, CRL-VLA) work in both simulation and real-world settings. Confidence is not 1.0 because real-world continual learning is a nascent field — most results are from controlled benchmarks (LIBERO) or limited real-world setups, and long-horizon, open-ended deployment scenarios remain largely unstudied.
