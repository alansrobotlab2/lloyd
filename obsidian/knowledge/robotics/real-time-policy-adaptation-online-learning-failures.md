---
type: medium-research
tags:
  - robotics/policy-adaptation
  - robotics/online-learning
  - robotics/failure-recovery
  - ai/offline-to-online-rl
  - ai/safe-reinforcement-learning
  - ai/continual-learning
domain: robotics
date: 2026-07-28
last_verified: 2026-08-28
sources:
  - url: "https://arxiv.org/html/2601.07821"
    title: "FARL: Failure-Aware Offline-to-Online RL with Self-Recovery for Real-World Manipulation"
  - url: "https://finch.agibot.com/research/lwd"
    title: "Learning while Deploying: Fleet-Scale Reinforcement Learning"
  - url: "https://openreview.net/forum?id=DbBD2aT1OG"
    title: "USR: Unified Latent Steering and Residual Refinement for Online Diffusion Policy Improvement"
  - url: "https://arxiv.org/abs/2508.21065"
    title: "Learning on the Fly: Rapid Policy Adaptation via Differentiable Simulation"
  - url: "https://arxiv.org/abs/2602.03973"
    title: "VLS: Steering Pretrained Robot Policies via Vision-Language Models"
  - url: "https://openreview.net/forum?id=6wd38R8L0Z"
    title: "FINO: Flow Matching with Injected Noise for Offline-to-Online RL"
  - url: "https://arxiv.org/html/2606.05660"
    title: "Safe Embodied AI for Long-horizon Tasks: A Cross-layer Analysis"
  - url: "https://rpg.ifi.uzh.ch/docs/RSS26_Ren.pdf"
    title: "Continual Learning in the Real World (RSS 2026)"
  - url: "https://arxiv.org/abs/2606.27353"
    title: "Continual Robot Policy Learning via Variational Neural Dynamics"
  - url: "https://yuan-liu-lifelong-rft.github.io/"
    title: "Towards Long-Lived Robots: Continual Learning VLA Models via RFT"
  - url: "https://arxiv.org/html/2605.26820"
    title: "Can VLA Models Learn from Real-World Data Continually without Forgetting?"
  - url: "https://arxiv.org/abs/2503.18684"
    title: "OMLA: Efficient Continual Adaptation with Online Meta-Learned Adapters"
  - url: "https://arxiv.org/abs/2606.07217"
    title: "WIZARD: Robotic Policy Adaptation via Weight-Space Meta-Learning"
  - url: "https://rpl.cs.utexas.edu/publications/2026/04/01/xiao-iclr26-pld/"
    title: "PLD: Self-Improving VLAs with Data Generation via Residual RL"
  - url: "https://arxiv.org/html/2411.16627"
    title: "ITPS: Inference-Time Policy Steering through Human Interactions"
---

# Real-Time Policy Adaptation for Robots — Online Learning from Failed Interactions

## Summary

Real-time policy adaptation enables robots to refine control policies during deployment by learning from failed interactions, near-misses, and corrective human interventions. The dominant paradigm is **offline-to-online reinforcement learning (O2O-RL)**: a pretrained policy is deployed and continuously improved through real-world experience. The central challenge is that online exploration inherently produces failures — spilling objects, entering unsafe states — which are costly in physical environments. Recent work addresses this through failure-aware safety critics with recovery policies, fleet-scale data flywheels, latent-space refinement of diffusion policies, differentiable-simulation-based adaptation, and training-free inference-time steering via vision-language models.

## Key Facts

- **FARL** (Li et al., arXiv:2601.07821) integrates a world-model-based safety critic + recovery policy, reducing intervention-requiring failures by 73.1% while improving task performance by 11.3% over baseline O2O-RL on Franka Emika Panda.
- **LWD** (Agibot, 2026) treats deployment as a continuous training loop across robot fleets. Trajectories (successes, failures, interventions) feed a shared replay buffer with Distributional Implicit Value Learning (DIVL) and Q-learning with Adjoint Matching (QAM) for flow-based action heads. Evaluated on G1 dual-arm robots across 8 real-world tasks.
- **USR** (Zhu et al., ICLR 2026) uses a lightweight actor for latent-space steering + residual refinement of frozen diffusion policies, avoiding full fine-tuning and catastrophic forgetting.
- **Learning on the Fly** (Pan et al., IEEE RA-L 2026) achieves sub-5-second policy adaptation via differentiable simulation and online residual dynamics learning, validated on agile quadrotors under real-world disturbances.
- **VLS** (Duan et al., arXiv:2602.03973) enables training-free inference-time adaptation by having VLMs synthesize differentiable reward functions that steer pretrained generative policies — no gradient updates needed.
- **FINO** (ICLR 2026) addresses exploration-collapse in O2O-RL by injecting noise into flow-matching policy training to discover actions beyond the offline dataset.
- **RoboMD** (ICLR 2026 poster) diagnoses robot vulnerabilities via semantic potential fields, enabling proactive avoidance of unobserved failure modes without explicit failure demonstrations.
- **Continual learning** (RSS 2026, Ren et al.) enables online specialization under deployment degradation without overestimating capabilities or imposing unnecessary conservatism.
- **Variational Neural Dynamics** (arXiv:2606.27353) reduces large-disturbance hover and tracking errors by 65.7% and 53.3% over SOTA by sampling diverse dynamics from a variational latent model.
- **VLA continual learning** (Liu et al., RSS 2026): post-training RL fine-tuning achieves 22% success rate gain over SFT on LIBERO using only 20% of training data, with replay-free catastrophic forgetting prevention.
- **Real-world forgetting** (arXiv:2605.26820): forgetting is ~16× worse in physical deployment than simulation suggests, driven by visual similarity and action primitive overlap. Experience replay reduces NBT from +80.0 to +5.0.
- **Pretrained VLA models resist forgetting** (Kim et al., ICLR 2025): GR00T N1.5 achieves near-zero backward transfer (NBT = 0.007), essentially no forgetting. Knowledge is dormant, not erased — recovery fine-tuning restores peak performance in <10% of original steps.

### Meta-Learning-Based Adaptation

- **OMLA** (arXiv:2503.18684): Online Meta-Learned Adapters learn a shared adapter prior via meta-learning, enabling knowledge transfer between tasks without catastrophic forgetting. Achieves 0.86 average FWT on LIBERO-OBJECT vs. 0.71 for standard LoRA. Uses similarity-based anchor sampling to make meta-learning computationally tractable for vision-language policies.

- **WIZARD** (arXiv:2606.07217): Weight-space inference for zero-shot adaptation from robotic demonstration. A meta-network maps task evidence (language + short demo video) directly to LoRA adapter weights in a single forward pass. Achieves ~2× improvement on unseen dataset collections and ~14× on unseen tasks, with no action labels or test-time optimization required.

### Self-Improvement from Failures

- **PLD** (Xiao et al., ICLR 2026): Probe, Learn, Distill — a three-stage framework for VLA self-improvement. Stage 1: lightweight residual actors probe failure regions of the VLA generalist. Stage 2: residual RL collects targeted data from failure regions. Stage 3: distillation merges improvements back into the base policy. Enables VLAs to self-improve directly from real-world failures without human intervention.

### Inference-Time Steering

- **ITPS** (Wang et al., arXiv:2411.16627): Inference-Time Policy Steering lets frozen generative policies be guided by real-time human interactions (point goals, trajectory sketches, physical corrections) via conditional sampling with likelihood constraints. Stochastic sampling achieves the best alignment-constraint satisfaction trade-off. No fine-tuning needed — the policy remains frozen.

## Related (vault entities)

- [[VLA Online Fine-Tuning Continual Learning]] — Continual learning for VLA models with experience replay
- [[Offline-to-Online Reinforcement Learning]] — O2O-RL as a deployment paradigm
- [[Safe Reinforcement Learning]] — Safety constraints and constrained MDPs
- [[Multi-Agent Task Decomposition]] — Hierarchical planning in multi-agent systems
- [[World-Gymnast]] — RL in world models (contrasted with online adaptation)
- [[Online Fine-Tuning]] — Competing pressures: rapid adaptation vs. catastrophic forgetting prevention
- [[Continual Learning]] — Lifelong learning without forgetting
- [[Safe Embodied AI]] — Cross-layer safety analysis for long-horizon tasks

## Open Questions

- **Failure predictor generalization**: How well do world-model safety critics generalize to unseen failure modes beyond training demonstrations?
- **Automatic failure discovery**: Can robots discover novel failure modes autonomously, or does the safety critic need exhaustive failure demonstrations?
- **Cross-task failure transfer**: Does learning to avoid failures in one task help with related tasks?
- **Optimal human intervention integration**: Should corrections be reward signals, demonstrations, or policy constraints?
- **Sample efficiency**: How many failures before a recovery policy is reliable enough for unsupervised deployment?
- **Real-time vs. batch adaptation tradeoffs**: How does sub-5-second adaptation compare to fleet-scale batch updates in final policy quality?
- **Fleet diversity effects**: Does multi-robot failure data accelerate learning or introduce noise?
- **Safety-performance coupling**: Under what conditions does safety gating hurt learning enough to offset its benefits?
- **Real-world VLA forgetting at scale**: How much does forgetting resistance degrade under truly open-world deployment?
- **Scaling laws for continual robot learning**: Is there a relationship between pretrained model scale and catastrophic forgetting robustness?
- **Weight-space generalization**: How far can meta-learned adapter generation (WIZARD) generalize to truly novel task distributions beyond training suite boundaries?
- **PLD failure coverage**: What fraction of real-world failure modes can PLD's residual actors actually discover, and what requires explicit failure demonstrations?
- **Inference-time steering overhead**: At what computational cost does ITPS conditional sampling become impractical for high-frequency control loops?

## Sources

1. Li et al., "FARL: Failure-Aware Offline-to-Online RL with Self-Recovery" — [arXiv:2601.07821](https://arxiv.org/html/2601.07821)
2. Agibot/AgileX, "Learning While Deploying: Fleet-Scale RL" — [finch.agibot.com/research/lwd](https://finch.agibot.com/research/lwd)
3. Zhu et al., "USR: Unified Latent Steering and Residual Refinement" — [OpenReview](https://openreview.net/forum?id=DbBD2aT1OG)
4. Pan et al., "Learning on the Fly: Rapid Policy Adaptation via Differentiable Simulation" — [arXiv:2508.21065](https://arxiv.org/abs/2508.21065)
5. Duan et al., "VLS: Steering Pretrained Robot Policies via Vision-Language Models" — [arXiv:2602.03973](https://arxiv.org/abs/2602.03973)
6. FINO: Flow Matching with Injected Noise — [OpenReview](https://openreview.net/forum?id=6wd38R8L0Z)
7. RoboMD: Semantic Potential Fields for Robot Failure Diagnosis — [arXiv:2606.05660](https://arxiv.org/html/2606.05660)
8. Ren et al., "Continual Learning in the Real World" — [RSS 2026](https://rpg.ifi.uzh.ch/docs/RSS26_Ren.pdf)
9. Continual Robot Policy Learning via Variational Neural Dynamics — [arXiv:2606.27353](https://arxiv.org/abs/2606.27353)
10. Liu et al., "Towards Long-Lived Robots: Continual Learning VLA Models via RFT" — [Project Page](https://yuan-liu-lifelong-rft.github.io/)
11. Zhu et al., "Can VLA Models Learn Continually without Forgetting?" — [arXiv:2605.26820](https://arxiv.org/html/2605.26820)
12. Kim et al., "Pretrained VLAs are Surprisingly Resistant to Forgetting" — [arXiv:2603.03818](https://arxiv.org/abs/2603.03818)
13. OMLA: Efficient Continual Adaptation with Online Meta-Learned Adapters — [arXiv:2503.18684](https://arxiv.org/abs/2503.18684)
14. WIZARD: Robotic Policy Adaptation via Weight-Space Meta-Learning — [arXiv:2606.07217](https://arxiv.org/abs/2606.07217)
15. Xiao et al., "PLD: Self-Improving Vision-Language-Action Models" — [UT Austin ICLR 2026](https://rpl.cs.utexas.edu/publications/2026/04/01/xiao-iclr26-pld/)
16. Wang et al., "ITPS: Inference-Time Policy Steering through Human Interactions" — [arXiv:2411.16627](https://arxiv.org/html/2411.16627)

## Confidence

**0.85**: High confidence in the synthesized coverage. Core findings are grounded in peer-reviewed sources (ICLR 2026, RSS 2026, arXiv 2025–2026) with real-world robot validation. Multiple independent labs (UT Austin, HKU, Agibot, ICL-ETH Zurich) converge on the offline-to-online RL paradigm with failure-aware adaptation. Confidence capped because: (a) real-world open-world deployment remains largely unvalidated beyond controlled benchmarks; (b) fleet-scale approaches (LWD) lack external replication; (c) several sources (FINO, RoboMD) are conference submissions with uncertain acceptance; (d) the simulation-reality gap in forgetting metrics (arXiv:2605.26820) suggests current benchmarks may overstate robustness.