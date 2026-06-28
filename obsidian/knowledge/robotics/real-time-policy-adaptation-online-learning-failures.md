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
date: 2026-07-15
last_verified: 2026-07-15
sources:
  - url: "https://arxiv.org/html/2601.07821"
    title: "FARL: Failure-Aware Offline-to-Online RL with Self-Recovery for Real-World Manipulation"
  - url: "https://finch.agibot.com/research/lwd"
    title: "Learning while Deploying: Fleet-Scale Reinforcement Learning"
  - url: "https://openreview.net/forum?id=DbBD2aT1OG"
    title: "USR: Unified Latent Steering and Residual Refinement for Online Diffusion Policy Improvement"
  - url: "https://arxiv.org/abs/2508.21065"
    title: "Learning on the Fly: Rapid Policy Adaptation via Differentiable Simulation"
  - url: "https://neurips.cc/virtual/2025/loc/san-diego/124003"
    title: "Towards Unified Expressive Policy Optimization for Robust Robot Learning"
---

# Real-Time Policy Adaptation for Robots — Online Learning from Failed Interactions

## Summary

Real-time policy adaptation enables robots to refine their control policies during deployment by learning from failed interactions, near-misses, and corrective human interventions. The dominant paradigm is **offline-to-online reinforcement learning (O2O-RL)**, where a pretrained policy (from demonstrations or simulation) is deployed and continuously improved through real-world experience. The central challenge is that online exploration inherently produces failures — spilling objects, knocking things out of reach, entering unsafe states — which are costly in physical environments. Recent work addresses this through failure-aware safety critics, recovery policies, fleet-scale data flywheels, and lightweight latent-space adaptation that avoids retraining full models.

## Key Findings

### 1. Failure-Aware O2O-RL (FARL) — Preventing failures during exploration
Li et al. (arXiv:2601.07821) introduce **Failure-Aware Offline-to-Online RL (FARL)**, which integrates a world-model-based safety critic and a recovery policy to prevent Intervention-requiring Failures (IR Failures) during online fine-tuning. The safety critic predicts failure states learned from curated failure demonstrations, while the recovery policy steers the agent back to safe state-action pairs before failures materialize. Evaluated on a Franka Emika Panda robot, FARL reduces IR failures by **73.1%** while improving average task performance by **11.3%** over baseline O2O-RL. The framework is built on Uni-O4 (on-policy PPO unification) with an "advantage correction" analysis showing that failure-aware exploration simultaneously improves learning and safety.

### 2. Learning While Deploying (LWD) — Fleet-scale data flywheel
Agibot's LWD framework (2026) treats deployment as a continuous training loop rather than a fixed evaluation. A pretrained VLA policy is deployed across a robot fleet; trajectories — including successes, failures, partial progress, and human interventions — are aggregated into a shared online replay buffer for offline and online RL updates. The updated policy is redeployed, creating a closed-loop data flywheel. Two key algorithmic components: **Distributional Implicit Value Learning (DIVL)** handles heterogeneous fleet data by learning distributions over action values (not scalars), reducing overestimation from OOD maximization. **Q-learning with Adjoint Matching (QAM)** enables policy extraction for flow-based action heads by reformulating critic-guided optimization as local regression along the flow trajectory, avoiding unstable backprop through the full generative process. Evaluated on G1 dual-arm robots across 8 real-world tasks including 3-5 minute long-horizon manipulation.

### 3. USR — Lightweight online refinement of diffusion policies
Zhu et al. (ICLR 2026) propose **Unified Steering and Residual Refinement (USR)** for sample-efficient online adaptation of diffusion-based policies. A lightweight actor outputs (a) latent noise to steer the diffusion process toward promising modes and (b) residual corrections to adapt beyond the pretrained policy's support. This unified design stabilizes training by combining stable mode selection with flexible refinement, avoiding the sample inefficiency of full model fine-tuning. Validated on MultiModalBench and a physical robot VLA improvement task.

### 4. Learning on the Fly — Sub-5-second real-time adaptation
Pan et al. (IEEE RA-L 2026, CoRL 2025) enable rapid policy adaptation via **differentiable simulation**. Online residual dynamics learning models the discrepancy between analytical predictions and real-world measurements, while gradient-based policy optimization updates the policy within the differentiable simulator. Residual dynamics learning, policy adaptation, and real-world deployment run in parallel across multiple threads. The system adapts to unseen disturbances within **5 seconds** of training, validated on agile quadrotor control under various real-world disturbances.

## Failure Modes and Learning Signals

The research identifies several categories of failure-derived learning signals:

| Signal Type | Description | Used By |
|---|---|---|
| **IR Failures** | Intervention-requiring failures (spilled water, broken glass, unreachable objects) | FARL, LWD |
| **Near-misses** | States that would have failed but were avoided by recovery policy | FARL (world model trained on near-miss data) |
| **Partial progress** | Trajectories that made progress but didn't complete | LWD, continual learning frameworks |
| **Human interventions** | Corrections applied by operators during deployment | LWD, RECALL |
| **Residual dynamics mismatch** | Gap between simulated and real dynamics | Learning on the Fly |

## Algorithmic Approaches to Failure Handling

### Safety Critic + Recovery Policy (FARL)
- Train a latent world model on failure trajectories to predict constraint violations
- Train a recovery policy on "escape" demonstrations (how to avoid/recover from near-failure states)
- During online exploration, safety critic monitors states; if failure predicted, recovery policy redirects
- Theoretical justification via advantage correction: safe exploration doesn't degrade learning, it improves it

### Fleet-Scale Replay (LWD)
- Treat all deployment data as training signals (successes + failures + interventions)
- DIVL learns value distributions rather than scalars, handling heterogeneous task mixtures
- Multi-step TD targets propagate sparse terminal rewards through long episodes
- Single generalist policy improved across diverse tasks simultaneously

### Latent-Space Refinement (USR)
- Freeze the pretrained diffusion policy's core parameters
- Train a lightweight actor that steers latent space + applies residual corrections
- Avoids catastrophic forgetting from full fine-tuning
- Sample-efficient: works with limited online interaction budget

### Differentiable Simulation Adaptation (Learning on the Fly)
- Learn residual dynamics online (real world minus simulator)
- Update policy gradient within differentiable simulator
- Sub-second adaptation to new disturbances
- Parallelizes dynamics learning, policy update, and real-world execution

## Key Challenges

1. **IR Failure cost**: In real-world manipulation, failures during exploration often require human intervention (re-resetting objects, cleaning spills). This is the primary bottleneck for deploying O2O-RL beyond controlled labs.

2. **Distribution shift**: The online deployment distribution differs from offline training data. Methods must handle OOD states without performance collapse.

3. **Heterogeneous fleet data**: Multi-robot fleets produce trajectories with different tasks, horizons, reward structures, and intervention patterns. Value learning must be stable across this mixture.

4. **Flow-based policy extraction**: Modern VLA policies use generative action heads (flow matching, diffusion). Traditional RL gradients don't apply cleanly — likelihoods are intractable and backprop through multi-step generation is unstable.

5. **Long-horizon credit assignment**: Failures in multi-minute tasks (e.g., brewing tea, making cocktails) may originate from errors many steps back. Standard TD learning struggles with sparse, delayed failure signals.

## Related

- [[VLA Online Fine-Tuning Continual Learning]] — Continual learning for VLA models with experience replay
- [[Multi-Agent Task Decomposition]] — Hierarchical planning in multi-agent systems
- [[Offline-to-Online Reinforcement Learning]] — O2O-RL as a deployment paradigm
- [[Safe Reinforcement Learning]] — Safe RL constraints and constrained MDPs

## Open Questions

- **Generalization of failure predictors**: How well do world-model-based safety critics generalize to unseen failure modes not covered by training demonstrations?
- **Failure mode coverage**: Can a robot automatically discover novel failure modes during deployment, or does the safety critic need exhaustive failure demonstrations?
- **Cross-task failure transfer**: Does learning to avoid failures in one task (e.g., spilling water) help with related tasks (e.g., pouring juice)?
- **Human intervention integration**: What is the optimal way to incorporate human corrections into the learning loop — as reward signals, demonstration data, or policy constraints?
- **Sample efficiency of failure-based learning**: How many failures does a robot need to encounter before the recovery policy is reliable enough to deploy unsupervised?
- **Real-time vs. batch adaptation**: How does sub-5-second adaptation (Learning on the Fly) compare to fleet-scale batch updates (LWD) in terms of final policy quality and safety guarantees?
- **Multi-robot failure diversity**: Does a fleet of robots with different environments produce complementary failure data that accelerates learning, or does it create noise?
- **Safety-performance tradeoff**: FARL shows both safety improvement and performance gain, but is this always the case? Under what conditions does safety gating limit exploration enough to hurt learning?

## Sources

1. [Li et al., "FARL: Failure-Aware Offline-to-Online RL with Self-Recovery for Real-World Manipulation"](https://arxiv.org/html/2601.07821) — arXiv:2601.07821 (2026). Primary source for failure-aware RL with safety critic + recovery policy. Demonstrates 73.1% IR failure reduction on real robot.

2. [Agibot/AgileX, "Learning While Deploying: Fleet-Scale RL"](https://finch.agibot.com/research/lwd) — Fleet-scale O2O-RL data flywheel. Introduced DIVL and QAM for heterogeneous fleet learning. Evaluated on G1 dual-arm robots.

3. [Zhu et al., "USR: Unified Latent Steering and Residual Refinement"](https://openreview.net/forum?id=DbBD2aT1OG) — ICLR 2026 submission. Lightweight online adaptation for diffusion policies via latent steering + residual refinement.

4. [Pan et al., "Learning on the Fly: Rapid Policy Adaptation via Differentiable Simulation"](https://arxiv.org/abs/2508.21065) — IEEE RA-L 2026, CoRL 2025. Sub-5-second real-time policy adaptation using online residual dynamics learning.

5. [NeurIPS 2025 Workshop, "Towards Unified Expressive Policy Optimization"](https://neurips.cc/virtual/2025/loc/san-diego/124003) — Analysis of O2O-RL challenges: limited multimodal coverage and distributional shift during online adaptation.

## Confidence

**0.75**: Moderate-high confidence. The findings are grounded in peer-reviewed papers (ICLR, CoRL, IEEE RA-L, NeurIPS) and a detailed industry research page (Agibot LWD). FARL and Learning on the Fly include real-world robot validation. Confidence is not higher because (a) real-world failure-based learning is a nascent subfield with limited independent replication, (b) most benchmarks are controlled environments (LIBERO, Franka Panda manipulation) rather than truly open-world deployment, and (c) fleet-scale approaches (LWD) are from a single lab with no external validation yet.
