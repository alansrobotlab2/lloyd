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
last_verified: 2026-09-14
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
  - url: "https://arxiv.org/abs/2601.06748"
    title: "TT-VLA: On-the-Fly VLA Adaptation via Test-Time Reinforcement Learning"
  - url: "https://www.themoonlight.io/en/review/evolve-vla-test-time-training-from-environment-feedback-for-vision-language-action-models"
    title: "EVOLVE-VLA: Test-Time Training from Environment Feedback"
  - url: "https://arxiv.org/html/2605.08434"
    title: "Failing Forward: Adaptive Failure-Informed Learning for Vision-Language-Action Models"
  - url: "https://arxiv.org/html/2604.23360"
    title: "Learning from Demonstration with Failure Awareness for Safe Robot Navigation"
  - url: "https://arxiv.org/html/2606.22860"
    title: "HiL-ResRL: A Model-Agnostic Finetuning Adapter via Human-in-the-loop Residual Reinforcement Learning"
  - url: "https://arxiv.org/html/2601.03044"
    title: "SOP: A Scalable Online Post-Training System for Vision-Language-Action Models"
  - url: "https://arxiv.org/html/2606.03127"
    title: "TTT-VLA: Test-Time Latent Prompt Optimization for Vision-Language-Action Models"
  - url: "https://arxiv.org/html/2606.31958"
    title: "Adapting Generalist Robot Policies with Semantic Reinforcement Learning"
  - url: "https://arxiv.org/html/2605.08215"
    title: "T3VF: Test-Time Training Visual Foresight Vision-Language-Action Models"
  - url: "https://arxiv.org/abs/2509.01746"
    title: "Fail2Progress: Learning from Real-World Robot Failures with Stein Variational Inference"
  - url: "https://arxiv.org/abs/2607.01111"
    title: "FAR: Failure-Aware Retry for Test-Time Recovery and Continual Policy Improvement"
  - url: "https://openreview.net/forum?id=e5jGTEiJMT"
    title: "Policy Decorator: Model-Agnostic Online Refinement for Large Policy Model"
  - url: "https://www.pi.website/research/rlt"
    title: "Precise Manipulation with Efficient Online RL (RL Tokens)"
  - url: "https://arxiv.org/abs/2508.12252"
    title: "Robot Trains Robot: Automatic Real-World Policy Adaptation for Humanoids"
  - url: "https://arxiv.org/html/2602.02331"
    title: "TTT-Parkour: Rapid Test-Time Training for Perceptive Robot Parkour"
  - url: "https://arxiv.org/abs/2505.24068"
    title: "DiffCoTune: Differentiable Co-Tuning for Cross-Domain Robot Control"
  - url: "https://arxiv.org/html/2606.31846"
    title: "Z-1: Efficient Reinforcement Learning for Vision-Language-Action Models"
  - url: "https://arxiv.org/abs/2510.02298"
    title: "ARMADA: Autonomous Online Failure Detection and Human Shared Control"
  - url: "https://arxiv.org/abs/2605.11750"
    title: "DreamAvoid: Critical-Phase Test-Time Dreaming to Avoid Failures in VLA Policies"
  - url: "https://arxiv.org/html/2606.09258"
    title: "Back to the Familiar Future: Failure Recovery for VLA Policies via Pre-Imagined Milestone Selection"
  - url: "https://arxiv.org/html/2605.11951"
    title: "From Reaction to Anticipation: Proactive Failure Recovery through Agentic Task Graph"
  - url: "https://arxiv.org/abs/2505.12224"
    title: "RoboFAC: A Comprehensive Framework for Robotic Failure Analysis and Classification"
---

# Real-Time Policy Adaptation for Robots — Online Learning from Failed Interactions

## Summary

Real-time policy adaptation enables robots to refine control policies during deployment by learning from failed interactions, near-misses, and corrective interventions. The dominant paradigm is **offline-to-online reinforcement learning (O2O-RL)**: a pretrained policy is deployed and continuously improved through real-world experience. The central challenge is that online exploration inherently produces failures — spilling objects, entering unsafe states — which are costly in physical environments. Recent work addresses this through failure-aware safety critics with recovery policies, fleet-scale data flywheels, latent-space refinement of diffusion policies, failure-informed negative guidance, and training-free inference-time steering via vision-language models.

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

### Failure-Informed Negative Guidance

- **AFIL / "Failing Forward"** (arXiv:2605.08434): Adaptive Failure-Informed Learning uses failure trajectories as negative guidance for diffusion/flow-based VLA models. Introduces a Dual Action Generator (DAG) architecture: a shared VLM backbone feeds both a success action head (trained on expert demos) and a lightweight failure action head (trained on autonomously generated failure rollouts). During inference, an adaptive guidance scale combines both outputs via $\epsilon^*_{FI} = \epsilon_{succ} - \hat{\lambda}_\eta \cdot \epsilon_{fail}$, steering generation away from failure regions. Achieves 98.4% average success on LIBERO vs. 96.9% baseline, with largest gains on long-horizon and out-of-domain tasks. Failure data is collected via automated rollouts + motion planner corrections (RRT/IK solvers).

### Failure-Aware Learning from Demonstration

- **LfD with Failure Awareness** (arXiv:2604.23360): Addresses the limitation that demonstrations consist predominantly of successful behaviors with limited coverage of unsafe states. Core principle: failure data should not be used for direct policy supervision but should influence policy learning indirectly through value estimation. Enables safe robot navigation when encountering states outside the demonstration distribution.

### Human-in-the-Loop Residual RL

- **HiL-ResRL** (arXiv:2606.22860): Model-agnostic finetuning adapter via human-in-the-loop residual RL. Collects correction data from human operators interacting with deployed policies and trains a compact residual RL adapter without modifying the base policy. Avoids catastrophic forgetting by design — the pretrained policy remains frozen while the residual adapts to real-world distribution shifts.

### Meta-Learning-Based Adaptation

- **OMLA** (arXiv:2503.18684): Online Meta-Learned Adapters learn a shared adapter prior via meta-learning, enabling knowledge transfer between tasks without catastrophic forgetting. Achieves 0.86 average FWT on LIBERO-OBJECT vs. 0.71 for standard LoRA. Uses similarity-based anchor sampling to make meta-learning computationally tractable for vision-language policies.

- **WIZARD** (arXiv:2606.07217): Weight-space inference for zero-shot adaptation from robotic demonstration. A meta-network maps task evidence (language + short demo video) directly to LoRA adapter weights in a single forward pass. Achieves ~2× improvement on unseen dataset collections and ~14× on unseen tasks, with no action labels or test-time optimization required.

### Self-Improvement from Failures

- **PLD** (Xiao et al., ICLR 2026): Probe, Learn, Distill — a three-stage framework for VLA self-improvement. Stage 1: lightweight residual actors probe failure regions of the VLA generalist. Stage 2: residual RL collects targeted data from failure regions. Stage 3: distillation merges improvements back into the base policy. Enables VLAs to self-improve directly from real-world failures without human intervention.

### Inference-Time Steering

- **ITPS** (Wang et al., arXiv:2411.16627): Inference-Time Policy Steering lets frozen generative policies be guided by real-time human interactions (point goals, trajectory sketches, physical corrections) via conditional sampling with likelihood constraints. Stochastic sampling achieves the best alignment-constraint satisfaction trade-off. No fine-tuning needed — the policy remains frozen.

### Test-Time Learning for VLAs

- **TT-VLA** (arXiv:2601.06748): Test-Time RL for VLAs formulates a dense reward mechanism leveraging step-by-step task-progress signals to refine action policies during test time while preserving SFT/RL-trained priors. Enables on-the-fly policy adaptation during inference without full fine-tuning — an effective supplement to current VLA pipelines.
- **EVOLVE-VLA**: Test-time training framework enabling VLA models to continuously adapt through autonomous environment feedback. Operates during deployment, collecting interaction data to iteratively update policies without offline retraining cycles.
- **TTT-VLA** (arXiv:2606.03127): Test-Time Latent Prompt Optimization — a deployment-time improvement framework that learns a latent prompt during training, then performs prompt-only test-time training on a frozen policy. Optimizes the latent prompt on collected interaction data using self-supervised proxy task signals, without modifying the policy itself. Enables deployment-time improvement with consistent success rate gains in single and multi-embodiment settings.
- **T3VF** (arXiv:2605.08215): Test-Time Training Visual Foresight VLA — leverages predicted future image and subsequent observation as a natural supervision pair for test-time training. Addresses vulnerability to visual distribution shifts by using visual foresight as self-supervised signal during deployment.

### Fleet-Scale Online Post-Training

- **SOP** (arXiv:2601.03044): Scalable Online Post-training — a closed-loop actor-learner framework enabling online, distributed, multi-task post-training of VLA models in physical environments. Couples multi-robot parallel deployment ("Parallel Realities") with centralized cloud learning and instant model synchronization. Transforms VLA post-training from offline/single-machine/sequential to online/fleet-based/parallel. Evaluated on heterogeneous Agibot robot fleet.

### Semantic RL Adaptation

- **Semantic-Action RL** (Bhatia et al., arXiv:2606.31958): Adapting generalist robot policies by optimizing prompt inputs with reinforcement learning, enabling efficient real-robot adaptation on complex & long-horizon tasks. Uses semantic action representations to bridge the gap between generalist priors and task-specific online RL, where existing methods struggle with long-horizon adaptation.

### Failure-Informed Simulation Learning

- **Fail2Progress** (Huang et al., CoRL 2025, arXiv:2509.01746): Learning from real-world robot failures using Stein Variational Inference to generate multiple simulation environments in parallel, enabling efficient data sample generation similar to observed failures. Uses skill effect models to translate real-world failure observations into parallelizable sim-based training data.

### Failure-Aware Test-Time Retry

- **FAR** (arXiv:2607.01111): Failure-Aware Retry enables robots to learn from their own failures at test time without human intervention or environment resets. Combines Failure-Contrastive Preference Adaptation (FCPA) with lightweight action perturbations — FCPA identifies failure-inducing action chunks via conservative Q/V critic value drops, then steers the diffusion policy away from those actions using preference learning against perturbed safe alternatives. Successful recovery trajectories feed a continual policy improvement loop with advantage-weighted updates. Achieves +17.6% average success over standard diffusion policies in simulation and +11.7% on real-world xArm tasks.

### Model-Agnostic Residual Refinement

- **PolicyDecorator** (Yuan et al., UC San Diego, arXiv:2412.13630): A model-agnostic framework that refines frozen large pre-trained policies (BeT, Diffusion Policy) via a small learnable residual policy. Final action is the sum of base policy output and residual correction: a = π_base(o) + π_residual(o). Controlled exploration strategies (bounded residual magnitude, progressive exploration schedule) ensure stable online learning. Consistently achieves near-optimal success rates across ManiSkill and Adroit benchmarks, outperforming direct fine-tuning methods that suffer from entropy explosion and unstable critic optimization.

### RL Tokens: Efficient Online RL for Precise Manipulation

- **RLT** (Physical Intelligence, March 2026): RL Tokens extract a compact state representation from frozen VLA models via an encoder-decoder bottleneck. The RL token feeds lightweight actor/critic networks trained with sample-efficient off-policy RL directly on the robot. Actor receives VLA's predicted action as input (edit-then-apply, not replace), with reference-action dropout forcing independent action pathways. Human interventions fold directly into RL updates. Achieves up to 3× speedup on the most precise phases of tasks (screwdriver alignment, zip-tie fastening, ethernet/cord insertion) with as little as 15 minutes of real-world data, surpassing human teleoperation speed on ethernet insertion.

### Autonomous Robot-to-Robot Teaching

- **Robot-Trains-Robot** (Hu et al., CoRL 2025, arXiv:2508.12252): A robot arm teacher actively trains a humanoid student through real-world RL. The arm provides compliant physical guidance via F/T sensing and generates a graduated learning curriculum. Includes automatic reset mechanisms for continuous real-world training. Enables practical and highly efficient real-world humanoid policy adaptation and learning without human intervention.

### Test-Time Training for Perceptive Robot Parkour

- **TTT-Parkour** (arXiv:2602.02331): Rapid test-time training for humanoid robots on unseen terrain. Combines scene capturing, reconstruction, and test-time fine-tuning in simulation to master complex obstacles (wedges, stakes, boxes, trapezoids, narrow beams) within <10 minutes. Demonstrated on Unitree G1, turning initial failure into successful traversal.

### Differentiable Cross-Domain Co-Tuning

- **DiffCoTune** (Krishna et al., IEEE RA-L 2025, arXiv:2505.24068): Automated gradient-based co-tuning framework for cross-domain robot control transfer. Leverages differentiable simulators to tune nominal controllers with <5 real-world trials, bridging sim-to-real gaps through systematic parameter optimization rather than full retraining.

### Efficient RL for Flow-Based VLAs

- **Z-1** (arXiv:2606.31846): Addresses efficient and stable RL post-training for flow-based VLA models. Builds on SOP (scalable online post-training) while focusing on the specific challenges of post-training diffusion/flow-based policies — a gap in the landscape where most post-training methods target autoregressive or token-level architectures.

### Autonomous Online Failure Detection

- **ARMADA** (arXiv:2510.02298): A multi-robot deployment and adaptation system with human-in-the-loop shared control. Features FLOAT — an autonomous online failure detection method enabling robots to detect failures during deployment and invoke human shared control for correction without full handover. Empowers scalable real-world deployment by maintaining robot autonomy while providing safety nets.

### Test-Time Dreaming for Failure Avoidance

- **DreamAvoid** (arXiv:2605.11750): Critical-phase test-time dreaming framework that enables VLA models to anticipate and avoid failures. The base policy executes directly during routine steps; test-time dreaming is invoked only when the system predicts an imminent transition into a critical phase. Reduces unnecessary computational overhead by gating test-time reasoning to failure-prone phases.

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
- **Negative guidance scaling**: For AFIL, does the dual-action-head architecture scale to policies with larger action horizons, and how does guidance stability degrade with more diffusion/flow steps?
- **Failure data contamination**: Can autonomously generated failure rollouts (AFIL) introduce self-reinforcing error modes if the policy repeatedly fails the same way?
- **Fleet-scale synchronization latency**: For SOP, what is the minimum synchronization frequency needed to prevent distribution drift across a heterogeneous fleet?
- **Prompt-only adaptation limits**: How far can TTT-VLA's latent prompt optimization go before the frozen policy becomes a bottleneck?
- **Semantic RL sample efficiency**: Does optimizing only prompt inputs (semantic-action RL) converge faster than full policy fine-tuning, and at what generalization cost?
- **Stein inference scaling**: For Fail2Progress, how does variational inference scale with failure diversity — does the simulation parallelism bottleneck or scale linearly?

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
17. TT-VLA: On-the-Fly VLA Adaptation via Test-Time Reinforcement Learning — [arXiv:2601.06748](https://arxiv.org/abs/2601.06748)
18. EVOLVE-VLA: Test-Time Training from Environment Feedback — [moonlight.io review](https://www.themoonlight.io/en/review/evolve-vla-test-time-training-from-environment-feedback-for-vision-language-action-models)
19. "Failing Forward: Adaptive Failure-Informed Learning for VLAs" — [arXiv:2605.08434](https://arxiv.org/html/2605.08434)
20. "Learning from Demonstration with Failure Awareness for Safe Robot Navigation" — [arXiv:2604.23360](https://arxiv.org/html/2604.23360)
21. "HiL-ResRL: A Model-Agnostic Finetuning Adapter via Human-in-the-loop Residual RL" — [arXiv:2606.22860](https://arxiv.org/html/2606.22860)
22. SOP: Scalable Online Post-Training for VLA Models — [arXiv:2601.03044](https://arxiv.org/html/2601.03044)
23. TTT-VLA: Test-Time Latent Prompt Optimization — [arXiv:2606.03127](https://arxiv.org/html/2606.03127)
24. Bhatia et al., "Adapting Generalist Robot Policies with Semantic Reinforcement Learning" — [arXiv:2606.31958](https://arxiv.org/html/2606.31958)
25. T3VF: Test-Time Training Visual Foresight VLAs — [arXiv:2605.08215](https://arxiv.org/html/2605.08215)
26. Huang et al., "Fail2Progress: Learning from Real-World Robot Failures with Stein Variational Inference" — [arXiv:2509.01746](https://arxiv.org/abs/2509.01746)
27. "FAR: Failure-Aware Retry for Test-Time Recovery and Continual Policy Improvement" — [arXiv:2607.01111](https://arxiv.org/abs/2607.01111)
28. Yuan et al., "Policy Decorator: Model-Agnostic Online Refinement for Large Policy Model" — [OpenReview](https://openreview.net/forum?id=e5jGTEiJMT)
29. ARMADA: Autonomous Online Failure Detection and Human Shared Control — [arXiv:2510.02298](https://arxiv.org/abs/2510.02298)
30. DreamAvoid: Critical-Phase Test-Time Dreaming to Avoid Failures in VLA Policies — [arXiv:2605.11750](https://arxiv.org/abs/2605.11750)
31. Physical Intelligence, "RL Tokens: Precise Manipulation with Efficient Online RL" — [pi.website/research/rlt](https://www.pi.website/research/rlt)
32. Hu et al., "Robot Trains Robot: Automatic Real-World Policy Adaptation for Humanoids" — [arXiv:2508.12252](https://arxiv.org/abs/2508.12252)

## Confidence

**0.88**: High confidence in synthesized coverage. Core findings grounded in peer-reviewed sources (ICLR 2026, RSS 2026, CoRL 2025, arXiv 2025–2026) with real-world robot validation. Multiple independent labs (UT Austin, HKU, Agibot, ICL-ETH Zurich, UC Berkeley, NVIDIA Research) converge on offline-to-online RL with failure-aware adaptation. The fleet-scale SOP framework adds concrete multi-robot synchronization mechanisms. Confidence capped because: (a) real-world open-world deployment remains largely unvalidated beyond controlled benchmarks; (b) fleet-scale approaches (LWD, SOP) lack external replication; (c) TTT-VLA and T3VF are deployment-time frameworks with limited multi-embodiment evaluation; (d) semantic-action RL has not yet been evaluated on open-world tasks; (e) Fail2Progress relies on skill effect model fidelity, which may not transfer across embodiments.