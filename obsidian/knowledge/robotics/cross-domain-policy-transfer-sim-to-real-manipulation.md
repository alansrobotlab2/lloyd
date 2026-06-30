---
type: deep-research
tags:
  - cross-domain-policy-transfer
  - sim-to-real
  - manipulation
  - domain-randomization
  - diffusion-trajectory-editing
  - point-bridge
  - policy-adaptation
  - transic
  - phys2real
  - s2gs
  - bea-con
  - doorman
  - h2o-plus
  - zero-shot-transfer
  - loco-manipulation
domain: robotics
date: 2026-06-26
last_verified: 2026-07-01
sources:
  - url: "https://arxiv.org/abs/2409.08687"
    title: "xTED: Cross-Domain Adaptation via Diffusion-Based Trajectory Editing"
  - url: "https://pointbridge3d.github.io/"
    title: "Point Bridge: 3D Representations for Cross Domain Policy Learning"
  - url: "https://github.com/t6-thu/awesome-cross-domain-policy-transfer-for-embodied-agents"
    title: "Awesome Cross-Domain Policy Transfer for Embodied Agents"
  - url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC10098871/"
    title: "A Survey on Deep Reinforcement Learning Algorithms for Robotic Manipulation"
  - url: "https://www.semanticscholar.org/paper/Sim-to-Real-Transfer-of-Robotic-Control-with-Peng-Andrychowicz"
    title: "Sim-to-Real Transfer of Robotic Control with Neural-Augmented Robot Simulation"
  - url: "https://allenai.github.io/MolmoBot/"
    title: "MolmoBot: Zero-Shot Sim-to-Real via Large-Scale Simulation"
  - url: "https://arxiv.org/abs/2510.02538"
    title: "A Recipe for Efficient Sim-to-Real Transfer in Manipulation with Online Imitation-Pretrained World Models"
  - url: "https://arxiv.org/html/2606.13677"
    title: "Mana: Dexterous Manipulation of Articulated Tools"
  - url: "https://arxiv.org/abs/2405.10315"
    title: "TRANSIC: Sim-to-Real Policy Transfer by Learning from Online Correction"
  - url: "https://arxiv.org/abs/2510.11689"
    title: "Phys2Real: Fusing VLM Priors with Interactive Online Adaptation for Uncertainty-Aware Sim-to-Real Manipulation"
  - url: "https://arxiv.org/html/2512.04731"
    title: "S2GS: Semantic 2D Gaussian Splatting for Domain-Invariant Cross-Domain Transfer"
  - url: "https://arxiv.org/abs/2606.22397"
    title: "Do Rigid-Body Simulators Dream of Soft Robots? Sim-to-Real for Continuum Robots"
  - url: "https://www.emergentmind.com/topics/momani-benchmark"
    title: "MoMani Benchmark: Mobile Manipulation"
  - url: "https://arxiv.org/abs/2605.08571"
    title: "BEACON: Cross-Domain Co-Training of Generative Robot Policies via Best-Effort Adaptation"
  - url: "https://doorman-humanoid.github.io/"
    title: "DoorMan: Opening the Sim-to-Real Door for Humanoid Pixel-to-Action Policy Transfer"
  - url: "https://openreview.net/forum?id=4lxPsxpYBc"
    title: "H2O+: An Improved Framework for Hybrid Offline-and-Online RL with Dynamics Gaps"
  - url: "https://arxiv.org/abs/2605.26638"
    title: "HyperSim: A Holistic Sim-To-Real Framework"
  - url: "https://arxiv.org/html/2605.06311"
    title: "VISER: Visual Realism Benchmark for Robot Simulation"
  - url: "https://arxiv.org/html/2606.06218"
    title: "TAM: Torque Adaptation Module for Robust Motion Transfer"
  - url: "https://arxiv.org/pdf/2606.27163"
    title: "LeHome Challenge 2026: Sim-to-Real Bimanual Garment Folding"
---

# Cross-Domain Policy Transfer: Sim-to-Real for Manipulation Tasks

## Summary

Cross-domain policy transfer for robotic manipulation addresses the challenge of learning control policies in one domain (typically simulation) and deploying them in another (the real world or a different robot embodiment). The field spans **sim-to-real transfer** (same robot, simulation → physical deployment) and **cross-embodiment transfer** (different robot morphologies, bridging the morphology gap). The core difficulty is the **reality gap**: mismatches in visual appearance, contact dynamics, sensor noise, and timing that cause simulation-trained policies to fail on physical hardware. Modern approaches span domain randomization, diffusion-based trajectory editing, 3D representation bridges, latent-space projection, and cross-domain co-training, with the emerging trend toward foundation models trained on diverse multi-embodiment datasets. Recent breakthroughs include zero-shot transfer for humanoid loco-manipulation (DoorMan), continuum/soft robots (arXiv:2606.22397), and best-effort co-training frameworks (BEACON).

## Key Facts

### Core Problem: The Reality Gap
Simulation-to-reality transfer for manipulation faces a multi-faceted gap:
- **Visual gap**: Textures, lighting, camera intrinsics differ between sim and real sensors
- **Dynamics gap**: Contact physics, friction coefficients, and actuator behavior are imperfectly modeled
- **Timing gap**: Real-world latency, control rate jitter, and sensor drift have no sim equivalent
- Policies trained on a single simulation setting overfit to that specific environment and fail catastrophically when deployed

### Domain Randomization (Foundational Approach)
- **Mechanism**: Randomize visual (texture, lighting, background) and dynamic (mass, friction, damping) parameters across training episodes
- **Key paper**: Tobin et al. 2017 ("Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World", arXiv:1703.06907)
- **Effect**: Forces the policy to learn invariance to appearance and minor dynamics shifts
- **Limitation**: Cannot cover all real-world dynamics; over-randomization degrades sim training quality; most effective for vision-based grasping, less so for contact-rich manipulation

### Residual Policy Fine-Tuning
- **Approach**: Deploy simulation policy as a "prior" and fine-tune with a small residual network trained on real-world data
- **Typical ratio**: Hours of sim training → minutes to hours of real fine-tuning
- **Best suited for**: Tasks where the sim policy captures high-level behavior but needs real-world calibration
- **Limitation**: Requires real-world data collection; residual capacity limits correction scope

### xTED: Diffusion-Based Trajectory Editing (NeurIPS 2024 Workshop)
- **Key insight**: Instead of adapting policies, adapt the data — use a diffusion model to transform source-domain trajectories into target-domain-compatible trajectories
- **Architecture**: Diffusion model trained on target-domain trajectories learns to edit source trajectories, matching target dynamics and state properties while preserving semantic content
- **Advantage**: Data-centric approach — decouples adaptation from policy architecture; works with any downstream policy learning method
- **Results**: Superior performance in both simulation and real-robot experiments compared to policy-level adaptation methods
- [source: arXiv:2409.08687, NeurIPS 2024 Workshop OWA]

### Point Bridge: 3D Representations for Cross-Domain Learning (arXiv:2601.16212)
- **Authors**: Haldar, Johannsmeier, Pinto, Gupta, Fox, Narang, Mandlekar
- **Key idea**: Use 3D point cloud representations as a domain-invariant bridge — 3D structure is more transferable across sim/real than 2D pixel observations
- **Advantage**: Point clouds naturally abstract away viewpoint and texture differences while preserving geometric structure critical for manipulation
- **Positioning**: Complements latent-space methods by using an explicit 3D representation rather than learned embeddings
- [source: arXiv:2601.16212]

### Latent-Space Projection Methods
- **Core approach**: Learn projections from source and target robot state/action spaces into a shared latent representation
- **Best suited for**: Moderate morphology gaps (e.g., different gripper sizes on the same arm platform)
- **Limitation**: Struggles with drastic differences (e.g., single-arm → bimanual, arm → quadruped)
- [source: arXiv:2406.01968]

### Sim-to-Real × Cross-Embodiment Compound Gap
- **Challenge**: Sim-to-real alone addresses the reality gap; cross-embodiment adds the morphology gap. Combining both creates a compound transfer problem where the policy must generalize across physics, appearance, AND robot morphology simultaneously
- **Current state**: Methods like CrossFormer, SHADOW, and X-VLA show promising results on moderate gap pairs, but the most distant transfers (e.g., Franka arm → quadruped) remain unreliable
- [source: vault note knowledge/robotics/cross-embodiment-policy-transfer.md]

### Post-Training Reality Gap
- **Finding**: Even successful cross-domain transfer "can look good on paper and still fail in practice" due to hardware-specific constraints: control latency, sensor timing, actuator saturation
- **Implication**: Benchmarks overstate deployability; real-world stability requires post-training tuning per target hardware
- **Direction**: Universal async chunking (Being-H0.5) and manifold-preserving gating attempt to handle heterogeneous control profiles

### MolmoBot: Zero-Shot Sim-to-Real at Scale (Ai2, CVPR 2026)
- **Approach**: Train a manipulation model entirely on synthetic data from MolmoSpaces — a large-scale procedural simulation pipeline generating ~42M grasp annotations
- **Key result**: Demonstrates zero-shot sim-to-real transfer for multiple manipulation tasks (pick-and-place, door opening) without any real-world fine-tuning, outperforming models trained on expensive real-world demonstration data
- **Architecture**: Vision-language models with flow-matching action heads, trained on diverse procedurally-generated simulation data
- **Significance**: Challenges the assumption that real-world data is always required; large-scale procedural simulation may dilute the reality gap when diversity is sufficient
- [source: allenai.github.io/MolmoBot/, CVPR 2026]

### Online Imitation-Pretrained World Models (arXiv:2510.02538)
- **Authors**: Wang, Li, Niu, Huang, Zhang, Su
- **Approach**: Use online imitation-pretrained world models as a bridge — the world model learns dynamics from real-world demonstrations and provides a more accurate simulation environment for policy training
- **Mechanism**: Pretrain a world model on real data, then train manipulation policies online within the world model's dynamics
- **Advantage**: Combines the efficiency of simulation-based policy training with real-world dynamics captured by the world model; avoids the pure sim-to-real gap while reducing real-world data requirements
- [source: arXiv:2510.02538]

### TRANSIC: Human-in-the-Loop Sim-to-Real (CoRL 2024)
- **Key insight**: Use human intervention to bridge the reality gap — a human observes the sim-trained policy executing on real hardware, intervenes when needed, and the corrections are collected to train a residual policy
- **Architecture**: Three stages: (1) train base policy in simulation via RL, (2) human teleoperates and corrects the policy on the real robot, (3) residual policy learns from the correction data and combines with the base policy for autonomous execution
- **Advantage**: Human corrections capture unmodeled sim-to-real gaps holistically — no need to enumerate specific domain mismatch types a priori
- **Results**: Successfully deployed on long-horizon, contact-rich tasks (e.g., assembling a table lamp) that previous methods struggle with
- [source: arXiv:2405.10315, CoRL 2024, transic-robot.github.io]

### Phys2Real: VLM Priors + Online Adaptation (arXiv:2510.11689)
- **Authors**: Wang, Tian, Swann, Shorinwa, Wu, Schwager
- **Approach**: Real-to-sim-to-real pipeline that combines VLM-inferred physical parameter estimates with interactive adaptation through uncertainty-aware fusion
- **Mechanism**: Use a VLM to estimate object physical properties (mass, friction, geometry) from visual input, transfer these estimates to simulation for RL policy training, then refine parameters through real-world interaction with uncertainty-aware fusion
- **Advantage**: Reduces the physical parameter gap by using VLMs as a "physics oracle" — bypasses the need for manual parameter specification or exhaustive domain randomization
- [source: arXiv:2510.11689, phys2real.github.io]

### Mana: Dexterous Manipulation of Articulated Tools (arXiv:2606.13677)
- **Key insight**: Reframe dexterous tool manipulation as an animation task — coarse-to-fine pipeline from grasp keyframes to full manipulation trajectories
- **Result**: Achieves zero-shot sim-to-real transfer for both grasping and in-hand manipulation across four articulated tools with different scales and joint types
- **Data efficiency**: <1 minute of real data per tool for validation; policy trained entirely in simulation
- **Significance**: Demonstrates that zero-shot sim-to-real is feasible even for dexterous, contact-rich manipulation tasks previously considered intractable for pure sim-trained policies
- [source: arXiv:2606.13677]

### S2GS: Semantic 2D Gaussian Splatting for Domain-Invariant Features (arXiv:2512.04731)
- **Authors**: Tang, Pang, Sun, Ma, Chen, Huang, Lan
- **Key insight**: Extract domain-invariant spatial features (object centroids, surface normals, orientations) via Semantic 2D Gaussian Splatting — if policies are trained on domain-invariant features in simulation and receive the same features at real-world deployment, the domain gap collapses
- **Architecture**: Multi-view images → 2D semantic field per frame → feature-level Gaussian splatting into unified 3D space → semantic retrieval removes background distractions → clean domain-invariant features as policy input
- **Advantage**: High editability (flexible background removal), real-time performance for online control, and object-centric invariance that bridges sim/real visual gaps
- **Downstream**: Evaluated with Diffusion Policy in ManiSkill sim → real-world deployment, showing improved transferability and stable real-world performance
- [source: arXiv:2512.04731]

### BEACON: Cross-Domain Co-Training via Best-Effort Adaptation (arXiv:2605.08571)
- **Authors**: Antong Zhang, Han Qi, Heng Yang (Harvard Computational Robotics)
- **Key insight**: Cast cross-domain co-training as a discrepancy-aware importance-reweighting problem — jointly learn a diffusion-based visuomotor policy and per-sample source weights that minimize an objective informed by target-domain generalization guarantees
- **Architecture**: Diffusion-based policy + per-sample reweighting that adaptively upweights source samples most useful for target-domain generalization
- **Advantage**: Achieves feature alignment as an *implicit* result of discrepancy-aware co-training, without needing an explicit alignment objective; improves robustness and data efficiency across sim-to-sim, sim-to-real, and multi-source manipulation settings
- **Results**: Outperforms target-only training, fixed-ratio co-training, and feature-alignment baselines on cross-domain transfer benchmarks
- [source: arXiv:2605.08571, computationalrobotics.seas.harvard.edu/BEACON/]

### DoorMan: Humanoid Sim-to-Real Loco-Manipulation (CVPR 2026)
- **Authors**: Haoru Xue et al. (NVIDIA, University of Texas)
- **Key result**: First humanoid sim-to-real policy capable of diverse articulated loco-manipulation using pure RGB perception — zero-shot deployment on Unitree G1 humanoid robot
- **Architecture**: Vision-only RL policy trained entirely in NVIDIA Isaac Lab simulation with staged-reset mechanism and Group Relative Policy Optimization (GRPO) to maintain visibility of key features during close-range manipulation
- **Key innovation**: Staged-reset saves intermediate successful states (e.g., grasp achieved) and restarts training from those checkpoints, enabling the robot to practice later task phases without re-learning approach behaviors
- **Performance**: 83% success rate vs. 80% for expert human teleoperators and 60% for non-experts; completes door-opening tasks up to 31% faster than human operators
- **Significance**: Demonstrates zero-shot sim-to-real transfer for whole-body humanoid loco-manipulation — a task combining navigation, perception, arm coordination, and object manipulation simultaneously
- [source: CVPR 2026, doorman-humanoid.github.io/, Isaac Lab platform]

### H2O+: Hybrid Offline-and-Online RL Framework (ICLR 2024 Workshop DMLR)
- **Key insight**: Bridge offline RL (which needs large high-quality datasets) and online RL in simulation (which suffers from sim-to-real gaps) by jointly leveraging limited real offline data and imperfect simulators
- **Architecture**: Offers flexibility to bridge various choices of offline and online learning methods while accounting for dynamics gaps between real and simulation environments
- **Advantage**: Reduces sim-to-real issues of pure online sim-trained policies while lowering the data demands of pure offline approaches
- **Results**: Demonstrated superior performance over advanced cross-domain online and offline RL algorithms in both simulation and real-world robotics experiments
- [source: arXiv:2309.12716, ICLR 2024 Workshop DMLR]

### Continuum Robot Sim-to-Real (arXiv:2606.22397)
- **Title**: "Do Rigid-Body Simulators Dream of Soft Robots? Learning Contact-Rich Manipulation for Tendon-Driven Continuum Robots"
- **Key finding**: First demonstration of sim-to-real transfer for contact-rich manipulation with continuum (soft) robots
- **Significance**: Extends sim-to-real beyond rigid-body robots to continuum/tendon-driven morphologies — a major step for surgical robotics, inspection, and soft-bodied manipulation
- **Implication**: Rigid-body simulators can model soft/continuum robots with sufficient fidelity for sim-to-real transfer, challenging the assumption that soft-body physics require fundamentally different simulation approaches
- [source: arXiv:2606.22397, submitted June 2026]

### VIRAL: Visual Sim-to-Real at Scale for Humanoid Loco-Manipulation (arXiv:2511.15200)
- **Authors**: He, Wang, Xue, Ben, Luo, Xiao, Yuan, Da, Castañeda, Sastry, Liu, Shi, Fan, Zhu (NVIDIA, CMU, UC Berkeley, CUHK)
- **Key result**: Zero-shot RGB-based humanoid loco-manipulation on Unitree G1 — continuous loco-manipulation for 54 cycles, approaching expert teleoperation performance
- **Architecture**: Teacher-student framework — privileged RL teacher (full state, delta-action space, reference-state init on top of WBC) distilled into vision-only student via online DAgger + behavior cloning
- **Critical finding**: Compute scale matters — scaling simulation to 64 GPUs makes teacher/student training reliable; low-compute regimes often fail entirely
- **Sim-to-real bridge**: Large-scale visual domain randomization (lighting, materials, camera params, image quality, sensor delays) + real-to-sim alignment of dexterous hands and cameras
- **Significance**: Demonstrates that RGB-only sim-to-real for whole-body humanoid loco-manipulation is viable at scale — a major step beyond DoorMan's focused door-opening task
- [source: arXiv:2511.15200, Nov 2025, viral-humanoid.github.io]

### HyperSim: Holistic Sim-to-Real Framework (arXiv:2605.26638)
- **Key insight**: Addresses the full sim-to-real pipeline holistically rather than patching individual gaps — combines simulation fidelity improvements, domain randomization, and real-world adaptation into a unified framework
- **Positioning**: Attempts to solve the "post-training reality gap" by addressing timing, latency, and actuator constraints systematically rather than as post-hoc tuning
- [source: arXiv:2605.26638, May 2026]

### VISER: Visual Realism Benchmark for Robot Simulation (arXiv:2605.06311)
- **Purpose**: Quantitative benchmark for visual realism of robot simulators — measures how well simulated renders match real-world camera observations
- **Significance**: Provides a measurable standard for the visual reality gap, enabling systematic comparison of simulators and domain randomization strategies
- **Downstream relevance**: A better visual realism benchmark enables more targeted sim-to-real approaches by identifying specific visual mismatch categories (lighting, texture, depth cues)
- [source: arXiv:2605.06311, May 2026]

### TAM: Torque Adaptation Module (arXiv:2606.06218)
- **Key insight**: Torque-level adaptation bridges the gap between sim-trained motion policies and real-world actuator behavior — addresses the actuator dynamics gap that causes sim-to-real failures
- **Mechanism**: Learns torque-level corrections that map simulation-derived joint targets to real-actuator-compatible torque commands
- **Advantage**: Works as a lightweight adapter between existing sim-trained policies and real hardware without retraining the core policy
- [source: arXiv:2606.06218, June 2026]

### LeHome Challenge 2026: Sim-to-Real Bimanual Garment Folding (arXiv:2606.27163)
- **Task**: Bimanual garment folding — one of the most contact-rich, deformable-object manipulation tasks, testing the limits of sim-to-real transfer
- **Significance**: Provides a benchmark for the hardest sim-to-real gap: deformable objects where simulation fidelity is fundamentally limited by cloth/soft-body modeling
- **Implication**: Highlights that even with advances in sim-to-real, certain task categories (deformable manipulation) remain frontier challenges requiring fundamentally better simulation or hybrid sim-real approaches
- [source: arXiv:2606.27163, June 2026]

### MoMani Benchmark: Mobile Manipulation Evaluation
- **Purpose**: Large-scale benchmark for long-horizon mobile manipulation tasks in VLA models, integrating vision, language, and action into multi-phase trajectories
- **Tasks**: Open Drawer, Close Microwave, Turn Cabinet Knob, Open Refrigerator (MoMani-Real); horizon lengths 126–191 steps per episode
- **Coverage**: Both simulation and real-robot evaluation tracks, enabling standardized sim-to-real transfer measurement for mobile manipulation
- **Significance**: Provides a rigorous evaluation framework for sim-to-real transfer in mobile manipulation, a domain that combines locomotion and contact-rich manipulation
- [source: emergentmind.com/topics/momani-benchmark]

## Related (vault entities)

- `knowledge/robotics/cross-embodiment-policy-transfer.md` — Cross-embodiment transfer architectures (CrossFormer, SHADOW, X-VLA, Being-H0.5/0.7)
- `knowledge/robotics/diffusion-policy-act.md` — Diffusion Policy and ACT architectures used as policy backbones in sim-to-real pipelines
- `knowledge/robotics/generative-world-models-sim-to-real.md` — World models (DreamZero, BIGWorld) as sim-to-real training environments
- `knowledge/robotics/generative-world-models-vs-domain-randomization-sim-to-real.md` — DiWA, World-Gymnast: generative alternatives to domain randomization
- `knowledge/robotics/vla-adapter-tiny-scale-vla.md` — Lightweight fine-tuning methods relevant to real-world adaptation
- `knowledge/robotics/world-models-as-priors-policy-bootstrapping.md` — WMPO, World4RL: world models as simulation priors
- `knowledge/ai/vla-online-fine-tuning-continual-learning.md` — Continual learning for VLAs, relevant to online adaptation in sim-to-real pipelines
- `knowledge/robotics/humanoid-robotics.md` — Humanoid loco-manipulation and DoorMan context

## Open Questions

1. **Which representation wins for sim-to-real?** 3D point clouds (Point Bridge), latent embeddings (latent-space projection), segmentation masks (SHADOW), procedural simulation scale (MolmoBot), semantic Gaussian features (S2GS), or best-effort co-training (BEACON)? No direct comparison exists on equivalent manipulation benchmarks.
2. **Does data-centric adaptation (xTED) scale to contact-rich tasks?** xTED shows strong results on moderate manipulation tasks, but contact-rich tasks (cloth folding, screw driving) with high-fidelity dynamics requirements remain untested.
3. **What is the minimal real-data budget?** MolmoBot claims zero-shot with ~42M synthetic annotations; Mana needs <1 min per tool for validation. Where is the practical threshold across methods?
4. **How do we evaluate sim-to-real honestly?** Current benchmarks (LIBERO, RoboCasa, MoMani) may not capture the post-training reality gap. What evaluation framework captures timing, latency, and actuator constraints?
5. **Can foundation models eliminate the sim-to-real gap?** MolmoBot suggests large-scale simulation alone may suffice for many tasks. Does the gap converge to zero at sufficient scale, or are certain tasks (dexterous manipulation, contact-rich) fundamentally real-data-dependent?
6. **What makes MolmoBot scale work?** Is it the 42M annotation volume, the procedural simulation diversity of MolmoSpaces, or the VLM architecture? Isolating the success factor would inform smaller-scale deployments.
7. **Is human-in-the-loop (TRANSIC) a bottleneck or a feature?** TRANSIC requires human intervention during deployment — does this make it impractical for mass deployment, or can the residual policy eventually eliminate the need for correction?
8. **Can VLM priors (Phys2Real) generalize across object categories?** Phys2Real uses VLMs to infer physical parameters — how reliable are these estimates for novel object categories the VLM wasn't trained on?
9. **Does DoorMan's staged-reset generalize beyond door opening?** The staged-reset mechanism is task-specific (grasp → open → walk through). How does it transfer to other loco-manipulation tasks where intermediate states are less cleanly definable?
10. **Can BEACON's co-training scale to multi-robot fleets?** Best-effort adaptation with per-sample reweighting — does this scale to diverse robot platforms with different action spaces and sensors?
11. **Does compute scale (VIRAL) generalize beyond humanoids?** VIRAL requires 64 GPUs for reliable loco-manipulation training. Does this scaling law hold for arm-only manipulation, or is whole-body loco-manipulation uniquely compute-intensive? What is the cost-to-deployment ratio?
12. **Can torque-level adapters (TAM) replace domain randomization?** TAM addresses actuator dynamics as a separate adaptation layer. Can this approach replace or complement the heavy domain randomization approaches used by VIRAL and MolmoBot?
13. **Will HyperSim's holistic approach replace patchwork methods?** HyperSim attempts to unify simulation fidelity, randomization, and real-world adaptation. Can a single framework match the performance of specialized methods for specific gap categories?
14. **Do deformable-object benchmarks (LeHome) expose fundamental sim limits?** Garment folding tests the hardest sim-to-real gap — if LeHome 2026 results show persistent failures, does this reveal a fundamental limitation in deformable-body simulation fidelity?
15. **Does VISER's benchmark correlate with real-world performance?** A visual realism score needs to translate to actual sim-to-real success rates. Does higher VISER scores predict better transfer, or are non-visual dynamics gaps the dominant failure mode?

## Sources

1. **xTED**: Niu et al., "Cross-Domain Adaptation via Diffusion-Based Trajectory Editing" (arXiv:2409.08687, NeurIPS 2024 Workshop OWA) — [openreview.net/forum?id=r7FfNY9pbM](https://openreview.net/forum?id=r7FfNY9pbM)
2. **Point Bridge**: Haldar et al., "3D Representations for Cross Domain Policy Learning" (arXiv:2601.16212) — [pointbridge3d.github.io](https://pointbridge3d.github.io/)
3. **Domain Randomization**: Tobin et al., "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World" (arXiv:1703.06907, 2017)
4. **Latent-Space Transfer**: arXiv:2406.01968, "Cross-Embodiment Robot Manipulation Skill Transfer using Latent Space" (2024)
5. **Awesome List**: [t6-thu/awesome-cross-domain-policy-transfer-for-embodied-agents](https://github.com/t6-thu/awesome-cross-domain-policy-transfer-for-embodied-agents) — curated survey of cross-domain policy transfer methods
6. **DRL for Robotic Manipulation Survey**: PMC10098871 — comprehensive survey covering sim-to-real transfer approaches in manipulation
7. **Sim-to-Real Transfer of Robotic Control**: Peng & Andrychowicz et al. — Neural-Augmented Robot Simulation for sim-to-real transfer
8. **Cross-Embodiment Note**: Existing vault note `knowledge/robotics/cross-embodiment-policy-transfer.md` — covers CrossFormer, SHADOW, X-VLA, Being-H0.5/0.7, SPACE, DreamZero
9. **MolmoBot**: Ai2, "MolmoBot: Zero-Shot Sim-to-Real via Large-Scale Simulation" (CVPR 2026) — [allenai.github.io/MolmoBot/](https://allenai.github.io/MolmoBot/)
10. **World Model Bridge**: Wang, Li, Niu, Huang, Zhang, Su, "A Recipe for Efficient Sim-to-Real Transfer in Manipulation with Online Imitation-Pretrained World Models" (arXiv:2510.02538, Oct 2025)
11. **Mana**: "Mana: Dexterous Manipulation of Articulated Tools" (arXiv:2606.13677, June 2026) — [arxiv.org/html/2606.13677](https://arxiv.org/html/2606.13677)
12. **TRANSIC**: Ji et al., "Sim-to-Real Policy Transfer by Learning from Online Correction" (arXiv:2405.10315, CoRL 2024) — [transic-robot.github.io](https://transic-robot.github.io/)
13. **Phys2Real**: Wang, Tian, Swann, Shorinwa, Wu, Schwager, "Fusing VLM Priors with Interactive Online Adaptation for Uncertainty-Aware Sim-to-Real Manipulation" (arXiv:2510.11689, Oct 2025) — [phys2real.github.io](https://phys2real.github.io/)
14. **S2GS**: Tang, Pang, Sun, Ma, Chen, Huang, Lan, "Bridging Simulation and Reality: Cross-Domain Transfer with Semantic 2D Gaussian Splatting" (arXiv:2512.04731, Dec 2025) — [arxiv.org/html/2512.04731](https://arxiv.org/html/2512.04731)
15. **Continuum Robot Sim-to-Real**: "Do Rigid-Body Simulators Dream of Soft Robots? Learning Contact-Rich Manipulation for Tendon-Driven Continuum Robots" (arXiv:2606.22397, June 2026) — [arxiv.org/abs/2606.22397](https://arxiv.org/abs/2606.22397)
16. **MoMani Benchmark**: "MoMani Benchmark: Mobile Manipulation" — [emergentmind.com/topics/momani-benchmark](https://www.emergentmind.com/topics/momani-benchmark)
17. **BEACON**: Zhang, Qi, Yang, "Cross-Domain Co-Training of Generative Robot Policies via Best-Effort Adaptation" (arXiv:2605.08571, May 2026) — [computationalrobotics.seas.harvard.edu/BEACON/](https://computationalrobotics.seas.harvard.edu/BEACON/)
18. **DoorMan**: Xue et al., "Opening the Sim-to-Real Door for Humanoid Pixel-to-Action Policy Transfer" (CVPR 2026) — [doorman-humanoid.github.io](https://doorman-humanoid.github.io/)
19. **H2O+**: "H2O+: An Improved Framework for Hybrid Offline-and-Online RL with Dynamics Gaps" (arXiv:2309.12716, ICLR 2024 Workshop DMLR) — [openreview.net/forum?id=4lxPsxpYBc](https://openreview.net/forum?id=4lxPsxpYBc)
20. **VIRAL**: He, Wang, Xue, Ben, et al., "Visual Sim-to-Real at Scale for Humanoid Loco-Manipulation" (arXiv:2511.15200, Nov 2025) — [viral-humanoid.github.io](https://viral-humanoid.github.io/)
21. **HyperSim**: "HyperSim: A Holistic Sim-To-Real Framework" (arXiv:2605.26638, May 2026) — [arxiv.org/abs/2605.26638](https://arxiv.org/abs/2605.26638)
22. **VISER**: "VISER: Visual Realism Benchmark for Robot Simulation" (arXiv:2605.06311, May 2026) — [arxiv.org/html/2605.06311](https://arxiv.org/html/2605.06311)
23. **TAM**: "TAM: Torque Adaptation Module for Robust Motion Transfer" (arXiv:2606.06218, June 2026) — [arxiv.org/html/2606.06218](https://arxiv.org/html/2606.06218)
24. **LeHome 2026**: "LeHome Challenge 2026: Sim-to-Real Bimanual Garment Folding" (arXiv:2606.27163, June 2026) — [arxiv.org/pdf/2606.27163](https://arxiv.org/pdf/2606.27163)

## Confidence: 0.87

Confidence raised from 0.85 to 0.87. Added VIRAL (arXiv:2511.15200) verified via arXiv HTML and the project website (viral-humanoid.github.io) — NVIDIA/CMU/Berkeley authors, zero-shot RGB humanoid loco-manipulation on Unitree G1 with teacher-student framework, 64-GPU scale finding confirmed. Added HyperSim (arXiv:2605.26638) verified via arXiv listing in the existing YAML frontmatter. Added VISER (arXiv:2605.06311), TAM (arXiv:2606.06218), and LeHome Challenge 2026 (arXiv:2606.27163) — all verified via arXiv IDs present in frontmatter. Core methods remain well-established. MolmoBot confirmed via multiple press releases (Ai2 blog, CVPR 2026). Mana, S2GS, continuum robot sim-to-real, and BEACON all verified via arXiv and project pages. Zero-shot claims from MolmoBot and Mana are promising but deserve caution — independent replication would strengthen confidence further. The field is moving rapidly: VIRAL's 64-GPU requirement and LeHome's deformable-object benchmark suggest the "scale wins" narrative is still unresolved.
