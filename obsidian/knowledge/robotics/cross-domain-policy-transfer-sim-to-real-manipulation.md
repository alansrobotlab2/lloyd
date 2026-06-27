---
type: quick-research
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
  - continuum-robot-sim-to-real
  - momani-benchmark
domain: robotics
date: 2026-06-26
last_verified: 2026-07-08
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
  - url: "https://knowledge/robotics/cross-embodiment-policy-transfer.md"
    title: "Cross-Embodiment Policy Transfer (existing vault note)"
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
---

# Cross-Domain Policy Transfer: Sim-to-Real for Manipulation Tasks

## Summary

Cross-domain policy transfer for robotic manipulation encompasses the challenge of learning control policies in one domain (typically simulation) and deploying them effectively in another (the real world or a different robot embodiment). The field spans **sim-to-real transfer** (same robot, simulation → physical deployment) and **cross-embodiment transfer** (different robot morphologies, bridging the morphology gap). The core difficulty is the **reality gap**: mismatches in visual appearance, contact dynamics, sensor noise, and timing that cause simulation-trained policies to fail on physical hardware. Modern approaches span domain randomization, diffusion-based trajectory editing, 3D representation bridges, and latent-space projection, with the emerging trend toward foundation models trained on diverse multi-embodiment datasets.

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

### Continuum Robot Sim-to-Real (arXiv:2606.22397)
- **Title**: "Do Rigid-Body Simulators Dream of Soft Robots? Learning Contact-Rich Manipulation for Tendon-Driven Continuum Robots"
- **Key finding**: First demonstration of sim-to-real transfer for contact-rich manipulation with continuum (soft) robots
- **Significance**: Extends sim-to-real beyond rigid-body robots to continuum/tendon-driven morphologies — a major step for surgical robotics, inspection, and soft-bodied manipulation
- **Implication**: Rigid-body simulators can model soft/continuum robots with sufficient fidelity for sim-to-real transfer, challenging the assumption that soft-body physics require fundamentally different simulation approaches
- [source: arXiv:2606.22397, submitted June 2026]

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

## Open Questions

1. **Which representation wins for sim-to-real?** 3D point clouds (Point Bridge), latent embeddings (latent-space projection), segmentation masks (SHADOW), or procedural simulation scale (MolmoBot)? No direct comparison exists on equivalent manipulation benchmarks.
2. **Does data-centric adaptation (xTED) scale to contact-rich tasks?** xTED shows strong results on moderate manipulation tasks, but contact-rich tasks (cloth folding, screw driving) with high-fidelity dynamics requirements remain untested.
3. **What is the minimal real-data budget?** MolmoBot claims zero-shot with ~42M synthetic annotations; Mana needs <1 min per tool for validation. Where is the practical threshold across methods?
4. **How do we evaluate sim-to-real honestly?** Current benchmarks (LIBERO, RoboCasa) may not capture the post-training reality gap. What evaluation framework captures timing, latency, and actuator constraints?
5. **Can foundation models eliminate the sim-to-real gap?** MolmoBot suggests large-scale simulation alone may suffice for many tasks. Does the gap converge to zero at sufficient scale, or are certain tasks (dexterous manipulation, contact-rich) fundamentally real-data-dependent?
6. **What makes MolmoBot scale work?** Is it the 42M annotation volume, the procedural simulation diversity of MolmoSpaces, or the VLM architecture? Isolating the success factor would inform smaller-scale deployments.
7. **Is human-in-the-loop (TRANSIC) a bottleneck or a feature?** TRANSIC requires human intervention during deployment — does this make it impractical for mass deployment, or can the residual policy eventually eliminate the need for correction?
8. **Can VLM priors (Phys2Real) generalize across object categories?** Phys2Real uses VLMs to infer physical parameters — how reliable are these estimates for novel object categories the VLM wasn't trained on?

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

## Confidence: 0.83

Confidence raised from 0.82 to 0.83. Core methods remain well-established. MolmoBot is confirmed via multiple press releases (Ai2 blog, The Letter, CVPR 2026) and the allenai project page, though full technical details from the CVPR paper text have not been read. Mana (arXiv:2606.13677) and the world-model recipe (arXiv:2510.02538) are verified via arXiv abstracts and supplementary project pages. S2GS (arXiv:2512.04731) verified via full HTML text on arXiv. Continuum robot sim-to-real (arXiv:2606.22397) verified via arXiv abstract — full text pending. Some specifics (exact annotation counts, architecture details) come from press summaries rather than peer-reviewed full text. The zero-shot claims from MolmoBot and Mana are promising but deserve caution — zero-shot performance claims need independent replication to be considered reliable.
