# Cross-Domain Policy Transfer — Sim-to-Real for Manipulation Tasks

## Summary

Cross-domain policy transfer addresses the challenge of learning robot control policies in simulation (or another domain) and deploying them on physical hardware. The core obstacle is the **reality gap** — mismatches in visual appearance, contact dynamics, sensor noise, and timing that cause simulation-trained policies to fail on real robots. Modern methods span domain randomization, diffusion-based trajectory editing, 3D representation bridges, latent-space projection, world-model-bridged training, and human-in-the-loop correction. The field is rapidly converging toward foundation models trained at massive scale on procedural simulation data, with several approaches demonstrating zero-shot sim-to-real transfer for increasingly complex tasks.

## Key Facts

- **Domain randomization** (Tobin et al., 2017) remains the foundational approach: randomize visual (textures, lighting, camera pose) and dynamic (mass, friction, damping) parameters during training to force policy invariance. Effective for vision-based grasping but struggles with contact-rich manipulation; over-randomization degrades training quality. Practitioners typically start with visual randomization alone, then incrementally add physics and dynamics axes. A well-configured pipeline uses YAML/JSON randomization configs with logged per-episode parameters for post-hoc failure analysis. [Tobin et al. arXiv:1703.06907; Claru DR pipeline guide]

- **xTED** (NeurIPS 2024 Workshop): Data-centric adaptation via diffusion-based trajectory editing. Transforms source-domain trajectories into target-domain-compatible ones by learning a diffusion model on target data. Decouples adaptation from policy architecture, working with any downstream learning method. [Niu et al. arXiv:2409.08687]

- **Point Bridge** (arXiv:2601.16212): Uses 3D point clouds as domain-invariant representations that abstract away viewpoint/texture differences while preserving geometric structure critical for manipulation. Complements latent-space methods by using explicit 3D representations. [Haldar et al.]

- **MolmoBot** (CVPR 2026, Allen AI): Zero-shot sim-to-real via large-scale procedural simulation (~42M synthetic grasp annotations from MolmoSpaces). Outperforms models trained on expensive real-world demonstration data. Challenges the assumption that real-world data is always required. [allenai.github.io/MolmoBot/]

- **DoorMan** (CVPR 2026): First zero-shot humanoid loco-manipulation on Unitree G1 — vision-only RL with staged-reset mechanism, 83% success rate vs 80% for expert teleoperators. Completes door-opening 31% faster than humans. [doorman-humanoid.github.io/]

- **VIRAL** (arXiv:2511.15200): Teacher-student framework for RGB-only humanoid loco-manipulation. Critical finding: compute scale matters — 64 GPUs required for reliable teacher/student training; low-compute regimes fail entirely. [viral-humanoid.github.io/]

- **BEACON** (arXiv:2605.08571): Cross-domain co-training via discrepancy-aware importance reweighting. Achieves feature alignment implicitly through co-training without explicit alignment objectives; outperforms fixed-ratio co-training and feature-alignment baselines on sim-to-real manipulation benchmarks. [Zhang, Qi, Yang, Harvard]

- **TRANSIC** (CoRL 2024): Human-in-the-loop correction — train base policy in simulation via RL, human teleoperates and corrects on real robot, residual policy learns from corrections for autonomous execution. Captures unmodeled sim-to-real gaps holistically. Achieves 81% average success across four real-robot tasks vs 45% for best baseline. [Ji et al. arXiv:2405.10315]

- **S2GS** (arXiv:2512.04731): Semantic 2D Gaussian Splatting extracts domain-invariant spatial features (centroids, normals, orientations) that collapse sim/real visual gaps. Real-time performance for online control. [Tang et al.]

- **Phys2Real** (arXiv:2510.11689): Uses VLMs as "physics oracles" to estimate object properties (mass, friction, geometry) from visual input, transfers to simulation for RL training, then refines through real-world interaction with uncertainty-aware fusion. [Wang et al.]

- **Mana** (arXiv:2606.13677): Zero-shot sim-to-real for dexterous manipulation of articulated tools. Coarse-to-fine pipeline from grasp keyframes to full trajectories. <1 min real data per tool for validation. [arXiv:2606.13677]

- **Continuum robot sim-to-real** (arXiv:2606.22397): First demonstration of sim-to-real transfer for tendon-driven continuum (soft) robots. Rigid-body simulators model soft morphologies with sufficient fidelity for deployment. [arXiv:2606.22397]

- **TAM** (arXiv:2606.06218): Torque adaptation module — lightweight adapter between sim-trained policies and real hardware that addresses actuator dynamics without retraining the core policy. [arXiv:2606.06218]

- **Post-training reality gap**: Even successful cross-domain transfer can fail on physical hardware due to control latency, sensor timing jitter, and actuator saturation not captured in simulation. Benchmarks overstate deployability; real-world stability requires per-hardware post-training tuning.

- **Two-phase training recipe** (industry standard): Phase 1 — pretrain on 50K-500K domain-randomized synthetic episodes. Phase 2 — fine-tune on 500-5K real-world demonstrations at 0.1x-0.01x learning rate. Hybrid approach consistently outperforms either method alone.

## Related (vault entities)

- `knowledge/robotics/cross-embodiment-policy-transfer.md` — Cross-embodiment transfer architectures
- `knowledge/robotics/diffusion-policy-act.md` — Diffusion Policy and ACT as policy backbones
- `knowledge/robotics/generative-world-models-sim-to-real.md` — World models as training environments
- `knowledge/robotics/vla-online-fine-tuning-continual-learning.md` — Continual learning for VLAs
- `knowledge/robotics/humanoid-robotics.md` — Humanoid loco-manipulation context

## Open Questions

1. **Which representation wins?** 3D point clouds, latent embeddings, segmentation masks, semantic Gaussian features, or best-effort co-training? No head-to-head comparison exists on equivalent benchmarks.
2. **Does data-centric adaptation (xTED) scale to contact-rich tasks?** Cloth folding, screw driving, and deformable-object manipulation remain untested.
3. **Minimal real-data budget**: MolmoBot claims zero-shot with 42M synthetic annotations; Mana needs <1 min per tool. Where is the practical threshold?
4. **Can foundation models eliminate the sim-to-real gap?** Does the gap converge to zero at sufficient scale, or are certain tasks fundamentally real-data-dependent?
5. **Is human-in-the-loop (TRANSIC) a bottleneck or feature?** Can the residual policy eventually eliminate correction needs?
6. **Can VLM priors (Phys2Real) generalize to novel object categories?** Reliability of VLM-inferred physics for unseen objects remains untested.
7. **Does compute scale (VIRAL's 64 GPUs) generalize beyond humanoids?** Is whole-body loco-manipulation uniquely compute-intensive?
8. **Can torque adapters (TAM) replace domain randomization?** Or do they complement it?
9. **Will HyperSim's holistic approach replace patchwork methods?**
10. **Do deformable-object benchmarks (LeHome 2026) expose fundamental simulation limits?**

## Sources

1. Tobin et al., "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World" (arXiv:1703.06907, 2017)
2. Niu et al., "Cross-Domain Adaptation via Diffusion-Based Trajectory Editing" (arXiv:2409.08687, NeurIPS 2024 Workshop)
3. Haldar et al., "Point Bridge: 3D Representations for Cross Domain Policy Learning" (arXiv:2601.16212)
4. Allen AI, "MolmoBot: Zero-Shot Sim-to-Real via Large-Scale Simulation" (CVPR 2026)
5. Xue et al., "DoorMan: Opening the Sim-to-Real Door for Humanoid Pixel-to-Action Policy Transfer" (CVPR 2026)
6. He et al., "VIRAL: Visual Sim-to-Real at Scale for Humanoid Loco-Manipulation" (arXiv:2511.15200)
7. Zhang, Qi, Yang, "BEACON: Cross-Domain Co-Training via Best-Effort Adaptation" (arXiv:2605.08571)
8. Ji et al., "TRANSIC: Sim-to-Real Policy Transfer by Learning from Online Correction" (arXiv:2405.10315, CoRL 2024)
9. Tang et al., "S2GS: Semantic 2D Gaussian Splatting" (arXiv:2512.04731)
10. Wang et al., "Phys2Real: Fusing VLM Priors with Interactive Online Adaptation" (arXiv:2510.11689)
11. "Mana: Dexterous Manipulation of Articulated Tools" (arXiv:2606.13677)
12. "Do Rigid-Body Simulators Dream of Soft Robots?" (arXiv:2606.22397)
13. "TAM: Torque Adaptation Module" (arXiv:2606.06218)
14. "HyperSim: A Holistic Sim-To-Real Framework" (arXiv:2605.26638)
15. "VISER: Visual Realism Benchmark" (arXiv:2605.06311)
16. "LeHome Challenge 2026: Sim-to-Real Bimanual Garment Folding" (arXiv:2606.27163)
17. Wang, Li, Niu et al., "Efficient Sim-to-Real with Online Imitation-Pretrained World Models" (arXiv:2510.02538)
18. t6-thu/awesome-cross-domain-policy-transfer-for-embodied-agents (GitHub curated survey)
19. Claru, "How to Set Up a Domain Randomization Pipeline" (claru.ai/guides/how-to-setup-domain-randomization-pipeline) — practitioner guide on DR pipeline setup
20. Muratore et al., "Robot Learning From Randomized Simulations: A Review" (PMC9038844, 2022)

## Confidence

0.87: Core methods (domain randomization, residual fine-tuning, latent-space transfer) are well-established with multiple independent replications. Emerging 2024-2026 methods (MolmoBot, DoorMan, VIRAL, BEACON, TRANSIC, S2GS, Mana) verified via arXiv IDs, project websites, and institutional publications (NVIDIA, Allen AI, Harvard). Zero-shot claims from MolmoBot and Mana are promising but deserve caution — independent replication would strengthen confidence. The field is moving rapidly; VIRAL's 64-GPU requirement and LeHome's deformable-object benchmark suggest the "scale wins" narrative is still unresolved. New practical DR pipeline guidance from Claru adds practitioner-level detail not captured in academic papers.