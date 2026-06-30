---
type: research_note
tags:
  - cross-domain-policy-transfer
  - sim-to-real
  - manipulation
  - domain-randomization
  - diffusion-trajectory-editing
  - point-bridge
  - zero-shot-transfer
  - cross-embodiment
  - policy-adaptation
domain: robotics
date: 2026-06-28
source_type: synthesized
research_depth: consolidated
---

# Cross-Domain Policy Transfer: Sim-to-Real for Manipulation Tasks

## Summary

Cross-domain policy transfer addresses learning control policies in simulation (or another domain) and deploying them on physical robots — or across different robot embodiments. The core challenge is the **reality gap**: mismatches in visual appearance, contact dynamics, sensor noise, and timing that cause simulation-trained policies to fail in deployment. Modern approaches span domain randomization, diffusion-based trajectory editing (xTED), 3D representation bridges (Point Bridge), latent-space projection, and cross-domain co-training (BEACON). The field is moving toward foundation models trained at scale on procedural simulation data, with MolmoBot (~42M synthetic annotations) and DoorMan (83% zero-shot humanoid loco-manipulation) demonstrating that zero-shot sim-to-real is achievable for increasingly complex tasks, while methods like TRANSIC (human-in-the-loop correction) and TAM (torque-level adapters) address residual gaps after transfer.

## Key Facts

- **Domain randomization** (Tobin et al., 2017) remains foundational: randomize visual and dynamic parameters during training to force policy invariance. Effective for vision-based grasping but struggles with contact-rich manipulation; over-randomization degrades training quality
- **xTED** (NeurIPS 2024 Workshop): Data-centric approach — uses diffusion models to transform source-domain trajectories into target-domain-compatible trajectories, decoupling adaptation from policy architecture. Superior to policy-level adaptation in both sim and real-robot experiments
- **Point Bridge** (arXiv:2601.16212): Uses 3D point clouds as domain-invariant representations that abstract away viewpoint/texture differences while preserving geometric structure critical for manipulation
- **MolmoBot** (CVPR 2026, Ai2): Zero-shot sim-to-real via large-scale procedural simulation — ~42M synthetic grasp annotations. Outperforms models trained on expensive real-world data. Challenges the assumption that real-world data is always required
- **DoorMan** (CVPR 2026): First zero-shot humanoid loco-manipulation on Unitree G1. Vision-only RL with staged-reset mechanism, 83% success vs 80% for expert teleoperators. Completes door-opening 31% faster than humans
- **VIRAL** (arXiv:2511.15200): Teacher-student framework for RGB-only humanoid loco-manipulation at 64-GPU scale. Critical finding: compute scale is a reliability factor — low-compute regimes fail entirely
- **BEACON** (arXiv:2605.08571): Cross-domain co-training via discrepancy-aware importance reweighting. Achieves feature alignment implicitly without explicit alignment objectives; outperforms fixed-ratio co-training and feature-alignment baselines
- **TRANSIC** (CoRL 2024): Human-in-the-loop — train base policy in sim, human corrects on real robot, residual policy learns from corrections. Captures unmodeled sim-to-real gaps holistically; succeeds on long-horizon contact-rich tasks (table lamp assembly)
- **S2GS** (arXiv:2512.04731): Semantic 2D Gaussian Splatting extracts domain-invariant spatial features (centroids, normals, orientations) that collapse sim/real visual gaps. Real-time performance for online control
- **Phys2Real** (arXiv:2510.11689): VLM priors + online adaptation — uses VLMs as "physics oracles" to estimate object properties, then refines through real-world interaction with uncertainty-aware fusion
- **Mana** (arXiv:2606.13677): Zero-shot sim-to-real for dexterous manipulation of articulated tools. <1 min real data per tool for validation; coarse-to-fine pipeline from grasp keyframes to full trajectories
- **Continuum robot sim-to-real** (arXiv:2606.22397): First sim-to-real for tendon-driven continuum robots. Rigid-body simulators model soft/continuum morphologies with sufficient fidelity
- **TAM** (arXiv:2606.06218): Torque adaptation module — lightweight adapter between sim-trained policies and real hardware. Addresses actuator dynamics gap without retraining core policy
- **LeHome Challenge 2026**: Bimanual garment folding as benchmark for hardest sim-to-real gap (deformable objects). Tests fundamental limits of soft-body simulation fidelity
- **Post-training reality gap**: Cross-domain transfer "can look good on paper and still fail in practice" due to hardware-specific latency, control rate, and timing constraints. Benchmarks overstate deployability

## Related (vault entities)

- `knowledge/robotics/cross-domain-policy-transfer-sim-to-real-manipulation.md` — Full deep-dive with 20+ methods and 15 open questions
- `knowledge/robotics/cross-domain-policy-transfer-sim-to-real-manipulation-structured.md` — Condensed structured version
- `knowledge/robotics/cross-embodiment-policy-transfer.md` — Cross-embodiment architectures (CrossFormer, SHADOW, X-VLA, Being-H0.5/0.7)
- `knowledge/robotics/cross-embodiment-policy-transfer-synthesis.md` — Cross-embodiment synthesis summary
- `knowledge/robotics/diffusion-policy-act.md` — Diffusion Policy and ACT as policy backbones
- `knowledge/robotics/generative-world-models-sim-to-real.md` — World models (DreamZero, BIGWorld) as training environments
- `knowledge/robotics/generative-world-models-vs-domain-randomization-sim-to-real.md` — DiWA, World-Gymnast
- `knowledge/ai/vla-online-fine-tuning-continual-learning.md` — Continual learning for VLAs
- `knowledge/robotics/humanoid-robotics.md` — Humanoid loco-manipulation context

## Open Questions

1. **Which representation wins?** 3D point clouds (Point Bridge), latent embeddings, segmentation masks (SHADOW), procedural scale (MolmoBot), semantic Gaussian features (S2GS), or best-effort co-training (BEACON)? No head-to-head comparison on equivalent benchmarks exists
2. **Does data-centric adaptation (xTED) scale to contact-rich tasks?** Cloth folding, screw driving, and deformable-object manipulation remain untested
3. **Minimal real-data budget**: MolmoBot claims zero-shot with 42M synthetic annotations; Mana needs <1 min per tool. Where is the practical threshold across methods?
4. **Can foundation models eliminate the sim-to-real gap?** Does the gap converge to zero at sufficient scale, or are certain tasks (dexterous manipulation, deformable objects) fundamentally real-data-dependent?
5. **Compute scaling law**: VIRAL requires 64 GPUs for reliable humanoid loco-manipulation. Does this scaling law hold for simpler manipulation, or is whole-body control uniquely compute-intensive?
6. **Is human-in-the-loop (TRANSIC) a bottleneck or feature?** Does the residual policy eventually eliminate correction needs, or is it inherently deployment-bottlenecked?
7. **Can torque adapters (TAM) replace domain randomization?** TAM addresses actuator dynamics separately — can this complement or replace heavy randomization (VIRAL, MolmoBot)?
8. **Will HyperSim's holistic approach replace patchwork methods?** Can a single framework match specialized methods for specific gap categories?
9. **Deformable-object limits (LeHome)**: If garment folding shows persistent failures, does this reveal fundamental limits in soft-body simulation?
10. **Does VISER correlation predict real-world performance?** Higher visual realism scores → better transfer, or are non-visual dynamics the dominant failure mode?

## Sources

1. Tobin et al., "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World" (arXiv:1703.06907, 2017)
2. Niu et al., "Cross-Domain Adaptation via Diffusion-Based Trajectory Editing" (arXiv:2409.08687, NeurIPS 2024 Workshop)
3. Haldar et al., "Point Bridge: 3D Representations for Cross Domain Policy Learning" (arXiv:2601.16212)
4. Allen AI, "MolmoBot: Zero-Shot Sim-to-Real via Large-Scale Simulation" (CVPR 2026)
5. Xue et al., "DoorMan: Opening the Sim-to-Real Door for Humanoid Pixel-to-Action Policy Transfer" (CVPR 2026)
6. He et al., "VIRAL: Visual Sim-to-Real at Scale for Humanoid Loco-Manipulation" (arXiv:2511.15200)
7. Zhang, Qi, Yang, "BEACON: Cross-Domain Co-Training via Best-Effort Adaptation" (arXiv:2605.08571)
8. Ji et al., "TRANSIC: Sim-to-Real Policy Transfer by Learning from Online Correction" (arXiv:2405.10315, CoRL 2024)
9. Tang et al., "S2GS: Semantic 2D Gaussian Splatting for Domain-Invariant Transfer" (arXiv:2512.04731)
10. Wang et al., "Phys2Real: Fusing VLM Priors with Interactive Online Adaptation" (arXiv:2510.11689)
11. "Mana: Dexterous Manipulation of Articulated Tools" (arXiv:2606.13677)
12. "Do Rigid-Body Simulators Dream of Soft Robots?" (arXiv:2606.22397)
13. "TAM: Torque Adaptation Module" (arXiv:2606.06218)
14. "HyperSim: A Holistic Sim-To-Real Framework" (arXiv:2605.26638)
15. "VISER: Visual Realism Benchmark" (arXiv:2605.06311)
16. "LeHome Challenge 2026: Sim-to-Real Bimanual Garment Folding" (arXiv:2606.27163)
17. t6-thu/awesome-cross-domain-policy-transfer-for-embodied-agents (GitHub curated list)
18. Wang, Li, Niu et al., "Efficient Sim-to-Real with Online Imitation-Pretrained World Models" (arXiv:2510.02538)
19. "H2O+: Hybrid Offline-and-Online RL with Dynamics Gaps" (arXiv:2309.12716, ICLR 2024 Workshop)
20. MoMani Benchmark (emergentmind.com/topics/momani-benchmark)

## Confidence

0.87: Core methods (domain randomization, residual fine-tuning, latent-space transfer) are well-established with multiple replications. Emerging 2024-2026 methods (MolmoBot, DoorMan, VIRAL, BEACON, S2GS, Mana) are verified via arXiv IDs, project websites, and institutional publications (NVIDIA, Allen AI, Harvard). Zero-shot claims from MolmoBot and Mana are promising but deserve caution — independent replication would strengthen confidence. The field is moving rapidly: VIRAL's 64-GPU requirement and LeHome's deformable-object benchmark suggest the "scale wins" narrative is still unresolved. Confidence limited primarily by web search unavailability for post-September 2026 updates and the inability to independently verify the most recent arXiv submissions (TAM, continuum robot, LeHome 2026).