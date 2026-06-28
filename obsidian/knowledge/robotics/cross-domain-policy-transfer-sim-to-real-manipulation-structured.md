## Summary

Cross-domain policy transfer for robotic manipulation bridges the "reality gap" — the mismatch between simulation-trained policies and real-world deployment. Core approaches include domain randomization (randomizing visual/dynamic parameters), diffusion-based trajectory editing (xTED: adapting data rather than policies), 3D representation bridges (Point Bridge: using point clouds as domain-invariant intermediaries), and online adaptation via world models or human correction (TRANSIC). The field has moved toward foundation models trained at scale on procedural simulation data, with MolmoBot demonstrating zero-shot sim-to-real transfer using ~42M synthetic annotations, and DoorMan showing zero-shot humanoid loco-manipulation on a Unitree G1.

## Key Facts

- **Domain randomization** (Tobin et al. 2017) remains the foundational approach: randomizing textures, lighting, mass, and friction during training to force policy invariance, but struggles with contact-rich tasks and cannot cover all real-world dynamics
- **xTED** (NeurIPS 2024 Workshop) uses diffusion-based trajectory editing as a data-centric approach — transforms source-domain trajectories into target-domain-compatible ones, decoupling adaptation from policy architecture
- **Point Bridge** uses 3D point clouds as domain-invariant representations, abstracting away viewpoint and texture differences while preserving geometric structure critical for manipulation
- **MolmoBot** (CVPR 2026) achieves zero-shot sim-to-real via large-scale procedural simulation (42M grasp annotations), challenging the assumption that real-world data is always required
- **DoorMan** (CVPR 2026) demonstrates first zero-shot humanoid loco-manipulation policy on Unitree G1 with 83% success rate — comparable to expert teleoperators (80%)
- **BEACON** frames cross-domain co-training as discrepancy-aware importance reweighting, achieving feature alignment implicitly through co-training without explicit alignment objectives
- **S2GS** uses semantic 2D Gaussian splatting to extract domain-invariant spatial features (centroids, normals, orientations) that collapse sim/real visual gaps
- **Continuum robot sim-to-real** (arXiv:2606.22397) shows rigid-body simulators can model soft/continuum robots with sufficient fidelity, extending sim-to-real beyond traditional rigid robots

## Related (vault entities)

- `knowledge/robotics/cross-embodiment-policy-transfer.md` — Cross-embodiment transfer architectures (CrossFormer, SHADOW, X-VLA)
- `knowledge/robotics/diffusion-policy-act.md` — Diffusion Policy and ACT architectures as policy backbones
- `knowledge/robotics/generative-world-models-sim-to-real.md` — World models (DreamZero, BIGWorld) as sim-to-real training environments
- `knowledge/robotics/generative-world-models-vs-domain-randomization-sim-to-real.md` — DiWA, World-Gymnast: generative alternatives to domain randomization
- `knowledge/robotics/vla-adapter-tiny-scale-vla.md` — Lightweight fine-tuning for real-world adaptation
- `knowledge/robotics/world-models-as-priors-policy-bootstrapping.md` — WMPO, World4RL: world models as simulation priors
- `knowledge/ai/vla-online-fine-tuning-continual-learning.md` — Continual learning for VLAs
- `knowledge/robotics/humanoid-robotics.md` — Humanoid loco-manipulation and DoorMan context

## Open Questions

1. **Which representation wins for sim-to-real?** No direct comparison exists between 3D point clouds, latent embeddings, segmentation masks, semantic Gaussian features, and best-effort co-training on equivalent benchmarks.
2. **Does data-centric adaptation (xTED) scale to contact-rich tasks?** Untested on cloth folding, screw driving, and other high-fidelity dynamics tasks.
3. **What is the minimal real-data budget?** MolmoBot claims zero-shot; Mana needs <1 min per tool. Where is the practical threshold?
4. **Can foundation models eliminate the sim-to-real gap?** Does the gap converge to zero at sufficient scale, or are certain tasks fundamentally real-data-dependent?
5. **Is human-in-the-loop (TRANSIC) scalable?** Does the residual policy eventually eliminate the need for correction, or is it inherently a deployment bottleneck?
6. **Can VLM priors (Phys2Real) generalize across novel object categories?** Reliability of VLM-inferred physical parameters for unseen objects remains untested.

## Sources

- **Survey**: Haoyi Niu et al., "A Comprehensive Survey of Cross-Domain Policy Transfer for Embodied Agents" (arXiv:2402.04580, IJCAI 2024) — [GitHub awesome list](https://github.com/t6-thu/awesome-cross-domain-policy-transfer-for-embodied-agents)
- **xTED**: arXiv:2409.08687 — Diffusion-based trajectory editing for cross-domain adaptation
- **Point Bridge**: Haldar et al., arXiv:2601.16212 — 3D point cloud representations as domain-invariant bridge
- **MolmoBot**: Allen AI, CVPR 2026 — Zero-shot sim-to-real via large-scale simulation
- **DoorMan**: Xue et al., CVPR 2026 — Humanoid zero-shot loco-manipulation on Unitree G1
- **VIRAL**: arXiv:2511.15200 — 64-GPU scale teacher-student framework for humanoid loco-manipulation
- **BEACON**: Zhang, Qi, Yang, arXiv:2605.08571 — Best-effort co-training via discrepancy-aware reweighting
- **S2GS**: Tang et al., arXiv:2512.04731 — Semantic 2D Gaussian splatting for domain-invariant features
- **Phys2Real**: Wang et al., arXiv:2510.11689 — VLM priors + online adaptation for uncertainty-aware manipulation
- **TRANSIC**: Ji et al., arXiv:2405.10315 — Human-in-the-loop sim-to-real via online correction (CoRL 2024)
- **Mana**: arXiv:2606.13677 — Dexterous manipulation of articulated tools with zero-shot transfer
- **TAM**: arXiv:2606.06218 — Torque adaptation module for actuator dynamics gap
- **H2O+**: arXiv:2309.12716 — Hybrid offline-and-online RL with dynamics gaps (ICLR 2024 Workshop)

## Confidence

0.87: Core methods (domain randomization, residual fine-tuning, latent-space transfer) are well-established with multiple replications. Emerging methods (MolmoBot, DoorMan, VIRAL) are verified via arXiv, project websites, and institutional publications (NVIDIA, Allen AI). Zero-shot claims are promising but independent replication would strengthen confidence. The field is moving rapidly with 2024-2026 literature still consolidating.
