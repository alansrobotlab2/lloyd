---
segment: knowledge
type: research_note
tags:
  - robot-evaluation
  - benchmark
  - bridgedata-v2
  - open-x-embodiment
  - rt-x
  - vla
  - sim-to-real
  - simpler
  - libero
  - maniskill
  - sim-to-sim-to-real
  - robot-arena
  - robot-arena-infinity
  - hug-bench
  - radar-bench
  - vlabench
  - robogate
  - omninavbench
  - being-h05
  - vla-data-volume
  - droid
domain: robotics
researched_at: 2026-11-15T00:00:00Z
last_verified: 2026-11-15
source_type: synthesized
research-depth: deep
---

# Scalable Robot Evaluation Benchmarks: BridgeData V2 and Open X-Embodiment

## Summary

Scalable robot evaluation benchmarks are standardized test suites used to measure how well robot manipulation policies generalize across tasks, environments, and embodiments. **BridgeData V2** provides a real-world WidowX 250 manipulation corpus of 60,096 trajectories across 24 real-world kitchen environments and 13 manipulation skills, serving as both a training dataset and evaluation benchmark. **Open X-Embodiment (OXE)** is the largest cross-embodiment dataset with over 1 million curated trajectories spanning 22 robot embodiments and 527 skills from 21 institutions, defining the RT-X model family's evaluation protocol of training on diverse robots and testing on held-out tasks at partner institutions. The field is shifting from single-embodiment evaluation toward cross-platform generalization metrics, with simulation-based evaluation (SIMPLER) achieving r > 0.85 correlation with real-world performance, enabling scalable low-cost evaluation that accelerates generalist VLA model development.

## Key Facts

### BridgeData V2 (Berkeley RAIL, CoRL 2023, arXiv:2308.12952)
- **60,096 trajectories** collected across **24 real-world kitchen environments** and **13 manipulation skills**
- Hardware: WidowX 250 (6-DOF arm), VR-controller teleoperation, 5 Hz control frequency, avg 38 timesteps per trajectory
- Skills include pick-and-place (carrots on plates), object rearrangement (spoons on towels, eggplants in baskets), stacking cubes
- Available as standalone dataset with richer metadata (failure modes, intervention timestamps, negative examples) and as a subset of Open X-Embodiment
- Serves as the standard evaluation environment for generalist VLA policies alongside Google Robot setups
- Represents 22% of OXE's real-robot (non-sim) subset despite being only 6% of total trajectory count

### Open X-Embodiment (Google DeepMind + 21 institutions, arXiv:2310.08864)
- **970,000+ curated trajectories** (1M+ total) across **22 robot embodiments**, **527 skills**, from **21 contributing institutions**
- Embodiments span single-arm manipulators (Franka Emika FR3), dual-arm systems, mobile manipulators, and dexterous hands
- Data stored in **RLDS format** (Apache Parquet, columnar compression) with lazy-loading support
- **RT-X model family**: RT-1-X (35M-parameter transformer, OXE-only training) and RT-2-X (VLM-backed)
- RT-1-X evaluated via **3,600 real-world trials** across 6 robot platforms at Berkeley, Freiburg, NYU, Stanford, and USC
- **Key result**: RT-1-X outperforms dataset-specific models by 50% on held-out tasks from contributing institutions
- OpenVLA pre-training used 970K-trajectory curated subset; curation step filters for data quality and instruction coverage
- OXE is now treated as **shared infrastructure** — most 2026 teams fine-tune from pretrained models rather than pre-training from scratch

### SIMPLER: Simulated Evaluation Framework (arXiv:2405.05941)
- **Problem**: Real-world robot evaluation is expensive, slow, and hard to reproduce
- **Approach**: Simulation environments replicating real robot setups (BridgeData V2 + Google Robot)
- **Key metrics**:
  - **Pearson correlation (r)**: r > 0.85 for Google Robot tasks, r = 0.890 for BridgeData V2
  - **Mean Maximum Rank Violation (MMRV)**: 0.014 — measures ranking accuracy across policies
- Addresses two gaps:
  - **Control gap**: System identification of PD gains, feed-forward terms, execution rates
  - **Visual gap**: Green-screening real-world backgrounds onto simulated scenes; texture matching
- Validates against RT-1, RT-1-X, RT-2-X, Octo-Base, Octo-Small
- Simulated evaluation predicts real-world performance better than validation action MSE

### VLA Training Data Volume Reference
| Task Type | Demo Volume Range | Key Systems |
|-----------|-------------------|------------|
| VLA pre-training (from scratch) | 500K – 1M+ trajectories | OpenVLA, RT-2, Octo |
| Fine-tune: single pick-and-place (fixed) | 50 – 200 demos | OpenVLA fine-tuning studies |
| Fine-tune: pick-and-place (multi-object) | 200 – 1,000 demos | BridgeData V2, ALOHA |
| Fine-tune: multi-step manipulation | 1,000 – 5,000 demos | Language Table, RoboAgent |
| Dexterous bimanual manipulation | 5,000 – 50,000 demos | pi-zero (Physical Intelligence) |
| Mobile manipulation (nav + manipulation) | 10,000 – 100,000 demos | OXE mobile subset, DROID |
| Humanoid whole-body control | 50,000 – 500,000 demos | GR00T N1, Figure 02, Unitree G1 |

### Evaluation Benchmarks Ecosystem
- **LIBERO**: Simulation benchmark for object rearrangement tasks; widely used for VLA evaluation. Current best models score <50% on 5+ step chains (LIBERO Long-Horizon).
- **LIBERO-Pro** (2025): Extended LIBERO with more diverse scenarios and distractors.
- **LIBERO-Plus** (2025): Additional task variants and evaluation dimensions.
- **LIBERO-Mem** (Nov 2025, arXiv:2511.11478): Memory-intensive manipulation benchmark for long-horizon, temporally entangled, and repetitive subgoals; tests robust memory rather than short-term perception
- **ManiSkill**: Digital twin framework supporting real-to-sim evaluation; includes BridgeData V2 evaluation tasks
- **RoboSuite**: Physics-based manipulation benchmark with task variety; Diffusion Policy achieves 80%+ on complex tasks
- **RoboCasa**: Home-environment manipulation benchmark; Being-H0.5 achieves 53.9% on low-res RGB
- **RobotArena ∞** (2025, arXiv:2510.23571): Scalable real-to-sim benchmarking system — builds a system of distribution where human users reset scenes, run robot policies, and evaluate resulting executions. Targets DROID scenes and evaluates policies fine-tuned on DROID. Bridges the gap between simulation benchmarks and real-world distribution shifts.
- **HUG-BENCH**: Robotic grasping benchmark with sim-to-real validation
- **VLABench** (ICCV 2025): Large-scale benchmark for language-conditioned robotics manipulation with long-horizon reasoning tasks
- **RoboGate** (2025/2026): Industrial robotics benchmark using Isaac Sim; revealed a "cross-simulator gap" — GR00T N1.6 scores 97.65% on LIBERO (MuJoCo) but 0% on RoboGate (Isaac Sim), same model/robot/task
- **RADAR** (Feb 2026, arXiv:2602.10980): Real-world Autonomous Dynamics And Reasoning benchmark; introduces systematic environmental dynamics, spatial reasoning tasks, and autonomous 3D evaluation. Key finding: sensor noise drops 3D IoU from 0.261 to 0.068, exposing severe fragility under physical variation
- **KinDERBench** (2026): Physical reasoning benchmark suite from the KinDER framework; evaluates embodied reasoning and planning capabilities across robotic systems
- **OmniNavBench** (May 2026, arXiv:2605.09441): Unified benchmark for general-purpose navigation assessing cross-skill coordination and cross-embodiment generalization. Introduces compositional complexity for multi-skill coordination evaluation.
- **DROID** (Toyota Research / Berkeley, arXiv:2403.12945): 76K trajectories, 564 scenes, 86 tasks across 18 research labs; primary training corpus for Octo and RT-X alongside OXE. Prioritizes environment diversity (564 distinct physical environments) over embodiment diversity.

### Post-June 2026 Benchmarks
- **PhAIL** (May 2026, arXiv:2605.29710): Real-Robot VLA Benchmark introducing distributional methodology. Proposes Time-to-Success CDF as an evaluation primitive instead of binary success rate — captures the full distribution of how fast policies succeed, not just whether they do. Critiques existing real-robot evaluation protocols for statistical insufficiency.
- **UMI-Bench 1.0** (June 2026, arXiv:2606.10382): Local-first real-robot benchmark for UMI-style (under-measured-intervention) wrist-view manipulation policies. Standardizes the full data-to-evaluation pipeline: demonstration collection, episode specification, scene reset, policy execution, logging, scoring, and task-factor analysis. Designed for reproducibility across labs.
- **RobotValues** (June 2026, arXiv:2606.03312): Evaluates household robots on value-conflict decision-making across 10K scenarios. Each instance presents a household image with multiple plausible robot actions that prioritize different human values (privacy, safety, convenience, cleanliness). Moves beyond task-success metrics to capture ethical and social reasoning in household robots.
- **MoMani Benchmark** (March 2026): Automated, large-scale framework for mobile manipulation training and evaluation. Covers long-horizon mobile manipulation tasks requiring coordinated navigation + manipulation — a growing evaluation gap as VLA models move beyond stationary arms.
- **Visual Realism Benchmark** (May 2026, arXiv:2605.06311): Benchmark for visually realistic simulation evaluation of robot manipulation policies. Addresses the visual domain gap between simulation and reality that existing benchmarks ignore.
- **WM4VLA Benchmark v4** (April 2026): Dataset curation benchmark filtering for arm visibility, minimum frame count, and per-dataset allocation for VLA model evaluation.

### Quantitative Results Summary
| Benchmark | Model | Success Rate / Metric | Notes |
|-----------|-------|-------------|-------|
| OXE held-out tasks | RT-1-X | +50% vs dataset-specific | 6 institutions, 3,600 trials |
| SIMPLER (Google Robot) | RT-1, RT-1-X, RT-2-X, Octo | r > 0.85 sim-real correlation | 4 tasks |
| SIMPLER (BridgeData V2) | Multiple policies | r = 0.890, MMRV = 0.014 | WidowX setup |
| OXE pretraining → fine-tune | RT-X | 80% less fine-tuning data needed | 500–1,000 trajectories vs 5,000–10,000 |
| LIBERO Long-Horizon (5+ steps) | All known VLAs | <50% | Error propagates across steps |
| RoboSuite complex manipulation | Diffusion Policy | 80%+ | Simulation benchmark |
| RoboCasa | Being-H0.5 | 53.9% (low-res RGB) | Home-environment benchmark |
| RADAR | Various | 3D IoU: 0.261 → 0.068 (under noise) | Sensor noise fragility |
| RoboGate cross-sim | GR00T N1.6 | 97.65% LIBERO → 0% RoboGate | Cross-simulator gap |

## Related (vault entities)
- `knowledge/robotics/cross-embodiment-policy-transfer.md` — OXE/RT-X as training foundation; CrossFormer, X-VLA, SHADOW
- `knowledge/robotics/cross-domain-policy-transfer-sim-to-real-manipulation.md` — Sim-to-real pipeline context
- `knowledge/robotics/cross-embodiment-policy-transfer-synthesis.md` — Cross-embodiment transfer synthesis
- `knowledge/robotics/real-time-policy-adaptation-online-learning-failures.md` — Online learning for policy adaptation
- Open X-Embodiment (knowledge graph entity) — Dataset provenance, scale metrics
- RT-X models (knowledge graph entity) — Model architecture and training
- BridgeData V2 (knowledge graph entity) — Dataset specifications
- DROID (knowledge graph entity) — Environment-diverse training corpus

## Open Questions

1. **Benchmark coverage gap**: Most evaluation focuses on tabletop kitchen tasks (BridgeData V2) and single-arm manipulation. How do benchmarks cover bimanual, mobile manipulation, legged robots, and outdoor deployment? *Progress*: MoMani Benchmark (2026) addresses mobile manipulation, but standardized cross-embodiment coverage remains incomplete.

2. **Real robot evaluation cost**: Running 3,600 real-world trials across 6 institutions is a massive effort. Can SIMPLER-level simulation reliability reach r > 0.95 to make real evaluation unnecessary for most research?

3. **Cross-simulator gap**: RoboGate revealed that models scoring near-perfectly on MuJoCo-based LIBERO can score 0% on Isaac Sim equivalents. What is the right simulation fidelity target, and should benchmarks mandate multi-simulator evaluation?

4. **Reality gap in real benchmarks**: RADAR (Feb 2026) showed 3D IoU drops from 0.261 to 0.068 under sensor noise — even "real-world" benchmarks overfit to static laboratory conditions. How do we stress-test models against high-entropy dynamics?

5. **Spatial-physical intelligence**: Current benchmarks emphasize semantic instruction following over geometric reasoning. Do VLAs truly understand 3D scenes or exploit 2D correlations?

6. **Standardized leaderboard**: Unlike vision-language (HELM, Big-Bench), robot learning lacks a unified leaderboard. The VLA Leaderboard (CodeSOTA) and RoboGate are emerging but not yet adopted as a standard.

7. **Shared infrastructure maturity**: With OXE, DROID, and BridgeData V2 now serving as shared pre-training infrastructure, how much does the field need to invest in benchmark standardization vs. continuing to build larger datasets?

8. **Value-aligned evaluation**: RobotValues (June 2026) exposes a blind spot — existing benchmarks evaluate task completion, not ethical decision-making. Should household robot benchmarks incorporate value-conflict scenarios as standard evaluation?

9. **Distributional evaluation methods**: PhAIL (May 2026) argues that binary success rates are statistically insufficient. Should the field adopt Time-to-Success CDF as a standard evaluation primitive?

## Sources

1. **BridgeData V2**: Walke et al., "BridgeData V2: A Dataset for Robot Learning at Scale" (CoRL 2023, arXiv:2308.12952) — [rail-berkeley.github.io/bridgedata](https://rail-berkeley.github.io/bridgedata/)
2. **Open X-Embodiment / RT-X**: Open X-Embodiment Collaboration, "Open X-Embodiment: Robotic Learning Datasets and RT-X Models" (arXiv:2310.08864) — [robotics-transformer-x.github.io](https://robotics-transformer-x.github.io/)
3. **SIMPLER**: Black et al., "Evaluating Real-World Robot Manipulation Policies in Simulation" (arXiv:2405.05941) — [simpler-env.github.io](https://simpler-env.github.io/)
4. **DROID Dataset**: Brohan et al., "DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset" (arXiv:2403.12945) — [droid-dataset.github.io](https://droid-dataset.github.io/)
5. **RobotArena ∞**: arXiv:2510.23571 — Scalable real-to-sim benchmarking system
6. **LIBERO-Mem**: arXiv:2511.11478 (Nov 2025) — Memory-intensive long-horizon manipulation benchmark
7. **RADAR**: Chen et al., "RADAR: Benchmarking Vision-Language-Action Generalization via Real-World Dynamics" (arXiv:2602.10980, Feb 2026)
8. **VLABench**: OpenMOSS, "VLABench: A Large-Scale Benchmark for Language-Conditioned Robotics Manipulation" (ICCV 2025) — [github.com/OpenMOSS/VLABench](https://github.com/OpenMOSS/VLABench)
9. **OmniNavBench**: arXiv:2605.09441 (May 2026) — Cross-skill coordination and cross-embodiment navigation benchmark
10. **RoboGate Cross-Simulator Gap**: [RoboGate VLA Benchmark](https://www.robogate.io/vla)
11. **VLA Training Data Volume**: [Claru.ai](https://claru.ai/blog/vla-training-data-volume) (April 2026) — VLA pre-training and fine-tuning data volume reference
12. **Awesome VLA Benchmarks**: [JFan5/awesome-vla-benchmarks](https://github.com/JFan5/awesome-vla-benchmarks) — Curated benchmark registry
13. **VLA Leaderboard**: [CodeSOTA Robotics](https://www.codesota.com/robotics) — RT-2 vs OpenVLA vs Pi0 vs Octo comparison
14. **KinDERBench** (2026) — Physical reasoning benchmark suite
15. **ICLR 2026 VLA Analysis**: Breuss, "State of Vision-Language-Action Research at ICLR 2026" — Survey of 164 VLA submissions
16. **Being-H0.5**: Scaling Human-Centric Robot Learning for Cross-Embodiment Generalization — [liner.com](https://liner.com/review/beingh05-scaling-humancentric-robot-learning-for-crossembodiment-generalization)
17. **PhAIL**: arXiv:2605.29710 (May 2026) — Real-Robot VLA Benchmark with distributional methodology
18. **UMI-Bench 1.0**: arXiv:2606.10382 (June 2026) — Local-first real-robot wrist-view manipulation benchmark
19. **RobotValues**: arXiv:2606.03312 (June 2026) — Household robot value-conflict evaluation benchmark
20. **MoMani Benchmark** (March 2026) — Mobile manipulation evaluation framework
21. **Visual Realism Benchmark**: arXiv:2605.06311 (May 2026) — Simulation visual fidelity evaluation
22. **WM4VLA Benchmark v4** (April 2026) — [HuggingFace](https://huggingface.co/datasets/zywu2115/WM4VLA_benchmark_v4) — VLA dataset curation benchmark
23. **VLA Survey 2026**: arXiv:2604.23001 (April 2026) — "Vision-Language-Action in Robotics: A Survey of Datasets..."

## Confidence: 0.92

BridgeData V2, Open X-Embodiment, and SIMPLER are well-documented with consistent arXiv papers and project sites. The VLA data volume reference (Claru.ai, April 2026) provides concrete, cross-referenced numbers matching published results from OpenVLA, Octo, pi-zero, and GR00T papers. RobotArena ∞, OmniNavBench, and RADAR are sourced directly from arXiv. The cross-simulator gap (RoboGate) and cross-embodiment results (Being-H0.5, RT-1-X) are confirmed across multiple sources. Confidence remains at 0.92 due to the depth of corroborating sources spanning the core benchmark landscape and concrete data-volume reference points.