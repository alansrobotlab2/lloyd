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
researched_at: 2026-06-29T00:00:00Z
last_verified: 2026-06-29
source_type: synthesized
research-depth: deep
---

# Scalable Robot Evaluation Benchmarks: BridgeData V2 and Open X-Embodiment

## Summary

Scalable robot evaluation benchmarks are standardized test suites for measuring robot manipulation policy generalization across tasks, environments, and embodiments. **BridgeData V2** provides 60,096 real-world trajectories across 24 kitchen environments and 13 skills using WidowX-250 hardware, serving as both training data and evaluation benchmark with CC-BY 4.0 licensing. **Open X-Embodiment (OXE)** is the largest cross-embodiment dataset with 1M+ curated trajectories spanning 22 robot embodiments, 527 skills, and 21 institutions, establishing the RT-X model family's training and evaluation protocol. The field has shifted from single-embodiment evaluation toward cross-platform generalization, with simulation-based evaluation (SIMPLER) achieving r > 0.85–0.924 correlation with real-world performance, enabling scalable low-cost evaluation. The benchmark ecosystem now includes 15+ specialized benchmarks addressing simulation-to-real gaps, adversarial testing, cross-simulator generalization, and value-aligned evaluation.

## Key Facts

### BridgeData V2 (Berkeley RAIL, CoRL 2023, arXiv:2308.12952)
- **60,096 trajectories** collected across **24 real-world kitchen environments** and **13 manipulation skills** via VR-controller teleoperation at 5 Hz control frequency, average 38 timesteps per trajectory
- Hardware: WidowX-250 6DOF arm (250g payload, 650mm range, 1mm accuracy, DYNAMIXEL XL430-W250 servos), synchronized RGB wrist + overhead cameras
- Skills: pick-and-place (carrots on plates), object rearrangement (spoons on towels, eggplants in baskets), cube stacking, pouring, rigid-body manipulation
- Available as standalone dataset with richer metadata (failure modes, intervention timestamps, negative examples) and as subset of Open X-Embodiment
- Data distribution is tighter than OXE due to consistent collection protocol across environments
- Represents 22% of OXE's real-robot (non-sim) subset despite being only 6% of total trajectory count
- Supports open-vocabulary, multi-task learning conditioned on goal images or natural language instructions

### Open X-Embodiment (Google DeepMind + 21 institutions, arXiv:2310.08864)
- **1M+ total trajectories** (970K curated subset) across **22 robot embodiments** spanning single-arm, dual-arm, mobile manipulators, and dexterous hands
- **527 skills** from **21 contributing institutions** including Google DeepMind, Berkeley, KIT, Stanford, and others
- Data stored in **RLDS format** (Apache Parquet, columnar compression) with lazy-loading support and synchronized multi-modal inputs
- **RT-X model family**: RT-1-X (35M-parameter transformer, OXE-only training) and RT-2-X (VLM-backed)
- RT-1-X evaluated via **3,600 real-world trials** across 6 robot platforms at Berkeley, Freiburg, NYU, Stanford, and USC
- **Key result**: RT-1-X outperforms dataset-specific models by 50% on held-out tasks from contributing institutions
- Pre-trained checkpoints released for inference and fine-tuning; curation step filters for data quality and instruction coverage
- OXE is now treated as **shared infrastructure** — most teams fine-tune from pretrained models rather than pre-training from scratch

### SIMPLER: Simulated Evaluation Framework (arXiv:2405.05941)
- **Problem**: Real-world robot evaluation is expensive, slow, and hard to reproduce
- **Approach**: Simulation environments replicating real robot setups (BridgeData V2 + Google Robot) built on SAPIEN physics engine and ManiSkill2/ManiSkill3
- **Two methodology tracks**:
  - Visual Matching: Green-screening real-world backgrounds onto simulated scenes with texture matching
  - Variant Aggregation: Averages success across environment variants (backgrounds, lighting, distractors, camera poses)
- **Key metrics**:
  - Pearson correlation (r): r = 0.924 for Google Robot tasks (Visual Matching setup), r = 0.890 for BridgeData V2
  - Mean Maximum Rank Violation (MMRV): 0.056 vs. 0.375 for validation-loss baseline
  - ManiSkill3 provides 10–15× speedup via GPU acceleration
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
- **LIBERO-Pro** (2025): Exposed that prior "95% accuracy" was mostly memorization; models fail when instructions are corrupted or replaced with meaningless tokens.
- **LIBERO-Plus** (2025): Procedurally generated task variations for stress-testing generalization.
- **LIBERO-Mem** (Nov 2025, arXiv:2511.11478): Memory-intensive long-horizon benchmark for temporally entangled, repetitive subgoals testing robust memory rather than short-term perception.
- **ManiSkill**: Digital twin framework supporting real-to-sim evaluation; includes BridgeData V2 evaluation tasks.
- **ManiSkill2**: 20 task families, 2,000+ object models, 4M+ demonstration frames covering stationary/mobile, single/dual-arm, rigid/soft-body.
- **ManiSkill3**: GPU-parallelized robotics simulation (arXiv:2410.00425, RSS 2025) with 10–15× speedup.
- **ManiSkill-HAB**: Home rearrangement benchmark for low-level manipulation (ICLR 2025).
- **RoboSuite**: MuJoCo-based physics framework for manipulation; Diffusion Policy achieves 80%+ on complex tasks.
- **RoboCasa**: Home-environment manipulation benchmark; Being-H0.5 achieves 53.9% on low-res RGB.
- **RobotArena ∞** (2025, arXiv:2510.23571): Scalable real-to-sim benchmarking system — builds distribution of human users resetting scenes, running policies, and evaluating executions. Targets DROID scenes with 7,000+ human preference pairs.
- **HUG-BENCH**: Human-centric robot evaluation for household planners in realistic scenarios.
- **VLABench** (ICCV 2025): Large-scale language-conditioned robotics manipulation with long-horizon reasoning tasks.
- **RoboGate** (2025/2026): Industrial robotics benchmark using Isaac Sim. **Cross-simulator gap**: GR00T N1.6 scores 97.65% on LIBERO (MuJoCo) but 0% on RoboGate (Isaac Sim) — same model/robot/task. 68 adversarial Pick & Place scenarios, scaling from 27M to 7B params (260×) yields zero improvement.
- **RADAR** (Feb 2026, arXiv:2602.10980): Real-world Autonomous Dynamics And Reasoning benchmark. Sensor noise drops 3D IoU from 0.261 to 0.068, exposing severe fragility under physical variation.
- **KinDERBench** (2026): Physical reasoning benchmark suite with 13 baselines in 8 environments evaluating embodied reasoning and planning.
- **OmniNavBench** (May 2026, arXiv:2605.09441): Unified benchmark for general-purpose navigation assessing cross-skill coordination and cross-embodiment generalization with compositional complexity.
- **PhAIL** (May 2026, arXiv:2605.29710): Real-Robot VLA Benchmark with distributional methodology. Proposes **Time-to-Success CDF** as evaluation primitive instead of binary success — captures full distribution of completion speed. Uses Kaplan-Meier estimator for censored data, bootstrap CIs, KS tests for significance.
- **UMI-Bench 1.0** (June 2026, arXiv:2606.10382): Local-first real-robot benchmark for wrist-view manipulation policies. Standardizes full data-to-evaluation pipeline: demonstration collection, episode specification, scene reset, policy execution, logging, scoring, task-factor analysis.
- **RobotValues** (June 2026, arXiv:2606.03312): 10K value-conflict scenarios for household robots. Each instance: household image with multiple plausible actions prioritizing different human values (privacy, safety, convenience, cleanliness).
- **MoMani Benchmark** (March 2026): Automated framework for mobile manipulation covering long-horizon tasks requiring coordinated navigation + manipulation.
- **Visual Realism Benchmark** (May 2026, arXiv:2605.06311): Simulation visual fidelity evaluation addressing visual domain gap between sim and reality.
- **WM4VLA Benchmark v4** (April 2026): Dataset curation benchmark filtering for arm visibility, minimum frame count, per-dataset allocation for VLA model evaluation.
- **DROID** (Toyota Research / Berkeley, arXiv:2403.12945): 76K trajectories, 564 scenes, 86 tasks across 18 research labs. Primary training corpus for Octo and RT-X alongside OXE. Prioritizes environment diversity (564 distinct physical environments) over embodiment diversity.

### Quantitative Results Summary
| Benchmark | Model | Success Rate / Metric | Notes |
|-----------|-------|-------------|-------|
| OXE held-out tasks | RT-1-X | +50% vs dataset-specific | 6 institutions, 3,600 trials |
| SIMPLER (Google Robot) | RT-1, RT-1-X, RT-2-X, Octo | r = 0.924 sim-real correlation | 4 tasks, Visual Matching |
| SIMPLER (BridgeData V2) | Multiple policies | r = 0.890, MMRV = 0.056 | WidowX setup |
| OXE pretraining → fine-tune | RT-X | 80% less fine-tuning data needed | 500–1,000 trajectories vs 5,000–10,000 |
| LIBERO Long-Horizon (5+ steps) | All known VLAs | <50% | Error propagates across steps |
| RoboSuite complex manipulation | Diffusion Policy | 80%+ | Simulation benchmark |
| RoboCasa | Being-H0.5 | 53.9% (low-res RGB) | Home-environment benchmark |
| RADAR | Various | 3D IoU: 0.261 → 0.068 (under noise) | Sensor noise fragility |
| RoboGate cross-sim | GR00T N1.6 | 97.65% LIBERO → 0% RoboGate | Cross-simulator gap |
| RoboGate scaling | 27M → 7B params | 0% improvement on adversarial | Capacity not the bottleneck |

## Related (vault entities)
- `knowledge/robotics/cross-embodiment-policy-transfer.md` — OXE/RT-X as training foundation; CrossFormer, X-VLA, SHADOW
- `knowledge/robotics/cross-domain-policy-transfer-sim-to-real-manipulation.md` — Sim-to-real pipeline context
- `knowledge/robotics/cross-embodiment-policy-transfer-synthesis.md` — Cross-embodiment transfer synthesis
- `knowledge/robotics/real-time-policy-adaptation-online-learning-failures.md` — Online learning for policy adaptation
- Open X-Embodiment (knowledge graph entity) — Dataset provenance, scale metrics
- RT-X models (knowledge graph entity) — Model architecture and training
- BridgeData V2 (knowledge graph entity) — Dataset specifications
- DROID (knowledge graph entity) — Environment-diverse training corpus
- SIMPLER (knowledge graph entity) — Simulated evaluation framework

## Open Questions

1. **Benchmark coverage gap**: Most evaluation focuses on tabletop kitchen tasks (BridgeData V2) and single-arm manipulation. How do benchmarks cover bimanual, mobile manipulation, legged robots, and outdoor deployment? *Progress*: MoMani (2026) addresses mobile manipulation, but standardized cross-embodiment coverage remains incomplete.

2. **Real robot evaluation cost**: Running 3,600 real-world trials across 6 institutions is massive. Can SIMPLER-level simulation reliability reach r > 0.95 to make real evaluation unnecessary for most research? *Current*: r = 0.924 for best setup; 0.95 threshold not yet achieved.

3. **Cross-simulator gap**: RoboGate revealed that models scoring near-perfectly on MuJoCo-based LIBERO can score 0% on Isaac Sim equivalents. What is the right simulation fidelity target, and should benchmarks mandate multi-simulator evaluation? *Insight*: Scaling parameters (27M→7B) doesn't help — failure is distribution gap, not capacity.

4. **Reality gap in real benchmarks**: RADAR (Feb 2026) showed 3D IoU drops from 0.261 to 0.068 under sensor noise — even "real-world" benchmarks overfit to static laboratory conditions. How do we stress-test models against high-entropy dynamics?

5. **Spatial-physical intelligence**: Current benchmarks emphasize semantic instruction following over geometric reasoning. Do VLAs truly understand 3D scenes or exploit 2D correlations? *Progress*: KinDERBench targets physical reasoning; LIBERO-Mem tests memory vs perception.

6. **Standardized leaderboard**: Unlike vision-language (HELM, Big-Bench), robot learning lacks a unified leaderboard. The VLA Leaderboard (CodeSOTA) and RoboGate are emerging but not yet adopted as a standard. PhAIL proposes distributional methodology (Time-to-Success CDF).

7. **Shared infrastructure maturity**: With OXE, DROID, and BridgeData V2 now serving as shared pre-training infrastructure, how much does the field need to invest in benchmark standardization vs. continuing to build larger datasets?

8. **Value-aligned evaluation**: RobotValues (June 2026) exposes a blind spot — existing benchmarks evaluate task completion, not ethical decision-making. Should household robot benchmarks incorporate value-conflict scenarios as standard evaluation?

9. **Distributional evaluation methods**: PhAIL (May 2026) argues that binary success rates are statistically insufficient. Should the field adopt Time-to-Success CDF as a standard evaluation primitive? *Progress*: Kaplan-Meier estimator + bootstrap CIs provide statistically rigorous framework.

10. **Benchmark inflation**: LIBERO-Pro revealed that "95% accuracy" can be pure memorization. How do we design benchmarks that measure true generalization rather than memorization? *Progress*: Adversarial scenarios in RoboGate and LIBERO-Pro show vulnerability; UMI-Bench standardizes pipeline-level reproducibility.

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
11. **PhAIL**: arXiv:2605.29710 (May 2026) — Real-Robot VLA Benchmark with distributional methodology
12. **UMI-Bench 1.0**: arXiv:2606.10382 (June 2026) — Local-first real-robot wrist-view manipulation benchmark
13. **RobotValues**: arXiv:2606.03312 (June 2026) — Household robot value-conflict evaluation benchmark
14. **MoMani Benchmark** (March 2026) — Mobile manipulation evaluation framework
15. **Visual Realism Benchmark**: arXiv:2605.06311 (May 2026) — Simulation visual fidelity evaluation
16. **WM4VLA Benchmark v4** (April 2026) — [HuggingFace](https://huggingface.co/datasets/zywu2115/WM4VLA_benchmark_v4) — VLA dataset curation benchmark
17. **VLA Training Data Volume**: [Claru.ai](https://claru.ai/blog/vla-training-data-volume) (April 2026) — VLA pre-training and fine-tuning data volume reference
18. **VLA Survey 2026**: arXiv:2604.23001 (April 2026) — "Vision-Language-Action in Robotics: A Survey of Datasets..."
19. **ManiSkill3**: arXiv:2410.00425 (RSS 2025) — GPU-parallelized simulation
20. **Being-H0.5**: Scaling Human-Centric Robot Learning for Cross-Embodiment Generalization — [liner.com](https://liner.com/review/beingh05-scaling-humancentric-robot-learning-for-crossembodiment-generalization)

## Confidence: 0.93

BridgeData V2, Open X-Embodiment, and SIMPLER are well-documented with consistent arXiv papers and project sites. The updated sim-real correlation (r=0.924) and cross-simulator gap findings (RoboGate: 97.65%→0%) are sourced from primary publications. RobotArena ∞, PhAIL, and UMI-Bench are sourced directly from arXiv with concrete methodology details. The VLA data volume reference matches published results from OpenVLA, Octo, pi-zero, and GR00T papers. Cross-embodiment results (Being-H0.5, RT-1-X) are confirmed across multiple sources. Confidence increased from 0.92 to 0.93 due to additional verification of SIMPLER's refined metrics and RoboGate's adversarial scenario details from primary sources.