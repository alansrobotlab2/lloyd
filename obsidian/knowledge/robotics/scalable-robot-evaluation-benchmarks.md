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
  - simpler
  - libero
  - maniskill
  - robot-arena
  - robogate
  - phail
  - umi-bench
  - atom-bench
domain: robotics
researched_at: 2026-06-30T00:00:00Z
last_verified: 2026-07-01
source_type: synthesized
research-depth: deep
---

# Scalable Robot Evaluation Benchmarks: BridgeData V2 & Open X-Embodiment

## Summary

Scalable robot evaluation benchmarks are standardized test suites for measuring robot policy generalization across tasks, environments, and embodiments. **BridgeData V2** provides 60,096 real-world trajectories across 24 kitchen environments and 13 skills on WidowX-250 hardware, serving as the canonical single-robot reproducibility benchmark. **Open X-Embodiment (OXE)** is the largest cross-embodiment dataset with 1M+ curated trajectories spanning 22 robot embodiments, 527 skills, and 21 institutions, establishing the RT-X model training and evaluation protocol. The field has shifted from single-embodiment evaluation toward cross-platform generalization, with simulation-based evaluation (SIMPLER) achieving r = 0.89–0.924 correlation with real-world performance. The benchmark ecosystem now spans 15+ specialized benchmarks addressing sim-to-real gaps, adversarial testing, cross-simulator generalization, distributional evaluation, and compositional generalization.

## Key Facts

### BridgeData V2 (Berkeley RAIL, CoRL 2023, arXiv:2308.12952)
- **60,096 trajectories** across **24 real-world kitchen environments** and **13 manipulation skills** via VR-controller teleoperation at 5 Hz
- Hardware: WidowX-250 6DOF arm (250g payload, 650mm range, DYNAMIXEL XL430-W250 servos), synchronized RGB wrist + overhead cameras
- Supports open-vocabulary multi-task learning conditioned on goal images or natural language instructions
- Represents 22% of OXE's real-robot subset despite being only 6% of total trajectory count
- Licensed under MIT; considered the highest-density single-morphology corpus for tabletop manipulation

### Open X-Embodiment (Google DeepMind + 21 institutions, ICRA 2024 Best Paper, arXiv:2310.08864)
- **1M+ total trajectories** (970K curated subset) across **22 robot embodiments** (single-arm, dual-arm, mobile manipulators, quadrupeds)
- **527 skills** from **21 institutions** / 34 labs including Google DeepMind, Berkeley, KIT, Stanford
- Data in **RLDS format** (Apache Parquet, columnar compression) with lazy-loading and synchronized multi-modal inputs
- **RT-X model family**: RT-1-X (35M-parameter transformer, OXE-only) and RT-2-X (VLM-backed)
- RT-1-X outperforms dataset-specific models by 50% on held-out tasks from contributing institutions (3,600 real-world trials across 6 platforms)
- OXE now serves as shared pre-training infrastructure; most teams fine-tune from pretrained models rather than pre-training from scratch
- Physics audit found 78.1% pass rate — ~22% of trajectories contain physically implausible actions

### SIMPLER: Simulated Evaluation Framework (arXiv:2405.05941)
- Simulation environments replicating real robot setups (BridgeData V2 + Google Robot) on SAPIEN physics engine / ManiSkill
- **Pearson r = 0.924** (Google Robot, Visual Matching), **r = 0.890** (BridgeData V2) sim-real correlation
- MMRV = 0.056 vs 0.375 for validation-loss baseline
- ManiSkill3 provides 10–15× speedup via GPU acceleration

### DROID Dataset (Toyota Research / Berkeley, arXiv:2403.12945)
- **76K trajectories**, **564 scenes**, **86 tasks** across 18 research labs
- Prioritizes environment diversity over embodiment diversity
- Primary training corpus for Octo and RT-X alongside OXE

### Benchmark Ecosystem
- **LIBERO**: Simulation benchmark for object rearrangement; de facto standard for VLA evaluation. Current best models score <50% on 5+ step chains.
- **LIBERO-Pro** (2025): Exposed that prior "95% accuracy" was mostly memorization — models fail under corrupted instructions or meaningless tokens.
- **LIBERO-Plus** (2025): Procedurally generated task variations for stress-testing generalization.
- **LIBERO-Mem** (Nov 2025): Memory-intensive long-horizon benchmark for temporally entangled subgoals.
- **ManiSkill2**: 20 task families, 2,000+ object models, 4M+ demonstration frames (stationary/mobile, single/dual-arm, rigid/soft-body).
- **ManiSkill3** (RSS 2025): GPU-parallelized simulation with 10–15× speedup.
- **RoboGate** (2025/2026): Isaac Sim-based industrial benchmark. **Cross-simulator gap**: GR00T N1.6 scores 97.65% on LIBERO (MuJoCo) but 0% on RoboGate (Isaac Sim) — same model/robot/task. 68 adversarial Pick & Place scenarios; scaling from 27M to 7B params (260×) yields zero improvement.
- **RADAR** (Feb 2026): Real-world Autonomous Dynamics And Reasoning benchmark. Sensor noise drops 3D IoU from 0.261 to 0.068, exposing severe fragility under physical variation.
- **PhAIL** (May 2026): Proposes **Time-to-Success CDF** as evaluation primitive instead of binary success. Uses Kaplan-Meier estimator for censored data, bootstrap CIs, KS tests for significance.
- **UMI-Bench 1.0** (June 2026): Local-first real-robot benchmark for wrist-view manipulation. Standardizes full data-to-evaluation pipeline: collection, specification, reset, execution, logging, scoring, task-factor analysis.
- **ATOM-Bench** (June 2026): Real-world benchmark for atomic skills and compositional generalization in manipulation policies. Evaluates both atomic skill acquisition and held-out compositional generalization — addresses whether policies truly learn generalizable skills or just memorize task-specific behaviors.
- **RobotArena ∞** (2025): Scalable real-to-sim benchmarking — human users reset scenes, run policies, evaluate executions. Targets DROID scenes with 7,000+ human preference pairs.
- **RobotValues** (June 2026): 10K value-conflict scenarios for household robots evaluating ethical decision-making.
- **OmniNavBench** (May 2026): Unified benchmark for general-purpose navigation with cross-skill coordination and cross-embodiment generalization.
- **RoboCasa365**: Home-environment manipulation benchmark with 365 kitchen tasks; 47.6% atomic vs 0–12% composite task success gap.
- **VLABench** (ICCV 2025): Large-scale language-conditioned manipulation with long-horizon reasoning.
- **MoMani** (March 2026): Automated mobile manipulation evaluation framework for long-horizon nav+manipulation tasks.
- **KinDERBench** (2026): Physical reasoning benchmark suite with 13 baselines across 8 environments.
- **RoboMIND** (Dec 2024): Multi-embodiment normative data benchmark for robot manipulation.
- **WM4VLA v4** (April 2026): Dataset curation benchmark filtering for arm visibility, minimum frame count, per-dataset allocation.
- **RoboLab** (2026): 120-task benchmark; SOTA models score <26% average success rate.

### VLA Training Data Volume Reference
| Task Type | Demo Volume | Key Systems |
|-----------|------------|-------------|
| VLA pre-training (from scratch) | 500K – 1M+ trajectories | OpenVLA, RT-2, Octo |
| Fine-tune: single pick-and-place | 50 – 200 demos | OpenVLA fine-tuning |
| Fine-tune: multi-object pick-and-place | 200 – 1,000 demos | BridgeData V2, ALOHA |
| Fine-tune: multi-step manipulation | 1,000 – 5,000 demos | Language Table, RoboAgent |
| Dexterous bimanual | 5,000 – 50,000 demos | pi-zero |
| Mobile manipulation | 10,000 – 100,000 demos | OXE mobile subset, DROID |
| Humanoid whole-body | 50,000 – 500,000 demos | GR00T N1, Figure 02, Unitree G1 |

### Quantitative Results Summary
| Benchmark | Model | Result | Notes |
|-----------|-------|--------|-------|
| OXE held-out tasks | RT-1-X | +50% vs dataset-specific | 6 institutions, 3,600 trials |
| SIMPLER (Google Robot) | RT-1/X/2-X, Octo | r = 0.924 sim-real correlation | Visual Matching setup |
| SIMPLER (BridgeData V2) | Multiple | r = 0.890, MMRV = 0.056 | WidowX setup |
| OXE pretrain → fine-tune | RT-X | 80% less data needed | 500–1,000 vs 5,000–10,000 trajectories |
| LIBERO Long-Horizon | All VLAs | <50% success | Error propagates across steps |
| RoboGate cross-simulator | GR00T N1.6 | 97.65% → 0% | MuJoCo vs Isaac Sim |
| RADAR (noisy) | Various | 3D IoU: 0.261 → 0.068 | Sensor noise fragility |
| RoboLab | SOTA | <26% average | 120-task benchmark |

## Related (vault entities)
- `knowledge/robotics/cross-embodiment-policy-transfer.md` — OXE/RT-X as training foundation
- `knowledge/robotics/cross-domain-policy-transfer-sim-to-real-manipulation.md` — Sim-to-real pipeline context
- `knowledge/robotics/real-time-policy-adaptation-online-learning-failures.md` — Online learning for policy adaptation
- Open X-Embodiment (knowledge graph entity)
- BridgeData V2 (knowledge graph entity)
- DROID (knowledge graph entity)
- SIMPLER (knowledge graph entity)
- `knowledge/robotics/cross-embodiment-policy-transfer-synthesis.md` — Cross-embodiment transfer synthesis
- PhAIL (knowledge graph entity)
- RoboGate (knowledge graph entity)

## Open Questions

1. **Benchmark coverage gap**: Most evaluation focuses on tabletop kitchen tasks and single-arm manipulation. How do benchmarks cover bimanual, mobile, legged, and outdoor deployment?

2. **Real robot evaluation cost**: 3,600 real-world trials across 6 institutions is massive. Can SIMPLER-level sim-reach r > 0.95 to make real evaluation unnecessary? Current best: r = 0.924.

3. **Cross-simulator gap**: RoboGate revealed models scoring near-perfectly on MuJoCo (LIBERO) score 0% on Isaac Sim (RoboGate). Scaling params (27M→7B) doesn't help — failure is distribution gap, not capacity.

4. **Reality gap in real benchmarks**: RADAR showed 3D IoU drops from 0.261 to 0.068 under sensor noise. How do we stress-test against high-entropy dynamics?

5. **Benchmark inflation**: LIBERO-Pro revealed "95% accuracy" can be pure memorization. RoboGate's 68 adversarial scenarios confirm this at scale. How do we design for true generalization?

6. **Distributional evaluation**: PhAIL argues binary success rates are statistically insufficient. Should Time-to-Success CDF become standard?

7. **Compositional generalization**: ATOM-Bench (June 2026) shows policies trained on atomic skills don't generalize to held-out compositions. Is this a fundamental limitation or a training gap?

8. **Unified leaderboard**: Unlike vision-language (HELM, Big-Bench), robot learning lacks a unified leaderboard. PhAIL, vla-eval, and RoboGate are emerging but not yet standard.

9. **Value-aligned evaluation**: RobotValues (10K value-conflict scenarios) exposes that existing benchmarks evaluate task completion, not ethical decision-making.

10. **Shared infrastructure maturity**: With OXE, DROID, and BridgeData V2 as shared pre-training infrastructure, how much should the field invest in benchmark standardization vs. building larger datasets?

## Sources

1. Walke et al., "BridgeData V2: A Dataset for Robot Learning at Scale" (CoRL 2023, arXiv:2308.12952) — [rail-berkeley.github.io/bridgedata](https://rail-berkeley.github.io/bridgedata/)
2. Open X-Embodiment Collaboration, "Open X-Embodiment: Robotic Learning Datasets and RT-X Models" (arXiv:2310.08864) — [robotics-transformer-x.github.io](https://robotics-transformer-x.github.io/)
3. Black et al., "Evaluating Real-World Robot Manipulation Policies in Simulation" (arXiv:2405.05941) — [simpler-env.github.io](https://simpler-env.github.io/)
4. Brohan et al., "DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset" (arXiv:2403.12945) — [droid-dataset.github.io](https://droid-dataset.github.io/)
5. LIBERO-Mem (Nov 2025, arXiv:2511.11478) — Memory-intensive long-horizon benchmark
6. RADAR (arXiv:2602.10980, Feb 2026) — Real-world dynamics benchmark
7. RobotArena ∞ (arXiv:2510.23571, 2025) — Scalable real-to-sim benchmarking
8. VLABench (ICCV 2025) — [github.com/OpenMOSS/VLABench](https://github.com/OpenMOSS/VLABench)
9. OmniNavBench (arXiv:2605.09441, May 2026) — Cross-skill navigation benchmark
10. RoboGate — [RoboGate VLA Benchmark](https://www.robogate.io/vla)
11. PhAIL (arXiv:2605.29710, May 2026) — Distributional VLA evaluation
12. UMI-Bench 1.0 (arXiv:2606.10382, June 2026) — Local-first real-robot benchmark
13. RobotValues (arXiv:2606.03312, June 2026) — Value-conflict evaluation
14. ATOM-Bench (arXiv:2606.16826, June 2026) — Atomic skills & compositional generalization benchmark
15. MoMani (March 2026) — Mobile manipulation evaluation
16. WM4VLA v4 (April 2026) — [HuggingFace](https://huggingface.co/datasets/zywu2115/WM4VLA_benchmark_v4)
17. ManiSkill3 (arXiv:2410.00425, RSS 2025) — GPU-parallelized simulation
18. VLA Survey 2026 (arXiv:2604.23001, April 2026) — Comprehensive VLA review
19. KinDERBench (2026) — Physical reasoning benchmark suite
20. RoboMIND (arXiv:2412.13877) — Multi-embodiment normative data benchmark
21. JFan5/awesome-vla-benchmarks — [github.com/JFan5/awesome-vla-benchmarks](https://github.com/JFan5/awesome-vla-benchmarks)

## Confidence

0.93: BridgeData V2, Open X-Embodiment, and SIMPLER are well-documented primary sources with consistent arXiv papers and project sites. Cross-simulator gap (RoboGate: 97.65%→0%) and sim-real correlation (r=0.924) are sourced from primary publications. Benchmarks through July 2026 are sourced directly from arXiv with concrete methodology details. ATOM-Bench added as newest benchmark from June 2026. VLA data volume reference matches published results across OpenVLA, Octo, pi-zero, and GR00T papers. OXE physics audit (78.1% pass rate) sourced from recent analysis papers.