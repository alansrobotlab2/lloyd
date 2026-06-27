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
  - hug-bench
  - radar-bench
  - vlabench
  - robogate
domain: robotics
researched_at: 2026-06-06T12:00:00Z
source_type: synthesized
research-depth: medium
---

# Scalable Robot Evaluation Benchmarks: BridgeData V2, Open X-Embodiment, and the Benchmarking Landscape

## Summary

Scalable robot evaluation benchmarks are the standardized test suites used to measure how well robot manipulation policies generalize across tasks, environments, and embodiments. **BridgeData V2** provides a real-world WidowX 250 manipulation corpus (60K trajectories, 24 environments) used as both a training dataset and evaluation benchmark. **Open X-Embodiment (OXE)** is the largest cross-embodiment dataset (1M+ trajectories, 22 embodiments) and defines the RT-X model family's evaluation protocol — training on diverse robots, testing on held-out tasks at partner institutions. **SIMPLER** introduced simulation-based evaluation that correlates with real-world success (r = 0.89), while broader benchmarking suites (LIBERO, ManiSkill, RoboSuite) cover simulation environments. The field is moving toward scalable, low-cost evaluation to accelerate generalist VLA model development.

## Key Facts

### BridgeData V2 (Berkeley RAIL, CoRL 2023, arXiv:2308.12952)
- **60,096 trajectories** collected across **24 real-world kitchen environments** and **13 manipulation skills**
- Hardware: WidowX 250 (6-DOF arm), VR-controller teleoperation, 5 Hz control frequency, avg 38 timesteps per trajectory
- Skills include pick-and-place (carrots on plates), object rearrangement (spoons on towels, eggplants in baskets), stacking cubes
- Available as standalone dataset with richer metadata (failure modes, intervention timestamps, negative examples) and as a subset of Open X-Embodiment
- Serves as the standard evaluation environment for generalist VLA policies alongside Google Robot setups

### Open X-Embodiment (Google DeepMind + 21 institutions, arXiv:2310.08864)
- **970,000+ trajectories** across **22 robot embodiments**, **527 skills**, from **21 contributing institutions**
- Embodiments span single-arm manipulators (Franka Emika FR3), dual-arm systems, mobile manipulators, and dexterous hands
- Data stored in **RLDS format** (Apache Parquet, columnar compression) with lazy-loading support
- **RT-X model family**: RT-1-X (35M-parameter transformer, OXE-only training) and RT-2-X (VLM-backed)
- RT-1-X evaluated via **3,600 real-world trials** across 6 robot platforms at Berkeley, Freiburg, NYU, Stanford, and USC
- **Key result**: RT-1-X outperforms dataset-specific models by 50% on held-out tasks from contributing institutions

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

### Evaluation Benchmarks Ecosystem
- **LIBERO**: Simulation benchmark for object rearrangement tasks; widely used for VLA evaluation. Current best models score <50% on 5+ step chains (LIBERO Long-Horizon).
- **LIBERO-Pro** (2025): Extended LIBERO with more diverse scenarios and distractors.
- **LIBERO-Plus** (2025): Additional task variants and evaluation dimensions.
- **LIBERO-Mem** (Nov 2025, arXiv:2511.11478): Memory-intensive manipulation benchmark for long-horizon, temporally entangled, and repetitive subgoals; tests robust memory rather than short-term perception
- **ManiSkill**: Digital twin framework supporting real-to-sim evaluation; includes BridgeData V2 evaluation tasks
- **RoboSuite**: Physics-based manipulation benchmark with task variety; Diffusion Policy achieves 80%+ on complex tasks
- **RoboCasa**: Home-environment manipulation benchmark; Being-H0.5 achieves 53.9% on low-res RGB
- **Habitat**: Navigation-focused benchmark
- **RobotArena** (emerging): Scalable benchmarking via real-to-sim translation
- **HUG-BENCH**: Robotic grasping benchmark with sim-to-real validation
- **VLABench** (ICCV 2025): Large-scale benchmark for language-conditioned robotics manipulation with long-horizon reasoning tasks
- **RoboGate** (2025/2026): Industrial robotics benchmark using Isaac Sim; revealed a "cross-simulator gap" — GR00T N1.6 scores 97.65% on LIBERO (MuJoCo) but 0% on RoboGate (Isaac Sim), same model/robot/task
- **RADAR** (Feb 2026, arXiv:2602.10980): Real-world Autonomous Dynamics And Reasoning benchmark; introduces systematic environmental dynamics, spatial reasoning tasks, and autonomous 3D evaluation. Key finding: sensor noise drops 3D IoU from 0.261 to 0.068, exposing severe fragility under physical variation
- **DROID** (Toyota Research / Berkeley, arXiv:2403.12945): 76K trajectories, 564 scenes, 86 tasks across 18 research labs; primary training corpus for Octo and RT-X alongside OXE

### VLA Evaluation Protocol
- Standard pipeline: Train policy on large corpus → Evaluate on held-out tasks in SIMPLER or real WidowX/Google Robot stations
- Success rate is the primary metric (task completion ratio over N trials)
- Cross-embodiment evaluation tests whether a model trained on diverse robots transfers to held-out embodiments
- Distribution shift evaluation: measure robustness to camera pose changes, texture variations, lighting, and distractors

### Quantitative Results Summary
| Benchmark | Model | Success Rate | Notes |
|-----------|-------|-------------|-------|
| OXE held-out tasks | RT-1-X | +50% vs dataset-specific | 6 institutions, 3,600 trials |
| SIMPLER (Google Robot) | RT-1, RT-1-X, RT-2-X, Octo | r > 0.85 sim-real correlation | 4 tasks |
| SIMPLER (BridgeData V2) | Multiple policies | r = 0.890, MMRV = 0.014 | WidowX setup |
| OXE pretraining → fine-tune | RT-X | 80% less fine-tuning data needed | 500-1,000 trajectories vs 5,000-10,000 |
| LIBERO Long-Horizon (5+ steps) | All known VLAs | <50% | Error propagates across steps |
| RoboSuite complex manipulation | Diffusion Policy | 80%+ | Simulation benchmark |
| RoboCasa | Being-H0.5 | 53.9% (low-res RGB) | Home-environment benchmark |

## Related (vault entities)
- `knowledge/robotics/cross-embodiment-policy-transfer.md` — OXE/RT-X as training foundation; CrossFormer, X-VLA, SHADOW
- `knowledge/robotics/cross-domain-policy-transfer-sim-to-real-manipulation.md` — Sim-to-real pipeline context
- Open X-Embodiment (knowledge graph entity) — Dataset provenance, scale metrics
- RT-X models (knowledge graph entity) — Model architecture and training
- BridgeData V2 (knowledge graph entity) — Dataset specifications

## Open Questions
1. **Benchmark coverage gap**: Most evaluation focuses on tabletop kitchen tasks (BridgeData V2) and single-arm manipulation. How do benchmarks cover bimanual, mobile manipulation, legged robots, and outdoor deployment?
2. **Real robot evaluation cost**: Running 3,600 real-world trials across 6 institutions is a massive effort. Can SIMPLER-level simulation reliability reach r > 0.95 to make real evaluation unnecessary for most research?
3. **Cross-simulator gap**: RoboGate revealed that models scoring near-perfectly on MuJoCo-based LIBERO can score 0% on Isaac Sim equivalents. What is the right simulation fidelity target, and should benchmarks mandate multi-simulator evaluation?
4. **Reality gap in real benchmarks**: RADAR (Feb 2026) showed 3D IoU drops from 0.261 to 0.068 under sensor noise — even "real-world" benchmarks overfit to static laboratory conditions. How do we stress-test models against high-entropy dynamics?
5. **Spatial-physical intelligence**: Current benchmarks emphasize semantic instruction following over geometric reasoning. Do VLAs truly understand 3D scenes or exploit 2D correlations?
6. **Standardized leaderboard**: Unlike vision-language (HELM, Big-Bench), robot learning lacks a unified leaderboard. The VLA Leaderboard (CodeSOTA) and RoboGate are emerging but not yet adopted as a standard.

## Sources
1. **BridgeData V2**: Walke et al., "BridgeData V2: A Dataset for Robot Learning at Scale" (CoRL 2023, arXiv:2308.12952) — [rail-berkeley.github.io/bridgedata](https://rail-berkeley.github.io/bridgedata/)
2. **Open X-Embodiment / RT-X**: Open X-Embodiment Collaboration, "Open X-Embodiment: Robotic Learning Datasets and RT-X Models" (arXiv:2310.08864) — [robotics-transformer-x.github.io](https://robotics-transformer-x.github.io/)
3. **SIMPLER**: Black et al., "Evaluating Real-World Robot Manipulation Policies in Simulation" (arXiv:2405.05941) — [simpler-env.github.io](https://simpler-env.github.io/)
4. **Awesome VLA Benchmarks**: [JFan5/awesome-vla-benchmarks](https://github.com/JFan5/awesome-vla-benchmarks) — Curated benchmark registry
5. **BridgeData V2 dataset analysis**: [truelabel.ai/models/bridgedata-v2-model](https://truelabel.ai/models/bridgedata-v2-model) — Dataset structure and usage
6. **OXE procurement analysis**: [truelabel.ai/glossary/open-x-embodiment](https://www.truelabel.ai/glossary/open-x-embodiment) — Scale, composition, and limitations
7. **VLA Leaderboard**: [CodeSOTA Robotics](https://www.codesota.com/robotics) — RT-2 vs OpenVLA vs Pi0 vs Octo comparison
8. **DROID Dataset**: Brohan et al., "DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset" (arXiv:2403.12945) — [droid-dataset.github.io](https://droid-dataset.github.io/)
9. **LIBERO-Mem**: arXiv:2511.11478 (Nov 2025) — Memory-intensive long-horizon manipulation benchmark
10. **RADAR**: Chen et al., "RADAR: Benchmarking Vision-Language-Action Generalization via Real-World Dynamics" (arXiv:2602.10980, Feb 2026) — Autonomous 3D evaluation with physical dynamics
11. **VLABench**: OpenMOSS, "VLABench: A Large-Scale Benchmark for Language-Conditioned Robotics Manipulation" (ICCV 2025) — [github.com/OpenMOSS/VLABench](https://github.com/OpenMOSS/VLABench)
12. **RoboGate Cross-Simulator Gap**: [RoboGate VLA Benchmark](https://www.robogate.io/vla) — GR00T N1.6 scores 97.65% on LIBERO (MuJoCo) vs 0% on RoboGate (Isaac Sim)
13. **ICLR 2026 VLA Analysis**: Breuss, "State of Vision-Language-Action Research at ICLR 2026" — Survey of 164 VLA submissions

## Confidence: 0.90
BridgeData V2 and Open X-Embodiment are well-documented (arXiv papers, project sites, GitHub repos) with consistent reported numbers across sources. SIMPLER's metrics (r = 0.890, MMRV = 0.014) are drawn directly from the arXiv paper and alphaXiv summary. The RT-X evaluation protocol (3,600 trials, 6 institutions, +50% improvement) is confirmed across multiple sources including the original paper and secondary analyses. Slightly reduced from 0.95 due to the broader benchmark ecosystem (LIBERO, ManiSkill, RoboSuite) being referenced at a higher level without deep dive into each; the focus remains on BridgeData V2 and OXE as the primary evaluation anchors.
