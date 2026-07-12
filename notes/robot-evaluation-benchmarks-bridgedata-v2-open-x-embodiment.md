# Scalable Robot Evaluation Benchmarks: BridgeData V2, Open X-Embodiment, and VLA Evaluation

## 1. BridgeData V2

### Overview
BridgeData V2 is a large-scale, real-world robotic manipulation dataset released by Berkeley RAIL Lab (Walke et al., 2023). It is designed to facilitate research in scalable robot learning, supporting both goal-conditioned and language-conditioned learning methods.

### Dataset Statistics
| Metric | Value |
|--------|-------|
| **Total Trajectories** | 60,096 (50,365 expert demonstrations + 9,731 scripted/autonomous) |
| **Skills** | 13 distinct skills |
| **Environments** | 24 distinct environments (toy kitchens, sinks, tabletops, laundry machines) |
| **Objects** | 100+ distinct objects |
| **Robot Platform** | WidowX 250 (6DOF robot arm, ~$4,000 total cost) |
| **Sensing** | RGBD over-the-shoulder camera, 2x randomized-pose RGB cameras, wrist-mounted RGB camera |
| **Resolution** | 640x480 images |
| **Control Frequency** | 5 Hz |
| **Action Space** | 7D (6D Cartesian end-effector pose + gripper discrete open/close) |
| **Data Collection** | 84% human (VR teleoperation), 16% scripted/autonomous |
| **Labels** | Natural language instructions (annotated post-hoc via crowdsourcing) |

### Skills Covered
1. Pick-and-place (foundational)
2. Pushing
3. Reorienting objects
4. Opening doors
5. Closing doors
6. Opening drawers
7. Closing drawers
8. Wiping surfaces
9. Folding cloths
10. Stacking blocks
11. Twisting knobs
12. Flipping switches
13. Sweeping granular media (with tool)
14. Turning faucets
15. Zipping zippers

(13 skills counting bidirectional motions as single skills)

### Environments (24, grouped into 4 categories)
- **Toy Kitchens** (7 distinct): combinations of sinks, stoves, microwaves
- **Tabletops**: various tabletop setups
- **Standalone Sinks**: toy sink environments
- **Other**: toy laundry machines, etc.

### Key Evaluation Results (from BridgeData V2 paper)
Evaluation on **seen tasks** (in-distribution) — success rates averaged over 10 trials:

| Method | Open Drawer | Sweep Beans | Fold Cloth | Stack Block | Put Corn | Put Carrot | Flip Pot | Put Eggplant | Average |
|--------|-----------|------------|-----------|------------|----------|-----------|----------|------------|---------|
| GCBC | 0.4 | 0.9 | 0.4 | 0.4 | 0.9 | 0.7 | 0.1 | 0.1 | **0.49** |
| D-GCBC | 0.6 | 0.9 | 0.7 | 0.2 | 0.8 | 0.4 | 0.1 | 0.2 | **0.49** |
| ACT | 0.5 | 0.9 | 0.7 | 0.3 | 0.8 | 0.1 | 0.0 | 0.0 | **0.41** |
| RT-1 | 1.0 | 0.6 | 0.9 | 0.0 | 0.0 | 0.8 | 0.4 | 0.2 | **0.49** |

Evaluation on **unseen tasks** (OOD generalization):
- GCBC: 0.60, D-GCBC: 0.55, RT-1: 0.50 across unseen objects/environments

**Cross-institution generalization** (Lab 1 → Lab 2):
- RT-1 maintained best cross-lab transfer (0.47 → 0.40)
- Goal-conditioned methods showed more degradation

### Key Papers
- **Walke et al., 2023** — "BridgeData V2: A Dataset for Robot Learning at Scale" (CoRL 2023)
  - arXiv: [2308.12952](https://arxiv.org/abs/2308.12952)
  - Website: [rail-berkeley.github.io/bridgedata](https://rail-berkeley.github.io/bridgedata/)
  - Code: [rail-berkeley/bridge_data_v2](https://github.com/rail-berkeley/bridge_data_v2)

---

## 2. Open X-Embodiment (OpenX)

### Overview
Open X-Embodiment is the largest open-source real robot dataset, released by a collaboration including Google DeepMind, Stanford, and others (Katz et al., 2023). It unifies diverse robot datasets into a common format for cross-embodiment learning.

### Dataset Statistics
| Metric | Value |
|--------|-------|
| **Total Trajectories** | 1,000,000+ (≈1M+) |
| **Robot Embodiments** | 22 different robot platforms |
| **Skills** | 527 distinct skills |
| **Tasks** | 160,266 distinct tasks |
| **Robot Types** | Single-arm robots, bi-manual robots, quadrupeds |
| **Data Format** | Unified format, accessible via TensorFlow Datasets (TFDS) or Google Cloud Storage |
| **Component Datasets** | ~15 component datasets including BridgeData V2, DROID, RT-1 data, Roboset, and others |

### Included Component Datasets
The Open X-Embodiment mixture combines data from:
- **BridgeData V2** (~60K trajectories, WidowX)
- **DROID** (76K trajectories, 564 scenes, 86 tasks, 52 buildings)
- **RT-1** data (Google warehouse robot)
- **Roboset** (~98.5K trajectories, 6 skills, 11 environments)
- **RH20T** (13K trajectories, 41 skills, 50 environments)
- **Calvin** (simulated kitchen dataset)
- And others (TACO Play, Leap, etc.)

The Franka robot is the most commonly represented embodiment.

### RT-X Models
The release includes **RT-X model checkpoints** for:
- **RT-1-X**: Trained on Open X-Embodiment dataset; showed 50% average improvement over RT-1 on cross-embodiment tasks
- **RT-2-X**: 55B parameter Internet-pretrained VLA, fine-tuned on robot data
- Models support inference and fine-tuning on diverse robot embodiments

### Key Papers
- **Katz et al., 2023** — "Open X-Embodiment: Robotic Learning Datasets and RT-X Models"
  - arXiv: [2310.08864](https://arxiv.org/abs/2310.08864)
  - Website: [robotics-transformer-x.github.io](https://robotics-transformer-x.github.io/)
  - GitHub: [google-deepmind/open_x_embodiment](https://github.com/google-deepmind/open_x_embodiment)

---

## 3. Benchmark Comparison: BridgeData V2 vs Open X-Embodiment

| Aspect | BridgeData V2 | Open X-Embodiment |
|--------|--------------|-------------------|
| **Scope** | Single robot (WidowX), focused manipulation | Multi-robot, multi-embodiment |
| **Trajectories** | 60,096 | 1,000,000+ |
| **Skills** | 13 | 527 |
| **Environments** | 24 | 100+ (across all components) |
| **Embodiments** | 1 (WidowX 250) | 22 (Franka, WidowX, etc.) |
| **Collection** | Controlled, systematic | Aggregated from many sources |
| **Focus** | Deep coverage, generalization axes | Breadth, cross-embodiment transfer |
| **Use Case** | Fine-tuning benchmark, generalization evaluation | Foundation model pretraining |
| **Evaluation** | Standardized tasks with controlled OOD tests | Less standardized, dataset-level |

**Key Differences:**
1. **BridgeData V2** is a curated benchmark designed for controlled generalization evaluation (novel objects, novel scenes, novel instructions). It provides a well-defined test suite.
2. **Open X-Embodiment** is a training corpus designed for foundation model pretraining across embodiments. Its primary value is data diversity and scale.
3. OpenVLA is trained on a curated 970K trajectory subset of Open X-Embodiment, then evaluated *on* BridgeData V2 tasks — making BridgeData V2 the *evaluation benchmark* and Open X-Embodiment the *training data*.

---

## 4. VLA Model Evaluation: OpenVLA and the Benchmark Ecosystem

### OpenVLA Model Architecture
| Component | Details |
|-----------|---------|
| **Parameters** | 7 billion (7B) |
| **VLM Backbone** | LLaVA-7B (Llama 2 language model + vision encoder) |
| **Vision Encoder** | Fused DINOv2 (spatial features) + SigLIP (semantic features) |
| **Resolution** | 224 × 224 px |
| **Action Output** | Discretized action tokens in LLM vocabulary |
| **Training Data** | 970K trajectories from Open X-Embodiment |
| **Training** | Full fine-tuning of LLaVA-7B on robot data |
| **Paper** | Kim et al., 2024, CoRL 2024 (arXiv: 2406.09246) |

### Evaluation Suite

#### BridgeData V2 WidowX Evaluation (17 tasks, 10 trials each = 170 rollouts)
Tasks organized by generalization axis:
- **Visual generalization**: unseen backgrounds, distractors, object colors/appearances
- **Motion generalization**: unseen object positions, new initial configurations
- **Semantic generalization**: novel object-instruction pairings, multi-object grounding
- **Language grounding**: multi-instruction tasks requiring correct object selection

**Results (OpenVLA vs. baselines):**
| Model | Parameters | BridgeData V2 Avg Success |
|-------|-----------|--------------------------|
| **OpenVLA** | 7B | **~71.3%** (bfloat16) |
| RT-2-X | 55B | ~54.8% |
| RT-1-X | 335M | Much lower |
| Octo | 1B | Much lower |

OpenVLA outperformed the 55B RT-2-X by **16.5% absolute success rate** across 29 evaluation tasks on both WidowX and Google Robot embodiments, despite having **7× fewer parameters**.

#### Google Robot Evaluation (12 tasks, 5 trials each = 60 rollouts)
- In-distribution and out-of-distribution tasks on the mobile manipulator
- OpenVLA and RT-2-X attained comparable performance on Google robot
- OpenVLA significantly outperformed RT-2-X on BridgeData V2 tasks

#### Quantized Inference Performance (8 representative BridgeData V2 tasks, 80 rollouts)
| Precision | Bridge Success | VRAM |
|-----------|---------------|------|
| bfloat16 | 71.3 ± 4.8% | 16.8 GB |
| int8 | 58.1 ± 5.1% | 10.2 GB |
| **int4** | **71.9 ± 4.7%** | **7.0 GB** |

4-bit quantization matches bfloat16 performance while using half the memory.

### Fine-Tuning Results
**LoRA fine-tuning** on OpenVLA:
- LoRA (r=32) achieves best performance-compute trade-off
- Trains only **1.4% of model parameters**
- Matches full fine-tuning performance
- Evaluated across 7 diverse manipulation tasks (pick-and-place to table cleaning)

**LIBERO Simulation Benchmark** (v2 addition):
OpenVLA fine-tuned via LoRA on four LIBERO suites:
- LIBERO-Spatial
- LIBERO-Object
- LIBERO-Goal
- LIBERO-10 (LIBERO-Long)

Fine-tuned OpenVLA significantly outperforms fine-tuned Octo on LIBERO, especially on tasks requiring language grounding in multi-task settings with multiple objects.

---

## 5. Related Benchmarks

### RLBench (Robot Learning Benchmark)
- **100 unique hand-designed tasks** ranging from simple reaching to multi-stage manipulation
- Built on **MuJoCo** physics engine
- Tasks include: target reaching, door opening, drawer manipulation, etc.
- Emphasis on few-shot learning and meta-learning
- Tasks written in Python, easy to modify
- **Paper**: [RLBench (arXiv:1909.12271)](https://arxiv.org/abs/1909.12271)
- **GitHub**: [stepjam/RLBench](https://github.com/stepjam/RLBench)
- **Website**: [sites.google.com/view/rlbench](https://sites.google.com/view/rlbench)

### LIBERO
- **4 task suites, 40 total tasks**: Spatial, Object, Goal, Long Horizon (LIBERO-10/Long)
- Franka arm robot in tabletop manipulation setting
- Designed for evaluating generalization in simulated environments
- Language-conditioned tasks
- Used as evaluation benchmark for VLA fine-tuning

### CALVIN
- Simulated kitchen environment dataset
- Included in Open X-Embodiment
- Focuses on long-horizon tasks in kitchen settings

### Other Notable Benchmarks
- **RH20T**: 13K trajectories, 41 skills, 50 environments
- **DROID**: 76K trajectories, 350 hours, 564 scenes, 86 tasks across 52 buildings
- **Roboset**: 98.5K trajectories, 6 skills, 11 environments
- **ManiSkill / ManiSkill2**: UC San Diego simulation benchmark
- **FetchBench**: Simulation benchmark for robot fetching tasks

---

## 6. Evaluation Metrics Summary

### Common Metrics
| Metric | Description | Used By |
|--------|------------|---------|
| **Success Rate** | Binary task completion, averaged over trials | BridgeData V2, all benchmarks |
| **In-Distribution vs OOD** | Separate evaluation on seen vs unseen generalization | BridgeData V2, Google Robot |
| **Cross-Institution Transfer** | Performance at different labs/institutions | BridgeData V2 |
| **Per-Task Breakdown** | Task-level analysis across generalization axes | OpenVLA evaluations |
| **Inference Speed** | Hz control frequency achievable | OpenVLA |
| **Memory Footprint** | VRAM requirements | OpenVLA |

### What Makes These Benchmarks Different

1. **BridgeData V2** — The gold standard for real-robot manipulation evaluation. Controlled OOD axes (visual, motion, semantic), systematic evaluation protocol, single-robot reproducibility.

2. **Open X-Embodiment** — Not an evaluation benchmark per se; a training corpus. Value is in data scale and diversity for pretraining foundation models.

3. **RLBench** — Simulation-based, 100 tasks, emphasis on few-shot/meta-learning. Complementary to real-robot benchmarks.

4. **LIBERO** — Simulation benchmark for long-horizon, language-conditioned tasks. Used for VLA fine-tuning evaluation.

5. **Google Robot Evaluation** — Real-world warehouse mobile manipulator. Tests cross-platform transfer and real-world robustness.

---

## Key Source URLs

| Resource | URL |
|----------|-----|
| OpenVLA paper (arXiv) | https://arxiv.org/abs/2406.09246 |
| OpenVLA website | https://openvla.github.io/ |
| OpenVLA GitHub | https://github.com/openvla/openvla |
| OpenVLA OpenReview | https://openreview.net/pdf?id=ZMnD6QZAE6 |
| BridgeData V2 paper | https://arxiv.org/abs/2308.12952 |
| BridgeData V2 website | https://rail-berkeley.github.io/bridgedata/ |
| BridgeData V2 GitHub | https://github.com/rail-berkeley/bridge_data_v2 |
| Open X-Embodiment paper | https://arxiv.org/abs/2310.08864 |
| Open X-Embodiment website | https://robotics-transformer-x.github.io/ |
| Open X-Embodiment GitHub | https://github.com/google-deepmind/open_x_embodiment |
| RLBench | https://sites.google.com/view/rlbench |
| RLBench GitHub | https://github.com/stepjam/RLBench |
| LIBERO | https://libero-robotics.github.io/ |
| VLA Benchmarks (curated list) | https://github.com/JFan5/awesome-vla-benchmarks |