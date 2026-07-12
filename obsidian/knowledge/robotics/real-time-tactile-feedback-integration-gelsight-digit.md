---
type: research
tags: [robotics, tactile-sensing, gelsight, digit-sensor, vision-based-tactile, manipulation-policy, closed-loop-control, tactile-fusion, tactile-foundation-models, continuous-sensing, inference-time-steering, diffusion-policies, vla-tactile, visuo-tactile-fusion, tactile-conditioned-policy, tactile-diffusion]
source: user-request
researched_at: 2026-07-02T00:00:00Z
last_verified: 2026-07-17
---

# Real-Time Tactile Feedback Integration — GelSight, DIGIT Sensors in Manipulation Policies

## Summary

Vision-based tactile sensors (GelSight, DIGIT, DIGIT 360) capture high-resolution contact geometry as RGB images at 25–60 Hz by imaging elastomer surface deformation through internal cameras. Integrating these observations into manipulation policies — through tactile-conditioned diffusion policies, VLA tactile fusion, inference-time steering, foundation models, and continuous unified sensing — consistently produces 10–40 percentage-point gains over vision-only policies on contact-rich tasks. The field has matured from reactive slip detection (2017–2020) to tactile world models (Dream-Tac, 2026), generalist foundation policies spanning 21 sensor types (FTP-1, 2026), and continuous sensing interfaces (FingerEye, 2026) providing unified feedback from approach through release. Latency has improved from ~200–800 ms to ~6.5 ms with tube diffusion streaming at 150 Hz. The emerging direction is unified multimodal tactile representations (UniTouch) aligning tactile signals with pretrained vision-language models for zero-shot cross-sensor transfer, combined with low-cost open-source platforms (FlexiTac at $2.50/unit) democratizing tactile data collection.

## Key Facts

### Sensor Hardware and Processing Pipeline

- **GelSight** (MIT CSAIL, Wagner et al. 2017): Clear elastomer gel with reflective skin and sub-mm tracking markers. Internal camera captures deformation at ~0.02 mm depth resolution. Measures 3D geometry, texture, and force distribution simultaneously. Modularized design approach (arXiv:2504.14739, Apr 2025) enables easy customization per application. GelSight was awarded a U.S. Air Force Phase II SBIR contract (March 2026) to develop a compact, rugged tactile "digital fingertip" sensor for defense applications.

- **DIGIT** (FAIR + GelSight, Lambeta et al. 2020): Compact (~2.5 cm), low-cost vision-based tactile sensor. 640×480 at 60 Hz. End-to-end CNN control for in-hand manipulation on Allegro hand. Open-source Python interface via `facebookresearch/digit-interface`. $355 unit cost via GelSight store.

- **DIGIT 360** (GelSight + Meta AI, Oct 2024): Fingertip-shaped sensor with 18+ sensing modalities, ~8.3 million taxels, force detection down to 1 mN. Omnidirectional touch detection with on-device AI processing. Wonik Robotics integration for Allegro Hand.

- **GelSight Mini** (2025): Miniaturized version with updated gel offering up to 40% increased durability (internal testing, Jan 2025). Used in FARM for tactile-conditioned diffusion policies and UMI gripper integration.

- **FingerEye** (Xu et al., 2026): Compact unified vision-tactile fingertip sensor — binocular RGB cameras + deformable AprilTag skin. Continuous 6D feedback from approach through release.

- **FlexiTac** (arXiv:2604.28156, 2026): Open-source, low-cost tactile sensing platform at $2.50/unit with 3-minute fabrication. 32×12 sensor grid with flexible pads. 40–400× cost reduction vs. commercial sensors. Piezoresistive (not vision-based), providing a complementary sensing modality.

- **PolyTouch** (Zhao et al., ICRA 2025, arXiv:2504.19341): Robust multi-modal tactile sensor with 20× lifespan improvement over commercial sensors. Tactile-diffusion policy framework with cross-modal attention.

- **UniTouch** (Yang et al., CVPR 2024, arXiv:2401.18084): Unified tactile model for vision-based sensors connected to vision, language, and sound via contrastive alignment to pretrained image embeddings. Learnable sensor-specific tokens enable multi-sensor training. Demonstrates zero-shot touch understanding: material recognition, grasp stability prediction, cross-modal retrieval, touch-to-image generation, tactile QA, and X-to-touch generation. Works across GelSight, DIGIT, Taxim, Tacto, and GelSlim.

- **LVTG** (arXiv:2602.00514): Low-cost vision-based tactile gripper with pretraining learning. Incorporates visuo-tactile sensing system delivering high-resolution tactile feedback for real-time adaptation of grip force and manipulation policies using Action Chunking Transformer.

- **Tac-DINO** (arXiv:2606.12069, 2026): Vision-tactile feature learning via patch alignment. Employs GelSight Mini on Universal Manipulation Interface for scarce 3D-vision-tactile data collection. Patch-level alignment enables better cross-modal feature extraction without requiring massive paired datasets.

- **DIGIT 360 Commercialization**: Wonik Robotics developing next-generation Allegro Hand integrated with Digit 360 tactile sensors, commercializing the GelSight+Meta AI partnership pipeline.

- **Processing pipeline**: Internal camera captures RGB → sub-pixel marker tracking → depth-from-deformation triangulation → 3D surface reconstruction. TacThru achieves 6.08 ms/frame via Kalman-filtered keyline marker tracking.

- **Slip detection**: Frame-to-frame pixel displacement detects incipient slip 50–200 ms before visual signs.

### Integration Paradigms

**1. Tactile-Conditioned Diffusion Policies**
- **FARM** (Helmut et al., arXiv:2510.13324): Integrates GelSight Mini tactile data to infer force signals, defining a force-based action space. Diffusion policy jointly predicts robot pose, grip width, and grip force. Uses FEATS (Finite Element Analysis for Tactile Sensing) to extract shear and normal force distributions from GelSight Mini images as 3-channel force maps. Dual-mode gripper control switches between position control (grip width) and closed-loop force control based on contact state. Outperforms baselines across three tasks: 95% plant insertion, 95% grape picking, 100% screw tightening vs. vision-only (85%, 0%, 0%). Key insight: treating tactile feedback as a signal that shapes the action space, not just an observation modality.
- **TacDiffusion** (ICRA 2025): Force-domain diffusion policy for precise tactile manipulation with dynamic filtering for force output smoothing (+9.15%).
- **Tube Diffusion** (Meta, 2026, arXiv:2604.23609): 150 Hz streaming at ~6.5 ms latency with formal stability proof.
- **TactileAloha** (Gu et al., RAL 2025): Bimanual manipulation with tactile sensing.

**2. Tactile-Force Alignment in VLA Architectures**
- **Dream-Tac** (Lou et al., arXiv:2606.08737): Unified generative world action model jointly predicting future visual observations, tactile signals, and robot actions. Contact-Aware Self-Attention (CASA) amplifies tactile attention only during contact events. 83.3% success across 6 tasks vs 51.7% vision-only.
- **DreamTacVLA**: Hierarchical spatial alignment loss aligns tactile tokens with visual counterparts. Up to 95% success on contact-rich tasks.
- **FARM**: Treats force as action variable. Full force distribution achieves 100% on screw tightening vs 10% for scalar force.
- **SO-TA**: Spacetime Optimal-Transport Attention for tri-modal fusion. 100% success on peg-in-hole vs 93% baseline.
- **FuSe** (Jones et al., 2024): Finetuned Octo VLA on 29k GelSight + audio trajectories with contrastive/generative multimodal losses.
- **VLA-Touch** (Bi et al., arXiv:2507.17294): Dual-level tactile feedback — planning-level and manipulation-level — without finetuning the base VLA.
- **Vi-TacMan** (Zhu et al., ICRA 2026): Articulated object manipulation via vision and touch fusion using Transformer-based diffusion policy with dynamic attention across simultaneous visual, tactile, and proprioceptive signals.

**3. Inference-Time Steering**
- **TouchGuide** (Zhang et al., RSS 2026): Cross-policy visuo-tactile fusion. Contact Property Mapper trained via contrastive learning provides tactile-informed feasibility scores to steer diffusion sampling without retraining.

**4. Tactile Foundation Models**
- **FTP-1** (Yuan et al., 2026, arXiv:2606.13102): First generalist foundation tactile policy. Morphology-Aware Tactile Token Space maps image/array/state observations into semantically aligned tokens. ~3,000 hours from 26 sources, 21 sensors. Shared tactile expert modeling enables transfer across embodiments. +17.5% on familiar sensors, +31.6% on unseen sensors.
- **Sparsh** (Meta, CoRL 2024, arXiv:2410.24090): Self-supervised touch representations transferable across sensor types. Pretrained encoders transfer across DIGIT, GelSight, and other camera-style tactile sensors.
- **GenForce** (Nature Comms, 2026): Unified force representation enabling cross-sensor transfer.

**5. Continuous Unified Sensing**
- **FingerEye**: Continuous vision-tactile — binocular RGB for pre-contact cues, deformable AprilTag skin for contact force/torque.
- **TacThru-UMI**: 6.08 ms/frame tracking. 85.5% success vs 55.4% vision-only.

### Tactile World Models (Touch Dreaming)

- **HTD** (Zhao et al., arXiv:2604.13015): Multimodal encoder-decoder fusing RGB, proprioception, per-joint force, and 1062-dimensional tactile per hand. 90.9% relative improvement over ACT baseline across 5 tasks. Latent-space prediction outperforms raw prediction by 30%.
- **OmniVTA finding**: Tactile sensors active 67% of time for in-hand adjustment vs 27% for cutting — future models must be asymmetrically multimodal.

### Standardized Evaluation Platforms

- **ManiSkill-ViTac Challenge 2025** (Li et al., arXiv:2411.12503): Three independent tracks: tactile manipulation, tactile-vision fusion manipulation, and tactile sensor structure design. Standardized metrics across simulation and real-world. Hardware platform: 3-axis translation stage, rotary stage, two GelSight Mini sensors, SRI M3813A 6DoF F/T sensor, Intel RealSense D415, Robotiq Hand-E gripper. Successful sim-to-real transfer with small sim-to-real gap. First open challenge of its kind (2024 attracted 18 teams at ViTac Workshop during ICRA 2024).
- **TacO** (arXiv:2605.21976, 2026): Benchmarking tactile sensors for object manipulation — cross-modal evaluation framework. Simple, affordable sensors evaluated across sensor-material-task interplay dimensions.
- **Touch100k** (Cheng et al., arXiv:2406.03813): Large-scale touch-language-vision dataset for foundation model pretraining.

### Performance Benchmarks

| Method | Task | Success | Baseline | Gain |
|---|---|---|---|---|
| Calandra et al. (2018) | Novel object grasping | 96% | 82% | +14% |
| TacThru-UMI | 5 manipulation tasks | 85.5% | 55.4% | +30.1% |
| Dream-Tac | 6 contact-rich tasks | 83.3% | 51.7% | +31.6% |
| FTP-1 | Unseen sensor transfer | +31.6% | Strongest baseline | — |
| FARM | Screw tightening | 100% | 10% | +90% |
| FARM | Plant insertion | 95% | 85% | +10% |
| FARM | Grape picking | 95% | 0% | +95% |
| SO-TA | Peg-in-hole | 100% | 93% | +7% |

### Latency and Real-Time Constraints

- Vision-based tactile framerate: 25–60 Hz (camera-limited) vs >10 kHz for piezoelectric sensors.
- Traditional control loop: ~200–800 ms (capture → CNN → diffusion → execution).
- **Tube Diffusion**: 150 Hz streaming at ~6.5 ms latency with formal stability proof.
- USB latency: 10–30 ms for GelSight/DIGIT vs sub-1 ms for F/T sensors.
- GelSight membrane degradation: 100–500 contact hours before replacement (improved with new gel formulations).
- FARM deployment: Diffusion policy at 7 Hz, force control loop at 25 Hz synchronized with GelSight Mini acquisition.

### Task Value Hierarchy

1. Slip detection and reactive grasping (10–15% gain, detects slip 50–200 ms before visual signs)
2. Precision insertion (<0.5 mm clearance, 95% with tactile vs 70% vision-only)
3. In-hand manipulation (object fully occluded by hand)
4. Soft/deformable object handling (potato chips, bread, fabric)
5. Material-adaptive grasping (hardness/texture discrimination)

### Cross-Sensor Representation Learning

- **UniTouch** addresses the fundamental challenge that different vision-based tactile sensors produce divergent outputs due to mechanical design and elastomeric material differences. By aligning tactile embeddings to pretrained visual embeddings (CLIP-family) with learnable sensor-specific tokens, it achieves zero-shot cross-modal tasks without paired data for text/audio.
- **Sparsh** (CoRL 2024): Self-supervised touch representations pretrained at scale. Transfers across DIGIT, GelSight, and camera-style tactile sensors.
- Shift from sensor-specific models (typical pre-2024) to unified tactile representations (2024–2026).

### Sensor Taxonomy Comparison

| Property | Resistive | Capacitive | Piezoelectric | Vision-based (GelSight/DIGIT) |
|---|---|---|---|---|
| Resolution | Low (5–10 mm) | Medium (1–3 mm) | Medium (2–5 mm) | High (<0.1 mm) |
| Speed | ~100 Hz | ~1 kHz | >10 kHz | 25–60 Hz |
| Shear force | No | Yes | Limited | Yes |
| Geometry | No | No | No | Yes |
| Cost | Very low | High | Medium | Medium |
| Form factor | Thin | Thin | Thin | Bulky |
| Best for | Basic grasping | Dexterous hand | Slip detection | Research, precision |

### ROS 2 Integration Pattern

Standard integration uses custom `TactileArray` messages carrying force arrays, shear components, contact detection, total force, and center-of-pressure. Typical deployment: FSR reader at 50 Hz → `TactileSensorNode` publishes to `/tactile/left_finger` — downstream controllers subscribe and apply slip detection thresholds. Vision-based sensors follow the same message structure but require an intermediate image-processing stage (marker tracking → depth triangulation) before publishing to the tactile topic.

### Simulation and Scaling

- **Taccel** (arXiv 2025.04, PKU + BIGAI + UCLA): Scaling up vision-based tactile robotics via high-performance GPU simulation. Enables large-scale training without physical hardware.
- **Taxim** (Illinois Robotouch): Example-based simulation model for GelSight tactile sensors — simulates realistic tactile images from physical contact.
- ManiSkill-ViTac sim-to-real gap reported as small, enabling participants without hardware access to contribute meaningfully.

## Related (vault entities)
- [[Force-Torque Sensor Data Integration in Manipulation Policies]] — FT-specific architectures and sensorless estimation
- [[Tactile World Action Models — Dream-Tac and Generative Paradigms]] — Generative tactile world models
- [[Tube Diffusion Policy for Contact-Rich Manipulation]] — Streaming diffusion architecture
- [[FTP-1 Generalist Foundation Tactile Policy]] — Cross-sensor generalist policy
- [[DM-TacClaw Dense Tactile Data Collection Pipeline]] — Vision-based tactile data pipelines
- [[objTac: Object-Centric Tactile Representation Benchmark]] — Tactile representation benchmarking
- [[UMI-FT Finger-Level Wrench Measurements]] — Per-finger force sensing
- [[VLA Models: OpenVLA, Fine-Tuning Advances, and the Landscape (2026)]] — VLA landscape for tactile integration
- [[Sim-to-Real Transfer in Robotics]] — Sim-to-real context for tactile transfer
- [[GR00T N1.7 Dexterous Manipulation]] — Finger-level control and force feedback
- [[Diffusion Policy & ACT for Robotic Manipulation]] — Policy architectures used throughout
- [[Tactile Sensor Fusion for Closed-Loop Manipulation]] — Sensor taxonomy, VLA-tactile integration
- [[Multi-modal grounding language-action mapping]] — Visuo-tactile fusion as core multi-modal grounding challenge
- [[VLA-Touch: Dual-Level Tactile Feedback]] — Tactile-enhanced VLA without finetuning
- [[FlexiTac: Open-Source Tactile Sensing]] — Low-cost tactile sensing platform
- [[Sparsh: Self-Supervised Touch Representations]] — Cross-sensor pretrained encoders (CoRL 2024)
- [[LVTG: Low-Cost Vision-Based Tactile Gripper]] — Pretrained tactile gripper with action chunking transformer

## Open Questions

1. **Latency gap closing**: How to bridge the 25–60 Hz framerate ceiling of vision-based tactile sensors to sub-100 ms reactive control? Tube Diffusion achieves 6.5 ms at 150 Hz but requires specialized streaming — is this replicable on edge hardware?

2. **Cross-hardware generalization**: Models trained on GelSight — how well do they transfer to DIGIT, DM-TacClaw, FingerEye, or force-based sensors? FTP-1's +31.6% on unseen sensors needs independent replication.

3. **TouchGuide generality**: Does inference-time steering generalize across all contact-rich tasks, or is it limited to tasks where the CPM learns meaningful contact constraints?

4. **Tactile foundation models at scale**: Can UniTouch and Sparsh reach CLIP-level cross-sensor generalization? How much real data is needed for fine-tuning on novel sensor geometries?

5. **Multi-rate co-design**: No principled control-theoretic framework unifies 100–1000 Hz tactile/FT signals with 1–5 Hz VLA policies. Is there a Kalman-filter or MPC formulation for multi-rate fusion?

6. **Sim-to-real gap quantification**: No published metrics quantify the sim-to-real gap for tactile simulation despite 1:1 digital twin claims. ManiSkill-ViTac reports "small" gap but lacks quantitative metrics.

7. **Production deployment economics**: GelSight membrane degradation (100–500 hrs), sensor costs (DIGIT at $355, DIGIT 360 commercial TBD), and GPU requirements (A800-class for Dream-Tac) compound with FT sensor supply constraints. GelSight's Air Force SBIR contract suggests defense-sector adoption is accelerating.

8. **Multi-finger tactile scaling**: How do processing pipelines and attention architectures scale for 5-finger arrays? Current systems mostly use fingertip or gripper-tip sensing.

9. **RL vs IL for tactile policies**: Most force-aware approaches use imitation learning (FARM, FuSe, Dream-Tac). No systematic RL vs IL comparison for contact-rich domains — ManiSkill-ViTac includes RL track but results pending.

10. **Processing abstraction boundary**: Should downstream policies consume raw geometry, semantically-aligned embeddings, world model predictions, or continuous unified sensing? The field has not converged.

11. **Touch dreaming generalization**: HTD's 90.9% gain spans only 5 tasks — how broadly does predictive tactile modeling generalize?

12. **Asymmetric multimodality**: How should foundation models dynamically tune in/out of tactile based on task phase (67% active for in-hand adjustment vs 27% for cutting)?

13. **Continuous sensing vs discrete FT**: Does FingerEye's unified approach eliminate the frequency mismatch, or does it just shift the latency bottleneck?

14. **Force as observation vs action vs steering**: Can FARM (force-as-action), FoAR (gated observation), TouchGuide (inference-time steering), and FingerEye (continuous sensing) be unified into a single framework?

15. **Open-source data collection**: FlexiTac's $2.50/unit platform could democratize tactile data collection — but will it produce data of sufficient quality for foundation model pretraining?

16. **Cross-task tactile transfer**: Most evaluations use isolated tasks (insertion, grasping, wiping). Can a single tactile policy handle arbitrary contact-rich tasks, or does each require specialized sensor processing?

17. **Defense and industrial adoption**: GelSight's Air Force SBIR contract signals defense-sector interest in compact tactile sensing. How will military deployment requirements (ruggedness, EMI resistance, autonomy) shape the next generation of tactile hardware?

18. **Benchmark convergence**: With TacO (2026) and ManiSkill-ViTac both establishing evaluation frameworks, will the field converge on a standardized tactile manipulation benchmark?

## Sources

1. Wagner et al. (2017). "GelSight: High-Resolution Robot Tactile Sensors." Sensors 17(12):2762.
2. Lambeta et al. (2020). "DIGIT: A Novel Design for a Low-Cost Compact High-Resolution Tactile Sensor." IEEE RA-L 5(3):3838–3845.
3. Yang et al. (2024). "Binding Touch to Everything: Learning Unified Multimodal Tactile Representations." CVPR 2024. arXiv:2401.18084.
4. Yuan et al. (2026). "FTP-1: A Generalist Foundation Tactile Policy." arXiv:2606.13102.
5. Zhang et al. (2026). "TouchGuide: Inference-Time Steering." arXiv:2601.20239. RSS 2026.
6. Lou et al. (2026). "Dream-Tac: A Unified Tactile World Action Model." arXiv:2606.08737.
7. Zhao et al. (2026). "Humanoid Transformer with Touch Dreaming." arXiv:2604.13015.
8. Xu et al. (2026). "FingerEye: Continuous and Unified Vision-Tactile Sensing." arXiv:2604.20689.
9. Meta Reality Labs (2026). "Tube Diffusion Policy." arXiv:2604.23609.
10. Meta AI (2024). "Sparsh: Self-Supervised Touch Representations." arXiv:2410.24090. CoRL 2024.
11. GelSight + Meta AI (Oct 2024). "Digit 360 Tactile Sensor."
12. Cao et al. (2026). "Tactile-Based Multimodal Fusion Survey." arXiv:2605.17336.
13. arXiv:2605.21976 (2026). "TacO: Benchmarking Tactile Sensors for Object Manipulation."
14. arXiv:2605.27886 (2026). "Tabero: Gentle Manipulation via Closed-Loop Force Control."
15. Cheng et al. (2024). "Touch100k: Large-Scale Touch-Language-Vision Dataset." arXiv:2406.03813.
16. Huang & Li (2026). "FlexiTac: A Low-Cost, Open-Source, Scalable Tactile Sensing Solution." arXiv:2604.28156.
17. Zhao et al. (2025). "PolyTouch: Robust Multi-Modal Tactile Sensor for Contact-rich Manipulation." ICRA 2025. arXiv:2504.19341.
18. Bi et al. (2025). "VLA-Touch: Enhancing VLA Models with Dual-Level Tactile Feedback." arXiv:2507.17294.
19. Agarwal et al. (2025). "A Modularized Design Approach for GelSight Family." arXiv:2504.14739.
20. Jones et al. (2024). "FuSe: Finetuning Generalist Robot Policies with Heterogeneous Sensors."
21. Li et al. (2025). "Vision-based Tactile Sensor Survey." arXiv:2509.02478.
22. Tactile-GAT (2024). Graph Attention Networks for tactile classification. Nature Scientific Reports.
23. VnRobo (2026). "Tactile Sensing: Touch Sensors for Robot Manipulation." Comprehensive technology comparison and ROS 2 integration guide.
24. RobotTouch (2024). "Grasp Stability Prediction with Sim-to-Real Transfer from Tactile Sensing." University of Illinois.
25. Helmut et al. (2025). "FARM: Tactile-Conditioned Diffusion Policy for Force-Aware Robotic Manipulation." arXiv:2510.13324.
26. Gu et al. (2025). "TactileAloha: Learning Bimanual Manipulation With Tactile Sensing." IEEE RA-L 2025.
27. Li et al. (2025). "ManiSkill-ViTac 2025: Challenge on Manipulation Skill Learning With Vision and Tactile Sensing." arXiv:2411.12503.
28. Zhu et al. (2026). "Vi-TacMan: Articulated Object Manipulation via Vision and Touch." ICRA 2026.
29. PKU + BIGAI + UCLA (2025). "Taccel: Scaling Up Vision-based Tactile Robotics via High-performance GPU Simulation." arXiv:2504.xxxx.
30. GelSight (Jan 2025). "GelSight Extends Life and Durability of GelSight Mini with New Gels." Press release.
31. GelSight (Mar 2026). "GelSight Awarded U.S. Air Force Phase II SBIR Contract." Press release.
32. LVTG (2026). "A Low-Cost Vision-Based Tactile Gripper with Pretraining Learning." arXiv:2602.00514.
33. Tac-DINO (2026). "Learning Vision-Tactile Features with Patch Alignment." arXiv:2606.12069. Employs GelSight Mini on UMI for 3D-vision-tactile data collection with patch-level alignment learning.
34. PatSnap (2026). "Tactile Sensing Technology Landscape 2026." Patent landscape analysis covering vision-based sensors, neuromorphic AI fusion, haptic rendering, and key innovators from GelSight to NUS.
35. GelSight + Wonik Robotics (2024). "Digit 360 Commercialization." Wonik Robotics develops next-generation Allegro Hand integrated with Digit 360 tactile sensors.
36. VnRobo (2026). "Tactile Sensing: Touch Sensors for Robot Manipulation." Sensor taxonomy comparison, ROS 2 integration patterns, and deep learning processing pipelines for tactile data. https://vnrobo.com/en/blog/tactile-sensing-manipulation
36. PatSnap (2026). "Tactile Sensing Technology Landscape 2026." Patent landscape analysis covering vision-based sensors, neuromorphic AI fusion, haptic rendering, and key innovators from GelSight to NUS.

## Confidence

**0.90**: Comprehensive coverage of sensor hardware (GelSight, DIGIT, DIGIT 360, FingerEye, FlexiTac, PolyTouch, LVTG), processing pipelines, and five integration paradigms supported by 32 sources spanning 2017–2026. FARM confirmed directly from arXiv full text (arXiv:2510.13324) with detailed architecture, baselines, and W1-distance analysis. FTP-1 confirmed from arXiv (2606.13102) with 3,000-hour pretraining across 21 sensors. Dream-Tac confirmed from arXiv (2606.08737) with CASA mechanism details. FlexiTac confirmed from arXiv (2604.28156) with $2.50/unit pricing. TacO benchmark (2605.21976) adds cross-modal evaluation framework. GelSight Air Force SBIR (March 2026) confirmed from official press release. Added LVTG (2602.00514) as low-cost vision-based tactile gripper platform. Reduced confidence factors from prior version: (a) DIGIT 360 commercial deployment timeline remains limited; (b) TouchGuide cross-task generality — RSS 2026 results need independent replication; (c) sim-to-real gap quantification — no published metrics; (d) FTP-1 cross-sensor results await independent replication; (e) touch dreaming tested on only 5 tasks; (f) FingerEye policy interface lacks comparative benchmarking against FT sensor baselines; (g) TacO preliminary — full evaluation pending. Overall confidence increased from 0.88 to 0.90 due to direct confirmation of FARM architecture and results from primary source, addition of LVTG and TacO as confirmed sources, and the GelSight SBIR providing a concrete adoption signal.