---
type: research-note
tags: [robotic-grasping, part-level-grasping, yolo-seg, sam2, instance-segmentation, vision-language-action, foundation-models]
domain: robotics
date: 2026-06-30
last_verified: 2026-06-30
sources:
  - url: "https://arxiv.org/abs/2512.02609"
    title: "SAM2Grasp: Prompt-Conditioned Temporal Action Prediction for Robotic Grasping"
  - url: "https://arxiv.org/abs/2408.00714"
    title: "SAM 2: Segment Anything in Images and Videos"
  - url: "https://github.com/ultralytics/ultralytics"
    title: "Ultralytics YOLO (YOLOv8/v9/v10/v11 — instance segmentation)"
  - url: "https://docs.ultralytics.com/models/sam-2"
    title: "SAM 2: Segment Anything Model 2 — Ultralytics Integration"
---

# Real-Time Object Recognition for Robotic Grasping: YOLO-seg and SAM2 for Part-Level Grasping

## Summary

Part-level robotic grasping targets specific functional sub-components (handles, knobs, edges) rather than treating objects as monolithic entities. The dominant paradigm chains a fast instance segmentation model (YOLO-seg) to extract graspable regions, then uses a foundation model like SAM2 for fine-grained promptable mask refinement and temporal tracking. **SAM2Grasp** (Wu et al., CVTE, arXiv 2512.02609) demonstrates this architecture end-to-end: a YOLO-seg detector proposes bounding-box prompts, SAM2 tracks the target across video frames via streaming memory, and a lightweight ACT (Action Chunking Transformer) head converts segmented regions into grasp poses. The full pipeline achieves 87.8% simulation success and 97.0% real-world bin-picking success with asynchronous inference at 20 Hz policy + 100 Hz control.

## Key Facts

- **YOLO-seg as the fast front-end**: YOLO's instance segmentation head (YOLOR/DeepSort-style post-processing) detects graspable regions at 30–60 FPS on RTX-class hardware. It produces bounding boxes and coarse masks that serve as prompt inputs for downstream foundation models. Ultralytics YOLO8+ supports instance segmentation with a single forward pass, making it the practical speed choice for on-device grasp pipelines.

- **SAM2 for temporal tracking and mask refinement**: Meta's SAM2 (ICLR 2025) extends SAM to video via streaming memory architecture, enabling promptable segmentation that tracks objects across frames. In grasping pipelines, an initial bounding-box prompt (from YOLO-seg) designates the target object, and SAM2 autonomously tracks it without further prompts — critical for handling occlusion and partial visibility during approach maneuvers.

- **SAM2Grasp architecture** (Wu et al., CVTE, 2025): Frozen SAM2 backbone + lightweight ACT policy head. Two-stage training: offline feature extraction from frozen SAM2, then supervised training of only the ACT head. This avoids catastrophic forgetting and is significantly faster than end-to-end fine-tuning. Asynchronous deployment: 20 Hz policy thread + 100 Hz control thread. Results: 87.8% sim success, 97.0% real-world average, 77% success at 40% frame dropout, 66% at 60%.

- **Instance segmentation → grasp proposal paradigm**: The dominant pipeline stages are: (1) segment objects/parts via YOLO-seg or SAM2, (2) sample candidate grasp poses within segmented regions, (3) filter/score candidates using a discriminator or diffusion-based policy. This is more robust than end-to-end pixel-to-grasp because it explicitly reasons about object geometry.

- **Part-level vs. object-level grasping**: Part-level grasping enables task-oriented actions (rotate doorknobs, press buttons, open cabinets) by identifying functional sub-components. It generalizes better across unseen objects but demands more compute. Benchmarks like OCID-grasp provide grasp-specific part annotations for training.

- **Speed tradeoffs**: YOLO-seg handles the latency-critical detection layer; SAM2 handles the accuracy-critical mask refinement. Together they form a speed–accuracy Pareto-optimal pair for real-time grasping, though SAM2 inference (~200–500 ms per frame on RTX 3090) requires asynchronous buffering to maintain 20 Hz policy loops.

## Related (vault entities)
- [[Real-Time VLM-Based Perception for Mobile Manipulation: Fused Depth-Pose-Segmentation Pipeline]]
- [[Vision-Language-Action Models]]
- [[Semantic Segmentation Robotics]]
- [[Mobile Manipulation]]
- [[Edge AI for Robotics]]
- [[Part-level grasping]]
- [[Instance segmentation → grasp proposal]]
- [[SAM2Grasp]]
- [[OCID-grasp]]
- [[Robotic grasping systems]]

## Open Questions

- How does SAM2's streaming memory interact with multi-object scenes? Current work assumes single-target tracking — multi-object part-level grasping remains underexplored.
- What is the optimal prompt strategy for SAM2 in grasping? Bounding boxes from YOLO-seg work well, but point/box-text prompts (language-conditioned) could enable open-vocabulary part-level grasping.
- Can the YOLO-seg + SAM2 pipeline be distilled to a single model for edge deployment (Jetson Orin-class)? Current two-model approaches strain onboard compute.
- How does SAM2's temporal tracking quality degrade under rapid gripper motion (self-occlusion, viewpoint change)? Robustness to approach-angle variance is uncharacterized.
- What benchmarks exist for comparing SAM2-based grasping against diffusion-based grasp policies (DiffusionGrasp, GraspDiffusion) on standardized datasets?
- Can SAM2's promptable interface enable language-driven part-level grasping ("grasp the handle of the blue cup") without retraining?

## Sources

- Wu et al., "SAM2Grasp: Prompt-Conditioned Temporal Action Prediction for Robotic Grasping" (CVTE, arXiv 2512.02609, 2025) — primary source for SAM2+YOLO-seg grasping pipeline
- Kirillov et al., "SAM 2: Segment Anything in Images and Videos" (Meta FAIR, ICLR 2025, arXiv 2408.00714) — SAM2 architecture and streaming memory
- Ultralytics YOLO documentation — instance segmentation head capabilities and speed benchmarks
- SAM2Grasp framework summary (vault entity) — 97.0% real-world bin picking, asynchronous 20 Hz/100 Hz deployment
- Instance segmentation → grasp proposal overview (vault entity) — dominant paradigm for segmentation-anchored grasp generation
- Part-level grasping overview (vault entity) — functional sub-component targeting vs. object-level grasping
- VLM perception pipeline note — broader context on fused depth-pose-segmentation pipelines [[Real-Time VLM-Based Perception for Mobile Manipulation]]

## Confidence

0.82: The core architectural claim (YOLO-seg as fast detector → SAM2 as temporal tracker/mask refiner → ACT policy head) is directly sourced from SAM2Grasp (arXiv 2512.02609) with concrete performance numbers (87.8% sim, 97.0% real). SAM2's capabilities (streaming memory, promptable video segmentation) are well-documented from the original SAM2 paper (ICLR 2025). The YOLO-seg speed profile is well-established from Ultralytics benchmarks. Confidence is tempered because: (1) SAM2Grasp is a single-group result — independent replication is needed; (2) real-world success rates depend heavily on the specific bin-picking scenario; (3) the YOLO-seg + SAM2 combination is demonstrated primarily in SAM2Grasp; broader adoption across the community is still emerging; (4) multi-object and language-conditioned grasping with this architecture remain research frontiers.