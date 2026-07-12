---
title: Multi-Agent Task Decomposition — Hierarchical Planning with Subgoal Reasoning
tags:
  - robotics/multi-agent
  - ai/planning
  - ai/task-decomposition
  - ai/hierarchical-planning
  - ai/subgoal-reasoning
  - robotics/task-allocation
  - ai/llm-planning
  - ai/neuro-symbolic
  - research/domain-research
created: 2026-07-13
updated: 2026-08-16
confidence: 0.92
---

# Multi-Agent Task Decomposition — Hierarchical Planning with Subgoal Reasoning

## Summary

Multi-agent task decomposition breaks high-level natural-language goals into coordinated subgoals distributed across multiple robotic agents. Hierarchical planning organizes decomposition into structured layers: abstract plans refine progressively into concrete executable actions via subgoal trees, behavior trees, or HTN methods. Modern approaches combine classical planning (HTN, PDDL, MRTA) with LLM-based decomposition that recursively builds subgoal trees — root goal → child subgoals → primitive leaf actions — constraining each level to reason only about its parent node rather than the full context. The core challenge is the contextual and logical gap: as horizons extend, attention dilutes and abstract-to-concrete reasoning degrades; hierarchical architectures mitigate this by bounding each planning step to a manageable abstraction level.

## Key Facts

### Foundational Planning Frameworks

- **Hierarchical Task Networks (HTN)** — Classical paradigm where tasks decompose through learned methods specifying when/how to break abstract tasks into subtasks. Modern work (arXiv:2511.12901) integrates LLM-based chatbots with HTN planners, learning HTN methods from interaction traces rather than hand-coding them.
- **Multi-Robot Task Allocation (MRTA)** — Concurrently assigning robots to goals + generating coordinated trajectories. Ranges from centralized optimization to decentralized market-based, consensus, and coalition formation approaches. Recent work uses multimodal multi-objective evolutionary algorithms and deep RL for dynamic allocation under communication constraints.
- **PDDL-based Planning** — Planning Domain Definition Language remains the standard interface between high-level reasoning and classical planners (FF, Fast Downward). LLM frameworks generate PDDL problems from natural language goals, then solve them with classical planners for guaranteed correctness.
- **Dec-SGTS (Decentralized Sub-Goal Tree Search)** — AAAI 2021: hierarchical, anytime algorithm for decentralized multi-agent planning that integrates subgoal-pair connection and evaluation with tree expansion. Solves MMDPs by sharing subgoal-level information with enriched semantics. Traded computation time for solution quality in large joint action spaces.

### Hierarchical Architecture Patterns

- **Two-Level Hierarchical Decomposition** — Upper-layer LLM agent decomposes tasks and assigns subgoals to lower-layer agents, which generate PDDL problems solved by classical planners. "LLM brain + planner body" architecture separates reasoning from execution.
- **Subgoal Tree Construction (STEP, arXiv:2506.21030)** — Builds a tree where root = high-level goal, each level recursively decomposes parent subgoals, leaf nodes = primitive actions. A decomposition model uses foundation LLMs; a termination model checks mappability (can this map to a primitive?) and consistency (embodiment affordances + environment constraints). Achieves 34% on VirtualHome WAH-NL, 25% on real robots, outperforming flat LLM planners.
- **ReAcTree (arXiv:2511.02424, Choi et al., 2025)** — Hierarchical LLM agent tree with control flow nodes (sequence, fallback, parallel) inspired by Behavior Trees. Each agent node independently handles a subgoal via ReAct-style reasoning. Two memory systems: (1) episodic memory retrieves goal-specific examples from past runs; (2) working memory serves as a shared blackboard for cross-agent awareness. Achieves 61% goal success on WAH-NL with Qwen 2.5 72B, nearly doubling ReAct's 31%.
- **GRHP (Graph-Fused Hierarchical Planning, Li et al., 2025)** — Decouples planning into large models for high-level semantic decomposition and small models for precise action generation. Fuses semantic graphs (from LLM reasoning) with environmental graphs via cross-graph attention to address the grounding gap between abstract instructions and executable actions.
- **Behavior Trees** — Reactive, composable structures combining hierarchical decomposition with sensor-driven condition evaluation. EmboTeam (arXiv:2601.11063) grounds LLM reasoning into reactive BTs for multi-robot teams. MRBTP provides theoretical soundness and completeness guarantees for BT-based multi-robot planning.
- **Agentic Robot (arXiv:2505.23450)** — Brain-inspired SAP-driven framework: (1) LRM-based planner decomposes instructions into structured subgoals guided by an Atomic Skill Library; (2) VLA executor generates continuous control commands from subgoals + visual input; (3) VLM verifier assesses subgoal completion and triggers recovery. Achieves 79.6% average success on LIBERO, outperforming SpatialVLA by +6.1% and OpenVLA by +7.4% on long-horizon tasks. SAP ensures standardized protocols for information exchange, progress monitoring, and error recovery throughout execution.
- **RoboBrain (arXiv:2502.21257)** — Unified brain model mapping abstract instructions to concrete manipulation actions via a hierarchical framework. Produces clear step-subgoal sequences for verifiable task decomposition, with later versions (2.0, 2.5) adding spatiotemporal reasoning and dense temporal value estimation.
- **MADRA (AAMAS 2026, arXiv:2511.21460)** — Multi-Agent Debate for Risk-Aware Embodied Planning. Multiple LLM-based agents debate the safety of instructions, guided by a critical evaluator scoring responses on logical soundness, risk identification, evidence quality, and clarity. Features a hierarchical cognitive collaborative planning framework integrating safety, memory, planning, and reflection modules for self-evolution. Achieves >90% rejection of unsafe tasks while minimizing false rejections on SafeAware-VH benchmark (800 annotated instructions on VirtualHome).
- **LGC-MARL** — LLM-Based Graph Collaboration MARL (Jia et al., Mar 2025). Decomposes complex instructions using LLMs, then coordinates agents with a graph-based policy and critic-modeled feedback for multi-agent reinforcement learning. Integrates language models into graph-structured MARL, enhancing planning, credit assignment, and scalability.
- **H-WM (Hierarchical World Model, arXiv:2602.11291, Chen et al., 2026)** — Bilevel world model combining a high-level logical world model (LLM fine-tuned for symbolic planning) with a low-level visual world model conditioned on logical states. Generates compact latent visual subgoals grounded in perceptual space. Addresses compounding error accumulation in end-to-end VLA policies by providing stable intermediate guidance across both logical and visual hierarchies. Introduces a training dataset aligning robot motion with symbolic states, actions, and visual observations.
- **TeamWeaver (2026)** — Hybrid planning framework combining LLM semantic reasoning with MIQP (Mixed-Integer Quadratic Programming) constraint optimization. LLM decomposes natural-language instructions into structured subtasks; MIQP solver handles optimal task allocation and scheduling under resource constraints. Provides interpretable allocation decisions (from LLM decomposition) with provable optimality guarantees (from MIQP). Addresses a key gap: pure LLM decomposers lack optimality guarantees for allocation, while pure optimizers lack semantic understanding of natural-language goals.

### LLM-Based Subgoal Decomposition

- **LLM-inferred subgoals match human expert decompositions** while preserving plan correctness — demonstrated by converting N-agent problems into N single-agent sub-problems via LLM-based decomposition.
- **Least-to-Most Decomposition** — Sequential subgoal generation (solve each before generating the next), reducing error accumulation vs. simultaneous generation.
- **Hierarchical LLM-Based Multi-Agent Framework (Kawabe & Takano, arXiv:2602.21670)** — Two-layer architecture with TextGrad-inspired prompt optimization and meta-prompt sharing. On MAT-THOR: 95% on compound tasks, 84% on complex, 60% on vague (+2/+7/+15 pp over LaMMA-P). Ablation: hierarchical structure contributes +59, prompt optimization +37, meta-prompt sharing +4 percentage points.
- **H²R (Hierarchical Hindsight Reflection, arXiv:2509.12810)** — Maps natural-language task specs into structured intermediate subgoals, enhanced by hindsight reflection for multi-task LLM agents.
- **Other frameworks**: SMART-LLM (ICRA 2024), BrainBody-LLM, LLaMAR, Emergent Mind survey, CrewAI hierarchical process, Lil'Log (Lilian Weng, 2023).

### Neuro-Symbolic Planning Approaches

- **LLMTAMP (ICRA 2025)** — Neuro-symbolic language models decompose long-sequential goals into multi-level subgoals, achieving much faster planning than pure symbolic methods while maintaining high accuracy.
- **Online HTN Learning** — Learning hierarchical task network methods from LLM interaction traces, enabling chatbots to learn decomposition rules from experience.

### Core Challenges

- **Contextual Gap** — Long task sequences introduce redundant information into LLM context, causing attention dilution. Hierarchical decomposition mitigates by using parent nodes (not the full task) as context for each level.
- **Logical Gap** — Extended sequences correspond to increasingly abstract linguistic instructions. Inference from abstract directives to concrete actions degrades with complexity. Subgoal trees constrain each decomposition step to a manageable abstraction level.
- **Embodiment Constraints** — LLMs overlook robot capability limits (single-arm manipulation, payload, workspace boundaries). Leaf termination models must check executability given specific affordances.
- **Multi-Agent Coordination** — Decomposition must account for inter-agent dependencies, shared resources, collision avoidance, and communication constraints. Decentralized decomposition scales better but risks subgoal conflicts.
- **Error Propagation** — Static plan-following agents suffer compounding errors; end-to-end visuomotor policies lack introspection. SAP-driven closed-loop designs enable autonomous error detection and recovery.

### Benchmark & Evaluation Landscape

- **MAT-THOR** — Multi-agent household benchmark for long-horizon heterogeneous robot planning; Kawabe & Takano achieve 95% on compound tasks.
- **VirtualHome WAH-NL** — Multi-room embodied task benchmark; STEP: 34%, ReAcTree: 61% with Qwen 2.5 72B.
- **ALFRED** — Visual reasoning for long-horizon embodied tasks in simulated environments.
- **LIBERO** — Long-horizon robotic benchmark for bimanual manipulation; Agentic Robot achieves 79.6% average success.
- **MANet / Habitat-Matterport3D** — Multi-agent navigation with collision avoidance and communication constraints.
- **SafeAware-VH** — Safety evaluation for embodied planning; MADRA achieves >90% unsafe-task rejection on 800 annotated VirtualHome instructions.

## Related (vault entities)

- [[Real-Time Policy Adaptation — Online Learning from Failures]] — Online learning closes the gap between planned subgoals and executed outcomes
- [[VLM Perception — Mobile Manipulation Fused Pipeline]] — Perception-grounded planning for embodied subgoal execution
- [[Embodied Spatial Reasoning — Multi-Room Navigation with Visual Memory]] — Spatial reasoning for subgoal-aware navigation
- [[Cross-Domain Policy Transfer — Sim-to-Real Manipulation]] — Transfer learning for subgoal policies across embodiments
- [[Vision-Language-Action Models]] — VLAs as primitive action executors within hierarchical planning pipelines
- [[Multi-Modal Grounding for Agents — Language-to-Action Mapping in VLMs]] — Grounding layer connecting subgoal language to physical actions

## Open Questions

- How do hierarchical planners handle subgoal conflicts when multiple agents decompose shared tasks independently?
- Can online learning of HTN methods generalize beyond the training domain without hand-coded method libraries?
- What is the optimal depth for subgoal trees — how many decomposition levels before overhead exceeds flat planning?
- How do we formally verify that a hierarchical decomposition preserves original goal semantics (i.e., leaf actions collectively achieve the root goal)?
- Can LLM-based decomposition reliably handle temporal constraints and hard deadlines in multi-agent scenarios?
- How do current approaches scale beyond 5–10 agents — do centralized decomposers become bottlenecks?
- What is the role of shared memory / joint situational awareness in hierarchical multi-agent planning?
- How effective are prompt optimization techniques across different LLM backbones — does prompt quality scale with model capability or saturate?
- Can decentralized subgoal tree search scale to large teams without communication overhead (cf. Dec-SGTS and MMDP approaches)?
- How well do SAP-driven frameworks transfer across different embodiments and task domains — is the Atomic Skill Library generalizable?

## Sources

- ReAcTree: Hierarchical LLM Agent Trees with Control Flow (arXiv:2511.02424, Choi et al., 2025) — Dynamic agent trees, episodic + working memory; 61% on WAH-NL
- Agentic Robot: Brain-Inspired SAP Framework (arXiv:2505.23450) — LRM planner → VLA executor → VLM verifier; 79.6% on LIBERO
- RoboBrain: Unified Brain Model (arXiv:2502.21257) — Hierarchical mapping from abstract to concrete manipulation; RoboBrain 2.0/2.5 with spatiotemporal reasoning
- GRHP: Graph-Fused Hierarchical Planning (Li et al., 2025) — Large-model semantics fused with small-model actions via cross-graph attention
- Hierarchical LLM-Based Multi-Agent Framework with Prompt Optimization (arXiv:2602.21670, Kawabe & Takano, 2026) — Two-layer LLM planner; 95% on MAT-THOR compound tasks
- STEP: Subgoal Tree Embodied Planner (arXiv:2506.21030) — Closed-loop subgoal tree construction; 34% on WAH-NL
- EmboTeam: Grounding LLM Reasoning into Reactive Behavior Trees (arXiv:2601.11063, 2026) — Multi-robot planning with BT execution
- Dec-SGTS: Decentralized Sub-Goal Tree Search (AAAI 2021) — Anytime hierarchical algorithm for decentralized multi-agent coordination
- SMART-LLM (Purdue, ICRA 2024) — LLM task analysis + RL path planning for multi-robot teams
- Online Learning of HTN Methods (arXiv:2511.12901, 2025) — Learning HTN methods via LLM interaction
- LLMTAMP: Neuro-Symbolic Task Planning (ICRA 2025) — Multi-level subgoal decomposition
- H²R: Hierarchical Hindsight Reflection (arXiv:2509.12810) — Multi-task subgoal mapping with hindsight
- "Large language models for multi-robot systems: a survey" (Springer, 2026) — Comprehensive LLM integration survey
- "Multi-agent Task Planning using Classical Planning Methods" (DYALab, ICRA 2025 workshop) — N-agent to N single-agent conversion
- LLaMAR: LM-based Long-Horizon Planner — Cognitive architecture for partial observability
- Emergent Mind: Hierarchical Goal Decomposition — Survey of decomposition methods
- CrewAI Hierarchical Process — Industry framework for hierarchical multi-agent management
- Lil'Log: LLM Powered Autonomous Agents (Lilian Weng, 2023) — Foundational subgoal decomposition framework
- MADRA: Multi-Agent Debate for Risk-Aware Embodied Planning (arXiv:2511.21460, AAMAS 2026) — Multi-agent debate framework with hierarchical cognitive collaboration; >90% unsafe-task rejection on SafeAware-VH
- LGC-MARL: LLM-Based Graph Collaboration MARL (Jia et al., Mar 2025) — Graph-structured MARL with LLM decomposition and critic-modeled feedback
- "Multi-agent Embodied AI: Advances and Future Directions" (Science China, 2026) — Comprehensive survey covering multi-agent control, planning, learning for multi-agent interaction, and embodied AI collaboration
- SAMA: Semantically Aligned Task Decomposition in Multi-Agent RL (OpenReview) — Language-grounded RL for subgoal-conditioned policies per agent; demonstrates considerable advantages in sample efficiency over state-of-the-art ASG methods on sparse-reward tasks
- H-WM: Robotic Task and Motion Planning Guided by Hierarchical World Model (arXiv:2602.11291, Chen et al., 2026) — Bilevel world model combining logical (symbolic) and visual (perceptual) state prediction for stable intermediate VLA guidance
- TeamWeaver: Hybrid LLM + Optimization Planning (2026) — LLM decomposes natural-language instructions; MIQP solver handles optimal task allocation; provides interpretable + provably optimal multi-robot planning

## Confidence

**0.88**: Strong evidence from 2024–2026 literature on LLM-based hierarchical decomposition (STEP, ReAcTree, Agentic Robot, RoboBrain, EmboTeam, SMART-LLM, HTN learning). The subgoal tree architecture, SAP-driven closed-loop design, and decentralized tree search are well-documented with experimental results across multiple benchmarks. Classical HTN and MRTA foundations are mature. Confidence is slightly below 0.90 because: (1) multi-agent coordination of hierarchical plans remains less studied than single-agent decomposition; (2) real-world deployment evidence is limited to lab-scale experiments; (3) the field evolves rapidly with new methods monthly, so coverage is current but not exhaustive; (4) several promising approaches (Agentic Robot, RoboBrain 2.5) lack independent replication yet.