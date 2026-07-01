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
confidence: 0.85
---

# Multi-Agent Task Decomposition — Hierarchical Planning with Subgoal Reasoning

## Summary

Multi-agent task decomposition is the process of breaking complex high-level goals into manageable sub-tasks and distributing them across multiple autonomous agents (robots or AI agents). Hierarchical planning organizes this decomposition into structured layers: high-level abstract plans are progressively refined into concrete, executable actions through subgoal reasoning. The field combines classical planning approaches (Hierarchical Task Networks, multi-robot task allocation) with modern LLM-based methods that use natural language to recursively decompose goals into subgoal trees, enabling long-horizon planning that flat planners cannot reliably solve. The core challenge is managing contextual and logical gaps: as task horizons extend, the distance between abstract instructions and concrete actions grows, causing reasoning to degrade. Hierarchical architectures mitigate this by constraining each planning level to reason only about its immediate parent node rather than the full task context.

## Key Facts

### Foundational Planning Frameworks

- **Hierarchical Task Networks (HTN)** — Classical planning paradigm where tasks are decomposed hierarchically through methods that specify when and how to decompose abstract tasks into subtasks. HTNs provide explicit domain knowledge encoding, enabling planners to stipulate global constraints on plans. Modern work (Online Learning of HTN Methods, arXiv:2511.12901) integrates LLM-based chatbots with HTN planners, learning HTN methods from interaction traces.

- **Multi-Robot Task Allocation (MRTA)** — The problem of concurrently assigning robots to goals and generating coordinated trajectories. Ranges from centralized optimization (solving allocation + planning jointly) to decentralized approaches (market-based, consensus algorithms, coalition formation). Recent work uses multimodal multi-objective evolutionary algorithms and deep reinforcement learning for dynamic allocation under communication constraints.

- **PDDL-based Planning** — Planning Domain Definition Language remains the standard interface between high-level reasoning and classical planners (FF, Fast Downward). LLM-based frameworks generate PDDL problems from natural language goals, then solve them with classical planners for guaranteed correctness.

### Hierarchical Architecture Patterns

- **Two-Level Hierarchical Decomposition** — An upper-layer LLM agent decomposes high-level tasks and assigns subgoals to lower-layer agents, which generate PDDL problems solved by classical planners. This "LLM brain + planner body" architecture separates reasoning from execution.

- **Subgoal Tree Construction** (STEP, arXiv:2506.21030) — Builds a hierarchical tree where:
  - Root = high-level natural language goal
  - Each level recursively decomposes parent subgoals into child subgoals
  - Leaf nodes = primitive executable actions
  - A **subgoal decomposition model** breaks complex tasks using foundation LLMs
  - A **leaf node termination model** determines when decomposition is sufficient by checking mappability (can this map to a primitive action?) and consistency (does this satisfy embodiment affordances and environment constraints?)
  - Achieves 34% success rate on VirtualHome WAH-NL benchmark and 25% on real robots, outperforming flat LLM planners

- **Behavior Trees** — Reactive, composable planning structures combining hierarchical decomposition with sensor-driven condition evaluation. EmboTeam (arXiv:2601.11063) grounds LLM reasoning into reactive behavior trees for multi-robot teams, combining high-level task decomposition with low-level reactive execution.

### LLM-Based Subgoal Decomposition

- **LLM-inferred subgoals match human expert decompositions** while preserving plan correctness — demonstrated in multi-agent planning methods that convert N-agent problems into N single-agent sub-problems via LLM-based subgoal decomposition.

- **Least-to-Most Decomposition** — Sequential subgoal generation where each subgoal is solved before generating the next, reducing error accumulation vs. generating all subgoals simultaneously.

- **Sub-Task Planner (SP)** — Explicitly structures planning through hierarchical, graph-based, or sequential sub-task generation. Addresses contextual and logical challenges in robotics, embodied AI, and automated agents.

- **Multi-Agent LLM Frameworks**:
  - **SMART-LLM** (Purdue, ICRA 2024) — Converts high-level task instructions into multi-robot task plans using LLMs for task analysis, then delegates to RL-based path planners for execution
  - **EmboTeam** (arXiv:2601.11063) — Hierarchical multi-robot task planning framework with LLM-driven decomposition co-optimized with reactive behavior tree execution
  - **BrainBody-LLM** — Two-module architecture: Brain-LLM for high-level reasoning and plan decomposition, Body module for physically feasible execution

### Neuro-Symbolic Planning Approaches

- **Neuro-Symbolic Language Models** (LLMTAMP, ICRA 2025) — Decompose long-sequential goals into multi-level subgoals using neuro-symbolic language models, achieving much faster planning than pure symbolic methods while maintaining high accuracy.

- **Integrated LLM-HTN Planning** — Online learning of HTN methods from LLM interaction traces, enabling chatbots to learn when and how to decompose tasks through experience rather than hand-coded method libraries.

### Core Challenges

- **Contextual Gap** — Long task sequences introduce redundant information into LLM input context. Extended sequences cause attention dilution, impairing reasoning. Hierarchical decomposition mitigates this by using parent nodes (not the full task) as context for each level.

- **Logical Gap** — Extended sequences correspond to increasingly abstract linguistic instructions. The inference from abstract directives to concrete actions degrades as task complexity grows. Subgoal trees reduce this by constraining each decomposition step to a manageable abstraction level.

- **Embodiment Constraints** — LLMs often overlook robot capability constraints (e.g., single-arm manipulation, payload limits, workspace boundaries). Leaf node termination models must check whether subgoals are executable given the specific embodiment's affordances.

- **Multi-Agent Coordination** — Decomposition must account for inter-agent dependencies, shared resources, collision avoidance, and communication constraints. Decentralized decomposition scales better but risks subgoal conflicts.

### Benchmark & Evaluation Landscape

- **VirtualHome WAH-NL** — Multi-room, long-horizon embodied task benchmark; STEP achieves 34% success rate
- **ALFRED** — Visual reasoning for long-horizon embodied tasks in simulated environments
- **LIBERO** — Long-horizon robotic benchmark for bimanual manipulation tasks
- **Multi-Agent Navigation (MANet, Habitat-Matterport3D)** — Multi-agent coordination benchmark with collision avoidance and communication constraints

## Related (vault entities)

- [[Real-Time Policy Adaptation — Online Learning from Failures]] — Complementary approach: online learning closes the gap between planned subgoals and executed outcomes
- [[VLM Perception — Mobile Manipulation Fused Pipeline]] — Perception-grounded planning for embodied subgoal execution
- [[Embodied Spatial Reasoning — Multi-Room Navigation with Visual Memory]] — Spatial reasoning component needed for subgoal-aware navigation
- [[Cross-Domain Policy Transfer — Sim-to-Real Manipulation]] — Transfer learning for subgoal policies across embodiments
- [[Vision-Language-Action Models]] — VLAs as primitive action executors within hierarchical planning pipelines
- [[Multi-Modal Grounding for Agents — Language-to-Action Mapping in VLMs]] — Grounding layer connecting subgoal language to physical actions

## Open Questions

- How do hierarchical planners handle subgoal conflicts when multiple agents decompose shared tasks independently?
- Can online learning of HTN methods generalize beyond the training domain without hand-coded method libraries?
- What is the optimal depth for subgoal trees — how many decomposition levels before the overhead exceeds flat planning?
- How do we formally verify that a hierarchical decomposition preserves the original goal semantics (i.e., leaf actions collectively achieve root goal)?
- Can LLM-based decomposition reliably handle temporal constraints and hard deadlines in multi-agent scenarios?
- How do current approaches scale beyond 5–10 agents — do centralized decomposers become bottlenecks?
- What is the role of shared memory / joint situational awareness in hierarchical multi-agent planning?

## Sources

- STEP: Subgoal Tree Embodied Planner (arXiv:2506.21030, Zhou et al.) — Closed-loop subgoal tree construction with decomposition and termination models
- EmboTeam: Grounding LLM Reasoning into Reactive Behavior Trees (arXiv:2601.11063, 2026) — Hierarchical multi-robot task planning with behavior tree execution
- SMART-LLM (Purdue, ICRA 2024) — Multi-agent robot task planning using LLMs for task analysis and RL for path planning
- Online Learning of HTN Methods (arXiv:2511.12901, 2025) — Learning hierarchical task network methods via LLM-based interaction
- LLMTAMP: Fast and Accurate Task Planning using Neuro-Symbolic Language Models (ICRA 2025) — Neuro-symbolic multi-level subgoal decomposition
- BrainBody-LLM — Hierarchical LLM-based planning with Brain-LLM for high-level reasoning and Body module for execution
- "Large language models for multi-robot systems: a survey" (Springer, 2026) — Comprehensive survey of LLM integration in multi-robot systems
- "Multi-agent Task Planning using Classical Planning Methods" (DYALab, ICRA 2025 workshop) — N-agent to N single-agent sub-problem conversion via LLM subgoal decomposition
- Emergent Mind: Hierarchical Goal Decomposition — Survey of hierarchical decomposition methods
- CrewAI Hierarchical Process — Industry framework for hierarchical multi-agent task management
- Lil'Log: LLM Powered Autonomous Agents (Lilian Weng, 2023) — Foundational framework for subgoal decomposition in autonomous agents

## Confidence

**0.85**: Strong evidence from 2024–2026 literature on LLM-based hierarchical decomposition (STEP, EmboTeam, SMART-LLM, HTN learning). The subgoal tree architecture and closed-loop decomposition models are well-documented. Classical HTN and MRTA foundations are mature. Confidence is reduced from 0.90 because: (1) multi-agent coordination of hierarchical plans remains less studied than single-agent decomposition; (2) real-world deployment evidence is limited to a few lab-scale experiments; (3) the field is evolving rapidly with new methods monthly, so coverage is current but not exhaustive.