---
type: reference
tags: [autoagent, self-improving-agents, agent-harness, meta-agent, research]
source: https://www.youtube.com/watch?v=RoaPvj9Ovug
date: 2026-04-04
summary: "AutoAgent applies the Auto Research pattern to agent harness optimization - using meta agents to autonomously improve prompts, tools, and orchestration rather than ML training code."
---

# AutoAgent & Self-Improving Agents

---

## Attribution Note

⚠️ **Video Attribution Discrepancy:** The YouTube video attributes the project to "Kevin Guo" on X, but the actual public AutoAgent repository is by **Jiabin Tang, Tianyu Fan, Chao Huang** (HKUDS organization). The Kevin Guo attribution could not be verified and may be incorrect or refer to a different project.

- **Official Repository:** [HKUDS/AutoAgent](https://github.com/HKUDS/AutoAgent) (~8,769 stars)
- **Paper:** [arxiv.org/abs/2502.05957](https://arxiv.org/abs/2502.05957)
- **License:** MIT

---

## Overview

**AutoAgent** (HKUDS) is a framework that enables users to create and deploy LLM agents through natural language alone. The video discusses applying the "Auto Research" pattern to agent harness optimization - instead of optimizing ML training code (like Karpathy's Auto Research), it optimizes the agent harness itself - prompts, tools, and orchestration logic.

**Core Insight:** Every domain needs a different harness, and harness engineering can be automated through meta-agent experimentation.

---

## Core Concept: Harness Optimization

### What is a "Harness"?

The "harness" refers to all code surrounding an LLM that determines:
- System prompts and identity (e.g., SOUL.md, IDENTITY.md)
- Tool configurations and routing
- Context management and memory structure
- Orchestration logic between agents

**Key Finding:** The harness around a fixed LLM can produce a **6x performance gap** on the same benchmark [[Meta-Harness Paper](https://arxiv.org/abs/2603.28052)].

### The Auto Research Pattern

Originally from Andrej Karpathy's [Auto Research](https://github.com/karpathy/autoresearch) project:
1. Agent modifies `train.py`
2. Trains for 5-minute fixed budget
3. Checks validation metrics (val_bpb)
4. Keep/discard decision
5. Repeat overnight

**AutoAgent adapts this pattern:**
- Edit harness → Run benchmark → Check results → Repeat
- Same loop, different target (agent harness instead of training code)

---

## AutoAgent Architecture

Based on the video description and related work:

### Two-Agent System
1. **Meta Agent**
   - Spins up parallel sandboxes
   - Runs task agents on evaluation tasks
   - Reads results and reasoning traces
   - Decides what to keep vs. revert

2. **Task Agent**
   - Starts with minimal tooling (bash only)
   - Reads `program.md` for research direction
   - Experiments with `agent.py` (its own code)
   - Connects to domain-specific benchmarks via adapter

### The Loop
```
Edit harness (agent.py)
    ↓
Run benchmark (spreadsheet_bench / terminal_bench)
    ↓
Check results + reasoning traces
    ↓
Keep or revert
    ↓
Repeat
```

### Key Insight
> "You're not writing Python anymore. You're just writing the markdown file. The human programs the agent, and the agent programs the code."

Domain experts define good instructions in `program.md`, and the meta agent discovers domain-specific tooling, verification loops, and orchestration logic autonomously.

---

## Benchmarks

### From Video
- **spreadsheet_bench** - Spreadsheet task benchmark
- **terminal_bench** - Terminal task benchmark

### Related Work: Terminal-Bench 2.0
From Meta-Harness research [[Paper](https://arxiv.org/abs/2603.28052)]:
- 89 tasks across difficulty levels (Easy: 4, Medium: 55, Hard: 30)
- Results: 76.4% on Terminal-Bench 2.0 (Claude Opus 4.6)
- Earlier iterations: ~46.5% on 19-task hard subset

---

## Related Work in Self-Improving Agents

### 1. Meta-Harness (Stanford/MIT/KRAFTON)
**Authors:** Yoonho Lee, Roshen Nair, Qizheng Zhang, Kangwook Lee, Omar Khattab, Chelsea Finn
**Paper:** https://arxiv.org/abs/2603.28052
**Project Page:** https://yoonholee.com/meta-harness/
**Artifact:** https://github.com/stanford-iris-lab/meta-harness-tbench2-artifact

**Architecture:**
- Meta agent (proposer) reads execution traces from prior candidates
- Proposes harness edits (prompts, context management, tool configs)
- Evaluates on held-out benchmark tasks
- Full filesystem access for analysis

**Results:** 76.4% on Terminal-Bench 2.0

### 2. SiriuS (Self-improving Multi-agent Systems)
**Authors:** Zhao, Wanjia; Yuksekgonul, Mert; Wu, Shirley; Zou, James
**Paper:** https://arxiv.org/pdf/2502.04780 (NeurIPS 2025)
**Repository:** https://github.com/zou-group/sirius

**Architecture:**
- Multi-agent system with Actor, Judgment, Critic roles
- Experience library of successful trajectories
- Bootstraps from failed trajectories

### 3. recursive-improve (Kayba AI)
**Repository:** https://github.com/kayba-ai/recursive-improve

**Approach:**
- Capture execution traces via `ri.patch()`
- Analyze failure patterns
- Apply targeted fixes
- Keep-or-revert evaluation loop

### 4. Harness Evolver
**Repository:** https://github.com/raphaelchristi/harness-evolver

**Features:**
- Meta-Harness optimization as Claude Code plugin
- Multi-agent evolution with LangSmith backend
- Self-organizing proposers
- Rubric-based evaluation
- Pareto front selection

### 5. Other Notable Projects
- **ADAS** (ICLR 2025) - Automated Design of Agentic Systems [[GitHub](https://github.com/shengranhu/ADAS)]
- **SICA** (ICLR 2025 Workshop) - Self-improving coding agent [[GitHub](https://github.com/MaximeRobeyns/self_improving_coding_agent)]
- **HGM (Huxley-Godel Machine)** - Self-improvement for coding agents [[GitHub](https://github.com/metauto-ai/hgm)]
- **GEPA** (ICLR 2026 Oral) - Genetic-Pareto for prompt evolution [[GitHub](https://github.com/gepa-ai/gepa)]
- **AI-Scientist** - Full automated scientific discovery [[GitHub](https://github.com/SakanaAI/AI-Scientist)]
- **AutoResearch Pattern (Karpathy-style):** [ChrisGoesGolfing](https://github.com/chrispyspearbit/ChrisGoesGolfing) - Autonomous code iteration with feedback loops

---

## Key Concepts

### Harness Engineering
The practice of designing the code that surrounds and orchestrates LLMs. This includes:
- Prompt design and system identity
- Tool selection and configuration
- Memory and context management
- Multi-agent coordination patterns

**Emerging Insight:** Harness engineering may become an automated process where agents engineer their own harnesses, similar to how AI is now writing code.

### Domain Expertise
> "Domain experts are going to be really valuable with these types of projects, because they're going to be able to define what are good instructions for the outcomes that you want from these different meta agents."

The value shifts from writing the harness to defining success criteria in natural language.

### Abstraction Layer
This represents "just another level of abstraction, similar to code." Just as code syntax is increasingly written by AI models, harnesses may also be engineered by agents:
- Define what success looks like
- Point the meta agent at the benchmark
- Return in 24 hours to see results

---

## Open Questions

1. **"AutoAgent by Kevin Guo"** - Could not locate a public GitHub repository or paper under this exact name. May be:
   - A private/internal project
   - A very recent release (post-research cutoff)
   - Using different naming conventions
   - The "Kevin Guo" attribution may be from a different context

2. **"spreadsheet_bench"** - This specific benchmark was not found in public repositories

3. **Video specifics** - The description matches Meta-Harness concepts closely but with different naming conventions

4. **Scalability** - How does this approach scale to complex, multi-step workflows?

5. **Safety** - What guardrails are needed when agents can modify their own orchestration logic?

---

## Research Path (Source Tree)

```
Primary Source:
└── https://www.youtube.com/watch?v=RoaPvj9Ovug
    └── "Self Improving Agents in 5 Minutes"
        ├── Kevin Guo (X post reference)
        ├── AutoAgent concept
        ├── Auto Research inspiration
        └── Benchmarks: spreadsheet_bench, terminal_bench

Thread 1: Auto Research (Original Inspiration)
└── https://github.com/karpathy/autoresearch
    └── Karpathy's autonomous ML research pattern

Thread 2: Meta-Harness (Closest Match)
├── https://arxiv.org/abs/2603.28052 (Paper)
├── https://yoonholee.com/meta-harness/ (Project Page)
└── https://github.com/stanford-iris-lab/meta-harness-tbench2-artifact (Artifact)

Thread 3: Self-Improving Agent Projects
├── https://github.com/zou-group/sirius (SiriuS)
├── https://github.com/kayba-ai/recursive-improve
├── https://github.com/raphaelchristi/harness-evolver
├── https://github.com/shengranhu/ADAS (ADAS)
├── https://github.com/MaximeRobeyns/self_improving_coding_agent (SICA)
├── https://github.com/metauto-ai/hgm (HGM)
├── https://github.com/gepa-ai/gepa (GEPA)
└── https://github.com/SakanaAI/AI-Scientist (AI-Scientist)

Thread 4: Benchmark Research
└── Terminal-Bench 2.0 (89 tasks, 3 difficulty levels)
```

---

## Implications for Agent Development

### For Lloyd (This Project)
1. **Harness optimization is a viable research direction** - The pattern of "edit harness → run benchmark → check results → repeat" is proven
2. **Meta-agent architecture** - Consider implementing a proposer/evaluator pattern for tool discovery
3. **Benchmark development** - Need domain-specific benchmarks (spreadsheet, terminal, etc.)
4. **Program.md concept** - Natural language instructions for meta agents could be valuable

### Open Research Questions for Lloyd
- Can we implement a lightweight harness evolver for tool discovery?
- What benchmarks would be most valuable for our use cases?
- How do we balance automation with safety when agents modify their own orchestration?

---

## Sources Consulted

1. https://www.youtube.com/watch?v=RoaPvj9Ovug - "Self Improving Agents in 5 Minutes" by Developers Digest (Primary source, transcript fetched)
2. https://github.com/HKUDS/AutoAgent - AutoAgent (HKUDS) - Official repository (~8,769 stars)
3. https://arxiv.org/abs/2502.05957 - AutoAgent paper (HKUDS)
4. https://arxiv.org/abs/2603.28052 - Meta-Harness paper (Lee et al., March 2026)
5. https://yoonholee.com/meta-harness/ - Meta-Harness project page
6. https://github.com/stanford-iris-lab/meta-harness-tbench2-artifact - Meta-Harness artifact
7. https://github.com/karpathy/autoresearch - Karpathy's Auto Research
8. https://github.com/chrispyspearbit/ChrisGoesGolfing - AutoResearch pattern implementation
9. https://github.com/zou-group/sirius - SiriuS (NeurIPS 2025)
10. https://arxiv.org/pdf/2502.04780 - SiriuS paper
11. https://github.com/kayba-ai/recursive-improve - Recursive improve framework
12. https://github.com/raphaelchristi/harness-evolver - Harness Evolver
13. https://github.com/shengranhu/ADAS - ADAS (ICLR 2025)
14. https://github.com/MaximeRobeyns/self_improving_coding_agent - SICA
15. https://github.com/metauto-ai/hgm - Huxley-Godel Machine
16. https://github.com/gepa-ai/gepa - GEPA (ICLR 2026 Oral)
17. https://github.com/SakanaAI/AI-Scientist - AI-Scientist

---

## Research Notes

**Attribution Correction:** The video mentions "Kevin Guo" as the creator of AutoAgent, but the public repository is by Jiabin Tang, Tianyu Fan, Chao Huang (HKUDS). This attribution discrepancy could not be resolved.

**Video Content:** The transcript describes a meta-agent architecture that optimizes agent harnesses (prompts, tools, orchestration) rather than ML training code, applying the Auto Research pattern to a different domain.

---

*Research conducted: 2026-04-04*
*Note updated: 2026-04-04*
*Note written by: Lloyd Research Agent*
