---
type: reference
tags: [ai, agents, self-improving, autonomous-systems, meta-learning]
source: https://www.youtube.com/watch?v=RoaPvj9Ovug
date: 2026-04-04
summary: "AutoAgent extends Karpathy's AutoResearch concept from ML training optimization to general agent harness engineering, using meta-agents to autonomously improve task agents across any domain."
---

# Self-Improving Agents: AutoAgent & AutoResearch

## Summary

Developers Digest's video "Self Improving Agents in 5 Minutes" introduces **AutoAgent**, a project by Kevin Goo that extends Andrej Karpathy's AutoResearch concept. While AutoResearch optimizes ML training code overnight, AutoAgent optimizes the **agent harness itself** — prompts, tools, and orchestration logic — enabling autonomous domain-specific agent engineering.

The core insight: instead of manually engineering agent systems, define what success looks like in `program.md` and let a meta-agent discover optimal harness configurations through iterative experimentation.

---

## Key Concepts

### The Self-Improving Loop

Both AutoResearch and AutoAgent follow the same iterative pattern:

1. **Edit** the target file (train.py for AutoResearch, agent.py for AutoAgent)
2. **Run** on benchmark tasks
3. **Evaluate** results against metrics
4. **Keep or discard** based on performance
5. **Repeat** autonomously

This loop runs overnight, allowing hundreds of experiments to accumulate domain-specific optimizations that no human would manually engineer.

### AutoResearch (Karpathy) — ML Training Optimization

**Repository**: https://github.com/karpathy/autoresearch

**Architecture**:
- `prepare.py` — Fixed constants, data prep, tokenizer (untouched)
- `train.py` — Model architecture, optimizer, training loop (agent edits this)
- `program.md` — Human-written instructions for the agent

**Key Design**:
- 5-minute training budget per experiment
- Single GPU (H100)
- Val_bpb (validation bits per byte) as the metric
- The agent "programs in natural language" — humans write `program.md`, agent writes `train.py`

### AutoAgent — Agent Harness Optimization

**Repository**: https://github.com/kevinrgu/autoagent (2,400 ⭐, created 2026-04-02)

**Architecture**:
```
agent.py                       -- single-file harness under test
  editable harness section     -- prompt, registries, tools, routing
  fixed adapter section        -- Harbor integration + trajectory serialization
program.md                     -- meta-agent instructions + directive
Dockerfile.base                -- base image
.agent/                        -- optional agent workspace artifacts
tasks/                         -- benchmark tasks
jobs/                          -- Harbor job outputs
results.tsv                    -- experiment log
```

**Key Components**:

1. **Meta Agent**: Autonomous researcher that iterates on the harness. Reads `program.md` for instructions and modifies `agent.py` to improve performance.

2. **Task Agent**: The actual agent harness being improved, defined in `agent.py`:
   - `SYSTEM_PROMPT`, `MODEL`, `MAX_TURNS` — agent configuration
   - `create_tools(environment)` — tool definitions
   - `create_agent(environment)` — agent construction with handoffs/sub-agents
   - `run_task(environment, instruction)` — orchestration logic

3. **Fixed Adapter Boundary**: Section in `agent.py` marked with comments that the meta-agent cannot modify. Contains Harbor integration and trajectory serialization.

### Benchmarks

AutoAgent uses the **Harbor benchmark framework**:

- **spreadsheet_bench**: Spreadsheet task evaluation
- **terminal_bench**: Terminal/command-line task evaluation

Task structure:
```
tasks/my-task/
  task.toml           -- config (timeouts, metadata)
  instruction.md      -- prompt sent to the agent
  tests/
    test.sh           -- entry point, writes /logs/reward.txt
    test.py           -- verification
  environment/
    Dockerfile        -- task container
  files/              -- reference files
```

**Metrics**:
- Primary: `passed` (number of tasks passed)
- Secondary: `avg_score` (average score across tasks)
- Cost tracking: `cost_usd`

---

## Design Philosophy

### 1. Program the Meta-Agent, Not the Harness
Edit `program.md` with natural language instructions. Let the agent edit the code. This is "programming in natural language" — the human programs the agent, and the agent programs the code.

### 2. Score-Driven Iteration
Every experiment produces a numeric score. Keep if better, discard if not. This creates a simple but powerful evolutionary pressure.

### 3. Simplicity Criterion
Equal performance with simpler code is a win. The system naturally discovers elegant solutions.

### 4. NEVER STOP
The agent runs autonomously until explicitly interrupted. This is what research looks like from here on out.

### 5. Domain-Specific Tooling
Each domain needs different harness engineering. AutoAgent allows optimization of:
- Domain-specific tooling
- Verification loops
- Orchestration logic
- Model selection (cheaper models for specific tasks vs. monolithic harness)

---

## Implications

### For Agent Engineering

1. **Harness engineering requires domain + model expertise**: AutoAgent discovers patterns that require understanding both the domain and how models behave.

2. **Multiple harnesses, not one monolith**: Organizations may benefit from optimized harnesses at different parts of the stack, potentially using different (cheaper) models for different tasks.

3. **Domain experts become valuable**: They define what good instructions look like and what outcomes to optimize for.

### For AI Development

1. **Another level of abstraction**: Just as code moved from manual syntax to AI-written code, harness engineering may move from manual to agent-discovered configurations.

2. **Define success, wait 24 hours**: Point the meta-agent at a problem, define what success looks like, and let it discover optimal configurations.

3. **Autonomous discovery**: Overnight, agents can discover domain-specific tooling, verification loops, and orchestration logic that nobody programmed.

---

## Papers & Links

### Primary Sources
- **AutoAgent GitHub**: https://github.com/kevinrgu/autoagent
- **AutoResearch GitHub**: https://github.com/karpathy/autoresearch
- **Developers Digest Video**: https://www.youtube.com/watch?v=RoaPvj9Ovug
- **Kevin Goo X Announcement**: https://x.com/kevingu/status/2039843234760073341
- **Karpathy AutoResearch Tweet**: https://x.com/karpathy/status/2029701092347630069

### Related Forks (AutoResearch)
- MacOS (MLX): https://github.com/miolini/autoresearch-macos
- MacOS (Apple Silicon): https://github.com/trevin-creator/autoresearch-mlx
- Windows (RTX): https://github.com/jsegov/autoresearch-win-rtx
- AMD: https://github.com/andyluo7/autoresearch

### Company
- **Third Layer** (Kevin Goo): https://www.thirdlayer.inc

---

## Sources

1. Developers Digest, "Self Improving Agents in 5 Minutes", YouTube, 2026. https://www.youtube.com/watch?v=RoaPvj9Ovug
2. Kevin Goo, AutoAgent GitHub Repository. https://github.com/kevinrgu/autoagent
3. Andrej Karpathy, AutoResearch GitHub Repository. https://github.com/karpathy/autoresearch
4. Harbor Benchmark Framework (referenced in AutoAgent docs).
