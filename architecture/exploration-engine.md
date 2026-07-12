---
segment: architecture
tags: [architecture]
type: notes

---

# Exploration Engine (#183) — Architecture

> The piece that makes Lloyd actively self-improving rather than just self-repairing. Generates hypotheses,tests them synthetically,and graduates winners into production through the verification pipeline.

## Dependencies

| Dependency | What It Provides | Required Before |
|-----------|-----------------|----------------|
| #181 (Composite Quality Score) | Metric to optimize against | Phase 2 |
| #182 (Rapid Eval Harness) | Fast synthetic evaluation | Phase 2 |
| #180 (Verification System) | Real-world confirmation after eval pass | Phase 3 |
| #178 (Intelligence Pipeline) | External hypothesis sources | Phase 4 |

**Build order:** #181 → #182 → This (#183). Phases 1 can start before dependencies are ready.

---

## Phase 1 — Hypothesis Generation Framework

**Goal:** Build the machinery that identifies improvement opportunities and formalizes them as testable hypotheses.

### Hypothesis Schema

```jsonl
{
  "id": "hyp-2026-03-22-001",
  "timestamp": "2026-03-22T04:00:00Z",
  "source": "pattern-mining",
  "surface": "agents/lloyd/TOOLS.md",
  "category": "response-format",
  "hypothesis": "Adding explicit code-block formatting guidance reduces orchestrator rework rate",
  "rationale": "3 corrections in past 2 weeks about code formatting in responses",
  "change_spec": {
    "file": "agents/lloyd/TOOLS.md",
    "type": "append",
    "content": "## Code Formatting\n- Always use fenced code blocks with language hints..."
  },
  "expected_outcome": "Fewer formatting corrections,higher first-attempt success",
  "risk_level": "low",
  "status": "proposed",
  "eval_score_before": null,
  "eval_score_after": null,
  "outcome": null,
  "branch": null
}
```

### Hypothesis Sources (Internal)

Four generators,each implemented as a function the exploration agent calls:

1. **Correction Pattern Miner**
   - Input: `memory/corrections.md` + recent daily notes
   - Logic: Cluster negative corrections by topic/surface. If ≥2 corrections in same area within 14 days,generate hypothesis to address root cause.
   - Example: "3 corrections about verbose responses → hypothesis: tighten SOUL.md conciseness guidance"

2. **Performance Plateau Detector**
   - Input: Quality score history from #181 (once available)
   - Logic: If composite score flat (±2%) for 14+ days,identify lowest-scoring dimension and propose intervention.
   - Example: "Delegation accuracy stuck at 78% for 3 weeks → hypothesis: add task-type classification step before orchestrator dispatch"

3. **Session Friction Analyzer**
   - Input: Session logs (daily notes,tool call patterns)
   - Logic: Identify repeated multi-step sequences that could be simplified,or recurring user clarification requests.
   - Example: "User asks 'what model?' 4 times in a week → hypothesis: always mention model in session greeting"

4. **Strategy Gap Scanner**
   - Input: `memory/exploration/strategy.jsonl` (tactical knowledge from past experiments)
   - Logic: Find surfaces with no experiments in 30+ days,or task types with no strategy entries.
   - Example: "No experiments on delegation thresholds since initial setup → hypothesis: test lower delegation threshold for research tasks"

### Improvement Surfaces Registry

Each surface has metadata controlling experiment behavior:

```jsonl
{
  "surface_id": "soul-md",
  "path": "agents/lloyd/SOUL.md",
  "risk_level": "medium",
  "requires_approval": false,
  "max_concurrent_experiments": 1,
  "min_eval_score_delta": 0.05,
  "rollback_strategy": "git-revert",
  "last_experiment": null
}
```

**Surfaces catalog (initial):**

| Surface | Path | Risk | Approval Required |
|---------|------|------|-------------------|
| SOUL.md | `agents/lloyd/SOUL.md` | medium | no |
| AGENTS.md | `agents/lloyd/AGENTS.md` | high | yes |
| TOOLS.md | `agents/lloyd/TOOLS.md` | low | no |
| Orchestrator config | `openclaw.json` agents section | high | yes |
| Skill procedures | `skills/*/SKILL.md` | medium | no |
| Subagent prompts | agent task templates | medium | no |
| Delegation rules | AGENTS.md routing section | high | yes |

### Phase 1 Deliverables

- [] `memory/exploration/hypotheses.jsonl` — hypothesis log
- [] `memory/exploration/surfaces.jsonl` — surfaces registry
- [] Hypothesis generation functions (4 internal generators)
- [] Autonomy task: `exploration-generate` — runs nightly,generates 1-3 hypotheses
- [] Hypothesis dedup: skip if same surface + similar change_spec exists in last 30 days

---

## Phase 2 — Experiment Execution Pipeline

**Goal:** Take a hypothesis,apply it on a git branch,run it through the eval harness,and record results.

**Hard dependency:** #182 (Rapid Eval Harness) must be operational.

### Experiment Lifecycle

```
Hypothesis (proposed)
    │
    ▼
Create experiment branch: `exp/<hypothesis-id>`
    │
    ▼
Apply change_spec to target file(s) on branch
    │
    ▼
Run Rapid Eval Harness (#182) ag