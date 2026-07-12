---
segment: architecture
tags: [architecture]
type: notes

---

# Self-Directed Improvement Planner (#179)

> Turns evaluation outputs and intelligence findings into concrete,prioritized improvement tasks — closing the gap between "we found a problem" and "we're fixing it."

## System Context

The planner sits at the center of the self-learning stack:

```
#178 Intelligence Pipeline ──┐
#177 Evaluation Engine ───────┤
Corrections log ──────────────┼──► IMPROVEMENT PLANNER ──► Auto-execute (low risk)
Autonomy QA signals ──────────┤                        ──► Backlog items (high risk)
Strategy Memory ──────────────┘                        ──► Change Registry (#180)
```

**Dependencies:**
- **#177 (Evaluation Engine)** — provides session scorecards,self-critique signals,outcome metrics
- **#178 (Intelligence Pipeline)** — provides external development findings and relevance scores
- **#180 (Verification System)** — consumes change tags,measures outcomes,feeds back results

**Key insight:** Several input signals exist TODAY (corrections,daily notes,autonomy task logs). We don't need to wait for #177/#178 to start building.

## Execution Model: Idler + Autonomy Scheduler

The planner does NOT run as a single weekly batch job. It leverages the existing **idler agent** and **autonomy scheduler** to run continuously throughout the day,GPU-gated like all other background work.

### Why Not Weekly?

- Corrections happen during sessions — waiting a week to act on "stop over-delegating" is absurd
- The idler is already GPU-gated and pulling tasks whenever the GPU is free
- Signal collection,synthesis,and execution are naturally separable autonomy tasks
- Multiple small runs beat one large batch: faster feedback,less stale signal accumulation

### Task Decomposition

The planner breaks into **three autonomy tasks** that run independently at different frequencies:

| Task | Frequency | Agent | Description |
|------|-----------|-------|-------------|
| **Signal Collector** | 4×/day (~6h) | memory | Parse corrections,daily notes,autonomy logs → raw signals |
| **Synthesis & Planning** | 2×/day (~12h) | memory | Aggregate signals,cross-ref strategy memory,generate candidates |
| **Auto-Executor** | 1×/day | idler | Execute low-risk candidates,tag changes for #180,create backlog items for high-risk |

All three are GPU-gated via the standard autonomy dispatch:
- GPU < 30% → run
- GPU ≥ 30% → backoff,retry at next poll
- No interference with foreground user sessions

### Reactive Triggers

Beyond scheduled runs,certain signals can trigger an **immediate mini-synthesis** via the idler:

- **3+ corrections on the same topic within 24h** → urgent signal,synthesize and propose immediately
- **Autonomy task failure** → immediate QA review of failed task
- **Strategy memory rollback from #180** → re-evaluate related candidates

These use the idler's existing task dispatch — just higher priority.

---

## Phase 1: Foundation & Governance Model

**Goal:** Establish the planner's decision framework,logging infrastructure,and governance rules before any automated changes happen.

### 1.1 Decision Log Structure

```
memory/improvement-planner/
├── decisions/
│   └── YYYY-MM-DD.md          # Daily decision log
├── governance.yml              # Risk classification rules
├── candidates/
│   └── YYYY-MM-DD.jsonl       # Raw improvement candidates before filtering
└── strategy/
    └── patterns.jsonl          # "What worked/failed" tactical memory
```

**Decision log entry format:**
```json
{
  "id": "imp-20260323-001",
  "timestamp": "2026-03-23T05:00:00Z",
  "source": "corrections|eval|intel|autonomy-qa|pattern",
  "signal": "User corrected delegation behavior 3x this week",
  "analysis": "AGENTS.md delegation threshold too aggressive for quick lookups",
  "proposed_action": "Adjust AGENTS.md 'Handle Directly' section to include...",
  "risk_tier": "high",
  "disposition": "proposed|auto-executed|deferred|rejected",
  "backlog_id": null,
  "change_id": null,
  "rationale": "Changes agent routing rules → high risk tier"
}
```

### 1.2 Governance Rules

Two-tier classification with explicit rule set:

**Auto-Execute (Low Risk):**
| Category | Examples | Constraint |
|----------|----------|------------|
| SOUL.md personality tuning | Tone adjustments,vibe refinements | No structural changes |
| Skill creation/updates | New skills,skill content edits | No tool permission changes |
| Knowledge vault updates | New docs,fact corrections | Write-allowed segments only |
| Monitoring task creation | New autonomy scan/check tasks | Read-only tasks only |
| Morning briefing content | Section additions,format tweaks | No delivery changes |
| Strategy memory updates | Pattern recording,outcome logging | Append-only |

**Propose-and-Wait (High Risk):**
| Category | Examples | Reason |
|----------|----------|--------|
| AGENTS.md changes | Routing rules,delegation thresholds,boundaries | Changes capability envelope |
| openc