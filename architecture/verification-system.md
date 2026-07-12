---
segment: architecture
type: reference
tags: [architecture]
date: 2026-03-22
status: planned
related:
  - projects/lloyd/architecture/autonomy-system.md
  - projects/lloyd/architecture/index.md
  - projects/lloyd/architecture/nightly-reflection.md

---

# Closed-Loop Verification System

**Backlog:** #180  
**Created:** 2026-03-22  
**Status:** Planned (deep dive complete,ready for implementation)

## Problem Statement

The system makes changes nightly — prompt edits,config tweaks,skill updates,memory consolidation — but never checks if they helped. Nightly reflection might undo yesterday's improvement or keep last week's regression. Without measurement,"improvement" is just hope.

## Design Principles

1. **Every change is a hypothesis.** Tag it,measure it,decide.
2. **Build measurement before automation.** Manual review with good data beats automated review with bad data.
3. **Rollback is cheap,regressions are expensive.** Default to revert on ambiguity.
4. **Degrade gracefully without #177.** Use proxy metrics (corrections,session length,rework) until the evaluation engine exists.
5. **Strategy memory is the real output.** The registry is the mechanism; the knowledge of "what works" is the value.

## Architecture Overview

```
Change Source                    Change Registry                  Verification
─────────────                    ───────────────                  ────────────
Nightly reflection ──┐                                           
Improvement planner ─┤──→ Tag & Register ──→ changes.jsonl ──→ Review Window
Manual edits ────────┘    (hypothesis,(structured log)   Expires
Idler agent ─────────┘    baseline,│
                           criteria)                                ▼
                                                    Idler / Autonomy Scheduler
                                                    (checks 4-6x/day via GPU-gated dispatch)
                                                              Measure Outcome
                                                                    │
                                                    ┌───────────────┼───────────────┐
                                                    ▼               ▼               ▼
                                                Graduated       Extended        Rolled Back
                                                    │               │               │
                                                    ▼               ▼               ▼
                                              Strategy Memory   Longer Window   Git Revert +
                                              "X works for Y"   (need data)    Strategy Memory
                                                                               "X failed for Y"
```

## Execution Model: Idler + Autonomy Scheduler

Verification runs are **not nightly batch jobs**. They leverage the existing idler agent and autonomy scheduler for continuous,GPU-gated execution throughout the day.

### Why Idler,Not Cron?
- **Review windows expire throughout the day** — a change made Monday evening shouldn't wait until Saturday night's cron to get reviewed
- **GPU gating is built in** — verification review is lightweight (read registry,count corrections,compare) but the LLM reasoning for decision-making benefits from the 122B model
- **Idler already has the dispatch loop** — no new infrastructure needed,just a new task type
- **Faster feedback** — checking 4-6x/day means decisions happen within hours of a review window expiring,not the next overnight run

### Task Scheduling
| Task | Runs Per Day | Agent | Tag | Priority |
|------|-------------|-------|-----|----------|
| Verification Review | 4-6x (autonomy scheduler) | `memory` | `periodic` | medium |
| Metric Collection | 4-6x (piggybacks on review) | `memory` | `periodic` | low |
| Strategy Memory Sync | 1x (nightly,after all reviews) | `memory` | `periodic` | low |
| Archive Cleanup | 1x (weekly) | `memory` | `audit` | low |

### Autonomy Task Definition
```json
{
  "name": "Verification Review",
  "agent": "memory",
  "tag": "periodic",
  "runs_per_day": 6,
  "skill": "verification-management",
  "description": "Check for expired review windows,collect proxy metrics,make graduate/rollback/extend decisions"
}
```

### Execution Flow
```
Idler polls backlog → Verification Review task due
  → GPU gate check (< 30% utilization)
  → Read changes.jsonl for active entries with review_due <= now
  → For each expired entry:
     1. Collect current metric (correction count,rework rate,etc.)
     2. Compare against baseline
     3. Decision: graduate / roll back / extend / expire-neutral
     4. Update registry entry
     5. If rollback: execute git revert immediately
     6. Write strategy memory entry
  → Mark task complete,return to idler pool
```

### Change Tagging Also Runs Continuously
Changes don't only happen at night. The orchestrator,manual edits,and im