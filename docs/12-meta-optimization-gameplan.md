# Meta-Optimization: AutoAgent-Inspired Self-Improvement Gameplan

> **Date**: 2026-04-04  
> **Status**: Proposal  
> **References**: [kevinrgu/autoagent](https://github.com/kevinrgu/autoagent), `~/obsidian/autonomy/`, `autonomy.py`

---

## 1. Problem Statement

Lloyd's autonomy system already has a self-improvement task (#29) and a nightly reflection chain (#38→39a→39→40→47→48), but these operate without formal measurement. Task #29 tunes timeouts, frequencies, and priorities based on heuristics — it has no baseline, no before/after comparison, and no rollback mechanism. Config changes from #40 are applied blind.

The [autoagent](https://github.com/kevinrgu/autoagent) project demonstrates a different approach: a **hill-climbing optimization loop** where a meta-agent modifies the agent's own prompt, tools, and orchestration, benchmarks the result, and keeps improvements while reverting regressions. The core insight is simple — **measure before you change, measure after, keep only what helps**.

This gameplan proposes integrating that pattern into Lloyd's existing autonomy infrastructure, upgrading the self-improvement loop from heuristic tuning to evidence-based optimization.

---

## 2. What AutoAgent Does (Relevant Subset)

AutoAgent's architecture:

```
program.md (objectives)
    │
    ▼
Meta-Agent (Claude w/ extended thinking)
    │
    ├── Reads current agent.py (harness + prompt + tools)
    ├── Proposes a modification
    ├── Runs benchmark suite
    ├── Compares score to baseline
    └── Keeps change if score improved, reverts if not
```

Key design choices worth adopting:
1. **Fixed/modifiable boundary** — clear separation of what the optimizer can and cannot touch
2. **Snapshot + rollback** — every modification is versioned; revert is cheap
3. **Score-driven decisions** — no change is accepted without measured improvement
4. **Extended thinking** — budget for complex reasoning about what to change and why
5. **Single-file harness** — keeps the modifiable surface small and parseable

What's NOT relevant: Harbor container isolation (distrobox handles this), standardized benchmark suites (Lloyd is a general-purpose assistant, not a benchmark runner), OpenAI provider fallback.

---

## 3. Current State Analysis

### 3.1 Existing Tasks That Touch Self-Improvement

| Task | What It Does | Gap |
|------|-------------|-----|
| **#29 Self-Improvement Loop** | Tunes timeouts, frequencies, priorities | No baseline metrics, no before/after, no rollback |
| **#40 Nightly Reflection — Config** | Applies config improvements from signal report | Blind application — no measurement of effect |
| **#38→39a→39 Reflection Chain** | Signal detection → knowledge analysis → knowledge write | Good at extracting learnings, but doesn't feed metrics to #29 |
| **#45 Trajectory Extraction** | Captures tool calls to daily JSONL | Raw data only — no scoring or quality metrics |
| **#46 Trajectory Skill Mining** | Mines error patterns into new skills | Focused on error recovery, not optimization |

### 3.2 Existing Tasks With Issues

| Task | Issue | Recommendation |
|------|-------|----------------|
| **#6 Morning Briefing** | `status: draft` — not running since 2026-03-31 | Reactivate or retire. If the skill works, set to `up_next`. If not, mark `archived` to stop it cluttering the scheduler's scan. |
| **#35 Daily Backlog Triage** | `status: draft` — not running since 2026-03-31 | Same — either reactivate or archive. |
| **#41 Email + Calendar** | BLOCKED — missing tools (email_recent, calendar_events) | Keep draft until Thunderbird MCP or Google Workspace OAuth is configured. Add a `blocked_reason` field so the scheduler skips it without burning cycles. |

### 3.3 What's Missing

1. **Metrics collection** — No structured performance data. Run records capture duration and success/fail, but not task-specific quality scores.
2. **Baseline snapshots** — No versioned snapshots of SOUL.md, config.yaml, or skill files before changes.
3. **Rollback mechanism** — If a config change makes things worse, there's no automated revert.
4. **Scoring framework** — No way to compute "did this change help?" beyond eyeballing activity logs.
5. **Safety boundary** — Autonomy tasks run with `bypassPermissions` and could edit anything, including `server.py` or `autonomy.py` itself.

---

## 4. Proposed Architecture

### 4.1 Overview

```
┌─────────────────────────────────────────────────────┐
│                  Meta-Optimization Loop              │
│                                                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐       │
│  │ Measure  │───▶│ Propose  │───▶│ Evaluate │──┐    │
│  │ Baseline │    │ Change   │    │ Result   │  │    │
│  └──────────┘    └──────────┘    └──────────┘  │    │
│       ▲                                   │    │    │
│       │              ┌────────────────────┘    │    │
│       │              ▼                         │    │
│       │    ┌──────────────────┐                │    │
│       │    │ Keep or Rollback │────────────────┘    │
│       │    └──────────────────┘                      │
│       │              │                               │
│       └──────────────┘                               │
└─────────────────────────────────────────────────────┘
                       │
          Feeds into existing tasks:
          #38 Signals, #40 Config, #46 Skill Mining
```

### 4.2 Fixed / Modifiable Boundary

Inspired by autoagent's "FIXED ADAPTER BOUNDARY":

**Modifiable by the optimizer (safe to mutate):**
- `SOUL.md` — personality, behavioral instructions
- `config.yaml` — model selection, timeouts, agent settings
- `~/obsidian/autonomy/*.md` — task frequencies, priorities, dependencies
- `~/obsidian/skills/*/SKILL.md` — skill instructions
- `prompt_builder.py` prompt fragments — template strings only

**Fixed (never touched by autonomy tasks):**
- `server.py` — API endpoints, SSE bridge
- `autonomy.py` — scheduler logic
- `mcp-servers/*.py` — tool implementations
- `web/` — frontend code
- `prompt_builder.py` code logic — only the template strings are modifiable

This boundary should be enforced, not just documented. See Phase 2 below.

### 4.3 Metrics Framework

A lightweight scoring system stored alongside run records:

```yaml
# ~/obsidian/autonomy/runs/<task_id>/<run_id>.md (extended frontmatter)
---
run_id: run_29_20260404_030000
task_id: 29
status: success
duration_seconds: 58.3
metrics:
  task_success_rate_7d: 0.94      # % of all tasks that succeeded in last 7 days
  avg_duration_7d: 42.1           # mean task duration across all tasks
  error_rate_7d: 0.06             # % of tasks that failed
  silent_rate_7d: 0.31            # % of tasks that returned [SILENT]
  scheduler_utilization_7d: 0.72  # % of ticks that found a due task
  skill_hit_rate_7d: 0.88        # % of tasks where skill_path resolved
  config_snapshot: abc123         # git short hash or content hash of config.yaml
  soul_snapshot: def456           # content hash of SOUL.md
---
```

A Python module (`metrics.py`) computes these from run records:

```python
def compute_system_metrics(days: int = 7) -> dict:
    """Scan run records, compute aggregate health metrics."""
    ...

def snapshot_config() -> str:
    """Return content hash of config.yaml + SOUL.md."""
    ...

def compare_snapshots(before: str, after: str) -> dict:
    """Diff two config snapshots, return what changed."""
    ...
```

---

## 5. Implementation Phases

### Phase 1: Metrics Collection (foundation)

**Goal**: Give the self-improvement loop real numbers to work with.

**New file**: `~/lloyd/metrics.py`
- `compute_system_metrics(days=7)` — scans `~/obsidian/autonomy/runs/` for run records, computes success rate, avg duration, error rate, silent rate, scheduler utilization
- `compute_task_metrics(task_id, days=7)` — per-task version of the above
- `snapshot_modifiable_state()` — hashes config.yaml + SOUL.md + task frontmatter, returns a snapshot ID
- `save_snapshot(snapshot_id)` — copies current modifiable files to `~/obsidian/autonomy/snapshots/<id>/`
- `restore_snapshot(snapshot_id)` — copies snapshot files back, overwriting current

**New file**: `~/obsidian/autonomy/snapshots/` directory for versioned state.

**Changes to `autonomy.py`**:
- After each `run_task()` completes, call `compute_task_metrics()` and include in run record frontmatter
- No behavioral change — just data collection

**Effort**: Small. Mostly file scanning and YAML parsing.

### Phase 2: Safety Boundary Enforcement

**Goal**: Prevent autonomy tasks from modifying fixed infrastructure.

**Changes to `autonomy.py`**:
- Add a `MODIFIABLE_PATHS` allowlist:
  ```python
  MODIFIABLE_PATHS = [
      Path.home() / "lloyd" / "SOUL.md",
      Path.home() / "lloyd" / "config.yaml",
      Path.home() / "obsidian" / "autonomy",
      Path.home() / "obsidian" / "skills",
      Path.home() / "obsidian" / "memory",
  ]
  ```
- When building `ClaudeAgentOptions` for autonomy runs, add file-write restrictions via `disallowed_tools` or a wrapper that validates paths before writing. The simplest approach: a pre-execution subliminal that says "You may only modify files under these paths: ..." and rely on the model respecting it (soft boundary). A hard boundary would require a custom MCP tool that validates paths, which is Phase 3 work.

**Changes to task #29**:
- Update the skill to explicitly state it may only modify task frontmatter (frequencies, priorities, timeouts) and must not touch skill content or SOUL.md.

**Effort**: Small. Mostly prompt/config changes.

### Phase 3: Upgrade Self-Improvement Loop (#29)

**Goal**: Transform task #29 from heuristic tuning into an autoagent-style hill-climbing optimizer.

**New skill**: `~/obsidian/skills/autonomy-self-improvement/SKILL.md` (rewrite)

The upgraded loop runs daily and follows this cycle:

```
1. MEASURE — call metrics.py to compute 7-day system metrics
2. COMPARE — load the last snapshot's metrics, compare to current
3. DIAGNOSE — identify the worst-performing area:
   - High error rate on specific task? → tune its timeout or frequency
   - High silent rate? → task is running too often, reduce frequency
   - Low scheduler utilization? → tasks are too infrequent, increase RPD
   - Specific task consistently slow? → check skill complexity
4. PROPOSE — draft a single, minimal change (one variable at a time)
5. SNAPSHOT — save current state via metrics.snapshot_modifiable_state()
6. APPLY — make the change
7. LOG — write the proposal, rationale, and snapshot ID to the run record
```

The **evaluation** happens on the next run (24h later):
```
1. MEASURE current metrics
2. LOAD previous run's snapshot ID and metrics
3. COMPARE — did the change help?
4. If regression detected (>5% worse on any key metric):
   - ROLLBACK to previous snapshot
   - LOG the rollback with reason
5. If improvement or neutral:
   - KEEP the change
   - LOG the result
```

**Key constraint**: Only one change per cycle. Multiple simultaneous changes make it impossible to attribute improvement or regression.

**Effort**: Medium. Requires metrics.py from Phase 1 and careful skill authoring.

### Phase 4: Config Reflection Upgrade (#40)

**Goal**: Wire the nightly reflection config task into the metrics/snapshot system.

**Changes to task #40 skill**:
- Before applying any config change, call `snapshot_modifiable_state()`
- After applying, log the snapshot ID and what changed
- On next run, check if the previous change caused regression (same compare logic as #29)
- If #29 and #40 both want to change config, #40 defers — it writes a "proposed change" to the signal report instead of applying directly

**Dependency change**: #40 should `depends_on: 29` to avoid conflicting mutations.

**Effort**: Small. Skill rewrite + dependency update.

### Phase 5: Trajectory Quality Scoring

**Goal**: Extend trajectory extraction (#45) to include quality scores, enabling the optimizer to measure conversation quality, not just task health.

**Changes to task #45 skill**:
- After extracting tool calls, compute per-session quality signals:
  - `tool_error_rate` — % of tool calls that returned errors
  - `correction_count` — number of user corrections detected (regex for "no", "wrong", "actually", "I meant")
  - `turn_count` — total turns (proxy for efficiency)
  - `tool_diversity` — unique tools used / total tool calls (proxy for skill breadth)
- Write these to the trajectory JSONL as metadata fields

**New metrics in `metrics.py`**:
- `compute_conversation_quality(days=7)` — aggregates trajectory quality scores
- Feed these into the self-improvement loop (#29) as additional signals

**Effort**: Medium. Requires parsing session transcripts and computing heuristics.

### Phase 6: SOUL.md Optimization (advanced)

**Goal**: Allow the optimizer to propose and test changes to SOUL.md behavioral instructions.

This is the most powerful and most dangerous phase. AutoAgent's core value proposition is that it can improve the agent's own prompt — but SOUL.md is Lloyd's identity.

**Guardrails**:
- Only the **Instructions** and **Tool Usage** sections of SOUL.md are modifiable — identity, personality, and core values are fixed
- Maximum 1 SOUL.md change per week (not per day)
- Changes must include a hypothesis ("I expect this change to reduce correction_count by X%")
- Automatic rollback if correction_count increases by >10% over the following week
- All SOUL.md mutations are logged to a dedicated `~/obsidian/autonomy/soul-changelog.md`

**Implementation**:
- Extend #29's skill to include SOUL.md in its optimization scope (weekly cadence)
- Add `soul_snapshot` to metrics framework
- Gate behind a config flag: `agent.soul_optimization: false` (opt-in)

**Effort**: Medium. Mostly skill authoring + guardrails. The hard part is evaluating "did this prompt change actually help?" — correction_count and user satisfaction signals are noisy.

---

## 6. Adjustments to Existing Autonomy Tasks

### Tasks to Modify

| Task | Change | Rationale |
|------|--------|-----------|
| **#29 Self-Improvement** | Rewrite skill per Phase 3. Add `depends_on: null` (runs independently). Keep `runs_per_day: 1`. | Core of this gameplan |
| **#40 Nightly Config** | Add snapshot/rollback. Set `depends_on: 29`. | Prevent conflicting mutations |
| **#45 Trajectory Extraction** | Add quality scoring per Phase 5. | Feed optimizer with conversation-level signals |
| **#38 Nightly Signals** | No code change, but add a "metrics delta" section to signal report format. | Give downstream tasks (#39a, #40) access to trend data |

### Tasks to Reactivate or Archive

| Task | Action | Rationale |
|------|--------|-----------|
| **#6 Morning Briefing** | Set `status: archived` | Has been `draft` since 2026-03-31. The skill may have worked under hermes but likely needs rewriting for Lloyd's MCP-based toolset. Archive to declutter the scheduler; restore when the skill is updated. |
| **#35 Daily Backlog Triage** | Set `status: up_next` | The skill path points to `skills/backlog-triage/SKILL.md` (relative path). If the skill exists and works, reactivate. If not, archive like #6. |
| **#41 Email + Calendar** | Add `blocked_reason: "missing email/calendar tools"`, keep `status: up_next` but add logic to skip blocked tasks | Running every 15min just to log "BLOCKED" wastes cycles. Either pause it or add blocked-task skipping to the scheduler. |

### New Task: Metrics Snapshot (#52)

A lightweight task that runs every 6 hours, computes and stores system metrics without proposing any changes. This creates a continuous metrics baseline independent of the optimizer's cadence.

```yaml
---
id: 52
name: Metrics Snapshot
frequency: daily
runs_per_day: 4
priority: low
status: up_next
skill_path: ~/obsidian/skills/metrics-snapshot/SKILL.md
timeout_seconds: 120
preemptible: true
---
```

---

## 7. Scheduler Enhancement: Blocked Task Skipping

Currently, `_is_task_due()` checks frequency, last_run, dependencies, and preferred_hours — but not whether a task is known to be blocked. Task #41 demonstrates the problem: it runs every 15 minutes, fails with "BLOCKED: missing tools", and logs success anyway.

**Proposed change to `autonomy.py`**:

```python
def _is_task_due(task: dict, all_tasks: list[dict]) -> bool:
    # ... existing checks ...
    
    # Skip tasks with known blockers
    blocked = str(task.get("blocked_reason", "") or "").strip()
    if blocked:
        return False
    
    # Skip tasks with high failure counts (circuit breaker)
    failure_count = int(task.get("failure_count") or 0)
    max_retries = int(task.get("max_retries") or 3)
    if failure_count >= max_retries:
        return False
    
    return True
```

This is a small, safe change that prevents wasted cycles on tasks that can't succeed.

---

## 8. Implementation Order & Dependencies

```
Phase 1: Metrics Collection
    │
    ├── Phase 2: Safety Boundary (can parallelize with Phase 1)
    │
    ▼
Phase 3: Upgrade #29 (requires Phase 1)
    │
    ├── Phase 4: Upgrade #40 (requires Phase 3)
    │
    ├── Phase 5: Trajectory Scoring (independent, can parallelize)
    │
    ▼
Phase 6: SOUL.md Optimization (requires Phase 3 + Phase 5 + weeks of metrics data)
```

**Quick wins (can do immediately):**
- Scheduler blocked-task skipping (Section 7)
- Archive/reactivate stale tasks (Section 6)
- Add `blocked_reason` to #41

**Phase 1+2** are the foundation and should be done together before anything else.

**Phase 6** is explicitly deferred until the metrics pipeline has been running for at least 2 weeks, providing a stable baseline.

---

## 9. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Optimizer makes config worse | Medium | Medium | Snapshot + automatic rollback (Phase 3) |
| SOUL.md mutation degrades personality | Low | High | Weekly cadence, identity sections frozen, opt-in flag (Phase 6) |
| Metrics are noisy/misleading | Medium | Low | 7-day rolling windows, >5% threshold for regression detection |
| Optimizer and reflection chain conflict | Medium | Medium | Dependency chain (#40 depends_on #29), single-change-per-cycle rule |
| Snapshot storage grows unbounded | Low | Low | Prune snapshots older than 30 days, keep only those referenced by run records |

---

## 10. Success Criteria

After 30 days of operation:
- System-wide task success rate ≥ 95% (currently unmeasured)
- Zero unintended config regressions (all changes tracked and reversible)
- Self-improvement loop has made ≥ 5 measured, beneficial changes
- No manual intervention required to fix optimizer-caused issues
- Metrics dashboard shows clear trend data (success rate, duration, error rate over time)
