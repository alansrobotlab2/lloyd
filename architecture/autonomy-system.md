---
segment: architecture
tags: [architecture,lloyd]
status: active
type: reference

---

# Autonomy System Architecture

**Created:** 2026-03-22
**Last updated:** 2026-04-02
**Status:** Active (GPU-gated dispatch implemented,nightly chain + dream consolidation operational)

## Overview

The autonomy system enables multi-agent collaboration through a shared backlog with GPU-aware task dispatch. Four specialized agent types pull from a single backlog,with LLM dispatch gated by GPU utilization to avoid interfering with foreground tasks.

**Storage:** Vault markdown files (migrated from SQLite 2026-03-29). Task files at `~/obsidian/autonomy/{id}-{slug}.md`,run files at `~/obsidian/autonomy-runs/{task-id}/{run-id}.md`,config at `~/obsidian/autonomy/_config.md`. Three consumers (MCP tools,MC extension,idler daemon) read/write markdown directly. QMD indexes both `autonomy` and `autonomy-runs` collections for searchability.

## Design Principles

1. **Sense -> Analyze -> Act** — ingest raw data,reflect on it,then improve from it
2. **One job per task** — each task has a single clear responsibility with no overlap
3. **Fail forward** — `stale_bypass_hours` lets dependent tasks run with stale input rather than blocking the whole chain when one step fails
4. **Preemptible by default** — low-priority maintenance yields to high-priority work
5. **Closed-loop learning** — trajectories feed skill mining,reflection feeds dream,dream feeds skills management. Nothing is write-only.

## Agent Types

### Memory Agent (Periodic Jobs)
- **Role:** Scheduled automation tasks
- **Tasks:** Nightly fact extraction,index rebuild,periodic memory capture,relationship discovery
- **Model:** `local-llm-120b/Qwen3.5-122B-A10B` (GPU 1)
- **Schedule:** Cron-based (15m periodic,nightly 2:00-5:15 AM PST)
- **Delivery:** Silent (`delivery.mode: "none"`) unless errors occur

### Operator Agent (Backlog Review)
- **Role:** Analysis,code changes,config updates
- **Tasks:** Implementation work,bug fixes,system maintenance
- **Model:** `anthropic/claude-opus-4-6` (main agent) or `local-llm-120b` for subagents
- **Trigger:** Manual assignment or backlog tagging (`tag="implementation"`)

### Idler Agent (Deep Maintenance)
- **Role:** Continuous background work when GPU idle
- **Tasks:** Memory audit,relationship discovery,skill harvest,backlog triage,research synthesis
- **Model:** `local-llm-120b/Qwen3.5-122B-A10B` (GPU 1)
- **Dispatch:** GPU-gated (polls every 250ms,dispatches when GPU < 30%)
- **Pattern:** GPU idle -> pull next task; GPU busy -> wait,doesn't compete

### Researcher Agent (Content Ingestion)
- **Role:** External content collection and synthesis
- **Tasks:** GitHub monitoring,RSS feeds,documentation,research papers
- **Model:** `local-llm-120b/Qwen3.5-122B-A10B` (GPU 1)
- **Trigger:** Scheduled or manual assignment

## Task Inventory (19 active tasks)

### High-Frequency (hourly or more)

| ID | Name | Freq | Priority | Role |
|----|------|------|----------|------|
| #25 | Memory Capture | hourly | high | Raw session transcript extraction to daily notes |
| #33 | Groundskeeper Loop | 15min | low | Fix broken links,stale facts,orphans (5/run) |
| #34 | Groundskeeper Research | hourly | low | Deep web research for thin profiles/stubs |
| #41 | Email+Calendar Monitor | 15min | medium | Poll email/calendar,surface relevant items |

### Multi-Daily

| ID | Name | Freq | Priority | Role |
|----|------|------|----------|------|
| #24 | Data Pipeline | 6x/day | high | Fact extraction -> inference -> relations -> index |
| #36 | Groundskeeper Survey | 4x/day | low | Scan vault,rebuild issue queue for #33/#34 |

### Daily (Daytime)

| ID | Name | Freq | Priority | Depends | Role |
|----|------|------|----------|---------|------|
| #35 | Backlog Triage | daily | medium | -- | Evaluate inbox items,promote/close |
| #45 | Trajectory Extraction | daily | low | -- | Capture worker session tool calls to JSONL |
| #46 | Trajectory Skill Mining | daily | medium | #45 | Mine error patterns,dispatch Opus to author skills |
| #30 | Intelligence Pipeline | daily | medium | -- | GitHub + YouTube scan,score,vault write |
| #29 | Self-Improvement Loop | daily | medium | -- | System metrics and meta-optimization |

### Nightly Chain (2-5am PST)

| ID | Name | Priority | Depends | Bypass | Role |
|----|------|----------|---------|--------|------|
| #38 | Reflection: Signals | high | -- | 36h | Detect correction signals from daily notes |
| #39a | Reflection: Knowledge Analysis | high | #38 | 36h | Derive updates,write JSON handoff |
| #39 | Reflection: Knowle