---
segment: architecture
tags: [architecture,lloyd]
backlog: 177
date: 2026-03-22
related:
- architecture/nightly-reflection.md
- architecture/autonomy-system.md
- architecture/index.md
relations:
  related-to:
  - projects/lloyd/plans/document-relations-retrieval.md
  - architecture/agents.md
  - architecture/backlog.md
  - architecture/infrastructure.md
  - architecture/morning-briefing.md
  - architecture/nightly-skills-management.md
  - architecture/nightly-vault-maintenance.md
  - architecture/skills.md
  - architecture/tools.md
  - architecture/voice.md
  - architecture/memory.md
  - architecture/evaluation-engine.md
tags: [architecture]
status: planned
type: reference

---













# Evaluation Engine Architecture

**Backlog:** #177 — Self-Learning: Evaluation Engine & Feedback Loops
**Status:** Planned (architecture design phase)
**Dependencies:** None (this is foundational — #179 and #180 depend on it)
**Execution Model:** Autonomy system (idler agent + GPU-gated dispatch)

## Problem Statement

Lloyd has a strong modification engine (reflection pipeline,config application,skill creation) but a weak evaluation engine. Changes are applied based on correction signals,but there's no systematic measurement of whether those changes actually improved outcomes. The self-improvement loop is optimizing blind.

**Current state:**
- Corrections are logged but not quantified over time
- Reflection pipeline applies changes but doesn't measure before/after impact
- Strategy decisions (approach A vs. B) aren't tracked against outcomes
- Autonomy task quality isn't sampled or scored
- No self-critique — the only learning signal comes from user corrections (sparse)

**Goal:** Build the measurement layer that closes the loop. Every change should have a measurable outcome. Every session should produce evaluation data.

## Design Principles

### Autonomy-Native

The evaluation engine — like all of Lloyd's automation — runs on the **idler agent** via the **autonomy scheduler**. No fixed cron times. No overnight batches. The idler dispatches evaluation tasks whenever GPU is idle,multiple times per day.

| Property | Value |
|----------|-------|
| **Timing** | Whenever GPU is idle — could be 2 PM or 3 AM |
| **Frequency** | Multiple times per day — evaluate sessions while they're fresh |
| **Model** | Local 122B on GPU 1 — effectively free |
| **Freshness** | Sessions evaluated within hours of completion |
| **Contention** | GPU-gated — never competes with foreground work |
| **Scheduling** | `runs_per_day` — autonomy scheduler handles timing |

### Continuous Processing

Evaluation is a **continuous background process**,not a batch job:

```
Sessions happen throughout the day
         │
         v
Periodic Memory Capture (every 15m) writes session transcripts
         │
         v
Idler picks up Session Scoring task (runs_per_day: 4)
  └─ Scores any un-scored sessions since last run
         │
         v
Idler picks up Self-Critique task (runs_per_day: 3)
  └─ Reviews scored sessions,generates synthetic corrections
         │
         v
Idler picks up Strategy Extraction task (runs_per_day: 2)
  └─ Extracts decision patterns from critiqued sessions
         │
         v
Data accumulates in metrics store — always fresh
         │
         v
Signal Processing reads evaluation data whenever it runs
  └─ Self-critique signals already waiting
```

**Key insight:** Evaluation data is always fresh. Any downstream consumer (signal processing,config application,morning briefing) reads whatever has accumulated — no timing dependencies to manage.

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                  DATA SOURCES                        │
│                                                      │
│  Session Transcripts    Corrections.md    Backlog    │
│  Git History            Autonomy Runs    Tool Logs   │
└──────────┬──────────────────┬──────────────┬────────┘
           │                  │              │
           v                  v              v
┌─────────────────────────────────────────────────────┐
│         COLLECTION LAYER (Autonomy Tasks)            │
│         Idler Agent — GPU-Gated — Local 122B         │
│                                                      │
│  Session Scorecards  │  Self-Critique  │  Autonomy   │
│  (4x/day)            │  (3x/day)       │  QA (2x/wk) │
└──────────┬──────────────────┬──────────────┬────────┘
           │                  │              │
           v                  v              v
┌─────────────────────────────────────────────────────┐
│              METRICS STORE                           │
│                                                      │
│  memory/metrics/sessions/YYYY-MM-DD.jsonl            │
│  memory/metrics/self-critique/YYYY-MM-DD.jsonl       │
│  memory/metrics/autonomy-qa/YYYY-MM-DD.jsonl         │
│  memory/metrics/strategy/decisions.jsonl              │
│  memory/metrics/weekly-rollup.md       