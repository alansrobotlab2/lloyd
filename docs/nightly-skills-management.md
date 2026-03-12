---
tags:
  - lloyd
  - architecture
type: reference
segment: projects
---

# Nightly Skills Management Architecture

An automated pipeline that mines Lloyd's session transcripts for reusable procedural knowledge, creates and maintains a skills library, and self-corrects incorrect or stale skills.

## Schedule

- **Time:** 3:00 AM PST (daily)
- **Agent:** `memory` (isolated session)
- **Model:** Claude Opus 4.6
- **Budget:** <$15 per run
- **Sequence:** Runs after [[nightly-vault-maintenance]] (2am), before [[nightly-reflection]] (4am)
- **Skill file:** [`nightly-skills-management/SKILL.md`](../../skills/nightly-skills-management/SKILL.md)

## Overview

Every interaction between Alan and Lloyd generates session transcripts stored as JSONL files. These transcripts contain procedural knowledge -- troubleshooting steps, corrections, behavioral rules, gotchas -- that would otherwise be lost between sessions. The skills management pipeline automatically surfaces this knowledge, packages it as reusable skills, and maintains the library over time.

This is a key component of Lloyd's self-improvement architecture, alongside [[nightly-reflection]] (mental models, MEMORY.md consolidation, config improvements) and [[nightly-vault-maintenance]] (structural hygiene).

## Three-Stage Pipeline

```
Session Transcripts (JSONL)          Skills Library
~/.openclaw/agents/*/sessions/       ~/obsidian/skills/
         |                                  ^
         v                                  |
+---------------------+          +----------------------+
|  Stage 1: Extract   |          |  Stage 3: Dedup &    |
|  extract-session-   |          |  Consolidate         |
|  log.py --hours 24  |          |  - Inventory all     |
|                     |          |    custom + built-in  |
|  Output: .log files |          |  - Flag overlaps     |
|  per session        |          |  - Merge redundant   |
+--------+------------+          +----------^-----------+
         |                                  |
         v                                  |
+---------------------+          +----------+-----------+
|  Stage 2: Evaluate  |--------->|  Create / Update     |
|  Read each .log,    |          |  Skills              |
|  apply signal       |          |  ~/obsidian/skills/  |
|  detection criteria |          |  <name>/SKILL.md     |
+---------------------+          +----------------------+
```

### Data Flow

1. **Source**: Session JSONL files from `~/.openclaw/agents/main/sessions/`, `~/.openclaw/agents/memory/sessions/`, and `~/.openclaw/logs/cc-instances/`
2. **Extraction**: Python script produces one `.log` file per session in `~/obsidian/memory/skill-maintenance/YYYY-MM-DD/`
3. **Evaluation**: Each log is read and assessed against signal detection criteria
4. **Output**: New or updated skills in `~/obsidian/skills/<name>/SKILL.md`, report in `~/obsidian/memory/skill-maintenance/YYYY-MM-DD/report.md`

## Signal Detection Criteria

The evaluation stage looks for seven categories of skill-worthy patterns:

### 1. Corrections
Alan corrected Lloyd's behavior or approach. Extract as a guardrail -- update an existing skill with a warning, or create a new skill with the correct procedure.

### 2. Remember Requests
Alan explicitly asked to preserve a procedure ("from now on", "always do X"). Create a skill capturing the procedure exactly.

### 3. Failure-to-Fix Chains
Something broke, investigation found a non-obvious root cause after multiple steps. Extract the diagnostic path and fix as a troubleshooting skill.

### 4. Behavioral Rules
Corrections that establish ongoing patterns ("never do X", "always Y"). Extract as a guardrail skill or update an existing skill's constraints.

### 5. Troubleshooting Playbooks
Failure-to-fix chains where the root cause was not obvious from the symptom -- even one-time incidents. Extract as: symptom, investigation steps, root cause, fix.

### 6. Stale or Incorrect Skill Steps
Evidence that an existing skill's instructions are wrong or outdated. Two failure modes:
- **(A) Wrong skill selected**: Fix description/tags, leave steps alone
- **(B) Right skill, wrong steps**: Update the actual procedure

### 7. Generic Patterns in Project-Specific Sessions
Many sessions involve project-specific work but contain generic troubleshooting or operational patterns underneath. Rule: if the ROOT CAUSE or DIAGNOSTIC APPROACH would apply to a different project, extract it -- even if the session context was specific.

### What to Skip
- Bug fixes with obvious causes (non-obvious = skill-worthy)
- Simple delegation (read context, dispatch orchestrator) with no novel steps
- Casual conversation, greetings, status checks
- Procedures already well-covered by existing skills
- Sessions where context is project-specific AND root cause is also project-specific

## Components

### Extraction Script
- **Path**: `~/obsidian/agents/memory/scripts/extract-session-log.py`
- **Usage**: `python3 extract-session-log.py --date YYYY-MM-DD` or `--hours 24`
- **Output**: One `.log` file per session with one line per event (`user:`, `tool:`, `lloyd:`)
- **Filters**: Strips metadata, daily_notes blocks, system prompts, heartbeats

### Output Artifacts
- **Daily reports**: `~/obsidian/memory/skill-maintenance/YYYY-MM-DD/report.md`
- **Session logs**: `~/obsidian/memory/skill-maintenance/YYYY-MM-DD/*.log`
- **Skills**: `~/obsidian/skills/<name>/SKILL.md`
- **Git snapshots**: Pre-run and post-run commits for rollback safety

## Skills Library Inventory

As of 2026-03-10, the library contains **34 skills**. The extraction pipeline produced 13 new skills and updated 10 existing skills from the initial two-week catchup (2026-02-24 to 2026-03-09).

### Skills Created by Extraction

| Skill | Category | Description |
|-------|----------|-------------|
| `openclaw-plugin-authoring` | Gotchas | Manifest fields, ES module format, streaming hook limitations |
| `stale-process-cleanup` | Troubleshooting | Kill lingering GPU/port/WS processes after service restarts |
| `git-filter-repo` | Troubleshooting | `--invert-paths` deletes from working directory, not just history |
| `secret-migration` | Procedure | SecretRef env var migration, distrobox restart gotcha |
| `backlog-as-handoff` | Behavioral | Using backlog descriptions as self-contained handoff docs |
| `config-wipe-recovery` | Troubleshooting | Silent failures after upgrades/cleanup wipe config/data dirs |
| `gpu-device-mismatch` | Troubleshooting | CUDA device ID errors, LD_LIBRARY_PATH in systemd |
| `https-migration-gotchas` | Troubleshooting | HTTP-to-HTTPS breaks service-to-service calls |
| `browser-audio-pipeline` | Troubleshooting | Frame size mismatches, VAD state corruption, WS architecture |
| `cron-stuck-recovery` | Troubleshooting | `runningAtMs` stuck state, force-run regression |
| `local-llm-gotchas` | Gotchas | Reasoning tokens eating max_tokens, model swap procedure |
| `qmd-collection-management` | Troubleshooting | Symlink gotchas, manual reindex, missing collections |
| `distrobox-service-setup` | Troubleshooting | Running services without systemd, socket path issues |

### Skills Updated by Extraction

| Skill | Key Additions |
|-------|--------------|
| `orchestrator-delegation` | Backlog lifecycle, scope discipline, model selection, gateway restart guardrail |
| `websearch` | Bare URL trigger for knowledge file creation |
| `discord-voice-setup` | Corrected DAVE encryption guidance (was wrong), added troubleshooting |
| `mc-ui-change` | Session label tracing, autoplay gotcha, TypeScript drift, summary debugging |
| `voice-mode` | GPU device assignment, ALSA sample rate, WebSocket input mode, hooks payload correction |
| `stale-process-cleanup` | vLLM child process pattern, WS diagnosis, systemd restart race |
| `config-wipe-recovery` | Orchestrator full-file overwrite pattern, git history recovery |
| `claude-code-subagent` | Binary path correction |
| `secret-migration` | Unsupported SecretRef fields, git history cross-reference |
| `restart-openclaw` | ClawDeck section deprecated |

## Design Decisions

### Why Skills, Not Memory?
MEMORY.md captures *what Lloyd knows* (facts, preferences, operational state). Skills capture *how to do things* (procedures, gotchas, diagnostic paths). Skills are discoverable via `skills_search`, loadable on-demand, and can be shared across agents. Memory is personal and session-scoped.

### Why Nightly, Not Real-Time?
Real-time skill extraction during sessions would interrupt flow and add latency. Nightly batch processing lets the pipeline evaluate a full day's work with hindsight, cross-reference across sessions, and identify patterns that only emerge when viewing multiple interactions together.

### Why Three Stages?
- **Extract** is mechanical (Python script) -- fast, deterministic, cheap
- **Evaluate** requires judgment (LLM) -- needs to read context, assess novelty, check for duplicates
- **Dedup** requires global view (LLM) -- must see the full library to find overlaps

Separating them allows independent improvement. The extraction script can be optimized without touching evaluation criteria, and criteria can be refined without changing the extraction format.

### Criteria Evolution
The evaluation criteria went through three iterations during the initial catchup:

| Version | Key Change | Yield |
|---------|-----------|-------|
| v1 (strict) | Required 3+ occurrences, dismissed "project-specific" and "one-time" | 2 skills, 1 update |
| v2 (loosened) | Dropped recurrence requirement, added troubleshooting playbooks + behavioral rules | +3 skills, +1 update |
| v3 (anti-filter) | Extract generic patterns from project-specific sessions, stale skill detection with A/B distinction | +8 skills, +13 updates |

The biggest unlock was v3's "project-specific" anti-filter: most work involves specific projects, but root causes and diagnostic approaches are often generic and reusable.

### Self-Correcting
The pipeline corrected an existing skill during the catchup -- `discord-voice-setup` was giving wrong advice about DAVE encryption. The "stale or incorrect skill steps" detection category with A/B distinction (wrong skill selected vs right skill with wrong steps) prevents the pipeline from corrupting good skills while still catching genuinely incorrect ones.

## Cost Profile

| Run Type | Typical Cost | Duration |
|----------|-------------|----------|
| Nightly (1 day, ~25 sessions) | $3-6 | 3-5 minutes |
| Catchup (14 days, ~150 sessions) | $3-15 | 10-18 minutes |

All runs use Claude Opus 4.6 for evaluation quality. Budget cap: $15 per run.

## Known Issues

1. **Delivery config**: The cron job has `delivery: "announce"` but no channel specified -- errors with "Discord recipient is required." Job completes fine, report is written to file. Should either set a specific channel or switch to `delivery: "none"`.
2. **Timeout**: Set to 10 minutes. First successful run took ~3 minutes, but heavy days could take longer. Monitor and adjust.
3. **Large sessions**: Sessions over 30KB are skimmed (first 200 lines + keyword search). Some procedural detail may be missed in very long sessions.
4. **ClawhHub dedup in Stage 2**: The evaluation stage now automatically sees ClawhHub catalog matches via unified `skills_search`, eliminating the need for a separate ClawhHub check step.

## Nightly Automation Sequence

| Time (PST) | Job | Purpose | Skill |
|------------|-----|---------|-------|
| 2:00 AM | `daily-vault-maintenance` | Tag hygiene, frontmatter validation, structure review | [SKILL.md](../../skills/nightly-vault-maintenance/SKILL.md) |
| 3:00 AM | `nightly-skills-management` | Skills extraction, evaluation, dedup | [SKILL.md](../../skills/nightly-skills-management/SKILL.md) |
| 4:00 AM | `nightly-reflection` | Self-improvement, mental models, MEMORY.md consolidation | [SKILL.md](../../skills/nightly-reflection/SKILL.md) |
| Every 15m | `periodic-memory-capture` | Extract transcripts to daily notes | [SKILL.md](../../skills/periodic-memory-capture/SKILL.md) |

## Related Docs

- [[nightly-reflection]] -- Nightly Reflection (self-improvement, mental models)
- [[nightly-vault-maintenance]] -- Vault Maintenance (structural hygiene)
- [[memory]] -- Memory System Architecture
- [[skills]] -- Skill System
- [Nightly Skills Management Skill](../../skills/nightly-skills-management/SKILL.md) -- Current implementation
