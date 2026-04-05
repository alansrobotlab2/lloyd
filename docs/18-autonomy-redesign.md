# 15 — Autonomy Redesign: From Maintenance to Learning

**Date:** 2026-04-05
**Status:** Planned

## Problem

Lloyd's autonomy pipeline has 19 tasks (16 active), but almost all are maintenance or extraction:
- Cleaning broken links, deduping entities, capturing sessions, fixing config
- The system digests what happened but never proactively seeks new knowledge
- Groundskeeper alone consumed ~16 hours of GPU time on janitorial work (379 runs)
- Memory Capture runs hourly but usually no-ops ("no new sessions")
- YouTube intelligence feeds are dead (63/66 returning 404)
- 3 tasks stuck in draft limbo since March

The pipeline is great at reflection but has zero active learning.

## Current Task Inventory

### Active (16 tasks)

| ID | Name | Freq | Model | Runs | Notes |
|----|------|------|-------|------|-------|
| 24 | Data Pipeline | 6x/day | 122B | 71 | Fact extraction, inference, relations |
| 25 | Memory Capture | hourly | 122B | 204 | Usually no-ops |
| 29 | Self-Improvement Loop | daily | 122B | 51 | Config experiments on branches |
| 30 | Intelligence Pipeline | daily | 122B | 6 | GitHub works, YouTube broken |
| 33 | Groundskeeper Loop | every-15min | 122B | 379 | Broken link fixing, massive compute |
| 34 | Groundskeeper Research | hourly | 122B | 161 | Often 0 candidates |
| 36 | Groundskeeper Survey | daily | 122B | 17 | Rebuilds groundskeeper queue |
| 37 | Nightly Skills Mgmt | daily | 122B | 8 | Trajectory-based skill creation |
| 38 | Nightly Reflection — Signals | daily | 122B | 6 | Signal detection from corrections |
| 39a | Nightly Reflection — Knowledge Analysis | daily | 122B | 10 | Derives memory updates |
| 39 | Nightly Reflection — Knowledge Write | daily | 122B | 9 | Writes to MEMORY.md |
| 40 | Nightly Reflection — Config | daily | 122B | 9 | Applies config changes |
| 45 | Trajectory Extraction | daily | 35B | 7 | Tool calls → JSONL |
| 46 | Trajectory Skill Mining | daily | 122B | 11 | Error patterns → skills |
| 47 | Dream Consolidation | daily | 122B | 4 | Usually no-ops (already consolidated) |
| 48 | Entity Resolution Sweep | daily | 122B | 3 | Finding 0 duplicates |

### Draft / Abandoned (3 tasks)

| ID | Name | Last Run | Notes |
|----|------|----------|-------|
| 6 | Morning Briefing | Mar 31 | Draft since March |
| 35 | Daily Backlog Triage | Mar 31 | Draft since March |
| 41 | Email + Calendar Monitoring | Apr 4 | Draft since creation |

### Dependency Chain (nightly pipeline)

```
38 (Signals)
 → 39a (Knowledge Analysis)
   → 39 (Knowledge Write)
     → 40 (Config)
       → 48 (Entity Resolution)
         → 47 (Dream Consolidation)
           → 37 (Skills Mgmt)
             → 6 (Morning Briefing) [draft]
```

## Redesign: 4-Tier Architecture

### Tier 1: Sense — Gather New Information

Purpose: Bring new data into the system from external and internal sources.

#### Intelligence Pipeline (keep, fix)
- **ID:** 30
- **Freq:** daily
- **Model:** 122B
- **Changes:**
  - Fix or replace YouTube scanning (63/66 RSS feeds dead)
  - Consider broader web monitoring: Hacker News, ArXiv, domain-specific blogs
  - Expand beyond 4 GitHub repos — add repos for tools Lloyd depends on (Claude SDK, vLLM, etc.)

#### Memory Capture (keep, reduce frequency)
- **ID:** 25
- **Freq:** 4x/day (was hourly)
- **Model:** 35B (was 122B)
- **Changes:**
  - Gate on new session count, not clock — skip if <2 new sessions
  - 35B is sufficient for extraction work
  - Saves ~20 no-op runs per day

#### Trajectory Extraction (keep as-is)
- **ID:** 45
- **Freq:** daily
- **Model:** 35B
- **Changes:** None needed. Working well.

#### Email + Calendar Monitoring (resurrect or kill)
- **ID:** 41
- **Freq:** 4x/day (if resurrected)
- **Model:** 35B
- **Decision needed:** Either commit to making this work or delete the task file. Draft limbo is waste.

### Tier 2: Learn — Actively Acquire Knowledge

Purpose: Proactively explore, research, and build understanding. This is the biggest gap in the current system.

#### Deep Dive Research (NEW)
- **Freq:** daily
- **Model:** 122B
- **Timeout:** 1800s
- **Description:** Pick 1 topic from the interest graph (robotics, reinforcement learning, sim-to-real, Claude SDK, vLLM, local LLM optimization, etc.). Perform web search, read primary sources, write a structured knowledge note to the vault. Topic selection should balance:
  - Recency (new developments in tracked areas)
  - Depth (shallow knowledge that deserves going deeper)
  - Relevance (connected to active projects)
- **Output:** One knowledge note per run in `knowledge/research/`
- **Key design:** Maintain a topic queue seeded from intelligence pipeline hits, user interests, and knowledge gaps identified by reflection. Don't just pick randomly.

#### Documentation Digester (NEW)
- **Freq:** daily
- **Model:** 122B
- **Timeout:** 900s
- **Description:** Monitor changelogs, release notes, and documentation updates for tools in the stack:
  - Claude Agent SDK / Claude Code
  - vLLM / local inference
  - Isaac Lab / Isaac GR00T
  - Obsidian ecosystem
  - Key Python packages
- **Output:** Structured diff notes — what changed, what matters to us, any action items
- **Key design:** Maintain a watched-sources list. Check each source's latest version against last-checked version. Only process actual changes.

#### Cross-Domain Synthesis (NEW)
- **Freq:** weekly (or 2x/week)
- **Model:** 122B
- **Timeout:** 1800s
- **Description:** Take 2-3 knowledge areas and find non-obvious connections:
  - "How does recent robotics sim-to-real work relate to our local LLM fine-tuning approach?"
  - "What patterns from the Claude SDK design could improve Lloyd's tool architecture?"
  - Bridge personal projects with professional interests
- **Output:** Synthesis notes in `knowledge/synthesis/`
- **Key design:** Not freeform brainstorming — grounded in actual vault knowledge. Read existing notes, find gaps or connections, write structured analysis.

### Tier 3: Reflect — Process What Was Learned

Purpose: Consolidate, correct, and improve based on accumulated experience. The nightly pipeline is the strongest part of the current system.

#### Nightly Reflection — Signals (keep)
- **ID:** 38, freq: daily, model: 122B

#### Nightly Reflection — Knowledge Analysis (keep)
- **ID:** 39a, freq: daily, model: 122B

#### Nightly Reflection — Knowledge Write (keep)
- **ID:** 39, freq: daily, model: 122B

#### Nightly Reflection — Config (keep)
- **ID:** 40, freq: daily, model: 122B

#### Trajectory Skill Mining (keep)
- **ID:** 46, freq: daily, model: 122B

#### Self-Improvement Loop (keep, fix)
- **ID:** 29, freq: daily, model: 122B
- **Fix:** Git commit bug — "Change applied to branch but not committed"

#### Nightly Skills Management (keep)
- **ID:** 37, freq: daily, model: 122B

### Tier 4: Maintain — Vault Hygiene (heavily reduced)

Purpose: Keep the knowledge base clean. Currently over-indexed here — scale back significantly.

#### Groundskeeper Survey (reduce)
- **ID:** 36
- **Freq:** weekly (was daily)
- **Model:** 35B (was 122B)
- **Rationale:** Queue doesn't change fast enough to justify daily rebuilds.

#### Groundskeeper Loop (reduce drastically)
- **ID:** 33
- **Freq:** daily (was every-15min)
- **Model:** 35B (was 122B)
- **Changes:**
  - One batch per day, cap at 20 items
  - Use 35B — link fixing doesn't need 122B reasoning
  - Saves ~95% of current groundskeeper compute

#### Entity Resolution Sweep (reduce)
- **ID:** 48
- **Freq:** weekly (was daily)
- **Model:** 35B (was 122B)
- **Rationale:** Last run found 0 true duplicates across 528 entities. Weekly is plenty.

#### Dream Consolidation (reduce)
- **ID:** 47
- **Freq:** weekly (was daily)
- **Model:** 122B
- **Changes:**
  - Gate on >50 new sessions since last run
  - Usually a no-op because nightly reflection already consolidates

### Kill / Merge List

| ID | Name | Action | Reason |
|----|------|--------|--------|
| 34 | Groundskeeper Research | **Kill** | Merge enrichment work into new Deep Dive Research task. Don't enrich thin vault profiles separately. |
| 24 | Data Pipeline (6x/day) | **Reduce to 2x/day** | Volume doesn't justify 6 runs. Consider merging with Memory Capture as a single "ingest" step. |
| 6 | Morning Briefing | **Decide: resurrect or delete** | Draft since March. Either build the skill properly or remove the task file. |
| 35 | Daily Backlog Triage | **Decide: resurrect or delete** | Same — draft limbo is noise in the task list. |

## Impact Summary

### Before
- ~50+ runs/day
- ~95% maintenance/extraction
- Zero active learning
- Groundskeeper dominates compute
- Memory Capture mostly no-ops

### After
- ~25 runs/day
- 3 new learning tasks filling freed compute
- Daily research deepening knowledge
- Documentation monitoring catching upstream changes
- Weekly cross-domain synthesis finding connections
- Maintenance scaled to actual need

### Compute Reallocation

| Category | Before (runs/day) | After (runs/day) |
|----------|-------------------|-------------------|
| Sense | ~30 (mostly no-ops) | ~6-8 |
| Learn | 0 | 2-3 |
| Reflect | ~8 | ~8 |
| Maintain | ~100+ | ~2 |
| **Total** | **~140+** | **~18-21** |

The freed 122B compute goes to the three new Tier 2 learning tasks, which are the highest-value use of local inference.

## Implementation Order

1. **Quick wins (day 1):**
   - Reduce Groundskeeper Loop to daily, switch to 35B
   - Reduce Memory Capture to 4x/day, switch to 35B
   - Reduce Data Pipeline to 2x/day
   - Kill Groundskeeper Research (#34)
   - Decide on draft tasks (#6, #35, #41) — resurrect or delete

2. **New learning tasks (day 2-3):**
   - Write Deep Dive Research skill + task file
   - Write Documentation Digester skill + task file
   - Fix Intelligence Pipeline YouTube scanning

3. **Weekly tasks (day 4):**
   - Switch Groundskeeper Survey, Entity Resolution, Dream Consolidation to weekly
   - Write Cross-Domain Synthesis skill + task file

4. **Validation (week 2):**
   - Monitor run quality for new learning tasks
   - Verify reduced maintenance tasks aren't creating drift
   - Check that knowledge output is actually useful in conversations

## Open Questions

- What specific topics/domains should seed the Deep Dive Research queue?
- Should Documentation Digester track any sources beyond the ones listed?
- Is weekly the right cadence for Cross-Domain Synthesis, or should it be triggered by accumulated knowledge volume?
- Should Morning Briefing be resurrected as a "daily digest" that summarizes what learning tasks found?
- Should we add a periodic benchmarking task that measures Lloyd's actual capability improvement over time?
