---
tags:
  - lloyd
  - architecture
type: reference
segment: projects
---

# Nightly Reflection Architecture

Recursive self-improvement for Lloyd, inspired by [Karpathy's auto-research pattern](../../knowledge/ai/karpathy-auto-research.md). The core loop -- modify, evaluate, keep/discard, repeat -- runs as the final job in the nightly automation sequence.

## Schedule

- **Time:** 4:00 AM PST (daily)
- **Agent:** `memory` (isolated session)
- **Model:** Claude Opus 4.6
- **Budget:** <$20 per run
- **Sequence:** Runs after [[nightly-vault-maintenance]] (2am) and [[nightly-skills-management]] (3am)
- **Skill file:** [`nightly-reflection/SKILL.md`](../../skills/nightly-reflection/SKILL.md)

## Design Principle

Karpathy's system edits `train.py`, trains for 5 minutes, measures validation loss, keeps improvements. Lloyd's equivalent:

| Karpathy | Lloyd |
|----------|-------|
| `program.md` (natural language instructions) | SOUL.md, AGENTS.md, TOOLS.md |
| `train.py` (code being optimized) | `.openclaw` configs (agent defs, tool allowlists, extensions) |
| Validation loss (evaluation metric) | Correction rate, task success, reinforcement signals |
| 5-minute training window | 1 day of live interaction |
| Git commits per experiment | Git snapshots per nightly run |

## Two Improvement Surfaces

Self-improvement touches two separate repos with different risk profiles:

### Surface 1: Obsidian Vault (`~/obsidian`)

The "soul" -- personality, behavior, knowledge, skills.

| File | What It Controls | Risk |
|------|-----------------|------|
| `agents/lloyd/SOUL.md` | Personality, tone, interaction style | Low |
| `agents/lloyd/AGENTS.md` | Task routing, delegation rules, memory protocol | Medium |
| `agents/lloyd/TOOLS.md` | Response format, on-demand skill references | Low |
| `agents/lloyd/MEMORY.md` | Long-term factual memory | Low |
| `skills/*/SKILL.md` | Procedural knowledge | Low |
| `memory/mental-models.md` | Understanding of Alan's patterns | Low |

### Surface 2: `.openclaw` Config (`~/.openclaw`)

The "wiring" -- runtime configuration that affects tool access, model routing, and service behavior.

| File | What It Controls | Risk |
|------|-----------------|------|
| `openclaw.json` | Agent config, tool allowlists, channel bindings, model routing | High |
| `extensions/agent-orchestrator/agents/*.ts` | Subagent prompts, thinking/effort, maxTurns | Medium |
| `extensions/voice-tools/index.ts` | Voice pipeline hooks | Medium |
| `extensions/mission-control/index.ts` | MC backend behavior | Medium |

## Nightly Run Flow

Seven phases, executed in order by the SKILL.md procedure.

### Phase 1: Pre-Flight Snapshots

Take baseline snapshots of both repos before any modifications.

```bash
# Vault snapshot
cd ~/obsidian && git add -A && git diff --cached --quiet || git commit -m "nightly: pre-snapshot $(date +%Y-%m-%d)"

# .openclaw snapshot
cd ~/.openclaw && git add -A && git diff --cached --quiet || git commit -m "nightly: pre-snapshot $(date +%Y-%m-%d)"
```

These are the rollback points if anything goes wrong.

### Phase 2: Signal Collection

Gather all behavioral signals from the past day:

- **Corrections log** (`memory/corrections.md`) -- positive and negative signals
- **Daily notes** (last 3 days) -- decisions, frustrations, praise, patterns
- **Task outcomes** -- orchestrator success/failure rates from backlog activity notes
- **Session metadata** -- tool call patterns, delegation frequency, error rates

### Phase 3: Pattern Analysis

Analyze collected signals for actionable patterns:

- **Recurring negatives (3+ occurrences)** -- strong signal, propose fix
- **Recurring positives (3+ occurrences)** -- codify to protect the pattern
- **New single signals** -- log for accumulation, don't act yet
- **Cross-surface patterns** -- e.g., delegation failure might need both AGENTS.md rule + orchestrator config change

Classify each proposed change by target surface:
- `vault:soul` -- SOUL.md personality/tone changes
- `vault:agents` -- AGENTS.md behavioral rules
- `vault:tools` -- TOOLS.md format/skill references
- `vault:skills` -- new or modified skills
- `config:agent` -- openclaw.json agent/tool config
- `config:orchestrator` -- subagent definitions
- `config:extension` -- extension behavior

### Phase 4: Existing Tasks (Preserved)

The core nightly reflection tasks from SKILL.md:

1. **Self-Improvement Review** -- process corrections, apply SOUL.md/AGENTS.md fixes
2. **Mental Models** -- update `memory/mental-models.md` from recent daily notes
3. **MEMORY.md Consolidation** -- distill daily notes into agents/lloyd/MEMORY.md, keep under 200 lines

### Phase 5: Config Improvements

For `.openclaw` changes identified in Phase 3:

1. **Propose** -- generate specific edits with rationale
2. **Validate** -- sanity check: does the JSON still parse? Are tool names valid? Are model aliases correct?
3. **Apply** -- write changes to the config files
4. **Log** -- record every change in `memory/learnings/YYYY-MM-DD.md` with:
   - File changed
   - Exact diff (before/after)
   - Rationale (which signal triggered it)
   - Surface classification
   - Risk level

### Phase 6: Post-Improvement Snapshots

Commit all changes across both repos:

```bash
# Vault improvements
cd ~/obsidian && git add -A && git diff --cached --quiet || git commit -m "nightly improvements: $(date +%Y-%m-%d)"

# .openclaw improvements
cd ~/.openclaw && git add -A && git diff --cached --quiet || git commit -m "nightly improvements: $(date +%Y-%m-%d)"
```

### Phase 7: Summary Report

Write a summary to `memory/learnings/YYYY-MM-DD.md` covering:
- Signals processed (count by type)
- Changes made (by surface)
- Changes deferred (insufficient signal strength)
- Metrics snapshot (correction rate, task success rate for the day)

Deliver summary via announce so Alan sees it in the morning.

## Evaluation Metrics

Track these over time to measure whether self-improvement is actually improving things:

| Metric | Source | Direction |
|--------|--------|-----------|
| Negative correction rate | `corrections.md` entries per day | Down better |
| Positive reinforcement rate | `corrections.md` entries per day | Up better |
| Orchestrator task success rate | Backlog task outcomes | Up better |
| Tool calls per task | Session logs | Down better (efficiency) |
| Rollbacks triggered | Git history | Down better |

Initial phase: collect baselines for 2 weeks before using metrics to gate auto-apply.

## Safety Guardrails

1. **Git snapshots bracket every run** -- pre and post, both repos
2. **One small change at a time** -- no wholesale rewrites
3. **3-occurrence threshold** -- signals must recur before triggering changes
4. **Additive preference** -- prefer additions/clarifications over removals
5. **JSON validation** -- any `.openclaw` config change must parse cleanly before commit
6. **Budget cap** -- <$20 per nightly run
7. **Learnings log** -- full changelog for Alan's visibility
8. **No gateway restarts** -- config changes take effect on next manual restart; Alan controls when runtime changes go live

## Rollback

If a change causes issues:
```bash
# Vault rollback
cd ~/obsidian && git log --oneline -5  # find pre-snapshot commit
cd ~/obsidian && git revert <commit>

# .openclaw rollback
cd ~/.openclaw && git log --oneline -5
cd ~/.openclaw && git revert <commit>
```

Gateway restart needed after `.openclaw` rollback.

## Nightly Automation Sequence

| Time (PST) | Job | Purpose | Skill |
|------------|-----|---------|-------|
| 2:00 AM | `daily-vault-maintenance` | Tag hygiene, frontmatter validation, structure review | [SKILL.md](../../skills/nightly-vault-maintenance/SKILL.md) |
| 3:00 AM | `nightly-skills-management` | Skills extraction, evaluation, dedup | [SKILL.md](../../skills/nightly-skills-management/SKILL.md) |
| 4:00 AM | `nightly-reflection` | Self-improvement, mental models, MEMORY.md consolidation | [SKILL.md](../../skills/nightly-reflection/SKILL.md) |
| Every 15m | `periodic-memory-capture` | Extract transcripts to daily notes | [SKILL.md](../../skills/periodic-memory-capture/SKILL.md) |

## Future Enhancements

- **Auto-apply gating** -- once metrics baseline is established, low-risk vault changes auto-merge; `.openclaw` changes always queue for review
- **A/B testing** -- run variant configs for N sessions, compare metrics
- **Replay evaluation** -- re-run past interactions against modified config, score difference
- **Multi-agent knowledge sharing** -- orchestrator runs contribute to shared findings log
- **Metric dashboard** -- MC page showing improvement trends over time

## Related Docs

- [[memory]] -- Memory System Architecture (Tier 2 nightly reflection)
- [[nightly-skills-management]] -- Skills Management (procedural self-improvement)
- [[nightly-vault-maintenance]] -- Vault Maintenance (structural hygiene)
- [[agents]] -- Agent System
- [Karpathy Auto-Research](../../knowledge/ai/karpathy-auto-research.md) -- Source inspiration
- [Nightly Reflection Skill](../../skills/nightly-reflection/SKILL.md) -- Current implementation
