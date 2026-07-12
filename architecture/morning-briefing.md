---
segment: architecture
tags: [architecture,lloyd]
relations:
  related-to:
  - autonomy/38-nightly-reflection-signals.md
  - architecture/nightly-skills-management.md
  - architecture/skills.md
  - architecture/tools.md
  - architecture/memory.md
  - architecture/autonomy-system.md
  - architecture/evaluation-engine.md
  - architecture/index.md
  - architecture/infrastructure.md
tags: [architecture]
type: reference

---














# Morning Briefing Architecture

Daily synthesis of overnight pipeline results and actionable morning context. Delivered to Alan via announce mode when he starts his day.

## Schedule

- **Time:** 7:00 AM PST (daily)
- **Agent:** `memory` (isolated session)
- **Model:** Claude Sonnet 4.6
- **Budget:** <$5 per run
- **Delivery:** Announce mode
- **Sequence:** Runs after all nightly pipeline jobs complete (last nightly job ends ~5:30 AM)
- **Skill file:** [`morning-briefing/SKILL.md`](../../skills/morning-briefing/SKILL.md)

## Purpose

The nightly pipeline generates several reports: signal processing,behavior test results,prompt audit scores,skill management reports,and vault maintenance logs. Without synthesis,Alan would need to read 5+ files to understand what happened overnight. The morning briefing consolidates this into a single,concise summary.

## Data Sources

| Source | Path | Purpose |
|--------|------|---------|
| Reflection changelog | `memory/learnings/YYYY-MM-DD.md` | Signals detected and changes applied |
| Signal report | `memory/reflection/signals-latest.md` | Signal classifications and counts |
| Test results | `memory/reflection/test-results-latest.md` | Behavior test pass/fail |
| Prompt audit | `memory/reflection/prompt-audit-latest.md` | Prompt quality scores |
| Day synthesis | `memory/reflection/day-synthesis-latest.md` | Yesterday's open threads |
| Backlog | Backlog API | Current item status |

## Output

Single file: `memory/reflection/morning-briefing-latest.md`

Sections:
1. **Overnight Pipeline** -- 3-5 bullet point headlines
2. **Today's Calendar** -- events (when integration available)
3. **Email** -- flagged items (when integration available)
4. **Backlog -- Attention Needed** -- top 3-5 actionable items
5. **Open Threads** -- from yesterday's synthesis
6. **Today's Focus** -- 1-2 suggested priorities

Target length: under 50 lines.

## Nightly Automation Sequence

| Time (PST) | Job | Purpose | Skill |
|------------|-----|---------|-------|
| 1:30 AM | `reflection-synthesis` | Extract key decisions,open questions,corrections quality | [SKILL.md](../../skills/day-end-synthesis/SKILL.md) |
| 2:00 AM | `reflection-vault` | Tag hygiene,frontmatter validation,structure review | [SKILL.md](../../skills/nightly-vault-maintenance/SKILL.md) |
| 3:00 AM | `reflection-skills` | Skills extraction,evaluation,dedup,effectiveness | [SKILL.md](../../skills/nightly-skills-management/SKILL.md) |
| 4:00 AM | `reflection-signals` | Signal detection & classification | [SKILL.md](../../skills/nightly-reflection-signals/SKILL.md) |
| 4:20 AM | `reflection-knowledge` | Mental models,MEMORY.md,vault propagation,pattern analysis | [SKILL.md](../../skills/nightly-reflection-knowledge/SKILL.md) |
| 4:40 AM | `reflection-audit` | System prompt quality audit & drift detection | [SKILL.md](../../skills/nightly-prompt-audit/SKILL.md) |
| 4:55 AM | `reflection-test` | Synthetic behavior tests & regression suite | [SKILL.md](../../skills/nightly-behavior-test/SKILL.md) |
| 5:15 AM | `reflection-config` | Apply fixes from signals + audit + test failures,git commits,summary | [SKILL.md](../../skills/nightly-reflection-config/SKILL.md) |
| **7:00 AM** | **`morning-briefing`** | **Synthesize overnight results,backlog,calendar** | [SKILL.md](../../skills/morning-briefing/SKILL.md) |
| Every 15m | `periodic-memory-capture` | Extract transcripts to daily notes | [SKILL.md](../../skills/periodic-memory-capture-lloyd/SKILL.md) |
| Sunday 1:00 AM | `reflection-backlog` | Backlog staleness,blocked items,deprioritization | [SKILL.md](../../skills/weekly-backlog-hygiene/SKILL.md) |

## Related Docs

- [[nightly-reflection]] -- Nightly Reflection (self-improvement pipeline)
- [[nightly-vault-maintenance]] -- Vault Maintenance
- [[nightly-skills-management]] -- Skills Management
- [[memory]] -- Memory System Architecture

