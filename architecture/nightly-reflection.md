---
segment: architecture
relations:
  related-to: []
tags: [architecture]
type: reference

---
















# Nightly Reflection Architecture

Recursive self-improvement for Lloyd,inspired by [Karpathy's auto-research pattern](../../knowledge/ai/karpathy-auto-research.md). The core loop -- modify,evaluate,keep/discard,repeat -- runs as the final sequence in the nightly automation pipeline.

## Schedule

Five sequential jobs in the reflection block:

| Time (PST) | Job | Skill | Purpose |
|------------|-----|-------|---------|
| 4:00 AM | Signal Processing | [`nightly-reflection-signals/SKILL.md`](../../skills/nightly-reflection-signals/SKILL.md) | Signal detection & classification |
| 4:20 AM | Knowledge Consolidation | [`nightly-reflection-knowledge/SKILL.md`](../../skills/nightly-reflection-knowledge/SKILL.md) | Mental models,MEMORY.md,vault propagation,pattern analysis |
| 4:40 AM | Prompt Audit | [`nightly-prompt-audit/SKILL.md`](../../skills/nightly-prompt-audit/SKILL.md) | System prompt quality audit & drift detection |
| 4:55 AM | Behavior Test | [`nightly-behavior-test/SKILL.md`](../../skills/nightly-behavior-test/SKILL.md) | Synthetic behavior tests & regression suite |
| 5:15 AM | Config Application | [`nightly-reflection-config/SKILL.md`](../../skills/nightly-reflection-config/SKILL.md) | Apply fixes from signals + audit + test failures,git commits,summary |

- **Agent:** `memory` (isolated session per job)
- **Model:** Claude Opus 4.6
- **Budget:** <$25 combined per night (reflection jobs only)
- **Sequence:** Runs after [[morning-briefing|reflection-synthesis]] (1:30am),[[nightly-vault-maintenance|reflection-vault]] (2am),and [[nightly-skills-management|reflection-skills]] (3am)
- **Handoff:** Signal report at `~/lloyd/_pipeline/reflection/signals-latest.md`,audit issues at `~/lloyd/_pipeline/reflection/prompt-audit-latest.md`,test failures at `~/lloyd/_pipeline/reflection/test-results-latest.md`

### Why 5 Jobs?

The original single-session reflection ran 7 phases in one agent turn. By phase 5,the agent was juggling corrections data,mental model updates,MEMORY.md diffs,and config change decisions simultaneously. Splitting into focused jobs gives each agent a clean context window dedicated to one job.

Cross-job dependencies are file-based: the signal report,audit issues,and test failures are written by earlier jobs and consumed by `reflection-config` which runs last.

## Design Principle

Karpathy's system edits `train.py`,trains for 5 minutes,measures validation loss,keeps improvements. Lloyd's equivalent:

| Karpathy | Lloyd |
|----------|-------|
| `program.md` (natural language instructions) | SOUL.md,AGENTS.md,TOOLS.md |
| `train.py` (code being optimized) | `.openclaw` configs (agent defs,tool allowlists,extensions) |
| Validation loss (evaluation metric) | Correction rate,task success,reinforcement signals |
| 5-minute training window | 1 day of live interaction |
| Git commits per experiment | Git snapshots per nightly run |

## Tiered Signal Thresholds

Signals are classified into two tiers with different action thresholds:

| Tier | Threshold | What qualifies |
|------|-----------|---------------|
| **Explicit** | **1 occurrence** → act immediately | Direct corrections ("don't do X"),stated preferences,explicit praise tied to specific behavior |
| **Inferred** | **2+ occurrences** → act | Implicit frustration,ambiguous signals,inferred patterns from daily notes |

Explicit corrections are direct instructions — the user shouldn't have to repeat themselves. Inferred patterns need confirmation before acting.

## Two Improvement Surfaces

Self-improvement touches two separate repos with different risk profiles:

### Surface 1: Obsidian Vault (`~/obsidian`)

The "soul" -- personality,behavior,knowledge,skills.

| File | What It Controls | Risk |
|------|-----------------|------|
| `agents/lloyd/SOUL.md` | Personality,tone,interaction style | Low |
| `agents/lloyd/AGENTS.md` | Task routing,delegation rules,memory protocol | Medium |
| `agents/lloyd/TOOLS.md` | Response format,on-demand skill references | Low |
| `agents/lloyd/MEMORY.md` | Long-term factual memory | Low |
| `skills/*/SKILL.md` | Procedural knowledge | Low |
| `memory/mental-models.md` | Understanding of Alan's patterns | Low |

### Surface 2: `.openclaw` Config (`~/.openclaw`)

The "wiring" -- runtime configuration that affects tool access,model routing,and service behavior.

| File | What It Controls | Risk |
|------|-----------------|------|
| `openclaw.json` | Agent config,tool allowlists,channel bindings,model routing | High |
| `extensions/agent-orchestrator/agents/*.ts` | Subagent prompts,thinking/effort,maxTurns | Medium |
| `extensions/voice-tools/index.ts` | Voice pipeline hooks | Medium |
| `extensions/mission-control/index.ts` | MC backend behavior | Medium |

## Job 1: Signal Processing

**Skill:** `nightly-reflection-signals/SKILL.md`

1. **Pre-Flight Snapshot