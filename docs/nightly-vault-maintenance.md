---
tags:
  - lloyd
  - architecture
type: reference
segment: projects
---

# Nightly Vault Maintenance Architecture

Automated structural hygiene for the Obsidian vault. Keeps tags consistent, frontmatter valid, and doc organization clean so that search, retrieval, and cross-referencing remain reliable.

## Schedule

- **Time:** 2:00 AM PST (daily)
- **Agent:** `memory` (isolated session)
- **Model:** Claude Opus 4.6
- **Budget:** <$20 per run
- **Sequence:** Runs first in the nightly sequence, before [[nightly-skills-management]] (3am) and [[nightly-reflection]] (4am)
- **Skill file:** [`nightly-vault-maintenance/SKILL.md`](../../skills/nightly-vault-maintenance/SKILL.md)

## Purpose

The Obsidian vault is Lloyd's long-term knowledge store. As content accumulates from daily notes, skill extraction, architecture docs, and manual edits, structural drift is inevitable: tags proliferate, frontmatter becomes inconsistent, docs end up in wrong directories, and orphan files lose discoverability. Vault maintenance catches this drift nightly before it compounds.

## Four Tasks

### Task 1: Tag Hygiene

Normalize the vault's tag namespace.

1. Run `vault_overview(detail="tags")` to get all tags with frequencies
2. Identify and fix:
   - **Duplicate/overlapping tags** (e.g. "llm" vs "large-language-model", "ai" vs "artificial-intelligence")
   - **Too-specific tags** used on only 1-2 docs that could be merged into broader tags
   - **Inconsistent naming** (hyphens vs underscores, plural vs singular)
3. For each merge/rename: pick canonical tag name, update all affected docs
4. Log changes to `memory/vault-maintenance/YYYY-MM-DD.md`

### Task 2: Recently Touched Docs -- Tag Review

Ensure recently modified docs have accurate, complete tags.

1. Query git log for docs modified in the last 7 days
2. For each modified doc:
   - Read content and assess whether tags are accurate and complete
   - Identify new tags that should be added based on content
   - Check if related docs should also get those tags (use `mem_search` to find related docs)
   - Update tags on all affected docs

### Task 3: Frontmatter Validation

Enforce frontmatter standards across the vault.

1. Scan vault segments for docs with missing or incomplete frontmatter
2. Required fields: `tags` (array), `segment` (matches directory)
3. Recommended: `summary` (1-line description)
4. Fix missing frontmatter -- add tags based on content and location
5. Flag any files in wrong directories for their segment

### Task 4: Structure Review

Identify structural issues that hurt discoverability and maintainability.

1. Look for large docs (>200 lines) that could be split into focused sub-docs
2. Look for orphan docs (no tags, no inbound links from other docs)
3. Suggest or execute reorganization where clear wins exist
4. Be conservative -- only move/split files when the benefit is obvious

## Guardrails

- **Git snapshots bracket every run** -- pre-run and post-run commits for rollback safety
- **One retry max per task** -- if a task fails, log the failure and move to the next task
- **Budget cap** -- <$20 per run
- **Hands off agent files** -- DO NOT modify `agents/lloyd/SOUL.md`, `AGENTS.md`, or `MEMORY.md` (those are [[nightly-reflection]] territory)
- **No deletions** -- reorganize, merge, or flag for review; never delete files
- **Full changelog** -- log all changes via `mem_write(path="memory/vault-maintenance/YYYY-MM-DD.md")`

## Output Artifacts

- **Maintenance log**: `~/obsidian/memory/vault-maintenance/YYYY-MM-DD.md` -- record of all changes made
- **Git snapshots**: Pre-run and post-run commits for rollback safety

## Nightly Automation Sequence

| Time (PST) | Job | Purpose | Skill |
|------------|-----|---------|-------|
| 2:00 AM | `daily-vault-maintenance` | Tag hygiene, frontmatter validation, structure review | [SKILL.md](../../skills/nightly-vault-maintenance/SKILL.md) |
| 3:00 AM | `nightly-skills-management` | Skills extraction, evaluation, dedup | [SKILL.md](../../skills/nightly-skills-management/SKILL.md) |
| 4:00 AM | `nightly-reflection` | Self-improvement, mental models, MEMORY.md consolidation | [SKILL.md](../../skills/nightly-reflection/SKILL.md) |
| Every 15m | `periodic-memory-capture` | Extract transcripts to daily notes | [SKILL.md](../../skills/periodic-memory-capture/SKILL.md) |

Vault maintenance runs first because clean tags and frontmatter improve the accuracy of `mem_search` and `skills_search` calls made by the later jobs.

## Related Docs

- [[nightly-reflection]] -- Nightly Reflection (self-improvement, runs at 4am)
- [[nightly-skills-management]] -- Skills Management (extraction pipeline, runs at 3am)
- [[memory]] -- Memory System Architecture
- [[infrastructure]] -- Infrastructure (runtime environment, systemd services)
- [Nightly Vault Maintenance Skill](../../skills/nightly-vault-maintenance/SKILL.md) -- Current implementation
