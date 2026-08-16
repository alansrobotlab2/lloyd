---
segment: agents
generated: 2026-08-15 02:00 PST
data_range: 2026-08-13 to 2026-08-15
---

# Signal Report — 2026-08-15

## Queued Signals (met threshold)

### Explicit (act on first occurrence)

| ID | Date | Type | Category | Description | Source |
|----|------|------|----------|-------------|--------|
| 1  | 2026-06-20 | correction | task-scoping | Stop turning every task into a meta-task about the task — just do the thing, don't overcomplicate | corrections.md |
| 2  | 2026-06-20 | correction | skill-authoring | Don't create NEW skills for every task — update existing ones | corrections.md |
| 3  | 2026-06-18 | correction | note-formatting | Stop adding 'lloyd:' prefix to notes | corrections.md |
| 4  | 2026-06-19 | correction | task-scoping | I don't want a full audit. I want you to just look at the last 3 days | corrections.md |
| 5  | 2026-06-19 | correction | task-scoping | Stop over-engineering the skill-consolidation pipeline | corrections.md |
| 6  | 2026-06-19 | correction | task-scoping | Don't add your own commentary to the report | corrections.md |
| 7  | 2026-06-19 | correction | knowledge-graph | don't create entities about the session itself | corrections.md |
| 8  | 2026-06-19 | correction | knowledge-graph | Session entity pollution — 26 entities, 349+ neighbors on generic `session` hub | corrections.md |
| 9  | 2026-06-19 | correction | tool-use | don't over-apply the which → command -v fix | corrections.md |
| 10 | 2026-06-19 | correction | truthfulness | "you're making up what I asked for" — agent misinterprets user request | corrections.md |
| 11 | 2026-06-20 | correction | memory | Meta-correction: "stop adding lloyd prefix" correction NOT in memory.md | corrections.md |
| 12 | 2026-06-21 | correction | truthfulness | "I never said that" — agent misquotes user | corrections.md |
| 13 | 2026-06-19 | correction | task-scoping | don't create entities about the session itself | corrections.md |

### Inferred (met 2+ threshold)

| ID | Date | Type | Category | Description | Frequency | Source |
|----|------|------|----------|-------------|-----------|--------|
| 1  | 2026-06-19/20 | pattern | task-scoping | Over-engineering tasks: turning simple requests into multi-phase pipelines with unnecessary meta-work | 4x | corrections.md + daily notes |
| 2  | 2026-06-19/20/21 | pattern | truthfulness | Insisting on own interpretation when corrected — agent doubles down on misreading instead of accepting correction | 3x | corrections.md + daily notes |
| 3  | 2026-06-19/20 | pattern | knowledge-graph | Creating non-graph entities: creating entity facts about sessions and internal processes instead of user-facing topics | 2x | corrections.md + daily notes |
| 4  | 2026-06-19/20 | pattern | persistence | Claiming fixes applied but not persisted: agent reports fix applied, but correction not actually written to memory.md | 2x | corrections.md + daily notes |
| 5  | 2026-06-19/21 | pattern | tool-use | Tool failures on non-existent `~/obsidian/sessions/` directory — agent attempts to read from sessions dir that doesn't exist | 2x | daily notes + tool-patterns |

## Pending Signals (below threshold)

- "Stop adding commentary to reports" — 1 occurrence, monitor for pattern
- "Don't create entities about the session itself" — 1 occurrence, monitor for pattern

## Tool Failure Patterns

- **Tool:** `cat ~/obsidian/sessions/` — **Error type:** FILE_NOT_FOUND — **Occurrences:** 2 — **Recommendation:** Guard daily note reads with existence check before cat; standardize to `~/obsidian/memory/learnings/` path
- **Tool:** `bash` (vault audit) — **Error type:** timeout during large directory traversal — **Occurrences:** 1 — **Recommendation:** Use background tasks for large find operations; limit scope with maxdepth
- **Tool:** `git commit` — **Error type:** working tree clean, no changes to commit — **Occurrences:** 1 — **Recommendation:** Check git status before attempting commit; skip if clean

## Positive Patterns to Reinforce

- **Pattern:** `which` → `command -v` POSIX portability fix — **Evidence:** 1 praise entry in corrections.md (2026-06-20) — **Action:** Encode as skill or update existing bash-error-handling skill with POSIX portability guard
- **Pattern:** Browser-based extraction for SPA product pages — **Evidence:** 1 success entry in tool-patterns-latest.md — **Action:** Update browser-content-extraction skill with SPA fallback pattern
- **Pattern:** Skill consolidation work acknowledged as valuable — **Evidence:** 1 success entry in tool-patterns-latest.md — **Action:** Continue skill-consolidation pipeline; document as positive pattern
