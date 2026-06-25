---
segment: agents
generated: 2026-06-24 17:30 PST
data_range: 2026-06-21 to 2026-06-24
---

# Signal Report — 2026-06-24

## Queued Signals (met threshold)

### Explicit (act on first occurrence)

| ID | Date | Type | Category | Description | Source |
|----|------|------|----------|-------------|--------|
| 1 | 2026-06-22 | correction | skill-discipline | Stopped at 'no results' instead of trying a broader search for skill-maintenance | daily note |
| 2 | 2026-06-21 | correction | knowledge-graph | Rejected fact_add for non-discoverable items (session, config update, tool fix) | daily note |
| 3 | 2026-06-21 | correction | knowledge-graph | Must use knowledge_capture pipeline, never fact_add directly | daily note |
| 4 | 2026-05-08 | correction | tool-use | Called deferred tools without ToolSearch schema loading first | corrections.md |
| 5 | 2026-05-08 | correction | tool-use | Port mismatch — assumed docs correct without verifying actual port | corrections.md |
| 6 | 2026-03-31 | correction | skill-discipline | Skipped skill check before gateway restart | corrections.md |
| 7 | 2026-03-29 | correction | skill-discipline | Skipped skills_search before tool calls and dispatches | corrections.md |
| 8 | 2026-03-29 | correction | pipeline-dispatch | Raw subagent spawn instead of pipeline-dispatch | corrections.md |
| 9 | 2026-03-29 | correction | scope-preservation | Merged two user links into one task | corrections.md |

### Inferred (met 2+ threshold)

| ID | Date | Type | Category | Description | Frequency | Source |
|----|------|------|----------|-------------|-----------|--------|
| 1 | 2026-06-17–23 | correction | scope-creep | Repeated over-engineering (creating skills for simple tasks, expanding scope beyond request) | 6+ | daily notes |
| 2 | 2026-06-18–19 | correction | communication | Sycophantic responses ('you're absolutely right', 'excellent point') instead of direct disagreement | 5+ | daily notes |
| 3 | 2026-06-17–24 | correction | communication | Over-verbose responses (600–800 word paragraphs where 3–4 sentences suffice) | 10+ | daily notes |
| 4 | 2026-06-19–24 | correction | tool-use | Tool call failures not caught (e.g., grep errors, write failures, empty results) | 6+ | daily notes |
| 5 | 2026-06-17–24 | praise | delegation | Correctly delegated complex tasks to orchestrator/worker model | 8+ | daily notes |
| 6 | 2026-06-18–24 | praise | communication | Direct concise responses well-received when they happen | 7+ | daily notes |
| 7 | 2026-06-17–19 | workflow | delegation | Research → review pipeline works well when followed | 4+ | daily notes |
| 8 | 2026-06-22–23 | correction | knowledge-graph | Repeated attempts to create metadata entities in KG | 3+ | daily notes |

## Key Patterns This Cycle

**Most Active Negative:** Scope creep and over-engineering — consistently building elaborate systems (skills, pipelines, multi-file operations) for simple one-shot tasks. The #1 trust erosion pattern across the entire recent period.

**Most Active Positive:** When Lloyd responds directly and concisely, or correctly delegates to the orchestrator/worker model, Alan builds on the output without pushback. The positive reinforcement is consistent: "exactly right" appears 13+ times when behavior is correct.

**Recurring Theme:** Knowledge graph discipline. Multiple corrections about not putting session metadata, dates, or transient facts into the graph. Only discoverable concepts belong — everything else goes to daily notes.

**Session Data Note:** Enriched session files exist at ~/lloyd/sessions/ but extraction logs are minimal — no structured session-level extraction data (e.g., tool call success/failure rates per session) available for deeper analysis. The session files contain raw tool result JSONs but no consolidated extraction reports.
