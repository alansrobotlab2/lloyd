---
segment: agents
generated: 2026-06-15 02:00 PST
data_range: 2026-06-10 to 2026-06-14
---

# Signal Report — 2026-06-15

## Queued Signals (met threshold)

### Explicit (act on first occurrence)

| ID | Date | Type | Category | Description | Source |
|----|------|------|----------|-------------|--------|
| 1  | 2026-06-10 | correction | tool-use | `python` command not found — must always use `python3` in bash heredocs and inline scripts | corrections.md |
| 2  | 2026-06-10 | correction | tool-use | `browser_navigate` opens new session/tab each time, closing previous one — multi-step browser workflows lose state | corrections.md |
| 3  | 2026-06-10 | correction | tool-use | `http_fetch` on JS-rendered SPAs captures initial blank page before client-side JS executes | corrections.md |
| 4  | 2026-06-10 | correction | tool-use | `http_fetch` to localhost:8091 (local LLM) times out on large inputs (>50KB) | corrections.md |
| 5  | 2026-06-10 | correction | tool-use | `http_fetch` on Amazon blocked by anti-bot protection | corrections.md |
| 6  | 2026-06-10 | correction | tool-use | `vault_search` default timeout (300s) insufficient for larger repos | corrections.md |
| 7  | 2026-06-10 | correction | delegation | Merging multiple distinct user-provided links into single task — each link must get its own subagent/task | corrections.md |
| 8  | 2026-06-10 | correction | skill-use | Using `sessions_spawn` instead of skill protocol's specified `pipeline-dispatch` | corrections.md |
| 9  | 2026-06-10 | correction | skill-use | Failing to read SKILL.md before acting — skill resolution is mandatory prerequisite | corrections.md |
| 10 | 2026-06-11 | correction | tool-use | Sending raw HTML (>50KB) to local LLM via `http_fetch` causes timeouts — extract text first via browser | corrections.md |
| 11 | 2026-06-11 | correction | tool-use | `browser_tabs` not used to manage tabs within single browser instance — sequential navigate→close cycles lose state | corrections.md |
| 12 | 2026-06-11 | correction | knowledge | `fact_add` calls during nightly reflection create useless session entities — pollutes knowledge graph | corrections.md |
| 13 | 2026-06-12 | correction | tool-use | Bash variable name typos (`RESULT` vs `result`, `URL` vs `url`) cause silent failures — use consistent UPPERCASE | corrections.md |
| 14 | 2026-06-12 | correction | behavior | Overcomplicating responses when user wants direct, practical answers | corrections.md |
| 15 | 2026-06-12 | correction | behavior | Stopping mid-batch to ask "should I continue?" — commit to completing announced operations without interruption | corrections.md |
| 16 | 2026-06-14 | correction | delegation | Merging YouTube transcript tasks — each video link must get separate parallel session, not merged | daily note |

### Inferred (met 2+ threshold)

| ID | Date | Type | Category | Description | Frequency | Source |
|----|------|------|----------|-------------|-----------|--------|
| 1  | 2026-06-10/11 | pattern | tool-use | Repeated `http_fetch` failures on content extraction — browser pipeline is more reliable | 6+ | corrections.md |
| 2  | 2026-06-10/12 | pattern | tool-use | Browser session state loss — sequential `browser_navigate` calls without `browser_tabs` management | 3+ | corrections.md |
| 3  | 2026-06-11/12 | pattern | tool-use | Tool path mismatches (e.g., sessions migration) causing cascading failures | 2+ | corrections.md |
| 4  | 2026-06-10/13 | pattern | delegation | Scope reduction: combining multiple user inputs into fewer tasks than provided | 2+ | corrections.md |
| 5  | 2026-06-12/14 | pattern | knowledge | Knowledge graph pollution: session entities, dates-as-entities, pipeline artifacts | 2+ | corrections.md |

## Pending Signals (below threshold)

- **Overthinking responses** — appears in multiple contexts but not yet at 2+ distinct occurrences with clear behavioral prescription. Monitor.
- **Refusal to use browser tools for web scraping** — single explicit occurrence in daily notes, not yet patterned.
- **Failing to recognize tool path mismatches** — single occurrence, monitor.

## Tool Failure Patterns

- **Tool:** `http_fetch` — **Error type:** JS SPA blank page capture — **Occurrences:** 2+ — **Recommendation:** Always use `browser_navigate` → `browser_wait` → `browser_snapshot` for SPAs. `http_fetch` only for server-rendered HTML.
- **Tool:** `http_fetch` — **Error type:** Amazon anti-bot block — **Occurrences:** 6+ — **Recommendation:** Fall back to browser tools for Amazon. If browser fails, report failure rather than retrying fetch.
- **Tool:** `http_fetch` — **Error type:** Localhost timeout on large payloads — **Occurrences:** 1 — **Recommendation:** Extract text via browser first, never send raw HTML (>50KB) to localhost:8091.
- **Tool:** `python` — **Error type:** Command not found — **Occurrences:** 4 — **Recommendation:** Always use `python3` in bash invocations. Use `set -u` to catch undefined variables.
- **Tool:** `vault_search` — **Error type:** Default timeout insufficient — **Occurrences:** 1 — **Recommendation:** Use 300s for small searches; 600s for large repos (>5000 files).
- **Tool:** `browser_navigate` — **Error type:** Session state loss — **Occurrences:** 3+ — **Recommendation:** Use `browser_tabs` to manage tabs within single browser instance. Never chain sequential navigate→close cycles.

## Positive Patterns to Reinforce

- **Pattern:** YouTube transcript pipeline (`browser_navigate` → `browser_snapshot` → extract text) — **Evidence:** 2026-06-14 daily note confirms working across sessions — **Action:** Encode as gold standard reference workflow
- **Pattern:** Multi-link parallel dispatch — each URL gets its own subagent — **Evidence:** corrections.md explicit threshold guidance, works well when followed — **Action:** Enforce 1:1 mapping rule
- **Pattern:** Direct, practical communication style — **Evidence:** corrections.md praises concise responses, no fluff — **Action:** Keep responses focused on practical tradeoffs
- **Pattern:** Browser session lifecycle management — **Evidence:** vault-maintenance/2026-06-13.md shows improved browser workflow — **Action:** Encode browser session management as standard practice
- **Pattern:** Knowledge graph discipline — **Evidence:** corrections.md and vault-maintenance notes show progress — **Action:** Maintain `safety_passed` rate on knowledge graph integrity