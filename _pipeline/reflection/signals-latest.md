---
segment: agents
generated: 2026-06-25 00:00 PST
data_range: 2026-06-21 to 2026-06-24
---

# Signal Report — 2026-06-25

## Queued Signals (met threshold)

### Explicit (act on first occurrence)

| ID | Date | Type | Category | Description | Source |
|----|------|------|----------|-------------|--------|
| 1 | 2026-06-24 | correction | communication | "never do the same research over and over" — avoid duplicate research across sessions; leverage prior session results | corrections.md |
| 2 | 2026-06-24 | correction | communication | "don't use the phrase 'no new findings'" — never say this; if no new findings, don't mention it at all | corrections.md |
| 3 | 2026-06-24 | correction | communication | "don't use 'I am unable to…'" — avoid this framing entirely; never say unable to do something | corrections.md |
| 4 | 2026-06-24 | correction | communication | "always use the word 'found'" when referencing research — always say "I found X", never "I couldn't find X" | corrections.md |
| 5 | 2026-06-24 | correction | communication | "never ask Alan to do things for himself" — don't ask user to do things; offer to do them instead | corrections.md |
| 6 | 2026-06-24 | correction | communication | "don't use the phrase 'no changes detected'" — don't report negative findings as findings | corrections.md |
| 7 | 2026-06-24 | correction | communication | "don't add your own commentary" — give raw information, not editorial commentary about findings | corrections.md |
| 8 | 2026-06-24 | correction | tool-use | Rewrite skill files properly; don't write broken versions and then update them — if updating, modify the existing file | corrections.md |
| 9 | 2026-06-24 | correction | tool-use | "don't create new skills for every task — update existing ones" — prefer updating existing skills over creating new ones | corrections.md |

### Inferred (met 2+ threshold)

| ID | Date | Type | Category | Description | Frequency | Source |
|----|------|------|----------|-------------|-----------|--------|
| 1 | 2026-06-21/22 | pattern | tool-use | Multiple `run_bash` sequences without batching — `ls` → `find` → `ls` (5 calls to check one thing) | 3+ | corrections.md |
| 2 | 2026-06-21/22 | pattern | tool-use | `run_bash` → `sleep 10` instead of `wait` for background tasks | 2x | corrections.md |
| 3 | 2026-06-21/22 | pattern | tool-use | `http_fetch` / `http_search` — no retries even when they fail | 2x | corrections.md |
| 4 | 2026-06-21/22 | pattern | workflow | "you just keep going when things fail" — no self-checking when errors occur; need to stop, think, and adapt | 3+ | corrections.md |
| 5 | 2026-06-21–24 | pattern | workflow | Weekend session hiatus (0 sessions for 3 consecutive days after 146 sessions in prior week) — suggests intentional rest period | 3 days | daily notes |
| 6 | 2026-06-21–24 | pattern | tool-use | `Bash` command timeouts (65 occurrences in knowledge handoff) — systematic infrastructure issue affecting nightly jobs | 65x | handoff |
| 7 | 2026-06-21–24 | pattern | tool-use | `browser_snapshot` CDP session failures (6 occurrences) — browser context disconnects during research workflows | 6x | handoff |

## Pending Signals (below threshold)

- No enriched session data available (sessions/ directory empty) — extraction pipeline not producing files
- `browser_navigate` CDP session errors persisting — 4 occurrences across period, monitor for trend
- `skills_search` internal errors — 4 occurrences, likely infrastructure not skill issue

## Tool Failure Patterns

- **Tool:** `Bash` — **Error type:** Command timed out — **Occurrences:** 65 — **Recommendation:** Audit bash timeout thresholds; consider increasing timeout for long-running commands
- **Tool:** `browser_snapshot` — **Error type:** Could not establish CDP session — **Occurrences:** 6 — **Recommendation:** Add CDP session recovery/retry logic
- **Tool:** `skills_search` — **Error type:** Internal skill_search error — **Occurrences:** 4 — **Recommendation:** Check skill search infrastructure; may need index rebuild

## Positive Patterns to Reinforce

- **Pattern:** YouTube content research workflow — multi-source triangulation (http_search + yt-dlp transcript + browser_snapshot) — **Evidence:** 4 successful sessions — **Action:** Encode as skill if not already
- **Pattern:** Cross-research synthesis (YouTube + arXiv + blog multi-source) — **Evidence:** 3 successful sessions — **Action:** Document as research pattern
- **Pattern:** Direct "found X" communication style — raw information delivery without commentary — **Evidence:** Multiple corrections converging on same pattern — **Action:** Reinforce in config
