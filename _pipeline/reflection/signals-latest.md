---
segment: agents
generated: 2026-08-23 22:13 PDT
data_range: 2026-08-21 to 2026-08-23
---

# Signal Report — 2026-08-23

**Sources read:** `memory/corrections.md` (no new entries since 2026-05-08), daily notes 08-21/08-22 (read; 08-23 note MISSING), derived sessions `~/lloyd/_pipeline/vault-derived/sessions/2026-08-22..23/` (10 sessions), trajectories `~/lloyd/_pipeline/trajectories/2026-08-22/23.jsonl` (1+1 sessions). Window spans 08-22 23:21 PDT → 08-24 01:16 PDT, centered on the new-box/OS migration.

## Queued Signals (met threshold)

### Explicit (act on first occurrence)

| ID | Date | Type | Category | Description | Source |
|----|------|------|----------|-------------|--------|
| 1  | 2026-08-23 | tool-failure | tool-use | `youtube-transcript-api` import fails on first try in 6 of 8 transcript sessions: `ModuleNotFoundError` under system python3.14 and wrong-venv attempts; recovery each time = `~/lloyd/.venvs/lloyd/bin/python3` (has v1.2.4) or `pip install`. SKILL.md (`youtube-transcript`) does not pin the interpreter path, so every run re-discovers it. Encode the venv interpreter + pin in the skill's first step. | derived sessions 08-23 (iv30d4, ivfdf0, iv3006, iv5b3e, ivc769, iv8c65) |

### Inferred (met 2+ threshold)

| ID | Date | Type | Category | Description | Frequency | Source |
|----|------|------|----------|-------------|-----------|--------|
| 1  | 2026-08-21/22/23 | pipeline | memory-capture | Daily note for 2026-08-23 was never generated (last note = 08-22); `memory-capture.log` still stale at 06-03. 3rd consecutive nightly cycle with the same flag — the job is not running on the new system. In scope of backlog #383 (autonomy job health diagnosis); recommend treating as the first fix in that pass. | 3x consecutive | daily notes + prior signal reports |

## Pending Signals (below threshold)

- **Bash (YouTube API) — transient `502 Bad Gateway` from YouTube on retry** — 1 occurrence (08-23, iv30d4); handled cleanly with backoff + re-fetch. Monitor; 2nd occurrence → add backoff/retry note to `youtube-transcript-error-handling`.
- **vault_search returned empty results on new system** — `vault_search("communication style ...")` → `{"results": []}` across all 12 collections (08-22, iv0484). Recovered via grep/find fallback on disk. Could be a qmd index gap vs. the lloyd-mcp tool backend; needs one deliberate probe with known-present content before acting. 1 occurrence. Monitor.
- **qmd surface friction** — `localhost:9001` refused connection (wrong/legacy port; service actually on 8181) + one MCP call rejected with `-32000 Not Acceptable: Client must accept both application/json and text/event-stream` (08-23, ivc769). Both worked around; verify qmd port config and accept-headers on the next qmd maintenance pass. 1 occurrence each.
- **User question left unanswered mid-session** — chrome-extension lookup (08-23, ivb492) delivered findings but not the actual answer ("how do I load this into Chrome") until an `[INNER VOICE]` nudge forced the concrete steps. Also surfaced a real repo gap: `chrome-extension/manifest.json` missing on disk and never in git (build assumes it). 1 occurrence. Monitor; repo fix is an action for Alan's queue, not a behavior config.

## Tool Failure Patterns

- **Tool:** `Bash` (youtube-transcript-api via python) — **Error type:** `ModuleNotFoundError: No module named 'youtube_transcript_api'` under `/usr/bin/python3` (3.14) and non-lloyd venvs; secondary: `uvx` executable-name error (`An executable named youtube-transcript-api is not provided`), and one API-level `TypeError: 'FetchedTranscriptSnippet' object is not subscriptable` after a version drift — **Occurrences:** 6 in 8 sessions (08-23) — **Recommendation:** fix in `youtube-transcript` SKILL.md: first line of the protocol = `PY=~/lloyd/.venvs/lloyd/bin/python3` (pinned `youtube-transcript-api==1.2.4`), use `FetchedTranscriptSnippet.text` attribute access, never bare `python3`/`uvx`.
- **Tool:** `Bash` (YouTube API) — **Error type:** transient `502 Bad Gateway` on a re-fetch — **Occurrences:** 1 — **Recommendation:** monitor; backoff pattern worked (see pending above).
- **Tool:** `Bash` (qmd/MCP probes) — **Error type:** wrong port (9001 refused) + `Not Acceptable` accept-header rejection — **Occurrences:** 1 each — **Recommendation:** config/verify pass, not a skill change (see pending above).

## Positive Patterns to Reinforce

- **Pattern:** Truncation guardrail now holds across the migration window — all 8 transcript sessions (08-23/24: Huberman, async-Python, DHH Omarchy, Maker Z1, Orca Slicer, Etched, smart-people talk, plus 08-22 re-runs) explicitly verified tail completeness (end-of-outro / final sentence / duration-vs-lengthSeconds check on the Etched video) before summarizing. Zero cut-offs since the 08-19 guardrail (6 total lifetime, none new). **Evidence:** 8/8 sessions — **Action:** no change; keep the guardrail in `youtube-transcript` SKILL.md. Note the Etched session shows the *strongest* form yet: a suspicious duration-sum vs max-start-time mismatch was investigated (durations/overlap drift) and resolved via oEmbed `lengthSeconds` before concluding.
- **Pattern:** Read-skill-first discipline emerging — the two most recent transcript sessions (iv8c65, iv7acd) opened with `skills_search` → `youtube-transcript` SKILL.md read *before* any extraction attempt; the earlier six did not (and each paid the discovery cost in retry steps). The 03-29 corrections (skills-check mandatory) are showing through. **Evidence:** 2 consecutive sessions — **Action:** reinforce in `youtube-transcript` SKILL.md preamble ("Step 0: you read this skill — proceed to the venv interpreter step"), no new skill needed.
- **Pattern:** Post-migration verification sequence working well — (a) full systems check via `system-health-check` skill with clean interpretation (load 11 correctly attributed to transient CUDA build + Chrome, not a fault), (b) QMD-on-GPU + vault-index confirmation via a *live search of yesterday's incident topic* hitting the incident file at 100% — using a known-real query as the index smoke test. **Evidence:** 2 sessions (08-22 ivе8da, 08-23 ivc769) — **Action:** add the live-known-topic probe as the standard "vault index healthy?" check in `qmd-index-maintenance` / `system-health-check`.
- **Pattern:** Empty `vault_search` → grep/find disk fallback recovered the communication-style answer in one pass (08-22 iv0484) — the vault-retrieval fallback chain (search → grep → read) behaves as specified. **Evidence:** 1 session (paired with the pending vault_search-empty flag) — **Action:** none; existing protocol held.

## Notes for downstream jobs

- **Job 2 (knowledge):** entity-graph incident state unchanged in-window (still in "writers churning rebuild / recovery pending" per 08-22 incident file); the 08-22 QMD live-search hit confirms the incident knowledge file itself is indexed and retrievable. No new knowledge writes required from this window beyond the youtube-transcript venv fix and the daily-note-missing flag.
- **Job 3 (config):** only actionable config surface is the `youtube-transcript` SKILL.md interpreter pin (explicit signal 1) and the backlog #383 escalation of memory-capture (inferred signal 1). Everything else in-window was already handled by the 08-22 run (timeout cap 1800s, broken-baseline guardrail, root-cause-triage pattern).
- **Data gap:** no daily note for 2026-08-23 exists; this report relied on derived-session + trajectory sources for the 08-23 window. If memory-capture is dead (inferred signal 1), future runs will have the same gap — factor into #383 prioritization.

_Generated by nightly-reflection-signals (Job 1/3) on 2026-08-23 22:13 PDT_
