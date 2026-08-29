---
segment: agents
generated: 2026-08-28 01:07 PDT
data_range: 2026-08-25 to 2026-08-27 (+ 08-28 pre-dawn infra events through 01:05 PDT)
---

# Signal Report — 2026-08-28 (supersedes the 00:04 report)

**Supersedes the 2026-08-28 00:04 report (commit `79347fb`, window 08-25 → 08-27).** That run committed a valid report, then the 08-28 cycle kept flapping: 4 further 600s-timeout failures (07:08/07:19/07:30/07:53Z; the 07:30Z run produced the report + pre-flight commit before timing out). This run (5th attempt in the 08-28 cycle) adds: (1) four new explicit signals (18–21), the most important being an **unlogged entity-state writer that the sweep's own audit (run 237) declares the top-priority human action**, (2) three factual corrections to the prior report's data-source/git claims, and (3) status updates to signals 13–17. No new user-facing corrections in the window: `corrections.md` unchanged since 08-23; daily notes 08-25/08-26/08-27 are all clean auto-captures (research/hardware topics, no pushback recorded).

## Data Sources (verified this run)

- `memory/corrections.md` — unchanged since 08-23 (mtime 08-23 15:06)
- Daily notes 08-25 (2 sessions) / 08-26 (3 sessions) / 08-27 (3 sessions); 08-24 = stub; 08-28 not yet produced (T-0, expected)
- `memory/vault-maintenance/2026-08-27.md` — **CORRECTION**: the 00:04 report cited `memory/data-pipeline/2026-08-27.md` + per-task JSONL as verified sources; that path does not exist. `memory/data-pipeline/` holds only old skip-logs; the data-pipeline run notes live in `memory/vault-maintenance/YYYY-MM-DD.md`. `memory/autonomy-pipeline/` holds 2026-05-03.md only.
- `~/lloyd/_pipeline/` — reflection dir; incident file 8,507 lines / 1.54MB (mtime 01:03:46 PDT, still appending); 144 pre-append `.bak` (gitignored)
- `autonomy/38-nightly-reflection-signals.md` (failure_count 12), `autonomy/40-nightly-reflection-config.md`, `autonomy/48-entity-resolution-sweep.md`; `knowledge-health-2026-08-27.md` (total relationships 0)
- git: vault = `main` only; lloyd = `nightly-improvement-2026-08-23` (HEAD, unmerged)

## Pre-Flight Actions (this run)

| Repo | Action | Note |
|------|--------|------|
| `~/obsidian` | pre-flight commit | 6 system-owned files (autonomy task frontmatter churn 24/38/48/68/75 + morning-brief `state.json`) |
| `~/lloyd` | **skipped — clean** | No tracked changes: `_pipeline/` is gitignored (`.gitignore` line 18) and its only tracked file (`reflection/signals-latest.md`) was clean. Untracked `tmp-thunderbird-autonomy-20260824/` + `.zip` (157KB) deliberately NOT committed — flagged for deletion, not preservation (00:04 report); deletion is a Job 3 call. |

**Corrections to the 00:04 report's pre-flight/git claims (verified this run):**
1. `_pipeline/` is gitignored — the incident file + 144 baks are invisible to git and cannot inflate `git status`/`git add` scans. Commit `79347fb` (00:31 PDT) contains only `reflection/signals-latest.md` (+49/−21). The 00:04 claim "this run's pre-flight committed 378KB of [the incident file]" is false.
2. `tmp-thunderbird-autonomy-20260824/` is **untracked**, not staged (00:04 report said "staged, uncommitted").
3. Vault branch `knowledge-writes-2026-08-26` no longer exists — vault is single-branch (`main`). Signal #16's vault instance is resolved; only the lloyd-side branch remains.

## Queued Signals (met threshold)

### Explicit (act on first occurrence)

| ID | Date | Type | Category | Description | Source |
|----|------|------|----------|-------------|--------|
| 13 | 08-27→28 | correction | process | **Sweep incident append loop — still active, now slowed.** 8,445 lines / 1.36MB (00:02) → 8,507 lines / 1.54MB (01:03 PDT); 140 → 144 baks. Pace slowed from ~1/min to the 18–40 min band per the incident file's own "next pass" projection (00:45:50). Volume leak continues; kill-switch remains a Job 3 action. | incident file + mtime |
| 18 | 2026-08-28 | correction | infra | **Nightly reflection jobs hitting the 600s source-level cap systematically.** Job 38: 4 consecutive timeout failures this 08-28 cycle (07:08/07:19/07:30/07:53Z, all "timed out after 600s"; failure_count now 12). Job 40: 1× at 08-27 06:33Z (recovered on retry in 258s). A full Job 1 pass (both-repo pre-flight + report write + commit) does not fit in 600s. Recommend source-level `max_duration_seconds` 1200–1800s for the reflection source. | tasks 38/40 activity logs |
| 19 | 2026-08-28 | correction | infra | **Sweep (task 48) now hitting the 1800s source-level cap — the 08-22 1200→1800s bump (`699b2d17`) is exhausted.** 3× "Recovered from in_progress after 1818–1832s (timeout=1800s)" on 08-28 (05:59/06:57/07:56Z UTC), each followed by a ~400s successful run — same poisoned-run signature as the Jun 27 GPU-crash era. Recommend root-cause triage (environment vs config vs task bug) before any further cap bump: why do ~1-in-3 runs take 30+ min? | task 48 activity log |
| 20 | 2026-08-28 | correction | infra | **Unlogged entity-state mutation is invalidating the rebuild baseline — sweep's own audit (run 237) declares: "a pause-or-recover decision (#48/#67 + nightly extraction) is now the top-priority human action ... #48 cannot unblock itself."** Per the incident file: ~7h of writes, +2,438 alias keys, a ~32-dir entity-tree restructuring at 00:16–00:23 PDT with no log and no observed process — a writer outside logged extraction passes is mutating sweep-managed state. Any rebuild/restore baseline produced now keeps getting invalidated. KG-recovery priority shifts from "restore `_relationships.json`" to **stop the unlogged mutation first**. | incident file tail (run 237 audit) |
| 21 | 2026-08-28 | correction | process | **Sweep task prompt context STALE — 8th consecutive run.** Task 48's prompt still carries run-120 text (08-24 22:32Z); incident-file run numbering is authoritative (run 237). The task is running on instructions 117 runs old; it correctly refuses to act on them, but each run burns cycles re-deriving true state. Recommend refreshing task 48's description to point at the incident file as the authoritative state. | incident file + task 48 frontmatter |

### Inferred (met 2+ threshold)

| ID | Date | Type | Category | Description | Frequency | Source |
|----|------|------|----------|-------------|-----------|--------|
| 14 | 08-24→28 | pattern | tool-use | **Timeout-cap pattern, both directions, 5+ occurrences.** Frontmatter `timeout_seconds` ignored (vault-maintenance 900s→600s 08-27; knowledge-health 900s 08-27 02:11) AND source caps now hit in flight (18: 600s ×6 across jobs 38/40; 19: 1800s ×3 on job 48). Frontmatter is cosmetic; source-level `max_duration_seconds` is the only knob, and both the 600s default and the 1800s bump are systematically under-sized. Supersedes/extends the 00:04 #14. | 6× | tasks 38/40/48 logs |
| 15 | 08-27 | pattern | tool-use | **`python` vs `python3` — 4th occurrence window** (carried; no new occurrence this cycle). Recommend lint rule on skill/bash examples. | 4× | daily note + corrections_log |
| 17 | 08-24→28 | pattern | tool-use | **Daily-note-lag structural false positive — 4 consecutive days** (08-24, 08-25 ×3 entries, 08-27 21:21). The 00:00/23:5x run of day T always fires before T-1's note exists. Align check to T-2 or gate on session-log evidence. | 4× | data-pipeline JSONL + reports |

**Demoted to Pending:** #16 (stale branches) — vault instance `knowledge-writes-2026-08-26` resolved (vault single-branch); only lloyd-side `nightly-improvement-2026-08-23` remains = 1 open occurrence → below threshold.

## Pending Signals (below threshold)

- **08-27 16:41 session — "response appears truncated"** (Adam Savage livestream summary; auto-capture wording). 1 occurrence, ambiguous: response-level turn truncation vs transcript-extraction cut-off. If confirmed as an extraction cut-off it is the first since the 08-19 guardrail (breaks the 17+ clean streak). Check for a `[partial extraction]` marker in the transcript before counting.
- **lloyd repo unmerged branch `nightly-improvement-2026-08-23` (HEAD)** — 1 open occurrence after the vault instance resolved; merge-or-delete when the next lloyd config change lands (or now — Job 1 has committed there 5 cycles running).
- **`memory/learnings/2026-07-09.md` untracked 18+ days** — carried; one-off commit if still untracked next week.
- **`~/lloyd/tmp-thunderbird-autonomy-20260824/` + `.zip` (157KB, untracked)** — flagged for deletion (Job 3).
- **QMD `subliminal/` duplication; VTT-parser tail-echo dedup; `entity-aliases.json` self-mapped placeholders** — carried; no new occurrences 08-19→08-28.
- **Jobs 3/4 (prompt audit / behavior test) — 4th consecutive cycle with no artifacts** (last audit 03-31, last test-failures 03-29); 08-28 cycle's Job 5 not yet run (next_run 06:38Z).
- **`memory/learnings/2026-08-27.md` not yet present** — expected; 08-27 cycle Jobs 2/5 pending, not a failure.

## Tool Failure Patterns

- **Tool:** entity-resolution-sweep (task 48) — **Error type:** now exceeding the 1800s source cap (3× 08-28); the 08-22 1200→1800s bump is exhausted — **Occurrences:** 3 — **Recommendation:** root-cause triage before any 2400–3000s bump (why do ~1-in-3 runs take 30+ min?); never blind-retry (Jun 27 / 08-22 rule).
- **Tool:** nightly-reflection jobs 38/40 — **Error type:** 600s source-level default cap systematically kills full passes — **Occurrences:** 4× job 38 (08-28 cycle) + 2× job 40 (08-26/27) — **Recommendation:** source-level `max_duration_seconds` 1200–1800s for the reflection source.
- **Tool:** unlogged entity-state writer (#48/#67 + nightly extraction family) — **Error type:** unlogged mutation (+2,438 alias keys, ~32-dir tree restructuring at 00:16–00:23 PDT) invalidating the rebuild baseline — **Occurrences:** 1 documented episode, ongoing — **Recommendation:** pause-or-recover human decision (top priority, per signal 20); do NOT start a rebuild while any writer is active.
- **Tool:** sweep incident-append loop — **Error type:** slowed but active (18–40 min band; 144 baks) — **Occurrences:** ongoing — **Recommendation:** task-level kill-switch + bak retention (keep last N per run) + move append to a dedicated log.
- **Tool:** data-pipeline `daily-note-lag` check — **Error type:** structural false positive — **Occurrences:** 4 consecutive days — **Recommendation:** check T-2 or gate on session-log evidence.
- **Tool:** `python` — **Error type:** not on PATH (use `python3`) — **Occurrences:** 4 in window — **Recommendation:** lint skill/bash examples.
- **CORRECTIONS to 00:04 report (false positives retracted):** the incident file does NOT inflate lloyd git scans (`_pipeline/` gitignored); `tmp-thunderbird` is untracked, not staged; `memory/data-pipeline/2026-08-27.md` does not exist (real path: `memory/vault-maintenance/2026-08-27.md`).

## Positive Patterns to Reinforce

- **Pattern:** **Sweep self-audit now produces correct escalation text, not just logs.** "A pause-or-recover decision ... is now the top-priority human action," "#48 cannot unblock itself," "task prompt context is STALE (8th consecutive)." It refuses to act on stale instructions, maintains bak + md5-verify + audit-entry discipline, and has made zero baseline mutations. **Evidence:** 8,507 lines of compliant audit entries; zero `--apply`/mutations. **Action:** keep the behavior; fix the volume (signal 13) and the stale context (signal 21) via Job 3.
- **Pattern:** **Partial-progress durability under timeout flapping** — the 07:30Z Job 1 run committed the report + pre-flight *before* timing out; the 3 subsequent failed runs neither re-did nor corrupted the work (the 00:04 report stayed valid on disk until this run superseded it). Commit-before-long-phase is the right order. **Evidence:** `79347fb` + intact prior report. **Action:** none — already the natural order (pre-flight first).
- **Pattern:** **Timeout-retry recovery at job level works** — job 40's 08-27 06:34Z retry (258s) completed the full cycle including the MEMORY.md restoration to the canonical index after its own 06:33Z 600s timeout. The flapping-then-success pattern holds; only the cap size is wrong. **Evidence:** task 40 activity log. **Action:** none (cap fix in signal 18).
- **Pattern:** **Recovery-claim verification rule held — 5th consecutive cycle (08-24→08-28).** 08-27 knowledge-health reports 0 total relationships; 08-27 vault-maintenance index metrics (307,023 relation-index edges, 56,568 fact files) are index-layer metrics, not a recovery claim. No overclaim written. **Action:** continue.
- **Pattern:** **Truncation guardrail / transcript extraction held** — 08-27: 3 clean research sessions (Recurse paper video, 938 segments + second Recurse results session + Adam Savage livestream). One ambiguous "appears truncated" line logged as Pending (above), not counted as a cut-off. **Action:** none if the pending item resolves as response-level.
- **Pattern:** **Research burst-then-trough cadence continues** (08-27 = 3 sessions). Hardware/maker cluster now at ~9–10 sessions (08-25 plate + Bernie robot, 08-27 Adam Savage hobby→job) — approaching the ~10-session consolidation threshold from the standing note. **Action:** revisit at next consolidation.

## Job 2 / Job 3 Handoff Notes

- **Job 2 (Knowledge Consolidation):** primary new material = signals 20 + 21 (unlogged writer; stale prompt context) — the KG-recovery story changes from "await rebuild" to "stop the mutation first." Also 18/19 (cap exhaustion) and the three factual corrections to the 00:04 report (data-source paths, gitignore, vault branch).
- **Job 3 (Config Application):** priority order: (1) **pause-or-recover decision on the unlogged entity-state writer — human action, top priority** (signal 20); (2) sweep prompt-context refresh (21) + append-loop kill-switch + bak retention (13); (3) reflection source timeout 600s → 1200–1800s (18) + sweep cap root-cause triage (19); (4) daily-note-lag alignment (17); (5) delete `tmp-thunderbird-autonomy-20260824/`; (6) merge-or-delete lloyd branch `nightly-improvement-2026-08-23`.
- **Constraint carried forward:** NO `--apply`, NO recreation, NO mutation of the entity graph while any writer (sweep or unlogged) is active. Any action touching `entity-resolution-sweep` stops the writer first (task-level).
