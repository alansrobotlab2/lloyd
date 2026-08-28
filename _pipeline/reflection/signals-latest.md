---
segment: agents
generated: 2026-08-28 00:04 PST
data_range: 2026-08-25 to 2026-08-27
---

# Signal Report — 2026-08-28

**Supersedes the 2026-08-27 22:57 PST run (signals-latest.md, 12 queued signals).** This run added: (1) pre-flight commits in both repos (vault `69f9fdc8` / `254f37b2`; lloyd `7e6a965`), (2) one new explicit signal (#13 — incident file append loop), (3) reclassification of two pending items into queued patterns (#14, #15), and (4) a corrected "data sources" caveat (previous run claimed `extraction_logs.jsonl` did not exist; the pipeline's enrichment stream is `memory/data-pipeline/`, which does exist and was in scope). All 12 previously queued signals remain in effect (see prior report; not restated here).

## Data Sources (verified)

- `memory/corrections.md` (unchanged through 08-27)
- Daily notes 08-25 / 08-26 / 08-27 (incl. new `memory/vault-maintenance/2026-08-27.md`)
- `memory/data-pipeline/2026-08-27.md` + per-task JSONL (2026-08-25 … 2026-08-27) — the "enriched session data"
- `memory/autonomy-pipeline/` — 08-24 … 08-27 task JSONL + reports
- `~/lloyd/_pipeline/` — reflection dir, extraction log, incident file + 140 pre-append baks, bg-*.log
- `autonomy/24-data-pipeline.md`, `autonomy/38-nightly-reflection-signals.md`, `autonomy/48-entity-resolution-sweep.md`

## Pre-Flight Actions (this run)

| Repo | Commit | Contents |
|------|--------|----------|
| `~/obsidian` | `69f9fdc8` | data-pipeline 08-27 (12 files) + 14 daily notes incl. 08-27 + vault-maintenance/2026-08-27 + 2026-02-22 fixup |
| `~/obsidian` | `254f37b2` | 00:00 nightly reflection commit (daily-note + vault-maintenance timestamp rewrite) |
| `~/lloyd` | `7e6a965` | entity-resolution-sweep-incident-2026-08-22.md 0→378KB (8,445 lines); baks already gitignored |
| `~/lloyd` (staged, uncommitted) | `3e0eeca` | `tmp-thunderbird-autonomy-20260824/` — **NOT committed**: transient Thunderbird autonomy-task probe artifacts; should be deleted, not preserved. Flag for Job 3. |

**Live-writer collision (vault):** after the pre-flight commits, the 00:00 data-pipeline cycle re-modified `autonomy/24-data-pipeline.md` and `autonomy/48-entity-resolution-sweep.md`. Left uncommitted deliberately — committing mid-write risks capturing a torn state. Next cycle's pre-flight will pick them up.

## Queued Signals (met threshold)

### Explicit (act on first occurrence)

| ID | Date | Type | Category | Description | Source |
|----|------|------|----------|-------------|--------|
| 13 | 2026-08-27 | correction | process | **Incident file append loop (still active at report time).** `entity-resolution-sweep-incident-2026-08-22.md` grew 378KB (8,445 lines) → 1.36MB (~8,650 lines) between 23:35 and 00:02. Each append creates a byte-identical pre-append `.bak` (140 baks / 122MB, all gitignored). Appends now ~1/min (was ~20-40 min at 23:35). File is now the hottest write in `~/lloyd/_pipeline/`, inflating every `git status`/`git add` scan on that repo. The sweep task itself is not visible in `ps` (likely detached bg/subagent), so the writer cannot be stopped by killing a process — must be stopped at the task level. | live observation (this run) + incident file diff |

### Inferred (met 2+ threshold)

| ID | Date | Type | Category | Description | Frequency | Source |
|----|------|------|----------|-------------|-----------|--------|
| 14 | 08-24→08-27 | pattern | tool-use | **Task frontmatter `timeout_seconds` ignored — confirmed 2×.** (a) entity-resolution-sweep poisoned 3× at the 1200s source cap before the 1200→1800s config fix (`699b2d17`). (b) 08-27 vault-maintenance note: `vault-maintenance` "timed out at 900s (frontmatter limit ignored; pool uses 600s default)." Task authors setting timeouts in frontmatter believe they've changed the cap when they haven't. Fix at source level (done for sweep; `vault-maintenance` not yet). | 2× | daily note + corrections_log (08-22) + config commit |
| 15 | 08-23→08-27 | pattern | tool-use | **`python` vs `python3` — 4th occurrence window.** 08-27 daily note (10:08 session): `python3` explicitly required — `python` not on PATH. Same class as the Jun 13 / Jul 23 corrections_log entries. Recommend a lint rule on skill/bash examples (`python ` → `python3 `) rather than more memory entries. | 4× | daily note + corrections_log |
| 16 | 08-26→08-27 | pattern | infra | **Stale `knowledge/` git branch in vault.** Unmerged `knowledge-writes-2026-08-26` (83 commits, "08-26 23:50", behind 5) found again at the 08-27 21:21 vault-maintenance check. Same family as lloyd repo's unmerged `nightly-improvement-2026-08-23`: reflection/knowledge jobs commit to ephemeral branches that never merge or delete. Branches accumulate in both repos. | 2× | vault-maintenance note + USER.md infrastructure |
| 17 | 08-24→08-27 | pattern | infra | **Recurring `daily note production lag` warning — 4 consecutive days.** 08-24 (22:58): 1/1 missing (T-1). 08-25 (20:27, 23:54, 00:13): 1/1 missing (T-1). 08-27 (21:21): 1/1 missing (T-1). The 00:00/23:5x runs of T's own day always fire before T-1's note exists → structural false-positive unless the check window or the capture deadline is aligned. Either align the lag-check to T-2 (or gate on session-log evidence, as the 08-22 stub backfill did) or accept one warning/day and stop reporting it. | 4× | data-pipeline JSONL + reports |

## Pending Signals (below threshold)

- **08-27 02:11 `knowledge-health` timeout at 900s frontmatter limit** — 1 occurrence, same root cause as #14; will queue automatically if it recurs (would then be 3×).
- **`memory/learnings/2026-07-09.md` untracked for 18+ days** (last seen untracked in the 08-09 vault-maintenance report; still present in the 08-27 21:21 commit) — monitor; if still untracked next week, queue as a one-off commit.
- **`~/lloyd/tmp-thunderbird-autonomy-20260824/` (157KB, staged)** — transient artifacts left staged in the lloyd repo by the 08-24 Thunderbird bridge restoration work; 08-27 vault-maintenance note flags them for deletion. Job 3 to delete (user-owned dir, not protected — but deletion of user data is a Job 3 config decision, not a signal).
- **QMD `subliminal/` search duplication** — open since 08-23, no new occurrence in window; stays on USER.md as an open issue, not a fresh signal.
- **VTT-parser tail-echo dedup fix** — still pending since 08-19; no new occurrences 08-19→08-27 (guardrail held, 17+ clean extractions). Remains a known-pending fix, not a new signal.
- **`entity-aliases.json` degenerate self-mapping** (14,487 `'#222' -> '#222'` placeholders regenerated 08-26 16:05 PDT) — single occurrence, subsumed by the KG-recovery workstream; no second independent occurrence to queue.

## Tool Failure Patterns

- **Tool:** `entity-resolution-sweep` (nightly, bg) — **Error type:** writer active while baseline `_relationships.json` is still missing — appends to incident file at ~1/min, one byte-identical 1.36MB `.bak` per append (140 baks / 122MB at 00:02). Not a crash; a resource leak + signal flood. The 08-27 daily note already lists the kill-switch: halt the sweep's incident-append step (or route append-only to a log file the sweep owns). **Recommendation:** Job 3 — task-level kill switch for the append loop + `.bak` cleanup (keep last N per run, e.g. run236) + move incident-append to a dedicated log. **Constraint (from 08-22 incident rule): NO apply/recreation/mutation of the entity graph while the writer is active.**
- **Tool:** `git status`/`git add` on `~/lloyd` — **Error type:** hot 1.36MB file (`entity-resolution-sweep-incident-2026-08-22.md`) rewritten ~1/min makes repo-wide scans noisy and risks committing a torn mid-append state (this run's pre-flight committed 378KB of it; the file grew further afterward — the committed snapshot is a valid point-in-time, not a loss). **Recommendation:** fold into the sweep kill-switch above; additionally add a targeted `git add <path>` for the incident file instead of whole-repo staging until the writer is stopped.
- **Tool:** `vault-maintenance` (task) — **Error type:** frontmatter `timeout_seconds=900` ignored; effective pool default 600s; task reported as "timed out at 900s" (misattribution of which limit fired). **Recommendation:** same as #14 — source-level `max_duration_seconds` for `vault-maintenance`.
- **Tool:** `python` — **Error type:** not on PATH (use `python3`) — **Occurrences:** 4 in window (08-27 daily note explicit). **Recommendation:** lint skill/bash examples; keep python3 rule in MEMORY.md (already present).
- **Tool:** data-pipeline `daily-note-lag` check — **Error type:** structural false positive every night (fires before T-1 note exists) — **Occurrences:** 4 consecutive days. **Recommendation:** check T-2, or gate on session-log evidence before warning.

## Positive Patterns to Reinforce

- **Pattern:** **Sweep append loop is *defensive and self-verifying*, not destructive.** Every append writes a pre-append `.bak`, md5-verifies it identical to the source, and logs the exact line/byte count (e.g. `run236-pre.bak`, 8,444 lines / 1,516,134 B, md5 `ff999d137adb18555d3bcca396ec6bf9`, "verified identical to source pre-copy"). The 08-22 rule ("never `--apply` against a broken baseline") is being honored under load: no apply, no recreation, no mutation, no service restart while the writer is active. **Evidence:** 8,445 → ~8,650 lines of compliant audit entries over ~13h; zero baseline mutations. **Action:** keep the rule; fix the *volume* (signal #13), not the *behavior*. This is the correct response to the destroyed graph — document it as the canonical "writer-active" posture in the entity-resolution-sweep SKILL.md.
- **Pattern:** **Pre-flight commit discipline held across both repos this run** — 3 clean commits, zero torn writes in committed state (the 00:00 data-pipeline collision was detected *after* commit and left alone rather than force-committed mid-write). **Evidence:** `69f9fdc8`, `254f37b2`, `7e6a965` all complete and internally consistent. **Action:** no change; the leave-alive-writer-behavior is the correct one — encode as a one-line note in the vault-commit.sh header comment ("if files re-dirty during commit, stop and leave them; do not force-commit mid-write").
- **Pattern:** **Recovery-claim verification rule held for a 4th consecutive cycle (08-24, 08-25, 08-26, 08-27).** `_relationships.json` still missing; `facts/` rebuilding (1,837 → 55,024 files); extraction reports 299 relationships (degraded non-zero metric, *not* a recovery claim). No overclaim written. **Evidence:** 08-27 21:21 vault-maintenance note + 08-26 23:15 UTC disk check. **Action:** continue; the rule is working — no change needed.
- **Pattern:** **Truncation guardrail held 08-19→08-27: zero cut-offs, 17+ clean transcript extractions**, all tail-verified before summarizing. **Evidence:** USER.md research-pattern window + transcript-continuity entry. **Action:** no change; the 08-19 guardrail is validated. (VTT-parser dedup fix remains pending — see Pending.)
- **Pattern:** **Multi-source cross-validation workflow** (YouTube → GitHub → arXiv, browser_evaluate + page.transcriptExtractor) produced a clean burst-then-trough cadence 08-12→08-27 with no new cut-offs, no new failures, and hardware/maker cluster now standing as a third research cluster (CNC, 3D printing, robotics, gym; 8+ sessions). **Evidence:** USER.md research patterns. **Action:** revisit cluster at next consolidation if it grows past ~10 sessions (per the standing note).

## Job 2 / Job 3 Handoff Notes

- **Job 2 (Knowledge Consolidation):** primary new material = signal #13 (append loop), #14/#15 (timeout/python3 patterns), #16 (stale branches), #17 (daily-note-lag false positive). The sweep-append-loop behavior (Positive #1) is the most important thing to encode *correctly* — it is not a failure, it is a resource leak in a well-behaved system.
- **Job 3 (Config Application):** action items in priority order: (1) kill-switch for the sweep incident-append loop + `.bak` retention policy (unblocks the `~/lloyd` repo noise); (2) `vault-maintenance` source-level timeout; (3) daily-note-lag check alignment (T-2 or evidence-gated); (4) delete `tmp-thunderbird-autonomy-20260824/` (staged in lloyd repo); (5) merge-or-delete `knowledge-writes-2026-08-26` (vault) and `nightly-improvement-2026-08-23` (lloyd).
- **Constraint carried forward:** NO `--apply`, NO recreation, NO mutation of the entity graph while the sweep writer is active. Any Job 3 action that touches `entity-resolution-sweep` must stop the writer first (task-level, not process-level).
