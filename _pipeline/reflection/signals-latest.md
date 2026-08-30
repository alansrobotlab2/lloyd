---
segment: agents
generated: 2026-08-30 02:17 PDT
data_range: 2026-08-27 to 2026-08-29
---

# Signal Report — 2026-08-30

## Pipeline Health (meta — Job 1 self-assessment)

- **Job 1 restored (4th consecutive-cycle gap closed).** 08-29 cycle artifacts all present: this report, `knowledge-handoff-2026-08-29.md`, `knowledge-health-2026-08-29.md` (all dated 08-29 02:0x PDT). The "Job 1 no artifacts on 08-25" flag from the 08-26 cycle is now resolved; Job 1 ran 08-26 through 08-29.
- **6 days of USER.md writes were lost, then recovered.** On-disk `~/obsidian/lloyd/USER.md` (mtime 08-26 23:06) contains no corrections_log entries for 08-28, 08-29, or 08-30 — those writes only exist in the 08-30 01:56 pre-flight commit `d496729a` in the lloyd repo (branch `nightly-improvement-2026-08-23`, 8 commits ahead of main, 12 files, 457 insertions). Disk and repo diverged.
- **Job 3 (prompt audit) now 4+ consecutive cycles without artifacts** (last audit 03-31). Job 4 (behavior test) last produced output 03-29. Escalation unchanged: backlog #383 items 2/3.
- **Morning brief degraded run persists** — 08-29 08:18 PDT run: 0 emails (vs ~15-21/day baseline), 0 calendar events, 3/4 health checks failed; 08-30 08:40 and 09:07 runs healthy (10 emails, 1 event, 5/5 checks). State file confirms `degraded_runs: 0` after two healthy runs, but the single degraded day is unexplained — IMAP/Thunderbird bridge gap between ~08-29 02:00 and 08-30 08:40.

## Queued Signals (met threshold)

### Explicit (act on first occurrence)

| ID | Date | Type | Category | Description | Source |
|----|------|------|----------|-------------|--------|
| 1 | 2026-08-30 | failure | infrastructure | Job 1/Job 2 artifacts (USER.md writes, handoff, health report) written to a file/dataset layer, not the on-disk vault — 6 days of corrections_log entries (08-28→08-30) exist only in the lloyd-repo git tree, not on disk at `~/obsidian/lloyd/USER.md`. Recovery-claim rule (08-24) implies: Job 2 must verify on-disk vault state after write, not just commit | lloyd repo diff vs on-disk file; vault pre-flight commit `d496729a` |
| 2 | 2026-08-29 | failure | infrastructure | `entity-resolution-sweep` now writes pre-run `.bak` snapshots of the incident file (2.2MB) on every attempt — 243 `.bak` files totaling 307M in `~/lloyd/_pipeline` (whole dir 756M). Writer-fix behavior is working but the backup strategy is unbounded; sweep still blocked on the path-scope exclusion part of the fix | `_pipeline` dir listing; incident file |
| 3 | 2026-08-29 | correction | pipeline-health | Job 3 (prompt audit) has produced no artifacts for 4+ consecutive nightly cycles (last: 03-31). Job 4: no output since 03-29. This is the same family as the 08-25 Job 1 loss — the nightly reflection pipeline has per-job failure modes that silently skip. Needs a job-level artifact-existence check (each job asserts its own output file exists at cycle end) rather than relying on downstream jobs noticing | reflection dir mtimes; backlog #383 items 2/3 |

### Inferred (met 2+ threshold)

| ID | Date | Type | Category | Description | Frequency | Source |
|----|------|------|----------|-------------|-----------|--------|
| 1 | 2026-08-27 | tool-use | youtube-transcript | 3rd distinct youtube-transcript-api v1.2.4 failure mode in 7 days: first-attempt extraction of F1XqqVNBa4o ("Dual-Memory Optimization for Self-Improving AI Agents") died with a Python traceback (`LENGTH_S=unknown` then exception in the script); recovered on retry. Combined with 08-23 (not-subscriptable) and 08-26 (`.end` AttributeError) — the SKILL.md guardrail (snippets expose `.text` only) is insufficient; the API surface keeps shifting | 3x distinct failure modes, 08-23/08-26/08-27 | trajectory `2026-08-27.jsonl` (ivfe0c, 2 sessions); USER.md tool_reliability |
| 2 | 2026-08-28 | tool-use | morning-brief | Morning-brief data-source flakiness: 08-29 08:18 run was fully degraded (0 emails/0 events, 3/4 health checks), then 08-30 runs fully healthy with no config change. Unexplained 24h IMAP/Thunderbird-bridge gap — same post-migration infrastructure fragility family as the 08-22/08-24 Thunderbird bridge restoration | 1 degraded day, 2 healthy days after | morning-brief state.json; trajectory `2026-08-28.jsonl` (4 degraded-health checks: cpu/memory/disk + health) |
| 3 | 2026-08-28 | pipeline-health | trajectory-extraction | Trajectory files are partial or missing again: `2026-08-28.jsonl` holds only the 09:05 morning-brief entry (missing any 08-28 user sessions — though 08-29's daily note shows the 08:52 and 10:44 PDT sessions landed on 08-29's file by timezone); `2026-08-29.jsonl` and `2026-08-30.jsonl` missing. Same residual gap the 08-26/08-27 cycles flagged; watermark (24711) shows the extractor is running, so this is a date-bucketing/late-write issue, not a dead pipeline | 3 consecutive cycles flagged (08-26, 08-27, now) | trajectories dir; .watermark.json |

## Pending Signals (below threshold)

- Vault working tree carries 5 dirty `autonomy/*.md` files with `failure_count`/`status`/`updated` churn (e.g. task #38: `up_next` → `in_progress`, `failure_count` 3→4) — task-manager runtime state leaking into version-controlled skill docs. 1 occurrence observed, monitor; if it recurs after the next pre-flight commit, needs a `.gitignore` for runtime frontmatter or a runtime-state carve-out
- New `personal/poems/` vault directory created by the 08-29 08:52 session (James McCrae poem) — fine, but note the assistant unilaterally created a new top-level vault segment; no user pushback recorded. Monitor for a stated preference
- 08-29 nylon/PCT 3D-printing research session (13:47 PDT) — hardware/maker cluster session count now ~9; approaching the ~10-session threshold where it becomes a standing third cluster per the 08-26 research-pattern entry
- QMD `subliminal/` search-result duplication — still no resolution recorded in window; minor, no new occurrence

## Tool Failure Patterns

- **Tool:** `youtube-transcript-api` (v1.2.4) — **Error type:** API-drift: first-attempt Python traceback on F1XqqVNBa4o (`LENGTH_S=unknown` → exception) — **Occurrences:** 1 (but 3rd distinct drift variant in 7 days) — **Recommendation:** stop patching per-error-variant in SKILL.md; pin to a single known-good extraction path or vendor the transcript fetch so the tool's internal surface changes stop biting Lloyd. The read-skill-first discipline (08-26 signal) is working but can't out-run an unstable API
- **Tool:** morning-brief IMAP/calendar sources — **Error type:** degraded run 08-29 08:18 (0 emails, 0 events, 3/4 health checks failed), self-recovered 08-30 — **Occurrences:** 1 — **Recommendation:** when a morning-brief run reports degraded, log the specific failing source (IMAP vs calendar vs health checks) into the run report — current reports don't say which leg broke, making 24h-gap diagnosis impossible
- **Tool:** entity-resolution-sweep writer — **Error type:** `.bak` backup growth unbounded (243 files / 307M in ~8 days; incident file alone has snapshots every attempt) — **Occurrences:** continuous — **Recommendation:** rotate: keep last N (e.g. 3) `.bak` per source file, prune the rest; part of the same writer-fix backlog item as the path-scope exclusion
- **Tool:** trajectory extractor — **Error type:** date-bucketing gap — sessions land in wrong/missing daily jsonl (08-28 file has only the 09:05 MB entry; 08-29 and 08-30 files missing) — **Occurrences:** 3 cycles — **Recommendation:** check whether the extractor buckets by session-start vs session-end timezone and whether late-flushed sessions drop out of the day file

## Positive Patterns to Reinforce

- **Pattern:** youtube-transcript retry-and-verify discipline — first-attempt failure on 08-27 recovered cleanly with one retry, extraction completed (1463 lines saved), session proceeded to note-write without user intervention — **Evidence:** 08-27 (1), extends the 17+ clean-session streak from 08-19→08-26 across the window — **Action:** keep the SKILL.md guardrail; escalate to the pin/vendor recommendation above only if a 4th distinct drift variant appears
- **Pattern:** graceful fallback when standard extraction fails — 08-29 08:52 Substack poem: standard extractor failed, assistant fell back to raw HTTP fetch, saved the artifact, created a sensible vault location, and offered relocation instead of forcing a choice — **Evidence:** 1 clean session, zero user pushback — **Action:** note as a working pattern (fallback chain + non-pushy placement offer); no skill change needed
- **Pattern:** recovery-claim rule held (5th consecutive cycle) — KG still at 0 relationships / `_relationships.json` missing; the 08-29 health report and handoff both recorded the destroyed state without a recovery claim; `entity-aliases.json` correctly reported as degraded (14,487 self-mapped placeholder entries) rather than "repaired" — **Evidence:** 08-29 knowledge-health + handoff, verified against disk 08-30 — **Action:** preserve; recovery plan unchanged (remote backup or `classify-relationships-v4.py` rebuild with pre-apply edge-count sanity check vs the 12,131 baseline)
- **Pattern:** incident-file preservation discipline — the 2.2MB `entity-resolution-sweep-incident-2026-08-22.md` (updated 08-30 02:15) survived the writer change with pre-write backups intact; the "never `--apply` against a broken baseline" rule from 08-22 held through the sweep's continued dry-run attempts — **Evidence:** incident file + 243 intact `.bak` snapshots — **Action:** preserve; the same discipline is now the model for the new Job 1/2 on-disk verification requirement (Explicit signal 1)
