---
segment: agents
generated: 2026-08-18 23:26 PST
data_range: 2026-08-16 to 2026-08-18
---

# Signal Report — 2026-08-18

**Input data status:** Daily notes for 08-16 (changelog only) and 08-18 (no session) were sparse. 08-17 has 4 auto-captured sessions (Qwen 3.8 27B ×2, FreeCAD tutorial ×2). `corrections.md` has NO new entries — last update 2026-05-08 (stale ~3.5 months). This run is driven almost entirely by extraction logs + the 08-19 daily note (autonomy task #48, ran 08-18 23:19 PST).

## Queued Signals (met threshold)

### Explicit (act on first occurrence)

| ID | Date | Type | Category | Description | Source |
|----|------|------|----------|-------------|--------|
| 1 | 2026-08-18 | data-integrity | tool-use | Entity-alias table has 2,677 pre-existing entries failing the `normalize_full(alias) == normalize_full(canonical)` invariant (unchanged this run, old=new=2,677). Look like note-filename/title-mapping aliases from another writer (e.g. `2026-04-03.md` → `2026-04-03`). Flagged in daily note as "needs human review of who writes `entity-aliases.json`". Action: identify all writers of entity-aliases.json; decide which writer owns title-mapping aliases or quarantine that entry class | 2026-08-19 daily note (task #48, 06:19Z) |

### Inferred (met 2+ threshold)

| ID | Date | Type | Category | Description | Frequency | Source |
|----|------|------|----------|-------------|-----------|--------|
| 1 | 2026-08-05/09/17 | pattern | tool-use | Transcript-extraction cut-off: video summary truncated mid-sentence/mid-explanation (FreeCAD bike stem 08-17; prior hits 08-05 TencentDB ×2, 08-09 Robotics Tech). Recurring across 4 distinct sessions over 2.5 weeks. Action: add length guard to the transcript summary pipeline — detect truncation (summary doesn't end cleanly / source transcript not fully covered) and either continue extraction or flag the note as partial | 4x cumulative (1x this window) | extraction (08-17 session) + prior notes in USER.md |
| 2 | 2026-08-12/17 | pattern | tool-use | yt-dlp selected for transcript pull then fails (missing JS runtime — no node in environment). Wrong tool choice for this environment; fallback to youtube-transcript-api recovered. Action: drop yt-dlp from the transcript fallback chain entirely in this environment (already documented in USER.md; this is a re-occurrence confirming the rule should be a hard skill guardrail) | 2x | extraction (08-17 session) + 08-12 record |

## Pending Signals (below threshold)

- **corrections.md staleness** — no entries since 2026-05-08 while daily notes + extraction logs carry continuing signal volume (this run proves it). 1 occurrence, monitor: if next run also finds zero new corrections.md entries, treat corrections.md as a dead input and stop pre-flight loading it (or wire daily-note corrections into it).
- **Entity-resolution: 96 AMBIGUOUS clusters** — all SUFFIX_AMBIGUOUS, stable set across 08-19 passes (02:11Z, 03:53Z, 23:19Z); norm-key diff 0/0. Byte-identical = no drift, nothing to act on yet, but top clusters (NVIDIA d=106+11, CLAUDE d=53, Isaac GR00T d=42) are candidates for manual alias decisions. Monitor.

## Tool Failure Patterns

- **Tool:** yt-dlp — **Error type:** missing JS runtime (no node) — **Occurrences:** 2 (08-12, 08-17) — **Recommendation:** hard guardrail in transcript skill: never select yt-dlp for transcripts in this environment; chain is youtube-transcript-api → browser_evaluate(page.transcriptExtractor) → VTT parsing.
- **Tool:** transcript-summary generation — **Error type:** output truncated mid-sentence — **Occurrences:** 4 (08-05 ×2, 08-09, 08-17) — **Recommendation:** fix — post-hoc truncation check (final sentence incomplete or transcript tail uncovered) + one continuation pass; if still truncated, mark the vault note `[partial extraction]`.
- **Tool:** entity-resolution sweep (task #48) — **Error type:** in-session edit accidentally truncated the 96-row ambiguous table — **Occurrences:** 1 — **Recommendation:** guardrail — sweep sessions must regenerate derived markdown tables from the source jsonl (`entity-merges-*.jsonl`), never hand-edit; this run self-corrected (regenerated, verified 96 rows).

## Positive Patterns to Reinforce

- **Pattern:** Transcript fallback chain (yt-dlp fails → youtube-transcript-api recovers) completed the FreeCAD tutorial extraction end-to-end — the fallback chain encoded from the 08-16 signals run is working in production — **Evidence:** 08-17 session (Qwen/FreeCAD batch); plus 08-16 config change — **Action:** no new encoding needed; the 2× yt-dlp re-failure supports making the skip-yt-dlp rule a hard skill guardrail instead of a soft note.
- **Pattern:** Verify-then-apply autonomy workflow for entity merges: dry run → apply SAFE only → post-apply dry run → byte-level diff against prior passes → backups before mutation → no restarts performed when not needed — **Evidence:** 08-19 task #48 session (2 merges applied cleanly, 96-row artifact verified, 2 backups taken) — **Action:** encode as a guardrail reference in the entity-resolution/autonomy skill: the dry-run-verify-backup-apply sequence is the template for batch vault mutations.
- **Pattern:** Self-recovery from in-session artifact corruption — the truncated 96-row table was detected by the post-apply verification and regenerated from source jsonl with row-count verification — **Evidence:** 1 session (08-19 task #48) — **Action:** reinforce: "verify derived artifacts against source data after every batch mutation" as an explicit step in sweep-type autonomy tasks.

## Notes for downstream jobs

- Job 2 (Knowledge Consolidation): the 4 auto-captured 08-17 research sessions (Qwen 3.8 27B, FreeCAD 1.1) are research-content, not signal-content; their knowledge already surfaced in prior USER.md consolidation (Qwen 3.8 27B "DeepSeek moment", FreeCAD "make internal lines" fix). Do not re-propagate.
- Job 3 (Config Application): no config changes warranted this run. Signal 1 (entity-alias writers) is an investigation task, not a config change; consider queueing it as a backlog item rather than acting tonight.
