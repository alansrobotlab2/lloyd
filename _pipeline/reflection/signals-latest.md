---
segment: agents
generated: 2026-08-22 22:07 PDT
data_range: 2026-08-19 to 2026-08-22
---

# Signal Report — 2026-08-22

> **CRITICAL — user attention required:** the 08-22 nightly next-gen extraction destroyed the entity graph. Full incident file: `~/lloyd/_pipeline/entity-resolution-sweep-incident-2026-08-22.md` (writer attribution, damage inventory, recommended actions). This must reach Alan, not just the pipeline.

## Queued Signals (met threshold)

### Explicit (act on first occurrence)

| ID | Date | Type | Category | Description | Source |
|----|------|------|----------|-------------|--------|
| 1 | 2026-08-22 | incident | pipeline | **Nightly next-gen extraction destroyed the entity graph.** `_relationships.json` (12,131 edges / 7,260 active) deleted with no backup; `memory-graph/` (merge history, 107-cluster hand-review artifact) deleted; `entity-aliases.json` truncated mid-write (5,532 → 1,198 keys, repaired, corrupt original preserved); `facts/` tree wholesale-rebuilt (66,124 → 1,837 dirs, ~97,700 facts). No backup system covers `_pipeline/`; gitignored → no local recovery. Writer: nightly extraction bg tasks 16:13–17:47 PDT (attributed via bg-*.log forensics). Detected by entity-resolution-sweep #48 dry-run crash (FileNotFoundError). | incident file (170 lines) |
| 2 | 2026-08-22 | tool-error | tool-use | **Task frontmatter `timeout_seconds` is NOT read by pool.py** — the effective cap is the source-level `max_duration_seconds`. #48 sweep (needs ~30 min) poisoned 3× at the old 1200s cap. Fixed: `scheduled-task` bumped 1200 → 1800 (config.yaml, committed `699b2d17`, comment documents the frontmatter-ignored fact). Encode this in a skill so the next task author doesn't set a frontmatter timeout and assume it works. | session `ive8da` + config.yaml |

### Inferred (met 2+ threshold)

| ID | Date | Type | Category | Description | Frequency | Source |
|----|------|------|----------|-------------|-----------|--------|
| 1 | 2026-08-22 | pattern | tool-use | **Poisoned-worker reset, 2nd occurrence with a distinct root cause** (Jun 27 GPU crash → 29 items, environmental; 08-22 timeout cap → 3 items, config). Both times: identified root cause before reset, no blind retry. Pattern = root-cause triage (environmental vs config vs task-bug) precedes reset. | 2x | session + USER.md tool_reliability |
| 2 | 2026-08-19/21 | pattern | pipeline | **Stale/missing job outputs flagged 3 consecutive runs** (prompt-audit 03-31, behavior-test 03-29/04-07, memory-capture.log 06-03) — each run defers to "a one-time autonomy-task diagnosis pass" and the pass never runs. Deferral is itself the failure now. | 3x flags | daily notes 08-19/08-20/08-21 |

## Pending Signals (below threshold)

- Read tool `not found` on absolute-path read (1 occurrence, 08-18) — no new occurrence this window; continuing to monitor, 2nd triggers a guardrail note in `read-not-found-handling`.
- `http_search` rate-limit (08-19, 1 occurrence, 0 retries) — no recurrence; closing.
- Entity-sweep guardrail "don't regenerate tables from empty sections without confirming source data is genuinely empty" (08-18, 1 occurrence, held last run) — no recurrence this window; still below threshold.

## Tool Failure Patterns

- **Tool:** entity-resolution-sweep dry-run — **Error type:** `FileNotFoundError: _relationships.json` (external cause: graph deleted by nightly writer) — **Occurrences:** 1 (08-22) — **Recommendation:** NOT a sweep fix. Restore from remote backup if one exists, else rebuild via `scripts/memory/classify-relationships-v4.py`; sanity-check edge counts against pre-incident baseline (12,131) before any `--apply` run.
- **Tool:** worker pool (scheduled-task source) — **Error type:** task timeout poisoning (3× at 1200s cap for a ~30-min task) — **Occurrences:** 3 — **Recommendation:** fixed (1800s cap, committed). Residual action: encode the frontmatter-ignored fact (signal 2) as a skill/guardrail so timeouts are set at source level.
- **Tool:** nightly extraction writer (next-gen memory pipeline) — **Error type:** destructive wholesale rewrite of `facts/` tree + unscoped deletion of graph files + mid-write truncation of `entity-aliases.json`, with no pre-write backup — **Occurrences:** 1 (critical severity) — **Recommendation:** add pre-write `.bak` + path-scope exclusion (must never touch `_relationships.json` or `memory-graph/`) before the next nightly run; sweep already backs up — the writer should match. **Gate the next sweep `--apply` behind this fix.**

## Positive Patterns to Reinforce

- **Pattern:** Incident forensics + safe containment (08-22) — dry-run crash → log/mtime forensics attributed the deletion to an external writer → sweep took no destructive action → repaired the truncated alias table while preserving the corrupt original → produced a complete incident file (evidence, damage inventory, recommended actions). — **Evidence:** 1 session (08-22) — **Action:** encode as a skill (`pipeline-forensics` / incident-report template): on unexpected file loss, (1) attribute the writer before touching anything, (2) never `--apply` against a broken baseline, (3) repair with corrupt originals preserved, (4) write the incident file.
- **Pattern:** Root-cause fix over blind retry (08-22) — 3 poisoned worker attempts diagnosed as a cap defect (not a task bug); fixed the source-level cap and documented it in the config comment instead of raising retries. — **Evidence:** 1 session — **Action:** reinforce (consistent with poisoned-worker reset pattern in USER.md tool_reliability).
- **Pattern:** KG classifier clean resume (08-21) — 2,556 candidate edges → 0 new / 2,556 cache hits in 351s (resume fingerprint clean), 38 upgrades landed with pre-apply backup, edge count verified 12,093 → 12,131. — **Evidence:** 1 large verified run — **Action:** maintain; backup-before-apply + verified-count reporting is the working part.
- **Pattern:** Truncation guardrail holding — no new transcript cut-offs in the window (08-19/08-20 notes verified complete; 08-21 research session clean). — **Evidence:** 3 clean runs since guardrail added — **Action:** maintain (guardrail closed the 5-cut-off loop; no further action).

## Carried-Forward Open Items (for Jobs 2/3)

- `memory-capture.log` stale (last entry 2026-06-03) — **3rd consecutive flag**, never checked. Needs the one-time check at the next autonomy-task diagnosis pass.
- Stale job outputs: `nightly-behavior-test` (03-29), `prompt-audit` (03-31), `vault-propagate` state (03-29/04-07) — **3rd consecutive flag**, deferred 3×. Recommend Jobs 2/3 stop deferring and schedule the diagnosis pass as an explicit backlog task.
- Incident #1 (08-22) recovery is the top priority for the next autonomy pass; the sweep's `--apply` run must stay blocked until graph recovery + writer fix land.
