---
segment: agents
generated: 2026-08-29 01:18 PDT
data_range: 2026-08-26 to 2026-08-28
---

# Signal Report — 2026-08-29

**Pre-flight commits:** vault `49b6a829` (2 memory files), lloyd `8f81b947` (3 files). Data sources: daily notes 08-26/27/28 (near-empty stubs — no session content, no keyword hits), `memory/corrections.md`, `_pipeline/reflection/knowledge-health-2026-08-28.md`, `_pipeline/reflection/knowledge-write-error-2026-08-28.md`, `_pipeline/entity-resolution-sweep-incident-2026-08-22.md` (runs through 294), USER.md research/infrastructure/corrections sections.

## Queued Signals (met threshold)

### Explicit (act on first occurrence)

| ID | Date | Type | Category | Description | Source |
|----|------|------|----------|-------------|--------|
| 1 | 2026-08-28 | failure | pipeline-health | Reflection pipeline degradation continues: Jobs 3/4 (prompt audit, behavior test) still no artifacts for 3+ consecutive cycles; **08-28 cycle lost its #39a analysis handoff** (newest handoff was 26.7h old at the 08-28 01:44 run → Job 2 correctly skipped by freshness guard). This is the 4th cycle with a missing artifact somewhere in the chain (08-25 lost Job 1; 08-28 lost Job 39a). Escalation lives in backlog #383 items 2/3. | knowledge-write-error-2026-08-28.md |
| 2 | 2026-08-28 | escalation | entity-graph | **#48 incident: all 4 recovery gates still FAILING at run 294 (00:39 PDT)** — `_relationships.json` absent (217th verification), `memory-graph/` absent, no graph vs 12,131 baseline, writer fix unlanded (39th consecutive unpatched run — unguarded `shutil.rmtree` at `nightly_extraction.py:215-216`). Writer keeps landing 1 relation/pass into the destroyed graph (index 295, moving metric — **not a recovery claim**). Escalation re-sent 13th consecutive time, ~9.5h with no human action. Only remote backup restore or v4 rebuild remain viable; **pause-or-recover decision is the top-priority human action**. | incident file runs 293/294 |
| 3 | 2026-08-28 | pattern | tool-hygiene | **Incident file self-amplifying growth**: `entity-resolution-sweep-incident-2026-08-22.md` = 1.9 MB with **187 pre-append `.bak` files totaling 217 MB** inside gitignored `_pipeline/` (no backup covers it). Every #48 run appends + snapshots the whole file. Growth rate ~50-90 KB/run; at this rate the archive exceeds the file itself weekly. | disk inspection |
| 4 | 2026-08-28 | failure | task-config | **#48 task prompt stale for 37 consecutive runs** — prompt still carries run-120 text (cites pass 47, scale 20,782/52,063, "index frozen at 299", "run 120"). Incident-file run numbering is authoritative (now run 294). The prompt no longer describes the actual state; every run must re-derive it. | incident file run 294 |
| 5 | 2026-08-24/25 | correction | path-migration | **Health checks probed legacy paths after the 08-22 migration** — 08-24/08-25 KG checks looked at pre-migration locations; post-migration graph state lives at `~/lloyd/_pipeline/vault-derived/facts/`. Explicit rule written to USER.md: "check there first." Health-check/reporting tooling (knowledge-health-report.py et al.) needs path updates to match the post-migration layout. | USER.md infrastructure |

### Inferred (met 2+ threshold)

| ID | Date | Type | Category | Description | Frequency | Source |
|----|------|------|----------|-------------|-----------|--------|
| 1 | 2026-08-24→08-28 | pattern | memory-capture | **Session capture gap**: `trajectories/2026-08-26.jsonl` missing, `2026-08-25.jsonl` holds only 2 entries, `2026-08-24.jsonl` absent; daily notes 08-21/08-24 are write-gap stubs; 08-26/27/28 daily notes contain no session content despite 08-25 = 2 sessions and 08-26 = 3 sessions per the research-pattern record. Same family as backlog #383 diagnosis — narrow scope to why 08-24-type days drop capture and whether the 08-24 Thunderbird-bridge session is recoverable from other logs. | 3 days | USER.md memory-capture status |
| 2 | 2026-08-23 + 08-26 | pattern | tool-use | **youtube-transcript-api v1.2.4 API drift, 2nd variant in 7 days** (08-23: not-subscriptable; 08-26: `AttributeError: ... no attribute 'end'`). Both recovered with a single clean retry, but the library's snippet API is drifting under us. SKILL.md guardrail already added: v1.2.4 snippets expose `.text` only — never `.end`/`.start`; compute tail via max segment start time or oEmbed `lengthSeconds`. Second costed occurrence → reinforce the preamble (see Positive #2). | 2x in 7d | USER.md tool_reliability |
| 3 | 2026-08-26 | pattern | skill-discipline | **Read-skill-first discipline lapping**: 1 of 3 transcript sessions on 08-26 was skill-first (skills_search → SKILL.md before extraction); the non-skill-first session paid 1 retry on the `.end` attribute. Second costed occurrence — the cost is directly attributable to skipping SKILL.md. 2x threshold met. | 2x | USER.md tool_reliability |

## Pending Signals (below threshold)

- **VTT-parser tail-echo duplicate lines** — still pending fix (08-19 origin, re-flagged 08-24, zero new cut-offs since; guardrail held). 1 open defect, monitor.
- **QMD MCP subliminal/ search-result duplication** — open minor issue since 08-23 verification, no fix. 1 occurrence, monitor.
- **Chrome extension `manifest.json` gap** — build script expects a file that does not exist; unresolved since 08-23. 1 occurrence, monitor.
- **Run-scale measurement inconsistency in #48** (run 292's scale line internally inconsistent vs run 293's canary; dotdir-boundary handling wobbles between runs' find series) — self-corrected by run 293's adjudication; 1 occurrence, monitor for recurrence.

## Tool Failure Patterns

- **Tool:** youtube-transcript-api v1.2.4 — **Error type:** snippet-API drift (`.end` AttributeError; earlier not-subscriptable) — **Occurrences:** 2 in 7 days — **Recommendation:** SKILL.md guardrail already encoded (`.text` only; tail via max segment start / oEmbed `lengthSeconds`); reinforce the skill-first preamble so the guardrail is actually read before extraction (inferred signal #3).
- **Tool:** VTT parser fallback — **Error type:** tail-echo duplicate trailing lines — **Occurrences:** 1 (08-19, re-flagged 08-24) — **Recommendation:** dedup fix still pending; queue for a fix pass (low priority — zero cut-offs since guardrail).
- **Tool:** knowledge-health-report.py / KG health checks — **Error type:** legacy-path probing post-migration (checked pre-migration locations on 08-24/08-25) — **Occurrences:** 2 cycles — **Recommendation:** update health-check tooling to post-migration path `~/lloyd/_pipeline/vault-derived/facts/`; add a path-existence assertion so a wrong-root check is loud, not silent.
- **Tool:** autonomy task prompt (#48) — **Error type:** stale prompt (37 runs of run-120 text) — **Occurrences:** ongoing — **Recommendation:** refresh the task description to point at the incident file's latest run + the 4 recovery gates + the pending human decision; or have the prompt say "read the incident file tail for current state."

## Positive Patterns to Reinforce

- **Pattern:** **No-recovery-claim rule held** — handoffs/memory made zero false restoration claims for a 3rd consecutive cycle (08-24/08-25/08-26); the 08-26 disk re-check (23:15 UTC) explicitly recorded "No recovery claimed." The 08-24 overclaim incident's corrective rule is working. — **Evidence:** 3 clean cycles + explicit disk-verified checks — **Action:** encode as a permanent rule in the knowledge-write skill config (recovery claims require a same-cycle fresh health-check/disk verification).
- **Pattern:** **Transcript truncation guardrail held 08-19→08-26** — 17+ clean sessions, zero cut-offs; all 3 of 08-26's extractions (103/485/562 segments) tail-verified (outro/CTA/head+tail) before summarizing; the 08-26 API drift recovered in one retry. — **Evidence:** 17+ sessions — **Action:** guardrail is already in the youtube-transcript SKILL.md; reinforce with the skill-first preamble fix (inferred #3).
- **Pattern:** **Dry-run-before-apply against a broken baseline** — #48's guardrail reproduced byte-identical stderr (731 B, md5-verified) across runs 273–294, exit 1 direct-echo-verified, zero apply/mutation events against the destroyed graph. The "never `--apply` against a broken baseline" rule from the 08-22 incident held for ~7 consecutive days. — **Evidence:** 22+ consecutive clean dry runs — **Action:** preserve; candidate for a general skill (destructive-op pre-flight: verify baseline integrity + canary before any `--apply`).
- **Pattern:** **Root-cause triage before reset** — 3rd occurrence (Jun 27 GPU crash 29-item reset; 08-22 sweep poisoning ×3 with environmental-vs-config-vs-task-bug triage preceding reset; 08-28 incident forensics distinguishing writer passes from #48 activity via pgrep/btime). — **Evidence:** 3 sessions — **Action:** encode as a poisoned-worker/troubleshooting skill update.
- **Pattern:** **Post-migration/post-reinstall verification in one pass** — 08-22 systems check (11 supervisord services, 3 GPUs), 08-23 QMD check (server up + vault index currency + recent-topic probe), 08-26 KG disk re-check at the new path. — **Evidence:** 3 consecutive cycles — **Action:** keep the one-pass verification checklist (service → retrieval layer → data state); fold the post-migration path update into it.
- **Pattern:** **Research workflow cadence stable** — burst-then-trough is the dominant cadence (08-23 = 9-session peak → 08-25/26 troughs of 2-3); hardware/maker cluster now standing as a third research cluster (8 sessions 08-23/25/26, video-first + structured-summary workflow). — **Evidence:** 8 sessions across 3 days — **Action:** Job 2 — revisit at next consolidation if the cluster grows past ~10 sessions.
