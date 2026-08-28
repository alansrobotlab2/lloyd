---
segment: agents
generated: 2026-08-27 22:58 PDT
data_range: 2026-08-25 to 2026-08-27
---

# Signal Report — 2026-08-27

## Queued Signals (met threshold)

### Explicit (act on first occurrence)

| ID | Date | Type | Category | Description | Source |
|----|------|------|----------|-------------|--------|
| 1 | 2026-08-26 | tool-failure | transcript | youtube-transcript-api v1.2.4 second drift variant: `AttributeError: 'FetchedTranscriptSnippet' object has no attribute 'end'` on first fetch attempt (session 20260826_230120_ivdd9f, one-person-business video, 562 seg). Cost = 1 retry; recovered using `.start`-only tail computation. Guardrail already added to youtube-transcript SKILL.md: v1.2.4 snippets expose `.text` only; never `.end`/`.start`; compute tail via max segment start time or oEmbed `lengthSeconds` | derived sessions + trajectory |
| 2 | 2026-08-25→27 | failure | data-pipeline | vault-derived session store truncates ALL files at ~2,014 chars mid-sentence (17/17 files across 08-25/26/27 end mid-word). Daily-note auto-captures inherit the truncation — 3 of 8 sessions show "response appears truncated" or end mid-sentence (08-26 11:34, 16:02; 08-27 16:41). Root cause is a fixed output buffer, NOT the LLM: the 08-27 daily note itself was written at 16:41 by a live session, so the truncation happens in the capture/store layer after generation. Downstream consumers (this job, daily notes, memory-capture) all degrade silently. | derived sessions + daily notes |
| 3 | 2026-08-27 | infra | pipeline | `entity-resolution-sweep` task running every ~6 min (226–235 runs/day) appends a full ~1MB `FileNotFoundError: _relationships.json` traceback block to `_pipeline/entity-resolution-sweep-incident-2026-08-22.md` each failed cycle. File grew from 502KB (01:43) to 1.5MB (07:08) to 2.4MB (21:45) on 08-27 and keeps growing ~2.5MB/cycle-hour. File is gitignored (no size guardrail anywhere). The task cannot succeed while `_relationships.json` is destroyed (08-22 incident), so it will keep hammering at 4x/hour forever until the KG is recovered or the task is disabled/deduped. Needs: (a) dedup the error block to one per state-change, (b) cap or truncate the file, (c) backoff/disable the schedule while the graph is unrecovered. Escalation candidate for backlog #383 / #378. | pipeline dir mtimes + incident file tail |
| 4 | 2026-08-26/27 | failure | data-pipeline | Trajectory capture incomplete: `trajectories/2026-08-27.jsonl` does not exist (watermark stopped 08-26 23:38; 08-27 has 8 sessions with 3 tool errors — the only day in range with zero trajectory coverage). `2026-08-26.jsonl` holds only 3 entries (17,419 bytes) for 3 sessions. This is the same capture-gap family as the 08-24 stub daily notes — diagnosis belongs in backlog #383's narrowed scope (why do specific days drop capture). | trajectories dir |

### Inferred (met 2+ threshold)

| ID | Date | Type | Category | Description | Frequency | Source |
|----|------|------|----------|-------------|-----------|--------|
| 5 | 2026-08-26/27 | pattern | tool-use | Non-skill-first transcript sessions pay the drift tax: 08-26 23:01 (ivdd9f, no skills_search) burned 1 retry on the `.end` drift error; 08-27 09:21 (ivfe0c) had no skills_search and its derived file shows the traceback before the clean fetch. 08-26 20:56 (iv8eca, skill-first: skills_search → skills_read → extract) had 0 errors. 2 of 3 transcript sessions skill-first in the window; the non-compliance correlated with the error. Reinforce the SKILL.md preamble (2nd costed occurrence per 08-26 note). | 2 costed occurrences | derived sessions + trajectories |

## Pending Signals (below threshold)

- Daily-note auto-capture truncation (signal 2) is already explicit via the store evidence, but the *fix* (buffer size) is unconfirmed — monitor until a full-length derived file appears to confirm the buffer is the cause, not the generator.
- oEmbed `lengthSeconds` probe for the 08-27 16:41 Adam Savage session (417 lines saved to /tmp/yt/oMnpatXgMzU.txt) returned a valid oEmbed response — the completeness-check fallback path works; 1 occurrence, no action needed.

## Tool Failure Patterns

- **Tool:** youtube-transcript-api (pinned venv) — **Error type:** `AttributeError: 'FetchedTranscriptSnippet' object has no attribute 'end'` (second drift variant after the 08-23 not-subscriptable failure) — **Occurrences:** 1 (08-26, 23:01) — **Recommendation:** guardrail encoded in SKILL.md; skill-first discipline (signal 5) is the preventive control; consider pinning the venv to a tested minor version if a third variant appears.
- **Tool:** entity-resolution-sweep (scheduled task) — **Error type:** `FileNotFoundError: _relationships.json` appended unboundedly, ~1MB/cycle — **Occurrences:** 226–235 failed cycles on 08-27 alone — **Recommendation:** dedup error log + file cap + disable schedule while KG unrecovered (signal 3). This is the 4th occurrence of the root-cause-triage-before-reset pattern (Jun 27 GPU crash → 29 items; 08-22 poisoning 3×; this unbounded append) — the pattern is now standing, encode it as a skill if #383 adds a reset task.
- **Tool:** vault-derived session store — **Error type:** silent truncation at ~2,014 chars — **Occurrences:** 17/17 files in 08-25→27 window — **Recommendation:** find the fixed buffer in the capture layer; add a truncation marker (`[truncated]`) so downstream jobs can detect it; this job's own inputs were degraded — flag to #383 diagnosis.
- **Tool:** trajectory extraction — **Error type:** missing daily file (08-27) / partial file (08-26) — **Occurrences:** 2 — **Recommendation:** part of the 08-24-type capture-gap family; same #383 scope.

## Positive Patterns to Reinforce

- **Pattern:** skill-first transcript extraction — skills_search → skills_read(youtube-transcript) → single-shot fetch with pinned venv → tail verification. — **Evidence:** 08-26 20:56 session (iv8eca): 0 errors, 6 tools, clean read of the 81-char SEGCOUNT marker; 08-27 09:21 (ivfe0c): 938 segments, tail-verified (outro reached), zero cut-offs. — **Action:** reinforce SKILL.md preamble (already planned per 08-26 note); the correlation with the non-skill-first cost (signal 5) justifies it.
- **Pattern:** truncation guardrail holding — zero transcript cut-offs 08-19 → 08-27 (9th clean day). 08-27's 938-segment extraction (largest of the window) tail-verified on first pass. — **Action:** no change; the guardrail is now 9 days proven, eligible to be treated as baseline behavior rather than a watch item at next consolidation.
- **Pattern:** ambient_decide surfacing well — 08-26 20:56 session correctly surfaced the time-sensitive RSSC meeting event to the user with a reasoned message (ties to the 09-05 presentation commitment). — **Evidence:** 1 clean use with good reasoning quality. — **Action:** none needed; monitor.
- **Pattern:** batch multi-session research on the Recurse paper (08-27 09:21 + 14:30, dual-memory harness memory layer + results +23 points across GPT-5.6/Qwen 3.6) — same structured video-first + summary workflow, no corrections, no cut-offs. — **Evidence:** 2 clean sessions, user built on both without pushback. — **Action:** research content belongs in projects/ai/research/ at next consolidation (memory-capture scope, not this job's).

## Notes for Job 2 (Knowledge Consolidation)

- Signal 2 (store truncation) is the most consequential finding of this cycle: it silently degrades every downstream consumer including the daily notes and this job's own inputs. If Job 2 writes derived-session facts from these files, they are truncated — prefer the daily-note auto-captures (shorter, less likely to hit the buffer) or re-read from source when possible.
- Signal 3 (unbounded incident-file growth) is an active disk/health hazard independent of the KG recovery work; recommend Job 3 or the backlog carry the dedup+cap+disable fix.
- Research content for 08-27 (Recurse paper) and 08-26 (DeepSeek harness, SFP+, one-person business) is queued for projects/ai/research/ per the standing consolidation pattern.
