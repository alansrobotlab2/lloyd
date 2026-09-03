---
segment: architecture
relations:
  related-to: []
tags: [architecture]
type: reference

---

# Nightly Reflection Architecture

**Last updated:** 2026-09-03 (rewritten from audited reality — the previous
version was truncated after Job 1, described five jobs at fixed clock times,
budgeted "<$25/night" against Claude Opus, and listed handoff files that have
never existed.)

The nightly chain turns the day's signals into durable changes: what Alan
corrected, what the system learned, and what should therefore change in memory
and config.

## The chain as it actually runs

```
#38 signals ──► #42 analysis ──► #39 knowledge write ──► #40 config
     22-04         22-04            23-04                  23-04

#56 trajectory extraction (01-02, independent)
     └─► #57 mining (23-04)      #51 relation linking (23-04)
```

All four reflection jobs run on the **local** model (`primary`), not Opus. There
is no dollar budget; the cost is GPU-hours.

| ID | Job | Reads | Writes |
|----|-----|-------|--------|
| #38 | Signals | `memory/corrections.md`, last 3 daily notes | `_pipeline/reflection/signals-latest.md` |
| #42 | Knowledge analysis | signals-latest.md | `_pipeline/reflection/knowledge-handoff-<date>.md` (validated by `scripts/validate_handoff.py`) |
| #39 | Knowledge write | the handoff | `memory/mental-models.md`, `people/alan/profile.md`, completion note |
| #40 | Config | signals + handoff | `~/lloyd/config.yaml`, `memory/learnings/<date>.md`, commits on `main` |

**Jobs 3 and 4 do not exist.** Earlier docs and #40's own data-load referenced a
prompt audit and a behavior test writing `prompt-audit-issues.md`,
`prompt-audit-latest.md`, `test-failures.md` and `test-results-latest.md`. No
autonomy task has ever written any of them. #40 no longer reads them.

## Scheduling

Windows are `preferred_hours` in machine-local time (PDT), not fixed clock
times, and each job also waits on `depends_on`. The chain takes roughly three
hours end to end when healthy. Before 2026-09-03 only #38 and #42 had windows,
so any upstream slip could push #39/#40 into the middle of the next day.

## The output contract (why this chain kept producing nothing)

Every job in this chain **claims its output file in its first turns**, then
enriches it in place, and flips a `status:` field to `complete` as its last act.

This is not stylistic. These jobs used to investigate exhaustively and write at
the very end, so a run that hit `max_turns` produced *nothing* while still
reporting success. On 2026-09-03 a 1429-second #38 run left `signals-latest.md`
untouched from two days earlier, and #42, #39 and #40 all consumed that stale
file believing it was current. #39 separately did its real writes and then died
during bookkeeping, so `mental-models.md` was genuinely updated while the record
said nothing had happened.

The recurring failure mode is looping on "one last verification pass" until the
turn limit kills the run. #42 is the reference implementation: a machine-checkable
output contract plus a validator it runs against itself.

A related trap: `compaction.microcompact` clears older tool results mid-turn, so
a job that re-reads what it already read can loop indefinitely. If a reflection
job reports "my earlier reads were cleared from context", that is this.

## Guardrails

- **Commit on `main`.** #40 commits config changes directly; it must start and
  end on `main` and merges any leftover `nightly-improvement-*` branch first.
  An earlier guardrail said the opposite and stranded 8+ commits on an unmerged
  branch.
- **Empty output is a failure**, recorded as one, with backoff. It used to be
  recorded as success and unblock the rest of the chain.
- **A failed upstream does not satisfy `depends_on`** — the gate reads
  `last_run`, which only successes set.
- `stale_bypass_hours` lets a downstream job run on stale input rather than
  block the chain forever. #38 and #40 set it; it is honoured only when the
  upstream is not currently running.

## Verification

Retrieval quality is measured, not assumed: task #82 runs the 20-query eval at
06:00 and records the trend, after #81's index maintenance at 05:00. If a night's
writes hurt recall, that is where it shows up.
