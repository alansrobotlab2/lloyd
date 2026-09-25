# Rationale-per-line curation of loaded memory (#1488): dry run 2026-09-25

**Verdict: proposal + tooling landed; applying the skill patch is owed to a human.**
The sleep-time "what to know today" arm was **not built**, because the channel
the item names cannot carry it (see the last section).

## What is here

- **`scripts/maintenance/vault-user-md-rationale-curation.patch`**, the vault
  half. It changes the `nightly-reflection-knowledge-write` skill in two ways:
  - Every USER.md line the job adds gets a ledger row in
    `lloyd/memory/user-md-ledger.md`. That is a topic file, never loaded. The
    row records why the line is loaded, its origin, a checkable `retire_when`,
    the `check` command, and a `checked` date.
  - A new Step 2a-ter runs when USER.md is within 1 KiB of its ceiling. It
    backfills at most 10 missing rows, runs the `check` of the 10 stalest rows,
    and archives (never deletes) at most 5 lines a night. A line is archived
    only when its check output shows its `retire_when` has happened. Each
    archived line cites that output and its uptake row. The procedure is in a
    new sibling file, `step-2a-ter-curation.md`.
  - `git -C ~/obsidian apply --check` passes against vault HEAD (the skill's
    last commit is `1a72649f`).
- **`scripts/memory/memory_ledger.py status`** (tracked code, read-only) is the
  deterministic half. It reports which loaded lines lack a row, which rows no
  longer name a loaded line, the headroom left, and the stalest-checked rows.
  It covers USER.md and the MEMORY.md index.
  - Rows are anchored on the first 60 characters of the entry. That anchor had
    0 collisions in both files.
  - It reads the `- [project] (date)` prefix that `memory_add` puts on a topic
    entry.
  - Tests: `tests/test_memory_ledger.py`.
- **`eval/run_user_md_curation_dryrun.py`** is the curator's first pass,
  one-shot on the primary, run against copies. Its `score` subcommand reports
  bytes and tokens freed and scores a hand audit.

## Dry run: copies of USER.md (16,368 / 16,384 B) and MEMORY.md (20,466 / 25,600 B)

Raw outputs are in `~/lloyd-data/eval/1488/`. The run was one call per file,
with thinking off and temperature 0.

| | entries | ledger rows | sidecar size | proposed retire | proposed relocate | bytes if applied | tokens if applied |
|---|---|---|---|---|---|---|---|
| USER.md | 55 | 55 | 8,677 B | 0 | 1 (176 B) | 16,368 → 16,192 | 4,088 → 4,042 |
| MEMORY.md | 85 | 85 | 11,458 B | 3 (674 B) | 22 (4,782 B) | 20,466 → 15,010 | 5,848 → 4,210 |

**The ledger backfill works mechanically but is mostly not checkable.**
- Every entry got a row.
- But 43 of 55 USER.md `retire_when` conditions, and 31 of 85 MEMORY.md ones,
  are "X changes" with no path, number or item a check could test.
- Both sidecars fit under the 32 KiB topic ceiling, and neither costs a
  prompt byte.

**Without disk evidence the curator is blind to exactly the lines this exists
to catch.** It retired nothing in USER.md. I checked the conditions it wrote
itself against disk, and **4 kept USER.md lines are already false**:

| # | line (abridged) | what disk says |
|---|---|---|
| 2 | header: "No loader enforces it (`app/memory_ceiling` is absent)" | `app/memory_ceiling.py` exists and refuses over-ceiling memory writes |
| 17 | "Interrupting speech is click-only… no acoustic barge-in" | `livekit.barge_in` on since 2026-09-24 (config.yaml); its own `retire_when` was "acoustic barge-in is implemented" |
| 18 | "retraining is blocked (training tree lost)" | `69a52d2f` retrained `hey_lloyd.onnx`; the model filenames changed |
| 36 | KG store at `~/lloyd/_pipeline/vault-derived/kg.sqlite` | moved to `~/lloyd-data/…` on 2026-09-22; `~/lloyd/_pipeline` does not exist |

MEMORY.md's `[feedback] Interrupting voice output is click-only` line is stale
the same way, and was also kept.

Lines 17 and 18 alone are 756 B of retirable text. That is 47× today's 16 B of
headroom. Line 2 is 672 B and line 36 is 679 B; both need rewriting rather than
retiring.

**Audited proposals:**
- USER.md relocate: 1 of 1 right.
- MEMORY.md retire: 2 of 3 right (Wilson [0.208, 0.939]). #51 is wrong: it is a
  standing "do not carry these two false claims" correction.
- MEMORY.md relocates were not audited line by line. MEMORY.md has 5.1 KB of
  headroom, so the patched step would not run on it tonight.

**What that means for the design:** a curator that only reads the file does not
make headroom. What does is running each row's `check` with tools, which the
patch requires and which is why it keeps a line whose check cannot be run. The
rationale ledger's value is that it tells the curator which command to run. So
the backfill prompt in the nightly step has to demand a *checkable* condition.
The dry run shows a model does not produce one unprompted (43/55 vague). The
patch's row format and Step 2 say so.

**No behaviour bench was run.** The dry-run curator produced no curated USER.md
worth an A/B (one relocated pointer line). #1425's 20-probe A/B already
measured that removing single loaded lines moved nothing past its noise floor
(2/20 both arms).

## The sleep-time arm, not built

`app/sessions_io.enqueue_ambient_prefetch` is an in-memory queue keyed by
**session id**. It lives in the backend process and is drained by that
session's next turn.
- A nightly pass has no session to address: tomorrow's chat does not exist yet.
- A backend restart drops the queue.

A "what to know today" note would need a new file-backed next-session channel
read by `prefetch`. That is a code change whose value only P0 (#1480) could
measure. It is recorded on the item rather than built blind.

## Owed to a person

- Apply the patch: `git -C ~/obsidian apply scripts/maintenance/vault-user-md-rationale-curation.patch`.
  Then watch the first curation run's completion note. It changes what a
  nightly job may remove from a loaded file, so it is not trivially safe.
- The four stale USER.md lines and the stale MEMORY.md feedback line above are
  real today whether or not the patch is applied. Correcting them is a vault
  edit this sweep was not allowed to make.
