---
id: 79
name: Retention sweep
status: up_next
frequency: daily
description: Sweep every unbounded store under the pipeline root; LEDGER_ARCHIVE_AGE_DAYS
  (14) for the automod ledger, WORKTREE_DIR_MAX_AGE_DAYS (7) for a worktree left by
  a finished round, VOICE_TURNS_MAX_AGE_DAYS (90, deliberately not the 30 the file
  stores use, because a voice turn is small and an audio turn is huge), and sessions
  deleted >30d on mtime (BACKGROUND_SESSION_ARCHIVE_AGE_DAYS) when idle.
---

# Retention sweep

Fixture for `tests/test_constant_quotes.py` (#2317): the three ways a task
description states a script's window, in the spellings
`autonomy/79-retention-sweep.md` really uses.

`LEDGER_ARCHIVE_AGE_DAYS (90)` in the front matter above would be the drift this
guard exists for — but it is not in the front matter above, and the numbers in the
Activity Log below are prose the check must never read.

## Activity Log

- 2026-09-20 — ran with LEDGER_ARCHIVE_AGE_DAYS (90) and deleted 11 ledgers
