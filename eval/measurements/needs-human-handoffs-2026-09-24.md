# Needs-human hand-offs that were undone (2026-09-24)

Follow-up to the #904 measure: most `needs-human` hand-offs the reconciler
made were reversed within minutes. Diagnosis, fix, and a read-only replay of
the fixed rule over the live ledger (`~/.local/state/lloyd-automod/promotions.jsonl`,
19,102 rows, last row 2026-09-24T20:11Z).

## Definitions

- **Hand-off**: a `status_moved` row to `draft` whose reason is the
  reconciler's spent park ("… a human decides", or the infra park "every
  implement turn this item got failed on the stack being down"). These are
  the moves that put `needs-human` on an item.
- **Reversal**: the item's first later row among `backlog_retriage`,
  `backlog_umbrella_unfolded`, `round_abandoned` (the reaper then
  `set_status(up_next)` with no ledger row: the "unrecorded" path),
  `backlog_implement phase=reopened` (`reopen_item`), a `started` turn, a
  `status_moved` back to `up_next`/`in_progress`, or a close.

These are not the #904 script's exact definitions (it counted 75 of 103
reversed: 33 re-triage, 13 reopen, 23 unrecorded); the re-triage path matches
it exactly at 33.

## Week to 2026-09-24T20:11Z: 103 hand-offs, 95 undone, median 12.9 min

| reversal path | n | what the fixed rule does |
|---|---|---|
| reaper `round_abandoned` → `up_next` (unrecorded) | 27 | 26 held `in_progress`: the round was still open when the hand-off was made |
| loop re-triage | 42 | 33 held (second life owed); 7 still handed off (deferral to an open blocker, re-triaged once it closed); 2 not reproduced |
| `reopen_item` (a person) | 12 | 10 still handed off; 2 not reproduced |
| moved back to `in_progress` | 5 | 5 held `in_progress` |
| closed (landed later) | 6 | 6 held `in_progress` (the round was mid-landing) |
| umbrella unfold | 3 | 2 held (an umbrella's unfold is always owed); 1 not reproduced |
| **not undone (stuck)** | 8 | **8 still handed off — none lost** |

**72 of the 103 hand-offs would not have been made** (37 under a live round,
35 ahead of an owed second life). All 8 hand-offs that stuck are still made,
as are the 10 reopens a person acted on. 9 hand-offs (flagged
`old-does-not-reproduce`) are not reproduced by the pre-fix code on the
prefix either, because the code or the ledger shape moved since (e.g. the
external/infra re-offer rules of 09-18 and later).

30 days: 236 hand-offs, 228 undone, 108 withheld, 8 stuck all still handed
off (only 141 reproduce under today's pre-fix code: the older ledger predates
several outcome rules, so the 7-day window is the one to read).

## Evidence, one per path

- **Round alive (unrecorded reaper reversal)**, #1143, 2026-09-18: turn
  `finished` 02:46:23Z on SM_20260918_024457 while its detached gate ran
  (preflight 02:45:44, tests 02:54:19, review 03:04:16); hand-off 02:46:25Z;
  the reaper abandoned the round at 03:19:37Z and set `up_next` with no row;
  next turn started 03:42:25Z. Same shape #435 today: hand-off 18:02:23Z,
  tests rung 18:04:33Z, review 18:11:42Z, abandoned 18:11:46Z.
- **Re-triage owed**, #1240, 2026-09-18: round abandoned 15:42:00Z, the
  turn-end reconcile handed it off 15:42:02Z, housekeeping re-triaged it
  15:42:40Z (38 s). #1149: hand-off 01:15:08Z, re-triage 01:36:16Z.
- **Reopen by a person**, #1151 (hand-offs 09-18, 09-21, 09-23, reopened each
  time hours to days later) — real decisions; still handed off.
- **Stuck**, #1342 and #731 (09-21) deferred to each other; re-triage waits on
  an open deferral target, so the cycle only a person can break. Kept.

## Cause

`desired_statuses`' spent branch parked every `spent` outcome
`draft` + `needs-human`, consulting neither of the two rules the second-life
passes already use:

1. `items_with_unfinished_rounds` (latest row `started`, a live promoted round,
   the round in `current.json`, its worktree, a live gate or land marker).
   `retriage_spent_items` and `unfold_spent_umbrellas` skip those items; the
   reconciler parked them. Its own in-flight check covered `started` and a live
   land marker only, never an open worktree or a running gate.
2. Whether a second life is owed. The turn-end `reconcile_statuses` in
   `autocode.execute` runs without the re-triage pass beside it, so a first
   spend was tagged and then untagged at the next housekeeping pass.
   `_escalate_review_disagreement` already refused to announce "needs you"
   while that was owed (`_second_life_owed`); the reconciler did not ask.

## Fix

In the spent branch, before tagging: an item in `items_with_unfinished_rounds`
stays `in_progress`; an item whose second life is owed
(`backlog.second_life_owed`, lifted out of `autocode._second_life_owed`, which
now delegates) goes to `draft` untagged — unless it is waiting on an open
deferral target (`_open_deferral_targets`), which keeps today's hand-off.
`retriage_enabled` (the `retriage_spent` switch) is passed by housekeeping,
the turn-end reconcile and `round board-pass`; with it off a first spend is a
person's at once, as before.

Replay: `eval/measurements/needs_human_handoff_replay.py`.
