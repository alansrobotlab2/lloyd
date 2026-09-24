# Board decisions: the first reading (#904, 2026-09-24)

`scripts/automod/board_decisions.py` joins every promotion into `up_next` to
the ledger event that made it and to what became of it. Read-only over the
live `~/.local/state/lloyd-automod/promotions.jsonl` (19,102 rows) and the
lloyd board, 2026-09-24 20:45Z. Reproduce with
`python -m scripts.automod.round board-decisions [--days N] [--summary]`.

## Where it lives (the item's placement question, decided)

A deterministic function in `scripts/automod`, surfaced through
`backlog.board_health` as `decisions` — so the dashboard's backlog payload (and
one line on the backlog panel) and the board steward's `<board_health>` block
carry it with no new job, route or poll — plus `round board-decisions` for the
per-item listing. Not a seventh autonomy job and not an extension of #76: both
put a model turn in front of what is a join over data already on disk, and #76
already times out when the fleet is in trouble. Cost: 29 ms warm per call,
`board_health` 0.61 s → 0.65 s (median of 5), inside the existing 60 s TTL.

## 7 days (2026-09-17 20:45Z → 09-24 20:45Z)

| promotions | 427 | triage 383 · unrecorded 23 · reopen 13 · reconcile 5 · gate 2 · group_triage 1 · steward 0 |
|---|---|---|
| per day | 2 30 88 114 90 35 32 36 | no zero-promotion day |
| terminal | landed 237 · closed off the ledger 87 · needs-human 68 · closed 14 · returned 12 · still open 9 | |
| done rate | 79.2% (landed 55.5%) | median dwell 14.9 h |

| retired in window | retired | later reopened | after the loop's re-triage | by `reopen_item` |
|---|---|---|---|---|
| needs-human | 103 | **75** | 33 | 13 |
| landed (closed) | 278 | 2 | 0 | 0 |
| closed by the loop | 109 | 0 | 0 | 0 |
| parked by the sweep | 8 | 0 | 0 | 0 |

30 days: 807 promotions, done 66.5% (landed 41.8%), dwell 12.9 h; needs-human
175 of 262 reversed, parked 5 of 68, closed 0 of 397, expired 0 of 13. The 30-day
longest zero streak (13 d) is the stretch before the status pipeline existed
(09-09), not a hold.

## What it says

- **Closing retirements are never reversed** (0 of 109 in 7 d, 0 of 410 in 30 d
  counting expiry). The sweep's and triage's retire calls hold up.
- **A needs-human hand-off is not a retirement in practice**: 73% come back into
  the pool within the week, a median 48 min after the hand-off (42 of 74 dated
  reversals inside an hour). 33 went through the loop's own
  second life (`retriage_spent_items`), 13 were a `reopen_item`, and 23 were a
  round starting on an item no ledger event had put back (`unrecorded`) — the
  status reconciler handing an item to a person while its round was still in
  review (e.g. #1143 on 09-18: `spent` at 02:46, review at 03:04, round
  re-started 03:42). That flapping is worth an item of its own; this measure
  only makes it visible.
- **Autonomy #35 is invisible to the ledger**: it promotes through the MCP
  store, which writes no ledger row, so its picks can only show up as
  `unrecorded` entries when a round takes them. The 7-day `unrecorded` 23 is
  dominated by the flapping above, so #35's own contribution cannot be
  separated from this data. The steward runs `apply: false` and has promoted 0.
- **87 promotions closed off the ledger** — the 09-24 hand sweeps — count as
  done with no terminal timestamp; they inflate the 7-day done rate from 58.8%
  (ledger-only) to 79.2%.

No threshold is set: what rate, dwell or reopen count is wrong enough to act on
needs a week of traffic and Alan's judgment (the item's second human clause).
