# #1484 — mid-turn ephemeral recall from the tool stream (2026-09-25)

**Verdict: rejected** (time-boxed scoped probe; nothing wired).

## The question

Would appending ≤3 fresh vault hits every N tool calls, queried from the turn's
own tool stream, put in front of the model documents it goes on to need? The
KV-prefix concern is moot by construction — an appended `state_anchor` never
touches the system prompt or earlier history — so the question is only whether
the appended tokens buy anything.

## Method (`eval/run_midturn_recall_probe.py`, offline, no model)

The 60 newest sessions on disk with a turn of ≥ 20 tool calls (their longest
turn each; mostly autocode/autotriage workers, the long-turn population the item
is about). Every 10 calls: query = the model's own captions for those 10 calls +
the basenames of the paths they touched; prefetch's hybrid leg
(`_search_vault`, lex+vec, the ~55 ms call); keep ≤3 hits the turn had not
already seen (turn-start `<vault-context>`, anything it read, earlier anchors;
#1511 self-hits dropped). A hit is **anticipated** when a later call in the same
turn names that document — the model fetched it anyway, so appending it earlier
could have saved the fetch. This is a floor on usefulness (a hit the model would
have used but never fetched is invisible); the turn-start prefix's own vault hits
are scored the same way as the reference.

Raw: `~/lloyd-data/eval/1484/probe.json`.

## Result

| | n | used later | rate (Wilson 95%) |
|---|---|---|---|
| mid-turn anchor hits | 249 hits over 144 anchors, 60 turns | 4 | **0.016** (0.006–0.041) |
| turn-start prefix hits (reference) | 28 | 0 | 0.000 (0–0.121) |

137 of 144 anchors found something "new", so the anchor would fire almost every
time, appending a mean 1,604 chars per long turn; 4 of 60 turns ever went on to
fetch an appended document. Query cost p50 136 ms (run beside another eval's qmd
load).

## Why rejected

Precision 1.6% means ~98% of what the anchor appends is never touched — cost
without a measured benefit, on the one population it was meant for. The
reference row says the turn-start prefix is no better by this floor, which is a
finding about prefix usefulness for worker turns (worth P0's "use" metric), not
a case for adding a second, per-10-calls copy of it. #1481 owns the long-turn
loss evidence; nothing here argues for reopening this before P0 (#1480) exists.
