# #1483 — one djev read per turn as an intent gate / store router (2026-09-25)

**Verdict: rejected.** Nothing wired into prefetch; the harness
(`eval/run_prefetch_route_eval.py`, frozen question inside) is kept so a faster
djev or a better schema can be re-priced in one run. Raw:
`~/lloyd-data/eval/1483/route.json` (and `route_probe.json`, an earlier run).

## What was asked

One `choice` question over the user's message, negative first: none / web / vault
/ facts / sessions / skill. A SKIP gate (P(none) ≥ θ → no search-derived
`<context>`; ambient and IDE still render) and a store router (argmax names the
store) were priced against labelled turns, with the injected characters each turn
really gets from `prefetch_context`.

## Latency — the disqualifier

| set | n | djev p50 | p90 |
|---|---|---|---|
| gold (doc questions) | 79 | 313 ms | 324 ms |
| tool-choice prompts | 20 | 320 ms | 642 ms |
| real user turns with a skill | 51 | 339 ms | 1,128 ms |
| real user turns, no skill | 24 | 323 ms | 963 ms |
| ten acknowledgements | 10 | 80 ms | 81 ms |

The whole prefetch budget is 300 ms and the deployed turn lands at ~62 ms p50.
A read that costs 313–339 ms on an ordinary message does not fit the window: in
parallel it misses the wall and fails open (so it gates nothing), serially it
quintuples every turn's prefetch. Only the one-line acknowledgements come back
in ~80 ms, and those are exactly the turns prefetch already treats specially
(`MIN_MESSAGE_LEN`, `_CONTINUATION_RE`, no skill hint).

## Quality, had it been free

| θ (skip if P(none) ≥ θ) | gold skipped | tool-choice skipped | skill turns skipped (skill lost) | no-skill turns skipped | chars saved on the 75 real turns |
|---|---|---|---|---|---|
| 0.5 | 1/79 | 0/20 | 4/51 | 7/24 | 62,330 / 478,372 (13.0%) |
| 0.8 | 0/79 | 0/20 | 2/51 | 7/24 | 47,129 (9.9%) |
| 0.9 | 0/79 | 0/20 | 2/51 | 6/24 | 46,447 (9.7%) |
| 0.99 | 0/79 | 0/20 | 1/51 | 4/24 | 29,118 (6.1%) |

At every θ that spares the gold set the gate still drops a needed skill on 1–2 of
51 skill turns ("ok now run a test. lets hear some tts", P(none) 1.0). No
tool-choice prompt is ever skipped at θ ≥ 0.8, so that eval's rendered prompts —
and its result — are identical with the gate on; it was not re-run on the primary.

The store router is not usable: the argmax is `facts` on 67 of the 79 gold
questions, every one of which is answered by a vault document; `vault` on 4.

## Bottom line

~10% fewer injected characters on real turns, at 1–2 lost skills per 51 and a
~320 ms read that cannot fit the 300 ms window. Revisit only if djev's per-read
cost on a ~20-word state drops under ~100 ms.
