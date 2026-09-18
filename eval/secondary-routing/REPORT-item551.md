# Secondary engine: quality on the work it is routed to (item #551)

- run: `2026-09-16T00:16:06+00:00` · repeats: 3 · items per job: {'title': 4, 'capture': 4, 'facts': 4, 'focus': 4, 'voice': 4} · arms: secondary vs primary
- tolerance: secondary keeps a job within **5.0** composite points and **+0.05** defect rate of the primary
- judge pass: off

Composite = format compliance (30) + anchor recall (30) + length budget (25) + uniqueness (15). Defect = over-long or a near-duplicate line inside one output.

- arms separate: True (every arm reached a distinct engine endpoint)
- router agrees with this measurement: True (no decision recorded yet (eval has not been run at the repeat floor))
- repeats are warm: repeat 2 and 3 replay a prompt the engine has already cached, so wall seconds favour the later repeats equally in both arms and `max` is the cold one. The primary's hit rate is on the dashboard (`prefix_cache_hit_rate`); the secondary's is reported `null` because llama.cpp's cached-token counter cannot be turned into a rate (f7028ce).
- `n` is trials, `items × repeats` is inputs replayed: the decision floor is on repeats (`3`), not on `n`, so a wide run of one repeat each still cannot route anything.

## session title  (`title`)

| arm | n | items × repeats | score mean ± spread | format ok | defect rate | wall s | out tok | tok/s |
|---|---|---|---|---|---|---|---|---|
| secondary | 12 | 4 × 3 | 62.92 ± 45.0 (σ 18.54) | 50% | 0% | 0.23 (max 0.49) | 9.2 | 39.6 |
| primary | 12 | 4 × 3 | 80.0 ± 10.0 (σ 3.54) | 100% | 0% | 0.17 (max 0.19) | 7.8 | 46.4 |

**Decision: `flip_to_primary`** — outside tolerance: +17.1 pts (margin 5.0) — the 1.4x latency cost is not buying parity


## post-session capture  (`capture`)

| arm | n | items × repeats | score mean ± spread | format ok | defect rate | wall s | out tok | tok/s |
|---|---|---|---|---|---|---|---|---|
| secondary | 12 | 4 × 3 | 90.83 ± 20.0 (σ 6.07) | 100% | 0% | 0.82 (max 1.32) | 74.2 | 90.6 |
| primary | 12 | 4 × 3 | 80.42 ± 55.0 (σ 18.76) | 83% | 0% | 0.63 (max 1.46) | 62.1 | 98.5 |

**Decision: `keep_secondary`** — within tolerance: -10.4 pts vs a 5.0-pt margin, defect +0.000 vs a +0.050 margin, at 1.3x the primary's wall time


## durable fact extraction  (`facts`)

| arm | n | items × repeats | score mean ± spread | format ok | defect rate | wall s | out tok | tok/s |
|---|---|---|---|---|---|---|---|---|
| secondary | 12 | 4 × 3 | 81.25 ± 45.0 (σ 16.35) | 75% | 0% | 0.85 (max 1.25) | 76.2 | 89.5 |
| primary | 12 | 4 × 3 | 72.5 ± 50.0 (σ 19.53) | 75% | 0% | 0.63 (max 1.17) | 68.5 | 108.3 |

**Decision: `keep_secondary`** — within tolerance: -8.8 pts vs a 5.0-pt margin, defect +0.000 vs a +0.050 margin, at 1.3x the primary's wall time


## focus/topic extraction  (`focus`)

| arm | n | items × repeats | score mean ± spread | format ok | defect rate | wall s | out tok | tok/s |
|---|---|---|---|---|---|---|---|---|
| secondary | 12 | 4 × 3 | 85.0 ± 30.0 (σ 10.61) | 100% | 0% | 0.36 (max 0.47) | 23.5 | 64.6 |
| primary | 12 | 4 × 3 | 86.25 ± 30.0 (σ 10.83) | 100% | 0% | 0.29 (max 0.59) | 15.8 | 53.8 |

**Decision: `keep_secondary`** — within tolerance: +1.2 pts vs a 5.0-pt margin, defect +0.000 vs a +0.050 margin, at 1.2x the primary's wall time


## voice summary  (`voice`)

| arm | n | items × repeats | score mean ± spread | format ok | defect rate | wall s | out tok | tok/s |
|---|---|---|---|---|---|---|---|---|
| secondary | 12 | 4 × 3 | 85.71 ± 12.5 (σ 5.12) | 100% | 0% | 0.59 (max 0.88) | 58.9 | 100.3 |
| primary | 12 | 4 × 3 | 84.62 ± 24.0 (σ 6.87) | 100% | 0% | 0.49 (max 1.1) | 40.8 | 83.1 |

**Decision: `keep_secondary`** — within tolerance: -1.1 pts vs a 5.0-pt margin, defect +0.000 vs a +0.050 margin, at 1.2x the primary's wall time


## Routing

- on primary now (`app.secondary_models.JOBS_ON_PRIMARY`): `[]`
- this run recommends on primary: `['title']`
- router agrees with the measurement: **True** — no decision recorded yet (eval has not been run at the repeat floor)
- a flip is executed by adding the job to `JOBS_ON_PRIMARY` in `app/secondary_models.py` and re-running this eval: the secondary arm lifts the pin and the primary arm sets it, so both arms keep measuring the same router and a confirmation run stays comparable with the one that recommended the flip.
- after a flip, the downstream check is one command: `python3 eval/secondary_routing_eval.py --downstream <YYYY-MM-DD>` — the over-long/duplicate rate over the 7 days ending that date against the 7 before it, read from the auto-captured entries in the daily notes.

---

Interpretation guard: the composite is mechanical, so it measures shape and grounding, not truth. Use the judge lines as corroboration only — the video this came from showed a harness pairing beating a bigger model, so a single number is not a winner. The latency ratio in each decision is the price being paid for whatever parity is claimed.
