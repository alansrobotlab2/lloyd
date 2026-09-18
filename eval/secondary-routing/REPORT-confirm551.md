# Secondary engine: quality on the work it is routed to (item #551)

- run: `2026-09-16T00:17:29+00:00` · repeats: 3 · items per job: {'title': 4, 'capture': 4, 'facts': 4, 'focus': 4, 'voice': 4} · arms: secondary vs primary
- tolerance: secondary keeps a job within **5.0** composite points and **+0.05** defect rate of the primary
- judge pass: off

Composite = format compliance (30) + anchor recall (30) + length budget (25) + uniqueness (15). Defect = over-long or a near-duplicate line inside one output.

- arms separate: True (every arm reached a distinct engine endpoint)
- router agrees with this measurement: True (router and measurement agree: on_primary=['title'])
- repeats are warm: repeat 2 and 3 replay a prompt the engine has already cached, so wall seconds favour the later repeats equally in both arms and `max` is the cold one. The primary's hit rate is on the dashboard (`prefix_cache_hit_rate`); the secondary's is reported `null` because llama.cpp's cached-token counter cannot be turned into a rate (f7028ce).
- `n` is trials, `items × repeats` is inputs replayed: the decision floor is on repeats (`3`), not on `n`, so a wide run of one repeat each still cannot route anything.

## session title  (`title`)

| arm | n | items × repeats | score mean ± spread | format ok | defect rate | wall s | out tok | tok/s |
|---|---|---|---|---|---|---|---|---|
| secondary | 12 | 4 × 3 | 66.25 ± 45.0 (σ 17.81) | 58% | 0% | 0.17 (max 0.25) | 9.6 | 55.1 |
| primary | 12 | 4 × 3 | 79.58 ± 10.0 (σ 3.8) | 100% | 0% | 0.16 (max 0.19) | 7.7 | 47.1 |

**Decision: `flip_to_primary`** — outside tolerance: +13.3 pts (margin 5.0) — the 1.1x latency cost is not buying parity


## post-session capture  (`capture`)

| arm | n | items × repeats | score mean ± spread | format ok | defect rate | wall s | out tok | tok/s |
|---|---|---|---|---|---|---|---|---|
| secondary | 12 | 4 × 3 | 90.42 ± 20.0 (σ 5.94) | 100% | 0% | 0.63 (max 0.81) | 70.8 | 112.6 |
| primary | 12 | 4 × 3 | 80.83 ± 60.0 (σ 19.24) | 83% | 0% | 0.57 (max 0.8) | 64.4 | 112.8 |

**Decision: `keep_secondary`** — within tolerance: -9.6 pts vs a 5.0-pt margin, defect +0.000 vs a +0.050 margin, at 1.1x the primary's wall time


## durable fact extraction  (`facts`)

| arm | n | items × repeats | score mean ± spread | format ok | defect rate | wall s | out tok | tok/s |
|---|---|---|---|---|---|---|---|---|
| secondary | 12 | 4 × 3 | 77.08 ± 45.0 (σ 16.26) | 67% | 0% | 0.68 (max 1.09) | 73.8 | 109.0 |
| primary | 12 | 4 × 3 | 72.92 ± 50.0 (σ 17.38) | 75% | 0% | 0.56 (max 0.78) | 74.7 | 134.2 |

**Decision: `keep_secondary`** — within tolerance: -4.2 pts vs a 5.0-pt margin, defect +0.000 vs a +0.050 margin, at 1.2x the primary's wall time


## focus/topic extraction  (`focus`)

| arm | n | items × repeats | score mean ± spread | format ok | defect rate | wall s | out tok | tok/s |
|---|---|---|---|---|---|---|---|---|
| secondary | 12 | 4 × 3 | 85.0 ± 30.0 (σ 10.61) | 100% | 0% | 0.33 (max 0.35) | 23.5 | 72.1 |
| primary | 12 | 4 × 3 | 86.25 ± 30.0 (σ 10.83) | 100% | 0% | 0.25 (max 0.29) | 15.8 | 62.0 |

**Decision: `keep_secondary`** — within tolerance: +1.2 pts vs a 5.0-pt margin, defect +0.000 vs a +0.050 margin, at 1.3x the primary's wall time


## voice summary  (`voice`)

| arm | n | items × repeats | score mean ± spread | format ok | defect rate | wall s | out tok | tok/s |
|---|---|---|---|---|---|---|---|---|
| secondary | 12 | 4 × 3 | 85.29 ± 12.5 (σ 4.96) | 100% | 0% | 0.59 (max 0.7) | 61.2 | 102.9 |
| primary | 12 | 4 × 3 | 82.58 ± 22.5 (σ 5.73) | 100% | 0% | 0.41 (max 0.51) | 40.7 | 98.1 |

**Decision: `keep_secondary`** — within tolerance: -2.7 pts vs a 5.0-pt margin, defect +0.000 vs a +0.050 margin, at 1.4x the primary's wall time


## Routing

- on primary now (`app.secondary_models.JOBS_ON_PRIMARY`): `['title']`
- this run recommends on primary: `['title']`
- router agrees with the measurement: **True** — router and measurement agree: on_primary=['title']
- a flip is executed by adding the job to `JOBS_ON_PRIMARY` in `app/secondary_models.py` and re-running this eval: the secondary arm lifts the pin and the primary arm sets it, so both arms keep measuring the same router and a confirmation run stays comparable with the one that recommended the flip.
- after a flip, the downstream check is one command: `python3 eval/secondary_routing_eval.py --downstream <YYYY-MM-DD>` — the over-long/duplicate rate over the 7 days ending that date against the 7 before it, read from the auto-captured entries in the daily notes.

---

Interpretation guard: the composite is mechanical, so it measures shape and grounding, not truth. Use the judge lines as corroboration only — the video this came from showed a harness pairing beating a bigger model, so a single number is not a winner. The latency ratio in each decision is the price being paid for whatever parity is claimed.
