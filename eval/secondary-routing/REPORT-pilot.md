# Secondary engine: quality on the work it is routed to (item #551)
> **Superseded by `REPORT-item551.md`, and kept as evidence of the defect this round
> fixed.** The `flip_to_primary` for `voice` below came from 4 items at **1 repeat
> each**: the `decide()` of that run compared a *trial count* against the 3-repeat
> floor, and 4 trials cleared it. It is not a routing decision and the router does
> not act on it — the same run at 3 repeats per item (`REPORT-confirm551.md`) routes
> `title` instead. `defect rate` is 0.000 on every arm here, so this pilot also
> contributed no over-long/duplicate evidence to any verdict.

- run: `2026-09-11T08:21:04+00:00` · repeats: 1 · items per job: {'voice': 4} · arms: secondary vs primary
- tolerance: secondary keeps a job within **5.0** composite points and **+0.05** defect rate of the primary
- judge pass: on (reported only, never decisive)

Composite = format compliance (30) + anchor recall (30) + length budget (25) + uniqueness (15). Defect = over-long or a near-duplicate line inside one output.

## voice summary  (`voice`)

| arm | n | score mean ± spread | format ok | defect rate | wall s | out tok |
|---|---|---|---|---|---|---|
| secondary | 4 | 66.5 ± 40.0 (σ 15.71) | 75% | 0% | 0.54 (max 0.7) | 52.2 |
| primary | 4 | 72.5 ± 10.0 (σ 4.33) | 100% | 0% | 0.93 (max 1.11) | 32.8 |

**Decision: `flip_to_primary`** — outside tolerance: +6.0 pts (margin 5.0) — the 0.6x latency cost is not buying parity

- judge (primary engine, corroboration only): secondary mean 2.0/2
- judge (primary engine, corroboration only): primary mean 1.75/2
