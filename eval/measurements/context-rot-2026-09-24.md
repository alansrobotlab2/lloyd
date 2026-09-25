# Context rot: Qwen3.8-Flash-Next, 2026-09-24 (P7)

**Verdict: inconclusive.** no length passes the rule in at least one shape; do not move the trigger on this run

**Reading (coordinator, 2026-09-25): keep `compaction.microcompact` at 0.72 / 0.52; no config change.**
The runner's strict rule returns *inconclusive* because the `session` shape
"fails at the base length". That failure is not rot: 10 of the 12 misses in
the whole run are `session`/`distract4` cells graded `undecided`, where the
model names both the pre-migration port and the current one and spends the
64-token answer budget explaining the conflict instead of answering. It is
as frequent at 50k (0.67) as at 240k (0.87) — flat in length, which is the
one thing a rot curve would not be. Every `single` and `multi3` cell is
1.0 up to 240k in both shapes (one multi3 miss at 240k), and `repo` holds
≥ 0.93 everywhere. Nothing here says Qwen3.8 degrades before the window
ends, so the plan's `L* ≥ 200k → keep 0.72` branch is the honest reading.
Cold TTFT is linear, ~4.1 s per 50k (20.8 s at 240k); warm ≤ 1.1 s.
Follow-up if the distractor condition should gate a future run: raise its
`max_tokens` or ask for the port alone, so hedging cannot read as a miss.
Raw rows: `eval/baselines/context-rot-2026-09-24.json`.

Runner: `eval/run_context_rot_eval.py`. Commit `b4ac86f684`. Engine: {"served": "Qwen3.8-Flash-Next-nvfp4", "root": "/home/alansrobotlab/lloyd/agent-services/llm/models/Inferact-Qwen3.8-Flash-Next-NVFP4", "max_model_len": 262144, "cache_config": {"cache_dtype": "fp8", "kv_cache_size_tokens": 844969, "block_size": 3200}}. Pool: pool already paused by ['operator']; leaving it alone.

## repo

| length | single | multi3 | distract4 | cold TTFT p50 | warm TTFT p50 | prompt p50 |
|---|---|---|---|---|---|---|
| 50,000 | 1.0 | 1.0 | 1.0 | 4.056 | 0.578 | 50312 |
| 100,000 | 1.0 | 1.0 | 1.0 | 8.371 | 0.626 | 100031 |
| 150,000 | 1.0 | 1.0 | 1.0 | 12.84 | 0.912 | 150101 |
| 200,000 | 1.0 | 1.0 | 0.9333 | 17.385 | 1.121 | 199851 |
| 240,000 | 1.0 | 1.0 | 1.0 | 20.762 | 1.085 | 240091 |

distract4 by depth:

| length | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 |
|---|---|---|---|---|---|
| 50,000 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 100,000 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 150,000 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 200,000 | 1.0 | 0.6667 | 1.0 | 1.0 | 1.0 |
| 240,000 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

L* (repo) = 150000; base A(50000) = 1.0; first fail: {"length": 200000, "A": 0.9333, "low_positions": {"0.3": 0.6667}}

## session

| length | single | multi3 | distract4 | cold TTFT p50 | warm TTFT p50 | prompt p50 |
|---|---|---|---|---|---|---|
| 50,000 | 1.0 | 1.0 | 0.6667 | 3.973 | 0.487 | 49571 |
| 100,000 | 1.0 | 1.0 | 0.6 | 8.202 | 0.621 | 100316 |
| 150,000 | 1.0 | 1.0 | 0.8667 | 12.444 | 0.742 | 149176 |
| 200,000 | 1.0 | 1.0 | 1.0 | 16.941 | 0.768 | 200852 |
| 240,000 | 1.0 | 0.9556 | 0.8667 | 20.474 | 0.93 | 239504 |

distract4 by depth:

| length | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 |
|---|---|---|---|---|---|
| 50,000 | 0.6667 | 0.3333 | 0.6667 | 0.6667 | 1.0 |
| 100,000 | 0.6667 | 0.3333 | 0.6667 | 0.6667 | 0.6667 |
| 150,000 | 0.6667 | 1.0 | 1.0 | 0.6667 | 1.0 |
| 200,000 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 240,000 | 1.0 | 1.0 | 1.0 | 0.3333 | 1.0 |

L* (session) = None; base A(50000) = 0.6667; first fail: {"length": 50000, "A": 0.6667, "low_positions": {"0.3": 0.3333}}

## Cost side (usage.db, last 14 days, read-only)

```
{
  "window_days": 14,
  "turns": 576,
  "busy_seconds": 110287,
  "busy_seconds_per_day": 7878,
  "peak_bands": {
    "0k-50k": 27,
    "50k-100k": 184,
    "100k-150k": 270,
    "150k-200k": 95,
    "200k-240k": 0,
    "240k-infk": 0
  },
  "current_trigger_tokens": 151303,
  "soak_baseline": {
    "days": 3,
    "turns": 576,
    "prefix_misses": 528,
    "reprefill_tokens": 51935681,
    "turns_with_misses": 164
  },
  "extra_compactions": null,
  "verdict": "not applicable: the trigger does not move"
}
```

Re-run whenever `models.primary.expect_model` changes.
