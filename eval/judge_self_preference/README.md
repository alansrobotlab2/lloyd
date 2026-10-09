# `judge_self_preference` — does the judge prefer its own family?

```bash
cd ~/lloyd
.venvs/lloyd/bin/python eval/judge_self_preference/run_arms.py \
    --raw $LLOYD_DATA/eval/judge_self_preference/arms-$(date -u +%Y-%m-%dT%H%M%SZ).jsonl \
    --report $LLOYD_DATA/eval/judge_self_preference/report.md
```

`durable_write_judge`'s corpus re-scored in four arms that differ only in one
attribution line above the note: `repeat` (the baseline prompt sent a second time —
the engine's own floor), `none`, `self`, `human`. Judge, rubric, examples and sample
text are byte-identical across them by construction. Every delta is read against this
run's own `none` arm, because the published 32.4% was judged on a different model.

The runner prints the engine before its first call, gives an unjudged arm row exactly
one more call, and writes no report while any arm row of any sample is still unjudged
after that retry (#2465). Offline: nothing outside `eval/` and `tests/` imports it.

**Measured 2026-10-09: 200 calls on `primary`, and the result is negative.** The `self`
arm flagged *more* than the unattributed one (pass rate on `bad` 65.0 % against 75.0 %),
so no self-preference effect was found; the finding and the per-arm figures are recorded
beside the 32.4 % figure it clears, in `eval/durable_write_judge/report.md`. Raw rows:
`~/lloyd-data/eval/owedcheck-2254/arms-final.jsonl`, report
`~/lloyd-data/eval/owedcheck-2254/REPORT.md`.

Two things that pass taught, both now pinned in `tests/test_judge_self_preference.py`.
The judging budget is `JUDGE_MAX_TOKENS = 1200`, because at the 400 this directory
shipped with six of the 200 replies truncated mid-JSON and the run correctly refused to
report; and `run_arms.engine_call` has to match `judge.call_engine`'s real contract,
because it did not — it passed a `temperature` keyword the engine never took and unpacked
a 2-tuple from a dict, so the first pass could only be driven from outside this repo.
