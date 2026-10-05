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

The runner prints the engine before its first call and writes no report while any arm
row of any sample is unjudged. Offline: nothing outside `eval/` and `tests/` imports
it, and no number here has been measured yet — the ~200-call pass is owed (#2254).
