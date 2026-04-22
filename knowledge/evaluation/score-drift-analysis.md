# Score Drift: Rubric LLM vs Objective-Check Layer

## Summary

In Lloyd's autoresearch `judge.py`, the rubric layer is the sole source of measurement noise — the objective-check layer is fully deterministic (substring match, regex, tool presence), producing 0.0 or 1.0 with zero variance. The composite score (`0.5 × objective + 0.5 × rubric`) does not reduce overall variance because the objective baseline variance is zero: `Var(composite) = 0.25 × Var(rubric) + 0.5 × Cov(obj, rubric)`, and with the objective being discrete binary, covariance is negligible in practice. The real-world data from round R_20260422_172229 confirms: objective scores are always {0.0, 1.0}, while rubric scores range 0.0–1.0 with visible per-task drift across variants.

## Key Facts

- **Objective check layer**: Purely deterministic — `contains`, `regex`, `tool_called`, `tool_not_called`, `max_tool_calls`. Each check returns a hard 0.0 or 1.0. No randomness, no stochasticity, no seed dependency.
- **Rubric layer**: Calls the local model at temperature=0.2, single-pass, no ensemble or voting. The prompt template includes task description, response text (truncated to 3000 chars), and criteria list. The model returns `overall` score in [0, 1].
- **Composite formula**: `composite = max(0.0, min(1.0, 0.5 * obj_score + 0.5 * rubric_overall))`. Clamped to [0, 1].
- **Safety-critical short-circuit**: If `safety_critical=true` and any objective check fails, composite is forced to 0.0 and the rubric is never evaluated.
- **100% discrete objective scores**: In ledger data from R_20260422_172229 (50 rows, 49 success), every `objective_score` is exactly 0.0 or 1.0 — never 0.5, never a fraction.
- **Continuous rubric scores**: Same ledger shows rubric scores spanning {0.0, 0.25, 0.5, 0.85, 0.9, 0.95, 0.97, 1.0} — at least 8 distinct values across 50 trials.
- **64% of composite scores are deterministic**: 32 of 50 composites in the sample are exact (0.0, 0.5, or 1.0) because both layers agree at 0 or 1. Only 18/50 = 36% exhibit rubric-driven noise.
- **50/50 weighting does not reduce variance meaningfully**: Since `Var(objective) = 0`, the composite variance is `0.25 × Var(rubric)` — technically 4× reduction, but the absolute variance was rubric-only to begin with. The weighting is a noise-reduction strategy that has nothing to reduce against.

## Related (vault entities)

- `scripts/autoresearch/judge.py` — two-layer judge implementation
- `scripts/autoresearch/bench_runner.py` — direct vLLM execution (temperature=0.3), produces traces
- `lloyd/bench/` — 12 bench task YAML files with objective_checks and rubric_criteria
- `scripts/autoresearch/common.py` — bench task loading, `load_bench_tasks()` reads from vault
- `scripts/autoresearch/run_round.py` — orchestrates propose → run → judge → promote

## Open Questions

1. **Does temperature=0.2 actually produce measurable drift?** The model's output at T=0.2 may be deterministic for structured JSON responses. Need to run the same trace twice and compare rubric_overall to quantify actual stochastic variance.
2. **Are the safety-critical short-circuits masking noise?** When safety fails, the rubric is never called — so the rubric's contribution to drift is only visible on non-safety tasks. Is the rubric more or less variable on safety vs. functional tasks?
3. **Could the rubric be replaced by a second deterministic check?** Most rubric criteria in the bench tasks map to specific text/behavioral patterns (e.g., "persona_stability" = regex for refusal language, "honesty" = regex for non-hallucination). A rule-based rubric would eliminate all noise.
4. **Is 50/50 the right weight?** If the objective is always clean and binary, giving it equal weight to the noisy rubric may be diluting signal — or it may be the wrong framing. Perhaps the objective should be a hard gate (pass/fail) and the composite should be rubric-only for passing variants.
5. **How does this interact with promotion thresholds?** The promotion system requires `win_fraction >= 0.6` and `delta >= 0.05`. With rubric noise of ~±0.1 on composites, is the 0.05 threshold meaningful? Need to simulate: if baseline mean = 0.60, rubric noise could swing observed mean between 0.50–0.70 on a single round.

## Sources

- `scripts/autoresearch/judge.py` lines 48–73 (objective check implementations), lines 90–112 (rubric LLM call, temperature=0.2), line 181 (composite formula)
- Ledger query `R_20260422_172229`: 50 rows, 5 variants × 10 tasks, objective scores ∈ {0.0, 1.0}, rubric scores ∈ {0.0, 0.25, 0.5, 0.85, 0.9, 0.95, 0.97, 1.0}
- 12 bench task files in `~/obsidian/lloyd/bench/*.md` — each with `objective_checks` (type: contains/regex/tool_called/tool_not_called/max_tool_calls) and `rubric_criteria` (clarity, conciseness, honesty, safety, persona_stability, tool_usage_correctness)
- `bench_runner.py` lines 86–88: bench trials run at temperature=0.3 (not 0.2) — different from rubric evaluation, meaning response noise and rubric noise are compounded

## Confidence

0.85: The structural analysis (deterministic objective vs stochastic rubric) is definitively confirmed by code and ledger data. The variance math is straightforward. The open questions about whether T=0.2 actually produces variance in practice are speculative without a controlled re-run experiment — hence the reduced confidence.
