# Skills index with descriptions (P5), 2026-09-25

**Verdict: ship `desc_push` (`skills.index.descriptions: true`); do not ship
`desc_pull` (`prefetch.skills.push` stays true).**

Runner: `eval/run_prefetch_cost_eval.py --queries eval/skill_match_queries.yaml
--n 100 --arms injected,injected_aa,desc_push,desc_pull`, 51 labelled turns ×
4 arms, sandboxed `pt-eval-` sessions, primary with the worker pool paused,
blind judge. Raw: `eval/baselines/skills-index-2026-09-25.json`.

Paired differences against `injected` (today):

| contrast | right_skill_reached | judge score | prefill tokens |
|---|---|---|---|
| injected_aa (A/A) | +0.02 | −0.06 | −1,192 (noise; A/A median abs 9,205) |
| desc_push | +0.02 (ns) | 0.00 (ns) | −892 (ns) |
| desc_pull | **−0.22 (sig., CI −0.34…−0.10)** | −0.20 (ns) | +1,589 (ns) |

desc_pull vs desc_push: 12 expected skills lost, 1 gained.

## Reading

The plan's rule: ship `desc_push` if `right_skill_reached` ≥ baseline − A/A
noise and the judge is not lower — it is +0.02 and 0.00, so it ships. The
index costs ~1.9k tokens more in the cached system prefix (69,951 vs 68,096)
and one cold prefill per open session after the restart. `desc_pull` would
ship only if it reached the right skill at least as often as `injected` and
cut prefill ≥ 15%; it lost a fifth of the right skills and cut nothing, so
bodies stay pushed.
