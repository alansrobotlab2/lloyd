# The `desc_frontload` arm (#4375): the 78 × 4 measured run was RULED NOT SPENT

**Verdict: NON-RUN.** The full labelled run — 78 queries × 4 arms = 312 arm-runs on the
primary engine with the worker pool held — was **RULED NOT SPENT** by owed-check on
#4375's owed entry on **2026-11-21** (`20261121_013851_owedcheck_4d0e`). That run has
never been executed, so this note carries **no `right_skill_reached` rate, no judge
score, no paired difference and no per-arm result for the arm** — none exists to quote,
and a table of them here would be a fabrication. What it does carry is the three things
the ruling rested on and every reader since has had to re-derive: the **cost bound**, the
**effect ceiling**, and **which question this arm cannot answer at all**.

The filename carries the ruling's date (2026-11-21), because that is the date this note is
about. **Every count below was re-measured on 2026-11-25**, the day this note was written,
by the round that landed it. Nothing here is carried forward from #4375 or from #4460's
triage; each figure names the tree and the instrument that produced it, because two of
them are not true in every tree (see "Why the ceiling is 7 and not 1").

Only owed-check can grant the run. No round may re-open the ruling; re-proposing it is a
re-litigation, and the numbers below are the reason it was refused.

## The cost bound, with its inputs spelled out

- **78 queries.** `grep -c 'id:' eval/skill_match_queries.yaml` → `78`, run 2026-11-25 on
  the live checkout. The labelled set the P5 runs read (`--queries eval/skill_match_queries.yaml`).
- **4 arms**, the quartet #4375's own ruling counted (`78 queries x 4 arms = 312
  arm-runs ~= 4.3 h exclusive primary with the pool self-held (#2498)`, recorded on
  #4375's owed entry). The runner's `--arms` default is a quartet too
  (`DEFAULT_ARMS`, `eval/run_prefetch_cost_eval.py:140`).
- **312 arm-runs** = 78 × 4.
- **199.5 s per arm-run** — one sample, not a mean. From the probe artifact below:
  `summary.arms.injected.n` is `1`, its `seconds_avg` is `199.5`, its `iterations_avg`
  is `9.0` against `config.max_turns` `8`, and its `completed_rate` is `0.0`. So the
  figure is a single turn of the **slower** arm that ran out of turns and never completed.
- **concurrency 4** — `config.concurrency` in the same artifact.

312 × 199.5 s = 62,244 arm-seconds, ÷ concurrency 4 = **15,561 s ≈ 4.3 h** of wall clock
on the primary with the pool held. Treat it as the **upper bound it is**: the one sample
of the arm under test ran **42.5 s** (`summary.arms.desc_frontload.seconds_avg`, also
n=1, `iterations_avg` 6.0, `completed_rate` 1.0), and 312 × 42.5 ÷ 4 = 3,315 s ≈ **0.9 h**.
A later pass that re-derives 0.9 h is not contradicting this note — the honest interval is
**0.9 h to 4.3 h**, and the ruling was made against the conservative end. Either way the
cost buys a comparison over a population of one index line (next section), which is why it
was refused.

## The effect ceiling: one index line of 209, character-neutral

Measured **2026-11-25** in the live checkout `~/lloyd` (the tree production imports) with
`.venvs/lloyd/bin/python`, comparing `advertised_index_block("desc_push")` against
`advertised_index_block("desc_frontload")` from `eval/run_prefetch_cost_eval.py` — i.e.
the two `<available_skills>` blocks the two arms actually run under, diffed with `difflib`
— with the arm enabled (`FRONTLOAD_ENABLED = True`, and `--frontload` defaults to `on`,
`eval/run_prefetch_cost_eval.py:1143`):

- index lines differing: **1** of 209 (0.48%),
- block char totals: **11,970** for `desc_push` and **11,970** for `desc_frontload` —
  the rewrite is length-exact, so the two totals are equal by construction and the
  difference is text position and nothing else,
- the arm's own tally over the same render: 208 descriptions seen, **21 rewritten**, 119
  already trigger-visible, 68 with no trigger the clip could show.

So the whole measurable difference between production and the arm, on today's tree, is
**one line's clause order out of 209 lines**. Against the A/A noise floor of ±0.02 that
#4375 recorded for a 78-query run, the only reachable outcomes of spending the hours above
were a null or a broken noise floor.

### Why the published ceiling is 7 and not 1

The delta is a function of **which** skills are granted a description, and that set is
granted in usage-rank order against the 12,000-char budget (`app/prompt_builder.py`,
`SKILLS_INDEX_BUDGET_CHARS`), so it rotates with the daily chat telemetry. Two trees, same
vault, same code, measured the same minute on 2026-11-25:

- **live checkout `~/lloyd`** (rank counts present, n=147): `clipped 68`, `trigger-leading
  37`, delta **1**, both blocks 11,970 chars.
- **a round worktree** (`app.paths` anchors `LLOYD_DATA` inside the worktree, so
  `_skill_rank_counts()` comes back empty and the granted set is the fallback order):
  `clipped 62`, `trigger-leading 25`, delta **7**, both blocks 11,989 chars. The seven
  lines that differ there today: `alfie-monitoring`, `backlog-triage`,
  `codebase-inspection`, `discord-social`, `entity-vectors-rebuild`, `fact-improvement`,
  `gate-to-land-lag-daily-record`.

This note therefore publishes **7** as the ceiling — the largest delta reachable on a tree
that cannot see production's rank, which is every tree a test run opens — and names 1 as
production's own reading today. The hard structural cap is the tally's `rewritten` count,
**21** of 208: no line can differ unless its description is rewritten, and only a
described skill's line is rendered at all. `tests/test_prefetch_cost_eval.py` re-renders
both blocks and fails if the live delta exceeds the published 7.

**A red node there does not mean the arm regressed.** It means the granted set moved
enough that this note's ceiling is stale, and the fix is a dated re-publication of the
number, not a code change. That is the owed post-landing check, stated on #4460.

## The one line the arm renders worse — prove command, not figure

`fact-improvement` is the skill whose line the enabled arm changes on production's tree,
and **the arm's version is worse**: the clause it front-loads is a caution, not a trigger.
Its line becomes `- fact-improvement — without deleting anything unless told to.Run the
nightly fact-quality improvement pass…` — the caution is hoisted to the front of the
visible 100-character clip and the real trigger ("Run the nightly fact-quality improvement
pass over the knowledge graph") is pushed past the cut. The mechanism is
`TRIGGER_WORDS` (`eval/run_prefetch_cost_eval.py:322`), which contains `"unless"`:
`frontload_description` moved the first clause whose trigger word landed inside its own
100 characters, and for this description that word is in a caution.

Do not trust that sentence — re-run it. #4373's own prove script is the instrument for the
library counts, and the render pair is the instrument for the delta:

```
cd ~/lloyd && .venvs/lloyd/bin/python - <<'PY'
import re, difflib
from agent_mcp.skills import iter_active_skills
from app.prompt_builder import (_skill_walk_roots, _skills_index_lines,
                               _skills_index_settings, _skill_rank_counts,
                               skill_index_description)
act = list(iter_active_skills(overlay=None, roots=_skill_walk_roots()))
s = _skills_index_settings(); c = _skill_rank_counts()
L = _skills_index_lines(act, c, budget=s['budget_chars'], max_desc=s['max_description_chars'])
d = {a.name: skill_index_description(a.frontmatter) for a in act}
clip = [l for l in L if l.endswith('…')]
nm = [l.lstrip('- ').split(' —')[0].strip() for l in clip]
lead = lambda n: bool(re.match(r'^\s*Use (when|to|for)\b', d.get(n, ''), re.I))
print('lines', len(L), 'clipped', len(clip), 'trigger-leading', sum(lead(n) for n in nm))
# the render pair, as the two arms run it
import importlib.util
spec = importlib.util.spec_from_file_location('rpce', 'eval/run_prefetch_cost_eval.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
a = m.advertised_index_block('desc_push').splitlines()
b = m.advertised_index_block('desc_frontload').splitlines()
for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
    if tag != 'equal':
        for l in a[i1:i2]: print('PROD :', l)
        for l in b[j1:j2]: print('ARM  :', l)
PY
```

The `lead()` predicate printed above is #4373's own, and it is what makes "worse"
checkable rather than an opinion: the arm's new leading clause for `fact-improvement`
fails it, while for the other six lines named above it passes.

## What this arm cannot answer: who gets a description

`right_skill_reached_tail_n` is **0 by construction** for this arm, not as a finding. The
tail population is the queries whose expected skill is *not* in the advertised set; this
arm advertises all 208 skills (`advertised_skill_count: 71` is not it — see below), so the
tail is empty, `right_skill_reached_tail_rate` is `null` and `tail_query_ids` is `[]`
(the probe's `advertised_skill_count: 208` is the advertised set, and
`described_skill_count: 71` the described one — the runner keeps the two apart on
purpose).
Reading those zeros as a result is the trap the runner itself flags at
`eval/run_prefetch_cost_eval.py:284` ("every arm would report
`right_skill_reached_tail_n: 0` forever").

So the who-gets-a-description question is carried by exactly one figure here,
`described_skill_count`, and this note does not answer it either. It records the shape of
the population today, from #4373's script in the live checkout: **208** skill lines, of
which **68** carry a clipped description and **3** carry one that fits under the 100-char
cap — **71 described** — leaving **137** names-only lines (68 + 3 + 137 = 208, and the
71 equals `described_skill_count: 71` in the 2026-11-21 probe, which is a coincidence of
two different dates, not a carry). The 137 names-only lines are the population a
front-load arm with a changed *length* would move, and this arm changes no length.

## #4373's landed rewrite is what collapsed this population to one line

Trigger-leading went **24 → 36** when #4373 landed — vault commit `20e4764a8`, 15
`skills/*/SKILL.md` descriptions front-loaded length-neutrally, measured at `clipped 69`
on the landing day. That vault rewrite is the reason the arm's measurable population
collapsed from the ~7 lines its round could see to the 1 line production shows: the
descriptions the arm would have moved had already been moved in the committed bytes, which
is why the tally reports 119 lines `already_visible`. #4375's own ruling names the same
cause ("which is what collapsed the measurable population from the round's ~7 lines to 1").

Today the same script prints `clipped 68 trigger-leading 37`. The counts move with the
granted set, so the **ratio** is the comparable quantity, exactly as #4373's acceptance
check says: 24/68 before the rewrite, 36/69 at landing, 37/68 today.

## Provenance of every figure in this note

- **Library counts** (`lines 209`, `clipped 68`, `trigger-leading 37`, 137 names-only,
  71 described) — measured **2026-11-25**, live checkout `~/lloyd`,
  `.venvs/lloyd/bin/python`, by #4373's own prove script quoted above.
- **The render delta** (1 line, 11,970 / 11,970 chars, tally 21 rewritten / 119
  already-visible / 68 no-visible-trigger) — measured **2026-11-25**, live checkout, the
  two `advertised_index_block` renders diffed with `difflib`.
- **The second tree's reading** (7 lines, 11,989 chars, `clipped 62 trigger-leading 25`) —
  measured **2026-11-25** in `/home/alansrobotlab/lloyd-work/SM_20261125_222702/home/lloyd`
  through the round's absolute interpreter, whose `app.paths` warning says out loud that
  its anchored data root makes such a scan "a measurement of nothing". It is published as
  the rank-less reading, never as production's.
- **The cost inputs** (199.5 s, 42.5 s, n=1, `iterations_avg`, `completed_rate`,
  `max_turns 8`, `concurrency 4`) — read **2026-11-25** out of
  `/home/alansrobotlab/lloyd-data/eval/baselines/frontload-probe-4375-20261121-014037.json`
  (`ran_at` `2026-11-21T09:44:12.824601+00:00`, `label` `frontload-probe-4375`, sha256
  `e2c87da2cd7ac4c88bffe027b1aa4d4493bd753688b0d3cbf900f86c1d240087`). It holds **one**
  record, `id` `smq001`, whose expected skill is `morning-brief-and-triage`: a
  script-mode smoke of the two arms, **not** a 78-query run.
- **`--frontload off` renders production's bytes** is attributed to the **in-process
  render comparison**, not to that artifact: the artifact ran **enabled**
  (`frontload_rewrite_enabled: true`, `frontload_prompt_equals_production: false`), so it
  proves nothing about the disabled render. Measured 2026-11-25 by rendering
  `advertised_index_block` and `system_prompt_for` for both arms with
  `FRONTLOAD_ENABLED = False`: both equal production byte for byte, and the tally comes
  back all-zero with `enabled: False`. Already pinned by
  `tests/test_prefetch_cost_eval.py::test_a_disabled_rewrite_renders_production_and_the_report_says_so`.
- **The artifact's role is script-mode proof only**: it shows the accepted arm name
  executes end-to-end outside the suite, and that `frontload_report` printed
  `incomplete: one of the two arms has no clean sample` (recorded on #4375's owed entry)
  rather than a verdict off one turn.
- **Artifact size as committed bytes are owed elsewhere**: `wc -l -c` on that runtime path
  printed `195 10521` on 2026-11-25. **No copy of this JSON is added to this repo by this
  note and none is owed here** — `eval/baselines/**` and `*.json` are both ignored, and
  `.gitignore` is a path no round may write (the #3989 ruling shape, precedent
  `0f3ef03b`). Whether the runtime bytes get repo history at all is owed-check's call, not
  a diff's; today they sit outside the retention sweep's stores (store 17 covers
  `eval/tool-quality/`, not `eval/baselines/`), so naming the path here does not expire on
  a sweep.

## Register: the figures this note publishes

The prose above is the argument; this block is the same numbers so a node or a later job
can read them without re-parsing sentences. Every value is dated and instrumented by the
Provenance section; `tests/test_prefetch_cost_eval.py` parses this block and re-derives the
two that are live (`render_delta_live` is today's reading, so it is pinned only as
`≤ render_delta_ceiling`).

```
measured_on: 2026-11-25
render_delta_ceiling: 7
render_delta_live: 1
render_delta_rankless_tree: 7
render_delta_structural_cap: 21
push_block_chars: 11970
frontload_block_chars: 11970
index_lines: 209
clipped_lines: 68
trigger_leading: 37
described_skill_count: 71
names_only_lines: 137
cost_queries: 78
cost_arms: 4
cost_arm_runs: 312
cost_seconds_per_arm_run: 199.5
cost_concurrency: 4
cost_hours_upper: 4.3
cost_seconds_faster_sample: 42.5
cost_hours_lower: 0.9
```

## Owed after this note lands

- Whether to spend the 0.9–4.3 h after all stays owed-check's ruling — **RULED NOT SPENT,
  2026-11-21** — and is not re-opened by anything in this file.
- On the next daily rank rotation: re-run the prove command above and confirm the live
  delta is still ≤ the published 7. If it grows past it, re-publish the number with a new
  date; the pinning node is written to fail loudly rather than silently age.
- Whether the probe JSON gets repo history (above).
- The vault witness copy of that JSON under `backlog/data/` and its own `wc -l -c` figure.
