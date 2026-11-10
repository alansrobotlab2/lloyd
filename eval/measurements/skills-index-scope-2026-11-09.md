# Skills index scope: does every chat need every skill name? 2026-11-09

**Verdict: keep the full index; `skills.index.chat_scope` stays `all` (absent).**
Hiding the 53 autonomy-bound skills from chat (`scoped`) changed nothing measurable,
and dropping the index (`no_index`) lost the right skill on 3 turns and gained it on
none. The switch stays in the code, off, for a library where the hidden set is large
enough to matter.

Runner: `eval/run_prefetch_cost_eval.py --queries eval/skill_match_queries.yaml --n 100
--arms desc_push,scoped,no_index --label skills-index-scope-2026-11-09-r2`, 51 labelled
chat turns (153 cells), primary `:8096`, worker pool held and drained, concurrency 4,
seed 562. Paired bootstrap CIs (`eval/stats.py`), arm minus `desc_push`. Artifacts in
`~/lloyd-data/eval/skills-index-scope-2026-11-09/` (the `-191214` file is a first run
killed at 47 of 255 cells by an editor restart; it is not read here).

## Arms

- `desc_push` — production: descriptions on, prefetch push on, every skill listed.
- `scoped` — the same with `chat_scope: discoverable`: the 53 skills an autonomy task
  binds by `skill_name`/`skill_path` (and any `surface:` without chat) are left out of
  the chat index behind one `skills_search` line.
- `no_index` — the same with no `<available_skills>` block at all; prefetch push and
  `skills_search` / `skills_read` still work.

## Results (n = 51)

| arm | system prefix | right skill reached | judge (0-2) | prefill tokens | completed |
|---|---|---|---|---|---|
| desc_push | 62,718 | 0.73 | 1.69 | 38,438 | 0.88 |
| scoped | 62,622 | 0.72 | 1.55 | 37,451 | 0.88 |
| no_index | 59,816 | 0.67 | 1.54 | 38,075 | 0.96 |

| contrast vs desc_push | right skill | judge | prefill tokens |
|---|---|---|---|
| scoped | +0.000 (0 lost, 0 gained, n=50) | −0.16 (−0.45…+0.12, n=49) | −1,272 (−6,863…+4,118) |
| no_index | −0.059 (−0.137…0.000, p=0.08; 3 lost, 0 gained) | −0.14 (−0.40…+0.12, n=50) | −363 (−5,972…+4,856) |

`no_index` lost `opencode` on smq043 and smq047 and `voice-clone-sample` /
`youtube-content` on smq055 — skills whose names, not the user's words, are what
points at them. One `scoped` cell (smq070) timed out at 900 s and is excluded from
its pairs.

## Why scoped saved 96 tokens, not thousands

The described index is budgeted (`budget_chars` 12,000), and the description grant
walks the rank until the budget is spent. Hiding 53 names frees budget that the next
skills down the rank immediately spend on their descriptions, so the scoped prefix is
almost the same size — a different 12,000 characters, not fewer. A real saving would
need a smaller budget too, and nothing here says the model would be better off with
less; the judge moved the wrong way (not significantly) for both arms that shrank
what it was shown.

## What this does not test

The library-size claim (#4027's `injected_pruned` arm, the top 140 by rank) is a
different question, a different baseline (`injected`), and its own write-up in
`skills-pruned-4027-2026-11-09.md`.
