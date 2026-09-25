# Harness replay diff — P13.1-3, 2026-09-24

Instrument: `eval/run_harness_replay_diff.py` (this commit). Base `8ac6f4a8`
(a detached worktree), HEAD = P13.1-3.

## Sample

`sample --n 200` over `~/lloyd-data/sessions/` (688 files, read-only; the
sample was copied to a scratch dir and reconstructed there). Stratified
round-robin over source × batch shape × size, 41 strata: autocode 45,
autotriage 23, autonomy 34, review 19, bench/benchmine 27, youtube-digest 12,
board-steward 9, session-distill 9, deep-research 8, arch-review 6, browser 3,
mission-control (chat) 5. Only 8 chat sessions exist on disk, so chat is thin;
every one of the loop's paths is the same code for every platform.

- 200 sessions → 236 user turns → 5,456 scripted iterations, 6,314 tool calls.
- 965 multi-call iterations, 28 of them wider than 3 calls.
- 75 sessions (a third, chosen by hash) carry perturbations: context-overflow
  rejection (40 recoveries), a stalled stream with one retry (34), state
  anchors (758 `harness.anchor_fired`), a notification drain, a disallowed
  `Glob`, a PreToolUse deny and inject, a terminal observer inject, and a short
  `max_turns` with the toolless wrap-up. Usage is replayed from each row's
  stats, so the meter is measured and relief fires (340 `harness.context_relief`).
- Each turn runs twice: `seq` (parallel off — production's setting) and `par`
  (on, `max_concurrency` 3). 472 replayed turns per tree, ~74 s per tree.

## Results

Channels: events `(type, call_id, sha1(content))` without timing fields;
final `chat_messages` `(role, tool_call_id, sha1(content), sha1(tool_calls),
has_reasoning, has_reasoning_content)`; session event log `(event,
sha1(data))`; hook trace (every PreToolUse / PostToolUse / OnEvent with the
message-list length and tail it saw); any raised error; request count.

| run | turns | clean | whitelisted | failures |
|---|---|---|---|---|
| A/A base vs base (step 1+2 run) | 472 | 472 | 0 | 0 |
| base vs steps 1+2 (`TurnState`, phases) | 472 | 472 | 0 | 0 |
| A/A base vs base (step 3 run) | 472 | 472 | 0 | 0 |
| base vs steps 1+2+3 (one dispatch path) | 472 | 470 | 2 | 0 |

The two whitelisted turns are `par` arm, `events` + `trace` only, same
multiset of `(type, call_id)`: `20260922_170739_autonomy_8ff6#0` (a 4-call
batch) and `20260922_183426_deepresearch_85e4#0` (5 calls), both wider than
the concurrency of 3 — the one intended change (a call is announced when it is
admitted). History, event log, errors and request counts: zero diffs, both
arms, all steps.

## Does the instrument see a real change?

Mutation check: HEAD with the history append deferred to the end of the batch
(the old parallel path's rule applied at concurrency 1). Against base: 144
`seq`-arm turns fail, every one in the hook trace (a PreToolUse for call₂ no
longer sees result₁), plus 130 `par`-arm trace changes the whitelist absorbs;
final history is identical, which is why the trace channel exists.

## Notes

- The first A/A run was not clean (34 history diffs): a spilled result names
  its path in the text, and each run had its own scratch data dir. One dir,
  emptied per run, fixed it. The `par` arm's reordering delays were timer-based
  and flaked under load (6 A/A interleaves); they are counted in event-loop
  passes now.
- Commands: `sample --n 200 --out $SCRATCH/fx`, then
  `all --base $SCRATCH/base --head <tree> --fixtures $SCRATCH/fx/fixtures.json --out $SCRATCH/runN`.
