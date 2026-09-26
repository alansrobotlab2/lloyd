# Cleared tool results: the free route, observation stubs, lossless relief (2026-09-25)

Backlog #1514 (blocks #1481), #1481, #1499. Runner: `eval/run_compaction_recall_eval.py`
(#600's planted-fact harness), late probe, the live primary (Qwen3.8-Flash-Next NVFP4 on
vLLM :8096), every run under the EXCLUSIVE primary lock with the worker pool paused. Raw
artifacts: `~/lloyd-data/eval/1514/{main,rung4,pilot}/result.json`. Recall is two columns
(distinctive codename / ambiguous current port), never blended; intervals are Wilson 95%
per arm and paired bootstrap (10,000 resamples, `eval/stats.py`) between arms on the same
sessions.

## Verdicts

| item | verdict | switch |
|---|---|---|
| #1514 | route **named and measured; not deployed** — the in-band clause made recall worse | `compaction.microcompact.name_session_record`, default off |
| #1481 | **landed off** — no recall gain over `tool_clear` (+0.05 / +0.05, ns), +1.0 s TTFT per turn; the idempotence fix ships unconditionally | `compaction.microcompact.observation_stubs`, default off |
| #1499 | **rejected / already done** — rung 4 has spilled before truncating since 2026-09-11; spill vs the pre-09-11 lossy rung: 10/10 vs 9/10, Δ −0.1 [−0.3, 0.0] | none (instrumentation only) |

## First: the 12/20 the whole chain was built on was a harness leak

`EvalPool` served Read and Grep over the whole scratch root — every arm's and every seed's
session JSON and spill files, each planting a *different* codename and port. A Grep for
`billing-east` therefore answered with strangers' ports. On the 2026-09-24 baseline,
`tool_clear` Grepped in 19 of 20 rows and in **all 8** of its failing rows (production: 5/20,
none: 2/20), and its failing answers are the model refusing a pile of conflicting values
("looks like injected content"). The pool now sees only the probe session's own record
(`session_id`), Grep honours `path`, and the session JSON is written `indent=2` as
`sessions_io` writes it. With that fixed, `tool_clear` is **17/20 and 19/20** on the
same generator, not 12/20. #1481's motivating number was mostly the leak.

## #1514 / #1481: `tool_clear` vs `self_record` vs `observation` (n = 20, 10 at 90k + 10 at 130k)

| arm | distinctive | ambiguous | TTFT iter 1, median | TTFT turn total, median | tool calls / turn | recall_observation / turn | record reads / turn |
|---|---|---|---|---|---|---|---|
| tool_clear | 17/20 [.64, .95] | 19/20 [.76, .99] | 4.42 s | 8.85 s | 4.85 | — | 0 |
| self_record | 14/20 [.48, .85] | 14/20 [.48, .85] | 4.99 s | 9.70 s | 4.00 | — | 0.2 |
| observation | 18/20 [.70, .97] | 20/20 [.84, 1.0] | 4.85 s | 9.73 s | 4.95 | 0.85 | 0.2 |
| production (reference, later batch) | 20/20 [.84, 1.0] | 20/20 [.84, 1.0] | 1.14 s | 5.89 s | 2.40 | — | 0 |

`production` ran in a second batch on the same 20 sessions, so compare its TTFT and
tool-call columns, not wall time. Against it, `observation` is −0.10 [−0.25, 0.00] /
identical at +3.9 s TTFT per turn and +2.55 tool calls; aggressive clearing loses, and
neither switch closes the gap to not clearing at 0.2/0.1.

Paired, same 20 sessions:

| comparison | distinctive Δ | ambiguous Δ | TTFT turn total Δ | tool calls Δ | wall Δ |
|---|---|---|---|---|---|
| self_record − tool_clear | −0.15 [−0.40, +0.10] | **−0.25 [−0.45, −0.10]** | +0.69 s [−0.20, +1.40] | −0.85 [−2.95, +0.80] | −3.8 s [−9.9, +2.4] |
| observation − tool_clear | +0.05 [−0.10, +0.25] | +0.05 [0.00, +0.15] | **+1.01 s [+0.15, +1.83]** | +0.10 [−1.85, +1.80] | **+6.0 s [+0.4, +11.4]** |
| observation − self_record | +0.20 [−0.05, +0.45] | **+0.30 [+0.10, +0.50]** | +0.32 s [−0.30, +0.98] | +0.95 [−0.30, +2.40] | **+9.8 s [+6.3, +12.9]** |

- **The free route is not free.** One clause naming `sessions/<sid>.json` and the spill
  directory on every cleared marker adds ~250 chars per marker (turn-start freed 101k
  tokens against `tool_clear`'s 116k, median) and was used in 3 of 20 rows. It cost
  ambiguous recall significantly. Left off; the route is documented where a cleared result
  is described (`app/harness/tool_result_spill.py::session_record_route`), which is #1514's
  acceptance, and the arm is #1481's named baseline.
- **Observation stubs beat the free route but not the thing it amends.** The stub's verbatim
  head carried the planted result onto the wire in 16 of 20 rows and `recall_observation`
  was called 0.85 times per turn, yet recall over `tool_clear` moved by one row per fact
  (ns) while every turn paid ~1 s more TTFT and 6 s more wall: the heads keep ~400 chars
  per cleared result in the prompt. House rule: a slower change needs a gain clear of the
  interval — not met. Code lands behind the switch, off; the tool is hidden from any turn
  whose relief writes no stubs, so the catalog is unchanged.
- **Re-relief is idempotent now (#1481 clause 2, on for everyone).** The spill-aware pass
  re-selected already-reduced `<persisted-output>` blocks on every pass: the second pass
  appended a duplicate `[preview dropped …]` line (a rewrite of a cached message) and every
  pass counted a clear. `usage.db`, last ~100 recorded turns: rung 1 was named on 60 relief
  passes and was the last rung on 35 of them with **0 tokens freed on all 35**. A stub is
  never selected again; a second pass over its own output is byte-identical and counts 0.

## #1499: what a relief rung drops

**Step 1, census** (`usage.db` compaction records, 100 turns with a record, 42 with relief;
`harness.context_relief` events, 82 passes over 64 turns in the retained event logs):

| rung | passes naming it | turns | what it freed |
|---|---|---|---|
| 1 tool_results | 60 | 42 | 0 tokens on the 35 passes it was the last rung (the re-reduction above) |
| 2 reasoning | 44 | 12 | 211,786 tokens over the 6 passes it was the last rung (median 50.5k) |
| 3 arguments | 11 | 5 | 266,423 argument chars over 11 passes (median 16.5k) |
| 4 truncate | 19 | 8 | 570,893 tokens over 19 passes as last rung (median 29.1k); 736,456 chars truncated (median 21.8k) |

Every rung that drops *tool* content already spills first and names the file: rung 1
(microcompact, 2026-09-05), rung 3 (2026-09-11), rung 4 (2026-09-11). The item's premise
("the cut is unrecoverable inside the turn") is the pre-09-11 rung 4. The one rung that drops
content with no in-band handle is rung 2 (preserved reasoning); that reasoning is persisted as
`role="thinking"` rows in the session record, i.e. behind #1514's route, and this probe cannot
plant a fact in reasoning. The relief event now carries `truncated_chars_freed` and
`argument_chars_freed` (the usage record had them; the event, which survives a turn with no
usage row, did not).

**Steps 2–4, probe** (n = 10 at 90k, rung 1 held off by naming every tool non-compactable,
turn-start pass off, target 0.3; the planted result was on the wire at iteration 1 and
dropped by rung 4 in **30/30** rows):

| arm | distinctive | ambiguous | TTFT turn total, median | tool calls | truncated chars freed, median |
|---|---|---|---|---|---|
| rung4 (today: spill + path) | 10/10 [.72, 1] | 10/10 | 3.92 s | 2.9 | 333,885 |
| rung4_lossy (pre-09-11: no spill) | 9/10 [.60, .98] | 10/10 | 3.36 s | 2.3 | 332,260 |
| rung4_self_record (+ #1514 clause) | 10/10 | 10/10 | 4.58 s | 3.2 | 313,491 |

Paired vs `rung4`: lossy distinctive −0.1 [−0.3, 0.0], TTFT −0.66 s [−1.15, −0.22];
self_record recall identical, TTFT +0.80 s [+0.40, +1.23]. Freed characters per event are
within 0.5% between spill and lossy. **Step 5, cost**: the spill costs nothing in freed
characters; its only cost is the re-read it enables (+0.6 tool calls, +0.66 s TTFT per turn).
Recall did not rise *materially* over the lossy counterfactual (the model re-reads the
original file just as well), and the mechanism is already live, so no flag is added.

## Limits

- Synthetic sessions; the planted fact lives in a file the probe can always re-read, which is
  the kindest case for every arm and the reason spill vs lossy is 10 vs 9.
- Session rows are raw, not D1 pointers (as in the 09-24 baseline), so the session record holds
  the planted result verbatim; in production a >2k result's row is a pointer whose file is in
  the same spill directory the route names.
- n = 20 (main) and 10 (rung 4); a gain below ~0.15 per fact is inside the intervals.
- `production_self_record` / `production_observation` are defined but were not run: neither
  switch gained at the aggressive thresholds where clearing is heaviest, and production
  clears far less, so there was no gain left to look for at 0.72/0.52.
