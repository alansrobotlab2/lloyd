# Compaction recall: the `tool_clear`/`sidecar` two-pass pair, 2026-10-14 (backlog #2682)

**Verdict: the sidecar arm is NOT below `tool_clear` — these figures open #2487's ship-on branch; the default still stays False in this commit, because the flip is a separate gated round.**

Runner: `eval/run_compaction_recall_eval.py`. Artifacts — the two passes over ONE
`--data-root`, both committed:
`eval/baselines/compaction-recall-sidecar-pair-pass1-2026-10-14.json` and
`eval/baselines/compaction-recall-sidecar-pair-pass2-2026-10-14.json`. Engine:
the already-running primary (Qwen3.8-Flash-Next on vLLM :8096); no second
`agent-llm-primary` was booted and no service was restarted.

## Why the second pass is the whole point

#2511 made the generator spill for real (`_spill_turn`, a body over
`SPILL_THRESHOLD_CHARS`), so every `tool`-shape row ends with a cleared preview
behind a `<persisted-output>` pointer — but a row is one turn, and
`load_and_compact_session` runs *before* relief can record anything. Pass one can
only book reductions into the per-session
`sessions/<sid>.microcompact-reduced.json` sidecar (#2689's second writer); pass
two is the first load that can apply one. One run of `--arms tool_clear,sidecar`
therefore reads `sidecar_applied == 0` everywhere and the two rates are the same
prompt graded twice — which is where #2682's first attempt died.

This pair is two passes. Pass one booked a non-empty `reduced_call_ids` on
**12 of 12** sidecar sessions and applied nothing (0 across
both arms, as it must). Pass two, over the same root — the same 12 session keys,
the same content hashes, the same `tools.json` — read those sidecars and applied
12 reductions (12 of 12 sidecar rows) while all 12
`tool_clear` rows applied **0**. The pair is non-vacuous, so the
difference below is attributable to the read-time reduction.

## The four numbers (pass two, the one that applied)

| arm | kept | distinctive | ambiguous | applied |
|---|---|---|---|---|
| `tool_clear` | 12 | 0.92 [0.65, 0.99] (11/12) | 0.92 [0.65, 0.99] (11/12) | 0 |
| `sidecar` | 12 | 1.00 [0.76, 1.00] (12/12) | 1.00 [0.76, 1.00] (12/12) | 12 |

Pass one, the same prompts graded with nothing applied — the row that makes pass
two's delta mean something:

| arm | kept | distinctive | ambiguous | applied |
|---|---|---|---|---|
| `tool_clear` | 12 | 0.92 [0.65, 0.99] (11/12) | 0.92 [0.65, 0.99] (11/12) | 0 |
| `sidecar` | 12 | 1.00 [0.76, 1.00] (12/12) | 1.00 [0.76, 1.00] (12/12) | 0 |

Paired (pass two, `sidecar` against `tool_clear` over the 12 sessions both
kept): distinctive +0.08 [+0.00, +0.25]; ambiguous
+0.08 [+0.00, +0.25]; first-token +1.13 [-2.29, +4.55] s; wall
-0.77 [-9.60, +7.99] s; tool calls +0.33 [-0.67, +1.17].

Cost, pass two: median wall 46.2 s
(`tool_clear`) vs 42.6 s (`sidecar`); median
first-token 4.7 s vs
4.7 s; mean tool calls
3.83 vs 4.17.

## What the pair can and cannot show

Each arm is graded on its own prompt, and the sidecar's prompt differs from
`tool_clear`'s by the reduction itself, so this is arm-vs-arm across sessions,
not one prompt measured twice — the same shape as every other arm in
`architecture/measurement.md`. With 12 sessions per arm the CIs cannot
resolve a difference much below 0.2, so the recall columns are directional; the
applied count is the unambiguous part, and it is the mechanism observed on the
real corpus for the first time.

## The branch these numbers select

0 of the two measures came out below `tool_clear` (`sidecar`
1.00 distinctive / 1.00 ambiguous against `tool_clear`
0.92 / 0.92).
Neither is below: the ship-on branch is open, and a separate gated round may flip `app/harness/options.py:287` to True and re-key the default-is-False nodes. That flip is not in this commit.
Either way the default stays False here. The two nodes in
`tests/test_compaction.py` that assert that default —
`test_the_shipped_default_does_not_consult_the_sidecar_at_all` (def :721, the
`__dataclass_fields__[...].default is False` assert at :727-728) and
`test_one_runs_options_value_arms_the_read_time_half_too` (def :1971, the same
assert at :1982-1983) — stay green, as does
`test_the_shipped_default_writes_no_sidecar_and_counts_what_it_always_did`
(def :2045), which pins the off behaviour through
`shipped["microcompact_sidecar"] is False` and `assert not sidecar.exists()`.
Those three are the re-key list for the flip round and clause 5 of #2682.

## How it was run, because the shape is not the CLI's own

Both passes ran under `LLOYD_DATA=<scratch>` with `--data-root <scratch>` and one
shared `--tools-snapshot <scratch>/tools.json`, so the synthetic sessions and
their spill files never touched `~/lloyd-data/sessions` and both passes graded an
identical tool set. The tree that minted the filler was pinned to one commit:
`_corpus_files` reads the checkout (`app/`, `agent_mcp/`, `scripts/`,
`architecture/`), so editing this repo mid-run would silently change the corpus
under the second pass.

A single 24-row process does not fit this box — the 16 G background-Bash scope
OOM-killed one at 21 rows on 2026-10-13. Each pass was therefore filled by four
3-session invocations (`--sessions 3 --sizes <one size>`, seeds 600/603/1600/1603
so the minted keys are the grid's own `s600-*`/`s1603-*`, depths in the grid's
cycle order) and then closed by one full-grid invocation with every row already
recorded, whose loop runs nothing and only reaches `_write` afterwards. Both
artifacts consequently carry the default grid's `args` and the same 12-entry
`sessions` block, which the rail asserts rather than assumes.

Kept and dropped: pass one 12 + 12 kept, 0 dropped; pass two 12 + 12 kept, 0 dropped; 0 errors in either.
Wall clock: ~1 min/row, ~50 min of live-primary time for the pair.
