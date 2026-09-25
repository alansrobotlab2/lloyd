# #1485 — chat transcripts in `vault_recall`; `session_recall` as a qmd query (2026-09-25)

**Verdict: landed off.** Both halves are built behind constants that default to
today's request byte for byte (`agent_mcp/vault.py::RECALL_EPISODIC_FLOORS`,
`agent_mcp/session.py::SESSION_RECALL_BACKEND`). #1511's echo rule applies on both
paths. Script: `eval/run_episodic_recall_eval.py`; raw:
`~/lloyd-data/eval/1485/` (`run1.json`, `run2_sessions1_head18.json`,
`run_json_era.json`, `synth_questions*.json`).

## Question sets

- **gold** — the 79 `vault_recall_queries.yaml` rows with `expect_docs`: must not
  regress.
- **episodic** — P0 (#1480) does not exist, so a known-item proxy: for 33 chat
  transcripts (three-part ids, ≥2 user turns, probes excluded, seed 25) the primary
  wrote the question Alan would ask weeks later to find that chat, told not to copy
  any 4-word phrase; a question #1511's echo test flags is discarded. Expected doc =
  that transcript. Synthetic, n=33 — a proxy for P0's multi-session category, not a
  substitute. (A daily-note-summary route was tried first and abandoned: capture
  minutes mislabelled most summaries, and content-checked labels left n=5.)

Arms interleaved per query with rotated order, live qmd daemon + djev (queried,
never loaded), `limit` 10, paired bootstrap 95%.

## vault_recall + episodic floors

| arm B | set | doc_hit A → B | diff [95%] | MRR diff [95%] | p50 A / B |
|---|---|---|---|---|---|
| A/A (today twice) | gold | 0.608 → 0.595 | -0.013 [-0.063, +0.025] | -0.016 [-0.038, +0.001] / run1 -0.024 [-0.049, -0.004] | — |
| sessions 1 + autonomy-runs 1, head 16 | gold | 0.608 → 0.633 | +0.025 [-0.051, +0.101] | **-0.041 [-0.082, -0.004]** | 464 / 494 ms |
| same | episodic | 0.000 → 0.727 | +0.727 [+0.576, +0.879] | +0.480 [+0.343, +0.616] | 448 / 491 ms |
| **sessions 1, head 18** | gold | 0.608 → 0.608 | +0.000 [-0.063, +0.076] | -0.014 [-0.059, +0.032] | 422 / 452 ms |
| **same** | episodic | 0.000 → 0.697 | +0.697 [+0.546, +0.849] | +0.440 [+0.309, +0.574] | 477 / 529 ms |

Today's recall cannot reach a transcript at all (0/33). The sessions-only variant
(now the constants' default shape) is non-inferior on the doc set within the A/A
noise and finds the chat 70% of the time; `autonomy-runs` added nothing the
episodic set measures and cost doc MRR. With it on, 41 of 79 gold queries carried
a transcript somewhere in their top 10.

**Why not flipped:** the regression pin does not index `sessions` and
`evalpin.production_payload` does not send `extra` — flipping would make every later
promotion's regression check judge a request production no longer sends (head 18,
no sessions). That, and a synthetic question set, make this a human flip after
P0. Owed by a human: add `sessions` to `~/.config/qmd/evalpin.yml`, teach
`production_payload` the `extra` key, re-run on P0's multi-session category, flip.

## session_recall: tokens vs qmd

| set | n | hit@5 tokens → qmd | MRR tokens → qmd | p50 |
|---|---|---|---|---|
| JSON era (chats since 2026-09-22, `--days 7`) | 10 | 0.9 → 0.8, diff -0.1 [-0.4, +0.2] | 0.667 → 0.750, +0.083 [-0.25, +0.40] | 0.7 ms → 62 ms |
| all episodic (`--days 30`) | 32 | 0.000 → 0.719 [+0.56, +0.88] | 0.000 → 0.623 | 0.8 ms → 164 ms |

Where both backends can see the chat they are indistinguishable (n=10) and qmd is
~100x slower. The 0.72-vs-0 row is the 2026-09-22 wipe, not a ranking win: the
JSON the token scorer reads is gone for every chat before 09-22 while their
markdown exports survived. At the default 7-day window that gap closes on its own
by 2026-09-29. Not flipped.
