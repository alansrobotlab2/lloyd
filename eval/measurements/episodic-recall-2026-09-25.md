# #1485 — chat transcripts in `vault_recall`; `session_recall` as a qmd query (2026-09-25)

**Verdict (second pass, below): episodic floor ON** (`retrieval.recall.episodic_floors:
true`, chat transcripts only); **session_recall stays on the token scorer**
(`retrieval.session_recall.backend: tokens`). The first pass (next sections) landed
both off behind module constants; #1511's echo rule applies on both paths. Script:
`eval/run_episodic_recall_eval.py`; raw: `~/lloyd-data/eval/1485/` (`run1.json`,
`run2_sessions1_head18.json`, `run_json_era.json`, `synth_questions*.json`).

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

## Second pass (same day): the pin taught, LloydMemEval, one fix, flipped

**What held the flip, and what was actually true.** The pin's snapshot is a
`VACUUM INTO` of the whole live index, and qmd's `/query` filters on
`documents.collection`, so the pin *always* carried the 670 `sessions`
documents; `evalpin.yml`'s `collections: {}` only emptied `store_collections`
(paths, contexts, defaults), which a request naming its collections never
reads. The real gap was `production_payload`: it listed `VAULT_SEGMENTS` only,
so the floor was filtered out of `collectionFloor`. Now:

- `production_payload` appends the shape's `extra` exactly as
  `_qmd_daemon_search` does (a test asserts the two payloads are equal key by key,
  flag on and off); the flag is `retrieval.recall.episodic_floors`, read by
  `vault.recall_episodic_floors()`, the one reader both use.
- `PinnedCorpus` refuses a snapshot holding no document of a collection
  production's request names (`snapshot_collections`, `missing_collections`,
  per-collection counts in provenance) and records `pin_config_report` (pin yml
  vs `index.yml`: collections, embed model).
- `~/.config/qmd/evalpin.yml` is now a copy of `index.yml` (backup
  `evalpin.yml.bak-2026-09-25`), so the pin's `store_collections` is
  production's too. There is no committed evalpin template in the repo.
- `session_recall`'s switch moved to `retrieval.session_recall.backend`.

**One fix, found by the pinned run.** The `sessions` collection is not only
chats: of 670 exports, 179 are three-part chat ids, 475 four-part background
runs (347 `youtubed`, 54 `backlogs`, autocode, autotriage, deep research —
exported there before `sessions-background/` existed) and 16 e2e fixtures.
Unfiltered, 48/79 gold queries carried a transcript in their top 10 and those
(e.g. `e2e_selfmod363_…` for "backlog item 363") displaced gold documents.
`RECALL_EPISODIC_CHAT_ONLY` keeps only `YYYYMMDD_HHMMSS_<tag>` transcripts, in
`vault_recall` and in `session_recall`'s qmd backend.

### Pinned doc gold (86 queries, 79 with docs), `PinnedCorpus` + djev replay file, regression.lock

Arms differ only by config overlay. Every djev read was fresh (172/arm) — B's
pool differs from A's, so B cannot replay A; `AA` is A again on its own fresh
reads (the A/A floor for a fresh-djev arm). Raw `~/lloyd-data/eval/1485/pin2/`.

| run | B vs A | doc_hit | doc_recall | MRR | NDCG@10 | p50 paired |
|---|---|---|---|---|---|---|
| run1, unfiltered | on vs off | -0.025 [-0.076, +0.025] | -0.013 [-0.063, +0.032] | +0.006 [-0.022, +0.038] | -0.010 [-0.043, +0.024] | +11.8 ms |
| run1 | AA vs A | 0 | +0.006 | -0.010 [-0.034, +0.012] | -0.006 | +7.4 ms |
| **run2, chat only** | **on vs off** | **+0.000 [-0.051, +0.051]** | +0.006 [-0.032, +0.044] | **+0.031 [-0.001, +0.066]** | +0.013 [-0.023, +0.053] | **+9.5 ms** |
| run2 | AA vs A | 0 | 0 | 0 | 0 | +7.0 ms |

Chat-only, 14/79 gold queries carry a transcript in their top 10. No doc-gold
loss beyond noise; MRR moves the right way.

### LloydMemEval v1 (dev, n=267): what the `recall` arms can and cannot see

Its sessions exist only in the set's JSON, so against the live index
`vault_recall` cannot reach them and neither prefetch arm calls `vault_recall`.
So: `run_memory_eval.py export-sessions` writes the 425 dev sessions in the
chat-export shape (`# id`, `# iso`, `user:`/`lloyd:`, chat-shaped ids; the
holdout is never opened), a prepared pin (`memevalpin`, :8183) is the live
snapshot plus those files added to its `sessions` collection (`qmd update` +
`embed` on the copy: 425 new, 670 unchanged; the other collections untouched),
and two new arms ask `vault_recall` (top 10, path + snippet, no facts) with the
floor off / on. Everything else — distractors are the other 422 synthetic
sessions and the 670 real exports — is production's. Driver
`~/lloyd-data/eval/1485/memeval_pin.py`; raw `…/memeval/run1/`.

Retrieval (no model; evidence session in the top 10, gold value in the snippets):

| category | n | evidence any off→on | evidence all | gold in snippets off→on [95%] |
|---|---|---|---|---|
| multi_session | 48 | 0 → 0.688 | 0 → 0.313 | 0.083 → 0.417, +0.333 [+0.208, +0.458] |
| temporal | 57 | 0 → 0.825 | 0 → 0.597 | 0.579 → 0.702, +0.123 [+0.018, +0.228] |
| all | 267 | 0 → 0.626 | 0 → 0.509 | 0.303 → 0.584, +0.281 [+0.225, +0.341] |

Answers (primary, thinking on, rules judge + djev for mixed; `correct_strict`,
paired bootstrap):

| category | n | recall | recall_episodic | diff [95%] |
|---|---|---|---|---|
| single_session | 57 | 0.404 | 0.579 | +0.175 [+0.053, +0.298] |
| **multi_session** | **48** | **0.021** | **0.292** | **+0.271 [+0.146, +0.396]** |
| knowledge_update | 53 | 0.075 | 0.585 | +0.509 [+0.377, +0.642] |
| temporal | 57 | 0.070 | 0.491 | +0.421 [+0.298, +0.544] |
| preference | 52 | 0.135 | 0.192 | +0.058 [-0.039, +0.154] |
| **all** | **267** | **0.146** | **0.434** | **+0.288 [+0.232, +0.348]** |

Multi-session is still retrieval-bound: the floor is 1, so both sessions reach
the top 10 for 31% of questions. Caveats: the set is synthetic (8.6% defective
in its spot audit) and the prepared index holds 425 sessions of one style.

### Real past chats (live daemon, chat-only filter; 33 synthetic questions about real chats)

doc_hit 0.000 → 0.727 [+0.576, +0.879], MRR +0.444 [+0.308, +0.582] (first pass,
unfiltered: 0.697); live gold in the same run doc_hit +0.000 [-0.063, +0.076], MRR
+0.020 [-0.019, +0.060]; p50 424 → 450 ms.

### session_recall (re-run, chat-only)

Where both backends see the chat (n=10): hit@5 0.9 → 0.8, -0.1 [-0.4, +0.2];
0.7 → 65 ms p50. All 32 older chats: 0 → 0.81, which is the 09-22 JSON wipe, not
ranking, and closes by 2026-09-29. **Not flipped.**

### Verdict

`retrieval.recall.episodic_floors: true` (chat only). The gain clears its
interval everywhere it was measured and the doc set does not move beyond the
A/A floor; +9.5 ms p50. Takes effect at the next `lloyd-mcp` restart; the next
regression check compares a parent without the floor against a commit with it,
which is the change measured here.
