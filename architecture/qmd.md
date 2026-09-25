---
title: qmd — the vault search engine, Lloyd's fork, and how it runs
status: implemented
date: 2026-09-25
---

# qmd — the vault search engine, Lloyd's fork, and how it runs

qmd (upstream `tobi/qmd`) indexes markdown into SQLite and answers hybrid
queries: BM25 over an FTS5 table, cosine over chunk embeddings, fused, optionally
cross-encoder reranked. Lloyd runs **one build of it**, a fork, as a long-lived
daemon on GPU 0. This doc is qmd itself: the fork, the daemon, the index and its
config, the API Lloyd depends on, the jobs that keep it healthy, and how it
fails. How `vault_recall` *uses* it — the pool, djev ranking, the gold set and
every measured change — is `architecture/retrieval.md`.

## 1. One build: the fork in `~/lloyd/qmd`

- **A separate clone, not a submodule.** `~/lloyd/qmd` is gitignored by this repo
  (`/qmd/` in `.gitignore`, no `.gitmodules`). Branch `lloyd`, pushed to
  `origin` = `alansrobotlab2/qmd`; upstream is `tobi/qmd`. `WORKLOG.md` and
  `GAMEPLAN.md` are force-added to the branch and carry the fork's own history.
- **Everything runs it.** The daemon, the watcher, the nightly cleanup timer, task
  #81, the regression pin and the `qmd` on PATH (`~/.local/bin/qmd` →
  `~/lloyd/qmd/bin/qmd`) all execute `~/lloyd/qmd/dist/cli/qmd.js`. The published
  `@tobilu/qmd` was uninstalled on 2026-09-19 because a second build is a second
  definition of how the index is written; `tests/test_qmd_single_build.py` pins it.
- **Fork changes are human-landed.** No automod round may edit `qmd/**`: the gate
  never builds or tests the fork, the review cannot see it, and a rollback cannot
  revert it. The landing path is: the fork's build, then its suite
  (`node scripts/test-all.mjs`), then a commit on `lloyd`, then a push, then
  rebuilding `dist/` in `~/lloyd/qmd` and restarting the daemon.
  `python -m scripts.qmd_fork_landing --fork <a cp -a copy>` answers whether a fork
  sha is in that state (never point it at the live checkout: the build rewrites
  the served `dist/`).
- **Editing `src/` changes nothing live.** The daemon serves `dist/`, and a node
  process keeps what it loaded until it restarts.

What the fork adds over upstream, newest first. `WORKLOG.md` covers the 09-07
and 09-19 work in §6–7 and the two 2026-09-21 changes (fork commits `fa71e57`
and `db52729`) in §8, the 2026-09-24 cache fix (`d01b049`) in §9, and the
2026-09-25 fusion depth (`079c9c9`) in §10:

| change | what it does | since |
|---|---|---|
| `QMD_FUSION_DEPTH` (#1475) | under global fusion each leg is fused this deep instead of cut to the candidate limit before RRF; unset keeps the old cut. Lloyd serves 100 | 2026-09-25 |
| a re-index keeps `llm_cache` (#1366) | `qmd update`, `qmd collection add` and SDK `update()` no longer empty the rerank cache; keys are content-addressed, the prune to 1,000 newest bounds it | 2026-09-24 |
| `update` counts pending against the configured model | the hint printed pending against the built-in default, so a non-default embed model read every hash as unembedded — 10,745 of them — on every watcher cycle | 2026-09-21 |
| `lexMode: "or"` | a lex search ORs its terms; AND stays the default | 2026-09-21 |
| `collectionFloor` map | under global fusion, each named collection's best N per search join the candidates | 2026-09-19 |
| `fusion: "global"` | one ranking across collections by score, instead of one RRF list per collection | 2026-09-19 |
| `meta` on `/query` | `reranked`, `rerankFallback`, `ms`, `phases`; `/health` counts rerank fallbacks | 2026-09-19 |
| incremental vector-index refresh | a write no longer forces a full in-memory rebuild on the next query | 2026-09-19 |
| in-memory exact vector index (`src/vecindex.ts`) | sqlite-vec brute-forced every `MATCH` and could not scope to a collection | 2026-09-07 |
| `rerankWindowChars`, rerank/idle knobs, `skipRerank` on REST | see §3 | 2026-09-07 |

## 2. The daemon and its index

```
supervisord program agent-qmd-daemon
  /usr/bin/node ~/lloyd/qmd/dist/cli/qmd.js mcp --http --port 8181      (GPU 0)
  index  ~/.cache/qmd/index.sqlite        (the default index name, "index")
  config ~/.config/qmd/index.yml          (collections + models)
  REST   POST /query, GET /health          MCP  /mcp
  log    ~/lloyd-data/logs/services/agent-qmd-daemon.err — one line per /query with phase ms
```

- **Binds `[::1]:8181` only.** An IPv4-literal probe cannot see it
  (`scripts/service_health_check.py` asks both).
- **The index** holds `documents` (path, title, hash, collection, active),
  `content` (whole file text by hash), `documents_fts` (FTS5 over path, title and
  body; BM25 weights 1.5 / 4.0 / 1.0), `content_vectors` + `vectors_vec` (one
  vector per chunk, keyed `hash_seq`; `content_vectors` itself is keyed
  `(hash, seq)` and carries the `model` + `embed_fingerprint` that decide what
  still counts as pending — §3), `llm_cache` (cached rerank scores and query
  expansions, pruned to the 1,000 newest) and `store_config` (the config hash it
  last synced). Scores survive a re-index: both cache keys are
  content-addressed (rerank on `{query, model, chunk}`, expansion on
  `{query, model}`), so a changed document simply gets new keys and its old
  score ages out, and since fork `d01b049` (#1366) neither `qmd update`, `qmd
  collection add` nor the SDK `update()` empties the table — upstream wiped it
  on every re-index, so a score lived one watcher cycle. The bound is the prune
  to the 1,000 newest inside `setCachedResult` (it fires on ~1% of writes, so
  the table can run a little past 1,000); the explicit clears are `qmd cleanup`
  (§5) and `store.clearCache()`.
- **What gets indexed.** Every `*.md` under a collection's path, **except any
  path with a dot-prefixed component** (`skills/.archived/**` is never indexed).
  Front matter is NOT stripped: it is in the FTS body and in chunk 0's embedding.
  The title is the first `#` or `##` heading anywhere in the file (a fork branch,
  `title-fix`, prefers front matter and the H1 and was measured neutral once the
  OR keyword leg landed, so it is not merged).
- **Collections** (14): the 11 the recall searches (`memory`, `knowledge`,
  `projects`, `personal`, `work`, `skills`, `architecture`, `lloyd`,
  `autonomy`, `backlog`, `people`) plus `autonomy-runs`, `sessions` and
  `subliminal`. The recall never names those three: `subliminal` is the whole
  vault as one collection, and a hit that arrives through it is folded back onto
  its segment path by `agent_mcp/vault.py::_qmd_normalize_global`. They are still
  embedded, so they count toward every re-embed. The vault's
  `architecture/` is an old OpenClaw-era set; the current docs in
  `~/lloyd/architecture/` are not a collection (measured, not adopted:
  `architecture/retrieval.md` §4).
- **Paths come back percent-encoded** (`encodeQmdPath`, per segment). Lloyd
  decodes them at `agent_mcp/vault.py::_qmd_post` (`qmd_file`).

## 3. Models, and the one place they are set

| role | model | where it runs |
|---|---|---|
| embed | Qwen3-Embedding-0.6B Q8_0 (since 2026-09-21; embeddinggemma-300M before) | every query's vec leg; every `embed` |
| rerank | Qwen3-Reranker-0.6B Q8_0 (cross-encoder) | only when a request asks for `rerank` — the recall's fallback path, `vault_search`, backlog dedupe |
| generate | qmd-query-expansion-1.7B | qmd's own `query` expansion; Lloyd's REST calls never ask for it (measured, not adopted) |

- **`models:` in `~/.config/qmd/index.yml` is the switch**, and it beats the
  `QMD_EMBED_MODEL` variable (`src/llm.ts::resolveEmbedModel`). The daemon, the
  watcher, task #81 and the cleanup timer all read it. A named index reads its
  own `~/.config/qmd/<name>.yml` (`QMD_CONFIG_DIR` redirects the directory): the
  regression pin reads `evalpin.yml`, which must name the same embed model,
  because the pin serves a copy of production's vectors. The committed template
  is `agent-services/conf/qmd-index.yml`; task #81 reports template↔live drift.
- **Changing the embed model is a full re-embed, and it starts by itself.**
  Pending is counted *per configured model* — `getHashesNeedingEmbedding` joins
  on `model` + `embed_fingerprint`, not on the content hash — so the moment
  `models:` names a different embed model every hash in the index reads as
  unembedded, and the watcher's next `embed` begins rewriting the live index
  beside the running daemon — the watcher has no cap of its own, so it is still
  the first responder to that edit (#1367 leaves it open). Task #81's backfill
  refuses it instead: `scripts/maintenance/qmd_index_maintenance.py` compares
  `pending_embeddings()` with the `documents` count the same job reports and, past
  `EMBED_PENDING_MAX_RATIO` — `0.25`, a quarter of the index — invokes no
  `qmd embed` at all, recording `model_change_suspected` with the pending count,
  the denominator it used and the configured embed model in its dated report, and
  exiting 0 because a refused guard is the check working. A deliberate switch is
  still run by hand (or as a side copy, as below); the cap protects the
  unattended path, not the operation. Two models cannot share `vectors_vec`: it
  is keyed `hash_seq` with no model column, and a dimension change is the only
  case that gets caught, as a
  hard error where the vec0 table is created (768 → 1024 for Qwen3) — an
  interrupted same-dimension re-embed simply leaves both models' vectors in one
  table. A changed *title* does not change a content hash, so re-titling alone
  never re-embeds anything. That is why the 2026-09-21 switch was built as a side
  copy with every collection re-embedded (~45k chunk vectors at the time, ~1 h at
  15–25 chunks/s on the 3090), caught up with `update` + `embed` against a scratch
  config, and swapped the files with the watcher and daemon stopped; the old index
  is kept as `index.sqlite.bak-gemma-20260921`. For the live count, read the
  daemon's own `GET /health → vecIndex.vectors` rather than any figure here.
- **Daemon knobs** live in `agent-qmd-daemon.conf`'s `environment=` and nowhere
  else (the regression pin reads them from there):
  - `QMD_RERANK_WINDOW_CHARS=1200` — the reranker reads a 1200-char window of the
    best chunk (MRR 0.504 against 0.484 for whole chunks, and faster);
  - `QMD_RERANK_CONTEXT_SIZE=2048` — enough for that window, ~0.9 GB less VRAM;
  - `QMD_RERANK_PARALLELISM=4` — NOT a speed knob: 4/8/16 contexts measure the
    same, the cross-encoder is compute-bound on the 3090;
  - `QMD_LLM_IDLE_TIMEOUT_MS=0` — models stay resident;
  - `QMD_FUSION_DEPTH=100` — each global-fusion leg fused 100 deep, not 20
    (`architecture/retrieval.md` §3.6).

## 4. The API Lloyd uses

`POST /query` with pre-expanded searches (Lloyd never asks qmd to expand):

| key | meaning |
|---|---|
| `searches` | `[{type, query}]`, type `lex`, `vec` or `hyde` — single-line, no `-term` on vec (sanitize first) |
| `collections` | names to search; omitted means all |
| `limit`, `candidateLimit` | rows returned; rows fused before any rerank. Set both explicitly |
| `rerank` / `skipRerank` | cross-encoder on/off — always explicit |
| `fusion: "global"` | one score-merged ranking across the named collections |
| `collectionFloor` | a number or `{collection: n}` — floor rows are appended after the fused head |
| `lexMode: "or"` | OR the lex terms (default AND) |
| `lexWeight`, `minScore`, `intent`, `rerankWindowChars` | tuning; Lloyd leaves them unset |

The reply carries `results` (`file` as `qmd://collection/path`, encoded; `title`,
`snippet`, `score`) and `meta` (`reranked`, `rerankFallback`, `ms`, `phases`).
`GET /health` carries uptime, rerank counters and `vecIndex` (vectors, full builds,
incremental refreshes). Callers in Lloyd:

- `agent_mcp/vault.py` — the recall doc leg (`recall_doc_leg_shape`), `vault_search`,
  entity lookups; `_qmd_post` is the door that folds `meta` into
  `app/qmd_health.py`;
- `agent_mcp/backlog_similar.py` — write-time dedupe (vec only, reranked, `backlog`);
- `app/routers/memory.py` — Mission Control's memory search;
- `scripts/automod/evalpin.py` — the regression pin's warm-up
  (`production_payload`, read from the recall's own shape).

The dedupe and Mission Control callers are **not** behind that door: each opens
its own `urllib` request (`agent_mcp/backlog_similar.py::semantic_candidates`,
`app/routers/memory.py::memory_search`), reads only `results`, and so reports no
`meta` to `qmd_health` — a rerank that could not run on either path is invisible,
and rule A of the dedupe silently degrades to the lexical rule. `memory_search`
also sends the raw query without `_qmd_sanitize` (`semantic_candidates` now
sanitizes). Both re-checked in the code on 2026-09-25. The gap is #1498; it
was first recorded on 09-22 inside #302, an older and already-closed rerank
item, so until then it had no open owner.

## 5. Keeping it healthy

- **Watcher** (`agent-qmd-watcher`, `agent-services/scripts/qmd-watcher.sh`):
  inotify on the vault and the session export tree, a 2 s debounce (capped at
  10 s) and a 60 s cooldown between `qmd update` + `embed` runs — a no-op `update` re-hashes every
  file (~7.5 s), so the debounce alone did not bound it while the automod loop
  writes continuously.
- **Nightly cleanup** (`lloyd-qmd-cleanup.timer`, 04:45): `qmd cleanup` prunes
  orphaned vectors, drops inactive document records and orphaned content hashes,
  **empties `llm_cache`** and vacuums — the unit's own log for 2026-09-22 reads
  3,932 chunks, 48 documents, 24 hashes and 58 cached responses. This is now the
  only routine emptying of the cache (a re-index keeps it since #1366, §2), so
  rerank scores live up to a day, pruned to the 1,000 newest in between. Unpruned, the
  vectors once reached 99.5% of rows and a 24 GB index, and they displace real
  results.
- **Task #81** (`scripts/maintenance/qmd_index_maintenance.py`): orphan prune,
  embedding backfill, template↔live drift report. It no longer stops the daemon
  (it did on 8 of 8 runs, for one to four documents the watcher would have
  embedded anyway). It also reports vec0 occupancy (live rows over allocated
  slots, from the sqlite-vec shadow tables) with a `need_capacity` verdict the
  orphan ratio cannot give, and the footprint as main + `-wal` + `-shm` (#844).
  Neither cleanup nor VACUUM reclaims a dead vec0 slot; the verdict is report-only,
  and the rebuild it points at is a hand side-copy-and-swap like the 09-21
  model switch.
- **A rerank that could not run says so.** No VRAM for a ranking context used to
  be an HTTP 200 with fusion-order results; `meta.reranked` is false, the daemon
  counts it, and `app/qmd_health.py` logs and announces it.

## 6. Eval pins

`scripts/automod/evalpin.PinnedCorpus` snapshots the index (`VACUUM INTO`) and
serves it from a second daemon on its own port with production's program
environment, so an eval measures a frozen corpus. A pin shares GPU 0 with
production; the regression runner's pins ran 42% of wall time on 2026-09-21 and
slowed production's reranks ~1.8×. Experiments use their own index name and port
(`reap_stale` matches only its own), and prepared snapshots (a re-embedded or
re-titled copy) are served by patching `snapshot()` — the pattern
`architecture/retrieval.md` §2 describes. A candidate fork build is measured
the other way round: the pin serves the same snapshot, and a patched
`production_daemon()` swaps the `qmd.js` path for the candidate worktree's
`dist/cli/qmd.js` and adds any environment it needs. That is how #1475's
fusion depth was measured before the fork change landed. Run the candidate
once with no new setting first; it has to reproduce production's numbers, or
the build rather than the setting is what changed.

## 7. Performance (2026-09-21)

- **Fusion alone** (rerank off, global, ~32 rows): ~50–200 ms per request,
  phases in the daemon log (`fts`, `embed`, `vec`, `chunk`). Re-checked
  2026-09-25 with `QMD_FUSION_DEPTH=100`: 119–140 ms warm. The first query
  after a daemon restart pays ~2 s of embedding-model load.
- **Cross-encoder**: ~56 rows/s on a quiet 3090 — 40 rows ≈ 1.8 s, 240 rows ≈
  5–7 s under the GPU's normal load. The recall no longer pays it except on its
  fallback path; djev orders the pool on GPU 2 instead.
- **Any write to the index** used to force a full blocking rebuild of the
  in-memory vector index on the next query (+0.85 s); since the incremental
  refresh it is ~165 ms for a small change.
- **Measurement traps**: a TTS restart runs a ~4 min compile that pins GPU 0 and
  makes qmd read 4× slow; an eval pin on GPU 0 does the same to production;
  repeated query text is answered from the rerank cache in ~0.1–0.25 s (a fresh
  cross-encoder recall of the same shape: ~7 s cold, ~4 s rerank), and since
  #1366 that hit survives watcher cycles until the nightly cleanup (§2, §5) — so
  a "warm" number is a cache hit unless the query text is new, however many
  vault writes came between.

## 8. Files

- `~/lloyd/qmd` (fork): `src/store.ts` (indexing, FTS, fusion, `structuredSearch`),
  `src/vecindex.ts`, `src/llm.ts` (models), `src/mcp/server.ts` (REST), `WORKLOG.md`
- `agent-services/supervisor/conf.d/agent-qmd-daemon.conf`, `agent-services/scripts/qmd-watcher.sh`
- `agent-services/systemd/lloyd-qmd-cleanup.service` + `.timer` (the user-scope pair symlinked into `~/.config/systemd/user/`)
- `agent-services/conf/qmd-index.yml` (template), `~/.config/qmd/index.yml`, `~/.config/qmd/evalpin.yml`
- `scripts/maintenance/qmd_index_maintenance.py`, `scripts/qmd_fork_landing.py`, `scripts/automod/evalpin.py`
- `app/qmd_health.py`, `agent_mcp/vault.py`
- tests: `test_qmd_single_build.py`, `test_qmd_index_template.py`, `test_qmd_index_maintenance.py`, `test_qmd_fork_landing.py`, `test_qmd_query_shape.py`, `test_qmd_health.py`, `test_qmd_maintenance_health.py`, `test_service_health_check_qmd.py`, `test_qmd_doc_claims.py`

## Review log

- **2026-09-25** — `current`. Re-checked against the tree, both configs and
  the running daemon: fork at `079c9c9` on `lloyd`, level with `origin/lloyd`;
  the six `QMD_*` knobs in `agent-qmd-daemon.conf` (now including
  `QMD_FUSION_DEPTH=100`, §1 and §3); the 14 collections and the 11 the recall
  searches (`VAULT_SEGMENTS`); the three models in `index.yml` and
  `evalpin.yml`; BM25 weights 1.5 / 4.0 / 1.0; watcher 2 s / 10 s / 60 s;
  cleanup at 04:45; `EMBED_PENDING_MAX_RATIO = 0.25`; the kept Gemma backup;
  every path in §8. Corrected: §4 cited #302 for the two callers that bypass
  `_qmd_post`, but #302 is an older, closed rerank item that the finding had
  been merge-appended into, so the gap was unowned; it is #1498 now, and
  `semantic_candidates` has since started sanitizing. §8 was missing
  `test_qmd_maintenance_health.py`. Added: §6's candidate-build pin pattern and
  §7's warm figure at fusion depth 100. #1367 put its cap on task #81 only;
  the watcher still has no guard of its own, as §3 says.
- **2026-09-22** — `current`; the mechanism still runs (fork at `db52729` with
  `dist/` rebuilt and pushed to `origin/lloyd`, daemon serving `[::1]:8181` off
  that tree, #81 `up_next`, template↔live drift reporting 2 items) but three
  sentences were wrong, and all three were about *when work starts on its own*:
  pending embeddings are counted **per configured model**, so an embed-model edit
  is a trigger rather than inert and task #81's backfill had then no cap at all —
  it has one now, `EMBED_PENDING_MAX_RATIO` in §3 (#1367);
  `update()` empties `llm_cache` whole on every watcher cycle, so the cache both
  §2's retention and §7's measurement trap rely on is empty in practice — 0 rows
  after 13 h and 83 reranked documents (#1366; fixed 2026-09-24 in fork
  `d01b049`: a re-index keeps the cache, pruned to the 1,000 newest); and `_qmd_post` is not the one
  door, because `app/routers/memory.py` and `agent_mcp/backlog_similar.py` open
  their own sockets and read only `results` — their rerank fallbacks go uncounted
  and `memory_search` sends an unsanitized query (#302). Corrected too: the
  nightly cleanup also empties the cache and drops inactive documents and orphaned
  hashes, not only vectors; `lexMode` and the `update` pending-hint fix had no
  `WORKLOG.md` section (added as §8 on 2026-09-24); `agent-qmd-daemon.conf`'s
  documented one-line revert names the published `@tobilu/qmd` that was
  uninstalled on 09-19 (#1368); and §8 was missing `test_qmd_query_shape.py` and
  `test_service_health_check_qmd.py`. No vector or document count is pinned here
  any more — §3 says to read `GET /health → vecIndex.vectors`.
