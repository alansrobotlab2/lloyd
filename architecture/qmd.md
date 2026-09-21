---
title: qmd — the vault search engine, Lloyd's fork, and how it runs
status: implemented
date: 2026-09-21
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

What the fork adds over upstream, newest first (`WORKLOG.md` has each one):

| change | what it does | since |
|---|---|---|
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
  log    agent-services/logs/agent-qmd-daemon.err — one line per /query with phase ms
```

- **Binds `[::1]:8181` only.** An IPv4-literal probe cannot see it
  (`scripts/service_health_check.py` asks both).
- **The index** holds `documents` (path, title, hash, collection, active),
  `content` (whole file text by hash), `documents_fts` (FTS5 over path, title and
  body; BM25 weights 1.5 / 4.0 / 1.0), `content_vectors` + `vectors_vec` (one
  vector per chunk, keyed `hash_seq`), `llm_cache` (1,000 most recent rerank
  scores) and `store_config` (the config hash it last synced).
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
- **Changing the embed model is a full re-embed.** Vectors of two models cannot
  share `vectors_vec` (the dimension is fixed at creation; 768 → 1024 for Qwen3),
  and a changed title or model does not change a content hash, so nothing
  re-embeds by itself. The 2026-09-21 switch built a side copy with every
  collection re-embedded (~45k chunk vectors, ~1 h at 15–25 chunks/s on the
  3090), caught it up with `update` + `embed` against a scratch config, and
  swapped the files with the watcher and daemon stopped; the old index is kept as
  `index.sqlite.bak-gemma-20260921`.
- **Daemon knobs** live in `agent-qmd-daemon.conf`'s `environment=` and nowhere
  else (the regression pin reads them from there):
  - `QMD_RERANK_WINDOW_CHARS=1200` — the reranker reads a 1200-char window of the
    best chunk (MRR 0.504 against 0.484 for whole chunks, and faster);
  - `QMD_RERANK_CONTEXT_SIZE=2048` — enough for that window, ~0.9 GB less VRAM;
  - `QMD_RERANK_PARALLELISM=4` — NOT a speed knob: 4/8/16 contexts measure the
    same, the cross-encoder is compute-bound on the 3090;
  - `QMD_LLM_IDLE_TIMEOUT_MS=0` — models stay resident.

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
  entity lookups; `_qmd_post` is the one door and folds `meta` into
  `app/qmd_health.py`;
- `agent_mcp/backlog_similar.py` — write-time dedupe (vec only, reranked, `backlog`);
- `app/routers/memory.py` — Mission Control's memory search;
- `scripts/automod/evalpin.py` — the regression pin's warm-up
  (`production_payload`, read from the recall's own shape).

## 5. Keeping it healthy

- **Watcher** (`agent-qmd-watcher`, `agent-services/scripts/qmd-watcher.sh`):
  inotify on the vault and the session export tree, a 2 s debounce (capped at
  10 s) and a 60 s cooldown between `qmd update` + `embed` runs — a no-op `update` re-hashes every
  file (~7.5 s), so the debounce alone did not bound it while the automod loop
  writes continuously.
- **Nightly cleanup** (`lloyd-qmd-cleanup.timer`, 04:45): `qmd cleanup` prunes
  orphaned vectors. Unpruned, they once reached 99.5% of rows and a 24 GB index,
  and they displace real results.
- **Task #81** (`scripts/maintenance/qmd_index_maintenance.py`): orphan prune,
  embedding backfill, template↔live drift report. It no longer stops the daemon
  (it did on 8 of 8 runs, for one to four documents the watcher would have
  embedded anyway).
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
`architecture/retrieval.md` §2 describes.

## 7. Performance (2026-09-21)

- **Fusion alone** (rerank off, global, ~32 rows): ~50–200 ms per request,
  phases in the daemon log (`fts`, `embed`, `vec`, `chunk`).
- **Cross-encoder**: ~56 rows/s on a quiet 3090 — 40 rows ≈ 1.8 s, 240 rows ≈
  5–7 s under the GPU's normal load. The recall no longer pays it except on its
  fallback path; djev orders the pool on GPU 2 instead.
- **Any write to the index** used to force a full blocking rebuild of the
  in-memory vector index on the next query (+0.85 s); since the incremental
  refresh it is ~165 ms for a small change.
- **Measurement traps**: a TTS restart runs a ~4 min compile that pins GPU 0 and
  makes qmd read 4× slow; an eval pin on GPU 0 does the same to production;
  repeated query text is answered from the rerank cache in ~0.2 s.

## 8. Files

- `~/lloyd/qmd` (fork): `src/store.ts` (indexing, FTS, fusion, `structuredSearch`),
  `src/vecindex.ts`, `src/llm.ts` (models), `src/mcp/server.ts` (REST), `WORKLOG.md`
- `agent-services/supervisor/conf.d/agent-qmd-daemon.conf`, `agent-services/scripts/qmd-watcher.sh`
- `agent-services/conf/qmd-index.yml` (template), `~/.config/qmd/index.yml`, `~/.config/qmd/evalpin.yml`
- `scripts/maintenance/qmd_index_maintenance.py`, `scripts/qmd_fork_landing.py`, `scripts/automod/evalpin.py`
- `app/qmd_health.py`, `agent_mcp/vault.py`
- tests: `test_qmd_single_build.py`, `test_qmd_index_template.py`, `test_qmd_index_maintenance.py`, `test_qmd_fork_landing.py`, `test_qmd_health.py`
