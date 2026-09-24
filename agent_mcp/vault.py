#!/usr/bin/env python3
"""
Lloyd MCP Server: Vault — obsidian vault read/write/search and hybrid recall.

Tools:
    vault_read, vault_write, vault_overview, vault_search, vault_recall (5 tools)

Vault root: ~/obsidian/
QMD daemon: http://localhost:8181/query

Split out of agent_mcp/memory.py as part of Task #340 PR 5. Owns:
    - Vault file read/write with case-insensitive resolution
    - Audit log for writes (~/obsidian/memory/audit/writes.jsonl)
    - QMD daemon hybrid search (BM25 + vector) with stopword cleanup
    - vault_recall, the parallel doc + fact retrieval entry point

Imports from agent_mcp.retrieval for the fact-side of vault_recall
(entity extraction, graph expansion, fact ranking) — the shared core
also used by agent_mcp.facts.
"""

import asyncio
import concurrent.futures
import datetime
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

from app.atomic_io import commit_lock, write_text_durable
from app.config import service_url
# The loaded-memory byte ceiling, shared with `memory_add`/`memory_replace` and
# Write/Edit. Stdlib-only module, number and wording owned by `prompt_surface`
# (#1010).
from app.memory_ceiling import memory_write_error
from mcp.types import Tool

from agent_mcp._shared import (
    VAULT,
    ErrorCode,
    _QUERY_STOPWORDS,
    _err,
    _wrap,
    get_bound_session,
)
from agent_mcp.retrieval import (
    FACT_GODNODE_THRESHOLD,
    FACT_RANK_CAP_SEED,
    FACT_RANK_CAP_GRAPH,
    _edge_counts_or_empty,
    extract_entities_from_query,
    fact_matches_tokens,
    fact_query_tokens,
    fact_score,
    get_facts_sync,
    graph_weighted_neighbors,
)
import math

logger = logging.getLogger("lloyd-vault")

# ── Constants ────────────────────────────────────────────────────────────────

AUDIT_LOG_DIR = VAULT / "memory" / "audit"
AUDIT_LOG_FILE = AUDIT_LOG_DIR / "writes.jsonl"
QMD_DAEMON_URL = service_url("qmd", "http://localhost:8181/query")

# The qmd collections vault_search fans out over and vault_recall filters on.
# `facts` is deliberately absent: qmd's `facts` collection points at
# ~/obsidian/facts, which exists and is empty — the real fact tree is
# _pipeline/vault-derived/facts and reaches retrieval through the knowledge
# graph and fact layer, not qmd. Until 2026-09-07 it was listed anyway, so one
# of every twelve fan-out requests went to an empty collection. Indexing the
# real fact tree in qmd (61k files, ~3x the index) is a separate, measurable
# decision; see qmd/GAMEPLAN.md item 4.
VAULT_SEGMENTS = [
    "memory", "knowledge", "projects", "personal", "work", "skills",
    "architecture", "lloyd", "autonomy", "backlog", "people",
]
# Over-request factor for the unrestricted path in `_qmd_daemon_search`, and the
# ceiling on it. `limit` is the size of the *answer*; these size the *pool* it is
# ranked out of.
#
# The factor used to be 2, for two reasons that are now both on the other side of
# the argument. (1) `subliminal` re-indexes the whole vault, so every hit arrived
# twice — that duplication is gone now the unrestricted path names its segments
# (#504). (2) qmd derives its retrieval width from `limit`, so a bigger ask widens
# the pool — true, and now the whole point: for the eval query
# `kg-maintenance-tasks` a 40-row pool yields 0 of its 5 expected documents, 60
# yields 1, and 240 yields all five (measured 2026-09-14 on the live daemon, named
# segments, rerank on). 3x keeps a small caller's pool a real pool (limit 5 -> 15)
# and `QMD_POOL_MAX` stops the factor running away for a caller already asking for
# hundreds — `vault_search`'s `max_results` reaches it, and so does the doc leg's
# own `RECALL_DOC_POOL`.
#
# "Measured free" was true of the *global* arm. On the named arm — the one #504
# switches to — the ask is the only thing that decides how much qmd does. What it
# costs was quoted here for a while as a pair of ~100-200 ms figures labelled
# "rerank on, warm"; they were CACHED REPEATS. `_qmd_daemon_search` warns two
# screens below that the daemon caches query embeddings so a sequential A/B
# measures arm order, and the figures priced here had been taken off exactly that
# cache. Re-priced 2026-09-18 by two rounds, 3 and 5 samples per cell, live
# daemon, named segments, a fresh never-seen query text per sample, rerank ON:
# 2,161-2,291 ms FRESH at a 40-row pool, 3,672-4,027 ms FRESH at 240. The
# identical repeat of a query the daemon had already run at 240 came back in
# 119-183 ms, CACHED — which is the shape those old figures had, and it understates
# the same unseen request by 20-34x. The rerank is the whole difference: the same
# 240-row ask with the cross-encoder skipped returns in 150-210 ms FRESH, so what
# the widening bought is roughly 1.5-1.7 s of cross-encoding rows 41-240 per named
# recall.
#
# So the widening is not free, and the number that governs it is no longer a
# sample in a comment: `LATENCY_BUDGET_MS` in
# `workers/sources/automod_regression.py` sets the ceiling this shape is read
# against (nightly 4,800 ms, set over the worst nightly run on disk at 4,408.0 ms,
# which is `nightly-20260904-20260904-060219.json` — nine days before #504) and
# reports one that crosses it (#1129).
QMD_POOL_FACTOR = 3
QMD_POOL_MAX = 240
# The size of the pool `_vault_recall`'s document leg ranks its answer out of,
# asked for by name because it is the recall path's decision, not a property of the
# search: `vault_search` and the entity lookup ask for a document list and take
# `QMD_POOL_FACTOR` times it, which is right for them and not nearly deep enough
# here. The value is `QMD_POOL_MAX` for one reason — for the query this replaced,
# "which autonomy tasks maintain the knowledge graph?", the daemon returned 1 of the
# 5 expected files at a 60-row pool, 3 at 120 and all 5 at 240 (live daemon, named
# segments, rerank on, measured 2026-09-14). Asking for the whole pool also means
# `QMD_POOL_FACTOR` has nothing left to widen, so the ask, the fold and the
# pre-rank slice are one number stated once.
RECALL_DOC_POOL = QMD_POOL_MAX

# ── Global fusion (2026-09-19): rank 40 well-chosen rows instead of 240 ───────
#
# Why the pool above had to be 240 is a fusion artefact, not a property of the
# corpus. qmd handed RRF one list per search per collection, and RRF reads ranks
# only: every collection's #1 ties with every other's, so a document ranked fifth
# in the one relevant collection lands near fused position 50 of an eleven-
# collection request. #504 kept it by ranking everything, and the cross-encoder
# is compute-bound on the 3090 (~56 rows/s; 4, 8 and 16 ranking contexts measure
# 4.03 / 3.98 / 4.16 s) — so a recall cost ~4.2 s alone, and at loop depth 4
# several of them queue on one daemon: 20-60 s each, past the 15 s client timeout
# (p99 20.4 s over 13,852 recalls; 1.3% timed out).
#
# The fork's `fusion: "global"` runs each search ONCE across the named
# collections and merges by score (BM25 and cosine are comparable inside one
# index), so the fused head is the best documents anywhere. What a global ranking
# costs is a small collection outscored wholesale: `autonomy` (36 task files —
# front matter and an activity log, poor lexical AND semantic matches for a
# natural question) lost every expected file, ranks 116-352. `collectionFloor`
# guarantees that collection's best five per search a look from the reranker.
#
# Pinned eval, 20 queries, full `_vault_recall`, rerank cache cleared per arm
# (qmd/WORKLOG.md 7.4 has every arm):
#
#     fusion      pool  floor        doc_hit  doc_recall  MRR    NDCG@10  recall
#     collection  240   -            1.00     0.610       0.497  0.582    ~4.7 s   <- was
#     collection   40   -            0.80     0.525       0.477  0.515
#     global       40   -            0.95     0.558       0.512  0.588    ~1.3 s
#     global       40   autonomy=5   1.00     0.578       0.532  0.603    ~1.4 s   <- deployed 09-19
#     global       60   autonomy=5   1.00     0.546 MRR, 0.602 NDCG — inside the noise of 40
#
# n=20, so read 0.02 as noise: hit rate is at parity, MRR and NDCG are up, and
# doc_recall is down 0.03 (about one expected document across the set). The
# floor was chosen by reading this eval's misses, which is a fitting risk stated
# here rather than hidden; what justifies it is the structure of that collection,
# and the durable fix is making task files retrievable (index their
# `description`), after which the floor can go.
#
# **Re-measured on the 87-query set (2026-09-21, #1335), and the autonomy-only
# floor did NOT hold.** One pinned snapshot per comparison, paired per query
# against collection-240, 95% bootstrap intervals; p50 is per-query recall on the
# pin beside the live daemon:
#
#     arm                               doc_hit                doc_recall             MRR     p50
#     collection 240 (was)              0.563                  0.411                  0.166   7.5 s
#     global 40, autonomy=5             -0.115 [-0.195,-0.034] -0.085 [-0.158,-0.018] +0.025  1.9 s
#     global 40, autonomy/arch/skills=5 -0.069 [-0.138,+0.000] -0.050 [-0.110,+0.008] +0.035  2.2 s  <- deployed
#     global 40, every collection=2     -0.092 [-0.172,-0.023] -0.071 [-0.138,-0.008] +0.024  2.1 s
#     global 60, autonomy/arch/skills=5 -0.057 [-0.115,+0.000] -0.041 [-0.091,+0.004] +0.039  2.8 s
#     global 80, autonomy=5             -0.103 [-0.172,-0.046] -0.083 [-0.144,-0.029] +0.029  3.1 s
#
# NDCG@10 is inside +-0.06 of zero on every arm. What global fusion loses is HITS,
# and it loses them the way it lost autonomy's: the small collections are
# outscored wholesale. `skills/retrieval-eval/SKILL.md` alone was four of the
# twelve queries global-40 lost (global rank 78 or past 240, where per-collection
# fusion had it at 13-50). A wider pool barely helps (80 is no better than 40);
# floors on the small collections do. `architecture` (36 docs) and `skills` (249)
# join `autonomy` (36). The remaining gap is -0.07 on doc_hit with an interval
# that only just reaches zero: the best FAST setting measured, not a proven
# equivalent. The 7.5 s per recall that buys the rest is not on offer (Alan,
# 2026-09-21); a deep pool at speed is #1336's route (djev on GPU 2).
#
# `RECALL_QMD_FUSION = "collection"` is the kill switch: it restores the old
# request exactly, 240-row pool included. An older daemon ignores the two new
# keys, which would mean per-collection fusion at a 40-row pool — the worst arm
# above — so `recall_doc_pool()` is what the doc leg asks for, never the
# constant directly.
RECALL_QMD_FUSION = "global"
RECALL_GLOBAL_DOC_POOL = 40
RECALL_COLLECTION_FLOOR = {"autonomy": 5, "architecture": 5, "skills": 5}


# ── djev ranks the recall (#1336, 2026-09-21) ────────────────────────────────
#
# qmd's cross-encoder is compute-bound on the 3090 it shares with TTS, the
# desktop and the regression pin (~2.2 s p50 for the 40-row pool above, 5-8 s
# under the pin). djev (GPU 2) ranks listwise in one read. Measured on the
# 87-query pinned eval, every arm reordering EXACTLY the same candidates:
#
#     same <=32-row pool, ordered by     doc_hit  MRR    NDCG@10  recall p50
#     qmd cross-encoder                  0.425    0.193  0.214    1.04 s
#     djev, 160 chars, one read          0.471    0.249  0.274    0.51 s
#     djev, full candidates, one read    0.437    0.226  0.251    0.87 s
#     fusion order, nothing              0.414    0.155  0.179    0.15 s
#
# djev beat the cross-encoder on the same pool (doc_hit +0.046 [+0.011,+0.092],
# NDCG +0.060 [+0.004,+0.114]), and shorter candidates beat longer ones. The
# shape then chosen, against the cross-encoder path above on the same pin:
#
#     global 40 + floors 5 + cross-encoder     0.494  0.201  0.234    2.18 s
#     head 20 + floors 2/2/2, djev 160 x1      0.517  0.256  0.275    0.52 s  <- deployed
#       paired: doc_hit +0.023 [-0.057,+0.103]  doc_recall -0.005 [-0.067,+0.058]
#               MRR +0.055 [-0.011,+0.122]      NDCG +0.041 [-0.028,+0.109]
#
# Equivalent on every metric and 4x faster, which is a win by Alan's rule. A
# repeat of one read gave the same outcome on 87/87 queries, though its scores
# do not repeat exactly (`architecture/djev.md` §8.2); `samples:
# "auto"` cost 200 ms and ranked worse; 100 chars lost hits and 240 ranked
# worse. The ceiling is the pool, not djev: no ~32-row pool holds as many
# findable documents as collection-240 did, and djev cannot rank more than the
# 32 rows one 128-token canvas holds (`djev.CANVAS_CHUNK_QUESTIONS`).
#
# `RECALL_RERANKER = "qmd"` is the kill switch: the request above, byte for
# byte. A djev that does not answer (down, timed out, a split canvas) sends the
# recall down that same path, counted and announced by `app/qmd_health.py`, so
# an outage costs speed, never quality. `djev.enabled` false means "qmd".
RECALL_RERANKER = "djev"

# ── The keyword leg ORs its terms (2026-09-21) ───────────────────────────────
#
# qmd's lex leg used to AND every non-stopword term, so a natural-language
# question matched only documents holding every one of its words: over the
# 87-query set the lex leg alone had an expected document within its top 32
# for 11% of them (AND) against 36% (OR), and within its top 240 for 13%
# against 54%. End to end on one pin, djev ranking, paired against AND:
#
#     doc_hit +0.011  doc_recall +0.033  MRR +0.028  NDCG +0.042 [-0.005,+0.094]
#     p50 510 ms against 494 ms
#
# and on the cross-encoder fallback path doc_hit +0.046, doc_recall +0.050.
# `lexWeight` 0.5 and 2 measured inside the noise of 1, so it stays unset. The
# fork's `lexMode` is opt-in (qmd fork, `buildFTS5Query`); a daemon that predates
# it ignores the key and ANDs, which is exactly the old request.
RECALL_LEX_MODE = "or"
RECALL_DJEV_HEAD = 20
RECALL_DJEV_FLOOR = {"autonomy": 2, "architecture": 2, "skills": 2}
RECALL_DJEV_POOL = 32     # head + floors can never pass it: 20 + (2+2+2) x 2 searches
RECALL_DJEV_CHARS = 160
RECALL_DJEV_SAMPLES = 1
RECALL_DJEV_TIMEOUT_S = 4.0


def recall_reranker() -> str:
    """Who orders the recall's document pool: "djev" or "qmd"."""
    if RECALL_RERANKER != "djev":
        return "qmd"
    try:
        from app import djev
        return "djev" if djev.enabled() else "qmd"
    except Exception:  # noqa: BLE001 — no djev client means the cross-encoder
        return "qmd"


def recall_doc_leg_shape(reranker: str | None = None) -> dict:
    """What the recall's document leg asks qmd for. The one definition the doc
    leg and `scripts/automod/evalpin.production_payload` both read."""
    if (reranker or recall_reranker()) == "djev":
        return {"limit": RECALL_DJEV_POOL, "candidateLimit": RECALL_DJEV_HEAD,
                "rerank": False, "floor": dict(RECALL_DJEV_FLOOR), "lexMode": RECALL_LEX_MODE}
    pool = RECALL_GLOBAL_DOC_POOL if RECALL_QMD_FUSION == "global" else RECALL_DOC_POOL
    return {"limit": pool, "candidateLimit": pool, "rerank": RECALL_QMD_RERANK,
            "floor": dict(RECALL_COLLECTION_FLOOR), "lexMode": RECALL_LEX_MODE}


def recall_doc_pool() -> int:
    """Rows the recall doc leg asks qmd to return: the pool its ranker orders."""
    return recall_doc_leg_shape()["limit"]

VAULT_EXCLUDE_DIRS = {"templates", "images"}
VAULT_EXCLUDE_FILES = {"tags.md"}

# Demotion patterns for auto-generated "memory churn" files. These all
# lexically contain whatever the user was discussing recently and crowd
# out canonical knowledge sources. Covers:
#  - `memory/YYYY-MM-DD.md` — daily session transcripts
#  - `memory/pipeline/**` — auto-generated pipeline state / skill candidate files
# Other memory/ subdirs (learnings, alan, autonomy-pipeline) NOT demoted —
# they tested as containing genuinely-useful aggregate content.
# Tunable via `demote_daily_logs` + `demote_factor` params on _vault_recall.
_DAILY_LOG_RE = re.compile(
    r"(?:^|/)memory/(\d{4}-\d{2}-\d{2}\.md|pipeline/)"
)
DAILY_LOG_DEMOTE_FACTOR = 0.4

# vault_recall's production defaults, named so eval/run_eval.py can import
# them instead of restating them. The nightly eval ran with graph_rerank
# False and alpha 0.5 while production ran True and 0.3 — it was measuring a
# configuration nothing serves (2026-09-03 review).
# graph_rerank went default-on 2026-05-12 for a measured +6-13% MRR. That
# measurement was never repeated after the graph was rebuilt, and the eval
# that would have caught it ran with rerank OFF — it was scoring a
# configuration production did not serve.
#
# Measured 2026-09-04 on the 20-query set, with the god-node penalty fixed
# (it had been a no-op: the degree map was cased, the voter keys lowercased,
# so every lookup missed and every voter was divided by the same constant):
#
#     rerank off              MRR 0.500   NDCG@10 0.601   3445 ms
#     rerank on, alpha 0.30   MRR 0.386   NDCG@10 0.497   4297 ms
#     rerank on, alpha 0.50   MRR 0.411   NDCG@10 0.503
#     rerank on, alpha 0.70   MRR 0.402   NDCG@10 0.520
#     rerank on, alpha 0.85   MRR 0.419   NDCG@10 0.539
#     rerank on, alpha 0.30, penalty still broken  MRR 0.436  NDCG@10 0.535
#
# Off wins at every alpha and is faster. The knob stays; the default flips.
# That was n=20. Scaled from the 87-query set, where a paired MRR difference
# carries a 95% interval of about +-0.06 (#1335, 2026-09-21), a paired gap at
# n=20 is uncertain to about +-0.11: "off wins" rests on off winning at all five
# alphas, not on any single gap being clear of noise.
RECALL_GRAPH_RERANK = False
RECALL_RERANK_ALPHA = 0.3      # only consulted when rerank is explicitly on

# djev re-ranking of the head of the pool, on GPU 2's idle decision engine.
# OFF, and the constant is what makes flipping it an adoption decision with an
# eval result behind it rather than a default that drifted on.
#
# Measured 2026-09-20 by inverse-cloze over 14 trials of 16 candidates: djev
# MRR 0.766 / recall@1 0.64, against 0.498 / 0.29 for lexical Jaccard and
# 0.146 / 0.00 for random, at 662 ms for 16. But Jaccard is a WEAK baseline
# and the comparison that decides adoption is against qmd's own reranker on
# the labelled set — `eval/run_eval.py --djev-rerank`, which is the only
# reason this arm exists in the handler at all. A shadow log has no labels
# and cannot answer it.
#
# 12 rather than 16: listwise `label_mass` measured 0.446, 0.807 and 0.965 at
# n=16 on three corpora, so 16 is the edge of the safe window. `app/djev.py`
# carries the numbers.
#
# Re-measured on the 87-query set over global fusion (#1335, 2026-09-21, one
# pinned snapshot, two runs per arm, paired per query against the baseline):
# top 12 is +0.053/+0.055 MRR, 95% interval [-0.002,+0.112] and [-0.001,+0.107],
# 15-16 queries better against 7-8 worse; top 8 is +0.048/+0.050 with intervals
# down to -0.007. Consistent in sign, half the n=20 step, never clear of zero, and
# +0.4-0.5 s on every recall. A change that is slower has to show a gain, so the
# arm stays off; a djev that REPLACES the cross-encoder only has to match it
# (#1336).
RECALL_DJEV_RERANK = False
RECALL_DJEV_RERANK_TOP = 12

# ── Saying so when a knob the caller set cannot take effect (#1372) ─────────
#
# #1336 made djev the recall's RANKER, so the first arm of the dispatch takes
# every production call and the `elif` below it that reads `djev_rerank` runs
# zero times — the knob is now inert twice over, once from `RECALL_DJEV_RERANK`
# and once from the ranker. It was still advertised in `vault_recall`'s own tool
# schema when #1372 was filed, and the triage probe confirmed what a caller
# learned from sending it: nothing. Zero shadow rows, zero `_djev_rerank_pool`
# calls, a normal-looking recall — djev-as-ranker when djev-as-reranker was
# asked for, a different decision, reported as silence.
#
# So a result that was not shaped by a knob the caller set names the knob and
# why. A log line is not the fix: the caller reads the result, and the
# alternative — refusing the call — would make an eval harness that passes the
# knob unconditionally unable to recall at all.
RECALL_UNUSED_KNOBS_KEY = "recall_knobs_ignored"
RECALL_ARM_UNUSED_NOTE = (
    "djev_rerank / djev_rerank_top took no effect: the djev rerank arm applies "
    "only when qmd is the recall's ranker, and djev ranked this recall, so the "
    "ordering you got is djev-as-ranker, not djev-as-reranker")
RECALL_KNOB_STRIPPED_NOTE = ("stripped at the tool boundary: eval-only knob, "
                             "and production holds its measured value")
#: The two eval knobs whose inertness has a second cause besides the strip.
_RECALL_ARM_KNOBS = frozenset({"djev_rerank", "djev_rerank_top"})

# Evaluation knobs: read out of `params` by `_vault_recall` for its in-process
# callers (`eval/run_eval.py`, the retrieval tests), and deliberately NOT
# settable by a tool call. Until 2026-09-23 the `vault_recall` schema declared
# all seven, which cost every turn their descriptions for knobs that 0 of 98
# recorded calls ever passed and that production holds at measured values.
# Dropping them from the schema alone would have left them readable from the
# wire under names no schema declares, the surface the #843 review refused for
# `seed_top_k`, so `call_tool` strips them before the handler sees the
# arguments. tests/test_retrieval.py pins both halves.
RECALL_EVAL_KNOBS = frozenset({
    "expand_graph", "graph_rerank", "rerank_alpha", "graph_top_k", "graph_hops",
    "djev_rerank", "djev_rerank_top",
})
RECALL_DEMOTE_DAILY_LOGS = True
RECALL_GRAPH_TOP_K = 5
RECALL_GRAPH_HOPS = 1
# Production's default for graph-expanded recall, named so a claim about it can
# be checked against something. `_vault_recall` reads exactly this when the
# caller omits `expand_graph`, and `eval/run_eval.py` compares its own run
# against it. The eval runs the graph expanded as a deliberate measurement
# choice, so production-match for this knob is false on every default eval run —
# it used to be asserted true by a conjunction term that compared a `store_true`
# flag with itself (#1000).
RECALL_EXPAND_GRAPH = False
# How many of the query's ranked entities become seeds. Raised from 5 to 10 in
# the graph-consistency work: ties at low scores can knock out the canonical
# entity — "Knowledge Graph Consistency" and "Knowledge Graph System" both
# score 0.27 for one query, and with k=5 the canonical one is cut off — so the
# wider slice won and stayed. What did NOT win was the number itself: it lived
# here as the literal `[:10]` inside `_vault_recall` and in `eval/run_eval.py`
# as two literal `[:5]`s, so from 2026-09-11 the nightly eval scored entity
# metrics against a seed set production never assembles (#843). `entity_hit` is
# a union with seeds FIRST and largest, and the error is one-directional: a
# query whose 6th-to-10th seed is the gold entity is a miss in the eval and a
# hit in production. One name, imported by the eval, is the whole fix.
# Deliberately NOT in the `vault_recall` tool schema, and — unlike `demote_factor`,
# `grep_code` and `graph_lookup`, which the schema also omits but this module
# still reads out of `params` — deliberately NOT readable from `params` either.
# `_vault_recall` is registered as the `vault_recall` handler (vault.py:1430) and
# `call_tool` hands it the client's argument dict raw, so a width taken from
# `params` would be an agent-settable retrieval knob under a name no schema
# declares — the "stray key changes retrieval" surface the review of #843
# refused. It is a keyword-only argument instead, so only an in-process caller
# can set it, and the only one that does is the eval, whose job is measuring a
# width. tests/test_retrieval.py pins that a raw MCP call ignores the key.
RECALL_SEED_TOP_K = 10

# qmd's own cross-encoder reranker, distinct from graph_rerank above. This
# client sent `skipRerank: true` on every request from the day it was
# written, and published qmd 2.8.3 silently ignored that key — it only
# reads `rerank` — so vault_recall has been reranked all along. The fork
# in ~/lloyd/qmd honours the flag, which is how the cost of skipping was
# finally measured, 2026-09-07, corpus pinned on identical index snapshots:
#
#     qmd rerank on    MRR 0.484   NDCG@10 0.590   doc_hit 0.95
#     qmd rerank off   MRR 0.323   NDCG@10 0.450   doc_hit 0.85
#
# n=20; at that size a paired MRR gap is uncertain to about +-0.11 (scaled from
# the +-0.06 measured at n=87, #1335), so 0.16 is clear of it, if not by much. The old docstring's "rarely changes top-1"
# was measured against a daemon that never turned it off. Prefetch still
# skips it explicitly (prefetch.py) because it runs inside a latency budget.
RECALL_QMD_RERANK = True

# Canonical-source prefixes for graph_lookup boost. When a graph-derived
# entity name resolves to a file under one of these prefixes, treat it as
# a strong signal — the user almost certainly wants this file, not the
# memory log that mentions it.
_CANONICAL_PREFIXES = (
    "autonomy/", "skills/", "backlog/", "architecture/", "knowledge/", "facts/",
)

# Source-code grep fallback. QMD's index covers `~/obsidian/` only, so
# questions about Lloyd internals (`vault_recall`, `FACT_GODNODE_THRESHOLD`,
# etc.) return nothing relevant. When the query mentions identifier-like
# tokens AND QMD's hit count is thin, fall back to ripgrep over the
# code roots and merge results.
LLOYD_HOME = Path(__file__).resolve().parent.parent
# `LLOYD_CODE_ROOT` repoints the grep corpus, the way `LLOYD_FACTS_ROOT` and
# `LLOYD_KG_DB` repoint the fact tree and the knowledge graph.
#
# It exists because this retriever searches the repository it ships in, which
# makes the code both the thing under test and part of the corpus. The paired
# quality comparison checks the previous commit out into a worktree and runs
# both arms against the live vault — and each arm then grepped ITS OWN source.
# Measured 2026-09-07 on a promotion that touched only an inject string:
# `lloyd-vllm-rel` returned six different files per arm, and ndcg10 and mrr_doc
# each moved 0.0060 with the qmd corpus pinned and no retrieval code changed at
# all. Any commit large enough to add prose to `app/` or `scripts/` moves the
# document metrics, for reasons that have nothing to do with retrieval quality.
#
# Pointing both arms at one tree makes the document corpus identical across
# them. Unset, behaviour is exactly as before: grep this checkout.
_CODE_ROOT = Path(os.environ["LLOYD_CODE_ROOT"]).resolve() \
    if os.environ.get("LLOYD_CODE_ROOT") else LLOYD_HOME
# The same tree, named. `eval/run_eval.py` records it in its artifact (#1374)
# because it is the second half of the document corpus: two runs whose qmd
# vector counts agree but whose keyword leg grepped different checkouts did not
# score the same thing. It is a public name so the recorder reads THE value the
# grep below uses rather than re-deriving it from the env a second way.
LLOYD_CODE_ROOT = _CODE_ROOT
LLOYD_CODE_ROOTS = [
    _CODE_ROOT / "agent_mcp",
    _CODE_ROOT / "app",
    _CODE_ROOT / "scripts",
    _CODE_ROOT / "workers",
]
LLOYD_CODE_PREFIX = str(_CODE_ROOT) + "/"
# Match Python-style identifiers >=4 chars that look code-like:
#   - have an underscore (vault_recall, _relationships)
#   - OR have 2+ uppercase chars (FACT_GODNODE, KGMentionClassifier)
#   - OR have a dot-extension (vault.py, relationships.json)
_IDENT_RE = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]{3,}\b")
_DOTTED_FILE_RE = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]{2,}\.(?:py|json|md|yaml|yml|toml|sh)\b")


def _identifier_tokens(query: str) -> list[str]:
    """Pull identifier-like tokens from a query for the grep fallback.
    Returns dedup'd list preserving order; bounded to 4 to cap subprocess work.
    """
    out, seen = [], set()
    for m in _DOTTED_FILE_RE.finditer(query):
        t = m.group(0)
        if t.lower() not in seen:
            seen.add(t.lower()); out.append(t)
    for m in _IDENT_RE.finditer(query):
        t = m.group(0)
        if "_" in t or sum(1 for c in t if c.isupper()) >= 2:
            if t.lower() not in seen:
                seen.add(t.lower()); out.append(t)
    return out[:4]


def _grep_lloyd_code(query: str, limit: int = 8, timeout: float = 2.0) -> list[dict]:
    """Ripgrep the Lloyd code roots for identifier-like tokens in `query`.
    Returns QMD-shaped result dicts so they merge cleanly. Empty if no
    identifier tokens or rg fails."""
    idents = _identifier_tokens(query)
    if not idents:
        return []
    roots = [str(r) for r in LLOYD_CODE_ROOTS if r.exists()]
    if not roots:
        return []
    found: dict[str, dict] = {}
    for rank, ident in enumerate(idents):
        try:
            proc = subprocess.run(
                ["rg", "--files-with-matches", "--type-add=src:*.{py,ts,tsx,js,sh,yaml,yml,toml,json}",
                 "--type", "src", "-F", ident, *roots],
                capture_output=True, text=True, timeout=timeout,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue
        # Sorted before the `limit` cut. rg searches in parallel and prints
        # files as its threads finish, so the same query on the same tree
        # admitted a different 8 files from run to run (7 of 81 gold queries
        # on 2026-09-21). That made the recall's pool, and so djev's input,
        # vary, and the regression check read the difference as a change.
        for path in sorted((proc.stdout or "").splitlines()):
            if not path:
                continue
            rel = path.removeprefix(LLOYD_CODE_PREFIX)
            if rel in found:
                continue
            # Score: 0.5 for top match per ident, decaying; behind QMD rank-1 (1.0)
            # but ahead of QMD rank-3 (0.33). Multiple-ident match files rise.
            base_score = 0.5 / (1 + 0.15 * rank)
            found[rel] = {
                "file": rel,
                "title": Path(path).name,
                "snippet": f"[code match: {ident}]",
                "score": round(base_score, 4),
            }
            if len(found) >= limit:
                break
        if len(found) >= limit:
            break
    return list(found.values())

CONSOLIDATION_MIN_RESULTS = 4
CONSOLIDATION_TIMEOUT = 10


def _consolidation_endpoint() -> tuple[str, str]:
    """Resolve (chat_completions_url, model_name) for vault consolidation.

    Uses the alias resolver so when `secondary_enabled: false` the call
    routes to primary instead of the dead :8091 endpoint.
    """
    try:
        from app.config import resolve_model_alias, _get_model_cfg
    except Exception:
        return ("", "")
    name = resolve_model_alias("secondary")
    cfg = _get_model_cfg(name) or {}
    base = cfg.get("base_url") or cfg.get("env", {}).get("ANTHROPIC_BASE_URL", "")
    if not base:
        return ("", name)
    return (f"{base.rstrip('/')}/v1/chat/completions", name)

CONSOLIDATION_SYSTEM_PROMPT = (
    "You are a memory consolidation engine. Your job is to take raw search results "
    "from a knowledge vault and produce a concise, deduplicated, well-structured "
    "consolidation that directly answers the user's query.\n\n"
    "Rules:\n1. Deduplicate overlapping content.\n2. Preserve specific facts, dates, names, numbers.\n"
    "3. Return a JSON object with a 'summary' key.\n4. Be concise — under 400 words."
)


# ── QMD helpers ──────────────────────────────────────────────────────────────

def _qmd_sanitize(query: str) -> str:
    """Strip control chars and collapse qmd query-syntax operators to spaces.

    qmd's vec/hyde parser treats `-term` as negation (Google-style) and throws
    HTTP 500 when it sees one. Replace hyphens (and other operator chars qmd
    treats as syntax) with spaces before the query hits either the vec or lex
    leg. The lex path already tokenizes on non-word boundaries so this is a
    no-op there; the vec path gets identical clean input. Fixes #325.
    """
    q = re.sub(r"[\x00-\x1f\x7f]", " ", query)
    q = re.sub(r"[-+\"]", " ", q)
    return re.sub(r"\s+", " ", q).strip()


def _qmd_strip_stopwords(query: str) -> str:
    """Strip stopwords from the FTS5 lex-leg query.

    Uses the wider _QUERY_STOPWORDS (see #327) rather than _ENTITY_STOPWORDS
    so natural-language question framing gets fully removed from BM25 signal.

    Short tokens (len<2) are kept when purely digits so version numbers like
    "Qwen3.5" → tokens [qwen3, 5] don't lose the fractional part.
    """
    words = [
        w for w in re.findall(r"\b\w+\b", query.lower())
        if w not in _QUERY_STOPWORDS and (len(w) >= 2 or w.isdigit())
    ]
    return " ".join(words) if words else query


def _qmd_log(msg: str) -> None:
    print(f"[qmd] {msg}", file=sys.stderr, flush=True)


class QmdUnavailable(RuntimeError):
    """The qmd daemon did not answer. Retrieval **failed**; it did not find nothing.

    Before #407 every non-HTTP-500 problem ended in `return None`, and every
    caller defaulted that to an empty list — so a hung or dead daemon read as an
    empty corpus everywhere except `logs/mcp.err`, 23 `search failed:
    TimeoutError('timed out')` lines behind which the agent was told there were
    no documents. The message is what the MCP handler shows the model
    (`_err(str(exc), ...)`), which is why it names the daemon and says in so
    many words that zero hits is not the finding.

    A `RuntimeError` because this is an operational condition, not a bad
    argument. A caller that would rather degrade than fail — there is exactly
    one, the opt-in enrichment in `_lookup_one` — catches it by name, which is
    the thing the old blanket `except Exception` could not express.
    """


# What a qmd query may cost a live turn. An eval is not a live turn: it would
# rather wait than score an answer it never received, so the paired regression
# check raises this through the environment (`automod_regression.QMD_TIMEOUT_ENV`).
QMD_TIMEOUT_S = 15.0


def _qmd_timeout() -> float:
    """`LLOYD_QMD_TIMEOUT_S` if it is a positive number, else 15 s. Read per
    call: a default bound at import is one no caller can move."""
    try:
        value = float(os.environ.get("LLOYD_QMD_TIMEOUT_S") or 0)
    except ValueError:
        value = 0.0
    return value if value > 0 else QMD_TIMEOUT_S


def qmd_file(file: str) -> str:
    """A qmd result's `file`, decoded. The daemon percent-encodes every path
    segment (`encodeQmdPath`), so `people/Ali Behrouz.md` arrived as
    `people/Ali%20Behrouz.md` and a `vault_read` of the cited path failed — 18
    indexed notes have a space, `+`, `#` or `&` in their path."""
    if not isinstance(file, str) or not file.startswith("qmd://"):
        return file
    return "qmd://" + urllib.parse.unquote(file[len("qmd://"):])


def _qmd_post(payload: dict) -> list:
    req = urllib.request.Request(
        QMD_DAEMON_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=_qmd_timeout()) as resp:
        data = json.loads(resp.read())
    # The daemon says whether the rerank this request asked for actually ran
    # (`app/qmd_health.py`). It never raises and never blocks the recall: an
    # unreranked answer is a worse answer, not a missing one.
    try:
        from app import qmd_health
        qmd_health.note_response(payload.get("rerank") is True, data.get("meta"))
    except Exception:  # noqa: BLE001
        pass
    return [
        {
            "file": qmd_file(r.get("file", "")),
            "title": r.get("title", ""),
            "snippet": r.get("snippet", ""),
            "score": r.get("score", 0),
        }
        for r in data.get("results", [])
    ]


def _qmd_normalize_global(rows: list, allowed: set) -> list:
    """Fold a whole-index result set back onto the vault-segment view.

    The global-scan path (see `_qmd_daemon_search`) searches every collection
    in the index, which is a superset of `VAULT_SEGMENTS` in two ways:

      - `subliminal` is the entire vault under one collection, so every hit
        arrives twice — once as `qmd://knowledge/x.md` and once as
        `qmd://subliminal/knowledge/x.md`. Strip the prefix and they are the
        same document.
      - `autonomy-runs` (3,746 files) is indexed but has never been part of
        what vault retrieval searches.

    Keeping only paths whose first component is a known segment reproduces
    the old corpus exactly and drops `autonomy-runs`, `agents/` and
    `templates/` in one rule. Deduping keeps the higher score: the same file
    can arrive from two collections with slightly different RRF scores.
    """
    best: dict = {}
    for r in rows:
        path = r.get("file", "").removeprefix("qmd://")
        path = path[len("subliminal/"):] if path.startswith("subliminal/") else path
        head = path.split("/", 1)[0]
        if head not in allowed:
            continue
        prev = best.get(path)
        if prev is None or float(r.get("score", 0) or 0) > float(prev.get("score", 0) or 0):
            best[path] = {**r, "file": f"qmd://{path}"}
    return sorted(best.values(), key=lambda r: -float(r.get("score", 0) or 0))


def _qmd_daemon_search(query: str, limit: int, collections: list,
                      skip_rerank: bool = not RECALL_QMD_RERANK,
                      legs: tuple[str, ...] = ("lex", "vec"),
                      lex_query: Optional[str] = None,
                      exact_pool: Optional[int] = None,
                      candidate_limit: Optional[int] = None,
                      floor: Optional[dict] = None,
                      lex_mode: Optional[str] = None) -> list:
    """Send a lex and/or vec query to the qmd daemon.

    **Returns a list, or raises `QmdUnavailable`.** An empty list means the
    daemon answered and nothing matched; a daemon that timed out, refused the
    connection, or answered with a status is an exception, because "zero
    documents" and "no answer" are different findings and the caller is the only
    one who can say which one it got (#407).

    DEFAULT: rerank on (see RECALL_QMD_RERANK for the measurement). The
    request carries an explicit `rerank` boolean — that is the key every
    qmd version reads; `skipRerank` was a client-side name that published
    2.8.3 ignored, so passing skip_rerank=True never actually skipped
    anything until the fork. Pass skip_rerank=True only inside a latency
    budget, as prefetch does.

    `legs` selects which search legs run. Measured 2026-09-07 on the fork,
    twelve collections in one request: ~90 ms without the reranker, ~700 ms
    with it (whole chunks, QMD_RERANK_PARALLELISM=4). Published 2.8.3 took
    ~2 s either way because sqlite-vec scanned every vector once per
    collection. The prefetch path runs `("lex",)` inside its latency budget
    and the full hybrid as a straggler whose result carries over to the
    next turn.

    **An unrestricted search names `VAULT_SEGMENTS`; a restricted one names its
    subset.** Both now say what they are willing to keep, and
    `_qmd_normalize_global` still folds the unrestricted reply back onto the
    segment view (strip `subliminal/`, drop non-vault heads, dedupe keeping the
    best score), so what changed is which documents are *reachable*, not which are
    allowed.

    Empty used to buy speed. qmd falls back to an exact cosine scan —
    `WHERE hash_seq IN (400 placeholders)` per batch — for any collection under
    `COLLECTION_VEC_EXACT_SCAN_MAX` (20,000 chunks), and every segment here is
    under it, so an empty list (which the REST handler resolves to `undefined`)
    took sqlite-vec's native `MATCH` path: one ANN scan instead of eleven exact
    ones. That is still true and still costs, and the price stated here used to be
    a CACHED-REPEAT price — figures taken by re-sending one query the daemon had
    already embedded, which is the measurement this docstring's own `skip_rerank`
    paragraph warns against ("the daemon caches query embeddings, so a sequential
    A/B measures arm order, not arm"). Re-priced 2026-09-18 by two rounds (3 and 5
    samples per cell) on never-seen query text per sample against the live daemon,
    rerank on — every figure below says which sample it is:
    2,161-2,291 ms FRESH named at a 40-row pool, 3,672-4,027 ms FRESH named at
    240, the identical repeat of a query just run at 240 in 119-183 ms CACHED, and
    150-210 ms FRESH rerank-OFF at 240, which is the fetch leg alone. So the
    widening #504 bought is roughly 1.5-1.7 s of extra cross-encoder per named
    recall (and the rerank over the whole width is 3.5-3.9 s of the ~3.7 s a named
    240-row recall costs), and it is
    governed by `LATENCY_BUDGET_MS` in `workers/sources/automod_regression.py`
    (nightly ceiling 4,800 ms) rather than by a sample in this comment: the
    nightly eval averaged 707.8 ms FRESH on 2026-09-13, before the widening, and
    4,368.4 ms FRESH on 2026-09-16 after it — both artifacts in
    `eval/baselines/` (#1129).

    What it bought in recall turned out to be the thing it was breaking. The
    global reply is one top-k whose width qmd derives from `limit`, and
    `subliminal` re-indexes the whole vault, so for the eval query
    "which autonomy tasks maintain the knowledge graph?" a 40-slot global ask came
    back with 36 rows of which `_qmd_normalize_global` kept 21 — and not one of the
    five autonomy task files the query expects was among them, while their
    neighbour `autonomy/39-` was. Named, at the pool `RECALL_DOC_POOL` asks for, all
    five arrive (ranks 14, 17, 35, 39, 175). That is #504: the doc leg's only
    remaining zero, and the mechanism behind it is #1005.

    The fairness property is the one the restricted path has always relied on.
    qmd's `collections` array is a per-collection quota (`ftsLimit = limit * 10`,
    plus an exact vec scan per collection), so naming the segments means `autonomy`
    — 242 files, 1% of the corpus — is competing for autonomy's slots instead of
    against the whole index. The original write-up recorded that hazard as the cost
    ("a global top-k can starve a small collection... two of five small-collection
    probes lost their top hit", qmd #791/#803); this change pays the *other* side of
    that trade, and #504 is the probe for whether paying it is net-positive across
    all 20 queries. The 826ms -> 462ms and 4.2s -> 269ms pairs in the history below
    were measured on other builds of the other arm; they are kept as history, not as
    the justification, and the re-measurement is a finding on #504.

    A **scope-restricted** search is untouched: its own subset, `[:limit]` back
    with no fold, and its pool equal to its answer — so `prefetch`'s nine-collection
    request on the turn-critical path does not inherit the widened ask.

    History, as recorded: the switch to empty was made against published 2.8.3,
    where it read as 4.2s -> 269ms; the fork closed most of that gap on its own, and
    a re-measurement on 2026-09-09 (fresh query per sample, rotated arm order — the
    daemon caches query embeddings, so a sequential A/B measures arm order, not arm)
    found the margin survived at 1.8x rather than 5x.

    `lex_query`, when given, replaces the lex leg's text. The lex leg is
    FTS5 with implicit AND (every term must match), so it wants a short,
    high-signal term list, while the vec leg benefits from the full
    focus-enriched sentence. Without this both legs got the same string and
    the hybrid's lex component returned nothing on enriched queries.
    """
    query = _qmd_sanitize(query)
    if not query:
        return []
    # Apply the same stopword strip to BOTH legs. Conversational framing
    # ("tell me about X", "show me Y") drifts the vec embedding away from
    # content. Identical inputs are fine — lex (BM25) and vec (embedding)
    # do fundamentally different matching, so they still produce
    # complementary signal.
    stripped = _qmd_strip_stopwords(query)
    lex_q = stripped
    if lex_query:
        lex_q = _qmd_strip_stopwords(_qmd_sanitize(lex_query)) or stripped

    # Unrestricted == "every segment we know about", and it now says so: naming
    # the segments puts the scan on qmd's per-collection path, where each one is
    # fetched for its own quota before fusion. Anything narrower is a deliberate
    # scope and keeps its own shape exactly as it had it.
    allowed = set(VAULT_SEGMENTS)
    unrestricted = allowed.issubset(set(collections or []))
    # `limit` sizes the answer; `pool` sizes what it is ranked out of, because a
    # document that is not in the reply is not in the answer at any rank. The
    # restricted path keeps pool == limit — it gets `[:limit]` back with no fold,
    # so a wider ask there buys nothing and prefetch is on a latency budget.
    pool = min(limit * QMD_POOL_FACTOR, QMD_POOL_MAX) if unrestricted else limit
    # The recall doc leg names its pool outright (`recall_doc_pool`): under global
    # fusion the number was measured, and the factor would turn its 40 into 120.
    if exact_pool is not None and unrestricted:
        pool = max(1, min(int(exact_pool), QMD_POOL_MAX))
    payload = {
        "searches": [{"type": leg, "query": lex_q if leg == "lex" else stripped}
                     for leg in legs],
        "limit": pool,
        # qmd applies `candidateLimit` as `fused.slice(0, candidateLimit)` BEFORE
        # its cross-encoder rerank, and its default is RERANK_CANDIDATE_LIMIT = 40.
        # So `limit` alone does not widen what gets ranked: a 240-row ask that
        # leaves this at 40 has its other 200 rows returned but never scored, which
        # is the same miss with a bigger payload (recorded: 0 of 5 expected files
        # at pool 40, all 5 at 240). Always explicit, like `rerank` below.
        # A caller may fuse a smaller head than it asks back: the djev ranker
        # (#1336) takes the fused head plus the floors' rows, all returned.
        "candidateLimit": candidate_limit if (candidate_limit and unrestricted) else pool,
        "collections": list(VAULT_SEGMENTS) if unrestricted else collections,
        # Always explicit. Omitting it means "the daemon's default", which
        # is rerank-on today and is not something this client should lean on.
        # The stash this came from made the key conditional; that undoes a
        # deliberate decision from 91a59f9 and is not part of the global-scan
        # change.
        "rerank": not skip_rerank,
    }
    # Only where several collections are named: one collection has nothing to
    # fuse across, and the prefetch lex leg is restricted and budgeted.
    if RECALL_QMD_FUSION == "global" and len(payload["collections"] or []) > 1:
        payload["fusion"] = "global"
        floor_map = RECALL_COLLECTION_FLOOR if floor is None else floor
        floor_sent = {c: n for c, n in floor_map.items() if c in payload["collections"]}
        if floor_sent:
            payload["collectionFloor"] = floor_sent
    if lex_mode and lex_mode != "and" and "lex" in legs:
        payload["lexMode"] = lex_mode

    def _finish(rows: list) -> list:
        if not unrestricted:
            return rows
        # Slice to the pool, not to `limit`. Trimming the folded reply back to the
        # size of the answer is where a deep ask goes to die: qmd has already
        # ranked it, the caller still only ever sees `limit` documents, and every
        # file between rank `limit` and the pool — `autonomy/67-` at recorded rank
        # 175 among them — is unreachable again. `_vault_recall`'s doc leg keeps
        # its own `[:RECALL_DOC_POOL]`; that is where the answer gets cut.
        return _qmd_normalize_global(rows, allowed)[:pool]

    try:
        return _finish(_qmd_post(payload))
    except urllib.error.HTTPError as e:
        # Rerank context OOM manifests as HTTP 500. One-shot retry without
        # rerank so callers see documents instead of silent zero-hits.
        if e.code == 500 and payload["rerank"]:
            try:
                payload["rerank"] = False
                _qmd_log("rerank failed (HTTP 500) — retrying with rerank off")
                # `_finish` on the retry too: a global scan that falls back to
                # rerank-off still returns whole-index rows that have to be
                # folded onto the segment view before a caller sees them.
                return _finish(_qmd_post(payload))
            except Exception as e2:
                _qmd_log(f"rerank-off retry also failed: {e2!r}")
                raise QmdUnavailable(
                    "qmd retrieval failed: the daemon 500'd with rerank on and "
                    f"the rerank-off retry also failed ({e2!r}) — these are "
                    "not zero matches"
                ) from e2
        _qmd_log(f"HTTPError {e.code}: {e.reason}")
        raise QmdUnavailable(
            f"qmd retrieval failed: the daemon returned HTTP {e.code} "
            f"({e.reason}) — these are not zero matches"
        ) from e
    except (OSError, urllib.error.URLError) as e:
        # The two daemon shapes, both of which used to buy one stderr line and a
        # `None` that each caller then defaulted to an empty list. TimeoutError:
        # up and not answering — the 23 `search failed: TimeoutError('timed
        # out')` lines in logs/mcp.err. URLError/ConnectionRefusedError: not
        # running at all.
        # `URLError` is itself an `OSError`, so this is one family; the name is
        # spelled twice to say that. The log line stays because it is the only
        # per-failure record the daemon path has, and the raise is what makes it
        # reach anyone.
        _qmd_log(f"search failed: {e!r}")
        raise QmdUnavailable(
            f"qmd retrieval failed: the daemon did not answer ({e!r}) — these "
            "are not zero matches"
        ) from e
    except Exception:
        # Anything else is this client's own bug (a bad fold in `_finish`, a
        # malformed reply, a typo), not an outage. It propagates as itself: the
        # tool handler still turns it into an error reply, so it is no longer
        # silent, but it must not be filed under "the daemon did not answer" —
        # that report would send the next reader to the daemon.
        _qmd_log("unexpected error in the qmd search path (not a daemon outage)")
        raise


def _consolidate_results(query: str, results: list) -> Optional[dict]:
    if len(results) < CONSOLIDATION_MIN_RESULTS:
        return None
    url, model_name = _consolidation_endpoint()
    if not url:
        return None
    parts = [f"Query: {query}\n\nSearch Results:\n"]
    for i, r in enumerate(results, 1):
        parts.append(f"--- Result {i} (score: {r.get('score', 'N/A')}) ---")
        parts.append(f"File: {r.get('citation', r.get('path', ''))}")
        snippet = r.get("snippet", "")[:2000]
        parts.append(f"Content:\n{snippet}\n")
    payload = json.dumps({
        "model": model_name,
        "messages": [{"role": "system", "content": CONSOLIDATION_SYSTEM_PROMPT}, {"role": "user", "content": "\n".join(parts)}],
        "temperature": 0.0, "max_tokens": 1000,
    }).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=CONSOLIDATION_TIMEOUT) as resp:
            data = json.loads(resp.read())
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        if not content:
            return None
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
        return json.loads(content)
    except Exception:
        return None


def _graph_rerank(
    documents: list[dict],
    seed_entities: list[str],
    weighted_neighbors: list,
    alpha: float = 0.5,
) -> list[dict]:
    """Re-rank documents using fact-graph topological voting (TGS-RAG Phase 3).

    For each doc, extract entities mentioned in its title+snippet. Each
    seed/neighbor entity contributes a vote weighted by its graph weight
    (seeds=1.0, neighbors=their graph_weighted_neighbors weight) and
    divided by log(1+degree) so god-nodes (e.g. "lloyd" with ~991 edges)
    don't dominate. Final score = alpha * QMD_score + (1-alpha) * normalized_topo.

    Returns a new list of dicts sorted by combined score, with `_qmd_score`
    and `_topo_score` annotations preserved for inspection/eval.
    """
    if not documents or (not seed_entities and not weighted_neighbors):
        return documents

    voters: dict[str, float] = {}
    for s in seed_entities or []:
        if s:
            voters[s.lower()] = 1.0
    for ent, w in (weighted_neighbors or []):
        key = (ent or "").lower()
        if key and key not in voters:
            voters[key] = float(w)

    if not voters:
        return documents

    # Precompute per-voter score contribution (degree penalty baked in)
    # and a word-boundary regex. This collapses what used to be a full
    # `extract_entities_from_query(doc_text)` call per doc — which
    # iterates ~2,700 entity dirs — into a tight regex scan over ~15
    # voter terms. Drops rerank latency by ~10x.
    # Lowercased, because `voters` is keyed on the lowercased name. Against
    # the cased map every lookup missed, so `degree` read 1 for every voter
    # and the god-node penalty divided them all by the same constant — a
    # no-op dressed as a penalty (2026-09-03 review).
    edge_counts = _edge_counts_or_empty(ci=True)
    voter_contribs: dict[str, float] = {}
    voter_patterns: dict[str, re.Pattern] = {}
    for vk, vw in voters.items():
        if len(vk) < 2:
            continue
        degree = max(edge_counts.get(vk, 1), 1)
        voter_contribs[vk] = vw / math.log(1 + degree + math.e)
        voter_patterns[vk] = re.compile(r"(?<!\w)" + re.escape(vk) + r"(?!\w)", re.IGNORECASE)

    def _topo(text: str) -> float:
        if not text:
            return 0.0
        score = 0.0
        for vk, pat in voter_patterns.items():
            if pat.search(text):
                score += voter_contribs[vk]
        return score

    raw = []
    for d in documents:
        text = " ".join([
            str(d.get("path") or ""),
            str(d.get("title") or ""),
            str(d.get("snippet") or ""),
        ])
        raw.append(_topo(text))

    max_topo = max(raw) if raw else 0.0
    rescored = []
    for d, t in zip(documents, raw):
        norm = (t / max_topo) if max_topo > 0 else 0.0
        qmd = float(d.get("score", 0) or 0)
        combined = alpha * qmd + (1 - alpha) * norm
        rescored.append({
            **d,
            "score": round(combined, 6),
            "_qmd_score": round(qmd, 6),
            "_topo_score": round(norm, 6),
        })
    rescored.sort(key=lambda x: x["score"], reverse=True)
    return rescored


def _run_vault_search(query: str, max_results: int, min_score: float, scope: str, consolidate: bool) -> dict:
    scope_prefixes = []
    if scope:
        for item in scope.split(","):
            item = item.strip()
            if item:
                scope_prefixes.append(item.rstrip("/") + "/")

    coll_list = VAULT_SEGMENTS
    if scope:
        scope_segs = [s.strip().rstrip("/") for s in scope.split(",") if s.strip()]
        coll_list = [s for s in scope_segs if s in VAULT_SEGMENTS] or VAULT_SEGMENTS

    # Run QMD search across the requested collections AND the source-code
    # grep fallback in parallel (lever 3) when not scope-restricted.
    def _do_qmd():
        # One request, whatever the collection count. This used to fan out one
        # request per collection through a 4-worker pool, which was strictly
        # worse than not doing it: the qmd daemon is a single node process that
        # serves requests serially — 8 concurrent requests take exactly 8x one
        # request, measured — so the pool only queued them. Same query, same
        # answer: 9571ms fanned out against 3971ms as one call. The threads
        # bought nothing and paid twelve times for the per-collection scan that
        # `_qmd_daemon_search` now avoids entirely.
        #
        # No empty-list default here: a daemon that did not answer raises
        # `QmdUnavailable`; it is not a search that found nothing. The raise
        # crosses the pool on `qmd_fut.result()` below and comes back as an
        # error reply in `_vault_search`. The grep leg's hits are deliberately
        # not handed back as if they were vault results during an outage.
        #
        # What #504's wider pool costs THIS caller, paired on the live daemon
        # (medians of 5 alternating runs, 2026-09-14): `max_results=10` asked 20
        # rows in 16 ms before, asks 30 in 19 ms now — +3 ms, because the pool is
        # a filter on one search, not a fan-out (12 named collections at 240 rows
        # costs 322 ms against 43 ms for 40 globally; a 12x scan would be ~516).
        # `_lookup_entity_facts` pays +2 ms on each of its six `limit=2` calls.
        # The only ask that moves materially is `_vault_recall`'s doc leg, 43 ms
        # -> 322 ms, and that is the ask this change exists to widen.
        return _qmd_daemon_search(query, max_results, coll_list)

    def _do_grep():
        # Skip grep when caller restricted scope — they want vault-only results.
        if scope_prefixes:
            return []
        return _grep_lloyd_code(query, limit=max(max_results, 8))

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        qmd_fut = pool.submit(_do_qmd)
        grep_fut = pool.submit(_do_grep)
        all_raw = qmd_fut.result()
        grep_results = grep_fut.result()

    if grep_results:
        existing = {r.get("file", "") for r in all_raw}
        for gr in grep_results:
            if gr.get("file") not in existing:
                all_raw.append(gr)

    # Lever 1: demote memory daily logs so they don't dominate canonical sources.
    for r in all_raw:
        path_for_demote = r.get("file", "").removeprefix("qmd://").removeprefix("obsidian/")
        if _DAILY_LOG_RE.search(path_for_demote):
            r["_pre_demote_score"] = r.get("score", 0)
            r["score"] = round(float(r.get("score", 0)) * DAILY_LOG_DEMOTE_FACTOR, 6)
    all_raw.sort(key=lambda x: float(x.get("score", 0)), reverse=True)

    parsed = []
    for r in all_raw:
        file_val = r.get("file", "")
        path = file_val.removeprefix("qmd://")
        if path.startswith("obsidian/"):
            path = path.removeprefix("obsidian/")
        score = float(r.get("score", 0))
        if score < min_score:
            continue
        if scope_prefixes and not any(path.startswith(p) for p in scope_prefixes):
            continue
        snippet = re.sub(r"@@[^@]*@@\s*(?:\([^)]*\)\s*)?", "", r.get("snippet", "")).strip()
        snippet = re.sub(r"^\d+:\s*", "", snippet, flags=re.MULTILINE).strip()
        parsed.append({"path": path, "score": round(score, 4), "snippet": snippet[:300], "citation": path})

    trimmed = parsed[:max_results]

    consolidated = None
    if consolidate and len(trimmed) >= CONSOLIDATION_MIN_RESULTS:
        consolidated = _consolidate_results(query, trimmed)

    return {
        "results": trimmed, "mode": "hybrid",
        "consolidated": consolidated is not None,
        "consolidated_summary": consolidated,
        "collections_searched": coll_list,
    }


# ── Vault helpers ────────────────────────────────────────────────────────────

def _resolve_case_insensitive(rel_path: str) -> Optional[Path]:
    current = VAULT
    for segment in Path(rel_path).parts:
        exact = current / segment
        if exact.exists():
            current = exact
            continue
        try:
            matches = [e for e in current.iterdir() if e.name.lower() == segment.lower()]
        except OSError:
            return None
        if not matches:
            return None
        current = matches[0]
    return current if current.is_file() else None


def _normalize_vault_path(path: str) -> tuple[str | None, str | None]:
    """Reduce a caller-supplied path to a vault-relative POSIX path.

    Callers routinely pass the documented ``~/obsidian/...`` or absolute
    ``/home/.../obsidian/...`` forms. ``Path.__truediv__`` does NOT expand
    ``~`` (only ``expanduser()`` does), so ``VAULT / "~/obsidian/x"`` silently
    creates a literal ``~`` directory *inside* the vault — the escape guard
    passes because the path is still under VAULT. That produced the recurring
    stray ``~/obsidian/`` tree (2026-07 incident). Strip the known vault
    prefixes and reject anything that points outside the vault.

    Returns ``(relative_path, None)`` on success or ``(None, error_message)``.
    """
    raw = path.strip()
    if not raw:
        return None, "path is required"
    vault_str = str(VAULT)
    p = raw
    if p.startswith("~/obsidian/"):
        p = p[len("~/obsidian/"):]
    elif p in ("~/obsidian", "~"):
        p = ""
    elif p == vault_str or p.startswith(vault_str + "/"):
        p = p[len(vault_str):]
    elif p.startswith("/") or p.startswith("~"):
        # Absolute path outside the vault, or a ~/<other-root> home path
        # (e.g. ~/lloyd/...). These are never valid vault targets.
        return None, f"path must be vault-relative or under {vault_str}, got {raw!r}"
    p = p.lstrip("/")
    if not p:
        return None, "path resolves to the vault root, not a file"
    parts = Path(p).parts
    if "~" in parts or ".." in parts:
        return None, f"invalid path segment in {raw!r}"
    return p, None


def _sha256_bytes(data: bytes) -> str:
    """The digest the change ledger records, called through the ledger itself.

    One hashing rule across the two records matters: the audit row's
    ``pre_sha256`` is only a join key if it is the same number the ledger index
    carries for the same bytes.
    """
    try:
        from agent_mcp import _change_ledger
        return _change_ledger.sha256_bytes(data)
    except Exception:  # noqa: BLE001 — hashlib is the same function, inlined
        return hashlib.sha256(data).hexdigest()


def _ledger_scope() -> "tuple[str, str] | None":
    """The (session, turn) this write belongs to, or None for no ledger.

    Delegates to the lookup ``Write``/``Edit`` use rather than restating it: one
    rule decides which calls are attributable, so this route cannot quietly end
    up outside the ledger by keeping its own copy of the predicate. Returns None
    for a call with no bound session or no turn id — a worker turn, a direct
    in-process caller — which is the degrade path, not an error.
    """
    try:
        from agent_mcp.builtin_fs import _ledger_scope as _fs_ledger_scope
        return _fs_ledger_scope(get_bound_session())
    except Exception:  # noqa: BLE001
        logger.warning("change ledger: scope lookup failed for a vault write",
                       exc_info=True)
        return None


def _audit_keys(scope: "tuple[str, str] | None") -> "tuple[str, str]":
    """The `(session, turn)` to put in the audit row, known halves only.

    The ledger needs both or nothing (`_ledger_scope` returns None), but the log
    should still say which session wrote the file. A turn-less call — a worker turn,
    whose session's ambient turn id is empty — is the common shape, and an empty
    session means an in-process caller, since an aggregator call that omitted its
    session id is refused before it gets here (#1053).
    """
    if scope:
        return scope
    return (get_bound_session(), "")


def _read_pre_bytes(target: Path) -> "bytes | None":
    """The bytes about to be replaced, or None for a create / an unreadable file.

    Read inside ``commit_lock`` on purpose: a pre-image is only an undo point if
    it is what this write actually displaced, and the lock is what makes "nothing
    else wrote between the read and the replace" true. Unreadable degrades to a
    write with no pre-image rather than to a failed write.
    """
    try:
        return target.read_bytes() if target.is_file() else None
    except OSError:
        return None


def _ledger_record(scope: "tuple[str, str] | None", *, target: Path, path: str,
                   pre_bytes: "bytes | None", post_sha: str) -> None:
    """Open this turn's entry for `target`, snapshot the pre-image, commit it.

    Same three ledger calls `Write`/`Edit` make (`begin`, `snapshot_pre`,
    `commit`), so a `vault_write` lands in the one index a turn's `Write`s
    land in and reverts through the one revert that reads it.

    Called after the bytes are on disk — `pre_bytes` was read under the writer
    lock, so the snapshot carries the content this write displaced while the
    file-lock section stays read-plus-write. Every failure is logged and
    swallowed: a write that fails because the undo bookkeeping failed is
    strictly worse than a write with no undo.
    """
    if scope is None:
        return
    try:
        from agent_mcp import _change_ledger, _task_registry
        session_id = get_bound_session()
        entry = _change_ledger.begin(
            scope, real=os.path.realpath(target), path=path,
            op="write" if pre_bytes is not None else "create",
            call_id=_task_registry.current_call_id.get(),
            # Set only when a subagent made the change, so a footer can say
            # which of a turn's writes came from a Task rather than the turn.
            via_session=session_id if session_id.startswith("task:") else "",
        )
        _change_ledger.snapshot_pre(scope, entry, pre_bytes)
        _change_ledger.commit(scope, entry, post_sha)
    except Exception:
        logger.warning("change ledger: record failed for %s", path, exc_info=True)


def _audit_write(path: str, byte_count: int, *, session: str = "", turn: str = "",
                 pre_sha256: str = "", post_sha256: str = "") -> None:
    """Append the write's row to the vault audit log.

    `session`/`turn`/`pre_sha256`/`post_sha256` are the join keys. Without them
    the row was the only record that a whole-file overwrite happened at all and
    named neither the turn that wrote it nor whether anything is restorable, so
    the log that sees the write could not reach the ledger that holds the
    pre-image — given "a note changed at 09:04:12Z" there was no way to answer
    either question. Empty strings mean the write was unattributable (no bound
    session/turn); the hashes are about the bytes and are recorded regardless.
    """
    try:
        AUDIT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        entry = {"timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(), "agent_id": "lloyd", "path": path, "bytes": byte_count, "action": "write",
                 "session": session, "turn": turn,
                 "pre_sha256": pre_sha256, "post_sha256": post_sha256}
        with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


# ── Tool handlers ────────────────────────────────────────────────────────────

def _vault_read(params: dict) -> dict:
    path, norm_err = _normalize_vault_path(params.get("path", ""))
    if norm_err:
        return _err(norm_err, ErrorCode.MISSING_PARAM if "required" in norm_err else ErrorCode.PATH_ESCAPE)
    try:
        target = VAULT / path
        if not target.resolve().is_relative_to(VAULT.resolve()):
            return _err("path escapes vault root", ErrorCode.PATH_ESCAPE)
        if not target.exists():
            resolved = _resolve_case_insensitive(path)
            if resolved is None:
                return _err(f"File not found: {path}", ErrorCode.NOT_FOUND)
            target = resolved
        text = target.read_text(encoding="utf-8", errors="replace")
        start_line = int(params.get("start_line", 0))
        num_lines = int(params.get("num_lines", 0))
        if start_line > 0 or num_lines > 0:
            lines = text.splitlines()
            start = max(0, start_line - 1)
            end = (start + num_lines) if num_lines > 0 else len(lines)
            text = "\n".join(lines[start:end])
        return {"path": path, "text": text or "(empty file)"}
    except Exception as exc:
        return _err(str(exc), ErrorCode.INTERNAL)


def _guard_knowledge_type(path: str, content: str) -> tuple[str, dict | None, str | None]:
    """Normalise a ``knowledge/`` note's ``type`` on write, or say why not.

    Returns ``(content_to_write, error_result, rewritten_from)``. Only one of
    the last two is ever non-``None``: an error means the write must not happen,
    a ``rewritten_from`` names the retired spelling the content carried before
    it was replaced by its canonical value. Runs BEFORE the mkdir/write, so the
    refused path creates no file and no directory.

    This is the writer-side half of the #370 vocabulary (#872). The tree was
    consolidated onto ``okf_taxonomy.CANONICAL_TYPES`` while every writer stayed
    on the pre-consolidation literals and no write path looked at ``type:`` at
    all — so a nightly job invented ``type: note``, the file landed, and every
    automod promotion on this box was blocked until someone found the word
    (#780). ``vault_write`` is the path the research skills are told to use, so
    it is where the vocabulary is enforced: an alias is rewritten to its
    canonical value, an invented value is refused by name.
    """
    if not path.startswith("knowledge/") or not path.endswith(".md"):
        return content, None, None
    try:
        from scripts.vault import okf_taxonomy
    except Exception as exc:  # noqa: BLE001
        # A guard whose input cannot be read reports a verdict it cannot
        # justify — four confirmed instances in lloyd/MEMORY.md. Failing open
        # here would mean an unimportable vocabulary silently re-admits any
        # value, which is the exact state this function exists to end.
        return content, _err(
            f"cannot validate knowledge/ `type`: scripts.vault.okf_taxonomy is "
            f"unimportable ({exc})", ErrorCode.INTERNAL), None
    try:
        rewritten, replaced = okf_taxonomy.normalize_document_type(content)
    except okf_taxonomy.KnowledgeTypeError as exc:
        return content, _err(str(exc), ErrorCode.INVALID_PARAM,
                             invalid_type=exc.value, path=path), None
    except Exception as exc:  # noqa: BLE001
        return content, _err(f"cannot validate knowledge/ `type`: {exc}",
                             ErrorCode.INTERNAL), None
    return rewritten, None, replaced


def _vault_write(params: dict) -> dict:
    path, norm_err = _normalize_vault_path(params.get("path", ""))
    if norm_err:
        return _err(norm_err, ErrorCode.MISSING_PARAM if "required" in norm_err else ErrorCode.PATH_ESCAPE)
    content = params.get("content", "")
    try:
        content, type_err, replaced = _guard_knowledge_type(path, content)
        if type_err is not None:
            return type_err
        target = VAULT / path
        if not target.resolve().is_relative_to(VAULT.resolve()):
            return _err("path escapes vault root", ErrorCode.PATH_ESCAPE)
        # The third lane onto the same two files (`memory_add`, Write/Edit, this),
        # so it carries the same ceiling — a guard on one lane only is a guard on
        # whichever lane the writer did not choose. Refusal, not commit: it returns
        # before the lock, the ledger and the mkdir, like the escape check above it.
        ceiling_msg = memory_write_error(target, content)
        if ceiling_msg:
            return _err(ceiling_msg, ErrorCode.INVALID_PARAM)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Same lock key the memory tools and Write/Edit take: `vault_write` with
        # path `lloyd/MEMORY.md` is the same state as `memory_add`, spelled a
        # third way, and a whole-file overwrite here used to drop whatever the
        # other lane had appended. The lock is off-tree (`lock_file_for`) — a
        # sibling `<name>.lock` here would be a new file in the vault, and
        # `lloyd/.research-queue.lock` is that precedent, tracked in git.
        scope = _ledger_scope()
        try:
            with commit_lock(target):
                pre_bytes = _read_pre_bytes(target)
                write_text_durable(target, content)
        except TimeoutError as exc:
            return _err(str(exc), ErrorCode.LOCK_TIMEOUT, path=path)
        byte_count = len(content.encode("utf-8"))
        post_sha = _sha256_bytes(content.encode("utf-8"))
        pre_sha = _sha256_bytes(pre_bytes) if pre_bytes is not None else ""
        _ledger_record(scope, target=target, path=path, pre_bytes=pre_bytes,
                       post_sha=post_sha)
        audit_session, audit_turn = _audit_keys(scope)
        _audit_write(path, byte_count, session=audit_session, turn=audit_turn,
                     pre_sha256=pre_sha, post_sha256=post_sha)
        result = {"success": True, "path": path, "bytes": byte_count}
        if replaced:
            # Say so: a writer that asked for one spelling and got another needs
            # to see it, or it keeps asking for the retired one.
            from scripts.vault import okf_taxonomy
            result["type_normalized"] = {
                "from": replaced,
                "to": okf_taxonomy.normalize_type(replaced),
            }
        return result
    except Exception as exc:
        return _err(str(exc), ErrorCode.INTERNAL)


def _vault_overview(params: dict) -> dict:
    try:
        if not VAULT.exists():
            return _err(f"Vault not found: {VAULT}", ErrorCode.NOT_FOUND)
        totals: dict = {}
        grand_total = 0
        for segment in VAULT_SEGMENTS:
            seg_dir = VAULT / segment
            if not seg_dir.is_dir():
                totals[segment] = 0
                continue
            count = sum(1 for f in seg_dir.rglob("*.md") if f.name not in VAULT_EXCLUDE_FILES and not any(p in VAULT_EXCLUDE_DIRS for p in f.parts))
            totals[segment] = count
            grand_total += count
        return {"total_files": grand_total, "segments": totals}
    except Exception as exc:
        return _err(str(exc), ErrorCode.INTERNAL)


def _vault_search(params: dict) -> dict:
    query = params.get("query", "").strip()
    if not query:
        return _err("query is required", ErrorCode.MISSING_PARAM, results=[])
    try:
        return _run_vault_search(query, int(params.get("max_results", 10)), float(params.get("min_score", 0.0)), params.get("scope", ""), params.get("consolidate", True))
    except Exception as exc:
        return _err(str(exc), ErrorCode.INTERNAL, results=[])


def _djev_doc_text(doc: dict) -> str:
    """One pool row as the decision engine sees it: the title, then the
    snippet qmd already returned. No disk read — the point of ranking a
    shortlist is that everything it needs is already in hand."""
    title = str(doc.get("title") or doc.get("path") or "")
    snippet = str(doc.get("snippet") or "")
    return f"{title}\n{snippet}" if snippet else title


def shadow_seams_dark_by_dispatch() -> dict[str, str]:
    """Shadow seams this module's dispatch cannot reach, and what lifts it.

    A row count can report that a seam is quiet; it cannot report that a seam is
    unreachable, because a dead hook and a day with nothing to record are the
    same stream of nothing. #1372 is exactly that pair: `rerank` stopped
    recording on 2026-09-21 because #1336 put djev's ranking in the arm above the
    hook, and the only surfaces that read the recorder reported three live seams.
    So the module that owns the dispatch answers the reachability question, and
    `djev_status` relays it.

    Reads `recall_reranker()` rather than the constant, so pulling the kill
    switch lifts the verdict as immediately as it lifts the hook.
    """
    if recall_reranker() == "djev":
        return {"rerank": (
            "unreachable: `ranker == \"djev\"` is the first arm of the "
            "`_vault_recall` dispatch and returns before the shadow hook's `elif`. "
            "Pull the kill switch (RECALL_RERANKER=\"qmd\") with the engine "
            "answering and the seam records again.")}
    return {}


def _recall_knob_report(asked: set[str], ranker: str, arm_on: bool) -> dict[str, str]:
    """Which knobs the CALLER set this call did not act on, and why (#1372).

    Empty for the defaults: production sets no knob per call, so the ordinary
    recall gets no explanation it did not ask for. Non-empty whenever a key
    arrived in `params` and the dispatch could not use it — which, since #1336,
    is every value of either arm knob while djev is the ranker.
    """
    if not asked:
        return {}
    if ranker == "djev":
        return {k: RECALL_ARM_UNUSED_NOTE for k in sorted(asked)}
    if "djev_rerank_top" in asked and not arm_on:
        # The switch itself took effect — leaving it off is what sent the recall
        # down the cross-encoder and the shadow hook — but the width beside it
        # is read only inside the arm, so it was never looked at.
        return {"djev_rerank_top": "no effect: the djev rerank arm was off, "
                                   "so the width was never read"}
    return {}


def _djev_rerank_pool(documents: list[dict], query: str, top: int) -> list[dict]:
    """Reorder the HEAD of the pool through djev; the tail keeps its order.

    The arm, as distinct from the ranker. Reached only when qmd is the recall's
    ranker: the dispatch calls it under `elif djev_rerank:`, and djev-ranking is
    the arm above it that returns first (#1336). djev does not reorder a pool it
    already ordered, so a caller that sets the knob while djev ranks is told so
    in its own result — `RECALL_UNUSED_KNOBS_KEY`.

    Only the head, because djev is a final-stage reranker over a shortlist and
    nothing else: above 32 questions the server splits the canvas into
    separate shared contexts whose scores are not comparable, and `label_mass`
    is already falling at 32. Fanning a 240-row pool across chunks and sorting
    the union produces an artefact that looks exactly like a ranking.

    Fail-open at every exit. `None` from the client means the engine did not
    answer — or answered across a canvas split, which it refuses to sort — and
    the pool keeps the order qmd gave it.
    """
    try:
        from app import djev
        n = min(int(top or 0), len(documents), djev.RANK_MAX_N)
        if n < 2:
            return documents
        head, tail = documents[:n], documents[n:]
        rows = djev.rank(query, [_djev_doc_text(d) for d in head],
                         seam="recall_arm")
        if rows is None:
            return documents
        return [head[r["index"]] for r in rows] + tail
    except Exception as e:  # noqa: BLE001 — an advisory reranker never fails a recall
        logger.debug("djev rerank arm: %s", e)
        return documents


def _djev_rank_recall(documents: list[dict], query: str) -> list[dict] | None:
    """djev orders the recall's whole pool (#1336); `None` when it did not answer.

    The pool is at most `RECALL_DJEV_POOL` rows, which one canvas holds, so
    nothing is cut. Each document's fusion score is kept as `_fusion_score` and
    `score` becomes djev's, so a consumer that re-sorts by score keeps djev's
    order rather than undoing it.
    """
    try:
        from app import djev
        head, tail = documents[:RECALL_DJEV_POOL], documents[RECALL_DJEV_POOL:]
        if len(head) < 2:
            return documents
        rows = djev.rank(query, [_djev_doc_text(d) for d in head],
                         seam="recall_rank", timeout=RECALL_DJEV_TIMEOUT_S,
                         chars=RECALL_DJEV_CHARS, samples=RECALL_DJEV_SAMPLES,
                         max_n=RECALL_DJEV_POOL)
        if not rows:
            return None
        ordered = []
        for r in rows:
            d = head[r["index"]]
            d["_fusion_score"] = d.get("score", 0)
            d["score"] = round(float(r["score"]), 6)
            ordered.append(d)
        return ordered + tail
    except Exception as e:  # noqa: BLE001 — a ranker failure is a fallback, not a failed recall
        logger.warning("djev recall ranking failed: %s", e)
        return None


def _djev_shadow_rerank(documents: list[dict], query: str) -> None:
    """Record what djev would have ordered, beside what production returns.

    Off-thread and drop-tolerant: one `put_nowait` on a bounded queue that drops
    rather than waits. NOT reached while djev is the recall's ranker — the
    dispatch calls this only under `elif reranker is None:` and the arm above it
    returns first for every `ranker == "djev"` call, so while
    `RECALL_RERANKER = "djev"` (production since #1336) this seam is
    structurally dark, and `tests/test_djev_rerank_arm.py` pins that by running
    the recall and counting recorder calls rather than by looking for this call
    in the source. The name "rerank seam" is historical: production stopped
    reranking here on 2026-09-21, so the seam now answers a rollback question —
    what djev's order would have been on the pool qmd's cross-encoder is
    ordering — and fires only with the kill switch pulled to "qmd" and the
    engine still answering.

    The state and question set are built by a lambda the shadow WORKER runs —
    the recall path holds the texts already and must not spend even a string
    concatenation on an observation nobody is waiting for.
    """
    try:
        from app import djev, djev_shadow
        if not djev_shadow.enabled("rerank"):
            return
        head = documents[:djev.RANK_DEFAULT_N]
        if len(head) < 2:
            return
        texts = [_djev_doc_text(d) for d in head]
        djev_shadow.shadow(
            seam="rerank",
            state=lambda: djev.rank_state(query, texts),
            questions=lambda: djev.rank_questions(texts),
            # Production's ordering of the same slice — the thing djev's is
            # being recorded beside.
            actual=[d.get("path") for d in head],
            meta={"query": query[:300], "pool_size": len(documents),
                  "scores": [round(float(d.get("score", 0) or 0), 4) for d in head]},
        )
    except Exception as e:  # noqa: BLE001 — a recorder never reaches its caller
        logger.debug("djev rerank shadow: %s", e)


def _vault_recall(params: dict, *, seed_top_k: int | None = None,
                  reranker: str | None = None) -> dict:
    """Combined recall over documents, entity facts and graph neighbours.

    `params` is what the `vault_recall` tool receives — `call_tool` hands this
    handler the client's argument dict raw — so every retrieval knob read out of it is agent-settable.
    `seed_top_k` is keyword-only precisely so it is NOT one of them: it is absent
    from the tool schema, a stray `{"seed_top_k": 3}` arriving over the wire
    changes nothing, and the only caller that sets it is `eval/run_eval.py`, in
    process, which is the one component whose job is measuring a seeding width
    (#843 review). `None` means "production's width", resolved against
    `RECALL_SEED_TOP_K` HERE rather than as a signature default, so a def-time
    binding cannot freeze a number the constant later moved off.
    """
    query = params.get("query", "").strip()
    if not query:
        return _err("query is required", ErrorCode.MISSING_PARAM, documents=[], facts=[])
    limit = int(params.get("limit", 20))
    include_facts = params.get("include_facts", True)
    expand_graph = bool(params.get("expand_graph", RECALL_EXPAND_GRAPH))
    # graph_rerank default-on as of 2026-05-12 — the perf optimization
    # (regex over voters instead of full entity scan) drops latency from
    # ~2s extra to near-zero, while the MRR lift (+6-13%) is consistent.
    # Alpha defaults to 0.3 (graph-heavy) per the May 11 alpha sweep.
    graph_rerank = bool(params.get("graph_rerank", RECALL_GRAPH_RERANK))
    rerank_alpha = float(params.get("rerank_alpha", RECALL_RERANK_ALPHA))
    demote_daily_logs = bool(params.get("demote_daily_logs", RECALL_DEMOTE_DAILY_LOGS))
    demote_factor = float(params.get("demote_factor", DAILY_LOG_DEMOTE_FACTOR))
    # Graph expansion breadth/depth. Both were hardcoded (top_k=5, hops=1)
    # until 2026-08-06 (#380): with only 5 neighbour slots, low-weight edge
    # types can never surface — a `mentions` edge scores 0.3 × 0.8 = 0.24
    # against typed edges at 0.4–0.6, so density changes were invisible to
    # retrieval and to the eval. hops=1 also meant the "multi-hop" eval
    # category was in fact being served by single-hop expansion.
    # Defaults preserve the historic behaviour exactly.
    graph_top_k = int(params.get("graph_top_k", RECALL_GRAPH_TOP_K))
    graph_hops = int(params.get("graph_hops", RECALL_GRAPH_HOPS))
    # Resolved at call time from the constants, like every knob above, so a
    # test that moves `RECALL_DJEV_RERANK` moves what this call does.
    #
    # `asked` is the set the CALLER sent, not the set that took effect — the
    # defaults are production's own and are not the caller's to be told about
    # (#1372). Both keys are read here, so both are reported when they are
    # inert; `djev_rerank_top` is read only inside the arm, which is why
    # `djev_rerank: false` under qmd still reports the width as unused.
    _asked = set(params) & {"djev_rerank", "djev_rerank_top"}
    djev_rerank = bool(params.get("djev_rerank", RECALL_DJEV_RERANK))
    djev_rerank_top = int(params.get("djev_rerank_top", RECALL_DJEV_RERANK_TOP))
    # Who orders the document pool (#1336). Keyword-only and never read from
    # `params`: a stray key must not choose the retriever. Only this function's
    # own fallback passes it.
    ranker = reranker or recall_reranker()
    doc_shape = recall_doc_leg_shape(ranker)

    # Seed width. The number and why it is 10 rather than 5 are at
    # RECALL_SEED_TOP_K; the reason it had to stop being a literal here is that
    # eval/run_eval.py restated it as `[:5]` and scored entity metrics on the
    # smaller set (#843). Unlike the eight knobs above, this one is NOT read out
    # of `params`, so no client key can move it — see the docstring. Resolution
    # happens here rather than in the signature so the constant is consulted at
    # call time, which is what lets a test move the constant and watch the width
    # move with it.
    seed_entities = [
        e for e, _ in
        extract_entities_from_query(query)[
            :(RECALL_SEED_TOP_K if seed_top_k is None else int(seed_top_k))]]

    # If graph_rerank is requested, we need neighbors regardless of expand_graph,
    # because rerank uses them as voters. Force graph expansion in that case.
    need_neighbors = expand_graph or graph_rerank
    weighted_neighbors: list[tuple[str, float]] = []
    if need_neighbors and seed_entities:
        weighted_neighbors = graph_weighted_neighbors(
            seed_entities, top_k=graph_top_k, hops=graph_hops
        )

    def _do_search():
        # Daemon is the only search path. A CLI subprocess fallback was removed
        # 2026-04-20: on daemon failure it hits the same broken state, then eats
        # 30s before returning empty. The orphaned function that comment
        # outlived — still defined, still zero call sites, still making this
        # read as a two-tier system — was deleted by #407 rather than rewired,
        # because the 2026-04-20 reasoning still holds.
        #
        # No empty-list default here: an outage propagates to `_vault_recall`
        # and comes back as an error, which is the whole point. A genuine
        # zero-hit answer is an empty list and still merges with the grep and
        # graph legs as usual.
        #
        # Ask for the pool, not for `limit` (#504). The pool is not decoration: for
        # the eval query "which autonomy tasks maintain the knowledge graph?", the
        # live daemon returned 1 of its 5 expected files from a 60-row pool, 3 from
        # 120, and all 5 from 240 — a document that never enters the reply is missing
        # at every rank, so no re-ranking downstream could have found it.
        # `_qmd_daemon_search` slices its folded reply at the same `pool`, so nothing
        # here re-cuts it; the list gets smaller further down only at the `[:limit]`
        # that hands the answer over.
        return _qmd_daemon_search(query, doc_shape["limit"], VAULT_SEGMENTS,
                                  skip_rerank=not doc_shape["rerank"],
                                  exact_pool=doc_shape["limit"],
                                  candidate_limit=doc_shape["candidateLimit"],
                                  floor=doc_shape["floor"],
                                  lex_mode=doc_shape.get("lexMode"))

    def _do_code_grep():
        if not params.get("grep_code", True):
            return []
        return _grep_lloyd_code(query, limit=8)

    def _do_graph_lookup():
        """Phase 2 (reframed): for each top seed + graph neighbor, look
        up its canonical file via a focused QMD search of the entity name.
        Surfaces autonomy/N-*.md, skills/<slug>/SKILL.md, backlog files
        that lexically match the entity but don't share the original
        query's tokens.

        Eval verdict (2026-05-12): on average HURTS single-entity queries
        by injecting alternatives that displace the perfect QMD top-1
        match. HELPS the 'hard' cross-domain category. Net regression
        when default-on (-12% MRR overall). Default-off, opt-in via
        `graph_lookup: True`."""
        if not params.get("graph_lookup", False):
            return []
        # Use top 4 seeds + top 4 neighbors. Cap entity name length to skip
        # noisy compound names like "autonomy tasks API" that just retrieve
        # the original query's results again.
        entity_pool = []
        for e in (seed_entities or [])[:4]:
            if e and 3 <= len(e) <= 60 and e not in entity_pool:
                entity_pool.append(e)
        if need_neighbors:
            for e, _w in (weighted_neighbors or [])[:4]:
                if e and 3 <= len(e) <= 60 and e not in entity_pool:
                    entity_pool.append(e)
        if not entity_pool:
            return []

        def _lookup_one(ent):
            # The one caller still allowed to degrade instead of fail, and the
            # reason is narrow: `graph_lookup` is opt-in enrichment (default off,
            # -12% MRR when on per the 2026-05-12 eval), so a transient failure
            # on one entity must not throw away a recall whose main search
            # already succeeded. What changed with #407 is the shape of the
            # swallow: `except Exception` also ate real bugs, and defaulting the
            # None made a daemon failure identical to "this entity has no file".
            # Now only `QmdUnavailable` degrades, and it is logged — and on a
            # real outage `_do_search` raises for the same daemon a few
            # milliseconds later, so the caller still gets an error, not a
            # quietly-shortened answer.
            try:
                hits = _qmd_daemon_search(ent, 2, VAULT_SEGMENTS)
            except QmdUnavailable as e:
                _qmd_log(f"graph_lookup: skipping entity {ent!r}: {e}")
                hits = []
            return [(ent, h) for h in hits[:2]]

        # Parallel per-entity lookup. Cap workers to avoid swamping QMD.
        canonical: dict[str, dict] = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(len(entity_pool), 6)
        ) as pool:
            for pairs in pool.map(_lookup_one, entity_pool):
                for ent, h in pairs:
                    f = h.get("file", "")
                    if not f or f in canonical:
                        continue
                    rel_path = f.removeprefix("qmd://").removeprefix("obsidian/")
                    is_canonical = any(rel_path.startswith(p) for p in _CANONICAL_PREFIXES)
                    qmd_score = float(h.get("score", 0) or 0)
                    if is_canonical:
                        # Promote canonical files into top-N but don't
                        # overtake QMD rank-1 (1.0). Cap at 0.85 so a
                        # legitimate query→canonical lexical match (1.0)
                        # still wins, but graph-derived canonicals beat
                        # demoted memory logs (0.4) and QMD rank-2 (0.5).
                        boosted = min(0.85, qmd_score + 0.35)
                    else:
                        # Weak promote: don't outrank ANY strong QMD hit.
                        boosted = min(0.5, qmd_score * 0.6)
                    canonical[f] = {
                        **h,
                        "score": round(boosted, 4),
                        "_via_graph_entity": ent,
                        "_qmd_score_at_lookup": qmd_score,
                        "_canonical_boost": is_canonical,
                    }
        return list(canonical.values())

    def _do_facts():
        # Returns (facts, graph_facts, fact_reads). `fact_reads` is the
        # accounting #1250 added: how many of the per-entity fact reads this
        # call attempted produced no facts because the READ failed, plus the
        # first failure's repr. Before this the loop's only exit on a bad read
        # was `continue`, so a leg that read nothing at all — every read
        # raising, or the read path answering every entity with an error —
        # returned `[]` and was indistinguishable from a leg that read
        # everything and matched nothing. The eval then scored that 0.0, the
        # paired promotion check compared 0.0 against a healthy arm's 0.375,
        # and an unrelated commit became a rollback reason. "Read nothing" is a
        # different answer from "matched nothing", and to be one it has to be
        # carried out of here.
        fact_reads: dict = {"n_failed": 0, "first_error": None}
        if not include_facts:
            return [], [], fact_reads
        # Query-aware fact ranking (Fix A, #322).
        qtoks = fact_query_tokens(query)

        def _note_failed_read(repr_text: str) -> None:
            fact_reads["n_failed"] += 1
            if fact_reads["first_error"] is None:
                fact_reads["first_error"] = repr_text

        def _collect(entity_names: list[str], godnode_threshold: int) -> list[dict]:
            """Pull facts from entities, applying god-node guardrail (Fix C).
            Tags each fact with its source `entity` so downstream consumers
            (re-ranking, eval, UI) can attribute facts back to a node.

            A read that failed is counted, never discarded. Two shapes count
            (#1250): a raised exception, repr'd `<Type>: <message>`, and a
            `get_facts_sync` reply that carries an `error` key with no facts
            (`{"error": "Entity not found: X", "facts": []}`,
            `agent_mcp/retrieval.py:169-171`), repr'd `error: <message>`. The
            second is the one the old `if not ef: continue` swallowed as
            though entity resolution had simply found nothing to return.
            """
            out: list[dict] = []
            for ent in entity_names:
                try:
                    entity_data = get_facts_sync(ent)
                except Exception as exc:
                    _note_failed_read(f"{type(exc).__name__}: {exc}")
                    continue
                if not isinstance(entity_data, dict):
                    _note_failed_read(f"{type(entity_data).__name__}: fact read "
                                      f"returned {type(entity_data).__name__}, not a mapping")
                    continue
                ef = entity_data.get("facts") or []
                if not ef:
                    if entity_data.get("error"):
                        _note_failed_read(f"error: {entity_data['error']}")
                    continue
                if len(ef) > godnode_threshold and qtoks:
                    kept = [f for f in ef if fact_matches_tokens(f, qtoks)]
                    if not kept:
                        continue
                    ef = kept
                resolved_entity = entity_data.get("entity") or ent
                out.extend({**f, "entity": resolved_entity} for f in ef)
            return out

        def _rank(candidates: list[dict], cap: int) -> list[dict]:
            if not candidates:
                return []
            scored = [
                (fact_score(f, qtoks), float(f.get("confidence", 0.5)), idx, f)
                for idx, f in enumerate(candidates)
            ]
            scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
            return [f for _s, _c, _i, f in scored[:cap]]

        seed_pool = _collect(seed_entities, FACT_GODNODE_THRESHOLD)
        facts = _rank(seed_pool, FACT_RANK_CAP_SEED)

        graph_facts: list[dict] = []
        if expand_graph and weighted_neighbors:
            neighbor_names = [e for e, _w in weighted_neighbors[:graph_top_k]]
            graph_pool = _collect(neighbor_names, FACT_GODNODE_THRESHOLD)
            graph_facts = _rank(graph_pool, FACT_RANK_CAP_GRAPH)

        # The accounting leaves this function with the facts, or it is worth
        # nothing: the failures happened inside `_collect`, two loops up from the
        # caller that has to report them.
        return facts, graph_facts, fact_reads

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            search_fut = pool.submit(_do_search)
            facts_fut = pool.submit(_do_facts)
            grep_fut = pool.submit(_do_code_grep)
            graph_lookup_fut = pool.submit(_do_graph_lookup)
            raw_results = search_fut.result()
            facts, graph_facts, fact_reads = facts_fut.result()
            grep_results = grep_fut.result()
            graph_lookup_results = graph_lookup_fut.result()
        # Merge alternate-retrieval sources: dedupe by file path; first
        # wins (QMD primary, then grep, then graph-lookup).
        existing_files = {r.get("file", "") for r in raw_results}
        for gr in grep_results:
            if gr.get("file") not in existing_files:
                raw_results.append(gr)
                existing_files.add(gr.get("file", ""))
        for gl in graph_lookup_results:
            if gl.get("file") not in existing_files:
                raw_results.append(gl)
                existing_files.add(gl.get("file", ""))
        documents = []
        # The pre-rank pool is `RECALL_DOC_POOL` wide, not `limit * 3` wide. Both
        # legs of this function now agree on that number: `_do_search` asks the
        # daemon for it, and this is where it would quietly be thrown away again —
        # `limit * 3` is 60 for the eval's 20, and the recorded ranks of the five
        # files `kg-maintenance-tasks` is scored against are 14, 17, 35, 39 and 175,
        # so a 3x slice loses one of them on the way to the ranker even after the
        # daemon handed it back. It stays a floor on `limit` so a caller asking for
        # more than the pool still gets the list it asked for.
        #
        # `pool_size` was conditional on "we re-sort, so over-fetch". That is now
        # unconditional: `demote_daily_logs` re-sorts the pool by *path shape* alone,
        # so its slice is not a quality order and narrowing it is a quality decision
        # made by a path regex.
        pool_size = max(doc_shape["limit"], limit)
        prerank_pool = raw_results[:pool_size]
        for r in prerank_pool:
            path = r.get("file", "").removeprefix("qmd://")
            if path.startswith("obsidian/"):
                path = path.removeprefix("obsidian/")
            documents.append({
                "path": path,
                "title": r.get("title", ""),
                "snippet": r.get("snippet", ""),
                "score": r.get("score", 0),
            })
        if demote_daily_logs:
            for d in documents:
                if _DAILY_LOG_RE.search(d.get("path") or ""):
                    d["_pre_demote_score"] = d.get("score", 0)
                    d["score"] = round(float(d.get("score", 0)) * demote_factor, 6)
            documents.sort(key=lambda x: float(x.get("score", 0) or 0), reverse=True)
        # ── djev: the eval arm, then the shadow seam ─────────────────────
        # `documents` holds qmd's reranked pool in the order production
        # returns it: the demote has sorted it and the slice has not run. Both
        # of the things below belong at exactly this point.
        #
        # The shadow hook's first draft sat inside `_graph_rerank`, which is
        # only reached under `if graph_rerank:` while `RECALL_GRAPH_RERANK` is
        # False — so it would have fired zero times in production and read as
        # a quiet seam rather than as a hook on dead code.
        #
        # They are exclusive because with the arm ON djev IS the decision, and
        # a shadow row comparing djev's ordering against djev's ordering is
        # not an observation. The eval mutes the recorder anyway
        # (`LLOYD_DJEV_SHADOW=0`); this keeps that true for anyone who flips
        # the constant by hand.
        #
        # With djev AS the ranker (#1336) neither runs: the pool arrives in
        # fusion order and djev orders it. A djev that does not answer sends the
        # whole recall down the cross-encoder path instead of serving fusion
        # order, which measured 0.05 MRR worse.
        #
        # That last paragraph is also why #1372 happened. "Neither runs" was
        # true of the code and false of every description of it — the schema
        # still advertised `djev_rerank`, the recorder's docstring still called
        # `rerank` a seam production passes through, and the file's own
        # reachability check was a source-text string that stayed green — so the
        # seam went dark with nothing saying so. Three things close that: a
        # caller who set either arm knob is told in the result
        # (`RECALL_UNUSED_KNOBS_KEY`), `shadow_seams_dark_by_dispatch()` says
        # which seam cannot fire and what would lift it, and
        # `tests/test_djev_rerank_arm.py` runs the recall and counts recorder
        # calls instead of reading for the call.
        if ranker == "djev":
            ranked = _djev_rank_recall(documents, query)
            if ranked is None:
                from app import qmd_health
                qmd_health.note_ranker(False, "djev did not answer")
                return _vault_recall(params, seed_top_k=seed_top_k, reranker="qmd")
            from app import qmd_health
            qmd_health.note_ranker(True)
            documents = ranked
        elif djev_rerank:
            documents = _djev_rerank_pool(documents, query, djev_rerank_top)
        elif reranker is None:
            # Not inside the fallback: that path exists because djev just failed
            # to answer, and a shadow row would queue another read at it.
            _djev_shadow_rerank(documents, query)
        if graph_rerank:
            documents = _graph_rerank(documents, seed_entities, weighted_neighbors, alpha=rerank_alpha)[:limit]
        else:
            documents = documents[:limit]
        result = {"documents": documents, "facts": facts, "query": query}
        # A knob the caller set that this call could not honour is named here,
        # not logged (#1372). Absent unless the caller actually sent one, so the
        # default recall — which is every recall production serves — pays
        # nothing for an explanation nobody asked for.
        _unused = _recall_knob_report(_asked, ranker, djev_rerank)
        if _unused:
            result[RECALL_UNUSED_KNOBS_KEY] = _unused
        # Fact-leg provenance (#1250). Always present, including when it is 0:
        # a consumer that has to distinguish "the fact leg read nothing" from
        # "the fact leg read everything and matched nothing" cannot do it from
        # an absent key, which would read the same as a version of this
        # function that never counted. `fact_read_first_error` is the first
        # failure's repr — `Type: message` for a raise, `error: <message>` for
        # a read path that answered with an error instead of facts — and is
        # None only when nothing failed. One repr, not a list: the point is to
        # name WHAT raised, and the count says how widespread it was.
        result["n_fact_reads_failed"] = int(fact_reads["n_failed"])
        result["fact_read_first_error"] = fact_reads["first_error"]
        if graph_facts:
            result["graph_expanded_facts"] = graph_facts
        if weighted_neighbors:
            result["graph_neighbors_used"] = [
                {"entity": e, "weight": round(w, 3)} for e, w in weighted_neighbors
            ]
        # NOTE: datetime event_date values from YAML-parsed fact frontmatter
        # are non-JSON-native. _wrap() in the dispatcher uses default=str to
        # serialize them. (Pre-existing data issue in the facts tree,
        # surfaced by #322 returning more facts.)
        return result
    except Exception as exc:
        return _err(str(exc), ErrorCode.INTERNAL, documents=[], facts=[])


# ── MCP registration ─────────────────────────────────────────────────────────

async def list_tools():
    return [
        Tool(name="vault_read", description="Use when you know the vault path; to find a note by topic, use vault_search first.\n\nRead a file from the obsidian vault. Path is vault-relative (e.g. 'memory/learnings/DAILY_NOTES.md'); a leading '~/obsidian/' is stripped automatically.", inputSchema={
            "type": "object", "properties": {"path": {"type": "string", "description": "Vault-relative path, e.g. 'knowledge/agents/foo.md'. Do not prefix with '~/' or an absolute home path."}, "start_line": {"type": "integer", "description": "1-based first line to return; omit to read from the top"}, "num_lines": {"type": "integer", "description": "How many lines to return from start_line; omit to read to the end"}}, "required": ["path"]}),
        Tool(name="vault_write", description="Use to create a vault file or overwrite one wholesale; to amend an existing note, vault_read it first.\n\nWrite content to a vault file. Audit-logged. Path is vault-relative (e.g. 'memory/learnings/DAILY_NOTES.md'); a leading '~/obsidian/' is stripped automatically.", inputSchema={
            "type": "object", "properties": {"path": {"type": "string", "description": "Vault-relative path, e.g. 'knowledge/agents/foo.md'. Do not prefix with '~/' or an absolute home path."}, "content": {"type": "string", "description": "Full file contents. This replaces the file wholesale — read it first if you mean to amend rather than overwrite."}}, "required": ["path", "content"]}),
        Tool(name="vault_overview", description="Summarize the obsidian vault: file counts per top-level segment, or the most-linked notes. Use this to orient before searching when you do not know what the vault holds.", inputSchema={
            "type": "object", "properties": {"detail": {"type": "string", "enum": ["summary", "hubs"], "description": "summary (default) counts files per segment; hubs lists the most-linked notes"}}}),
        Tool(name="vault_search", description="Use to find vault notes by topic when you do not know the path; then vault_read the winner.\n\nSearch the obsidian vault, combining BM25 keyword matching with vector similarity. Returns ranked excerpts with their vault paths; use vault_read to pull a full file.", inputSchema={
            "type": "object", "properties": {"query": {"type": "string", "description": "Natural-language or keyword query"}, "max_results": {"type": "integer", "description": "Maximum excerpts to return (default 10)"}, "min_score": {"type": "number", "description": "Drop results scoring below this threshold"}, "scope": {"type": "string", "description": "Restrict to a vault segment, e.g. 'knowledge' or 'memory/learnings'"}, "consolidate": {"type": "boolean", "description": "Summarize the hits into one synthesized answer instead of returning raw excerpts"}}, "required": ["query"]}),
        Tool(name="vault_recall", description="Use when a question spans documents and entity facts; for prose alone use vault_search instead.\n\nCombined recall: vault search and the facts of the entities the query names, in parallel.", inputSchema={
            "type": "object", "properties": {
                "query": {"type": "string", "description": "Natural-language query; entities mentioned in it are resolved and their facts returned alongside documents"},
                "limit": {"type": "integer", "description": "Documents returned (default 20)"},
                "include_facts": {"type": "boolean", "description": "Include entity facts (default true)"},
                "demote_daily_logs": {"type": "boolean", "description": f"Down-weight daily notes (default {RECALL_DEMOTE_DAILY_LOGS})"},
            }, "required": ["query"]}),
    ]


async def call_tool(name: str, arguments: dict):
    handlers = {
        "vault_read": _vault_read, "vault_write": _vault_write, "vault_overview": _vault_overview,
        "vault_search": _vault_search, "vault_recall": _vault_recall,
    }
    handler = handlers.get(name)
    stripped: list[str] = []
    if name == "vault_recall":
        # The eval knobs are not the client's to set; see RECALL_EVAL_KNOBS.
        raw = arguments or {}
        stripped = sorted(k for k in raw if k in RECALL_EVAL_KNOBS)
        arguments = {k: v for k, v in raw.items() if k not in RECALL_EVAL_KNOBS}
    if handler:
        # Handlers are sync and do subprocess/urllib I/O (QMD search, rg,
        # consolidation LLM call) with multi-second timeouts — run them in a
        # worker thread so the shared event loop never stalls.
        result = await asyncio.to_thread(handler, arguments)
        if stripped and isinstance(result, dict) and "error" not in result:
            # The strip is the right answer about retrieval and was still
            # silence as an answer to the client (#1372): a caller that sent
            # `djev_rerank: true` over the wire got a normal recall and no word
            # that its knob never reached the handler. Name it in the body. Not
            # onto an error result — an error body is about the failure.
            result = {**result, RECALL_UNUSED_KNOBS_KEY: {
                k: (f"{RECALL_KNOB_STRIPPED_NOTE}; {RECALL_ARM_UNUSED_NOTE}"
                    if k in _RECALL_ARM_KNOBS and recall_reranker() == "djev"
                    else RECALL_KNOB_STRIPPED_NOTE) for k in stripped}}
        return _wrap(result)
    return _wrap(_err(f"Unknown tool: {name}", ErrorCode.UNKNOWN_TOOL))
