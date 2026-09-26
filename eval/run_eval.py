#!/usr/bin/env python3
"""Vault retrieval eval runner.

Loads queries from `vault_recall_queries.yaml`, calls `_vault_recall` for
each, scores entity_hit / doc_hit / topk_overlap against expected entities
and docs, and writes a baseline JSON record to `baselines/`.

Usage:
    .venvs/lloyd/bin/python eval/run_eval.py
    .venvs/lloyd/bin/python eval/run_eval.py --label phase3-A --notes "graph-vote enabled, alpha=0.5"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
LLOYD_HOME = HERE.parent
sys.path.insert(0, str(LLOYD_HOME))

# The djev shadow recorder is MUTED for every eval run, and this line must
# stay above the `agent_mcp.vault` import: the lead shadow seam lives inside
# `_vault_recall`, so an eval that scored 20 queries against the pinned corpus
# would write 20 rows indistinguishable from production traffic — into the
# very `label_mass` distribution those rows' floors are supposed to be derived
# from. `setdefault`, so a caller who set it deliberately keeps their value.
os.environ.setdefault("LLOYD_DJEV_SHADOW", "0")

from agent_mcp.vault import (
    LLOYD_CODE_ROOT,
    QMD_DAEMON_URL,
    RECALL_DEMOTE_DAILY_LOGS,
    RECALL_DJEV_RERANK,
    RECALL_DJEV_RERANK_TOP,
    RECALL_EXPAND_GRAPH,
    RECALL_GRAPH_HOPS,
    RECALL_GRAPH_RERANK,
    RECALL_GRAPH_TOP_K,
    RECALL_RERANK_ALPHA,
    RECALL_SEED_TOP_K,
    _vault_recall,
)
from agent_mcp.facts import _extract_entities_from_query
from agent_mcp.retrieval import (
    recall_seeds as _recall_seeds, semantic_seed_k as _semantic_seed_k,
    semantic_seeding_record as _semantic_seeding_record,
)
from app.kg_store import StoreUnavailable, store
from app.paths import EVAL_BASELINES_DIR, VAULT_FACTS_ROOT, VAULT_KG_DB
# The absolute latency ceiling for THIS run's context, read from the one module
# that owns it. Before #1129 the runner wrote `latency_ms_avg` into every
# artifact and nothing anywhere read it, which is how a 708 ms → 4,230 ms step
# landed silently. Reading it here, at the moment the number is produced, is what
# makes the nightly budget a live check instead of a constant with a test.
from workers.sources.automod_regression import (CONTEXT_NIGHTLY, fact_leg_empty,
                                                fact_read_coverage, over_budget)
# This file runs both as `python eval/run_eval.py` (script dir on sys.path) and
# as `import eval.run_eval` from the tests; the second form needs the package
# spelling.
try:
    from eval import counterfactual as cf
except ImportError:  # pragma: no cover - script-dir invocation
    import counterfactual as cf

# The interval behind every number this file writes or prints (#696). Owned by
# its own stdlib module rather than restated here, for the reason stated at the
# top of `eval/stats.py`: the trend audit, this writer and the backtest all need
# the same arithmetic, and two copies of a statistic is how two sides come to
# disagree about whether a number moved. Same dual spelling as `counterfactual`:
# this file runs as a script AND as `import eval.run_eval` from the tests.
try:
    from eval import stats as evstats
except ImportError:  # pragma: no cover - script-dir invocation
    import stats as evstats

# The holdout leg's reserve rule (#1412). Same dual spelling as the two above.
try:
    from eval import retrieval_holdout as holdout
except ImportError:  # pragma: no cover - script-dir invocation
    import retrieval_holdout as holdout

# The DOCUMENT half of the corpus this run scores (#1374). Owned by its own
# stdlib module because `scripts/eval_trend_stats.py` has to read the same key
# with the same meaning: the writer and the reader disagreeing about what an
# absent key means is exactly the defect this is filed under.
from app import doc_corpus


def _corpus_provenance() -> dict:
    """What was actually scored, not just how it was scored.

    Every knob this run used is already recorded; the data underneath it was
    not. That gap is invisible in the headline numbers: with graph re-ranking
    off the graph never reorders documents, and the document leg queries the
    qmd daemon over the network, so an entirely empty fact tree and store
    still produce identical mrr_doc / ndcg10 / doc_hit_rate and `errors: 0`.
    Only the three entity metrics collapse. Recording the resolved paths (they
    honour LLOYD_FACTS_ROOT / LLOYD_KG_DB) plus the store's own counts is what
    lets a later reader tell the two runs apart.

    Raises StoreUnavailable if the store cannot be opened — that is a
    different failure from "opened and empty" and the caller must not
    collapse the two.

    Under ``doc`` sits the OTHER half of what this run scored (#1374). Every
    ``doc_hit`` / ``ndcg10`` / ``mrr_doc`` in the artifact came from the qmd
    daemon and from grepping a checkout, and neither was ever recorded: the
    recall reaches qmd over HTTP at whatever that daemon holds at that minute,
    and it re-embeds continuously (38 768 → 39 210 vectors inside one daemon
    process on 2026-09-22, no restart, no artifact that says so). So the eight
    fact keys below could tell a reader which KG produced a number and never
    which document corpus did. ``None`` for the block means the probe could not
    answer, and every reader downstream is required to call that *unknown* —
    never 0, never "identical".
    """
    facts_root = Path(VAULT_FACTS_ROOT)
    entity_dirs = 0
    if facts_root.is_dir():
        entity_dirs = sum(1 for child in facts_root.iterdir() if child.is_dir())
    stats = store().stats()
    return {
        "facts_root": str(facts_root),
        "kg_db": str(VAULT_KG_DB),
        "entity_dirs": entity_dirs,
        "edges_total": int(stats.get("edges_total", 0)),
        "edges_active": int(stats.get("edges_active", 0)),
        "aliases": int(stats.get("aliases", 0)),
        "entities": int(stats.get("entities", 0)),
        "facts": int(stats.get("facts", 0)),
        # The document half, probed at the moment the run scores it. `None` is a
        # value, not an omission: it is how "the daemon would not answer" reaches
        # a reader seven days later.
        doc_corpus.DOC_KEY: doc_corpus.collect(
            QMD_DAEMON_URL, code_root=LLOYD_CODE_ROOT),
    }


def _corpus_line(corpus: dict) -> str:
    doc = corpus.get(doc_corpus.DOC_KEY)
    # The doc half is printed beside the fact half because the line is the only
    # place a human sees both in one glance; `vectors=unknown` has to be
    # readable as unknown right there, in the same words the artifact uses.
    doc_part = ("vectors=unknown" if not isinstance(doc, dict) else
                f"vectors={doc['vectors']} mode={doc['corpus_mode']} "
                f"code_root={doc['code_root']}")
    return (f"[info] corpus facts_root={corpus['facts_root']} "
            f"entity_dirs={corpus['entity_dirs']} kg_db={corpus['kg_db']} "
            f"entities={corpus['entities']} edges_active={corpus['edges_active']} "
            f"aliases={corpus['aliases']} facts={corpus['facts']} "
            f"doc[{doc_part}]")


# The fact-leg guard is IMPORTED, never restated here (#1250).
# `workers/sources/automod_regression.py` owns it, because that side has to ask
# the same question of a BASELINE artifact written by unknown code while it
# compares two arms; this side asks it of the run it just produced. Two copies
# of a guard is how the two sides come to disagree about whether an arm measured
# anything at all, across a subprocess boundary no test can see from one side —
# which is the defect class this item is filed under.
def fact_leg_read_nothing(records: list[dict], corpus: dict) -> bool:
    """Whether this run's fact leg read NOTHING while the fact tree is not empty.

    `fact_leg_empty` over this run's records and its own corpus block. The half
    `corpus_ok` cannot see: `corpus_ok` is
    `bool(corpus["edges_active"]) and bool(corpus["entities"])`, the graph half
    only, and `corpus["facts"]` comes from the store index — what the index
    holds, not what recall could read. So an arm whose every fact read failed
    (the `except Exception: continue` in `agent_mcp/vault.py:_collect` discarded
    every one of them, with no log and no counter, until #1250) passed
    `corpus_ok` and was recorded with a real-looking 0.0.
    """
    return fact_leg_empty(fact_read_coverage(records),
                          int((corpus or {}).get("facts") or 0))


#: The legs a scored entity can come from, weakest-last. `seed` is NOT retrieval:
#: `seeds_extracted` is what the QUERY named (`recall_seeds()`), so an entity
#: carried only by that leg is the scorer being handed part of its own answer —
#: #1548, measured on `nightly-20260926`, where every one of the 43 `entity_hit`
#: queries matched an entity the seed leg held, and the field that used to
#: present "entities retrieval returned" led with those same seeds.
#: The other three are what retrieval returned: the entity a returned fact was
#: fetched for, the entity a graph-expanded fact was fetched for, and the graph
#: neighbours the walk used.
ENTITY_LEG_ORDER = ("seed", "fact", "graph_expanded_fact", "graph_neighbor")
#: The retrieval half of that tuple — an entity in one of these was returned.
RETRIEVAL_LEGS = ENTITY_LEG_ORDER[1:]


def _entity_legs(result: dict,
                 seeds: list[str] | None = None) -> dict[str, list[str]]:
    """The run's entity names grouped by the LEG each came from, in signal order.

    Deduped WITHIN a leg only (case-insensitively, original case preserved, same
    rule as `_entities_in_result`), and an entity that came from two legs appears
    in BOTH lists. That duplication is the point: the union below dedupes ACROSS
    legs, so in the union a seed hides the fact that also carried the entity, and
    nothing in the artifact could say which leg satisfied a matched entity. Keys
    are `ENTITY_LEG_ORDER`; every key is always present, possibly empty.
    """
    raw = {
        "seed": [str(s or "").strip() for s in (seeds or [])],
        "fact": [str(f.get("entity", "") or "").strip()
                 for f in (result.get("facts", []) or [])],
        "graph_expanded_fact": [str(f.get("entity", "") or "").strip()
                                for f in (result.get("graph_expanded_facts", []) or [])],
        "graph_neighbor": [str(n.get("entity", "") or "").strip()
                           if isinstance(n, dict) else str(n or "").strip()
                           for n in (result.get("graph_neighbors_used", []) or [])],
    }
    out: dict[str, list[str]] = {}
    for leg in ENTITY_LEG_ORDER:
        seen: set[str] = set()
        out[leg] = [e for e in raw[leg] if e and not (e.lower() in seen or seen.add(e.lower()))]
    return out


def _retrieval_entities(result: dict) -> list[str]:
    """Entities retrieval carried — fact, graph-expanded fact, graph neighbour.

    The seed leg excluded, so this is what `result_summary.fact_entities_top10`
    is supposed to mean and did not: that field used to be the first ten of
    `_entities_in_result(result, seeds)`, which is the seed leg FIRST, so on the
    2026-09-26 baseline its first seven entries were byte-identical to
    `seeds_extracted`, and an audit that intersected matched entities with it
    (as this item's own proving command did) was reading the query's own extracted
    entities as retrieval output. Order and dedupe rule as `_entity_legs`, minus
    the seed leg.
    """
    legs = _entity_legs(result)
    seen: set[str] = set()
    return [e for leg in RETRIEVAL_LEGS for e in legs[leg]
            if not (e.lower() in seen or seen.add(e.lower()))]


def _entities_in_result(result: dict, seeds: list[str] | None = None) -> list[str]:
    """Union of entity signals: seeds extracted from query, fact entities,
    graph_expanded facts entities, and graph_neighbors_used. Order = signal
    strength (seeds first).

    This is the list `entity_hit` is graded against, which is exactly the defect
    #1548 files: a gold entity the QUERY named scores as a hit with no fact,
    expanded fact or neighbour carrying it. It stays the graded list — the
    retrieval-carried rate (#1548 clause 2) is a SECOND aggregate beside it, not a
    rename of this one, because renaming the trended number would silently re-base
    every published floor. Ask `_entity_legs` which leg carried a hit, and
    `_retrieval_entities` what retrieval returned.
    """
    seen: set[str] = set()
    return [e for leg in ENTITY_LEG_ORDER for e in _entity_legs(result, seeds)[leg]
            if not (e.lower() in seen or seen.add(e.lower()))]


def _entity_leg_attribution(legs: dict[str, list[str]],
                            expected_raw: list[str]) -> dict[str, dict]:
    """For each gold entity `entity_hit` counted: WHICH LEG satisfied it (#1548 clause 1).

    `{gold_label: {"legs": [..], "retrieval_satisfied": bool}}`, keyed by the same
    `_norm`'d gold label `entities_matched` reports. Satisfaction is decided the way
    `entity_hit` decides it — `_entity_pair_satisfied` over the gold's spelling and
    its store canonical, against the same `_norm`'d pairs — but PER LEG, so a gold
    entity a returned fact carries says `["fact"]` and one the query's own extracted
    seed carries says `["seed"]`. `_entity_legs` keeps the legs apart for precisely
    this; the union in `_entities_in_result` cannot answer the question, because
    seeds come FIRST there and its case-insensitive dedupe then drops the
    fact-carried copy of any entity a seed also named — which is the whole of #1548.

    A gold entity that is not a hit is not a key: there is nothing to attribute. A
    key with `"legs": []` is possible and honest — a match can come from a leg's
    canonical while its raw name matches nothing — and `retrieval_satisfied` is then
    false, which is right: nothing in the result carried it.
    """
    def pairs(es):
        return [(_norm(e), _norm(_entity_canonical(e))) for e in es]

    per_leg = {leg: pairs(legs.get(leg, [])) for leg in ENTITY_LEG_ORDER}
    all_pairs = [pr for leg in ENTITY_LEG_ORDER for pr in per_leg[leg]]
    out: dict[str, dict] = {}
    for ent in expected_raw:
        ent_n, ent_c = _norm(ent), _norm(_entity_canonical(ent))
        if not _entity_pair_satisfied(ent_n, ent_c, all_pairs):
            continue
        hit_legs = [leg for leg in ENTITY_LEG_ORDER
                    if _entity_pair_satisfied(ent_n, ent_c, per_leg[leg])]
        out[ent_n] = {"legs": hit_legs,
                      "retrieval_satisfied": any(l in RETRIEVAL_LEGS for l in hit_legs)}
    return out


def _retrieval_entity_hit(rec: dict) -> bool:
    """Did a fact, a graph-expanded fact or a graph neighbour carry this hit?

    Reads the record rather than the result, so the same call works on a record
    this version wrote and on one an older run wrote: the scorer's boolean is
    preferred, and a record with neither it nor `scoring.entity_legs` — every
    artifact on disk before #1548 — counts as NOT carried. That is the honest
    reading of an unattributed hit, and it is the same rule #1547 set for an
    unrecorded seeding: absence is not a measurement, and it is never a guess.
    """
    sc = rec.get("scoring") or {}
    if "entity_hit_retrieval_carried" in sc:
        return bool(sc["entity_hit_retrieval_carried"])
    attrib = sc.get("entity_legs") or {}
    return any((v or {}).get("retrieval_satisfied") for v in attrib.values())


def _doc_paths(result: dict) -> list[str]:
    return [str(d.get("path", "")) for d in (result.get("documents") or [])]


def _norm(s: str) -> str:
    """Normalize for matching: lowercase, treat - and _ as equivalent."""
    return str(s or "").lower().replace("-", "_")


def _entity_canonical(name: str) -> str:
    """The store's canonical form of an entity name, or the name itself.

    The scorer compared two spellings and nothing else (#1260): `entity_matches`
    asked whether the gold string is a substring of a retrieved string, so an
    answer that names the SAME entity under its alias reads as a miss, and one
    whose canonical is *shorter* than the gold can never match at all — the 09-18
    record returned `Task #363` against a gold of `Backlog Item #363` and was
    scored a miss. Resolution runs on both sides through the one route that owns
    the question, `app.kg_store`'s alias table and entity registry.

    A name that does not resolve keeps its own spelling, and an unreadable store
    leaves the whole comparison exactly as it was. A scorer that degraded to
    "nothing resolves" on a broken store would report the resulting misses as a
    retrieval regression; the failure to look has to be invisible in the numbers,
    which is why it is not also reported as a zero here.
    """
    text = str(name or "").strip()
    if not text:
        return text
    try:
        return store().resolve(text) or text
    except StoreUnavailable:
        return text


def _entity_pair_satisfied(exp: str, exp_canon: str,
                           got_pairs: list[tuple[str, str]]) -> bool:
    """THE entity-label rule, written once. `exp`/`exp_canon` are the normalized
    gold and its canonical; `got_pairs` is the same pair for each name on the other
    side. An expected entity counts when a returned name either contains it as
    before, or IS it once both go through the store (#1260). Canonical forms are
    compared for *equality*, never substring, because resolution answers "is this
    the same entity", a question substring could not ask: the alias table maps `KG`
    and `Relationship Graph` onto `Knowledge Graph` and pointedly does NOT map
    `Graph` onto it, so a returned `Graph` must stay unsatisfied by a gold
    `Knowledge Graph` even though the two share a word. The substring rule stays as
    the additive half rather than being replaced: it is what carries an expectation
    written as a partial surface (`TGS-RAG Implementation` against the row `#363
    TGS-RAG Implementation`), and dropping it would retire a satisfiability route
    `tests/test_eval_corpus_guard.py` exists to defend.

    The pair arguments are pre-normalized and pre-resolved rather than resolved
    inside, because `_score` resolves each side once per query; a labeler-side
    caller uses `entity_label_satisfied`, which does the resolution for one label.
    Both routes reach the same two-clause predicate, which is the point: a second
    *expression* of this rule — even a correct one — is the defect this repo has
    catalogued twice, since two copies of one matcher drift and then the ceiling
    and the metric it bounds are measured under different definitions of a match
    while still looking commensurable.
    """
    return any(exp in got or (exp_canon == got_canon)
               for got, got_canon in got_pairs)


def entity_label_satisfied(label: str, choices: list[str]) -> bool:
    """Does any name in `choices` satisfy the gold label `label`? Same rule as
    `_score`, reached by way of `_entity_pair_satisfied`.

    Public because `eval/label_agreement_ceiling.py` asks this identical question of
    a second labeler's answer, and its ceiling is only commensurable with
    `entity_hit_rate` if "satisfied" means the same thing in both.

    A name that does not resolve keeps its own spelling and an unreadable store
    degrades to substring-only, exactly as `_entity_canonical` does — so an
    agreement figure computed against a broken store quietly becomes
    substring-only, which is why the ceiling artifact records the store it ran
    against instead of pretending the number is store-independent.
    """
    return _entity_pair_satisfied(
        _norm(label), _norm(_entity_canonical(label)),
        [(_norm(c), _norm(_entity_canonical(c))) for c in choices])


def _doc_pair_satisfied(exp: str, got_docs: list[str]) -> bool:
    """THE doc-label rule: the normalized gold is a substring of a normalized
    returned path, as `expect_docs` documents it. Substring only, no
    canonicalisation — there is no alias table for a path, and inventing one would
    be a second matcher again. `_score` and the second labeler both come through
    here; see `_entity_pair_satisfied` for why that matters."""
    return any(exp in got for got in got_docs)


def doc_label_satisfied(label: str, choices: list[str]) -> bool:
    """Public wrapper: does any path in `choices` satisfy gold label `label`?"""
    return _doc_pair_satisfied(_norm(label), [_norm(c) for c in choices])


def _doc_satisfied_by(exps: list[str], got: str) -> bool:
    """Does this ONE returned path satisfy any gold label? `_doc_pair_satisfied`
    read transposed — the rank scan walks the returned paths, so the roles swap and
    the predicate is reached by name rather than restated. A restatement here is how
    a rank scan and a recall count start disagreeing about what a hit is: `mrr_doc`
    would credit a document `doc_recall` did not count, and the two numbers would
    come from different definitions of the same event."""
    return any(_doc_pair_satisfied(exp, [got]) for exp in exps)


def _ndcg_at_k(got_docs: list[str], expected_docs: list[str], k: int = 10) -> float:
    """Binary-relevance NDCG@k. Each got_docs[i] (i<k) scores 1 if it
    matches any expected substring, else 0. IDCG is computed against the
    actual count of relevant docs found in top-k (not len(expected_docs)),
    because expectations are substrings and may each match multiple docs.
    Returns 0.0 when no relevant docs in top-k or no expectations."""
    if not expected_docs:
        return 0.0
    import math as _m
    # Same rule as the rank scan and the recall count, so one document cannot be a
    # hit for NDCG and a miss for doc_recall.
    rel = [1 if _doc_satisfied_by(expected_docs, got) else 0 for got in got_docs[:k]]
    num_rel = sum(rel)
    if num_rel == 0:
        return 0.0
    dcg = sum(r / _m.log2(i + 2) for i, r in enumerate(rel))
    idcg = sum(1.0 / _m.log2(i + 2) for i in range(num_rel))
    return dcg / idcg


def _score(query_spec: dict, result: dict, seeds: list[str] | None = None) -> dict:
    expected_raw = [str(e) for e in (query_spec.get("expect_entities") or [])]
    expected_entities = [_norm(e) for e in expected_raw]
    expected_doc_raw = [str(d) for d in (query_spec.get("expect_docs") or [])]
    expected_docs = [_norm(d) for d in expected_doc_raw]
    legs = _entity_legs(result, seeds)
    got_raw = _entities_in_result(result, seeds)
    got_paths_raw = _doc_paths(result)
    got_entities = [_norm(e) for e in got_raw]
    got_docs = [_norm(p) for p in got_paths_raw]

    # Both legs match through the ONE rule — `_entity_pair_satisfied` /
    # `_doc_pair_satisfied`, above — which the second labeler reaches by way of the
    # public wrappers. That is the whole reason the rule is a function and not a
    # closure here: `eval/label_agreement_ceiling.py` scores an independent
    # labeller's answer against these same gold labels, and a ceiling is only a
    # ceiling if "satisfied" means the same thing on both sides of the division.
    # The fact leg keeps substring alone — see the comment on `fact_entity_recall`
    # below; #1164's acceptance is written against that number, and a ceiling for
    # the fact leg is therefore not measured at all
    # (`UNMEASURED_METRICS` in label_agreement_ceiling.py).
    expected_canon = [_norm(_entity_canonical(e)) for e in expected_raw]
    got_canon = [_norm(_entity_canonical(e)) for e in got_raw]
    got_pairs = list(zip(got_entities, got_canon))

    entity_matches = [exp for exp, exp_canon in zip(expected_entities, expected_canon)
                      if _entity_pair_satisfied(exp, exp_canon, got_pairs)]
    # Computed once, recorded twice below: the per-entity legs and the query's
    # boolean are the same call, so the artifact can never carry a mark that
    # disagrees with its own attribution.
    entity_legs_attrib = _entity_leg_attribution(legs, expected_raw)
    doc_matches = [exp for exp in expected_docs
                   if _doc_pair_satisfied(exp, got_docs)]

    # Rank of FIRST matching expected doc in returned list (1-indexed; None if none).
    first_doc_rank = None
    for rank, got in enumerate(got_docs, start=1):
        if _doc_satisfied_by(expected_docs, got):
            first_doc_rank = rank
            break

    # Reciprocal rank (0 if not found). Useful for MRR-style aggregates.
    rr_doc = (1.0 / first_doc_rank) if first_doc_rank else 0.0
    ndcg10 = _ndcg_at_k(got_docs, expected_docs, k=10)

    # Facts-side scoring: how many expected entities appear in returned facts
    # (not just seeds/neighbors)? This isolates whether the entity attribution
    # makes its way into the fact retrieval, separate from the seed extractor.
    fact_entities = []
    for f in (result.get("facts") or []):
        e = _norm(f.get("entity", "") or "")
        if e:
            fact_entities.append(e)
    fact_matches = [exp for exp in expected_entities if any(exp in fe for fe in fact_entities)]
    fact_entity_recall = (len(fact_matches) / len(expected_entities)) if expected_entities else None

    return {
        "entity_hit": bool(entity_matches),
        "doc_hit": bool(doc_matches),
        "entity_recall": (len(entity_matches) / len(expected_entities)) if expected_entities else None,
        "doc_recall": (len(doc_matches) / len(expected_docs)) if expected_docs else None,
        "first_doc_rank": first_doc_rank,
        "rr_doc": round(rr_doc, 4),
        "ndcg10": round(ndcg10, 4),
        "fact_entity_recall": fact_entity_recall,
        "entities_matched": entity_matches,
        # WHICH LEG satisfied each matched entity (#1548), keyed by the `_norm`'d
        # gold label: `legs` names the entries of `ENTITY_LEG_ORDER`
        # (`seed`/`fact`/`graph_expanded_fact`/`graph_neighbor`) that carry it. Without it the
        # artifact cannot say whether `entity_hit` came from retrieval or from the
        # query naming its own gold entity, and `entity_hit_rate` reads as retrieval
        # when the seed extractor is what produced it.
        "entity_legs": entity_legs_attrib,
        # The one boolean the aggregate needs: THIS query's hit was carried by
        # retrieval, not by the query's own seeds. `any()` over the attribution
        # above, so the mark and the rate cannot disagree (#1548).
        "entity_hit_retrieval_carried": any(
            (v or {}).get("retrieval_satisfied") for v in entity_legs_attrib.values()),
        # Which of the query's expected entities came back through the FACT
        # pool. `fact_entity_recall` is the average of this list's length, and an
        # average cannot say which entity moved; this says which, so
        # `count_overreach_regressions` below can name the query and the entity
        # a fact-side write took out of the answer.
        "fact_entities_matched": fact_matches,
        "docs_matched": doc_matches,
    }


# The five retrieval knobs below (`graph_rerank`, `rerank_alpha`, `graph_top_k`,
# `graph_hops`, `seed_top_k`) default to production's value BY IMPORT, never by a
# restated literal. Until #498 the literals here were the pre-#322 settings —
# `rerank_alpha=0.5` against production's 0.3 — and 59cd7bf fixed only the
# argparse half, so any programmatic caller inherited a configuration nothing
# serves (its first version of `agent_mcp/fact_improvement.py:_fact_entity_recall`
# reported exactly such a number). `expand_graph` is deliberately NOT one of the
# four: production's default for that knob is False (`_vault_recall`) while the
# eval runs it on as a measurement choice, and #1000 owns that claim.
# tests/test_eval_scorer.py pins both directions: the signature equals the
# constants, and build_parser()'s defaults equal the signature.
def count_overreach_regressions(before: list[dict], after: list[dict]) -> list[dict]:
    """Queries whose expected entity was retrievable before and is not now.

    The number that reads a fact-side deletion as the regression it may be.
    `fact_entity_recall` cannot do this: it is an average over queries of the
    same tree, so expiring one side of a pair leaves the claim retrievable
    through its twin and reports no change, while expiring an entity's
    best-scoring fact ejects it from the ten-slot fact pool and moves the
    average — a movement that is blast radius, not quality. Averaging hides both
    cases; this compares two runs of the SAME queries and names the queries
    where an expected entity stopped being returned at all.

    Expiring the older twin of a near-duplicate yields no entry here (the
    surviving twin still carries the entity), which is the whole point: a
    correction is not a regression, an over-reach is.

    `before`/`after` are `run_eval()` return values, so this is testable over
    records without a corpus, and is how the nightly comparison is meant to be
    run against a snapshot copy via `LLOYD_FACTS_ROOT`/`LLOYD_KG_DB`.
    """
    after_by_query = {r["query"]: r for r in after}
    out = []
    for rec in before:
        now = after_by_query.get(rec["query"])
        if now is None:
            continue
        had = set(rec["scoring"].get("fact_entities_matched") or [])
        if not had:
            continue
        gone = sorted(had - set(now["scoring"].get("fact_entities_matched") or []))
        if gone:
            out.append({"query": rec["query"], "category": rec.get("category"),
                        "entities": gone})
    return out


# `expand_graph` is the one knob whose signature default is deliberately NOT
# the production constant. Production defaults it False (`RECALL_EXPAND_GRAPH`);
# the eval scores recall with the graph expanded on purpose, and scoring it
# closed is a different measurement, not a parity fix (#498 fixed the signature
# for the other knobs, #1000 makes only the claim about this one honest). What
# the eval applies is recorded in the baseline as `expand_graph`, and whether
# that equals production is recorded beside it as
# `expand_graph_matches_production`.
def run_eval(queries: list[dict], limit: int = 20, expand_graph: bool = True,
             graph_rerank: bool = RECALL_GRAPH_RERANK,
             rerank_alpha: float = RECALL_RERANK_ALPHA,
             demote_factor: float | None = None,
             graph_top_k: int = RECALL_GRAPH_TOP_K,
             graph_hops: int = RECALL_GRAPH_HOPS,
             seed_top_k: int = RECALL_SEED_TOP_K,
             djev_rerank: bool = RECALL_DJEV_RERANK,
             djev_rerank_top: int = RECALL_DJEV_RERANK_TOP,
             counterfactual: bool = True) -> list[dict]:
    records = []
    # Frozen perturbation records, loaded once. Absent or short is surfaced per
    # query below as an unscored counterfactual block rather than crashing the
    # run: the existing metrics are still worth having, and the compare step
    # already refuses to compare a baseline whose shape it cannot trust.
    perturbations = {}
    if counterfactual:
        try:
            perturbations = cf.load_records()
        except (FileNotFoundError, KeyError, yaml.YAMLError) as e:
            print(f"  [counterfactual] records unusable ({type(e).__name__}: {e})")
    for spec in queries:
        qid = spec.get("id")
        query = spec.get("query", "")
        if not query:
            continue
        t0 = time.perf_counter()
        try:
            recall_params: dict = {
                "query": query,
                "limit": limit,
                "expand_graph": expand_graph,
                "graph_rerank": graph_rerank,
                "rerank_alpha": rerank_alpha,
                "graph_top_k": graph_top_k,
                "graph_hops": graph_hops,
                # The adoption arm. It is a real parameter of the handler and
                # not a log, because the question — "does djev's ordering beat
                # qmd's own reranker" — can only be answered on the labelled
                # set, and a shadow row has no labels.
                "djev_rerank": djev_rerank,
                "djev_rerank_top": djev_rerank_top,
            }
            if demote_factor is not None:
                recall_params["demote_factor"] = demote_factor
            # The width goes to retrieval as a KEYWORD argument, never as a
            # `seed_top_k` key inside `recall_params`: `_vault_recall` is the
            # `vault_recall` MCP handler and its `params` dict is the client's
            # raw arguments, so a key put there is an undocumented tool
            # parameter the schema does not declare (#843 review). Keyword-only
            # keeps the property that matters here — the record's seeds are the
            # seeds retrieval actually used — while leaving the wire shape
            # exactly the schema's.
            result = _vault_recall(recall_params, seed_top_k=seed_top_k)
            err = None
        except Exception as e:
            result = {"documents": [], "facts": []}
            err = f"{type(e).__name__}: {e}"
        latency_ms = (time.perf_counter() - t0) * 1000

        # Bounded by the knob, whose default IS `RECALL_SEED_TOP_K`, so both
        # this record and the recall call above carry production's width by
        # default and neither repeats the number. Until #843 this line said
        # `[:5]` while `_vault_recall` sliced the same extractor at 10 —
        # `entity_hit` is a union with these seeds first and largest, so the
        # eval called a miss what production served as a hit, on 13 of the 20
        # bench queries, and the artifact reported `matches_production_defaults:
        # true` because the conjunction never looked at the seed count.
        # `recall_seeds` is the one definition vault_recall uses too (#1486):
        # the lexical head at `seed_top_k`, plus the semantic seeds when that
        # switch is on — so the record carries exactly the seeds retrieval used.
        if _semantic_seed_k():
            seeds = _recall_seeds(query, seed_top_k)
        else:
            seeds = [e for e, _ in
                     (_extract_entities_from_query(query) or [])[:seed_top_k]]
        scoring = _score(spec, result, seeds=seeds)

        rec = {
            "id": qid,
            "query": query,
            "category": spec.get("category"),
            "seeds_extracted": seeds,
            "result_summary": {
                "n_docs": len(result.get("documents") or []),
                "n_facts": len(result.get("facts") or []),
                "n_graph_facts": len(result.get("graph_expanded_facts") or []),
                "n_neighbors": len(result.get("graph_neighbors_used") or []),
                # #1250: what the fact leg FAILED to read, and the first
                # failure's repr. `n_facts: 0` alone cannot say whether the
                # tree was empty, every read failed, or nothing matched — and
                # the six zeroed arms of 2026-09-18 proved a reader cannot tell
                # those apart after the fact. Older artifacts simply lack these
                # keys, which is why the guard in `fact_read_coverage` reads
                # `n_facts` and never these.
                "n_fact_reads_failed": int(result.get("n_fact_reads_failed") or 0),
                "fact_read_first_error": result.get("fact_read_first_error"),
                "doc_paths_top10": _doc_paths(result)[:10],
                "fact_entities_top10": _retrieval_entities(result)[:10],
                "neighbors": [
                    {"entity": n.get("entity"), "weight": n.get("weight")}
                    for n in (result.get("graph_neighbors_used") or [])[:5]
                ],
            },
            "expected": {
                "entities": spec.get("expect_entities") or [],
                "docs": spec.get("expect_docs") or [],
            },
            "scoring": scoring,
            "latency_ms": round(latency_ms, 1),
            "error": err,
        }
        if counterfactual:
            _attach_counterfactual(rec, spec, result, seeds, recall_params,
                                   perturbations.get(qid), seed_top_k=seed_top_k)
            # Float where scoreable, None where not — summarize averages over
            # the non-None values, which is how a query that cannot be pinned
            # stays out of pinned_rate's denominator instead of inflating it.
            cfx = rec.get("counterfactual") or {}
            if cfx:
                scoring["counterfactual_moved_rate"] = (
                    1.0 if cfx["counterfactual_moved"] else 0.0)
                pinned = cfx["counterfactual_pinned"]
                scoring["counterfactual_pinned_rate"] = (
                    None if pinned is None else (1.0 if pinned else 0.0))
            else:
                scoring["counterfactual_moved_rate"] = None
                scoring["counterfactual_pinned_rate"] = None
        records.append(rec)
    return records


def _attach_counterfactual(rec: dict, spec: dict, result: dict, seeds: list[str],
                           recall_params: dict, perturbation: dict | None,
                           *,
                           seed_top_k: int = RECALL_SEED_TOP_K) -> None:
    """Run this query's perturbed twin through the SAME path and score both
    directions onto the record.

    Reusing `recall_params` with only `query` swapped is the whole point: the
    two arms must differ in the one constraint and nothing else, so any
    difference in the entity-level output is attributable to that constraint.
    `seed_top_k` is a parameter rather than a `recall_params` key — the dict is
    the MCP wire shape and the width is not in the `vault_recall` schema — and
    `run_eval` passes the same value it passed to the reference arm, so the
    variant's seed slice is bounded by it rather than by a width of its own: an
    arm cut to a different count from the arm retrieval seeded at makes
    `seed_moved` — the label #537 depends on — a comparison of two truncations
    instead of a comparison of two queries (#843).
    """
    if not perturbation:
        rec["counterfactual"] = None
        rec["counterfactual_error"] = "no perturbation record for this query id"
        return
    variant_query = perturbation.get("perturbed_query") or ""
    try:
        variant = _vault_recall({**recall_params, "query": variant_query},
                                seed_top_k=seed_top_k)
        variant_seeds = [
            e for e, _ in
            (_extract_entities_from_query(variant_query) or [])[:seed_top_k]]
        rec["counterfactual"] = cf.score_pair(perturbation, result, seeds,
                                              variant, variant_seeds)
        rec["counterfactual"]["seeds_variant"] = variant_seeds
    except Exception as e:
        # A variant that errored is not a retrieval failure; scoring it would
        # put a retriever bug report on a harness bug.
        rec["counterfactual"] = None
        rec["counterfactual_error"] = f"{type(e).__name__}: {e}"


# The seven scored metrics and how each one's run-to-run uncertainty is
# estimated (#696). A hit/miss rate is a binomial proportion and takes the
# closed-form Wilson interval; the other five are means of a per-query value
# that is not a count of successes — an mrr_doc of 0.447 is not "8.9 of 20" —
# so their interval comes from resampling the run's own per-query vector. The
# per-query values are already persisted (`records[].scoring`), so this adds
# arithmetic and nothing to collect.
CI_METRICS = {
    "entity_hit_rate": ("entity_hit", "wilson"),
    # The same denominator, the narrower numerator: only the hits a fact, a
    # graph-expanded fact or a graph neighbour carried. It exists because
    # `entity_hit_rate` alone is satisfiable by the query's own extracted seeds
    # (#1548), so a reader who takes it as "what retrieval returned" is reading the
    # seed extractor. Trend THIS one for retrieval quality; the gap between the two
    # is the seed-carried share, and it is reported, not hidden.
    "entity_hit_rate_retrieval_carried": ("entity_hit_retrieval_carried", "wilson"),
    "doc_hit_rate": ("doc_hit", "wilson"),
    "entity_recall_avg": ("entity_recall", "bootstrap"),
    "doc_recall_avg": ("doc_recall", "bootstrap"),
    "mrr_doc": ("rr_doc", "bootstrap"),
    "ndcg10": ("ndcg10", "bootstrap"),
    "fact_entity_recall_avg": ("fact_entity_recall", "bootstrap"),
}


def confidence_intervals(records: list[dict], *,
                         n_resamples: int = evstats.N_RESAMPLES,
                         seed: int = evstats.SEED) -> dict:
    """The 95 % interval beside each of the seven overall metrics.

    Entry shape: ``{"ci": [lo, hi], "n": int, "kind": "wilson"|"bootstrap"}``,
    plus `k` (the hit count) on the two rates. Every entry carries its own `n`
    because a 20-query run is not one denominator: `fact_entity_recall_avg` is
    scored only where the corpus HAS the entity row, and a query whose
    expectation list is empty scores None for a recall (#541) and leaves that
    metric's average. Averaging over 4 of 20 and reporting an interval over 20
    would be the zero-denominator failure in a new costume.

    `ci` is `[null, null]` — not `[0, 0]`, not `[1, 1]` — when there is nothing
    to bound: a `NaN` is not valid JSON and a baseline artifact has to stay
    parseable, so the empty case serialises as nulls and the printer says
    "no verdict". A one-query run gets nulls for the same reason: a percentile
    bootstrap on a single value reports that value back with zero width, which
    is not an interval.

    This is a WITHIN-run interval (how precisely this run measured this corpus).
    It is deliberately NOT the night-over-night interval — the nightly corpus is
    not pinned between runs, so a night-to-night comparison is unpaired and needs
    the wider `independent_bootstrap_ci` over the two runs' vectors, which
    `scripts/eval_trend_stats.py` owns. Quoting this block as if it bounded a
    delta between two nights is the misuse clause 5 of #696 names out loud.
    """
    out: dict = {}
    for metric, (field, method) in CI_METRICS.items():
        vals = [r["scoring"].get(field) for r in records]
        known = [v for v in vals if v is not None]
        if method == "wilson":
            n = len(known)
            if n == 0:
                out[metric] = {"ci": [None, None], "n": 0, "k": 0, "kind": "wilson"}
                continue
            k = sum(1 for v in known if v)
            lo, hi = evstats.wilson_ci(k, n)
            out[metric] = {"ci": [round(lo, 4), round(hi, 4)], "n": n, "k": k,
                           "kind": "wilson"}
        else:
            res = evstats.bootstrap_mean_ci(known, n_resamples=n_resamples, seed=seed)
            out[metric] = {
                "ci": ([None, None] if res["lo"] is None
                       else [round(res["lo"], 4), round(res["hi"], 4)]),
                "n": res["n"], "kind": "bootstrap",
            }
    out["params"] = {"confidence": 0.95, "n_resamples": n_resamples, "seed": seed}
    return out


def _fmt_ci(metric: str, overall: dict) -> str:
    """The interval suffix for one printed metric line: ` [0.300,0.701] n=20`.

    A zero or absent denominator prints `[no verdict]`, never a rate and never a
    bracket — that is the same rule `METRIC_NAN_POLICY` already applies to the
    stored value, extended to the printed line so a reader cannot take "0.00" of
    an unscored metric for a measured zero (#1260 fixed the fact metric's side of
    this in the artifact; this is its side of the page).
    """
    entry = (overall.get("ci95") or {}).get(metric)
    if not isinstance(entry, dict):
        return "  [no interval] n=?"
    n = entry.get("n") or 0
    ci = entry.get("ci")
    if n <= 0 or not ci or any(b is None for b in ci):
        return f"  [no verdict] n={n}"
    return f"  [{ci[0]:.3f},{ci[1]:.3f}] n={n}"


#: Which metrics have no gold-side ceiling, and why. Mirrors
#: `UNMEASURED_METRICS` in `eval/label_agreement_ceiling.py`; restated here as the
#: reporter's reason string so every metric gets a reason even when the artifact is
#: absent and nothing can be imported from it.
CEILING_UNMEASURED = {"fact_entity_recall_avg":
                      "the second labeler labels entities and document paths, not "
                      "fact rows, so no gold-side surrogate exists for the fact leg"}


def ceiling_context(queries: list[dict], *, scored_ids: list[str] | None = None) -> dict:
    """Read the label-agreement artifact and turn it into `summary.overall` fields.

    `scored_ids` must be the ids THIS run scored. The ceiling is re-averaged over
    exactly those queries, because a ratio is only meaningful when both halves are
    over the same experiment: `--limit 3` scores three queries, and dividing a
    three-query hit rate by an eighty-one-query ceiling produces a number that looks
    like a position on a scale and is nothing of the kind. `None` means the whole
    corpus, which is what the nightly does.

    The returned dict always carries these keys, whatever it finds:
      `label_agreement`        {"entity_label_agreement", "doc_label_agreement",
                                "labeler", "artifact", "reason"?}
      `ceiling`                {"kind", per-metric values, "reason"?}
      `<metric>_normalized`    score / ceiling, or null
      `<metric>_ceiling_kind`  which kind of ceiling that divisor was, or null

    Four states it distinguishes, because the whole subject of the field is that an
    absent or stale ceiling must not read as a measured one: no artifact; an artifact
    written by a stub labeler; an artifact whose `labels_sha256` no longer matches
    the corpus (commit `9b028e9` re-pointed 22 gold entity names under an unchanged
    id set, which the McNemar join on query ids cannot see); and an artifact whose
    stored agreement does not recompute from its own contents. Each arrives as
    `ceiling: null` plus a `reason` naming the state — never as a dropped key
    (invisible to the next reader) and never as a 0.0 (a measured floor that was
    never measured).

    The two ceilings in this file are different instruments and the normalized value
    names which one it divided by: `anchorless_query_count` bounds a query the seeds
    cannot anchor, which no retriever can hit, while `gold_label_surrogate` bounds
    what any answer agreeing with an independent labeller can score on these gold
    labels. A query can be anchorable and still be label-noise; conflating the two
    would let a seed defect excuse a label defect.

    The import is deliberately inside the function: a nightly must still emit its raw
    numbers if the labeler module is broken, and a reporter that dies because the
    instrument it annotates is unreadable would take the retrieval metric down with
    it. That failure is a `reason` string, not a traceback.
    """
    try:
        import eval.label_agreement_ceiling as lac
    except Exception as exc:
        return _ceiling_absent_fields(f"labeler module unavailable: {exc}")
    try:
        art = lac.load_artifact(expect_labels_sha256=lac.labels_sha256(queries))
    except lac.ArtifactRefused as refused:
        return _ceiling_absent_fields(str(refused))
    except Exception as exc:
        return _ceiling_absent_fields(f"label-agreement artifact unreadable: {exc}")

    # Recompute rather than read the stored block, and over THIS run's scored ids. The
    # stored `ceiling` is the corpus-wide one; a run that scored a subset must not
    # divide by it, and recomputing through `lac.ceiling` costs one pass over rows
    # already in memory.
    ceil = lac.ceiling(art, ids=scored_ids) if scored_ids is not None else art["ceiling"]
    agree = art["agreement"]
    fields: dict = {
        "label_agreement": {
            "entity_label_agreement": agree["entity_label_agreement"],
            "doc_label_agreement": agree["doc_label_agreement"],
            "entity_labels": agree["entity_labels"],
            "entity_labels_agreed": agree["entity_labels_agreed"],
            "entity_labels_offered": agree["entity_labels_offered"],
            "doc_labels": agree["doc_labels"],
            "doc_labels_agreed": agree["doc_labels_agreed"],
            "doc_labels_offered": agree["doc_labels_offered"],
            "labeler": art["labeler"],
            "artifact": art["_path"],
            "labels_sha256": art["corpus"]["labels_sha256"],
            # Named inline so a sub-0.80 agreement cannot be read without the labels
            # that caused it. The full per-query record lives in the artifact.
            "entity_disagreements": [f"{d['id']} gold={d['gold']}"
                                     for d in agree["disagreements"]["entity"]],
            "doc_disagreements": [f"{d['id']} gold={d['gold']}"
                                  for d in agree["disagreements"]["doc"]],
        },
        "ceiling": {"kind": ceil["kind"], "artifact": art["_path"],
                    "ran_at": art["ran_at"], "labeler": art["labeler"],
                    "n": ceil["n"], "excluded": ceil["excluded"],
                    "unmeasured": ceil["unmeasured"],
                    **{m: v for m, v in ceil["values"].items()}},
    }
    for metric in CI_METRICS:
        fields[f"{metric}_normalized"] = None
        fields[f"{metric}_ceiling_kind"] = (
            ceil["kind"] if ceil["values"].get(metric) is not None else None)
    return fields


def _ceiling_absent_fields(reason: str) -> dict:
    """The null-and-explain shape, used for every state with no usable ceiling."""
    fields: dict = {
        "label_agreement": {"entity_label_agreement": None,
                            "doc_label_agreement": None, "labeler": None,
                            "artifact": None, "reason": reason},
        "ceiling": {"kind": None, "reason": reason},
    }
    for metric in CI_METRICS:
        fields[f"{metric}_normalized"] = None
        fields[f"{metric}_ceiling_kind"] = None
    return fields


def _normalize_against_ceiling(fields: dict, overall: dict) -> None:
    """Fill `<metric>_normalized = score / ceiling` for each metric that has a
    ceiling, once the raw aggregates exist. Mutates `fields` in place.

    A ceiling of exactly 0.0 leaves the value null with a note beside it: it means no
    answer at all could satisfy these gold labels, so the ratio is 0/0 and a printed
    0.0 would claim a scored position where there is a division by nothing.
    """
    ceil = fields.get("ceiling") or {}
    for metric in CI_METRICS:
        cap = ceil.get(metric)
        score = overall.get(metric)
        if cap is None or score is None:
            continue
        # The kind travels with the ratio it describes — set here rather than left to
        # the caller, so no path can produce a normalized number whose divisor kind is
        # unknown. A ratio without its kind is the ambiguity this field exists to
        # remove: the seed-side and gold-side ceilings bound different failures.
        fields[f"{metric}_ceiling_kind"] = ceil.get("kind")
        if not cap:
            fields.setdefault("ceiling_notes", {})[metric] = (
                "ceiling is 0.0; score/ceiling undefined")
            continue
        fields[f"{metric}_normalized"] = round(score / cap, 4)


def summarize(records: list[dict]) -> dict:
    by_cat = defaultdict(list)
    for r in records:
        by_cat[r.get("category") or "?"].append(r)

    def avg(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 3) if xs else None

    def _cf(rec, key):
        # .get, not []: baselines written before #541 and the automod baseline
        # arm carry no perturbation block, and summarize runs over both.
        return rec["scoring"].get(key)

    # Each rate averages only the queries that were scoreable for that half, so
    # the denominators travel with the numbers: a pinned_rate over 4 of 20
    # queries is not comparable to one over 18, and the whole point of #541 is
    # that a low number here is a finding rather than a malfunction.
    moved_vals = [_cf(r, "counterfactual_moved_rate") for r in records]
    pinned_vals = [_cf(r, "counterfactual_pinned_rate") for r in records]
    anchorless = anchorless_queries(records)

    overall = {
        "n_queries": len(records),
        # The seed-side ceiling, beside the number it bounds. `entity_hit_rate`
        # cannot exceed (n - anchorless)/n until the residue has a recall arm to
        # reach it with (#1164); a run whose count moved is a run whose extractor
        # changed, not one whose search got better.
        "anchorless_query_count": len(anchorless),
        "anchorless_query_ids": anchorless,
        "entity_hit_rate": avg([1.0 if r["scoring"]["entity_hit"] else 0.0 for r in records]),
        # #1548: retrieval-only, by construction — a hit counts here only if a
        # fact, a graph-expanded fact or a graph neighbour carried a matched
        # entity. <= `entity_hit_rate` on any record set, and the difference is
        # exactly the queries whose only matched entity came from `seeds_extracted`
        # (or from nothing at all, which is the same reading: retrieval did not
        # carry it). Read alongside `fact_entity_recall_avg`, whose denominator is
        # the fact leg and which therefore cannot be seed-inflated either.
        "entity_hit_rate_retrieval_carried": avg(
            [1.0 if _retrieval_entity_hit(r) else 0.0 for r in records]),
        "doc_hit_rate": avg([1.0 if r["scoring"]["doc_hit"] else 0.0 for r in records]),
        "entity_recall_avg": avg([r["scoring"]["entity_recall"] for r in records]),
        "doc_recall_avg": avg([r["scoring"]["doc_recall"] for r in records]),
        "mrr_doc": avg([r["scoring"]["rr_doc"] for r in records]),
        "ndcg10": avg([r["scoring"]["ndcg10"] for r in records]),
        "fact_entity_recall_avg": avg([r["scoring"]["fact_entity_recall"] for r in records]),
        # ...and each with its 95 % interval and its own denominator (#696).
        "ci95": confidence_intervals(records),
        "latency_ms_avg": avg([r["latency_ms"] for r in records]),
        "errors": sum(1 for r in records if r.get("error")),
        "counterfactual_moved_rate": avg([v for v in moved_vals]),
        "counterfactual_pinned_rate": avg([v for v in pinned_vals]),
        "counterfactual_n_moved": len([v for v in moved_vals if v is not None]),
        "counterfactual_n_pinned": len([v for v in pinned_vals if v is not None]),
    }
    # `label_agreement`, `ceiling` and the per-metric `<metric>_normalized` /
    # `<metric>_ceiling_kind` go beside the raw aggregates (#654). Emitted for all
    # seven metrics — including `fact_entity_recall_avg`, which has no gold-side
    # ceiling at all — so a reader never has to guess whether an absent key meant
    # "no ceiling exists" or "the instrument has never run": an absent key and a
    # measured zero look identical to the next tool that reads this file, which is
    # the failure this whole field exists to close. `ceiling` arrives already
    # populated by `main`, which knows the corpus labels; `summarize` only places
    # the keys and does the division, so it stays a function of its records and
    # every existing caller keeps working unchanged.
    for metric in CI_METRICS:
        overall.setdefault(f"{metric}_normalized", None)
        overall.setdefault(f"{metric}_ceiling_kind", None)
    overall.setdefault("label_agreement", None)
    overall.setdefault("ceiling", None)

    per_cat = {}
    for cat, rs in by_cat.items():
        per_cat[cat] = {
            "n": len(rs),
            "entity_hit_rate": avg([1.0 if r["scoring"]["entity_hit"] else 0.0 for r in rs]),
            "entity_hit_rate_retrieval_carried": avg(
                [1.0 if _retrieval_entity_hit(r) else 0.0 for r in rs]),
            "doc_hit_rate": avg([1.0 if r["scoring"]["doc_hit"] else 0.0 for r in rs]),
            "entity_recall_avg": avg([r["scoring"]["entity_recall"] for r in rs]),
            "mrr_doc": avg([r["scoring"]["rr_doc"] for r in rs]),
            "ndcg10": avg([r["scoring"]["ndcg10"] for r in rs]),
            "fact_entity_recall_avg": avg([r["scoring"]["fact_entity_recall"] for r in rs]),
            "counterfactual_moved_rate": avg([_cf(r, "counterfactual_moved_rate") for r in rs]),
            "counterfactual_pinned_rate": avg([_cf(r, "counterfactual_pinned_rate") for r in rs]),
        }
    return {"overall": overall, "by_category": per_cat}


def _fmt_rate(value) -> str:
    """null and 0.0 are different facts and must not print the same. A run with
    no perturbation block reads 'null'; a run where nothing moved reads '0.00',
    and only one of those is a finding about retrieval."""
    return "null" if value is None else f"{value:.2f}"


def anchorless_queries(records: list[dict]) -> list[str]:
    """The ids of queries the seed extractor anchored on NOTHING the gold names.

    A query is anchorless when no expected entity appears among its extracted
    seeds — neither as the seed itself nor as a substring of it — so no fact-leg
    read, doc-leg search or graph traversal has anywhere to start. It is a
    property of the seeds, not of the answer: this reads `seeds_extracted`, which
    the run recorded from the same extractor production uses, never the returned
    entities.

    #1260. The harness has always measured the entity leg without saying how
    much of it was scorable, which is why three families of knob-proposing items
    (#569 seed scoring, #633/#634 graph arms, #843 seed width) could each look
    viable against a 0.5 that no knob of theirs could move. The count travels in
    `summary.overall`, so the ceiling is in every baseline artifact rather than
    in whoever last re-derived it by hand.
    """
    out = []
    for rec in records:
        expected = [_norm(e) for e in ((rec.get("expected") or {}).get("entities") or [])]
        if not expected:
            continue
        if rec.get("seeds_extracted") is None:
            # No seeds were RECORDED for this record (a synthetic record, or a
            # baseline written before the field existed). Zero recorded seeds and
            # no recorded seeds are different observations, and reporting the
            # second as the first is a verdict from a missing input — the count
            # would then measure which artifacts have the key.
            continue
        seeds = [_norm(s) for s in rec["seeds_extracted"]]
        if not any(exp in seed or seed in exp for exp in expected for seed in seeds):
            out.append(rec.get("id", "?"))
    return out


def _fmt_rate3(value) -> str:
    """`_fmt_rate` at the three decimals the fact metric is reported at. Same
    rule: `fact_entity_recall_avg` is null on a run whose fact leg read nothing
    (#1250), and a null must not print as 0.000."""
    return "null" if value is None else f"{value:.3f}"


def print_failures(failures: list[dict]) -> None:
    """The labelled defect list #541 asks for. Printed, not just filed in the
    JSON, because the nightly report is read by a person in chat."""
    bad = [f for f in failures if f.get("label")]
    if not bad:
        print("\nCounterfactual failures: none")
        return
    print("\nCounterfactual failures:")
    for f in bad:
        extra = f" ({'; '.join(f['pinned_failures'])})" if f.get("pinned_failures") else ""
        print(f"  {f['id']:<26} {f['axis_changed'] or '?':<10} {f['label']}"
              f"{' -> ' + str(f.get('new_value')) if f.get('new_value') else ''}{extra}")
    keying = cf.identity_keying_evidence(failures)
    if keying:
        print(f"  entity-name swaps the seed extractor did not notice "
              f"(evidence for/against #537): {', '.join(keying)}")


def print_table(records: list[dict], summary: dict) -> None:
    print(f"\n{'id':<26} {'cat':<10} {'eH':<4} {'dH':<4} {'eR':<6} {'dR':<6} {'rank':<5} {'NDCG':<6} {'fER':<6} {'mv':<4} {'pn':<4} {'axis':<10} {'latms':<7}")
    print("-" * 118)
    for r in records:
        s = r["scoring"]
        eh = "✓" if s["entity_hit"] else "✗"
        dh = "✓" if s["doc_hit"] else "✗"
        er = f"{s['entity_recall']:.2f}" if s["entity_recall"] is not None else "—"
        dr = f"{s['doc_recall']:.2f}" if s["doc_recall"] is not None else "—"
        rk = str(s["first_doc_rank"]) if s["first_doc_rank"] else "—"
        ndcg = f"{s['ndcg10']:.2f}"
        fer = f"{s['fact_entity_recall']:.2f}" if s["fact_entity_recall"] is not None else "—"
        cfx = r.get("counterfactual") or {}
        mv = ("✓" if cfx.get("counterfactual_moved") else "✗") if cfx else "—"
        pn = ("—" if not cfx else
              ("·" if cfx.get("counterfactual_pinned") is None else
               ("✓" if cfx.get("counterfactual_pinned") else "✗")))
        axis = cfx.get("axis_changed") or ""
        print(f"{r['id']:<26} {(r['category'] or ''):<10} {eh:<4} {dh:<4} {er:<6} {dr:<6} {rk:<5} {ndcg:<6} {fer:<6} {mv:<4} {pn:<4} {axis:<10} {r['latency_ms']:<7.0f}")
    print()
    o = summary["overall"]
    # fact_entity_recall sits next to MRR because it is the metric the fact
    # side of recall actually moves: MRR and NDCG score documents, and a
    # graph change can lift the facts returned without touching either.
    # `_fmt_rate`, not `or 0`: a null fact metric (the guard's "the fact leg read
    # nothing", #1250) and a measured 0.000 are different facts, and printing
    # both as 0.000 is how the zeroed arms read as ordinary results in the log.
    #
    # One metric per line now, each with its 95 % interval and its own
    # denominator (#696). The single packed line could not carry them and the
    # packing is what made the page unreadable as evidence: one query flipping
    # on a 20-query eval moves a rate by 5 points against an interval roughly 40
    # points wide, so a number copied off this page without its bracket is
    # exactly the reading `scripts/eval_trend_stats.py` now refuses to make.
    # `errors` keeps its place on the header line — it is a count, not a score,
    # and the zero-denominator rule below is about scores.
    print(f"Overall: n={o['n_queries']}  errors={o['errors']}")
    # Which ceiling the numbers below are to be read against, and the one reading
    # that is NOT licensed without a person's call (#654 acceptance): agreement is
    # the gate on whether a low entity hit rate is a retrieval defect at all.
    _agree = o.get("label_agreement") or {}
    if _agree.get("entity_label_agreement") is not None:
        _ent = _agree["entity_label_agreement"]
        _verdict = ("labels too ambiguous to call it a retrieval defect"
                    if _ent < 0.80 else
                    "labels hold; the entity hit rate stands as a retrieval result"
                    if _ent > 0.90 else
                    "labels partly ambiguous; read the normalized gap, not the raw")
        print(f"  entity_label_agreement={_ent}  "
              f"doc_label_agreement={_agree.get('doc_label_agreement')}  -> {_verdict}")
    for label, metric, fmt in (("MRR", "mrr_doc", _fmt_rate3),
                               ("NDCG10", "ndcg10", _fmt_rate3),
                               ("doc_hit", "doc_hit_rate", _fmt_rate),
                               ("doc_recall", "doc_recall_avg", _fmt_rate),
                               ("entity_hit", "entity_hit_rate", _fmt_rate),
                               ("entity_recall", "entity_recall_avg", _fmt_rate),
                               ("fact_entity_recall", "fact_entity_recall_avg", _fmt_rate3)):
        # A metric whose denominator is empty prints the rate as null (it was
        # never measured) and the interval as "no verdict" — never a number and
        # never a bracket, so a zero-denominator run cannot read as a pass here
        # either (#1260's rule, applied to the printed page).
        # The gold-side ceiling and the normalized score ride in ON the metric's own
        # line (#654), so a reader cannot copy the raw rate off this page without its
        # denominator — the same reason #696 put the CI on this line. `null` prints as
        # null: "no ceiling exists for this metric" and "ceiling is 0.0" are different
        # facts from a measured value, and a bare `-` would hide which.
        ceiling = o.get("ceiling") or {}
        kind = ceiling.get("kind")
        cap = ceiling.get(metric) if kind else None
        norm = o.get(f"{metric}_normalized")
        # What follows `score/ceiling=` is the RATIO, never the divisor. The label
        # reads as a division, so a first number that is the ceiling would have a
        # reader copy 0.691 off this page as the position while the position
        # (0.5357) sits behind it in parentheses — and on the nightly page the two
        # are one `=` apart. The divisor keeps its own named slot instead.
        if cap is None:
            suffix = f"   score/ceiling=null (ceiling=null kind={kind or 'null'})"
        else:
            suffix = (f"   score/ceiling={'null' if norm is None else norm}"
                      f" (ceiling={cap} kind={kind})")
        print(f"  {label:<20}{fmt(o.get(metric))}{_fmt_ci(metric, o)}{suffix}")
    # The labeler identity and, below 0.80, the disagreement set. The second half is
    # the clause that keeps this instrument from becoming an excuse: a low ceiling
    # reported without the labels that caused it is a reason to stop fixing entity
    # identification, and a low ceiling reported WITH them is a work list.
    agree = o.get("label_agreement") or {}
    if agree.get("entity_label_agreement") is None and agree.get("reason"):
        print(f"  {'gold_ceiling':<20}null — {agree['reason']}")
    elif agree:
        lab = agree.get("labeler") or {}
        print(f"  {'label_agreement':<20}entity={_fmt_rate(agree['entity_label_agreement'])}"
              f"  doc={_fmt_rate(agree['doc_label_agreement'])}"
              f"  (labeler={lab.get('model')}@{lab.get('endpoint')})")
        if (agree["entity_label_agreement"] or 0) < 0.80:
            ent = agree.get("entity_disagreements") or []
            print(f"  {'disagreement set':<20}{len(ent)} gold entity label(s): "
                  f"{', '.join(ent) if ent else '(none listed)'}")
    print(f"  {'avg_lat':<20}{o['latency_ms_avg']:.0f}ms")
    # Printed beside entity_hit/doc_hit because that gap is what #541 exists to
    # explain: retrieval finds a relevant document far more reliably than it
    # finds the right entity row.
    #
    # Each denominator prints OUT OF TOTAL — `pinned=0.90 (n=50/81)`, not `(n=50)`
    # (#763 clause 3). The bare count has been on this page since `af1e8c1`
    # (2026-09-09), and it shows how many queries were scored without saying how
    # many there were. The two legs are deliberately scored over different
    # populations — an entry with no `expected_pinned` feeds moved only — so
    # `(n=66)` beside `(n=50)` reads as one population with 16 rows missing for no
    # stated reason, and a nightly reader comparing the two trend lines compares
    # two unknown sets. The run's own `n_queries` is the out-of-total, so the
    # fraction also names which corpus was scored.
    # The entity leg's two rates on one line, because `entity_hit_rate` on its own
    # is satisfiable by the query's own extracted seeds (#1548) and the delta table
    # above prints it as the entity number. The gap IS the seed-carried share.
    _rc = o.get("entity_hit_rate_retrieval_carried")
    print(f"         entity leg: hit={_fmt_rate(o.get('entity_hit_rate'))} "
          f"of which retrieval-carried={_fmt_rate(_rc)} "
          f"(the rest came only from `seeds_extracted`)")
    mv, pn = o.get("counterfactual_moved_rate"), o.get("counterfactual_pinned_rate")
    total = o.get("n_queries", 0)
    print(f"         counterfactual: moved={_fmt_rate(mv)} "
          f"(n={o.get('counterfactual_n_moved', 0)}/{total})  "
          f"pinned={_fmt_rate(pn)} (n={o.get('counterfactual_n_pinned', 0)}/{total})")
    print("\nBy category:")
    for cat, s in summary["by_category"].items():
        print(f"  {cat:<10} n={s['n']:<3} entity_hit={s['entity_hit_rate']:.2f}  "
              f"doc_hit={s['doc_hit_rate']:.2f}  ent_recall={s['entity_recall_avg']:.2f}  "
              f"MRR={s['mrr_doc']:.3f}  NDCG10={s['ndcg10']:.3f}  "
              f"fER={_fmt_rate3(s['fact_entity_recall_avg'])}")


def build_parser() -> argparse.ArgumentParser:
    """The CLI's parser, as a function so a test can read its real defaults.

    A test that hand-mirrors these `add_argument` calls asserts about values it
    set itself and cannot fail (`tests/test_eval_scorer.py::test_eval_defaults_are_productions`
    is that tautology, and #999 tracks it). Reading this parser is the only
    check that can catch drift, which is why it is a function and not a line
    inside `main()`.

    Every retrieval knob defaults to the production constant it names, imported
    from `agent_mcp.vault`. The nightly eval used to run `graph_rerank=False,
    alpha=0.5` against a production that measured differently — it measured a
    configuration nothing serves, so a retrieval regression could not show up in
    it (2026-09-03 review; the CLI half was fixed in 59cd7bf, `run_eval()`'s
    signature in #498). Say "whatever the constant says" rather than naming a
    value: `RECALL_GRAPH_RERANK` is False today (agent_mcp/vault.py:144-168, the
    2026-09-04 sweep), and a stale comment above its read in `_vault_recall`
    still claims default-on — that comment is #1001.

    `--seed-top-k` is here because the parity flag has to be able to go false.
    A `RECALL_SEED_TOP_K == RECALL_SEED_TOP_K` term would reproduce #1000's
    `and not args.no_graph` defect one constant over — a conjunction can only be
    falsified by something that is not itself the constant — and the flag that
    cannot fail is why a run seeded at 5 against a production seeded at 10
    reported `matches_production_defaults: true` for eight days (#843).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default=str(HERE / "vault_recall_queries.yaml"))
    ap.add_argument("--label", default="baseline", help="Label embedded in the output filename")
    ap.add_argument("--notes", default="", help="Free-text notes saved with the run (e.g. config knobs)")
    ap.add_argument("--limit", type=int, default=20)
    # Defaults ARE production's, imported from agent_mcp.vault — see the
    # build_parser() docstring for why naming a value here would go stale.
    # Deliberately not defaulted to production: `RECALL_EXPAND_GRAPH` is False,
    # and the eval scores recall with the graph expanded on purpose. Whether
    # this run equals production is stated by `expand_graph_matches_production`
    # instead of by the parity flag (#1000).
    ap.add_argument("--no-graph", action="store_true",
                    help=f"Disable expand_graph (eval default: on; production: {RECALL_EXPAND_GRAPH})")
    ap.add_argument("--no-graph-rerank", dest="graph_rerank", action="store_false",
                    default=RECALL_GRAPH_RERANK,
                    help=f"Disable graph-vote re-ranking (production: {RECALL_GRAPH_RERANK})")
    ap.add_argument("--alpha", type=float, default=RECALL_RERANK_ALPHA,
                    help=f"Re-rank alpha: 1.0=pure QMD, 0.0=pure graph (production {RECALL_RERANK_ALPHA})")
    ap.add_argument("--demote-factor", type=float, default=None, help="Daily-log demote factor (default uses module constant 0.4)")
    ap.add_argument("--graph-top-k", type=int, default=RECALL_GRAPH_TOP_K,
                    help=f"Graph expansion breadth (production {RECALL_GRAPH_TOP_K})")
    ap.add_argument("--graph-hops", type=int, default=RECALL_GRAPH_HOPS,
                    help=f"Graph expansion depth (production {RECALL_GRAPH_HOPS})")
    ap.add_argument("--seed-top-k", type=int, default=RECALL_SEED_TOP_K,
                    help=f"How many query entities become recall seeds "
                         f"(production {RECALL_SEED_TOP_K})")
    # The djev adoption arm. Note the honest caveat before reading a result
    # off it: the labelled set is 20 queries / 50 doc labels and the noise
    # floor is 0.02 MRR (agent_mcp/vault.py:172), so this eval can adjudicate
    # a LARGE win or a LARGE loss and nothing in between.
    ap.add_argument("--djev-rerank", dest="djev_rerank", action="store_true",
                    default=RECALL_DJEV_RERANK,
                    help=f"Re-rank the head of the pool through djev "
                         f"(production {RECALL_DJEV_RERANK})")
    ap.add_argument("--djev-rerank-top", type=int, default=RECALL_DJEV_RERANK_TOP,
                    help=f"How many top documents djev re-orders "
                         f"(production {RECALL_DJEV_RERANK_TOP}, ceiling 16)")
    # Measuring the no-graph baseline on purpose is legitimate — it is how the
    # blind spot above was found. Everything else that reaches an empty corpus
    # got there by accident and must not be handed a well-formed score sheet.
    ap.add_argument("--allow-empty-corpus", action="store_true",
                    help="Score even when the fact tree / graph store is empty "
                         "(records corpus_ok: false)")
    ap.add_argument("--no-counterfactual", dest="counterfactual", action="store_false",
                    default=True,
                    help="Skip the perturbed twin of every query (halves the "
                         "run; summary then carries null counterfactual rates)")
    return ap


def build_run_config(args: argparse.Namespace) -> dict:
    """What this run was scored with, and whether that is what production serves.

    One module-level callable rather than an inline expression in `main()` for
    two reasons. It is the only place the parsed CLI meets the production
    constants, so it has to be reachable without a corpus — `main()`'s version
    of this conjunction could only be read off a finished baseline file, which
    means testing the flag costed a retrieval run and no test paid it, which is
    how the conjunction lost the seed count in the first place and stayed
    missing it while every artifact it stamped read `true`.

    `matches_production_defaults` is a conjunction over the parsed values
    against the constants — never one constant against itself, and never a CLI
    flag against itself. `and not args.no_graph` used to sit at the end of it: a
    `store_true` flag compared with its own absence, true on every run that did
    not pass the flag, so it could only ever turn the verdict off and could
    never disagree with production (#1000). Every term below is now a parsed
    value against an imported `RECALL_*` constant.

    `expand_graph` is deliberately NOT a term of that conjunction. Production
    defaults it False while the eval runs it on as a measurement choice, so the
    honest claim is its own field, `expand_graph_matches_production`, computed
    against `RECALL_EXPAND_GRAPH` — false on the default invocation, true only
    for a `--no-graph` run. Folding it into the conjunction instead would label
    every nightly run a non-production configuration, which is a different claim
    from the honest one.

    `semantic_seeding` is a second field of that same shape, for a knob that
    cannot be a conjunction term at all: #1486 (`dbfde750`, 2026-09-25) made the
    seed list the scores are built from union the lexical head with up to `k`
    semantic seeds, and it is read from `retrieval.entity_seeding.semantic`
    through `semantic_seed_k()`, not from argparse and not from a `RECALL_*`
    constant. So there is no parsed value to compare, and a term comparing the
    accessor with itself would be #1000 again. What the artifact owes the reader
    is not a parity verdict but the configuration itself — which is also what
    #843 was missing when the conjunction had no seed count: a night can now be
    asked which seeding defined its `seeds_extracted` instead of being assumed
    to have used today's. The night of 2026-09-25 (seeding off) and the night of
    2026-09-26 (seeding on) both read `matches_production_defaults: true` and
    are not comparable on the entity leg — that is what this field is for.
    """
    expand_graph = not args.no_graph
    return {
        "expand_graph": expand_graph,
        # Computed against production's constant, not against the flag that set
        # it. The default run expands the graph; production does not.
        "expand_graph_matches_production": expand_graph == RECALL_EXPAND_GRAPH,
        "graph_rerank": args.graph_rerank,
        "rerank_alpha": args.alpha,
        "graph_top_k": args.graph_top_k,
        "graph_hops": args.graph_hops,
        # The count the records' `seeds_extracted` were sliced at, in the
        # artifact itself: a baseline can now be asked how many seeds it scored
        # with instead of being assumed to have used the current constant.
        "seed_top_k": args.seed_top_k,
        # The OTHER half of what the seeds were (#1547): `seed_top_k` is the
        # lexical width, and since #1486 the seed list is that head UNIONED with
        # up to `k` semantic seeds, so `seed_top_k: 10` no longer says how many
        # seeds a record holds. Read through production's own accessor, so the
        # artifact and the recall under test cannot report different seedings.
        # A sibling field, never a term of the conjunction below: the
        # conjunction's shape is `args.<knob> == RECALL_<KNOB>`, and this knob is
        # neither parsed nor a constant — `expand_graph_matches_production` above
        # is the precedent for a knob that config owns.
        "semantic_seeding": _semantic_seeding_record(),
        "djev_rerank": args.djev_rerank,
        "djev_rerank_top": args.djev_rerank_top,
        "matches_production_defaults": (
            args.graph_rerank == RECALL_GRAPH_RERANK
            and args.alpha == RECALL_RERANK_ALPHA
            and args.graph_top_k == RECALL_GRAPH_TOP_K
            and args.graph_hops == RECALL_GRAPH_HOPS
            # The term whose absence is #843: 5 against 10 used to read `true`.
            and args.seed_top_k == RECALL_SEED_TOP_K
            # An arm run is NOT production, and the artifact has to say so or
            # a djev-reranked baseline is comparable with a plain one by
            # accident. `djev_rerank_top` is deliberately not a term: it means
            # nothing while the arm is off, and a conjunction that can be
            # falsified by an inert knob is #1000's defect one constant over.
            and args.djev_rerank == RECALL_DJEV_RERANK
        ),
    }


def main() -> int:
    args = build_parser().parse_args()

    spec_file = Path(args.queries)
    spec = yaml.safe_load(spec_file.read_text())
    queries = spec.get("queries") or []
    # #1412: a corpus carrying reserved holdout ids is scored only by the paired
    # check's holdout leg, and never under a label the nightly readers glob. The
    # refusal names counts and the label, never an id.
    refused = holdout.refusal(queries, args.label, os.environ, LLOYD_HOME)
    if refused:
        print(f"[fatal] {refused}", file=sys.stderr)
        return 2
    leg = "holdout" if holdout.is_holdout_corpus(queries, LLOYD_HOME) else "dev"
    print(f"[info] loaded {len(queries)} queries from {spec_file}")

    try:
        corpus = _corpus_provenance()
    except StoreUnavailable as exc:
        # Distinct from the empty case on purpose, and deliberately NOT
        # bypassable by --allow-empty-corpus: "I could not read the store" and
        # "the store is empty" are different facts about the world, and a flag
        # that means the second must never silently excuse the first.
        sys.stdout.flush()   # keep the fatal line after the context it follows
        print(f"[fatal] knowledge-graph store unreadable: {exc}", file=sys.stderr)
        print(f"        kg_db      = {VAULT_KG_DB}", file=sys.stderr)
        print(f"        facts_root = {VAULT_FACTS_ROOT}", file=sys.stderr)
        print("        --allow-empty-corpus does not apply: this store did not "
              "open at all.", file=sys.stderr)
        return 3

    # Two halves, one flag. The fact half is exactly the rule that has always
    # been here — an empty graph scores like a healthy one on the document
    # metrics. The doc half is #1374: a run whose vector count is zero scored
    # every `doc_hit` against an index that could not answer, which is the same
    # indistinguishability one directory over. An *unknown* count is neither:
    # `doc_corpus.doc_ok` answers True for it and the artifact says `doc: null`,
    # so a daemon that is simply down cannot fail a run it never served.
    fact_half_ok = bool(corpus["edges_active"]) and bool(corpus["entities"])
    doc_half_ok = doc_corpus.doc_ok(corpus)
    corpus_ok = fact_half_ok and doc_half_ok
    if not corpus_ok:
        if not args.allow_empty_corpus:
            sys.stdout.flush()
            print("[fatal] empty corpus — refusing to score. An empty graph "
                  "scores identically to a healthy one on mrr_doc, ndcg10 and "
                  "doc_hit_rate, so the result would be indistinguishable from "
                  "a real run.", file=sys.stderr)
            if not fact_half_ok:
                print(f"        facts_root = {VAULT_FACTS_ROOT}  "
                      f"(entity_dirs={corpus['entity_dirs']})", file=sys.stderr)
                print(f"        kg_db      = {VAULT_KG_DB}  "
                      f"(entities={corpus['entities']}, "
                      f"edges_active={corpus['edges_active']})", file=sys.stderr)
            if not doc_half_ok:
                # Named separately, because the two halves need different fixes
                # and the operator reading this decides which one to chase.
                print(f"        doc corpus = vectors=0 on the daemon at "
                      f"{(corpus.get(doc_corpus.DOC_KEY) or {}).get('health_url')} "
                      f"(index={(corpus.get(doc_corpus.DOC_KEY) or {}).get('index_path')}) "
                      "— every doc_hit in this run is an empty index, not a "
                      "retrieval result (#1374).", file=sys.stderr)
            # The half named here is the half the flag is being offered for:
            # "measure the no-graph baseline" is the wrong advice when the graph is
            # intact and the vector index is what came back empty.
            print("        Pass --allow-empty-corpus to measure the "
                  f"{'no-graph' if not fact_half_ok else 'empty-document-corpus'} "
                  "baseline deliberately.", file=sys.stderr)
            return 2
        print("[warn] empty corpus, proceeding under --allow-empty-corpus; "
              "this run records corpus_ok: false")

    records = run_eval(
        queries, limit=args.limit,
        # NOT `== RECALL_EXPAND_GRAPH`: production defaults the knob off, and
        # scoring the eval with the graph closed is a different measurement, not
        # a parity fix. #1000 makes only the claim honest.
        expand_graph=not args.no_graph,
        graph_rerank=args.graph_rerank,
        rerank_alpha=args.alpha,
        demote_factor=args.demote_factor,
        graph_top_k=args.graph_top_k,
        graph_hops=args.graph_hops,
        seed_top_k=args.seed_top_k,
        djev_rerank=args.djev_rerank,
        djev_rerank_top=args.djev_rerank_top,
        counterfactual=args.counterfactual,
    )
    summary = summarize(records)
    # The gold-side ceiling (#654), read HERE and not inside `summarize`: the
    # artifact has to be checked against the labels THIS run was scored against
    # (`labels_sha256`), and `summarize` is a pure function of its records that
    # eleven other callers and tests depend on it remaining. `summarize` has already
    # placed the keys with nulls, so this is a fill, not a reshape — and it happens
    # before `out` is assembled, so every consumer of `summary["overall"]` below
    # (`over_budget`, `print_table`, the written JSON) sees one consistent object
    # rather than two snapshots of it.
    _ceiling_fields = ceiling_context(
        queries,
        # The ids that were actually scored, not the ids that were asked about: an
        # errored query has no score to normalize, so including it would widen the
        # ceiling's denominator past the metric's.
        scored_ids=[r["id"] for r in records if not r.get("error")])
    _normalize_against_ceiling(_ceiling_fields, summary["overall"])
    summary["overall"].update(_ceiling_fields)
    # #1250: a fact leg that read NOTHING is not a fact score of zero. Every
    # per-entity read can fail — `agent_mcp/vault.py:_collect` used to discard
    # each failure with `except Exception: continue`, no log, no counter — and
    # the run then recorded `fact_entity_recall_avg: 0.0` beside `corpus_ok:
    # true` and `errors: 0`, against a corpus naming 315,462 facts. `corpus_ok`
    # could not catch it because it is `bool(edges_active) and bool(entities)`:
    # the graph half only, and `corpus.facts` is the store's index count, not
    # what recall could read. Nulling the metric and flipping `corpus_ok` is
    # what moves it from "a real 0.0 that the paired check turns into a
    # rollback reason for an unrelated commit" to "this arm did not measure".
    # The guard is TOTAL-across-queries on purpose: the healthy arm's per-query
    # fact counts are nineteen 10s and one 1, so a single query at 0 is a
    # legitimate score and must keep its number.
    fact_cov = fact_read_coverage(records)
    fact_leg_vacuous = fact_leg_read_nothing(records, corpus)
    if fact_leg_vacuous:
        summary["overall"]["fact_entity_recall_avg"] = None
        # The interval goes with the rate (#696). A bootstrap over a leg that
        # read nothing returns a tight bracket around zero, and a bracket beside
        # a nulled metric is worse than no bracket: it is a precision claim about
        # a number this run just declared it does not have.
        summary["overall"].setdefault("ci95", {})["fact_entity_recall_avg"] = {
            "ci": [None, None], "n": 0, "kind": "bootstrap",
            "suppressed": "fact leg read nothing on every query (#1260)",
        }
        for cat_summary in summary["by_category"].values():
            cat_summary["fact_entity_recall_avg"] = None
        corpus_ok = False
        sys.stdout.flush()
        print("[warn] the fact leg read NOTHING on every query while the store "
              f"indexes {corpus['facts']} facts — fact_entity_recall_avg is "
              "recorded as null, not 0.0, and this run records corpus_ok: false. "
              "A zeroed fact leg and an empty fact tree are the same reading, "
              "and neither is a measurement of retrieval quality.", file=sys.stderr)
        print(f"        facts_root = {VAULT_FACTS_ROOT}", file=sys.stderr)
        print(f"        kg_db      = {VAULT_KG_DB}", file=sys.stderr)
    # The labelled defect list is a deliverable of #541, not a debug aid: the
    # item's own step 5 is "decide which fix the taxonomy argues for", and that
    # decision needs the labels where the numbers are.
    failures = cf.label_failures(records)

    out = {
        "label": args.label,
        # Which leg of the retrieval eval this run scored (#1412). `holdout` records
        # live only in the paired check's scratch data roots.
        "leg": leg,
        "notes": args.notes,
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "limit": args.limit,
        # The run's retrieval shape and the parity verdict come from
        # `build_run_config`, the one callable where parsed CLI meets production
        # constants — written inline here, the conjunction sat behind a
        # subprocess boundary no test could grade, so nothing pinned it and the
        # seed term was never in it.
        **build_run_config(args),
        "demote_daily_logs": RECALL_DEMOTE_DAILY_LOGS,
        "corpus": corpus,
        "corpus_ok": corpus_ok,
        # What the fact leg read (#1250), recorded on EVERY run — including the
        # healthy ones, where `n_facts_total` is what lets a later "was this arm
        # zeroed?" question be answered without re-running anything. `empty` is
        # the named verdict; the counts are its evidence.
        "fact_leg": {**fact_cov, "empty": fact_leg_vacuous,
                     "facts_in_corpus": int(corpus.get("facts") or 0)},
        # Which frozen record set produced the variants. A selfmod round diffs
        # these two files against each other; recording the path (not just the
        # numbers) is what tells a reader whether two baselines are comparable.
        "counterfactual_records": str(cf.RECORD_PATH),
        "counterfactual_ran": bool(args.counterfactual),
        "counterfactual_failures": failures,
        "identity_keying_evidence": cf.identity_keying_evidence(failures),
        "summary": summary,
        "records": records,
        # The nightly context's ceiling applied to this run's own average
        # (`workers/sources/automod_regression.LATENCY_BUDGET_MS`). A report:
        # `over: true` is a finding to read, never a gate that fails the run —
        # latency is largely qmd's embedding cache, so the paired promotion check
        # compares quality and this compares against a fixed number.
        "latency_budget": over_budget(summary["overall"], CONTEXT_NIGHTLY),
    }
    out_path = EVAL_BASELINES_DIR / f"{args.label}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"[info] wrote {out_path}")
    print(_corpus_line(corpus))

    print_table(records, summary)
    verdict = out["latency_budget"]
    if verdict:
        flag = "OVER BUDGET" if verdict["over"] else "inside budget"
        print(f"\nLatency: {verdict['latency_ms_avg']:,.0f} ms average vs the "
              f"{verdict['context']} budget of {verdict['budget_ms']:,.0f} ms — "
              f"{flag}. Reported, not gated: quality is what the paired promotion "
              f"check compares; this is the absolute ceiling (#1129).")
    print_failures(failures)
    return 0


if __name__ == "__main__":
    sys.exit(main())
