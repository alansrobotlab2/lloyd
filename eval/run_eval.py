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
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
LLOYD_HOME = HERE.parent
sys.path.insert(0, str(LLOYD_HOME))

from agent_mcp.vault import (
    RECALL_DEMOTE_DAILY_LOGS,
    RECALL_GRAPH_HOPS,
    RECALL_GRAPH_RERANK,
    RECALL_GRAPH_TOP_K,
    RECALL_RERANK_ALPHA,
    _vault_recall,
)
from agent_mcp.facts import _extract_entities_from_query
from app.kg_store import StoreUnavailable, store
from app.paths import VAULT_FACTS_ROOT, VAULT_KG_DB
# This file runs both as `python eval/run_eval.py` (script dir on sys.path) and
# as `import eval.run_eval` from the tests; the second form needs the package
# spelling.
try:
    from eval import counterfactual as cf
except ImportError:  # pragma: no cover - script-dir invocation
    import counterfactual as cf


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
    }


def _corpus_line(corpus: dict) -> str:
    return (f"[info] corpus facts_root={corpus['facts_root']} "
            f"entity_dirs={corpus['entity_dirs']} kg_db={corpus['kg_db']} "
            f"entities={corpus['entities']} edges_active={corpus['edges_active']} "
            f"aliases={corpus['aliases']} facts={corpus['facts']}")


def _entities_in_result(result: dict, seeds: list[str] | None = None) -> list[str]:
    """Union of entity signals: seeds extracted from query, fact entities,
    graph_expanded facts entities, and graph_neighbors_used. Order = signal
    strength (seeds first)."""
    seen, out = set(), []
    for e in (seeds or []):
        e = str(e or "").strip()
        if e and e.lower() not in seen:
            seen.add(e.lower()); out.append(e)
    for f in result.get("facts", []) or []:
        e = str(f.get("entity", "") or "").strip()
        if e and e.lower() not in seen:
            seen.add(e.lower()); out.append(e)
    for f in result.get("graph_expanded_facts", []) or []:
        e = str(f.get("entity", "") or "").strip()
        if e and e.lower() not in seen:
            seen.add(e.lower()); out.append(e)
    for n in result.get("graph_neighbors_used", []) or []:
        e = str(n.get("entity", "") or "").strip()
        if e and e.lower() not in seen:
            seen.add(e.lower()); out.append(e)
    return out


def _doc_paths(result: dict) -> list[str]:
    return [str(d.get("path", "")) for d in (result.get("documents") or [])]


def _norm(s: str) -> str:
    """Normalize for matching: lowercase, treat - and _ as equivalent."""
    return str(s or "").lower().replace("-", "_")


def _ndcg_at_k(got_docs: list[str], expected_docs: list[str], k: int = 10) -> float:
    """Binary-relevance NDCG@k. Each got_docs[i] (i<k) scores 1 if it
    matches any expected substring, else 0. IDCG is computed against the
    actual count of relevant docs found in top-k (not len(expected_docs)),
    because expectations are substrings and may each match multiple docs.
    Returns 0.0 when no relevant docs in top-k or no expectations."""
    if not expected_docs:
        return 0.0
    import math as _m
    rel = [1 if any(exp in got for exp in expected_docs) else 0 for got in got_docs[:k]]
    num_rel = sum(rel)
    if num_rel == 0:
        return 0.0
    dcg = sum(r / _m.log2(i + 2) for i, r in enumerate(rel))
    idcg = sum(1.0 / _m.log2(i + 2) for i in range(num_rel))
    return dcg / idcg


def _score(query_spec: dict, result: dict, seeds: list[str] | None = None) -> dict:
    expected_entities = [_norm(e) for e in (query_spec.get("expect_entities") or [])]
    expected_docs = [_norm(d) for d in (query_spec.get("expect_docs") or [])]
    got_entities = [_norm(e) for e in _entities_in_result(result, seeds)]
    got_docs = [_norm(p) for p in _doc_paths(result)]

    entity_matches = [exp for exp in expected_entities if any(exp in got for got in got_entities)]
    doc_matches = [exp for exp in expected_docs if any(exp in got for got in got_docs)]

    # Rank of FIRST matching expected doc in returned list (1-indexed; None if none).
    first_doc_rank = None
    for rank, got in enumerate(got_docs, start=1):
        if any(exp in got for exp in expected_docs):
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
        "docs_matched": doc_matches,
    }


def run_eval(queries: list[dict], limit: int = 20, expand_graph: bool = True,
             graph_rerank: bool = False, rerank_alpha: float = 0.5,
             demote_factor: float | None = None,
             graph_top_k: int = 5, graph_hops: int = 1,
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
            recall_params = {
                "query": query,
                "limit": limit,
                "expand_graph": expand_graph,
                "graph_rerank": graph_rerank,
                "rerank_alpha": rerank_alpha,
                "graph_top_k": graph_top_k,
                "graph_hops": graph_hops,
            }
            if demote_factor is not None:
                recall_params["demote_factor"] = demote_factor
            result = _vault_recall(recall_params)
            err = None
        except Exception as e:
            result = {"documents": [], "facts": []}
            err = f"{type(e).__name__}: {e}"
        latency_ms = (time.perf_counter() - t0) * 1000

        seeds = [e for e, _ in (_extract_entities_from_query(query) or [])[:5]]
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
                "doc_paths_top10": _doc_paths(result)[:10],
                "fact_entities_top10": _entities_in_result(result, seeds)[:10],
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
                                   perturbations.get(qid))
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
                           recall_params: dict, perturbation: dict | None) -> None:
    """Run this query's perturbed twin through the SAME path and score both
    directions onto the record.

    Reusing `recall_params` with only `query` swapped is the whole point: the
    two arms must differ in the one constraint and nothing else, so any
    difference in the entity-level output is attributable to that constraint.
    """
    if not perturbation:
        rec["counterfactual"] = None
        rec["counterfactual_error"] = "no perturbation record for this query id"
        return
    variant_query = perturbation.get("perturbed_query") or ""
    try:
        variant = _vault_recall({**recall_params, "query": variant_query})
        variant_seeds = [e for e, _ in
                         (_extract_entities_from_query(variant_query) or [])[:5]]
        rec["counterfactual"] = cf.score_pair(perturbation, result, seeds,
                                              variant, variant_seeds)
        rec["counterfactual"]["seeds_variant"] = variant_seeds
    except Exception as e:
        # A variant that errored is not a retrieval failure; scoring it would
        # put a retriever bug report on a harness bug.
        rec["counterfactual"] = None
        rec["counterfactual_error"] = f"{type(e).__name__}: {e}"


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

    overall = {
        "n_queries": len(records),
        "entity_hit_rate": avg([1.0 if r["scoring"]["entity_hit"] else 0.0 for r in records]),
        "doc_hit_rate": avg([1.0 if r["scoring"]["doc_hit"] else 0.0 for r in records]),
        "entity_recall_avg": avg([r["scoring"]["entity_recall"] for r in records]),
        "doc_recall_avg": avg([r["scoring"]["doc_recall"] for r in records]),
        "mrr_doc": avg([r["scoring"]["rr_doc"] for r in records]),
        "ndcg10": avg([r["scoring"]["ndcg10"] for r in records]),
        "fact_entity_recall_avg": avg([r["scoring"]["fact_entity_recall"] for r in records]),
        "latency_ms_avg": avg([r["latency_ms"] for r in records]),
        "errors": sum(1 for r in records if r.get("error")),
        "counterfactual_moved_rate": avg([v for v in moved_vals]),
        "counterfactual_pinned_rate": avg([v for v in pinned_vals]),
        "counterfactual_n_moved": len([v for v in moved_vals if v is not None]),
        "counterfactual_n_pinned": len([v for v in pinned_vals if v is not None]),
    }

    per_cat = {}
    for cat, rs in by_cat.items():
        per_cat[cat] = {
            "n": len(rs),
            "entity_hit_rate": avg([1.0 if r["scoring"]["entity_hit"] else 0.0 for r in rs]),
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
    print(f"Overall: n={o['n_queries']}  MRR={o['mrr_doc']:.3f}  NDCG10={o['ndcg10']:.3f}  "
          f"fact_entity_recall={o['fact_entity_recall_avg'] or 0:.3f}")
    print(f"         entity_hit={o['entity_hit_rate']:.2f}  doc_hit={o['doc_hit_rate']:.2f}  "
          f"ent_recall={o['entity_recall_avg']:.2f}  doc_recall={o['doc_recall_avg']:.2f}  "
          f"avg_lat={o['latency_ms_avg']:.0f}ms  errors={o['errors']}")
    # Printed beside entity_hit/doc_hit because that gap is what #541 exists to
    # explain: retrieval finds a relevant document far more reliably than it
    # finds the right entity row.
    mv, pn = o.get("counterfactual_moved_rate"), o.get("counterfactual_pinned_rate")
    print(f"         counterfactual: moved={_fmt_rate(mv)} (n={o.get('counterfactual_n_moved', 0)})  "
          f"pinned={_fmt_rate(pn)} (n={o.get('counterfactual_n_pinned', 0)})")
    print("\nBy category:")
    for cat, s in summary["by_category"].items():
        print(f"  {cat:<10} n={s['n']:<3} entity_hit={s['entity_hit_rate']:.2f}  "
              f"doc_hit={s['doc_hit_rate']:.2f}  ent_recall={s['entity_recall_avg']:.2f}  "
              f"MRR={s['mrr_doc']:.3f}  NDCG10={s['ndcg10']:.3f}  fER={s['fact_entity_recall_avg'] or 0:.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default=str(HERE / "vault_recall_queries.yaml"))
    ap.add_argument("--label", default="baseline", help="Label embedded in the output filename")
    ap.add_argument("--notes", default="", help="Free-text notes saved with the run (e.g. config knobs)")
    ap.add_argument("--limit", type=int, default=20)
    # Defaults ARE production's, imported from agent_mcp.vault. The nightly
    # eval used to run graph_rerank=False, alpha=0.5 against a production
    # that runs True and 0.3 — it measured a configuration nothing serves,
    # so a retrieval regression could not show up in it (2026-09-03 review).
    ap.add_argument("--no-graph", action="store_true", help="Disable expand_graph (default: on)")
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
    args = ap.parse_args()

    spec_file = Path(args.queries)
    spec = yaml.safe_load(spec_file.read_text())
    queries = spec.get("queries") or []
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

    corpus_ok = bool(corpus["edges_active"]) and bool(corpus["entities"])
    if not corpus_ok:
        if not args.allow_empty_corpus:
            sys.stdout.flush()
            print("[fatal] empty corpus — refusing to score. An empty graph "
                  "scores identically to a healthy one on mrr_doc, ndcg10 and "
                  "doc_hit_rate, so the result would be indistinguishable from "
                  "a real run.", file=sys.stderr)
            print(f"        facts_root = {VAULT_FACTS_ROOT}  "
                  f"(entity_dirs={corpus['entity_dirs']})", file=sys.stderr)
            print(f"        kg_db      = {VAULT_KG_DB}  "
                  f"(entities={corpus['entities']}, "
                  f"edges_active={corpus['edges_active']})", file=sys.stderr)
            print("        Pass --allow-empty-corpus to measure the no-graph "
                  "baseline deliberately.", file=sys.stderr)
            return 2
        print("[warn] empty corpus, proceeding under --allow-empty-corpus; "
              "this run records corpus_ok: false")

    records = run_eval(
        queries, limit=args.limit,
        expand_graph=not args.no_graph,
        graph_rerank=args.graph_rerank,
        rerank_alpha=args.alpha,
        demote_factor=args.demote_factor,
        graph_top_k=args.graph_top_k,
        graph_hops=args.graph_hops,
        counterfactual=args.counterfactual,
    )
    summary = summarize(records)
    # The labelled defect list is a deliverable of #541, not a debug aid: the
    # item's own step 5 is "decide which fix the taxonomy argues for", and that
    # decision needs the labels where the numbers are.
    failures = cf.label_failures(records)

    out = {
        "label": args.label,
        "notes": args.notes,
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "limit": args.limit,
        "expand_graph": not args.no_graph,
        "graph_rerank": args.graph_rerank,
        "rerank_alpha": args.alpha,
        "graph_top_k": args.graph_top_k,
        "graph_hops": args.graph_hops,
        "demote_daily_logs": RECALL_DEMOTE_DAILY_LOGS,
        "matches_production_defaults": (
            args.graph_rerank == RECALL_GRAPH_RERANK
            and args.alpha == RECALL_RERANK_ALPHA
            and args.graph_top_k == RECALL_GRAPH_TOP_K
            and args.graph_hops == RECALL_GRAPH_HOPS
            and not args.no_graph
        ),
        "corpus": corpus,
        "corpus_ok": corpus_ok,
        # Which frozen record set produced the variants. A selfmod round diffs
        # these two files against each other; recording the path (not just the
        # numbers) is what tells a reader whether two baselines are comparable.
        "counterfactual_records": str(cf.RECORD_PATH),
        "counterfactual_ran": bool(args.counterfactual),
        "counterfactual_failures": failures,
        "identity_keying_evidence": cf.identity_keying_evidence(failures),
        "summary": summary,
        "records": records,
    }
    out_path = HERE / "baselines" / f"{args.label}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"[info] wrote {out_path}")
    print(_corpus_line(corpus))

    print_table(records, summary)
    print_failures(failures)
    return 0


if __name__ == "__main__":
    sys.exit(main())
