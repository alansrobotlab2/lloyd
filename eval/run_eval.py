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

from agent_mcp.vault import _vault_recall
from agent_mcp.facts import _extract_entities_from_query


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
             demote_factor: float | None = None) -> list[dict]:
    records = []
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

        records.append({
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
        })
    return records


def summarize(records: list[dict]) -> dict:
    by_cat = defaultdict(list)
    for r in records:
        by_cat[r.get("category") or "?"].append(r)

    def avg(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 3) if xs else None

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
        }
    return {"overall": overall, "by_category": per_cat}


def print_table(records: list[dict], summary: dict) -> None:
    print(f"\n{'id':<26} {'cat':<10} {'eH':<4} {'dH':<4} {'eR':<6} {'dR':<6} {'rank':<5} {'NDCG':<6} {'fER':<6} {'latms':<7}")
    print("-" * 100)
    for r in records:
        s = r["scoring"]
        eh = "✓" if s["entity_hit"] else "✗"
        dh = "✓" if s["doc_hit"] else "✗"
        er = f"{s['entity_recall']:.2f}" if s["entity_recall"] is not None else "—"
        dr = f"{s['doc_recall']:.2f}" if s["doc_recall"] is not None else "—"
        rk = str(s["first_doc_rank"]) if s["first_doc_rank"] else "—"
        ndcg = f"{s['ndcg10']:.2f}"
        fer = f"{s['fact_entity_recall']:.2f}" if s["fact_entity_recall"] is not None else "—"
        print(f"{r['id']:<26} {(r['category'] or ''):<10} {eh:<4} {dh:<4} {er:<6} {dr:<6} {rk:<5} {ndcg:<6} {fer:<6} {r['latency_ms']:<7.0f}")
    print()
    o = summary["overall"]
    print(f"Overall: n={o['n_queries']}  entity_hit={o['entity_hit_rate']:.2f}  doc_hit={o['doc_hit_rate']:.2f}  "
          f"ent_recall={o['entity_recall_avg']:.2f}  doc_recall={o['doc_recall_avg']:.2f}  "
          f"MRR={o['mrr_doc']:.3f}  NDCG10={o['ndcg10']:.3f}  fER={o['fact_entity_recall_avg'] or 0:.3f}  "
          f"avg_lat={o['latency_ms_avg']:.0f}ms")
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
    ap.add_argument("--no-graph", action="store_true", help="Disable expand_graph (default: on)")
    ap.add_argument("--graph-rerank", action="store_true", help="Enable Phase 3 graph-vote re-ranking")
    ap.add_argument("--alpha", type=float, default=0.5, help="Re-rank alpha: 1.0=pure QMD, 0.0=pure graph (default 0.5)")
    ap.add_argument("--demote-factor", type=float, default=None, help="Daily-log demote factor (default uses module constant 0.4)")
    args = ap.parse_args()

    spec_file = Path(args.queries)
    spec = yaml.safe_load(spec_file.read_text())
    queries = spec.get("queries") or []
    print(f"[info] loaded {len(queries)} queries from {spec_file}")

    records = run_eval(
        queries, limit=args.limit,
        expand_graph=not args.no_graph,
        graph_rerank=args.graph_rerank,
        rerank_alpha=args.alpha,
        demote_factor=args.demote_factor,
    )
    summary = summarize(records)

    out = {
        "label": args.label,
        "notes": args.notes,
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "limit": args.limit,
        "expand_graph": not args.no_graph,
        "graph_rerank": args.graph_rerank,
        "rerank_alpha": args.alpha,
        "summary": summary,
        "records": records,
    }
    out_path = HERE / "baselines" / f"{args.label}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"[info] wrote {out_path}")

    print_table(records, summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
