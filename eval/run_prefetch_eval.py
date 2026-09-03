#!/usr/bin/env python3
"""Prefetch-path retrieval eval.

`run_eval.py` measures the explicit `vault_recall` tool (reranking on, no
budget). This measures the vault leg of prefetch *as deployed*:

  lex     — `_search_vault_lex`: short AND sub-queries with a drop-a-term
            ladder, no rerank. This is what lands inside the 300 ms budget.
  hybrid  — `_search_vault(..., legs=("lex","vec"))`: the straggler whose
            result is carried over to the *next* turn.
  merged  — `_merge_vault_results(lex, hybrid)`: what a second turn on the
            same topic sees (its own lex hits + the carried semantic hits).

Each query is scored against the `expect_docs` substrings in
`vault_recall_queries.yaml` (paths relative to ~/obsidian). Entity
expectations are ignored — the facts worker is a separate path.

The queries are run as a *first turn* (a fresh SessionFocus updated with the
query), which is the hardest case for the lex leg: no prior-turn context.

Usage:
    .venvs/lloyd/bin/python eval/run_prefetch_eval.py
    .venvs/lloyd/bin/python eval/run_prefetch_eval.py --label after-ladder --notes "..."
    .venvs/lloyd/bin/python eval/run_prefetch_eval.py --skip-hybrid   # lex only, ~5 s
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
LLOYD_HOME = HERE.parent
sys.path.insert(0, str(LLOYD_HOME))

logging.disable(logging.CRITICAL)

import prefetch  # noqa: E402


def _norm(s: str) -> str:
    return str(s or "").lower().replace("-", "_")


def _path(r: dict) -> str:
    f = str(r.get("file") or "")
    return f[len("qmd://"):] if f.startswith("qmd://") else f


def _score(expected_docs: list[str], results: list[dict]) -> dict:
    exp = [_norm(d) for d in expected_docs]
    got = [_norm(_path(r)) for r in results]
    first_rank = None
    for rank, g in enumerate(got, start=1):
        if any(e in g for e in exp):
            first_rank = rank
            break
    matched = [e for e in exp if any(e in g for g in got)]
    return {
        "doc_hit": bool(matched),
        "first_doc_rank": first_rank,
        "rr_doc": round(1.0 / first_rank, 4) if first_rank else 0.0,
        "doc_recall": round(len(matched) / len(exp), 4) if exp else None,
        "docs_matched": matched,
        "n_results": len(results),
    }


def run(queries: list[dict], skip_hybrid: bool = False) -> list[dict]:
    records = []
    for spec in queries:
        query = spec.get("query", "")
        if not query:
            continue
        focus = prefetch.SessionFocus()
        focus.update(query)

        t0 = time.perf_counter()
        lex = prefetch._search_vault_lex(query, focus, deadline=None)
        lex_ms = (time.perf_counter() - t0) * 1000

        hybrid: list[dict] = []
        hybrid_ms = None
        if not skip_hybrid:
            t0 = time.perf_counter()
            hybrid = prefetch._search_vault_hybrid_and_stash(query, focus)
            hybrid_ms = (time.perf_counter() - t0) * 1000

        merged = prefetch._merge_vault_results(lex, hybrid)
        exp = spec.get("expect_docs") or []
        records.append({
            "id": spec.get("id"),
            "query": query,
            "category": spec.get("category"),
            "expected_docs": exp,
            "lex_subqueries": [" ".join(q) for q in focus.lex_subqueries(query)],
            "lex": {"latency_ms": round(lex_ms, 1),
                    "landed": lex_ms < prefetch.PREFETCH_BUDGET_MS,
                    "paths": [_path(r) for r in lex], **_score(exp, lex)},
            "hybrid": None if skip_hybrid else {
                "latency_ms": round(hybrid_ms, 1),
                "paths": [_path(r) for r in hybrid], **_score(exp, hybrid)},
            "merged": {"paths": [_path(r) for r in merged], **_score(exp, merged)},
        })
    return records


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


def summarize(records: list[dict]) -> dict:
    n = len(records)
    legs = ["lex", "merged"] + (["hybrid"] if records and records[0].get("hybrid") else [])
    out: dict = {"n": n}
    for leg in legs:
        rs = [r[leg] for r in records]
        out[f"{leg}_doc_hit"] = round(sum(1 for r in rs if r["doc_hit"]) / n, 3) if n else 0.0
        out[f"{leg}_mrr"] = round(sum(r["rr_doc"] for r in rs) / n, 3) if n else 0.0
        lat = [r["latency_ms"] for r in rs if "latency_ms" in r]
        if lat:
            out[f"{leg}_p50_ms"] = round(statistics.median(lat), 1)
            out[f"{leg}_p90_ms"] = round(_pct(lat, 0.9), 1)
    out["lex_landed_rate"] = round(sum(1 for r in records if r["lex"]["landed"]) / n, 3) if n else 0.0
    by_cat: dict[str, list] = defaultdict(list)
    for r in records:
        by_cat[str(r.get("category"))].append(r)
    out["by_category"] = {
        cat: {leg: round(sum(1 for r in rs if r[leg]["doc_hit"]) / len(rs), 3) for leg in legs}
        for cat, rs in sorted(by_cat.items())
    }
    return out


def print_table(records: list[dict], summary: dict) -> None:
    has_h = bool(records and records[0].get("hybrid"))
    hdr = f"{'id':28} {'cat':10} {'lex':>4} {'lex ms':>7} " + (f"{'hyb':>4} {'hyb ms':>7} " if has_h else "") + f"{'mrg':>4}  first lex sub-query"
    print(hdr); print("-" * len(hdr))
    for r in records:
        row = f"{str(r['id'])[:28]:28} {str(r['category'])[:10]:10} {'Y' if r['lex']['doc_hit'] else '.':>4} {r['lex']['latency_ms']:>7.0f} "
        if has_h:
            row += f"{'Y' if r['hybrid']['doc_hit'] else '.':>4} {r['hybrid']['latency_ms']:>7.0f} "
        row += f"{'Y' if r['merged']['doc_hit'] else '.':>4}  {(r['lex_subqueries'] or [''])[0][:40]}"
        print(row)
    print()
    for k, v in summary.items():
        if k != "by_category":
            print(f"  {k:18} {v}")
    print("  by_category:")
    for cat, v in summary["by_category"].items():
        print(f"    {cat:12} {v}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default=str(HERE / "vault_recall_queries.yaml"))
    ap.add_argument("--label", default="prefetch-baseline")
    ap.add_argument("--notes", default="")
    ap.add_argument("--skip-hybrid", action="store_true", help="Lex leg only (no 1-3 s vec calls)")
    args = ap.parse_args()

    spec = yaml.safe_load(Path(args.queries).read_text())
    queries = spec.get("queries") or []
    print(f"[info] loaded {len(queries)} queries; budget={prefetch.PREFETCH_BUDGET_MS}ms "
          f"min_score={prefetch.VAULT_MIN_SCORE} lex_max_terms={prefetch.VAULT_LEX_MAX_TERMS}")
    records = run(queries, skip_hybrid=args.skip_hybrid)
    summary = summarize(records)
    out = {
        "label": args.label, "notes": args.notes,
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "budget_ms": prefetch.PREFETCH_BUDGET_MS,
        "vault_min_score": prefetch.VAULT_MIN_SCORE,
        "lex_max_terms": prefetch.VAULT_LEX_MAX_TERMS,
        "lex_max_calls": prefetch.VAULT_LEX_MAX_CALLS,
        "summary": summary, "records": records,
    }
    out_path = HERE / "baselines" / f"{args.label}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"[info] wrote {out_path}")
    print_table(records, summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
