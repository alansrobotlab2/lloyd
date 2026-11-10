#!/usr/bin/env python3
"""eval/run_vault_search_eval.py — the `vault_search` tool's document quality, per pipeline.

Every recall measurement in `architecture/retrieval.md` §3.1/§3.2 was taken on
`vault_recall`, and the arm it adopted (OR keywords, djev ordering) never
reached `vault_search` — the tool whose description every prose question is
routed to. This scores `vault_search` on the same labelled corpus with the same
scorer (`run_eval._score`, documents only: the tool returns no facts), under
either document pipeline:

    legacy   AND keywords + qmd cross-encoder — what vault_search runs today
    recall   vault_recall's document leg (OR, djev, floors) without facts

Usage (pause the worker pool first if djev or the primary is in use —
`round restart` notes; this reads the LIVE qmd daemon and djev):

    .venvs/lloyd/bin/python eval/run_vault_search_eval.py --pipeline legacy
    .venvs/lloyd/bin/python eval/run_vault_search_eval.py --pipeline recall

`--compare` runs both, alternating per query so neither arm owns a warm cache,
and prints the paired difference. `consolidate` is off in every arm: it adds a
model call and changes no document rank.

The artifact goes to `app.paths.EVAL_VAULT_SEARCH_DIR` (`$DATA_ROOT/eval/vault-search/`).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from agent_mcp import vault  # noqa: E402
from app.paths import EVAL_VAULT_SEARCH_DIR  # noqa: E402
from eval.run_eval import _score  # noqa: E402

DOC_METRICS = ("doc_hit", "rr_doc", "ndcg10", "doc_recall")


def run_one(spec: dict, pipeline: str, limit: int) -> dict:
    query = spec.get("query", "")
    t0 = time.perf_counter()
    err = None
    try:
        out = vault._run_vault_search(query, limit, 0.0, "", False, pipeline=pipeline)
        docs = [{"path": r["path"]} for r in out.get("results") or []]
    except Exception as e:  # noqa: BLE001 — an outage is an error row, never a zero
        docs, err = [], f"{type(e).__name__}: {e}"
    ms = (time.perf_counter() - t0) * 1000
    scoring = _score(spec, {"documents": docs, "facts": []}, seeds=[])
    return {"id": spec.get("id"), "pipeline": pipeline, "error": err,
            "latency_ms": round(ms, 1), "doc_paths_top10": [d["path"] for d in docs[:10]],
            **{k: scoring.get(k) for k in DOC_METRICS}}


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r["error"]]
    out = {"n": len(rows), "errors": len(rows) - len(ok)}
    for k in DOC_METRICS:
        vals = [float(r[k]) for r in ok if r.get(k) is not None]
        out[k] = round(statistics.fmean(vals), 4) if vals else None
    lat = sorted(r["latency_ms"] for r in ok)
    out["latency_p50_ms"] = lat[len(lat) // 2] if lat else None
    return out


def paired(rows: list[dict], a: str, b: str) -> dict:
    by = {}
    for r in rows:
        by.setdefault(r["id"], {})[r["pipeline"]] = r
    out = {}
    for k in DOC_METRICS:
        diffs = [float(p[b][k]) - float(p[a][k]) for p in by.values()
                 if a in p and b in p and not p[a]["error"] and not p[b]["error"]
                 and p[a].get(k) is not None and p[b].get(k) is not None]
        out[k] = {"n": len(diffs), "mean_diff": round(statistics.fmean(diffs), 4) if diffs else None,
                  "wins": sum(d > 0 for d in diffs), "losses": sum(d < 0 for d in diffs)}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--queries", default=str(HERE / "vault_recall_queries.yaml"))
    ap.add_argument("--pipeline", choices=vault.VAULT_SEARCH_PIPELINES, default="legacy")
    ap.add_argument("--compare", action="store_true", help="run both pipelines, paired")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    queries = (yaml.safe_load(Path(args.queries).read_text()) or {}).get("queries") or []
    queries = [q for q in queries if q.get("query") and q.get("expect_docs")]
    arms = list(vault.VAULT_SEARCH_PIPELINES) if args.compare else [args.pipeline]
    rows: list[dict] = []
    for i, spec in enumerate(queries):
        order = arms if i % 2 == 0 else list(reversed(arms))
        for arm in order:
            rows.append(run_one(spec, arm, args.limit))

    report = {"measured_at": datetime.now(timezone.utc).isoformat(),
              "queries": args.queries, "limit": args.limit,
              "summary": {arm: summarize([r for r in rows if r["pipeline"] == arm]) for arm in arms},
              "rows": rows}
    if args.compare:
        report["paired_recall_minus_legacy"] = paired(rows, "legacy", "recall")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) if args.out else EVAL_VAULT_SEARCH_DIR / f"{'-'.join(arms)}-{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    print(f"[info] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
