#!/usr/bin/env python3
"""Measure whether a fact-side write is a correction or an over-reach, on a snapshot.

Backlog #874 clause 11. `fact_entity_recall` scores *which entities won slots*, so
it is flat when a loop retires one side of a near-duplicate (the survivor still
carries the claim) and drops when a write over-reaches and ejects an entity from
the ten-slot pool (`agent_mcp.retrieval.FACT_RANK_CAP_SEED`) — the 09-08 pass's
0.35 → 0.30 was that drop. `eval/run_eval.py::count_overreach_regressions` is the
number that separates the two; this script is the measurement that proves it,
against a built-in fixture on a **snapshot copy**, never the live tree.

The fixture — three entities in a fact tree this script creates. Every score below
is (query tokens present in the fact's text) / 4, which is what `_rank` ranks on
(`fact_score`), over a pool capped at 10 across all seeded entities
(`FACT_RANK_CAP_SEED`):

  Query 1 "Snapshot Alpha indexer throughput" expects `Snapshot Alpha Indexer`.
  That query seeds two directories:
    Snapshot Alpha Indexer  `state`: one fact naming all 4 tokens (1.0) — the head.
                            `usage`: one fact naming 2 of them (0.5).
    Snapshot Alpha          10 facts naming 3 of the 4 tokens (0.75), not expected.
  Twelve candidates, cap 10: the pool takes the head plus nine 0.75 rivals, so the
  expected entity answers **only** through its head fact. Expire that one fact and
  the cap backfills with rivals and the 0.5 straggler — the expected entity is gone
  from the answer.

  The expected entity is deliberately the *longer* name: the eval's fact matcher is
  substring-based (`exp in fe`, `eval/run_eval.py`), so an entity named
  `Snapshot Alpha` would still count as matched by a rival named
  `Snapshot Alpha Indexer`. Naming the expectation the superstring is what makes
  eviction observable rather than laundered by the matcher.

  Query 2 "Snapshot Beta queue port" expects `Snapshot Beta`, whose `state` holds
  two near-duplicates: `_token_overlap` is 0.667, over the loop's own 0.6 gate,
  differing only in the port they name. Expiring the older twin leaves the
  survivor, which answers the query just as well.

Two writes then run through the real `_fact_invalidate`, one fact each:

  Alpha  expires the top-scoring fact           → the entity loses its slot: OVER-REACH
  Beta   expires the older twin of the pair     → the survivor still answers: zero

Both arms must hold for the number to be usable as a gate: nonzero for the first,
zero for the second. `fact_entity_recall` over the same two states is printed
alongside: it cannot see the second write at all, and for the first it reports an
average over a query set rather than the entity that vanished — 0.05 of a 20-query
suite is what one ejected entity is worth there.

    .venvs/lloyd/bin/python scripts/memory/measure_overreach_snapshot.py --json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Where `app.paths` points by default. Refusing it is the only thing standing
# between "measure on a copy" and "expire facts on the live corpus".
LIVE_FACTS_ROOT = REPO / "_pipeline" / "vault-derived" / "facts"

# Alpha's query and the four tokens `fact_query_tokens` extracts from it; every
# score below is (tokens present in the fact's text) / 4.
ALPHA_QUERY = "Snapshot Alpha indexer throughput"
BETA_QUERY = "Snapshot Beta queue port"
# The expected name is the one whose answer is a single fact — `Snapshot Alpha` the
# entity is the rival that backfills the pool, so expecting it could never show an
# eviction.
QUERIES = [
    {"id": "alpha", "query": ALPHA_QUERY, "expect_entities": ["Snapshot Alpha Indexer"],
     "category": "snapshot"},
    {"id": "beta", "query": BETA_QUERY, "expect_entities": ["Snapshot Beta"],
     "category": "snapshot"},
]

# The expected entity's two files deliberately share the id `fact-001`, like 104 of
# the first 400 live entity dirs: the over-reach write names ONE fact and must mark
# exactly one, which is #874's identity fix exercised through the real store.
_FIXTURE: dict[str, dict[str, list[tuple[str, float, str]]]] = {
    "Snapshot Alpha Indexer": {
        # 4 of the 4 query tokens → 1.0, the head of the pool.
        "state": [("Snapshot Alpha indexer throughput is 40 rows per second.", 1.0,
                   "fact-001")],
        # 2 of 4 → 0.5 (its blob is the text + the category word), so it is 11th
        # under the cap and gets cut the moment the head is gone.
        "usage": [("Snapshot Alpha was retired from the nightly board.", 0.6,
                   "fact-001")],
    },
    # Seeded by query 1 (its name appears in it) but NOT expected by it: 10 facts at
    # 3 of 4 tokens → 0.75, enough to fill the ten-slot pool once the head is gone.
    # Its name is a SUBSTRING of the expected entity's, never a superstring — the
    # eval's matcher asks `exp in fe`, so these facts can never stand in for the
    # entity the query expects.
    "Snapshot Alpha": {
        "state": [(f"Snapshot indexer throughput baseline for shard {i}.", 0.8,
                   f"fact-{i:03d}") for i in range(1, 11)],
    },
    "Snapshot Beta": {
        "state": [("Snapshot Beta serves the queue on port 8090.", 0.9, "fact-001"),
                  ("Snapshot Beta serves the queue on port 8091.", 0.9, "fact-002")],
    },
}

# The two writes. Both substrings aim at exactly one stored fact.
OVERREACH = ("Snapshot Alpha Indexer", "40 rows per second")
NEAR_DUPLICATE = ("Snapshot Beta", "port 8090")


def build_fixture(root: Path) -> int:
    """Write the fixture tree; returns the number of facts written."""
    import yaml

    n = 0
    for entity, by_category in _FIXTURE.items():
        ed = root / entity
        ed.mkdir(parents=True, exist_ok=True)
        for category, facts in by_category.items():
            prepared = [{"fact": text, "confidence": conf, "category": category,
                         "id": fid, "created_at": "2026-09-01T00:00:00+00:00",
                         "valid_at": "2026-09-01", "expired_at": None,
                         "invalid_at": None, "provenance": "STATED",
                         "source_doc": None}
                        for text, conf, fid in facts]
            fm = {"type": "facts", "entity": entity, "category": category,
                  "facts": prepared}
            (ed / f"{entity}-{category}.md").write_text(
                f"---\n{yaml.dump(fm, sort_keys=False)}---\n\n# {entity}\n",
                encoding="utf-8")
            n += len(prepared)
    return n


def measure(facts_root: Path, kg_db: Path) -> dict:
    """Score the queries, run the two writes, score again. Same process, same tree."""
    from agent_mcp import vault

    # The doc leg is a qmd daemon call over the live vault. An empty list is that
    # function's genuine zero-hit answer, not an outage, and the clause is about
    # the fact leg; graph rerank is off so the ten-slot pool cap is the only
    # thing deciding which facts answer a query.
    vault._qmd_daemon_search = lambda *a, **k: []

    spec = importlib.util.spec_from_file_location("run_eval", REPO / "eval" / "run_eval.py")
    run_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_eval)
    from agent_mcp.facts import _fact_invalidate

    def _scored() -> list[dict]:
        return run_eval.run_eval(QUERIES, limit=5, expand_graph=False,
                                 graph_rerank=False, counterfactual=False)

    def _expired(entity: str, substr: str) -> int:
        return _fact_invalidate({"entity": entity, "fact_substring": substr,
                                 "ended": "2026-09-19"}).get("expired_count", -1)

    before = _scored()
    same_run = run_eval.count_overreach_regressions(before, before)
    n_overreach = _expired(*OVERREACH)
    n_near_dup = _expired(*NEAR_DUPLICATE)
    after = _scored()
    return {
        "facts_root": str(facts_root),
        "same_run_regressions": same_run,
        "expired_by_overreach_write": n_overreach,
        "expired_by_near_duplicate_write": n_near_dup,
        "regressions_after_writes": run_eval.count_overreach_regressions(before, after),
        "matched_entities_before": {r["query"]: r["scoring"]["fact_entities_matched"]
                                    for r in before},
        "matched_entities_after": {r["query"]: r["scoring"]["fact_entities_matched"]
                                   for r in after},
        "fact_entity_recall_before": {r["query"]: r["scoring"]["fact_entity_recall"]
                                      for r in before},
        "fact_entity_recall_after": {r["query"]: r["scoring"]["fact_entity_recall"]
                                     for r in after},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--facts-root", default=None,
                    help="fact tree to use (default: a fresh temp dir holding the fixture)")
    ap.add_argument("--kg-db", default=None, help="store path to use with it")
    ap.add_argument("--json", action="store_true", help="print one JSON object")
    args = ap.parse_args(argv)

    tmp = None
    if args.facts_root:
        facts_root = Path(args.facts_root).resolve()
    else:
        tmp = tempfile.mkdtemp(prefix="overreach-snapshot-")
        facts_root = Path(tmp) / "facts"
        facts_root.mkdir(parents=True)
        build_fixture(facts_root)
    if facts_root == LIVE_FACTS_ROOT.resolve():
        print(f"refusing to run against the live fact tree: {facts_root}",
              file=sys.stderr)
        return 2
    kg_db = Path(args.kg_db) if args.kg_db else facts_root.parent / "kg.sqlite"

    # Set before anything imports `app.paths`, which reads these at import time.
    os.environ["LLOYD_FACTS_ROOT"] = str(facts_root)
    os.environ["LLOYD_KG_DB"] = str(kg_db)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

    result = measure(facts_root, kg_db)
    result["fixture_facts"] = sum(len(fs) for cs in _FIXTURE.values() for fs in cs.values())
    if args.json:
        print(json.dumps(result))
        return 0
    print(f"snapshot: {facts_root} ({result['fixture_facts']} fixture facts)")
    print(f"  over-reach write expired {result['expired_by_overreach_write']} fact(s); "
          f"near-duplicate write expired {result['expired_by_near_duplicate_write']}")
    print(f"  same-run regressions (must be 0): {len(result['same_run_regressions'])}")
    print(f"  regressions after both writes (must be 1, Alpha): "
          f"{json.dumps(result['regressions_after_writes'])}")
    print(f"  matched before: {result['matched_entities_before']}")
    print(f"  matched after:  {result['matched_entities_after']}")
    print(f"  fact_entity_recall before: {result['fact_entity_recall_before']}")
    print(f"  fact_entity_recall after:  {result['fact_entity_recall_after']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
