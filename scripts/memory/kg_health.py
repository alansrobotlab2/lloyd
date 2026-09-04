#!/usr/bin/env python3
"""
Knowledge-Graph Health — Phase 0 instrumentation for backlog #380.

Emits a single JSON snapshot of fact-graph structural health so the
densification phases have a measurable before/after. Read-only: touches
nothing under FACTS_ROOT.

The headline metric is `node_coverage_pct` — the share of entity directories
that appear in at least one live edge. At the #380 baseline that was 0.46%
(294 of 63,860), which is why multi-hop traversal has nothing to walk.

`latent_relationship_entities` is the actionable gap: entities carrying a
`-relationship.md` fact file whose relations were never promoted into the
edge store. Counted as relationship-category facts whose `source_doc` no
edge cites — the old `rel_files - nodes_with_edges` subtraction compared two
unrelated populations and could read negative.

NOTE ON METRIC USE (#380 anti-Goodhart guard): edge_count and node_coverage_pct
are diagnostics, NOT success criteria. Success for #380 is eval MRR — multi-hop
in particular. Edge count is trivially inflatable; never optimise it directly.

Usage:
  python scripts/memory/kg_health.py                 # human-readable summary
  python scripts/memory/kg_health.py --json          # raw JSON to stdout
  python scripts/memory/kg_health.py --json -o FILE  # write snapshot to FILE
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

# Ensure app/ is importable when running this script standalone
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.paths import VAULT_FACTS_ROOT  # noqa: E402
from app.kg_store import StoreUnavailable, store  # noqa: E402


# ── Collection ───────────────────────────────────────────────────────────────


def scan_entities(root: Path) -> tuple[list[str], dict[str, int], int]:
    """Walk FACTS_ROOT once.

    Returns (entity_dir_names, category_counts, stray_file_count).

    Only directories are entities; plain files at the root (backup .bak files,
    entity-aliases.json, _relationships.json) are counted separately. Conflating
    the two inflates the entity count — the raw `ls | wc -l` at the #380 baseline
    read 64,862 against a true 63,860.
    """
    entities: list[str] = []
    categories: collections.Counter[str] = collections.Counter()
    stray_files = 0

    for entry in root.iterdir():
        if not entry.is_dir():
            stray_files += 1
            continue
        entities.append(entry.name)
        for f in entry.iterdir():
            if f.suffix != ".md":
                continue
            # Fact files are <Entity>-<category>.md
            categories[f.stem.rsplit("-", 1)[-1]] += 1

    return entities, dict(categories), stray_files


def load_edges() -> list[dict[str, Any]]:
    """Live (non-expired) edges from the store."""
    return store().edges.active()


# ── Graph structure ──────────────────────────────────────────────────────────


def connected_components(edges: list[dict[str, Any]]) -> list[int]:
    """Component sizes over the undirected projection, largest first.

    Undirected on purpose: this measures whether the graph is *reachable*, which
    is what multi-hop retrieval depends on. Direction matters for `fact_impact`
    (Phase 4), not for connectivity.
    """
    adj: dict[str, set[str]] = collections.defaultdict(set)
    for e in edges:
        src, tgt = e.get("source"), e.get("target")
        if not src or not tgt:
            continue
        adj[src].add(tgt)
        adj[tgt].add(src)

    seen: set[str] = set()
    sizes: list[int] = []
    for node in adj:
        if node in seen:
            continue
        # Iterative flood-fill — recursion would blow the stack on hub nodes.
        stack, size = [node], 0
        seen.add(node)
        while stack:
            cur = stack.pop()
            size += 1
            for nb in adj[cur]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        sizes.append(size)

    return sorted(sizes, reverse=True)


def degree_buckets(edges: list[dict[str, Any]]) -> dict[str, int]:
    """Bucketed degree distribution over nodes that have at least one edge."""
    deg: collections.Counter[str] = collections.Counter()
    for e in edges:
        src, tgt = e.get("source"), e.get("target")
        if not src or not tgt:
            continue
        deg[src] += 1
        deg[tgt] += 1

    buckets: collections.Counter[str] = collections.Counter()
    for d in deg.values():
        if d == 1:
            buckets["1"] += 1
        elif d <= 3:
            buckets["2-3"] += 1
        elif d <= 10:
            buckets["4-10"] += 1
        elif d <= 50:
            buckets["11-50"] += 1
        else:
            buckets["50+"] += 1
    return dict(buckets)


# ── Snapshot ─────────────────────────────────────────────────────────────────


def build_snapshot() -> dict[str, Any]:
    root = VAULT_FACTS_ROOT
    if not root.exists():
        raise SystemExit(f"facts root not found: {root}")

    entities, categories, stray_files = scan_entities(root)
    edges = load_edges()

    graph_nodes = {
        n
        for e in edges
        for n in (e.get("source"), e.get("target"))
        if n
    }
    components = connected_components(edges)

    st = store()
    alias_count = st.aliases.count()

    entity_count = len(entities)
    rel_files = categories.get("relationship", 0)

    # Entity names carrying 5+ words — the fragment-shaped tail that Phase 1
    # targets. Not itself a junk verdict; multi-word concepts are legitimate.
    long_names = sum(1 for e in entities if len(e.split()) >= 5)

    return {
        "captured_at": dt.datetime.now().isoformat(),
        "facts_root": str(root),
        "entities": {
            "count": entity_count,
            "stray_root_files": stray_files,
            "with_relationship_file": rel_files,
            "names_5plus_words": long_names,
            "names_5plus_words_pct": _pct(long_names, entity_count),
        },
        "edges": {
            "count": len(edges),
            "by_type": dict(
                collections.Counter(e.get("type") for e in edges).most_common()
            ),
            "by_provenance": dict(
                collections.Counter(e.get("provenance") for e in edges).most_common()
            ),
        },
        "graph": {
            "nodes_with_edges": len(graph_nodes),
            "node_coverage_pct": _pct(len(graph_nodes), entity_count),
            "component_count": len(components),
            "largest_component": components[0] if components else 0,
            "isolated_entities": entity_count - len(graph_nodes),
            "degree_distribution": degree_buckets(edges),
        },
        "aliases": {
            "count": alias_count,
            "coverage_pct": _pct(alias_count, entity_count),
        },
        # Work queue: relationship prose extracted but never promoted to an
        # edge. Measured against the fact index, so it counts the documents
        # the seeder still has to read rather than a difference of two
        # unrelated totals.
        "latent_relationship_entities": _latent_relationship_entities(st),
        # Cross-entity contamination, near-duplicate clusters, duplicate regrowth
        # (kg_hygiene.py). Added 2026-09-03 after 63 directories were found holding
        # facts about a different entity and nothing had measured it.
        "hygiene": _hygiene_section(),
        "fact_categories": dict(
            collections.Counter(categories).most_common()
        ),
    }


def _latent_relationship_entities(st) -> int:
    """Entities whose `relationship`-category facts have produced no edge.

    An entity counts as promoted once any active edge names it as a source;
    the rest are what `seed_relationship_edges.py` (or, since the extractor
    emits edges itself, the next extraction) still owes the graph.
    """
    with_prose = st.facts_idx.entities_with_category("relationship")
    if not with_prose:
        return 0
    sourced = {e["source"] for e in st.edges.active()}
    return len(with_prose - sourced)


def _hygiene_section() -> dict[str, Any]:
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import kg_hygiene
        snap = kg_hygiene.snapshot(VAULT_FACTS_ROOT, days=7)
        return {k: v for k, v in snap.items() if k in ("contamination", "near_duplicates", "regrowth")}
    except Exception as e:  # never let hygiene break the health snapshot
        return {"error": f"{type(e).__name__}: {e}"}


def _pct(n: int, total: int) -> float:
    return round(100.0 * n / total, 2) if total else 0.0


# ── Reporting ────────────────────────────────────────────────────────────────


def print_summary(s: dict[str, Any]) -> None:
    ent, edg, gr = s["entities"], s["edges"], s["graph"]

    print(f"Knowledge-Graph Health — {s['captured_at']}")
    print(f"  root: {s['facts_root']}\n")

    print(f"  entities                 {ent['count']:>8,}")
    print(f"  edges (live)             {edg['count']:>8,}")
    print(f"  nodes with >=1 edge      {gr['nodes_with_edges']:>8,}"
          f"   ({gr['node_coverage_pct']}% coverage)")
    print(f"  isolated entities        {gr['isolated_entities']:>8,}")
    print(f"  connected components     {gr['component_count']:>8,}"
          f"   (largest: {gr['largest_component']:,})")
    print(f"  alias coverage           {s['aliases']['count']:>8,}"
          f"   ({s['aliases']['coverage_pct']}%)")
    print()
    print(f"  PHASE 2 QUEUE — entities with relationship prose")
    print(f"  not yet promoted to edges {s['latent_relationship_entities']:>8,}")
    h = s.get("hygiene") or {}
    if "contamination" in h:
        c, n, r = h["contamination"], h["near_duplicates"], h["regrowth"]
        print("hygiene")
        print(f"  contaminated dirs         {c['dirs']:>8,}   ({c['foreign_facts']} facts about another entity)")
        print(f"  near-duplicate clusters   {n['clusters']:>8,}   {n['by_tier']}")
        print(f"  near-dup regrowth {r['days']}d      {r['near_dup_new']:>8,}   of {r['new_dirs']} new dirs")
    print()
    print(f"  names >=5 words          {ent['names_5plus_words']:>8,}"
          f"   ({ent['names_5plus_words_pct']}%)  <- Phase 1 target")
    print(f"  stray files in root      {ent['stray_root_files']:>8,}")

    if edg["by_provenance"]:
        print("\n  edges by provenance:")
        for k, v in edg["by_provenance"].items():
            print(f"    {v:>6,}  {k}")

    if gr["degree_distribution"]:
        print("\n  degree distribution:")
        for k in ("1", "2-3", "4-10", "11-50", "50+"):
            if k in gr["degree_distribution"]:
                print(f"    {gr['degree_distribution'][k]:>6,}  degree {k}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Knowledge-graph health snapshot (#380)")
    ap.add_argument("--json", action="store_true", help="emit raw JSON")
    ap.add_argument("-o", "--output", type=Path, help="write JSON snapshot to FILE")
    args = ap.parse_args()

    snapshot = build_snapshot()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(snapshot, indent=2))
        print(f"wrote {args.output}", file=sys.stderr)

    if args.json:
        print(json.dumps(snapshot, indent=2))
    else:
        print_summary(snapshot)


if __name__ == "__main__":
    main()
