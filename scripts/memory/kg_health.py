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
                                                     # (name FILE per run, in UTC:
                                                     # kg-health-<date -u +%Y-%m-%dT%H%M%SZ>.json)

A run that writes a snapshot ALSO rolls the entity-directory baseline forward
after it (`kg_hygiene.write_baseline`), because that file is what the hygiene
`regrowth` section diffs `new_dirs` against — the reference has to move with the
snapshots it feeds, and an inspection (`--json` to stdout) must not move it
(#1535). `--no-baseline-update` skips it; `--baseline FILE` points at another.
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
from app.kg_store import StoreUnavailable, canonical_edge_type, store  # noqa: E402


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


def build_snapshot(baseline_path=None) -> dict[str, Any]:
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
        # UTC with an offset, like the per-run `kg-health-<UTC>Z.json` name the
        # skill writes it under; a naive local stamp labelled a 20:58 PDT run
        # with the previous UTC day's date (#822).
        "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
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
            # One key per relation (#1161). Counting the raw column is how the
            # 2026-09-15 snapshot came to report `related_to` 5,046 and
            # `related-to` 49 as two relations: the histogram overstated the
            # vocabulary and any per-type rule read off it saw half the edges.
            # The fold is the store's own rule, so the snapshot and the store
            # cannot disagree about what a canonical spelling is.
            "by_type": dict(
                collections.Counter(
                    canonical_edge_type(e.get("type") or "") for e in edges
                ).most_common()
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
        "hygiene": _hygiene_section(baseline_path),
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


def _kg_hygiene():
    """kg_hygiene sits beside this script, not in the package."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import kg_hygiene
    return kg_hygiene


def _hygiene_section(baseline_path=None) -> dict[str, Any]:
    try:
        snap = _kg_hygiene().snapshot(VAULT_FACTS_ROOT, days=7, baseline_path=baseline_path)
        # Passed through whole, not reshaped: `regrowth` carries the baseline it
        # diffed against (`dirs_at_baseline`, `baseline_at`, `no_baseline_reason`)
        # and the snapshot is the only place a reader can check its number
        # against (#1535 clause 3). Filtering keys here is what made the field
        # unreadable once already.
        return {k: v for k, v in snap.items() if k in ("contamination", "near_duplicates", "regrowth")}
    except Exception as e:  # never let hygiene break the health snapshot
        return {"error": f"{type(e).__name__}: {e}"}


def _regrowth_cell(r: dict[str, Any]) -> str:
    """`kg_hygiene.describe`, with a failure that still refuses to print a bare
    count: `of 12027 new dirs` with no reference beside it is the #1535 defect,
    so even the error path says the reference is missing rather than omitting it."""
    try:
        return _kg_hygiene().describe(r)
    except Exception as e:
        return f"not measured: the regrowth reference could not be rendered ({type(e).__name__}: {e})"


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
        # The reference belongs beside the number, not in a footnote: the line
        # this replaces read "51 of 12027 new dirs", where 12027 was the whole
        # store and nothing on the page said so (#1535 clause 5).
        print(f"  near-dup regrowth         {_regrowth_cell(r)}")
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


def _advance_baseline(baseline_path=None) -> None:
    """Roll the regrowth reference forward to the directory set just snapshotted.

    Only a run that WRITES a snapshot advances it. Advancing on `--json` to
    stdout — someone poking at the numbers — would reset the reference under the
    next scheduled run and report zero growth, which is how an instrument ends up
    measuring the last thing that touched it.

    A warning, not a failed run: the snapshot is already on disk, and the next
    run then diffs against the older reference — a bigger number, still honest,
    still labelled with the date it is a delta from.
    """
    try:
        rec = _kg_hygiene().write_baseline(VAULT_FACTS_ROOT, baseline_path)
        print(f"advanced entity-dir baseline {rec['path']} "
              f"({rec['dirs']:,} dirs at {rec['captured_at']})", file=sys.stderr)
    except Exception as e:
        print(f"WARNING: entity-dir baseline not advanced: {type(e).__name__}: {e}",
              file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description="Knowledge-graph health snapshot (#380)")
    ap.add_argument("--json", action="store_true", help="emit raw JSON")
    ap.add_argument("-o", "--output", type=Path, help="write JSON snapshot to FILE")
    ap.add_argument("--baseline", type=Path, default=None,
                    help="entity-dir baseline file regrowth diffs against (default: kg_hygiene.BASELINE_PATH)")
    ap.add_argument("--no-baseline-update", action="store_true",
                    help="do not roll the entity-dir baseline forward after writing a snapshot")
    args = ap.parse_args()

    snapshot = build_snapshot(baseline_path=args.baseline)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(snapshot, indent=2))
        print(f"wrote {args.output}", file=sys.stderr)
        # After the snapshot, so the `new_dirs` it holds is the delta from the
        # reference that existed when this run started.
        if not args.no_baseline_update:
            _advance_baseline(args.baseline)

    if args.json:
        print(json.dumps(snapshot, indent=2))
    else:
        print_summary(snapshot)


if __name__ == "__main__":
    main()
