#!/usr/bin/env python3
"""Run v4 classifier on remaining ~11k mentions edges, apply back to graph.

Pre-filters to only classify edges not already in classified-v4.jsonl or classified-v4-fullgraph.jsonl.
Uses 8 concurrent vLLM calls. Merges results back to _relationships.json.
"""

import sys
import json
import os
import time
import hashlib
from datetime import datetime

sys.path.insert(0, '/home/alansrobotlab/lloyd/scripts/memory')
sys.path.insert(0, '/home/alansrobotlab/lloyd/_pipeline/memory-graph')

# Import from v4 classifier - note the hyphen in filename
import importlib.util
import importlib.machinery

# Load the v4 classifier module by file path
loader = importlib.machinery.SourceFileLoader(
    'classify_relationships_v4',
    '/home/alansrobotlab/lloyd/scripts/memory/classify-relationships-v4.py'
)
spec = importlib.util.spec_from_loader('classify_relationships_v4', loader)
classify_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(classify_module)

# Also load the apply module
apply_loader = importlib.machinery.SourceFileLoader('apply_v4_fullgraph', '/home/alansrobotlab/lloyd/scripts/memory/apply_v4_fullgraph.py')
apply_spec = importlib.util.spec_from_loader('apply_v4_fullgraph', apply_loader)
apply_module = importlib.util.module_from_spec(apply_spec)

# We need to mock some things for the apply module
import types
import sys
apply_module.sys = sys
apply_module.json = json
apply_module.time = time
apply_module.datetime = datetime
apply_module.json.dumps = json.dumps
apply_module.json.dump = json.dump

spec.loader.exec_module(apply_module)

from classify_relationships_v4 import (
    load_all_edges,
    load_entity_facts,
    load_entity_aliases,
    classify_single_edge,
    run_v4_classify,
    EDGE_CONTEXT_LIMIT,
    ENTITY_STOPWORDS,
    MAX_TOKENS,
    MODEL,
    TEMPERATURE,
    TYPE_CANDIDATES,
)

FACTS_DIR = "/home/alansrobotlab/obsidian/facts"
RELATIONSHIPS_FILE = f"{FACTS_DIR}/_relationships.json"
CLASSIFIED_OUTPUT = "/home/alansrobotlab/lloyd/_pipeline/memory-graph/classified-v4-batch.jsonl"

# Existing classified files to skip
EXISTING_FILES = [
    "/home/alansrobotlab/lloyd/_pipeline/memory-graph/classified-v4.jsonl",
    "/home/alansrobotlab/lloyd/_pipeline/memory-graph/classified-v4-fullgraph.jsonl",
]


async def main():
    print(f"Batch v4 Classifier Runner")
    print(f"==========================")
    print(f"API URL: {os.environ.get('LLOYD_API_URL', 'http://127.0.0.1:8096')}")
    print(f"Concurrent calls: 8")
    print()

    # Load graph data
    print("Loading graph data...")
    all_edges = load_all_edges(RELATIONSHIPS_FILE)
    entity_facts = load_entity_facts(FACTS_DIR)
    entity_aliases = load_entity_aliases(entity_facts)

    # Count mentions edges
    mentions_edges = [e for e in all_edges if e["type"] == "mentions" and not e.get("expired_at")]
    print(f"Total mentions edges: {len(mentions_edges):,}")

    # Load existing classified pairs to skip
    print("Loading existing classified pairs...")
    already_classified = set()
    for fpath in EXISTING_FILES:
        if os.path.exists(fpath):
            with open(fpath) as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        pair = (record["source"], record["target"])
                        already_classified.add(pair)
            print(f"  {fpath}: {sum(1 for _ in open(fpath))} records")

    print(f"Already classified pairs: {len(already_classified):,}")

    # Filter to unclassified
    remaining = [
        e for e in mentions_edges
        if (e["source"], e["target"]) not in already_classified
    ]
    print(f"Remaining to classify: {len(remaining):,}")
    print()

    if not remaining:
        print("No remaining edges to classify.")
        return

    # Run v4 classify with 8 concurrent calls
    print(f"Classifying {len(remaining):,} edges with 8 concurrent LLM calls...")
    t0 = time.time()
    results = await run_v4_classify(
        remaining,
        entity_facts,
        entity_aliases,
        all_edges,
        max_concurrent=8,
    )
    classify_time = time.time() - t0
    print(f"Classification done in {classify_time:.0f}s ({len(results)} edges)")

    # Save to classified output
    print(f"Saving to {CLASSIFIED_OUTPUT}...")
    with open(CLASSIFIED_OUTPUT, "w") as f:
        for r in results:
            f.write(json.dumps(r.__dict__) + "\n")

    # Apply results back to graph
    print("Applying results to _relationships.json...")
    apply_start = time.time()

    with open(RELATIONSHIPS_FILE) as f:
        data = json.load(f)
    edges = data["edges"]

    applied = 0
    skipped = 0
    errors = 0

    for result in results:
        # Find matching edge
        found = False
        for i, edge in enumerate(edges):
            if edge["source"] == result.source and edge["target"] == result.target and edge["type"] == result.original_type:
                if not edge.get("expired_at"):
                    if result.new_type and result.new_type != result.original_type:
                        edges[i]["type"] = result.new_type
                        edges[i]["confidence"] = result.confidence
                        edges[i]["_classified"] = True
                        edges[i]["_classify_timestamp"] = datetime.utcnow().isoformat() + "Z"
                        edges[i]["_classify_reason"] = result.reason[:500]
                        edges[i]["_classification_version"] = "v4"
                        edges[i]["_entity_type_hint"] = result.entity_type_hint
                        if result.direction_check:
                            edges[i]["_direction_check"] = result.direction_check
                        if result.verdict_adjustment:
                            edges[i]["_verdict_adjustment"] = result.verdict_adjustment
                        applied += 1
                    else:
                        edges[i]["_classified"] = True
                        edges[i]["_classify_timestamp"] = datetime.utcnow().isoformat() + "Z"
                        edges[i]["_classify_reason"] = result.reason[:500]
                        edges[i]["_classification_version"] = "v4"
                        skipped += 1
                    found = True
                    break
        if not found:
            errors += 1
            print(f"  Warning: edge {result.source} -> {result.target} ({result.original_type}) not found in graph")

    # Write back
    data["edges"] = edges
    with open(RELATIONSHIPS_FILE, "w") as f:
        json.dump(data, f, indent=2)

    apply_time = time.time() - apply_start

    print(f"Applied {applied} type changes")
    print(f"Updated {skipped} edges with metadata only")
    print(f"Not found: {errors}")
    print(f"Apply time: {apply_time:.0f}s")
    print()

    # Summary
    from collections import Counter
    new_types = Counter(r.new_type for r in results if r.new_type)
    print("Edge type distribution in batch:")
    for t, c in new_types.most_common():
        print(f"  {t}: {c}")

    total_time = classify_time + apply_time
    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f}m)")
    print("Done.")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
