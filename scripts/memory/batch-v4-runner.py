#!/usr/bin/env python3
"""Batch v4 classifier — runs remaining mentions through v4 pipeline with 8 concurrent calls, then applies back."""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent))

from classify_relationships_v4 import (
    EDGE_CONTEXT_LIMIT,
    ENTITY_STOPWORDS,
    MAX_TOKENS,
    MODEL,
    TEMPERATURE,
    TYPE_CANDIDATES,
    CONCEPT_KW,
    ENTITY_ALIAS_PATTERN,
    FILE_KW,
    PERSON_KW,
    ROLE_KW,
    SKILL_KW,
    SYSTEM_KW,
    TASK_KW,
    classify_single_edge,
    run_v4_classify,
    load_all_edges,
    load_entity_facts,
    load_entity_aliases,
)

FACTS_DIR = "/home/alansrobotlab/obsidian/facts"
RELATIONSHIPS_FILE = f"{FACTS_DIR}/_relationships.json"
CLASSIFIED_OUTPUT = "/home/alansrobotlab/lloyd/_pipeline/memory-graph/classified-v4-batch.jsonl"
CLASSIFIED_V4_JSONL = "/home/alansrobotlab/lloyd/_pipeline/memory-graph/classified-v4.jsonl"

async def build_classified_set() -> Set[Tuple[str, str]]:
    """Load already-classified pairs from classified-v4.jsonl."""
    pairs: Set[Tuple[str, str]] = set()
    if os.path.exists(CLASSIFIED_V4_JSONL):
        with open(CLASSIFIED_V4_JSONL) as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    pairs.add((record["source"], record["target"]))
    return pairs

async def classify_batch(
    all_edges: List[Dict[str, Any]],
    entity_facts: Dict[str, Dict[str, Any]],
    entity_aliases: Dict[str, Set[str]],
    classified_pairs: Set[Tuple[str, str]],
) -> List[Dict]:
    """Filter to unclassified mentions, run v4 classify, return results as dicts."""
    
    # Filter to active mentions NOT already classified
    mentions_edges = [
        e for e in all_edges
        if e["type"] == "mentions" and not e.get("expired_at")
        and (e["source"], e["target"]) not in classified_pairs
    ]
    
    print(f"Unclassified mentions: {len(mentions_edges)}")
    
    if not mentions_edges:
        return []
    
    # Run v4 classify with 8 concurrent calls
    results = await run_v4_classify(
        mentions_edges,
        entity_facts,
        entity_aliases,
        all_edges,
        max_concurrent=8,
    )
    
    # Convert to dicts for apply
    outputs = []
    for r in results:
        outputs.append({
            "source": r.source,
            "target": r.target,
            "original_type": r.original_type,
            "new_type": r.new_type,
            "confidence": r.confidence,
            "reason": r.reason,
            "quote_verified": r.quote_verified,
            "direction_check": r.direction_check,
            "verdict_adjustment": r.verdict_adjustment,
            "edge_hash": r.edge_hash,
            "entity_type_hint": r.entity_type_hint,
        })
    
    return outputs

async def apply_results(
    results: List[Dict],
    relationships_file: str,
) -> Dict[str, Any]:
    """Apply classified results back to _relationships.json."""
    
    with open(relationships_file) as f:
        data = json.load(f)
    edges = data["edges"]
    
    # Build hash map
    edge_map: Dict[str, Tuple[int, Dict]] = {}
    for i, e in enumerate(edges):
        h = hashlib.md5(f"{e['source']}|{e['target']}|{e.get('confidence', 0.8)}".encode()).hexdigest()
        edge_map[h] = (i, e)
    
    # Count changes
    upgrade_count = sum(1 for r in results if r.get("new_type") and r["new_type"] != r["original_type"] and r["new_type"] != "mentions")
    downgrade_count = sum(1 for r in results if r.get("new_type") and r["new_type"] == "mentions")
    unchanged_count = sum(1 for r in results if not r.get("new_type") or r["new_type"] == r["original_type"])
    
    # Apply updates
    for r in results:
        h = r.get("edge_hash", "")
        if h in edge_map:
            idx, e = edge_map[h]
            if r.get("new_type"):
                e["type"] = r["new_type"]
                e["confidence"] = r["confidence"]
                e["_classified"] = True
                e["_classify_version"] = "v4"
                e["_classify_reason"] = r.get("reason", "")
                if r.get("direction_check"):
                    e["_direction_check"] = r["direction_check"]
                e["_verdict_adjustment"] = r.get("verdict_adjustment", "")
                e["_entity_type_hint"] = r.get("entity_type_hint", "")
    
    # Save
    with open(relationships_file, "w") as f:
        json.dump({"edges": edges, "meta": data.get("meta", {})}, f, indent=2)
    
    return {
        "total_edges": len(edges),
        "results_count": len(results),
        "upgrades": upgrade_count,
        "downgrades": downgrade_count,
        "unchanged": unchanged_count,
    }

async def main():
    print("=" * 60)
    print("Batch v4 Classifier Runner")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    all_edges = load_all_edges(RELATIONSHIPS_FILE)
    entity_facts = load_entity_facts(FACTS_DIR)
    entity_aliases = load_entity_aliases(entity_facts)
    
    print(f"  Total edges: {len(all_edges)}")
    print(f"  Entities: {len(entity_facts)}")
    
    # Load existing classified pairs
    print("\nLoading existing classified pairs...")
    classified_pairs = await build_classified_set()
    print(f"  Existing pairs: {len(classified_pairs)}")
    
    # Classify remaining
    print(f"\nClassifying remaining mentions with 8 concurrent LLM calls...")
    t0 = time.time()
    results = await classify_batch(all_edges, entity_facts, entity_aliases, classified_pairs)
    classify_time = time.time() - t0
    print(f"  Classification done in {classify_time:.0f}s ({len(results)} edges)")
    
    if results:
        # Save classified output
        print(f"\nSaving classified output to {CLASSIFIED_OUTPUT}...")
        with open(CLASSIFIED_OUTPUT, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        
        # Apply to graph
        print(f"\nApplying to graph...")
        t1 = time.time()
        apply_result = await apply_results(results, RELATIONSHIPS_FILE)
        apply_time = time.time() - t1
        print(f"  Apply done in {apply_time:.0f}s")
        print(f"\n  Upgrades:    {apply_result['upgrades']}")
        print(f"  Downgrades:  {apply_result['downgrades']}")
        print(f"  Unchanged:   {apply_result['unchanged']}")
        
        total_time = classify_time + apply_time
        print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f}m)")
    else:
        print("No remaining edges to classify.")

if __name__ == "__main__":
    asyncio.run(main())
