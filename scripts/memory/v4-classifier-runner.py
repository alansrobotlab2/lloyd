#!/usr/bin/env python3
"""Run v4 classifier on remaining ~11k mention edges with 8 concurrent calls, merge back to graph."""

import sys
import json
import os
import time
import asyncio
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, '/home/alansrobotlab/lloyd/scripts/memory')

# Load v4 classifier module
import importlib.util
import importlib.machinery
loader = importlib.machinery.SourceFileLoader('classify_relationships_v4', '/home/alansrobotlab/lloyd/scripts/memory/classify-relationships-v4.py')
spec = importlib.util.spec_from_loader('classify_relationships_v4', loader)
classify_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(classify_module)
sys.modules['classify_relationships_v4'] = classify_module

from classify_relationships_v4 import classify_edge_v4, load_all_edges, _v2

FACTS_DIR = "/home/alansrobotlab/obsidian/facts"
REL_FILE = f"{FACTS_DIR}/_relationships.json"
CLASSIFIED_OUTPUT = "/home/alansrobotlab/lloyd/_pipeline/memory-graph/classified-v4-batch.jsonl"

LOG_FILE = "/home/alansrobotlab/lloyd/_pipeline/memory-graph/v4-classify-progress.log"
API_URL = "http://127.0.0.1:8096/v1/chat/completions"
MODEL = "Qwen/Qwen3.6-35B-A3B-FP8"
MAX_TOKENS = 4096
TEMPERATURE = 0.0
CONCURRENCY = 8

# Pre-filtered entity aliases and facts
ENTITY_ALIASES_CACHE = {}
ENTITY_FACTS_CACHE = {}


def load_entities():
    """Load all entity facts and aliases once."""
    global ENTITY_ALIASES_CACHE, ENTITY_FACTS_CACHE
    
    if ENTITY_FACTS_CACHE:
        return
    
    entity_facts = {}
    for dname in os.listdir(FACTS_DIR):
        fpath = os.path.join(FACTS_DIR, dname)
        if os.path.isdir(fpath):
            facts_file = os.path.join(fpath, "facts.json")
            if os.path.exists(facts_file):
                with open(facts_file) as f:
                    entity_data = json.load(f)
                entity_name = entity_data.get("entity_name", dname)
                entity_facts[entity_name] = entity_data
    
    entity_aliases = {}
    for entity_name, facts in entity_facts.items():
        for fact in facts.get("facts", []):
            aliases_text = fact.get("aliases", [])
            if isinstance(aliases_text, list):
                entity_aliases.setdefault(entity_name, set()).update(
                    a.strip() for a in aliases_text if a.strip()
                )
    
    ENTITY_FACTS_CACHE.update(entity_facts)
    ENTITY_ALIASES_CACHE.update(entity_aliases)


async def classify_one_async(edge, edge_context, entity_aliases, all_edges):
    """Classify a single edge using thread pool to run sync classify_edge_v4."""
    loop = asyncio.get_event_loop()
    
    def classify_sync():
        return classify_edge_v4(
            edge["source"], edge["target"], edge_context,
            API_URL, MODEL, TEMPERATURE, MAX_TOKENS,
            skip_direction_check=False, skip_context=False
        )
    
    return await loop.run_in_executor(ThreadPoolExecutor(1), classify_sync)


async def main():
    print("v4 CLASSIFIER RUNNER")
    print("=" * 60)
    print(f"API: {API_URL}")
    print(f"Concurrency: {CONCURRENCY}")
    print(f"Log: {LOG_FILE}")
    print()
    
    # Load entities once
    print("Loading entities...")
    load_entities()
    print(f"  Loaded {len(ENTITY_FACTS_CACHE)} entities, {len(ENTITY_ALIASES_CACHE)} aliases")
    
    # Load graph
    print("Loading graph...")
    with open(REL_FILE) as f:
        data = json.load(f)
    edges = data["edges"]
    
    # Get active mentions
    mentions = [e for e in edges if e["type"] == "mentions" and not e.get("expired_at")]
    print(f"  Active mentions: {len(mentions):,}")
    
    # Load already classified pairs
    print("Loading already classified pairs...")
    classified_pairs = set()
    for fpath in [
        "/home/alansrobotlab/lloyd/_pipeline/memory-graph/classified-v4.jsonl",
        "/home/alansrobotlab/lloyd/_pipeline/memory-graph/classified-v4-fullgraph.jsonl",
    ]:
        if os.path.exists(fpath):
            with open(fpath) as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        classified_pairs.add((rec["source"], rec["target"]))
    print(f"  Already classified: {len(classified_pairs):,}")
    
    # Filter to unclassified
    remaining = [e for e in mentions if (e["source"], e["target"]) not in classified_pairs]
    print(f"  Remaining to classify: {len(remaining):,}")
    print()
    
    if not remaining:
        print("Nothing to classify.")
        return
    
    # Build edge contexts for all remaining
    print("Building edge contexts...")
    edge_contexts = {}
    for edge in remaining:
        ctx = _v2._load_fact_snippets(edge["source"], edge["target"], 4000)
        edge_contexts[(edge["source"], edge["target"])] = ctx
    print(f"  Built {len(edge_contexts)} contexts")
    print()
    
    # Run classification with 8 concurrent calls
    print(f"Classifying {len(remaining):,} edges with {CONCURRENCY} concurrent calls...")
    log("Classification started")
    
    t_start = time.time()
    results = []
    failed = 0
    
    semaphore = asyncio.Semaphore(CONCURRENCY)
    
    async def classify_with_semaphore(edge):
        async with semaphore:
            ctx = edge_contexts.get((edge["source"], edge["target"]), None)
            if not ctx:
                return None, edge
            return await classify_one_async(edge, ctx, ENTITY_ALIASES_CACHE, edges)
    
    tasks = [classify_with_semaphore(edge) for edge in remaining]
    results_with_edge = await asyncio.gather(*tasks)
    
    for pair, result in results_with_edge:
        if result is None:
            failed += 1
        else:
            results.append((pair, result))
    
    elapsed = time.time() - t_start
    rate = len(results) / elapsed if elapsed > 0 else 0
    
    # Summary
    print()
    print("=" * 60)
    print("CLASSIFICATION COMPLETE")
    print("=" * 60)
    print(f"Processed: {len(results)}")
    print(f"Failed: {failed}")
    print(f"Time: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"Rate: {rate:.1f} edges/s")
    print()
    
    # Edge type distribution
    type_counts = Counter()
    for _, r in results:
        type_counts[r.get("type", "unknown") or "mentions"] += 1
    
    print("Edge type distribution:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c:,}")
    print()
    
    # Save classified output
    print(f"Saving to {CLASSIFIED_OUTPUT}...")
    with open(CLASSIFIED_OUTPUT, "w") as f:
        for pair, result in results:
            src, tgt = pair
            record = {
                "source": src,
                "target": tgt,
                "original_type": "mentions",
                "new_type": result.get("type"),
                "confidence": result.get("confidence"),
                "reason": result.get("reason", ""),
                "reason_quote": result.get("reason_quote"),
                "quote_verified": result.get("quote_verified"),
                "direction_check": result.get("direction_check"),
                "verdict_adjustment": result.get("verdict_adjustment"),
                "classified_at": datetime.now(timezone.utc).isoformat(),
            }
            f.write(json.dumps(record) + "\n")
    
    print(f"Saved {len(results)} records")
    
    # Apply to graph
    print("\nApplying to graph...")
    graph_edges = data["edges"]
    
    # Build hash map
    edge_map = {}
    for i, e in enumerate(graph_edges):
        key = (e["source"], e["target"], e["type"])
        edge_map[key] = i
    
    upgrades = 0
    downgrades = 0
    unchanged = 0
    not_found = 0
    
    for pair, result in results:
        source, target = pair
        # Find the mentions edge
        key = (source, target, "mentions")
        if key in edge_map:
            idx = edge_map[key]
            edge = graph_edges[idx]
            
            new_type = result.get("type")
            if new_type and new_type != "mentions":
                edge["type"] = new_type
                edge["confidence"] = result.get("confidence", 0.5)
                edge["_classified"] = True
                edge["_classify_timestamp"] = result.get("classified_at", datetime.now(timezone.utc).isoformat())
                edge["_classify_reason"] = result.get("reason", "")[:500]
                edge["_classification_version"] = "v4"
                
                if result.get("direction_check"):
                    edge["_direction_check"] = result["direction_check"]
                if result.get("verdict_adjustment") and result["verdict_adjustment"] != "none":
                    edge["_verdict_adjustment"] = result["verdict_adjustment"]
                    
                upgrades += 1
                if result.get("confidence", 1.0) < 0.6:
                    downgrades += 1
            else:
                unchanged += 1
                edge["_classified"] = True
                edge["_classify_timestamp"] = result.get("classified_at", datetime.now(timezone.utc).isoformat())
                edge["_classify_reason"] = result.get("reason", "")[:500]
                edge["_classification_version"] = "v4"
        else:
            not_found += 1
    
    # Write back
    data["edges"] = graph_edges
    with open(REL_FILE, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Upgrades: {upgrades:,}")
    print(f"Downgrades: {downgrades:,}")
    print(f"Unchanged: {unchanged:,}")
    print(f"Not found: {not_found:,}")
    print()
    
    # Final verification
    print("Verifying merged graph...")
    with open(REL_FILE) as f:
        final_data = json.load(f)
    final_edges = final_data["edges"]
    final_types = Counter(e["type"] for e in final_edges)
    
    print(f"\nFinal edge type distribution ({len(final_edges):,} edges):")
    for t, c in sorted(final_types.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c:,}")
    
    final_mentions = sum(1 for e in final_edges if e["type"] == "mentions")
    print(f"\nmentions: {final_mentions:,} ({100*final_mentions/len(final_edges):.1f}%)")
    unique_types = len([t for t, c in final_types.items() if c > 0])
    print(f"Unique edge types: {unique_types}")
    print()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
