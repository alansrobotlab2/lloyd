#!/usr/bin/env python3
"""Generate final entity resolution sweep report."""

import json
from pathlib import Path

RESULTS_FILE = Path("/tmp/entity_resolution_results.json")
MERGE_LOG_FILE = Path("/tmp/entity_merge_log.json")

def main():
    print("=" * 70)
    print("ENTITY RESOLUTION SWEEP - FINAL REPORT")
    print("=" * 70)
    print()

    # Load data
    with open(RESULTS_FILE) as f:
        results = json.load(f)

    with open(MERGE_LOG_FILE) as f:
        merge_log = json.load(f)

    initial_count = 753
    final_count = 735
    merged_count = len([m for m in merge_log['merges'] if m.get('files_merged', 0) > 0])

    print("SUMMARY")
    print("-" * 70)
    print(f"Initial entity count: {initial_count}")
    print(f"Final entity count: {final_count}")
    print(f"Entities merged: {initial_count - final_count}")
    print()

    print("HIGH CONFIDENCE MATCHES (score >= 0.85)")
    print("-" * 70)
    print(f"Total candidates: {results['high_confidence_count']}")
    print(f"Successfully merged: {merged_count}")
    print(f"Skipped (already processed): {results['high_confidence_count'] - merged_count}")
    print()

    print("MERGED ENTITIES:")
    for m in merge_log['merges']:
        if m.get('files_merged', 0) > 0:
            print(f"  {m['duplicate']} -> {m['canonical']} ({m['files_merged']} files)")

    print()
    print("MEDIUM CONFIDENCE MATCHES (flagged for review, 0.7-0.85)")
    print("-" * 70)
    print(f"Total candidates: {results['medium_confidence_count']}")
    print()

    print("CANDIDATES FOR MANUAL REVIEW:")
    for i, m in enumerate(results['medium_confidence_matches'][:20], 1):
        print(f"  {i}. {m['entity_a']} <-> {m['entity_b']} (score: {m['score']:.2f}, type: {m['match_type']})")

    if len(results['medium_confidence_matches']) > 20:
        print(f"  ... and {len(results['medium_confidence_matches']) - 20} more")

    print()
    print("ERRORS")
    print("-" * 70)
    if merge_log['errors']:
        for e in merge_log['errors']:
            print(f"  {e['entities']}: {e['reason']}")
    else:
        print("  No errors")

    print()
    print("=" * 70)
    print("END OF REPORT")
    print("=" * 70)

if __name__ == "__main__":
    main()
