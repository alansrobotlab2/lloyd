#!/usr/bin/env python3
"""
Entity Resolution Sweep - Identify and merge duplicate entities in the fact store.
Uses three-tier resolution: exact, fuzzy (Levenshtein + Jaccard), and substring containment.
"""

import os
import re
import sys
import json
from pathlib import Path
from typing import List, Tuple, Dict
import difflib

try:
    from rapidfuzz import fuzz, distance
except ImportError:
    print("rapidfuzz not installed, using fallback")
    fuzz = None
    distance = None

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.paths import VAULT_FACTS_ROOT as FACTS_DIR
EXCLUDE_DIRS = {"_pipeline", "_relationships.json"}

def normalize_name(name: str) -> str:
    """Normalize entity name for comparison."""
    # Remove special chars, lowercase, strip
    normalized = re.sub(r'[^\w\s]', '', name.lower()).strip()
    # Remove extra whitespace
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized

def exact_match(name_a: str, name_b: str) -> Tuple[bool, float]:
    """Check for exact (case-insensitive) match."""
    norm_a = normalize_name(name_a)
    norm_b = normalize_name(name_b)
    if norm_a == norm_b:
        return True, 1.0
    return False, 0.0

def substring_match(name_a: str, name_b: str) -> Tuple[bool, float]:
    """Check if one name is contained in the other."""
    norm_a = normalize_name(name_a)
    norm_b = normalize_name(name_b)

    if norm_a in norm_b or norm_b in norm_a:
        # Score based on length ratio
        ratio = min(len(norm_a), len(norm_b)) / max(len(norm_a), len(norm_b))
        return True, ratio
    return False, 0.0

def fuzzy_match(name_a: str, name_b: str) -> Tuple[bool, float]:
    """Compute fuzzy similarity using Levenshtein distance + token overlap."""
    norm_a = normalize_name(name_a)
    norm_b = normalize_name(name_b)

    # Use sequence matcher as fallback
    ratio = difflib.SequenceMatcher(None, norm_a, norm_b).ratio()

    # Token overlap (Jaccard similarity)
    tokens_a = set(norm_a.split())
    tokens_b = set(norm_b.split())

    if tokens_a and tokens_b:
        intersection = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)
        jaccard = intersection / union if union > 0 else 0
        # Weighted average: 70% Levenshtein, 30% Jaccard
        ratio = 0.7 * ratio + 0.3 * jaccard

    return ratio >= 0.5, ratio

def count_facts(entity_dir: Path) -> int:
    """Count fact files in an entity directory."""
    if not entity_dir.is_dir():
        return 0
    return len(list(entity_dir.glob("*.md")))

def get_entity_info(entity_name: str) -> Dict:
    """Get information about an entity."""
    # Handle directory names with spaces - they may be quoted
    entity_path = FACTS_DIR / entity_name
    if not entity_path.exists():
        # Try without quotes
        clean_name = entity_name.strip("'\"")
        entity_path = FACTS_DIR / clean_name

    return {
        "name": entity_name,
        "path": entity_path,
        "fact_count": count_facts(entity_path) if entity_path.exists() else 0,
        "exists": entity_path.exists()
    }

def find_matches(entities: List[str]) -> List[Dict]:
    """Find all potential duplicate matches."""
    matches = []
    n = len(entities)

    for i in range(n):
        for j in range(i + 1, n):
            name_a, name_b = entities[i], entities[j]

            # Skip if either doesn't exist
            info_a = get_entity_info(name_a)
            info_b = get_entity_info(name_b)
            if not info_a["exists"] or not info_b["exists"]:
                continue

            # Check exact match
            is_exact, score = exact_match(name_a, name_b)
            if is_exact:
                matches.append({
                    "entity_a": name_a,
                    "entity_b": name_b,
                    "match_type": "exact",
                    "score": score,
                    "info_a": info_a,
                    "info_b": info_b
                })
                continue

            # Check substring match
            is_substring, sub_score = substring_match(name_a, name_b)
            if is_substring:
                matches.append({
                    "entity_a": name_a,
                    "entity_b": name_b,
                    "match_type": "substring",
                    "score": sub_score,
                    "info_a": info_a,
                    "info_b": info_b
                })
                continue

            # Check fuzzy match
            is_fuzzy, fuzzy_score = fuzzy_match(name_a, name_b)
            if is_fuzzy and fuzzy_score >= 0.7:
                matches.append({
                    "entity_a": name_a,
                    "entity_b": name_b,
                    "match_type": "fuzzy",
                    "score": fuzzy_score,
                    "info_a": info_a,
                    "info_b": info_b
                })

    return matches

def main():
    """Main entry point for entity resolution sweep."""
    print("=" * 60)
    print("ENTITY RESOLUTION SWEEP")
    print("=" * 60)

    # Get all entity directories
    entity_names = []
    for item in FACTS_DIR.iterdir():
        if item.name in EXCLUDE_DIRS or not item.name[0].isalnum() and not item.name[0].isdigit():
            continue
        if item.is_dir() or (item.is_file() and item.name.endswith(".md")):
            entity_names.append(item.name)

    # Remove duplicates and clean
    entity_names = list(set(entity_names))
    print(f"\nTotal entities: {len(entity_names)}")

    # Find matches
    print("\nSearching for duplicate candidates...")
    matches = find_matches(entity_names)

    # Categorize by confidence
    high_confidence = [m for m in matches if m["score"] >= 0.85]
    medium_confidence = [m for m in matches if 0.7 <= m["score"] < 0.85]

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print("=" * 60)

    print(f"\nTotal matches found: {len(matches)}")
    print(f"High confidence (>= 0.85): {len(high_confidence)}")
    print(f"Medium confidence (0.7-0.85): {len(medium_confidence)}")

    if high_confidence:
        print(f"\n{'=' * 60}")
        print("HIGH CONFIDENCE MATCHES (Auto-merge candidates)")
        print("=" * 60)
        for i, match in enumerate(high_confidence, 1):
            print(f"\n{i}. {match['entity_a']} <-> {match['entity_b']}")
            print(f"   Match type: {match['match_type']}")
            print(f"   Score: {match['score']:.2f}")
            print(f"   Facts: {match['info_a']['fact_count']} vs {match['info_b']['fact_count']}")

    if medium_confidence:
        print(f"\n{'=' * 60}")
        print("MEDIUM CONFIDENCE MATCHES (Flag for review)")
        print("=" * 60)
        for i, match in enumerate(medium_confidence, 1):
            print(f"\n{i}. {match['entity_a']} <-> {match['entity_b']}")
            print(f"   Match type: {match['match_type']}")
            print(f"   Score: {match['score']:.2f}")

    # Save results to file for further processing
    results_file = Path("/tmp/entity_resolution_results.json")
    results_data = {
        "total_entities": len(entity_names),
        "total_matches": len(matches),
        "high_confidence_count": len(high_confidence),
        "medium_confidence_count": len(medium_confidence),
        "high_confidence_matches": [
            {k: v for k, v in m.items() if k not in ["info_a", "info_b"]}
            for m in high_confidence
        ],
        "medium_confidence_matches": [
            {k: v for k, v in m.items() if k not in ["info_a", "info_b"]}
            for m in medium_confidence
        ]
    }

    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Results saved to: {results_file}")
    print("=" * 60)

    return results_data

if __name__ == "__main__":
    main()
