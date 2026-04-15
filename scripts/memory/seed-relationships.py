#!/usr/bin/env python3
"""Bulk-seed the entity relationship store from existing data sources.

Extraction methods:
  1. Co-occurrence: entities sharing the same source_doc get "co_mentioned" edges
  2. Fact text mentions: entity A's fact text mentions entity B -> "mentions" edge
  3. Wiki-link co-occurrence: relations-index wiki-link pairs -> "wiki_link_co_occurrence" edges

Usage:
  python seed-relationships.py           # dry-run (default)
  python seed-relationships.py --dry-run # explicit dry-run
  python seed-relationships.py --apply   # write changes to disk
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

FACTS_DIR = Path.home() / "obsidian" / "facts"
RELATIONSHIPS_FILE = FACTS_DIR / "_relationships.json"
RELATIONS_INDEX = Path.home() / "lloyd" / "_pipeline" / "relations-index.json"

# Confidence levels per extraction method
CONFIDENCE = {
    "co_mentioned": 0.7,
    "mentions": 0.8,
    "wiki_link_co_occurrence": 0.6,
}

# Minimum entity name length to avoid false-positive matches
MIN_ENTITY_NAME_LEN = 5

# Generic entity names that match too broadly — skip for mentions extraction
ENTITY_STOPWORDS = {
    "test", "agent", "agents", "memory", "system", "state", "config",
    "update", "status", "event", "error", "general", "model", "tools",
    "skill", "skills", "plan", "plans", "notes", "data", "pipeline",
    "server", "service", "client", "task", "tasks", "build", "setup",
    "review", "research", "project", "debug", "audit", "queue", "cache",
    "proxy", "bridge", "index", "store", "report", "search", "query",
}

# Skip UUID-style entity names
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-", re.IGNORECASE)


def load_relationships() -> dict:
    """Load the existing relationship store."""
    if RELATIONSHIPS_FILE.exists():
        with open(RELATIONSHIPS_FILE) as f:
            return json.load(f)
    return {"edges": [], "schema_version": 1}


def save_relationships(data: dict) -> None:
    """Write the relationship store to disk."""
    with open(RELATIONSHIPS_FILE, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def build_existing_edge_set(edges: list[dict]) -> set[tuple[str, str, str]]:
    """Build a set of (source, target, type) for active (non-expired) edges."""
    result = set()
    for e in edges:
        if e.get("expired_at") is None:
            result.add((e["source"], e["target"], e["type"]))
    return result


def get_entity_names() -> list[str]:
    """Get canonical entity names from the facts directory structure."""
    entities = []
    for entry in os.listdir(FACTS_DIR):
        path = FACTS_DIR / entry
        if not path.is_dir():
            continue
        if entry.startswith(".") or entry.startswith("_"):
            continue
        entities.append(entry)
    return sorted(entities)


def make_edge(source: str, target: str, edge_type: str, source_doc: str | None = None) -> dict:
    """Create a new edge dict."""
    return {
        "source": source,
        "target": target,
        "type": edge_type,
        "confidence": CONFIDENCE[edge_type],
        "provenance": "EXTRACTED",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "expired_at": None,
        "source_doc": source_doc,
    }


def load_facts_for_entity(entity_dir: Path) -> list[dict]:
    """Load all facts from YAML files in an entity directory."""
    facts = []
    for fname in os.listdir(entity_dir):
        if not fname.endswith(".md"):
            continue
        fpath = entity_dir / fname
        try:
            content = fpath.read_text(encoding="utf-8")
            # Extract YAML frontmatter
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    frontmatter = yaml.safe_load(parts[1])
                    if frontmatter and isinstance(frontmatter.get("facts"), list):
                        for fact in frontmatter["facts"]:
                            if isinstance(fact, dict):
                                facts.append(fact)
        except Exception:
            continue
    return facts


def extract_co_mentioned(entities: list[str], existing: set) -> list[dict]:
    """Method 1: entities sharing the same source_doc get co_mentioned edges."""
    # Map source_doc -> set of entity names
    doc_to_entities: dict[str, set[str]] = defaultdict(set)

    for entity in entities:
        entity_dir = FACTS_DIR / entity
        if not entity_dir.is_dir():
            continue
        facts = load_facts_for_entity(entity_dir)
        for fact in facts:
            source_doc = fact.get("source_doc")
            if source_doc and fact.get("expired_at") is None:
                doc_to_entities[source_doc].add(entity)

    new_edges = []
    for doc, ent_set in doc_to_entities.items():
        ent_list = sorted(ent_set)
        for i in range(len(ent_list)):
            for j in range(i + 1, len(ent_list)):
                a, b = ent_list[i], ent_list[j]
                if (a, b, "co_mentioned") not in existing and (b, a, "co_mentioned") not in existing:
                    new_edges.append(make_edge(a, b, "co_mentioned", source_doc=doc))
                    existing.add((a, b, "co_mentioned"))

    return new_edges


def extract_mentions(entities: list[str], existing: set) -> list[dict]:
    """Method 2: entity A's fact text mentions entity B -> mentions edge."""
    # Build lookup: lowercase entity name -> canonical entity name
    # Skip short names, stopwords, and UUID-style names
    entity_set = {
        e.lower(): e for e in entities
        if len(e) >= MIN_ENTITY_NAME_LEN
        and e.lower() not in ENTITY_STOPWORDS
        and not UUID_RE.match(e)
    }

    # Pre-compile patterns for each entity (word-boundary, case-insensitive)
    patterns: dict[str, re.Pattern] = {}
    for lower_name, canonical in entity_set.items():
        # Escape for regex, use word boundaries
        escaped = re.escape(canonical)
        patterns[lower_name] = re.compile(r"\b" + escaped + r"\b", re.IGNORECASE)

    new_edges = []
    for entity in entities:
        # Skip source entities that are UUIDs, pure numbers, or stopwords
        if UUID_RE.match(entity) or entity.isdigit() or entity.lower() in ENTITY_STOPWORDS:
            continue
        entity_dir = FACTS_DIR / entity
        if not entity_dir.is_dir():
            continue
        facts = load_facts_for_entity(entity_dir)
        entity_lower = entity.lower()

        # Collect all fact text for this entity
        all_text = " ".join(f.get("fact", "") for f in facts if f.get("expired_at") is None)
        if not all_text:
            continue

        for target_lower, target_canonical in entity_set.items():
            # Skip self-references
            if target_lower == entity_lower:
                continue
            if patterns[target_lower].search(all_text):
                if (entity, target_canonical, "mentions") not in existing:
                    new_edges.append(make_edge(entity, target_canonical, "mentions"))
                    existing.add((entity, target_canonical, "mentions"))

    return new_edges


def path_to_entity_name(doc_path: str) -> str | None:
    """Extract an entity-like name from a vault document path.

    Examples:
      projects/lloyd/plans/discord-voice-integration.md -> Lloyd
      personal/people/family/ron-hokanson.md -> ron-hokanson
      knowledge/robotics/unitree-g1.md -> unitree-g1
      memory/2026-04-01.md -> None (date-based, not a meaningful entity)
    """
    parts = Path(doc_path).parts
    if not parts:
        return None

    # Skip pure date-based memory files
    if parts[0] == "memory":
        return None

    # For project paths, use the project name (second component)
    if parts[0] == "projects" and len(parts) >= 2:
        return parts[1]

    # For work paths, use the org or project name
    if parts[0] == "work" and len(parts) >= 2:
        return parts[1]

    # For personal/people paths, use the person name (stem of filename)
    if parts[0] == "personal" and "people" in parts:
        stem = Path(doc_path).stem
        return stem

    # For knowledge paths, use the filename stem
    if parts[0] == "knowledge":
        stem = Path(doc_path).stem
        return stem

    return None


def extract_wiki_link_co_occurrence(entities: list[str], existing: set) -> list[dict]:
    """Method 3: wiki-link relationships from the relations-index."""
    if not RELATIONS_INDEX.exists():
        return []

    with open(RELATIONS_INDEX) as f:
        data = json.load(f)

    # Build a case-insensitive lookup from entity names
    entity_lookup: dict[str, str] = {}
    for e in entities:
        entity_lookup[e.lower()] = e
        # Also index hyphenated/spaces variants
        entity_lookup[e.lower().replace(" ", "-")] = e
        entity_lookup[e.lower().replace("-", " ")] = e

    new_edges = []
    seen_pairs: set[tuple[str, str]] = set()

    for rel in data.get("relationships", []):
        if rel.get("type") != "wiki-link":
            continue

        source_path = rel["source"]
        target_path = rel["target"]

        # Skip self-links
        if source_path == target_path:
            continue

        source_entity = path_to_entity_name(source_path)
        target_entity = path_to_entity_name(target_path)

        if not source_entity or not target_entity:
            continue

        # Resolve to canonical entity names
        src_canonical = entity_lookup.get(source_entity.lower())
        tgt_canonical = entity_lookup.get(target_entity.lower())

        if not src_canonical or not tgt_canonical:
            continue

        # Skip self-references
        if src_canonical == tgt_canonical:
            continue

        # Deduplicate within this extraction (order-independent)
        pair = tuple(sorted([src_canonical, tgt_canonical]))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        a, b = pair
        if (a, b, "wiki_link_co_occurrence") not in existing and (b, a, "wiki_link_co_occurrence") not in existing:
            new_edges.append(make_edge(a, b, "wiki_link_co_occurrence"))
            existing.add((a, b, "wiki_link_co_occurrence"))

    return new_edges


def main():
    parser = argparse.ArgumentParser(description="Bulk-seed entity relationship store")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true", default=True, help="Preview changes without writing (default)")
    group.add_argument("--apply", action="store_true", help="Write changes to disk")
    args = parser.parse_args()

    if args.apply:
        args.dry_run = False

    # Load current state
    rel_data = load_relationships()
    edges_before = len(rel_data["edges"])
    existing = build_existing_edge_set(rel_data["edges"])

    print(f"Relationship store: {RELATIONSHIPS_FILE}")
    print(f"Edges before: {edges_before}")
    print()

    # Get entity list
    entities = get_entity_names()
    print(f"Entities found: {len(entities)}")
    print()

    # Method 1: Co-occurrence
    print("--- Method 1: Co-occurrence in source_doc ---")
    co_edges = extract_co_mentioned(entities, existing)
    print(f"  New co_mentioned edges: {len(co_edges)}")
    for e in co_edges[:5]:
        print(f"    {e['source']} -- co_mentioned --> {e['target']}  (doc: {e['source_doc']})")
    if len(co_edges) > 5:
        print(f"    ... and {len(co_edges) - 5} more")
    print()

    # Method 2: Fact text mentions
    print("--- Method 2: Fact text entity mentions ---")
    mention_edges = extract_mentions(entities, existing)
    print(f"  New mentions edges: {len(mention_edges)}")
    for e in mention_edges[:5]:
        print(f"    {e['source']} -- mentions --> {e['target']}")
    if len(mention_edges) > 5:
        print(f"    ... and {len(mention_edges) - 5} more")
    print()

    # Method 3: Wiki-link co-occurrence
    print("--- Method 3: Wiki-link co-occurrence ---")
    wiki_edges = extract_wiki_link_co_occurrence(entities, existing)
    print(f"  New wiki_link_co_occurrence edges: {len(wiki_edges)}")
    for e in wiki_edges[:5]:
        print(f"    {e['source']} -- wiki_link_co_occurrence --> {e['target']}")
    if len(wiki_edges) > 5:
        print(f"    ... and {len(wiki_edges) - 5} more")
    print()

    # Summary
    all_new = co_edges + mention_edges + wiki_edges
    total_new = len(all_new)
    edges_after = edges_before + total_new

    print("=" * 50)
    print(f"Edges before:        {edges_before}")
    print(f"  + co_mentioned:    {len(co_edges)}")
    print(f"  + mentions:        {len(mention_edges)}")
    print(f"  + wiki_link_co_occurrence: {len(wiki_edges)}")
    print(f"  = total new:       {total_new}")
    print(f"Edges after:         {edges_after}")
    print()

    if args.dry_run:
        print("[DRY RUN] No changes written. Use --apply to write.")
    else:
        rel_data["edges"].extend(all_new)
        save_relationships(rel_data)
        print(f"[APPLIED] Wrote {edges_after} edges to {RELATIONSHIPS_FILE}")


if __name__ == "__main__":
    main()
