#!/usr/bin/env python3
"""
Merge duplicate entities based on resolution results.
Merges high-confidence matches (score >= 0.85) and preserves fact history.
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime

FACTS_DIR = Path("/home/alansrobotlab/obsidian/facts")
RESULTS_FILE = Path("/tmp/entity_resolution_results.json")

def get_canonical_entity(info_a: dict, info_b: dict) -> tuple:
    """Determine which entity should be canonical (keep more facts)."""
    if info_a["fact_count"] >= info_b["fact_count"]:
        return info_a["name"], info_b["name"]
    return info_b["name"], info_a["name"]

def merge_entity_files(source_dir: Path, target_dir: Path, source_name: str, target_name: str) -> list:
    """Move and rename fact files from source to target entity."""
    merged_files = []

    if not source_dir.exists() or not source_dir.is_dir():
        return merged_files

    for fact_file in source_dir.iterdir():
        if not fact_file.is_file() or not fact_file.suffix == ".md":
            continue

        # Read the fact file to get entity info
        try:
            content = fact_file.read_text()
        except Exception as e:
            print(f"  Error reading {fact_file.name}: {e}")
            continue

        # Create new filename with target entity name
        new_name = fact_file.name.replace(source_name, target_name)
        # Handle case sensitivity in filenames
        new_name = new_name.replace(source_name.lower(), target_name.lower())

        target_path = target_dir / new_name

        # If target file exists, append content with separator
        if target_path.exists():
            print(f"  File collision: {new_name} already exists in target")
            # Append with marker
            with open(target_path, "a") as f:
                f.write(f"\n\n--- MERGED FROM: {source_name} ({datetime.now().isoformat()}) ---\n")
                f.write(content)
        else:
            # Move and rename
            shutil.move(str(fact_file), str(target_path))

        merged_files.append({
            "source": str(fact_file),
            "target": str(target_path),
            "original_name": fact_file.name,
            "new_name": new_name
        })

    return merged_files

def main():
    """Main entry point for entity merging."""
    print("=" * 60)
    print("ENTITY MERGE - High Confidence Matches Only")
    print("=" * 60)

    # Load results
    if not RESULTS_FILE.exists():
        print(f"ERROR: Results file not found: {RESULTS_FILE}")
        return

    with open(RESULTS_FILE, "r") as f:
        results = json.load(f)

    high_conf_matches = results.get("high_confidence_matches", [])
    print(f"\nHigh confidence matches to process: {len(high_conf_matches)}")

    if not high_conf_matches:
        print("No high confidence matches to merge.")
        return

    # Track merges
    merge_log = {
        "timestamp": datetime.now().isoformat(),
        "total_to_merge": len(high_conf_matches),
        "merges": [],
        "errors": []
    }

    # Process each match (limit to 10 per run as per guardrails)
    max_merges = 10
    processed = 0

    for match in high_conf_matches:
        if processed >= max_merges:
            print(f"\nReached maximum of {max_merges} merges for this run.")
            break

        entity_a = match["entity_a"]
        entity_b = match["entity_b"]
        score = match["score"]

        print(f"\n{'=' * 40}")
        print(f"Processing: {entity_a} <-> {entity_b}")
        print(f"Score: {score:.2f}")

        # Clean names (remove quotes)
        clean_a = entity_a.strip("'\"")
        clean_b = entity_b.strip("'\"")

        # Determine canonical
        dir_a = FACTS_DIR / clean_a
        dir_b = FACTS_DIR / clean_b

        # Handle directory names with spaces (may be quoted in filesystem)
        if not dir_a.exists():
            # Try to find actual directory
            for item in FACTS_DIR.iterdir():
                if item.name.replace("'", "").replace('"', "") == clean_a:
                    dir_a = item
                    break

        if not dir_b.exists():
            for item in FACTS_DIR.iterdir():
                if item.name.replace("'", "").replace('"', "") == clean_b:
                    dir_b = item
                    break

        if not dir_a.exists() or not dir_b.exists():
            print(f"  ERROR: One or both directories don't exist")
            merge_log["errors"].append({
                "entities": [entity_a, entity_b],
                "reason": "Directory not found"
            })
            continue

        # Determine canonical (more facts wins)
        facts_a = len([f for f in dir_a.iterdir() if f.is_file() and f.suffix == ".md"]) if dir_a.exists() else 0
        facts_b = len([f for f in dir_b.iterdir() if f.is_file() and f.suffix == ".md"]) if dir_b.exists() else 0

        if facts_a >= facts_b:
            canonical, duplicate = clean_a, clean_b
            canonical_dir, duplicate_dir = dir_a, dir_b
        else:
            canonical, duplicate = clean_b, clean_a
            canonical_dir, duplicate_dir = dir_b, dir_a

        print(f"  Canonical: {canonical} ({max(facts_a, facts_b)} facts)")
        print(f"  Duplicate: {duplicate} ({min(facts_a, facts_b)} facts)")

        # Merge files
        print(f"  Merging files from {duplicate} to {canonical}...")
        merged = merge_entity_files(duplicate_dir, canonical_dir, duplicate, canonical)

        if merged:
            print(f"  Merged {len(merged)} file(s)")
            merge_log["merges"].append({
                "canonical": canonical,
                "duplicate": duplicate,
                "files_merged": len(merged),
                "file_details": merged
            })
        else:
            print(f"  No files to merge (duplicate was empty)")
            merge_log["merges"].append({
                "canonical": canonical,
                "duplicate": duplicate,
                "files_merged": 0,
                "note": "Duplicate was empty"
            })

        # Remove empty duplicate directory
        try:
            # Remove any remaining files
            for item in duplicate_dir.iterdir():
                if item.is_file():
                    item.unlink()

            # Remove directory if empty
            if not any(duplicate_dir.iterdir()):
                duplicate_dir.rmdir()
                print(f"  Removed empty directory: {duplicate}")
        except Exception as e:
            print(f"  Warning: Could not remove directory: {e}")
            merge_log["errors"].append({
                "entities": [entity_a, entity_b],
                "reason": f"Could not remove directory: {e}"
            })

        processed += 1

    # Save merge log
    merge_log_file = Path("/tmp/entity_merge_log.json")
    with open(merge_log_file, "w") as f:
        json.dump(merge_log, f, indent=2)

    print(f"\n{'=' * 60}")
    print("MERGE SUMMARY")
    print("=" * 60)
    print(f"Processed: {processed} merges")
    print(f"Successful: {len([m for m in merge_log['merges'] if m.get('files_merged', 0) > 0])}")
    print(f"Errors: {len(merge_log['errors'])}")
    print(f"\nMerge log saved to: {merge_log_file}")

    return merge_log

if __name__ == "__main__":
    main()
