#!/usr/bin/env python3
"""Backfill the `provenance` field onto existing facts in ~/obsidian/facts/.

Provenance rules (applied only to facts that lack the field):
  - EXTRACTED  if source_doc contains "http" or "arxiv"
  - INFERRED   if confidence < 0.7
  - STATED     otherwise (source_doc is empty/null/local path)

Usage:
  python fact-provenance-backfill.py           # dry-run (default)
  python fact-provenance-backfill.py --apply   # actually write changes
"""

import argparse
import os
import re
import sys
from pathlib import Path

import yaml


FACTS_DIR = Path.home() / "obsidian" / "facts"

# Regex to split frontmatter from body.  Matches the opening and closing "---".
FRONTMATTER_RE = re.compile(r"^---\n(.*?\n)---\n?(.*)", re.DOTALL)


def determine_provenance(fact: dict) -> str:
    """Return the provenance label for a single fact dict."""
    confidence = fact.get("confidence")
    if confidence is not None and confidence < 0.7:
        return "INFERRED"

    source_doc = fact.get("source_doc")
    if source_doc and isinstance(source_doc, str):
        lower = source_doc.lower()
        if "http" in lower or "arxiv" in lower:
            return "EXTRACTED"

    return "STATED"


def process_file(path: Path, *, apply: bool) -> dict:
    """Process a single markdown fact file.

    Returns a stats dict: {updated: int, skipped: int, error: str | None}
    """
    stats = {"updated": 0, "skipped": 0, "error": None}

    try:
        raw = path.read_text(encoding="utf-8")
    except Exception as e:
        stats["error"] = str(e)
        return stats

    m = FRONTMATTER_RE.match(raw)
    if not m:
        stats["error"] = "no frontmatter found"
        return stats

    fm_text = m.group(1)
    body = m.group(2)

    try:
        data = yaml.safe_load(fm_text)
    except yaml.YAMLError as e:
        stats["error"] = f"YAML parse error: {e}"
        return stats

    if not isinstance(data, dict):
        stats["error"] = "frontmatter is not a mapping"
        return stats

    facts = data.get("facts")
    if not facts or not isinstance(facts, list):
        # No facts list -- nothing to do
        return stats

    changed = False
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        if "provenance" in fact:
            stats["skipped"] += 1
            continue
        fact["provenance"] = determine_provenance(fact)
        stats["updated"] += 1
        changed = True

    if changed and apply:
        new_fm = yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)
        new_content = f"---\n{new_fm}---\n{body}"
        path.write_text(new_content, encoding="utf-8")

    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true", default=True, help="Show what would change without writing (default)")
    group.add_argument("--apply", action="store_true", help="Actually write changes to disk")
    args = parser.parse_args()

    apply = args.apply

    if not FACTS_DIR.is_dir():
        print(f"ERROR: facts directory not found: {FACTS_DIR}", file=sys.stderr)
        sys.exit(1)

    mode = "APPLY" if apply else "DRY RUN"
    print(f"=== Fact Provenance Backfill ({mode}) ===")
    print(f"Scanning: {FACTS_DIR}\n")

    files_processed = 0
    files_modified = 0
    total_updated = 0
    total_skipped = 0
    errors = []

    for md_path in sorted(FACTS_DIR.rglob("*.md")):
        if not md_path.is_file():
            continue

        stats = process_file(md_path, apply=apply)
        files_processed += 1

        if stats["error"]:
            errors.append((md_path, stats["error"]))
            continue

        if stats["updated"] > 0:
            files_modified += 1
            rel = md_path.relative_to(FACTS_DIR)
            print(f"  {'WRITE' if apply else 'WOULD UPDATE'}: {rel}  "
                  f"({stats['updated']} facts updated, {stats['skipped']} already had provenance)")

        total_updated += stats["updated"]
        total_skipped += stats["skipped"]

    print(f"\n--- Summary ---")
    print(f"Files scanned:          {files_processed}")
    print(f"Files {'modified' if apply else 'to modify'}:         {files_modified}")
    print(f"Facts updated:          {total_updated}")
    print(f"Facts already had prov: {total_skipped}")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for p, msg in errors:
            print(f"  {p.relative_to(FACTS_DIR)}: {msg}")

    if not apply and total_updated > 0:
        print(f"\nRe-run with --apply to write changes.")


if __name__ == "__main__":
    main()
