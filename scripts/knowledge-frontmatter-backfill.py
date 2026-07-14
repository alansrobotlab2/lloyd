#!/usr/bin/env python3
"""
Backfill missing frontmatter fields on existing knowledge pages.

Infers source_type, domain, last_synthesized from existing page content.
Non-destructive: only ADDS missing fields, never overwrites existing ones.

Run: python3 ~/lloyd/scripts/knowledge-frontmatter-backfill.py [--dry-run]
"""

import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

KNOWLEDGE_DIR = Path(os.path.expanduser("~/obsidian/knowledge"))
LOG_FILE = KNOWLEDGE_DIR / "_log.md"

# Skip these files
SKIP_FILES = {"KNOWLEDGE_SCHEMA.md", "_log.md", "idle-worker-tasks.md", "knowledge.md"}

# Type → source_type mapping
SYNTHESIZED_TYPES = {
    "deep-research", "medium-research", "research", "research-note",
    "synthesis", "quick-research",
}
CAPTURED_TYPES = {"stack-update"}
PRIMARY_TYPES = {"notes", "work-notes", "reference", "hub", "knowledge"}

# URL patterns indicating captured content
CAPTURED_URL_PATTERNS = [
    r"youtube\.com",
    r"youtu\.be",
    r"dev\.azure\.com",
    r"wiki\.aveva\.com",
]

DRY_RUN = "--dry-run" in sys.argv


def parse_frontmatter(content: str) -> tuple[dict | None, str, str]:
    """Parse YAML frontmatter from markdown content.

    Returns (frontmatter_dict, frontmatter_raw, body) or (None, '', content).
    """
    if not content.startswith("---"):
        return None, "", content

    end = content.find("\n---", 3)
    if end == -1:
        return None, "", content

    fm_raw = content[4:end]  # skip opening ---\n
    body = content[end + 4:]  # skip closing ---\n

    # Simple YAML parsing — enough for our frontmatter
    fm = {}
    current_key = None
    current_list = None

    for line in fm_raw.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # List item
        if stripped.startswith("- ") and current_key:
            if current_list is None:
                current_list = []
            current_list.append(stripped[2:].strip())
            fm[current_key] = current_list
            continue

        # Key-value
        if ":" in stripped:
            if current_list is not None:
                current_list = None

            colon_idx = stripped.index(":")
            key = stripped[:colon_idx].strip()
            value = stripped[colon_idx + 1:].strip()

            if value == "" or value == "|":
                current_key = key
                current_list = None if value == "|" else []
                fm[key] = current_list if current_list is not None else ""
            elif value.startswith("[") and value.endswith("]"):
                fm[key] = [v.strip().strip("'\"") for v in value[1:-1].split(",") if v.strip()]
                current_key = key
                current_list = None
            else:
                fm[key] = value.strip("'\"")
                current_key = key
                current_list = None

    return fm, fm_raw, body


def infer_source_type(fm: dict, body: str) -> str | None:
    """Infer source_type from existing frontmatter and content."""
    page_type = fm.get("type", "").lower()

    # Explicit type mappings
    if page_type in SYNTHESIZED_TYPES:
        return "synthesized"
    if page_type in CAPTURED_TYPES:
        return "captured"

    # Check for source URLs indicating capture
    source = fm.get("source", "")
    sources = fm.get("sources", [])
    all_sources = []
    if isinstance(source, str) and source:
        all_sources.append(source)
    if isinstance(sources, list):
        for s in sources:
            if isinstance(s, str):
                all_sources.append(s)
            elif isinstance(s, dict):
                all_sources.append(s.get("url", ""))

    # YouTube or external wiki → captured
    for src in all_sources:
        for pattern in CAPTURED_URL_PATTERNS:
            if re.search(pattern, src, re.IGNORECASE):
                return "captured"

    # Multiple research sources → synthesized
    if len(all_sources) >= 3:
        return "synthesized"

    # Has sources but few → could be either; if type is notes/reference → primary
    if page_type in PRIMARY_TYPES:
        return "primary"

    # Single external source → captured
    if len(all_sources) == 1 and ("http" in all_sources[0] or "www" in all_sources[0]):
        return "captured"

    # Default: if it has any external source, captured; else primary
    if all_sources:
        return "captured"

    return "primary"


def infer_domain(filepath: Path) -> str:
    """Infer domain from directory path."""
    rel = filepath.relative_to(KNOWLEDGE_DIR)
    parts = rel.parts
    if len(parts) > 1:
        return parts[0]
    return "general"


def add_frontmatter_fields(content: str, filepath: Path) -> tuple[str, list[str]]:
    """Add missing frontmatter fields. Returns (new_content, changes_list)."""
    fm, fm_raw, body = parse_frontmatter(content)
    changes = []

    if fm is None:
        # No frontmatter at all — skip, too risky to auto-add
        return content, ["SKIPPED: no frontmatter"]

    additions = []

    # source_type
    if "source_type" not in fm:
        inferred = infer_source_type(fm, body)
        if inferred:
            additions.append(f"source_type: {inferred}")
            changes.append(f"source_type: {inferred}")

    # domain
    if "domain" not in fm:
        domain = infer_domain(filepath)
        additions.append(f"domain: {domain}")
        changes.append(f"domain: {domain}")

    # segment (should always be 'knowledge')
    if "segment" not in fm:
        additions.append("segment: knowledge")
        changes.append("segment: knowledge")

    # last_synthesized (for synthesized pages, use file mtime)
    if "last_synthesized" not in fm:
        inferred_type = fm.get("source_type") or infer_source_type(fm, body)
        if inferred_type == "synthesized":
            mtime = datetime.fromtimestamp(filepath.stat().st_mtime, tz=timezone.utc)
            date_str = mtime.strftime("%Y-%m-%d")
            additions.append(f"last_synthesized: {date_str}")
            changes.append(f"last_synthesized: {date_str}")

    if not additions:
        return content, []

    # Insert additions before the closing ---
    new_fm_lines = fm_raw.rstrip().split("\n")
    for addition in additions:
        new_fm_lines.append(addition)

    new_content = "---\n" + "\n".join(new_fm_lines) + "\n---" + body
    return new_content, changes


def main():
    if DRY_RUN:
        print("=== DRY RUN MODE — no files will be modified ===\n")

    total = 0
    modified = 0
    skipped = 0
    all_changes = []

    for md_file in sorted(KNOWLEDGE_DIR.rglob("*.md")):
        if md_file.name in SKIP_FILES:
            continue
        # Skip source summary subdirs (immutable)
        if "sources" in md_file.parts:
            continue

        total += 1
        content = md_file.read_text(encoding="utf-8")
        new_content, changes = add_frontmatter_fields(content, md_file)

        if not changes:
            continue

        if "SKIPPED" in changes[0]:
            skipped += 1
            if DRY_RUN:
                rel = md_file.relative_to(KNOWLEDGE_DIR)
                print(f"  SKIP {rel}: {changes[0]}")
            continue

        modified += 1
        rel = md_file.relative_to(KNOWLEDGE_DIR)
        all_changes.append((rel, changes))

        if DRY_RUN:
            print(f"  {rel}:")
            for c in changes:
                print(f"    + {c}")
        else:
            md_file.write_text(new_content, encoding="utf-8")

    print(f"\n{'[DRY RUN] ' if DRY_RUN else ''}Results:")
    print(f"  Total pages scanned: {total}")
    print(f"  Modified: {modified}")
    print(f"  Skipped (no frontmatter): {skipped}")
    print(f"  Unchanged: {total - modified - skipped}")

    # Summary by source_type
    type_counts = {}
    for rel, changes in all_changes:
        for c in changes:
            if c.startswith("source_type:"):
                st = c.split(":")[1].strip()
                type_counts[st] = type_counts.get(st, 0) + 1

    if type_counts:
        print(f"\n  source_type distribution:")
        for st, count in sorted(type_counts.items()):
            print(f"    {st}: {count}")

    # Append to log if not dry run
    if not DRY_RUN and modified > 0:
        today = datetime.now().strftime("%Y-%m-%d")
        log_entry = f"## [{today}] lint | Frontmatter backfill — {modified} pages updated (source_type, domain, segment, last_synthesized)\n"
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_entry)
        print(f"\n  Appended log entry to {LOG_FILE}")


if __name__ == "__main__":
    main()
