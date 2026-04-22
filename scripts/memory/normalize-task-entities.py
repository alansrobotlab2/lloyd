#!/usr/bin/env python3
"""Task-number entity normalizer.

Canonicalizes every task-like entity dir to `Task #N`. Merges facts from
duplicate/variant dirs into the canonical, rewrites edges in
`_relationships.json`, and updates `entity-aliases.json`.

Conventions recognised and migrated (per backlog #312 evidence):
    Task #N, Task N, Task_N, backlog_N, backlog_item_N, backlog_task_N

Bare numeric dirs (e.g. `235`) are only migrated when another task-form for
the same N already exists — otherwise the number might not be a task.

Usage:
    python scripts/memory/normalize-task-entities.py            # dry run
    python scripts/memory/normalize-task-entities.py --apply    # mutate
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

FACTS = Path("/home/alansrobotlab/obsidian/facts")
RELATIONSHIPS = FACTS / "_relationships.json"
ALIASES = FACTS / "_aliases.json"

# Priority: the higher the index, the more we prefer this form's body as
# the "primary" when merging (filenames, metadata, etc.). `hash` is already
# canonical so it beats everything.
CONVENTION_PRIORITY = {
    "hash": 6,
    "space": 5,
    "underscore": 4,
    "backlog": 3,
    "backlog_task": 2,
    "backlog_item": 1,
    "bare": 0,
}


def parse_task_entity(name: str):
    """Return (task_id:int, convention:str) or (None, None)."""
    if m := re.match(r"^Task\s*#(\d{1,4})$", name):
        return int(m.group(1)), "hash"
    if m := re.match(r"^Task\s+(\d{1,4})$", name):
        return int(m.group(1)), "space"
    if m := re.match(r"^Task_(\d{1,4})$", name):
        return int(m.group(1)), "underscore"
    if m := re.match(r"^backlog_task[_-]?(\d{1,4})$", name, re.I):
        return int(m.group(1)), "backlog_task"
    if m := re.match(r"^backlog_item[_-]?(\d{1,4})$", name, re.I):
        return int(m.group(1)), "backlog_item"
    if m := re.match(r"^backlog[_-](\d{1,4})$", name, re.I):
        return int(m.group(1)), "backlog"
    if re.match(r"^\d{1,4}$", name):
        return int(name), "bare"
    return None, None


def build_plan():
    groups: dict[int, list[tuple[str, str]]] = defaultdict(list)
    for entry in FACTS.iterdir():
        if not entry.is_dir():
            continue
        tid, conv = parse_task_entity(entry.name)
        if tid is None:
            continue
        groups[tid].append((entry.name, conv))

    plan = []
    for tid in sorted(groups):
        variants = groups[tid]
        # Bare-only group: skip (we can't be sure it's a task).
        if all(conv == "bare" for _, conv in variants):
            continue
        canonical = f"Task #{tid}"
        variants.sort(key=lambda x: -CONVENTION_PRIORITY[x[1]])
        # Primary = the variant we rename to canonical (or canonical itself
        # if it already exists).
        primary = canonical if any(n == canonical for n, _ in variants) else variants[0][0]
        others = [name for name, _ in variants if name != primary]
        if primary == canonical and not others:
            # Already canonical, nothing to do.
            continue
        plan.append({
            "task_id": tid,
            "canonical": canonical,
            "primary": primary,
            "others": others,
        })
    return plan


def count_facts(entity_dir: Path) -> int:
    return len(list(entity_dir.glob("*.md")))


def count_edges(rel_data: dict, entity: str) -> int:
    return sum(
        1 for edge in rel_data.get("edges", [])
        if edge.get("source") == entity or edge.get("target") == entity
    )


def migrate_dir(src: Path, dst: Path, old_prefix: str, new_prefix: str, apply: bool):
    """Move every file in src into dst, renaming `{old_prefix}-…` to
    `{new_prefix}-…`. On filename collision, append `-dupeN` to the incoming
    basename."""
    moved = []
    for f in src.iterdir():
        if not f.is_file():
            continue
        new_name = f.name
        if new_name.startswith(f"{old_prefix}-"):
            new_name = f"{new_prefix}-" + new_name[len(old_prefix) + 1:]
        target = dst / new_name
        if target.exists():
            base, ext = target.stem, target.suffix
            n = 2
            while (dst / f"{base}-dupe{n}{ext}").exists():
                n += 1
            target = dst / f"{base}-dupe{n}{ext}"
        moved.append((f, target))
        if apply:
            shutil.move(str(f), str(target))
    if apply:
        try:
            src.rmdir()
        except OSError as e:
            print(f"    WARN: could not remove {src} — {e}", file=sys.stderr)
    return moved


def rewrite_edges(rel_data: dict, renames: dict[str, str]) -> int:
    rewrites = 0
    for edge in rel_data.get("edges", []):
        if edge.get("source") in renames:
            edge["source"] = renames[edge["source"]]
            rewrites += 1
        if edge.get("target") in renames:
            edge["target"] = renames[edge["target"]]
            rewrites += 1
    # Dedupe identical edges produced by the rewrite.
    seen = set()
    deduped = []
    dedup_count = 0
    for edge in rel_data.get("edges", []):
        key = (
            edge.get("source"),
            edge.get("target"),
            edge.get("type"),
            edge.get("valid_at"),
        )
        if key in seen:
            dedup_count += 1
            continue
        seen.add(key)
        deduped.append(edge)
    rel_data["edges"] = deduped
    return rewrites, dedup_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    plan = build_plan()
    print(f"=== Task-number normalization plan ({len(plan)} groups) ===\n")

    rel_data = json.loads(RELATIONSHIPS.read_text(encoding="utf-8"))
    aliases = json.loads(ALIASES.read_text(encoding="utf-8")) if ALIASES.exists() else {}

    renames: dict[str, str] = {}
    for item in plan:
        renames[item["primary"]] = item["canonical"]
        for other in item["others"]:
            renames[other] = item["canonical"]
        # Trim self-rename (canonical→canonical) — happens when primary is
        # already canonical.
        if renames.get(item["canonical"]) == item["canonical"]:
            del renames[item["canonical"]]

    # Print the plan
    for item in plan:
        tid = item["task_id"]
        canonical = item["canonical"]
        primary = item["primary"]
        others = item["others"]
        p_dir = FACTS / primary
        p_facts = count_facts(p_dir) if p_dir.exists() else 0
        p_edges = count_edges(rel_data, primary)
        action = "rename" if primary != canonical else "(canonical exists)"
        print(f"  #{tid} canonical={canonical!r}")
        print(f"    primary: {primary!r} ({p_facts} facts, {p_edges} edges) — {action}")
        for other in others:
            o_dir = FACTS / other
            o_facts = count_facts(o_dir) if o_dir.exists() else 0
            o_edges = count_edges(rel_data, other)
            print(f"    merge:   {other!r} ({o_facts} facts, {o_edges} edges)")

    print(f"\n=== Summary ===")
    print(f"  entity renames:  {len(renames)}")
    print(f"  affected tasks:  {len(plan)}")

    if not args.apply:
        print("\n  (dry run) — pass --apply to mutate")
        return

    print("\n=== Applying ===\n")
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    # Back up
    shutil.copy(RELATIONSHIPS, FACTS / f"_relationships.{ts}.pre-taskrename.bak.json")
    if ALIASES.exists():
        shutil.copy(ALIASES, FACTS / f"entity-aliases.{ts}.pre-taskrename.bak")
    print(f"  backup timestamp: {ts}")

    # Step 1: for each plan group, migrate primary + others → canonical dir
    for item in plan:
        canonical = item["canonical"]
        primary = item["primary"]
        others = item["others"]
        c_dir = FACTS / canonical
        p_dir = FACTS / primary

        if primary == canonical:
            # canonical dir exists, nothing to do for primary itself
            pass
        elif not c_dir.exists():
            # Simple rename
            p_dir.rename(c_dir)
            # Fix file prefixes inside
            for f in c_dir.iterdir():
                if f.is_file() and f.name.startswith(f"{primary}-"):
                    new_name = f"{canonical}-" + f.name[len(primary) + 1:]
                    f.rename(c_dir / new_name)
            print(f"  renamed: {primary!r} → {canonical!r}")
        else:
            # Canonical dir already exists (shouldn't happen because we'd
            # have picked it as primary), but handle safely.
            migrate_dir(p_dir, c_dir, primary, canonical, apply=True)
            print(f"  merged:  {primary!r} → {canonical!r}")

        for other in others:
            o_dir = FACTS / other
            if o_dir.exists():
                migrate_dir(o_dir, c_dir, other, canonical, apply=True)
                print(f"  merged:  {other!r} → {canonical!r}")

    # Step 2: rewrite edges
    rewrites, deduped = rewrite_edges(rel_data, renames)
    RELATIONSHIPS.write_text(json.dumps(rel_data, indent=2) + "\n", encoding="utf-8")
    print(f"\n  edges rewritten: {rewrites}, deduped: {deduped}")

    # Step 3: update aliases
    alias_adds = 0
    for old, new in renames.items():
        if aliases.get(old) != new:
            aliases[old] = new
            alias_adds += 1
        # Lowercase version too (memory.py checks both)
        if aliases.get(old.lower()) != new:
            aliases[old.lower()] = new
            alias_adds += 1
    ALIASES.write_text(json.dumps(aliases, indent=2, sort_keys=True), encoding="utf-8")
    print(f"  alias entries added/updated: {alias_adds}")

    print(f"\n  DONE. Backups at .pre-taskrename.bak (ts={ts})")


if __name__ == "__main__":
    main()
