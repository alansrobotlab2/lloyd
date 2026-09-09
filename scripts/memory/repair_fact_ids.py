#!/usr/bin/env python3
"""Renumber colliding fact IDs in a fact tree.

An ID is only meaningful if it names one fact. `fact_invalidate`, `fact_resolve`
and every revert report address a fact by `<file, id>`, and when that pair
matches twice they act on whichever the reader happens to find first.

Two writers created the collisions. The extractor used to restart its numbering
at 1 on every run (fixed 2026-09-03 by `app.fact_ids`), and every entity **merge**
concatenates two independently-numbered `facts:` lists — `_merge_facts_lists`
dedups by fact text and cannot see an ID (fixed 2026-09-08 by `dedupe_ids`).
Neither fix repaired the files already written, which is what this does.

The first holder keeps the ID: it is the one any outside record already names.
Later collisions get the next free ID for their category. No fact is dropped and
no text is touched — only the `id` field of a fact that was sharing one.

    python scripts/memory/repair_fact_ids.py                     # dry run, live tree
    python scripts/memory/repair_fact_ids.py --apply
    python scripts/memory/repair_fact_ids.py --facts-dir ... --db ... --apply
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LLOYD = HERE.parent.parent
sys.path.insert(0, str(LLOYD))

from app.paths import VAULT_FACTS_ROOT, VAULT_KG_DB  # noqa: E402
from app.fact_ids import dedupe_ids  # noqa: E402
from app.atomic_io import atomic_write_text, locked_file  # noqa: E402
from agent_mcp.facts import _parse_fact_frontmatter, _write_fact_frontmatter  # noqa: E402


def _duplicate_pairs(db: Path) -> int:
    from app.kg_store import KGStore
    st = KGStore(db)
    try:
        return st._query(
            "SELECT COUNT(*) n FROM (SELECT file_path, fact_id FROM facts_idx "
            "WHERE fact_id IS NOT NULL GROUP BY file_path, fact_id HAVING COUNT(*) > 1)"
        )[0]["n"]
    finally:
        st.close()


def _collisions(facts: list) -> int:
    seen, n = set(), 0
    for f in facts:
        if not isinstance(f, dict) or not f.get("id"):
            continue
        if f["id"] in seen:
            n += 1
        else:
            seen.add(f["id"])
    return n


def _repair_one(path: Path) -> int:
    """Re-read under the lock and rewrite. Returns how many IDs moved.

    The read has to happen inside `locked_file`, not before it: the extractor
    runs four worker threads and `fact_add` fires from chat turns, and both do
    read-modify-write on these files. A repair that read first and wrote after
    would drop whatever landed in between -- which is the very failure the lock
    exists to prevent, committed by the tool that is meant to be fixing them.
    """
    with locked_file(path):
        raw = path.read_text(encoding="utf-8")
        fm = _parse_fact_frontmatter(raw)
        if not fm:
            return 0
        facts = fm.get("facts")
        if not isinstance(facts, list):
            return 0
        moved = dedupe_ids(facts, fm.get("category"))
        if moved:
            body = raw.split("---", 2)[-1]
            atomic_write_text(path, _write_fact_frontmatter(fm) + body)
        return moved


def repair(facts_dir: Path, apply: bool) -> dict:
    files = renumbered = scanned = 0
    changed: list[dict] = []
    for path in sorted(facts_dir.rglob("*.md")):
        if path.name.endswith(".lock"):
            continue
        scanned += 1
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            continue
        fm = _parse_fact_frontmatter(raw)
        if not fm:
            continue
        facts = fm.get("facts")
        if not isinstance(facts, list) or len(facts) < 2:
            continue
        # Survey unlocked -- cheap, and only decides which files to open again.
        collisions = _collisions(facts)
        if not collisions:
            continue
        if apply:
            # Re-counted under the lock; the file may have changed since.
            collisions = _repair_one(path)
            if not collisions:
                continue
        files += 1
        renumbered += collisions
        changed.append({"file": str(path.relative_to(facts_dir)), "renumbered": collisions,
                        "facts": len(facts)})
    return {"scanned": scanned, "files": files, "renumbered": renumbered, "changed": changed}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--facts-dir", type=Path, default=VAULT_FACTS_ROOT)
    ap.add_argument("--db", type=Path, default=VAULT_KG_DB)
    ap.add_argument("--apply", action="store_true", help="write; default is a dry run")
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--no-reindex", action="store_true",
                    help="skip the reindex (the index still holds the old IDs)")
    args = ap.parse_args()

    if not args.facts_dir.is_dir():
        print(f"no fact tree at {args.facts_dir}", file=sys.stderr)
        return 2

    before = _duplicate_pairs(args.db) if args.db.exists() else None
    print(f"tree: {args.facts_dir}\ndb:   {args.db}")
    if before is not None:
        print(f"duplicate (file, fact_id) pairs before: {before:,}")

    res = repair(args.facts_dir, args.apply)
    print(f"scanned {res['scanned']:,} files; "
          f"{res['renumbered']:,} colliding IDs across {res['files']:,} files")
    for c in res["changed"][:10]:
        print(f"    {c['file']}  ({c['renumbered']} of {c['facts']} facts)")
    if len(res["changed"]) > 10:
        print(f"    … and {len(res['changed']) - 10} more")

    if not args.apply:
        print("\n(dry run — pass --apply to write)")
        return 0

    if not args.no_reindex and args.db.exists():
        from app.kg_store import KGStore
        st = KGStore(args.db)
        try:
            st.facts_idx.reindex(root=args.facts_dir)
            print(f"reindexed: {st.stats()}")
        finally:
            st.close()
        after = _duplicate_pairs(args.db)
        print(f"duplicate (file, fact_id) pairs after: {after:,}")
        res["after"] = after
    res["before"] = before

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
