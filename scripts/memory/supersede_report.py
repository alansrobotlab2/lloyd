#!/usr/bin/env python3
"""How many active facts come from an EARLIER version of their note. Read-only.

For every vault note the facts index names as a `source_doc`, compares the
`source_hash` each active fact was extracted under with the hash of the note's
bytes on disk now, and reports the rows on an old hash — the population
`knowledge_graph.supersede_on_reextract` (agent_mcp/fact_supersede.py) would
judge. Notes matching the config's `exempt_globs` are counted separately, and a
note that no longer exists is its own line (its facts cannot be "superseded";
they have no current bytes at all).

Reads through `app.kg_store` (the store's own read API); writes nothing.

    .venvs/lloyd/bin/python scripts/memory/supersede_report.py [--top 25] [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_mcp.fact_supersede import is_exempt, load_config  # noqa: E402
from app.atomic_io import hash_bytes  # noqa: E402


def build_report(rows: list[dict], vault_root: Path, exempt_globs) -> dict:
    by_doc: dict[str, dict[str, int]] = defaultdict(dict)
    for r in rows:
        by_doc[r["source_doc"]][r["source_hash"]] = int(r["n"])
    notes = []
    totals = {"notes": 0, "active": 0, "on_old_hash": 0, "exempt_on_old_hash": 0,
              "missing_note_rows": 0, "missing_notes": 0}
    for doc, hashes in by_doc.items():
        active = sum(hashes.values())
        totals["notes"] += 1
        totals["active"] += active
        path = vault_root / doc
        try:
            current = hash_bytes(path.read_bytes())
        except OSError:
            totals["missing_notes"] += 1
            totals["missing_note_rows"] += active
            continue
        old = sum(n for h, n in hashes.items() if h != current)
        if not old:
            continue
        exempt = is_exempt(doc, exempt_globs)
        totals["exempt_on_old_hash" if exempt else "on_old_hash"] += old
        notes.append({"source_doc": doc, "active": active, "on_old_hash": old,
                      "old_hashes": sum(1 for h in hashes if h != current),
                      "current_hash_present": current in hashes, "exempt": exempt})
    notes.sort(key=lambda n: -n["on_old_hash"])
    return {"totals": totals, "notes": notes}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    from app.kg_store import StoreUnavailable, store
    from app.paths import VAULT_ROOT
    try:
        rows = store().facts_idx.active_counts_by_source_hash()
    except StoreUnavailable as exc:
        print(f"store unavailable: {exc}", file=sys.stderr)
        return 2
    cfg = load_config()
    report = build_report(rows, Path(VAULT_ROOT), cfg.exempt_globs)
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    t = report["totals"]
    print(f"mode={cfg.mode}  notes={t['notes']}  active={t['active']}")
    print(f"on an old hash: {t['on_old_hash']} (non-exempt)  "
          f"+ {t['exempt_on_old_hash']} (exempt feeds)")
    print(f"notes gone from the vault: {t['missing_notes']} holding {t['missing_note_rows']} active rows")
    print()
    for n in [n for n in report["notes"] if not n["exempt"]][:args.top]:
        print(f"{n['on_old_hash']:>7}/{n['active']:<7} {n['old_hashes']:>3} old hash(es)  "
              f"{'' if n['current_hash_present'] else '[current bytes never extracted] '}"
              f"{n['source_doc']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
