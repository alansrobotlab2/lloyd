#!/usr/bin/env python3
"""Count entities whose active facts share one id across category files.

#874's acceptance check, kept runnable. `app/fact_ids.next_fact_id` numbers
`<prefix>-NNN` within one file, so `fact-001` is a handle inside a file and an
alias for several facts across an entity — which is why every fact-addressing
writer now names the file it writes in (`agent_mcp/facts.py::_apply_fact_marks`).
The backfill site stays `scripts/memory/repair_fact_ids.py`; until it runs, this
is the number that says how much of the corpus the scope rule is protecting.

Read-only. Exit 1 if the tree is missing, so a cron can tell a bad scan from a
clean one.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.paths import VAULT_FACTS_ROOT  # noqa: E402


def scan(root: Path, limit: int = 0) -> dict:
    dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    if limit:
        dirs = dirs[:limit]
    collide, sample, ids = [], [], 0
    for d in dirs:
        counts: collections.Counter = collections.Counter()
        for p in d.glob("*.md"):
            text = p.read_text(encoding="utf-8", errors="replace")
            end = text.find("\n---", 3)
            if not text.startswith("---") or end == -1:
                continue
            try:
                import yaml
                fm = yaml.safe_load(text[3:end]) or {}
            except Exception:      # noqa: BLE001 - an unparsable file is a gap, not a crash
                continue
            for f in fm.get("facts") or []:
                if (isinstance(f, dict) and f.get("id")
                        and not f.get("expired_at") and not f.get("invalid_at")):
                    counts[f["id"]] += 1
                    ids += 1
        shared = [i for i, n in counts.items() if n > 1]
        if shared:
            collide.append(d.name)
            if len(sample) < 5:
                sample.append({"entity": d.name, "id": shared[0],
                               "facts": counts[shared[0]]})
    return {"entities_scanned": len(dirs), "active_facts_seen": ids,
            "entities_with_shared_id": len(collide),
            "share": round(len(collide) / len(dirs), 4) if dirs else None,
            "worst_examples": sample, "root": str(root)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=400,
                    help="entity dirs to scan (0 = all; all takes minutes)")
    ap.add_argument("--root", default=None)
    args = ap.parse_args()
    root = Path(args.root) if args.root else VAULT_FACTS_ROOT
    if not root.is_dir():
        print(f"fact tree missing: {root}", file=sys.stderr)
        return 1
    print(json.dumps(scan(root, args.limit), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
