#!/usr/bin/env python3
"""Scan the session store for fabricated reasoning traces (#1510).

    .venvs/lloyd/bin/python -m scripts.thinking_fidelity_scan
    .venvs/lloyd/bin/python -m scripts.thinking_fidelity_scan --list
    .venvs/lloyd/bin/python -m scripts.thinking_fidelity_scan --root /path/to/sessions

Prints the flagged count and the number of reasoning blocks scanned on the
same line, because "0 flagged" and "0 scanned" are otherwise the same string —
the mistake `app/uptake.py` records for a store that was not there at all. The
default root is the PRODUCTION data root read through
`app.data_root.production_data_root()`, not `app.paths.SESSIONS_DIR`: under the
test suite the latter is an empty scratch root, and a scan of it would cheerfully
report a clean store.

Exit codes: 0 a store was scanned (whatever it held), 2 the root does not exist,
3 the root exists but held no reasoning block to scan — the answer that must
never look like a pass.

The verdict is lexical (see `app/thinking_fidelity.py`). What an operator does
with the list — re-score, exclude from Inner Voice and judge inputs, or declare
the recorded reasoning untrustworthy until re-scored — is deliberately not this
script's call.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from app.data_root import production_data_root  # noqa: E402
from app.thinking_fidelity import scan_store  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="",
                    help="session store to scan (default: the production data "
                         "root's sessions/ dir)")
    ap.add_argument("--list", action="store_true",
                    help="also print one line per session file with a flagged "
                         "block, oldest first")
    args = ap.parse_args(argv)

    root = Path(args.root).expanduser() if args.root \
        else production_data_root() / "sessions"
    if not root.is_dir():
        print(f"thinking-trace fidelity: no session store at {root} — nothing "
              "was scanned, which is NOT a clean verdict", file=sys.stderr)
        return 2

    scan = scan_store(root)
    print(f"root: {root}")
    print(scan.report())
    if args.list:
        for name in scan.flagged_files:
            print(f"  flagged: {name}")

    if scan.blocks == 0:
        print(f"  {scan.files} session file(s) held no reasoning block at all: "
              "this scan could not discriminate either way", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
