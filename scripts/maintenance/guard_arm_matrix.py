#!/usr/bin/env python3
"""Print the guard-by-dispatch-path matrix, derived from the tree (#1963).

    python scripts/maintenance/guard_arm_matrix.py            # this checkout
    python scripts/maintenance/guard_arm_matrix.py --root ~/lloyd

One row per production dispatch path that builds a turn, naming the hook-installed
guards it can arm. This is what `architecture/guard-coverage.md`'s guard-by-path
section tells a reader to run. Reads syntax only — no turn is built, nothing is
imported from the tree it inspects beyond this repo's own AST helpers. Exit 0
when the table printed; 1 when no dispatch path was found, since an empty table
from a scan that matched nothing is not a clean result.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main(argv: list[str] | None = None) -> int:
    from app.harness import guard_arm_matrix as G

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=str(REPO),
                        help="the checkout to read (default: this one)")
    parser.add_argument("--capabilities", action="store_true",
                        help="also print each worker source's capability "
                             "envelope and what reaches beyond it (#2269)")
    args = parser.parse_args(argv)
    matrix = G.guard_arm_matrix(Path(args.root).expanduser())
    if not matrix:
        print(f"no dispatch path found under {args.root}", file=sys.stderr)
        return 1
    print(G.render(matrix))
    if args.capabilities:
        # #2269, on this report because this report is the one that already
        # asks "which guards are armed on which dispatch path" — the reach
        # question is asked of the same paths, and a source reaching a durable
        # tool outside its declaration should turn up where an unarmed registry
        # turns up, not in a report nobody reads.
        #
        # Behind a flag, and reading the envelopes from the tree this script was
        # IMPORTED from while the guard columns are read from `--root`: a
        # capability set is a set of tool names, so answering the reach question
        # statically would mean importing a source module from a checkout that
        # is not the running one. An empty `--root` therefore reports the guard
        # columns of that tree against this one's declarations, and says so.
        caps = G.capability_matrix(Path(args.root).expanduser())
        print(f"\nworker sources with a capability envelope: {len(caps)}"
              f" (declared sets read from {REPO})\n")
        print(G.render_capabilities(caps))
        found = G.capability_findings(Path(args.root).expanduser())
        print(f"\ncapability findings: {len(found)}")
        for f in found:
            print(f"  [{f.reason}] {f.target}: {f.extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
