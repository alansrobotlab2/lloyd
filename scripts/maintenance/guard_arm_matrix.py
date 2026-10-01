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
    args = parser.parse_args(argv)
    matrix = G.guard_arm_matrix(Path(args.root).expanduser())
    if not matrix:
        print(f"no dispatch path found under {args.root}", file=sys.stderr)
        return 1
    print(G.render(matrix))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
