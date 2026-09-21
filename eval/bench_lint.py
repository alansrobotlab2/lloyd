"""CLI entry point for the bench validity lint (#646).

The implementation is ``scripts/autoresearch/bench_lint.py`` — this is the
``eval/``-side handle the item named, so a nightly job or a CI step calls the
same code the round calls, over the same ``cfg.paths.bench_dir``.

    python eval/bench_lint.py            # markdown report, exit 0
    python eval/bench_lint.py --strict   # exit 1 while any task is lint-invalid
    python eval/bench_lint.py --json     # machine-readable

``--strict`` is deliberately NOT the default. The lint flags and a person
decides (#646's deferred clause): the live bench currently has lint-invalid
tasks whose checks a human has not yet tightened or retired, so a strict
default would make every nightly red for a state the board already records.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.autoresearch.bench_lint import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
