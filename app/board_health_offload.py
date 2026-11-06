"""`backlog.board_health` in a child process, for the dashboard.

`board_health` takes ~10-13 s of pure-Python work per call (front matter for
every backlog item, twice; the ledger re-sorted per round; the owed checks).
The dashboard calls it once a scorecard TTL from a thread inside the backend,
and a thread there holds the GIL against the event loop: a py-spy profile on
2026-11-05 put 47% of the backend's samples in it while `/health` stalled for
up to 5 s. The same call in a child process costs the same CPU on another
core and none of the loop's time.

Run as `python -m app.board_health_offload <ledger> <backlog_dir>`; prints
the result as JSON. `run()` is the parent side and returns None on any
failure, like the in-process call it replaces.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

#: Generous: the call measured 9.7-12.8 s on 2026-11-05 and grows with the board.
TIMEOUT_S = 120.0


def run(ledger: Path, backlog_dir: Path, *, timeout: float = TIMEOUT_S) -> dict[str, Any] | None:
    """`board_health(ledger, backlog_dir=backlog_dir)` computed in a child."""
    from app.paths import LLOYD_HOME
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "app.board_health_offload", str(ledger), str(backlog_dir)],
            cwd=str(LLOYD_HOME), capture_output=True, text=True, timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    try:
        out = json.loads(proc.stdout)
    except ValueError:
        return None
    return out if isinstance(out, dict) else None


def _main(argv: list[str]) -> int:
    from scripts.automod import backlog as B
    ledger, backlog_dir = Path(argv[0]), Path(argv[1])
    if not backlog_dir.exists():
        print("null")
        return 0
    print(json.dumps(B.board_health(ledger, backlog_dir=backlog_dir), default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
