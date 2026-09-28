#!/usr/bin/env python3
"""Scan the session store for fabricated reasoning traces (#1510).

    .venvs/lloyd/bin/python -m scripts.thinking_fidelity_scan
    .venvs/lloyd/bin/python -m scripts.thinking_fidelity_scan --list
    .venvs/lloyd/bin/python -m scripts.thinking_fidelity_scan --root /path/to/sessions
    .venvs/lloyd/bin/python -m scripts.thinking_fidelity_scan --record

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

`--record` (#1656) is the persistence half. #1510's ruling says the guard is to
be gated on "the marker rate falling", and a rate needs a series: until now the
number above existed only in the terminal that printed it, so the question the
ruling asks had no after to compare its before against — the same gap
`scripts/iv_metrics_record.py` closed for the IV metrics. One compact JSON row
per run is appended to `_pipeline/metrics/thinking-fidelity-prevalence.jsonl`
under the production data root (`--series` overrides it, and the tests do), and
printed, so an unattended run can quote the line the way the IV recorder's job
is told to quote its breach line.

The row carries its numerator AND its denominator — `flagged` beside `blocks`,
`files_with_flags` beside `files`, `exempt_blocks` on a field of its own, and
`scanned`, which is "at least one block was read" — because a prevalence series
whose zero is ambiguous is worse than no series: the first reader of
`flagged: 0` asks whether the store was clean or empty, and only `blocks` can
answer. So `flagged_rate` is `null` on a zero-block run rather than `0.0`, the
same distinction the exit codes draw (3, not 0).

The verdict is lexical (see `app/thinking_fidelity.py`). What an operator does
with the list — re-score, exclude from Inner Voice and judge inputs, or declare
the recorded reasoning untrustworthy until re-scored — is deliberately not this
script's call. What #1656 did settle is one of those uses: the IV/judge
exclusion lives in `app/inner_voice/session_input.py`, which asks
`app.thinking_fidelity` the same question this script asks the store.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from app.data_root import production_data_root  # noqa: E402
from app.thinking_fidelity import StoreScan, scan_store  # noqa: E402

#: Where the prevalence series lives, relative to the production data root —
#: beside the other measured series (`_pipeline/metrics/kg-health-*.json`).
PREVALENCE_RELATIVE = (
    Path("_pipeline") / "metrics" / "thinking-fidelity-prevalence.jsonl")


def prevalence_row(scan: StoreScan) -> dict:
    """The one machine-readable row a scan produces, denominator and all.

    Field by field the same numbers `StoreScan.report()` prints, so the prose
    and the series cannot disagree about what a run measured. `scanned` and the
    `null` rate are the anti-confusion pair: a run over an empty store is
    `{scanned: false, blocks: 0, flagged_rate: null}` and a clean store is
    `{scanned: true, blocks: 40964, flagged: 0, flagged_rate: 0.0}`.
    """
    now = datetime.now(timezone.utc)
    return {
        "date": now.date().isoformat(),
        "recorded_at": now.isoformat(timespec="seconds"),
        "root": str(scan.root),
        "files": scan.files,
        "files_with_flags": scan.files_with_flags,
        "blocks": scan.blocks,
        "flagged": scan.flagged,
        "exempt_blocks": scan.blocks_exempt_meta,
        "exempt_files": scan.files_exempt_meta,
        "unreadable": scan.unreadable,
        "scanned": scan.blocks > 0,
        "flagged_rate": (round(scan.flagged / scan.blocks, 6)
                         if scan.blocks else None),
    }


def record_prevalence(row: dict, series: Path) -> Path:
    """Append the row as one JSON line and return the file it went into."""
    series.parent.mkdir(parents=True, exist_ok=True)
    with series.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    return series


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="",
                    help="session store to scan (default: the production data "
                         "root's sessions/ dir)")
    ap.add_argument("--list", action="store_true",
                    help="also print one line per session file with a flagged "
                         "block, oldest first")
    ap.add_argument("--record", action="store_true",
                    help="print the prevalence row as one JSON line and append "
                         "it to the prevalence series")
    ap.add_argument("--series", default="",
                    help=f"series file to append to (default: <data root>/"
                         f"{PREVALENCE_RELATIVE})")
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

    if args.record:
        row = prevalence_row(scan)
        series = (Path(args.series).expanduser() if args.series
                  else production_data_root() / PREVALENCE_RELATIVE)
        record_prevalence(row, series)
        print(f"recorded: {json.dumps(row, sort_keys=True)}")
        print(f"  appended to {series}")

    if scan.blocks == 0:
        print(f"  {scan.files} session file(s) held no reasoning block at all: "
              "this scan could not discriminate either way", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
