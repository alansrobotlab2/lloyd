#!/usr/bin/env python3
"""Scan the session store for fabricated reasoning traces (#1510).

    .venvs/lloyd/bin/python -m scripts.thinking_fidelity_scan
    .venvs/lloyd/bin/python -m scripts.thinking_fidelity_scan --list
    .venvs/lloyd/bin/python -m scripts.thinking_fidelity_scan --root /path/to/sessions
    .venvs/lloyd/bin/python -m scripts.thinking_fidelity_scan --record
    .venvs/lloyd/bin/python -m scripts.thinking_fidelity_scan --per-day --since 2026-09-26
    .venvs/lloyd/bin/python -m scripts.thinking_fidelity_scan --record --per-day

Prints the flagged count and the number of reasoning blocks scanned on the
same line, because "0 flagged" and "0 scanned" are otherwise the same string —
the mistake `app/uptake.py` records for a store that was not there at all. The
default root is the PRODUCTION data root read through
`app.data_root.production_data_root()`, not `app.paths.SESSIONS_DIR`: under the
test suite the latter is an empty scratch root, and a scan of it would cheerfully
report a clean store.

Exit codes: 0 a store was scanned (whatever it held), 2 the root does not exist,
3 the root exists but held no reasoning block to scan — the answer that must
never look like a pass. A `--since` that is not a `YYYY-MM-DD` date is argparse's
own exit 2 with the usage line, before any file is read.

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

`--per-day` (#1770) is the half `--record` could not answer, and it is the reason
the row above is not enough. A whole-store snapshot moves only when the aggregate
moves, and an aggregate over ~59,000 reasoning blocks is dominated by history:
new sessions falling to zero fabrication barely register in it, so the question
#1510's ruling asks — "is the marker rate falling?" — is a PER-DAY question and a
whole-store series cannot pose it. `--per-day` therefore rolls the store up by
the `YYYYMMDD` date prefix of each session filename (`20260926_101010_chat_1a2b
is 2026-09-26`) and prints one line per calendar day: day, flagged, blocks, rate.
`--since YYYY-MM-DD` bounds that roll-up to that date onward and says nothing
about the whole-store row beside it, which always covers everything. The grouping
is a filename read; every NUMBER in a day row is `app.thinking_fidelity`'s,
computed through the same `scan_file` call `scan_store` makes, so the e206f304
meta-session exemption is inherited rather than restated beside it.

Two properties a day series has to keep straight, and the fields that keep them:

  - **A day is still a row when it holds nothing scannable.** A day whose files
    hold no reasoning block comes back `blocks: 0, flagged_rate: null,
    scanned: false`, never omitted. A missing day in a day series reads as "the
    defect was quiet that day", which is the exact ambiguity the whole-store row
    above exists to kill, reproduced one level down.
  - **Day rows and the whole-store row share one file, so each names its own
    denominator.** `scope` is `"day"` (with `day: "YYYY-MM-DD"`, and `date` set
    to that day because that is what the row measures) or `"whole_store"` (with
    `date` the run date). Without it `flagged_rate: 0.0045` would mean two
    different things in the same series, and the rate read off the wrong row is
    the rate that lies.

The day rows and the whole-store row are two reads of a live store — the day rows
from one walk of the glob, the whole-store row from `scan_store`'s — so a session
being written while the scan runs can appear in one and not the other. Every row
carries its own `recorded_at` and its own `files`, so the disagreement is
visible rather than silent; the day rows are the ones the ruling reads.

`--record --per-day` together is what the daily autonomy task runs (its
description is the command line, since the task body reaches no worker), so the
series grows without anyone at a terminal — the gap #1770 was filed for.

The closed-day figures this shipped against, so a later reader can tell a moved
number from a wrong one: 2026-09-26 flagged 32 of 10413 blocks (0.31%) and
2026-09-27 flagged 54 of 11956 (0.45%), measured 2026-09-29. A day still being
written is a snapshot, not a constant — 2026-09-28 measured 10005 blocks while it
was open and 10324 after it closed — so compare day rows only across CLOSED days.

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
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from app.data_root import production_data_root  # noqa: E402
from app.thinking_fidelity import (  # noqa: E402
    StoreScan, scan_files, scan_store)

#: Where the prevalence series lives, relative to the production data root —
#: beside the other measured series (`_pipeline/metrics/kg-health-*.json`).
PREVALENCE_RELATIVE = (
    Path("_pipeline") / "metrics" / "thinking-fidelity-prevalence.jsonl")


#: What a row's denominator covers. Both kinds land in the one series file
#: (#1770), so a reader has to be able to tell a day's rate from the store's —
#: `flagged_rate: 0.0045` measured over one day and over ~59,000 blocks are
#: different facts about different populations, and they used to be
#: indistinguishable because only the whole-store kind existed.
SCOPE_DAY = "day"
SCOPE_WHOLE_STORE = "whole_store"

#: Session files are named `YYYYMMDD_HHMMSS_<kind>_<id>.json`, so the calendar
#: day a session belongs to is readable off its name and grouping needs no second
#: JSON parse. The prefix is validated as a real date, so a name like
#: `20261345_…json` is reported as undated instead of sorted into a day that does
#: not exist — the same refusal `--since` makes of a non-date.
SESSION_DAY_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})_")


def day_of(name: str) -> str:
    """The ISO calendar day a session filename names, or "" when it names none.

    "" is not a silent drop: `roll_up_by_day` collects those names and `main`
    prints how many there were, because a day roll-up whose denominator quietly
    excludes files it could not date is a rate over an unstated population.
    """
    hit = SESSION_DAY_RE.match(name)
    if not hit:
        return ""
    try:
        return date(int(hit.group(1)), int(hit.group(2)),
                    int(hit.group(3))).isoformat()
    except ValueError:                    # 20261345_ matches; 2026-13-45 does not
        return ""


def since_iso(value: str) -> str:
    """argparse `type=` for `--since`: a `YYYY-MM-DD` in, an ISO date out.

    Normalises through `date` rather than comparing the raw string, so
    `2026-9-26` and `2026-09-26` bound the same window and a value that is not a
    date fails at the flag instead of silently bounding the roll-up to nothing.

    `""` passes through: argparse applies `type=` to the DEFAULT as well as to a
    typed value, so an empty string is the unbounded window the flag carries when
    it is absent — not a malformed date.
    """
    if not value:
        return ""
    try:
        return datetime.strptime(value, "%Y-%m-%d").date().isoformat()
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--since wants a YYYY-MM-DD date, got {value!r}") from None


class DayRollup:
    """One `StoreScan` per calendar day, plus the files that are in no day row.

    A plain class and not a `@dataclass` on purpose: this module is loaded by
    `spec_from_file_location` from `tests/test_thinking_fidelity_scan_prevalence_row.py`
    without being put in `sys.modules`, and `@dataclass` resolving a string
    annotation looks its module up there — under 3.12 that dies with
    `'NoneType' object has no attribute '__dict__'` at import, before any test
    runs. A script that jobs import by path has to stay importable by a loader
    that registers nothing.
    """

    def __init__(self) -> None:
        self.days: dict[str, StoreScan] = {}
        self.undated: list[str] = []

    def report(self) -> str:
        """One line per day, flagged over blocks, oldest day first."""
        lines = []
        for day, scan in self.days.items():
            rate = (f"{scan.flagged / scan.blocks:.4f}" if scan.blocks
                    else "n/a (no reasoning block that day)")
            lines.append(
                f"per-day {day}: flagged {scan.flagged} of {scan.blocks} blocks "
                f"(rate {rate}; {scan.files} session file(s), "
                f"{scan.blocks_exempt_meta} exempt as meta-session)")
        if self.undated:
            lines.append(f"  {len(self.undated)} session file(s) have no "
                         "YYYYMMDD_ name prefix and are in NO day row (they are "
                         "still in the whole-store row): "
                         + ", ".join(sorted(self.undated)[:5]))
        return "\n".join(lines)


def roll_up_by_day(root: Path, since: str = "") -> DayRollup:
    """Group `root`'s session files by the day in their name, scan each group.

    Each day's numbers come from `app.thinking_fidelity.scan_files` — the same
    walk `scan_store` makes, one `scan_file` call per file — so the meta-session
    exemption and the marker set are inherited, not re-implemented here (#1770
    clause 1). `since` is a normalised ISO date or ""; it bounds the ROLL-UP, and
    the whole-store row written beside it is unaffected by it.

    A day inside the window whose files hold no reasoning block still gets a scan
    (`blocks == 0`, which the row renders as a null rate, not a zero rate): the
    day exists in the store, so omitting it would be the ambiguity this series
    exists to remove. A day the store holds no files for is a different fact — no
    row is invented for it, and the printed window says what was scanned.
    """
    by_day: dict[str, list[Path]] = {}
    out = DayRollup()
    for path in sorted(root.glob("*.json")):
        day = day_of(path.name)
        if not day:
            out.undated.append(path.name)
        elif not since or day >= since:
            by_day.setdefault(day, []).append(path)
    for day in sorted(by_day):
        out.days[day] = scan_files(root, by_day[day])
    return out


def _measurements(scan: StoreScan) -> dict:
    """The numerator/denominator fields every row carries, whatever its scope.

    Field by field the same numbers `StoreScan.report()` prints, so the prose and
    the series cannot disagree about what a run measured. `scanned` and the
    `null` rate are the anti-confusion pair: a run over an empty store is
    `{scanned: false, blocks: 0, flagged_rate: null}` and a clean store is
    `{scanned: true, blocks: 40964, flagged: 0, flagged_rate: 0.0}`.
    """
    return {
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


def prevalence_row(scan: StoreScan,
                   scope: str = SCOPE_WHOLE_STORE) -> dict:
    """The machine-readable row for a whole-store scan, denominator and all.

    `scope` is what the denominator covers, and it is a field rather than an
    implication because `--record --per-day` appends day rows into the same file
    as this one.
    """
    now = datetime.now(timezone.utc)
    return {
        "date": now.date().isoformat(),
        "scope": scope,
        "recorded_at": now.isoformat(timespec="seconds"),
        "root": str(scan.root),
        **_measurements(scan),
    }


def day_row(day: str, scan: StoreScan) -> dict:
    """The machine-readable row for one calendar day's slice of the store.

    `date` is the DAY, not the run: this row measures 2026-09-26 whether it is
    written on the 27th or the 3rd, and a series reader groups on `date`.
    `recorded_at` is when the scan ran, so a restated day is distinguishable from
    a fresh one. `day` repeats the same ISO date under the name the clause uses.
    """
    return {
        "date": day,
        "scope": SCOPE_DAY,
        "day": day,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "root": str(scan.root),
        **_measurements(scan),
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
                         "it to the prevalence series; with --per-day the day "
                         "rows are appended before it, oldest day first")
    ap.add_argument("--series", default="",
                    help=f"series file to append to (default: <data root>/"
                         f"{PREVALENCE_RELATIVE})")
    ap.add_argument("--per-day", action="store_true",
                    help="also print one line per calendar day — day, flagged, "
                         "blocks, rate — for the days the store holds, grouped "
                         "by the session filename's YYYYMMDD prefix; with "
                         "--record, append those day rows too")
    ap.add_argument("--since", type=since_iso, default="",
                    help="with --per-day: roll up only days on or after this "
                         "YYYY-MM-DD date (the whole-store row still covers "
                         "everything)")
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

    rollup = roll_up_by_day(root, args.since) if args.per_day else None
    if rollup is not None:
        report = rollup.report()
        if report:
            print(report)
        if not rollup.days:
            print(f"per-day: no session file is dated on or after "
                  f"{args.since or 'any date in the store'}, so the roll-up held "
                  "nothing — no day was discriminated, which is NOT a clean "
                  "verdict", file=sys.stderr)

    if args.record:
        series = (Path(args.series).expanduser() if args.series
                  else production_data_root() / PREVALENCE_RELATIVE)
        day_rows = ([day_row(day, s) for day, s in rollup.days.items()]
                    if rollup is not None else [])
        for one in day_rows:
            record_prevalence(one, series)
        row = prevalence_row(scan)
        record_prevalence(row, series)
        if day_rows:
            print(f"recorded {len(day_rows)} day row(s) "
                  f"({day_rows[0]['day']} .. {day_rows[-1]['day']}), oldest "
                  "first, then the whole-store row")
        print(f"recorded: {json.dumps(row, sort_keys=True)}")
        print(f"  appended to {series}")

    if scan.blocks == 0:
        print(f"  {scan.files} session file(s) held no reasoning block at all: "
              "this scan could not discriminate either way", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
