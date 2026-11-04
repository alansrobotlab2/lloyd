#!/usr/bin/env python3
"""Recount the reaper's reap-while-promoted conflicts, one named population per run.

What it re-derives
------------------
The #3600 recount, read-only: for every `round_abandoned` row that carries an
`item_id` at instant T, does ANOTHER round of that same item (the item's rounds come
from `backlog.implement_history`) hold a `promoted` row at ts <= T whose commit is in
neither `backlog._reverted_commits` nor any `settled` row at ts <= T. That is an
abandon taken while a promotion of the same item was still under observation — the
board move #3600 closed, and the only standing verification that it stayed closed.

The predicate is not restated here. The item's rounds come from `implement_history`,
the reverts from `_reverted_commits`, the settle from `_ledger_events("settled")`, and
the only arithmetic this file owns is `ts <= T` and a commit comparison. A `settled`
row is matched on `commit` because a `settled` row carries no `round_id` at all —
0 of the 381 on the live ledger when `workers/sources/board_steward.py:535-538`
measured it — so scoping the settle test by round instead would make every promotion
look unsettled and inflate `conflicts` out of nothing.

The baseline, and which population it belongs to
------------------------------------------------
Measured 2026-11-04 over the live state dir, `--ledger` default, merged population:

    population=merged abandons=374 conflicts=4   items 1685, 2694, 3112, 3524
    population=merged conflicts_since_fix=0

**`conflicts=4` is a merged-population figure and decays the other way from the live
one**: its rows are the rotations, so they are permanent as long as the archives are
kept, while `conflicts=4`'s live-file subset is not. The same run over `--live-only`
on 2026-11-04 reads `abandons=317 conflicts=2 items=3112,3524`. That 2 is a dated
reading, NOT a baseline, and the retention sweep is already eating it: rotation is
per-row and keyed on the row's own month
(`scripts/groundskeeper/retention-sweep.py:130`, `:2511`, `:2650`), and
`SM_20261014_095321`'s `promoted` row (item 2694) is archive-only while its
`round_abandoned` row is still live. When `SM_20261021_172454`'s promotion rotates the
live count reads 1, then 0 — a docstring that called 2 the baseline would be read as a
regression a month from now, which is the exact failure this item exists to prevent,
pointed at itself. `abandons=` is population-dependent for the same reason (317 vs
374, a 57-row gap on 2026-11-04), which is why every number this prints is labelled
with the population it was counted over rather than printed bare.

The load-bearing number is `conflicts_since_fix`: the conflicts whose abandon landed
strictly after #3600's fix. It is the only figure both populations agree on, and it
cannot move unless the guard breaks again — a rotation removes rows from the live file
but never rewrites a `created_at`.

Why `--ledger` is not a population switch
-----------------------------------------
Population is `merged` by default (`state.ledger_rows_merged`, so the
`promotions-archive-*.jsonl.gz` rotations count) or live-alone under `--live-only`,
and nothing else selects it. `--ledger` points at a different file for a test that
plants a ledger; the archives it reads are always the ones beside that file
(`state.ledger_archives`), so it changes which store is measured, never which
population of it. There is deliberately no `--record` and no `--schedule`: owed-check
ruled on #3600's owed entry 2 at 2026-11-04T02:14:31 that "the recount stays an
on-demand probe", and #2759 shipped `gate_to_land_lag.py` with a `--record` flag and
no caller, which #2858 then had to file the caller for. This script has no write path
of any kind — no journal, no `--record`, nothing under any data root.

Usage
-----
    cd ~/lloyd && .venvs/lloyd/bin/python scripts/maintenance/reap_round_conflicts.py
    cd ~/lloyd && .venvs/lloyd/bin/python scripts/maintenance/reap_round_conflicts.py --live-only

Exit 0 iff `conflicts_since_fix` is 0. The historical conflicts do not make it fail:
they predate the fix and are the reason the guard exists.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

#: Resolved from `__file__`, never cwd (app.paths' rule). Appended, not inserted at
#: [0], so a `--ledger` aimed at another checkout's file does not resolve
#: `scripts.automod` back to this checkout — the sibling probes make the same
#: distinction in their docstrings and it is what `tests/test_reap_round_conflicts.py`
#: relies on when it aims this probe at a planted file.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

#: #3600's guard landed at this instant. `git log -1 --format='%cI' 68425af5` answers
#: `2026-10-30T11:36:03-07:00`, which is `2026-10-30T18:36:03Z` — the COMMITTER stamp,
#: not #3600's settle time (its landed note records settle at 18:46:21Z, ten minutes
#: later); using the settle would silently widen the window by that much and let a
#: conflict in the gap read as historical.
FIX_INSTANT = "2026-10-30T18:36:03Z"
FIX_COMMIT = "68425af5"
FIX_TS = datetime.fromisoformat(FIX_INSTANT.replace("Z", "+00:00")).timestamp()


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(float(ts), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _stamp(row: dict) -> str:
    """The row's own `created_at`, falling back to its epoch only if it lacks one."""
    text = str(row.get("created_at") or "")
    return text if text else _iso(row.get("ts") or 0.0)


def recount(ledger: Path, *, merged: bool) -> dict:
    """The as-of-T conflicts over one population of one ledger.

    `merged=True` is the live file plus every rotation (the default population);
    `merged=False` is the live file alone. Both halves of a conflict — the abandon and
    the promotion it collided with — are read over the SAME population, because a
    mixed read would report a conflict the ledger cannot show, or hide one.
    """
    from scripts.automod import backlog as B

    population = "merged" if merged else "live"
    history = B.implement_history(ledger, with_archives=merged)
    reverted = set(B._reverted_commits(ledger, merged=merged))
    settles = [(str(d.get("commit") or ""), float(d.get("ts") or 0.0))
               for d in B._ledger_events(ledger, "settled", require_item=False,
                                         merged=merged)]
    promoted_by_round: dict[str, list[dict]] = {}
    for d in B._ledger_events(ledger, "promoted", require_item=False, merged=merged):
        if d.get("round_id"):
            promoted_by_round.setdefault(str(d["round_id"]), []).append(d)
    abandons = [d for d in B._ledger_events(ledger, "round_abandoned", merged=merged)
                if d.get("item_id") is not None]

    conflicts: list[dict] = []
    for a in abandons:
        item = int(a["item_id"])
        t = float(a.get("ts") or 0.0)
        reaped = str(a.get("round_id") or "")
        for attempt in history.get(item, []):
            other = str(attempt.get("round_id") or "")
            # `other` is empty for every `started` row: none of the 715 on the live
            # ledger carries a `round_id`, so those rows contribute no rounds and are
            # skipped rather than counted as a round named "".
            if not other or other == reaped:
                continue
            for promo in promoted_by_round.get(other, []):
                if float(promo.get("ts") or 0.0) > t:
                    continue
                commit = str(promo.get("commit") or "")
                if commit in reverted:
                    continue
                if any(c == commit and s <= t for c, s in settles):
                    continue
                conflicts.append({
                    "created_at": _stamp(a),
                    "ts": t,
                    "item_id": item,
                    "reaped_round": reaped,
                    "promoted_round": other,
                    "promoted_created_at": _stamp(promo),
                    "commit": commit,
                })
                break

    conflicts.sort(key=lambda r: r["ts"])
    since_fix = [r for r in conflicts if r["ts"] > FIX_TS]
    return {
        "population": population,
        "ledger": str(ledger),
        "abandons": len(abandons),
        "conflicts": conflicts,
        "since_fix": len(since_fix),
    }


def format_report(result: dict) -> list[str]:
    """The printed lines. Every number rides the population name it was counted over."""
    pop = result["population"]
    items = sorted({c["item_id"] for c in result["conflicts"]})
    lines = [
        f"population={pop} ledger={result['ledger']}",
        f"population={pop} abandons={result['abandons']} conflicts={len(result['conflicts'])}"
        f" items={','.join(str(i) for i in items) if items else '-'}",
    ]
    for c in result["conflicts"]:
        lines.append(
            f"population={pop} conflict created_at={c['created_at']} item={c['item_id']}"
            f" reaped={c['reaped_round']} promoted={c['promoted_round']}"
            f" promoted_created_at={c['promoted_created_at']} commit={c['commit'][:8]}")
    lines.append(
        f"population={pop} conflicts_since_fix={result['since_fix']}"
        f" (abandons after {FIX_INSTANT}, {FIX_COMMIT}) exit={0 if result['since_fix'] == 0 else 1}")
    return lines


class ProbeArgumentParser(argparse.ArgumentParser):
    """An unknown flag gets the reason that matters, not just the usage line.
    `argparse` is what makes the flag surface checkable at all —
    `tests/test_reap_round_conflicts.py` reads the registered flags off the syntax tree,
    which it could not do against a hand-rolled `if "--live-only" in args`. This probe
    takes two arguments and no way to store an answer: owed-check ruled on #3600's owed
    entry 2 at 2026-11-04T02:14:31 that the recount stays an on-demand probe, and a
    `--record` flag is how a probe nobody scheduled becomes a nightly row (#2759 shipped
    one with no caller and #2858 had to file the caller afterwards)."""

    def error(self, message: str):
        super().error(f"{message} — this probe takes only --live-only and --ledger, and "
                      "writes nothing")


def main(argv: list[str] | None = None) -> int:
    parser = ProbeArgumentParser(
        prog="reap_round_conflicts.py",
        description="Recount #3600's reap/promoted conflicts over one named ledger "
                    "population. Read-only.",
        epilog="read-only: prints a recount and an exit code, writes nothing")
    parser.add_argument("--live-only", action="store_true",
                        help="count the live promotions.jsonl alone instead of the "
                             "merged live+archives population (the default)")
    parser.add_argument("--ledger", metavar="PATH", type=Path,
                        help="count this ledger file instead of state.LEDGER_PATH")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    from scripts.automod import state as S
    result = recount(args.ledger or S.LEDGER_PATH, merged=not args.live_only)
    for line in format_report(result):
        print(line)
    return 0 if result["since_fix"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
