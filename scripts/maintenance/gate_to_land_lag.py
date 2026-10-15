#!/usr/bin/env python3
"""Gate-pass→promote lag, and the moved-tree catch the ledger cannot count (#2759).

Two numbers decide whether a staleness stamp is worth building for the
verdict→action lane of #2758: how long a gate's verdict stays actionable before the
promoter acts on it, and how often the landing's state-identity check has actually
caught the tree moving. Both have been computable from the promotion ledger since
the first landing and nothing published either one, so every "stamp this lane"
proposal argued with no denominator. `scripts/automod/landed_outcomes.py` measures
spans that contain the whole implementer turn (its p50s are hours);
`scripts/automod/scorecard.py` sums a gate's own rung seconds. Neither spans
verdict→action.

So this is a reader, not an instrument: it joins the newest passing `gate` row
(`ok is true`) to the `promoted` row of the same `round_id`, across
`promotions.jsonl` **and** every `promotions-archive-YYYYMM.jsonl.gz` beside it —
measured 2026-10-15, 26,798 of the 47,767 ledger rows are archived, so a reader that
took only the live file described under 60 % of the corpus and reported the shorter
lag.

Read the three absences in the output as loudly as the numbers:

  * **The promote-time identity refuse leaves no ledger row.** Its raise
    (`promote.py`: "worktree HEAD moved since the gate ran … re-gate before
    landing") sits *before* the `try:` that appends an event, so it escapes
    `promote()` with no `append_event`: 0 of 47,767 rows contain that message and no
    file under the state dir does. Its catch count is therefore **unknown**, and this
    report never prints a `0` for it. The countable half is the gate preflight's
    catch, whose `live HEAD moved` detail `gate.py` does write.
  * **A caught-and-recovered move is invisible too.** `_regate_after_move` re-runs
    the ladder, and the ordinary `gate` rows are its only trace, so a recovery cannot
    be separated from a routine re-gate.
  * **The other two #2758 lanes are not computable from any store.** measure→publish
    and read→claim have no "state at next read" row anywhere, so their lag needs a
    writer, not a reader. Said here rather than printed as an empty table.

Giving those three a row of their own is new instrumentation, outside this read-only
item, and owed to a ruling once the published absence is on the record.

    python scripts/maintenance/gate_to_land_lag.py [--state-dir PATH] [--record] [--json]

Exit 0 either way: this is a report, not a guard, and a thin join is printed with the
`n` that makes it thin. `--record` appends **one** JSON row per invocation to
`gate_to_land_lag.jsonl` in the automod state dir (`scripts/automod/state.py`'s
`STATE_DIR`, so `LLOYD_AUTOMOD_STATE` relocates it); nothing else is ever written.
`promotions.jsonl` is the audit trail of what code runs in production, and a report
that appended to it would be forging history.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
# Appended, not inserted: `app/` holds top-level modules that would shadow a package
# of the same name for anything else in this process (same rule as
# `scripts/maintenance/voice_turn_trend.py`).
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

#: The report's own bucket: a verdict that waited ten minutes is already outside the
#: window a staleness stamp would have closed, so this count is the one that says how
#: often the lane is running on stale evidence at all.
OVER_BUCKET_S = 600.0

#: The ceiling is NOT a constant here. It is `automod.landing.idle_hard_max_wait_s`,
#: read through `scripts.automod.state.landing_cfg` out of the shipped `config.yaml`,
#: because the day that value moves, a literal 4500 turns this count into folklore.
CEILING_KEY = "idle_hard_max_wait_s"

#: The report's own file, beside `landed_outcomes.jsonl` and `scorecard.jsonl` in the
#: automod state dir — never inside the checkout, where the guardian alerts on runtime
#: data hourly (#1667, the same rule `landed_outcomes.py` follows).
RECORD_NAME = "gate_to_land_lag.jsonl"

#: How many dropped round ids get named per line. The count is the figure; the names
#: exist so a reader can chase one round down in the ledger instead of taking the
#: count on trust. More than this and the line stops being readable.
NAMED_DROPS = 5

#: The substring `gate.py` writes when the tree moved under a round ("live HEAD
#: moved: <head> != base <base> …"). Counted because it is IN the ledger; the
#: promote-time message is not, which is the other half of the line below.
MOVED_TREE_TEXT = "live HEAD moved"


def _percentile(values: list[float], q: float) -> Optional[float]:
    """The `q`-th percentile (`q` is a fraction, not a percent), or None for no sample.

    Linear interpolation between the two nearest ranks — `numpy.percentile`'s default
    method, the definition `scripts/maintenance/voice_turn_trend.py` also uses — so a
    number printed here and one recomputed from the same rows in numpy agree. An
    off-by-one interpolation is a "regression" nobody can reproduce, and this table is
    what a staleness proposal gets graded on.
    """
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    low, high = math.floor(pos), math.ceil(pos)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def _state_dir() -> Path:
    from scripts.automod import state as S
    # Read off the module at call time, never copied at import: `STATE_DIR` is the
    # directory the loop offers, and `LLOYD_AUTOMOD_STATE` is the one hook
    # `scripts/automod/state.py` gives a caller for relocating it.
    return Path(S.STATE_DIR)


def record_path(state_dir: Optional[Path] = None) -> Path:
    return (Path(state_dir) if state_dir else _state_dir()) / RECORD_NAME


def ledger_files(state_dir: Path) -> list[Path]:
    """The live ledger, then every archive beside it, oldest month first.

    The glob is the whole point: the retention sweep rolls old rows into
    `promotions-archive-<YYYYMM>.jsonl.gz` on its own schedule, so a reader that
    hard-coded one month — or read only the live file — silently redefines its own
    population. A missing live file is tolerated (the sweep can roll its last rows
    out); the reader then sees `ledger_lines: 0` rather than a traceback.
    """
    return [Path(state_dir) / "promotions.jsonl",
            *sorted(Path(state_dir).glob("promotions-archive-*.jsonl.gz"))]


def iter_ledger_rows(files) -> Iterator[tuple[Optional[dict], bool, bool]]:
    """Yield `(row, from_archive, moved_tree)` per non-empty ledger line.

    One streaming pass, because the live ledger plus its archives is 47,767 rows and
    the archives are gzip: holding decoded rows would cost more than the figures need.
    `row` is None for a line that is not a JSON object — a truncated final line is
    normal in an appended JSONL, and the caller counts those out loud instead of
    letting a partial read look like a complete one. `moved_tree` is decided on the
    RAW line rather than the decoded dict, because the count the item asks for is "rows
    whose text contains `live HEAD moved`" and an undecodable line still has text; it
    must not also vanish from that count.
    """
    for path in files:
        path = Path(path)
        if not path.is_file():
            continue
        archive = path.suffix == ".gz"
        try:
            # Not `opener = gzip.open if archive else path.open`: `path.open` is a
            # bound method, so calling it as `opener(path, …)` passes the path twice.
            handle = (gzip.open(path, "rt", encoding="utf-8", errors="replace")
                      if archive else path.open("rt", encoding="utf-8", errors="replace"))
        except OSError:
            continue
        with handle as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                moved = MOVED_TREE_TEXT in line
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    yield None, archive, moved
                    continue
                yield (row if isinstance(row, dict) else None), archive, moved


def _epoch(row: dict) -> Optional[float]:
    """The row's timestamp: `ts`, else `created_at`, else None (counted, not guessed)."""
    ts = row.get("ts")
    if isinstance(ts, (int, float)):
        return float(ts)
    stamp = row.get("created_at")
    if isinstance(stamp, str):
        try:
            return datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def _ceiling() -> Optional[float]:
    """`automod.landing.idle_hard_max_wait_s` from the shipped config, or None.

    None is reported, never defaulted: guessing 4500 here is the exact thing clause 1
    forbids, and an unreadable config makes the count meaningless, so the line says so.
    """
    from scripts.automod import state as S
    try:
        raw = S.landing_cfg().get(CEILING_KEY)
    except Exception:
        return None
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def measure(state_dir=None) -> dict:
    """The join, plus every denominator and drop that qualifies it."""
    directory = Path(state_dir) if state_dir else _state_dir()
    files = ledger_files(directory)

    gate_ts: dict[str, float] = {}
    gate_live: set[str] = set()
    promoted_ts: dict[str, float] = {}
    promoted_live: set[str] = set()
    lines = gate_rows = promoted_rows = unreadable = 0
    no_id = no_stamp = moved_rows = 0

    for row, archive, moved in iter_ledger_rows(files):
        lines += 1
        if moved:
            moved_rows += 1
        if row is None:
            unreadable += 1
            continue
        event = row.get("event")
        if event == "gate":
            if row.get("ok") is not True:
                continue
            gate_rows += 1
            rid = row.get("round_id")
            if not rid:
                no_id += 1
                continue
            stamp = _epoch(row)
            if stamp is None:
                no_stamp += 1
                continue
            gate_ts[rid] = max(stamp, gate_ts.get(rid, float("-inf")))
            if not archive:
                gate_live.add(rid)
        elif event == "promoted":
            promoted_rows += 1
            rid = row.get("round_id")
            if not rid:
                no_id += 1
                continue
            stamp = _epoch(row)
            if stamp is None:
                no_stamp += 1
                continue
            promoted_ts[rid] = max(stamp, promoted_ts.get(rid, float("-inf")))
            if not archive:
                promoted_live.add(rid)

    joined = sorted(set(gate_ts) & set(promoted_ts))
    lags = [promoted_ts[r] - gate_ts[r] for r in joined]
    only_promoted = sorted(set(promoted_ts) - set(gate_ts))
    only_gate = sorted(set(gate_ts) - set(promoted_ts))
    ceiling = _ceiling()

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "state_dir": str(directory),
        "ledger_files": [str(p) for p in files if Path(p).is_file()],
        "ledger_lines": lines,
        "unreadable_rows": unreadable,
        "gate_pass_rows": gate_rows,
        "promoted_rows": promoted_rows,
        "rows_without_round_id": no_id,
        "rows_without_timestamp": no_stamp,
        "n": len(lags),
        "p50": _percentile(lags, 0.50),
        "p90": _percentile(lags, 0.90),
        "p99": _percentile(lags, 0.99),
        "max": max(lags) if lags else None,
        "min": min(lags) if lags else None,
        "negative": sum(1 for v in lags if v < 0),
        # `bucket_s` travels with the count: if `OVER_BUCKET_S` is ever retuned, a
        # row read back from the record still says which threshold it was counted
        # against, which is the same reason `ceiling_s` is a field and not a caption.
        "bucket_s": OVER_BUCKET_S,
        "over_bucket": sum(1 for v in lags if v > OVER_BUCKET_S),
        "over_ceiling": (sum(1 for v in lags if v > ceiling)
                         if ceiling is not None else None),
        "ceiling_s": ceiling,
        "rounds_only_in_archives": sum(1 for r in joined
                                       if r not in gate_live and r not in promoted_live),
        "promoted_without_passing_gate": len(only_promoted),
        "promoted_without_passing_gate_ids": only_promoted[:NAMED_DROPS],
        "gate_without_promoted": len(only_gate),
        "gate_without_promoted_ids": only_gate[:NAMED_DROPS],
        "live_head_moved_rows": moved_rows,
    }


def _fmt(value: Optional[float]) -> str:
    return "?" if value is None else f"{value:g} s"


def render(row: dict) -> str:
    """The report. Every percentile line carries its `n`, and every drop is a line."""
    denom = f"(n={row['n']})"
    out = [
        "gate-pass→promote lag, from the promotions ledger and its archives (#2759)",
        f"  ledger lines read: {row['ledger_lines']} over {len(row['ledger_files'])} "
        f"file(s); unreadable rows skipped: {row['unreadable_rows']}",
        f"  rounds joined: {row['n']} (passing gate rows seen "
        f"{row['gate_pass_rows']}, promoted rows seen {row['promoted_rows']})",
        f"  joined from an archive only: {row['rounds_only_in_archives']} {denom}",
    ]
    for label in ("p50", "p90", "p99", "max", "min"):
        out.append(f"  {label:<4s}{_fmt(row[label]):>10s} {denom}")
    out.append(f"  negative lags (promoted stamped before its newest passing gate "
               f"row — the promoter's row beat the ladder's last one): "
               f"{row['negative']} {denom}")
    out.append(f"  over {OVER_BUCKET_S:g} s: {row['over_bucket']} {denom}")
    if row["ceiling_s"] is None:
        out.append(f"  over {CEILING_KEY}: ? (the key was unreadable in config.yaml, "
                   f"so nothing was graded against it) {denom}")
    else:
        out.append(f"  over {CEILING_KEY}={row['ceiling_s']:g} s: "
                   f"{row['over_ceiling']} {denom}")

    # The drops, counted and named, so a thin join cannot read as a short lag and a
    # reader can chase one round down instead of trusting a count.
    out.append(f"  promoted with no passing gate row: "
               f"{row['promoted_without_passing_gate']}"
               + (f" ({', '.join(row['promoted_without_passing_gate_ids'])}"
                  + ("…" if row["promoted_without_passing_gate"] > NAMED_DROPS else "")
                  + ")" if row["promoted_without_passing_gate_ids"] else ""))
    out.append(f"  passing gate with no promoted row: {row['gate_without_promoted']}"
               + (f" ({', '.join(row['gate_without_promoted_ids'])}"
                  + ("…" if row["gate_without_promoted"] > NAMED_DROPS else "")
                  + ")" if row["gate_without_promoted_ids"] else ""))
    if row["rows_without_round_id"] or row["rows_without_timestamp"]:
        out.append(f"  rows no join can use: {row['rows_without_round_id']} without a "
                   f"round_id, {row['rows_without_timestamp']} without a timestamp")

    out.append(f"  ledger rows containing {MOVED_TREE_TEXT!r}: "
               f"{row['live_head_moved_rows']} — the gate-preflight catch, which "
               f"gate.py does write; the promote-time identity refuse appends no "
               f"ledger row, so its catch count is unknown, not zero, and a "
               f"caught-and-recovered move is indistinguishable from a routine re-gate")
    out.append("  not computable from any store (owed #2758): measure→publish and "
               "read→claim have no state-at-next-read row, so their lag needs a "
               "writer, not this reader")
    return "\n".join(out)


def record(row: dict, path: Optional[Path] = None) -> Path:
    """Append exactly one JSON line. The only write this script makes."""
    target = Path(path) if path else record_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    return target


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Gate-pass→promote lag and moved-tree catches, read-only")
    ap.add_argument("--state-dir", default=None,
                    help="automod state dir (default: the one "
                         "scripts/automod/state.py resolves)")
    ap.add_argument("--record", action="store_true",
                    help=f"append one JSON row to {RECORD_NAME} in that directory")
    ap.add_argument("--json", action="store_true", help="print the row as JSON")
    args = ap.parse_args(argv)

    state_dir = Path(args.state_dir) if args.state_dir else _state_dir()
    row = measure(state_dir=state_dir)
    print(json.dumps(row, indent=2, sort_keys=True) if args.json else render(row))
    if args.record:
        print(f"\nrecorded → {record(row, record_path(state_dir))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
