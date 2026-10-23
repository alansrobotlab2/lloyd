#!/usr/bin/env python3
"""Whether the declared scheduled-task ceiling may be flipped to enforcement (#2927).

`app/scheduled_ceiling.py` wired the 33-name `scheduled-task` ceiling into the live
turn in SHADOW mode (#2670): the ceiling is compiled, the turn is dispatched exactly
as it was before, and every name it would have refused is appended to a durable
ledger. Enforcement is a later commit behind one operator switch
(`LLOYD_SCHEDULED_CEILING_MODE=enforce`), and #2670's owed entry set the bar for
taking it: **7 shadow days in that store with no would-strip name above 5% of turns
and zero tool-name refusals.** Nothing computed that bar before now — the ledger had
no reader, so "are we ready" was answerable only by hand-counting JSONL. This is that
reader.

Five readings, then ONE verdict:

  * **shadow dates** — the count of distinct dates the shadow rows cover. Turns are
    the wrong denominator for this bar: #2670 asked for seven *days*, because the
    fleet's exposure is a function of which jobs ran, and one busy night of twenty
    turns is not seven days of whatever the schedule decides to run.
  * **would-strip names** — each name any row recorded, with its count and its rate
    over the TOTAL row count (every mode). Sorted by count, each marked `OVER` past
    the bar.
  * **unmeasured rows** — rows carrying no `would_strip` list. These are NOT zeros:
    `TurnSurface.witness` deliberately omits the key when the ceiling could not
    compile (`compile_error`), because a metric that reads its own missing input as
    a zero is the failure this box has been burned on repeatedly. They sit in the
    denominator of the rates above, so the count is printed beside the rates rather
    than folded into them silently.
  * **enforce rows / rows refusing a name** — under shadow a refusal is impossible
    by construction (nothing is dispatched away), so this is 0 and says nothing about
    readiness. It is printed anyway because the same reader has to be the one that
    answers after a flip, when a non-empty `would_strip` on an `enforce` row IS a
    tool call that was refused — and a reader that starts reporting that number only
    once it is interesting would be a reader nobody trusts at the moment it is.

  * **rows carrying a would-strip name** — printed, deliberately NOT a bar. The per-name
    table divides each name by the total separately, so four names at two rows each all
    read 4.8% and the verdict is PASS, while the rows carrying any name at all are 8 of 42
    (19.0%) and under enforcement every one of them is a turn that lost a tool. Row-level
    and name-level are different questions, and the number the flip actually hurts jobs
    with cannot be derived from the table: it needs a count no line above prints. The live
    ledger measured 33 of 354 = 9.3% on 2026-10-23 with no single name above 6.5%.

The verdict is PASS only when the first four hold: at least `MIN_SHADOW_DATES` shadow
dates, every would-strip name at or under `NAME_RATE_CEILING` of the total rows, no
unmeasured row, and no enforce-mode row refusing a name. Everything else is FAIL with
one reason line per failed bar, naming the number and the bar it missed.

What this deliberately does NOT decide:

  * **Whether a name should be licensed instead of tolerated.** `automod_vault_land`
    is the only name over the bar today, and the way it comes under is a `grants:`
    frontmatter block on the eight nightly jobs that already hold the tool
    (`app/harness/policy.py::validate_task_grants` reads a frontmatter block as
    *the human wrote it, so the human is the issuer* — which makes each block a
    person's act, not a round's). This reader prints the rate and the scopes; the
    licence is Alan's call. It must not be gamed here: suppressing the reading to
    reach PASS is the same mistake as widening the ceiling to reach 0.
  * **Flipping the switch.** `--ledger` reads a copy; the flip itself is an operator
    edit to the service unit, taken only on a PASS, and the paired test that
    `run_task`'s `allowed_tools` equals the compiled envelope goes in that commit
    (#2670's owed entry 1).

Exit status is the verdict: 0 on PASS, 1 on FAIL, so the daily run can be a check and
not a grep.

`--record` appends the reading as one JSONL row to `flip-gate-readings.jsonl`, in the
same directory as the ledger, and changes nothing else — the same stdout, the same exit
code. It exists because every reading here is perishable: the reader recomputes over the
whole ledger, so as the fleet keeps writing, the total every rate divides by moves, and
the PASS a flip is authorised on cannot be re-derived afterwards (post-flip the ledger
carries `enforce` rows, which is exactly the change that alters the denominator). A FAIL
is recorded exactly as a PASS is: appending is a write of evidence, not an escalation —
this script raises nothing, notifies nothing and files nothing on a red reading, and the
red ones are the rows a later reader most needs.

    python -m scripts.maintenance.scheduled_ceiling_flip_gate [--ledger PATH] [--record]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import Counter
from pathlib import Path

from app.scheduled_ceiling import MODE_ENFORCE, MODE_SHADOW, ledger_path, read_rows

#: Where `--record` appends its rows. Named for what it holds rather than for the script,
#: because it is the artifact a later reader goes looking for.
READINGS_FILENAME = "flip-gate-readings.jsonl"

#: How many distinct shadow dates the flip needs (#2670's owed entry 1: "7 shadow
#: days in the clause-3 store"). Counted as dates, never as rows — see the module
#: docstring for why rows are the wrong unit for this one bar.
MIN_SHADOW_DATES = 7

#: The per-name ceiling: a name may appear in at most this fraction of ALL rows
#: (#2670: "no would-strip name above 5% of turns"). A name's rate is per TURN, not
#: per call: a row's `would_strip` is a de-duplicated set, so one turn that called
#: the same excluded tool eleven times contributes one.
NAME_RATE_CEILING = 0.05

#: The line the verdict lives on. The whole report carries exactly one of these, so
#: a later reader (or the owed-check job) can take the decision off one grep instead
#: of re-deriving it from the table.
VERDICT_PREFIX = "verdict: "


def readings_path() -> Path:
    """`DATA_ROOT/scheduling/flip-gate-readings.jsonl` — the ledger's sibling.

    Derived from `app.scheduled_ceiling.ledger_path()` rather than from its own env var or
    from `--ledger`, for two reasons. The readings belong beside the witness they measure,
    under the data root and not in the tree a run can destroy (#2213, #1799). And a test
    that redirects the ledger through `LLOYD_SCHEDULED_CEILING_LEDGER` — the knob the
    reader's own tests already use — then redirects the readings with it, instead of
    appending a test row into the file the fleet is reading. `--ledger` is deliberately NOT
    consulted for the destination: a handed-in copy is often under `/tmp`, and readings
    filed next to it would be deleted with it, which is the failure mode this file exists
    to prevent. The row records which file the reading came from in its `ledger` field, so
    a reading of a copy stays labelled as one.
    """
    return ledger_path().parent / READINGS_FILENAME


def record_reading(m: dict, *, source: str, path: Path) -> None:
    """Append this reading as one JSONL row. The only write this script ever makes.

    The row carries the reading, not a summary of it — including `failures` verbatim, so a
    later reader can re-check the reason against the code that produced it rather than
    against a paraphrase. `ts` is UTC-aware because a naive local stamp in a machine-facing
    row is read as UTC by every reader after it.
    """
    row = {
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
        "ledger": source,
        "total_rows": m["total_rows"],
        "modes": m["modes"],
        "shadow_dates": m["shadow_dates"],
        "names": m["names"],
        "unmeasured": m["unmeasured"],
        "compile_errors": m["compile_errors"],
        "rows_with_strip": m["rows_with_strip"],
        "enforce_rows": m["enforce_rows"],
        "enforce_refusals": m["enforce_refusals"],
        "failures": m["failures"],
        "verdict": m["verdict"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def measure(rows: list[dict]) -> dict:
    """The four readings and the verdict, from the rows `read_rows` returned.

    Pure: no file access, no env, no clock. The date is read from the row, because
    `TurnSurface.witness` stamped it in UTC when the turn ended — re-deriving a date
    from `ts` here would be a second parse of a value the writer already normalised,
    and a row without one is malformed data to report, not to repair.
    """
    total = len(rows)
    modes = Counter(str(r.get("mode") or "") for r in rows)
    shadow_dates = sorted({str(r["date"]) for r in rows
                           if r.get("mode") == MODE_SHADOW and r.get("date")})
    counts: Counter = Counter()
    for row in rows:
        for name in row.get("would_strip") or ():
            counts[str(name)] += 1
    names = [{"name": n, "count": c, "rate": (c / total) if total else 0.0}
             # Count desc, then name asc: the order must not move when two names
             # tie, or a daily diff shows a re-shuffle where nothing changed.
             for n, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    # A row is "unmeasured" when it carries no `would_strip` key at all. `[]` means
    # measured clean and counts as zero reach; a missing key means the compile failed
    # and the turn measured nothing, which is a different fact with the same shape.
    unmeasured = [r for r in rows if "would_strip" not in r]
    compile_errors = sum(1 for r in unmeasured if r.get("compile_error"))
    # ROWS, not name-instances: a turn that reached for three excluded tools is one turn
    # that would have lost tools, and enforcement is judged on how many jobs it hurts.
    # A row with no `would_strip` key measured nothing, so it is not counted here either.
    rows_with_strip = sum(1 for r in rows if r.get("would_strip"))
    enforce_rows = [r for r in rows if r.get("mode") == MODE_ENFORCE]
    enforce_refusals = [r for r in enforce_rows if r.get("would_strip")]

    failures: list[str] = []
    if total == 0:
        # An absent or empty ledger is not a clean bill of health: it is the same
        # reading-the-wrong-file case that makes every other empty store here a red.
        failures.append("the ledger holds 0 rows, so the ceiling has never been "
                        "shadow-measured (run a scheduled task, or pass --ledger)")
    if len(shadow_dates) < MIN_SHADOW_DATES:
        failures.append(f"shadow dates {len(shadow_dates)} < {MIN_SHADOW_DATES} "
                        f"required")
    over = [n for n in names if n["rate"] > NAME_RATE_CEILING]
    for entry in over:
        failures.append(f"{entry['name']} reached in {entry['count']} of {total} "
                        f"rows = {entry['rate'] * 100:.1f}% > "
                        f"{NAME_RATE_CEILING * 100:.1f}%")
    if unmeasured:
        failures.append(f"{len(unmeasured)} row(s) carry no would_strip measurement "
                        f"({compile_errors} naming a compile_error) — they sit in "
                        "the denominator above and measured nothing")
    if enforce_refusals:
        failures.append(f"{len(enforce_refusals)} enforce-mode row(s) refused a tool "
                        "name")

    return {
        "total_rows": total,
        "modes": dict(modes),
        "shadow_dates": shadow_dates,
        "names": names,
        "unmeasured": len(unmeasured),
        "compile_errors": compile_errors,
        "rows_with_strip": rows_with_strip,
        "enforce_rows": len(enforce_rows),
        "enforce_refusals": len(enforce_refusals),
        "failures": failures,
        "verdict": "FAIL" if failures else "PASS",
    }


def format_report(m: dict, *, source: str = "") -> str:
    """The report the owed-check job reads. One `verdict:` line, at the end."""
    lines = ["scheduled-ceiling flip gate (#2670)",
             f"ledger: {source}" if source else "ledger: (rows handed in)"]
    modes = m["modes"]
    line = (f"rows: {m['total_rows']}   shadow {modes.get(MODE_SHADOW, 0)}   "
            f"enforce {modes.get(MODE_ENFORCE, 0)}   "
            f"unmeasured {m['unmeasured']}")
    if m["compile_errors"]:
        line += f" ({m['compile_errors']} with a compile_error)"
    lines.append(line)
    dates = m["shadow_dates"]
    lines.append(f"shadow dates: {len(dates)}"
                 + (f"   ({', '.join(dates)})" if dates else ""))
    lines.append(f"would-strip names (count / rate over all {m['total_rows']} rows; "
                 f"bar: every name <= {NAME_RATE_CEILING * 100:.1f}%):")
    if not m["names"]:
        lines.append("  (none — no row recorded a name outside the ceiling)")
    for entry in m["names"]:
        flag = "  OVER" if entry["rate"] > NAME_RATE_CEILING else ""
        lines.append(f"  {entry['name']:<26} {entry['count']:>4}   "
                     f"{entry['rate'] * 100:>5.1f}%{flag}")
    strip_rate = (m["rows_with_strip"] / m["total_rows"]) if m["total_rows"] else 0.0
    lines.append(f"rows carrying a would-strip name: {m['rows_with_strip']} of "
                 f"{m['total_rows']} = {strip_rate * 100:.1f}%   (under enforce, each "
                 "of these is a turn refused a tool name — the per-name table above "
                 "cannot show this: four names at 4.8% each are four PASS lines and "
                 "eight refused turns)")
    lines.append(f"enforce rows: {m['enforce_rows']}   rows refusing a name: "
                 f"{m['enforce_refusals']}   (shadow dispatches nothing away, so "
                 "this is 0 by construction until a flip is taken)")
    for reason in m["failures"]:
        lines.append(f"  ! {reason}")
    lines.append(VERDICT_PREFIX + m["verdict"])
    return "\n".join(lines)


def _record_or_warn(m: dict, *, source: str, path: Path) -> bool:
    """Record, and say so on STDERR if the record could not be written.

    stdout stays byte-for-byte the report (it is the machine-read artifact), and the exit
    code stays the verdict, so a failed record cannot change what a caller decided — but it
    must not be silent either: a reading nobody kept is exactly the perishable evidence this
    flag exists to stop losing. stderr is where that goes precisely because the check's own
    output is not its channel.
    """
    try:
        record_reading(m, source=source, path=path)
        return True
    except OSError as exc:
        print(f"wrote no reading to {path}: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return False


def main(argv: list[str] | None = None) -> int:
    """Read the ledger, print the report, exit 0 on PASS and 1 on FAIL."""
    ap = argparse.ArgumentParser(
        prog="scheduled_ceiling_flip_gate",
        description="Read the shadow ledger and say whether the scheduled-task "
                    "ceiling may be enforced yet (#2670, #2927).")
    ap.add_argument("--ledger", default="",
                    help="a ledger copy to read instead of the live one (the "
                         "witness file a report cites). Still parsed by "
                         "app.scheduled_ceiling.read_rows, so a malformed tail line "
                         "is skipped the same way the live read skips it.")
    ap.add_argument("--record", action="store_true",
                    help="append this reading as one JSONL row to "
                         f"{READINGS_FILENAME} beside the live ledger, on PASS or "
                         "FAIL, before exiting with the same report and the same "
                         "code. The reading is perishable: the rates are over the "
                         "total, and the total moves with every turn and with the "
                         "flip itself.")
    # `sys.argv[1:]` and not `[]`: the empty list made every command-line flag a no-op.
    # `--record` then wrote nothing while still exiting on the verdict, and `--ledger
    # /tmp/copy.jsonl` read the LIVE ledger behind a report that named the copy — a
    # silently ignored flag is worse than a refused one, and the only reason no test saw
    # it is that a test hands `main()` its argv explicitly and never goes through a shell.
    argv = list(argv) if argv is not None else sys.argv[1:]
    args = ap.parse_args(argv)

    source = Path(args.ledger).expanduser() if args.ledger else ledger_path()
    measured = measure(read_rows(source))
    print(format_report(measured, source=str(source)))
    if args.record:
        _record_or_warn(measured, source=str(source), path=readings_path())
    return 0 if measured["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
