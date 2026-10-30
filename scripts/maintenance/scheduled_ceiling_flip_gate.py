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

Five readings, then TWO verdicts (#3415): `verdict: PRE-FLIP <V>`, the four bars over the
SHADOW rows, and `verdict: POST-FLIP HEALTH <V>`, what enforcement has actually refused
over the ENFORCE rows. They are separate lines because they answer separate questions, and
after the flip they routinely disagree: the pre-flip bars can hold while enforcement
refuses a licensed name, and the first unlicensed refusal of a night must not read as the
pre-flip evidence having collapsed.

  * **shadow dates** — the count of distinct dates the shadow rows cover. Turns are
    the wrong denominator for this bar: #2670 asked for seven *days*, because the
    fleet's exposure is a function of which jobs ran, and one busy night of twenty
    turns is not seven days of whatever the schedule decides to run.
  * **would-strip names** — each name a SHADOW row recorded, with its count and its rate
    over the SHADOW row count. Not over every row: an enforce row's `would_strip` is a
    refusal that already happened and belongs to the post-flip verdict, and a denominator
    that kept growing with enforce rows would make the pre-flip bars go greener the longer
    enforcement ran. Sorted by count, each marked `OVER` past the bar.
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

PRE-FLIP is PASS only when the first four hold: at least `MIN_SHADOW_DATES` shadow dates,
every shadow-side would-strip name at or under `NAME_RATE_CEILING` of the shadow rows, and
no unmeasured row. There is deliberately NO pre-flip bar on an enforce row refusing a name:
after the flip that is the ceiling doing what it was flipped to do, and a bar there would
report FAIL at the moment the flip starts working. Enforcement is judged by `post_flip_health`
below, which reads FAIL on two facts and only two — a refusal of a name its own grant_scope
holds a live grant for, or the enforce refusal rate over `NAME_RATE_CEILING` — and reports an
unlicensed refusal as the ceiling working. Each FAIL prints one reason line per failed bar,
naming the number and the bar it missed. The exit code is 0 only when BOTH verdicts pass.

What this deliberately does NOT decide:

  * **Whether a name should be licensed instead of tolerated.** `automod_vault_land` is
    the name that has sat nearest the bar (23 of 624 shadow rows = 3.7% on 2026-10-30), and
    the way it comes under is a `grants:`
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
#: The line the PRE-FLIP verdict lives on. Prefix-match readers use
#: `tests/...::verdict_lines`, which matches `PRE-FLIP ` and `POST-FLIP ` first, so the two
#: cannot be confused by a `startswith("verdict: ")` on a report that has two of them.
VERDICT_PREFIX = "verdict: "
PRE_FLIP_PREFIX = "PRE-FLIP "
POST_FLIP_PREFIX = "POST-FLIP HEALTH "


def live_licensed_pairs(*, store=None) -> tuple[set[tuple[str, str]], str]:
    """Which (grant_scope, tool name) pairs a live grant licenses, and how it was found.

    Returns `(pairs, note)`; `note` is "" on a clean read and names the failure otherwise,
    with `pairs` empty in that case. Empty is NOT a clean read — it is the reason
    `post_flip_health` distinguishes "no grant" from "no grant store", because the two have
    opposite meanings about a refusal: one is the ceiling working, the other is this reader
    unable to tell.

    Matching is `(row["scope"], normalize_tool_name(row["tool_pattern"]))` against the
    ledger row's `grant_scope` and name — the same exact-equality-after-normalisation the
    store's own lookups do (`app/harness/policy.py:755`, `:1119`), so this reader cannot
    license a name the live path would have refused. `tool_pattern` is a literal tool name
    in every row today, and nothing here widens it to a glob: a grant minted as `B*` would
    match nothing here and nothing there, which is the drift this avoids.

    `store=` is the seam the tests use; the default calls `default_store()`, which opens
    the live grant database — read-only, `live()` is a SELECT of unexpired, unrevoked rows.
    """
    try:
        if store is None:
            from app.harness.policy import default_store
            store = default_store()
        rows = store.live()
    except Exception as exc:  # noqa: BLE001 — an unreadable store must be a NOTE, not a guess
        return set(), f"{type(exc).__name__}: {exc}"
    try:
        from app.harness.policy import normalize_tool_name
        pairs = {(str(r.get("scope") or ""), normalize_tool_name(r.get("tool_pattern")))
                 for r in rows}
    except Exception as exc:  # noqa: BLE001
        return set(), f"{type(exc).__name__}: {exc}"
    return pairs, ""


def post_flip_health(rows: list[dict], *, enforce_rows: list[dict] | None = None,
                     enforce_refusals: list[dict] | None = None,
                     licensed_pairs: set[tuple[str, str]] | None = None,
                     store=None) -> dict:
    """What enforcement has actually done, judged on the ENFORCE rows alone.

    Two ways this reads FAIL, and only two:

      1. a refusal of a name whose scope holds a live grant. That is the ceiling taking
         something away that a person licensed by hand — the ban exception #2670's whole
         design turns on — and it is the only refusal on this ledger that is a bug.
      2. the rate of refusals over enforce rows above NAME_RATE_CEILING. That bar is
         deliberately the same number as the pre-flip per-name bar: enforcement refusing a
         name more than 5% of the time is the same fact the pre-flip bar was a prediction
         of, now measured instead of forecast.

    A refusal of a name the scope holds NO live grant for is the ceiling working as
    designed, and is reported that way rather than as a failure. That distinction is the
    reason this verdict exists at all: a naive "any enforce refusal is red" would have the
    first unlicensed night report a failure at the exact moment the flip starts succeeding,
    and the reader's next move would be a rollback of a change that is doing its job.
    Clause 6 of the item asks the same of a human: no revert until someone names the
    refused name and says whether it was licensed, which is precisely the pair this
    function prints.

    An unreadable grant store with refusals on the ledger is FAIL: `licensed_pairs` empty
    would otherwise read every refusal as unlicensed and print a clean POST-FLIP HEALTH
    that rests on not having been able to look. With no refusals there is nothing for the
    store to have decided, so an unreadable store is a note and the verdict stands.

    Returns {verdict, failures, licensed, detail, rate}. `detail` is the
    name/scope/licensed triples, so a red is actionable from stdout alone.
    """
    if enforce_rows is None:
        enforce_rows = [r for r in rows if r.get("mode") == MODE_ENFORCE]
    if enforce_refusals is None:
        enforce_refusals = [r for r in enforce_rows if r.get("would_strip")]
    rate = (len(enforce_refusals) / len(enforce_rows)) if enforce_rows else 0.0

    store_note = ""
    if licensed_pairs is None:
        if enforce_refusals:
            licensed_pairs, store_note = live_licensed_pairs(store=store)
        else:
            licensed_pairs, store_note = set(), ""

    failures: list[str] = []
    licensed: list[str] = []
    detail: list[dict] = []
    for r in enforce_refusals:
        scope = str(r.get("grant_scope") or "")
        for name in sorted(set(r.get("would_strip") or ())):
            is_licensed = (scope, str(name)) in licensed_pairs
            detail.append({"name": str(name), "scope": scope, "licensed": is_licensed})
            if is_licensed:
                licensed.append(f"{name} ({scope})")

    licensed = sorted(set(licensed))
    if licensed:
        failures.append(
            f"{len(licensed)} name(s) refused under enforce while their scope holds a "
            f"live grant: {', '.join(licensed)}")
    if rate > NAME_RATE_CEILING:
        failures.append(
            f"enforce refusal rate {rate * 100:.1f}% ({len(enforce_refusals)} of "
            f"{len(enforce_rows)} enforce rows) > "
            f"{NAME_RATE_CEILING * 100:.1f}%")
    if store_note:
        failures.append(
            f"the grant store could not be read ({store_note}) while "
            f"{len(enforce_refusals)} enforce row(s) refused a name, so no refusal could "
            "be judged licensed or unlicensed — this is a red to go and look at, not a "
            "clean read")

    return {"verdict": "FAIL" if failures else "PASS",
            "failures": failures,
            "licensed": licensed,
            "detail": detail,
            "rate": rate}


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
        "enforce_refusal_rate": m.get("enforce_refusal_rate"),
        "licensed_refusals": m.get("licensed_refusals"),
        "post_flip": m.get("post_flip"),
        "post_flip_failures": m.get("post_flip_failures") or [],
        "failures": m["failures"],
        "verdict": m["verdict"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")


def measure(rows: list[dict], *, licensed_pairs: set[tuple[str, str]] | None = None) -> dict:
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
    # The per-name bars ask "how often did a turn reach for a name outside its envelope",
    # and only a SHADOW turn is a measurement of that: an enforce row's `would_strip` is a
    # refusal that already happened, counted by `post_flip_health` on its own side. Once
    # the flip is taken every new row is enforce, so a rate kept over `rows` would divide
    # the last week's shadow reaches by a total that keeps growing with enforce rows and
    # read as improvement — the bar would go green on a ledger where enforcement is
    # refusing more every day. Hence the denominator is named in the line it prints.
    shadow = [r for r in rows if r.get("mode") != MODE_ENFORCE]
    counts: Counter = Counter()
    for row in shadow:
        for name in row.get("would_strip") or ():
            counts[str(name)] += 1
    names = [{"name": n, "count": c, "rate": (c / len(shadow)) if shadow else 0.0}
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
    # Also shadow-only, because this is the PREDICTOR of what enforcement would do: an
    # enforce row's refusal is not a prediction, it is the thing itself and is reported as
    # a rate over enforce rows below.
    rows_with_strip = sum(1 for r in shadow if r.get("would_strip"))
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
        failures.append(f"{entry['name']} reached in {entry['count']} of {len(shadow)} "
                        f"rows = {entry['rate'] * 100:.1f}% > "
                        f"{NAME_RATE_CEILING * 100:.1f}%")
    if unmeasured:
        failures.append(f"{len(unmeasured)} row(s) carry no would_strip measurement "
                        f"({compile_errors} naming a compile_error) — they sit in "
                        "the denominator above and measured nothing")
    # There is deliberately no bar here on `enforce_refusals`. Before the flip it could
    # only ever fire on a stray enforce row, and after the flip it would fire on the FIRST
    # refusal of an unlicensed name — which is the ceiling doing the job it was flipped to
    # do. What enforcement must be judged on is whether it refused something the scope was
    # licensed for, and how much of the enforce traffic that is: `post_flip_health`.
    # `licensed_pairs=None` means "ask the store", which `post_flip_health` does ONCE for
    # the whole reading — never per row, because `live()` is a query against the same
    # workers.db the scheduler is writing and a per-row read would let one report mix two
    # moments of a store that changes during a fleet night.
    post = post_flip_health(rows, enforce_rows=enforce_rows,
                            enforce_refusals=enforce_refusals,
                            licensed_pairs=licensed_pairs)

    return {
        "total_rows": total,
        "shadow_rows": len(shadow),
        "modes": dict(modes),
        "shadow_dates": shadow_dates,
        "names": names,
        "unmeasured": len(unmeasured),
        "compile_errors": compile_errors,
        "rows_with_strip": rows_with_strip,
        "enforce_rows": len(enforce_rows),
        "enforce_refusals": len(enforce_refusals),
        "enforce_refusal_rate": post["rate"],
        "licensed_refusals": post["licensed"],
        "post_flip": post["verdict"],
        "post_flip_failures": post["failures"],
        "licensed_refusal_detail": post["detail"],
        "failures": failures,
        # `verdict` stays the PRE-FLIP verdict: the four bars' exit code is what the
        # recorder and every existing reader of a stored reading already consumes, and
        # clause 3 of this item is precisely that removing the flip variable restores that
        # reading. The enforce side gets its own key and its own line.
        "verdict": "FAIL" if failures else "PASS",
    }


def format_report(m: dict, *, source: str = "") -> str:
    """The report the owed-check job reads. One `verdict:` line, at the end."""
    lines = ["scheduled-ceiling flip gate (#2670)",
             f"ledger: {source}" if source else "ledger: (rows handed in)"]
    # Every rate on the shadow side is over the SHADOW rows, and the line says so. The
    # total is still printed above it, so nothing is hidden — but a bar that asks "how
    # often does a shadow turn reach outside its envelope" must not answer with a
    # denominator that enforcement is busy growing.
    strip_denom = m.get("shadow_rows", m["total_rows"])
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
    lines.append(f"would-strip names (count / rate over the {strip_denom} shadow rows; "
                 f"bar: every name <= {NAME_RATE_CEILING * 100:.1f}%):")
    if not m["names"]:
        lines.append("  (none — no row recorded a name outside the ceiling)")
    for entry in m["names"]:
        flag = "  OVER" if entry["rate"] > NAME_RATE_CEILING else ""
        lines.append(f"  {entry['name']:<26} {entry['count']:>4}   "
                     f"{entry['rate'] * 100:>5.1f}%{flag}")
    # The rate is over the shadow rows, so the printed denominator has to BE the shadow
    # rows: `2 of 45 = 4.8%` is arithmetic the reader cannot do (2/45 is 4.4%), and after
    # the flip the total grows with enforce rows while this predictor's denominator does
    # not. The total is on the line above, so naming 42 here hides nothing.
    strip_rate = (m["rows_with_strip"] / strip_denom) if strip_denom else 0.0
    lines.append(f"rows carrying a would-strip name: {m['rows_with_strip']} of "
                 f"{strip_denom} shadow rows = {strip_rate * 100:.1f}%   (under enforce, each "
                 "of these is a turn refused a tool name — the per-name table above "
                 "cannot show this: four names at 4.8% each are four PASS lines and "
                 "eight refused turns)")
    rate = m.get("enforce_refusal_rate",
                 (m["enforce_refusals"] / m["enforce_rows"]) if m["enforce_rows"] else 0.0)
    lines.append(f"enforce rows: {m['enforce_rows']}   rows refusing a name: "
                 f"{m['enforce_refusals']}   enforce refusal rate: {rate * 100:.1f}%")
    if not m["enforce_rows"]:
        lines.append("  (shadow dispatches nothing away, so the enforce numbers are 0 by "
                     "construction until a flip is taken; the POST-FLIP verdict below is "
                     "then vacuously PASS and says nothing about enforcement)")
    for entry in m.get("licensed_refusal_detail") or []:
        verdict_word = ("LICENSED BY A LIVE GRANT" if entry["licensed"]
                        else "no live grant for that scope — the ceiling working")
        scope = entry["scope"] or "(no scope)"
        lines.append(f"    refused {entry['name']} in {scope}: {verdict_word}")
    for reason in m["failures"]:
        lines.append(f"  ! {reason}")
    lines.append(VERDICT_PREFIX + PRE_FLIP_PREFIX + m["verdict"])
    for reason in m.get("post_flip_failures") or []:
        lines.append(f"  !! {reason}")
    lines.append(VERDICT_PREFIX + POST_FLIP_PREFIX + m.get("post_flip", "PASS"))
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
    # The exit code is "is anything wrong", which after the flip is EITHER verdict; which
    # half failed is readable from the two `verdict:` lines, not from the code.
    return 0 if (measured["verdict"] == "PASS"
                 and measured.get("post_flip", "PASS") == "PASS") else 1


if __name__ == "__main__":
    sys.exit(main())
