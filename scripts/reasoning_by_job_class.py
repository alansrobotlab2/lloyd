#!/usr/bin/env python3
"""#2576 — what default-on thinking buys, per job class, against that class's outcome.

    cd ~/lloyd && .venvs/lloyd/bin/python -m scripts.reasoning_by_job_class
    ... --days 14                      # longer window (default 7)
    ... --all                          # the whole store, no window
    ... --json                         # the same table, machine-readable
    ... --markdown                     # the vault report's table form
    ... --witness-out F                # commit the .md witness the report cites
    ... --workers-db <db> --usage-db <db>   # point it at fixtures

Run it from `~/lloyd`: `app.paths` resolves the two default store paths inside
whichever tree the module was imported from, so the same command from a round
worktree aims at that worktree's empty `.lloyd-data/` — where `_read_only` refuses
to invent a store, which is the sound outcome, and a worktree scan that *did* find
one would print a clean table over nothing. A round measures production by naming
`--workers-db`/`--usage-db` explicitly, and `corpus.workers_db`/`corpus.usage_db`
in `--json` record which bytes any given run opened.

Why the two stores have to be joined at all
-------------------------------------------
`usage.db`'s `usage` table records one row per turn and carries a
`reasoning_tokens` column, but it has NO job-class column: nothing in that store
says whether a turn was an autocode attempt or a triage sweep. The class exists
only on the other side of the fence — `workers.db`'s `runs.meta_json.session_ids`
names the sessions each run drove, written by `workers/pool.py` from the harness
session list. So the class table is a cross-store join, and until it existed the
question "is the engine's default reasoning level earning its tokens on THIS
class of job?" could only be answered by hand-held queries that no one could
re-run — which is why #2576 asks for a script and not a number.

Engine default, for the record: the main loop sends no thinking knob at all
(`app/harness/client.py` builds the payload without one), so every interactive
turn and every worker turn reasons at whatever the launcher's
`--reasoning-parser` flag leaves on. This script prices the tokens;
`scripts/thinking_effort_probe.py` measures what the engine actually honours.
Neither one moves anything — #2576 is report-only, and the flag lives in
`agent-services/bin/start-*.sh`, which a person edits.

Definitions, because the window decides the answer
--------------------------------------------------
* **Window.** Filtered on BOTH stores: runs with `started_at` inside it, usage
  rows with `ts` inside it. Its end is the NEWEST record in the usage store, not
  `datetime.now()` (the `side_effect_traffic_census.py` rule): two runs over an
  unchanged corpus print byte-identical output, so a table quoted into an item
  cannot disagree with the next run. `--all` drops the filter entirely.
* **Class.** The `source` of the earliest run whose `meta_json.session_ids` name
  a usage row's `session_id`. A session named by runs of two different sources
  would be a real ambiguity, so the footer always prints how many there are
  (measured 0 on this box over 2,715 sessions in the 2026-10-04 window); the
  earliest run's class wins so the attribution is deterministic either way.
* **Covered subset.** `reasoning_tokens` is NULL on some rows, and a NULL is not
  a zero — averaging it in would report a measured-and-cheap class next to a
  not-measured one as though they were the same finding. So EVERY token figure on
  a row is over the rows of that class whose `reasoning_tokens` is non-null, and
  the coverage count is printed on the row itself, as the denominator it is.
* **Displacement columns (#2798).** `mean_tool_calls` and `errors_per_tool_call`
  are over that class's rows carrying BOTH `tool_calls` and `tool_errors` —
  counted on the row as `tool_rows` — and `length_bound_turns` counts rows whose
  `stop_reason` is exactly `length` over the rows carrying a stop reason at all,
  counted as `stop_rows`. Both denominators travel with their figures for the same
  reason `cover` does: 566 of the live store's 12,105 rows predate the telemetry
  migration (`app/usage_store.py::_TELEMETRY_COLUMNS`: "NULL = unmeasured … never
  zero"), so a mean over `turns` would price an unmeasured turn as a turn that
  called no tools. This is the column that makes the crowding-out mechanism
  *falsifiable* on this box instead of argued: the mechanism needs a budget the
  turn actually hits, and a `length_bound_turns` of 0 across a week of real
  traffic is the visible absence of that, not a claim that thinking is free.
* **p90 reasoning tokens.** Nearest-rank over the covered values sorted
  ascending: the value at `ceil(0.9 * n)`, 1-based. Stated because two sane
  percentile definitions disagree on small n, and this table gets quoted.
* **Reasoning share.** `sum(reasoning_tokens) / sum(output_tokens)` over the
  covered subset, not a mean of per-row ratios — a 2-token turn and a 30k-token
  turn must not count equally. `reasoning_tokens` is a SUBSET of
  `output_tokens` on this engine (measured: 0 of 2,877 attributed rows in the
  2026-10-04 window carry reasoning above output), so the share is "how much of
  the bill was thinking", not "how much on top of the bill".
* **Outcome rate.** `success / (success + failed)`. `skipped` and `interrupted`
  are excluded from the denominator because neither is an attempt that could have
  gone either way — a `skipped` run never reached the engine (depth gate,
  landing hold), and an `interrupted` one was cut by a restart — but both are
  printed beside the rate, because a class that is 100 % of 3 attempted runs and
  a class that is 100 % of 623 are not the same finding. Hence the rate always
  carries its `n`, and `THIN` marks a denominator under `THIN_DENOM` (30).
* **`unattributed`.** Usage rows whose `session_id` is named by NO run in the
  window: the interactive residual, and the only row without an outcome rate.
  It is a row and not a footnote because a third of the corpus is otherwise
  silently gone from a class table, and 3,283 of 10,278 rows were in it when
  #2576 was triaged. Usage rows sitting outside the window are counted in the
  footer for the same reason: nothing in the store may vanish without a line.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.paths import USAGE_DB, WORKERS_DB, describe_tree  # noqa: E402 — data root

#: The row for turns no worker run claims. Named in #2576's acceptance, so it is
#: a stable label and not a display string.
UNATTRIBUTED = "unattributed"

#: A denominator below this is printed with THIN: too few attempted runs for an
#: outcome rate to say anything about the class (#1266's exclusion discipline).
THIN_DENOM = 30

#: The one `usage.stop_reason` value that records a turn cut off by the budget
#: rather than by the model deciding to stop (`stop`), the turn cap (`max_turns`)
#: or a cancel. #2798's displacement claim is that thinking crowds out executed
#: commands INSIDE a fixed budget, so this value, and only this value, is the
#: budget being hit — counted per class as `length_bound_turns`.
LENGTH_STOP_REASON = "length"


@dataclass
class ClassRow:
    """One row of the table: the turns of one job class and that class's runs."""

    source: str
    sessions: set[str] = field(default_factory=set)
    turns: int = 0
    reasoning: list[int] = field(default_factory=list)   # covered rows only
    output_covered: int = 0                              # sum, covered rows only
    prompt: list[int] = field(default_factory=list)      # EVERY row with a count
    status_counts: dict[str, int] = field(default_factory=dict)
    # ── displacement (#2798) ────────────────────────────────────────────────
    # Tool-call and error counts, appended IN PAIRS over the rows that carry a
    # tool-call counter, so the two sums below are always over one row set.
    tool_calls: list[int] = field(default_factory=list)
    tool_errors: list[int] = field(default_factory=list)
    stop_rows: int = 0        # rows of this class carrying a stop reason
    length_bound: int = 0     # ... and that reason was 'length'

    @property
    def tool_rows(self) -> int:
        """Rows of this class with a non-null `tool_calls` — the tool denominator."""
        return len(self.tool_calls)

    @property
    def mean_tool_calls(self) -> float | None:
        """Tool calls per measured turn, over the rows that carry the counter.

        NULL is never averaged in as a zero, for the same reason `mean_reasoning`
        is over the covered subset: a turn whose telemetry never landed is not a
        turn that called no tools, and mixing the two would price a class the
        engine reported nothing about as the cheapest class on the box.
        """
        return sum(self.tool_calls) / self.tool_rows if self.tool_rows else None

    @property
    def errors_per_tool_call(self) -> float | None:
        """`sum(tool_errors) / sum(tool_calls)`, aggregate like `share`.

        A mean of per-turn ratios would weight a turn that made one call the same
        as one that made a hundred. Returns None when nothing was measured, and
        also when the calls total is zero while errors are not — a ratio with a
        zero denominator and a nonzero numerator has no value to print, and the
        two sums on the row say so exactly.
        """
        if not self.tool_rows:
            return None
        calls, errors = sum(self.tool_calls), sum(self.tool_errors)
        if not calls:
            return None if errors else 0.0
        return errors / calls

    @property
    def length_bound_turns(self) -> int | None:
        """Turns that ended `stop_reason='length'`; None if none was measured.

        This is the column the crowding-out mechanism is tested against: thinking
        displacing executed commands only happens inside a budget the turn
        actually hits, and `length` is that hit. A class whose rows carry no stop
        reason at all reports None — printing 0 there would read as "measured and
        never hit the ceiling", which is a finding, not a blank.
        """
        return self.length_bound if self.stop_rows else None

    @property
    def covered(self) -> int:
        return len(self.reasoning)

    @property
    def mean_reasoning(self) -> float | None:
        return sum(self.reasoning) / self.covered if self.covered else None

    @property
    def p90_reasoning(self) -> int | None:
        if not self.covered:
            return None
        ordered = sorted(self.reasoning)
        return ordered[max(0, math.ceil(0.9 * self.covered) - 1)]

    @property
    def mean_output(self) -> float | None:
        return self.output_covered / self.covered if self.covered else None

    @property
    def mean_prompt(self) -> float | None:
        """Mean prompt tokens per turn, over EVERY row that carries a count.

        Deliberately NOT the covered subset: this is the price of a per-request
        thinking knob (a knob changes the stamped system message, so the request's
        whole prompt leaves the prefix cache — `app/harness/finalizer.py:24-37`
        declines one for this reason), and the price is charged on requests whose
        reasoning counter came back NULL just as much as on the ones that did. A
        class with no counted prompt at all reports None, not zero: zero would say
        the turns were free.
        """
        return sum(self.prompt) / len(self.prompt) if self.prompt else None

    @property
    def share(self) -> float | None:
        """Reasoning tokens as a fraction of output tokens, covered subset."""
        if not self.covered or not self.output_covered:
            return None
        return sum(self.reasoning) / self.output_covered

    @property
    def outcome(self) -> dict[str, Any]:
        """`success/(success+failed)`, with the excluded counts kept on the row."""
        sc = self.status_counts
        success, failed = sc.get("success", 0), sc.get("failed", 0)
        denom = success + failed
        excluded = {k: v for k, v in sc.items() if k not in ("success", "failed")}
        return {
            "rate": (success / denom) if denom else None,
            "n": denom,
            "thin": bool(denom) and denom < THIN_DENOM,
            "success": success,
            "failed": failed,
            "excluded": excluded,
        }

    def as_dict(self) -> dict[str, Any]:
        o = self.outcome
        return {
            "class": self.source,
            "sessions": len(self.sessions),
            "turns": self.turns,
            "reasoning_coverage": self.covered,
            "mean_reasoning_tokens": (round(self.mean_reasoning, 1)
                                      if self.mean_reasoning is not None else None),
            "p90_reasoning_tokens": self.p90_reasoning,
            "mean_output_tokens": (round(self.mean_output, 1)
                                   if self.mean_output is not None else None),
            "mean_input_tokens": (round(self.mean_prompt, 1)
                                  if self.mean_prompt is not None else None),
            "input_rows": len(self.prompt),
            "reasoning_share_of_output": (round(self.share, 4)
                                          if self.share is not None else None),
            # The sums the share and the means are derived from, at full precision.
            # Without them the committed witness cannot prove itself: `share` is
            # `round(sum_r/sum_o, 4)` and the two means are each rounded to 0.1, so
            # re-deriving the share from the means can only ever agree to ~1e-4, and
            # a check with a tolerance that wide cannot catch a share that drifted
            # from its own sums. With them, the pinning test is exact.
            "reasoning_total": sum(self.reasoning) if self.covered else None,
            "output_total": self.output_covered if self.covered else None,
            # ── displacement (#2798) ──────────────────────────────────────────
            # Each ratio travels with the count of rows it was derived over, as
            # `reasoning_coverage` and `input_rows` already do, and with its two
            # sums at full precision so a witness can prove its own rounding.
            "mean_tool_calls": (round(self.mean_tool_calls, 1)
                                if self.mean_tool_calls is not None else None),
            "errors_per_tool_call": (round(self.errors_per_tool_call, 4)
                                     if self.errors_per_tool_call is not None else None),
            "length_bound_turns": self.length_bound_turns,
            "tool_rows": self.tool_rows,
            "stop_rows": self.stop_rows,
            "tool_calls_total": (sum(self.tool_calls) if self.tool_rows else None),
            "tool_errors_total": (sum(self.tool_errors) if self.tool_rows else None),
            "outcome_rate": (round(o["rate"], 4) if o["rate"] is not None else None),
            "outcome_n": o["n"],
            "outcome_thin": o["thin"],
            "success": o["success"],
            "failed": o["failed"],
            "skipped": o["excluded"].get("skipped", 0),
            "interrupted": o["excluded"].get("interrupted", 0),
        }


def _read_only(db: Path) -> sqlite3.Connection:
    """Open a store WITHOUT creating it: a missing store is a finding, not an
    empty one. `sqlite3.connect` on a bare path fabricates a 0-byte database,
    and a census that prints a clean empty table is the worst kind of zero."""
    if not db.exists():
        raise SystemExit(f"{db}: no such store (refusing to create one)")
    return sqlite3.connect(f"file:{db}?mode=ro", uri=True)


def build_table(*, workers_db: Path, usage_db: Path, since: str | None) -> dict[str, Any]:
    """The class table over the two stores. Pure read; returns dicts, prints nothing.

    `since` is an ISO date compared as a STRING against `runs.started_at` and
    `usage.ts` — both are stored `%Y-%m-%dT%H:%M:%S`, so a lexical compare is the
    right one and no timezone math is invented here.
    """
    where = " WHERE started_at >= ?" if since else ""
    args: tuple = (since,) if since else ()
    con = _read_only(workers_db)
    try:
        run_rows = con.execute(
            f"select source, started_at, status, meta_json from runs{where}"
            " order by started_at, rowid", args).fetchall()
    finally:
        con.close()

    # Earliest run wins, so a session named by two sources still gets a
    # deterministic class instead of a dictionary-order one.
    session_class: dict[str, str] = {}
    collisions: set[str] = set()
    classes: dict[str, ClassRow] = {}
    for source, _started, status, meta_json in run_rows:
        row = classes.setdefault(source, ClassRow(source=source))
        row.status_counts[status] = row.status_counts.get(status, 0) + 1
        try:
            sids = json.loads(meta_json or "{}").get("session_ids") or []
        except (TypeError, ValueError):
            sids = []
        for sid in sids:
            if not isinstance(sid, str):
                continue
            seen = session_class.get(sid)
            if seen is None:
                session_class[sid] = source
                classes[source].sessions.add(sid)
            elif seen != source:
                collisions.add(sid)

    ucon = _read_only(usage_db)
    try:
        # `stop_reason`, `tool_calls` and `tool_errors` are the #2798 displacement
        # columns: NULL on every row written before the telemetry migration
        # (`app/usage_store.py::_TELEMETRY_COLUMNS`), which is why each figure
        # below carries its own denominator instead of dividing by `turns`.
        cols = ("session_id, reasoning_tokens, output_tokens, input_tokens,"
                " stop_reason, tool_calls, tool_errors")
        if since:
            rows = ucon.execute(
                f"select {cols} from usage where ts >= ?", (since,)).fetchall()
            outside = ucon.execute(
                "select count(*) from usage where ts < ?", (since,)).fetchone()[0]
        else:
            rows = ucon.execute(f"select {cols} from usage").fetchall()
            outside = 0
        newest = ucon.execute("select max(ts) from usage").fetchone()[0]
    finally:
        ucon.close()

    unattributed = ClassRow(source=UNATTRIBUTED)
    null_sessions = 0
    for sid, reasoning, output, prompt, stop_reason, tool_calls, tool_errors in rows:
        target = classes.get(session_class[sid]) if sid in session_class else None
        if target is None:
            target = unattributed
            if sid is None:
                null_sessions += 1
        target.turns += 1
        if prompt is not None:
            target.prompt.append(int(prompt))
        if stop_reason is not None:
            target.stop_rows += 1
            if stop_reason == LENGTH_STOP_REASON:
                target.length_bound += 1
        if tool_calls is not None and tool_errors is not None:
            # Paired, deliberately: an error count without its call count would put
            # errors over a denominator that never counted them, and a call count
            # without its error count would divide by a total missing its own
            # numerator's rows. Both columns come from the same insert
            # (`app/usage_store.py::_TELEMETRY_COLUMNS`); a row missing either one
            # is out of both figures, and `tool_rows` is the count that says so.
            target.tool_calls.append(int(tool_calls))
            target.tool_errors.append(int(tool_errors))
        if reasoning is not None:
            # Both sums of the share grow together, or `mean_output` and `share`
            # would be over different row sets than the reasoning figures on the
            # same row — an uncovered row's output tokens belong to the turn
            # count, never to a token column.
            target.reasoning.append(int(reasoning))
            target.output_covered += int(output or 0)
    # The row still carries a session count, because "3 turns in 1 session" and
    # "3 turns in 3 sessions" are different claims about the interactive residual.
    unattributed.sessions = {sid for sid, *_rest in rows
                             if sid not in session_class}

    table = sorted(classes.values(), key=lambda r: r.source)
    total_turns = sum(r.turns for r in table) + unattributed.turns
    return {
        # The corpus identity, because `app.paths` resolves inside whichever tree
        # the module was imported from: run this from an automod worktree with no
        # explicit `--workers-db` and the defaults point at a near-empty
        # `.lloyd-data/`, printing a clean table over nothing. A measurement that
        # cannot name the store it read is how a `tree=worktree` zero gets quoted
        # into an item as a finding.
        "corpus": {"describe_tree": describe_tree(),
                   "workers_db": str(workers_db), "usage_db": str(usage_db)},
        "window": {"since": since, "until": newest},
        "runs": len(run_rows),
        "sessions_named_by_runs": len(session_class),
        "multi_class_sessions": len(collisions),
        "usage_rows": len(rows),
        "usage_rows_outside_window": outside,
        "turns_attributed": total_turns - unattributed.turns,
        "null_session_id_rows": null_sessions,
        "rows": [r.as_dict() for r in table] + [unattributed.as_dict()],
    }


#: The three #2798 fields a row must carry for the displacement columns to exist
#: at all. Their ABSENCE is the signal: a row loaded from the committed 2026-10-11
#: witness has no tool-call counter in it, so a table of those rows has no
#: displacement column — a column of dashes over an unmeasured corpus would read
#: as "measured, and the turns made no calls".
DISPLACEMENT_KEYS = ("mean_tool_calls", "errors_per_tool_call",
                     "length_bound_turns")

#: Markdown header labels for the same three, in cell order.
DISPLACEMENT_COLUMNS = ("meanT", "errPerCall", "lenBound")


def displacement_cells(row: dict[str, Any], *, with_n: bool = True) -> list[str]:
    """The three displacement cells for ONE row, read from that row's own fields.

    Every figure carries the count it was derived over, as `outcome` carries its
    `n`: a mean of 2.4 calls over 40 measured turns and one of 2.4 over 4,000 are
    different findings, and a reader cannot tell them apart without the
    denominator. `errPerCall` shares `meanT`'s denominator (both counters are
    written by the same insert), which is why it does not repeat it.

    The sentinel is the caller's: the aligned table prints a hyphen, the markdown
    one the report's en dash, and a class whose rows carry no counter prints
    whichever it is — never a 0, which would say the turns called nothing rather
    than that nothing was counted. A class measured with no tool call and no
    length-bound turn prints real zeros.
    """
    def cell(key: str, spec: str, denom: str | None) -> str:
        value = row.get(key)
        if value is None:
            return ""            # the caller substitutes its own dash
        text = format(value, spec)
        n = row.get(denom) if denom else None
        return f"{text} n={n:,}" if n is not None else text

    tool_denom = None if not with_n else "tool_rows"
    stop_denom = None if not with_n else "stop_rows"
    return [cell("mean_tool_calls", ",.1f", tool_denom),
            cell("errors_per_tool_call", ",.3f", None),
            cell("length_bound_turns", ",d", stop_denom)]


def aligned_displacement(row: dict[str, Any]) -> str:
    """The three cells padded to the aligned table's widths, `-` for an absent one.

    Column widths 9/10/7 with the `n=` dropped, for the same reason the aligned
    `outcome` cell prints `96%` where the markdown one prints `96 % n=1133`: the
    aligned page is a CLI table, and its denominators are the `--json` row's
    `tool_rows` and `stop_rows`, which its legend points at.
    """
    mean_t, err, length_bound = displacement_cells(row, with_n=False)

    def pad(text: str, width: int) -> str:
        return "-".rjust(width) if not text else text.rjust(width)

    return f"{pad(mean_t, 9)}{pad(err, 10)}{pad(length_bound, 7)}"


def render(result: dict[str, Any]) -> str:
    """The human-readable table. Every column is a definition, not a guess."""
    out: list[str] = []
    w = result["window"]
    c = result["corpus"]
    out.append(f"#2576 reasoning-by-job-class  window: since={w['since'] or '(all)'} "
               f"until={w['until'] or '(empty store)'}  runs={result['runs']}")
    out.append(f"corpus: {c['describe_tree']}")
    out.append(f"stores: workers={c['workers_db']} usage={c['usage_db']}")
    out.append("")
    # #2798: the displacement columns appear only when some row of THIS table
    # carries them. A table built from the committed 2026-10-11 witness has no
    # such keys at all, and printing three columns of dashes over it would claim
    # the witness was measured and came out with no tool calls, which is the
    # inversion every denominator on this page exists to prevent.
    displacing = any(k in row for row in result["rows"] for k in DISPLACEMENT_KEYS)
    header = (f"{'class':<18}{'sess':>6}{'turns':>7}{'cover':>7}"
              f"{'meanR':>9}{'p90R':>9}{'meanOut':>9}{'meanIn':>9}{'share':>8}"
              f"{'outcome':>10}{'s/f/s/i':>16}"
              + (f"{'meanT':>9}{'err/call':>10}{'lenB':>7}" if displacing else ""))
    out.append(header)
    out.append("-" * len(header))
    for row in result["rows"]:
        pct = "n/a" if row["outcome_rate"] is None else (
            f"{row['outcome_rate'] * 100:.0f}%" + (" THIN" if row["outcome_thin"] else ""))
        share = "-" if row["reasoning_share_of_output"] is None else (
            f"{row['reasoning_share_of_output'] * 100:.1f}%")
        counts = (f"{row['success']}/{row['failed']}/"
                  f"{row['skipped']}/{row['interrupted']}")

        def cell(value: float | None, width: int = 9) -> str:
            """`-` for "no covered turn", never a number.

            A `-1` in a token column is a measurement-shaped object that measures
            nothing, and a zero-width default would say "zero tokens" about a class
            whose engine reported nothing at all. `cover` above is the column that
            says which of the two it is.
            """
            return "-".rjust(width) if value is None else f"{value:>{width}.0f}"

        out.append(
            f"{row['class']:<18}{row['sessions']:>6}{row['turns']:>7}"
            f"{row['reasoning_coverage']:>7}"
            f"{cell(row['mean_reasoning_tokens'])}"
            f"{cell(row['p90_reasoning_tokens'])}"
            f"{cell(row['mean_output_tokens'])}"
            f"{cell(row['mean_input_tokens'])}"
            f"{share:>8}{pct:>10}{counts:>16}"
            + (aligned_displacement(row) if displacing else ""))
    out.append("")
    out.append(f"outcome rate = success/(success+failed); skipped/interrupted excluded "
               f"from that denominator and printed beside it as s/f/s/i; "
               f"THIN = denominator under {THIN_DENOM}, with its n attached.")
    out.append("meanR/p90R/meanOut are over the COVERED subset (non-null "
               "reasoning_tokens), whose count is the `cover` column; "
               "p90 = nearest-rank at ceil(0.9*n); share = sum(reason)/sum(output).")
    out.append("meanIn is mean PROMPT tokens over every row of the class that carries "
               "a count — not the covered subset — because it is the prefill a "
               "per-request thinking knob re-charges when its stamped system message "
               "leaves the prefix cache (app/harness/finalizer.py:24-37).")
    if displacing:
        out.append("meanT = mean tool calls per turn and err/call = tool errors per tool "
                   "call, both over that class's rows carrying a tool-call counter; "
                   "lenB = turns that ended stop_reason='length', over the rows carrying a "
                   "stop reason. Their denominators are the --json row's `tool_rows` and "
                   "`stop_rows`; a `-` means NO row of that class carries that counter, "
                   "which is not the same finding as a turn that called nothing.")
    out.append(f"turns attributed {result['turns_attributed']} of {result['usage_rows']} "
               f"usage rows in the window; unattributed = the interactive residual, "
               f"no run names its session.")
    out.append(f"usage rows outside the window (not in any figure above): "
               f"{result['usage_rows_outside_window']}; rows with a NULL session_id: "
               f"{result['null_session_id_rows']}; sessions named by more than one job "
               f"class: {result['multi_class_sessions']} (earliest run's class wins).")
    return "\n".join(out)


#: Column order and formatting for the markdown rendering. The vault report
#: reprints this table, and its previous copy was hand-transcribed — which has
#: already drifted once (a `THIN` marker on a class whose witness denominator is
#: 136, i.e. not thin) — so the rows now have exactly one formatter, and the
#: report's copy is pinned to its output.
MD_COLUMNS = ("class", "sess", "turns", "cover", "meanR", "p90R", "meanOut",
              "meanIn", "share", "outcome", "s/f/s/i")

#: Em-dash-free en dash for "no covered turn in this class", matching the report.
NO_VALUE = "–"


def _md_cell(row: dict[str, Any], *, displacement: bool = False) -> list[str]:
    """One row's markdown cells, derived only from its witness fields.

    `displacement` appends the three #2798 cells. It is decided by the TABLE
    (`markdown_rows`), never per row, because a table whose rows have different
    cell counts is a ragged markdown table — and the report's own pin asserts
    every row has the header's pipe count.
    """
    def num(value: float | None) -> str:
        return NO_VALUE if value is None else f"{value:,.0f}"

    share = row["reasoning_share_of_output"]
    o = row["outcome_rate"]
    if o is None:
        outcome = "n/a"
    else:
        outcome = f"{o * 100:.0f} % n={row['outcome_n']}"
        if row["outcome_thin"]:
            outcome += " THIN"
    cells = [row["class"], f"{row['sessions']:,}", f"{row['turns']:,}",
             f"{row['reasoning_coverage']:,}", num(row["mean_reasoning_tokens"]),
             num(row["p90_reasoning_tokens"]), num(row["mean_output_tokens"]),
             num(row["mean_input_tokens"]),
             (NO_VALUE if share is None else f"{share * 100:.1f} %"),
             outcome,
             f"{row['success']}/{row['failed']}/{row['skipped']}/{row['interrupted']}"]
    if displacement:
        cells += [c or NO_VALUE for c in displacement_cells(row)]
    return cells


def markdown_rows(result: dict[str, Any]) -> str:
    """The class table as the vault report prints it, from the same dict as `render`.

    Highest reasoning share first, then by class name, and a class with no covered
    turn last — so the order carries the finding the report is about (which class
    spends the most of its bill on thinking) instead of an alphabetical one that
    happens to interleave the four turn-less classes through the middle. No bold:
    emphasis cannot be derived from a field, and a formatter that had to be told
    which cells to embolden is a formatter that can drift from the data.
    """
    def sort_key(row: dict[str, Any]) -> tuple[int, float, str]:
        share = row["reasoning_share_of_output"]
        return (share is None, -(share or 0.0), row["class"])

    # The displacement columns exist only for a table whose rows carry them, so
    # the committed 2026-10-11 witness — whose rows have no such keys — renders
    # byte-identically and the report's byte-for-byte pin of it stays green.
    displacing = any(k in row for row in result["rows"] for k in DISPLACEMENT_KEYS)
    columns = MD_COLUMNS + (DISPLACEMENT_COLUMNS if displacing else ())
    lines = ["| " + " | ".join(columns) + " |",
             "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in sorted(result["rows"], key=sort_key):
        lines.append("| " + " | ".join(_md_cell(row, displacement=displacing)) + " |")
    return "\n".join(lines)


WITNESS_HEAD = """# #{item} per-job-class reasoning share and outcome rate

Committed witness for #{item} (`knowledge/thinking-effort-per-job-class-2026-10-11.md`). Produced by `scripts/reasoning_by_job_class.py {flags} --workers-db … --usage-db …` and rewritten verbatim by that command's `--witness-out`, over the run-derived `window` that scan closed on. `--all` keys its window off the newest usage row, so this file is the pinned corpus: the same command next week prints a different table by construction.

`corpus.workers_db` and `corpus.usage_db` are the bytes that were opened, and they are what the numbers describe. `corpus.describe_tree` is a different fact — the tree the *code* was imported from — and a round worktree has no `.lloyd-data/` of its own to scan, so the stores must be named explicitly there; an unnamed default inside a worktree either resolves to the empty one or the script refuses to open it. Read the two paths, not the tree line.

Numbers only — a class name, counts, token totals. No prompt, no turn content, no session id, which is the property that lets a census of every turn in the store be committed at all.
"""


def write_witness(result: dict[str, Any], path: Path, *, flags: str) -> None:
    """Commit the scan as the tracked `.md` witness the report cites.

    A tracked `.md` holding one fenced JSON block, because `*.json` is ignored
    repo-wide and `.gitignore` is a denied path for an automod round, so the
    negation that would track a raw `.json` witness is not this script's to
    write. `scripts/reasoning_by_job_class.py --all --witness-out <path>` is the
    whole route, which is the point: the report quotes these bytes, and hand
    pasting them is the step that drifted before.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(WITNESS_HEAD.format(item=2576, flags=flags)
                    + "\n```json\n"
                    + json.dumps(result, indent=2, sort_keys=True)
                    + "\n```\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers-db", type=Path, default=WORKERS_DB)
    ap.add_argument("--usage-db", type=Path, default=USAGE_DB)
    ap.add_argument("--days", type=int, default=7,
                    help="window length, counted back from the newest usage row")
    ap.add_argument("--all", action="store_true", help="no window: the whole store")
    ap.add_argument("--json", action="store_true", dest="as_json")
    ap.add_argument("--markdown", action="store_true",
                    help="print the vault report's table form instead of the aligned one")
    ap.add_argument("--witness-out", type=Path, default=None,
                    help="write the tracked .md witness the report cites to this path")
    args = ap.parse_args(argv)

    since = None
    if not args.all:
        probe = _read_only(args.usage_db)
        try:
            newest = probe.execute("select max(ts) from usage").fetchone()[0]
        finally:
            probe.close()
        if not newest:
            raise SystemExit(f"{args.usage_db}: usage table is empty")
        from datetime import datetime, timedelta
        since = (datetime.strptime(newest[:19], "%Y-%m-%dT%H:%M:%S")
                 - timedelta(days=args.days)).strftime("%Y-%m-%dT%H:%M:%S")

    result = build_table(workers_db=args.workers_db, usage_db=args.usage_db,
                         since=since)
    if args.witness_out:
        write_witness(result, args.witness_out,
                      flags="--all" if args.all else f"--days {args.days}")
    if args.as_json:
        print(json.dumps(result, indent=2))
    elif args.markdown:
        print(markdown_rows(result))
    else:
        print(render(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
