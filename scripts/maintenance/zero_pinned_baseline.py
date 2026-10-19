#!/usr/bin/env python3
"""The count #706's zero-pinned invariant has never had a referent in code (#3022).

The invariant, carried from #522 into #706's owed entry, is a sentence about a
NUMBER: "tasks pinned at exactly 0.00 must not increase". Before this module nothing
in the tree computed that number, so the sentence could not be true or false of any
state the code was in — `git grep -n "must not increase" -- '*.py'` at 13d24da8
returns 0 hits, `git grep -niE "zero[_-]?pinned|pinned[_-]?(at|baseline)"` finds no
counter, and all 30 test nodes of ``tests/test_bench_invariants.py`` are structural
over the task manifest (unique ids, required fields, the safety-critical set) with
none opening the ledger. The one place the phenomenon is named in prose is
``scripts/autoresearch/promote.py`` ("always fails sits at mean composite exactly
0.00"), and prose counts nothing. So a promoted mined task could add a zero-pinned
task and no gate, report or test would have anything to notice.

WHAT A TASK HAS TO DO TO BE PINNED
    The unit is the per-trial row of the research ledger: one row is one trial of one
    task by one variant in one round (a row with no ``event`` key — the ``spec`` /
    ``split`` / ``round_summary`` / ``decision`` rows are 929 of the file's 12,792 and
    carry no score). A task is pinned when the MAXIMUM ``composite_score`` over its
    trials stamped inside the trailing window is EXACTLY 0.00 — the best thing it
    managed in the window is nothing, which is what "always fails" means. Max, not
    min: one 0.00 trial beside a 0.50 trial is a task that can score, and a min would
    file it as broken. Equality is exact and carries no epsilon, because the
    invariant is spelled "exactly 0.00"; a task whose best trial is 0.001 is not
    pinned and must not read as if it were.

THE NULL ROWS ARE THE FILTER THAT DECIDES THE NUMBER
    Of the ledger's 11,863 per-trial rows, 3,491 carry ``"composite_score": null`` —
    the not-rankable shape ``scripts/autoresearch/judge.py`` writes for #416/#646/#698
    — and every one of them sits inside a default-length window. A reader that coerced
    null to ``0.0`` would report a far larger count than 9 and look plausible doing it,
    so a null (or absent, or non-numeric) score is DROPPED, never read as zero. Same
    selector as ``scripts/autoresearch/promotion_fp_rate.py::per_task_rows``:
    ``not row.get("event") and row.get("composite_score") is not None``.

WHAT 9 IS, AND WHAT IT IS NOT
    ``DECLARED_BASELINE_ZERO_PINNED`` below is a dated ruling, not a property of the
    tree. The window slides every day, and the split file rotated two tasks on the day
    the nine were counted — one of them, ``bench_005_replay_memory_update``, is itself
    one of the nine. A measured count that no longer equals the declared one is a
    re-declaration owed to a person (with its date and its definition), not a
    regression this report alarms on, so it prints the drift and still exits 0. The
    definition is the other half of the number: the same filter restricted to
    ``variant_id.startswith("BASELINE")`` returned 10 on the day the constant was set,
    because it adds ``bench_020_skill_inventory_coverage_gap``. #706's 2026-10-02
    ruling used that BASELINE-only spelling; the 2026-10-19 ruling that chose the
    constant chose all rows. This module is the all-rows definition, so the constant
    and its command agree; a counter on the other one would disagree with its own
    number.

NOT THIS MODULE
    ``scripts/autoresearch/coverage_leg.py`` prints a list of graded failures it calls
    "unreachable (today's failure, pass@n never passes)". It is the nearest reader in
    spirit and measures a different thing: pass@n over repeated draws the coverage arm
    takes itself, not the maximum ``composite_score`` of trials already in the ledger.
    It is neither imported nor replaced here. And the second defect #3022's triage
    found — the promote endpoint in ``app/routers/workers.py`` consults only the task
    defect and the assertion-denial guards, never the ``out_of_band`` band verdict the
    bench-mining calibration entry point (``workers/sources/bench_mine.py``) writes, so
    a candidate whose calibration mean is 0.0 promotes with HTTP 200 and ADDS a
    zero-pinned task — is a refusal that belongs in the promotion lane. This file closes
    the "nothing computes the number" half only, which is the half that made the
    invariant unsayable.

    THAT ENTRY POINT IS NAMED HERE BY ITS MODULE, NOT ITS FUNCTION. #1769's guard
    ``tests/test_relabel_stale_bench_calibration.py::test_no_call_to_the_calibration_entry_point_under_scripts_maintenance``
    is a plain ``git grep`` over all of ``scripts/maintenance/``, prose included, so
    spelling its name anywhere in this directory red-suites the tree for a sentence.
    The guard is a property of the directory, not of this file; the name is one
    fragment away, which is why the wording above reads as it does.

USAGE
    python scripts/maintenance/zero_pinned_baseline.py [--ledger PATH]
                                                      [--window-days N] [--now ISO-8601]
                                                      [--json]

    ``--json`` prints the measurement and its denominators as one object, for a
    nightly report to read. There is deliberately no ``--record``: this report writes
    nothing anywhere, so no run of it can become the input the next run measures
    against. Exit 0 whether or not the measurement agrees with the declaration — the
    only nonzero exit is a ledger that cannot be read or a stamp that cannot be
    parsed, and both refusals name the declared baseline and the window they could not
    measure.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
# Appended, not inserted: `app/` holds top-level modules that would shadow a package
# of the same name for anything else in this process (the rule
# `scripts/maintenance/gate_to_land_lag.py` states in full).
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from app.paths import PIPELINE_DIR, data_root_for_tree, production_data_root  # noqa: E402

#: Where the per-trial ledger sits under the runtime data root. One spelling, so the
#: reader that resolves a root and the fallback that does not cannot drift apart.
LEDGER_RELPATH = Path("_pipeline") / "research" / "ledger.jsonl"

#: The path named when no data root holds a ledger at all.
DEFAULT_LEDGER = PIPELINE_DIR / "research" / "ledger.jsonl"

#: THE ONLY WINDOW LENGTH WRITTEN INTO THIS FILE. Every default — the helper's, the
#: report's, the CLI's — reads this constant, because a second literal is a second
#: definition of the window the declared baseline was measured under, and the two
#: would drift the day somebody nudged one of them. ``tests/
#: test_zero_pinned_baseline.py`` counts the numeric literals and fails on a second.
DEFAULT_WINDOW_DAYS = 30

#: #706's declared baseline: the number of bench tasks pinned at exactly 0.00 that a
#: person ruled on 2026-10-19, over the all-rows definition above and a default-length
#: trailing window — 9 of the 28 tasks with a scored trial in it. Recompute it with:
#:     python scripts/maintenance/zero_pinned_baseline.py
#: which reads the live ledger over ``DEFAULT_WINDOW_DAYS`` and prints the ids beside
#: this number. A measurement that disagrees is a re-declaration owed to a person with
#: a new date, not a value for this constant to chase.
DECLARED_BASELINE_ZERO_PINNED = 9

#: The date that ruling was made, printed beside every report so a reader can never
#: mistake the dated measurement for a property of the tree.
DECLARED_ON = "2026-10-19"

def default_ledger() -> Path:
    """The real ledger, resolved lazily so importing this module touches no disk.

    The ledger lives under the runtime data root, and an automod worktree's root has
    none, so a run from inside a round would otherwise resolve to a path that does not
    exist. Search order — the data root of the tree named by ``LLOYD_HOME``, this
    process's data root, then production's — and the first that actually holds the
    file wins; the fallback is :data:`DEFAULT_LEDGER`. The same order as
    ``scripts/autoresearch/replay_frontier_selection.py::default_ledger``, for the same
    reason: an import-time ``exists()`` is a filesystem probe inside every import of
    this module, including the gate's.
    """
    env_tree = os.environ.get("LLOYD_HOME", "")
    roots = ([data_root_for_tree(Path(env_tree).expanduser())] if env_tree else []) + [
        PIPELINE_DIR.parent, production_data_root()]
    for root in roots:
        candidate = root / LEDGER_RELPATH
        if candidate.is_file():
            return candidate
    return DEFAULT_LEDGER


def parse_stamp(raw: Any) -> datetime | None:
    """A row's ``created_at`` as an aware UTC datetime, or None when there is none.

    Ledger stamps are written ``...Z`` (the ISO-8601 UTC form ``judge.py`` appends); a
    stamp that arrives naive is read as UTC, which is what every other reader on this
    box already assumes of a stamp with no offset. A stamp that will not parse at all
    is ``None`` — and a row with no usable stamp is DROPPED rather than counted,
    because a row cannot be placed in a window it cannot be dated into.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _score_of(row: Mapping[str, Any]) -> float | None:
    """The row's score if it has a numeric one, else None (never 0.0).

    ``None``, an absent key, and a non-numeric value all land here together, and a
    boolean too — ``True`` is an ``int`` to Python and a score of 1 by accident is not
    a score. None of them is read as 0.00: that coercion is the single way this count
    over-reports, and 3,491 of the live rows are null.
    """
    raw = row.get("composite_score")
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    return None


@dataclass(frozen=True)
class Measurement:
    """One window's count, with every denominator that makes it readable.

    ``tasks_scored`` is the population the count is a share OF: tasks with at least
    one usable trial in the window. Without it "9" has no scale, and a report that
    cannot print "9 of 28" cannot tell a corpus that grew from a corpus that rotted.
    """

    #: Task ids whose best trial in the window is exactly 0.00, sorted.
    pinned: tuple[str, ...] = ()
    window_days: int = DEFAULT_WINDOW_DAYS
    started_at: datetime | None = None
    ended_at: datetime | None = None
    #: Tasks with at least one scored, dated, in-window trial.
    tasks_scored: int = 0
    #: The trials behind those tasks.
    trials_used: int = 0
    rows_read: int = 0
    rows_event: int = 0
    #: Rows with no score at all — ``"composite_score": null`` or the key absent. The
    #: not-rankable shape, and the single largest way this count could be over-stated.
    rows_null_score: int = 0
    #: Rows carrying a score that is not a number (a string, a bool): corrupt, dropped.
    rows_bad_score: int = 0
    rows_undated: int = 0
    rows_no_task: int = 0
    rows_outside_window: int = 0
    #: JSON lines the file-level reader could not parse (0 for a row-level measure).
    rows_unparsed: int = 0
    #: Ledger the file-level reader opened, when there was one.
    ledger: Path | None = None
    #: ``json.loads`` failures kept for a human, capped so a rotted file cannot grow RAM.
    unparsed_examples: list[str] = field(default_factory=list, repr=False)

    @property
    def pinned_count(self) -> int:
        return len(self.pinned)

    @property
    def agrees_with_declaration(self) -> bool:
        return self.pinned_count == DECLARED_BASELINE_ZERO_PINNED

    def as_dict(self) -> dict[str, Any]:
        """The measurement as JSON-ready scalars, for ``--json`` and a nightly row."""
        return {
            "invariant": "zero-pinned tasks must not increase (#706)",
            "declared_baseline": DECLARED_BASELINE_ZERO_PINNED,
            "declared_on": DECLARED_ON,
            "definition": ("max composite_score per task over per-trial rows stamped "
                           "inside a trailing window is exactly 0.00; null scores dropped"),
            "ledger": str(self.ledger) if self.ledger else None,
            "window_days": self.window_days,
            "window_started_at": (self.started_at.isoformat().replace("+00:00", "Z")
                                  if self.started_at else None),
            "window_ends_at": (self.ended_at.isoformat().replace("+00:00", "Z")
                               if self.ended_at else None),
            "pinned_count": self.pinned_count,
            "pinned": list(self.pinned),
            "tasks_scored": self.tasks_scored,
            "trials_used": self.trials_used,
            "agrees_with_declaration": self.agrees_with_declaration,
            "rows_read": self.rows_read,
            "rows_event": self.rows_event,
            "rows_null_score": self.rows_null_score,
            "rows_bad_score": self.rows_bad_score,
            "rows_undated": self.rows_undated,
            "rows_no_task": self.rows_no_task,
            "rows_outside_window": self.rows_outside_window,
            "rows_unparsed": self.rows_unparsed,
        }


def measure(rows: Iterable[Mapping[str, Any]], *,
            window_days: int = DEFAULT_WINDOW_DAYS,
            now: datetime | None = None,
            rows_unparsed: int = 0,
            ledger: Path | None = None,
            unparsed_examples: list[str] | None = None) -> Measurement:
    """Group ledger rows by task and return the ones pinned at exactly 0.00.

    ``rows`` is any iterable of ledger row dicts — a parsed file, or a fixture's list.
    A row counts only when it is not an ``event`` row, carries a numeric
    ``composite_score``, carries a parseable ``created_at``, names a ``task_id``, and
    is stamped no more than ``window_days`` before ``now`` (default: this instant).
    Every rejection is counted in the returned denominators, because a count whose
    drops are invisible is the number that made #706 unsayable in the first place.
    """
    moment = now or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    cutoff = moment - timedelta(days=window_days)
    best: dict[str, list[float]] = defaultdict(list)
    counts = {"read": 0, "event": 0, "null": 0, "bad": 0, "undated": 0, "no_task": 0,
              "outside": 0}
    for row in rows:
        counts["read"] += 1
        if row.get("event"):
            counts["event"] += 1
            continue
        if row.get("composite_score") is None:
            # No score recorded. The explicit null and the absent key are one fact to
            # this reader — both are the #416/#646/#698 not-rankable shape — and
            # neither is 0.00.
            counts["null"] += 1
            continue
        score = _score_of(row)
        if score is None:
            counts["bad"] += 1  # a score that is not a number: a corrupt row
            continue
        stamp = parse_stamp(row.get("created_at"))
        if stamp is None:
            counts["undated"] += 1
            continue
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            counts["no_task"] += 1
            continue
        if stamp < cutoff:
            counts["outside"] += 1
            continue
        best[task_id].append(score)
    pinned = sorted(task for task, scores in best.items() if max(scores) == 0.00)
    return Measurement(
        pinned=tuple(pinned),
        window_days=window_days,
        started_at=cutoff,
        ended_at=moment,
        tasks_scored=len(best),
        trials_used=sum(len(v) for v in best.values()),
        rows_read=counts["read"],
        rows_event=counts["event"],
        rows_null_score=counts["null"],
        rows_bad_score=counts["bad"],
        rows_undated=counts["undated"],
        rows_no_task=counts["no_task"],
        rows_outside_window=counts["outside"],
        rows_unparsed=rows_unparsed,
        ledger=ledger,
        unparsed_examples=list(unparsed_examples or []),
    )


def pinned_at_zero(rows: Iterable[Mapping[str, Any]],
                   window_days: int = DEFAULT_WINDOW_DAYS,
                   *, now: datetime | None = None) -> list[str]:
    """The task ids pinned at exactly 0.00, sorted — the invariant's referent.

    Thin over :func:`measure`, which is where the filters live; the two cannot
    disagree because there is only one pass.
    """
    return list(measure(rows, window_days=window_days, now=now).pinned)


def read_rows(path: Path) -> tuple[list[dict], list[str]]:
    """Parse a ledger file into row dicts, and return the JSON lines that failed.

    The whole file is materialised on purpose: 12,792 rows is a few megabytes and every
    sibling reader of this ledger (``promotion_fp_rate._rows``) does the same. A line
    that will not parse is kept OUT of the rows and handed back as text, because a
    silently skipped line is how a reader reports a count over a corpus it never
    actually read.
    """
    rows: list[dict] = []
    unparsed: list[str] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                if len(unparsed) < 5:
                    unparsed.append(text[:200])
                continue
            if isinstance(parsed, dict):
                rows.append(parsed)
            elif len(unparsed) < 5:
                unparsed.append(text[:200])
    return rows, unparsed


def measure_from_ledger(path: Path, *, window_days: int = DEFAULT_WINDOW_DAYS,
                        now: datetime | None = None) -> Measurement:
    """Read ``path`` and measure it — the CLI's whole job."""
    rows, unparsed = read_rows(path)
    return measure(rows, window_days=window_days, now=now, rows_unparsed=len(unparsed),
                   ledger=path, unparsed_examples=unparsed)


def report_line(m: Measurement, *, declared: int = DECLARED_BASELINE_ZERO_PINNED,
                declared_on: str = DECLARED_ON) -> str:
    """The one line the invariant is worth: declaration, window, count and ids together.

    All four on a single line, so the count can never be forwarded without the window
    it was measured over or the declaration it is being compared to — which is the
    exact way #706's number got orphaned from its definition the first time.
    """
    ids = ", ".join(m.pinned) if m.pinned else "(none)"
    return (f"zero-pinned #706: declared baseline {declared} ({declared_on}, all "
            f"per-trial rows, max composite_score) | measured {m.pinned_count} of "
            f"{m.tasks_scored} tasks pinned at exactly 0.00 over the trailing "
            f"{m.window_days} days | ids: {ids}")


def drift_line(m: Measurement, *, declared: int = DECLARED_BASELINE_ZERO_PINNED,
               declared_on: str = DECLARED_ON) -> str | None:
    """A sentence about a moved count — never a verdict, and never a nonzero exit.

    ``None`` when the measurement matches the declaration. The wording is deliberately
    not an alarm: the window slides daily and the split rotates tasks, so a moved count
    is evidence for the next ruling, not a regression to bisect.
    """
    if m.pinned_count == declared:
        return None
    return (f"#706 baseline moved: declared {declared} ({declared_on}, trailing "
            f"{m.window_days} days), measured {m.pinned_count}. Expected drift is not a "
            "regression — a person or the owed-check ruling re-declares the constant "
            "with its date and its definition (all rows, or BASELINE_* rows only; that "
            "choice is the one that moves the number).")


def cannot_line(reason: str, *, window_days: int, ledger: Path,
                declared: int = DECLARED_BASELINE_ZERO_PINNED,
                declared_on: str = DECLARED_ON) -> str:
    """A refusal to measure, naming the baseline and the window it could not check.

    A refusal that prints only "no such file" is the report that reads as a clean
    result to whatever reads it next: it has to say WHICH invariant went unchecked.
    """
    return (f"zero-pinned #706: cannot measure — {reason} (declared baseline {declared} "
            f"of {declared_on}, trailing {window_days} days). The count is unknown, not "
            f"zero. Pass --ledger PATH; default resolved to {ledger}.")


def parse_now(raw: str) -> datetime:
    """``--now`` as an aware UTC datetime, so a report can be re-run for a window."""
    parsed = parse_stamp(raw)
    if parsed is None:
        raise ValueError(f"unparseable --now stamp {raw!r}; use ISO-8601 UTC, "
                         "e.g. 2026-10-19T12:00:00Z")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """The CLI's arguments, built apart from ``main`` so a test can read the defaults.

    ``--window-days`` defaults to :data:`DEFAULT_WINDOW_DAYS` and ``--ledger`` to
    ``None``, which :func:`main` resolves through :func:`default_ledger` — neither
    default is a second copy of the number or the path it stands for.
    """
    parser = argparse.ArgumentParser(
        prog="zero_pinned_baseline.py",
        description="Recount the bench tasks pinned at exactly 0.00 and print them "
                    "beside #706's declared baseline. A report: it writes nothing.")
    parser.add_argument("--ledger", type=Path, default=None,
                        help="per-trial research ledger (default: the one resolved "
                             "under the runtime data root)")
    parser.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS,
                        help="trailing window in days (default: DEFAULT_WINDOW_DAYS)")
    parser.add_argument("--now", default=None, metavar="ISO-8601",
                        help="window anchor (default: this instant)")
    parser.add_argument("--json", action="store_true",
                        help="print the measurement as one JSON object")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    ledger = args.ledger if args.ledger is not None else default_ledger()
    try:
        moment = parse_now(args.now) if args.now else datetime.now(timezone.utc)
    except ValueError as exc:
        print(cannot_line(str(exc), window_days=args.window_days, ledger=ledger),
              file=sys.stderr)
        return 2
    if not ledger.is_file():
        print(cannot_line("no ledger file at the path given",
                          window_days=args.window_days, ledger=ledger), file=sys.stderr)
        return 1

    m = measure_from_ledger(ledger, window_days=args.window_days, now=moment)
    doc = m.as_dict()
    if args.json:
        print(json.dumps(doc, indent=2, sort_keys=True))
        return 0
    print(f"ledger: {m.ledger} | rows read {m.rows_read} | per-trial scored {m.trials_used}"
          f" | dropped: {m.rows_event} event, {m.rows_null_score} with no score, "
          f"{m.rows_bad_score} non-numeric score, {m.rows_undated} undated, "
          f"{m.rows_no_task} taskless, {m.rows_outside_window} outside the window, "
          f"{m.rows_unparsed} unparsed")
    print(f"window: {doc['window_started_at']} → {doc['window_ends_at']}")
    print(report_line(m))
    drift = drift_line(m)
    if drift:
        print(drift)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
