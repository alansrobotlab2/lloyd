"""One weighted composite index over the eval surfaces that already emit dated artifacts (#2597).

Lloyd has more than ten measurement instruments and no answer to "is Lloyd better
than last month". Each one publishes into its own file — the nightly retrieval
baseline, `scripts/automod/scorecard.py`, `scripts/automod/landed_outcomes.py`,
`scripts/funnel_census.py`, `app/tool_quality.py`, `app/uptake.py`,
`scripts/iv_metrics_record.py` — and each is read, if at all, on its own day.
This script aggregates the arms that already exist instead of adding another one:
it reads the newest dated artifact of each, tiers each by the maturity rubric
below, and publishes ONE dated row with a declared-weight total and an
equal-weight total beside it.

Three rules make the number worth reading, and each is the reason the file exists:

1. **Absence scores `missing` and drops out of the denominator.** A dead
   instrument must not read as green, and it must not read as zero either. This
   is the rule `scripts/eval_trend_stats.py` already applies to a drift term
   ("a drift term that reads 0 when the truth is unmeasured is the failure mode
   this whole file exists to remove") and `scripts/funnel_census.py` applies to an
   unmeasurable stage (a named refusal with its denominator, not a rate). Here it
   is applied to the instruments themselves. A component whose artifact is absent
   contributes `missing`; a component whose newest artifact is older than its own
   declared cadence contributes `stale`. Neither enters the numerator *or* the
   denominator, and the renormalisation is printed. Fewer than
   `MIN_SCORED_COMPONENTS` measured components and the total REFUSES rather than
   publishing a percentage that a single surviving instrument could set to 100.

2. **Tier is a mechanical test, not an opinion.** 0 = no artifact at the declared
   path. 1 = an artifact exists but is not arriving on its cadence, or nothing
   reads it against a prior row. 2 = rows arrive within the declared cadence
   window AND the row names a consumer that compares an artifact against a prior
   row. The tier is the load-bearing column: Lloyd's own record says a measurement
   that is merely *observable* is not a measurement (#460: "`iv_grade.py` can
   compute it, but its output only exists in whoever's terminal ran it"), and
   #2507 and #2275 each landed an instrument that no scheduler ever ran. A tier
   column over the instrument list turns those three scattered rulings into one
   standing read. **A tier-2 claim with no named consumer is a tier-1 row**, which
   is why `Consumer` is data on the component and not a comment.

3. **The total is printed twice** — under the declared weights and under equal
   weights — because hand-set weights are this design's weakest move (the weights
   below have no derivation, and the item says so). If the two disagree about the
   direction of the week-over-week move, the headline is "the weights decide the
   answer", and that disagreement is the deliverable, not an embarrassment.

Units are declared per component and only `rate` components (a 0..1 share) are
allowed into the average; `funnel_census` is carried in the table with its unit
and its stage-declared denominator but excluded from the total, because its own
round finding is that its stages divide by artifacts and rows interchangeably. A
value that is not inside 0..1 is reported `not-a-rate` rather than averaged, which
is what keeps a count or a latency from becoming a percentage.

**v1 is report-only.** No gate, promotion rule or rollback path reads this row.
`tests/test_eval_instrument_index.py::test_no_gate_promotion_or_rollback_path_reads_the_index`
proves it over every module in the tree by AST scan (a reference through a
variable or an `-m` argument counts, prose does not), and
`::test_the_publisher_is_not_imported_by_the_boot_path` proves it by import, with a
control that runs the same probe once WITH the publisher imported so the negative
verdict is a measurement rather than a probe that cannot report True. Wiring the
named consumer the item defers (backlog triage priority, #693's ranking by
measured value) is deliberately not done here, so the "another report nobody reads"
risk is deferred, not retired.

Usage:

    cd ~/lloyd && .venvs/lloyd/bin/python -m scripts.eval_instrument_index [--print]

    # one run over a snapshot of the stores instead of the live data root:
    ... -m scripts.eval_instrument_index --roots snapshot.json --out-dir /tmp/idx
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = 1
ITEM = "#2597"
SERIES_NAME = "instrument-index.jsonl"

#: A composite over one instrument is not a composite: it is that instrument's own
#: number wearing a total's clothes, which is the flat-headline error this file is
#: about. Below this many measured components the totals refuse and say what is
#: missing instead of publishing the survivor's value as a percentage.
MIN_SCORED_COMPONENTS = 2

#: Rows of this series needed before a component can be called flat: three prior
#: rows plus the one being published. Below it the answer is `insufficient-series`,
#: never `flat` — `scorecard.jsonl` holds 2 rows and `uptake` 3 today, and reading
#: either of those as a flat line would be exactly the crying-wolf failure the
#: item's risks name.
FLAT_WINDOW_ROWS = 4

#: Two floats this close are the same number. Used only for the flat-headline
#: test, where the question is "has this value moved at all across the window".
FLAT_EPSILON = 1e-9

#: The only unit the composite is allowed to average: a 0..1 share, oriented so
#: that higher is better after `quality()` flips the down-is-better arms.
AVERAGEABLE_UNIT = "rate"

_RATE_FRACTION = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s*$")
_ISO_IN_NAME = re.compile(r"(\d{4}-\d{2}-\d{2})")
_COMPACT_IN_NAME = re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)")


# ── the components ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Consumer:
    """The thing that reads this instrument's artifact against a PRIOR row.

    Tier 2 is "tracked and analysed over time", and the only evidence of that is
    a reader that compares rows. A consumer that merely copies or renders the
    newest row (`app/routers/dashboard.py` recomputing the scorecard,
    `scripts/iv_metrics_mirror.py` mirroring the IV series) leaves the instrument
    at tier 1, which is why `compares_rows` is a field rather than implied by the
    consumer's existence.
    """

    script: str
    how: str
    compares_rows: bool


@dataclass(frozen=True)
class Instrument:
    """One measurement surface, declared exactly as its writer declares it.

    `artifact` resolves at CALL time through the writer's own module constant,
    never through a path re-spelled here: an index that re-typed
    `eval/tool-quality` would keep reporting a location after the writer moved,
    which is the failure `app/tool_quality.py`'s "resolved at call time" comment
    and `app/paths.py`'s worktree warning both exist to stop. It returns either a
    directory (then `pattern` selects the dated files in it) or one series file.
    """

    key: str
    title: str
    artifact: Callable[[], Path]
    pattern: str | None
    value_path: str
    unit: str
    #: Which direction is better: "up" (a hit rate) or "down" (a dropped rate).
    polarity: str
    weight: float
    weight_why: str
    #: Days between runs as the WRITER declares them, or None for an instrument
    #: with no declared cadence at all. `cadence_days is None` and
    #: `hand_run_by_ruling` travel together: such an instrument cannot reach
    #: tier 2 (nothing is arriving on a schedule to be tracked over time), and it
    #: is never flagged stale, because a cadence check that flags a deliberate
    #: design is worse than no check.
    cadence_days: int | None
    hand_run_by_ruling: bool
    #: The autonomy task file that runs the writer, or None = hand-run-or-none.
    scheduled_by: str | None
    consumer: Consumer | None
    #: For a single-file JSONL series, the key holding the row's own timestamp;
    #: None means take the date from the filename.
    date_key: str | None = None
    ci95_path: str | None = None
    n_path: str | None = None
    #: The artifact's own statement of what its denominator is, when it carries
    #: one. Read into the row so a share is never quoted over an unknown N.
    unit_path: str | None = None
    #: False = reported, never averaged into a total.
    averageable: bool = True


def _baselines_dir() -> Path:
    from app import paths as p
    return p.EVAL_BASELINES_DIR


def _tool_quality_dir() -> Path:
    from app import paths as p
    return p.TOOL_QUALITY_DIR


def _funnel_dir() -> Path:
    from app import paths as p
    from scripts import funnel_census as fc
    return p.DATA_ROOT.joinpath(*fc.FUNNEL_ARTIFACT_SUBDIR)


def _scorecard_series() -> Path:
    from scripts.automod import scorecard as sc
    return sc.scorecard_path()


def _landed_outcomes_series() -> Path:
    from scripts.automod import landed_outcomes as lo
    return lo.landed_outcomes_path()


def _iv_series() -> Path:
    from scripts import iv_metrics_record as iv
    return iv.DEFAULT_SERIES_PATH


def _uptake_dir() -> Path:
    from app import uptake
    return uptake.UPTAKE_ARTIFACT_DIR


#: The table. Seven instruments, weights summing to 1.0, every path resolved
#: through the writer's own constant — which is how the two stores the item calls
#: out end up IN the table instead of being skipped for living where the data root
#: does not reach: the automod state dir holds `scorecard.jsonl` and
#: `landed_outcomes.jsonl` outside `DATA_ROOT`, and uptake's default output is
#: inside the code tree.
INSTRUMENTS: tuple[Instrument, ...] = (
    Instrument(
        key="retrieval_nightly",
        title="Nightly retrieval non-regression baseline",
        artifact=_baselines_dir,
        pattern="nightly-*.json",
        value_path="summary.overall.doc_hit_rate",
        unit="rate",
        polarity="up",
        weight=0.28,
        weight_why="weighted by reach: every prompt this box builds is built out "
                   "of the retrieval this arm scores, and it is the only component "
                   "carrying its own Wilson interval (#696)",
        cadence_days=1,
        hand_run_by_ruling=False,
        scheduled_by="autonomy/82-nightly-retrieval-eval.md",
        consumer=Consumer("scripts/eval_trend_stats.py",
                          "walks the nightly baselines directory and compares each "
                          "night against the prior one (McNemar transitions)",
                          compares_rows=True),
        ci95_path="summary.overall.ci95.doc_hit_rate.ci",
        n_path="summary.overall.ci95.doc_hit_rate.n",
    ),
    Instrument(
        key="iv_metrics",
        title="Inner Voice metrics series (dropped verdicts per LLM call)",
        artifact=_iv_series,
        pattern=None,
        value_path="dropped_rate",
        unit="rate",
        polarity="down",
        weight=0.20,
        weight_why="weighted by coverage: it observes every turn of every session "
                   "on the box, so it is the widest-behaviour arm in the table",
        cadence_days=1,
        hand_run_by_ruling=False,
        scheduled_by="autonomy/86-nightly-iv-metrics-series.md",
        consumer=Consumer("scripts/iv_metrics_record.py",
                          "`_prior_breaching` compares the last row's rate against "
                          "the median of the prior `--window-rows` rows",
                          compares_rows=True),
        n_path="llm_calls",
        date_key="recorded_at",
    ),
    Instrument(
        key="automod_scorecard",
        title="Unattended-loop scorecard (acceptance hit rate)",
        artifact=_scorecard_series,
        pattern=None,
        value_path="acceptance.hit_rate",
        unit="rate",
        polarity="up",
        weight=0.18,
        weight_why="weighted by decisions informed: it is the only read of whether "
                   "the loop is earning its keep, and the loop's own dashboard tab "
                   "is rendered from it",
        cadence_days=7,
        hand_run_by_ruling=False,
        scheduled_by=None,
        consumer=Consumer("app/routers/dashboard.py",
                          "calls `scorecard.compute()` fresh for the loop tab, so it "
                          "renders the newest numbers but never compares one stored "
                          "row against another",
                          compares_rows=False),
        n_path="acceptance.with_outcome",
        date_key="computed_at",
    ),
    Instrument(
        key="landed_outcomes",
        title="Landed-change undo rate by author (agent arm)",
        artifact=_landed_outcomes_series,
        pattern=None,
        value_path="by_author.agent.later_undo_7d",
        unit="rate",
        polarity="down",
        weight=0.14,
        weight_why="weighted by what it alone measures: the cost side of a landing "
                   "— how often a landed commit is undone within 7 days — has no "
                   "other arm",
        cadence_days=1,
        hand_run_by_ruling=False,
        scheduled_by="autonomy/97-landed-outcomes-daily-record.md",
        consumer=Consumer("scripts/automod/landed_outcomes.py",
                          "`compute` takes a window and prints one row; nothing "
                          "compares one stored row against another",
                          compares_rows=False),
        date_key="generated_at",
    ),
    Instrument(
        key="tool_quality",
        title="Tool-call quality (non-error calls that returned nothing)",
        artifact=_tool_quality_dir,
        pattern="[0-9][0-9][0-9][0-9]-*.json",
        value_path="corpus_rates.empty",
        unit="rate",
        polarity="down",
        weight=0.10,
        weight_why="weighted by call volume against recoverability: it sees every "
                   "tool call, but an empty return is retried inside the same turn "
                   "rather than ending it",
        cadence_days=7,
        hand_run_by_ruling=False,
        scheduled_by=None,
        consumer=Consumer("app/tool_quality.py",
                          "prunes its own store past RETENTION_DAYS and prints the "
                          "run; no reader compares one dated file against another",
                          compares_rows=False),
    ),
    Instrument(
        key="uptake_probe",
        title="Uptake probe classifier precision",
        artifact=_uptake_dir,
        pattern="uptake-*.json",
        value_path="classifier.precision",
        unit="rate",
        polarity="up",
        weight=0.05,
        weight_why="weighted by thinness: hand-labelled over a 23-positive label "
                   "file, so a small weight rather than a zero, and no weight at "
                   "all would hide that the store is in the code tree",
        cadence_days=None,
        hand_run_by_ruling=True,
        scheduled_by=None,
        consumer=Consumer("scripts/uptake_probe.py",
                          "scores the classifier against the pinned label file and "
                          "exits 3 below the 0.70 precision floor; it does not "
                          "compare one dated table against another",
                          compares_rows=False),
        n_path="classifier.n_positives",
    ),
    Instrument(
        key="funnel_census",
        title="Durable-artifact funnel census (surfaced stage)",
        artifact=_funnel_dir,
        pattern="funnel-*.json",
        value_path="stages[surfaced].rate",
        unit="share",
        polarity="up",
        weight=0.05,
        weight_why="weighted by breadth against unit honesty: it censuses every "
                   "durable artifact, but its own round finding is that its stages "
                   "divide by artifacts and rows interchangeably, so it is reported "
                   "and not averaged",
        cadence_days=7,
        hand_run_by_ruling=False,
        scheduled_by=None,
        averageable=False,
        consumer=Consumer("scripts/funnel_census.py",
                          "writes the census; #2507's round finding is that nothing "
                          "schedules it and nothing reads one census against the "
                          "next",
                          compares_rows=False),
        ci95_path="stages[surfaced].ci95",
        n_path="stages[surfaced].n",
        unit_path="stages[surfaced].unit",
    ),
)


# ── reading an artifact ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class Reading:
    """What is actually on disk for one instrument, as at `today`.

    `status` is one of: `measured` (a value, arriving on cadence), `missing` (no
    artifact at the declared path), `stale` (an artifact older than the declared
    cadence), `no-value` (an artifact whose declared pointer yields nothing — a
    named refusal, in the census's own idiom, never a zero) and `not-a-rate` (a
    value outside 0..1, so not a share and not averageable).
    """

    key: str
    status: str
    artifact_path: str
    newest_artifact: str | None
    age_days: int | None
    value: float | None
    denominator: int | None
    ci95: list[float] | None
    declared_unit: str | None
    detail: str

    @property
    def contributes(self) -> bool:
        """The single test for "this instrument is in the total". Absence and
        staleness are excluded structurally: no other status passes."""
        return self.status == "measured"


def _dig(doc: Any, path: str) -> Any:
    """Follow `a.b[c].d` through nested dicts and lists.

    The bracket segment addresses a list entry by its `stage` name, which is how
    the funnel census's `stages` list is keyed — the one shape in this table that
    is not a plain dotted pointer. Any miss returns None, which the caller turns
    into `no-value` rather than a zero.
    """
    cur = doc
    for seg in path.split("."):
        if cur is None:
            return None
        name, _, bracket = seg.partition("[")
        if name:
            if not isinstance(cur, dict) or name not in cur:
                return None
            cur = cur[name]
        if bracket:
            want = bracket.rstrip("]")
            if not isinstance(cur, list):
                return None
            hit = next((e for e in cur
                        if isinstance(e, dict) and str(e.get("stage")) == want), None)
            if hit is None:
                return None
            cur = hit
    return cur


def _coerce_value(raw: Any) -> tuple[float | None, int | None, list[float] | None]:
    """A number, the `"k / n"` string, or the rate-DICT — with the artifact's own
    interval when it carries one.

    Two writers publish a share as a record rather than a scalar, and the index has
    to read the shape the writer actually writes: `app/tool_quality.py::rate`
    emits `"12 / 300"` because it refuses to print a bare count, and
    `scripts/automod/landed_outcomes.py::rate` emits
    `{"k": 3, "n": 40, "rate": 0.075, "ci_low": …, "ci_high": …}` — Wilson bounds
    over the same denominator it prints, never folded into a string. A reader that
    understood only the scalar would publish an instrument that IS running as
    `no-value` forever, which is the exact failure this file exists to remove.
    Returns `(value, denominator, interval)`; the interval, when present, is the
    artifact's own and is never re-derived here.
    """
    if raw is None or isinstance(raw, bool):
        return None, None, None
    if isinstance(raw, (int, float)):
        return float(raw), None, None
    if isinstance(raw, Mapping):
        n = raw.get("n")
        denom = n if isinstance(n, int) and not isinstance(n, bool) else None
        rate = raw.get("rate")
        k = raw.get("k")
        value = (float(rate) if isinstance(rate, (int, float))
                 else (float(k) / denom
                       if isinstance(k, int) and not isinstance(k, bool) and denom
                       else None))
        lo, hi = raw.get("ci_low"), raw.get("ci_high")
        ci = ([float(lo), float(hi)]
              if all(isinstance(v, (int, float)) and not isinstance(v, bool)
                     for v in (lo, hi)) else None)
        return value, denom, ci
    m = _RATE_FRACTION.match(str(raw))
    if m:
        k, n = int(m.group(1)), int(m.group(2))
        return (float(k) / n if n else None), (n if n else None), None
    return None, None, None


def _date_from_name(name: str) -> date | None:
    m = _ISO_IN_NAME.search(name)
    if m:
        try:
            return date.fromisoformat(m.group(1))
        except ValueError:
            pass
    m = _COMPACT_IN_NAME.search(name)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    return None


def _newest_dated_file(store: Path, pattern: str) -> tuple[Path | None, date | None]:
    """The newest artifact by the DATE IN ITS NAME, falling back to mtime.

    A filename date is what "observed cadence from the files actually on disk"
    means; an mtime is only a fallback, and a store whose names carry no date at
    all reports its newest file rather than nothing. Two artifacts may share a
    day — the retrieval eval has run twice on a morning — so a same-day tie goes
    to the later write, which is the row a reader would open.
    """
    best: tuple[Path | None, date | None] = (None, None)
    best_key: tuple[int, float] = (-1, 0.0)
    for path in sorted(store.glob(pattern)):
        if not path.is_file():
            continue
        named = _date_from_name(path.name)
        mtime = path.stat().st_mtime
        day = named or datetime.fromtimestamp(mtime, timezone.utc).date()
        key = (day.toordinal(), mtime)
        if best[1] is None or key > best_key:
            best, best_key = (path, day), key
    return best


def _tail_lines(path: Path, max_bytes: int = 65536) -> list[str]:
    """The last lines of a growing JSONL series, read from the end.

    Precedent: `scripts/iv_metrics_record.py::_prior_series` refuses to read a
    series whole for the same reason — a store that has accumulated thousands of
    rows is not read into memory to look at its newest one.
    """
    with path.open("rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        fh.seek(max(0, size - max_bytes))
        chunk = fh.read()
    return chunk.decode("utf-8", "replace").splitlines()


def _newest_jsonl_row(path: Path) -> dict | None:
    for line in reversed(_tail_lines(path)):
        line = line.strip()
        if not line:
            continue
        try:
            doc = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(doc, dict):
            return doc
    return None


def _as_day(raw: Any) -> date | None:
    if raw is None:
        return None
    text = str(raw).strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        pass
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def read_instrument(instr: Instrument, *, today: date,
                    root_override: Path | None = None) -> Reading:
    """One component's row of evidence, read from disk as at `today`."""
    declared = root_override if root_override is not None else instr.artifact()
    missing = Reading(key=instr.key, status="missing", artifact_path=str(declared),
                      newest_artifact=None, age_days=None, value=None,
                      denominator=None, ci95=None, declared_unit=None,
                      detail=f"no artifact at the declared path {declared}")
    if declared.is_dir():
        path, day = _newest_dated_file(declared, instr.pattern or "*")
        if path is None:
            return missing
    elif declared.is_file():
        path, day = declared, None
    else:
        return missing

    doc: Any
    if path.suffix == ".jsonl" or instr.date_key:
        doc = _newest_jsonl_row(path)
        if doc is None:
            return Reading(key=instr.key, status="no-value", artifact_path=str(path),
                           newest_artifact=day.isoformat() if day else None,
                           age_days=None, value=None, denominator=None, ci95=None,
                           declared_unit=None,
                           detail=f"{path.name} holds no readable JSON row")
        # A growing series has no date in its own filename (`scorecard.jsonl` is
        # one file forever), so its age is the newest ROW's timestamp. The row
        # wins over an mtime wherever it carries one: a copied or restored store's
        # mtime is a fact about the copy, not about when the instrument ran.
        if instr.date_key:
            stamped = _as_day(_dig(doc, instr.date_key))
            if stamped is not None:
                day = stamped
    else:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return Reading(key=instr.key, status="no-value", artifact_path=str(path),
                           newest_artifact=day.isoformat() if day else None,
                           age_days=None, value=None, denominator=None, ci95=None,
                           declared_unit=None,
                           detail=f"{path.name} unreadable: {type(exc).__name__}")

    age = (today - day).days if day else None
    raw = _dig(doc, instr.value_path)
    value, denominator, value_ci = _coerce_value(raw)
    if denominator is None and instr.n_path:
        n = _dig(doc, instr.n_path)
        denominator = int(n) if isinstance(n, int) and not isinstance(n, bool) else None
    # The interval the artifact carries beside its own rate wins over a pointed-to
    # one: a writer publishing `ci_low`/`ci_high` next to `rate` has already bound
    # the bounds to the denominator it printed (#696).
    ci_raw = _dig(doc, instr.ci95_path) if instr.ci95_path else None
    pointed = ([float(x) for x in ci_raw]
               if isinstance(ci_raw, (list, tuple)) and len(ci_raw) == 2 else None)
    ci95 = value_ci or pointed
    declared_unit = _dig(doc, instr.unit_path) if instr.unit_path else None

    base = dict(artifact_path=str(path),
                newest_artifact=day.isoformat() if day else None, age_days=age,
                value=value, denominator=denominator, ci95=ci95,
                declared_unit=declared_unit if isinstance(declared_unit, str) else None)

    if value is None:
        return Reading(key=instr.key, status="no-value",
                       detail=f"{path.name} carries no value at `{instr.value_path}` "
                              f"(a named refusal, not a zero)", **base)
    if not 0.0 <= value <= 1.0:
        return Reading(key=instr.key, status="not-a-rate",
                       detail=f"{value} is outside 0..1, so `{instr.value_path}` is "
                              f"not a share and may not enter the total", **base)
    if (instr.cadence_days is not None and not instr.hand_run_by_ruling
            and age is not None and age > instr.cadence_days):
        return Reading(key=instr.key, status="stale",
                       detail=f"newest artifact is {age} d old against a declared "
                              f"cadence of {instr.cadence_days} d "
                              f"(scheduled_by={instr.scheduled_by or 'no task file'})",
                       **base)
    return Reading(key=instr.key, status="measured",
                   detail=f"newest artifact {path.name}"
                          + (f", {age} d old" if age is not None else ""), **base)


# ── tier, quality, movement ───────────────────────────────────────────────────

def tier_of(instr: Instrument, reading: Reading) -> int:
    """0 does not exist, 1 is observable, 2 is tracked and analysed over time.

    Tier 2 requires BOTH edges of "tracked over time": rows arriving inside the
    declared cadence window, and a named consumer that compares an artifact
    against a prior row. Drop the consumer and the same fresh artifact is a
    tier-1 row, which is the clause under test here as much as in #460's history.
    """
    if reading.status == "missing":
        return 0
    if instr.cadence_days is None:
        # Hand-run by ruling: nothing is arriving on a cadence to be tracked.
        return 1
    fresh = reading.age_days is not None and reading.age_days <= instr.cadence_days
    if not fresh:
        return 1
    if instr.consumer is None or not instr.consumer.compares_rows:
        return 1
    return 2


def quality(reading: Reading, instr: Instrument) -> float | None:
    """The reading on the one axis the composite is allowed to average: 0..1,
    higher is better. A `down`-is-better arm is flipped, so a dropped rate of
    0.045 is a quality of 0.955 and both are on the same scale as a hit rate."""
    if not reading.contributes or reading.value is None:
        return None
    q = reading.value if instr.polarity == "up" else 1.0 - reading.value
    return round(min(1.0, max(0.0, q)), 6)


def movement(reading: Reading, instr: Instrument, prior_quality: float | None) -> str:
    """Did this component move, given the interval its own artifact carries?

    A component whose `ci95` covers the prior value may not be scored as moving:
    #696 bounds the sampling error per metric precisely so a re-run of the same
    corpus cannot be read as an improvement. Cross-night comparisons are
    independent-form, never paired — the corpus drifts between nights (#608).
    """
    if prior_quality is None:
        return "no-prior-row"
    q = quality(reading, instr)
    if q is None:
        return "not-scored"
    if reading.ci95:
        lo, hi = sorted(reading.ci95)
        if instr.polarity != "up":
            lo, hi = 1.0 - hi, 1.0 - lo
        if lo <= prior_quality <= hi:
            return "flat-within-interval"
    if abs(q - prior_quality) < FLAT_EPSILON:
        return "unchanged"
    return "up" if q > prior_quality else "down"


def flat_check(key: str, history: Sequence[float], q_now: float | None,
               composite_moved: bool) -> str:
    """The flat-headline detector: a number everyone quotes that no longer
    discriminates. Needs `FLAT_WINDOW_ROWS` rows to say anything at all, and
    says `insufficient-series` rather than `flat` without them."""
    values = list(history) + ([q_now] if q_now is not None else [])
    if len(values) < FLAT_WINDOW_ROWS:
        return f"insufficient-series({len(values)} of {FLAT_WINDOW_ROWS} rows)"
    window = values[-FLAT_WINDOW_ROWS:]
    if all(abs(v - window[0]) < FLAT_EPSILON for v in window) and composite_moved:
        return (f"flat over {FLAT_WINDOW_ROWS} rows while the composite moved: "
                f"candidate vanity metric")
    return "moving"


# ── the row ───────────────────────────────────────────────────────────────────

def totals_of(readings: Sequence[tuple[Instrument, Reading, float | None]]) -> dict:
    """Declared-weight and equal-weight totals over the SAME scored set.

    Fewer than `MIN_SCORED_COMPONENTS` measured components and the whole block
    refuses. Excluded components are named with their status, and the weight sum
    they would have carried is printed, so the renormalisation is visible rather
    than implied by a number that silently changed its denominator.
    """
    scored = [(i, q) for (i, r, q) in readings if r.contributes and q is not None
              and i.averageable]
    excluded = [{"key": i.key, "status": r.status, "weight": i.weight,
                 "unit": i.unit,
                 "reason": (r.detail if not r.contributes
                            else (None if i.averageable else
                                  f"unit `{i.unit}` is not `{AVERAGEABLE_UNIT}`: "
                                  f"reported, not averaged"))}
                for (i, r, q) in readings
                if not (r.contributes and q is not None and i.averageable)]
    block = {"scored": len(scored), "of": len(readings), "excluded": excluded,
             "min_scored": MIN_SCORED_COMPONENTS}
    if len(scored) < MIN_SCORED_COMPONENTS:
        return dict(block, status="refused", declared_weights=None, equal_weights=None,
                    renormalised_weight_sum=None,
                    reason=f"only {len(scored)} of {len(readings)} instruments "
                           f"reported a measured value; a composite over fewer than "
                           f"{MIN_SCORED_COMPONENTS} is that instrument's own number "
                           f"wearing a total's clothes, so the index refuses")
    weight_sum = sum(i.weight for (i, _) in scored)
    declared = sum(q * i.weight for (i, q) in scored) / weight_sum
    equal = sum(q for (_, q) in scored) / len(scored)
    return dict(block, status="ok", declared_weights=round(declared, 4),
                equal_weights=round(equal, 4),
                renormalised_weight_sum=round(weight_sum, 4))


def sign_of(delta: float | None) -> int | None:
    if delta is None:
        return None
    if delta > FLAT_EPSILON:
        return 1
    if delta < -FLAT_EPSILON:
        return -1
    return 0


def weight_agreement(prior_rows: Sequence[dict]) -> dict:
    """Has the declared weighting ever changed the DIRECTION of the move?

    The weights here have no derivation, and the equal-weight line beside them is
    the only available defence. If across every comparison they have never
    disagreed, the declared weighting is decorative and this row says so in its
    own header rather than waiting for someone to notice.
    """
    comparisons = 0
    disagreements = 0
    for row in prior_rows:
        t = row.get("totals") or {}
        if t.get("status") != "ok":
            continue
        deltas = row.get("delta_vs_prior") or {}
        if deltas.get("declared_weights") is None or deltas.get("equal_weights") is None:
            continue
        comparisons += 1
        if sign_of(deltas["declared_weights"]) != sign_of(deltas["equal_weights"]):
            disagreements += 1
    return {"prior_rows_compared": comparisons,
            "sign_disagreements": disagreements,
            "decorative": (None if comparisons == 0
                           else disagreements == 0)}


def delta_vs_prior(totals: dict, prior: dict | None) -> dict:
    prior_totals = (prior or {}).get("totals") or {}
    out = {"prior_generated_at": (prior or {}).get("generated_at"),
           "declared_weights": None, "equal_weights": None}
    if (totals.get("status") != "ok" or prior_totals.get("status") != "ok"):
        return out
    out["declared_weights"] = round(totals["declared_weights"]
                                    - prior_totals["declared_weights"], 4)
    out["equal_weights"] = round(totals["equal_weights"]
                                 - prior_totals["equal_weights"], 4)
    return out


def headline_of(totals: dict, deltas: dict, agreement: dict,
                dead_or_stale: Sequence[dict]) -> str:
    """The one line a person reads. The sign-disagreement case is the point of
    the whole artifact, so it outranks the number itself."""
    if totals.get("status") != "ok":
        return f"the index refuses: {totals['reason']}"
    s, e = sign_of(deltas.get("declared_weights")), sign_of(deltas.get("equal_weights"))
    if s is not None and e is not None and s != e:
        return (f"the weights decide the answer: the declared-weight total moved "
                f"{'up' if s > 0 else 'down'} "
                f"({deltas['declared_weights']:+}) while the equal-weight total "
                f"moved {'up' if e > 0 else 'down'} ({deltas['equal_weights']:+})")
    named = ", ".join(d["key"] for d in dead_or_stale) or "none"
    extra = ""
    if agreement.get("decorative") is True:
        extra = ("; the declared weights have never changed the direction of the "
                 "move, so on this series they are decorative")
    return (f"{totals['scored']} of {totals['of']} instruments measured; "
            f"declared-weight {totals['declared_weights']}, equal-weight "
            f"{totals['equal_weights']}; missing or stale: {named}{extra}")


def build_row(instruments: Sequence[Instrument] = INSTRUMENTS, *,
              now: datetime | None = None,
              root_overrides: Mapping[str, Path] | None = None,
              out_dir: Path | None = None,
              code_root: Path | None = None) -> dict:
    """Assemble one dated row. Reads the artifacts and nothing else; writes only
    when `out_dir` is given, and then only to read back its own prior rows."""
    from app import paths as p

    stamp = now or datetime.now(timezone.utc)
    today = stamp.date()
    code = Path(code_root) if code_root else p.LLOYD_HOME
    overrides = dict(root_overrides or {})

    readings: list[tuple[Instrument, Reading, float | None]] = []
    for instr in instruments:
        override = overrides.get(instr.key)
        reading = read_instrument(instr, today=today, root_override=override)
        readings.append((instr, reading, quality(reading, instr)))

    prior_rows = read_series(out_dir) if out_dir else []
    prior = prior_rows[-1] if prior_rows else None
    totals = totals_of(readings)
    deltas = delta_vs_prior(totals, prior)

    scored_keys = {i.key for (i, r, q) in readings if r.contributes and q is not None
                   and i.averageable}
    composite_moved = sign_of(deltas.get("declared_weights")) not in (None, 0)

    components: list[dict] = []
    for instr, reading, q in readings:
        history = [v for row in prior_rows
                   for c in (row.get("components") or [])
                   if c.get("key") == instr.key and isinstance(c.get("quality"), (int, float))
                   for v in [float(c["quality"])]]
        resolved = Path(reading.artifact_path)
        components.append({
            "key": instr.key,
            "title": instr.title,
            "instrument_path": str(resolved.parent if resolved.is_file()
                                   or resolved.suffix else resolved),
            "artifact_path": reading.artifact_path,
            "inside_code_tree": code in resolved.parents,
            "outside_data_root": p.DATA_ROOT not in resolved.parents,
            "newest_artifact": reading.newest_artifact,
            "age_days": reading.age_days,
            "status": reading.status,
            "tier": tier_of(instr, reading),
            "weight": instr.weight,
            "weight_why": instr.weight_why,
            "unit": reading.declared_unit or instr.unit,
            "declared_unit": instr.unit,
            "averageable": instr.averageable,
            "polarity": instr.polarity,
            "value": reading.value,
            "quality": q,
            "denominator": reading.denominator,
            "ci95": reading.ci95,
            "contributes": bool(reading.contributes and q is not None
                                and instr.averageable),
            "in_total_denominator": instr.key in scored_keys,
            "movement": movement(reading, instr, _prior_quality(prior_rows, instr.key)),
            "flat_check": flat_check(instr.key, history, q, composite_moved),
            "cadence_days": instr.cadence_days,
            "hand_run_by_ruling": instr.hand_run_by_ruling,
            "scheduled_by": instr.scheduled_by,
            "consumer": (None if instr.consumer is None else {
                "script": instr.consumer.script, "how": instr.consumer.how,
                "compares_rows": instr.consumer.compares_rows}),
            "detail": reading.detail,
        })

    dead_or_stale = [{"key": c["key"], "instrument_path": c["artifact_path"],
                      "status": c["status"], "why": c["detail"],
                      "scheduled_by": c["scheduled_by"]}
                     for c in components
                     if c["status"] in ("missing", "stale", "no-value")]

    agreement = weight_agreement(prior_rows)
    return {
        "schema": SCHEMA,
        "item": ITEM,
        "generated_at": stamp.isoformat(timespec="seconds"),
        "date": today.isoformat(),
        "tree": p.describe_tree(),
        "series": str(Path(out_dir) / SERIES_NAME) if out_dir else None,
        "history_rows": len(prior_rows),
        "allowed_average": (
            f"only components whose unit is `{AVERAGEABLE_UNIT}` (a 0..1 share, "
            f"flipped by polarity so higher is better). A count, a latency or a "
            f"share over an undeclared denominator is reported and excluded: "
            f"this table mixes writers, and #2507's own finding is that its "
            f"stages divide by artifacts and rows interchangeably."),
        "weights_sum_declared": round(sum(i.weight for i in instruments), 4),
        "components": components,
        "totals": totals,
        "delta_vs_prior": deltas,
        "weight_agreement": agreement,
        "dead_or_stale": dead_or_stale,
        "not_a_component_of_itself": (
            "this series is the index's own artifact and is deliberately absent "
            "from its own table: a component that reads its own prior row would "
            "make the flat-headline test circular"),
        "headline": headline_of(totals, deltas, agreement, dead_or_stale),
    }


def _prior_quality(prior_rows: Sequence[dict], key: str) -> float | None:
    for row in reversed(list(prior_rows)):
        for c in (row.get("components") or []):
            if c.get("key") == key and isinstance(c.get("quality"), (int, float)):
                return float(c["quality"])
    return None


# ── the series ────────────────────────────────────────────────────────────────

def series_path(out_dir: Path) -> Path:
    return Path(out_dir) / SERIES_NAME


def read_series(out_dir: Path) -> list[dict]:
    """Every prior row, oldest first. A malformed line is skipped, not fatal:
    one unparseable row must not cost the whole trend the way a bad ledger row
    costs `landed_outcomes` its gate."""
    path = series_path(Path(out_dir))
    if not path.is_file():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            doc = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(doc, dict):
            rows.append(doc)
    return rows


def append_row(row: dict, out_dir: Path) -> Path:
    """Append one dated row. Creates nothing outside its own directory, and
    writes nothing into the code tree — the publisher's only output is this
    file, which is what keeps `git status --porcelain` clean after a run."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = series_path(out)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def _num(value: float | None, spec: str = ".3f") -> str:
    return "" if value is None else f"{value:{spec}}"


def render(row: dict) -> str:
    lines = [f"# eval instrument index  {row['date']}  ({row['item']})",
             f"tree: {row['tree']}",
             f"HEADLINE: {row['headline']}",
             "",
             f"{'component':20s} {'status':10s} {'tier':4s} {'wt':5s} {'unit':6s} "
             f"{'value':>7s} {'qual':>6s} {'N':>5s}  newest      scheduled_by",
             "-" * 108]
    for c in row["components"]:
        ci = (f" [{c['ci95'][0]:.3f},{c['ci95'][1]:.3f}]" if c["ci95"] else "")
        lines.append(
            f"{c['key']:20s} {c['status']:10s} {c['tier']:<4d} {c['weight']:<5.2f} "
            f"{c['unit']:6s} {_num(c['value']):>7s} {_num(c['quality']):>6s} "
            f"{('' if c['denominator'] is None else str(c['denominator'])):>5s}  "
            f"{str(c['newest_artifact']):11s}  {c['scheduled_by'] or 'none'}{ci}")
    t = row["totals"]
    lines += ["",
              f"totals: {t['status']}  scored {t['scored']}/{t['of']}  "
              f"declared-weight {t['declared_weights']}  equal-weight {t['equal_weights']}"]
    for c in row["components"]:
        if not c["contributes"]:
            lines.append(f"  excluded from the total: {c['key']} — {c['status']}")
    lines.append(f"delta vs prior: {row['delta_vs_prior']}")
    lines.append(f"weights: {row['weight_agreement']}")
    return "\n".join(lines)


class RootsFileError(ValueError):
    """The `--roots` snapshot cannot be honoured as an override."""


def _load_roots(path: str | None) -> dict[str, Path] | None:
    """Read a `--roots` snapshot file into `build_row`'s `root_overrides`.

    Every key has to name a declared component. `build_row` resolves overrides by
    key and silently ignores one that matches nothing, so a typo in a snapshot
    would mean the run reads the live store for that arm while looking
    fixture-driven — a thing that reads as overridden while it is not, which is
    the exact failure this index refuses in its own components.
    """
    if path is None:
        return None
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RootsFileError(f"{path} must hold a JSON object of component key -> path, "
                             f"not {type(raw).__name__}")
    known = sorted(i.key for i in INSTRUMENTS)
    unknown = sorted(set(raw) - set(known))
    if unknown:
        raise RootsFileError(f"{path} names component(s) {unknown} that are not in the "
                             f"table; the declared keys are {known}")
    return {k: Path(v) for k, v in raw.items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m scripts.eval_instrument_index",
        description="Publish the weighted eval instrument index (#2597). "
                    "Report-only: nothing gates, promotes or rolls back on it.")
    ap.add_argument("--out-dir", default=None,
                    help="series directory (default: app.paths.EVAL_INSTRUMENT_INDEX_DIR)")
    ap.add_argument("--now", default=None,
                    help="ISO timestamp to stamp the row with (reproducible runs)")
    ap.add_argument("--roots", default=None,
                    help="JSON file mapping component key -> artifact path, so one run "
                         "reads a snapshot of the stores instead of the live data root "
                         "(the same purpose --now serves for the clock). A key that is "
                         "not a declared component is a refusal, exit 2.")
    ap.add_argument("--no-record", action="store_true",
                    help="print the row and append nothing")
    ap.add_argument("--print", dest="show", action="store_true",
                    help="print the rendered table")
    args = ap.parse_args(argv)

    from app import paths as p
    out = Path(args.out_dir) if args.out_dir else Path(p.EVAL_INSTRUMENT_INDEX_DIR)
    stamp = datetime.fromisoformat(args.now) if args.now else datetime.now(timezone.utc)
    try:
        overrides = _load_roots(args.roots)
    except (RootsFileError, json.JSONDecodeError) as exc:
        print(f"refusing to publish: {exc}", file=sys.stderr)
        return 2
    row = build_row(now=stamp, out_dir=out, root_overrides=overrides)
    if not args.no_record:
        append_row(row, out)
    if args.show or args.no_record:
        print(render(row))
    else:
        print(f"{row['date']}: {row['headline']}")
    # Report-only in v1 means nothing gates on this row — it does not mean the
    # exit code lies. A refused total exits 1 so a caller cannot read an index
    # that measured nothing as a run that succeeded.
    return 0 if row["totals"]["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
