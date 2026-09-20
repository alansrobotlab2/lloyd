#!/usr/bin/env python3
"""Append one row to the Inner Voice metrics series: `~/lloyd/_pipeline/reflection/iv-metrics.jsonl`.

Backlog #460. `scripts/iv_grade.py` can compute intervention rate, landed rate,
miss rate, tokens/turn, latency and the observer's dropped-verdict count, and has
always printed them to a terminal and then forgotten them. Over 2026-09-01..09-12
the measured baseline is 353 dropped verdicts out of 9,027 LLM calls (0.039), and
#458 was written off ONE hand-run of this script on 09-08; 37 more drops
accumulated in the following four days with nobody told, because the number only
existed in a terminal. That is the shape of the gap: the measurement works, the
*series* does not exist, so every question needing a before/after has no after.

This is the persistence half, and it is deliberately dumb:

  cd ~/lloyd && python3 scripts/iv_grade.py --json --since "$(date -d '26 hours ago' +%Y-%m-%dT%H:%M:%S)" \\
      | python3 scripts/iv_metrics_record.py --hours 26

It reads the grader's JSON on stdin, appends exactly one line, and prints a
one-line verdict. **It opens no database at all.** `iv_grade.py:1-6` states the
grader is read-only and must stay that way because the observer is writing that
table live, and `tests/integration/test_iv_guards.py:46` guards the table from
tests; so the series is a plain JSONL file and never a new column or a second
writer on `usage.db`.

`usage.db` is absent from this file on purpose, and `--out` exists so the
destination is never hardcoded — the tests run this against a fixture root.

Window bounds. `inner_voice_observations.created_at` is written **local-naive**, while
UTC runs hours ahead of it, so a bound authored in a different clock does not shift the
window — it moves one end of it, silently. On this box (UTC−7) a `date -u` bound names a
local instant 7 hours later than intended, and the window loses its **oldest** 7 hours:
measured 2026-09-13 on one fixed 26-hour window, 182 LLM calls with the local bound
against 130 with `date -u`, `last` identical both ways. The query succeeds either way, so
the wrong one looks like a quiet night. That hazard is still here and is still this
script's reason for taking the bound as a string.

What is no longer here is #835's second, worse face. A date-only bound or SQLite's
`datetime('now')` renders with a *space*, and the stored rows carry a `T`; a raw
comparison of the two was decided by the separator (`T` 0x54 above space 0x20), so every
row of the bound's own day counted as "after the bound" whatever its hour — a 3-hour
window once reported 3,634 rows whose honest count was 0, and the two separator forms of
one instant disagreed 170 vs 4. `iv_grade.py`'s `WINDOW_CLAUSE` now normalises both sides
with `replace(..., 'T', ' ')` before comparing, so the window is decided by the clock and
the two forms cannot disagree. Local authorship is still mandatory: normalising the
separator does nothing about an instant that was wrong to begin with.

This script stores the bound it was handed verbatim and stamps `until` from the same
local clock the rows use, so the window in the row is the window that was asked for,
and the job that hands it the bound is told to hand it in local wall clock.

Exit codes: 0 normal · 2 sustained breach (see the threshold block) · 3 nothing
usable on stdin.

Exit 2 is this process's, and it is *not* the autonomy run's exit code. The nightly
gets here through an agent's Bash tool, so 2 is that tool call's result; the run's own
status is the agent's turn, which exits 0 whether or not the series breached. What
closes that seam is not this script — it is the printed verdict line, which on a breach
contains the literal text `exit code 2`, which is what `_detect_silent_failures`
(`autonomy.py:26-33`, regex at `:29`) scans a run's final prose for. Quote the line and the run is
flagged; that is the only automated surface, and the task's `description`, which is what
`_build_task_prompt` injects, is what tells the agent to quote the line. See `EXIT_BREACH` for why the number is in the prose.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Dropped verdicts / LLM calls. Provisional: the measured baseline for
#: 2026-09-01..09-12 was 0.039 (353 drops over 9,027 LLM calls — 234 at the 12 s
#: deadline, 118 at 5 s), so this is ~2.5x today's rate and is meant to catch a
#: step change, not normal wobble. Alan sets the real value and the channel a
#: breach reaches; that is a person-decision left open on #460, and the test
#: (`tests/test_iv_metrics_series.py`) pins that this default and the number in
#: the task file agree rather than pinning the magnitude.
DEFAULT_THRESHOLD = 0.10
#: Rows the median is taken over: 7 nights at one row a night. A threshold on a
#: single night is a coin flip — the fleet's own history is one bad night at 2.5x
#: the baseline followed by a clean one.
DEFAULT_WINDOW_ROWS = 7
#: Rates below this many can flag but cannot breach. Three nights is the smallest set
#: that is a trend rather than a reading — and a first night, alone in the file, would
#: otherwise alert about nothing but the series existing.
MIN_BREACH_ROWS = 3
#: Extra rows read behind the median window when counting unreadable ones.
READ_SLACK = 100
#: Sustained breach. 2 is the "refused/anomaly" code this repo already uses
#: (`scripts/skill_verdicts.py:320`, `scripts/validate_handoff.py:70`). Named so the
#: value returned and the value printed in the verdict prose cannot drift apart —
#: `_verdict` embeds it in the line the runner scans for `exit code [1-9]`, so a
#: retuned code that only reached the `return` would silently un-arm the alert.
EXIT_BREACH = 2
#: Nothing usable on stdin (empty, unparseable, or a windowless report).
EXIT_NO_INPUT = 3


def _env_threshold() -> float | None:
    raw = os.environ.get("IV_METRICS_DROPPED_THRESHOLD", "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        print(f"iv-metrics: ignoring unparseable IV_METRICS_DROPPED_THRESHOLD="
              f"{raw!r}", file=sys.stderr)
        return None


def _local_now_text() -> str:
    """`until`, in the same shape `created_at` is stored in.

    Local-naive on purpose — matching the column, not the server. See the module
    docstring and #835: `datetime.utcnow()` here would move `until` 7-8 hours
    ahead of the newest row it is claiming to bound.
    """
    return datetime.datetime.now().isoformat(timespec="microseconds")


def _local_today() -> str:
    return datetime.date.today().isoformat()


def _rate(numer: float | None, denom: float | None) -> float | None:
    return None if not denom else round(numer / denom, 4)


def _dropped_verdicts(report: dict) -> tuple[int, int, dict]:
    """(dropped, error_total, per_deadline) from the grader's `cost.errors`.

    `iv_grade.py:257-259` buckets rows carrying an `error` by
    `error.split(':')[0]`, so a deadline miss arrives as `timeout after 12.0s` —
    the deadline text is part of the key, which is what makes "did a retuned
    deadline help?" a field comparison rather than a re-read of prose. A dropped
    verdict is recorded `action='noop'` with `error` set
    (`app/inner_voice/observer.py:926` writes the label, `:956-960` folds it into a
    noop), so `http_error` and anything else carrying an `error` also produced no
    verdict and belongs in the numerator. Because the grader's counter only sees
    truthy `error` values, every key here is evidence of a dropped call; the
    `no reason` guard below is defensive against a future grader that counts the
    healthy majority in the same map.
    """
    errors = report.get("cost", {}).get("errors", {}) or {}
    dropped, error_total, by_deadline = 0, 0, {}
    for key, count in errors.items():
        count = int(count or 0)
        if key == "no reason":
            continue
        error_total += count
        if key.startswith("timeout"):
            by_deadline[key] = count
        dropped += count
    return dropped, error_total, by_deadline


def over_bound(row: dict) -> bool:
    """Does this row's dropped-verdict count exceed its own recorded bound?

    The one-line check #460 asks a consumer to be able to do — `dropped_verdicts`
    against `threshold * llm_calls`, both read off the row, with no prose parsing and
    no re-derivation of the rate. Kept as a function rather than an expression inside
    `_row` so the nightly `flagged` field and anyone reading the file afterwards apply
    *the same* comparison: a threshold two places in the file, one of which can be
    edited, is how a bound stops meaning anything. Strictly `>` — a row sitting
    exactly on the bound is not a breach of it.

    Rows with no LLM calls or no bound answer False rather than raising or dividing:
    a night with no traffic has no rate to compare, and "could not measure" must not
    become "measured clean" — but it is `dropped_rate: null` in the row that says so,
    not this function silently reporting False for a row that did have a rate.
    """
    llm_calls = row.get("llm_calls") or 0
    threshold = row.get("threshold")
    if not llm_calls or threshold is None:
        return False
    return (row.get("dropped_verdicts") or 0) > threshold * llm_calls


def _row(report: dict, *, window_hours: float | None, threshold: float,
         threshold_source: str, window_rows: int) -> dict:
    """The JSONL row: every field a delta needs, numeric and flat.

    Field names follow `iv_grade.py`'s own report: `cost` carries the counts
    (`observations`, `turns`, `llm_calls`, `errors`), `precision_proxy` the landed
    rate, `recall_proxy` the miss rate. `landed_rate`/`miss_rate` arrive as `None`
    when there was nothing to score — no injects, or no session transcript to
    compare — and are stored as null rather than 0.0, so an unmeasurable night can
    never be read as "scored and clean". A wrong `--hours` likewise cannot
    masquerade as a real change: the grader reports the bound it was handed, not
    the span it covered, so the requested span sits beside `since`.
    """
    scope = report.get("scope", {}) or {}
    cost = report.get("cost", {}) or {}
    precision = report.get("precision_proxy", {}) or {}
    recall = report.get("recall_proxy", {}) or {}
    llm_calls = int(cost.get("llm_calls", 0) or 0)
    dropped, error_total, by_deadline = _dropped_verdicts(report)
    return {
        # window: the bound as passed, the data's own ends, and the local instant
        # of measurement. `until` is local because `created_at` is local (#835).
        "since": scope.get("since"),
        "until": _local_now_text(),
        "window_hours": window_hours,
        "first": scope.get("first"),
        "last": scope.get("last"),
        # volumes
        "observations": int(cost.get("observations", 0) or 0),
        "turns": int(cost.get("turns", 0) or 0),
        "llm_calls": llm_calls,
        # the dropped-verdict rate: this series' alertable number
        "dropped_verdicts": dropped,
        "timeout_by_deadline": by_deadline,
        "error_total": error_total,
        "dropped_rate": _rate(dropped, llm_calls),
        # quality proxies, straight from the grader
        "landed_rate": precision.get("landed_rate"),
        "miss_rate": recall.get("miss_rate"),
        # cost
        "observer_ms_per_turn": cost.get("observer_ms_per_turn"),
        "input_tokens_per_turn": cost.get("input_tokens_per_turn"),
        # threshold state, so a breach is reconstructible from the file alone
        "threshold": threshold,
        "threshold_source": threshold_source,
        "window_rows": window_rows,
        "flagged": over_bound({"llm_calls": llm_calls, "threshold": threshold,
                               "dropped_verdicts": dropped}),
        "models": sorted((cost.get("models") or {}).keys()),
        "recorded_at": datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="seconds"),
    }


def _prior_rates(path: Path, keep: int) -> tuple[list[float], int]:
    """Dropped rates of the last `keep` readable rows, oldest first.

    `floor` bounds the backward read at `keep + floor` rows so a series that has
    accumulated thousands of rows is not read whole for a 7-row median; the floor
    only ever engages if rows carry no parseable date. Malformed lines — the
    realistic result of a run killed mid-append — are counted and reported rather
    than dropped silently: a series that is quietly rotting must not report
    "no breach" because its own tail stopped parsing.
    """
    if not path.exists():
        return [], 0
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    tail = lines[-(keep + READ_SLACK):]
    rates, malformed = [], 0
    for line in tail:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        rate = row.get("dropped_rate")
        if isinstance(rate, (int, float)) and not isinstance(rate, bool):
            rates.append(float(rate))
    return rates[-keep:], malformed


def _verdict(row: dict, rates: list, malformed: int) -> tuple[bool, str]:
    """(breach, one-line report). The comparison is over fields, never prose.

    A row is `flagged` when its own rate exceeds the bound. A *breach* is the
    median of the last `window_rows` rates exceeding it, and only once at least
    `MIN_BREACH_ROWS` rates are on the table: one bad night is a report, a run of
    them is a fault. Median rather than "every row" so a single healthy night can
    stop an ongoing breach, and median rather than "any row" so a single spike
    cannot start one — the asymmetry a fixed bound needs to stay worth reading at
    night. No floor on `llm_calls`: a row too small to trust is still a row, and
    dropping rows from the denominator for being small is how a guard ends up
    reading its own missing input and reporting a verdict it cannot justify.
    """
    basis = (rates[-row["window_rows"]:] + [row["dropped_rate"]]
             if row["dropped_rate"] is not None else [])
    breach = (len(basis) >= MIN_BREACH_ROWS
              and statistics.median(basis) > row["threshold"])
    row["breach"] = breach
    row["breach_basis_rows"] = len(basis)
    row["malformed_prior_rows"] = malformed
    rate_text = ("n/a" if row["dropped_rate"] is None
                 else f"{row['dropped_rate']:.4f}")
    parts = [
        f"iv-metrics: since={row['since']} llm_calls={row['llm_calls']} "
        f"dropped={row['dropped_verdicts']} rate={rate_text} "
        f"threshold={row['threshold']} (from {row['threshold_source']}, "
        f"median of {len(basis)} row(s), floor {MIN_BREACH_ROWS})",
        f"landed={row['landed_rate']} miss={row['miss_rate']}",
    ]
    if row["timeout_by_deadline"]:
        parts.append("timeouts=" + ",".join(
            f"{k}:{v}" for k, v in sorted(row["timeout_by_deadline"].items())))
    if malformed:
        parts.append(f"WARNING {malformed} unreadable prior row(s) in the series")
    if breach:
        parts[0] = "BREACH " + parts[0]
        # The token the autonomy runner actually reads. This process's exit code
        # never reaches the scheduler: the nightly runs the pipeline through an
        # agent's Bash tool, so 2 is that tool's result, not the task's, and the
        # run's own exit status is the agent's turn — which completes successfully
        # whether or not anything was wrong. The one automated surface that exists
        # is `_detect_silent_failures` (`autonomy.py:26-33`, regex at `:29`) scanning the run's
        # FINAL PROSE for `exit code [1-9]`. So the verdict line has to name its own
        # exit code, and the task tells the agent to quote this line verbatim: the
        # alert then survives an agent that describes the night in calm prose,
        # because the trigger is a substring of what it was told to paste.
        parts.append(f"dropped-verdict breach: exit code {EXIT_BREACH}")
    elif row["flagged"]:
        # Deliberately without the token above: one bad night is a report, not a
        # fault, and a run whose summary trips the failure detector when nothing is
        # sustained is how indicators get ignored (autonomy.py:35-44 records 33 false
        # positives in a week doing exactly that).
        parts[0] = "flagged (not sustained) " + parts[0]
    return breach, " | ".join(parts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hours", type=float, default=None,
                    help="the span `--since` covered, recorded as metadata; the "
                         "grader reports its bound, not its width")
    ap.add_argument("--threshold", type=float, default=None,
                    help=f"dropped/llm_calls bound (default "
                         f"{DEFAULT_THRESHOLD}, or $IV_METRICS_DROPPED_THRESHOLD)")
    ap.add_argument("--window-rows", type=int, default=DEFAULT_WINDOW_ROWS,
                    help="rows the median is taken over")
    ap.add_argument("--out", default=str(REPO_ROOT / "_pipeline" / "reflection"
                                         / "iv-metrics.jsonl"),
                    help="series file to append one row to")
    args = ap.parse_args(argv)

    raw = sys.stdin.read()
    if not raw.strip():
        print("iv-metrics: nothing on stdin; expected `iv_grade.py --json` piped in",
              file=sys.stderr)
        return EXIT_NO_INPUT
    try:
        report = json.loads(raw)
    except json.JSONDecodeError as exc:
        # The grader prints prose and exits 0 when no rows match, so a parse
        # failure is the usual sign that a window bound selected nothing. Loud.
        print(f"iv-metrics: stdin is not a grader JSON report ({exc}); first "
              f"80 chars: {raw[:80]!r}", file=sys.stderr)
        return EXIT_NO_INPUT
    if not isinstance(report, dict) or "scope" not in report:
        print("iv-metrics: not an iv_grade report", file=sys.stderr)
        return EXIT_NO_INPUT
    since = (report.get("scope") or {}).get("since")
    if not str(since or "").strip() or str(since) == "all time":
        # A windowless report is "all time": its rate would sit in the series
        # beside nightly rows and make every delta meaningless. Refuse the row
        # rather than poison the trend, and fail rather than look healthy.
        print("iv-metrics: report has no window bound (`since` is null) — run "
              "iv_grade.py with --since; refusing to store an all-time row",
              file=sys.stderr)
        return EXIT_NO_INPUT

    env_threshold = _env_threshold()
    threshold = (args.threshold if args.threshold is not None
                 else env_threshold if env_threshold is not None
                 else DEFAULT_THRESHOLD)
    source = ("--threshold" if args.threshold is not None
              else "IV_METRICS_DROPPED_THRESHOLD" if env_threshold is not None
              else "default")

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    rates, malformed = _prior_rates(out, args.window_rows)
    row = _row(report, window_hours=args.hours, threshold=threshold,
               threshold_source=source, window_rows=args.window_rows)
    # Annotate before the append: `breach` and `malformed_prior_rows` have to be
    # IN the stored row, or reconstructing why a night alerted means re-running
    # the median against a file that has since grown.
    breach, verdict = _verdict(row, rates, malformed)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")

    print(verdict)
    if breach:
        # Returned for anyone running this by hand or in a pipeline that checks it.
        # Not the alert: the nightly arrives through an agent's Bash tool, so this is
        # that tool call's status and the run's own exit status is the agent's turn,
        # which is 0 either way. The alert is the prose `_verdict` printed above —
        # it carries the literal text "exit code 2", which `_detect_silent_failures`
        # (`autonomy.py:26-33`, regex at `:29`) matches out of the run's summary — and the task's
        # the run's report body is where a person sees it. Pinned by
        # tests/test_iv_metrics_series.py::a_breach_verdict_trips_the_runner_detector.
        print(verdict, file=sys.stderr)
    return EXIT_BREACH if breach else 0


if __name__ == "__main__":
    sys.exit(main())
