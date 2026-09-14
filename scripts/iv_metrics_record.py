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

Window bounds. `inner_voice_observations.created_at` is written **local-naive**,
while SQLite's own `datetime('now')` is UTC, and #835 documents the consequence: a
UTC bound against local rows is decided lexically against the stored ISO `T`, which
sorts above a space, so *every* row of the bound's own day compares as "greater
than the bound" regardless of hour — a 3-hour window reported 3,634 rows whose
honest count was 0. This script stores the bound it was handed verbatim and stamps
`until` from the same local clock the rows use, so the skew is visible in the row
instead of silently dropping the newest hours.

Exit codes: 0 normal · 2 sustained breach (see the threshold block) · 3 nothing
usable on stdin. An exit code, not a sentence, because the autonomy runner
consumes exit codes reliably and prose only when told.
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
    (`app/inner_voice/observer.py:926` writes the label, `:954-959` folds it into a
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
        "flagged": bool(dropped > threshold * llm_calls) if llm_calls else False,
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
    elif row["flagged"]:
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
        return 3
    try:
        report = json.loads(raw)
    except json.JSONDecodeError as exc:
        # The grader prints prose and exits 0 when no rows match, so a parse
        # failure is the usual sign that a window bound selected nothing. Loud.
        print(f"iv-metrics: stdin is not a grader JSON report ({exc}); first "
              f"80 chars: {raw[:80]!r}", file=sys.stderr)
        return 3
    if not isinstance(report, dict) or "scope" not in report:
        print("iv-metrics: not an iv_grade report", file=sys.stderr)
        return 3
    since = (report.get("scope") or {}).get("since")
    if not str(since or "").strip() or str(since) == "all time":
        # A windowless report is "all time": its rate would sit in the series
        # beside nightly rows and make every delta meaningless. Refuse the row
        # rather than poison the trend, and fail rather than look healthy.
        print("iv-metrics: report has no window bound (`since` is null) — run "
              "iv_grade.py with --since; refusing to store an all-time row",
              file=sys.stderr)
        return 3

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
        # Distinct from 0 and from 3 so a runner that never reads the prose still
        # sees a sustained fault: 2 is the "refused/anomaly" code this repo already
        # uses (`scripts/skill_verdicts.py:320`, `scripts/validate_handoff.py:70`),
        # and `_detect_silent_failures` (`autonomy.py:26-32`) flags an "exit code 2"
        # appearing in a run's summary even when the turn itself succeeded.
        print(verdict, file=sys.stderr)
    return 2 if breach else 0


if __name__ == "__main__":
    sys.exit(main())
