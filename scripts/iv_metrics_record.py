#!/usr/bin/env python3
"""Append one row to the Inner Voice metrics series: `~/lloyd-data/_pipeline/reflection/iv-metrics.jsonl`.

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
(`autonomy.py:27-33`, regex at `:30`) scans a run's terminal block for
(`autonomy.py:2268-2270`). Quote the line and the run is flagged, and the task's
`description`, which is what `_build_task_prompt` injects, is what tells the agent to
quote it. See `EXIT_BREACH` for why the number is in the prose.

That prose channel is the one a #460-class failure can still lose: it depends on a model
choosing to paste a token, and a calm paraphrase of the night erases the breach. Since
#1145 there is a second channel that is code's, and it does not care what the run says.
On the night a breach *starts* — this row is a breach and the newest previous row was
not, per `_prior_breaching` — this process calls the guardian's
`announce(level="warning")`: journal line plus desktop toast, voice off, and never
`alert()`, which would also append a ledger row and file a backlog task on every
subsequent night the bound stayed broken. One warning per breach, from `announce_breach`,
and a failed announcement costs neither the row nor the exit code.
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
sys.path.insert(0, str(REPO_ROOT))
from app.paths import PIPELINE_DIR  # noqa: E402

#: Dropped verdicts / LLM calls. **0.05 is Alan's ruling (#460), applied to this
#: constant by #1288 on 2026-09-20; the task file had said 0.05 since vault
#: 1eb0735e.** The measured baseline is 0.039 over 2026-09-01..09-12 — 353 drops over
#: 9,027 LLM calls: 234 at the 12 s deadline, 118 at 5 s; re-measured off `usage.db`
#: on 2026-09-20 at 0.0386 over the same local days. The retired 0.10 could not have
#: caught the one history that mattered: the degraded 2026-09-05..09-11 stretch ran
#: .0548 · .118 · .0341 · .0856 · .0526 · .0137 · .0086 per local day (320 drops over
#: 7,530 calls), so its 7-row median of 0.0526 was under 0.10 and 09-06's 0.118 was
#: the only night over it — one night, where `MIN_BREACH_ROWS` needs three. #458 was
#: written off a single hand-run inside that week. 0.05 sits between the two: above
#: the healthy weeks — 0.014 since, and the 9 rows recorded up to 2026-09-20 run
#: 0.0..0.0203 — and below that median, so it fires on a run of bad nights and stays
#: quiet otherwise. The other half of that ruling was the *channel* a breach reaches
#: (#1145), and it is `announce(level="warning")`, once per breach, from code — see
#: `announce_breach`; `EXIT_BREACH`'s prose remains the report-side half, the one that
#: depends on the nightly quoting it. The tests pin this default both to the number
#: stated in `~/obsidian/autonomy/86-*.md` and to 0.05 itself, so neither can drift
#: back silently.
DEFAULT_THRESHOLD = 0.05
#: Rows the median is taken over: 7 nights at one row a night. A threshold on a
#: single night is a coin flip — the fleet's own history is one night at 0.118
#: followed by nights back at 0.014 and 0.009, which is a spike to read, not a week
#: to announce. Seven nights is also the window the ruled 0.05 was measured against:
#: that stretch's median is over it, every healthy week's median is under it.
DEFAULT_WINDOW_ROWS = 7
#: Rates below this many can flag but cannot breach. Three nights is the smallest set
#: that is a trend rather than a reading — and a first night, alone in the file, would
#: otherwise alert about nothing but the series existing.
MIN_BREACH_ROWS = 3
#: Extra rows read behind the median window when counting unreadable ones.
READ_SLACK = 100
#: Sustained breach. 2 is the "refused/anomaly" code this repo already uses
#: (`scripts/skill_verdicts.py:633`, `scripts/validate_handoff.py:70`). Named so the
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


def _announce_enabled() -> bool:
    """Off only when someone says so: `IV_METRICS_ANNOUNCE=0`.

    The mute exists for the test suite, and it is the recorder's own rather than the
    guardian's `LLOYD_JOURNAL_ALERTS`/`LLOYD_DESKTOP_ALERTS` pair for one reason: the
    recorder decides whether to speak at all, so the switch has to sit where that
    decision is made. Without it, every suite run that reaches a breach appends a real
    line to the live journal and puts a real toast on the user's screen — which is
    precisely the 2026-09-07 incident `notify.py:_channel_on` documents ("a fake
    critical incident in the live journal is the same pollution the `external` gate was
    added for"). Production never sets it, so a breach always announces.
    """
    return os.environ.get("IV_METRICS_ANNOUNCE", "1").strip().lower() not in (
        "0", "false", "no", "off", "")


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
    (`app/inner_voice/observer.py:932` writes the label, `:960-966` folds it into a
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


def _numeric(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def breach_basis(row: dict, prior_rates: list) -> list[float]:
    """The rates the median is taken over: the last `window_rows` prior rates plus this
    row's own — empty when the row has no rate, which is what "nothing to breach" means.

    ONE definition of the basis, shared by `_verdict` (which decides tonight),
    `_prior_breaching` (which decides whether last night was a breach) and `main` (which
    puts the median in the announcement). A second copy is how the toast could quote a
    median the decision did not use, and how "was last night a breach" could be
    re-derived over a different window than the one that night used.
    """
    own = row.get("dropped_rate")
    if not _numeric(own):
        return []
    # `is None`, not falsy: `--window-rows 0` is a real (if useless) request and
    # silently widening it back to 7 would make the basis a different window than the
    # one the caller asked for and the one the row stores.
    window = row.get("window_rows")
    window = DEFAULT_WINDOW_ROWS if window is None else int(window)
    return list(prior_rates or [])[-window:] + [float(own)]


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


def _prior_breaching(last: dict | None, prior_rates: list[float]) -> bool:
    """Was the newest existing row a breach, as that night computed it?

    Read from the row's own stored `breach` when it has one, and re-derived from its
    own stored `threshold`/`window_rows` when it does not — a re-derivation rather than
    a rate comparison, because "the median of the last 7 rows is over the bound" is not
    recoverable from one number. This is what makes "did a breach just *start*?" a
    question code can answer: without it the only signal is "this row is a breach", and
    a breach that has run for five nights would announce itself every night at 08:00.

    Rows appended before #460 stored no `breach` field, and a test fixture seeds prior
    rows as `{"since": ..., "dropped_rate": ...}` with no other keys, so the
    re-derivation is the normal path here, not a fallback for rot.
    """
    if not last:
        return False
    stored = last.get("breach")
    if isinstance(stored, bool):
        return stored
    threshold = last.get("threshold")
    if not _numeric(threshold):
        return False
    basis = breach_basis(last, prior_rates)
    return len(basis) >= MIN_BREACH_ROWS and statistics.median(basis) > threshold


def _prior_series(path: Path, keep: int) -> tuple[list[float], int, bool]:
    """(rates of the last `keep` readable rows oldest-first, malformed count,
    was-the-newest-row-a-breach).

    `floor` bounds the backward read at `keep + floor` rows so a series that has
    accumulated thousands of rows is not read whole for a 7-row median; the floor
    only ever engages if rows carry no parseable date. Malformed lines — the
    realistic result of a run killed mid-append — are counted and reported rather
    than dropped silently: a series that is quietly rotting must not report
    "no breach" because its own tail stopped parsing.

    The newest row that actually parses decides the third value, even if a malformed
    line sits behind it: a corrupt tail must not read as "the breach just started".
    """
    if not path.exists():
        return [], 0, False
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    tail = lines[-(keep + READ_SLACK):]
    rates, malformed = [], 0
    last_row: dict | None = None
    last_basis: list[float] = []
    for line in tail:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if isinstance(row, dict):
            # Snapshot the rates preceding THIS row, so the newest parsed row's breach
            # is re-derived over the basis its own run used — not over a list built by
            # blindly dropping the last element, which is only that row's own rate when
            # the newest row has a readable rate at all.
            last_row, last_basis = row, list(rates)
        rate = row.get("dropped_rate")
        if isinstance(rate, (int, float)) and not isinstance(rate, bool):
            rates.append(float(rate))
    return rates[-keep:], malformed, _prior_breaching(last_row, last_basis)


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
    basis = breach_basis(row, rates)
    breach = (len(basis) >= MIN_BREACH_ROWS
              and statistics.median(basis) > row["threshold"])
    row["breach"] = breach
    row["breach_basis_rows"] = len(basis)
    # Stored, not recomputed at announcement time, for two reasons: the toast quotes the
    # number the decision was made on rather than one re-derived later, and a later
    # reader can reconstruct *why* this night alerted from the row alone — which is what
    # `threshold`/`window_rows`/`breach_basis_rows` are already in the file for.
    row["breach_median"] = (round(float(statistics.median(basis)), 6)
                            if basis else None)
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
        # is `_detect_silent_failures` (`autonomy.py:27-33`, regex at `:30`) scanning the run's
        # terminal block for `exit code [1-9]`. So the verdict line has to name its own
        # exit code, and the task tells the agent to quote this line verbatim: the
        # alert then survives an agent that describes the night in calm prose,
        # because the trigger is a substring of what it was told to paste.
        parts.append(f"dropped-verdict breach: exit code {EXIT_BREACH}")
    elif row["flagged"]:
        # Deliberately without the token above: one bad night is a report, not a
        # fault, and a run whose summary trips the failure detector when nothing is
        # sustained is how indicators get ignored (autonomy.py:36-43 records 33 false
        # positives in a week doing exactly that).
        parts[0] = "flagged (not sustained) " + parts[0]
    return breach, " | ".join(parts)


def breach_announcement(row: dict) -> tuple[str, str]:
    """Toast head and body for a breach that just STARTED. Pure, so it can be pinned.

    Every number is read back out of the row the recorder just wrote — including
    `breach_median` and `breach_basis_rows`, which `_verdict` stored rather than leaving
    the toast to re-derive, so the sentence cannot quote a median or a row count the
    decision did not use. That is why the parenthetical repeats `basis` instead of
    `window_rows`: the window is what was asked for, the basis is what was medianed, and
    a three-row series asked for seven used to say "median of 7 rows" in the toast beside
    a verdict line saying `median of 3 row(s)` — two bases for one decision, in the same
    message pair, which is the thing this function exists not to do.
    """
    basis = row.get("breach_basis_rows") or MIN_BREACH_ROWS
    median = row.get("breach_median")
    median_txt = f"{median:.4f}" if _numeric(median) else "n/a"
    return (
        "Inner Voice metrics: dropped-verdict breach",
        f"{basis} nightly rows medianed {median_txt} dropped verdicts per observer call, "
        f"over the bound of {row.get('threshold')} "
        f"({row.get('threshold_source', 'default')}, median of "
        f"{basis} row(s), floor {MIN_BREACH_ROWS}). Window "
        f"since={row.get('since')}: {row.get('dropped_verdicts')} of "
        f"{row.get('llm_calls')} observer calls returned no verdict. One warning per "
        f"breach; the series is "
        f"_pipeline/reflection/iv-metrics.jsonl.")


def announce_breach(row: dict) -> bool:
    """Send ONE guardian announcement for a breach that just started.

    `announce`, never `alert` — the ruling on #1145. `alert()` is the recording fan-out:
    it appends a ledger row, writes ALERT.md, and routes to `_backlog_task` for
    `critical`/`trigger`/`needs_human` (`notify.py:118-160`, backlog gate at
    `:157-159`), so a bound breach that lasts five nights through `alert` would file
    five tasks and five ledger rows for one condition. `announce()` (`notify.py:162-188`)
    has none of those channels — its whole body is `_journal`, `_desktop` and `_speak` —
    which is why a recurring condition can use it. This passes `voice=False`, so two of
    the three: a journal line and a toast, no spoken sentence. Same import route
    `app/prefix_miss.py:_announce` and `scripts/automod/promote.py:announce` take; the
    guardian lives outside the package and is not importable by name.

    This is the whole point of #1145: the reaching-a-person part is CODE's job. The
    recorder's exit 2 is a Bash tool result inside an agent turn, the turn exits 0
    either way, and the only thing upstream ever read was the literal text `exit code
    2` in the run's terminal block (`autonomy.py:2268-2270`, `_detect_silent_failures`) — so a
    run that paraphrased the verdict lost the breach. Whatever the nightly report says,
    the toast goes out.
    """
    if not _announce_enabled():
        # Loud, not silent. A mute that prints nothing is indistinguishable, in the run
        # record, from a night that decided not to announce — and this line is the only
        # place a reader of that record can see that the fan-out was switched off rather
        # than that it ran. Production never sets the mute, so seeing it means someone
        # did.
        print("iv-metrics: breach NOT announced (IV_METRICS_ANNOUNCE is off)",
              file=sys.stderr)
        return False
    try:
        gdir = REPO_ROOT / "agent-services" / "guardian"
        if not (gdir / "notify.py").is_file():
            return False
        if str(gdir) not in sys.path:
            sys.path.insert(0, str(gdir))
        import gstate
        import notify as notify_mod
        import policy

        notifier = notify_mod.Notifier(
            ledger=gstate.AutomodState(Path(policy.AUTOMOD_STATE)).ledger,
            state_dir=Path(policy.GUARDIAN_STATE),
            vault_root=policy.VAULT_ROOT,
            voice=False,
            voice_window=policy.VOICE_REPEAT_SECONDS,
        )
        title, body = breach_announcement(row)
        results = notifier.announce(title, body, level="warning")
        detail = (", ".join(f"{k}={'ok' if v else 'off'}"
                            for k, v in sorted(results.items())) or "no channel")
        if not any(results.values()):
            # Every room channel declined — muted, or the box has no session bus. Saying
            # "announced breach via journal=off, desktop=off" would report a success with
            # a footnote, and a permanently mute-proof-looking run record is how an alert
            # channel that stopped working reads as a clean series.
            print(f"iv-metrics: breach NOT announced — no channel reported success "
                  f"({detail})", file=sys.stderr)
            return False
        print(f"iv-metrics: announced breach via {detail}", file=sys.stderr)
        return True
    except Exception as exc:  # noqa: BLE001 — the bell never costs the row
        # Loud on stderr, and NOT raised: a missing guardian checkout, an unreadable
        # state dir or a dead session bus must not turn "the row is appended and the
        # exit code is 2" into "the nightly recorded nothing". `_announce` and
        # `promote.announce` swallow theirs silently; this one is on stderr because
        # the whole item is that a breach can go unannounced.
        print(f"iv-metrics: breach announcement FAILED ({exc}); the row is still "
              f"appended and the exit code is still {EXIT_BREACH}", file=sys.stderr)
        return False


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
    ap.add_argument("--out", default=str(PIPELINE_DIR / "reflection" / "iv-metrics.jsonl"),
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
    rates, malformed, was_breaching = _prior_series(out, args.window_rows)
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
        # NOT the alert: the nightly arrives through an agent's Bash tool, so this is
        # that tool call's status and the run's own exit status is the agent's turn,
        # which is 0 either way. Two things happen now, and neither depends on the
        # run's wording:
        #   * the verdict line printed above carries the literal text `exit code 2`,
        #     which `_detect_silent_failures` (`autonomy.py:27-33`, regex at `:30`)
        #     matches out of the run's terminal block — that is the report-side half, and it
        #     still needs the agent to quote the line;
        #   * and if this breach is the FIRST row of a run of them, one guardian
        #     announcement goes out from here, in code, whatever the model then says.
        #     That second part is #1145: the exit code stops at the tool result and the
        #     prose depends on a paraphrase, so a breach could vanish with the run's
        #     phrasing. The toast cannot.
        print(verdict, file=sys.stderr)
        if was_breaching:
            print("iv-metrics: breach was already present in the last row; not "
                  "announced again (one warning per breach)", file=sys.stderr)
        else:
            announce_breach(row)
    return EXIT_BREACH if breach else 0


if __name__ == "__main__":
    sys.exit(main())
