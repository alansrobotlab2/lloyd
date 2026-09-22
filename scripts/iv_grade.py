#!/usr/bin/env python3
"""Inner Voice — retrospective grading over recorded observations.

Read-only analysis. Nothing in the chat path depends on this, and it must
stay that way: the observer's job is to watch the primary, not to be
watched synchronously by a third thing.

The subsystem records everything needed to judge itself — every decision,
its trigger, its content, its cost, and the session it belongs to — and
until now nothing read any of it back. That left no answer to the two
questions that decide whether Inner Voice is worth its cost:

  * PRECISION — when it intervened, did the primary actually change course?
  * RECALL    — when it stayed silent at the end of a turn, did the user
                immediately have to correct the agent?

Both are proxies, and they are labelled as proxies in the output. An
inject "landed" if the primary's next assistant message did something
other than repeat itself; a terminal noop "missed" if the user's very next
message reads like a correction. Neither is ground truth. They are good
enough to spot a regression and to compare two prompts, which is what the
tuning loop actually needs.

UNITS — the one thing that used to be wrong here (backlog #995). Two different
quantities are reported, and only one of them is a round-trip:

  * `observer ms SUMMED / turn` is a SUM: every LLM millisecond in the window
    divided by turns. A turn costs several calls (measured 09-01..09-17: 1,189
    turns over 9,967 LLM calls = 8.4 calls/turn), so this is an order of
    magnitude larger than any single request.
  * `observer ms PER CALL` is p50/p90/p99 of `latency_ms` over LLM rows that
    carry no error — the actual round-trip distribution.

Every deadline in the subsystem (`inner_voice.observer.timeout_seconds`,
`async_timeout_seconds`) is PER CALL, so only the second pair may be compared
to it. Printed alone, the summed line caused #458 to argue "mean observer
latency is 25.8 s/turn against a 12 s deadline … the deadline is arithmetically
doomed": both halves were unit errors. The deadline sits ABOVE p99 (p50 1.8 s /
p90 4.1 s / p99 9.2 s over that window).

Percentiles exclude error rows because a failed call's latency is its own
deadline cut-off (stamped after the request was abandoned), not a completed
round-trip — 369 error rows in the 09-01 window range 1.6 s to 34.8 s.

Usage:
    python scripts/iv_grade.py                    # all sessions
    python scripts/iv_grade.py --session <id>
    python scripts/iv_grade.py --since '2026-09-16 04:00:00'   # LOCAL wall clock
    python scripts/iv_grade.py --json             # machine-readable
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_LLOYD_HOME = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_LLOYD_HOME))

DB_PATH = _LLOYD_HOME / "usage.db"
SESSIONS_DIR = _LLOYD_HOME / "sessions"

INTERVENTIONS = ("inject", "cancel", "ambient", "clarify")

# A row whose decision came from a deterministic guard rather than the model.
# These cost no LLM call, so `_was_llm_call` already excludes them from the
# cost tables — which also made them invisible in the per-trigger interventions
# column, and `pretool` vanished from the report entirely even on the evening it
# fired 19 injects. They also trivially satisfy the precision proxy: they only
# ever fire mid-turn, so the loop always continues afterwards, and 19 false
# positives read as a 1.0 landed rate. Score them apart.
_DETERMINISTIC_REASON_RE = re.compile(r"^(?:deterministic:|fast-path:)", re.IGNORECASE)


def _is_deterministic(row: dict) -> bool:
    return bool(_DETERMINISTIC_REASON_RE.match(row.get("reason") or ""))

# The user's next message reading as a correction is the strongest cheap
# signal that a terminal noop was wrong. Deliberately narrow: a follow-up
# question is normal conversation, not a correction.
_CORRECTION_RE = re.compile(
    r"\b(?:you (?:didn'?t|did not|never|forgot|missed|skipped)"
    r"|that'?s (?:not|wrong|incorrect)|not what i (?:asked|wanted|meant)"
    r"|try again|you were supposed to|i asked (?:you )?(?:to|for)"
    r"|still (?:broken|failing|not working|wrong)|actually,? no"
    r"|finish (?:it|the)|incomplete|you stopped)\b",
    re.IGNORECASE,
)


def _rows(conn: sqlite3.Connection, where: str, params: list) -> list[dict]:
    conn.row_factory = sqlite3.Row
    sql = f"""SELECT id, session_id, turn_id, sequence_in_turn, trigger, action,
                     reason, content, related_tool, input_tokens, output_tokens,
                     cache_read, latency_ms, model, error, created_at
              FROM inner_voice_observations {where} ORDER BY id"""
    return [dict(r) for r in conn.execute(sql, params)]


def _session_messages(session_id: str) -> list[dict]:
    p = SESSIONS_DIR / f"{session_id}.json"
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text()).get("messages") or []
    except Exception:
        return []


def _text_of(msg: dict) -> str:
    content = msg.get("content")
    if isinstance(content, str):
        return content
    out = []
    for chunk in content or []:
        if isinstance(chunk, dict) and chunk.get("type") == "text":
            out.append(chunk.get("text") or "")
    return "".join(out)


def _grade_injects(rows: list[dict]) -> dict[str, Any]:
    """Did an inject change what the primary did next?

    Proxy: an inject that is followed, later in the same turn, by at least
    one more assistant_message decision means the loop continued and the
    primary read the nudge. An inject with nothing after it in the turn
    means the turn ended anyway — the nudge bought nothing.
    """
    by_turn: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_turn[r["turn_id"]].append(r)

    landed = stranded = 0
    deterministic = 0
    deterministic_turns: Counter = Counter()
    stranded_examples: list[dict] = []
    for turn_id, turn_rows in by_turn.items():
        turn_rows.sort(key=lambda r: r["sequence_in_turn"])
        for i, r in enumerate(turn_rows):
            if r["action"] != "inject":
                continue
            if _is_deterministic(r):
                # A guard inject always fires mid-turn, so "the loop continued"
                # is true by construction and says nothing about whether the
                # nudge was right. Count them; do not score them.
                deterministic += 1
                deterministic_turns[turn_id] += 1
                continue
            later = turn_rows[i + 1:]
            continued = any(x["trigger"] == "assistant_message" for x in later)
            if continued:
                landed += 1
            else:
                stranded += 1
                if len(stranded_examples) < 5:
                    stranded_examples.append({
                        "turn_id": turn_id,
                        "trigger": r["trigger"],
                        "reason": (r["reason"] or "")[:110],
                    })
    total = landed + stranded
    worst = deterministic_turns.most_common(3)
    return {
        "injects": total,
        "landed": landed,
        "stranded": stranded,
        "landed_rate": round(landed / total, 3) if total else None,
        "stranded_examples": stranded_examples,
        "deterministic_injects": deterministic,
        # More than a couple of guard injects in one turn is the shape of a
        # miscalibrated guard, not of a primary in trouble.
        "deterministic_worst_turns": [
            {"turn_id": t, "injects": n} for t, n in worst
        ],
    }


def _grade_terminal_noops(rows: list[dict]) -> dict[str, Any]:
    """Did a turn the observer signed off on draw a correction?

    Proxy for recall. Looks at the `result`-trigger decision for each turn
    and asks whether the user's next message reads like a correction.
    """
    by_session: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["trigger"] == "result":
            by_session[r["session_id"]].append(r)

    checked = missed = 0
    missed_examples: list[dict] = []
    for session_id, result_rows in by_session.items():
        msgs = _session_messages(session_id)
        if not msgs:
            continue
        user_texts = [
            (m.get("timestamp") or "", _text_of(m))
            for m in msgs
            if m.get("role") == "user" and not str(m.get("source") or "").startswith("inner_voice_")
        ]
        for r in result_rows:
            if r["action"] not in ("noop", "acknowledge_complete"):
                continue
            ts = r["created_at"] or ""
            nxt = next((t for stamp, t in user_texts if stamp > ts), None)
            if nxt is None:
                continue
            checked += 1
            if _CORRECTION_RE.search(nxt):
                missed += 1
                if len(missed_examples) < 5:
                    missed_examples.append({
                        "session_id": session_id,
                        "turn_id": r["turn_id"],
                        "observer_reason": (r["reason"] or "")[:90],
                        "user_said": nxt.strip()[:130],
                    })
    return {
        "terminal_noops_with_a_following_user_message": checked,
        "followed_by_correction": missed,
        "miss_rate": round(missed / checked, 3) if checked else None,
        "miss_examples": missed_examples,
    }


def _was_llm_call(r: dict) -> bool:
    """Did this row cost a round-trip?

    Keyed on measured spend rather than on the reason text. Reason
    prefixes have drifted (`fast-path:`, `Fast-path:`, and since v5
    `observation-only:` for pretool rows written with no call at all), and
    a text match silently reclassified whole triggers as expensive.
    """
    return bool(r.get("input_tokens") or r.get("latency_ms"))


def _percentile(sorted_vals: list[int], pct: int) -> int | None:
    """Nearest-rank percentile of an ascending list; `pct` is a whole percent.

    Nearest-rank (`ceil(pct·n/100)`, 1-based) rather than an interpolated
    quantile: the value reported is then always a latency the observer really
    took, and the arithmetic is integer-only, so `p99` of ten fixture rows is
    pinned exactly the same way in a test as on 9,891 live rows. An empty
    distribution returns None instead of raising — a window whose every LLM row
    errored has no completed round-trip to report, which is a measurement of
    "nothing to measure", not a crash.
    """
    n = len(sorted_vals)
    if not n:
        return None
    rank = -(-pct * n // 100)                      # ceil(pct*n/100), 1-based
    return sorted_vals[min(rank, n) - 1]


def _ms_text(value: int | None) -> str:
    """Milliseconds for the report, or the shape that says 'not measured'."""
    return "not measured" if value is None else f"{value:,}"


def _cost(rows: list[dict]) -> dict[str, Any]:
    turns = {r["turn_id"] for r in rows}
    llm = [r for r in rows if _was_llm_call(r)]
    # Per-call round-trips, error rows excluded (see the module docstring). A row
    # that spent tokens but recorded no latency has no duration to place in a
    # distribution, so it is out of the denominator too — `latency_ms_per_call_n`
    # prints that denominator beside the percentiles rather than leaving a reader
    # to assume it is `llm_calls`.
    lat_ok = sorted(r["latency_ms"] for r in llm
                    if not r["error"] and r["latency_ms"] is not None)
    in_tok = sum(r["input_tokens"] or 0 for r in rows)
    cached = sum(r["cache_read"] or 0 for r in rows)
    by_trigger: dict[str, dict[str, int]] = defaultdict(
        lambda: {"calls": 0, "in_tok": 0, "ms": 0, "interventions": 0, "guard": 0}
    )
    for r in llm:
        b = by_trigger[r["trigger"]]
        b["calls"] += 1
        b["in_tok"] += r["input_tokens"] or 0
        b["ms"] += r["latency_ms"] or 0
        if r["action"] in INTERVENTIONS:
            b["interventions"] += 1
    # Guard interventions cost no LLM call, so they are absent from `llm` —
    # which is why `pretool` disappeared from this table entirely on the
    # evening the repetition guard fired 19 times. Count them separately so a
    # trigger that is spending nothing but acting a lot is still visible.
    for r in rows:
        if r["action"] in INTERVENTIONS and _is_deterministic(r):
            by_trigger[r["trigger"]]["guard"] += 1
    return {
        "observations": len(rows),
        "turns": len(turns),
        "llm_calls": len(llm),
        "fast_path_share": round(1 - len(llm) / len(rows), 3) if rows else None,
        "input_tokens": in_tok,
        "cached_input_tokens": cached,
        "cache_hit_rate": round(cached / in_tok, 3) if in_tok else None,
        "input_tokens_per_turn": round(in_tok / len(turns)) if turns else 0,
        # SUM, not a round-trip: total LLM ms divided by turns (#995). The key
        # name stays for compatibility — `scripts/iv_metrics_record.py:290` reads
        # this exact key into the nightly series, and renaming it would break the
        # trend at the seam and every row already written. Only the PRINTED label
        # carries the unit.
        "observer_ms_per_turn": round(
            sum(r["latency_ms"] or 0 for r in llm) / len(turns)
        ) if turns else 0,
        # Per-call percentiles of the same `latency_ms`, over non-error rows. New
        # keys, so the recorder's existing rows and its fixture are untouched.
        "latency_ms_per_call_n": len(lat_ok),
        "latency_ms_per_call_p50": _percentile(lat_ok, 50),
        "latency_ms_per_call_p90": _percentile(lat_ok, 90),
        "latency_ms_per_call_p99": _percentile(lat_ok, 99),
        "by_trigger": {k: dict(v) for k, v in sorted(by_trigger.items())},
        "models": dict(Counter(r["model"] or "?" for r in rows)),
        "errors": dict(Counter(
            (r["error"] or "").split(":")[0] for r in rows if r["error"]
        )),
    }


#: The `--since` predicate, kept as one SQL expression so the query cannot drift
#: from the text `--help` describes.
#:
#: Backlog #835. `created_at` is written local-naive ISO with a `T`
#: (`usage_store.record_inner_voice_observation`), while a bound produced by
#: SQLite's `datetime('now', …)` or by `date '+%Y-%m-%d %H:%M:%S'` renders with a
#: SPACE. SQLite compares TEXT with BINARY collation, so
#: `'2026-09-16T00:16:07' >= '2026-09-16 04:00:00'` is decided at the separator —
#: `T` (0x54) sorts above space (0x20) — and never reaches the time digits. A
#: space-form bound therefore kept every row of the bound's own day and everything
#: after it: 170 rows where the honest count was 119 on the day #835 was triaged.
#: Normalising both sides to the space form makes the comparison a clock
#: comparison, and makes the two separator forms of one instant agree — which is
#: the acceptance check.
#:
#: The function blocks index use, so the read is a full scan. Accepted: 58,455 rows
#: (measured 2026-09-20) scan in ~25 ms, in a read-only, off-critical-path grader
#: (the module docstring and `scripts/iv_metrics_record.py:19-23` are why it stays
#: off the chat path).
#:
#: Only `T` is replaced, and a `%Y`-rendered date never contains one. A NULL
#: `created_at` comes back NULL from `replace()` and is excluded by the
#: comparison — which is what a row with no timestamp should do.
WINDOW_CLAUSE = "replace(created_at, 'T', ' ') >= replace(?, 'T', ' ')"

#: Names the clock, which is the part that was missing: `--since` is the only
#: time-bounded query on this table, and an unqualified "ISO date lower bound"
#: made `date -u` look like the natural choice on a box whose rows run hours
#: behind UTC.
_SINCE_HELP = (
    "Lower bound on created_at, compared in LOCAL wall clock — the clock "
    "created_at is written in (local-naive ISO). Not UTC: UTC runs hours ahead "
    "of these rows here, so a `date -u` or datetime('now') bound silently drops "
    "the oldest hours of the window. Both separator forms of one instant "
    "('YYYY-MM-DD HH:MM:SS' and 'YYYY-MM-DDTHH:MM:SS') select the same rows."
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", help="restrict to one session id")
    ap.add_argument("--since", help=_SINCE_HELP)
    ap.add_argument("--json", action="store_true", help="emit JSON")
    args = ap.parse_args()

    if not DB_PATH.exists():
        print(f"no usage db at {DB_PATH}", file=sys.stderr)
        return 1

    where, params = [], []
    if args.session:
        where.append("session_id = ?")
        params.append(args.session)
    if args.since:
        where.append(WINDOW_CLAUSE)
        params.append(args.since)
    clause = (" WHERE " + " AND ".join(where)) if where else ""

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    rows = _rows(conn, clause, params)
    if not rows:
        print("no observations matched")
        return 0

    report = {
        "scope": {
            "session": args.session or "all",
            "since": args.since or "all time",
            "first": rows[0]["created_at"],
            "last": rows[-1]["created_at"],
        },
        "cost": _cost(rows),
        "precision_proxy": _grade_injects(rows),
        "recall_proxy": _grade_terminal_noops(rows),
        "action_mix": dict(Counter(r["action"] for r in rows).most_common()),
    }

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    c, p, r = report["cost"], report["precision_proxy"], report["recall_proxy"]
    print(f"\nInner Voice grading — {report['scope']['session']}, "
          f"{report['scope']['first'][:10]} to {report['scope']['last'][:10]}")
    print("=" * 66)
    print(f"\nCOST  {c['turns']} turns, {c['observations']} observations, "
          f"{c['llm_calls']} LLM calls")
    print(f"  fast-path share        {c['fast_path_share']}")
    print(f"  input tokens / turn    {c['input_tokens_per_turn']:,}")
    # Two units, spelled out (#995). The SUMMED line is total LLM ms over turns —
    # a turn costs several calls — and the PER CALL line is the round-trip
    # distribution the observer's per-call deadlines are actually set against.
    print(f"  observer ms SUMMED / turn  {c['observer_ms_per_turn']:,}"
          f"   (sum of {c['llm_calls']} LLM calls / {c['turns']} turns — "
          f"NOT a round-trip)")
    print(f"  observer ms PER CALL       "
          f"p50 {_ms_text(c['latency_ms_per_call_p50'])}"
          f"  p90 {_ms_text(c['latency_ms_per_call_p90'])}"
          f"  p99 {_ms_text(c['latency_ms_per_call_p99'])}"
          f"   (per call, ms; {c['latency_ms_per_call_n']:,} LLM calls with no error)")
    print(f"  cache hit rate         {c['cache_hit_rate']}"
          f"{'   (0.0 means vLLM lacks --enable-prompt-tokens-details)' if not c['cache_hit_rate'] else ''}")
    print(f"  served by              {c['models']}")
    if c["errors"]:
        print(f"  errors                 {c['errors']}")
    print("\n  by trigger:")
    print(f"    {'trigger':<20}{'calls':>7}{'interv':>8}{'guard':>7}"
          f"{'in_tok':>12}{'tok/interv':>12}")
    for trig, b in c["by_trigger"].items():
        per = f"{b['in_tok'] // b['interventions']:,}" if b["interventions"] else "—"
        print(f"    {trig:<20}{b['calls']:>7}{b['interventions']:>8}"
              f"{b.get('guard', 0):>7}{b['in_tok']:>12,}{per:>12}")

    print(f"\nPRECISION PROXY  (did an inject keep the primary working?)")
    print(f"  model-judged injects   {p['injects']}")
    print(f"  loop continued after   {p['landed']}")
    print(f"  turn ended anyway      {p['stranded']}")
    print(f"  landed rate            {p['landed_rate']}")
    for ex in p["stranded_examples"]:
        print(f"    stranded [{ex['trigger']}] {ex['reason']}")
    print(f"  guard injects          {p['deterministic_injects']}"
          f"   (not scored — they always fire mid-turn, so 'the loop "
          f"continued' is true by construction)")
    for t in p["deterministic_worst_turns"]:
        if t["injects"] >= 3:
            print(f"    !! {t['injects']} guard injects in turn {t['turn_id']} "
                  f"— check the guard, not the primary")

    print(f"\nRECALL PROXY  (did a signed-off turn draw a correction?)")
    print(f"  terminal noops checked {r['terminal_noops_with_a_following_user_message']}")
    print(f"  followed by correction {r['followed_by_correction']}")
    print(f"  miss rate              {r['miss_rate']}")
    for ex in r["miss_examples"]:
        print(f"    missed: observer said {ex['observer_reason']!r}")
        print(f"            user then said {ex['user_said']!r}")

    print("\nBoth rates are PROXIES, not ground truth — see the module docstring.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
