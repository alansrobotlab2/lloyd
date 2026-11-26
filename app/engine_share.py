"""What share of a turn's wall clock the engine actually owns (#4240).

A token-speed win is bounded by the share of wall clock that tokens occupy.
Kundel's argument about OpenAI's 8x-faster tier — the part of that talk that
transfers — is exactly that inequality: "if half your time or even longer is
spent on running compilers or running end-to-end tests and validations, you
might not actually notice that much the speed up". Lloyd has a queue of
inference-throughput arms (#2144 draft-model fine-tune for speculative
decoding, #3771 KV-prefix pre-warm, #4128 fp4 weight quantization, #2019
prefix-reuse pricing, and the landed #1267) and none of them is gated on the
one number that bounds its own payoff. This module computes that number
locally, because the video offers no measurement of it — the transferable
content is the inequality, not a share.

**Both quantities already exist, on the two sides of one subtraction.**

* Numerator — engine time. `app/harness/loop.py` stamps each iteration with
  `duration_ms` measured from `it.started_at` to `iteration_ended_at`, both
  inside `_stream_iteration`: it is the model request's span (prefill, stream,
  stream retries) and it stops *before* tool dispatch. Every writer persists
  it: `app/routers/messages.py` onto each per-iteration `assistant_message`
  stats block, `app/run_recorder.py` likewise.
* Denominator — turn wall clock. `usage.duration_ms` is one row per turn,
  written by both writers from the stream's `result` event.

Everything else (tool dispatch, hooks, prefetch, compaction, queue waits) is
the difference, and nothing names it per cause. `usage.tool_ms_total` names
part of it — that is the only reason this report prints it beside every engine
share, and it is a *floor* on the non-engine remainder, not a breakdown.

**The numerator has to be scoped, or the share is meaningless.** The same
`stats` dict object an iteration writes is copied onto several rows, so an
unguarded sum of `assistant.stats.duration_ms` counts one iteration several
times; and the *turn*-level stats block — `stats_dict = stream_stats` at
`app/routers/messages.py:1610` — also lands on the final assistant entry, and
its `duration_ms` is the whole turn. The triage measurement over 7 days of this
box's transcripts put the damage precisely: the unscoped reading reports p50
1.867 and a share above 1.0 for **1,258 of 1,503** joined sessions (measured
2026-11-17). So the scoping rule is the instrument:

* an entry counts only if its stats carry `iteration` and do **not** carry
  `num_turns` — the per-iteration shape, not the turn-total shape;
* measurements are de-duplicated on `(turn_id, iteration)`, last writer wins,
  because the identical dict is written on the assistant row *and* the
  tool-call row of one iteration (and on the reconstructed tail rows);
* any entry shape is accepted, not only `role == "assistant"`, because an
  iteration that streamed tool calls with no text is never written on an
  assistant row at all — `build_assistant_text_entry` runs only
  `if evt.get("tool_calls") and full_response.strip()` — and its engine time
  would otherwise be silently missing from the numerator. Every row that
  qualifies carries the same dict for its iteration, so widening the role does
  not double anything the de-dupe has not already collapsed.

**A missing duration is never a fast engine.** Rows the reconstructed paths
write (`_tool_pair` with `evt=None`, `run_recorder._unpersisted_pairs`) can
carry stats that are *absent* rather than zero, so an unguarded sum undercounts
engine time and the share reads low — the silent direction. That is why coverage
is published as `k / N` per class and why a class under `COVERAGE_FLOOR` prints
`coverage too thin to interpret` instead of a number, the shape #2378's "join
too weak to interpret" clause uses.

**A share above 1.0 is a clock disagreement, not a result.** The engine cannot
cost more than the turn that contains it, so a session whose de-duplicated
engine ms exceeds the sum of its `usage.duration_ms` is named as a clock
mismatch and gets no ratio. One-clock rule (#1001 lineage): `usage.ts` is naive
UTC (the column default is `strftime(...,'now')`) and session timestamps are
naive local, so this module never compares the two — it windows the corpus by
session *file name* and reads every usage row of the sessions it selected,
which needs no clock arithmetic across the two frames at all.

**The unit of observation is one session, not one turn.** `usage.db` has no
`turn_id` column, so the join is by `session_id` and the denominator is the sum
of that session's rows. Per-turn attribution is what #2275's declined
dispatch-side field would buy; the item's step 7 re-opens that seam only if a
class reaches 0.5 with n >= 30.

**Gate-rung time is a separate denominator, and this says so.** #2378 measured
228.5 h of ledger gate/review-rung subprocess time over 1468 rounds. That time
is inside an automod *round's* wall clock, not inside any *turn's*: the ladder
runs detached in the aggregator while the round's session sits between turns.
So the ladder figure is printed beside the automod classes for scale and is
explicitly excluded from every denominator here — a faster model cannot touch
it, and this instrument does not claim it is part of a turn either.

**Report-only.** Nothing is promoted, gated or auto-disabled by this file. The
one write it performs is its own dated row set under
`app.paths.ENGINE_SHARE_DIR`, pruned past `RETENTION_DAYS`, exactly as
`app/tool_quality.py` bounds its own store (#2275's precedent). Its output ends
with the decision sentence the item asks for; if a run prints no ruling, the
instrument is disabled and should be deleted (#543 close-out discipline).

**Seams.** One process boundary: this is run as `python -m app.engine_share`
by whatever schedules it, and it reads what two *other* writers leave on disk —
`app/routers/messages.py` and `app/run_recorder.py` — plus `usage.db`. The
write-side half of #4240 is `turn_usage.TurnTelemetry.engine_ms_total`, which
folds the same quantity off the live event stream and fills the long-dead
`usage.duration_api_ms` column through the dict both writers already splat; it
keys on the iteration number for the same reason `deduped_iteration_ms` keys on
`(turn_id, iteration)`, so the column and this numerator are one quantity rather
than two clocks. `tests/test_usage_store_telemetry.py` puts a test across that
boundary on both writers; `tests/test_engine_share_report.py` seeds transcripts
with `app.transcript_entries`, the builders the writers themselves use, so the
reader is pinned against the shapes that are really persisted rather than shapes
invented here.
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

from app.atomic_io import atomic_write_text

#: Schema tag of the emitted row set, so a later reader of a dated file knows
#: which scoping rule produced it.
SCHEMA = "engine-share-v1"

#: Below this fraction of a class's sessions carrying measured engine time, the
#: class publishes coverage and no share. 0.6 is the item's stated floor; the
#: triage reading over 7 days put the pooled figure at 1,297/1,503 = 0.863, so
#: the floor is a guard against the reconstructed-row hole, not a normal state.
COVERAGE_FLOOR = 0.6

#: A class smaller than this is printed but never rules on: the item's own
#: publication bar ("a class needs n >= 30 to publish").
MIN_CLASS_N = 30

#: The reading that would flip a decode arm from capped to supported: an engine
#: that owns at least half of a publishable class's wall clock. It is also the
#: number the item names for re-opening #2275's per-tool field, so it appears
#: once, here, and the report prints it rather than restating it.
DECODE_DECISIVE_SHARE = 0.5

DEFAULT_DAYS = 7
RETENTION_DAYS = 30

#: The throughput arms the decision sentence is allowed to rule on, in the item's
#: numbering, each with the leg of the engine's time it attacks. A decode arm
#: cannot win more than the engine's own share, and a prefill arm only the
#: prefill part of it, which these columns cannot separate.
THROUGHPUT_ARMS = (
    ("#2144", "decode", "fine-tune the speculative-decoding draft model on the "
                        "session corpus"),
    ("#3771", "prefill", "pre-warm the KV prefix for predictable worker prompts"),
    ("#4128", "decode", "fp4 weight quantization for agent tasks"),
    ("#2019", "prefill", "price candidate arms by success-gated prefix reuse"),
)

#: `source` values that are the automod loop's own turns, so the report can name
#: the round denominator next to them.
AUTOMOD_SOURCES = ("autocode", "automod-review")

# A session id's `source` column collapses to a job class here. `autonomy-task`
# is one class, not 90: the mint rule puts the task number in the session
# `source`, and per-task classes are all under the publication bar while the
# path they share is one job class. The empty source falls back to `platform`,
# which is what names the interactive paths (`mission-control`, `browser`).
_SOURCE_PREFIXES = (("autonomy-task", "autonomy-task"),)


def job_class(doc: Mapping[str, Any]) -> str:
    """The class a session belongs to, read off its own metadata.

    Never guessed from a title or a path: `sessions/*.json` carries `source` and
    `platform` because the writer that made the session named it.
    """
    source = str(doc.get("source") or "").strip()
    if source:
        for prefix, name in _SOURCE_PREFIXES:
            if source.startswith(prefix):
                return name
        return source
    return str(doc.get("platform") or "").strip() or "unknown"


def is_iteration_stats(stats: Any) -> bool:
    """True for a per-iteration stats block, false for a turn-total one.

    The two shapes are distinguished by their keys, the same way #859's closing
    instrument distinguishes them: the turn row has no `iteration` key, and the
    per-iteration row never carries `num_turns`. Both conditions are checked so
    neither key can be added to one shape by a later writer without the other
    shape starting to leak into the numerator.
    """
    return (isinstance(stats, Mapping)
            and "iteration" in stats
            and "num_turns" not in stats)


def deduped_iteration_ms(entries: Any) -> "dict[tuple[str, Any], int]":
    """One engine duration per `(turn_id, iteration)`, last writer wins.

    Accepts any entry that carries a per-iteration stats block — the assistant
    text row, the tool-call row and the reconstructed tail rows all carry the
    same dict for one iteration — so an iteration that streamed tool calls with
    no text is not missing from the numerator. Values are the ms the engine's
    own request span measured; a stats block whose `duration_ms` is missing or
    non-integer contributes nothing, which is the "unmeasured, not zero" rule.
    """
    out: "dict[tuple[str, Any], int]" = {}
    for entry in entries or []:
        if not isinstance(entry, Mapping):
            continue
        stats = entry.get("stats")
        if not is_iteration_stats(stats):
            continue
        duration = stats.get("duration_ms")
        if not isinstance(duration, int) or isinstance(duration, bool):
            continue
        key = (str(entry.get("turn_id") or ""), stats.get("iteration"))
        out[key] = max(0, duration)
    return out


def engine_ms_of_session(doc: Mapping[str, Any]) -> tuple[int, int, int]:
    """`(engine_ms, iterations_persisted, iterations_measured)` for one session.

    * `iterations_persisted` is the de-duplicated count of per-iteration rows.
      It is printed beside the turn count because it is *always* smaller: both
      writers flush a per-iteration stats row only when a tool call follows the
      iteration — both `app/routers/messages.py` and `app/run_recorder.py` guard
      it with `if evt.get("tool_calls") and <text>.strip():` — so the iteration that answered
      in prose alone is never on disk and the share reads low by at least one
      iteration per turn. The figure is printed so a reader can see how much of
      a turn the numerator cannot see; closing it is a transcript-shape change,
      not this report's job.
    * `iterations_measured` counts those rows where `duration_ms` is a non-zero
      integer. A session with zero of them has *no* measurement, and the caller
      must not read its absence as an engine that answered instantly.
    """
    durations = deduped_iteration_ms(doc.get("messages") or [])
    return (sum(durations.values()), len(durations),
            sum(1 for ms in durations.values() if ms > 0))


def percentile(values: "list[float]", q: float) -> "float | None":
    """Nearest-index quantile of an unsorted list; None when empty.

    The index rule is copied from `gate_duration_stats` verbatim —
    `min(n - 1, int(q * (n - 1) + 0.5))` on the sorted list — so this table's p90
    and the gate ladder's p90 are the same statistic read the same way, and a
    reader comparing them is not comparing two definitions. It is a nearest-index
    pick, not an interpolated quantile: for 10 sessions it lands on the 9th of the
    sorted shares, and for 4 sessions `p50` is the 3rd.

    A mean would be the wrong statistic whatever the rule, because the whole point
    of the table is that the classes differ by an order of magnitude.
    """
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))]


# --------------------------------------------------------------- the reader --
def read_usage_totals(db_path: Path) -> "dict[str, dict[str, Any]]":
    """Per-session totals from `usage.db`, opened read-only.

    Every usage row is read, not a window: `usage.ts` is naive UTC while
    session timestamps are naive local, and comparing the two frames is the
    #1001 mistake this module avoids by never windowing the database at all. A
    session's rows all belong to that session, and the corpus window is applied
    to the session files instead.

    Read-only (`mode=ro`) for the reason `usage_store.read_rows_readonly` gives:
    a report must never write to — or migrate — a live `usage.db`. A missing
    column reads as None so an old store cannot crash the sweep.
    """
    import sqlite3

    out: "dict[str, dict[str, Any]]" = {}
    if not Path(db_path).is_file():
        return out
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        have = {r[1] for r in conn.execute("PRAGMA table_info(usage)")}
        want = ["duration_ms", "tool_ms_total", "tool_calls", "ttft_ms_first",
                "duration_api_ms", "tool_batch_ms_total"]
        sel = ", ".join(c if c in have else "NULL" for c in want)
        # The denominator of the completeness figure: `num_turns` is what the
        # harness counted as iterations, so `iterations / SUM(num_turns)` says
        # how much of a turn's engine time ever reached a stats row.
        iters = "SUM(num_turns)" if "num_turns" in have else "NULL"
        # Per TURN, not per session: "ttft measured" has to be a rate over the
        # same denominator the share uses, or a session with one measured turn
        # out of twenty reads as fully instrumented.
        ttft = ("SUM(CASE WHEN ttft_ms_first IS NOT NULL THEN 1 ELSE 0 END)"
                if "ttft_ms_first" in have else "NULL")
        rows = conn.execute(
            f"SELECT session_id, {sel}, COUNT(*) AS turns, {iters} AS iterations, "
            f"{ttft} AS ttft_turns FROM usage GROUP BY session_id").fetchall()
        for row in rows:
            rec: "dict[str, Any]" = {}
            for key, value in zip(["session_id", *want, "turns", "iterations",
                                   "ttft_turns"], row):
                rec[key] = value
            turn_ms = rec["duration_ms"]
            if not isinstance(turn_ms, int) or turn_ms <= 0:
                continue
            out[str(rec["session_id"])] = rec
    except sqlite3.Error:
        return out
    finally:
        conn.close()
    return out


def count_usage_rows(db_path: Path) -> "dict[str, int]":
    """`{rows, rows_with_api_ms}` over the whole store, read-only.

    `rows_with_api_ms` is the figure the item's post-landing check wants to see
    grow, printed daily so nobody has to run the query by hand to learn whether
    the column is still dead.
    """
    import sqlite3

    out = {"rows": 0, "rows_with_api_ms": 0}
    if not Path(db_path).is_file():
        return out
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        have = {r[1] for r in conn.execute("PRAGMA table_info(usage)")}
        out["rows"] = int(conn.execute("SELECT COUNT(*) FROM usage").fetchone()[0])
        if "duration_api_ms" in have:
            out["rows_with_api_ms"] = int(conn.execute(
                "SELECT COUNT(*) FROM usage WHERE duration_api_ms IS NOT NULL"
            ).fetchone()[0])
    except sqlite3.Error:
        pass
    finally:
        conn.close()
    return out


def _new_class(cls: str) -> "dict[str, Any]":
    return {"class": cls, "sessions": 0, "measured": 0, "engine_ms": 0,
            "turn_ms": 0, "tool_ms": 0, "tool_calls": 0, "ttft_turns": 0,
            "turns": 0, "iterations": 0, "iterations_expected": 0,
            "shares": [], "mismatched": [],
            # #4386: the sessions whose usage row carries `tool_batch_ms_total`
            # (NULL on every row written before it), and their own sums.
            "batch_sessions": 0, "batch_ms": 0, "batch_engine_ms": 0,
            "batch_turn_ms": 0}


def collect(*, sessions_dir: "Path | str", db_path: "Path | str",
            days: int = DEFAULT_DAYS, today: "str | date | None" = None,
            coverage_floor: float = COVERAGE_FLOOR,
            min_class_n: int = MIN_CLASS_N) -> "dict[str, Any]":
    """Engine share per job class, from the transcripts and `usage.db`.

    Pure over the two stores: reads them, writes nothing. The returned dict is
    what `format_report` renders and what the dated row set stores.
    """
    from app.tool_quality import load_transcript, transcript_paths

    root = Path(sessions_dir)
    store = Path(db_path)
    anchor = _as_day(today) or datetime.now().date()
    paths = transcript_paths(root, days=days, today=anchor)
    usage = read_usage_totals(store)

    classes: "dict[str, dict[str, Any]]" = {}
    joined = 0
    mismatched: "list[dict[str, Any]]" = []
    unreadable = 0
    for path in paths:
        doc = load_transcript(path)
        if not isinstance(doc, dict):
            unreadable += 1
            continue
        sid = str(doc.get("session_id") or path.stem)
        row = usage.get(sid)
        if row is None:
            continue
        joined += 1
        cls = classes.setdefault(job_class(doc), _new_class(job_class(doc)))
        engine_ms, iterations, measured_iters = engine_ms_of_session(doc)
        turn_ms = int(row["duration_ms"])
        cls["sessions"] += 1
        if not measured_iters:
            # Nothing on disk says how long the engine took, so this session
            # joins neither numerator nor denominator. Putting its `turn_ms` in
            # the denominator anyway is the unguarded sum the item warns about:
            # the ratio comes back low by exactly the engine time the rows do not
            # carry, which reads as a fast engine rather than an unmeasured one.
            continue
        cls["measured"] += 1
        if engine_ms > turn_ms:
            # The engine cannot cost more than the turn that contains it, so the
            # two clocks disagree (or a transcript outlived the rows its own
            # session wrote). Name it, and fold NOTHING from it into a total: a
            # session with a broken clock inflates every ratio it joins, which is
            # what first showed on live data on 2026-11-17, where an excluded
            # session's engine ms still sat in the class total and left the
            # autocode class's engine and tool shares summing to 1.107.
            cls["mismatched"].append(sid)
            mismatched.append({"session_id": sid, "engine_ms": engine_ms,
                               "turn_ms": turn_ms, "class": cls["class"]})
            continue
        cls["turns"] += int(row["turns"] or 0)
        cls["iterations"] += iterations
        cls["iterations_expected"] += int(row.get("iterations") or 0)
        cls["turn_ms"] += turn_ms
        cls["engine_ms"] += engine_ms
        cls["tool_ms"] += int(row["tool_ms_total"] or 0)
        batch = row.get("tool_batch_ms_total")
        if isinstance(batch, int) and not isinstance(batch, bool):
            # Same row as `duration_ms` and `tool_ms_total`, so the batch figure
            # is over its own turn; one span per batch, batches never overlap,
            # so engine + tool_batch stays a fraction of that turn.
            cls["batch_sessions"] += 1
            cls["batch_ms"] += max(0, batch)
            cls["batch_engine_ms"] += engine_ms
            cls["batch_turn_ms"] += turn_ms
        cls["tool_calls"] += int(row["tool_calls"] or 0)
        cls["ttft_turns"] += int(row.get("ttft_turns") or 0)
        if turn_ms > 0:
            cls["shares"].append(engine_ms / turn_ms)

    for cls in classes.values():
        cls["coverage"] = (cls["measured"] / cls["sessions"]) if cls["sessions"] else 0.0
        # Two different reasons not to print a share, and a reader has to be
        # able to tell them apart: thin coverage says the transcripts do not
        # carry the measurement, an empty `shares` list with good coverage says
        # every session that did carry one was thrown out for disagreeing clocks.
        cls["refusal"] = (None if (cls["coverage"] >= coverage_floor
                                   and cls["shares"])
                          else ("coverage too thin to interpret"
                                if cls["coverage"] < coverage_floor
                                else "no comparable session: every measured one "
                                     "disagreed on the clocks"))
        cls["publishable"] = cls["refusal"] is None
        cls["p50"] = percentile(cls["shares"], 0.5) if cls["publishable"] else None
        cls["p90"] = percentile(cls["shares"], 0.9) if cls["publishable"] else None
        cls["tool_share"] = (cls["tool_ms"] / cls["turn_ms"]) if cls["turn_ms"] else None
        # No pooled figure for a refused class either: the class total is exactly
        # the number the floor exists to stop anyone reading.
        cls["pooled"] = ((cls["engine_ms"] / cls["turn_ms"])
                         if (cls["publishable"] and cls["turn_ms"]) else None)
        # #4386: over the batch-measured sessions only — a NULL row joins neither
        # side, so pre-landing traffic cannot pull the figure toward zero.
        cls["batch_share"] = ((cls["batch_ms"] / cls["batch_turn_ms"])
                              if (cls["publishable"] and cls["batch_turn_ms"])
                              else None)
        cls["batch_pooled"] = ((cls["batch_engine_ms"] / cls["batch_turn_ms"])
                               if cls["batch_share"] is not None else None)

    publishable = [c for c in classes.values()
                   if c["publishable"] and c["sessions"] >= min_class_n]
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "days": days,
        "window_from": (anchor - timedelta(days=days)).isoformat(),
        "files_in_window": len(paths),
        "files_unreadable": unreadable,
        "sessions_joined": joined,
        "usage_rows": count_usage_rows(store),
        "coverage_floor": coverage_floor,
        "min_class_n": min_class_n,
        "classes": sorted(classes.values(), key=lambda c: (-c["sessions"], c["class"])),
        "mismatched": mismatched,
        "gate_ladder": gate_ladder_stats(),
        "decision": decide(publishable, min_class_n=min_class_n),
    }


def _as_day(value: "str | date | None") -> "date | None":
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value)).date()
    except ValueError:
        return None


def gate_ladder_stats() -> "dict[str, Any]":
    """How long a full gate ladder takes now, from the automod ledger.

    Reused, not re-derived (#2378's reuse ruling): the one definition lives in
    `scripts/automod/backlog.gate_duration_stats`. Printed for scale beside the
    automod classes and never folded into a denominator — the ladder runs
    detached, so its seconds are in a round's wall clock and not in any turn's.
    Unavailable (no ledger, no `scripts` on the path) reports `{"n": 0}`.
    """
    try:
        from scripts.automod import backlog as B, state as S
        return B.gate_duration_stats(S.LEDGER_PATH)
    except Exception:                                        # noqa: BLE001
        return {"n": 0}


def decide(publishable: "list[dict[str, Any]]",
           *, min_class_n: int = MIN_CLASS_N) -> "list[dict[str, Any]]":
    """The ruling the item requires: one line per throughput arm.

    A 2x decode win saves at most half of the engine's share of wall clock, so
    a decode arm is worth buying where some publishable class actually spends
    half its turn inside the engine, and capped where the engine is a minority
    of the time. The bar is a *class* at or above `DECODE_DECISIVE_SHARE` with
    `n >= min_class_n`; the pooled figure cannot settle it, because the whole
    point of the table is that the classes differ by an order of magnitude.

    Prefill arms get the weaker claim, and the honest one: a turn keeps only
    `ttft_ms_first`/`ttft_ms_max` — one number for the whole turn, the first
    chunk of its first iteration (`app/turn_usage.py:173-177`) — so the prefill
    part of the engine share is not summable out of these columns however many
    rows carry a TTFT, and the ruling says that instead of inventing a split.
    """
    strong = [c for c in publishable
              if c["sessions"] >= min_class_n
              and (c["p50"] or 0.0) >= DECODE_DECISIVE_SHARE]
    paid = max(strong, key=lambda c: (c["p50"] or 0.0), default=None)
    best = max(publishable, key=lambda c: (c["p50"] or 0.0), default=None)
    # The counterweight: the publishable class whose turn is most spent on tool
    # dispatch. Kundel's inequality cuts both ways, and a ruling that names only
    # the class where speed pays has not stated the case where it does not.
    bound = max(publishable, key=lambda c: (c["tool_share"] or 0.0), default=None)
    rulings: "list[dict[str, Any]]" = []
    for item, leg, what in THROUGHPUT_ARMS:
        if leg == "decode":
            if paid:
                verdict = "supports"
                why = (f"class {paid['class']} spends {_f(paid['p50'])} of its "
                       f"wall clock inside the engine (p50, n={paid['sessions']} "
                       f">= {min_class_n}), so a 2x decode win there buys up to "
                       f"{_f((paid['p50'] or 0.0) / 2)} of that class's time"
                       + (f"; it caps at {_f((bound['p50'] or 0.0) / 2)} on "
                          f"{bound['class']}, where tool dispatch is already "
                          f"{_f(bound['tool_share'])} of the turn"
                          if bound and bound is not paid else ""))
            else:
                # `best` is None when no class was publishable at all — no
                # session joined the two stores, or every class fell under the
                # floor. The arm is still capped, by the absence of a reading.
                verdict = "caps"
                best_p50 = (best["p50"] or 0.0) if best else 0.0
                why = ("no class at n>="
                       f"{min_class_n} reaches engine p50 "
                       f"{_f(DECODE_DECISIVE_SHARE)}"
                       + (f"; the highest publishable class is {best['class']} at "
                          f"{_f(best['p50'])} (n={best['sessions']})" if best
                          else "; no class published a share at all")
                       + f", so a 2x decode win caps at {_f(best_p50 / 2)} of even "
                         "that class")
        else:
            verdict = "caps"
            why = ("bounded by the engine share, which these columns cannot then "
                   "split into prefill and decode: a turn keeps one ttft "
                   "(ttft_ms_first, the first chunk of its first iteration), so "
                   "this has to be priced against #520's measured prefix-cache "
                   "hit rate rather than off a share it cannot isolate")
        rulings.append({"item": item, "leg": leg, "arm": what,
                        "verdict": verdict, "why": why})
    return rulings


def _f(value: "float | None") -> str:
    return "—" if value is None else f"{value:.3f}"


def _k(value: "int | None") -> str:
    return "0" if not value else f"{value:,}"


def format_report(report: Mapping[str, Any]) -> str:
    """The printed table. Every rate is k/N, every class prints its n.

    Three lines a reader is meant to grep for, each pinned by a test:
    `coverage too thin to interpret`, `clock mismatch`, and the `decision:`
    block naming a throughput item with `supports` or `caps`.
    """
    out: "list[str]" = []
    out.append(f"engine share — engine_ms / turn_ms   (schema {report['schema']})")
    out.append(f"window: session files dated {report['window_from']} or newer "
               f"(days={report['days']}); unit = one session")
    usage = report["usage_rows"]
    out.append(f"corpus: files_in_window={_k(report['files_in_window'])}, "
               f"unreadable={_k(report['files_unreadable'])}, "
               f"joined_to_usage={_k(report['sessions_joined'])}")
    out.append(f"usage.db: rows={_k(usage['rows'])}, "
               f"duration_api_ms filled on {_k(usage['rows_with_api_ms'])}"
               f"/{_k(usage['rows'])} rows")
    out.append("")
    out.append(f"{'class':<22} {'n':>5} {'coverage':>12} {'p50':>7} {'p90':>7} "
               f"{'engine_ms/turn_ms':>19} {'tool_ms/turn':>13} {'tool_calls':>11}")
    for cls in report["classes"]:
        coverage = f"{cls['measured']}/{cls['sessions']}"
        if not cls["publishable"]:
            out.append(f"{cls['class']:<22} {_k(cls['sessions']):>5} "
                       f"{coverage:>12} {'—':>7} {'—':>7} {'—':>19} "
                       f"{_share(cls['tool_share']):>13} {_k(cls['tool_calls']):>11}"
                       f"  {cls['refusal']}")
            continue
        out.append(f"{cls['class']:<22} {_k(cls['sessions']):>5} {coverage:>12} "
                   f"{_f(cls['p50']):>7} {_f(cls['p90']):>7} "
                   f"{_k(cls['engine_ms']) + '/' + _k(cls['turn_ms']):>19} "
                   f"{_share(cls['tool_share']):>13} {_k(cls['tool_calls']):>11}")
        out.append(f"    turns={_k(cls['turns'])}  pooled={_f(cls['pooled'])}  "
                   f"ttft measured on {cls['ttft_turns']}/{cls['turns']} turns"
                   + ("  (no ttft: this class reports none)"
                      if cls["ttft_turns"] == 0 else ""))
        if cls["publishable"] and cls["tool_share"] is not None:
            both = (cls["pooled"] or 0.0) + cls["tool_share"]
            out.append(f"    engine + tool over turn_ms: {_f(cls['pooled'])} + "
                       f"{_f(cls['tool_share'])} = {_f(both)}"
                       + ("  (>1 here is not a clock mismatch: engine_ms and "
                          "tool_ms_total are summed over different things, and "
                          "tool_ms_total gives each of several calls issued "
                          "together its own span, so on a class that "
                          "parallelises the two overlap)" if both > 1.0 else ""))
        if cls["publishable"]:
            out.append(_batch_line(cls))
        out.append(f"    engine iterations on disk: {cls['iterations']}"
                   + (f"/{cls['iterations_expected']} counted by the harness"
                      f" = {_frac(cls['iterations'], cls['iterations_expected'])}"
                      "  (an iteration that answered in prose alone is never"
                      " written, so this share reads low by what is missing)"
                      if cls["iterations_expected"] else " (harness count NULL)"))
        if cls["mismatched"]:
            out.append(f"    clock mismatch: {len(cls['mismatched'])} session(s) "
                       f"excluded, engine_ms > turn_ms: "
                       + ", ".join(cls["mismatched"][:5]))
    out.append("")
    total = sum(c["sessions"] for c in report["classes"])
    measured = sum(c["measured"] for c in report["classes"])
    engine = sum(c["engine_ms"] for c in report["classes"])
    turn = sum(c["turn_ms"] for c in report["classes"])
    tools = sum(c["tool_ms"] for c in report["classes"])
    out.append(f"all classes: coverage={measured}/{total}, "
               f"pooled engine_ms/turn_ms={_k(engine)}/{_k(turn)}"
               + (f" = {engine / turn:.3f}" if turn else ""))
    out.append(f"all classes: tool_ms_total/turn_ms={_k(tools)}/{_k(turn)}"
               + (f" = {tools / turn:.3f}" if turn else "")
               + "  (a SUM of per-call spans, not the wall-clock remainder: "
                 "several calls issued together each report their own span, so on "
                 "a class that parallelises engine + tool can exceed 1.0 without "
                 "either clock being wrong — see each class's engine + tool line)")
    if report["mismatched"]:
        out.append(f"clock mismatch: {len(report['mismatched'])} session(s) where "
                   f"the de-duplicated engine_ms exceeded turn_ms — no ratio "
                   f"printed for them (one-clock rule, #1001 lineage):")
        for row in report["mismatched"][:10]:
            out.append(f"    {row['session_id']} ({row['class']}): "
                       f"engine {row['engine_ms']:,} ms > turn {row['turn_ms']:,} ms")
    out.append(_automod_line(report))
    ladder = report.get("gate_ladder") or {}
    if ladder.get("n"):
        out.append(f"gate ladder, a SEPARATE denominator: a full gate takes median "
                   f"{ladder['median_s']} s / p90 {ladder['p90_s']} s over "
                   f"{ladder['n']} rounds — detached from every turn, so none of "
                   f"it is inside any share above")
    else:
        out.append("gate ladder: no ledger history (n=0), so no figure for the "
                   "separate automod denominator")
    out.append("")
    out.append("decision:")
    for ruling in report["decision"]:
        out.append(f"  {ruling['item']} ({ruling['leg']}: {ruling['arm']}) — "
                   f"{ruling['verdict'].upper()}: {ruling['why']}")
    if not report["decision"]:
        out.append("  no ruling produced — the instrument is disabled (#543)")
    out.append("")
    out.append("report-only: nothing here is promoted, gated or auto-disabled. "
               "Re-opening #2275's per-tool field needs a class at engine share "
               f">= {_f(DECODE_DECISIVE_SHARE)} with n >= {MIN_CLASS_N}.")
    return "\n".join(out)


def _batch_line(cls: Mapping[str, Any]) -> str:
    """#4386: `tool_batch_ms_total` beside the engine and `tool_ms_total` lines.

    One wall-clock span per batch of calls issued together, so unlike
    `tool_ms_total` it cannot count an overlap twice and `engine + tool_batch`
    over `turn_ms` is a fraction of the turn. Computed over the sessions whose
    row carries the column; a class with none says so on its own line and every
    other line it prints is unchanged.
    """
    if cls.get("batch_share") is None:
        return ("    engine + tool_batch over turn_ms: —  (tool_batch_ms_total is "
                "NULL on every row of this class: written before #4386)")
    both = (cls["batch_pooled"] or 0.0) + cls["batch_share"]
    return (f"    engine + tool_batch over turn_ms: {_f(cls['batch_pooled'])} + "
            f"{_f(cls['batch_share'])} = {_f(both)}  (tool_batch_ms_total "
            f"{_k(cls['batch_ms'])} ms over the {cls['batch_sessions']}/"
            f"{cls['measured']} measured sessions whose row carries it)")


def _share(value: "float | None") -> str:
    return "—" if value is None else f"{value:.3f}"


def _frac(numerator: int, denominator: int) -> str:
    return "—" if not denominator else f"{numerator / denominator:.3f}"


def _automod_line(report: Mapping[str, Any]) -> str:
    """The worked example the item asks the report to reach, printed as numbers.

    An automod round is the compile-bound case in its extreme form, so the
    sentence that prices a decode arm for the loop has to name both clocks: the
    engine's share of a round's *turns*, and the gate ladder's seconds in the
    round's *wall clock*. Kundel's inequality is exactly this shape — "if half
    your time or even longer is spent on running compilers or running end-to-end
    tests and validations, you might not actually notice that much the speed
    up" — and the ledger's 228.5 h over 1468 rounds (#2378's measurement) is
    why a faster model cannot buy back the second term at any speed.
    """
    automod = [c for c in report["classes"] if c["class"] in AUTOMOD_SOURCES]
    if not automod:
        return ("automod rounds: no session in the window carries an automod "
                "source, so no figure for the loop's own turns")
    engine = sum(c["engine_ms"] for c in automod)
    turn = sum(c["turn_ms"] for c in automod)
    sessions = sum(c["sessions"] for c in automod)
    share = engine / turn if turn else None
    half = (share or 0.0) / 2
    return (f"automod rounds: turns carry engine_ms/turn_ms={_k(engine)}/{_k(turn)}"
            + (f" = {share:.3f} over {sessions} sessions, so a 2x decode win caps "
               f"at {half:.3f} of a round's turn time and the gate ladder is the "
               f"bigger term" if share is not None else ""))


def prune_store(out_dir: Path, *, today: "date | None" = None,
                days: int = RETENTION_DAYS) -> "dict[str, int]":
    """Delete this store's dated files older than `days`. Never raises.

    Same rule `app.tool_quality.prune_store` states: age is the date in the
    name, not mtime, and only `^\\d{4}-\\d{2}-\\d{2}\\.json$` inside this
    directory is this store's to delete.
    """
    import re

    out = {"files": 0, "bytes": 0, "errors": 0}
    root = Path(out_dir)
    if not root.is_dir() or days <= 0:
        return out
    anchor = today or datetime.now().date()
    cutoff = anchor - timedelta(days=days - 1)
    for path in root.glob("*.json"):
        match = re.match(r"^(\d{4}-\d{2}-\d{2})\.json$", path.name)
        if not match:
            continue
        try:
            file_day = datetime.strptime(match.group(1), "%Y-%m-%d").date()
        except ValueError:
            continue
        if file_day >= cutoff:
            continue
        try:
            size = path.stat().st_size
            path.unlink()
            out["files"] += 1
            out["bytes"] += size
        except OSError:
            out["errors"] += 1
    return out


def sweep(*, sessions_dir: "Path | str | None" = None,
          db_path: "Path | str | None" = None,
          out_dir: "Path | str | None" = None,
          days: int = DEFAULT_DAYS,
          today: "str | date | None" = None,
          retention_days: int = RETENTION_DAYS,
          write: bool = True) -> "dict[str, Any]":
    """Collect the report, print it, and publish one dated row set.

    Defaults resolve *at call time* so a process whose data root moved (a test's
    scratch `LLOYD_DATA`, a round's worktree) never reads one tree and writes
    another. `write=False` prints without touching the store, which is what the
    CLI's `--dry-run` does.
    """
    from app import paths as _paths

    root = Path(sessions_dir) if sessions_dir else Path(_paths.SESSIONS_DIR)
    store = Path(db_path) if db_path else Path(_paths.USAGE_DB)
    report = collect(sessions_dir=root, db_path=store, days=days, today=today)
    report["text"] = format_report(report)
    if not write:
        return report
    out_root = Path(out_dir) if out_dir else Path(_paths.ENGINE_SHARE_DIR)
    out_root.mkdir(parents=True, exist_ok=True)
    anchor = _as_day(today) or datetime.now().date()
    payload = {k: v for k, v in report.items() if k != "text"}
    atomic_write_text(out_root / f"{anchor.isoformat()}.json",
                      json.dumps(payload, indent=1, sort_keys=True) + "\n")
    report["pruned"] = prune_store(out_root, today=anchor, days=retention_days)
    return report


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(
        description="Engine share of turn wall clock, per job class (#4240).")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS,
                        help=f"session-file window (default {DEFAULT_DAYS})")
    parser.add_argument("--sessions-dir", default=None,
                        help="transcript directory (default: app.paths.SESSIONS_DIR)")
    parser.add_argument("--db", default=None,
                        help="usage.db path (default: app.paths.USAGE_DB)")
    parser.add_argument("--out-dir", default=None,
                        help="row-set directory (default: app.paths.ENGINE_SHARE_DIR)")
    parser.add_argument("--today", default=None,
                        help="day the window ends at and the name the row set is"
                             " written under, as YYYY-MM-DD or YYYYMMDD (the"
                             " spelling a transcript file name carries);"
                             " unparseable means today's clock")
    parser.add_argument("--dry-run", action="store_true",
                        help="print without writing the dated row set")
    args = parser.parse_args(argv)
    report = sweep(sessions_dir=args.sessions_dir, db_path=args.db,
                   out_dir=args.out_dir, days=args.days, today=args.today,
                   write=not args.dry_run)
    print(report["text"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
