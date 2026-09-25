"""Reader for skill-injection telemetry (#435).

The counterpart to the rows `prefetch._emit_skill_match_events` appends per
prefetched turn. One call answers the two questions the instrument exists for,
over the last N days: per-skill **offer / loaded / ignored** counts, and whether
the store holds any of that at all.

`ignored` is the point of the whole thing. The loaded half has always been
reachable: `app/routers/_messages_subliminal.py` persists the injected block as
a `role="subliminal"` message and `app/uptake.py` parses the `<skill name=…>`
blocks out of it — through a proxy it names itself (`SKILL_PRESENCE_PROXY` at
`app/uptake.py:173`, `SKILL_PRESENCE_NOTE` at `:187-191`), because that proves
what *landed* and says nothing about what was matched and then not rendered.
This module answers the half nothing else could: what a turn was offered and
never got.

Two shapes a caller must be able to tell apart, so the return is a dict and not
a bare mapping of counts. `no_telemetry: True` means the window measured
nothing, so every count a caller might want is unmeasured; a skill entry means
it was measured, and `{"offers": 0, ...}` inside that would be a real
measurement of a skill that stopped being offered. Collapsing the two is the
failure mode this item was filed against: skill-lint's STALE bucket has printed
0 since April because `check_stale` returns early on every skill carrying
`status: active`, so nothing reaches the age comparison — not because zero
skills went stale. `tests/test_skill_lint_report_trust.py:71` is the test-enforced
version of that (`untrustworthy_categories() == ["DRIFT", "STALE"]`), and the same
trap here is one `res.get("offers", 0)` away.

The file is parsed here rather than through `app.event_log.read_events`, which
is that module's paginated UI reader: keyword-only `(session_id, *, offset,
limit=200)`, no root parameter, and a 200-row default page. A 30-day window over
a busy session is thousands of rows, so a counts reader built on a page of 200
would report the tail of each session and silently drop the rest — low numbers
that read as a dead skill. `tests/test_skill_injection_telemetry.py::
test_counts_are_not_clipped_by_the_event_reader_default_page` pins that against
the real default.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

#: Event type `prefetch.SKILL_MATCH_EVENT` writes — one row per reported offer.
#: Duplicated as a literal rather than imported from `prefetch` because
#: `prefetch` imports the MCP tool modules and this module has to stay
#: importable from scripts and jobs that never load them. The drift this risks
#: (a writer renamed, a reader silently counting zero forever) is what
#: `test_the_writer_and_the_reader_name_one_event_type` pins by comparing the two
#: constants directly.
SKILL_MATCH_EVENT = "prefetch.skill_match"

#: How many days a window covers when a caller does not say. Long enough that a
#: skill used weekly is not mistaken for dead. The window is the whole question,
#: so a retirement rule's own threshold stays a human's call (#434, skill-lint's
#: STALE bucket) rather than a constant hidden here.
DEFAULT_DAYS = 30

#: The event a tool call is logged under (`app/routers/messages.py` and
#: `app/run_recorder.py` both write it), and the tool whose calls are a *pull*:
#: the model asking for a skill's body itself. P5's pull arm injects no body, so
#: without this count its skills would read as never loaded.
TOOL_CALL_EVENT = "brain1.tool_call_proposed"
SKILLS_READ_TOOL = "skills_read"

__all__ = ["SKILL_MATCH_EVENT", "DEFAULT_DAYS", "TOOL_CALL_EVENT",
           "SKILLS_READ_TOOL", "skill_injection_counts"]

_SUFFIX = ".events.jsonl"


def skill_injection_counts(root, days: int = DEFAULT_DAYS) -> dict:
    """Per-skill offer / loaded / ignored counts over the last `days` days.

    `root` is a directory of `<session_id>.events.jsonl` — `app.paths.
    EVENT_LOGS_DIR` in production, a `tmp_path` in a test. Files are discovered
    by that suffix, so a stray file in the directory is not read and a session
    with no file simply contributes nothing. `days` is a positive number of days
    ending now.

    Returns::

        {"window_days": days, "since": <iso>, "until": <iso>,
         "events": <rows counted>, "sessions": <distinct sessions>,
         "skipped": <rows unusable>, "no_telemetry": <bool>,
         "skills": {<name>: {"offers": n, "loaded": n, "ignored": n,
                             "max_score": f}},
         "loaded_by_read": {<name>: n}, "reads": <skills_read calls counted>,
         "note": <one-line state of the window>}

    `loaded_by_read` (P5) counts `skills_read(name=…)` calls in the window, off
    the `brain1.tool_call_proposed` rows — the same rows
    `app.uptake.skills_read_by_session` parses. It is a separate mapping rather
    than a fourth key in each `skills` entry so an offer row stays exactly what
    it was, and so a skill that was read but never offered does not appear in
    `skills` as a measured zero-offer skill. It does not move `no_telemetry`,
    which is about the offer rows: a window with reads and no offers still has
    unmeasured offer counts.

    `skipped` is how many of this event type's rows the reader could not use at
    all — a line that failed to parse, an undated row, or one with no skill name
    or no score to attribute. A row that is simply outside the window is *not*
    skipped: it was placed, and the window excluded it, which is the window
    working. `max_score` is the highest score the skill reached in the window,
    the number that separates "never offered" from "offered at 3.1 and never
    rendered" — the first is dead, the second is a threshold one step away.

    `sessions` counts distinct session ids, the closest thing the event log has
    to a turn count: an event row carries no turn or message id and one session
    runs many prefetched turns, so this is a floor on sessions rather than a
    number of turns — which is what the post-landing person check needs to prove
    live traffic reached the new emitter at all.

    `offers` is the emitted row count, not a count of turns:
    `prefetch._reported_offers` caps what one turn writes at
    `SKILL_REPORT_TOP_K`, so a per-turn ceiling is not a total. `ignored` is
    `offers - loaded` — an offer that did not render into the turn's `<context>`
    block. Whether the model then acted on a skill it was *given* is not knowable
    from the event log and is not what this means; `landed` is a rendering fact.
    """
    until = datetime.now(timezone.utc)
    since = until - timedelta(days=max(0, int(days)))

    counts: dict[str, dict[str, float]] = {}
    events = 0
    skipped = 0
    sessions: set[str] = set()
    by_read: dict[str, int] = {}
    reads = 0
    for path in sorted(Path(root).glob(f"*{_SUFFIX}")):
        session_id = path.name[: -len(_SUFFIX)]
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n").strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # A torn tail line is normal after a crash mid-append
                    # (`log_event` holds a per-process lock, so a second writer
                    # can interleave). Skipping it is right; counting it is what
                    # keeps the skip from being invisible, and what makes a file
                    # of nothing-but-garbage read as no telemetry rather than as
                    # a healthy zero offers.
                    skipped += 1
                    continue
                if isinstance(row, dict) and row.get("event") == TOOL_CALL_EVENT:
                    name = _read_skill_name(row.get("data"))
                    if name is not None:
                        when = _row_time(row.get("ts"))
                        if when is not None and since <= when <= until:
                            by_read[name] = by_read.get(name, 0) + 1
                            reads += 1
                    continue
                if not isinstance(row, dict) or row.get("event") != SKILL_MATCH_EVENT:
                    continue        # another event type: not this reader's business
                when = _row_time(row.get("ts"))
                defect = _row_defect(row, when)
                if defect is not None:
                    # Undated, unattributable, or unscored: it cannot be placed
                    # in the window or pinned to a skill, and counting it under a
                    # caller's belief in either is worse than dropping it. The
                    # count is what keeps a store full of defective rows from
                    # reading as a quiet week.
                    skipped += 1
                    continue
                if when < since or when > until:
                    # Placed, and outside: that is the window working, not a
                    # defect. `when > until` is a clock skew or a seeded future
                    # stamp, and neither is evidence about this window.
                    continue
                data = row["data"]
                name = data["skill"]
                score = data["score"]
                bucket = counts.setdefault(
                    name, {"offers": 0, "loaded": 0, "ignored": 0, "max_score": 0.0})
                bucket["offers"] += 1
                bucket["max_score"] = max(bucket["max_score"], float(score))
                if data.get("landed") is True:
                    bucket["loaded"] += 1
                else:
                    # Only `landed: true` counts as loaded. A false row, or one
                    # that never said either way, is an offer that did not
                    # render — which is the count the instrument exists to make.
                    bucket["ignored"] += 1
                events += 1
                sessions.add(session_id)

    return {
        "window_days": days,
        "since": since.isoformat(timespec="seconds"),
        "until": until.isoformat(timespec="seconds"),
        "events": events,
        "sessions": len(sessions),
        "skipped": skipped,
        # True when nothing in the window was measured: an empty store, a store
        # holding only other event types, or rows that are all unparseable,
        # undated, out-of-window or unattributable.
        "no_telemetry": events == 0,
        "skills": counts,
        "loaded_by_read": by_read,
        "reads": reads,
        "note": (
            f"no telemetry: {SKILL_MATCH_EVENT} appears in none of the rows "
            f"under this root inside the window — every offer/ignore count "
            f"below is unmeasured, not zero"
            if events == 0 else
            f"{events} {SKILL_MATCH_EVENT} rows measured across "
            f"{len(sessions)} sessions"
        ),
    }


def _read_skill_name(data):
    """The skill a `skills_read` tool-call row asked for, or None.

    `data.args` is the call's arguments as a JSON *string* (a dict is accepted
    too); anything that is not a `skills_read` call with a non-empty `name` is
    not a read this reader can attribute.
    """
    if not isinstance(data, dict) or data.get("name") != SKILLS_READ_TOOL:
        return None
    args = data.get("args")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    name = args.get("name") if isinstance(args, dict) else None
    return name if isinstance(name, str) and name else None


def _row_defect(row: dict, when):
    """Why this `prefetch.skill_match` row cannot be counted, or None.

    Kept separate from the loop so the reason exists in one place: a row the
    reader cannot use is *reported* (`skipped`), not quietly dropped, which is
    the difference between "the writer stopped emitting" and "the writer
    changed shape" — the second is a bug to chase, the first is the answer.
    """
    if when is None:
        return "undated"
    data = row.get("data")
    if not isinstance(data, dict):
        return "no-data"
    name = data.get("skill")
    if not isinstance(name, str) or not name:
        return "no-skill-name"      # would mint a `""` skill that looks popular
    score = data.get("score")
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        return "no-score"           # a name with no score is not a scored offer
    return None


def _row_time(value):
    """Parse an event row's `ts` into an aware datetime, or None.

    `app.event_log._utc_iso` writes `"%Y-%m-%dT%H:%M:%S.<mmm>Z"` — a *string*,
    not an epoch float. Accepting only a number here is the mistake this function
    exists to avoid: it would filter every real row out and report no telemetry
    over a full store. A numeric epoch is accepted too (some producers write
    `time.time()` floats in their own payloads), and anything else is None.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if not isinstance(value, str) or not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        # A naive stamp is read as UTC, the way every other reader on this box
        # reads one. The alternative — refusing — would drop rows the writer
        # honestly produced; the known cost is the offset-less `ALERT.md`
        # `written:` field, which is not this file's format.
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed
