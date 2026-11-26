"""Withdraw unattended authority from an action class on measured outcomes (#4048).

#534's grant bounds scope, not judgment: nothing on the dispatch path keyed on
what happened the last time a kind of action ran unattended. This module is the
outcome half, as ONE more arm of the same check — `app.harness.policy.check_grants`
consults `withheld()` before its tier-1 early return, against rows in the same
`workers.db` the grants live in (`action_outcomes`, declared in
`workers/queue.py::GRANT_DDL` beside `authority_grants`). There is no second
policy path and no second store.

The pieces, each a decision recorded where it is made:

* **A frozen taxonomy** (`ACTION_CLASSES`, at most `MAX_CLASSES`): a class is an
  unattended terminal step (the tool that commits the effect) plus the ledger
  events that grade it. A class is admitted only if an outcome source
  INDEPENDENT of the actor's own success flag grades it today; everything else
  is in `EXCLUDED_CLASSES` with the reason, never in the taxonomy with an
  invented outcome.
* **A populator seeded only from the automod ledger** (`grade_event`,
  `populate`): `rollback_succeeded`, `regression_check`, `review`, `vault_land`.
  Each row names the event it came from (`source_event`, and `source_event_id`
  = sha256 of the ledger line's bytes, so re-reading a folded archive inserts
  nothing twice) and carries a NOT-NULL `evidence_ref` a person can grep for.
* **A resolver** (`resolve`): `probation` or `clearance`, from the rows alone.
* **An arm switch** (`class_arm_armed`) that is off by default and read from
  `harness.action_class_arm.enabled`. Nothing in the tree sets it; arming it
  changes what every unattended dispatch may do, which is Alan's call.

Enforcement is not prompt-dependent: stdlib plus the store, no model, no prompt
surface — the same property `tests/unit/test_grant_policy.py` pins on policy.py.
"""

from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Iterator, Mapping

logger = logging.getLogger("lloyd-harness-action-outcomes")

OUTCOME_TABLE = "action_outcomes"

CLEAN, BAD, NEUTRAL = "clean", "bad", "neutral"
OUTCOMES = (CLEAN, BAD, NEUTRAL)

PROBATION, CLEARANCE = "probation", "clearance"


# ---------------------------------------------------------------------------
# The numbers, and why each is what it is
# ---------------------------------------------------------------------------

#: K — graded outcomes a class needs before it may act unattended at all, and
#: the clean streak that restores clearance after a bad outcome.
#:
#: Rationale (the rule of three): after K consecutive clean outcomes and no bad
#: one, the 95% upper confidence bound on the class's per-action bad rate is
#: about 3/K. K = 30 bounds it at 10% — the rate at which an unattended action
#: is more often right than a person's review queue would be slow, and the
#: smallest round number whose bound is a single-digit-tens percentage. It is
#: deliberately NOT tuned to the backtest (the item's own risk note: a value
#: picked to make the backtest come out a way is a hand-set knob), and it is
#: below both mapped classes' monthly volume on this box (regression checks
#: ~25/day, vault landings ~22/day over 2026-09..11), so a demoted class can
#: earn its way back in days, not months — probation must not starve its own fix.
CLEARANCE_STREAK_K = 30

#: The window a bad outcome demotes for. 30 days because it is the horizon the
#: item names (`bad_outcomes_30d`) and the retention sweep folds the ledger
#: monthly; a K-long clean streak restores clearance inside it.
WINDOW_DAYS = 30

#: Classes, never more. The item's frozen bound: past eight, a class is a tool.
MAX_CLASSES = 8

#: `rollback_succeeded` triggers that grade the landed commit as bad. Ruling (a)
#: of #4048 is open; until it is made, `promote_failed` (4 of the 11 rows to
#: date) is booked NEUTRAL: it is the promote mechanism failing, not evidence
#: that the landed change was wrong. An unknown future trigger is neutral too —
#: a new detector is not admitted as a judge by default.
COUNTED_ROLLBACK_TRIGGERS = frozenset({"regression", "data_damage", "error_rate", "crash"})

#: Ruling (b) of #4048: does a pre-landing review-rung refusal grade the class?
#: False until ruled. A refused round never reached production, so it is not an
#: outcome of an unattended *action*; and on this ledger a refusal lands most
#: days (614 refusals with a named non-met clause, 2026-09..11), which would hold
#: `automod_land` in probation permanently — the "block always" setting the item
#: exists to avoid. With it False every `review` row is booked neutral (kept, so
#: the record shows it was seen).
REVIEW_REFUSAL_GRADES = False

#: A clause verdict that is not a refusal of the clause.
_PASSING_VERDICTS = frozenset({"met", "unnecessary", "rejected", "post_landing"})

#: A `vault_land` ok:false that is a no-op, not a refused landing: 291 of the
#: 384 ok:false rows to 2026-11-26 say "nothing to commit on those paths" — the
#: working tree already matched HEAD. Counting them bad would freeze the class on
#: an event in which nothing was written.
_VAULT_NOOP_MARK = "nothing to commit"


@dataclass(frozen=True)
class ActionClass:
    name: str
    #: The unattended terminal tool(s) the arm withholds while on probation.
    tools: frozenset[str]
    #: The ledger events that grade this class — its independent outcome source.
    sources: tuple[str, ...]
    #: Who grades, in a sentence: the independence argument.
    graded_by: str


#: The frozen taxonomy. Two classes, because two terminal steps have an outcome
#: source on the ledger that is not the actor's own run status.
ACTION_CLASSES: Mapping[str, ActionClass] = MappingProxyType({
    "automod_land": ActionClass(
        name="automod_land",
        tools=frozenset({"automod_land"}),
        sources=("rollback_succeeded", "regression_check", "review"),
        graded_by=("the guardian's rollback (`rollback_succeeded`) and the detached "
                   "paired-A/B regression check (`regression_check`, bad only with a "
                   "non-empty `confirmed_by`); the review rung per ruling (b)"),
    ),
    "vault_land": ActionClass(
        name="vault_land",
        tools=frozenset({"automod_vault_land"}),
        sources=("vault_land",),
        graded_by=("`automod_vault_land`'s own validators and vault review, which "
                   "refuse and revert the bytes the worker wrote (`vault_land` ok:false)"),
    ),
})

#: Classes considered and excluded, with why. A class with no independent
#: outcome source on the ledger is recorded here rather than graded on nothing.
EXCLUDED_CLASSES: Mapping[str, str] = MappingProxyType({
    "email_send": ("durable-external: zero `grant_dispatch` rows all-time on this box "
                   "(19 rows: autonomy_write_task 9, Bash 9, autonomy_delete_task 1) and "
                   "no event grades a sent mail"),
    "calendar_write": "durable-external: zero calls in `grant_dispatch`, no outcome event",
    "contacts_write": "durable-external: zero calls in `grant_dispatch`, no outcome event",
    "durable_memory_write": ("#552's uptake table grades entries for the consolidator, "
                             "not a ledger event; not seeded here"),
    "autotriage_close": ("a later stale/refuted verdict is the triage worker's own stream, "
                         "not independent of the actor"),
})

assert len(ACTION_CLASSES) <= MAX_CLASSES
assert not set(ACTION_CLASSES) & set(EXCLUDED_CLASSES)

#: tool name → class name, derived (never hand-kept beside the taxonomy).
TOOL_CLASS: Mapping[str, str] = MappingProxyType({
    tool: cls.name for cls in ACTION_CLASSES.values() for tool in cls.tools})


def class_for_tool(tool: str) -> str | None:
    return TOOL_CLASS.get(str(tool or ""))


# ---------------------------------------------------------------------------
# The arm switch
# ---------------------------------------------------------------------------

ARM_CONFIG_KEY = "harness.action_class_arm.enabled"


def class_arm_armed() -> bool:
    """`harness.action_class_arm.enabled`; False by default and on any failure.

    Only a literal `true` arms it. Nothing in the tree writes this key.
    """
    try:
        from app.config import CONFIG

        block = (((CONFIG or {}).get("harness") or {}).get("action_class_arm") or {})
        return block.get("enabled") is True
    except Exception:  # noqa: BLE001 — unreadable config is "disarmed"
        return False


# ---------------------------------------------------------------------------
# Grading ledger events
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GradedEvent:
    action_class: str
    outcome: str
    source_event: str
    source_event_id: str
    ts: str
    evidence_ref: str
    subject: str


#: Every `ts` in the table, and every bound compared against one, is written in
#: this one shape, so SQL's text comparison is time order.
_TS_FORMAT = "%Y-%m-%dT%H:%M:%S+00:00"


def _event_ts(event: dict) -> str:
    """The event's instant: the float `ts` the ledger writes first, else
    `created_at`; '' when it has neither."""
    ts = event.get("ts")
    if isinstance(ts, (int, float)) and not isinstance(ts, bool):
        return dt.datetime.fromtimestamp(float(ts), dt.timezone.utc).strftime(_TS_FORMAT)
    created = event.get("created_at")
    if isinstance(created, str) and created.strip():
        text = created.strip()
        text = text[:-1] + "+00:00" if text.endswith("Z") else text
        try:
            return _iso(dt.datetime.fromisoformat(text))
        except ValueError:
            return ""
    return ""


def line_id(line: bytes) -> str:
    """The id of a ledger row: its own bytes, newline stripped."""
    return hashlib.sha256(line.rstrip(b"\r\n")).hexdigest()[:24]


def _review_refusal_clauses(event: dict) -> list[int]:
    """Clause numbers a review REFUSED on. `ok` is not the field: gate.py writes
    `ok: False` only when the grader could not judge the diff (unreachable,
    unusable, or `unreliable` citations — "not a judgment of the diff"), and
    `ok: True, blocking: True` is the refusal. So a refusal is a judged review
    that blocked, naming a clause it did not find met."""
    if event.get("ok") is not True or event.get("blocking") is not True:
        return []
    out: list[int] = []
    for c in event.get("clauses") or []:
        if isinstance(c, dict) and str(c.get("verdict") or "") not in _PASSING_VERDICTS:
            try:
                out.append(int(c.get("clause")))
            except (TypeError, ValueError):
                continue
    return out


def grade_event(event: dict, source_event_id: str) -> GradedEvent | None:
    """One ledger row → one graded row, or None for an event this ladder does
    not read. Only the four named events are ever graded."""
    if not isinstance(event, dict):
        return None
    name = event.get("event")
    ts = _event_ts(event)
    if not ts:
        return None
    commit = str(event.get("commit") or "")[:12]

    def row(cls: str, outcome: str, ref: str, subject: str) -> GradedEvent:
        return GradedEvent(cls, outcome, str(name), source_event_id, ts,
                           f"promotions ledger {name}@{ts} {ref} [{source_event_id}]",
                           subject)

    if name == "rollback_succeeded":
        trigger = str(event.get("trigger") or "")
        outcome = BAD if trigger in COUNTED_ROLLBACK_TRIGGERS else NEUTRAL
        return row("automod_land", outcome, f"trigger={trigger or '?'} commit={commit}",
                   commit)
    if name == "regression_check":
        regressed = event.get("regressed")
        if regressed is False:
            return row("automod_land", CLEAN, f"commit={commit} regressed=false", commit)
        if regressed is True:
            confirmed = event.get("confirmed_by")
            outcome = BAD if confirmed else NEUTRAL
            note = "confirmed" if confirmed else "unconfirmed"
            return row("automod_land", outcome, f"commit={commit} regressed=true {note}",
                       commit)
        return None
    if name == "review":
        rid = str(event.get("round_id") or "")
        if not REVIEW_REFUSAL_GRADES:
            return row("automod_land", NEUTRAL, f"round={rid} (ruling (b) open)", rid)
        if event.get("ok") is not True:
            return row("automod_land", NEUTRAL, f"round={rid} grader-failure", rid)
        refused = _review_refusal_clauses(event)
        if refused:
            return row("automod_land", BAD,
                       f"round={rid} refused clauses={refused}", rid)
        if event.get("blocking") is True:
            return row("automod_land", NEUTRAL, f"round={rid} blocked-unnamed", rid)
        return row("automod_land", CLEAN, f"round={rid} pass", rid)
    if name == "vault_land":
        item = event.get("item_id")
        ref = f"item={item if item is not None else '-'} commit={commit or '-'}"
        if event.get("ok") is True:
            return row("vault_land", CLEAN, ref, commit)
        if event.get("ok") is False:
            errors = " ".join(str(e) for e in (event.get("errors") or []))
            if _VAULT_NOOP_MARK in errors:
                return row("vault_land", NEUTRAL, ref + " no-op", commit)
            return row("vault_land", BAD, ref + f" errors={errors[:160]!r}", commit)
        return None
    return None


def iter_ledger_lines(paths: Iterable[Path]) -> Iterator[bytes]:
    """Every non-blank line of each file, in the order given; `.gz` decodes."""
    for path in paths:
        p = Path(path)
        try:
            raw = gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes()
        except (OSError, EOFError):
            logger.warning("[action-outcomes] ledger file unreadable: %s", p)
            continue
        for line in raw.splitlines():
            if line.strip():
                yield line


def graded_from_lines(lines: Iterable[bytes]) -> list[GradedEvent]:
    """Graded rows in ledger order, one per distinct line (a row present in
    both an archive and the live file is graded once)."""
    seen: set[str] = set()
    out: list[GradedEvent] = []
    for line in lines:
        sid = line_id(line)
        if sid in seen:
            continue
        seen.add(sid)
        try:
            event = json.loads(line)
        except ValueError:
            continue
        g = grade_event(event, sid)
        if g is not None:
            out.append(g)
    return out


# ---------------------------------------------------------------------------
# The store half — the same workers.db as the grants
# ---------------------------------------------------------------------------


def populate(store: Any, ledger_paths: Iterable[Path], *,
             now: dt.datetime | None = None) -> int:
    """Insert one row per graded ledger event into `action_outcomes`. Idempotent
    on `source_event_id`. Returns how many rows were new.

    `store` is a `GrantStore` (its `_connect` applies the shared DDL)."""
    graded = graded_from_lines(iter_ledger_lines(ledger_paths))
    recorded = (now or dt.datetime.now(dt.timezone.utc)).astimezone(
        dt.timezone.utc).isoformat()
    added = 0
    with store._connect() as conn:
        for g in graded:
            cur = conn.execute(
                f"INSERT OR IGNORE INTO {OUTCOME_TABLE} (action_class, outcome,"
                " source_event, source_event_id, ts, evidence_ref, subject, recorded_at)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (g.action_class, g.outcome, g.source_event, g.source_event_id, g.ts,
                 g.evidence_ref, g.subject or None, recorded))
            added += cur.rowcount
    return added


def rows_for(store: Any, action_class: str, *, before: dt.datetime | None = None
             ) -> list[dict]:
    sql = f"SELECT * FROM {OUTCOME_TABLE} WHERE action_class=?"
    args: list[Any] = [action_class]
    if before is not None:
        sql += " AND ts < ?"
        args.append(_iso(before))
    sql += " ORDER BY ts ASC, id ASC"
    with store._connect() as conn:
        return [dict(r) for r in conn.execute(sql, args)]


def _iso(value: dt.datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc).strftime(_TS_FORMAT)


def _parse(ts: str) -> dt.datetime:
    v = dt.datetime.fromisoformat(ts)
    return v if v.tzinfo else v.replace(tzinfo=dt.timezone.utc)


@dataclass(frozen=True)
class Rollup:
    graded_outcomes: int
    bad_outcomes_30d: int
    clean_streak: int
    last_bad_ref: str | None
    last_ref: str | None


def rollup(rows: Iterable[Mapping[str, Any]], *, now: dt.datetime) -> Rollup:
    """`{graded_outcomes, bad_outcomes_30d, clean_streak}` over rows in time
    order. Neutral rows count for nothing — not graded, and they neither extend
    nor break a streak."""
    graded = bad_window = streak = 0
    last_bad = last = None
    horizon = now - dt.timedelta(days=WINDOW_DAYS)
    for r in rows:
        outcome = r["outcome"]
        if outcome not in (CLEAN, BAD):
            continue
        graded += 1
        last = r["evidence_ref"]
        if outcome == BAD:
            streak = 0
            last_bad = r["evidence_ref"]
            if _parse(r["ts"]) >= horizon:
                bad_window += 1
        else:
            streak += 1
    return Rollup(graded, bad_window, streak, last_bad, last)


@dataclass(frozen=True)
class Standing:
    action_class: str
    status: str
    why: str
    evidence_ref: str


def resolve_rows(action_class: str, rows: Iterable[Mapping[str, Any]], *,
                 now: dt.datetime) -> Standing:
    """`probation` or `clearance` for a class, from its rows alone.

    * an unrecognised class is on probation, never in clearance;
    * fewer than K graded outcomes is probation (no track record);
    * a K-long clean streak is clearance, whatever the window holds — the way
      back;
    * otherwise a bad outcome inside the window is probation;
    * otherwise clearance.
    """
    if action_class not in ACTION_CLASSES:
        return Standing(action_class, PROBATION,
                        "unrecognised action class (not in the frozen taxonomy)", "none")
    r = rollup(rows, now=now)
    if r.graded_outcomes < CLEARANCE_STREAK_K:
        return Standing(action_class, PROBATION,
                        f"{r.graded_outcomes} graded outcome(s), fewer than "
                        f"K={CLEARANCE_STREAK_K}",
                        r.last_ref or "none")
    if r.clean_streak >= CLEARANCE_STREAK_K:
        return Standing(action_class, CLEARANCE,
                        f"clean streak {r.clean_streak} >= K={CLEARANCE_STREAK_K}",
                        r.last_ref or "none")
    if r.bad_outcomes_30d:
        return Standing(action_class, PROBATION,
                        f"{r.bad_outcomes_30d} bad outcome(s) in the last {WINDOW_DAYS} d, "
                        f"clean streak {r.clean_streak} < K={CLEARANCE_STREAK_K}",
                        r.last_bad_ref or "none")
    return Standing(action_class, CLEARANCE,
                    f"no bad outcome in the last {WINDOW_DAYS} d", r.last_ref or "none")


def resolve(store: Any, action_class: str, *, now: dt.datetime) -> Standing:
    if action_class not in ACTION_CLASSES:
        return resolve_rows(action_class, [], now=now)
    return resolve_rows(action_class, rows_for(store, action_class, before=now), now=now)


def withheld(store: Any, *, tool: str, now: dt.datetime) -> Standing | None:
    """The arm's answer for one call: the probation Standing that withholds it,
    or None when the tool is no class's terminal step or the class is cleared."""
    cls = class_for_tool(tool)
    if cls is None:
        return None
    standing = resolve(store, cls, now=now)
    return standing if standing.status == PROBATION else None


def denial_reason(standing: Standing, *, tool: str, scope: str) -> str:
    return (f"action-class: '{tool}' is the unattended terminal step of class "
            f"'{standing.action_class}', which is on {PROBATION} ({standing.why}); "
            f"evidence_ref: {standing.evidence_ref}. Scope '{scope}' may stop at the "
            "branch and its gate report — the draft-only path — and a person may land "
            f"it. Clearance returns after {CLEARANCE_STREAK_K} consecutive clean "
            "graded outcomes (app/harness/action_outcomes.py).")
