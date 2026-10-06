"""Where skill bytes enter a prompt, and how many (#624).

A SKILL.md reaches the model by three routes and they cost very different
amounts. The chat turn-start injection is capped (`prefetch.SKILL_BODY_MAX`, 6000
chars, with an excerpt for the runner-up), so a 1,600-line skill costs a chat turn
the same as a 200-line one. The autonomy task prompt (`autonomy._build_task_prompt`)
and the worker prompt (`workers.sources._common.build_skill_prompt`) splice the
whole file in with no cap and carry it for the run. #624's premise — that shorter
bodies save tokens — is only true on the second kind, so the question "which route
cost what" has to be answerable before a spill pass is judged on it.

`record_skill_embed` writes one `skill.embedded` event into the run's own session
event log (the per-session machine record, `app/event_log.py`) and one
`SKILL_EMBED` INFO line, per skill per turn or run, tagged with the route. A
report over `event_logs/*.events.jsonl` can then sum `embedded_chars` by route.

Accounting, never the turn: nothing here raises.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("lloyd-server")

EVENT = "skill.embedded"

#: The two routes that get NO cut at all: both splice the whole `SKILL.md` into the
#: prompt and book `len(...)` of it — `autonomy._build_task_prompt` above, and
#: `workers.sources._common.build_skill_prompt`, booked by `deep_research.py`. Named as
#: literals rather than imported from `app.harness.skill_dispatch`, which owns the
#: constants: this module is imported by `app.autonomy` and by the workers, and the
#: harness package is an import edge neither should acquire for a string.
#: `tests/test_skill_embed.py::test_only_these_two_routes_are_treated_as_uncapped` pins
#: each literal against the constant that owns it.
UNCAPPED_ROUTES = ("autonomy_task", "worker_prompt")

#: #2272: a CHAR ceiling on those two routes, recorded and never enforced.
#:
#: Value: 6,000 chars, the number `app/prefetch.py:54` holds for one skill's worth of
#: prompt on the route that DOES cut (`SKILL_BODY_MAX`, the cap this module's own
#: docstring names). It is equal to that number by measurement, not by import: a soft
#: limit that shares a name with the hard cut could not move independently when the
#: ruling comes, so the equality is pinned instead —
#: `tests/test_skill_embed.py::test_the_soft_limit_equals_the_capped_route_s_cut_today`
#: fails if either side moves unseen. Chosen over any
#: percentile of the corpus below because a value read off these logs could only ever
#: describe what the uncapped routes already do, and that is the thing under review;
#: measured over that corpus, 274 of the 420 uncapped embeds sit above it today, so the
#: line marks traffic rather than endorsing it. Whether to cap or spill the two routes
#: is a SEPARATE ruling, to be made on the `over_soft_limit` rows this records (#2272
#: clause 5), and nothing here enforces it: no flag, no truncation, and no branch
#: anywhere that reads the stamp back.
#:
#: Re-derive the corpus this was measured on — per-route count, total and worst, over
#: the two uncapped routes ONLY. Route-scoped because a sum over every route inherits a
#: phantom: `skill_dispatch._walk_deliveries` books a `<skill>` tag quoted in prose with
#: no closing tag as everything after it (live example: a 10,115-char `prefetch` row for
#: `tool-call-args-truncation` in session `20261003_210909_owedcheck_731d`).
#: `SKILL_EMBED_LOG_GLOB` overrides the path, so a test can point it at a fixture log.
#:
#: $ python3 - <<'PY'
#: import glob, json, os, collections
#: ROUTES = ("autonomy_task", "worker_prompt")        # the routes with NO cut
#: pat = os.environ.get("SKILL_EMBED_LOG_GLOB") or os.path.expanduser(
#:     "~/lloyd-data/event_logs/*.events.jsonl")
#: by = collections.defaultdict(list)
#: for f in glob.glob(pat):
#:     for line in open(f, errors="replace"):
#:         try:
#:             row = json.loads(line)
#:         except Exception:
#:             continue
#:         d = row.get("data") or {}
#:         if row.get("event") == "skill.embedded" and d.get("route") in ROUTES:
#:             by[d["route"]].append(int(d.get("embedded_chars") or 0))
#: print(json.dumps({r: {"n": len(v), "sum": sum(v), "max": max(v) if v else 0}
#:                   for r, v in sorted(by.items())}))
#: PY
UNCAPPED_EMBED_SOFT_LIMIT = 6_000


def record_skill_embed(session_id: str | None, *, route: str, skill: str,
                       embedded_chars: int, source_chars: int | None = None,
                       truncated: bool | None = None,
                       turn_id: str | None = None) -> dict[str, Any]:
    """Record one skill body entering a prompt by `route`; return the record.

    `embedded_chars` is what the prompt carries; `source_chars` is the file's own
    size when the caller has it, so a capped route shows what it left out.
    Without a session id only the log line is written.

    #2272 adds the accounting only. On the two uncapped routes, an embed over
    `UNCAPPED_EMBED_SOFT_LIMIT` also stamps `over_soft_limit: True` and
    `soft_limit: <the limit>` — the LIMIT, not `embedded_chars`, which is already on the
    row, so a reader can see the line the row was judged against. A route with a hard
    cut is never stamped: its number is bounded by code, and flagging it would bury the
    uncapped traffic these rows exist to count. Nothing truncates, refuses, or branches
    on the stamp.
    """
    record: dict[str, Any] = {"route": route, "skill": skill,
                              "embedded_chars": int(embedded_chars)}
    if source_chars is not None:
        record["source_chars"] = int(source_chars)
    if truncated is not None:
        record["truncated"] = bool(truncated)
    if route in UNCAPPED_ROUTES and embedded_chars > UNCAPPED_EMBED_SOFT_LIMIT:
        # Set-only. Nothing anywhere reads this back, and nothing may until the
        # cap-or-spill ruling is made (#2272 clause 5).
        record["over_soft_limit"] = True
        record["soft_limit"] = UNCAPPED_EMBED_SOFT_LIMIT
    try:
        logger.info("SKILL_EMBED route=%s skill=%s embedded=%dc source=%s session=%s",
                    route, skill, record["embedded_chars"],
                    f"{source_chars}c" if source_chars is not None else "-",
                    session_id or "-")
        if session_id:
            from app import event_log
            event_log.log_event(session_id, EVENT, record, turn_id=turn_id)
    except Exception as exc:  # noqa: BLE001 — a lost record is never an error
        logger.debug("skill_embed: record failed for %s: %s", session_id, exc)
    return record


def record_context_skills(session_id: str | None, context_text: str,
                          turn_id: str | None = None) -> list[dict[str, Any]]:
    """Record every skill a turn-start `<context>` block carried, with its size."""
    try:
        from app.harness.skill_dispatch import skill_delivery_sizes
        sizes = skill_delivery_sizes(context_text or "")
    except Exception as exc:  # noqa: BLE001
        logger.debug("skill_embed: could not read the context block: %s", exc)
        return []
    return [record_skill_embed(session_id, route=d["route"], skill=d["name"],
                               embedded_chars=d["chars"], truncated=d["truncated"],
                               turn_id=turn_id)
            for d in sizes]
