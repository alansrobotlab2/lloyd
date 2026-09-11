"""Dashboard endpoint — one aggregated snapshot of system + agent state.

`GET /api/dashboard` returns everything the Mission Control dashboard
renders. Deliberately one endpoint rather than eight: the page polls on a
short interval, and eight concurrent requests per tick (times however
many browser tabs are open) is a lot of load to put on a box whose whole
job is to hold a 262k-token KV cache steady.

Every section is gathered independently and degrades on its own. A
wedged supervisord or an unreachable MCP aggregator turns one panel into
an error string; it never blanks the page. That is the point of a
dashboard — it is most useful exactly when something is broken, so it
must not be the second thing to break.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app import host_metrics, sessions_io, vllm_metrics
from app.sessions_io import is_background_session_name, is_user_session
from app.backlog_status import CLOSED_STATUSES

logger = logging.getLogger("lloyd-server")

router = APIRouter()

# The aggregator's loopback state route. Same host, same machine — if
# this is slow the aggregator is wedged, and a short timeout keeps that
# from becoming the dashboard's problem too.
_MCP_STATE_URL = "http://127.0.0.1:8500/state"
_MCP_TIMEOUT_S = 2.0


async def _gather(name: str, coro) -> tuple[str, Any]:
    """Await one section, converting any failure into an `error` payload."""
    try:
        return name, await coro
    except Exception as exc:
        logger.warning("dashboard section %r failed: %s", name, exc)
        return name, {"error": f"{type(exc).__name__}: {exc}"}


async def _to_thread(fn, *args) -> Any:
    """Run a blocking call off the event loop."""
    return await asyncio.to_thread(fn, *args)


# Sections that walk the vault are cached: the backlog is 300+ markdown
# files and its status counts do not change between 2-second polls.
# Live sections (engines, host, running turns) are never cached — they
# are the whole point of the page.
_VAULT_SCAN_TTL_S = 10.0
_cache: dict[str, tuple[float, Any]] = {}


def _cached(key: str, ttl: float, fn) -> Any:
    hit = _cache.get(key)
    now = time.monotonic()
    if hit is not None and now - hit[0] < ttl:
        return hit[1]
    value = fn()
    _cache[key] = (now, value)
    return value


# The closing delimiter of a YAML front-matter block: `---` alone on its
# own line. Anchored, because `str.split("---")` also fires on a `---` in
# prose or inside a quoted activity-log entry, which truncates the block
# at the wrong place and yields either a parse error or, worse, a partial
# dict that looks fine.
_FM_END_RE = re.compile(r"^---[ \t]*$", re.M)
_FM_CHUNK = 4096


def _frontmatter(path, limit: int = 65536) -> dict:
    """First YAML block of a markdown file, or {} if there isn't one.

    `limit` bounds the **front matter**, not the prefix we are willing to
    look at, and the file is read a chunk at a time so the ordinary case
    still costs one small read. The distinction is not academic. At a flat
    3000-byte prefix this silently dropped five backlog items, and the
    selection was not random: an item accumulates `activity_log` entries
    precisely by being worked on, so the two it hid were the two that were
    `in_progress`. A board that loses whatever is most active is worse than
    no board. Anything past `limit` with no closing `---` is malformed
    rather than large, and still yields {}.
    """
    import yaml

    try:
        with open(path, encoding="utf-8") as f:
            head = f.read(_FM_CHUNK)
            if not head.startswith("---"):
                return {}
            while (m := _FM_END_RE.search(head, 3)) is None and len(head) < limit:
                chunk = f.read(_FM_CHUNK)
                if not chunk:
                    break
                head += chunk
    # A file that will not decode is not front matter either, and it must not
    # take the whole section down with it.
    except (OSError, UnicodeError):
        return {}
    if m is None:
        return {}
    opening = head.find("\n")
    if opening < 0 or opening > m.start():
        return {}
    try:
        fm = yaml.safe_load(head[opening + 1:m.start()])
    except Exception:
        return {}
    # A block that parses to a list or a bare string is not front matter;
    # returning it would blow up on the caller's first `.get`.
    return fm if isinstance(fm, dict) else {}


# ── Sections ───────────────────────────────────────────────────────────


async def _agent_state() -> dict[str, Any]:
    """Subagents + background bash tasks, read from the MCP aggregator.

    Both live in the lloyd-mcp process, not this one — see the docstring
    on `agent_mcp.main.state`.
    """
    import httpx

    async with httpx.AsyncClient() as client:
        resp = await client.get(_MCP_STATE_URL, timeout=_MCP_TIMEOUT_S)
        resp.raise_for_status()
        return resp.json()


async def _primary_state() -> dict[str, Any]:
    """What the primary agent loop is doing right now."""
    from app.config import CONFIG

    active = sessions_io.active_sessions_snapshot()
    # The snapshot is pure in-memory queue state and stays that way — it is
    # also the automod promoter's idle gate. Titles live on disk, so they
    # are joined on here, off the loop and behind a TTL cache.
    if active:
        from app import session_titles

        titles = await asyncio.to_thread(
            session_titles.titles_for, [s["session_id"] for s in active]
        )
        for entry in active:
            entry["title"] = titles.get(entry["session_id"], "")

    models = CONFIG.get("models") or {}
    default_alias = (CONFIG.get("model") or {}).get("default", "primary")
    default_cfg = models.get(default_alias) or {}

    running = [s for s in active if s["running"]]
    return {
        "model": default_alias,
        "base_url": default_cfg.get("base_url", ""),
        "context_length": default_cfg.get("context_length"),
        "max_turns": (CONFIG.get("agent") or {}).get("max_turns"),
        "permission_mode": (CONFIG.get("agent") or {}).get("permission_mode", ""),
        "preserve_thinking_iterations": (CONFIG.get("harness") or {}).get(
            "preserve_thinking_iterations"
        ),
        "sessions": active,
        "running_count": len(running),
        "queued_count": sum(
            s["pending_user"] + s["pending_ambient"] for s in active
        ),
        "busy": bool(running),
    }


# Recent chats come off disk, and a session JSON carries its whole
# transcript — 100 files and 7.5 MB across the tree today. Re-reading all
# of that on a 2-second poll is the exact cost `session_titles.title_for`
# caches around, so this scan is bounded twice: it opens only a window of
# the newest files, and the parse is cached for longer than the poll.
#
# The window is safe because a file's mtime is never *earlier* than its
# `last_active` — a background writer (the titler, post-session capture,
# TodoWrite) only ever pushes a session's mtime later than its last real
# message. Ordering by mtime can promote a stale session above its true
# position, but it can never demote a fresh one out of the window.
_RECENT_SHOWN = 2
_RECENT_KEPT = 8
#: How many files the walk may open before giving up. This replaced a flat
#: `[:24]` slice of the newest files by mtime, which was the whole budget when
#: every file in the directory was a chat. On 2026-09-10, before autonomy runs
#: were recorded at all, 22 of the newest 24 were already background, and
#: recording adds ~170 a day more against ~14 chats. The walk now
#: skips background ids by NAME without opening them and stops at
#: `_RECENT_KEPT` user rows or this ceiling — bounded either way, and no
#: longer bounded by the fleet's throughput.
_RECENT_CEILING = 400
_RECENT_TTL_S = 10.0


def _caption_rate(data: dict) -> dict[str, int] | None:
    """The newest turn's tool-caption rate, or None if no turn recorded one.

    Scans backwards for the most recent assistant `stats` carrying the
    counters, because the ratchet is per-turn: a session that captioned its
    first turn and lost the habit on its second should read as the second.
    Cheap — the file is already parsed for this row — and bounded by the first
    hit rather than by the transcript length.
    """
    for msg in reversed(data.get("messages") or []):
        stats = msg.get("stats")
        if isinstance(stats, dict) and stats.get("tool_calls_total"):
            return {"total": int(stats["tool_calls_total"]),
                    "captioned": int(stats.get("tool_calls_captioned") or 0)}
    return None


def _scan_recent_sessions() -> list[dict[str, Any]]:
    """Parse the newest session files into rows, newest conversation first."""
    import json
    from datetime import datetime, timezone

    from app.paths import SESSIONS_DIR

    # Shared with `GET /api/sessions` so the two lists cannot disagree
    # about what "recent" means. Its docstring is the reason this sorts on
    # `last_active` rather than on the mtime it selected candidates by.
    from app.routers.sessions import _last_active_ts

    try:
        paths = sorted(
            SESSIONS_DIR.glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return []

    rows: list[dict[str, Any]] = []
    opened = 0
    for path in paths:
        # The filename fast path. A background run's id has four parts and a
        # chat's has three, so most of the directory can be skipped without
        # opening it — which is what keeps this scan bounded now that ~240
        # background sessions a day land in the same directory. For a
        # four-part id the shape is the whole decision here: it is skipped
        # unread. That is safe only while nothing that creates a user session
        # mints one, which `tests/test_session_platform_checks.py` pins at the
        # creators. A three-part id is still judged by its `platform` below.
        if is_background_session_name(path.name):
            continue
        if len(rows) >= _RECENT_KEPT or opened >= _RECENT_CEILING:
            break
        opened += 1
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        # Scheduled tasks have their own panel, and so do worker jobs. They
        # are not chats. One definition (`sessions_io.is_user_session`) rather
        # than a literal per reader — this one had never learned about
        # `worker`, which was already 479 of the directory's 643 files on
        # 2026-09-10.
        if not is_user_session(data):
            continue
        goal = data.get("goal") if isinstance(data.get("goal"), dict) else {}
        ts = _last_active_ts(path, data)
        rows.append({
            "session_id": data.get("session_id") or path.stem,
            "title": (data.get("title") or "").strip(),
            "preview": (data.get("preview") or "")[:160],
            # Explicit UTC: `last_active` on disk is a naive *local* stamp,
            # and handing that to `new Date()` in a browser on any other
            # offset silently shifts every row.
            "last_active": datetime.fromtimestamp(ts, timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "message_count": data.get("message_count") or 0,
            "platform": data.get("platform", ""),
            "inner_voice": bool(data.get("inner_voice")),
            "model": data.get("model", ""),
            "goal": goal.get("text", ""),
            "goal_achieved": bool(goal.get("achieved_at")),
            "todo_counts": _count_todos(data.get("todos") or []),
            # None until a turn records it — an old session predates the
            # counters and must not render as 0/0, which reads as a failure.
            "captions": _caption_rate(data),
            "_ts": ts,
        })

    rows.sort(key=lambda row: row["_ts"], reverse=True)
    for row in rows:
        row.pop("_ts", None)
    return rows[:_RECENT_KEPT]


def _recent_sessions() -> dict[str, Any]:
    """The chats that most recently finished talking.

    "Finished" means *no turn running or queued right now*: a live session
    is already on the panel beside this one, and rendering it in both
    costs the operator the only thing this half is for — what happened
    just before now.

    The live filter is applied outside the cache deliberately. The scan is
    what is expensive; membership in the run queue is free and changes the
    instant a turn is enqueued, so caching it would leave a chat that just
    started reading as finished for up to ten seconds.
    """
    rows = _cached("recent_sessions", _RECENT_TTL_S, _scan_recent_sessions)
    live = {s["session_id"] for s in sessions_io.active_sessions_snapshot()}
    return {
        "sessions": [r for r in rows if r["session_id"] not in live][:_RECENT_SHOWN]
    }


def _count_todos(todos: list) -> dict[str, int]:
    counts = {"pending": 0, "in_progress": 0, "completed": 0}
    for t in todos:
        if isinstance(t, dict):
            status = t.get("status", "")
            if status in counts:
                counts[status] += 1
    return counts


def _services() -> dict[str, Any]:
    """supervisord health for infra + lloyd processes."""
    from app.supervisor_client import (
        _INFRA_SERVICES,
        _LLOYD_SERVICES,
        _health,
        _port_open,
        _sup_state,
        _supervisor_all_lenient,
    )

    procs = _supervisor_all_lenient()
    out: list[dict[str, Any]] = []
    for group, table in (("infra", _INFRA_SERVICES), ("lloyd", _LLOYD_SERVICES)):
        for sid, (name, port) in table.items():
            proc = procs.get(sid)
            active, sub = _sup_state(proc)
            port_healthy = _port_open(port) if port else None
            out.append({
                "id": sid,
                "name": name,
                "group": group,
                "port": port,
                "state": active,
                "sub_state": sub,
                "port_healthy": port_healthy,
                "health": _health(active, port_healthy),
                "uptime": (proc or {}).get("description"),
            })
    unhealthy = [s["id"] for s in out if s["health"] != "healthy"]
    return {"services": out, "unhealthy": unhealthy, "total": len(out)}


# Queue states that mean "not finished". `completed` dominates the depth
# table (3,400+ rows on this box) and would drown the live numbers.
_OPEN_STATES = ("queued", "claimed", "running", "pending", "ready")


def _workers() -> dict[str, Any]:
    """Worker pool, per-source depth, and what just ran."""
    from app.config import CONFIG
    from workers.pool import get_pool
    from workers.queue import get_queue

    q = get_queue()
    depth = q.depth_by_source() if hasattr(q, "depth_by_source") else {}

    by_state: dict[str, int] = {}
    for _src, states in (depth or {}).items():
        if isinstance(states, dict):
            for state, n in states.items():
                by_state[state] = by_state.get(state, 0) + int(n)

    pool = None
    try:
        p = get_pool()
        pool = p.status() if p else {"running": False}
    except Exception:
        pool = {"running": False}

    # One row per configured source, with its open (unfinished) backlog
    # split out from the lifetime completed count.
    sources_cfg = (CONFIG.get("workers") or {}).get("sources") or {}
    sources = []
    for name, cfg in sources_cfg.items():
        d = depth.get(name, {}) or {}
        sources.append({
            "name": name,
            "enabled": bool((cfg or {}).get("enabled", False)),
            "open": sum(int(d.get(st, 0)) for st in _OPEN_STATES),
            "running": int(d.get("running", 0)),
            "completed": int(d.get("completed", 0)),
            "failed": int(d.get("failed", 0)),
            "poisoned": int(d.get("poisoned", 0)),
        })
    sources.sort(key=lambda r: (-r["running"], -r["open"], r["name"]))

    runs = []
    try:
        for r in (q.list_runs(limit=6) or []):
            runs.append({
                "run_id": r.get("run_id", ""),
                "source": r.get("source", ""),
                "status": r.get("status", ""),
                "started_at": r.get("started_at", ""),
                "duration_seconds": r.get("duration_seconds"),
                "summary": (r.get("summary") or "")[:120],
            })
    except Exception:
        pass

    return {
        "enabled": bool((CONFIG.get("workers") or {}).get("enabled", False)),
        "duplicate_effects_suppressed": _duplicate_effects_suppressed(),
        "pool": pool,
        "depth_by_source": depth,
        "by_state": by_state,
        "open_total": sum(by_state.get(st, 0) for st in _OPEN_STATES),
        "poisoned_total": by_state.get("poisoned", 0),
        "sources": sources,
        "recent_runs": runs,
    }


def _duplicate_effects_suppressed() -> int | None:
    """How many duplicate side effects the #544 ledger has refused.

    Read off the ledger file, not from a process counter, because the process
    that writes the ledger is the aggregator and the process serving this
    endpoint is the backend — an in-memory count would read zero forever.
    None means unreadable, which is reported as such rather than as a clean
    zero: a guard whose counter silently reads 0 is a guard that looks like it
    never had to fire.
    """
    try:
        from agent_mcp._tool_effects import suppressed_total
        return suppressed_total()
    except Exception:
        return None


def _autonomy() -> dict[str, Any]:
    """Scheduled-task fleet: what is due, what is running, what is broken.

    Task definitions are markdown under ~/obsidian/autonomy/, so the file
    walk is TTL-cached; the in-flight list comes from the live pool and
    is not.
    """
    from datetime import datetime, timezone
    from pathlib import Path

    autonomy_dir = Path.home() / "obsidian" / "autonomy"

    def _scan() -> dict[str, Any]:
        # A missing directory is an empty fleet, not a different shape. This
        # used to return a four-key dict here, and the page reads `overdue`,
        # `held` and `classifier` unguarded — on 2026-09-10 the vault was
        # gone and that short form was the `Cannot read properties of
        # undefined` that blanked Mission Control on its landing tab. One
        # return, one shape, whatever is on disk.
        paths = list(autonomy_dir.glob("*.md")) if autonomy_dir.exists() else []
        by_status: dict[str, int] = {}
        upcoming: list[dict[str, Any]] = []
        failing: list[dict[str, Any]] = []
        scheduled: list[tuple[dict[str, Any], dict]] = []
        all_fm: list[dict] = []
        total = 0
        for path in paths:
            # Only NN-name.md task files — skip _config.md, reports, notes.
            if not path.name[:1].isdigit():
                continue
            fm = _frontmatter(path)
            if not fm:
                continue
            total += 1
            all_fm.append(fm)
            status = str(fm.get("status") or "draft")
            by_status[status] = by_status.get(status, 0) + 1
            row = {
                "name": str(fm.get("name") or path.stem),
                "status": status,
                "frequency": str(fm.get("frequency") or ""),
                "next_run": _iso(fm.get("next_run")),
                "last_run": _iso(fm.get("last_run")),
                "blocked": None,
            }
            if status == "failed":
                failing.append(row)
            elif row["next_run"]:
                upcoming.append(row)
                scheduled.append((row, fm))

        # Ask the scheduler, not the clock, why each past-due task has not
        # run. `blocked` stays None if the classification itself fails, which
        # lands the row in `overdue` — the direction that surfaces a task
        # rather than hiding one.
        classifier = "naive"
        try:
            import autonomy

            for row, fm in scheduled:
                try:
                    row["blocked"] = autonomy.hold_reason(fm, all_fm)
                except Exception:
                    row["blocked"] = None
            classifier = "autonomy"
        except Exception as exc:
            logger.warning("autonomy hold classification unavailable: %s", exc)

        # Split overdue from genuinely-upcoming. Sorting them together and
        # calling the result "next up" is how 23 tasks whose next_run is
        # months in the past read as a healthy schedule: the soonest-first
        # sort puts the most overdue at the top, labelled as if it were
        # the next thing to run.
        #
        # `held` is the same trap one level in. A paused task, or a nightly
        # one at midday, is past its next_run for most of every day; calling
        # that overdue keeps the counter permanently lit and buries the task
        # that really did miss its window.
        now_iso = datetime.now(timezone.utc).isoformat()
        past_due = [r for r in upcoming if (r["next_run"] or "") < now_iso]
        overdue = [r for r in past_due if not r["blocked"]]
        held = [r for r in past_due if r["blocked"]]
        pending = [r for r in upcoming if (r["next_run"] or "") >= now_iso]
        overdue.sort(key=lambda r: r["next_run"] or "")        # worst first
        held.sort(key=lambda r: r["next_run"] or "")           # worst first
        pending.sort(key=lambda r: r["next_run"] or "")        # soonest first
        return {
            "total": total,
            "by_status": by_status,
            "overdue": overdue[:5],
            "overdue_count": len(overdue),
            "held": held[:5],
            "held_count": len(held),
            "upcoming": pending[:5],
            "failing": failing[:5],
            "classifier": classifier,
        }

    out = _cached("autonomy", _VAULT_SCAN_TTL_S, _scan)

    # Live: which scheduled tasks the pool is actually executing now.
    running: list[dict[str, Any]] = []
    try:
        from workers.pool import get_pool

        pool = get_pool()
        status = pool.status() if pool else {}
        now = datetime.now(timezone.utc)
        for job_id, meta in (status.get("in_flight") or {}).items():
            if (meta or {}).get("source") != "scheduled-task":
                continue
            started = _iso(meta.get("started_at"))
            elapsed = None
            if started:
                try:
                    dt = datetime.fromisoformat(started)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    elapsed = (now - dt).total_seconds()
                except ValueError:
                    pass
            running.append({
                "job_id": str(job_id),
                "kind": (meta or {}).get("kind", ""),
                "started_at": started,
                "elapsed_s": elapsed,
            })
    except Exception:
        pass

    return {**out, "running": running, "running_count": len(running)}


def _backlog() -> dict[str, Any]:
    """Backlog board: open work by status and by board."""
    from pathlib import Path

    backlog_dir = Path.home() / "obsidian" / "backlog"

    def _scan() -> dict[str, Any]:
        # Same rule as `_autonomy`: a missing board is empty, not shaped
        # differently. The early return here dropped `recent_open`.
        paths = list(backlog_dir.glob("*.md")) if backlog_dir.exists() else []
        by_status: dict[str, int] = {}
        # board -> {open, total}
        boards: dict[str, dict[str, int]] = {}
        total = 0
        umbrellas = grouped = 0
        recent: list[dict[str, Any]] = []
        for path in paths:
            if not path.name[:1].isdigit():
                continue
            fm = _frontmatter(path)
            if not fm:
                continue
            total += 1
            status = str(fm.get("status") or "draft")
            board = str(fm.get("board") or "default")
            by_status[status] = by_status.get(status, 0) + 1
            entry = boards.setdefault(board, {"open": 0, "total": 0})
            entry["total"] += 1
            if status not in _BACKLOG_CLOSED:
                entry["open"] += 1
                if fm.get("members") or "umbrella" in (fm.get("tags") or []):
                    umbrellas += 1
                if fm.get("group") is not None:
                    grouped += 1
                try:
                    mtime = path.stat().st_mtime
                except OSError:
                    mtime = 0.0
                recent.append({
                    "name": str(fm.get("name") or path.stem),
                    "status": status,
                    "board": board,
                    "mtime": mtime,
                })
        recent.sort(key=lambda r: r["mtime"], reverse=True)
        return {
            "total": total,
            "by_status": by_status,
            "by_board": sorted(
                ({"board": b, **v} for b, v in boards.items()),
                key=lambda r: -r["open"],
            ),
            "open_total": sum(
                n for st, n in by_status.items() if st not in _BACKLOG_CLOSED
            ),
            "recent_open": recent[:5],
            "umbrellas": umbrellas,
            "grouped": grouped,
        }

    return _cached("backlog", _VAULT_SCAN_TTL_S, _scan)


# Statuses that take a task off the board, canonical and legacy.
_BACKLOG_CLOSED = CLOSED_STATUSES


# The scorecard reads the whole ledger, the backlog and a week of git log.
# Its numbers move per round, not per poll: a minute is already generous.
_SCORECARD_TTL_S = 60.0


def _automod() -> dict[str, Any]:
    """The unattended loop's scorecard for the last 7 days (scripts/automod/
    scorecard.py) plus its live state. `compute` is read-only and stdlib; a
    failure here is this section's error string and nothing else's."""
    from scripts.automod import scorecard, state as S

    def _scan() -> dict[str, Any]:
        row = scorecard.compute(since_days=7.0)
        current = S.read_current() or {}
        return {**row, "enabled": bool(S.is_enabled()),
                "current": {"round_id": current.get("round_id"), "state": current.get("state")},
                "halted": bool(S.is_halted()), "broken": bool(S.is_broken())}

    return _cached("automod", _SCORECARD_TTL_S, _scan)


def _iso(value: Any) -> str:
    """Frontmatter dates arrive as str or datetime depending on the writer."""
    from datetime import date, datetime

    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value) if value else ""


def _usage() -> dict[str, Any]:
    """Local token accounting from usage.db.

    This is Lloyd's own record of what it spent against the local
    engines — distinct from the vLLM counters, which are per-engine and
    reset on restart.
    """
    import usage_store

    return {
        "last_hour": usage_store.summary(hours=1),
        "last_24h": usage_store.summary(hours=24),
        "last_7d": usage_store.summary(days=7),
        "daily": usage_store.history_daily(days=7),
        "by_model_24h": usage_store.model_breakdown(hours=24),
        # Prefix-cache misses on long re-admissions (app/prefix_miss.py) —
        # the 09-09 stall's signature, counted from the turns' own usage rows.
        "prefix_misses_1h": usage_store.prefix_miss_summary(hours=1),
        "prefix_misses_24h": usage_store.prefix_miss_summary(hours=24),
    }


async def _vllm() -> list[dict[str, Any]]:
    """Every engine's snapshot, with the primary's KV pressure on its row.

    The card's live gauge is one 2-second reading, and what the 09-09 stall
    turned on was sustained pressure, which one reading cannot show. The p90
    comes from `engine_pressure`'s background ring, so it is there even when
    nobody had the page open for the last five minutes.
    """
    from app import engine_pressure

    engines = await vllm_metrics.collect(vllm_metrics.configured_engines())
    pressure = engine_pressure.snapshot()
    for row in engines:
        if row.get("alias") == pressure["alias"]:
            row["pressure"] = pressure
    return engines


# ── Endpoint ───────────────────────────────────────────────────────────


@router.get("/api/dashboard")
async def get_dashboard():
    """One snapshot: host, engines, primary agent, subagents, services."""
    sections = await asyncio.gather(
        _gather("host", host_metrics.collect()),
        _gather("vllm", _vllm()),
        _gather("primary", _primary_state()),
        _gather("recent", _to_thread(_recent_sessions)),
        _gather("agents", _agent_state()),
        _gather("services", _to_thread(_services)),
        _gather("workers", _to_thread(_workers)),
        _gather("autonomy", _to_thread(_autonomy)),
        _gather("backlog", _to_thread(_backlog)),
        _gather("automod", _to_thread(_automod)),
        _gather("usage", _to_thread(_usage)),
    )
    payload: dict[str, Any] = dict(sections)
    payload["timestamp"] = time.time()
    return JSONResponse(payload)
