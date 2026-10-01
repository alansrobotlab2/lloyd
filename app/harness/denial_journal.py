"""One durable row per refusal, from every guard, in one file.

Before this module a refusal's only durable witness was the refusing guard's
own choice: the grant gate writes a `grant_dispatch` row, install provenance
appends to `supply-chain/provenance.jsonl`, and every other guard — the
hard-deny table, the protected-path deny-sets, the sync-registration and
service-control checks, the read-only bench sandbox, the outbound-content
gate, the desktop and sessionless refusals — emitted one `logger.warning`
into `server.err`, which with its ten rotations spans about 2.6 days. "How
often does guard X refuse, and in which session class?" had no answer older
than that, and a question about a threshold or a false-positive rate had no
corpus. OpenAPPA keeps every decision as a fact in its trajectory log beside
the hash of the policy it was made under; this is the same idea at the size
this box needs: an append-only JSONL file under the data root, one row per
refusal, carrying the guard, where it fired, the session's class, the tool,
the reason, and the commit the guard was running.

Two call sites cover every guard, so a new guard is journaled without
knowing this module exists:

* `HookRegistry.fire_pre_tool_use` (`app/harness/hooks.py`) — every
  PreToolUse deny, including a fail-closed gate that raised.
* `agent_mcp.main._refused_call` — every refusal the aggregator makes at
  dispatch, which is where a call runs whether or not a hook was installed.

The two Write-lane handlers (`builtin_fs`, `vault`) refuse through their own
error returns and call `record` themselves.

Fail-open, like the provenance journal and for the same reason: a journal
that can turn its own failure into a refusal, or delay a dispatch, is worse
than no journal. `record` never raises and never blocks on anything but one
local append. A lost row costs one warning line; the refusal it describes
already happened and already reached the model.

Not a decision input. Nothing reads this file on the dispatch path. Its
readers are the scorecard (row 16, "guard denials"), the dashboard, and
whoever asks the question above.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("lloyd-harness-denial-journal")

JOURNAL_NAME = "denials.jsonl"
#: Environment override, for tests and for a canary boot. Honoured by the
#: scorecard too, so the one variable moves both the writer and the reader.
JOURNAL_ENV = "LLOYD_DENIAL_JOURNAL"

#: Where a refusal was made. `hook` is a PreToolUse deny in the harness;
#: `dispatch` is the aggregator refusing at `call_tool`, which every call
#: passes whether or not a hook was installed.
WHERE = ("hook", "dispatch")

#: Session classes, from the id alone. `bench` is a sandboxed eval/bench id;
#: `subagent` a `task:*` child; `background` a worker/autonomy id;
#: `chat` anything else with an id; `none` an empty id.
CLASSES = ("chat", "background", "bench", "subagent", "none")

#: Hook callback qualname → guard name. A callback not listed is journaled
#: under its own qualname, so an unlisted gate is still counted, under a
#: name a reader can grep for.
_HOOK_GUARDS: dict[str, str] = {
    "_safety_pretool_cb": "safety",
    "_policy_pretool_cb": "grant",
    "_content_pretool_cb": "outbound_content",
    "_bench_corpus_pretool_cb": "bench_corpus",
    "write_scope_denial": "trial_write_scope",
}

_SAFETY_LABEL = re.compile(r"harness safety: blocked '([^']*)'(?: on '(.*)')?")

_commit_cache: str | None = None


def journal_path() -> Path:
    """Where the journal lives. Env override first, then the data root."""
    override = os.environ.get(JOURNAL_ENV, "").strip()
    if override:
        return Path(override)
    try:
        from app.paths import DENIAL_JOURNAL_PATH
        return Path(DENIAL_JOURNAL_PATH)
    except Exception:  # noqa: BLE001 — same fail-open as the provenance journal
        return Path(os.path.expanduser(f"~/.cache/lloyd/safety/{JOURNAL_NAME}"))


def session_class(session_id: str | None, *,
                  parent_of: Callable[[str], str | None] | None = None) -> str:
    """Classify a session by its id. Never raises."""
    sid = str(session_id or "")
    if not sid:
        return "none"
    try:
        if sid.startswith("task:"):
            return "subagent"
        try:
            from agent_mcp._tool_sandbox import is_sandboxed_session
            sandboxed = is_sandboxed_session(sid)
        except Exception:  # noqa: BLE001 — the harness may run without the aggregator package
            sandboxed = sid.startswith(("bench_", "pt-eval-"))
        if sandboxed:
            return "bench"
        from app.harness.service_control import is_background_session
        if is_background_session(sid, parent_of=parent_of):
            return "background"
    except Exception:  # noqa: BLE001
        pass
    return "chat"


def guard_for_hook(callback: Any) -> str:
    """The guard name a PreToolUse callback is journaled under."""
    name = getattr(callback, "__qualname__", None) or getattr(callback, "__name__", None) \
        or repr(callback)
    short = str(name).rsplit(".", 1)[-1]
    return _HOOK_GUARDS.get(short, str(name))


def label_of(reason: str) -> str:
    """The pattern label inside a safety refusal, else ''."""
    m = _SAFETY_LABEL.search(str(reason or ""))
    return m.group(1) if m else ""


def _commit() -> str:
    global _commit_cache
    if _commit_cache is None:
        try:
            from app.gitinfo import head_commit
            from app.paths import LLOYD_HOME
            _commit_cache = str(head_commit(LLOYD_HOME) or "")
        except Exception:  # noqa: BLE001
            _commit_cache = ""
    return _commit_cache


def record(*, guard: str, where: str, session_id: str | None, tool: str,
           reason: str, excerpt: str = "", label: str = "",
           parent_of: Callable[[str], str | None] | None = None,
           path: Path | None = None) -> bool:
    """Append one row. Returns True when the row was written. Never raises."""
    try:
        target = path if path is not None else journal_path()
        reason_text = str(reason or "")
        row = {
            "at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "guard": str(guard or "unknown"),
            "where": where if where in WHERE else "dispatch",
            "session": str(session_id) if session_id else None,
            "session_class": session_class(session_id, parent_of=parent_of),
            "tool": str(tool or ""),
            "label": str(label or label_of(reason_text)),
            "reason": reason_text[:400],
            "excerpt": str(excerpt or "")[:200],
            "commit": _commit(),
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        return True
    except Exception as exc:  # noqa: BLE001 — the guard outlives its own journal
        logger.warning("denial journal write failed (%s: %s); the refusal stands and "
                       "only this row is lost", type(exc).__name__, str(exc)[:160])
        return False


def record_hook_deny(callback: Any, *, session_id: str | None, tool: str,
                     deny: dict[str, Any]) -> bool:
    """Journal a PreToolUse deny dict as the hook registry returns it."""
    try:
        hso = (deny or {}).get("hookSpecificOutput") or {}
        reason = str(hso.get("permissionDecisionReason") or "")
    except Exception:  # noqa: BLE001
        reason = ""
    return record(guard=guard_for_hook(callback), where="hook", session_id=session_id,
                  tool=tool, reason=reason)


def rows(path: Path | None = None, *, since: float | None = None) -> list[dict[str, Any]]:
    """Every parseable row, oldest first; `since` is a UTC epoch floor."""
    target = path if path is not None else journal_path()
    out: list[dict[str, Any]] = []
    try:
        with target.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if since is not None and _epoch(row.get("at")) < since:
                    continue
                out.append(row)
    except FileNotFoundError:
        return []
    except Exception as exc:  # noqa: BLE001
        logger.warning("denial journal unreadable: %s", exc)
    return out


def _epoch(stamp: Any) -> float:
    try:
        return dt.datetime.fromisoformat(str(stamp)).timestamp()
    except Exception:  # noqa: BLE001
        return 0.0


def summarize(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts by guard, by session class, by `where`, and guard × class."""
    by_guard: dict[str, int] = {}
    by_class: dict[str, int] = {}
    by_where: dict[str, int] = {}
    cross: dict[str, dict[str, int]] = {}
    for r in entries:
        g = str(r.get("guard") or "unknown")
        c = str(r.get("session_class") or "none")
        w = str(r.get("where") or "dispatch")
        by_guard[g] = by_guard.get(g, 0) + 1
        by_class[c] = by_class.get(c, 0) + 1
        by_where[w] = by_where.get(w, 0) + 1
        cross.setdefault(g, {})[c] = cross.setdefault(g, {}).get(c, 0) + 1
    return {"total": len(entries), "by_guard": by_guard, "by_class": by_class,
            "by_where": by_where, "guard_by_class": cross}
