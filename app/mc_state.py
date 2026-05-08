"""Mission Control UI state mirror + navigate event bus.

Holds the frontend's reported {tab, focus_by_tab} so MCP tools can read
what the user is currently looking at. Pushes navigate commands back to
subscribed frontend clients via an asyncio fan-out.

Persists to ~/lloyd/mc-state.json so state survives backend restarts —
the bus itself is in-memory only (subscribers reattach on reconnect).
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.paths import LLOYD_HOME

logger = logging.getLogger("lloyd-server")

VALID_TABS = {
    "inner_voice", "chat", "backlog", "autonomy", "workers",
    "memory", "architecture", "skills", "tools", "services",
    "settings", "graph",
}

_STATE_PATH = LLOYD_HOME / "mc-state.json"

_state_lock = asyncio.Lock()
_state: dict[str, Any] = {
    "tab": "inner_voice",
    "focus_by_tab": {},
    "last_updated": None,
}

_subscribers: set[asyncio.Queue] = set()


def _load_from_disk() -> None:
    """Hydrate the in-memory mirror from disk on import.

    Bad/missing files leave defaults intact — no exception escapes module
    import.
    """
    if not _STATE_PATH.exists():
        return
    try:
        data = json.loads(_STATE_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            tab = data.get("tab")
            if tab in VALID_TABS:
                _state["tab"] = tab
            focus = data.get("focus_by_tab")
            if isinstance(focus, dict):
                _state["focus_by_tab"] = {
                    k: v for k, v in focus.items()
                    if k in VALID_TABS and isinstance(v, dict)
                }
            _state["last_updated"] = data.get("last_updated")
    except Exception as e:
        logger.warning("mc_state: failed to load %s: %s", _STATE_PATH, e)


def _persist() -> None:
    try:
        _STATE_PATH.write_text(
            json.dumps(_state, indent=2, default=str), encoding="utf-8"
        )
    except Exception as e:
        logger.warning("mc_state: failed to persist %s: %s", _STATE_PATH, e)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_focus(focus: Any) -> Optional[dict]:
    """Coerce client-supplied focus into the canonical {kind, id, label?} shape.

    Returns None when the client explicitly cleared focus or sent garbage.
    """
    if not isinstance(focus, dict):
        return None
    kind = focus.get("kind")
    fid = focus.get("id")
    if not kind or fid in (None, ""):
        return None
    out: dict[str, Any] = {"kind": str(kind), "id": str(fid)}
    if "label" in focus and focus["label"]:
        out["label"] = str(focus["label"])
    return out


async def get_state() -> dict:
    """Snapshot of the current mirrored state."""
    async with _state_lock:
        focus_for_tab = _state["focus_by_tab"].get(_state["tab"])
        return {
            "tab": _state["tab"],
            "focus": focus_for_tab,
            "focus_by_tab": dict(_state["focus_by_tab"]),
            "last_updated": _state["last_updated"],
        }


async def set_state(tab: Optional[str], focus: Any) -> dict:
    """Update the mirror from a frontend report.

    `tab` may be None (only updating focus); `focus` may be None (clearing
    the active tab's focus). Validates tab against VALID_TABS.
    """
    async with _state_lock:
        if tab is not None:
            if tab not in VALID_TABS:
                raise ValueError(f"unknown tab: {tab!r}")
            _state["tab"] = tab

        active = _state["tab"]
        normalized = _normalize_focus(focus)
        if normalized is None:
            _state["focus_by_tab"].pop(active, None)
        else:
            _state["focus_by_tab"][active] = normalized

        _state["last_updated"] = _now()
        _persist()

        return {
            "tab": _state["tab"],
            "focus": _state["focus_by_tab"].get(_state["tab"]),
            "last_updated": _state["last_updated"],
        }


def subscribe() -> asyncio.Queue:
    """Register a navigate-event subscriber; caller must call unsubscribe()."""
    q: asyncio.Queue = asyncio.Queue(maxsize=64)
    _subscribers.add(q)
    return q


def unsubscribe(q: asyncio.Queue) -> None:
    _subscribers.discard(q)


async def publish_navigate(tab: str, focus_id: Optional[str]) -> None:
    """Push a navigate command to every subscribed client.

    Slow/dead subscribers are dropped rather than blocking the publisher.
    """
    if tab not in VALID_TABS:
        raise ValueError(f"unknown tab: {tab!r}")
    payload = {"type": "navigate", "tab": tab, "focus_id": focus_id}
    dead: list[asyncio.Queue] = []
    for q in list(_subscribers):
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        _subscribers.discard(q)


_load_from_disk()
