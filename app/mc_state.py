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
    "dashboard",
    "inner_voice", "chat", "backlog", "autonomy", "workers",
    "memory", "architecture", "skills", "tools", "services",
    "settings", "graph", "ide",
}

_STATE_PATH = LLOYD_HOME / "mc-state.json"

_state_lock = asyncio.Lock()
_state: dict[str, Any] = {
    "tab": "inner_voice",
    "focus_by_tab": {},
    # IDE tab mirror — frontend reports {open_folder, visible_file, open_tabs}.
    # None until the user has opened the IDE tab at least once.
    "ide": None,
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
            ide = data.get("ide")
            if isinstance(ide, dict):
                _state["ide"] = _normalize_ide(ide)
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


def _normalize_ide(ide: Any) -> Optional[dict]:
    """Coerce client-supplied IDE state into the canonical shape.

    Returns None on garbage. Drops unknown keys; coerces tab list entries
    to strings; caps the tab list at 32 entries to keep the state compact.
    """
    if not isinstance(ide, dict):
        return None
    open_folder = ide.get("open_folder")
    visible_file = ide.get("visible_file")
    open_tabs = ide.get("open_tabs")
    out: dict[str, Any] = {}
    if isinstance(open_folder, str) and open_folder.strip():
        out["open_folder"] = open_folder
    if isinstance(visible_file, str) and visible_file.strip():
        out["visible_file"] = visible_file
    if isinstance(open_tabs, list):
        clean = [str(t) for t in open_tabs if isinstance(t, str) and t.strip()]
        out["open_tabs"] = clean[:32]
    if not out:
        return None
    return out


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
            "ide": dict(_state["ide"]) if _state.get("ide") else None,
            "last_updated": _state["last_updated"],
        }


def get_ide_snapshot() -> Optional[dict]:
    """Synchronous snapshot of just the IDE block.

    Used by prefetch context injection where the call site is sync and
    can't await get_state(). Reads without the lock — small TOCTOU window
    but the worst case is a one-turn-stale read, harmless.
    """
    ide = _state.get("ide")
    return dict(ide) if isinstance(ide, dict) else None


def get_focus_snapshot() -> dict:
    """Synchronous snapshot of the tab + per-tab focus map.

    Same trade as `get_ide_snapshot`: read without the lock so sync call
    sites (the dashboard aggregator runs its disk reads in a thread) can
    use it. Worst case is a one-poll-stale tab name.
    """
    return {
        "tab": _state.get("tab", ""),
        "focus_by_tab": dict(_state.get("focus_by_tab") or {}),
    }


_SENTINEL = object()


async def set_state(tab: Optional[str], focus: Any, ide: Any = _SENTINEL) -> dict:
    """Update the mirror from a frontend report.

    `tab` may be None (only updating focus); `focus` may be None (clearing
    the active tab's focus). `ide` is sentinel-defaulted: omitted means
    "leave IDE state unchanged"; an explicit None clears it; a dict updates
    it. Validates tab against VALID_TABS.
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

        prev_folder = (_state.get("ide") or {}).get("open_folder") if _state.get("ide") else None
        if ide is not _SENTINEL:
            if ide is None:
                _state["ide"] = None
            else:
                _state["ide"] = _normalize_ide(ide)

        new_folder = (_state.get("ide") or {}).get("open_folder") if _state.get("ide") else None
        # Rebind the inotify watcher whenever the open folder changes.
        # Lazy-imported to avoid bringing watchdog into modules that don't
        # need it during startup.
        if new_folder != prev_folder:
            try:
                from app import file_watcher
                file_watcher.bind(new_folder)
            except Exception as e:
                logger.warning("mc_state: file_watcher.bind failed: %s", e)

        _state["last_updated"] = _now()
        _persist()

        return {
            "tab": _state["tab"],
            "focus": _state["focus_by_tab"].get(_state["tab"]),
            "ide": dict(_state["ide"]) if _state.get("ide") else None,
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
    _fanout(payload)


VALID_IDE_ACTIONS = {"open_folder", "close_tab"}


async def publish_file_changed(path: str, *, deleted: bool = False) -> None:
    """Push a file-changed signal to every subscribed client.

    Fired by the inotify watcher when a file inside the IDE's open
    folder is modified, created, deleted, or renamed. The frontend
    decides what to do (silent reload, animate, conflict banner)
    based on whether it has the path open and whether the tab is dirty.
    """
    payload = {"type": "file_changed", "path": path, "deleted": bool(deleted)}
    _fanout(payload)


async def publish_close_modal(tab: str) -> None:
    """Push a close-modal command to every subscribed client.

    Counterpart to publish_navigate: where navigate(tab, focus_id) tells a
    page to open a modal/popup, close_modal(tab) tells it to dismiss
    whatever modal it currently has open. Pages that own no modals (e.g.
    workers, settings) ignore the event.
    """
    if tab not in VALID_TABS:
        raise ValueError(f"unknown tab: {tab!r}")
    payload = {"type": "close_modal", "tab": tab}
    _fanout(payload)


async def publish_ide_action(kind: str, path: str) -> None:
    """Push an IDE action (open_folder / close_tab) to every subscribed client.

    `ide_open_file` is intentionally NOT here — it reuses publish_navigate
    with tab="ide", focus_id=<path>, which the existing pendingFocus channel
    already handles. This bus is for richer IDE drives that don't fit a
    single focus_id.
    """
    if kind not in VALID_IDE_ACTIONS:
        raise ValueError(f"unknown ide action kind: {kind!r}")
    payload = {"type": "ide_action", "kind": kind, "path": path}
    _fanout(payload)


def _fanout(payload: dict) -> None:
    dead: list[asyncio.Queue] = []
    for q in list(_subscribers):
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        _subscribers.discard(q)


_load_from_disk()
