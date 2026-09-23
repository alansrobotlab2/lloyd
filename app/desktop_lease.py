"""Who may drive the desktop right now: the human, or Lloyd.

One seat, one pointer, one keyboard focus. When Lloyd clicks on Alan's real
Hyprland desktop the cursor moves and focus changes under Alan's hands, so
acting is a *lease* only a human hands out (Mission Control's Desktop tab,
``POST /api/desktop/lease``) and takes back — by the same toggle, by the
lease running out, or by simply moving the mouse (the tripwire in
``agent_mcp/desktop/guards.py``).

Rules, modelled on Hermes Agent's bot-desktop lease (``tools/bot_desktop/
lease.py``) and adapted to a shared seat:

* No file, an unreadable file, or an expired grant means **the human holds**.
  Hermes defaults to the agent on its private desktop; on the real seat the
  safe default is the opposite, and a corrupt file must fail closed.
* ``epoch`` increments on every grant and every revoke. The aggregator reads
  it before an action and again after; a result produced across a takeover
  is voided rather than reported.
* Nothing the agent can call grants a lease. The route is human-facing, and
  ``app/harness/safety.py`` refuses Bash that names it or this file.

Stdlib only: the backend writes it, the aggregator reads it, both import it.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from app.paths import DESKTOP_LEASE_PATH

MAX_TTL_MINUTES = 240


def _path() -> Path:
    return DESKTOP_LEASE_PATH


@contextmanager
def _locked() -> Iterator[None]:
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p.with_suffix(".lock"), "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _load() -> dict[str, Any]:
    try:
        data = json.loads(_path().read_text())
        if isinstance(data, dict):
            return data
    except FileNotFoundError:
        return {"holder": "human", "epoch": 0}
    except Exception:
        pass
    return {"holder": "human", "epoch": 0, "corrupt": True}


def _store(data: dict[str, Any]) -> None:
    p = _path()
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, p)


def read(now: float | None = None) -> dict[str, Any]:
    """The lease as it stands, with expiry applied (not written)."""
    now = time.time() if now is None else now
    data = _load()
    if data.get("holder") == "agent":
        exp = float(data.get("expires_at") or 0)
        if exp and now >= exp:
            data = dict(data, holder="human", expired=True)
    data["agent_holds"] = data.get("holder") == "agent"
    if data["agent_holds"] and data.get("expires_at"):
        data["remaining_s"] = max(0, int(float(data["expires_at"]) - now))
    return data


def grant(minutes: float, *, by: str = "mission-control",
          session_id: str = "") -> dict[str, Any]:
    minutes = max(1.0, min(float(minutes or 30), MAX_TTL_MINUTES))
    now = time.time()
    with _locked():
        cur = _load()
        data = {
            "holder": "agent",
            "granted_by": by,
            "session_id": session_id or "",
            "since": now,
            "expires_at": now + minutes * 60,
            "epoch": int(cur.get("epoch") or 0) + 1,
            "reason": "granted",
        }
        _store(data)
    return read(now)


def revoke(reason: str = "revoked", *, by: str = "mission-control") -> dict[str, Any]:
    now = time.time()
    with _locked():
        cur = _load()
        data = {
            "holder": "human",
            "epoch": int(cur.get("epoch") or 0) + 1,
            "since": now,
            "reason": reason,
            "revoked_by": by,
        }
        _store(data)
    return read(now)


def agent_may_act(session_id: str = "") -> tuple[bool, str, int]:
    """(allowed, reason, epoch). A grant pinned to one session covers only it."""
    data = read()
    epoch = int(data.get("epoch") or 0)
    if not data["agent_holds"]:
        if data.get("expired"):
            return False, "the desktop lease expired", epoch
        if data.get("corrupt"):
            return False, "the lease file is unreadable (fails closed)", epoch
        why = data.get("reason") or ""
        if why and why not in ("granted", "revoked"):
            return False, f"the human has control ({why})", epoch
        return False, "the human has control", epoch
    pinned = data.get("session_id") or ""
    if pinned and session_id and pinned != session_id:
        return False, "the lease was granted to a different session", epoch
    return True, "", epoch


def current_epoch() -> int:
    return int(read().get("epoch") or 0)


def note_seat(cursor: tuple[int, int] | None, window: str | None) -> None:
    """Record where Lloyd left the pointer and focus, for the tripwire."""
    with _locked():
        data = _load()
        if data.get("holder") != "agent":
            return
        if cursor is not None:
            data["last_cursor"] = [int(cursor[0]), int(cursor[1])]
        if window is not None:
            data["last_window"] = window
        data["last_action_at"] = time.time()
        _store(data)
