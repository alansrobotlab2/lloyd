"""Inner Voice on/off A/B on the chat path (IV plan R3).

The claim that IV "stays on for chat, where its value was measured" had no
measurement behind it: `iv_grade` reports two proxies and says so, and there
has never been a comparison of chats with the observer against chats without
it. This is that comparison's assignment half; `scripts/iv_ab_report.py` is
the reading half.

Assignment is a pure function of the session id, so the arm of any session can
be recomputed later without trusting a stored field — and is ALSO stamped on
the session file (`inner_voice_ab`) at creation, because the flags it sets are
what every reader of "is IV on here" already consults (`_session_iv_flags`,
the Inner Voice tab, the session listings). One definition of on.

Only a brand-new chat-shaped session created inside the window is assigned,
from `sessions_io._save_session_meta`'s create branch. A session created with
explicit flags (the Inner Voice tab's "+ new chat", `/goal`) already has a file
by then and is never touched; neither is a worker session.

Config, `inner_voice.ab`:
    enabled: false           # ships off; the window is the switch
    name: iv-chat-1          # salts the hash, so a new experiment reshuffles
    start: 2026-09-25        # local date, inclusive
    end: 2026-10-09          # local date, exclusive
    fraction_on: 0.5
"""

from __future__ import annotations

import datetime as _dt
import hashlib
from typing import Any

ARMS = ("on", "off")


def ab_config() -> dict[str, Any]:
    try:
        from app.config import CONFIG

        cfg = dict(((CONFIG.get("inner_voice") or {}).get("ab")) or {})
    except Exception:  # noqa: BLE001 — no config is no experiment
        cfg = {}
    cfg.setdefault("enabled", False)
    cfg.setdefault("name", "iv-chat-1")
    cfg.setdefault("fraction_on", 0.5)
    return cfg


def arm_for(session_id: str, *, name: str, fraction_on: float = 0.5) -> str:
    """Deterministic: the same id and experiment name always get the same arm."""
    h = hashlib.sha256(f"{name}:{session_id}".encode()).hexdigest()
    return "on" if int(h[:8], 16) / 0x1_0000_0000 < float(fraction_on) else "off"


def _as_date(v: Any) -> _dt.date | None:
    if isinstance(v, _dt.datetime):
        return v.date()
    if isinstance(v, _dt.date):
        return v
    if isinstance(v, str) and v.strip():
        try:
            return _dt.date.fromisoformat(v.strip()[:10])
        except ValueError:
            return None
    return None


def window_open(cfg: dict[str, Any], today: _dt.date | None = None) -> bool:
    if not cfg.get("enabled"):
        return False
    start, end = _as_date(cfg.get("start")), _as_date(cfg.get("end"))
    if start is None or end is None:
        return False
    today = today or _dt.date.today()
    return start <= today < end


def assignment(session_id: str, *, today: _dt.date | None = None,
               cfg: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """The fields a new chat session is created with, or None outside a window."""
    cfg = ab_config() if cfg is None else cfg
    if not session_id or not window_open(cfg, today):
        return None
    name = str(cfg.get("name") or "iv-chat-1")
    arm = arm_for(session_id, name=name, fraction_on=float(cfg.get("fraction_on", 0.5)))
    on = arm == "on"
    return {
        "inner_voice": on,
        "inner_voice_evaluate_user_turns": on,
        "inner_voice_ab": {"experiment": name, "arm": arm,
                           "assigned_on": (today or _dt.date.today()).isoformat()},
    }
