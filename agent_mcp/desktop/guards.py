"""Hard refusals for desktop input, checked before the lease and before anything moves.

Two families, both after Hermes Agent's ``tools/computer_use/tool.py`` (MIT):

* **Key combos** that end the session or take the machine away: lock, log
  out, power, the system menu. Canonicalised on ``+`` *and* ``-`` so
  ``ctrl-alt-delete`` cannot slip past a ``+`` table, aliases folded
  (``control``→``ctrl``, ``win``/``cmd``/``meta``→``super``), matched as a
  subset so extra modifiers do not help. The static table is joined at runtime
  by every Hyprland bind whose own description says lock/power/system/log out,
  read from ``hyprctl binds -j`` — omarchy's binds are the authority for what
  this desktop does, and a hand-copied list would drift from them.
* **Typed text**: Hermes's shell-shaped patterns for every target, and, when the
  focused window is a terminal, the same ``safety.check_bash_command`` Bash
  itself passes through. Typing ``rm -rf ~/obsidian`` into ``foot`` is the
  Bash tool by another name, so it gets the Bash tool's guards — vault,
  tree, sync registration, service control — and nothing weaker.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from typing import Iterable

KEY_ALIASES = {
    "control": "ctrl", "ctl": "ctrl", "option": "alt", "opt": "alt",
    "win": "super", "windows": "super", "cmd": "super", "command": "super",
    "meta": "super", "logo": "super", "mod4": "super", "⌘": "super", "⌥": "alt",
    "del": "delete", "esc": "escape", "return": "enter", "bksp": "backspace",
}
MODIFIERS = {"ctrl", "alt", "shift", "super"}

STATIC_BLOCKED: list[frozenset[str]] = [
    frozenset({"ctrl", "alt", "delete"}),
    frozenset({"ctrl", "alt", "backspace"}),   # X server kill
    frozenset({"super", "l"}),
    frozenset({"ctrl", "super", "l"}),         # omarchy: lock system
    frozenset({"ctrl", "super", "p"}),         # omarchy: power
    frozenset({"super", "escape"}),            # omarchy: system menu
    frozenset({"ctrl", "super", "i"}),         # omarchy: toggle idle lock
    frozenset({"xf86poweroff"}),
    frozenset({"alt", "f4"}),
]
for n in range(1, 13):                          # VT switches leave the session
    STATIC_BLOCKED.append(frozenset({"ctrl", "alt", f"f{n}"}))

_DANGEROUS_BIND = re.compile(
    r"\b(lock|power|system menu|log ?out|logout|exit hyprland|quit hyprland|"
    r"shut ?down|reboot|suspend|hibernate|close all)\b", re.I)
_BIND_MODS = {1: "shift", 4: "ctrl", 8: "alt", 64: "super"}

BLOCKED_TYPE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in (
    r"curl\s+[^|]*\|\s*(ba|z)?sh\b",
    r"wget\s+[^|]*\|\s*(ba|z)?sh\b",
    r"\bsudo\s+rm\s+-[rf]",
    r"\brm\s+-rf\s+/\s*$",
    r":\s*\(\)\s*\{\s*:\|:\s*&\s*\}",
    r"\bmkfs(\.\w+)?\s",
    r"\bdd\s+[^\n]*\bof=/dev/",
)]

TERMINAL_CLASSES = {
    "foot", "footclient", "alacritty", "kitty", "ghostty", "com.mitchellh.ghostty",
    "org.wezfurlong.wezterm", "wezterm", "xterm", "org.gnome.console",
    "org.gnome.terminal", "konsole", "terminator", "tilix",
}

_bind_cache: tuple[float, list[frozenset[str]]] = (0.0, [])


def canonical_keys(keys: str | Iterable[str]) -> list[str]:
    if isinstance(keys, str):
        raw = re.split(r"\s*[+]\s*|(?<=\w)-(?=\w)", keys.strip())
    else:
        raw = list(keys)
    out = []
    for k in raw:
        k = str(k).strip().lower()
        if not k:
            continue
        out.append(KEY_ALIASES.get(k, k))
    return out


def hyprland_dangerous_binds() -> list[frozenset[str]]:
    """Combos omarchy binds to lock/power/system actions. Cached 5 minutes."""
    global _bind_cache
    now = time.monotonic()
    if now - _bind_cache[0] < 300 and _bind_cache[1]:
        return _bind_cache[1]
    combos: list[frozenset[str]] = []
    try:
        out = subprocess.run(["hyprctl", "binds", "-j"], capture_output=True,
                             text=True, timeout=3).stdout
        for b in json.loads(out or "[]"):
            if not _DANGEROUS_BIND.search(str(b.get("description") or "")):
                continue
            mods = {v for bit, v in _BIND_MODS.items() if int(b.get("modmask") or 0) & bit}
            key = str(b.get("key") or "").lower()
            if key:
                combos.append(frozenset(mods | {KEY_ALIASES.get(key, key)}))
    except Exception:
        pass
    _bind_cache = (now, combos)
    return combos


def blocked_combo(keys: str | Iterable[str], modifiers: Iterable[str] = ()) -> str | None:
    """The reason a combo is refused, or None."""
    parts = set(canonical_keys(keys)) | set(canonical_keys(list(modifiers)))
    if not parts:
        return None
    for combo in STATIC_BLOCKED + hyprland_dangerous_binds():
        if combo and combo <= parts:
            return f"blocked key combo: {'+'.join(sorted(combo))}"
    return None


def blocked_text(text: str, *, window_class: str = "", session_id: str = "") -> str | None:
    """The reason typed text is refused, or None."""
    if not text:
        return None
    for pat in BLOCKED_TYPE_PATTERNS:
        if pat.search(text):
            return f"blocked pattern in typed text: {pat.pattern!r}"
    if window_class.lower() in TERMINAL_CLASSES:
        try:
            from app.harness.safety import check_bash_command
            match = check_bash_command(text, None, at_dispatch=True,
                                       session_id=session_id)
        except Exception as exc:   # pragma: no cover - fail closed
            return f"could not check typed shell text ({exc}); refusing"
        if match:
            label, excerpt = match
            return (f"typing into a terminal is Bash by another route — "
                    f"harness safety blocked {label!r} on {excerpt!r}")
    return None


def denied_window(window: dict, deny: Iterable[str]) -> str | None:
    """A window this module must never look at or touch."""
    hay = f"{window.get('class', '')} {window.get('title', '')}".lower()
    for d in deny:
        d = str(d).strip().lower()
        if d and d in hay:
            return f"window matches desktop.deny_windows entry {d!r}"
    return None
