"""The real seat: Hyprland, driven with its own tools.

Hermes Agent drives Linux through cua-driver, whose supported path is X11 /
XWayland (XTEST + AT-SPI) and whose Linux validation matrix does not list
Hyprland. Every client on this desktop is native Wayland, so this backend
uses what the compositor itself offers:

* windows, focus, cursor position: ``hyprctl -j`` (clients, activewindow,
  monitors, cursorpos) and ``hyprctl dispatch`` (focuswindow, movecursor,
  sendshortcut — the last delivers a hotkey to a window WITHOUT focusing it,
  the one true background input path Hyprland has);
* pixels: ``grim -T <stable_id>`` (ext-image-copy-capture) for a window —
  it reads the toplevel itself, which is what makes an occluded window or one
  on another workspace capturable; ``grim -g`` over the layout rectangle only
  for ``scope="screen"`` and for a visible window that has no toplevel id,
  since a region grab reads whatever is on screen there, occluder included.
  Either is scaled by ``grim -s`` so the longest edge fits ``max_dimension``
  (no Pillow needed);
* text and keys: ``wtype`` (virtual-keyboard protocol, to the focused window);
* elements and accessibility actions, plus a uinput pointer for buttons and
  wheels: the helper process (``agent-services/bin/lloyd-desktop-helper.py``).

Coordinates, one rule: every coordinate the model sees or sends is in
SCREENSHOT pixels of the capture it came from. ``Capture.to_screen`` maps
back to layout pixels with that capture's own origin and scale.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("lloyd-desktop")

HELPER = Path(__file__).resolve().parents[2] / "agent-services" / "bin" / "lloyd-desktop-helper.py"

WTYPE_KEYS = {
    "enter": "Return", "return": "Return", "escape": "Escape", "tab": "Tab",
    "backspace": "BackSpace", "delete": "Delete", "space": "space",
    "up": "Up", "down": "Down", "left": "Left", "right": "Right",
    "home": "Home", "end": "End", "pageup": "Page_Up", "page_up": "Page_Up",
    "pagedown": "Page_Down", "page_down": "Page_Down", "insert": "Insert",
    "menu": "Menu", "capslock": "Caps_Lock", "printscreen": "Print",
    "plus": "plus", "minus": "minus", "comma": "comma", "period": "period",
    "slash": "slash",
}
for _n in range(1, 25):
    WTYPE_KEYS[f"f{_n}"] = f"F{_n}"
WTYPE_MODS = {"ctrl": "ctrl", "alt": "alt", "shift": "shift", "super": "logo"}
HYPR_MODS = {"ctrl": "CTRL", "alt": "ALT", "shift": "SHIFT", "super": "SUPER"}


class DesktopError(Exception):
    def __init__(self, message: str, code: str = "error") -> None:
        super().__init__(message)
        self.code = code


async def _run(argv: list[str], *, timeout: float = 5.0,
               stdin: bytes | None = None) -> tuple[int, bytes, str]:
    proc = await asyncio.create_subprocess_exec(
        *argv, stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(proc.communicate(stdin), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise DesktopError(f"{argv[0]} timed out after {timeout}s", "timeout")
    return proc.returncode or 0, out, err.decode("utf-8", "replace").strip()


async def hyprctl_json(what: str) -> Any:
    rc, out, err = await _run(["hyprctl", what, "-j"])
    if rc != 0:
        raise DesktopError(f"hyprctl {what}: {err or rc}", "compositor")
    return json.loads(out or b"null")


async def dispatch(*args: str) -> str:
    rc, out, err = await _run(["hyprctl", "dispatch", *args])
    text = out.decode("utf-8", "replace").strip()
    if rc != 0 or (text and text.lower() not in ("ok",)):
        raise DesktopError(f"hyprctl dispatch {' '.join(args)}: {text or err}", "compositor")
    return text


# ─────────────────────────────────────────────────────────────────────────
# Windows
# ─────────────────────────────────────────────────────────────────────────

def _win(c: dict, active_addr: str, visible_ws: set[int]) -> dict:
    ws = c.get("workspace") or {}
    return {
        "address": c.get("address", ""),
        "stable_id": str(c.get("stableId") or ""),
        "class": c.get("class", "") or c.get("initialClass", ""),
        "title": c.get("title", ""),
        "pid": int(c.get("pid") or 0),
        "workspace": ws.get("name") or str(ws.get("id", "")),
        "workspace_id": int(ws.get("id") or 0),
        "x": int((c.get("at") or [0, 0])[0]), "y": int((c.get("at") or [0, 0])[1]),
        "width": int((c.get("size") or [0, 0])[0]), "height": int((c.get("size") or [0, 0])[1]),
        "focused": c.get("address") == active_addr,
        "visible": int(ws.get("id") or 0) in visible_ws and bool(c.get("mapped", True))
                   and not c.get("hidden", False),
        "floating": bool(c.get("floating")),
        "fullscreen": bool(c.get("fullscreen")),
        "xwayland": bool(c.get("xwayland")),
    }


async def windows() -> list[dict]:
    clients, active, monitors = await asyncio.gather(
        hyprctl_json("clients"), hyprctl_json("activewindow"), hyprctl_json("monitors"))
    active_addr = (active or {}).get("address", "") if isinstance(active, dict) else ""
    visible_ws: set[int] = set()
    for m in monitors or []:
        aw = (m.get("activeWorkspace") or {}).get("id")
        if aw is not None:
            visible_ws.add(int(aw))
        sw = (m.get("specialWorkspace") or {}).get("id")
        if sw:
            visible_ws.add(int(sw))
    out = [_win(c, active_addr, visible_ws) for c in clients or []
           if c.get("mapped", True) and (c.get("size") or [0, 0])[0] > 0]
    out.sort(key=lambda w: (not w["focused"], not w["visible"], w["class"].lower()))
    return out


async def monitors() -> list[dict]:
    return [{"name": m.get("name"), "x": int(m.get("x") or 0), "y": int(m.get("y") or 0),
             "width": int(round((m.get("width") or 0) / (m.get("scale") or 1))),
             "height": int(round((m.get("height") or 0) / (m.get("scale") or 1))),
             "focused": bool(m.get("focused"))}
            for m in (await hyprctl_json("monitors")) or []]


async def cursor_pos() -> tuple[int, int]:
    rc, out, err = await _run(["hyprctl", "cursorpos", "-j"])
    try:
        d = json.loads(out)
        return int(d["x"]), int(d["y"])
    except Exception:
        # Older builds print "x, y".
        a, b = out.decode().split(",")
        return int(float(a)), int(float(b))


async def active_address() -> str:
    a = await hyprctl_json("activewindow")
    return (a or {}).get("address", "") if isinstance(a, dict) else ""


def match_window(wins: list[dict], spec: str | None) -> dict:
    """Hermes's matching tiers, on this compositor's fields.

    Omitted → the focused window. ``0x…`` → that address. Otherwise exact
    class, exact title, class substring, title substring — and on no match an
    error listing what exists, never a silent fall back to the focused window.
    """
    if not wins:
        raise DesktopError("no windows are open", "no_window")
    if not spec:
        for w in wins:
            if w["focused"]:
                return w
        raise DesktopError("no window has focus; pass window=", "no_window")
    s = spec.strip()
    if s.startswith("0x"):
        for w in wins:
            if w["address"] == s:
                return w
    sl = s.lower()
    tiers = (
        lambda w: w["class"].lower() == sl,
        lambda w: w["title"].lower() == sl,
        lambda w: sl in w["class"].lower(),
        lambda w: sl in w["title"].lower(),
    )
    for tier in tiers:
        hits = [w for w in wins if tier(w)]
        if hits:
            hits.sort(key=lambda w: (not w["focused"], not w["visible"]))
            return hits[0]
    names = ", ".join(f"{w['class']} '{w['title'][:40]}'" for w in wins[:15])
    raise DesktopError(f"no window matches {spec!r}; open windows: {names}", "no_window")


# ─────────────────────────────────────────────────────────────────────────
# Capture
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class Capture:
    capture_id: str
    kind: str                     # "window" | "screen"
    origin_x: int                 # layout px of the image's top-left
    origin_y: int
    width: int                    # layout px covered
    height: int
    scale: float                  # image px per layout px
    window: dict | None = None
    elements: list[dict] = field(default_factory=list)
    at: float = field(default_factory=time.time)

    @property
    def image_size(self) -> tuple[int, int]:
        return max(1, round(self.width * self.scale)), max(1, round(self.height * self.scale))

    def to_screen(self, x: float, y: float) -> tuple[int, int]:
        return (int(round(self.origin_x + x / self.scale)),
                int(round(self.origin_y + y / self.scale)))

    def element_center_screen(self, el: dict) -> tuple[int, int] | None:
        b = el.get("bounds")
        if not b or self.window is None:
            return None
        return (int(self.window["x"] + b[0] + b[2] / 2),
                int(self.window["y"] + b[1] + b[3] / 2))


def scale_for(width: int, height: int, max_dimension: int) -> float:
    longest = max(width, height, 1)
    return min(1.0, float(max_dimension) / float(longest))


# JPEG at q85: a 1456-px UI frame is ~150 KB against ~1 MB as PNG, and the
# model's cost is set by pixels, not bytes. Request size, the spill file and
# the 24 MiB outbound cap are what the bytes buy.
_FORMAT_ARGS = ["-t", "jpeg", "-q", "85"]
MIME = "image/jpeg"


async def grab_window(stable_id: str, scale: float) -> bytes:
    """One window's own buffer, by its foreign-toplevel id.

    ``grim -T`` (ext-image-copy-capture) reads the toplevel itself, so the
    window need not be on the visible workspace and nothing drawn over it
    leaks in — measured 2026-09-23 on a Thunderbird window two workspaces
    away. Hyprland's ``stableId`` is that identifier.
    """
    argv = ["grim", "-T", stable_id, *_FORMAT_ARGS]
    if scale < 0.999:
        argv[1:1] = ["-s", f"{scale:.4f}"]
    rc, out, err = await _run(argv + ["-"], timeout=10)
    if rc != 0 or not out:
        raise DesktopError(f"grim -T failed: {err or rc}", "capture_failed")
    return out


async def grab(x: int, y: int, w: int, h: int, scale: float) -> bytes:
    argv = ["grim", "-g", f"{x},{y} {w}x{h}", *_FORMAT_ARGS]
    if scale < 0.999:
        argv[1:1] = ["-s", f"{scale:.4f}"]
    rc, out, err = await _run(argv + ["-"], timeout=10)
    if rc != 0 or not out:
        raise DesktopError(f"grim failed: {err or rc}", "capture_failed")
    return out


# ─────────────────────────────────────────────────────────────────────────
# Input
# ─────────────────────────────────────────────────────────────────────────

async def move_cursor(x: int, y: int) -> None:
    await dispatch("movecursor", str(int(x)), str(int(y)))


async def focus_window(address: str) -> None:
    await dispatch("focuswindow", f"address:{address}")


def split_combo(keys: list[str]) -> tuple[list[str], str]:
    mods = [k for k in keys if k in WTYPE_MODS]
    rest = [k for k in keys if k not in WTYPE_MODS]
    if len(rest) != 1:
        raise DesktopError(f"a key combo needs exactly one non-modifier key, got {rest or 'none'}",
                           "bad_keys")
    return mods, rest[0]


def _keysym(k: str) -> str:
    return WTYPE_KEYS.get(k.lower(), k if len(k) == 1 else k)


async def wtype_combo(mods: list[str], key: str) -> None:
    argv = ["wtype"]
    for m in mods:
        argv += ["-M", WTYPE_MODS[m]]
    argv += ["-k", _keysym(key)]
    for m in reversed(mods):
        argv += ["-m", WTYPE_MODS[m]]
    rc, _, err = await _run(argv)
    if rc != 0:
        raise DesktopError(f"wtype: {err or rc}", "input_failed")


async def wtype_text(text: str) -> None:
    # One argv per 400 chars keeps any single invocation short; wtype holds
    # no state between runs, so splitting changes nothing on the wire.
    for i in range(0, len(text), 400):
        rc, _, err = await _run(["wtype", "-d", "4", "--", text[i:i + 400]], timeout=60)
        if rc != 0:
            raise DesktopError(f"wtype: {err or rc}", "input_failed")


async def send_shortcut(mods: list[str], key: str, address: str) -> None:
    """A hotkey to one window, without focusing it (Hyprland ``sendshortcut``)."""
    mod = "".join(HYPR_MODS[m] for m in mods)
    await dispatch("sendshortcut", f"{mod},{_keysym(key)},address:{address}")


# ─────────────────────────────────────────────────────────────────────────
# Helper process
# ─────────────────────────────────────────────────────────────────────────

class Helper:
    """The stdio JSON-lines client for ``lloyd-desktop-helper.py``."""

    def __init__(self, python: str = "/usr/bin/python3", script: Path = HELPER) -> None:
        self.python = python
        self.script = script
        self.proc: asyncio.subprocess.Process | None = None
        self.lock = asyncio.Lock()
        self.ids = itertools.count(1)
        self.a11y_enabled = False

    async def _ensure(self) -> asyncio.subprocess.Process:
        if self.proc is not None and self.proc.returncode is None:
            return self.proc
        env = dict(os.environ)
        env.setdefault("PYTHONWARNINGS", "ignore")
        env["PYTHONWARNINGS"] = "ignore"
        self.proc = await asyncio.create_subprocess_exec(
            self.python, "-u", str(self.script),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL, env=env, limit=16 * 1024 * 1024)
        self.a11y_enabled = False
        return self.proc

    async def call(self, method: str, params: dict | None = None, *,
                   timeout: float = 15.0) -> dict:
        async with self.lock:
            proc = await self._ensure()
            rid = next(self.ids)
            line = json.dumps({"id": rid, "method": method, "params": params or {}}) + "\n"
            try:
                proc.stdin.write(line.encode())
                await proc.stdin.drain()
                while True:
                    raw = await asyncio.wait_for(proc.stdout.readline(), timeout)
                    if not raw:
                        raise DesktopError("desktop helper exited", "helper_died")
                    resp = json.loads(raw)
                    if resp.get("id") == rid:
                        break
            except (asyncio.TimeoutError, BrokenPipeError, ConnectionResetError) as exc:
                # A hung toolkit: kill the helper so the next call starts clean.
                # An accessibility call that timed out may or may not have run.
                await self.close()
                raise DesktopError(
                    f"desktop helper {method} did not answer ({type(exc).__name__}); "
                    f"outcome unknown — capture again before acting again",
                    "timeout_outcome_unknown") from exc
            if "error" in resp:
                raise DesktopError(resp["error"], resp.get("code") or "helper_error")
            return resp.get("result") or {}

    async def enable_a11y(self) -> None:
        if self.a11y_enabled:
            return
        try:
            await self.call("enable_a11y")
            self.a11y_enabled = True
        except DesktopError as exc:
            logger.warning("desktop: could not enable a11y: %s", exc)

    async def close(self) -> None:
        proc, self.proc = self.proc, None
        if proc is not None and proc.returncode is None:
            try:
                proc.stdin.close()
                await asyncio.wait_for(proc.wait(), 2)
            except Exception:
                proc.kill()
