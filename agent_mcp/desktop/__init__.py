"""Computer use on the real desktop: ``desktop_capture`` and ``desktop_act``.

Modelled on Nous Research's Hermes Agent ``computer_use`` tool (MIT,
github.com/NousResearch/hermes-agent, ``tools/computer_use/``): capture a
window as a screenshot plus a numbered list of its interactable elements,
act by element index (preferred) or by pixel, read the ``verdict`` each action
returns, re-capture to verify. Two tools instead of Hermes's one, because a
single tool cannot be both read-only (capture: plan mode, parallel batches)
and side-effecting (act: effect ledger, lease). ``architecture/desktop.md`` is
the long version.

What is Lloyd's own, not Hermes's:

* **The backend** is Hyprland-native (``hypr.py``): ``grim -T`` per window,
  ``hyprctl`` for windows/focus/cursor/``sendshortcut``, ``wtype`` for keys,
  and a system-python helper for AT-SPI and a uinput pointer. cua-driver's
  Linux path is X11/XWayland and does not list Hyprland.
* **The lease** (``app/desktop_lease.py``). The seat is shared with Alan, so
  every action needs a human-granted lease; capture does not. A tripwire
  revokes it the moment the pointer or focus moves under a human's hand.
* **Who may call it**: never a background, worker or bench session — refused
  at dispatch in ``agent_mcp/main.py`` before this module runs.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
import uuid
from collections import OrderedDict
from typing import Any

from mcp.types import CallToolResult, TextContent, Tool

from agent_mcp._shared import image_result, text_result
from agent_mcp.desktop import guards, hypr
from agent_mcp.desktop.hypr import Capture, DesktopError
from app import desktop_lease
from app.config import CONFIG, service_url

logger = logging.getLogger("lloyd-desktop")

LLOYD_API = os.environ.get("LLOYD_API_URL") or service_url("backend", "http://127.0.0.1:8080")

DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "helper_python": "/usr/bin/python3",
    "max_dimension": 1456,
    "max_elements": 200,
    "listed_elements": 60,
    "tripwire_px": 24,
    "deny_windows": ["1password", "bitwarden", "keepassxc", "authenticator",
                     "hyprlock", "polkit", "pinentry"],
    "push_frames": True,
    # How coordinates are spoken with the model: "pixels" of the screenshot,
    # or "norm1000" (0-1000 on each axis — how Qwen3-VL-family models ground).
    # "pixels" is an untested default, not a measured choice: nothing has
    # measured which space the primary grounds in (architecture/desktop.md §7,
    # #1421).
    "coordinate_space": "pixels",
}

ACTIONS = ("click", "double_click", "right_click", "middle_click", "drag",
           "scroll", "type", "key", "set_value", "wait", "focus_window")
_ACTION_HINTS = {"screenshot": "desktop_capture", "capture": "desktop_capture",
                 "hotkey": "key", "keypress": "key", "press": "key",
                 "left_click": "click", "tap": "click", "type_text": "type",
                 "write": "type", "focus": "focus_window", "focus_app": "focus_window",
                 "list_windows": "desktop_capture(mode='windows')", "sleep": "wait",
                 "move": "click", "select": "set_value"}


def cfg() -> dict[str, Any]:
    raw = CONFIG.get("desktop") or {}
    out = dict(DEFAULTS)
    if isinstance(raw, dict):
        out.update({k: v for k, v in raw.items() if v is not None})
    return out


# Per-session last capture: what element indices and coordinates refer to.
_captures: OrderedDict[str, Capture] = OrderedDict()
_helper: hypr.Helper | None = None
_act_lock = asyncio.Lock()


def _helper_client() -> hypr.Helper:
    global _helper
    if _helper is None:
        _helper = hypr.Helper(python=str(cfg().get("helper_python") or "/usr/bin/python3"))
    return _helper


def _remember(session_id: str, cap: Capture) -> None:
    _captures[session_id or ""] = cap
    _captures.move_to_end(session_id or "")
    while len(_captures) > 64:
        _captures.popitem(last=False)


def _session_id() -> str:
    try:
        from agent_mcp._task_registry import current_session_id
        return current_session_id.get() or ""
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────

_COORD = {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}

CAPTURE_DESCRIPTION = (
    "Look at Alan's desktop (Hyprland). Returns a screenshot of one window plus a "
    "numbered list of its interactable elements (role, name, bounds in screenshot "
    "pixels). Read-only — no lease needed. mode='som' (default) = image + elements; "
    "'vision' = image only; 'ax' = elements only (no image; cheapest, and the only "
    "useful mode for a model that cannot see); 'windows' = list open windows "
    "(class, title, address, workspace, focused). window= picks a window by "
    "address (0x…), class or title substring; omitted = the focused window. "
    "scope='screen' grabs the whole monitor (image only, no elements). Works on "
    "windows on other workspaces. Element indices are valid until the next capture."
)

ACT_DESCRIPTION = (
    "Act on Alan's desktop: click / double_click / right_click / middle_click / drag "
    "/ scroll / type / key / set_value / wait / focus_window. Needs the desktop "
    "lease, which only Alan grants from Mission Control's Desktop tab; without it "
    "the result says so — tell Alan what you need. Target with element=N from your "
    "LAST desktop_capture (preferred: clicks go through accessibility, the pointer "
    "does not move) or coordinate=[x,y] in that capture's screenshot pixels. "
    "Each result carries a verdict: effect confirmed|unverifiable, and decision "
    "done|verify_fresh_state|escalate — follow it; never repeat an input the "
    "verdict confirmed, re-capture to check an unverifiable one. "
    "capture_after=true returns a fresh capture of the same window. Refused: "
    "lock/logout/power key combos, shell-dangerous typed text, anything typed into a "
    "terminal that Bash itself would refuse, password fields, password managers. "
    "If Alan moves the mouse the lease is revoked and you stop."
)


async def list_tools() -> list[Tool]:
    # Always listed, refused at call time when `desktop.enabled` is false: a
    # list that emptied itself would hide the tools from the annotation
    # staleness test, the way CLAUDE.md records for code_graph.
    return [
        Tool(name="desktop_capture", description=CAPTURE_DESCRIPTION, inputSchema={
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["som", "vision", "ax", "windows"],
                         "description": "som (default) | vision | ax | windows"},
                "window": {"type": "string",
                           "description": "Address (0x…), class, or title substring. "
                                          "Omit for the focused window."},
                "scope": {"type": "string", "enum": ["window", "screen"],
                          "description": "window (default) or the whole monitor"},
                "find": {"type": "string",
                         "description": "Only list elements whose name or role contains "
                                        "this text (indices stay the capture's own)."},
            },
        }),
        Tool(name="desktop_act", description=ACT_DESCRIPTION, inputSchema={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": list(ACTIONS),
                           "description": "What to do; see the tool description."},
                "element": {"type": "integer",
                            "description": "1-based index from your last desktop_capture."},
                "coordinate": {**_COORD, "description": "[x, y] in the last capture's "
                               "screenshot pixels. Use only when no element fits."},
                "button": {"type": "string", "enum": ["left", "right", "middle"],
                           "description": "Mouse button, default left."},
                "modifiers": {"type": "array", "items": {
                    "type": "string", "enum": ["ctrl", "alt", "shift", "super"]},
                    "description": "Held during a click."},
                "from_element": {"type": "integer", "description": "drag: source element index."},
                "to_element": {"type": "integer", "description": "drag: target element index."},
                "from_coordinate": {**_COORD, "description": "drag: source [x, y]."},
                "to_coordinate": {**_COORD, "description": "drag: target [x, y]."},
                "direction": {"type": "string", "enum": ["up", "down", "left", "right"],
                              "description": "scroll direction."},
                "amount": {"type": "integer", "description": "Scroll ticks, default 3."},
                "text": {"type": "string", "description": "type: text to type at the focus."},
                "keys": {"type": "string",
                         "description": "key: combo like 'ctrl+s', 'enter', 'alt+tab'."},
                "value": {"type": "string",
                          "description": "set_value: new text for an entry, option label "
                                         "for a combo box, number for a slider."},
                "seconds": {"type": "number", "description": "wait: up to 30."},
                "window": {"type": "string",
                           "description": "focus_window/key target; defaults to the last "
                                          "capture's window."},
                "capture_after": {"type": "boolean",
                                  "description": "Return a fresh capture of the same "
                                                 "window after acting."},
            },
            "required": ["action"],
        }),
    ]


# ─────────────────────────────────────────────────────────────────────────
# Capture
# ─────────────────────────────────────────────────────────────────────────

def _norm() -> bool:
    return str(cfg().get("coordinate_space") or "pixels") == "norm1000"


def _fmt_bounds(b: list[int] | None, scale: float, image: tuple[int, int] = (0, 0)) -> str:
    if not b:
        return "@ bounds-unknown (act by element index)"
    px = [v * scale for v in b]
    if _norm() and image[0] and image[1]:
        iw, ih = image
        px = [px[0] * 1000 / iw, px[1] * 1000 / ih, px[2] * 1000 / iw, px[3] * 1000 / ih]
    return "@ ({}, {}, {}, {})".format(*(int(round(v)) for v in px))


def _summary(cap: Capture, *, mode: str, find: str, notes: list[str],
             listed: int) -> str:
    iw, ih = cap.image_size
    w = cap.window or {}
    head = (f"capture mode={mode} id={cap.capture_id} image={iw}x{ih} "
            f"screen={cap.width}x{cap.height} scale={cap.scale:.3f}"
            + (" coords=norm1000" if _norm() else ""))
    if w:
        head += (f" window='{w.get('title', '')[:80]}' class={w.get('class', '')} "
                 f"address={w.get('address', '')} workspace={w.get('workspace', '')}"
                 f"{' focused' if w.get('focused') else ''}"
                 f"{'' if w.get('visible') else ' (not on a visible workspace)'}")
    lines = [head]
    els = cap.elements
    if mode != "vision" and cap.kind == "window":
        shown = [(i, e) for i, e in enumerate(els, 1)
                 if not find or find.lower() in f"{e.get('role', '')} {e.get('name', '')}".lower()]
        lines.append(f"{len(els)} interactable element(s)"
                     + (f", {len(shown)} matching {find!r}" if find else "") + ":")
        for i, e in shown[:listed]:
            extra = []
            if e.get("states"):
                st = [s for s in e["states"] if s not in ("enabled", "sensitive")]
                if "enabled" not in e["states"] and "sensitive" not in e["states"]:
                    st.append("disabled")
                if st:
                    extra.append(",".join(st))
            if e.get("value"):
                extra.append(f"value={e['value'][:60]!r}")
            lines.append(f"  #{i} {e.get('role')} {e.get('name', '')[:120]!r} "
                         f"{_fmt_bounds(e.get('bounds'), cap.scale, cap.image_size)}"
                         + (f" [{'; '.join(extra)}]" if extra else ""))
        if len(shown) > listed:
            lines.append(f"  ... +{len(shown) - listed} more (pass find= to narrow)")
    lines += [f"({n})" for n in notes]
    return "\n".join(lines)


async def _push_frame(cap: Capture, png: bytes | None, tool: str, text: str) -> None:
    if not cfg().get("push_frames", True):
        return
    try:
        import httpx
        iw, ih = cap.image_size
        frame = {
            "tool": tool, "ts": time.time(), "capture_id": cap.capture_id,
            "window": {k: (cap.window or {}).get(k) for k in
                       ("title", "class", "address", "workspace", "focused")},
            "width": iw, "height": ih, "scale": cap.scale, "kind": cap.kind,
            "mime": hypr.MIME if png else None,
            "image_b64": base64.b64encode(png).decode() if png else None,
            "elements": [{"index": i, "role": e.get("role"), "name": (e.get("name") or "")[:80],
                          "bounds": [int(round(v * cap.scale)) for v in e["bounds"]]
                          if e.get("bounds") else None}
                         for i, e in enumerate(cap.elements[:300], 1)],
            "summary": text[:2000],
        }
        async with httpx.AsyncClient(timeout=5.0) as cli:
            await cli.post(f"{LLOYD_API}/api/desktop/state", json=frame)
    except Exception as exc:
        logger.debug("desktop frame push failed: %s", exc)


_pending: set[asyncio.Task] = set()


def _schedule_push(*a: Any) -> None:
    t = asyncio.create_task(_push_frame(*a))
    _pending.add(t)
    t.add_done_callback(_pending.discard)


async def capture(*, mode: str = "som", window: str | None = None,
                  scope: str = "window", find: str = "", session_id: str = "",
                  tool: str = "desktop_capture") -> tuple[str, bytes | None, Capture]:
    c = cfg()
    mode = mode if mode in ("som", "vision", "ax", "windows") else "som"
    max_dim = int(c.get("max_dimension") or 1456)
    deny = c.get("deny_windows") or []
    wins = await hypr.windows()
    notes: list[str] = []
    cid = uuid.uuid4().hex[:8]

    if mode == "windows":
        rows = [w for w in wins if not guards.denied_window(w, deny)]
        lines = [f"{len(rows)} window(s):"] + [
            f"  {w['address']} {w['class']} '{w['title'][:80]}' ws={w['workspace']} "
            f"{w['width']}x{w['height']}{' FOCUSED' if w['focused'] else ''}"
            f"{'' if w['visible'] else ' (hidden workspace)'}" for w in rows]
        return "\n".join(lines), None, Capture(cid, "list", 0, 0, 0, 0, 1.0)

    if scope == "screen":
        mons = await hypr.monitors()
        cx, cy = await hypr.cursor_pos()
        mon = next((m for m in mons if m["x"] <= cx < m["x"] + m["width"]
                    and m["y"] <= cy < m["y"] + m["height"]), mons[0] if mons else None)
        if mon is None:
            raise DesktopError("no monitor", "no_display")
        if any(guards.denied_window(w, deny) for w in wins if w["visible"]):
            raise DesktopError("a window on this screen is on desktop.deny_windows; "
                               "capture a specific window instead", "denied_window")
        scale = hypr.scale_for(mon["width"], mon["height"], max_dim)
        png = await hypr.grab(mon["x"], mon["y"], mon["width"], mon["height"], scale)
        cap = Capture(cid, "screen", mon["x"], mon["y"], mon["width"], mon["height"], scale)
        notes.append("full-screen capture has no element list; to act on a window, "
                     "desktop_capture(window=...) for its elements")
        _remember(session_id, cap)
        text = _summary(cap, mode="vision", find="", notes=notes, listed=0)
        return text, png, cap

    win = hypr.match_window(wins, window)
    why = guards.denied_window(win, deny)
    if why:
        raise DesktopError(why, "denied_window")
    scale = hypr.scale_for(win["width"], win["height"], max_dim)
    cap = Capture(cid, "window", win["x"], win["y"], win["width"], win["height"],
                  scale, window=win)

    png: bytes | None = None
    grab_task = None
    if mode in ("som", "vision"):
        if win.get("stable_id"):
            grab_task = asyncio.create_task(hypr.grab_window(win["stable_id"], scale))
        elif win["visible"]:
            grab_task = asyncio.create_task(
                hypr.grab(win["x"], win["y"], win["width"], win["height"], scale))
        else:
            notes.append("window is on a hidden workspace and has no toplevel id; "
                         "no image")
    if mode in ("som", "ax"):
        helper = _helper_client()
        await helper.enable_a11y()
        try:
            tree = await helper.call("tree", {
                "pid": win["pid"], "title": win["title"], "capture_id": cid,
                "max_elements": int(c.get("max_elements") or 200)})
            cap.elements = tree.get("elements") or []
            if not tree.get("found"):
                notes.append(tree.get("note") or "no accessibility tree")
            elif tree.get("truncated"):
                notes.append(f"element walk stopped at {len(cap.elements)}; "
                             "pass find= or act by coordinate for the rest")
            if not cap.elements and tree.get("found"):
                notes.append("the app exposes no interactable elements (custom-drawn "
                             "or remote desktop); act by coordinate from the image")
        except DesktopError as exc:
            notes.append(f"accessibility tree unavailable: {exc}")
    if grab_task is not None:
        try:
            png = await grab_task
        except DesktopError as exc:
            notes.append(f"no image: {exc}")
    _remember(session_id, cap)
    if any(e.get("role") == "password text" for e in cap.elements):
        notes.append("this window has a password field; never type into it — ask Alan")
    text = _summary(cap, mode=mode, find=find or "", notes=notes,
                    listed=int(c.get("listed_elements") or 60))
    return text, (png if mode != "ax" else None), cap


# ─────────────────────────────────────────────────────────────────────────
# Act
# ─────────────────────────────────────────────────────────────────────────

def _verdict(effect: str, path: str, hint: str = "") -> dict:
    decision = {"confirmed": "done", "unverifiable": "verify_fresh_state",
                "failed": "escalate"}.get(effect, "verify_fresh_state")
    v = {"effect": effect, "path": path, "decision": decision}
    if hint:
        v["hint"] = hint
    elif decision == "verify_fresh_state":
        v["hint"] = ("capture again (or pass capture_after=true) to see whether it "
                     "worked before doing anything else")
    elif decision == "escalate":
        v["hint"] = "try the next rung: coordinate= from the screenshot, or focus_window"
    return v


def _element(cap: Capture | None, idx: int | None) -> dict:
    if cap is None or cap.kind != "window":
        raise DesktopError("no window capture to take element indices from — "
                           "desktop_capture first", "stale")
    if not idx or not 1 <= idx <= len(cap.elements):
        raise DesktopError(f"element #{idx} is not in your last capture "
                           f"(1..{len(cap.elements)})", "stale")
    return cap.elements[idx - 1]


def _point(cap: Capture | None, *, element: int | None,
           coordinate: list[int] | None) -> tuple[int, int]:
    if coordinate:
        if cap is None:
            raise DesktopError("coordinate= needs a capture to be relative to", "stale")
        x, y = coordinate
        iw, ih = cap.image_size
        if _norm():
            if not (0 <= x <= 1000 and 0 <= y <= 1000):
                raise DesktopError(f"coordinate {coordinate} is outside 0-1000",
                                   "bad_coordinate")
            x, y = x * iw / 1000.0, y * ih / 1000.0
        if not (0 <= x <= iw and 0 <= y <= ih):
            raise DesktopError(f"coordinate {coordinate} is outside the {iw}x{ih} "
                               "screenshot", "bad_coordinate")
        return cap.to_screen(x, y)
    el = _element(cap, element)
    pt = cap.element_center_screen(el)
    if pt is None:
        raise DesktopError(f"element #{element} has no on-screen bounds; act on it by "
                           "index with an accessibility action, or use coordinate=",
                           "no_bounds")
    return pt


async def _require_visible(cap: Capture | None) -> None:
    """Pixel input lands on whatever is under the cursor: the window must be shown."""
    if cap is None or cap.window is None:
        return
    wins = await hypr.windows()
    w = next((w for w in wins if w["address"] == cap.window["address"]), None)
    if w is None:
        raise DesktopError("that window has closed; capture again", "stale")
    if not w["visible"]:
        raise DesktopError("pixel input needs the window on the visible workspace — "
                           "desktop_act(focus_window) first (visible to Alan), or use "
                           "an element's accessibility action", "not_visible")
    if (w["x"], w["y"]) != (cap.window["x"], cap.window["y"]) or \
            (w["width"], w["height"]) != (cap.window["width"], cap.window["height"]):
        raise DesktopError("the window moved or resized since your capture; capture "
                           "again", "stale")


async def _tripwire(session_id: str) -> str | None:
    """Revoke the lease if a human moved the pointer or focus since Lloyd's last action."""
    lease = desktop_lease.read()
    px = int(cfg().get("tripwire_px") or 24)
    last = lease.get("last_cursor")
    cur = await hypr.cursor_pos()
    why = None
    if last and (abs(cur[0] - last[0]) > px or abs(cur[1] - last[1]) > px):
        why = f"the pointer moved to {cur} since Lloyd's last action at {tuple(last)}"
    elif lease.get("last_window"):
        active = await hypr.active_address()
        if active and active != lease["last_window"]:
            why = "keyboard focus moved to another window since Lloyd's last action"
    if why:
        desktop_lease.revoke(f"tripwire: {why}", by="tripwire")
        await _notify("Lloyd stopped: you took the desktop back")
        return why
    if not last:
        desktop_lease.note_seat(cur, await hypr.active_address())
    return None


async def _notify(msg: str) -> None:
    try:
        await hypr._run(["hyprctl", "notify", "1", "4000", "rgb(ff9e3b)", msg])
    except Exception:
        pass


async def _after(session_id: str) -> None:
    await asyncio.sleep(0.15)
    try:
        desktop_lease.note_seat(await hypr.cursor_pos(), await hypr.active_address())
    except Exception:
        pass


async def act(args: dict, session_id: str) -> tuple[dict, bytes | None, str]:
    action = str(args.get("action") or "")
    if action not in ACTIONS:
        hint = _ACTION_HINTS.get(action)
        raise DesktopError(f"unknown action {action!r}"
                           + (f"; did you mean {hint}?" if hint else "")
                           + f" Actions: {', '.join(ACTIONS)}", "bad_action")
    if action == "wait":
        secs = max(0.0, min(30.0, float(args.get("seconds") or 1)))
        await asyncio.sleep(secs)
        return {"ok": True, "action": "wait", "seconds": secs}, None, ""

    ok, why, epoch = desktop_lease.agent_may_act(session_id)
    if not ok:
        raise DesktopError(f"{why}. Actions need the desktop lease, which only Alan "
                           "grants (Mission Control → Desktop). Say what you want to "
                           "do and ask Alan for it.", "no_lease")
    tripped = await _tripwire(session_id)
    if tripped:
        raise DesktopError(f"Alan took the desktop back ({tripped}); the lease is "
                           "revoked. Stop and tell Alan where you got to.",
                           "human_has_control")

    cap = _captures.get(session_id or "")
    c = cfg()
    deny = c.get("deny_windows") or []
    if cap is not None and cap.window is not None:
        w = guards.denied_window(cap.window, deny)
        if w:
            raise DesktopError(w, "denied_window")
    helper = _helper_client()
    button = str(args.get("button") or "left")
    mods = guards.canonical_keys(list(args.get("modifiers") or []))
    result: dict[str, Any] = {"action": action}
    target_window = cap.window if cap is not None else None

    async with _act_lock:
        if action in ("click", "double_click", "right_click", "middle_click"):
            if action == "right_click":
                button = "right"
            elif action == "middle_click":
                button = "middle"
            clicks = 2 if action == "double_click" else 1
            el_idx = args.get("element")
            if mods:
                why = guards.blocked_combo([], mods)
                if why:
                    raise DesktopError(why, "blocked")
            if el_idx and not args.get("coordinate") and button == "left" \
                    and clicks == 1 and not mods:
                el = _element(cap, int(el_idx))
                if el.get("role") == "password text":
                    raise DesktopError("that is a password field", "blocked")
                try:
                    r = await helper.call("act", {"capture_id": cap.capture_id,
                                                  "index": int(el_idx), "kind": "click"})
                except DesktopError as exc:
                    if exc.code == "stale":
                        raise
                    r = {"ok": False, "note": str(exc)}
                if r.get("ok"):
                    result.update(ok=True, element=int(el_idx),
                                  verdict=_verdict("unverifiable", r.get("path", "atspi"),
                                                   "accessibility action sent; the pointer "
                                                   "did not move. Capture to confirm."))
                    await _after(session_id)
                    return await _maybe_after(args, result, session_id, cap)
                result["fallback"] = r.get("note") or "no accessibility action"
            await _require_visible(cap)
            x, y = _point(cap, element=el_idx, coordinate=args.get("coordinate"))
            await hypr.move_cursor(x, y)
            # Modifiers are held on the helper's own uinput device: a `wtype -M`
            # releases its virtual keyboard, and with it the modifier, on exit.
            await helper.call("pointer", {"op": "click", "button": button,
                                          "clicks": clicks, "mods": mods})
            result.update(ok=True, screen_point=[x, y],
                          verdict=_verdict("unverifiable", "uinput_pixel"))

        elif action == "drag":
            await _require_visible(cap)
            fx, fy = _point(cap, element=args.get("from_element"),
                            coordinate=args.get("from_coordinate"))
            tx, ty = _point(cap, element=args.get("to_element"),
                            coordinate=args.get("to_coordinate"))
            await hypr.move_cursor(fx, fy)
            await helper.call("pointer", {"op": "down", "button": button})
            try:
                steps = 12
                for i in range(1, steps + 1):
                    await hypr.move_cursor(int(fx + (tx - fx) * i / steps),
                                           int(fy + (ty - fy) * i / steps))
                    await helper.call("pointer", {"op": "nudge"})
                    await asyncio.sleep(0.02)
            finally:
                await helper.call("pointer", {"op": "up", "button": button})
            result.update(ok=True, from_point=[fx, fy], to_point=[tx, ty],
                          verdict=_verdict("unverifiable", "uinput_drag"))

        elif action == "scroll":
            direction = str(args.get("direction") or "down")
            amount = max(1, min(50, int(args.get("amount") or 3)))
            if args.get("element") or args.get("coordinate"):
                await _require_visible(cap)
                x, y = _point(cap, element=args.get("element"),
                              coordinate=args.get("coordinate"))
            else:
                await _require_visible(cap)
                if cap is None or cap.window is None:
                    raise DesktopError("scroll needs a window capture, element or "
                                       "coordinate", "stale")
                w = cap.window
                x, y = w["x"] + w["width"] // 2, w["y"] + w["height"] // 2
            await hypr.move_cursor(x, y)
            await helper.call("pointer", {"op": "wheel", "direction": direction,
                                          "amount": amount})
            result.update(ok=True, screen_point=[x, y],
                          verdict=_verdict("unverifiable", "uinput_wheel"))

        elif action == "type":
            text = str(args.get("text") or "")
            if not text:
                raise DesktopError("type needs text=", "bad_args")
            active = await _active_window()
            klass = (active or {}).get("class", "")
            why = guards.blocked_text(text, window_class=klass, session_id=session_id)
            if why:
                raise DesktopError(why, "blocked")
            if args.get("element"):
                el = _element(cap, int(args["element"]))
                if el.get("role") == "password text":
                    raise DesktopError("that is a password field", "blocked")
                await helper.call("act", {"capture_id": cap.capture_id,
                                          "index": int(args["element"]), "kind": "focus"})
            if target_window and active and active.get("address") != target_window["address"]:
                raise DesktopError(
                    f"typing goes to the focused window, which is "
                    f"{active.get('class')} '{active.get('title', '')[:40]}', not your "
                    f"capture's {target_window['class']}. desktop_act(focus_window) "
                    f"first, or use set_value on an element (background).",
                    "input_target_mismatch")
            await hypr.wtype_text(text)
            result.update(ok=True, typed_chars=len(text),
                          verdict=_verdict("unverifiable", "wtype"))

        elif action == "key":
            keys = guards.canonical_keys(str(args.get("keys") or ""))
            if not keys:
                raise DesktopError("key needs keys=, e.g. 'ctrl+s'", "bad_args")
            why = guards.blocked_combo(keys, mods)
            if why:
                raise DesktopError(why, "blocked")
            kmods, key = hypr.split_combo(sorted(set(keys) | set(mods),
                                                 key=lambda k: (k not in hypr.WTYPE_MODS, k)))
            target = target_window
            if args.get("window"):
                target = hypr.match_window(await hypr.windows(), args["window"])
            active = await _active_window()
            if target and active and active.get("address") != target["address"]:
                # Background rung: Hyprland delivers a shortcut to a window that
                # does not have focus. Only combos — a bare key has no bind path.
                await hypr.send_shortcut(kmods, key, target["address"])
                result.update(ok=True, verdict=_verdict(
                    "unverifiable", "hypr_sendshortcut",
                    "sent to the window without focusing it; capture to confirm"))
            else:
                await hypr.wtype_combo(kmods, key)
                result.update(ok=True, verdict=_verdict("unverifiable", "wtype"))

        elif action == "set_value":
            el_idx = args.get("element")
            if not el_idx:
                raise DesktopError("set_value needs element=", "bad_args")
            el = _element(cap, int(el_idx))
            if el.get("role") == "password text":
                raise DesktopError("that is a password field", "blocked")
            value = str(args.get("value") if args.get("value") is not None else "")
            kind = "set_text" if el.get("role") in ("entry", "text", "terminal") else "set_value"
            why = guards.blocked_text(value, window_class=(cap.window or {}).get("class", ""),
                                      session_id=session_id)
            if why:
                raise DesktopError(why, "blocked")
            r = await helper.call("act", {"capture_id": cap.capture_id, "index": int(el_idx),
                                          "kind": kind, "value": value})
            if r.get("ok") and r.get("verified") is True:
                v = _verdict("confirmed", r.get("path", kind))
            elif r.get("ok"):
                v = _verdict("unverifiable", r.get("path", kind))
            else:
                v = _verdict("failed", r.get("path", kind), r.get("note") or r.get("code") or "")
            result.update(ok=bool(r.get("ok")), verdict=v)

        elif action == "focus_window":
            spec = args.get("window") or (target_window or {}).get("address")
            if not spec:
                raise DesktopError("focus_window needs window=", "bad_args")
            w = hypr.match_window(await hypr.windows(), spec)
            why = guards.denied_window(w, deny)
            if why:
                raise DesktopError(why, "denied_window")
            await hypr.focus_window(w["address"])
            active = await hypr.active_address()
            result.update(ok=active == w["address"], window=w["address"],
                          verdict=_verdict("confirmed" if active == w["address"] else "failed",
                                           "hypr_focuswindow",
                                           "window focused and on screen (Alan can see "
                                           "this); element indices from older captures "
                                           "may be stale — capture again"))

        if desktop_lease.current_epoch() != epoch:
            raise DesktopError("the lease changed while acting — Alan took control; "
                               "the result is void", "human_has_control")
        await _after(session_id)
    return await _maybe_after(args, result, session_id, cap)


async def _active_window() -> dict | None:
    wins = await hypr.windows()
    return next((w for w in wins if w["focused"]), None)


async def _maybe_after(args: dict, result: dict, session_id: str,
                       cap: Capture | None) -> tuple[dict, bytes | None, str]:
    if not args.get("capture_after") or not result.get("ok"):
        return result, None, ""
    await asyncio.sleep(0.35)
    window = (cap.window or {}).get("address") if cap and cap.window else None
    try:
        text, png, new_cap = await capture(mode="som", window=window,
                                           session_id=session_id, tool="desktop_act")
        _schedule_push(new_cap, png, "desktop_act", text)
        return result, png, text
    except DesktopError as exc:
        result["capture_after_error"] = str(exc)
        return result, None, ""


# ─────────────────────────────────────────────────────────────────────────
# MCP
# ─────────────────────────────────────────────────────────────────────────

def _error(exc: DesktopError | Exception) -> CallToolResult:
    code = getattr(exc, "code", "error")
    return text_result(json.dumps({"error": str(exc), "code": code}), is_error=True)


async def call_tool(name: str, arguments: dict) -> CallToolResult:
    arguments = arguments or {}
    if not cfg().get("enabled"):
        return _error(DesktopError("desktop computer use is disabled "
                                   "(desktop.enabled: false)", "disabled"))
    sid = _session_id()
    try:
        if name == "desktop_capture":
            text, png, cap = await capture(
                mode=str(arguments.get("mode") or "som"),
                window=arguments.get("window") or None,
                scope=str(arguments.get("scope") or "window"),
                find=str(arguments.get("find") or ""), session_id=sid)
            if cap.kind != "list":
                _schedule_push(cap, png, name, text)
            if png:
                return image_result(text, png, mime=hypr.MIME)
            return text_result(text, is_error=False)
        if name == "desktop_act":
            result, png, cap_text = await act(arguments, sid)
            body = json.dumps(result, indent=1)
            if cap_text:
                body = f"{body}\n\n{cap_text}"
            if png:
                return image_result(body, png, mime=hypr.MIME, is_error=False)
            return text_result(body, is_error=not result.get("ok", True))
    except DesktopError as exc:
        return _error(exc)
    except Exception as exc:   # pragma: no cover - defensive
        logger.exception("desktop: %s failed", name)
        return _error(DesktopError(f"{type(exc).__name__}: {exc}", "internal"))
    return _error(DesktopError(f"unknown tool {name}", "unknown_tool"))


async def shutdown() -> None:
    global _helper
    if _helper is not None:
        try:
            await _helper.call("pointer", {"op": "release_all"}, timeout=2)
        except Exception:
            pass
        await _helper.close()
        _helper = None
