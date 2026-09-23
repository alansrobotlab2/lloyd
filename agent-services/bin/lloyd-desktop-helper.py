#!/usr/bin/python3
"""Lloyd's desktop helper: AT-SPI and a virtual pointer, over stdio JSON lines.

Spawned and owned by the aggregator (``agent_mcp/desktop/hypr.py``). It runs on
the SYSTEM python on purpose: ``gi.repository.Atspi`` exists there (Arch's
``python-gobject`` + ``at-spi2-core``) and in no venv. It is stdlib + gi only,
so it needs nothing installed.

Why a separate process at all — the same seam Hermes Agent keeps between its
tool and cua-driver:

* AT-SPI is synchronous D-Bus into every toolkit on the desktop. A hung app
  stalls the call; a crashing binding takes the process with it. Here that
  costs a helper restart, not the aggregator that serves every tool.
* The virtual pointer is a /dev/uinput device. It lives exactly as long as
  this process, so a killed aggregator cannot leave a device with a button
  held down: the kernel releases it when the fd closes.

Protocol: one JSON object per line on stdin, ``{"id", "method", "params"}``;
one line back, ``{"id", "result"}`` or ``{"id", "error"}``. Methods are the
``m_*`` functions below. Nothing here decides policy (the lease, the
guards, which window) — the aggregator does, and this does what it is told.

Coordinates: element bounds are returned WINDOW-relative. Measured on
Hyprland 2026-09-23, AT-SPI's "screen" extents are window-relative too on
Wayland (a frame reports 0,0 wherever the compositor put it) and a widget
that is not laid out reports x=y=INT_MIN; both map to ``bounds: None`` here,
Hermes's zero-bounds rule, so no coordinate is ever derived from a sentinel.
"""

from __future__ import annotations

import warnings as _w
_w.filterwarnings("ignore", category=DeprecationWarning)

import fcntl
import json
import os
import struct
import sys
import time
import traceback

# ─────────────────────────────────────────────────────────────────────────
# uinput: a virtual relative mouse with three buttons and two wheels.
# Position comes from `hyprctl dispatch movecursor` (exact, in layout
# pixels); this device supplies what the compositor cannot synthesise —
# button presses, wheel ticks, and motion events while a button is held.
# ─────────────────────────────────────────────────────────────────────────

EV_SYN, EV_KEY, EV_REL = 0x00, 0x01, 0x02
SYN_REPORT = 0
REL_X, REL_Y, REL_HWHEEL, REL_WHEEL = 0x00, 0x01, 0x06, 0x08
BTN = {"left": 0x110, "right": 0x111, "middle": 0x112}
MOD_KEYS = {"ctrl": 29, "shift": 42, "alt": 56, "super": 125}

UI_SET_EVBIT = 0x40045564
UI_SET_KEYBIT = 0x40045565
UI_SET_RELBIT = 0x40045566
UI_DEV_SETUP = 0x405C5503
UI_DEV_CREATE = 0x5501
UI_DEV_DESTROY = 0x5502
BUS_VIRTUAL = 0x06

_EVENT = struct.Struct("llHHi")


class VirtualMouse:
    def __init__(self) -> None:
        self.fd = os.open("/dev/uinput", os.O_WRONLY | os.O_NONBLOCK)
        fcntl.ioctl(self.fd, UI_SET_EVBIT, EV_KEY)
        fcntl.ioctl(self.fd, UI_SET_EVBIT, EV_REL)
        for code in list(BTN.values()) + list(MOD_KEYS.values()):
            fcntl.ioctl(self.fd, UI_SET_KEYBIT, code)
        for code in (REL_X, REL_Y, REL_WHEEL, REL_HWHEEL):
            fcntl.ioctl(self.fd, UI_SET_RELBIT, code)
        name = b"lloyd-virtual-pointer"
        setup = struct.pack("HHHH80sI", BUS_VIRTUAL, 0x4C4C, 0x0001, 1,
                            name.ljust(80, b"\0"), 0)
        fcntl.ioctl(self.fd, UI_DEV_SETUP, setup)
        fcntl.ioctl(self.fd, UI_DEV_CREATE)
        # The compositor has to notice the new device before its first event
        # lands anywhere; measured ~100 ms on Hyprland.
        time.sleep(0.35)
        self.held: set[int] = set()

    def _emit(self, etype: int, code: int, value: int) -> None:
        now = time.time()
        sec = int(now)
        os.write(self.fd, _EVENT.pack(sec, int((now - sec) * 1e6), etype, code, value))

    def _syn(self) -> None:
        self._emit(EV_SYN, SYN_REPORT, 0)

    def nudge(self) -> None:
        """+1 then -1: a motion event at the current position, net zero."""
        self._emit(EV_REL, REL_X, 1)
        self._syn()
        self._emit(EV_REL, REL_X, -1)
        self._syn()

    def button(self, name: str, down: bool) -> None:
        code = BTN[name]
        self._emit(EV_KEY, code, 1 if down else 0)
        self._syn()
        (self.held.add if down else self.held.discard)(code)

    def key(self, code: int, down: bool) -> None:
        self._emit(EV_KEY, code, 1 if down else 0)
        self._syn()
        (self.held.add if down else self.held.discard)(code)

    def click(self, name: str = "left", clicks: int = 1, mods=()) -> None:
        codes = [MOD_KEYS[m] for m in mods if m in MOD_KEYS]
        for c in codes:
            self.key(c, True)
        try:
            for i in range(max(1, clicks)):
                self.button(name, True)
                time.sleep(0.03)
                self.button(name, False)
                if i + 1 < clicks:
                    time.sleep(0.06)
        finally:
            for c in reversed(codes):
                self.key(c, False)

    def rel(self, dx: int, dy: int) -> None:
        if dx:
            self._emit(EV_REL, REL_X, int(dx))
        if dy:
            self._emit(EV_REL, REL_Y, int(dy))
        self._syn()

    def wheel(self, direction: str, amount: int) -> None:
        code = REL_HWHEEL if direction in ("left", "right") else REL_WHEEL
        sign = 1 if direction in ("up", "right") else -1
        for _ in range(max(1, min(50, amount))):
            self._emit(EV_REL, code, sign)
            self._syn()
            time.sleep(0.015)

    def release_all(self) -> None:
        for code in list(self.held):
            self._emit(EV_KEY, code, 0)
            self._syn()
        self.held.clear()

    def close(self) -> None:
        try:
            self.release_all()
            fcntl.ioctl(self.fd, UI_DEV_DESTROY)
        finally:
            os.close(self.fd)


_mouse: VirtualMouse | None = None


def _get_mouse() -> VirtualMouse:
    global _mouse
    if _mouse is None:
        _mouse = VirtualMouse()
    return _mouse


# ─────────────────────────────────────────────────────────────────────────
# AT-SPI
# ─────────────────────────────────────────────────────────────────────────

_Atspi = None


def _atspi():
    global _Atspi
    if _Atspi is None:
        import gi
        gi.require_version("Atspi", "2.0")
        from gi.repository import Atspi
        Atspi.set_timeout(1500, 4000)
        _Atspi = Atspi
    return _Atspi


INTERACTIVE_ROLES = {
    "push button", "toggle button", "check box", "radio button", "menu item",
    "check menu item", "radio menu item", "menu", "combo box", "entry",
    "password text", "spin button", "slider", "link", "page tab", "list item",
    "tree item", "icon", "button", "switch", "toolbar", "text", "terminal",
    "document web",
}
# Roles only worth listing when they can take input.
EDITABLE_ONLY = {"text", "terminal", "document web"}
CLICK_ACTIONS = ("click", "press", "activate", "jump", "toggle", "open",
                 "select", "clickancestor", "showmenu")

_INT_MIN = -(2 ** 31)
_captures: dict[str, list] = {}
_capture_order: list[str] = []


def _bounds(obj, coord):
    try:
        e = obj.get_extents(coord)
    except Exception:
        return None
    if e.x <= _INT_MIN + 1 or e.y <= _INT_MIN + 1 or e.width <= 0 or e.height <= 0:
        return None
    if e.width > 20000 or e.height > 20000:
        return None
    return [int(e.x), int(e.y), int(e.width), int(e.height)]


def _states(obj):
    A = _atspi()
    try:
        ss = obj.get_state_set()
    except Exception:
        return set()
    out = set()
    for name in ("showing", "visible", "enabled", "sensitive", "focused",
                 "focusable", "editable", "checked", "selected", "expanded",
                 "active", "pressed"):
        st = getattr(A.StateType, name.upper(), None)
        if st is not None and ss.contains(st):
            out.add(name)
    return out


def _actions(obj):
    try:
        iface = obj.get_action_iface()
    except Exception:
        return []
    if iface is None:
        return []
    out = []
    try:
        for i in range(iface.get_n_actions()):
            out.append(iface.get_action_name(i))
    except Exception:
        pass
    return out


def _desktop_apps():
    A = _atspi()
    d = A.get_desktop(0)
    for i in range(d.get_child_count()):
        try:
            app = d.get_child_at_index(i)
        except Exception:
            continue
        if app is not None:
            yield app


def m_enable_a11y(params):
    """Tell every toolkit an assistive technology is listening.

    Chromium and Electron build no accessibility tree until the session's
    ``org.a11y.Status.IsEnabled`` is true; flipping it makes running
    instances build one retroactively (what cua-driver does too).
    """
    from gi.repository import Gio, GLib
    bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
    bus.call_sync("org.a11y.Bus", "/org/a11y/bus",
                  "org.freedesktop.DBus.Properties", "Set",
                  GLib.Variant("(ssv)", ("org.a11y.Status", "IsEnabled",
                                         GLib.Variant("b", True))),
                  None, Gio.DBusCallFlags.NONE, 2000, None)
    return {"enabled": True}


def m_apps(params):
    out = []
    for app in _desktop_apps():
        try:
            pid = app.get_process_id()
            wins = []
            for j in range(min(app.get_child_count(), 40)):
                w = app.get_child_at_index(j)
                if w is None:
                    continue
                wins.append({"title": w.get_name() or "", "role": w.get_role_name()})
            out.append({"name": app.get_name() or "", "pid": pid, "windows": wins})
        except Exception:
            continue
    return {"apps": out}


def _find_window(pid: int, title: str):
    best, best_score = None, -1
    for app in _desktop_apps():
        try:
            if app.get_process_id() != pid:
                continue
            n = app.get_child_count()
        except Exception:
            continue
        for j in range(min(n, 60)):
            try:
                w = app.get_child_at_index(j)
                if w is None:
                    continue
                name = w.get_name() or ""
                st = _states(w)
            except Exception:
                continue
            score = 0
            if title and name == title:
                score = 100
            elif title and name and (name in title or title in name):
                score = 60
            if "active" in st:
                score += 20
            if "showing" in st or "visible" in st:
                score += 5
            if score > best_score:
                best, best_score = w, score
    return best


def _collect(win, max_elements: int):
    A = _atspi()
    found = []
    seen = 0

    def keep(obj) -> dict | None:
        try:
            role = obj.get_role_name()
        except Exception:
            return None
        if role not in INTERACTIVE_ROLES:
            return None
        st = _states(obj)
        if "showing" not in st and "visible" not in st:
            return None
        if role in EDITABLE_ONLY and "editable" not in st:
            return None
        b = _bounds(obj, A.CoordType.WINDOW)
        try:
            name = obj.get_name() or ""
        except Exception:
            name = ""
        if not name and role not in ("entry", "password text", "text",
                                     "combo box", "spin button", "slider",
                                     "terminal", "document web"):
            try:
                name = obj.get_description() or ""
            except Exception:
                pass
            if not name:
                return None
        rec = {"role": role, "name": name[:200], "bounds": b,
               "states": sorted(st & {"focused", "checked", "selected",
                                       "expanded", "editable", "pressed",
                                       "enabled", "sensitive"}),
               "actions": _actions(obj)[:6]}
        if role in ("entry", "text", "spin button", "combo box") and role != "password text":
            try:
                t = obj.get_text_iface()
                if t is not None:
                    rec["value"] = (t.get_text(0, min(t.get_character_count(), 120)) or "")
            except Exception:
                pass
        if role == "slider":
            try:
                v = obj.get_value_iface()
                if v is not None:
                    rec["value"] = str(v.get_current_value())
            except Exception:
                pass
        return rec

    # Collection.get_matches is one D-Bus round trip for the whole window
    # where a walk is one per node; fall back to the walk when the toolkit
    # does not implement it.
    try:
        coll = win.get_collection_iface()
    except Exception:
        coll = None
    if coll is not None:
        try:
            ss = A.StateSet.new([])
            ss.add(A.StateType.SHOWING)
            rule = A.MatchRule.new(ss, A.CollectionMatchType.ALL, [],
                                   A.CollectionMatchType.NONE, [],
                                   A.CollectionMatchType.NONE, [],
                                   A.CollectionMatchType.NONE, False)
            matches = coll.get_matches(rule, A.CollectionSortOrder.CANONICAL,
                                       max_elements * 12, True)
            for obj in matches:
                seen += 1
                rec = keep(obj)
                if rec is not None:
                    found.append((obj, rec))
                    if len(found) >= max_elements:
                        break
            if found:
                return found, seen, "collection"
        except Exception:
            found = []

    stack = [(win, 0)]
    while stack and len(found) < max_elements and seen < max_elements * 40:
        node, depth = stack.pop()
        try:
            n = node.get_child_count()
        except Exception:
            continue
        kids = []
        for i in range(min(n, 400)):
            try:
                c = node.get_child_at_index(i)
            except Exception:
                continue
            if c is None:
                continue
            seen += 1
            rec = keep(c)
            if rec is not None:
                found.append((c, rec))
                if len(found) >= max_elements:
                    break
            if depth < 60:
                kids.append((c, depth + 1))
        stack.extend(reversed(kids))
    return found, seen, "walk"


def m_tree(params):
    pid = int(params.get("pid") or 0)
    title = str(params.get("title") or "")
    capture_id = str(params.get("capture_id") or "")
    max_elements = max(1, min(int(params.get("max_elements") or 200), 600))
    win = _find_window(pid, title)
    if win is None:
        return {"found": False, "elements": [],
                "note": "no accessible window for this pid — the app exposes no "
                        "AT-SPI tree (or a11y was just enabled; capture again)"}
    t0 = time.monotonic()
    pairs, seen, how = _collect(win, max_elements)
    objs = [p[0] for p in pairs]
    if capture_id:
        _captures[capture_id] = objs
        _capture_order.append(capture_id)
        while len(_capture_order) > 6:
            _captures.pop(_capture_order.pop(0), None)
    return {
        "found": True,
        "window_title": win.get_name() or "",
        "window_bounds": _bounds(win, _atspi().CoordType.WINDOW),
        "elements": [p[1] for p in pairs],
        "walked": seen,
        "truncated": len(pairs) >= max_elements,
        "method": how,
        "ms": int((time.monotonic() - t0) * 1000),
    }


def _obj(params):
    cap = str(params.get("capture_id") or "")
    idx = int(params.get("index") or 0)
    objs = _captures.get(cap)
    if objs is None:
        raise LookupError("stale: that capture is gone — capture again")
    if not 1 <= idx <= len(objs):
        raise LookupError(f"no element #{idx} in that capture (1..{len(objs)})")
    return objs[idx - 1]


def m_act(params):
    """One accessibility action on an element from a capture."""
    obj = _obj(params)
    kind = str(params.get("kind") or "click")
    value = params.get("value")
    if kind == "click":
        names = _actions(obj)
        iface = obj.get_action_iface()
        for want in CLICK_ACTIONS:
            for i, n in enumerate(names):
                if n.lower() == want:
                    ok = bool(iface.do_action(i))
                    return {"ok": ok, "path": f"atspi_action:{n}"}
        return {"ok": False, "code": "no_action",
                "note": f"element has no click-like action ({names})"}
    if kind == "focus":
        comp = obj.get_component_iface()
        ok = bool(comp and comp.grab_focus())
        return {"ok": ok, "path": "atspi_grab_focus"}
    if kind == "set_text":
        et = obj.get_editable_text_iface()
        if et is None:
            return {"ok": False, "code": "not_editable"}
        ok = bool(et.set_text_contents(str(value or "")))
        got = None
        try:
            t = obj.get_text_iface()
            got = t.get_text(0, t.get_character_count())
        except Exception:
            pass
        return {"ok": ok, "path": "atspi_set_text",
                "verified": got == str(value or "") if got is not None else None}
    if kind == "set_value":
        v = obj.get_value_iface()
        if v is not None:
            try:
                ok = bool(v.set_current_value(float(value)))
                return {"ok": ok, "path": "atspi_set_value",
                        "verified": abs(v.get_current_value() - float(value)) < 1e-6}
            except (TypeError, ValueError):
                pass
        sel = obj.get_selection_iface()
        target = str(value or "").strip().lower()
        if sel is not None:
            n = obj.get_child_count()
            for i in range(min(n, 500)):
                c = obj.get_child_at_index(i)
                if c is not None and (c.get_name() or "").strip().lower() == target:
                    ok = bool(sel.select_child(i))
                    return {"ok": ok, "path": "atspi_select_child"}
            # A combo box keeps its options one level down, in a menu/list.
            for i in range(min(n, 10)):
                c = obj.get_child_at_index(i)
                s2 = c.get_selection_iface() if c is not None else None
                if s2 is None:
                    continue
                for k in range(min(c.get_child_count(), 500)):
                    o = c.get_child_at_index(k)
                    if o is not None and (o.get_name() or "").strip().lower() == target:
                        ok = bool(s2.select_child(k))
                        return {"ok": ok, "path": "atspi_select_child"}
        return {"ok": False, "code": "no_such_value",
                "note": f"no option named {value!r} and no numeric value interface"}
    raise ValueError(f"unknown act kind {kind!r}")


def m_element_state(params):
    obj = _obj(params)
    out = {"states": sorted(_states(obj)),
           "bounds": _bounds(obj, _atspi().CoordType.WINDOW)}
    try:
        t = obj.get_text_iface()
        if t is not None and obj.get_role_name() != "password text":
            out["value"] = t.get_text(0, min(t.get_character_count(), 500))
    except Exception:
        pass
    return out


def m_pointer(params):
    op = str(params.get("op") or "")
    m = _get_mouse()
    if op == "nudge":
        m.nudge()
    elif op == "click":
        m.nudge()
        m.click(str(params.get("button") or "left"), int(params.get("clicks") or 1),
                list(params.get("mods") or []))
    elif op == "down":
        m.nudge()
        m.button(str(params.get("button") or "left"), True)
    elif op == "up":
        m.button(str(params.get("button") or "left"), False)
    elif op == "rel":
        m.rel(int(params.get("dx") or 0), int(params.get("dy") or 0))
    elif op == "wheel":
        m.nudge()
        m.wheel(str(params.get("direction") or "down"), int(params.get("amount") or 3))
    elif op == "release_all":
        m.release_all()
    else:
        raise ValueError(f"unknown pointer op {op!r}")
    return {"ok": True}


def m_ping(params):
    return {"pong": True, "pid": os.getpid()}


METHODS = {name[2:]: fn for name, fn in globals().items()
           if name.startswith("m_") and callable(fn)}


def main() -> int:
    out = sys.stdout
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            continue
        rid = req.get("id")
        fn = METHODS.get(str(req.get("method") or ""))
        try:
            if fn is None:
                raise ValueError(f"unknown method {req.get('method')!r}")
            resp = {"id": rid, "result": fn(req.get("params") or {})}
        except LookupError as exc:
            resp = {"id": rid, "error": str(exc), "code": "stale"}
        except Exception as exc:
            resp = {"id": rid, "error": f"{type(exc).__name__}: {exc}",
                    "trace": traceback.format_exc(limit=3)[-800:]}
        out.write(json.dumps(resp) + "\n")
        out.flush()
    if _mouse is not None:
        _mouse.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
