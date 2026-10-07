"""Run the built frontend in a headless browser and say what it did at runtime.

The `frontend` rung types, builds and unit-tests the frontend. Its own docstring
used to conclude from that that no runtime check was possible, because the
`:5173` dev server serves the LIVE tree — so mid-gate it grades a tree the diff
is not in. That reasoning is right about the dev server and wrong about the
conclusion: the rung already builds the round's own `web/` into a temporary
directory, and THAT is a runtime artifact of the tree under review. Serving it
on a spare loopback port and loading it in headless chromium is a probe of the
round, not of whatever happens to be checked out.

What the probe can tell you, deterministically and with no model call:

* `app-mounted`   — `#root` gained child elements.
* `page-rendered` — the loaded document has any text at all.
* `boundary-fallback` — Mission Control's error boundary is ON SCREEN. This is
  the check that matters, and it is the opposite of the obvious one: `web/src/App.tsx`
  wraps the tree in an `ErrorBoundary` that renders `Something went wrong` and
  logs a console error, so a component that throws on mount leaves `#root`
  NON-empty. A probe that only asked "is the root non-empty" passes a broken app.
* `pageerror`     — an uncaught exception, whatever the page went on to render.
* `console-error` — a message logged at `console.error`.
* `page-load`     — the document never finished loading inside the budget.

Console WARNINGS are counted and never judged. Mission Control's voice path logs
`The AudioContext was not allowed to start` and a LiveKit `mic publish failed:
NotSupportedError` in any headless browser with no user gesture; the live-app
baseline measured 2026-09-27 had zero console errors and twelve warnings, so a
check that treated a warning as failure would false-block on every frontend
landing in the repo.

Screenshots and the evidence artifact are written under the automod STATE dir,
never inside `~/lloyd` (the guardian alerts on runtime data in the checkout) and
never inside the build directory (the rung deletes that). Observe-only, like
`vet`: the verdict is recorded, and the false-block rate over ≥20 real landings
is the number that has to exist before a block is honest.
"""

from __future__ import annotations

import contextlib
import functools
import http.server
import json
import socket
import socketserver
import tempfile
import threading
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

from scripts.automod import state as S

#: What Mission Control's boundary renders when something throws inside it
#: (`web/src/App.tsx`). Matched on the rendered text, because that is what a
#: human sees; a DOM id would be a second thing to break.
BOUNDARY_FALLBACK_TEXT = "Something went wrong"

#: How much of the rendered text to keep. Enough to recognise the app in a verdict
#: months later; a whole Mission Control page is ~13 KB and does not belong in a
#: ledger row.
_VISIBLE_TEXT_CAP = 400

#: How long to let the document reach `load` before calling it a load failure.
#: Generous on purpose: a real Mission Control build fetches on boot.
LOAD_BUDGET_S = 30.0

#: How long after `load` to wait for the first render to land before reading the
#: DOM. Not a settle-the-world wait — `app-mounted` failing is a real finding, so
#: this budget is the difference between "slow" and "broken".
SETTLE_MS = 1500

#: The box's chromium. The one `agent_mcp/browser.py` drives is the playwright
#: build; the venv here is the base one, where `p.chromium.launch()` resolves
#: `chromium` from `PATH`. Passing an explicit path keeps the probe honest about
#: which binary ran, and `unavailable()` turns a missing one into a named skip.
CHROMIUM = "/usr/bin/chromium"

#: `/api/...` paths the real server answers as a server-sent event stream rather
#: than as a JSON document, so a stub must not answer them with JSON.
#: `app/routers/mc_ui.py` returns `StreamingResponse(..., media_type=
#: "text/event-stream")` for `/api/mc/events`, and
#: `web/src/hooks/useMcNavigationEvents.ts` opens it with
#: `new EventSource('/api/mc/events')`. The MIME type is not cosmetic: chromium
#: refuses the connection and logs it as a console ERROR —
#: `EventSource's response has a MIME type ("text/html") that is not
#: "text/event-stream". Aborting the connection.` — which is one of the three
#: console errors in the gate artifact `frontend_probe/SM_20261006_161018.json`,
#: where every one of them is a consequence of there being no server, not of the
#: build (that verdict carried `pageerrors: []` and no boundary fallback). So
#: an API stub that answers this route from its `"*"` JSON default does not merely
#: differ from production: it keeps the load probe red, and through
#: `frontend_layout.maybe_advance` it keeps the layout baseline from ever being
#: blessed.
SSE_PATHS: frozenset[str] = frozenset({"/api/mc/events"})


def probe_build(out_dir: Path, *, chromium: str | None = None,
                settle_ms: int = SETTLE_MS, shots_dir: Path | None = None,
                load_budget_s: float = LOAD_BUDGET_S,
                api_stub: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Load `out_dir` (a `vite build` output) headless and return a verdict.

    `{"skipped": "<why>"}` when the browser or the build is not there; otherwise
    `{"ok": bool, "checks": [<failing checks>], "metrics": {…}, "url": …}`. Every
    failing check is named and carries the text a fresh session needs to act on
    it without re-running anything: the console or page-error string, and — once
    anything has failed — a screenshot path that exists.

    `load_budget_s` and `settle_ms` are parameters for the same reason a test
    wants them: the shipped budgets are tuned for a real app, and a fixture that
    never finishes loading would otherwise cost 30 seconds to prove the branch.

    `api_stub` is handed straight to `serve_build` (#2327). Left None the page
    boots with no server behind it and Mission Control renders
    `Dashboard unavailable:` — a `console-error` verdict about this probe's own
    server, which is what has held the layout baseline on every frontend round so
    far (`maybe_advance` only stores when the probe passed). A caller that
    wants a verdict about the APP passes the frozen fixture
    (`layout_fixture.API_STUB`) — the same object `gate._frontend_probe` passes and
    the one the layout leg beside this probe serves by default.
    """
    why = unavailable(out_dir, chromium)
    if why:
        return {"skipped": why}
    exe = chromium or CHROMIUM
    out_dir = Path(out_dir)
    shots = Path(shots_dir) if shots_dir else (S.STATE_DIR / "frontend_probe")
    shots.mkdir(parents=True, exist_ok=True)

    pageerrors: list[str] = []
    console_errors: list[str] = []
    browser_noise: list[str] = []
    warnings: list[str] = []
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # noqa: BLE001 — an absent dependency is a skip, not a failure
        return {"skipped": f"playwright is not importable ({exc.__class__.__name__})"}

    with contextlib.ExitStack() as stack:
        try:
            base_url = stack.enter_context(serve_build(out_dir, api_stub=api_stub))
            pw = stack.enter_context(sync_playwright())
            browser = pw.chromium.launch(
                executable_path=exe, headless=True,
                args=["--no-sandbox", "--disable-setuid-sandbox",
                      "--disable-dev-shm-usage"])
        except Exception as exc:  # noqa: BLE001 — cannot start is a skip; a crash is a check
            return {"skipped": f"headless chromium could not start ({str(exc)[:180]})"}
        stack.callback(browser.close)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        # Two channels, both needed. An exception that escapes to the top of the
        # module graph goes to `pageerror` WITHOUT appearing in the console, so a
        # probe that listened only for `console` messages would report a page with
        # a dead script tree as healthy.
        page.on("pageerror", lambda e: pageerrors.append(str(e)))
        # The console listener has one exclusion, and it cost a measured bug to
        # find: chromium asks every page it loads for `/favicon.ico`, whether or
        # not the app referenced one, and the resulting 404 is logged as a
        # `console.error`. Measured here on a healthy two-file build:
        #     console.error: Failed to load resource: … 404 (File not found)
        #         @ url http://127.0.0.1:57011/favicon.ico
        # while the three requests the app actually makes all returned 200. A
        # zero-console-errors rule that counted it would have failed on EVERY
        # landing forever — the exact false-block the flake budget exists to
        # prevent, and the reason the probe is recorded before it gates.
        #
        # The exclusion is by the message's own `location.url`, so it cannot
        # swallow a real one: a missing hashed chunk 404s at `/assets/index-*.js`,
        # which is kept and named. And the request is not dropped silently — it is
        # counted into `browser_requested_404s`, so a future reader can see the
        # probe saw it and decided it was the browser's business, not the app's.
        def _see(m) -> None:
            kind = classify_console(m.type, str(m.location.get("url", "")))
            if kind == "error":
                console_errors.append(m.text)
            elif kind == "warning":
                warnings.append(m.text)
            elif kind == "browser-noise":
                browser_noise.append(f"{m.text} @ {m.location.get('url', '')}")
        page.on("console", _see)
        try:
            page.goto(base_url + "/", wait_until="load",
                      timeout=int(load_budget_s * 1000))
        except Exception as exc:  # noqa: BLE001
            # Capture what the browser did see BEFORE deciding the verdict: a load
            # that never completes usually means something on screen is wedged,
            # which is exactly the picture worth keeping. Suppressed, because a
            # page that will not load may also not screenshot, and an instrument
            # that cannot capture must still report the failure — `_fail` then
            # names no screenshot rather than naming one that does not exist.
            with contextlib.suppress(Exception):
                page.screenshot(path=str(shots / "broken.png"), full_page=True)
            return _fail(out_dir, shots, [{"check": "page-load",
                                          "problem": f"the built app never finished loading "
                                                     f"within {load_budget_s:g}s: "
                                                     f"{str(exc)[:200]}",
                                          "url": base_url + "/"}], base_url)
        page.wait_for_timeout(settle_ms)
        visible = page.evaluate("document.body?.innerText ?? ''")
        metrics = {
            "root_children": page.evaluate(
                "document.getElementById('root')?.children.length ?? -1"),
            "body_text_length": len(visible),
            # The text itself, not only its length. A verdict that records the
            # words on screen can be re-read by the next session without a browser;
            # one that records a count cannot distinguish "the app" from "a
            # directory listing" — which is precisely the mistake this probe made
            # once, serving the repo root and calling it a mounted app.
            "visible_text": visible[:_VISIBLE_TEXT_CAP],
            "boundary_fallback_present": BOUNDARY_FALLBACK_TEXT in visible,
            # Whether `#root` exists at all, reported separately from its child
            # count so the verdict can say which of the two DOM conditions the
            # mount assertion was decided under — on the success path too, so an
            # `ok` names what it looked at rather than only that nothing failed.
            "root_present": bool(page.evaluate("!!document.getElementById('root')")),
        }
        checks = evaluate(metrics, pageerrors, console_errors)
        shot = ""
        if checks:
            # Taken once, after the settle window, for any failing verdict — the
            # page is as good as it is going to get by now, and a second picture
            # of the same broken app tells a reader nothing new.
            shot = str(shots / "broken.png")
            with contextlib.suppress(Exception):
                page.screenshot(path=shot, full_page=True)
            if not Path(shot).is_file():
                shot = ""
            checks = [dict(c, screenshot=shot) for c in checks]
    verdict: dict[str, Any] = {"ok": not checks, "checks": checks,
                               "metrics": metrics, "url": base_url,
                               "pageerrors": pageerrors, "console_errors": console_errors,
                               "warnings": warnings,
                               "browser_requested_404s": browser_noise,
                               "screenshot": shot, "build_output": str(out_dir)}
    return verdict


def classify_console(message_type: str, location_url: str = "") -> str:
    """Route one browser console message to `error`, `warning`, or `browser-noise`.

    A pure function so the two decisions that decide this probe's false-block rate
    are testable without a browser, on every run, rather than only when a real
    chromium happens to make the request:

    - **A warning is never an error.** The live app measured 2026-09-27 logs 12
      console messages on mount in a headless browser with no user gesture — 0
      errors, all `warning` (AudioContext autoplay, LiveKit `mic publish failed:
      NotSupportedError`). Treating a warning as failure would refuse every
      frontend landing forever, and the app would be fine every time.
    - **A favicon 404 is the browser's own request.** Chromium asks for
      `/favicon.ico` whether or not the app references one, and the 404 arrives as
      a console *error* — measured on a healthy two-file build, at
      `location.url = http://127.0.0.1:<port>/favicon.ico`, while the app's three
      real requests all returned 200. It is kept, counted, and not judged, so the
      record still shows the probe saw it.

    Any other error URL is kept, including the one that matters most: a hashed
    chunk the build did not emit logs at `/assets/index-*.js`, and that is the
    round's problem, not the browser's.
    """
    if message_type == "warning":
        return "warning"
    if message_type != "error":
        return "ignored"
    if location_url.endswith("/favicon.ico"):
        return "browser-noise"
    return "error"


def unavailable(out_dir: Path | None, chromium: str | None = None) -> str:
    """Why there is nothing to probe, or "" when there is.

    Each reason is a sentence a gate record can carry, because the rung must
    record a named skip rather than pass quietly: `node_modules` and the browser
    are both untracked, and a box that lost one must say so instead of looking
    verified. Checked in this order on purpose — "your build output is missing"
    is a different problem from "this box has no chromium", and the rung has to
    end up blaming the right one.
    """
    if out_dir is None or not Path(out_dir).exists():
        return f"no build output at {out_dir}"
    d = Path(out_dir)
    if not (d / "index.html").exists():
        return f"build output {d} has no index.html"
    if not any(p.suffix in (".js", ".css") for p in d.rglob("*") if p.is_file()):
        return f"build output {d} has no script or stylesheet"
    if not Path(chromium or CHROMIUM).is_file():
        return f"no chromium at {chromium or CHROMIUM}"
    return ""


def evaluate(metrics: dict[str, Any], pageerrors: list[str],
             console_errors: list[str]) -> list[dict[str, Any]]:
    """The deterministic core, as named failing checks. Nothing here sees a pixel.

    Public because the boundary check is a property of THIS app, not of browsers
    generally, and it has to be pinned without hiring a browser to ask it.
    """
    fails: list[dict[str, Any]] = []
    if int(metrics.get("root_children", -1)) <= 0:
        fails.append({"check": "app-mounted",
                      "problem": f"#root rendered {metrics.get('root_children')} child "
                                 f"element(s) — the app did not mount",
                      "selector": "#root"})
    if not int(metrics.get("body_text_length", 0)):
        # An error document, or a page that rendered nothing: no check below can
        # tell a broken app from a broken load off an empty body, and calling this
        # `ok` would be the false pass the flake budget cannot pay for.
        fails.append({"check": "page-rendered",
                      "problem": "the loaded document had no text at all",
                      "selector": "body"})
    if metrics.get("boundary_fallback_present"):
        fails.append({"check": "boundary-fallback",
                      "problem": f"the error boundary is on screen ({BOUNDARY_FALLBACK_TEXT!r}) "
                                 f"— a component threw inside it. #root is non-empty, so "
                                 f"'app-mounted' CANNOT see this",
                      "text": BOUNDARY_FALLBACK_TEXT})
    for e in pageerrors:
        fails.append({"check": "pageerror", "problem": f"uncaught page error: {e}"})
    for c in console_errors:
        fails.append({"check": "console-error", "problem": f"console.error: {c}"})
    return fails


def _fail(out_dir: Path, shots: Path, checks: list[dict], url: str = "") -> dict[str, Any]:
    """A failing verdict from a page we could not read.

    The screenshot is named only when it is really on disk. A record that points
    at a PNG nobody wrote is worse than one that admits there is none: the next
    session opens the path, finds nothing, and cannot tell a capture that failed
    from a capture that was deleted.
    """
    shot = shots / "broken.png"
    path = str(shot) if shot.is_file() else ""
    # Every failing check carries the key, even when the answer is "no capture
    # exists": a reader iterating checks should not have to handle two shapes, and
    # a missing key would be read as "this check has evidence somewhere else".
    return {"ok": False, "checks": [dict(c, screenshot=path) for c in checks],
            "metrics": {}, "url": url, "screenshot": path, "build_output": str(out_dir)}


class _Handler(http.server.SimpleHTTPRequestHandler):
    """Static files, with the SPA's route fallback — and only for routes.

    `resolve_path` is not a method of `SimpleHTTPRequestHandler`; the hook is
    `translate_path`, so an override named for the other is dead code that every
    request walks straight past. Getting the fallback wrong is worse than not
    having it: served blindly, `/assets/index-9f2c.js` becomes an HTML document,
    chromium refuses it for its MIME type, and the probe reports a console error
    that is a lie about the round's diff while the app — healthy or broken —
    never got to speak. So: unknown path with NO extension → `index.html`, the
    way `vite dev` hands a client-side route to the router; unknown path WITH one
    → a real 404, because "the build did not emit this file" and "the app threw"
    are different findings and the status code is what tells them apart.
    """

    def log_message(self, *_args) -> None:      # silence: the gate's stdout is the ledger's
        pass

    #: Set by `serve_build(api_stub=...)`; None means `/api/...` is not served at
    #: all, which is what the load probe has always had.
    api_stub: Mapping[str, Any] | None = None

    def __init__(self, *args, api_stub: Mapping[str, Any] | None = None,
                 **kwargs) -> None:
        # Must be on the instance BEFORE `BaseHTTPRequestHandler.__init__`, which
        # handles the one request this instance exists for and returns done — set
        # it after and every request is answered as if the stub were absent.
        self.api_stub = api_stub
        super().__init__(*args, **kwargs)

    def translate_path(self, path: str = "") -> str:
        found = super().translate_path(path)
        if Path(found).exists():
            return found
        if Path(urlsplit(path).path).suffix:
            return found                       # a missing asset stays a 404
        return super().translate_path("/index.html")

    def _answer_api(self) -> bool:
        """Answer an `/api/...` request from `api_stub`; True if this answered it.

        `sort_keys` so the bytes a page sees are a function of the stub alone: a
        dict literal's own order is not part of its value, and a response whose
        bytes moved because a key was re-ordered is a diff with no cause.

        An event-stream route (`SSE_PATHS`) is the one path the stub does not
        answer from its own mapping — see `_answer_sse`.
        """
        stub = self.api_stub
        if stub is None:
            return False
        path = urlsplit(self.path).path
        if not path.startswith("/api/"):
            return False
        length = int(self.headers.get("Content-Length") or 0)
        if length:                              # drain the body or keep-alive stalls
            self.rfile.read(length)
        if path in SSE_PATHS and self.command == "GET":
            return self._answer_sse()
        key = path if path in stub else ("*" if "*" in stub else None)
        payload = b"{}" if key is None else json.dumps(stub[key], sort_keys=True).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
        return True

    def _answer_sse(self) -> bool:
        """Answer an EventSource poll the way the real route does: as an event stream.

        `Content-Type: text/event-stream` is the entire substance of this method.
        Answering `/api/mc/events` from the stub's `"*"` JSON default instead makes
        chromium log a console ERROR and abort the connection, so the probe would
        report a broken page over a page that only disagreed with the stub about a
        MIME type — and `gate._frontend_probe` keys the layout baseline's bless off
        this verdict passing, so the wrong header was starving a check that has
        never once run with a baseline to compare against.

        One unnamed `data:` frame, then the response is finished. Unnamed because a
        named one would be believed: the hook dispatches `event: navigate` straight
        into `setCurrentTab`, so a frame carrying a real tab would move the rendered
        page and the layout leg would fingerprint wherever it had been sent. No
        listener is attached to the nameless `message` event, so this frame reaches
        the page and changes nothing. Finished rather than held open because
        `BaseHTTPRequestHandler` speaks HTTP/1.0: an ended response is a complete
        body to the client, the hook's `onerror` reconnects and gets the same
        single-frame stream, and no handler thread is left parked inside a server
        the probe has already shut down.
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(b"data: {}\n\n")
        self.wfile.flush()
        return True

    def do_GET(self) -> None:                   # noqa: N802 - stdlib name
        if not self._answer_api():
            super().do_GET()

    def do_POST(self) -> None:                  # noqa: N802 - stdlib name
        if not self._answer_api():
            self.send_error(501, "Unsupported method ('POST')")


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


@contextlib.contextmanager
def serve_build(out_dir: Path, *, api_stub: Mapping[str, Any] | None = None):
    """Serve `out_dir` on a spare loopback port and yield its base URL.

    Port 0, so two gates never choose the same one and no rung has to know what
    `:5173` or the canary ports are doing; loopback only, so nothing on the LAN
    reaches a tree that is under review.

    `api_stub` is for a caller that needs the app to have DATA, which is not the
    same thing as needing it to load. Left None the server answers nothing under
    `/api/`, and Mission Control then boots into `Dashboard unavailable: ...`
    because its 5-second poll fails — the load probe is unaffected (the bundle
    still mounts, which is all it asserts) but a caller reading geometry would be
    reading an error message. When a mapping is passed it is consulted per path,
    with `"*"` as the default, and every `/api/...` request is answered from it as
    JSON — POST included, because the app reports its own tab with one and a
    `501 Unsupported method` there becomes a console error the caller did not ask
    for. Keys are request paths (`/api/dashboard`); values are JSON documents.
    The exception is `SSE_PATHS`, which answer as an event stream whatever the
    mapping says, because that shape is a property of the route and not of the
    data behind it (`_answer_sse`).
    """
    out_dir = Path(out_dir).resolve()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    # `directory` has to be passed explicitly: `SimpleHTTPRequestHandler` defaults
    # it to the process CWD, so without it this serves the working directory of
    # whatever called the probe — under pytest that is the repo root, and the
    # browser then renders a directory LISTING of the checkout and reports it as
    # the app (`#root`'s two children being the listing's `<pre>` and `<table>`,
    # and every script 404ing so no pageerror can ever fire). The probe would have
    # been quietly blind to every broken frontend it was ever pointed at.
    handler = functools.partial(_Handler, directory=str(out_dir),
                              api_stub=api_stub)

    class Bound(_Server):
        def __init__(self) -> None:
            super().__init__(("127.0.0.1", port), handler)

    with Bound() as httpd:
        thread = threading.Thread(target=httpd.serve_forever,
                                  kwargs={"poll_interval": 0.05}, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{port}"
        finally:
            httpd.shutdown()


def summary(verdict: dict[str, Any]) -> str:
    """One line for the rung detail and the ledger. Longest-first on the check
    name so `pageerror` and `console-error` never collapse into each other."""
    if verdict.get("skipped"):
        return f"probe SKIPPED ({verdict['skipped']})"
    checks = verdict.get("checks") or []
    if not checks:
        m = verdict.get("metrics") or {}
        return (f"probe ok ({m.get('root_children')} root children, "
                f"{m.get('body_text_length')} chars of text, no page or console errors)")
    named = ", ".join(c["check"] for c in checks[:4])
    more = f" (+{len(checks) - 4} more)" if len(checks) > 4 else ""
    return f"probe FAILED: {len(checks)} check(s) [{named}{more}] — {checks[0]['problem'][:180]}"


def run(out_dir: Path, *, api_stub: Mapping[str, Any] | None = None,
        **kw: Any) -> dict[str, Any]:
    """`probe_build` with any exception turned into a skip.

    A probe that can raise cannot be observe-only: an unexpected failure inside
    it would fail the rung, which is precisely the flakiness risk the recorded
    first week exists to measure before this check is allowed to stall a landing.

    `api_stub` is spelled out rather than left to `**kw` because it is the one
    keyword the gate passes (#2327): a caller of this signature can see that the
    rung's probe is given data, and a typo in it is a `TypeError` at the call site
    instead of a probe that quietly loads the app shell and calls it a verdict.
    """
    try:
        return probe_build(out_dir, api_stub=api_stub, **kw)
    except Exception as exc:  # noqa: BLE001 — an instrument that breaks says so, it does not fail the build
        return {"skipped": f"probe raised {exc.__class__.__name__}: {str(exc)[:180]}"}


def write_artifact(round_id: str, verdict: dict[str, Any]) -> Path:
    """Keep the evidence of a failing probe out of the checkout and off `gate.json`.

    `gate.json` is copied at landing and then deleted with the worktree, so a
    verdict that lives only in the rung detail has no surviving body — the same
    reason `vet`'s record rides the ledger event. This goes beside the ledger,
    under the automod state dir, and never under `~/lloyd/`.
    """
    import json
    import re
    d = S.STATE_DIR / "frontend_probe"
    d.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", round_id or "unknown")
    path = d / f"{safe}.json"
    path.write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")
    return path


def tmp_build_dir() -> Path:
    """A scratch outDir, kept separate from the rung's own mktemp so a probe
    cannot be reading a directory something else is emptying."""
    return Path(tempfile.mkdtemp(prefix="lloyd-fe-probe-"))
