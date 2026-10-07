"""The runtime probe's own tests (#1601): clauses 1, 2 and 3.

Real headless chromium, real `vite build`-shaped fixtures. Three things make
that tractable and each is load-bearing rather than convenient:

* the fixtures are HTML that does exactly what the probe asserts on — a
  `#root` that gains children, an `ErrorBoundary` that renders the app's real
  fallback copy, a throw that escapes to the `pageerror` channel — so a node
  fails when the CHECK is wrong, not when a bundler is;
* the served directory is the argument to the probe, never a dev server, which
  is the thing the rung must not touch (`:5173` serves the live tree, so a probe
  of it grades a tree the diff is not in);
* `app-mounted` is asserted on a fixture that renders text WITHOUT a root
  element, which the old "is the body non-empty" idea passes.

The skip-path nodes need no browser at all and are not skipped when one is
missing: a probe that cannot run is exactly the situation the skip exists to
describe. Browser nodes call `_require_browser()` instead of carrying a skip-if
MARKER, for a reason that is itself about this check — the honesty prechecks
count five dishonesty spellings in this file's RAW text, prose included, and that
marker's spelling is one of them, unconditionally blocking. A conditional skip in
one helper is the shape the same checker deliberately demotes to advice (#1204),
and it is one occurrence in the file instead of one per node.
"""

from __future__ import annotations

import io
import socket
import tokenize
from types import SimpleNamespace
from pathlib import Path

import pytest

from scripts.automod import frontend_probe as FP


def _require_browser() -> None:
    """Skip, conditionally, when chromium is not on this box.

    The reason is the claim: a browser question nobody can answer here must not
    be answered by a green row.
    """
    if not Path(FP.CHROMIUM).is_file():
        pytest.skip(f"no chromium at {FP.CHROMIUM} — a browser question cannot be "
                    f"answered on this box, and this file must not report that it was")


# The fixtures are the shapes the rung exists to catch, in the DOM the real app
# produces. `MOUNTED` is what a healthy Mission Control boot looks like from
# outside; the throw is what `web/src/App.tsx`'s boundary leaves behind.
MOUNTED = """
  const root = document.getElementById('root');
  const panel = document.createElement('div');
  panel.className = 'panel';
  panel.textContent = 'Mission Control — sessions, autonomy, GPU, voice';
  root.appendChild(panel);
"""

# Healthy, by the probe's own definition, and it warns: the voice path's
# AudioContext and LiveKit messages measured on the live app on 2026-09-27. A
# warning must never fail a check, so every warning here is a live trap for an
# over-eager future edit.
WARNINGS = """
  console.warn('The AudioContext was not allowed to start. It must be resumed '
               + '(or created) after a user gesture on the page.');
  console.warn('mic publish failed: NotSupportedError');
  console.info('LiveKit: connecting (this is info, not an error)');
"""

# The throw clause 1 needs: the boundary has ALREADY rendered its fallback (so
# `#root` is NON-empty — that is the whole of clause 3), it logged through the
# app's own console path, and the error then escapes to the `pageerror` channel,
# whose text is what clause 1 requires the verdict to carry.
THROW_INSIDE_BOUNDARY = """
  const root = document.getElementById('root');
  root.innerHTML = '<h2>Something went wrong</h2><p>Try reloading the page.</p>';
  const err = new TypeError('Cannot read properties of undefined (reading volume)');
  console.error('ErrorBoundary caught an error:', err);
  throw err;
"""

# An error with no boundary above it: nothing extra renders, nothing is logged,
# the exception simply escapes. The `pageerror` channel is the only witness.
UNCAUGHT_ONLY = """
  const root = document.getElementById('root');
  const panel = document.createElement('div');
  panel.textContent = 'Mission Control';
  root.appendChild(panel);
  throw new Error('boom from a module top level');
"""

# Text on the page, `#root` absent. The one shape a "body is non-empty" check
# passes and an `app-mounted` check refuses.
TEXT_WITHOUT_ROOT = """
  document.body.innerHTML = '<h1>Mission Control</h1><p>417x</p>';
"""

# Nothing at all: no children, no text. The document loaded; it rendered nothing.
RENDERS_NOTHING = """
  /* a bundle that boots, mounts nothing, and says nothing */
"""


def _build(tmp_path: Path, script: str, *, root_element: bool = True,
           extra: str = "", name: str = "build", root_inner: str = "") -> Path:
    """A directory shaped like a `vite build` output: one hashed asset and an
    `index.html` that references it the way the real one does, with an absolute
    `/assets/...` path so the probe's own static serving is what resolves it."""
    out = tmp_path / name
    out.mkdir(parents=True, exist_ok=True)
    body = (f'<div id="root">{root_inner or "<!-- app mounts here -->"}</div>'
            if root_element else '<!-- no root element at all -->')
    (out / "index.html").write_text(
        '<!doctype html>\n<html><head><meta charset="utf-8">'
        '<link rel="stylesheet" href="/assets/index-abc123.css"></head>\n'
        f"<body>{body}{extra}"
        '<script type="module" src="/assets/index-abc123.js"></script>'
        "</body></html>\n", encoding="utf-8")
    (out / "assets").mkdir(exist_ok=True)
    (out / "assets" / "index-abc123.css").write_text(".panel{display:grid}\n",
                                                     encoding="utf-8")
    (out / "assets" / "index-abc123.js").write_text(script, encoding="utf-8")
    return out


def _shots(tmp_path: Path) -> Path:
    d = tmp_path / "shots"
    d.mkdir(exist_ok=True)
    return d


# ── clause 1: a throwing mount is a failing check that carries the error ────

def test_a_component_that_throws_on_mount_produces_a_failing_check_with_its_error(tmp_path):
    """Clause 1, with clause 3's mechanism in the same fixture: the boundary has
    ALREADY rendered `Something went wrong` into `#root`, so the root is non-empty
    and the naive "did anything mount" question answers yes. The thrown message has
    to be in the verdict — a check that only says `boundary-fallback` gives a
    future session nothing to fix, and the item's acceptance is that every block
    carries evidence readable without re-running the probe.
    """
    _require_browser()
    shots = _shots(tmp_path)
    v = FP.run(_build(tmp_path, THROW_INSIDE_BOUNDARY), shots_dir=shots,
               chromium=FP.CHROMIUM, settle_ms=300)
    assert v["ok"] is False, v
    names = [c["check"] for c in v["checks"]]
    assert "boundary-fallback" in names, names
    assert "pageerror" in names, f"the throw must reach the pageerror channel: {v}"
    text = " | ".join(c["problem"] for c in v["checks"])
    assert "Cannot read properties of undefined" in text, text
    assert v["metrics"]["root_children"] >= 1, (
        "the root WAS non-empty — which is why a root-emptiness check is clause 3")
    for c in v["checks"]:
        assert c["problem"], c
        assert c["screenshot"] and Path(c["screenshot"]).is_file(), (
            f"check {c['check']} names no screenshot that exists: {c}")
    assert v["screenshot"] and Path(v["screenshot"]).is_file(), v


def test_an_uncaught_runtime_error_is_caught_by_the_pageerror_channel(tmp_path):
    """The channel the console is not. Nothing renders wrong and nothing is
    logged, so a probe listening only for console output would call this page
    healthy — `#root` has its panel and there is no fallback on screen.
    """
    _require_browser()
    v = FP.run(_build(tmp_path, UNCAUGHT_ONLY), shots_dir=_shots(tmp_path),
               chromium=FP.CHROMIUM, settle_ms=300)
    assert v["ok"] is False, v
    assert [c["check"] for c in v["checks"]] == ["pageerror"], v["checks"]
    assert "boom from a module top level" in v["checks"][0]["problem"], v["checks"]
    assert v["console_errors"] == [], "nothing was logged — this is the pageerror channel"


def test_a_page_that_renders_nothing_is_a_failing_check(tmp_path):
    """The document loaded and produced nothing: no children, no text. That is a
    finding, not an empty pass, and the two named checks say which half failed."""
    _require_browser()
    v = FP.run(_build(tmp_path, RENDERS_NOTHING), shots_dir=_shots(tmp_path),
               chromium=FP.CHROMIUM, settle_ms=300)
    assert v["ok"] is False, v
    assert [c["check"] for c in v["checks"]] == ["app-mounted", "page-rendered"], v["checks"]
    assert v["metrics"]["root_children"] == 0 and v["metrics"]["body_text_length"] == 0, v


def test_a_document_that_never_finishes_loading_is_reported_honestly(tmp_path):
    """The `page-load` branch, and it is a real hang rather than a mock: the
    fixture references an image on a port that accepts the connection and never
    answers, so the document genuinely cannot reach `load` and the browser's own
    timeout fires — the same code path a wedged boot takes on a real landing.

    `load_budget_s=0.6` is why this costs a second and not thirty. The shipped
    budget is generous because Mission Control fetches on boot; shrinking it here
    reaches the branch faster than faking it would, and what is asserted is the
    branch, not the number.
    """
    _require_browser()
    shots = _shots(tmp_path)
    stall = socket.socket()
    stall.bind(("127.0.0.1", 0))       # bound and never accepted → a black hole, not
    stall.listen(1)                    # a refused connection, which would load fine
    port = stall.getsockname()[1]
    try:
        build = _build(tmp_path, MOUNTED,
                       extra=f'<img src="http://127.0.0.1:{port}/hang.png">')
        v = FP.run(build, shots_dir=shots, chromium=FP.CHROMIUM, settle_ms=200,
                   load_budget_s=0.6)
    finally:
        stall.close()
    assert v["ok"] is False, v
    assert [c["check"] for c in v["checks"]] == ["page-load"], v["checks"]
    assert "never finished loading" in v["checks"][0]["problem"], v["checks"]
    # The check names a screenshot only when the capture happened, and it agrees
    # with the verdict's own field: a page that will not load may also not
    # screenshot, and naming a file that is not there is the false claim #721 is
    # about.
    assert v["checks"][0]["screenshot"] == v["screenshot"], v


# ── clause 2: the positive control ─────────────────────────────────────────

def test_console_messages_are_routed_by_the_pure_classifier():
    """The two routing decisions, pinned without a browser — because a browser
    cannot be relied on to make the request at the moment a test runs, and both
    decisions decide this probe's false-block rate.

    Measured live 2026-09-27 on the healthy app: the AudioContext and LiveKit
    messages arrive as `type == "warning"`, and chromium's `/favicon.ico` 404
    arrives as `type == "error"` at
    `location.url = http://127.0.0.1:57011/favicon.ico`, while the three requests
    the app itself made all returned 200. The first must not reach the error
    channel, the second must not be judged, and a missing hashed chunk — the
    failure this rung exists to catch — must be kept.
    """
    assert FP.classify_console("warning", "https://app/index.html") == "warning", (
        "a warning is a warning wherever it is logged from; the voice path warns "
        "from the page itself")
    assert FP.classify_console("error", "http://127.0.0.1:57011/favicon.ico") == \
        "browser-noise", "chromium asked for that file, the app never did"
    assert FP.classify_console("error", "http://127.0.0.1/assets/index-9f2c.js") == \
        "error", "a chunk the build never emitted IS the round's problem"
    assert FP.classify_console("error", "") == "error", (
        "an error with no URL is judged, not excused — the exclusion is by URL")
    assert FP.classify_console("error", "http://127.0.0.1/assets/favicon-helper.js") \
        == "error", "a real module whose name merely contains the word is judged too"
    assert FP.classify_console("log", "http://127.0.0.1/") == "ignored"


def test_a_healthy_build_produces_zero_failing_checks_even_though_it_warns(tmp_path):
    """Clause 2. The same probe, the same fixture machinery, zero failing checks
    — which is what stops a probe that always fails from passing clause 1. The
    warnings are the trap: the voice path's AudioContext and LiveKit messages are
    present on the live app in a headless browser with no gesture, so treating a
    warning as failure would false-block every frontend landing there is."""
    _require_browser()
    v = FP.run(_build(tmp_path, MOUNTED + WARNINGS), shots_dir=_shots(tmp_path),
               chromium=FP.CHROMIUM, settle_ms=300)
    assert v["ok"] is True, v
    assert v["checks"] == [], v["checks"]
    assert v["metrics"]["root_children"] == 1 and v["metrics"]["body_text_length"] > 0, v
    assert v["metrics"]["boundary_fallback_present"] is False, v
    assert FP.summary(v).startswith("probe ok ("), FP.summary(v)


def test_mounted_text_without_a_root_element_is_not_reported_healthy(tmp_path):
    """The narrower half of clause 2's job: the positive control must not be a
    text check in disguise. This page has real text and no `#root`, which is the
    shape a "body is non-empty" probe blesses.
    """
    _require_browser()
    v = FP.run(_build(tmp_path, TEXT_WITHOUT_ROOT, root_element=False),
               shots_dir=_shots(tmp_path), chromium=FP.CHROMIUM, settle_ms=300)
    assert v["ok"] is False, v
    assert [c["check"] for c in v["checks"]] == ["app-mounted"], v["checks"]
    assert v["metrics"]["body_text_length"] > 0, "the text really is there"


# ── clause 3: the boundary check, pinned without a browser too ─────────────

def test_the_boundary_fallback_is_a_failing_check_though_the_root_is_full(tmp_path):
    """Clause 3, live: the throw is caught by a boundary that renders the app's
    fallback, so `#root` has children, nothing escapes and nothing is logged —
    `app-mounted` passes and ONLY the boundary check fires.
    """
    _require_browser()
    caught = ("const root = document.getElementById('root');"
              "root.innerHTML = '<h2>Something went wrong</h2>';"
              "try { null.boom; } catch (e) { console.warn('boundary caught it'); }")
    v = FP.run(_build(tmp_path, caught), shots_dir=_shots(tmp_path),
               chromium=FP.CHROMIUM, settle_ms=300)
    assert v["ok"] is False, v
    assert [c["check"] for c in v["checks"]] == ["boundary-fallback"], v["checks"]
    assert v["metrics"]["root_children"] >= 1, "the root is full, as the real app's is"
    assert v["metrics"]["boundary_fallback_present"] is True, v["metrics"]
    assert v["pageerrors"] == [] and v["console_errors"] == [], (
        "the error was caught, so only the rendered fallback can see it")


def test_the_boundary_check_names_the_app_copy_it_looks_for():
    """Clause 3, without a browser. `app-mounted` is asserted as PASSING while the
    boundary check fires, which is the whole reason the boundary check exists: the
    two facts coexist on one page. The copy is compared to `web/src/App.tsx` so the
    probe cannot drift from what the app renders without a test reddening.
    """
    fails = FP.evaluate({"root_children": 3, "body_text_length": 417,
                         "boundary_fallback_present": True}, [], [])
    assert [f["check"] for f in fails] == ["boundary-fallback"], fails
    assert FP.BOUNDARY_FALLBACK_TEXT in fails[0]["problem"]
    healthy = FP.evaluate({"root_children": 3, "body_text_length": 417,
                           "boundary_fallback_present": False}, [], [])
    assert healthy == [], healthy

    src = Path(__file__).resolve().parent.parent / "web" / "src" / "App.tsx"
    if src.exists():
        assert FP.BOUNDARY_FALLBACK_TEXT in src.read_text(encoding="utf-8"), (
            "the probe looks for copy the app no longer renders, which would make "
            "the check permanently silent")


def test_an_uncaught_error_and_a_logged_error_are_two_named_checks():
    """The pure evaluation again: both channels, each named, and `problem` carries
    the text itself rather than a count of it.
    """
    fails = FP.evaluate({"root_children": 1, "body_text_length": 900,
                         "boundary_fallback_present": False},
                        ["TypeError: x is not a function", "ReferenceError: y"],
                        ["Failed to fetch /api/sessions: 500"])
    assert [f["check"] for f in fails] == ["pageerror", "pageerror", "console-error"], fails
    assert "TypeError: x is not a function" in fails[0]["problem"]
    assert "Failed to fetch /api/sessions: 500" in fails[2]["problem"]


# ── clause 5's mechanism, at the probe end: named skips ────────────────────

def test_an_absent_browser_is_a_named_reason_and_never_an_ok(tmp_path):
    """A missing browser is the one situation in which a probe may say nothing
    about the app, and it has to say THAT. `ok` absent, not `ok: True`: a caller
    that asks `not verdict.get("ok")` would read a skip carrying `ok` as a pass,
    and one that asks `verdict.get("ok") is False` would read it as a failure.
    """
    build = _build(tmp_path, MOUNTED, name="built-but-unbrowsed")
    missing = str(tmp_path / "no-such-chromium")
    assert "chromium" in FP.unavailable(build, chromium=missing), FP.unavailable(build)
    v = FP.run(build, shots_dir=_shots(tmp_path), chromium=missing)
    assert "chromium" in v["skipped"], v
    assert "ok" not in v, "a probe that could not run must not report ok"
    assert FP.summary(v) == f"probe SKIPPED ({v['skipped']})", FP.summary(v)


def test_a_missing_build_is_skipped_without_reaching_a_browser(tmp_path):
    """Named skip, and no browser needed to say so: the build directory is checked
    first, so a broken landing is never blamed on a missing chromium."""
    v = FP.run(tmp_path / "no-such-build-dir", shots_dir=_shots(tmp_path))
    assert "no build output" in v["skipped"], v
    assert "ok" not in v, v


def test_a_build_output_with_no_index_html_is_named_as_such(tmp_path):
    d = tmp_path / "half-a-build"
    d.mkdir()
    (d / "assets").mkdir()
    (d / "assets" / "index-x.js").write_text("console.log(1)")
    v = FP.run(d, shots_dir=_shots(tmp_path))
    assert "index.html" in v["skipped"], v


def test_a_build_output_with_no_assets_is_named_as_such(tmp_path):
    """An `index.html` with no script is a build that lied about its output. Not
    probed, and the reason says so rather than reporting a healthy empty page."""
    d = tmp_path / "shell-only"
    d.mkdir()
    (d / "index.html").write_text('<!doctype html><div id="root"></div>')
    v = FP.run(d, shots_dir=_shots(tmp_path))
    assert "no script or stylesheet" in v["skipped"], v


# ── the rule that decides the whole design ─────────────────────────────────

def test_the_served_page_is_the_build_output_it_was_given(tmp_path):
    """The clause-3 half of "does the probe look at the right tree": the verdict's
    own metrics must name the build it was handed.

    This node exists because of a bug the first version of this file could not
    catch. `serve_build` originally left `SimpleHTTPRequestHandler`'s `directory`
    at its default — the process CWD — so under pytest it served a directory
    LISTING of the checkout. Every assertion the old node made still passed:
    `#root` is absent in a listing, so `root_children` is -1 and the empty-root
    check stays silent; the listing has non-empty text; no script runs, so no error
    can fire; and the negative assertion ("no `live-tree-marker` in the text") was
    true of a listing too. The probe would have reported `probe ok` on every
    landing forever while never having seen the app.

    So the assertion is POSITIVE and it is about content the build wrote: the
    recorded `visible_text` must contain this build's marker. A check that cannot
    distinguish "verified" from "looking at the wrong thing" is worse than no
    check, because it is trusted.
    """
    _require_browser()
    built = _build(tmp_path, MOUNTED, name="built-tree", root_element=True,
                   root_inner="built-tree-marker")
    v = FP.run(built, shots_dir=_shots(tmp_path), chromium=FP.CHROMIUM, settle_ms=300)
    assert v["ok"] is True, v
    assert "built-tree-marker" in v["metrics"]["visible_text"], (
        "the probe graded something other than the build it was handed — this is "
        "the field that proves which tree was loaded", v["metrics"])
    assert v["metrics"]["root_present"] is True, (
        "the fixture has a #root; an `ok` without one means the mount assertion "
        "was never the thing being satisfied", v["metrics"])
    assert v["checks"] == [], v["checks"]
    assert v["screenshot"] == "", (
        "a healthy probe captures nothing: no check failed, so there is nothing to "
        "claim evidence about")
    assert v["build_output"] == str(built.resolve()), v["build_output"]


def test_a_missing_bundle_is_reported_as_a_404_and_not_a_mime_type_lie(tmp_path):
    """The same contract one level up, through a real browser: what a reader of the
    verdict sees when the build is short a file.

    The app cannot mount (the script that would have mounted it is absent), so
    `app-mounted` must fail, and the console must say the honest thing — a 404 for
    the URL that was asked for — rather than a MIME-type refusal that would point
    the next session at the probe's server instead of at the build.
    """
    _require_browser()
    build = tmp_path / "short-a-file"
    (build / "assets").mkdir(parents=True)
    # A real stylesheet sits beside the missing chunk on purpose: the probe
    # refuses to open a build directory with no asset in it at all ("the build
    # lied"), and that verdict would have answered the wrong question. Here the
    # build did produce files — it just did not produce THIS one.
    (build / "assets" / "existing.css").write_text(".panel{display:grid}")
    (build / "index.html").write_text(
        "<!doctype html><html><head>"
        '<link rel="stylesheet" href="/assets/existing.css"></head>'
        "<body><div id=\"root\"></div>"
        '<script type="module" src="/assets/index-9f2c.js"></script></body></html>')
    v = FP.run(build, shots_dir=_shots(tmp_path), chromium=FP.CHROMIUM, settle_ms=300)
    named = [c["check"] for c in v["checks"]]
    assert "app-mounted" in named, v["checks"]
    text = " | ".join(c["problem"] for c in v["checks"])
    assert "404" in text, f"the missing file must be named as a missing file: {text}"
    assert "MIME" not in text and "text/html" not in text, (
        f"the probe answered a script with the app shell: {text}")


def test_a_missing_asset_is_a_404_while_an_unknown_route_gets_the_shell(tmp_path):
    """The static server's own contract, testable without a browser: an unknown
    ROUTE gets `index.html` (that is how `/sessions/7` reaches the client-side
    router), and an unknown ASSET gets a real 404.

    Applying the shell to a hashed chunk is not cosmetic. Chromium asks for
    `/assets/index-9f2c.js`, is handed an HTML document, and refuses it for its
    MIME type — so the probe would log `Refused to execute script … because its
    MIME type ('text/html') is not executable`, an error describing the probe's own
    server rather than the round's diff, while the truth (the build never emitted
    the file) went unreported. A 404 is the same fact in a channel that cannot be
    misread.
    """
    build = tmp_path / "served"
    (build / "assets").mkdir(parents=True)
    (build / "index.html").write_text('<!doctype html><html><body><div id="root">shell'
                                      "</div></body></html>")
    with FP.serve_build(build) as url:
        route = _get(url + "/sessions/7")
        assert route.status == 200 and "shell" in route.body, route
        asset = _get(url + "/assets/index-9f2c.js")
        assert asset.status == 404, asset
        assert "shell" not in asset.body, "the SPA shell answered in place of a script"


def test_a_build_short_one_file_reports_a_missing_file_and_not_a_mime_type_lie(tmp_path):
    """The same contract through a real browser: what a reader of the verdict sees
    when the build is short one file.

    The app cannot mount (the script that would have mounted it is absent), so
    `app-mounted` must fail, and the console must name the missing file rather than
    complain about MIME types — the first sentence sends the next session to the
    build, the second sends them to this probe's server.
    """
    _require_browser()
    build = tmp_path / "short-a-file"
    (build / "assets").mkdir(parents=True)
    # A real stylesheet beside the missing chunk on purpose: the probe refuses to
    # open a build directory with no asset in it at all ("the build lied"), and
    # that verdict would have answered a different question. Here the build did
    # produce files; it just did not produce this one.
    (build / "assets" / "existing.css").write_text(".panel{display:grid}")
    (build / "index.html").write_text(
        '<!doctype html><html><head><link rel="stylesheet" href="/assets/existing.css">'
        '</head><body><div id="root"></div>'
        '<script type="module" src="/assets/index-9f2c.js"></script></body></html>')
    v = FP.run(build, shots_dir=_shots(tmp_path), chromium=FP.CHROMIUM, settle_ms=300)
    named = [c["check"] for c in v["checks"]]
    assert "app-mounted" in named, v["checks"]
    text = " | ".join(c["problem"] for c in v["checks"])
    assert "404" in text, f"the missing file must be named as a missing file: {text}"
    assert "MIME" not in text and "text/html" not in text, (
        f"the probe answered a script request with the app shell: {text}")


def _get(url: str):
    """A GET that reports a 404 as a result, not an exception: the status code is
    the thing under test, and `urlopen` raises on exactly it."""
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return SimpleNamespace(status=r.status, body=r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        return SimpleNamespace(status=e.code, body=e.read().decode("utf-8", "replace"))


def test_the_probe_never_names_the_dev_server_port():
    """A module-level rule, cheaply enforced: the dev server serves the live tree,
    so mid-gate it grades a tree the diff is not in. Not one reference in the
    probe's CODE — its prose may still explain why, which is what the comments in
    that file are for.
    """
    src_path = (Path(__file__).resolve().parent.parent / "scripts" / "automod"
                / "frontend_probe.py")
    code = _code_only(src_path.read_text(encoding="utf-8"))
    assert "5173" not in code, (
        "the dev-server port appears in executable code, not prose")


def _code_only(src: str) -> str:
    """The source with every string literal and comment blanked, via `tokenize`.

    The rule being tested is about *code*, and the file has good reason to discuss
    the port in prose, so the check needs a prose-blind view of it. Line-wise
    filtering and quote-counting both get this wrong — a docstring spanning lines,
    an apostrophe in prose — and a check whose view of the file is unreliable
    reports a violation that isn't there, which is the failure mode this whole
    probe exists to avoid.
    """
    out: list[str] = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        out.append(tok.string)
    return "".join(out)


# ── #2327: the probe's server has to be able to answer the API ──────────────
#
# Every node above runs against a build with nothing behind `/api/`, which is what
# `probe_build` has always handed `serve_build`: no `api_stub`, so the SPA shell
# answers a dashboard poll. That is a HOLLOW page, and the gate's own artifact for
# round SM_20261006_161018 shows what it costs — `ok: false` with three console
# errors and no page errors at all:
#
#     Failed to load sessions: SyntaxError: Unexpected token '<', "<!doctype "…
#     EventSource's response has a MIME type ("text/html") that is not
#         "text/event-stream". Aborting the connection.
#     Failed to load resource: the server responded with a status of 501
#         (Unsupported method ('POST'))
#
# All three are server absence, not the round's diff, and the rendered text opens
# `Dashboard unavailable:`. The layout leg beside this probe already fingerprints a
# POPULATED page (`frontend_layout._served` → `FP.serve_build(out_dir,
# api_stub=API_STUB)`), so the leg was reading the app while the load probe next to
# it read an error message — and because the rung's bless keys off `verdict["ok"]`
# (`gate._frontend_probe` → `maybe_advance(advanced=bool(verdict.get("ok")))`), a
# probe that is red on server absence also holds the layout baseline forever.
# These nodes pin the stub being wired through, and pin that the detection above
# still fires when the data is there.

#: A stub whose marker cannot appear in an `index.html`: the only way the string
#: reaches the rendered text is the stub's JSON being fetched and parsed by the
#: page. A green verdict that could also come from the shell would prove nothing.
PROBE_STUB: dict = {"/api/dashboard": {"marker": "stub-dashboard-9f2c"}, "*": {}}

#: What Mission Control actually does on boot, in the two channels that decide
#: this item: a `fetch('/api/dashboard')` whose failure it logs, and
#: `new EventSource('/api/mc/events')` (`useMcNavigationEvents.ts:38`). The real
#: server answers that second one as a stream (`app/routers/mc_ui.py:233`,
#: `StreamingResponse(..., media_type="text/event-stream")`).
API_CONSUMER = """
  fetch('/api/dashboard').then(r => r.json()).then(j => {
    const root = document.getElementById('root');
    const panel = document.createElement('div');
    panel.className = 'panel';
    panel.textContent = 'Dashboard marker=' + j.marker;
    root.appendChild(panel);
  }, e => console.error('Failed to load dashboard: ' + e));
  new EventSource('/api/mc/events');
"""

# The event stream on its own, mount done synchronously: the one thing this node
# is asking is whether chromium aborts the connection, so nothing else may be in
# flight while it answers.
EVENTS_ONLY = """
  new EventSource('/api/mc/events');
  const root = document.getElementById('root');
  const panel = document.createElement('div');
  panel.className = 'panel';
  panel.textContent = 'Mission Control — event stream opened';
  root.appendChild(panel);
"""

# A component that fails inside the app, logged the way the app logs it.
SEEDED_CONSOLE_ERROR = """
  console.error('TypeError: cannot read properties of undefined (reading volume)');
"""


def test_the_probe_hands_its_api_stub_to_the_page_it_loads(tmp_path):
    """Clause 1: `probe_build` and `run` take an `api_stub` and pass it to
    `serve_build`, so a build that fetches `/api/dashboard` renders the stub's JSON
    rather than the SPA's `index.html`.

    The assertion is about content only the stub can put on screen, because the
    failure this thread (line 115 of the probe) shipped with is a green-looking
    load of a page that never got its data: an HTTP-level check on `serve_build`
    would have passed while the probe kept calling it with no stub at all.
    """
    _require_browser()
    v = FP.run(_build(tmp_path, API_CONSUMER, name="stubbed"),
               shots_dir=_shots(tmp_path), chromium=FP.CHROMIUM, settle_ms=800,
               api_stub=PROBE_STUB)
    assert v["ok"] is True, v
    assert v["checks"] == [], v["checks"]
    assert "stub-dashboard-9f2c" in v["metrics"]["visible_text"], (
        "the page did not render the stub's JSON — which is the whole of the "
        f"`Dashboard unavailable:` artifact this item is about: {v['metrics']}")
    assert v["pageerrors"] == [] and v["console_errors"] == [], v


def test_the_same_build_without_a_stub_is_red_on_the_fetch_the_stub_answers(tmp_path):
    """The control the node above is measured against — and it is the gate's
    verdict today: `serve_build(out_dir)` with no stub answers `/api/dashboard`
    with the app shell, `r.json()` chokes on `<!doctype`, and the probe reports a
    `console-error` for an app that is not broken.

    If this node ever goes green, the probe has stopped reporting absence as
    absence, and the green verdict above stops proving anything.
    """
    _require_browser()
    v = FP.run(_build(tmp_path, API_CONSUMER, name="unstubbed"),
               shots_dir=_shots(tmp_path), chromium=FP.CHROMIUM, settle_ms=800)
    assert v["ok"] is False, v
    assert "console-error" in [c["check"] for c in v["checks"]], v["checks"]
    assert "stub-dashboard-9f2c" not in v["metrics"]["visible_text"], v["metrics"]


def test_the_stub_serves_the_event_route_as_an_event_stream_not_json(tmp_path):
    """Clause 3, at the HTTP level and with no browser: `/api/mc/events` answers
    `text/event-stream` with at least one `data:` frame.

    Before this change the `"*"` catch-all answered it `application/json`, and
    chromium's EventSource refused the MIME type — a console ERROR, which is why
    threading the stub alone would not have unblocked anything. The two sibling
    assertions are the same answer not spreading where it should not: a dashboard
    poll is still JSON, and a path the fixture does not name is still the `"*"`
    JSON default. The POST assertion is the artifact's third error — the same stub
    turned that `501 Unsupported method` into a 200.
    """
    build = tmp_path / "served"
    (build / "assets").mkdir(parents=True)
    (build / "index.html").write_text('<!doctype html><div id="root">shell</div>')
    with FP.serve_build(build, api_stub=PROBE_STUB) as url:
        events = _request(url, "/api/mc/events")
        assert events.status == 200, events
        assert events.content_type.startswith("text/event-stream"), (
            f"an EventSource client aborts on anything else: {events}")
        assert "data:" in events.body, f"the stream carried no frame: {events}"
        dashboard = _request(url, "/api/dashboard")
        assert dashboard.content_type.startswith("application/json"), dashboard
        assert "stub-dashboard-9f2c" in dashboard.body, dashboard
        other = _request(url, "/api/sessions")
        assert other.content_type.startswith("application/json"), other
        assert other.body.strip() == "{}", f"`*` is the default: {other}"
        post = _request(url, "/api/mc/navigation", method="POST")
        assert post.status == 200, f"the app posts its own tab; 501 became an error: {post}"


def test_the_pages_eventsource_does_not_abort_on_the_stub(tmp_path):
    """Clause 3, in the browser that has to believe it: opening
    `new EventSource('/api/mc/events')` against the stub-served build logs NO
    console error.

    This is the node that fails on a JSON answer, which is the half of the item
    that the plumbing alone cannot fix: the MIME abort was one of the three console
    errors in the gate artifact, and it survives every correct-looking `api_stub`
    that keeps answering the route with JSON.
    """
    _require_browser()
    v = FP.run(_build(tmp_path, EVENTS_ONLY, name="events-stubbed"),
               shots_dir=_shots(tmp_path), chromium=FP.CHROMIUM, settle_ms=800,
               api_stub=PROBE_STUB)
    assert v["console_errors"] == [], (
        "a stream answered with the wrong MIME type makes chromium log an ERROR and "
        f"abort the connection — that message, not the app: {v['console_errors']}")
    assert v["ok"] is True, v


def test_a_logged_console_error_still_fails_the_probe_with_the_stub_served(tmp_path):
    """Clause 4: the stub must not buy a green verdict over a broken app. The
    marker in the rendered text proves the data arrived; the named `console-error`
    proves the seeded failure was still seen.
    """
    _require_browser()
    v = FP.run(_build(tmp_path, API_CONSUMER + SEEDED_CONSOLE_ERROR,
                      name="stubbed-but-broken"),
               shots_dir=_shots(tmp_path), chromium=FP.CHROMIUM, settle_ms=800,
               api_stub=PROBE_STUB)
    assert v["ok"] is False, v
    names = [c["check"] for c in v["checks"]]
    assert "console-error" in names, names
    assert "reading volume" in " | ".join(c["problem"] for c in v["checks"]), v["checks"]
    assert "stub-dashboard-9f2c" in v["metrics"]["visible_text"], (
        "the stub was served, so this failure is the app's and not the missing "
        f"server: {v['metrics']}")


def test_a_rendered_error_boundary_still_fails_the_probe_with_the_stub_served(tmp_path):
    """Clause 4, the check that only the boundary can make, with data on screen.
    `#root` is full, the dashboard JSON is rendered below it, and the fallback copy
    is what decides the verdict.
    """
    _require_browser()
    caught = ("const root = document.getElementById('root');"
              "root.innerHTML = '<h2>Something went wrong</h2>';"
              "try { null.boom; } catch (e) { console.warn('boundary caught it'); }")
    v = FP.run(_build(tmp_path, caught + API_CONSUMER, name="stubbed-boundary"),
               shots_dir=_shots(tmp_path), chromium=FP.CHROMIUM, settle_ms=800,
               api_stub=PROBE_STUB)
    assert v["ok"] is False, v
    names = [c["check"] for c in v["checks"]]
    assert "boundary-fallback" in names, names
    assert "stub-dashboard-9f2c" in v["metrics"]["visible_text"], (
        f"the stub was served under the boundary: {v['metrics']}")


def _request(url: str, path: str, *, method: str = "GET"):
    """A request that reports headers, not just a body, and never raises.

    The content type IS the thing under test in clause 3, and `urlopen` hides it
    behind a case-insensitive mapping while also raising on a 4xx/5xx — which is
    exactly the status the stub used to answer a POST with.
    """
    import http.client
    from urllib.parse import urlsplit

    s = urlsplit(url)
    conn = http.client.HTTPConnection(s.hostname, s.port, timeout=10)
    try:
        conn.request(method, path,
                     body=b"{}" if method == "POST" else None,
                     headers={"Content-Type": "application/json"}
                     if method == "POST" else {})
        r = conn.getresponse()
        return SimpleNamespace(status=r.status,
                               content_type=r.getheader("Content-Type") or "",
                               body=r.read().decode("utf-8", "replace"))
    finally:
        conn.close()
