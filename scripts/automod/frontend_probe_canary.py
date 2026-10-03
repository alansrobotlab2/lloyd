"""Measure the frontend runtime probe against builds that are KNOWN to be broken.

The probe (`scripts/automod/frontend_probe.py`, #1601) ships observe-only, and its
whole value claim is "it would have caught this". Until this file that claim rested
on hand-written fixture assertions and no detection number: every node in
`tests/test_automod_frontend_probe.py` asks the probe about a shape its author chose
to write, which proves the probe can see a shape, not how often it sees one. This
module supplies the missing measurement — seed ≥10 broken builds, run the shipped
probe over each, report `detected n/m` — which is what #1872 asks for and what #1601
owed entry 2's number and entry 3's promotion ruling are gated on.

Why the seeds are hand-written build directories rather than real diffs
    A seed is an `index.html`, one hashed `.js` and one hashed `.css`, laid out the
    way `vite build` lays them out. No seed needs a real `vite build`, so no build
    input (`package.json`, `vite.config.*`, `tsconfig*`) is touched and the run costs
    seconds per seed rather than a build per seed. The fixture machinery is a copy of
    `tests/test_automod_frontend_probe.py`'s `_build`, deliberately: `scripts/` must
    not import from `tests/`, so the vocabulary is copied rather than shared.

Why every probe call passes an explicit `shots_dir`
    `probe_build` defaults its screenshot directory to `STATE_DIR / "frontend_probe"`
    — which is exactly where the gate's real per-round verdicts are written
    (`frontend_probe.py:419-422`), and exactly the directory #1601 owed entry 1
    counts when it tallies "≥20 recorded verdicts on real `web/` landings". A canary
    that let that default fire would create the directory and fill it with seeded
    fakes, and the real-verdict tally could then count fabricated rows. So: an
    explicit `shots_dir` inside this run's scratch directory on every call, and this
    run's artifact goes under `STATE_DIR / "frontend_probe_canary"` — beside, never
    inside, `STATE_DIR / "frontend_probe"`.

Why a seed may be declared `blind`, and what stops that from being a loophole
    Two shapes this item names are invisible to the probe as shipped (a throw that
    lands after the settle window has closed; a stylesheet that is served with a 200
    and matches no rule), and with a ≥90% bar over ≥10 seeds the tolerance is exactly
    one miss — so sinking those would refuse the run for a limitation of the
    instrument rather than a defect in it. A seed therefore carries an optional
    `blind_reason`, is scored `blind`, printed by name, and leaves both `n` and `m`.
    Two things keep the declaration honest. First, it is only honoured when the probe
    genuinely reported nothing: a declared-blind seed the probe DOES flag is scored
    `detected` and counted, so declaring a visible seed blind buys the run nothing.
    Second, the artifact and stdout both carry `rate_counting_blind_as_misses`
    alongside the rate, so the pessimistic number is on the record and the item that
    owes the scope decision (whether either shape should become a probe capability)
    gets the measurement rather than the exemption.

What this run is NOT
    It is not a gate. Nothing here is called by `gate.py`, and the probe stays
    observe-only until #1601 owed entry 3 is re-ruled on the number this writes.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from scripts.automod import frontend_layout as FL
from scripts.automod import frontend_probe as FP
from scripts.automod import state as S

#: The contract of #1872: a rate is only worth reporting over at least this many
#: must-detect seeds, and only at or above this fraction. Both ride the artifact
#: beside the measurement, so a reader never has to guess which bar produced it.
MIN_SEEDS = 10
MIN_DETECTION_RATE = 0.90

#: Sibling of `STATE_DIR / "frontend_probe"`, never inside it — see the docstring.
ARTIFACT_DIRNAME = "frontend_probe_canary"

#: The asset names `index.html` references. A seed that "never emitted" one of these
#: keeps the reference and drops the file, which is what a build that lied about its
#: own output looks like from the browser's side.
ASSET_JS = "index-abc123.js"
ASSET_CSS = "index-abc123.css"

#: One seed is a document that never reaches `load`. Under the shipped
#: `LOAD_BUDGET_S` (30 s) that costs half a minute to decide a branch the probe
#: decides the same way at any budget, so this one seed shortens it — the same
#: fixture-timing knob `tests/test_automod_frontend_probe.py` turns.
WEDGED_LOAD_BUDGET_S = 2.0

#: The asset the wedged-load seed asks for. Bound and listened, never accepted: the
#: request hangs rather than being refused, so `load` genuinely cannot complete.
WEDGED_ASSET = "/hang.png"

# ── the fixture vocabulary, copied from tests/test_automod_frontend_probe.py ──
#
# `MOUNTED` is what a healthy boot looks like from outside; the throw is what
# `web/src/App.tsx`'s error boundary leaves behind. `WARNINGS` is healthy-but-noisy:
# the AudioContext and LiveKit messages the live app logs with no user gesture. It
# belongs on the control, where it is a live trap for anyone who later tries to judge
# a console warning.

MOUNTED = """
  const root = document.getElementById('root');
  const panel = document.createElement('div');
  panel.className = 'panel';
  panel.textContent = 'Mission Control — sessions, autonomy, GPU, voice';
  root.appendChild(panel);
"""

WARNINGS = """
  console.warn('The AudioContext was not allowed to start. It must be resumed '
               + '(or created) after a user gesture on the page.');
  console.warn('mic publish failed: NotSupportedError');
  console.info('LiveKit: connecting (this is info, not an error)');
"""

THROW_INSIDE_BOUNDARY = """
  const root = document.getElementById('root');
  root.innerHTML = '<h2>Something went wrong</h2><p>Try reloading the page.</p>';
  const err = new TypeError('Cannot read properties of undefined (reading volume)');
  console.error('ErrorBoundary caught an error:', err);
  throw err;
"""

UNCAUGHT_ONLY = """
  const root = document.getElementById('root');
  const panel = document.createElement('div');
  panel.textContent = 'Mission Control';
  root.appendChild(panel);
  throw new Error('boom from a module top level');
"""

TEXT_WITHOUT_ROOT = """
  document.body.innerHTML = '<h1>Mission Control</h1><p>417x</p>';
"""

RENDERS_NOTHING = """
  /* a bundle that boots, mounts nothing, and says nothing */
"""

# ── the shapes #1872 names that the copied fixtures do not cover ─────────────

#: A module script that imports a chunk the build never emitted. The browser
#: resolves that reference at runtime, so the missing file surfaces as a 404 on the
#: console and a tree that never mounts — the "a 404, not a MIME-type lie" finding.
IMPORT_OF_A_MISSING_CHUNK = """
  import { panel } from '/assets/chunk-late-abc123.js';
  document.getElementById('root').appendChild(panel());
"""

#: First paint lands, then something throws while the probe is still inside its
#: settle window. This is the "component throwing after first paint" shape, seeded
#: where the shipped probe can actually see it.
THROW_AFTER_FIRST_PAINT = """
  const root = document.getElementById('root');
  const panel = document.createElement('div');
  panel.className = 'panel';
  panel.textContent = 'Mission Control — first paint landed';
  root.appendChild(panel);
  setTimeout(() => {
    throw new TypeError('sessionPanel is undefined (threw after first paint)');
  }, 400);
"""

#: Reads a global that no emitted file defines, while mounting.
UNDEFINED_GLOBAL_AT_MOUNT = """
  const endpoint = window.LLOYD_RUNTIME.endpoint;
  const root = document.getElementById('root');
  const panel = document.createElement('div');
  panel.textContent = 'Mission Control @ ' + endpoint;
  root.appendChild(panel);
"""

#: Mounts fine and reports a boot-time API failure through `console.error`. Healthy
#: DOM, broken start-up — the shape `page-rendered` cannot see.
BOOT_FETCH_FAILURE_LOGGED = MOUNTED + """
  console.error('Failed to fetch /api/sessions: 500 (Internal Server Error)');
"""

#: Builds real DOM and attaches it to a node that is itself never in the document:
#: the app did all its work into a detached tree.
MOUNT_INTO_A_DETACHED_NODE = """
  const panel = document.createElement('div');
  panel.textContent = 'Mission Control — built but never attached';
  const holder = document.createElement('div');
  holder.appendChild(panel);
"""

#: A healthy page that dies some seconds later — past the settle window, so the
#: shipped probe has already read the DOM when this fires. Hence the seed's
#: `blind_reason`, and hence the number this run reports for it.
THROW_LONG_AFTER_LOAD = MOUNTED + """
  setTimeout(() => { throw new Error('voice loop died long after load'); }, 5000);
"""

#: A stylesheet the browser accepts and that matches nothing on the page.
UNMATCHED_STYLESHEET = "#lloyd-node-that-is-not-in-the-dom{display:grid}\n"

HEALTHY_CSS = ".panel{display:grid}\n"


@dataclass(frozen=True)
class Seed:
    """One build directory, and what the probe is expected to make of it.

    `expect` is `broken` for a seed that must be detected and `control` for the
    healthy build. `blind_reason` is non-empty only for a seed the probe provably
    cannot see; it takes that seed out of the rate rather than sinking it.
    `throws_after_ms` is set only by a seed whose whole point is a timer, so the
    table itself says which side of `frontend_probe.SETTLE_MS` the throw lands on.
    """

    name: str
    script: str
    expect: str = "broken"
    blind_reason: str = ""
    root_element: bool = True
    root_inner: str = ""
    extra_html: str = ""
    emit_js: bool = True
    emit_css: bool = True
    css_body: str = HEALTHY_CSS
    throws_after_ms: int | None = None
    wedged_load: bool = False
    load_budget_s: float | None = None
    #: Which instrument the seed grades. `load` (every seed as shipped) is
    #: `frontend_probe`: mount, console, pageerror, error boundary. `layout` is
    #: #2130's fingerprint leg — `frontend_layout.capture_build` + `diff` — over a
    #: fixture app built by `layout_app`, because a layout seed has to control a
    #: stylesheet and a set of sections, not a mount script. The channel is a field
    #: rather than two tables so the artifact stays ONE table:
    #: `stylesheet_loads_but_matches_no_rule` has to be the same row before and
    #: after the leg landed, or the number moves and nobody can tell whether the
    #: instrument improved or the denominator did.
    channel: str = "load"
    #: For `channel="layout"`: which `LAYOUT_BREAKS` injection to apply. The seed
    #: names the mechanism; `LAYOUT_BREAKS` carries what it is supposed to move, so
    #: a seed cannot be scored against an expectation its own injection does not
    #: create.
    layout_kind: str = ""


#: The shipped seed table: 12 seeds that must be detected, exactly one healthy
#: control, and two declared blind. `--seeds` puts a floor under the first group; it
#: never truncates this one.
SEEDS: tuple[Seed, ...] = (
    Seed("healthy_build_that_warns", MOUNTED + WARNINGS, expect="control"),

    # The four break shapes #1872 names, as table entries.
    Seed("import_of_a_module_the_build_never_emitted", IMPORT_OF_A_MISSING_CHUNK),
    Seed("throw_after_first_paint_inside_the_settle_window", THROW_AFTER_FIRST_PAINT,
         throws_after_ms=400),
    Seed("undefined_global_read_at_mount", UNDEFINED_GLOBAL_AT_MOUNT),
    Seed("stylesheet_the_build_never_emitted", MOUNTED, emit_css=False),

    # The shapes the probe's own suite pins — seeded again here as builds, so the
    # number counts builds rather than assertions.
    Seed("throw_caught_by_the_error_boundary", THROW_INSIDE_BOUNDARY),
    Seed("uncaught_error_at_module_top_level", UNCAUGHT_ONLY),
    Seed("boots_and_renders_nothing", RENDERS_NOTHING),
    Seed("text_rendered_without_a_root_element", TEXT_WITHOUT_ROOT,
         root_element=False),
    Seed("bundle_script_the_build_never_emitted", MOUNTED, emit_js=False),
    Seed("boot_fetch_failure_logged_as_console_error", BOOT_FETCH_FAILURE_LOGGED),
    Seed("mounted_into_a_detached_node", MOUNT_INTO_A_DETACHED_NODE),
    Seed("document_never_finishes_loading", MOUNTED, wedged_load=True,
         load_budget_s=WEDGED_LOAD_BUDGET_S),

    # Declared blind: not visible to the probe as shipped, so the run names them and
    # keeps them out of the rate — while still reporting the rate they would cost.
    Seed("throw_long_after_the_settle_window", THROW_LONG_AFTER_LOAD,
         blind_reason=f"throws 5000 ms after load, past the {FP.SETTLE_MS} ms settle "
                      f"window at which the probe reads the DOM",
         throws_after_ms=5000),
    # The blind spot the layout leg exists to close, and the row the item's
    # acceptance check is written against. Read the mechanism before the name: the
    # shape this seed shipped as — a rule that matches NOTHING — is not a break at
    # all. A rule that matches nothing changed nothing, and no instrument can see a
    # change that did not happen; that literal variant stays a no-fire control
    # pinned in tests/test_frontend_layout_fingerprint.py. What ships here is the
    # shape a CSS edit actually takes on the way to breaking a component: the
    # stylesheet is still served with a 200, the selector no longer matches the node
    # it used to style, the console says nothing, the mount is clean, and one
    # section is laid out wrong. `dead_rule` renames the selector for exactly that.
    Seed("stylesheet_loads_but_matches_no_rule", MOUNTED,
         channel="layout", layout_kind="dead_rule"),
    Seed("shared_variable_shifts_a_section_the_diff_never_named", MOUNTED,
         channel="layout", layout_kind="shared_variable"),
    Seed("panel_hidden_by_a_rule_that_matches_it", MOUNTED,
         channel="layout", layout_kind="hidden_panel"),
    Seed("panel_widened_past_the_section_that_holds_it", MOUNTED,
         channel="layout", layout_kind="wide_panel"),
)

#: The four shapes the item requires, by the name of the seed that carries them.
REQUIRED_SHAPES = (
    "import_of_a_module_the_build_never_emitted",
    "throw_after_first_paint_inside_the_settle_window",
    "undefined_global_read_at_mount",
    "stylesheet_the_build_never_emitted",
)

#: The four words this module ever writes into a seed's `score`.
SCORES = ("detected", "missed", "blind", "control")


def must_detect_seeds(seeds: Iterable[Seed] = SEEDS) -> list[Seed]:
    """The seeds the run is measured on: broken, and not declared blind.

    A declared-blind seed is `expect == "broken"` — it is a broken build — but it is
    not one the probe is being graded on, so counting it here would print a seed
    count the rate's own denominator does not match.
    """
    return [s for s in seeds if s.expect == "broken" and not s.blind_reason]


def blind_seeds(seeds: Iterable[Seed] = SEEDS) -> list[Seed]:
    return [s for s in seeds if s.blind_reason]


def control_seeds(seeds: Iterable[Seed] = SEEDS) -> list[Seed]:
    return [s for s in seeds if s.expect == "control"]


@contextlib.contextmanager
def _stalled_port():
    """A port that accepts nothing: bound and listened, never `accept`ed.

    A refused connection would let the page load fine; a black hole is what makes the
    document genuinely unable to reach `load`, which is the branch this seed tests.
    """
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    try:
        yield sock.getsockname()[1]
    finally:
        sock.close()


def write_build(out: Path, seed: Seed, *, extra_html: str = "") -> Path:
    """Lay `seed` down as a `vite build` output: hashed assets under `assets/`,
    absolute `/assets/...` references in `index.html`, so the probe's own static
    serving is what resolves them.

    A seed with `emit_js` or `emit_css` false keeps the reference and drops the file
    — a build that did not produce one of its own outputs, which the probe has to
    tell apart from an app that threw.
    """
    out.mkdir(parents=True, exist_ok=True)
    (out / "assets").mkdir(exist_ok=True)
    body = (f'<div id="root">{seed.root_inner or "<!-- app mounts here -->"}</div>'
            if seed.root_element else '<!-- no root element at all -->')
    (out / "index.html").write_text(
        '<!doctype html>\n<html><head><meta charset="utf-8">'
        f'<link rel="stylesheet" href="/assets/{ASSET_CSS}"></head>\n'
        f"<body>{body}{seed.extra_html}{extra_html}"
        f'<script type="module" src="/assets/{ASSET_JS}"></script>'
        "</body></html>\n", encoding="utf-8")
    if seed.emit_css:
        (out / "assets" / ASSET_CSS).write_text(seed.css_body, encoding="utf-8")
    if seed.emit_js:
        (out / "assets" / ASSET_JS).write_text(seed.script, encoding="utf-8")
    return out


#: ---------------------------------------------------------------------------
#: The layout channel's fixture app (#2130)
#: ---------------------------------------------------------------------------
#:
#: A Mission Control-shaped page, not Mission Control. A layout seed has to control
#: a stylesheet and a set of sections; standing up the real dashboard would put the
#: round's own `web/` tree inside the measurement, and the canary's job is to grade
#: the leg against a break it created, not against whatever the app happened to do
#: today. The DOM is the dashboard's though, exactly: an `h2` per section (the
#: collector takes `section > h2` and skips a section without one), a wrapper `div`,
#: `.rounded-lg` panels carrying an `h3`, `.rounded-md` HealthPills — the selectors
#: `scripts/maintenance/dashboard_mobile_probe.py` reads. A fixture built with
#: invented class names collects an EMPTY panels list, which is the vacuous pass
#: that probe's own comment was written to refuse.

#: The seven headings the leg's `MIN_SECTIONS` denominator was measured off. Fewer
#: and the fixture trips its own no-verdict guard instead of testing the diff.
LAYOUT_HEADS = ("vLLM engines", "Lloyd agent", "Subagents & background tasks",
                "System", "Services", "Automation & work", "Tokens")

#: `<section class="sec sec-<slug>">`, so a seed can aim one rule at one section the
#: way a real shared-variable edit misses exactly one section.
LAYOUT_SLUGS = {"vLLM engines": "engines", "Lloyd agent": "agent",
                "Subagents & background tasks": "subagents", "System": "system",
                "Services": "services", "Automation & work": "automation",
                "Tokens": "tokens"}

#: The healthy sheet. `--side-pad` is consumed by TWO sections (`tokens` and
#: `automation`) on purpose: a variable no section shares would make the
#: shared-variable seed a single-section edit, which is not the ripple class.
LAYOUT_CSS = """
:root { --section-w: 700px; --panel-w: 340px; --side-pad: 200px; }
.sec { width: var(--section-w); }
.sec-tokens { width: calc(var(--section-w) - var(--side-pad)); }
.sec-automation { width: calc(var(--section-w) - var(--side-pad)); }
.sec > div > .rounded-lg { width: var(--panel-w); }
.sec > div > .rounded-md { width: 120px; }
"""

#: Every layout injection: what it appends to the sheet, what it deletes from it,
#: the sections it is SUPPOSED to move, and the path the round is assumed to have
#: changed. `changed_paths` deliberately names a theme file that is NOT the affected
#: section's owner for three of the four seeds: `UNTOUCHED` is the tag this leg
#: exists to produce, and a canary that could only ever show `in-diff` would be
#: grading an instrument that never reports the interesting case. `wide_panel` is
#: the opposite pairing so both tags appear in one artifact.
LAYOUT_BREAKS: dict[str, dict[str, Any]] = {
    "control": {"add": "", "kill": (), "affected": (), "owners": {},
                "changed_paths": ()},
    # The rule that sized Tokens is renamed, so the sheet still loads with a 200 and
    # its selector matches nothing: Tokens falls back to `.sec`'s width and grows by
    # exactly the pad it used to subtract.
    "dead_rule": {
        "add": ".sec-tokenz { width: calc(var(--section-w) - var(--side-pad)); }\n",
        "kill": (".sec-tokens { width: calc(var(--section-w) - var(--side-pad)); }",),
        "affected": ("Tokens",),
        "owners": {"Tokens": ["web/src/components/pages/TokensPage.tsx"]},
        "changed_paths": ["web/src/theme.css"],
    },
    # One `:root` value, edited for a reason that names no section, takes two of them
    # with it — the ripple class in its purest form, and the reason the leg grades
    # sections the diff never mentions.
    "shared_variable": {
        "add": ":root { --side-pad: 420px; }\n",
        "kill": (),
        "affected": ("Tokens", "Automation & work"),
        "owners": {"Tokens": ["web/src/components/pages/TokensPage.tsx"],
                   "Automation & work": ["web/src/components/pages/AutomationPage.tsx"]},
        "changed_paths": ["web/src/theme.css"],
    },
    # A panel that stops rendering. Kirschner's own example is an icon disappearing
    # somewhere else; this is that bug, and a width-only check would miss it — the
    # surviving panel is simply a different width and nothing looks missing.
    "hidden_panel": {
        # `:nth-child(2)`, NOT `:last-child`: Services' wrapper holds two pills
        # after its panels, so `:last-child` selects a pill, matches no panel, and
        # the seed would silently break nothing — a canary that breaks nothing and
        # reports a miss is an instrument reporting about itself.
        "add": ".sec-services > div > .rounded-lg:nth-child(2) {\n  display: none; }\n",
        "kill": (),
        "affected": ("Services",),
        "owners": {"Services": ["web/src/components/pages/ServicesPage.tsx"]},
        "changed_paths": ["web/src/theme.css"],
    },
    # A panel wider than the section holding it: the section's own scrollWidth grows
    # past its client width, which is the ONLY way this field set sees an overflow
    # that no other section notices. Aimed at the component that owns the section,
    # so the artifact also carries an `in-diff` row.
    "wide_panel": {
        "add": ".sec-system > div > .rounded-lg { width: 900px; }\n",
        "kill": (),
        "affected": ("System",),
        "owners": {"System": ["web/src/components/pages/SystemPage.tsx"]},
        "changed_paths": ["web/src/components/pages/SystemPage.tsx"],
    },
}

#: One viewport for the canary: the seeds aim at the leg's mechanisms, not at
#: breakpoint coverage, which `tests/test_frontend_layout_fingerprint.py` pins by
#: capturing one page at every configured width.
LAYOUT_WIDTHS: tuple[int, ...] = (1280,)


def layout_app(dest: Path, kind: str) -> Path:
    """Write the fixture app — whole, or broken in one named way.

    No vite: a *build* is just a directory the server can serve, and the layout
    channel needs an `index.html` and a stylesheet, so these seeds run on a box with
    no `node`. The load channel still builds through the real bundler because it
    grades a channel that only exists once a bundler has decided what ships.
    """
    if kind not in LAYOUT_BREAKS:
        raise ValueError(f"no such layout break: {kind!r} (of "
                         f"{sorted(LAYOUT_BREAKS)})")
    spec = LAYOUT_BREAKS[kind]
    dest.mkdir(parents=True, exist_ok=True)
    body = "\n".join(_layout_section(h) for h in LAYOUT_HEADS)
    css = LAYOUT_CSS
    for killed in spec["kill"]:
        if killed not in css:
            raise ValueError(f"layout break {kind!r} kills a rule the sheet "
                             f"does not contain: {killed!r}")
        css = css.replace(killed, "")
    (dest / "index.html").write_text(
        "<!doctype html><html><head><meta charset=utf-8>"
        '<link rel="stylesheet" href="/styles.css"></head>'
        f"<body><main><div id=\"root\">{body}</div></main></body></html>")
    (dest / "styles.css").write_text(css + spec["add"])
    return dest


def measure_layout_seed(seed: Seed, dir: Path, *,
                        shots_dir: Path | None = None) -> dict[str, Any]:
    """One layout seed: the same app built whole AND broken, both fingerprinted.

    The control is rebuilt per seed rather than measured once and shared, because a
    fingerprint is a property of the page AND the browser: share one control across
    four seeds and a browser that drifts mid-run — a font that lands late, a
    scrollbar that appears — becomes three confident "detected" rows and a rate
    nobody can read. Each seed pays for two loads and gets a control that saw the
    same browser process.

    Returns the same shape the load path returns (`checks`, `metrics`, or
    `skipped`), so `score_seed` and `summarise` stay the single definition of the
    rate across both channels.
    """
    if seed.channel != "layout":
        raise ValueError(f"{seed.name}: channel is {seed.channel!r}, not layout")
    spec = LAYOUT_BREAKS[seed.layout_kind]
    # The layout channel takes no screenshot, so this is where its evidence goes
    # instead: the two raw geometry captures the diff was computed from, which is
    # what a reviewer of a layout finding actually needs to see. Under
    # `--keep-shots`'s tree when one is passed, otherwise the seed's own scratch
    # directory. The artifact's `shots_dir` must name a directory that exists and
    # that this run owns — `Path("")` would be the CWD, which for this script is a
    # checkout.
    shots = Path(shots_dir or dir)
    shots.mkdir(parents=True, exist_ok=True)
    caps: dict[str, Any] = {}
    for stage in ("control", "broken"):
        layout_app(dir / stage, "control" if stage == "control" else seed.layout_kind)
        cap = FL.capture_build(dir / stage, widths=LAYOUT_WIDTHS,
                               chromium=FP.CHROMIUM)
        if cap.get("skipped"):
            return {"skipped": f"the layout leg could not run: {cap['skipped']}"}
        caps[stage] = cap
        (shots / f"{stage}-geometry.json").write_text(
            json.dumps(cap, indent=1, default=str), encoding="utf-8")
    unrendered = [f"{stage} at {v.get('width')}px ({v['no_verdict']})"
                  for stage, cap in caps.items()
                  for v in (cap["views"] or {}).values() if v.get("no_verdict")]
    if unrendered:
        # A no-verdict is not a miss: an app that never rendered says nothing about
        # whether the diff works, and scoring it either way would be a number about
        # the fixture rather than about the leg.
        return {"skipped": "the fixture produced no layout verdict: "
                           + ", ".join(unrendered)}
    base = FL.fingerprint(caps["control"]["views"],
                          round_id=f"canary-{seed.name}-control")
    current = FL.fingerprint(caps["broken"]["views"], round_id=f"canary-{seed.name}")
    checks = FL.diff(base, current,
                     owners={k: list(v) for k, v in spec["owners"].items()},
                     changed_paths=list(spec["changed_paths"]))
    return {"checks": checks,
            "expected_sections": list(spec["affected"]),
            "metrics": {"layout_sections": {
                w: len(v.get("sections") or {})
                for w, v in (caps["broken"]["views"] or {}).items()}},
            "layout": {"sections_per_width": {
                           w: len(v.get("sections") or {})
                           for w, v in (current["views"] or {}).items()},
                       "attribution": {c["check"]: c["attribution"] for c in checks}},
            "screenshot": "", "shots_dir": str(shots)}


def _layout_section(head: str) -> str:
    """One section of the fixture app, in the dashboard's shape. See `layout_app`."""
    pills = ('<div class="rounded-md">pill a</div>'
             '<div class="rounded-md">pill b</div>' if head == "Services" else "")
    panels = "".join(f'<div class="rounded-lg"><h3>{head} panel {i}</h3></div>'
                     for i in (1, 2))
    return (f'<section class="sec sec-{LAYOUT_SLUGS[head]}"><h2>{head}</h2>'
            f"<div>{panels}{pills}</div></section>")


def score_seed(seed: Seed, verdict: dict[str, Any]) -> dict[str, Any]:
    """One seed's score, from the verdict alone — the scoring rule in one place so
    the rate has exactly one definition.

    * `detected` requires at least one NAMED failing check. A verdict that says
      `ok: False` while naming nothing is not a detection, it is a contradiction, and
      counting it would inflate the number this module exists to report honestly.
    * a `control` seed is scored `control` whatever the probe says, and never
      `detected`. A control the probe fails is a false block: the run calls it that
      and exits non-zero, rather than letting it join a numerator.
    * a seed with a `blind_reason` is scored `blind` — but only when the probe
      reported nothing. If the probe named a failing check anyway the declaration was
      wrong, so the seed is `detected` and counted, which is what stops "declare it
      blind" from being a way to protect a seed from the bar.
    * a skipped verdict (`{"skipped": …}`, and no `ok`) is not a miss either: nothing
      was measured. It is scored `blind` with `probe_skipped` set, and `main` turns a
      run in which every seed skipped into an exit code rather than a rate.
    """
    named = [str(c.get("check", "")) for c in (verdict.get("checks") or [])]
    named = [c for c in named if c]
    base: dict[str, Any] = {"name": seed.name, "checks": named,
                            "expect": seed.expect, "declared_blind": bool(seed.blind_reason),
                            "blind_reason": seed.blind_reason,
                            "throws_after_ms": seed.throws_after_ms,
                            # The row says which instrument it grades, so a reader
                            # comparing two increments is not comparing a load number
                            # to a layout number by accident.
                            "channel": seed.channel,
                            # For a layout seed, the section each fired check names —
                            # `checks` is names only, and the whole point of the leg is
                            # WHICH section it blamed and on whose account.
                            "attribution": (verdict.get("layout") or {}).get("attribution")
                            or {}}
    if verdict.get("skipped"):
        return {**base, "score": "blind", "probe_skipped": True,
                "note": f"the probe could not run: {verdict['skipped']}"}
    if seed.expect == "control":
        return {**base, "score": "control",
                "note": ("" if not named else
                         f"the probe FAILED a healthy build: {', '.join(named)}")}
    if seed.channel == "layout":
        # Stricter than the load channel's bar, and it has to be. For a load seed a
        # fired check IS the finding; here a leg that fired on every section would
        # hit the right one by accident, and one that fired on the wrong section
        # while the styled one sits untouched found something else entirely. So the
        # check must carry the heading this seed's own injection moved — which is
        # also the only property a reviewer can act on ("Tokens moved"), as against
        # "the layout differs somewhere".
        expected = [str(e) for e in (verdict.get("expected_sections") or [])]
        hits = [c for c in named if any(e in c for e in expected)]
        if hits:
            return {**base, "score": "detected", "note": ""}
        return {**base, "score": "missed",
                "note": (f"{len(named)} layout check(s) fired but none named "
                         f"{expected}: {named[:4]}" if named else
                         "the fingerprint was identical, so the leg says ok over a "
                         "build whose layout the seed broke")}
    if named:
        return {**base, "score": "detected", "note": ""}
    if seed.blind_reason:
        return {**base, "score": "blind", "note": f"declared blind: {seed.blind_reason}"}
    if seed.throws_after_ms is not None and seed.throws_after_ms > FP.SETTLE_MS:
        return {**base, "score": "missed",
                "note": f"a throw at {seed.throws_after_ms} ms is past the "
                        f"{FP.SETTLE_MS} ms window the probe reads the DOM at"}
    return {**base, "score": "missed", "note": "the probe reported no failing check"}


def summarise(records: list[dict[str, Any]], *, min_seeds: int = MIN_SEEDS,
              min_rate: float = MIN_DETECTION_RATE) -> dict[str, Any]:
    """The rate, and what it excludes. `detected` and `missed` are the whole of `n`
    and `m`; `blind` and `control` are in neither.

    `rate_counting_blind_as_misses` is the same numerator over a denominator that
    charges every declared-blind seed as a miss. It is not the contract's number —
    the contract excludes provably-invisible seeds — but it is the number a reader
    deserves to see next to it, because it is the one that would be true if the blind
    declarations were the loophole they could be read as.
    """
    scored = [r for r in records if r["score"] in ("detected", "missed")]
    n = sum(1 for r in scored if r["score"] == "detected")
    m = len(scored)
    blind_missed = [r for r in records if r["score"] == "blind"
                    and r["expect"] == "broken" and not r.get("probe_skipped")]
    controls = [r for r in records if r["score"] == "control"]
    dirty = [r["name"] for r in controls if r.get("checks")]
    rate = (n / m) if m else 0.0
    pessimistic = (n / (m + len(blind_missed))) if (m + len(blind_missed)) else 0.0
    return {"detected": n, "seeds_measured": m, "rate": rate,
            "blind": [r["name"] for r in records if r["score"] == "blind"],
            "blind_as_misses": len(blind_missed),
            "rate_counting_blind_as_misses": pessimistic,
            "probe_skipped": [r["name"] for r in records if r.get("probe_skipped")],
            "control": [r["name"] for r in controls],
            "control_clean": not dirty, "control_failures": dirty,
            "thresholds": {"min_seeds": min_seeds, "min_rate": min_rate},
            "passes": bool(m >= min_seeds and rate >= min_rate and not dirty)}


def run_canary(*, root: Path | None = None, shots_dir: Path | None = None,
               seeds: Iterable[Seed] = SEEDS, min_seeds: int = MIN_SEEDS,
               min_rate: float = MIN_DETECTION_RATE) -> dict[str, Any]:
    """Seed every build, probe every seed, score every verdict, return the report.

    `root` is the scratch directory: one sub-directory per seeded build, and one
    per-seed `shots_dir` under `root/shots/`, so nothing this does can land in a
    checkout or in the gate's verdict directory. It defaults to
    `frontend_probe.tmp_build_dir()`, which exists for exactly this purpose.
    """
    root = Path(root) if root else FP.tmp_build_dir()
    # One evidence tree for the whole run, decided ONCE, so a record's `shots_dir`
    # never has to name a directory that was not made. Default is the scratch root
    # itself, not `root/shots`: the load channel's per-seed directory is already
    # `root/<name>`, and putting this channel's captures in the same place keeps one
    # directory per seed holding everything that seed produced.
    shots_root = Path(root) if shots_dir is None else Path(shots_dir)
    seeds = list(seeds)
    records: list[dict[str, Any]] = []
    for seed in seeds:
        if seed.channel == "layout":
            # No vite and no bundler for these: the layout channel grades a served
            # directory, and `layout_app` writes exactly the two files that makes
            # (`index.html`, `styles.css`), so the seeds run on a box with no node.
            try:
                verdict = measure_layout_seed(
                    seed, root / seed.name,
                    shots_dir=(shots_root / seed.name) if shots_root else None)
            except Exception as exc:                    # noqa: BLE001
                # Same rule as `FP.run`'s guard on the other side of this loop: an
                # instrument that explodes reports that it did not measure, and does
                # not take the run — and therefore the artifact and the rate — down
                # with it. A raise here would also lose the 13 rows already measured.
                verdict = {"skipped": f"the layout leg raised "
                                      f"{exc.__class__.__name__}: {str(exc)[:180]}"}
            record = score_seed(seed, verdict)
            record["build_dir"] = str(root / seed.name)
            record["shots_dir"] = str(verdict.get("shots_dir") or (root / seed.name))
            record["screenshot"] = str(verdict.get("screenshot") or "")
            record["metrics"] = dict(verdict.get("metrics") or {})
            record["layout"] = verdict.get("layout") or {}
            records.append(record)
            continue
        with contextlib.ExitStack() as stack:
            extra = ""
            if seed.wedged_load:
                port = stack.enter_context(_stalled_port())
                extra = f'<img src="http://127.0.0.1:{port}{WEDGED_ASSET}">'
            build = write_build(root / seed.name, seed, extra_html=extra)
            shots = root / "shots" / seed.name
            shots.mkdir(parents=True, exist_ok=True)
            kwargs: dict[str, Any] = {"shots_dir": shots, "chromium": FP.CHROMIUM}
            if seed.load_budget_s is not None:
                kwargs["load_budget_s"] = seed.load_budget_s
            # `FP.run` IS `probe_build`, with any exception turned into a named skip —
            # which is what a harness that measures an instrument wants: a probe that
            # explodes gets reported as "nothing measured", it does not crash the run.
            verdict = FP.run(build, **kwargs)
            record = score_seed(seed, verdict)
            record["build_dir"] = str(build)
            record["shots_dir"] = str(shots)
            record["screenshot"] = str(verdict.get("screenshot") or "")
            record["metrics"] = {k: v for k, v in (verdict.get("metrics") or {}).items()
                                 if k in ("root_children", "body_text_length",
                                          "root_present", "boundary_fallback_present")}
            records.append(record)
    report = summarise(records, min_seeds=min_seeds, min_rate=min_rate)
    report.update({
        "schema": 1,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scratch_root": str(root),
        "chromium": FP.CHROMIUM,
        "probe_budgets": {"settle_ms": FP.SETTLE_MS,
                          "load_budget_s": FP.LOAD_BUDGET_S,
                          "wedged_seed_load_budget_s": WEDGED_LOAD_BUDGET_S},
        "seeds_seeded": len(seeds),
        "seeds_required": min_seeds,
        "seeds": records,
    })
    return report


def write_run_artifact(report: dict[str, Any], *,
                       now: datetime | None = None) -> Path:
    """Keep the measurement where the owed-check job reads it, never in the gate's
    real-verdict directory.

    Two files, one document: the timestamped run keeps history, and `latest.json` is
    the stable pointer a later job can be told to open. `latest.json` is written LAST
    and written WHOLE — a `-latest` pointer that ever holds a skeleton is a pointer
    the next reader mistakes for a report — so the only ordering here is "after the
    run ended, the complete body".
    """
    d = S.STATE_DIR / ARTIFACT_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, default=str) + "\n"
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    run_path = d / f"run-{stamp}.json"
    run_path.write_text(payload, encoding="utf-8")
    (d / "latest.json").write_text(payload, encoding="utf-8")
    return run_path


def format_rows(report: dict[str, Any]) -> list[str]:
    """One line per seed: its score, its name, and the checks that earned it."""
    rows: list[str] = []
    for r in report["seeds"]:
        detail = ", ".join(r["checks"]) or r.get("note", "")
        rows.append(f"  [{r['score']}] {r['name']} — {detail}")
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m scripts.automod.frontend_probe_canary",
        description="Seed known-broken frontend builds and report what fraction the "
                    "runtime probe detects.")
    ap.add_argument("--seeds", type=int, default=MIN_SEEDS,
                    help=f"minimum number of must-detect seeds the run must cover "
                         f"(default {MIN_SEEDS}). The whole shipped table always "
                         f"runs: this is a floor the run refuses to report a rate "
                         f"below, not a slice of the table.")
    args = ap.parse_args(argv)

    available = len(must_detect_seeds())
    if args.seeds > available:
        print(f"frontend-probe canary: --seeds {args.seeds} asks for more must-detect "
              f"seeds than the shipped table holds ({available}); nothing measured.")
        return 2

    seeds = list(SEEDS)
    print(f"frontend-probe canary: {available} must-detect seeds, "
          f"{len(control_seeds(seeds))} control, {len(blind_seeds(seeds))} declared blind "
          f"(probe budgets: settle {FP.SETTLE_MS} ms, load {FP.LOAD_BUDGET_S:g} s)")

    root = FP.tmp_build_dir()
    report = run_canary(root=root, seeds=seeds, min_seeds=args.seeds)
    for row in format_rows(report):
        print(row)

    if report["probe_skipped"] and report["seeds_measured"] == 0:
        # Nothing was probed. A `detected 0/0` line here would hand whoever greps
        # for the pattern a fraction that was never measured; the honest output for
        # an instrument that could not run is that nothing was measured.
        print(f"not measured: the probe could not run "
              f"({report['seeds'][0].get('note', 'no reason given')})")
        print(f"scratch kept: {root}")
        return 2

    if report["blind"]:
        print(f"blind, excluded from n and m: {', '.join(report['blind'])}")
    if report["control_failures"]:
        print(f"FALSE BLOCK: the probe failed the healthy control — "
              f"{', '.join(report['control_failures'])}")
    print(f"detected {report['detected']}/{report['seeds_measured']} "
          f"({report['rate'] * 100:.1f}%), bar >="
          f"{report['thresholds']['min_rate'] * 100:.0f}% over >="
          f"{report['thresholds']['min_seeds']} seeds "
          f"(counting the {report['blind_as_misses']} blind seeds as misses: "
          f"{report['detected']}/{report['seeds_measured'] + report['blind_as_misses']} "
          f"= {report['rate_counting_blind_as_misses'] * 100:.1f}%)")

    path = write_run_artifact(report)
    report["artifact"] = str(path)
    print(f"artifact {path}")
    print(f"         {S.STATE_DIR / ARTIFACT_DIRNAME / 'latest.json'}")

    if report["passes"]:
        # Scratch goes only on a pass. On a failed measurement the seeded builds and
        # their screenshots are the evidence, and deleting them deletes the finding.
        shutil.rmtree(root, ignore_errors=True)
        return 0
    print(f"scratch kept: {root}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
