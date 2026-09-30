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
    Seed("stylesheet_loads_but_matches_no_rule", MOUNTED, css_body=UNMATCHED_STYLESHEET,
         blind_reason="the stylesheet is served with a 200 and matches no node, so no "
                      "channel the probe watches (load status, DOM, console, "
                      "pageerror) can carry it"),
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
                            "throws_after_ms": seed.throws_after_ms}
    if verdict.get("skipped"):
        return {**base, "score": "blind", "probe_skipped": True,
                "note": f"the probe could not run: {verdict['skipped']}"}
    if seed.expect == "control":
        return {**base, "score": "control",
                "note": ("" if not named else
                         f"the probe FAILED a healthy build: {', '.join(named)}")}
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


def run_canary(*, root: Path | None = None, seeds: Iterable[Seed] = SEEDS,
               min_seeds: int = MIN_SEEDS,
               min_rate: float = MIN_DETECTION_RATE) -> dict[str, Any]:
    """Seed every build, probe every seed, score every verdict, return the report.

    `root` is the scratch directory: one sub-directory per seeded build, and one
    per-seed `shots_dir` under `root/shots/`, so nothing this does can land in a
    checkout or in the gate's verdict directory. It defaults to
    `frontend_probe.tmp_build_dir()`, which exists for exactly this purpose.
    """
    root = Path(root) if root else FP.tmp_build_dir()
    seeds = list(seeds)
    records: list[dict[str, Any]] = []
    for seed in seeds:
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
