#!/usr/bin/env python
"""Measure Mission Control dashboard geometry at a set of viewports.

Usage: dashboard_mobile_probe.py URL [widths...] [--seed=PATH]

For each width reports whether any `section` overflows its own box, the width
of every panel inside every section (so two runs can be diffed), and the
elements whose content is being clipped — each one flagged `unreachable` when
nothing else in the page can get the reader to its full text.

`--seed=PATH` answers the question the live read cannot. Against a real backend
the clipped list is whatever the fleet happens to be called this hour: zero long
names is a clean report and a dashboard with nothing in it are the same number.
The seed is a saved `/api/dashboard` snapshot whose row labels are deliberately
long, so the denominator of "was every label I asked for rendered, and was any of
them clipped past reachability" is known BEFORE the page loads.
"""
import json
import re
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

WIDTHS = [320, 360, 390, 414, 1280]
# Measured 2026-09-27 off the dashboard's own section headings. The floor is a
# denominator guard, not a claim about the design: fewer sections than this
# means the page had not rendered when the probe read it, and "no section
# overflows" over an empty list prints the same zero as a healthy dashboard.
MIN_SECTIONS = 7

# The one request the dashboard makes (`web/src/api.ts:2592`, `dashboardApi.get`).
# Anchored on `/api/dashboard` rather than a bare `/dashboard` suffix: `API_BASE`
# is `/api` by default and the chrome side-panel build sets it to an absolute
# origin ending in `/api` (api.ts:949-957), so the suffix holds under either
# spelling while a bare `/dashboard` could also swallow a page navigation.
DASHBOARD_URL_RE = re.compile(r"/api/dashboard(\?.*)?$")

# An iPhone's own UA and pixel ratio, for a seeded phone measurement. A headless
# desktop UA at 390 px is a narrow window, not a phone: UA sniffing and
# `pointer: coarse` change what a component renders, and DPR is what a real
# device rasterises at. Applied only when a seed is loaded and the width is a
# phone's, so #1685's four unseeded geometry pins keep the context they were
# calibrated in.
IPHONE_UA = ("Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) "
             "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 "
             "Mobile/15E148 Safari/604.1")
PHONE_DPR = 3

# Every string the seeded run asks a page back for, as `(section, rows, field)`
# — one entry per span on the dashboard that is the only place its row's text
# appears (DashboardPage.tsx: TaskLine's `{task.name}`, the Running-now row's
# `{r.kind || r.job_id}`, the Failed row's `{t.name}`, the worker Sources
# `{src.name}`, Recent runs' `{r.source}` and `{r.summary}`, and Backlog
# Recently-touched's `{t.name}`). `running`/`kind` is here because that span
# falls back to `job_id` when `kind` is empty, so a seed row must carry one.
SEEDED_LABEL_PATHS = [
    ("autonomy", "overdue", "name"),
    ("autonomy", "held", "name"),
    ("autonomy", "upcoming", "name"),
    ("autonomy", "running", "kind"),
    ("autonomy", "failing", "name"),
    ("workers", "sources", "name"),
    ("workers", "recent_runs", "source"),
    ("workers", "recent_runs", "summary"),
    ("backlog", "recent_open", "name"),
]

JS = r"""
(seeded) => {
  const sel = el => {
    let s = el.tagName.toLowerCase();
    if (typeof el.className === 'string' && el.className.trim())
      s += '.' + el.className.trim().split(/\s+/).slice(0, 3).join('.');
    return s;
  };
  const isClipped = el => el.clientWidth > 0 && el.scrollWidth > el.clientWidth;
  // What makes a clipped label LOST rather than merely shortened: nothing. A
  // clipped label is still reachable in exactly two shapes — a non-empty
  // `title` on the element or any ancestor (hover on a desktop, the
  // long-press tooltip on iOS), or an `a`/`button` wrapping it, which carries
  // the reader to a screen where the full text is legible. Nothing else
  // counts: an `aria-label` is read aloud and never shown, and a sibling row
  // that happens to repeat the text is a different element. Returns the
  // exemption it found, or null for "the reader cannot get this back".
  const reach = el => {
    for (let n = el; n; n = n.parentElement) {
      const t = n.getAttribute ? n.getAttribute('title') : null;
      if (t && t.trim()) return 'title';
      const tag = n.tagName ? n.tagName.toLowerCase() : '';
      if (tag === 'a' || tag === 'button') return tag;
    }
    return null;
  };
  const sectionOf = el => {
    const sec = el.closest ? el.closest('section') : null;
    return sec ? ((sec.querySelector('h2') || {}).textContent || '').trim()
               : '(outside any section)';
  };
  const out = {};
  // Which device the browser thinks it is being told about. Carried in the
  // payload so the caller can assert the emulation it asked for was the
  // emulation it got: a 390 px desktop window and a 390 px iPhone are different
  // pages (UA sniffing and `pointer: coarse` change what a component renders),
  // and a phone measurement that silently ran as a narrow desktop window would
  // still print a number.
  out.env = {ua: navigator.userAgent, dpr: window.devicePixelRatio,
             coarse: matchMedia('(pointer: coarse)').matches};
  out.vw = window.innerWidth;
  out.pageOverflow = document.scrollingElement.scrollWidth - document.scrollingElement.clientWidth;
  out.sections = [];
  for (const sec of document.querySelectorAll('section')) {
    const head = sec.querySelector('h2')?.textContent?.trim();
    if (!head) continue;
    const r = sec.getBoundingClientRect();
    const kids = Array.from(sec.querySelectorAll(':scope > div > .rounded-lg, :scope > .rounded-lg, :scope > div > div > .rounded-lg'))
      .map(k => {
        const h = k.querySelector('h3')?.textContent?.trim();
        return {name: h || null, w: Math.round(k.getBoundingClientRect().width),
                sw: k.scrollWidth};
      });
    // Services' children are HealthPills, which carry `.rounded-md`, not the
    // `.rounded-lg` a Panel does. Without them the Services section reports an
    // EMPTY panels list, and a desktop assertion over an empty list passes
    // forever no matter how wrong the layout gets — the grader caught exactly
    // that about the first version of this. Collect them so the pin has a
    // denominator.
    const pills = Array.from(sec.querySelectorAll(':scope > div > .rounded-md, :scope > .rounded-md'))
      .map(k => ({name: (k.textContent || '').trim().slice(0, 20) || null,
                  w: Math.round(k.getBoundingClientRect().width),
                  sw: k.scrollWidth}));
    out.sections.push({
      head,
      cw: Math.round(r.width),
      sw: sec.scrollWidth,
      overflow: sec.scrollWidth - Math.round(r.width),
      panels: kids,
      pills,
      kidsTotal: kids.length + pills.length,
    });
  }
  // The clipped inventory, now with a verdict on each entry. Before #2202 this
  // list was built and thrown away: the Python side never read `out.clipped`,
  // and a flat list could not tell the one label that is merely long from the
  // one nobody can ever read in full.
  out.clipped = [];
  for (const el of document.querySelectorAll('body *')) {
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || !el.offsetParent) continue;
    if (!isClipped(el)) continue;
    const isPanel = el.classList.contains('rounded-lg');
    if (!isPanel && !['hidden','clip','auto','scroll'].includes(cs.overflowX)) continue;
    const t = (el.textContent || '').trim();
    if (!t) continue;
    out.clipped.push({sel: sel(el), sw: el.scrollWidth, cw: el.clientWidth,
                      text: t.slice(0, 50), unreachable: reach(el) === null});
  }
  // The cap is a payload limit, not a count: `clippedTotal` is what the list was
  // before it, so a caller that prints "N clipped" prints a real denominator
  // rather than 25.
  out.clippedTotal = out.clipped.length;
  out.unreachableTotal = out.clipped.filter(c => c.unreachable).length;
  out.clipped = out.clipped.slice(0, 25);

  // The seeded pass: what the fixture asked for, and what came back.
  //
  // FINDING an asked-for label on the page is a text lookup — the element with
  // no child elements whose own trimmed text equals it. The CHILDLESS rule is
  // load-bearing: a row div containing exactly this one span has the same
  // trimmed textContent as the span itself, and measuring the div instead would
  // report "not clipped" for a span that is. Uniqueness is a property of the
  // fixture (tests/fixtures/dashboard_long_labels.json holds seven distinct
  // strings), and if two elements ever do carry one label, the verdict taken is
  // the worst of them — the label counts as lost unless every copy of it is
  // reachable.
  //
  // The VERDICT is geometry and nothing else: `isClipped` compares scrollWidth
  // with clientWidth, and `reach` walks the ancestor chain. No seeded string is
  // ever compared against the collected `text` fields, which are 50-char slices
  // and could not match a whole label even when the label rendered in full.
  out.seeded = {asked: (seeded || []).length, rendered: 0, unreachable: 0,
                bySection: {}, missing: [], unreachableDetail: []};
  for (const label of (seeded || [])) {
    const hits = [];
    for (const el of document.querySelectorAll('body *')) {
      if (el.children.length > 0) continue;
      if ((el.textContent || '').trim() !== label) continue;
      hits.push(el);
    }
    if (hits.length === 0) {
      // Reported as a slice so a human can recognise WHICH label went missing;
      // nothing matches on it afterwards.
      out.seeded.missing.push(label.slice(0, 60));
      continue;
    }
    out.seeded.rendered += 1;
    const head = sectionOf(hits[0]);
    out.seeded.bySection[head] = (out.seeded.bySection[head] || 0) + 1;
    const lost = hits.filter(el => isClipped(el) && reach(el) === null);
    if (lost.length > 0) {
      out.seeded.unreachable += 1;
      if (out.seeded.unreachableDetail.length < 4) {
        out.seeded.unreachableDetail.push({label: label.slice(0, 60), section: head,
                                           sel: sel(lost[0]), sw: lost[0].scrollWidth,
                                           cw: lost[0].clientWidth});
      }
    }
  }
  return out;
}
"""


SECTION_POLL = """() => document.querySelectorAll('section').length > 0"""


def load_seed(path):
    """Read a seed snapshot off disk. The path is the caller's, so a missing
    fixture is a `FileNotFoundError` naming it, never a silent unseeded run."""
    return json.loads(Path(path).read_text())


def seeded_labels(payload, paths=SEEDED_LABEL_PATHS):
    """Every row label the seeded run intends to check, in fixture order.

    Derived from the payload rather than stored beside it: a hand-maintained
    second copy of the seven strings is a denominator that quietly stops
    matching what the page is fed, which is the failure this whole fixture
    exists to avoid. The instrument and its caller call THIS function, so the
    count the test prints and the count the browser was asked for cannot drift.
    """
    out, seen = [], set()
    for section, rows_key, leaf in paths:
        # A section that failed to build is `{"error": ...}`, not a state object,
        # and an older backend may send no section at all. Either yields no
        # labels, which the caller's denominator check then refuses.
        sec = payload.get(section)
        if not isinstance(sec, dict):
            continue
        rows = sec.get(rows_key) or []
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            val = row.get(leaf)
            if isinstance(val, str) and val.strip() and val not in seen:
                seen.add(val)
                out.append(val)
    return out


def dismiss_dev_overlay(page):
    """Remove vite's dev-mode error overlay if one is covering the page, and say so.

    The overlay is a `<vite-error-overlay>` custom element pinned inset-0 at a
    z-index above the app, and it swallows every click — including the one that
    opens the drawer. It fires on a cold dep-optimizer in a fresh worktree over a
    `Failed to resolve import "vscode"` inside vite's own pre-bundled CJS facade,
    which is vite's build cache and not the code under test: the app underneath
    renders in full. Left alone, that artifact turned every measured pin in this
    file into "dashboard did not load" the moment `web/node_modules` became
    reachable (#1685's four geometry pins went red for exactly this reason).

    Removing it does not hide a real compile error, for two reasons: the message
    is printed to stderr as it goes, and the page under a genuine error renders
    none of the labels, so `seed_problems`' short-render assertion is the check
    that actually decides. The overlay is chrome; the render is the measurement.
    """
    try:
        msg = page.evaluate(
            """() => {
              const o = document.querySelector('vite-error-overlay');
              if (!o) return null;
              const text = (o.shadowRoot ? o.shadowRoot.textContent : '') || '';
              o.remove();
              return text.slice(0, 300);
            }""")
    except Exception:
        return
    if msg:
        print(f"NOTE dev-server overlay removed before measuring: {msg}", file=sys.stderr)


def open_dashboard(page, timeout_ms=15000):
    """Mobile starts on the chat tab; open Dashboard from the drawer.

    Waits for the tab to actually be showing instead of sleeping a fixed number
    of milliseconds. The drawer's button list renders lazily, and on a cold
    vite compile a fixed settle once came back with ZERO sections measured —
    which the geometry check then reports as "nothing overflows", the exact
    shape of a pass that means nothing. If the drawer did not take, retry it.
    """
    for attempt in range(3):
        # Dismissed HERE, inside the loop, not once above it. The overlay is not a
        # start-up-only artifact in a linked worktree: over a borrowed
        # `node_modules` it re-fires DURING a load, and this element is `inset-0`
        # at a z-index above the app, so the click it intercepts is precisely the
        # drawer's. Dismissed once before the loop, an overlay that arrives after
        # that moment stood between every one of the three attempts and its
        # button, Playwright waited its own 30 s action timeout on a click that
        # could never land, and the pin came back `Timeout … waiting for
        # get_by_label("Open menu")` having measured nothing — which is why the
        # same instrument failure landed on a DIFFERENT node in every run of
        # #2298: a re-firing overlay looks like a random layout regression from
        # outside. `dismiss_dev_overlay`'s own docstring says the overlay
        # "swallows every click — including the one that opens the drawer", and
        # measured 6 of 6 loads of a round worktree carried it.
        dismiss_dev_overlay(page)
        menu = page.get_by_label("Open menu")
        try:
            if menu.count() > 0 and menu.first.is_visible():
                # Bounded by THIS pin's budget, not by Playwright's 30 s default:
                # three attempts that each wait half a minute for a click that
                # cannot land is 90 s of wall clock spent before the node says
                # anything, and under the gate's 8-way load that is how a pin
                # turns into a rung timeout.
                menu.first.click(timeout=timeout_ms)
                item = page.get_by_role("button", name="Dashboard")
                item.first.wait_for(state="visible", timeout=timeout_ms)
                item.first.click(timeout=timeout_ms)
        except Exception:
            if attempt == 2:
                raise
        try:
            page.wait_for_function(SECTION_POLL, timeout=timeout_ms)
            return
        except Exception:
            if attempt == 2:
                # Leave the exception to the caller's denominator guard, which
                # is what turns "no sections" into a refused verdict.
                return
            page.wait_for_timeout(1000)


def measure_page(browser, url, width, seed=None, mutate=None,
                 mutate_note="measure_page(mutate=…)"):
    """One viewport's dashboard geometry: open the tab, settle, measure.

    Split out of `run` so `tests/test_dashboard_responsive.py` asserts on THIS
    measurement instead of carrying a second copy of the JS that could drift
    from the instrument it is supposed to be checking.

    `mutate` is JS function source, run against the seeded label list once the
    dashboard is open and immediately before the measurement, and it exists for one
    question that no other shape of this file can answer: *does this measurement go
    red when the labels stop wrapping?* A class contract can be pinned by reading
    `DashboardPage.tsx`, but a verdict can only be shown to track a style by
    measuring the shipped page with that style put back. Doing it here rather than
    in a scratch checkout is the point — the app, its router, its real component
    tree and its real data stay under measurement, and only the declarations on the
    label elements change. It is printed when applied, because a measurement of an
    authored style must never be mistaken for one of the shipped page.

    `seed` is a `/api/dashboard` snapshot served in place of the backend's. It is
    intercepted at the ROUTE, not by starting a server: the dashboard makes one
    request per poll (`dashboardApi.get`), every other endpoint stays the live
    tree's, and the code under test is still the code being served — only its
    data is authored. With a seed at a phone width the context is an emulated
    iPhone (UA, DPR 3, touch); unseeded it stays the context #1685's four
    geometry pins were calibrated in, which is why those two paths do not share
    a line.
    """
    kwargs = dict(viewport={"width": width, "height": 844},
                  ignore_https_errors=True,
                  is_mobile=width < 768, has_touch=width < 768)
    if seed is not None and width < 768:
        kwargs["user_agent"] = IPHONE_UA
        kwargs["device_scale_factor"] = PHONE_DPR
    ctx = browser.new_context(**kwargs)
    try:
        page = ctx.new_page()
        errs = []
        page.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        if seed is not None:
            body = json.dumps(seed)
            ctx.route(DASHBOARD_URL_RE,
                      lambda route: route.fulfill(
                          status=200, content_type="application/json", body=body))
        page.goto(url, wait_until="load", timeout=45000)
        page.wait_for_timeout(2500)
        open_dashboard(page)
        page.wait_for_timeout(3000)
        labels = seeded_labels(seed) if seed else []
        if mutate:
            # The mutation is itself measured and printed, so a caller cannot claim a
            # differential whose treatment was never applied. The reflow after it is
            # not waited on: `scrollWidth` and `getBoundingClientRect` both force a
            # synchronous layout in Chromium, so the numbers below are post-mutation
            # whatever the frame scheduler is doing.
            applied = page.evaluate(mutate, labels)
            print(f"NOTE page mutated before measuring ({mutate_note}): {applied} "
                  f"element(s) at {width}px — the verdict below is of an authored "
                  "style, not of the shipped page", file=sys.stderr)
        res = page.evaluate(JS, labels)
        res["console_errors"] = errs[:3]
        return res
    finally:
        ctx.close()


def describe(res, width, seeded, floor=None):
    """The one line a viewport is worth printing, with its denominators.

    Shared by the CLI and the pytest pin (#2202 clause 5) so a run that passed
    and a run that measured nothing stay two different things on a terminal, and
    neither has to invent its own summary line that can drift from the other.
    """
    if "error" in res:
        return f"{width}px: probe error {res['error']}"
    clipped = res.get("clippedTotal", len(res.get("clipped") or []))
    line = (f"{width}px: {len(res['sections'])} sections"
            + (f" (floor {floor})" if floor is not None else "")
            + f", {clipped} clipped ({res.get('unreachableTotal', 0)} unreachable)")
    if seeded:
        s = res.get("seeded") or {}
        line += (f"; seeded labels {s.get('rendered', 0)}/{s.get('asked', 0)} rendered, "
                 f"{s.get('unreachable', 0)} unreachable, "
                 f"per section {s.get('bySection') or {}}")
    return line


def seed_problems(res, width):
    """Everything wrong with one viewport's seeded verdict, as messages.

    Two assertions, and the second is what makes the first mean something:
    "0 unreachable clipped labels" is also exactly what a page reports when the
    panel that holds them threw, rendered nothing, and therefore clipped nothing.
    So a short render fails in its own right, and a seed that yielded no labels at
    all fails too — a zero denominator is not a verdict, which is the same rule
    the section floor below already enforces against a page that never painted.
    """
    s = res.get("seeded") or {}
    asked, rendered = s.get("asked", 0), s.get("rendered", 0)
    out = []
    if asked == 0:
        return [f"{width}px: the seed yielded no labels, so '0 unreachable' was "
                "never going to mean anything"]
    if rendered < asked:
        out.append(f"{width}px: only {rendered} of {asked} seeded labels rendered "
                   f"(missing {s.get('missing')}) — a panel that rendered nothing "
                   "cannot also have clipped nothing")
    if s.get("unreachable", 0) > 0:
        out.append(f"{width}px: {s['unreachable']} seeded label(s) clipped with no "
                   f"title and no link to open: {s.get('unreachableDetail')}")
    return out


def run(url, widths, seed=None):
    result = {}
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for w in widths:
            try:
                res = measure_page(browser, url, w, seed=seed)
                result[str(w)] = res
            except Exception as exc:
                result[str(w)] = {"error": str(exc)[:300]}
        browser.close()

    # Exit on a verdict, not on "the browser did not crash". A cold vite
    # compile once returned zero sections at 414 px: `overflow: 0` on an empty
    # list is the same number as a clean dashboard, and reading it as a pass is
    # how a check with a zero denominator grades a broken tree as healthy.
    print(json.dumps(result, indent=1))
    problems = []
    for w, res in result.items():
        # One line per viewport, always — including the viewports that went on to
        # be refused by a floor. A run that measured nothing has to LOOK like that
        # on the terminal, not like a silent zero.
        print(describe(res, int(w), seed is not None, MIN_SECTIONS), file=sys.stderr)
        if "error" in res:
            problems.append(f"{w}px: probe error {res['error']}")
            continue
        if seed is not None:
            problems.extend(seed_problems(res, w))
        n = len(res["sections"])
        if n < MIN_SECTIONS:
            problems.append(f"{w}px: only {n} dashboard sections measured "
                            f"(expected >= {MIN_SECTIONS}) — no verdict available")
            continue
        for sec in res["sections"]:
            if sec["overflow"] > 0:
                problems.append(f"{w}px: '{sec['head']}' overflows its box by "
                                f"{sec['overflow']}px")
        if res["pageOverflow"] > 0:
            problems.append(f"{w}px: page scrolls sideways by {res['pageOverflow']}px")
    for line in problems:
        print(f"FAIL {line}", file=sys.stderr)
    print(f"{'FAIL' if problems else 'PASS'}: {len(result)} viewport(s) measured",
          file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    args = sys.argv[1:]
    seed_path, positional = None, []
    for a in args:
        if a.startswith("--seed="):
            seed_path = a.split("=", 1)[1]
        else:
            positional.append(a)
    url = positional[0] if positional else "http://127.0.0.1:5199/"
    ws = [int(a) for a in positional[1:]] or WIDTHS
    # A `--seed` path that does not exist is an error, not an unseeded run: the
    # whole point of the flag is a known denominator, and falling back to the
    # live backend would print a verdict the operator never asked for.
    seed = load_seed(seed_path) if seed_path else None
    sys.exit(run(url, ws, seed=seed))
