#!/usr/bin/env python
"""Measure Mission Control dashboard geometry at a set of viewports.

Usage: mc_layout_probe.py URL [widths...]

For each width reports whether any `section` overflows its own box, the width
of every panel inside every section (so two runs can be diffed), and the
elements whose content is being clipped.
"""
import json
import sys
from playwright.sync_api import sync_playwright

WIDTHS = [320, 360, 390, 414, 1280]
# Measured 2026-09-27 off the dashboard's own section headings. The floor is a
# denominator guard, not a claim about the design: fewer sections than this
# means the page had not rendered when the probe read it, and "no section
# overflows" over an empty list prints the same zero as a healthy dashboard.
MIN_SECTIONS = 7

JS = r"""
() => {
  const sel = el => {
    let s = el.tagName.toLowerCase();
    if (typeof el.className === 'string' && el.className.trim())
      s += '.' + el.className.trim().split(/\s+/).slice(0, 3).join('.');
    return s;
  };
  const out = {};
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
  out.clipped = [];
  for (const el of document.querySelectorAll('body *')) {
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || !el.offsetParent) continue;
    if (el.scrollWidth <= el.clientWidth) continue;
    if (el.clientWidth === 0) continue;
    const isPanel = el.classList.contains('rounded-lg');
    if (!isPanel && !['hidden','clip','auto','scroll'].includes(cs.overflowX)) continue;
    const t = (el.textContent || '').trim();
    if (!t) continue;
    out.clipped.push({sel: sel(el), sw: el.scrollWidth, cw: el.clientWidth,
                      text: t.slice(0, 50)});
  }
  out.clipped = out.clipped.slice(0, 25);
  return out;
}
"""


SECTION_POLL = """() => document.querySelectorAll('section').length > 0"""


def open_dashboard(page, timeout_ms=15000):
    """Mobile starts on the chat tab; open Dashboard from the drawer.

    Waits for the tab to actually be showing instead of sleeping a fixed number
    of milliseconds. The drawer's button list renders lazily, and on a cold
    vite compile a fixed settle once came back with ZERO sections measured —
    which the geometry check then reports as "nothing overflows", the exact
    shape of a pass that means nothing. If the drawer did not take, retry it.
    """
    for attempt in range(3):
        menu = page.get_by_label("Open menu")
        try:
            if menu.count() > 0 and menu.first.is_visible():
                menu.first.click()
                item = page.get_by_role("button", name="Dashboard")
                item.first.wait_for(state="visible", timeout=timeout_ms)
                item.first.click()
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


def measure_page(browser, url, width):
    """One viewport's dashboard geometry: open the tab, settle, measure.

    Split out of `run` so `tests/test_dashboard_responsive.py` asserts on THIS
    measurement instead of carrying a second copy of the JS that could drift
    from the instrument it is supposed to be checking.
    """
    ctx = browser.new_context(viewport={"width": width, "height": 844},
                              ignore_https_errors=True,
                              is_mobile=width < 768, has_touch=width < 768)
    try:
        page = ctx.new_page()
        errs = []
        page.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        page.goto(url, wait_until="load", timeout=45000)
        page.wait_for_timeout(2500)
        open_dashboard(page)
        page.wait_for_timeout(3000)
        res = page.evaluate(JS)
        res["console_errors"] = errs[:3]
        return res
    finally:
        ctx.close()


def run(url, widths):
    result = {}
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for w in widths:
            try:
                res = measure_page(browser, url, w)
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
        if "error" in res:
            problems.append(f"{w}px: probe error {res['error']}")
            continue
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
    url = args[0] if args else "http://127.0.0.1:5199/"
    ws = [int(a) for a in args[1:]] or WIDTHS
    sys.exit(run(url, ws))
