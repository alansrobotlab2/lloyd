"""Layout fingerprint: catch the CSS ripple the load probe cannot see.

Why this exists
---------------
`frontend_probe` verifies that Mission Control *loads*: `#root` mounted, no
console error, no pageerror, no error-boundary fallback. Its own canary measures
what that leaves unseen — the seed `stylesheet_loads_but_matches_no_rule` ships
as `"score": "blind", "checks": []` in
`~/.local/state/lloyd-automod/frontend_probe_canary/latest.json`, with the
reason recorded there: "the stylesheet is served with a 200 and matches no node,
so no channel the probe watches (load status, DOM, console, pageerror) can carry
it." A change to one component's CSS breaking *another* component's layout is
therefore invisible to the gate: the bundle is fine, the console is silent, the
mount is clean, and the page is visually broken.

What this adds
--------------
Per-section *geometry* of the served build at fixed viewport widths, collected by
the same JS `scripts/maintenance/dashboard_mobile_probe.py` uses
(`COLLECTOR_JS`, imported, not copied — the maintenance probe and this leg must
not be free to drift apart). The fingerprint is diffed against the one stored by
the last landing that passed, and every section that moved, grew, disappeared or
appeared becomes one named check — including the sections the round's diff never
names, tagged `UNTOUCHED`, which is the whole point of the leg.

Two decisions the item's text left open, settled here
-----------------------------------------------------
*What is served.* A layout leg on top of `serve_build` as written would have been
worse than nothing: with no backend behind it the dashboard never renders (the
page's own poll fails and Mission Control shows `Dashboard unavailable: ...`), so
the gate's probe record for SM_20261002_180956 carries `body_text_length: 183`
and three console errors. `MIN_SECTIONS` can never be met over an error message,
so the honest reading of that build is "no verdict", forever. `capture_build`
therefore serves the frozen snapshot in `layout_fixture` under `/api/`, which
makes rendering deterministic and keeps the fingerprint from moving because a
backlog item got filed. `frontend_probe`'s own load check is unchanged — it still
sees the app with no backend, which is what it has always measured.

*Which fields are compared.* Exactly `FINGERPRINT_FIELDS`, one module-level
constant, projected into the stored fingerprint at capture time. The owed
noise-floor measurement (#2130) is expected to shrink that tuple; because the
projection happens at write time and every stored fingerprint carries its own
`fields`, shrinking it needs no code change and never compares a field the new
set does not name. Fields whose value is *text* (the `clipped` list, panel
headings read from live rows) are excluded by construction: the live-data
variance preview over 8 reloads of the real dashboard saw `clipped` entries move
on six selectors while `cw`, `sw`, `overflow`, `panels` and `pills` did not move
at all.

Record-only
-----------
`gate.py` appends these checks to the frontend rung's report and its summary, and
never lets one turn the rung red — the `vet` observe-only precedent, and the same
grade-only ruling (#623) that already holds #1601's own promotion. Promotion is
the owed clause's numbers to earn: 10 real `web/` landings under a 10% false-fire
rate. The baseline advances only on a landing whose *other* frontend checks
passed and whose moving sections were all named by the diff, so the first ripple
cannot poison every later comparison.
"""
from __future__ import annotations

import contextlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.automod import frontend_probe as FP
from scripts.automod import state as S
from scripts.automod.layout_fixture import API_STUB
from scripts.maintenance.dashboard_mobile_probe import (
    JS as COLLECTOR_JS,
    MIN_SECTIONS,
    SECTION_POLL,
    open_dashboard,
)

#: The viewport widths a fingerprint is captured at. 390 is the narrowest width
#: Mission Control's own grid is documented against that is not a degenerate
#: 320 (where a 4-column stat strip is *expected* to fold), 1280 is the desktop
#: the `frontend` rung already loads at. Both are needed: a change that only
#: moves the mobile breakpoint and one that only moves the desktop grid are
#: different bugs, and a fingerprint that measures one of them grades half the
#: surface.
LAYOUT_WIDTHS: tuple[int, ...] = (390, 1280)

#: THE compared field set — one constant on purpose. The owed noise-floor
#: measurement (#2130) shrinks this tuple and nothing else: `fingerprint_of`
#: projects it in at capture time and every stored fingerprint carries its own
#: copy under `"fields"`, so a field that turns out to move reload-to-reload
#: leaves the comparison without a code change, and an old baseline carrying it
#: is simply not asked about. Text-bearing fields (`clipped`, panel headings read
#: from live rows) never entered: over 8 reloads of the real dashboard behind the
#: real API, `clipped` entries moved on six selectors while these five did not
#: move once.
FINGERPRINT_FIELDS: tuple[str, ...] = ("cw", "sw", "overflow", "panels", "pills")

#: Bump when the *shape* of a stored fingerprint changes, so an old file is
#: refused rather than diffed as if it were current.
SCHEMA = 1

#: Where the baseline and the leg's per-run artifacts live, beside the probe's
#: own `frontend_probe/`. Never inside the repo: the guardian alerts hourly on
#: in-tree runtime data.
LAYOUT_DIRNAME = "frontend_layout"

#: How long to wait for the page to render, and how long to sit after it did.
#: The wait is a poll for sections rather than a sleep because the dashboard
#: renders only when its `/api/dashboard` poll answers: over 8 reloads behind the
#: live API, two came back with ZERO sections at a fixed 2 s sleep — which is the
#: denominator guard firing on a page that was merely slow, the exact confusion
#: the maintenance probe's own docstring warns about.
RENDER_BUDGET_MS = 15000
SETTLE_MS = 1200
DRAWER_TIMEOUT_MS = 4000


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _brief(value: Any, limit: int = 72) -> str:
    """A field value short enough to read in a gate detail line."""
    text = json.dumps(value, sort_keys=True)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _normalise(field: str, value: Any) -> Any:
    """Project one collector field into the form it is compared in.

    Panel and pill entries come out of the collector as dicts; stored as lists of
    `[name, width, scrollWidth]` so the JSON round trip through the baseline file
    cannot change what a value equals. Dicts would sort by key on dump and come
    back with the key order of the file rather than the page.
    """
    if field in ("panels", "pills"):
        return [[(e.get("name") if isinstance(e, dict) else None),
                 int(e.get("w") or 0) if isinstance(e, dict) else 0,
                 int(e.get("sw") or 0) if isinstance(e, dict) else 0]
                for e in (value or [])]
    if field == "overflow":
        return int(value or 0)
    if field in ("cw", "sw"):
        return int(value or 0)
    return value


def fingerprint_of(raw: Mapping[str, Any], *, width: int,
                   fields: Sequence[str] = FINGERPRINT_FIELDS,
                   min_sections: int = MIN_SECTIONS) -> dict[str, Any]:
    """One viewport's fingerprint from one collector read.

    The denominator guard is the maintenance probe's (`MIN_SECTIONS`): fewer
    sections than that means the page had not rendered when it was read, and
    "nothing moved" measured over an empty list is the same zero as a healthy
    dashboard. When it fires the record carries a named no-verdict reason and an
    EMPTY `sections` map — a caller that diffs this against anything compares
    nothing, which is what it should get for a page that never appeared.
    """
    headed = [s for s in (raw.get("sections") or [])
              if str(s.get("head") or "").strip()]
    no_verdict = ""
    if len(headed) < min_sections:
        no_verdict = (f"layout-no-verdict:{width}px — only {len(headed)} section(s) "
                      f"rendered (need {min_sections}); the page had not rendered, "
                      f"so nothing here can be compared")
    seen: dict[str, int] = {}
    sections: dict[str, dict[str, Any]] = {}
    if not no_verdict:
        for sec in headed:
            head = str(sec["head"]).strip()
            seen[head] = seen.get(head, 0) + 1
            key = head if seen[head] == 1 else f"{head} #{seen[head]}"
            sections[key] = {f: _normalise(f, sec.get(f)) for f in fields}
    return {"width": width, "sections_measured": len(headed),
            "no_verdict": no_verdict, "sections": sections,
            "page_overflow": int(raw.get("pageOverflow") or 0)}


def _fields_in(views: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """The fields the stored records actually carry, read off the records."""
    return sorted({f for v in views.values()
                   for sec in (v.get("sections") or {}).values() for f in sec})


def fingerprint(views: Mapping[str, Mapping[str, Any]], *, round_id: str = "",
                build: str = "",
                fields: Sequence[str] | None = None) -> dict[str, Any]:
    """The storable document: every width's fingerprint under one `views` map.

    `"fields"` is read off the records rather than restated from the module
    constant, because the file must describe what it holds: a record projected
    under a shrunken `FINGERPRINT_FIELDS` has to say so, or a later reader would
    diff it against a baseline that carries a field it no longer compares and
    never find out why nothing fires.
    """
    return {"schema": SCHEMA, "captured_at": _now(), "round_id": round_id,
            "build": build,
            "fields": list(fields) if fields is not None
            else (_fields_in(views) or list(FINGERPRINT_FIELDS)),
            "views": {str(w): dict(v) for w, v in views.items()}}


@contextlib.contextmanager
def _served(out_dir: Path):
    """The build served with the frozen snapshot behind `/api/`."""
    with FP.serve_build(out_dir, api_stub=API_STUB) as base_url:
        yield base_url


def capture_build(out_dir: Path, *, widths: Sequence[int] = LAYOUT_WIDTHS,
                  chromium: str | Path | None = None,
                  api_stub: Mapping[str, Any] | None = API_STUB,
                  settle_ms: int = SETTLE_MS,
                  render_budget_ms: int = RENDER_BUDGET_MS,
                  shots_dir: Path | None = None) -> dict[str, Any]:
    """Collect one fingerprint per width from the build output in `out_dir`.

    Returns `{"skipped": why}` when there is no browser to collect with — the same
    honest shape as `frontend_probe`, because a leg that silently measured
    nothing would report a clean layout forever.
    """
    why = FP.unavailable(out_dir, str(chromium) if chromium else None)
    if why:
        return {"skipped": why}
    exe = str(chromium or FP.CHROMIUM)
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:                          # noqa: BLE001
        return {"skipped": f"playwright is not importable here: {exc}"}

    views: dict[str, Any] = {}
    try:
        with _served(out_dir) if api_stub is API_STUB \
                else FP.serve_build(out_dir, api_stub=api_stub) as base_url:
            with sync_playwright() as pw:
                browser = pw.chromium.launch(
                    executable_path=exe,
                    headless=True,
                    args=["--no-sandbox", "--disable-setuid-sandbox",
                          "--disable-dev-shm-usage"])
                try:
                    for width in widths:
                        views[str(width)] = _capture_one(
                            browser, base_url, width, settle_ms=settle_ms,
                            render_budget_ms=render_budget_ms,
                            shots_dir=shots_dir)
                finally:
                    browser.close()
    except Exception as exc:                          # noqa: BLE001
        return {"skipped": f"serving or collecting died: {exc.__class__.__name__}: "
                           f"{str(exc)[:200]}"}
    return {"views": views}


def _capture_one(browser: Any, base_url: str, width: int, *,
                 settle_ms: int, render_budget_ms: int,
                 shots_dir: Path | None) -> dict[str, Any]:
    """One viewport: load, wait for the dashboard to appear, read the geometry."""
    mobile = width < 768
    ctx = browser.new_context(viewport={"width": width, "height": 900},
                              is_mobile=mobile, has_touch=mobile)
    try:
        page = ctx.new_page()
        try:
            page.goto(base_url + "/", wait_until="load",
                      timeout=max(render_budget_ms, 10000))
        except Exception as exc:                      # noqa: BLE001
            return fingerprint_of({"sections": []}, width=width) | {
                "capture_error": f"the load timed out: {exc.__class__.__name__}"}
        try:
            page.wait_for_function(SECTION_POLL, timeout=render_budget_ms)
        except Exception:                             # noqa: BLE001
            # Mobile boots on the chat tab, so on a narrow viewport the dashboard
            # is behind the drawer. `open_dashboard` re-checks and returns; its own
            # failure is what the denominator guard below turns into a verdict.
            if mobile:
                with contextlib.suppress(Exception):
                    open_dashboard(page, timeout_ms=_drawer_budget(render_budget_ms))
        page.wait_for_timeout(settle_ms)
        raw = page.evaluate(COLLECTOR_JS)
        if shots_dir is not None:
            Path(shots_dir).mkdir(parents=True, exist_ok=True)
            with contextlib.suppress(Exception):
                page.screenshot(path=str(Path(shots_dir) / f"layout-{width}.png"))
        return fingerprint_of(raw or {"sections": []}, width=width)
    finally:
        ctx.close()


def _drawer_budget(render_budget_ms: int) -> int:
    """The drawer's own budget: short. A page that needs 15 s to open a drawer is
    not a page this leg is going to measure well, and the canary runs this table
    once per seed."""
    return max(1000, min(DRAWER_TIMEOUT_MS, render_budget_ms // 3))


def changed_sections(baseline: Mapping[str, Any], current: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every (width, section) pair whose projected fingerprint differs.

    Pure: two stored fingerprints in, a list out, no browser and no filesystem.
    A width present on one side only, or with a no-verdict on either side, is
    reported under `skipped_widths` rather than as a change — a page that did not
    render at 390 did not "change at 390".
    """
    out: list[dict[str, Any]] = []
    bviews, cviews = baseline.get("views") or {}, current.get("views") or {}
    for width in sorted(set(bviews) | set(cviews), key=lambda w: (len(w), w)):
        b, c = bviews.get(str(width)), cviews.get(str(width))
        if b is None or c is None or b.get("no_verdict") or c.get("no_verdict"):
            # Not comparable, which `skipped_widths` reports; never a "change".
            continue
        fields = list(current.get("fields") or FINGERPRINT_FIELDS)
        for key in sorted(set(b.get("sections") or {}) | set(c.get("sections") or {})):
            bf = (b.get("sections") or {}).get(key)
            cf = (c.get("sections") or {}).get(key)
            if bf is None or cf is None:
                out.append({"width": int(width), "section": key,
                            "kind": "appeared" if bf is None else "disappeared",
                            "fields": {f: {"was": bf, "now": cf} for f in fields},
                            "was": bf, "now": cf})
                continue
            moved = {f: {"was": bf.get(f), "now": cf.get(f)}
                     for f in fields if bf.get(f) != cf.get(f)}
            if moved:
                out.append({"width": int(width), "section": key, "kind": "moved",
                            "fields": moved, "was": bf, "now": cf})
    return out


def skipped_widths(baseline: Mapping[str, Any],
                   current: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Widths that could not be compared at all, with the reason each one.

    Separate from `changed_sections` on purpose: "this width moved" and "this
    width could not be read" are different findings, and folding the second into
    the first is how an unrendered page ends up reported as a layout break — or,
    read the other way, how a page that rendered on one side only gets graded as
    clean.
    """
    out: list[dict[str, Any]] = []
    bviews, cviews = baseline.get("views") or {}, current.get("views") or {}
    for width in sorted(set(bviews) | set(cviews), key=lambda w: (len(w), w)):
        b, c = bviews.get(str(width)), cviews.get(str(width))
        if b is None or c is None:
            out.append({"width": width,
                        "reason": "captured at this width on one side only"})
        elif b.get("no_verdict") or c.get("no_verdict"):
            out.append({"width": width,
                        "reason": str(b.get("no_verdict") or c.get("no_verdict"))})
    return out


def attribution(section: str, *, owners: Mapping[str, Sequence[str]] | None,
                changed_paths: Sequence[str]) -> str:
    """`in-diff` when a source file naming this section is among the changed paths.

    The mapping is built by `owners_from_sources` — a grep of `web/` for the
    section's heading literal, which is how a heading reaches the rendered DOM
    (`<Section title="System">`). A section the grep cannot resolve is
    `UNTOUCHED`, and so is one whose owner the round genuinely did not change:
    the untouched set is the finding this leg exists to produce, so the default
    errs toward naming it.
    """
    if not changed_paths:
        return "UNTOUCHED"
    paths = (owners or {}).get(section) or []
    changed = set(changed_paths)
    return "in-diff" if any(p in changed for p in paths) else "UNTOUCHED"


def diff(baseline: Mapping[str, Any], current: Mapping[str, Any], *,
         owners: Mapping[str, Sequence[str]] | None = None,
         changed_paths: Sequence[str] = ()) -> list[dict[str, Any]]:
    """One failing check per changed SECTION — not per field, not per viewport.

    A section that moved at both viewports is one finding about one section, with
    both widths in its detail; a section that did not move emits nothing at all.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for change in changed_sections(baseline, current):
        grouped.setdefault(change["section"], []).append(change)
    checks: list[dict[str, Any]] = []
    for section in sorted(grouped):
        moves = grouped[section]
        widths = ", ".join(f"{m['width']}px" for m in moves)
        parts: list[str] = []
        for m in moves:
            if m["kind"] != "moved":
                parts.append(f"{m['width']}px {m['kind']}")
                continue
            parts.append(f"{m['width']}px " + "; ".join(
                f"{f} {_brief(d['was'])}->{_brief(d['now'])}"
                for f, d in sorted(m["fields"].items())))
        checks.append({
            "check": f"layout-changed:{section}",
            "ok": False,
            "channel": "layout",
            "section": section,
            "kind": moves[0]["kind"],
            "attribution": attribution(section, owners=owners,
                                       changed_paths=changed_paths),
            "widths": [m["width"] for m in moves],
            "detail": f"{section} moved ({widths}): " + " | ".join(parts),
            "delta": {str(m["width"]): m["fields"] for m in moves},
        })
    return checks


def owners_from_sources(source_root: Path,
                        headings: Sequence[str]) -> dict[str, list[str]]:
    """Map each rendered heading to the `web/` sources that name it literally.

    `web/src/components/pages/DashboardPage.tsx` holds `<Section title="System">`,
    so the heading's own string is the join key from a rendered section back to a
    path in the round's diff. Quoted-or-bracketed containment, not a bare
    substring: "System" as a substring lives in half the tree, and an attribution
    that says `in-diff` because the word appeared somewhere is worse than the
    honest `UNTOUCHED`.
    """
    root = Path(source_root)
    want = [h for h in headings if h]
    found: dict[str, list[str]] = {h: [] for h in want}
    if not root.is_dir() or not want:
        return found
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in (".tsx", ".ts", ".jsx", ".js", ".css"):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        try:
            rel = "web/" + str(path.relative_to(root))
        except ValueError:
            rel = str(path)
        for head in want:
            if (f'"{head}"' in text or f"'{head}'" in text
                    or f"`{head}`" in text or f">{head}<" in text):
                found[head].append(rel)
    return found


def baseline_path() -> Path:
    return S.STATE_DIR / LAYOUT_DIRNAME / "baseline.json"


def load_baseline() -> dict[str, Any] | None:
    """The last stored fingerprint, or None. Never raises: a leg with no baseline
    has nothing to compare, which is a state it reports, not an error."""
    try:
        raw = json.loads(baseline_path().read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict) or raw.get("schema") != SCHEMA:
        return None
    return raw


def store_baseline(fp: Mapping[str, Any], *, round_id: str = "") -> Path:
    path = baseline_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = dict(fp)
    doc["advanced_by"] = round_id
    # Written whole, never appended: a half-written baseline would be read as a
    # fingerprint with sections missing, which is exactly the false fire this leg
    # is supposed to be able to tell apart from a real one.
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True))
    tmp.replace(path)
    return path


def maybe_advance(report: Mapping[str, Any], *, advanced: bool,
                  round_id: str = "") -> str:
    """Store the report's fingerprint if, and only if, this landing may bless it.

    The rule the item asked to be defined now: a baseline advances on a landing
    whose OTHER frontend checks passed, that produced a verdict at every width,
    and whose every moved section was named by the round's own diff. Layout
    changes legitimately on purpose — but then the diff names the section, and
    re-blessing is that landing's job. A landing whose diff says nothing about
    `Services` must not store a fingerprint with `Services` moved, because that
    is how the first ripple becomes the new truth and every later ripple reads as
    normal.
    """
    if not report.get("captured"):
        return f"held: the leg did not run ({report.get('skipped') or 'no capture'})"
    if not advanced:
        return "held: the frontend rung's other checks did not pass"
    if report.get("no_verdict"):
        widths = ", ".join(str(n.get("width")) for n in report["no_verdict"])
        return f"held: no layout verdict at {widths}px"
    untouched = [c for c in report.get("checks") or []
                 if c.get("attribution") == "UNTOUCHED"]
    if untouched:
        return (f"held: {len(untouched)} UNTOUCHED section(s) moved "
                f"({', '.join(str(c['section']) for c in untouched[:3])})")
    if not report.get("fingerprint"):
        return "held: no fingerprint was captured"
    store_baseline(report["fingerprint"], round_id=round_id)
    return "stored"


def run(out_dir: Path, *, changed_paths: Sequence[str] = (),
        chromium: str | Path | None = None, widths: Sequence[int] = LAYOUT_WIDTHS,
        source_root: Path | None = None, shots_dir: Path | None = None,
        round_id: str = "", baseline: Mapping[str, Any] | None = None,
        settle_ms: int = SETTLE_MS) -> dict[str, Any]:
    """Capture, diff against `baseline`, and project a record-only report.

    Writes only into `S.STATE_DIR`, never into the tree under review.
    """
    cap = capture_build(out_dir, widths=widths, chromium=chromium,
                        shots_dir=shots_dir, settle_ms=settle_ms)
    if cap.get("skipped"):
        return {"captured": False, "skipped": cap["skipped"], "checks": [],
                "no_verdict": [], "fingerprint": None, "widths": list(widths),
                "sections_compared": 0, "baseline": "not-read"}

    views = cap["views"]
    fp = fingerprint(views, round_id=round_id, build=str(out_dir))
    no_verdict = [{"check": f"layout-no-verdict:{v['width']}px",
                   "detail": v["no_verdict"], "width": v["width"]}
                  for v in views.values() if v.get("no_verdict")]
    sections = [k for v in views.values() for k in (v.get("sections") or {})]
    report: dict[str, Any] = {
        "captured": True, "skipped": "", "fingerprint": fp,
        "no_verdict": no_verdict, "widths": sorted(int(w) for w in views),
        "sections_compared": len(set(sections)), "checks": [],
        "baseline": "absent",
    }
    if baseline is None:
        # The first landing after this leg exists has nothing to compare against.
        # "no baseline" is a state, not a pass: it must not be readable as
        # "nothing moved", or the very first layout break after a state wipe is
        # certified by the absence of a file.
        report["baseline"] = "absent"
        return report
    report["baseline"] = "present"
    owners = (owners_from_sources(Path(source_root), sorted(set(sections)))
              if source_root is not None else None)
    report["checks"] = diff(baseline, fp, owners=owners,
                            changed_paths=list(changed_paths))
    # Widths the diff could not compare at all, computed and carried in the
    # artifact. A report that says `0 checks` because one side never had the width
    # has to read as no-verdict and not as clean, and the only way to know that from
    # the file alone is for the file to name the widths it dropped. The empty-list
    # case is the point: a diff over a page that was never laid out prints the same
    # zero as a stable page, which is the denominator rule #1752's shape.
    report["widths_unreadable"] = skipped_widths(baseline, fp)
    return report


def summary(report: Mapping[str, Any]) -> str:
    """One line for the rung detail and the ledger."""
    if not report or not report.get("captured"):
        return f"LAYOUT SKIPPED ({(report or {}).get('skipped') or 'no capture'})"
    widths = ", ".join(f"{w}px" for w in report.get("widths") or [])
    if report.get("no_verdict"):
        n = report["no_verdict"][0]
        return (f"LAYOUT NO VERDICT ({n['detail'].split(' — ')[0]} at {n['width']}px"
                f"; {len(report['no_verdict'])} width(s) unreadable)")
    if report.get("baseline") == "absent":
        return (f"LAYOUT BASELINE ABSENT ({report['sections_compared']} section(s) "
                f"fingerprinted at {widths}; nothing to compare yet)")
    checks = report.get("checks") or []
    # Read BEFORE the clean branch, not after it: `checks == []` is also what a
    # comparison over a page that was never laid out looks like, and a line that said
    # `LAYOUT ok` there would be the clean-looking zero the denominator rule exists
    # to forbid. The widths that could not be compared are named, so the reader sees
    # how much of the page the word "ok" is actually speaking for.
    unreadable = report.get("widths_unreadable") or []
    caveat = (f" — {len(unreadable)} width(s) NOT compared "
              f"({', '.join(str(w['width']) + 'px' for w in unreadable[:3])})"
              if unreadable else "")
    if not checks:
        return (f"LAYOUT PARTIAL, 0 checks ({report['sections_compared']} section(s) "
                f"compared at {widths} against the stored baseline){caveat}"
                if unreadable else
                f"LAYOUT ok ({report['sections_compared']} section(s) stable at "
                f"{widths} against the stored baseline)")
    untouch = [c for c in checks if c.get("attribution") == "UNTOUCHED"]
    named = ", ".join(f"{c['section']} {c['attribution']}" for c in checks[:4])
    return (f"LAYOUT MOVED {len(checks)} section(s) [{named}"
            f"{'.' if len(checks) <= 4 else '…'}] — {len(untouch)} UNTOUCHED")


def without_fingerprint(report: Mapping[str, Any]) -> dict[str, Any]:
    """The report as it goes into `gate.json`: everything but the fingerprint.

    A fingerprint is tens of kilobytes of numbers per landing; `gate.json` is
    copied onto the landing commit's message and read by a human. The fingerprint
    itself belongs in the leg's artifact directory and in the baseline file.
    """
    return {k: v for k, v in report.items() if k != "fingerprint"}


def write_artifact(round_id: str, report: Mapping[str, Any]) -> Path:
    """Record the run under the leg's state dir; never inside the repo."""
    path = S.STATE_DIR / LAYOUT_DIRNAME / (
        "latest.json" if round_id in ("", "latest") else f"{round_id}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1, sort_keys=True, default=str))
    return path

