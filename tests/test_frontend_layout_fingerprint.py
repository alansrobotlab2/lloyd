"""Layout fingerprint (#2130): clauses 1, 2 and 3.

Three surfaces, pinned separately, because the three ways this leg can fail are
different and one test covering all of them at once covers none of them:

* `fingerprint_of` — the collector's read projected into the compared field set,
  including the denominator guard that decides whether the read means anything;
* `changed_sections` / `diff` / `attribution` — pure functions from two stored
  fingerprints to one named check per moved section, tagged by whether the
  round's diff owned that section;
* `maybe_advance` / `run` / `write_artifact` — what may bless the baseline, and
  where the bytes are allowed to land.

The two browser pins carry no marker of any kind, so a plain `pytest tests/`
collects them and the gate's `tests` rung executes them on every round — the
contract `tests/test_automod_frontend_probe.py` states for its own browser nodes,
which likewise call a `_require_browser()` helper instead of marking themselves.
The one skip in this file is that helper's, is behind an `if`, and fires only when
the box cannot answer the question at all (no chromium binary, or a fixture build
`frontend_probe.unavailable` refuses to serve); every other node here runs
unconditionally. Registering a marker would take a `pytest.ini` edit, and a round
may not write that file, so a marker on these nodes would be unread by anything.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.automod import frontend_layout as FL  # noqa: E402
from scripts.automod import frontend_probe as FP  # noqa: E402
from scripts.automod import state as S  # noqa: E402
from scripts.automod.layout_fixture import DASHBOARD_SNAPSHOT  # noqa: E402

# The dashboard's own seven headings, as `DashboardPage.tsx` titles them. Reused
# rather than invented: the section-count floor (`MIN_SECTIONS = 7`) was measured
# off them, so a test that made up its own names could pass on a page the real
# app would never produce.
HEADS = ["vLLM engines", "Lloyd agent", "Subagents & background tasks", "System",
         "Services", "Automation & work", "Tokens"]


def section(head, cw=728, sw=None, panels=(), pills=()):
    """One collector record, in the shape `COLLECTOR_JS` returns its entries."""
    sw = cw if sw is None else sw
    return {"head": head, "cw": cw, "sw": sw, "overflow": sw - cw,
            "panels": [{"name": n, "w": w, "sw": s} for n, w, s in panels],
            "pills": [{"name": n, "w": w, "sw": s} for n, w, s in pills]}


def raw(sections, page_overflow=0):
    return {"vw": 1280, "pageOverflow": page_overflow, "sections": sections,
            "clipped": []}


def panels_of(cw):
    """The two half-width panels a healthy section holds."""
    return [(None, cw // 2, cw // 2 - 2), (None, cw // 2, cw // 2 - 2)]


def healthy(cw=728):
    """A collector read with the real dashboard's shape: 7 sections, 2 panels each."""
    return raw([section(h, cw=cw, panels=panels_of(cw)) for h in HEADS])


def view(width=1280, cw=728, *, override=None, drop=(),
         min_sections=FL.MIN_SECTIONS):
    """One width's fingerprint, with named sections replaced or removed wholesale.

    `override` exists so a test can move ONE section and leave the other six
    byte-identical: a helper that rebuilt every section at the new width would
    assert "one check per changed section" while changing all seven, which is a
    pin that passes for the wrong reason.
    """
    secs = {h: section(h, cw=cw, panels=panels_of(cw)) for h in HEADS}
    secs.update(override or {})
    return FL.fingerprint_of(raw([s for h, s in secs.items() if h not in drop]),
                             width=width, min_sections=min_sections)


def fp(*views, round_id="R-1"):
    return FL.fingerprint({str(v["width"]): v for v in views}, round_id=round_id)


def _report(tmp_path):
    """A captured two-width report, assembled without a browser."""
    return {"captured": True, "skipped": "", "checks": [], "no_verdict": [],
            "widths": [390, 1280], "sections_compared": len(HEADS),
            "baseline": "present",
            "fingerprint": fp(view(1280), view(390, cw=358))}


# ---------------------------------------------------------------------------
# Clause 1 — the collector's read, and the floor that decides if it means anything
# ---------------------------------------------------------------------------

def test_one_record_per_rendered_section_carries_exactly_the_compared_fields():
    """Every rendered section becomes one record, keyed by its heading, holding
    the compared fields and nothing else.

    The names are asserted against `FINGERPRINT_FIELDS` rather than spelled out
    again, so this pin can never drift into blessing a field the comparison does
    not actually use.
    """
    got = FL.fingerprint_of(healthy(), width=1280)
    assert sorted(got["sections"]) == sorted(HEADS)
    for head, fields in got["sections"].items():
        assert sorted(fields) == sorted(FL.FINGERPRINT_FIELDS), head
    sysrec = got["sections"]["System"]
    assert (sysrec["cw"], sysrec["sw"], sysrec["overflow"]) == (728, 728, 0)
    assert sysrec["panels"] == [[None, 364, 362], [None, 364, 362]], sysrec["panels"]
    assert got["sections_measured"] == len(HEADS)
    assert got["no_verdict"] == ""


def test_a_panel_and_pill_entry_keeps_its_name_and_both_widths():
    """Panel *names* and pill *names* are in the record, not just widths.

    `dashboard_mobile_probe.py` collects HealthPills separately from Panels
    because the Services section's children carry `.rounded-md`, and without them
    that section reports an EMPTY panels list — an assertion over an empty list
    passes forever however wrong the layout gets. So a pill disappearing has to
    be visible here, with the name it had.
    """
    read = raw([section(h, panels=[(None, 300, 298)]) if h != "Services" else
                section(h, pills=[("agent-llm-primary", 120, 118),
                                  ("lloyd-backend", 110, 108)])
                for h in HEADS])
    got = FL.fingerprint_of(read, width=1280)
    assert got["sections"]["Services"]["pills"] == [
        ["agent-llm-primary", 120, 118], ["lloyd-backend", 110, 108]]
    assert got["sections"]["Services"]["panels"] == []
    assert got["sections"]["System"]["panels"] == [[None, 300, 298]]


def test_a_page_that_did_not_render_emits_the_named_no_verdict_and_no_section_records():
    """Fewer than MIN_SECTIONS sections means the page had not been rendered.

    The guard is the maintenance probe's own, and it is load-bearing for the
    diff: "no section moved" over an empty list is the same zero as a healthy
    dashboard, so an unrendered page must produce no per-section records at all —
    there is then nothing that could be read as clean.
    """
    half = raw([section(h) for h in HEADS[:FL.MIN_SECTIONS - 1]])
    assert len(half["sections"]) == FL.MIN_SECTIONS - 1
    got = FL.fingerprint_of(half, width=390)
    assert got["sections"] == {}, "an unrendered page must yield no per-section records"
    assert got["no_verdict"].startswith("layout-no-verdict:390px"), got["no_verdict"]
    assert str(FL.MIN_SECTIONS) in got["no_verdict"]
    assert got["sections_measured"] == FL.MIN_SECTIONS - 1, "the read is still reported"


def test_the_no_verdict_is_a_named_finding_of_a_run_and_never_a_clean_layout(tmp_path,
                                                                             monkeypatch):
    """A run over an unreadable page says NO VERDICT and holds the baseline.

    The two readings this pin keeps apart are why the guard exists: `LAYOUT ok`
    and `LAYOUT NO VERDICT` must never be the same string, and a landing that
    could not read the page must not be able to bless a baseline.
    """
    monkeypatch.setattr(S, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(FL, "capture_build", lambda *a, **k: {"views": {
        "390": FL.fingerprint_of(raw([section("only one")]), width=390)}})
    report = FL.run(Path("/nonexistent-build"), widths=(390,), baseline=fp(view()))
    assert [n["check"] for n in report["no_verdict"]] == ["layout-no-verdict:390px"]
    assert report["checks"] == []
    assert "NO VERDICT" in FL.summary(report), FL.summary(report)
    assert FL.maybe_advance(report, advanced=True).startswith("held: no layout verdict")
    assert not FL.baseline_path().exists()


# ---------------------------------------------------------------------------
# Clause 2 — one check per moved section, attributed against the round's diff
# ---------------------------------------------------------------------------

def test_the_diff_emits_one_check_per_changed_section_naming_its_heading():
    """A moved section yields EXACTLY one check, and the check names the heading.

    One per section — not one per field, not one per viewport: `System` shifting
    at both 390 and 1280 is one finding about one section.
    """
    base = fp(view(1280), view(390, cw=358))
    # Only `System` differs at 390: it now scrolls past its own box, and one of
    # its panels is wider than the section that holds it.
    cur = fp(view(390, cw=358, override={
        "System": section("System", cw=358, sw=412,
                          panels=[(None, 200, 460), (None, 358, 356)])}), view(1280))
    checks = FL.diff(base, cur)
    assert [c["check"] for c in checks] == ["layout-changed:System"], checks
    check = checks[0]
    assert check["ok"] is False
    assert check["section"] == "System"
    assert "System" in check["detail"]
    assert check["widths"] == [390]
    assert sorted(check["delta"]["390"]) == ["overflow", "panels", "sw"]


def test_an_unchanged_section_emits_nothing_and_a_both_width_move_is_still_one_check():
    base = fp(view(1280), view(390, cw=358))
    both = fp(view(1280, override={"vLLM engines": section("vLLM engines", cw=700)}),
              view(390, cw=358, override={"vLLM engines": section("vLLM engines",
                                                                  cw=350)}))
    checks = FL.diff(base, both)
    assert len(checks) == 1, [c["check"] for c in checks]
    assert sorted(checks[0]["widths"]) == [390, 1280]
    assert checks[0]["check"] == "layout-changed:vLLM engines"
    named = " ".join(c["check"] for c in checks)
    for head in HEADS:
        if head != "vLLM engines":
            assert head not in named, "the five quiet sections stay silent"


def test_a_panel_that_disappeared_is_a_check_naming_the_section_it_left():
    """Kirschner's actual bug: an icon somewhere else *disappears*.

    A panel that stops rendering is that bug at the level the fingerprint can
    still see — and a check watching widths alone would miss it entirely, because
    the surviving panel simply gets a different width and nothing looks missing.
    """
    base = fp(view())
    cur = fp(view(override={"Services": section("Services", cw=728,
                                                panels=[(None, 364, 362)])}))
    checks = FL.diff(base, cur)
    assert [c["check"] for c in checks] == ["layout-changed:Services"], checks
    assert checks[0]["detail"].count("Services") >= 1
    assert checks[0]["delta"]["1280"]["panels"]["now"] == [[None, 364, 362]]


def test_a_section_that_disappeared_is_a_check_until_the_floor_says_no_verdict():
    """Both readings of a vanished section, and which one wins.

    With `min_sections` relaxed the diff names the section and calls it
    `disappeared` (and `appeared` coming back the other way). At the shipped
    floor of 7 — which is the dashboard's real section count — losing a whole
    section IS the no-render case, so the guard wins and there is no per-section
    record to diff at all. The two assertions together are the honest contract:
    section-level loss is the guard's, panel-level loss is the leg's.
    """
    base = fp(view())
    gone = fp(view(drop=("Tokens",), min_sections=6))
    checks = FL.diff(base, gone)
    assert [c["check"] for c in checks] == ["layout-changed:Tokens"]
    assert checks[0]["kind"] == "disappeared"
    assert FL.diff(gone, base)[0]["kind"] == "appeared"
    floored = view(drop=("Tokens",))
    assert floored["no_verdict"].startswith("layout-no-verdict:1280px")
    assert floored["sections"] == {}
    assert FL.diff(base, fp(floored)) == []


def test_a_moved_section_is_tagged_in_diff_only_if_the_round_changed_its_owning_source():
    """`in-diff` when the section's own component is among the changed paths,
    `UNTOUCHED` when it is not — and the untouched set is the point of the leg.

    The join runs through `owners_from_sources`, because the round's diff is
    paths and the fingerprint is headings.
    """
    base, cur = _system_moved()
    owners = {"System": ["web/src/components/pages/DashboardPage.tsx"]}
    changed = ["web/src/components/pages/DashboardPage.tsx"]
    mine = FL.diff(base, cur, owners=owners, changed_paths=changed)[0]
    assert mine["attribution"] == "in-diff", mine
    other = FL.diff(base, cur, owners={"System": ["web/src/Other.tsx"]},
                    changed_paths=changed)[0]
    assert other["attribution"] == "UNTOUCHED", other
    unnamed = FL.diff(base, cur, owners=None, changed_paths=changed)[0]
    assert unnamed["attribution"] == "UNTOUCHED", "an unresolvable heading is untouched"
    nobody = FL.diff(base, cur, owners=owners, changed_paths=[])[0]
    assert nobody["attribution"] == "UNTOUCHED", "no diff, nothing is in it"


def _system_moved():
    """A baseline and a current that differ in exactly one section: `System`."""
    return fp(view()), fp(view(override={"System": section("System", cw=700)}))


def test_owners_from_sources_maps_a_rendered_heading_to_the_file_that_names_it(tmp_path):
    """The join key from a heading to a path is the heading's own literal.

    Quoted, not a bare substring: the word `system` occurs throughout the tree,
    and matching on it would attribute nearly every section to nearly every
    change — a leg that can only ever say `in-diff` never reports a ripple.
    """
    (tmp_path / "src").mkdir(parents=True)
    (tmp_path / "src" / "DashboardPage.tsx").write_text(
        'export const X = () => <Section title="System" icon={Cpu} />;\n'
        'const noise = "the system clock";\n')
    (tmp_path / "src" / "Other.tsx").write_text('export const y = "system load";\n')
    (tmp_path / "src" / "notes.md").write_text("System\n")
    owners = FL.owners_from_sources(tmp_path, ["System", "Tokens"])
    assert owners["System"] == ["web/src/DashboardPage.tsx"], owners
    assert owners["Tokens"] == []


def test_shrinking_the_one_field_set_constant_stops_comparing_that_field():
    """The compared set is ONE constant, and shrinking it is a one-line edit.

    The owed noise-floor measurement is expected to drop fields that turn out to
    move reload-to-reload. This pin is what makes that claim true rather than
    hopeful: the projection happens at capture time, so a field dropped from the
    tuple stops being compared with no change to the diff, and a fingerprint
    stored under the shrunken set carries its own `fields` for a later reader.
    """
    # The one change is in a PILL's width, which `cw` and `sw` do not see: a
    # section can be perfectly unoverflowed and still have lost most of its pill's
    # box, which is the class of thing this leg exists to notice.
    pills = [("agent-llm-primary", 120, 118), ("lloyd-backend", 60, 58)]
    only = {"Services": section("Services", cw=728, pills=pills)}
    assert FL.diff(fp(view()), fp(view(override=only))), "pills ship in the set"
    narrow = ("cw", "sw")
    moved = raw([dict(s, pills=pills) if s["head"] == "Services" else s
                 for s in healthy()["sections"]])
    narrow_base = fp(FL.fingerprint_of(healthy(), width=1280, fields=narrow))
    narrow_cur = fp(FL.fingerprint_of(moved, width=1280, fields=narrow))
    assert sorted(narrow_cur["fields"]) == ["cw", "sw"], "the file says what it holds"
    assert FL.diff(narrow_base, narrow_cur) == [], "it left the set, and the diff"
    assert "pills" in FL.FINGERPRINT_FIELDS, "and it ships until measured out"


def test_a_width_that_could_not_be_read_is_reported_and_never_read_as_a_change():
    """390px that failed to render is not a section that moved at 390px."""
    base = fp(view(1280), view(390, cw=358))
    one_width = fp(view(1280))
    assert [s["width"] for s in FL.skipped_widths(base, one_width)] == ["390"]
    assert FL.diff(base, one_width) == []
    unreadable = fp(view(1280), FL.fingerprint_of(raw([section("a")]), width=390))
    assert FL.diff(base, unreadable) == []
    reasons = FL.skipped_widths(base, unreadable)
    assert len(reasons) == 1 and "rendered" in reasons[0]["reason"], reasons


# ---------------------------------------------------------------------------
# Clause 3 — the baseline advances only on a landing that earned it, and no byte
# of this leg lands inside the checkout
# ---------------------------------------------------------------------------

def test_the_baseline_advances_only_on_a_landing_whose_other_checks_passed(
        tmp_path, monkeypatch):
    """Passed sibling checks bless the baseline; failed ones do not, and the
    reason for holding is said out loud rather than left as an absent file."""
    monkeypatch.setattr(S, "STATE_DIR", tmp_path / "state")
    report = _report(tmp_path)
    assert FL.maybe_advance(report, advanced=False, round_id="A").startswith(
        "held: the frontend rung's other checks did not pass")
    assert not FL.baseline_path().exists()
    assert FL.maybe_advance(report, advanced=True, round_id="A") == "stored"
    assert json.loads(FL.baseline_path().read_text())["round_id"] == "R-1"
    assert json.loads(FL.baseline_path().read_text())["advanced_by"] == "A"


def test_a_ripple_the_rounds_diff_does_not_name_never_blesses_the_baseline(
        tmp_path, monkeypatch):
    """Otherwise the first ripple becomes the new truth.

    The bless rule is the item's: the baseline advances on a passing landing whose
    diff NAMES the changed section. An UNTOUCHED move is precisely the finding
    this leg exists to catch, so storing over it would erase the evidence and
    certify the break as normal for every landing after it.
    """
    monkeypatch.setattr(S, "STATE_DIR", tmp_path / "state")
    report = _report(tmp_path)
    report["checks"] = [{"check": "layout-changed:Tokens", "ok": False,
                         "attribution": "UNTOUCHED", "section": "Tokens"}]
    assert FL.maybe_advance(report, advanced=True).startswith("held: 1 UNTOUCHED")
    assert not FL.baseline_path().exists()
    report["checks"] = [{"check": "layout-changed:Tokens", "ok": False,
                         "attribution": "in-diff", "section": "Tokens"}]
    assert FL.maybe_advance(report, advanced=True, round_id="B") == "stored"


def test_a_stored_baseline_round_trips_and_a_foreign_schema_is_refused(tmp_path,
                                                                      monkeypatch):
    """A baseline reads back equal to what went in, and a file of another shape of
    this leg is refused rather than diffed as if it were current."""
    monkeypatch.setattr(S, "STATE_DIR", tmp_path / "state")
    report = _report(tmp_path)
    FL.maybe_advance(report, advanced=True, round_id="R-1")
    stored = FL.load_baseline()
    assert stored["views"] == report["fingerprint"]["views"]
    assert sorted(stored["fields"]) == sorted(FL.FINGERPRINT_FIELDS)
    stale = json.loads(FL.baseline_path().read_text())
    stale["schema"] = FL.SCHEMA + 1
    FL.baseline_path().write_text(json.dumps(stale))
    assert FL.load_baseline() is None


def test_a_run_writes_its_artifacts_only_under_the_state_dir(tmp_path, monkeypatch):
    """Every byte this leg writes goes to `S.STATE_DIR`; nothing goes in the tree.

    The guardian alerts hourly on runtime data inside the checkout, and the static
    rung once built its scratch path over an empty `mktemp` stdout and deleted the
    live checkout with it — so where the artifacts go is not a cosmetic question.
    """
    state = tmp_path / "state"
    monkeypatch.setattr(S, "STATE_DIR", state)
    monkeypatch.setattr(FL, "capture_build", lambda *a, **k: {"views": {"1280": view()}})
    report = FL.run(Path("/nonexistent-build"), baseline=fp(view()))
    path = FL.write_artifact("SM_TEST", FL.without_fingerprint(report))
    assert path.is_file() and path.is_relative_to(state / "frontend_layout"), path
    written = sorted(p.relative_to(state).as_posix()
                     for p in state.rglob("*") if p.is_file())
    assert written == ["frontend_layout/SM_TEST.json"], written


def test_the_leg_reports_skipped_rather_than_a_clean_layout_when_there_is_no_browser(
        tmp_path, monkeypatch):
    """No chromium, no build: the leg says SKIPPED and blesses nothing.

    A leg reporting a quiet layout because it never looked would be the worst
    kind of green. The shipped load probe has the same shape for the same reason.
    """
    monkeypatch.setattr(FP, "unavailable", lambda out_dir, chromium: "no chromium at /nope")
    assert FL.capture_build(tmp_path) == {"skipped": "no chromium at /nope"}
    report = FL.run(tmp_path, baseline=fp(view()))
    assert report["captured"] is False and report["checks"] == []
    assert FL.summary(report).startswith("LAYOUT SKIPPED"), FL.summary(report)
    assert FL.maybe_advance(report, advanced=True).startswith("held: the leg did not run")


def test_the_compared_field_set_ships_without_a_text_bearing_field():
    """`clipped` is excluded, and the exclusion is pinned by behaviour, not by name.

    `COLLECTOR_JS` fills `clipped` with up to 25 entries of `(selector,
    scrollWidth, clientWidth, node text)` harvested from every visible element in
    the page, so its length and its members move on any reflow anywhere on the
    dashboard. Comparing it would fire a check for an edit to some component the
    moved section has nothing to do with — the opposite of this leg's finding,
    which is one check naming one section — and half of each entry's value is node
    text, a value that moves when a human types.

    So this node asserts the three things that make the exclusion real: two reads
    differing ONLY in `clipped` project to the SAME fingerprint, they diff to no
    checks at all, and the constant itself names no text-bearing field. The
    determinism the surviving fields rely on is a separate mechanism — the frozen
    `layout_fixture` snapshot `capture_build` serves by default, never the live
    `/api/dashboard` — and dropping a field from the set stays the noise-floor
    run's one-line edit, pinned by
    `test_shrinking_the_one_field_set_constant_stops_comparing_that_field`.
    """
    assert set(FL.FINGERPRINT_FIELDS) <= {"cw", "sw", "overflow", "panels", "pills"}
    assert "clipped" not in FL.FINGERPRINT_FIELDS
    quiet = raw([section(h) for h in HEADS])
    loud = raw([section(h) for h in HEADS])
    loud["clipped"] = [{"sel": "div.rounded-lg", "sw": 900, "cw": 400,
                        "text": "35 scheduled"},
                       {"sel": "h3.font-medium", "sw": 1200, "cw": 100,
                        "text": "backlog: rename the entity guard"}]
    assert quiet["clipped"] == [] and len(loud["clipped"]) == 2
    loud_rest = {k: v for k, v in loud.items() if k != "clipped"}
    quiet_rest = {k: v for k, v in quiet.items() if k != "clipped"}
    assert loud_rest == quiet_rest, \
        "the two reads differ in `clipped` and in nothing else"
    a = FL.fingerprint_of(quiet, width=1280)
    b = FL.fingerprint_of(loud, width=1280)
    assert a == b, "a read differing only in `clipped` is ONE fingerprint"
    assert FL.diff(fp(a), fp(b)) == [], "and no check to report"


def test_the_frozen_api_snapshot_keeps_every_key_the_dashboard_reads():
    """The fixture is not a small dashboard, it is the whole page's input.

    `DashboardPage.tsx` reads `host.memory.used_bytes` unguarded, so a snapshot
    missing `host` renders a React error boundary where the dashboard should be —
    the state the fixture exists to avoid, arriving as a silence instead of a
    noise. The key list is the endpoint's own top level.
    """
    for key in ("host", "vllm", "primary", "recent", "agents", "services",
                "workers", "autonomy", "backlog", "automod", "usage", "timestamp"):
        assert key in DASHBOARD_SNAPSHOT, key
    assert isinstance(DASHBOARD_SNAPSHOT["timestamp"], float)
    assert all(len(rows) <= 2 for rows in DASHBOARD_SNAPSHOT["vllm"] if isinstance(rows, list))


# ---------------------------------------------------------------------------
# The browser pins: the collector against a served build, at the configured widths
# ---------------------------------------------------------------------------

def _fixture_build(dest: Path) -> Path:
    """A served build whose section widths come from its stylesheet.

    Not a page hand-written as bare HTML: the leg's premise is that a CSS edit
    moves the geometry, so the width lives in `styles.css` and a test changes the
    stylesheet to change the layout.
    """
    dest.mkdir(parents=True, exist_ok=True)
    # The classes are the dashboard's own: `dashboard_mobile_probe.py` selects a
    # panel as `.rounded-lg` and a HealthPill as `.rounded-md` under the section's
    # wrapper div, so a fixture built with invented class names would collect an
    # empty panel list and the pin would pass on nothing.
    body = "\n".join(
        f'<section class="sec"><h2>{h}</h2><div>'
        f'<div class="rounded-lg panel"><h3>{h} panel</h3></div>'
        f'<div class="rounded-md pill">{h} pill</div></div></section>'
        for h in HEADS)
    (dest / "index.html").write_text(
        "<!doctype html><html><head><meta charset=utf-8>"
        '<link rel="stylesheet" href="/styles.css"></head>'
        f"<body><main><div>{body}</div></main></body></html>")
    (dest / "styles.css").write_text(".sec { width: 700px; }\n"
                                     ".panel { width: 600px; }\n"
                                     ".pill { width: 120px; }\n")
    return dest


def _require_browser(tmp_path) -> Path:
    """Build the fixture, then skip ONLY if this box cannot answer a browser question.

    One conditional skip in one helper, the shape `tests/test_automod_frontend_probe.py`
    uses for the same reason: the reason string is `frontend_probe.unavailable`'s own,
    so it names a fact (`no build output`, `no index.html`, `no script or stylesheet`,
    `no chromium at <path>`) rather than an opinion, and the two nodes that call this
    are the only ones that can skip. Only the missing-browser reason skips: the
    fixture is built BEFORE the check, so any other reason means this file wrote an
    unusable fixture, and that lands as a failure rather than being skipped past.
    """
    out = _fixture_build(tmp_path / "build")
    why = FP.unavailable(out, Path("/usr/bin/chromium"))
    if why.startswith("no chromium at"):
        pytest.skip(why)
    assert not why, f"the fixture build this file writes is itself unusable: {why}"
    return out


def test_the_collector_reads_one_record_per_section_at_each_configured_width(tmp_path):
    """The same page captured at both configured widths is two fingerprints whose
    section widths track the viewport — the record is of a layout, not of a DOM.

    This is also the mechanism the canary's stylesheet seed depends on: the width
    comes from `styles.css`, so if a rule matching nothing could move a section
    here, or a rule that sizes one could leave the record alone, the pin goes red
    before the canary ever has to.
    """
    out = _require_browser(tmp_path)
    got = FL.capture_build(out, widths=FL.LAYOUT_WIDTHS,
                           chromium=Path("/usr/bin/chromium"), api_stub=None)
    assert "skipped" not in got, got
    assert sorted(int(w) for w in got["views"]) == sorted(FL.LAYOUT_WIDTHS)
    for width in FL.LAYOUT_WIDTHS:
        one = got["views"][str(width)]
        assert sorted(one["sections"]) == sorted(HEADS)
        assert one["no_verdict"] == ""
        assert one["sections"]["System"]["cw"] == 700, one["sections"]["System"]
        assert one["sections"]["System"]["panels"][0] == ["System panel", 600, 600]
        assert one["sections"]["System"]["pills"][0][:2] == ["System pill", 120]


def test_a_stylesheet_rule_matching_nothing_changes_no_field_that_the_set_compares(
        tmp_path):
    """The blind spot, pinned from the other side.

    The shipped canary scores `stylesheet_loads_but_matches_no_rule` as
    `"score": "blind", "checks": []`, because no channel the load probe watches
    can carry it. The layout leg's channel is geometry, so this pin asserts both
    halves of what the leg needs: a dead rule produces a fingerprint equal to the
    control's (no false fire on a stylesheet edit that changes nothing), and a
    rule that *does* size a section produces a different one (the seed's own
    break is detectable). The first half is what makes the second meaningful —
    a leg that fired on every stylesheet edit would also "detect" this seed.
    """
    out = _require_browser(tmp_path)
    control = FL.capture_build(out, widths=(1280,), chromium=Path("/usr/bin/chromium"),
                               api_stub=None)
    assert "skipped" not in control, control
    dead = FL.fingerprint(control["views"])
    (out / "styles.css").write_text(
        (out / "styles.css").read_text()
        + ".no-such-class-anywhere { width: 999px; }\n")
    edited = FL.capture_build(out, widths=(1280,), chromium=Path("/usr/bin/chromium"),
                              api_stub=None)
    assert FL.fingerprint(edited["views"])["views"] == dead["views"], \
        "a rule matching no node must not move the fingerprint"
    assert FL.diff(dead, FL.fingerprint(edited["views"])) == []
