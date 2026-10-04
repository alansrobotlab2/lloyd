"""knowledge-health-report.py — the Stale Facts section (#841).

The detector used to age a fact from `created_at` alone and skip anything that
carried `valid_at`. On the current store `created_at` exists on ~34% of active
facts and the oldest usable one is younger than the 60-day threshold, so
`find_stale_facts` returned 0 on every run and the report printed the clean
verdict while 140,179 of 315,784 active facts carried no date at all. Every
node below pins one of those three failure modes: the missing second date
source, the `valid_at` amnesty, and the clean verdict over an undatable store.

A fourth mode arrived with the rebuild that followed (#2179, the nodes under the
stamp-cohort heading): `created_at` present on ~100% of records and meaning the
re-derivation date rather than the recording date. That is the opposite input fault —
a date on everything, on no fact informative — and it makes the store read fresh for
as long as the stamp is younger than the threshold, then stale in one day. The cohort
nodes pin the guard that reads such a date as undated, and they hold #841's own aging
path intact on the way, because a fact that carries an `event_date` still ages from it.
"""
import importlib.util
import inspect
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("khr", ROOT / "scripts/memory/knowledge-health-report.py")
khr = importlib.util.module_from_spec(_spec); sys.modules["khr"] = khr; _spec.loader.exec_module(khr)

# A fixed clock: every age asserted below is an exact integer number of days.
NOW = datetime(2026, 9, 19, 12, 0, 0, tzinfo=timezone.utc)
THRESH = khr.STALE_DAYS_THRESHOLD  # 60


def _iso(days_ago: int) -> str:
    return (NOW - timedelta(days=days_ago)).isoformat()


def _write(root: Path, name: str, cat: str, facts: list[dict]) -> Path:
    """One fact file at <root>/<name>/<name>-<cat>.md holding `facts` verbatim."""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    fm = {"type": "facts", "entity": name, "category": cat,
          "facts": [dict({"entity": name, "confidence": 0.9, "category": cat}, **f) for f in facts]}
    p = d / f"{name}-{cat}.md"
    p.write_text(f"---\n{yaml.dump(fm, sort_keys=False)}---\n\n# {name} - {cat}\n", encoding="utf-8")
    return p


def _report(entities: dict, now: datetime = NOW) -> str:
    return khr.generate_report(
        khr.compute_entity_stats(entities),
        khr.compute_relationship_stats([], entities),
        [],
        khr.find_stale_facts(entities, now, THRESH),
        now,
        stale_unevaluable=khr.stale_coverage(entities),
    )


def _stale_section(report: str) -> str:
    start = report.index("## Stale Facts")
    rest = report[start + len("## Stale Facts"):]
    end = rest.find("\n## ")
    return rest if end == -1 else rest[:end]


# The line the contract fixes: `STALE_UNEVALUABLE: n of m facts carry no usable
# date (x%)`. n and m render with thousands separators, so they may hold commas.
UNEVALUABLE_RE = re.compile(
    r"STALE_UNEVALUABLE: ([\d,]+) of ([\d,]+) facts carry no usable date \((\d+(?:\.\d+)?)%\)")


# ── clause 1: event_date is a usable age source ───────────────────────────────

def test_event_date_only_fact_is_stale(tmp_path):
    """A fact with no `created_at` at all used to be skipped outright."""
    root = tmp_path / "facts"
    _write(root, "Asimov", "history", [
        {"fact": "published a novel", "event_date": _iso(200)},
    ])
    entities = khr.load_entities(root)

    stale = khr.find_stale_facts(entities, NOW, THRESH)

    assert [(s["entity"], s["category"], s["age_days"]) for s in stale] == [
        ("Asimov", "history", 200)]
    assert stale[0]["preview"] == "published a novel"
    # and it is counted as evaluable, not as an unevaluable fact
    assert khr.stale_coverage(entities) == (0, 1)


def test_undatable_and_expired_facts_are_not_reported_stale(tmp_path):
    """A missing date is 'unmeasured', never stale; a retired fact is neither."""
    root = tmp_path / "facts"
    _write(root, "Nodate", "state", [{"fact": "no date anywhere"}])
    _write(root, "Retired", "state", [
        {"fact": "expired long ago", "event_date": _iso(400), "expired_at": _iso(1)},
    ])
    entities = khr.load_entities(root)

    assert khr.find_stale_facts(entities, NOW, THRESH) == []
    # only the live fact counts toward coverage; the expired one is out of the
    # denominator entirely, so m is 1 and n is 1
    assert khr.stale_coverage(entities) == (1, 1)


# ── clause 2: the older of created_at and event_date decides the age ──────────

def test_both_dates_present_ages_from_the_older_event_date(tmp_path):
    """Recorded last week, about an event 200 days old: reported at ~200 days."""
    root = tmp_path / "facts"
    _write(root, "Conference", "state", [
        {"fact": "keynote given long ago, filed recently",
         "created_at": _iso(7), "event_date": _iso(200)},
    ])
    entities = khr.load_entities(root)

    stale = khr.find_stale_facts(entities, NOW, THRESH)

    assert [(s["entity"], s["age_days"]) for s in stale] == [("Conference", 200)]


def test_both_dates_present_ages_from_the_older_created_at(tmp_path):
    """The same rule mirrored: the older date wins whichever field holds it."""
    root = tmp_path / "facts"
    _write(root, "Rebuilt", "state", [
        {"fact": "filed long ago about a recent event",
         "created_at": _iso(200), "event_date": _iso(7)},
    ])
    entities = khr.load_entities(root)

    stale = khr.find_stale_facts(entities, NOW, THRESH)

    assert [(s["entity"], s["age_days"]) for s in stale] == [("Rebuilt", 200)]


def test_recent_fact_with_no_dates_is_not_stale_but_is_unevaluable(tmp_path):
    root = tmp_path / "facts"
    _write(root, "Fresh", "state", [{"fact": "recent event", "event_date": _iso(5)}])
    _write(root, "Blank", "state", [{"fact": "undatable"}])
    entities = khr.load_entities(root)

    assert khr.find_stale_facts(entities, NOW, THRESH) == []
    assert khr.stale_coverage(entities) == (1, 2)


# ── clause 3: valid_at re-bases the age instead of exempting the fact ─────────

def test_old_valid_at_is_still_reported(tmp_path):
    """A `valid_at` older than the threshold no longer buys a permanent amnesty."""
    root = tmp_path / "facts"
    _write(root, "Idcol", "state", [
        {"fact": "recorded 250 days ago, valid since 200 days ago",
         "created_at": _iso(250), "valid_at": _iso(200)},
    ])
    entities = khr.load_entities(root)

    stale = khr.find_stale_facts(entities, NOW, THRESH)

    # the age is re-based onto valid_at, so 200 days rather than 250
    assert [(s["entity"], s["age_days"]) for s in stale] == [("Idcol", 200)]


def test_recent_valid_at_re_bases_the_age_away_from_stale(tmp_path):
    root = tmp_path / "facts"
    _write(root, "Reaffirmed", "state", [
        {"fact": "recorded 250 days ago, re-affirmed 10 days ago",
         "created_at": _iso(250), "valid_at": _iso(10)},
    ])
    entities = khr.load_entities(root)

    assert khr.find_stale_facts(entities, NOW, THRESH) == []
    assert khr.stale_coverage(entities) == (0, 1)


def test_valid_at_cannot_age_a_fresh_fact_backward(tmp_path):
    """`valid_at` moves the clock forward only: a 7-day-old record stays fresh
    even when it asserts something that became valid 250 days ago."""
    root = tmp_path / "facts"
    _write(root, "LateEntry", "state", [
        {"fact": "filed 7 days ago about a 250-day-old validity",
         "created_at": _iso(7), "valid_at": _iso(250)},
    ])
    entities = khr.load_entities(root)

    assert khr.find_stale_facts(entities, NOW, THRESH) == []


# ── clause 4: the unevaluable line, printed whenever n > 0 ────────────────────

def test_unevaluable_line_names_n_and_m_over_active_facts(tmp_path):
    """2 undatable facts of 3 active ones: the count and the share are printed."""
    root = tmp_path / "facts"
    _write(root, "Blind", "state", [
        {"fact": "no date"},
        {"fact": "also no date", "valid_at": _iso(3)},
        {"fact": "dated and fresh", "event_date": _iso(3)},
    ])
    entities = khr.load_entities(root)

    report = _report(entities)
    section = _stale_section(report)
    match = UNEVALUABLE_RE.search(section)

    assert match is not None, section
    assert match.groups() == ("2", "3", "66.7")


def test_unevaluable_line_has_no_floor_below_which_it_is_silent(tmp_path):
    """One undatable fact in ten thousand still prints: n > 0 is the only gate."""
    root = tmp_path / "facts"
    facts = [{"fact": f"dated fact {i}", "event_date": _iso(3)} for i in range(9999)]
    facts.append({"fact": "the one undatable fact"})
    _write(root, "MostlyDated", "state", facts)
    entities = khr.load_entities(root)

    assert khr.stale_coverage(entities) == (1, 10000)
    section = _stale_section(_report(entities))
    match = UNEVALUABLE_RE.search(section)

    assert match is not None, section
    assert match.groups() == ("1", "10,000", "0.0")


# ── clause 5: the clean verdict needs zero stale AND zero undatable ───────────

def test_undatable_fixture_cannot_print_the_clean_verdict(tmp_path):
    """The #841 symptom: a whole store with no dates read as 'nothing to triage'."""
    root = tmp_path / "facts"
    _write(root, "Blind", "state", [{"fact": "no date"}, {"fact": "also no date"}])
    entities = khr.load_entities(root)

    assert khr.find_stale_facts(entities, NOW, THRESH) == []
    report = _report(entities)
    section = _stale_section(report)

    assert "No stale facts found" not in report
    assert UNEVALUABLE_RE.search(section) is not None


def test_clean_verdict_survives_when_every_fact_is_dated_and_fresh(tmp_path):
    root = tmp_path / "facts"
    _write(root, "Fresh", "state", [{"fact": "recent", "event_date": _iso(3)}])
    entities = khr.load_entities(root)

    report = _report(entities)

    assert khr.find_stale_facts(entities, NOW, THRESH) == []
    assert khr.stale_coverage(entities) == (0, 1)
    assert "*No stale facts found.*" in report
    assert "STALE_UNEVALUABLE" not in report


def test_missing_coverage_measurement_is_not_read_as_clean(tmp_path):
    """A caller that never measured coverage gets no clean verdict either — the
    same defect class as a guard reporting a verdict on an input it did not read."""
    root = tmp_path / "facts"
    _write(root, "Fresh", "state", [{"fact": "recent", "event_date": _iso(3)}])
    entities = khr.load_entities(root)
    report = khr.generate_report(khr.compute_entity_stats(entities),
                                 khr.compute_relationship_stats([], entities),
                                 [], [], NOW)

    assert "No stale facts found" not in report
    assert "STALE_UNEVALUABLE: coverage not measured" in report


# ── clause 6: none of this reaches the alarm channel ──────────────────────────

def _clean_hygiene() -> dict:
    return {"contaminated": [], "contaminated_dirs": 0, "foreign_facts": 0,
            "near_dup_clusters": 0, "near_dup_dirs": 0, "near_dup_tiers": {},
            "regrown": [], "new_dirs": 0, "regrowth_days": 7, "provenance": {}}


def test_alarm_inputs_do_not_include_staleness(tmp_path):
    """`_alarms` gains no stale/coverage parameter, so a non-zero stale count or
    an unevaluable line cannot reach the alarm channel or the exit code."""
    assert tuple(inspect.signature(khr._alarms).parameters) == (
        "store_stats", "hygiene", "duplicate_id_files", "baseline")
    assert (khr.EXIT_OK, khr.EXIT_ALARM) == (0, 2)

    alarms = khr._alarms({"edges_active": 53416}, _clean_hygiene(), 0, 0)
    assert alarms == []


def test_stale_table_stays_capped_while_the_count_is_reported(tmp_path):
    """52k stale facts must not turn the report into a 52k-row table."""
    root = tmp_path / "facts"
    facts = [{"fact": f"old claim {i}", "event_date": _iso(100 + i)}
             for i in range(THRESH + 10)]
    facts.append({"fact": "undatable one"})
    _write(root, "Pile", "state", facts)
    entities = khr.load_entities(root)

    report = _report(entities)
    section = _stale_section(report)
    rows = [ln for ln in section.splitlines()
            if ln.startswith("| ") and not ln.startswith("|---")
            and "Entity | Category" not in ln and not ln.startswith("| …")]

    assert len(rows) == khr.SECTION_ROW_CAP
    assert f"*{THRESH + 10 - khr.SECTION_ROW_CAP:,} more*" in section
    assert UNEVALUABLE_RE.search(section) is not None


# ── #1543: the section shows the shape of the backlog, and lists reviewable rows ─
#
# Measured live the day these were written, over `app.paths.VAULT_FACTS_ROOT` with the
# script's own `find_stale_facts`: 15,948 stale facts, p10 82 days, **median 148**, p90
# 576, max 49,942; 88.1% inside a year and 114 (0.71%) older than ten years. Sorting by
# age descending therefore spent all 50 rows on that 0.71% — the top row was `Dual
# Process Theory lineage includes William James's distinction`, 49,942 days, a historical
# fact with a real `event_date` that no review would invalidate — while the 14,053-fact
# band just past the threshold, which is the actual backlog, was listed nowhere. A
# reader of the section came away believing the store was a pile of century-old junk.

#: The band labels the section must print: one closed band per ladder bound, then one
#: open above the largest. Derived from the module's own constant so a boundary edit
#: moves this test rather than silently disagreeing with it; the lowest band opens one
#: day AT the staleness threshold, because `find_stale_facts` selects with
#: `age >= threshold_days` and so calls an exactly-threshold-old fact stale.
BAND_LABELS = [f"{lo:,}-{hi:,}" for lo, hi in zip(
    [THRESH, *[b + 1 for b in khr.AGE_BAND_BOUNDS_DAYS[:-1]]],
    khr.AGE_BAND_BOUNDS_DAYS)] + [f">{khr.AGE_BAND_BOUNDS_DAYS[-1]:,}"]


def test_the_printed_bands_cover_the_total_without_gaps_or_overlap(tmp_path):
    """Every stale fact lands in exactly one band, so the shares sum to the whole.

    Two holes this pins, both found by running the ladder against live data rather than
    against itself. (a) One band per bound with the last open above its own edge leaves a
    gap between the second-largest bound and the largest: a 1,000-day fact sits in no
    band. (b) The floor: `find_stale_facts` selects with `age >= threshold_days`, so an
    exactly-60-day fact is stale and counted in the total, and a floor of `THRESH + 1`
    dropped the 78 such facts the live store has out of the line beside it. Either hole
    prints shares summing to less than 100% under a total that counts every fact — the
    same misreading of the backlog this item is about, one line higher up.
    """
    assert len(BAND_LABELS) == len(khr.AGE_BAND_BOUNDS_DAYS) + 1, BAND_LABELS
    ages = [THRESH, 61, 90, 91, 365, 366, 730, 731, 1000, 3650, 3651]
    section = _one_entity_section(tmp_path, ages)
    line = _band_line(section)
    for label in BAND_LABELS:
        assert label in line, f"{label} missing from {line}"
    # Three facts in the lowest band: the threshold itself, one day past it, and the
    # first bound — the two edges a floor of `THRESH + 1` would lose are the first two.
    assert f"{THRESH}-90: 3 (27.3%)" in line, line
    assert "731-3,650: 3 (27.3%)" in line, line
    assert ">3,650: 1 (9.1%)" in line, line
    counts = [int(entry.split(": ", 1)[1].split(" ")[0].replace(",", ""))
              for entry in line.split(": ", 1)[1].split(" | ")]
    assert sum(counts) == len(ages), (
        f"the bands sum to {sum(counts)} of the {len(ages)} stale facts: a fact in no "
        "band is invisible in the line while still counted in the total")
    shares = [float(entry.split("(")[1].rstrip(")%")) for entry in
              line.split(": ", 1)[1].split(" | ")]
    assert abs(sum(shares) - 100.0) <= 0.3, f"shares sum to {sum(shares)}%: {line}"


def _one_entity_section(tmp_path, ages: list[int], name: str = "Pile") -> str:
    """The Stale Facts section for exactly one entity holding one stale fact per `age`."""
    root = tmp_path / "facts"
    _write(root, name, "state",
           [{"fact": f"claim at {a} days", "event_date": _iso(a)} for a in ages])
    return _stale_section(_report(khr.load_entities(root)))


def _listed_ages(section: str) -> list[int]:
    """The ages of the listed rows, in the order printed."""
    rows = [ln for ln in section.splitlines()
            if ln.startswith("| ") and not ln.startswith("|---")
            and "Entity | Category" not in ln and not ln.startswith("| …")]
    return [int(ln.rsplit("|", 2)[-2].strip()) for ln in rows]


def _band_line(section: str) -> str:
    lines = [ln for ln in section.splitlines() if ln.startswith("Age bands (days):")]
    assert len(lines) == 1, f"expected exactly one band line, got {len(lines)}"
    return lines[0]


def test_the_reviewable_band_is_one_of_the_printed_band_edges():
    """The two constants have to stay coupled, and nothing else in the file says so.

    The table selects on `REVIEWABLE_AGE_DAYS` while the printed distribution is cut by
    `AGE_BAND_BOUNDS_DAYS`. If those drift — 365 raised to 400, say — the counts in the
    band line would no longer add up to the reviewable/older split the listing sentence
    states, and the section would contradict itself one line apart.
    """
    assert khr.REVIEWABLE_AGE_DAYS in khr.AGE_BAND_BOUNDS_DAYS, (
        f"the listing cuts at {khr.REVIEWABLE_AGE_DAYS} days but the printed bands are "
        f"cut at {khr.AGE_BAND_BOUNDS_DAYS}, so no band edge matches the split the "
        "section describes")


def test_the_section_prints_one_band_line_of_counts_and_shares(tmp_path):
    """Clause 1: one line, one entry per band, each with its count and its percentage.

    Six facts, one per band, so every share is exactly 16.7% and a fact bucketed into the
    wrong band moves two numbers at once. The counts also have to add up to the total: a
    band line that silently lost the over-3,650 tail would put clause 4's "remain visible
    as counts" back into the dark.
    """
    section = _one_entity_section(tmp_path, [70, 120, 300, 500, 1000, 5000])

    line = _band_line(section)
    for label in BAND_LABELS:
        assert f"{label}: 1 (16.7%)" in line, f"{label} missing or miscounted: {line}"

    counts = [int(entry.split(": ", 1)[1].split(" ")[0].replace(",", ""))
              for entry in line.split(": ", 1)[1].split(" | ")]
    assert sum(counts) == 6, f"the band counts sum to {sum(counts)}, not the 6 reported"


def test_the_median_age_prints_beside_the_total(tmp_path):
    """Clause 2: the median is on the total's own line, not somewhere in the section.

    15,948 is a number a reader cannot act on; 15,948 with a median of 100 days says the
    backlog is one dense band just past the threshold. Three facts, so the median is an
    actual member of the fixture and a whole-day print: a `f"{x}"` on a float median
    would show `100.0` and a mean would show 166.
    """
    section = _one_entity_section(tmp_path, [61, 100, 300])

    total_line = next(ln for ln in section.splitlines() if "in total" in ln)
    assert "**3** in total" in total_line, total_line
    assert "median **100 days**" in total_line, (
        f"the median is not on the total's line: {total_line!r}")


def test_the_listed_rows_are_the_oldest_within_the_reviewable_band(tmp_path):
    """Clause 3: the rows are the oldest *reviewable* facts, not the century tail.

    Two historical facts stand above three reviewable ones, mirroring the live shape
    that motivated the item. The over-band ages must not appear as rows — and must still
    appear as counts, which is clause 4's first half read from the other side.
    """
    over, reviewable = [5000, 49942], [300, 200, 100]
    section = _one_entity_section(tmp_path, over + reviewable)

    assert _listed_ages(section) == sorted(reviewable, reverse=True)
    assert not any(f"| {a} |" in section for a in over), (
        "a fact older than the reviewable band is listed as a row")
    assert f"{khr.REVIEWABLE_AGE_DAYS:,} days" in section
    assert "3 of 5 are reviewable" in section.replace("**", ""), (
        "the section does not say how much of the total the listing covers")
    assert ">3,650: 2 (40.0%)" in _band_line(section), (
        "the over-band facts vanished instead of being counted")


def test_the_reviewable_listing_keeps_the_cap_the_ellipsis_and_the_total(tmp_path):
    """Clause 3, second half: capping the *reviewable* rows must not shrink the count.

    70 reviewable facts (100-169 days, the fixture the existing cap test uses) plus 40
    over-band (400-439). The listing is capped at `SECTION_ROW_CAP`, but the total and
    the `*N more*` ellipsis are over all 110 stale facts: an ellipsis that started
    counting only the selected band would report 20 more when 60 exist, which is the same
    misreading of the backlog size this item is about, one line further down.
    """
    root = tmp_path / "facts"
    _write(root, "Pile", "state",
           [{"fact": f"reviewable {i}", "event_date": _iso(100 + i)} for i in range(70)]
           + [{"fact": f"historical {i}", "event_date": _iso(400 + i)} for i in range(40)])
    section = _stale_section(_report(khr.load_entities(root)))

    assert len(_listed_ages(section)) == khr.SECTION_ROW_CAP
    assert _listed_ages(section) == sorted(range(100, 170), reverse=True)[:khr.SECTION_ROW_CAP]
    assert "**110** in total" in section, "the total no longer counts all stale facts"
    assert "*60 more*" in section, "the ellipsis counted only the reviewable band"


def test_the_section_names_the_gap_when_nothing_is_reviewable(tmp_path):
    """Clause 4, second half: no reviewable fact is said, not left as an empty table.

    Three facts all older than the band. An empty table under a non-zero count reads as
    "nothing stale" one line after the section said there are three, so the oldest rows
    print anyway with the reason said out loud — the listing degrades, the visibility
    does not.
    """
    section = _one_entity_section(tmp_path, [5000, 4200, 3800])

    assert "none of the 3" in section
    assert f"{khr.REVIEWABLE_AGE_DAYS:,}-day reviewable band" in section
    assert "historical" in section, "the section does not say why the rows are not candidates"
    assert _listed_ages(section) == [5000, 4200, 3800], (
        "an empty table where the item requires the oldest rows")
    assert "*more*" not in section, "three rows are under the cap, so no ellipsis is owed"
    assert ">3,650: 3 (100.0%)" in _band_line(section)


class _FakeEdges:
    def __init__(self, edges):
        self._edges = edges

    def all(self):
        return self._edges


class _FakeStore:
    """Just enough of `app.kg_store.store` for `main()`: edges and stats."""

    def __init__(self, edges, stats):
        self.edges = _FakeEdges(edges)
        self._stats = stats

    def stats(self):
        return self._stats


def test_main_prints_the_line_and_exits_zero(tmp_path, monkeypatch):
    """End-to-end: a run with both stale facts and undatable facts writes both
    lines, exits EXIT_OK, and posts no alarm — the alarm channel is unchanged."""
    root = tmp_path / "facts"
    out = tmp_path / "out"
    _write(root, "Pile", "state", [
        {"fact": "old claim", "event_date": _iso(200)},
        {"fact": "undatable claim"},
    ])
    # HOME is where the graph baseline lives; pointing it at tmp_path keeps the
    # baseline out of the verdict so the exit code under test is only about
    # staleness.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(khr, "_kg_store",
                        lambda: _FakeStore([{"source": "Pile", "target": "Other", "type": "uses"}],
                                           {"edges_active": 53416, "edges_total": 53416}))
    monkeypatch.setattr(khr, "fact_duplicate_stats",
                        lambda: {"unavailable": True, "reason": "store faked in test"})
    alerts: list = []
    monkeypatch.setattr(khr, "_alert", lambda alarms, path: alerts.append(alarms))
    monkeypatch.setattr(sys, "argv", ["knowledge-health-report.py",
                                      "--facts-dir", str(root),
                                      "--output-dir", str(out)])

    rc = khr.main()

    assert rc == khr.EXIT_OK
    assert alerts == []
    reports = list(out.glob("knowledge-health-*.md"))
    assert len(reports) == 1
    text = reports[0].read_text()
    assert UNEVALUABLE_RE.search(_stale_section(text)) is not None
    assert "old claim" in text
    assert "No stale facts found" not in text


# ── #2179: the stamp cohort — a created_at the bulk of the store shares ───────
#
# Measured live 2026-10-04 over the 130,115 active fact records: 99,707 (76.6%) carry
# `created_at: 2026-09-23` and the whole `created_at` field spans 11 days, because
# `clean_facts_directory()` (`scripts/memory/next-gen-memory/nightly_extraction.py:185`)
# wipes the entity tree and `fact_extractor.py:551` restamps every re-derived row with
# the run's date. 80,673 of those records carry no `event_date` at all, so the stamp is
# their only date and every one of them crossed the 60-day threshold on 2026-11-22
# together. The shipped rule aged them as fresh until that day and counted 2 facts
# unevaluable (`stale_coverage` returned `(2, 130115)`). These nodes pin the guard that
# reads such a date as provenance loss instead of freshness.

#: The guard's absolute floor, read from the shipped constant so a fixture is exactly at
#: the threshold the code names and one record under it is one `range()` away, not a copy
#: of a magic number.
COHORT = khr.STALE_DOMINANT_DATE_FLOOR

#: A `created_at` far enough back to be stale on the fixed clock. Every cohort fixture
#: uses it so that a guard which failed to arm is caught: the same facts, unguarded, are
#: unambiguously stale, which each node asserts before it asserts the guard held.
COHORT_AGE = 200

#: The cohort line, in the section's own words: its count, its percent of the active
#: records, the stamp date, and the ISO day the whole cohort crosses the threshold.
STAMP_RE = re.compile(
    r"STALE_STAMP_COHORT: ([\d,]+) facts \((\d+(?:\.\d+)?)% of ([\d,]+) active\) "
    r"carry no date but `created_at: (\d{4}-\d{2}-\d{2})`, a calendar date "
    r"([\d,]+) of them share.*?threshold on (\d{4}-\d{2}-\d{2})",
    re.S)


def _stamp_report(entities: dict, now: datetime = NOW) -> str:
    """`_report` plus the cohort measurement, which is what `main()` now passes."""
    return khr.generate_report(
        khr.compute_entity_stats(entities),
        khr.compute_relationship_stats([], entities),
        [],
        khr.find_stale_facts(entities, now, THRESH),
        now,
        stale_unevaluable=khr.stale_coverage(entities),
        stamp_cohort=khr.stale_stamp_cohort(entities),
    )


def test_a_dominant_created_at_cohort_is_undated_rather_than_stale(tmp_path):
    """Clause 1: an active fact with no `event_date` whose `created_at` date is carried
    by more than 25% of the store AND by at least `COHORT` records is not returned by
    `find_stale_facts` once older than the threshold, and `stale_coverage` counts it as
    unevaluable. The fixture plants exactly `COHORT` same-day records so the floor arms.
    """
    root = tmp_path / "facts"
    _write(root, "Rebuilt", "state",
           [{"fact": f"stamp claim {i}", "created_at": _iso(COHORT_AGE)}
            for i in range(COHORT)])
    _write(root, "Fresh", "state", [{"fact": "dated elsewhere", "event_date": _iso(5)}])
    # One record on the stamp day that owns dates of its own and was re-affirmed 3 days
    # ago: in the stamp date's record count, out of the cohort, and fresh under the
    # #841 rule. It is what keeps `date_records` and `cohort` from being the same number
    # in the dict below, so an implementation that reports one as the other is caught.
    _write(root, "Reaffirmed", "state",
           [{"fact": "restamped, but re-affirmed this week",
             "created_at": _iso(COHORT_AGE), "event_date": _iso(5), "valid_at": _iso(3)}])
    entities = khr.load_entities(root)

    # Not vacuous: on the shipped age rule every one of those records IS past the
    # threshold, so an absent guard would return COHORT stale facts rather than zero.
    past_threshold = [f for _name, entries in entities.items() for entry in entries
                      for f in entry["facts"]
                      if khr.stale_age_reference(f)
                      and (NOW - khr.stale_age_reference(f)).days >= THRESH]
    assert len(past_threshold) == COHORT, "the fixture must be stale without the guard"

    assert khr.find_stale_facts(entities, NOW, THRESH) == []
    # The cohort alone: both controls keep their own dates and stay evaluable, so the
    # guard sweeps up nothing it has no reason to doubt.
    assert khr.stale_coverage(entities) == (COHORT, COHORT + 2)

    cohort = khr.stale_stamp_cohort(entities)
    stamp = (NOW - timedelta(days=COHORT_AGE)).date()
    assert cohort == {"date": stamp.isoformat(), "date_records": COHORT + 1,
                      "active_total": COHORT + 2, "cohort": COHORT,
                      "flip_date": (stamp + timedelta(days=THRESH)).isoformat()}


def test_an_event_date_still_ages_a_fact_recorded_on_the_stamp_day(tmp_path):
    """Clause 2: the #841 aging path is not weakened. A fact carrying an `event_date`
    older than the threshold stays stale even when its `created_at` sits on the dominant
    date — excluding it would retire the very detection #841 shipped, and the 20,8xx
    stale total the live report prints is entirely `event_date`-driven (measured 0 of
    20,825 carry a null `event_date`).
    """
    root = tmp_path / "facts"
    _write(root, "Rebuilt", "state",
           [{"fact": f"stamp claim {i}", "created_at": _iso(COHORT_AGE)}
            for i in range(COHORT)])
    _write(root, "OldEvent", "state",
           [{"fact": "restamped on the rebuild, but about a 400-day-old event",
             "created_at": _iso(COHORT_AGE), "event_date": _iso(400)}])
    entities = khr.load_entities(root)

    cohort = khr.stale_stamp_cohort(entities)
    # The OldEvent row shares the stamp date, so it is in the date's record count and NOT
    # in the cohort: it owns a date of its own, which is the whole distinction.
    assert cohort["date_records"] == COHORT + 1
    assert cohort["cohort"] == COHORT

    stale = khr.find_stale_facts(entities, NOW, THRESH)
    assert [(s["entity"], s["age_days"]) for s in stale] == [("OldEvent", 400)]
    assert khr.stale_coverage(entities) == (COHORT, COHORT + 1)


def test_a_cohort_one_record_under_the_floor_still_ages_at_the_threshold(tmp_path):
    """Clause 3: under the absolute floor the guard stays silent, so a same-day corpus
    ages at exactly `STALE_DAYS_THRESHOLD` as it does today — the boundary the live store
    crosses by 99x and a hand-written fixture does not.
    """
    root = tmp_path / "facts"
    _write(root, "Rebuilt", "state",
           [{"fact": f"same-day claim {i}", "created_at": _iso(THRESH)}
            for i in range(COHORT - 1)])
    _write(root, "Fresh", "state", [{"fact": "recent", "event_date": _iso(3)}])
    entities = khr.load_entities(root)

    # One record short of the floor on 1,000 active records: 99.9% share, so the share
    # half of the guard is satisfied and the floor alone is what keeps it quiet.
    assert khr.stale_stamp_cohort(entities) is None
    stale = khr.find_stale_facts(entities, NOW, THRESH)
    assert len(stale) == COHORT - 1
    assert sorted({s["age_days"] for s in stale}) == [THRESH], \
        "facts must go stale at exactly the threshold, not a day past it"
    assert khr.stale_coverage(entities) == (0, COHORT)


def test_a_cohort_at_exactly_the_share_still_ages(tmp_path):
    """Clause 3's other half: at exactly 25.0% the share test does not fire either — the
    guard is `> STALE_DOMINANT_DATE_SHARE`, and the live cohort is at 76.6%, four times
    the line, so nothing is lost by keeping the comparison strict.
    """
    root = tmp_path / "facts"
    _write(root, "Rebuilt", "state",
           [{"fact": f"same-day claim {i}", "created_at": _iso(COHORT_AGE)}
            for i in range(COHORT)])
    _write(root, "MostlyFresh", "state",
           [{"fact": f"fresh {i}", "event_date": _iso(3)} for i in range(3 * COHORT)])
    entities = khr.load_entities(root)

    # 1,000 of 4,000 active records carry the date: above the floor, at the share.
    assert khr.stale_stamp_cohort(entities) is None
    stale = khr.find_stale_facts(entities, NOW, THRESH)
    assert len(stale) == COHORT
    assert khr.stale_coverage(entities) == (0, 4 * COHORT)


def test_the_section_names_the_cohort_and_the_day_it_flips(tmp_path):
    """Clause 4: the Stale Facts section prints the cohort on its own line — stamp date,
    count, percent of active records, and the ISO day it crosses the threshold — while
    the `STALE_UNEVALUABLE` line keeps its format and its `n`, and the stale total moves
    only by the fact that carries a real `event_date`.
    """
    root = tmp_path / "facts"
    _write(root, "Rebuilt", "state",
           [{"fact": f"stamp claim {i}", "created_at": _iso(COHORT_AGE)}
            for i in range(COHORT)])
    _write(root, "Blind", "state", [{"fact": "no date at all"}])
    _write(root, "OldEvent", "state",
           [{"fact": "old event", "created_at": _iso(COHORT_AGE), "event_date": _iso(400)}])
    entities = khr.load_entities(root)

    section = _stale_section(_stamp_report(entities))
    match = STAMP_RE.search(section)
    assert match is not None, section
    stamp = (NOW - timedelta(days=COHORT_AGE)).date()
    assert match.groups() == ("1,000", "99.8", "1,002", stamp.isoformat(), "1,001",
                              (stamp + timedelta(days=THRESH)).isoformat())

    # `n` keeps counting the facts the check cannot age, and the two populations stay
    # separable: 1,001 = the 1,000-record cohort + the one record with no date at all.
    unevaluable = UNEVALUABLE_RE.search(section)
    assert unevaluable is not None, section
    assert unevaluable.groups() == ("1,001", "1,002", "99.9")
    assert int(unevaluable.group(1).replace(",", "")) - COHORT == 1

    # The stale total is the `event_date`-driven fact alone. Had the guard swept facts
    # with real event dates in, this is the number the owed check says must not move.
    assert "**1** in total" in section, section


def test_a_stamp_day_whose_records_all_own_an_event_date_yields_no_cohort(tmp_path):
    """The empty end of the cohort: 1,000 records share one `created_at`, so a date does
    dominate the store, but every one of them carries an `event_date` of its own — so
    there is nothing undated to name, no cohort line prints, and all 1,000 age stale from
    those dates. This is the shape of the 2,533 live records that owe ruling #2, pinned as
    today's behaviour: they keep aging off the stamp and are NOT swept into the cohort.
    """
    root = tmp_path / "facts"
    _write(root, "Restamped", "state",
           [{"fact": f"restamped old event {i}", "created_at": _iso(COHORT_AGE),
             "event_date": _iso(400)} for i in range(COHORT)])
    entities = khr.load_entities(root)

    cohort = khr.stale_stamp_cohort(entities)
    assert cohort["date_records"] == COHORT and cohort["cohort"] == 0
    stale = khr.find_stale_facts(entities, NOW, THRESH)
    assert len(stale) == COHORT
    assert sorted({s["age_days"] for s in stale}) == [400]
    assert khr.stale_coverage(entities) == (0, COHORT)

    section = _stale_section(_stamp_report(entities))
    assert STAMP_RE.search(section) is None, "an empty cohort has nothing to name"
    assert f"**{COHORT:,}** in total" in section, section


def test_a_measured_cohort_suppresses_the_clean_verdict_on_its_own(tmp_path):
    """Clause 5's second half: the suppression is carried by the cohort measurement
    itself, not only by the coverage pair that counts it. Here the section is handed the
    pair every pre-#2179 caller computed — `(0, 1,000)`, the cohort still inside the
    fresh population as far as that pair knows — together with the cohort, and the clean
    verdict is still refused. Without this the verdict would hinge on one caller
    remembering to count the cohort a second time, in another function.
    """
    root = tmp_path / "facts"
    _write(root, "Rebuilt", "state",
           [{"fact": f"stamp claim {i}", "created_at": _iso(COHORT_AGE)}
            for i in range(COHORT)])
    entities = khr.load_entities(root)

    report = khr.generate_report(khr.compute_entity_stats(entities),
                                 khr.compute_relationship_stats([], entities),
                                 [], [], NOW,
                                 stale_unevaluable=(0, COHORT),
                                 stamp_cohort=khr.stale_stamp_cohort(entities))

    assert "*No stale facts found.*" not in report
    assert "No stale facts found" not in report
    assert STAMP_RE.search(_stale_section(report)) is not None


def test_the_unevaluable_line_keeps_its_n_when_no_date_dominates(tmp_path):
    """Clause 4's second half, on a corpus the guard must not touch: the shipped shape
    from `test_unevaluable_line_names_n_and_m_over_active_facts` — 2 dateless of 3 active
    — keeps the line's format, its `n` and its percent, and gains no cohort line, because
    nothing on that tree shares a date at the floor the guard names.
    """
    root = tmp_path / "facts"
    _write(root, "Blind", "state", [
        {"fact": "no date"},
        {"fact": "also no date", "valid_at": _iso(3)},
        {"fact": "dated and fresh", "event_date": _iso(3)},
    ])
    entities = khr.load_entities(root)

    assert khr.stale_stamp_cohort(entities) is None
    section = _stale_section(_stamp_report(entities))
    match = UNEVALUABLE_RE.search(section)

    assert match is not None, section
    assert match.groups() == ("2", "3", "66.7")
    assert STAMP_RE.search(section) is None, "no cohort measured, so no cohort line"


def test_a_stamp_only_corpus_cannot_print_the_clean_verdict(tmp_path):
    """Clause 5: a corpus whose only dated facts are dominant-cohort facts gets no clean
    verdict, the same way `test_undatable_fixture_cannot_print_the_clean_verdict` denies
    one to a dateless store — `find_stale_facts` returns nothing, so before #2179 this
    printed "*No stale facts found.*" over a quarter-million facts that had merely not
    reached 2026-11-22 yet.
    """
    root = tmp_path / "facts"
    _write(root, "Rebuilt", "state",
           [{"fact": f"stamp claim {i}", "created_at": _iso(COHORT_AGE)}
            for i in range(COHORT)])
    entities = khr.load_entities(root)

    assert khr.find_stale_facts(entities, NOW, THRESH) == []
    report = _stamp_report(entities)
    section = _stale_section(report)

    assert "No stale facts found" not in report
    assert STAMP_RE.search(section) is not None

    # The control that keeps this from being a size effect: the same 1,000 records, dated
    # on spread-out days instead of one stamp, is a corpus the section may call clean.
    spread = tmp_path / "spread"
    _write(spread, "Spread", "state",
           [{"fact": f"dated {i}", "created_at": _iso(1 + (i % 10))}
            for i in range(COHORT)])
    spread_entities = khr.load_entities(spread)

    assert khr.stale_stamp_cohort(spread_entities) is None
    assert khr.stale_coverage(spread_entities) == (0, COHORT)
    assert "*No stale facts found.*" in _stamp_report(spread_entities)


def test_the_nightly_run_prints_the_cohort_line_and_exits_zero(tmp_path, monkeypatch):
    """The wiring, across the process boundary the nightly job actually crosses: `main()`
    measures the cohort off the parsed tree and the dated report it writes carries the
    line — a guard only reachable from the test helper would be a guard the monitor never
    runs. Staleness still cannot reach the alarm channel or the exit code (#841 clause 6,
    which this change must not move).
    """
    root = tmp_path / "facts"
    out = tmp_path / "out"
    _write(root, "Rebuilt", "state",
           [{"fact": f"stamp claim {i}", "created_at": _iso(COHORT_AGE)}
            for i in range(COHORT)])
    _write(root, "Pile", "state", [
        {"fact": "old claim", "event_date": _iso(200)},
        {"fact": "undatable claim"},
    ])
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(khr, "_kg_store",
                        lambda: _FakeStore([{"source": "Rebuilt", "target": "Other", "type": "uses"}],
                                           {"edges_active": 53416, "edges_total": 53416}))
    monkeypatch.setattr(khr, "fact_duplicate_stats",
                        lambda: {"unavailable": True, "reason": "store faked in test"})
    alerts: list = []
    monkeypatch.setattr(khr, "_alert", lambda alarms, path: alerts.append(alarms))
    monkeypatch.setattr(sys, "argv", ["knowledge-health-report.py",
                                      "--facts-dir", str(root),
                                      "--output-dir", str(out)])

    rc = khr.main()

    assert rc == khr.EXIT_OK
    assert alerts == []
    reports = list(out.glob("knowledge-health-*.md"))
    assert len(reports) == 1
    section = _stale_section(reports[0].read_text())

    match = STAMP_RE.search(section)
    assert match is not None, section
    stamp = (NOW - timedelta(days=COHORT_AGE)).date()
    # 1,000 of the 1,002 active records: the cohort, the `event_date` row and the one
    # record with no date at all.
    assert match.groups()[:2] == ("1,000", "99.8")
    assert match.group(4) == stamp.isoformat()
    assert match.group(6) == (stamp + timedelta(days=THRESH)).isoformat()
    # Only the `event_date` fact is stale; the 1,000-record cohort is not in that count.
    assert "**1** in total" in section, section
    assert UNEVALUABLE_RE.search(section).groups() == ("1,001", "1,002", "99.9")


# ── edge-type cardinality (#546) ─────────────────────────────────────────────

CARDINALITY_RE = re.compile(r"^Edge-type cardinality: (PASS|FAIL) — ", re.M)


def _drifted_edges() -> list[dict]:
    """`mentions` at 60% of 100 active edges, two count-1 types and one at 3 —
    the shape the live store had when #546 was measured."""
    edges = [{"source": f"S{i}", "target": f"T{i}", "type": "mentions"} for i in range(60)]
    edges += [{"source": f"U{i}", "target": f"V{i}", "type": "uses"} for i in range(35)]
    edges += [{"source": f"D{i}", "target": f"E{i}", "type": "depends_on"} for i in range(3)]
    edges += [{"source": "A", "target": "B", "type": "informs"},
              {"source": "C", "target": "D", "type": "ships"}]
    return edges


def test_cardinality_line_lists_rare_types_and_the_dominant_share():
    """#546's list test, with the verdict's authorship pinned (#1820 clause 5).

    This fixture always held two kinds of rare type, which is why it is the clause-5
    fixture: `informs` and `ships` are names `app.kg_store.EDGE_TYPES` does not contain,
    while `depends_on` does. Under #1658 all three sat in one bucket — the FAIL named
    whichever of them were old enough to fail. Under #1820's rule the two junk names
    still FAIL the line and are the only names in the failing conditions, and
    `depends_on (3)` is still LISTED but contributes nothing to the verdict: the line
    reports every type under the floor, because a reader who cannot see a rare approved
    type cannot tell it from a store that stopped recording them. A fixture that
    exercised only the canonical half could not tell those two behaviours apart, and
    that ambiguity is what let a floor of 5 FAIL two legitimate `fact_relate` singletons
    (`conflicts_with` 44916, `describes` 48187) for a month.

    `uses (` staying absent is #546's own clause: 35 uses is above the floor, so a
    type that passes the floor must not be decorated with a count that reads as rare.
    """
    dist = {"mentions": 60, "uses": 35, "depends_on": 3, "informs": 1, "ships": 1}
    line = khr.edge_type_cardinality(dist)
    assert CARDINALITY_RE.match(line).group(1) == "FAIL"
    for rare in ("informs (1)", "ships (1)", "depends_on (3)"):
        assert rare in line
    assert "conditions failing: types below the 5-use floor (informs (1), ships (1))" \
        in line, line
    assert "depends_on" not in line.split("conditions failing:")[1], line
    assert "uses (" not in line  # 35 uses is above the floor
    assert "dominant type mentions is 60.0% of 100 active edges" in line


def test_cardinality_passes_a_spread_vocabulary_with_no_rare_type():
    line = khr.edge_type_cardinality({"uses": 40, "part_of": 35, "related_to": 25})
    assert CARDINALITY_RE.match(line).group(1) == "PASS"
    assert "types under 5 uses: none" in line


@pytest.mark.parametrize("dist,previous", [
    # A catch-all with no rare types. #1658 made that condition a trend, so it fails
    # on the ceiling (87% > 80%) or on a rise (51% over a 49% night) — not on 51%
    # alone, which is what FAILed every night while task #74 retyped the dominant
    # type down by ~2.5k edges a night.
    ({"mentions": 87, "uses": 13}, None),
    ({"mentions": 51, "uses": 49}, {"date": "2026-09-27", "share": 0.49}),
    # A rare type with no catch-all still fails on its own, with no history at all.
    ({"uses": 50, "part_of": 50, "informs": 4}, None),
])
def test_either_signal_alone_fails(dist, previous):
    """A runaway catch-all fails, and so does a rare type with no catch-all — they
    are two different drifts and each still fails without the other."""
    assert CARDINALITY_RE.match(
        khr.edge_type_cardinality(dist, previous=previous)).group(1) == "FAIL"


def test_cardinality_fail_is_report_output_and_still_exits_zero(tmp_path, monkeypatch):
    """The verdict is printed, not alarmed: a drifted store prints FAIL in the
    report and exits EXIT_OK with no alert, so `_alarms()`'s four conditions
    stay the only ones that move the exit code."""
    root = tmp_path / "facts"
    out = tmp_path / "out"
    _write(root, "Pile", "state", [{"fact": "fresh claim", "event_date": _iso(1)}])
    monkeypatch.setenv("HOME", str(tmp_path))
    edges = _drifted_edges()
    monkeypatch.setattr(khr, "_kg_store",
                        lambda: _FakeStore(edges, {"edges_active": len(edges),
                                                   "edges_total": len(edges)}))
    monkeypatch.setattr(khr, "fact_duplicate_stats",
                        lambda: {"unavailable": True, "reason": "store faked in test"})
    alerts: list = []
    monkeypatch.setattr(khr, "_alert", lambda alarms, path: alerts.append(alarms))
    monkeypatch.setattr(sys, "argv", ["knowledge-health-report.py",
                                      "--facts-dir", str(root),
                                      "--output-dir", str(out)])

    rc = khr.main()

    assert rc == khr.EXIT_OK
    assert alerts == []
    text = next(out.glob("knowledge-health-*.md")).read_text()
    m = CARDINALITY_RE.search(text)
    assert m is not None and m.group(1) == "FAIL"
    assert "informs (1)" in text and "mentions is 60.0%" in text


def test_alarms_take_no_edge_type_input():
    """Pinned by signature: `_alarms()` cannot see the type distribution, so no
    cardinality verdict can reach the exit code through it."""
    assert list(inspect.signature(khr._alarms).parameters) == [
        "store_stats", "hygiene", "duplicate_id_files", "baseline"]
    assert khr._alarms({"edges_active": 100}, {}, 0, 0) == []


# ── #1596: the conflicts_with resolution trace is counted, and counted only ───
#
# `fact_resolve_apply` has written a `conflicts_with` trace onto each loser since
# #1544, and nothing anywhere counted the records that carry one, so the only
# answer to "is the resolve path being used, and is the number rising?" was a
# hand-run grep over ~32,000 fact files. The live corpus measures 0 today —
# `contradiction_trace_coverage` over ~/lloyd-data/_pipeline/vault-derived/facts
# returned (0, 118,343) on 2026-09-28, over 12,483 entity dirs, with
# `stale_coverage` returning (2, 118,336) on the same load as the control that says
# the walk really read the facts. A line whose first job is to print a real zero is
# a measurement, so it prints its denominator beside its count (#841/#1543) and it
# never moves the exit code.

TRACE_COVERAGE_RE = re.compile(
    r"Fact records carrying a `conflicts_with` resolution trace \| "
    r"([\d,]+) of ([\d,]+) fact records")


def _settled_pair(root: Path, *, traced: bool) -> Path:
    """One entity, two records: a winner and a loser `fact_resolve_apply` settled.

    The loser carries `invalid_at` because the mark and the trace are one write, so
    a counter that walked only ACTIVE facts would report 0 over a corpus made of
    nothing but resolutions — the failure mode this fixture's shape exists to catch.
    `traced=False` is the same pair with the trace key stripped: the state the live
    corpus is in, and the state that must still print a number.
    """
    loser = {"fact": "the feature is disabled", "confidence": 0.5,
             "created_at": _iso(30), "invalid_at": _iso(1),
             "invalid_reason": "fact_resolve_apply: opposing_terms:enabled/disabled"}
    if traced:
        loser[khr.CONTRADICTION_TRACE_KEY] = {
            "type": "conflicts_with", "entity": "Lloyd",
            "file": "Lloyd/Lloyd-state.md", "fact_id": "stat-001",
            "fact": "the feature is enabled", "confidence": 0.9,
            "reason": "opposing_terms:enabled/disabled", "resolved_at": _iso(1)}
    return _write(root, "Lloyd", "state", [
        {"fact": "the feature is enabled", "confidence": 0.9, "created_at": _iso(2)},
        loser])


def _hygiene_report(entities: dict) -> str:
    """The report with the Hygiene section rendered and the trace line measured."""
    return khr.generate_report(
        khr.compute_entity_stats(entities),
        khr.compute_relationship_stats([], entities),
        [],
        khr.find_stale_facts(entities, NOW, THRESH),
        NOW,
        hygiene=_clean_hygiene(),
        stale_unevaluable=khr.stale_coverage(entities),
        trace_coverage=khr.contradiction_trace_coverage(entities))


def test_one_traced_loser_in_a_corpus_reports_one(tmp_path):
    """Clause 2: one traced loser plus untraced records reports 1, in the report."""
    root = tmp_path / "facts"
    _settled_pair(root, traced=True)
    # A record whose TEXT merely mentions the key is the live corpus's only `
    # conflicts_with` hit (SkillDAG-state.md is prose about the paper's own edge
    # vocabulary), so it is the control: counted, the number would be 2 not 1.
    _write(root, "SkillDAG", "state",
           [{"fact": "the paper's conflicts_with edge type", "created_at": _iso(3)}])
    entities = khr.load_entities(root)
    loser = [f for f in entities["Lloyd"][0]["facts"]
             if f["fact"] == "the feature is disabled"][0]
    assert not khr.is_fact_active(loser), (
        "the trace must sit on a retired record: it only ever exists on the loser "
        "of a pair, which the same write just invalidated")
    assert khr.contradiction_trace_coverage(entities) == (1, 3), (
        "counted once, over every record whether active or retired, and not out of "
        "a fact's text")
    m = TRACE_COVERAGE_RE.search(_hygiene_report(entities))
    assert m is not None, "the report prints no resolution-trace line at all"
    assert (m.group(1), m.group(2)) == ("1", "3"), m.groups()


def test_an_untraced_corpus_prints_zero_beside_its_label(tmp_path):
    """Clause 3, zero half: 0 renders as a measurement, not as an absent row."""
    root = tmp_path / "facts"
    _settled_pair(root, traced=False)
    entities = khr.load_entities(root)
    assert khr.contradiction_trace_coverage(entities) == (0, 2), (
        "the settled pair with its trace key stripped reads 0: the counter keys on "
        "the field, never on `invalid_at`, which the live corpus holds 115,911 of")
    m = TRACE_COVERAGE_RE.search(_hygiene_report(entities))
    assert m is not None, "a zero must still print the line"
    assert (m.group(1), m.group(2)) == ("0", "2"), m.groups()
    assert "not measured" not in m.group(0)


def test_the_trace_line_never_moves_the_alarm_exit_code(tmp_path, monkeypatch):
    """Clause 3, end to end: a corpus holding a real resolution exits EXIT_OK,
    posts no alarm, and `_alarms()` has no parameter able to carry the number.

    The alarm baseline is the real one (`active_edges` 37,867 on 2026-09-28), so
    the faked store reports above half of it and the exit code under test is about
    the trace and nothing else.
    """
    root = tmp_path / "facts"
    out = tmp_path / "out"
    _settled_pair(root, traced=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(khr, "_kg_store", lambda: _FakeStore(
        [{"source": "Lloyd", "target": "Other", "type": "uses"}],
        {"edges_active": 53416, "edges_total": 53416}))
    monkeypatch.setattr(khr, "fact_duplicate_stats",
                        lambda: {"unavailable": True, "reason": "store faked in test"})
    alerts: list = []
    monkeypatch.setattr(khr, "_alert", lambda alarms, path: alerts.append(alarms))
    monkeypatch.setattr(sys, "argv", ["knowledge-health-report.py",
                                      "--facts-dir", str(root),
                                      "--output-dir", str(out)])

    rc = khr.main()

    assert rc == khr.EXIT_OK, f"a traced record moved the exit code: rc={rc}"
    assert alerts == []
    text = next(out.glob("knowledge-health-*.md")).read_text()
    m = TRACE_COVERAGE_RE.search(text)
    assert m is not None and m.group(1) == "1", text
    assert list(inspect.signature(khr._alarms).parameters) == [
        "store_stats", "hygiene", "duplicate_id_files", "baseline"], (
        "_alarms() gained an input that can carry the trace count to the exit code")


# ── clause 5 (#1817): the counter has to see the writer that actually runs ────
# Everything above feeds the counter a corpus this file wrote by hand. That is the
# right shape for testing the counter and exactly why the line could read
# `0 of 119,167 fact records` while a daily job condemned facts: nothing in this file
# had ever asked what the improve loop's own write looks like to the report. So this
# node runs the real writer into a temp root and hands the report THAT root.

def test_the_daily_writer_s_mark_is_counted_by_the_report(tmp_path, monkeypatch):
    """One `confidence` apply by the improve loop is traced >= 1 to the report.

    Crosses the boundary the nightly line is built on: `agent_mcp.fact_improvement`
    writes the markdown, `knowledge-health-report.py` reads it back through its own
    loader. Before #1817 the pair was one-directional — the writer marked
    `invalid_at` through a text match with no extras, so the counter had nothing to
    find and the line measured a population the only writer that runs could not move.
    """
    import test_fact_improvement_confidence_trace as writer
    fi = writer.fi

    root = writer._built_tree(tmp_path, monkeypatch, list(writer._RECORDS))
    planned = fi.plan_entity("Idcol")
    action = next(a for a in planned["actions"] if a["kind"] == "confidence")
    result = fi.apply_action(action, writer._iso(1))
    assert result["expired_count"] == 1 and result["traces_written"] == 1, result

    # The report's OWN loader, pointed at the root the writer just wrote — not this
    # file's `_write`, which would only prove the counter matches this file's shape.
    entities = khr.load_entities(root)
    traced, total = khr.contradiction_trace_coverage(entities)
    assert total == 4, f"the report lost records on read: {total}"
    assert traced == 1, (
        f"the writer left a trace and the counter found {traced}: the two halves "
        "disagree about what a resolution record looks like")
    # And it is the invalidated record that carries it, so the denominator the line
    # prints (`of N fact records`) and the numerator are the same population.
    marked = [f for groups in entities.values() for group in groups
              for f in group["facts"] if f.get("invalid_at")]
    assert len(marked) == 1 and marked[0].get("conflicts_with"), marked
