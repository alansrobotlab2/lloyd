"""knowledge-health-report.py — the Stale Facts section (#841).

The detector used to age a fact from `created_at` alone and skip anything that
carried `valid_at`. On the current store `created_at` exists on ~34% of active
facts and the oldest usable one is younger than the 60-day threshold, so
`find_stale_facts` returned 0 on every run and the report printed the clean
verdict while 140,179 of 315,784 active facts carried no date at all. Every
node below pins one of those three failure modes: the missing second date
source, the `valid_at` amnesty, and the clean verdict over an undatable store.
"""
import importlib.util
import inspect
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

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
