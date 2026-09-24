"""#711: the per-skill activation eval — triggers, misses, false triggers.

Fixture skills go through the real loader and the real matcher
(`prefetch._search_skills` with a substituted inventory), so these pin the
definitions without reading the live vault. The checked-in corpus is checked
for its contract from the YAML alone.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

from scripts import skill_activation as A

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "eval" / "run_skill_activation_eval.py"


def _runner():
    spec = importlib.util.spec_from_file_location("run_skill_activation_eval", RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _skill(name, desc):
    return A.skill_from_text(name, f"---\nname: {name}\ndescription: {desc}\n---\n# {name}\n")


TEA = "tea-brewing"
RECORDS = [
    {"id": "p1", "turn": "how long should I steep oolong", "expected_skills": [TEA]},
    {"id": "p2", "turn": "what temperature for green leaves", "expected_skills": [TEA]},
    {"id": "p3", "turn": "steep the oolong again please", "expected_skills": [TEA]},
    {"id": "p4", "turn": "brew a pot of tea for me", "expected_skills": [TEA]},
    {"id": "p5", "turn": "the green leaves taste bitter, what temperature", "expected_skills": [TEA]},
    {"id": "n1", "turn": "make coffee in the french press", "expected_skills": ["coffee-making"]},
    {"id": "n2", "turn": "espresso shot is too sour", "expected_skills": ["coffee-making"]},
    {"id": "n3", "turn": "grind coffee beans coarse for french press", "expected_skills": ["coffee-making"]},
    {"id": "n4", "turn": "restart the backend server now", "expected_skills": [], "expected_empty": True},
    {"id": "n5", "turn": "what's the weather tomorrow afternoon", "expected_skills": [], "expected_empty": True},
]


def _inventory(tea_desc):
    return [_skill("coffee-making", "coffee french press espresso grinder beans"),
            _skill(TEA, tea_desc)]


def test_counts_misses_and_false_triggers_on_a_fixture():
    cases = A.build_cases(RECORDS, [TEA])
    res = A.evaluate(cases, _inventory("steep oolong leaves, coffee espresso french press"))
    t = res["skills"][TEA]
    # p2 and p5 need `green`/`temperature`, which this description lacks.
    assert (t["positives"], t["triggers"], t["misses"]) == (5, 3, 2)
    # n1 and n3 carry three coffee words each, enough for the excerpt slot.
    assert (t["negatives"], t["false_triggers"]) == (5, 2)
    assert t["recall"] == 0.6 and t["false_trigger_rate"] == 0.4
    by_id = {r["id"]: r for r in t["rows"]}
    assert by_id["n1"]["winner"] == "coffee-making" and by_id["n1"]["pass"] is False
    assert by_id["n1"]["expected"] == ["coffee-making"], "a negative names who should have fired"
    assert by_id["n4"]["expected"] == [] and by_id["n4"]["pass"] is True
    assert res["corpus"]["false_triggers"] == 2 and res["corpus"]["recall"] == 0.6


def test_a_rate_over_nothing_prints_its_denominator():
    cases = A.build_cases(RECORDS[5:], [TEA])  # negatives only
    res = A.evaluate(cases, _inventory("steep oolong leaves"))
    t = res["skills"][TEA]
    assert t["recall"] is None and t["false_trigger_rate"] == 0.0
    lines = "\n".join(_runner().render(res))
    assert "n/a (denominator 0)" in lines
    assert "0.0000" in lines


def test_positives_without_negatives_fail_the_run(monkeypatch, capsys):
    ev = _runner()
    only_pos = [r for r in RECORDS if r["expected_skills"] == [TEA]]
    monkeypatch.setattr(A, "load_spec", lambda *a, **k: {"skills": {TEA: {}}, "extra_records": []})
    monkeypatch.setattr(A, "load_records", lambda *a, **k: only_pos)
    monkeypatch.setattr(A, "base_skills", lambda: _inventory("steep oolong leaves"))
    monkeypatch.setattr("prefetch._get_skills_cached", lambda: _inventory("steep oolong leaves"))
    assert ev.main([]) == 1
    assert "positives and no negatives" in capsys.readouterr().err
    assert A.contract_errors(A.build_cases(only_pos, [TEA])) == [
        f"{TEA}: 5 positives and no negatives"]


def test_two_runs_report_identical_counts_and_rows():
    cases = A.build_cases(RECORDS, [TEA])
    inv = _inventory("steep oolong and green leaves at the right temperature")
    ev = _runner()
    one = ev.build_report(A.evaluate(cases, inv))
    two = ev.build_report(A.evaluate(cases, inv))
    one.pop("measured_at"), two.pop("measured_at")
    assert one == two
    row = one["rows"][0]
    assert set(row) >= {"id", "label", "winner", "pass"}
    assert one["matches_production_defaults"] is True
    assert {"graph_rerank", "rerank_alpha", "graph_top_k", "graph_hops"} <= set(one)


def test_the_baseline_lives_under_the_data_root():
    from app.paths import EVAL_BASELINES_DIR
    assert _runner().baseline_path() == EVAL_BASELINES_DIR / "skill_activation_baseline.json"


def test_activation_is_what_prefetch_injects(monkeypatch):
    """Top skill at the first threshold, the second only at SKILL_THRESHOLD_SECOND."""
    import prefetch
    a, b = {"name": "a"}, {"name": "b"}
    monkeypatch.setattr(prefetch, "_search_skills",
                        lambda toks, skills=None: [(9.0, a), (prefetch.SKILL_THRESHOLD_SECOND - 0.1, b)])
    assert A.injected("a long enough turn") == ["a"]
    monkeypatch.setattr(prefetch, "_search_skills",
                        lambda toks, skills=None: [(9.0, a), (prefetch.SKILL_THRESHOLD_SECOND, b)])
    assert A.injected("a long enough turn") == ["a", "b"]
    assert A.injected("hi") == [], "prefetch skips a message under MIN_MESSAGE_LEN"


def test_the_checked_in_corpus_is_derived_and_meets_its_contract():
    spec = A.load_spec()
    assert len(spec["skills"]) >= 5
    raw = yaml.safe_load(A.CASES.read_text())
    assert set(raw) <= {"skills", "extra_records"}, "cases come from the #557 set, not a copy"
    queries = {r["id"] for r in yaml.safe_load(A.QUERIES.read_text())["records"]}
    assert not queries & {r["id"] for r in spec["extra_records"]}
    assert all("authored" in r["source"] for r in spec["extra_records"])
    cases = A.build_cases(A.load_records(spec), spec["skills"])
    assert A.contract_errors(cases, 5) == []
    assert all(isinstance(e.get("recall_floor"), float) for e in spec["skills"].values())


@pytest.mark.live_vault
def test_every_covered_skill_is_live():
    active = {s["name"] for s in A.base_skills()}
    assert set(A.load_spec()["skills"]) <= active
