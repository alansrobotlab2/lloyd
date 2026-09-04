"""entity_semantic_gate.py — merge only on unanimous SAME; fail closed; cache."""
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "memory"))
import entity_semantic_gate as g  # noqa: E402


def _overview(root, name, definition=None, summary_line=None):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    fm = {"type": "overview", "entity": name, "category": "overview"}
    if definition is not None:
        fm["definition"] = definition
    body = f"# Summary\n\n{summary_line or ''}\n"
    (d / f"{name}-overview.md").write_text(f"---\n{yaml.dump(fm, sort_keys=False)}---\n\n{body}")


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "facts"
    _overview(r, "Intel", "Intel is a semiconductor company that released the Pro B70 GPU.")
    _overview(r, "Intel Pipeline", "Intel Pipeline is an information aggregation system scanning ArXiv.")
    _overview(r, "Morning Briefing", summary_line="An automated daily digest task that aggregates scored items.")
    _overview(r, "Morning Briefing System", "A proactive system that surfaces staleness in the daily digest.")
    return r


def test_definition_prefers_frontmatter_then_summary(root):
    assert g.entity_definition("Intel", root).startswith("Intel is a semiconductor")
    assert g.entity_definition("Morning Briefing", root).startswith("An automated daily digest")
    assert g.entity_definition("Nope", root) == ""


def test_parse_verdict():
    assert g.parse_verdict("SAME — both are the robot") == "SAME"
    assert g.parse_verdict("different, one is a company") == "DIFFERENT"
    assert g.parse_verdict("I think maybe") == "UNPARSED"


def _fake(answer):
    def fn(user):
        fn.seen = user
        return answer
    return fn


def test_unanimous_same_allows_merge(root, tmp_path):
    gate = g.SemanticGate(root, tmp_path / "cache.jsonl",
                          judge_fns={"a": _fake("SAME — same thing"), "b": _fake("SAME — yes")})
    v = gate.verdict("Morning Briefing System", "Morning Briefing")
    assert v["decision"] == "SAME"
    assert v["judges"]["a"]["reason"] == "same thing"
    assert v["cached"] is False
    # both definitions reached the judge
    assert "semiconductor" not in gate.judge_fns["a"].seen
    assert "daily digest" in gate.judge_fns["a"].seen


def test_any_different_routes_to_review(root, tmp_path):
    gate = g.SemanticGate(root, tmp_path / "cache.jsonl",
                          judge_fns={"a": _fake("SAME — corp"), "b": _fake("DIFFERENT — a scanner vs a company")})
    v = gate.verdict("Intel Pipeline", "Intel")
    assert v["decision"] == "REVIEW"
    assert v["judges"]["b"]["verdict"] == "DIFFERENT"


def test_judge_error_fails_closed(root, tmp_path):
    def boom(user):
        raise ConnectionError("vLLM down")
    gate = g.SemanticGate(root, tmp_path / "cache.jsonl",
                          judge_fns={"a": _fake("SAME — x"), "b": boom})
    v = gate.verdict("Intel Pipeline", "Intel")
    assert v["decision"] == "REVIEW"
    assert v["judges"]["b"]["verdict"] == "ERROR"


def test_unparsed_answer_fails_closed(root, tmp_path):
    gate = g.SemanticGate(root, tmp_path / "cache.jsonl", judge_fns={"a": _fake("hmm, hard to say")})
    assert gate.verdict("Intel Pipeline", "Intel")["decision"] == "REVIEW"


def test_verdicts_are_cached_by_definition(root, tmp_path):
    cache = tmp_path / "cache.jsonl"
    gate = g.SemanticGate(root, cache, judge_fns={"a": _fake("SAME — ok")})
    gate.verdict("Intel Pipeline", "Intel")
    assert gate.calls == 1
    assert cache.exists() and len(cache.read_text().splitlines()) == 1
    v2 = gate.verdict("Intel Pipeline", "Intel")
    assert v2["cached"] is True and gate.calls == 1
    # a fresh gate reloads the cache from disk
    gate2 = g.SemanticGate(root, cache, judge_fns={"a": _fake("DIFFERENT — changed my mind")})
    assert gate2.verdict("Intel Pipeline", "Intel")["decision"] == "SAME" and gate2.calls == 0
    # but a changed definition invalidates it
    _overview(root, "Intel", "Intel is now defined differently.")
    assert gate2.verdict("Intel Pipeline", "Intel")["decision"] == "REVIEW" and gate2.calls == 1


def test_default_judges_honour_secondary_switch(monkeypatch):
    from app.config import CONFIG
    monkeypatch.delenv("KG_JUDGES", raising=False)
    monkeypatch.setitem(CONFIG, "secondary_enabled", False)
    assert [a for a, _ in g.default_judges()] == ["primary"]
    monkeypatch.setitem(CONFIG, "secondary_enabled", True)
    assert [a for a, _ in g.default_judges()] == ["primary", "secondary"]
    monkeypatch.setenv("KG_JUDGES", "x=http://h/x, y=http://h/y")
    assert g.default_judges() == [("x", "http://h/x"), ("y", "http://h/y")]


def test_missing_definition_is_review_and_never_asks_a_judge(root, tmp_path):
    """A name with nothing on file can only be judged by its shape — refuse."""
    (root / "Fresh Restore Pipeline").mkdir()          # dir exists, no overview yet
    asked = []
    def spy(user):
        asked.append(user); return "SAME — looks the same"
    cache = tmp_path / "cache.jsonl"
    # poison the cache with a shape-only SAME to prove it is bypassed
    gate = g.SemanticGate(root, cache, judge_fns={"a": spy})
    v = gate.verdict("Fresh Restore Pipeline", "Intel")
    assert v["decision"] == "REVIEW" and asked == []
    assert v["judges"]["gate"]["verdict"] == "NO_DEFINITION"
    assert "Fresh Restore Pipeline" in v["judges"]["gate"]["reason"]
    assert not cache.exists() or cache.read_text() == ""
    # once a definition appears, the judge is consulted normally
    _overview(root, "Fresh Restore Pipeline", "A pipeline that restores things freshly.")
    assert gate.verdict("Fresh Restore Pipeline", "Intel")["decision"] == "SAME" and len(asked) == 1
