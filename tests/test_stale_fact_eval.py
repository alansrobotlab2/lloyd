"""#622 — the plant-then-supersede memory eval (`eval/run_stale_fact_eval.py`).

Every engine call is stubbed: the runner's `complete`, `judge_fn` and
`primary` hooks stand in for the primary, so nothing here needs :8096.

Run: .venvs/lloyd/bin/python -m pytest tests/test_stale_fact_eval.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

_spec = importlib.util.spec_from_file_location("run_stale_fact_eval", ROOT / "eval" / "run_stale_fact_eval.py")
R = importlib.util.module_from_spec(_spec)
sys.modules["run_stale_fact_eval"] = R
_spec.loader.exec_module(R)


def _sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


@pytest.fixture
def shadow(tmp_path, monkeypatch):
    monkeypatch.setenv("LLOYD_FACTS_ROOT", str(tmp_path / "facts"))
    monkeypatch.setenv("LLOYD_KG_DB", str(tmp_path / "kg.sqlite"))
    with R.restored_store_pointers():
        yield tmp_path


# ── the shadow guard ──────────────────────────────────────────────────────────

def test_guard_refuses_unset_and_live_paths(tmp_path):
    from app.data_root import ACCOUNT_HOME, PRODUCTION_DATA_ROOT
    with pytest.raises(R.LiveStoreRefused, match="LLOYD_FACTS_ROOT is not set"):
        R.check_shadow_paths({})
    live_db = PRODUCTION_DATA_ROOT / "_pipeline" / "vault-derived" / "kg.sqlite"
    with pytest.raises(R.LiveStoreRefused, match="LLOYD_KG_DB"):
        R.check_shadow_paths({"LLOYD_FACTS_ROOT": str(tmp_path / "f"), "LLOYD_KG_DB": str(live_db)})
    with pytest.raises(R.LiveStoreRefused, match="LLOYD_FACTS_ROOT"):
        R.check_shadow_paths({"LLOYD_FACTS_ROOT": str(ACCOUNT_HOME / "obsidian" / "x"),
                              "LLOYD_KG_DB": str(tmp_path / "kg.sqlite")})
    assert R.check_shadow_paths({"LLOYD_FACTS_ROOT": str(tmp_path / "f"),
                                 "LLOYD_KG_DB": str(tmp_path / "kg.sqlite")})


def test_runner_exits_nonzero_naming_the_error_on_a_live_path(monkeypatch, tmp_path, capsys):
    from app.data_root import PRODUCTION_DATA_ROOT
    monkeypatch.setenv("LLOYD_FACTS_ROOT", str(tmp_path / "facts"))
    monkeypatch.setenv("LLOYD_KG_DB", str(PRODUCTION_DATA_ROOT / "_pipeline" / "vault-derived" / "kg.sqlite"))
    assert R.main(["--dry-run", "--out", str(tmp_path / "o.json")]) == 2
    assert "LiveStoreRefused" in capsys.readouterr().err
    assert not (tmp_path / "o.json").exists()


# ── the corpus ────────────────────────────────────────────────────────────────

def test_corpus_counts_are_pinned():
    probes = R.build_corpus()
    assert R.corpus_counts(probes) == {
        "entities": 32, "facts_planted": 128, "facts_per_entity_min": 4,
        "facts_per_entity_max": 4, "superseded": 20, "controls": 12}
    assert R.build_corpus() == probes, "the corpus must be deterministic"
    for p in probes:
        if p.superseded:
            assert p.value != p.old_value
            assert p.old_value not in p.value and p.value not in p.old_value
            assert p.old_fact in [t for _c, t in p.facts] and p.supersede_fact not in [t for _c, t in p.facts]


# ── scoring ───────────────────────────────────────────────────────────────────

def _superseded():
    return next(p for p in R.build_corpus() if p.superseded and p.slot == "port")


def test_old_value_answer_is_a_miss_not_an_unanswered_probe():
    p = _superseded()
    assert R.score(p, f"curl http://localhost:{p.old_value}/health") == "stale"
    assert R.score(p, f"curl http://localhost:{p.value}/health") == "correct"
    assert R.score(p, "I have nothing on record for that.") == "none"
    assert R.score(p, None) == "error"
    both = f"use {p.value} (older note said {p.old_value})"
    assert R.score(p, both) == "mixed"
    assert R.settle("mixed", "current") == "correct"
    assert R.settle("mixed", "superseded") == "stale"
    assert R.settle("mixed", "both_unresolved") == "hedged"
    assert R.settle("mixed", None) == "error"


def test_an_accented_or_grouped_value_still_counts_as_the_value():
    city = R.Probe("X", "person", "person", "city", "q", True, "Valparaiso", "Tallinn", [],
                   match_new=["Valparaiso"], match_old=["Tallinn"])
    assert R.score(city, "Use Valparaíso's clock.") == "correct"
    budget = R.Probe("Y", "project", "project", "budget", "q", True, "16000", "9750", [],
                     match_new=["16000"], match_old=["9750"])
    assert R.score(budget, "Plan against 16,000 GPU-hours.") == "correct"
    assert R.score(budget, "It was 9,750.") == "stale"


# ── the arms ──────────────────────────────────────────────────────────────────

def test_supersession_is_written_through_the_real_writers(shadow):
    from app import kg_store
    probes = R.build_corpus()
    d = R.plant(probes, shadow, "facts_expired")
    R._point_store(d)
    p = _superseded()
    rows = kg_store.store().facts_idx.for_entity(p.entity, include_expired=True)
    old = [r for r in rows if r["fact"] == p.old_fact]
    new = [r for r in rows if r["fact"] == p.supersede_fact]
    assert len(old) == 1 and old[0]["expired_at"], "the old row is expired, not deleted"
    assert len(new) == 1 and not new[0]["expired_at"]
    assert new[0]["category"] == old[0]["category"] == p.slot


def test_stale_evidence_reaches_the_unfiltered_legs_and_not_the_filtered_one(shadow):
    probes = R.build_corpus()
    counts = {}
    for arm in ("facts_expired", "facts_appended", "prose_appended", "prose_consolidated"):
        d = R.plant(probes, shadow, arm)
        counts[arm] = sum(R.stale_evidence(p, R.render(p, d, arm)) for p in probes)
    assert counts["facts_expired"] == 0, "fact_get filters expired rows: the leg that already filters"
    assert counts["facts_appended"] == 20
    assert counts["prose_appended"] == 20, "the lloyd-segment prose leg carries the old value"
    assert counts["prose_consolidated"] == 20


def test_arms_differ_only_by_the_memory_they_hold(shadow):
    probes = R.build_corpus()
    p = _superseded()
    rendered = {arm: R.render(p, R.plant(probes, shadow, arm), arm) for arm in R.ARMS}
    base = rendered["stateless"]
    for arm, msgs in rendered.items():
        assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool"]
        assert msgs[1] == base[1] and msgs[2] == base[2], arm
        assert msgs[0]["content"].split("\n\n<memory>")[0] == base[0]["content"], arm
    assert "<memory>" not in base[0]["content"]
    assert "<memory>" in rendered["prose_appended"][0]["content"]
    assert json.loads(base[3]["content"]).get("error"), "stateless: the store has no such entity"
    assert json.loads(rendered["facts_appended"][3]["content"])["facts"]


# ── a whole run, engine stubbed ───────────────────────────────────────────────

def _stub_engine(probes_by_question):
    async def complete(client, base_url, model, messages, tools, max_tokens, tool_choice="auto"):
        assert tool_choice in ("auto", "none")
        assert sorted(t["function"]["name"] for t in tools) == ["fact_get", "memory_read"]
        p = probes_by_question[messages[1]["content"]]
        text = f"answer: {p.old_value if p.superseded else p.value}"
        return {"message": {"content": text}, "finish": "stop", "completion_tokens": 7}

    async def judge(client, base_url, model, probe, answer):
        raise AssertionError("no answer here names both values")

    return complete, judge


def test_full_run_reports_every_field_and_touches_nothing_live(shadow, tmp_path):
    from app.data_root import ACCOUNT_HOME, PRODUCTION_DATA_ROOT
    live_db = PRODUCTION_DATA_ROOT / "_pipeline" / "vault-derived" / "kg.sqlite"
    vault = ACCOUNT_HOME / "obsidian"
    before_db = _sha(live_db)
    before_mem = {f: _sha(vault / "lloyd" / f) for f in ("MEMORY.md", "USER.md")}
    porcelain = (lambda: subprocess.run(["git", "-C", str(vault), "status", "--porcelain", "--", "lloyd"],
                                        capture_output=True, text=True).stdout) if (vault / ".git").exists() else (lambda: "")
    before_git = porcelain()

    probes = R.build_corpus()
    complete, judge = _stub_engine({p.question: p for p in probes})
    out = tmp_path / "report.json"
    rc = R.main(["--samples", "2", "--out", str(out)], complete=complete, judge_fn=judge,
                primary=lambda _u: ("http://stub", "stub-model"))
    assert rc == 0
    report = json.loads(out.read_text())
    assert report["corpus"]["entities"] == 32 and report["corpus"]["superseded"] == 20
    assert report["corpus"]["controls"] == 12
    for arm in R.ARMS:
        block = report["summary"][arm]
        sa = block["stale_action"]
        assert (sa["numerator"], sa["denominator"], sa["rate"]) == (40, 40, 1.0)
        ctl = block["control_accuracy"]
        assert (ctl["numerator"], ctl["denominator"], ctl["rate"]) == (24, 24, 1.0)
        if arm != "stateless":
            gain = block["memory_gain"]
            assert gain["gain"] == 0.0 and gain["stateful_rate"] == gain["stateless_rate"]
    assert _sha(live_db) == before_db
    assert {f: _sha(vault / "lloyd" / f) for f in before_mem} == before_mem
    assert porcelain() == before_git


def test_run_trial_answers_tool_calls_from_the_arm_then_takes_the_text(shadow):
    import asyncio
    probes = R.build_corpus()
    d = R.plant(probes, shadow, "prose_appended")
    p = _superseded()
    seen = []

    async def complete(client, base_url, model, messages, tools, max_tokens, tool_choice="auto"):
        seen.append(messages[-1])
        if len(seen) == 1:
            return {"message": {"content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
                "name": "memory_read", "arguments": json.dumps({"file": "MEMORY.md"})}}]},
                "finish": "tool_calls", "completion_tokens": 3}
        return {"message": {"content": f"port {p.value}"}, "finish": "stop", "completion_tokens": 4}

    out = asyncio.run(R.run_trial(None, "u", "m", R.render(p, d, "prose_appended"), [], d,
                                  100, 4, complete))
    assert out["answer"] == f"port {p.value}" and out["iterations"] == 2
    assert p.old_fact in json.loads(seen[1]["content"])["content"], "memory_read read the shadow file"


# ── the flag: memory_tools.date_stamp_entries ─────────────────────────────────

def test_memory_add_is_unstamped_by_default_and_stamped_when_on(tmp_path, monkeypatch):
    import datetime
    from agent_mcp import session
    from app import config
    monkeypatch.setattr(session, "MEMORIES_ROOT", tmp_path)
    monkeypatch.setattr(session, "_entry_date", lambda: datetime.date(2026, 9, 2))
    monkeypatch.setitem(config.CONFIG, "memory_tools", {})
    assert session._memory_add({"file": "MEMORY.md", "entry": "- Quillmark runs on godwit-33."})["success"]
    monkeypatch.setitem(config.CONFIG, "memory_tools", {"date_stamp_entries": True})
    assert session._memory_add({"file": "MEMORY.md", "entry": "- Quillmark runs on dunlin-50."})["success"]
    assert session._memory_add({"file": "MEMORY.md", "entry": "plain line"})["success"]
    assert (tmp_path / "MEMORY.md").read_text().splitlines() == [
        "- Quillmark runs on godwit-33.",
        "- (2026-09-02) Quillmark runs on dunlin-50.",
        "(2026-09-02) plain line"]


def test_dated_arms_carry_the_date_through_consolidation(shadow):
    probes = R.build_corpus()
    d = R.plant(probes, shadow, "prose_consolidated_dated")
    text = (d.memories / "MEMORY.md").read_text()
    p = _superseded()
    assert f"- (2026-09-02) {p.supersede_fact}" in text
    assert f") {p.old_fact}" in text and f"- (2026-09-02) {p.old_fact}" not in text
    assert len(d.old_last) == 20 and 0 < sum(d.old_last.values()) < 20, "both orders occur"
