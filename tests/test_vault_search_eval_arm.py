"""eval/run_vault_search_eval.py scores vault_search per pipeline with run_eval's scorer."""
from __future__ import annotations

import json

import yaml

from agent_mcp import vault
from eval import run_vault_search_eval as ev


def test_compare_runs_both_pipelines_and_scores_documents(tmp_path, monkeypatch):
    q = tmp_path / "q.yaml"
    q.write_text(yaml.safe_dump({"queries": [
        {"id": "a", "query": "gearbox backlash", "expect_docs": ["knowledge/gear.md"]},
        {"id": "b", "query": "servo offsets", "expect_docs": ["knowledge/servo.md"]}]}))
    seen: list = []

    def fake(query, limit, min_score, scope, consolidate, *, pipeline=None):
        seen.append((pipeline, consolidate))
        hit = "knowledge/gear.md" if pipeline == "recall" else "knowledge/other.md"
        return {"results": [{"path": hit}]}

    monkeypatch.setattr(vault, "_run_vault_search", fake)
    out = tmp_path / "r.json"
    assert ev.main(["--queries", str(q), "--compare", "--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    assert {p for p, _ in seen} == {"legacy", "recall"} and not any(c for _, c in seen)
    assert rep["summary"]["recall"]["doc_hit"] == 0.5
    assert rep["summary"]["legacy"]["doc_hit"] == 0.0
    assert rep["paired_recall_minus_legacy"]["doc_hit"]["wins"] == 1


def test_an_outage_is_an_error_row_not_a_zero(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise vault.QmdUnavailable("down")
    monkeypatch.setattr(vault, "_run_vault_search", boom)
    row = ev.run_one({"id": "a", "query": "x", "expect_docs": ["k.md"]}, "legacy", 10)
    assert row["error"] and ev.summarize([row])["errors"] == 1
