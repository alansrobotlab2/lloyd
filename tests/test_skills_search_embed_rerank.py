"""skills_search's embedding re-order (skills.search_embed_rerank, on since 2026-11-10).

The chat prefetch leg re-orders its offers by embedding (2026-11-09); the tool
the model calls when the index missed one stayed lexical. This pins that the
tool path can take the same re-order, that it changes ORDER only, and that it is
off by default.
"""
from __future__ import annotations

import json

import pytest

from agent_mcp import skills as skills_mod
from app import skill_embed_rank

SKILLS = [
    {"name": "restart-services", "description": "restart lloyd services safely",
     "category": "ops", "tags": [], "body": "round restart"},
    {"name": "service-health", "description": "check lloyd services health",
     "category": "ops", "tags": [], "body": "status"},
]


@pytest.fixture
def two_skills(monkeypatch):
    monkeypatch.setattr(skills_mod, "load_skills_cached", lambda: [dict(s) for s in SKILLS])


def _names(payload: str) -> list[str]:
    return [r["name"] for r in json.loads(payload)["results"]]


def test_off_by_default(two_skills, monkeypatch):
    monkeypatch.setattr(skill_embed_rank, "tool_settings", lambda: None)
    called: list = []
    monkeypatch.setattr(skill_embed_rank, "rerank", lambda *a, **k: called.append(1))
    skills_mod._skills_search({"query": "lloyd services"})
    assert called == []


def test_on_it_reorders_and_keeps_lexical_scores(two_skills, monkeypatch):
    monkeypatch.setattr(skill_embed_rank, "tool_settings",
                        lambda: skill_embed_rank.tool_settings_from({}))
    monkeypatch.setattr(skill_embed_rank, "rerank",
                        lambda text, scored, skills, cfg: list(reversed(scored)))
    lex = skills_mod._skills_search_scored("restart lloyd services", embed=False)
    out = json.loads(skills_mod._skills_search({"query": "restart lloyd services"}))
    assert [r["name"] for r in out["results"]] == [s["name"] for _, s in reversed(lex)]
    assert {r["score"] for r in out["results"]} == {round(sc, 2) for sc, _ in lex}
    assert out["total"] == len(lex)


def test_a_failing_rerank_is_lexical_order(two_skills, monkeypatch):
    monkeypatch.setattr(skill_embed_rank, "tool_settings",
                        lambda: skill_embed_rank.tool_settings_from({}))

    def boom(*a, **k):
        raise RuntimeError("no model")
    monkeypatch.setattr(skill_embed_rank, "rerank", boom)
    lex = [s["name"] for _, s in skills_mod._skills_search_scored("restart lloyd services", embed=False)]
    assert _names(skills_mod._skills_search({"query": "restart lloyd services"})) == lex


def test_tool_settings_read_config(monkeypatch):
    from app import config
    monkeypatch.setattr(config, "CONFIG", {"skills": {"search_embed_rerank": True,
                                                      "embed_fusion": {"enabled": False, "weight": 3, "k": 8}}})
    cfg = skill_embed_rank.tool_settings()
    assert cfg["weight"] == 3.0 and cfg["k"] == 8 and cfg["wait_ms"] == skill_embed_rank.TOOL_WAIT_MS
    monkeypatch.setattr(config, "CONFIG", {"skills": {"embed_fusion": {"enabled": True}}})
    assert skill_embed_rank.tool_settings() is None


def test_the_shipped_config_turns_it_on_and_tests_hold_it_off():
    # On since 2026-11-10 (eval/measurements/skills-search-embed-2026-11-10.md);
    # conftest's autouse fixture keeps the 2.5 GB embedder out of test processes.
    import yaml
    from pathlib import Path
    from app import config
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text())
    assert cfg["skills"]["search_embed_rerank"] is True
    assert skill_embed_rank.tool_settings() is None


def test_the_eval_tool_arm_scores_both_orders(two_skills, monkeypatch):
    from eval import run_skill_match_eval as ev
    monkeypatch.setattr(skill_embed_rank, "_build_vectors", lambda skills: None)
    monkeypatch.setattr(skill_embed_rank, "rerank",
                        lambda text, scored, skills, cfg: list(reversed(scored)))
    lex = ev.tool_arm(False)("restart lloyd services")
    emb = ev.tool_arm(True)("restart lloyd services")
    assert lex and emb == list(reversed(lex))
