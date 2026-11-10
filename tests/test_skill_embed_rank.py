"""The chat skill matcher's embedding re-order (2026-11-09 review, finding 4).

The embedder is replaced by a fake that maps known texts to fixed unit vectors, so
these nodes pin the contract — order may move, scores and the pool may not, and
every failure is lexical order — without loading 2.5 GB of weights.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

import app.paths as paths
from app import prefetch
from app import skill_embed_rank as R
from app.config import CONFIG


def _sk(name, desc=""):
    return {"name": name, "description": desc, "tags": [], "body": "", "raw": name}


ALPHA, BETA, GAMMA = _sk("alpha", "first"), _sk("beta", "second"), _sk("gamma", "third")
SKILLS = [ALPHA, BETA, GAMMA]
AXIS = {"alpha": 0, "beta": 1, "gamma": 2}


def _fake_embed(text: str):
    v = np.zeros(4, dtype="float32")
    for name, i in AXIS.items():
        if name in text or name.upper() in text:
            v[i] = 1.0
    if not v.any():
        v[3] = 1.0
    return v / np.linalg.norm(v)


@pytest.fixture
def fusion(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "SKILL_EMBED_CACHE_PATH", tmp_path / "emb.json")
    monkeypatch.setattr(R, "_embed", _fake_embed)
    skills_cfg = dict(CONFIG.get("skills") or {})
    skills_cfg["embed_fusion"] = {"enabled": True, "weight": 2.0, "k": 16, "wait_ms": 2000}
    monkeypatch.setitem(CONFIG, "skills", skills_cfg)
    R._reset_for_tests()
    R._build_vectors(SKILLS)            # synchronously, as the background build would
    yield R.settings()
    R._reset_for_tests()


def test_off_by_default_shape_and_off_is_lexical(monkeypatch):
    skills_cfg = dict(CONFIG.get("skills") or {})
    skills_cfg.pop("embed_fusion", None)
    monkeypatch.setitem(CONFIG, "skills", skills_cfg)
    assert R.settings() is None
    scored = [(5.0, ALPHA), (4.0, BETA)]
    assert R.rerank("about BETA", scored, SKILLS) is scored


def test_the_cosine_moves_the_order_and_never_the_scores(fusion):
    scored = [(5.0, ALPHA), (4.5, BETA), (3.2, GAMMA)]
    out = R.rerank("the user asks about gamma", scored, SKILLS, fusion)
    assert [s["name"] for _, s in out] == ["gamma", "alpha", "beta"]
    assert sorted(sc for sc, _ in out) == sorted(sc for sc, _ in scored)
    assert {id(s) for _, s in out} == {id(s) for _, s in scored}, "the pool changed"


def test_only_the_top_k_moves(fusion):
    cfg = dict(fusion, k=2)
    scored = [(5.0, ALPHA), (4.5, BETA), (3.2, GAMMA)]
    out = R.rerank("beta please", scored, SKILLS, cfg)
    assert [s["name"] for _, s in out] == ["beta", "alpha", "gamma"]


def test_a_late_embed_is_lexical_order(fusion, monkeypatch):
    monkeypatch.setattr(R, "_embed", lambda text: (time.sleep(0.5), _fake_embed(text))[1])
    scored = [(5.0, ALPHA), (4.5, BETA)]
    assert R.rerank("beta", scored, SKILLS, dict(fusion, wait_ms=50)) is scored


def test_missing_vectors_answer_lexical_and_start_one_build(fusion, monkeypatch):
    R._reset_for_tests()
    calls = []
    monkeypatch.setattr(R, "_build_vectors", lambda skills: calls.append(len(skills)))
    scored = [(5.0, ALPHA), (4.5, BETA)]
    assert R.rerank("beta", scored, SKILLS, fusion) is scored
    assert R.rerank("beta", scored, SKILLS, fusion) is scored
    R._executor.submit(lambda: None).result(5)
    assert calls == [3], "the build did not start exactly once"


def test_an_edited_description_invalidates_its_vector(fusion):
    edited = dict(BETA, description="rewritten")
    assert R._vectors_for([ALPHA, edited, GAMMA]) is None


def test_the_prefetch_leg_skips_worker_sessions_and_applies_to_chat(fusion, monkeypatch):
    monkeypatch.setattr(prefetch, "_get_skills_cached", lambda: SKILLS)
    monkeypatch.setattr(prefetch, "_search_skills",
                        lambda toks: [(5.0, ALPHA), (4.5, BETA), (3.2, GAMMA)])
    chat = prefetch._search_skills_for_turn(set(), "gamma", "20261109_101500")
    worker = prefetch._search_skills_for_turn(set(), "gamma", "20261109_101500_autotriage_0fa8")
    assert chat[0][1]["name"] == "gamma"
    assert worker[0][1]["name"] == "alpha"


def test_a_failed_build_backs_off_instead_of_retrying_every_turn(fusion, monkeypatch, tmp_path):
    R._reset_for_tests()
    monkeypatch.setattr(paths, "SKILL_EMBED_CACHE_PATH", tmp_path / "empty.json")

    def no_model(text):
        raise FileNotFoundError("Qwen/Qwen3-Embedding-0.6B is not in the hub cache")

    monkeypatch.setattr(R, "_embed", no_model)
    R._building.set()
    R._build_vectors(SKILLS)
    assert not R._building.is_set()
    started = []
    monkeypatch.setattr(R, "_build_vectors", lambda skills: started.append(1))
    assert R._vectors_for(SKILLS) is None
    R._executor.submit(lambda: None).result(5)
    assert started == [], "a build was retried inside the back-off"
