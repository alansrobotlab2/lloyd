"""#1490: Skill2Query pseudo-queries per skill, inside the lexical scorer.

Pins: off is byte-for-byte the old scorer; a pseudo-query hit is a metadata
hit at `weight` and never double-counts a name/description/tag token; a
cache generated from another version of the skill is ignored; the generator's
prompt is built from the skill alone (the leakage rule); and prefetch reads
the index once per turn.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

import prefetch
from agent_mcp import skills as S

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "scripts" / "skill_pseudo_queries.py"


def _skill(name="poisoned-worker-troubleshoot", desc="Diagnose poisoned worker queue items",
           tags=("workers",), body="Look at workers.db and requeue items."):
    return {"name": name, "description": desc, "tags": list(tags), "body": body}


def _index(tmp_path, skill, queries, *, weight=2.0, mode="union", hash_=None):
    path = tmp_path / "pq.json"
    path.write_text(json.dumps({"prompt_version": 1, "skills": {
        skill["name"]: {"hash": hash_ or S.skill_text_hash(skill), "queries": queries}}}))
    key, table = S._load_pseudo_queries(path)
    return {"weight": weight, "mode": mode, "path": path, "key": key, "table": table}


def test_off_is_the_old_scorer(monkeypatch):
    monkeypatch.setattr(S, "pseudo_query_settings", lambda: None)
    sk = _skill()
    q = S._query_tokens("clear the poisoned workers please")
    old = 3.0 * len(q & S._skill_token_sets(sk)[0]) + 2.0 * len(q & S._skill_token_sets(sk)[1]) \
        + 1.5 * len(q & S._skill_token_sets(sk)[2]) \
        + 0.3 * min(len(q & S._skill_token_sets(sk)[3]), S._BODY_HITS_CAP)
    assert S._score_skill(sk, q) == pytest.approx(old)
    assert S._score_skill(sk, q, pseudo_queries=None) == pytest.approx(old)


def test_settings_default_off_and_sanitised(monkeypatch):
    from app import config
    monkeypatch.setitem(config.CONFIG, "skills", {"directories": []})
    assert S.pseudo_query_settings() is None
    monkeypatch.setitem(config.CONFIG, "skills",
                        {"pseudo_queries": {"enabled": True, "weight": "x", "mode": "nope",
                                            "path": "/nonexistent/pq.json"}})
    got = S.pseudo_query_settings()
    assert got["weight"] == 3.0 and got["mode"] == "union"
    assert got["path"] == Path("/nonexistent/pq.json")
    # "enabled" must be literally true: a truthy string is not a switch.
    monkeypatch.setitem(config.CONFIG, "skills", {"pseudo_queries": {"enabled": "yes"}})
    assert S.pseudo_query_settings() is None


def test_a_pseudo_query_hit_is_a_metadata_hit_at_weight(tmp_path):
    sk = _skill(desc="Diagnose queue items", tags=())
    q = S._query_tokens("my background jobs keep failing, fix them")
    assert S._score_skill(sk, q, pseudo_queries=None) == 0.0     # no metadata hit
    idx = _index(tmp_path, sk, ["my background jobs keep failing"], weight=2.0)
    hits = len(q & (S._query_tokens("my background jobs keep failing")))
    assert hits >= 2
    body = 0.3 * min(len(q & S._skill_token_sets(sk)[3]), S._BODY_HITS_CAP)
    assert S._score_skill(sk, q, pseudo_queries=idx) == pytest.approx(2.0 * hits + body)


def test_metadata_tokens_are_not_counted_twice(tmp_path):
    sk = _skill()
    q = S._query_tokens("poisoned worker")
    base = S._score_skill(sk, q, pseudo_queries=None)
    idx = _index(tmp_path, sk, ["clear the poisoned worker"], weight=3.0)
    assert S._score_skill(sk, q, pseudo_queries=idx) == pytest.approx(base)


def test_union_and_max_modes(tmp_path):
    sk = _skill(desc="x", tags=())
    q = S._query_tokens("backlog triage overnight")
    qs = ["show backlog", "triage overnight"]
    union = _index(tmp_path, sk, qs, weight=1.0, mode="union")
    assert S._pseudo_query_hits(sk, q, union) == 3
    sk2 = _skill(desc="x", tags=())
    mx = _index(tmp_path, sk2, qs, weight=1.0, mode="max")
    assert S._pseudo_query_hits(sk2, q, mx) == 2


def test_a_stale_cache_entry_is_ignored(tmp_path):
    sk = _skill(desc="x", tags=())
    q = S._query_tokens("background jobs failing")
    idx = _index(tmp_path, sk, ["background jobs failing"], hash_="0" * 40)
    assert S._pseudo_query_hits(sk, q, idx) == 0


def test_a_missing_file_gains_nothing(tmp_path):
    key, table = S._load_pseudo_queries(tmp_path / "absent.json")
    assert table == {}
    sk = _skill(desc="x", tags=())
    idx = {"weight": 3.0, "mode": "union", "path": tmp_path / "absent.json",
           "key": key, "table": table}
    assert S._pseudo_query_hits(sk, S._query_tokens("anything at all"), idx) == 0


def test_a_regenerated_file_replaces_the_memo(tmp_path):
    sk = _skill(desc="x", tags=())
    q = S._query_tokens("background jobs")
    idx = _index(tmp_path, sk, ["unrelated words"])
    assert S._pseudo_query_hits(sk, q, idx) == 0
    import os
    import time
    time.sleep(0.01)
    idx2 = _index(tmp_path, sk, ["background jobs"])
    os.utime(idx2["path"], None)
    idx2["key"], idx2["table"] = S._load_pseudo_queries(idx2["path"])
    assert S._pseudo_query_hits(sk, q, idx2) == 2


def test_prefetch_reads_the_index_once_per_turn(monkeypatch):
    calls = []
    monkeypatch.setattr(prefetch, "pseudo_query_index", lambda: calls.append(1))
    skills = [_skill(name=f"s{i}") for i in range(5)]
    prefetch._search_skills(S._query_tokens("poisoned workers"), skills=skills)
    assert calls == [1]


def _gen():
    spec = importlib.util.spec_from_file_location("skill_pseudo_queries", GEN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_generator_sees_the_skill_and_nothing_else():
    gen = _gen()
    sk = _skill(body="BODY-MARKER " * 10)
    prompt = gen.build_prompt(sk)
    assert sk["name"] in prompt and sk["description"] in prompt and "BODY-MARKER" in prompt
    # The leakage rule: the eval's labelled turns never reach the generator.
    src = GEN.read_text()
    for forbidden in ("skill_match_queries", "SESSIONS_DIR", "sessions/"):
        assert forbidden not in src
    assert gen.skill_hash(sk) == S.skill_text_hash(sk)


def test_parse_queries_strips_list_markers():
    gen = _gen()
    text = '1. clear the poisoned workers\n- "why are jobs stuck?"\n\nHere:\n* requeue them'
    assert gen.parse_queries(text) == ["clear the poisoned workers", "why are jobs stuck?",
                                       "requeue them"]


def test_the_eval_arm_restores_the_production_index(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "run_skill_match_eval", ROOT / "eval" / "run_skill_match_eval.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    before = prefetch.pseudo_query_index
    arm = ev.pseudo_query_arm(tmp_path / "absent.json", 2.0)
    arm("hello there")
    assert prefetch.pseudo_query_index is before


def test_hit_at_1_and_mrr_on_a_hand_computed_fixture():
    spec = importlib.util.spec_from_file_location(
        "run_skill_match_eval", ROOT / "eval" / "run_skill_match_eval.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    recs = [{"id": "a", "turn": "a", "expected_skills": ["x"]},
            {"id": "b", "turn": "b", "expected_skills": ["y"]},
            {"id": "c", "turn": "c", "expected_skills": ["z"]}]
    got = {"a": ["x", "q"], "b": ["q", "y"], "c": []}
    out = ev.score(recs, lambda t: got[t])
    assert out["hit@1"] == pytest.approx(0.3333)
    assert out["mrr@5"] == pytest.approx(0.5)             # (1 + 1/2 + 0) / 3
    same = ev.paired(out, out, "hit1")
    assert same["diff"] == 0 and same["n"] == 3
