"""The #600 planted-fact runner: generator, grading order, arms, gate.

Everything here is offline — the engine is a scripted `stream_chat`, the
tool pool is the runner's own stub, `/metrics` is patched out.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "run_compaction_recall_eval", ROOT / "eval" / "run_compaction_recall_eval.py")
R = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = R
_spec.loader.exec_module(R)


@pytest.fixture
def corpus(tmp_path):
    rng = random.Random(1)
    root = tmp_path / "tree"
    (root / "app").mkdir(parents=True)
    files = []
    for i in range(6):
        p = root / "app" / f"mod{i}.py"
        p.write_text("\n".join(
            f"def handler_{i}_{n}(config_value):  # timeout {rng.randint(1, 99)}"
            for n in range(400)))
        files.append(p)
    return root, files


def _planted():
    return R.Planted(passphrase="QUARTZ-HERON-2291", port="7419", old_port="7914",
                     distractor_ports={"billing-west": "7491", "search-east": "7149"})


# --- generator -------------------------------------------------------------

def test_session_plants_both_facts_in_one_tool_result_at_its_depth(corpus):
    root, files = corpus
    s = R.build_session(7, 20_000, 0.5, root=root, corpus=files)
    planted = s.messages[s.planted_index]
    assert planted["role"] == "tool" and planted["tool_call_id"] == "call_planted"
    text = planted["content"][0]["text"]
    assert s.planted.passphrase in text
    assert f"listens on port {s.planted.port} now" in text
    assert s.planted.old_port in text            # the ambiguous half
    frac = s.planted_index / len(s.messages)
    assert 0.3 < frac < 0.7
    assert s.est_tokens >= 20_000
    # filler is tool call/result pairs, and the probe asks for both facts
    assert sum(1 for m in s.messages if m["role"] == "tool") > 5
    assert "CODENAME:" in s.probe and "PORT:" in s.probe


def test_session_is_deterministic_and_distractors_share_the_digits(corpus):
    root, files = corpus
    a = R.build_session(11, 15_000, 0.1, root=root, corpus=files)
    b = R.build_session(11, 15_000, 0.1, root=root, corpus=files)
    assert a.sha256 == b.sha256
    ports = a.planted.all_ports()
    assert len(ports) >= 4
    assert len({"".join(sorted(p)) for p in ports}) == 1


# --- grading: code decides before any judge --------------------------------

def test_labelled_lines_are_graded_by_code():
    p = _planted()
    v = R.grade_answer("blah\nCODENAME: QUARTZ-HERON-2291\nPORT: 7419", p)
    assert v == {"distinctive": "hit", "ambiguous": "hit"}
    v = R.grade_answer("CODENAME: COBALT-WREN-1111\nPORT: 7914", p)
    assert v == {"distinctive": "wrong", "ambiguous": "wrong"}   # old port is wrong


def test_unlabelled_answer_falls_back_to_the_whole_text():
    p = _planted()
    v = R.grade_answer("The codename is quartz-heron-2291 and it moved to 7491.", p)
    assert v == {"distinctive": "hit", "ambiguous": "wrong"}
    v = R.grade_answer("I can't recall either of those.", p)
    assert v == {"distinctive": "undecided", "ambiguous": "undecided"}


def test_the_judge_sees_only_what_the_code_left_undecided():
    p = _planted()
    seen = []

    def judge(fact, text, planted):
        seen.append(fact)
        return "hit"

    out = R.decide("CODENAME: QUARTZ-HERON-2291\nPORT: 7149", p, judge)
    assert seen == []                            # code decided both
    assert out["final"] == {"distinctive": "hit", "ambiguous": "wrong"}

    out = R.decide("CODENAME: QUARTZ-HERON-2291\nno idea on the port", p, judge)
    assert seen == ["ambiguous"]
    assert out["code"]["ambiguous"] == "undecided"
    assert out["final"]["ambiguous"] == "hit" and out["judged"] == ["ambiguous"]


# --- the gate reads the #1078 record ----------------------------------------

def test_each_arm_is_valid_only_when_its_own_feature_fired():
    none_rec = {"mechanisms": [], "turn_start": {"tokens_freed": 0, "mechanisms": []}}
    fired_rec = {"mechanisms": ["relief:intra_turn"],
                 "turn_start": {"tokens_freed": 0, "mechanisms": []},
                 "relief": [{"reason": "intra_turn", "rungs": ["tool_results"],
                             "freed_tokens": 5000}]}
    ran_nothing = {"mechanisms": ["relief:intra_turn"],
                   "relief": [{"reason": "intra_turn", "rungs": ["tool_results"],
                               "freed_tokens": 0}]}
    assert R.valid_for_arm("none", R.fired(none_rec))
    assert not R.valid_for_arm("none", R.fired(fired_rec))
    assert R.valid_for_arm("production", R.fired(fired_rec))
    # a ladder pass that freed nothing did not exercise the feature
    assert not R.valid_for_arm("production", R.fired(ran_nothing))
    assert not R.valid_for_arm("tool_clear", R.fired(None))


def test_overlay_swaps_and_restores_the_compaction_block():
    from app.config import CONFIG
    before = json.dumps(CONFIG.get("compaction"), sort_keys=True, default=str)
    with R.compaction_overlay(R.ARMS["none"]["compaction"]) as merged:
        assert CONFIG["compaction"]["microcompact"]["enabled"] is False
        assert CONFIG["compaction"]["mode"] == "truncate"
        assert merged["microcompact"]["enabled"] is False
    assert json.dumps(CONFIG.get("compaction"), sort_keys=True, default=str) == before


# --- the stub pool -----------------------------------------------------------

def test_pool_serves_the_session_files_and_notes_recovery(tmp_path):
    p = _planted()
    pool = R.EvalPool([], {"notes/deploy.md": f"codename `{p.passphrase}`\n"
                                              f"billing-east relay listens on port {p.port} now"},
                      tmp_path, tree_root=tmp_path, planted=p)
    out = asyncio.run(pool.call_tool("Read", {"file_path": "notes/deploy.md"}))
    assert not out["is_error"] and p.passphrase in out["content"]
    assert pool.recovered == {"distinctive", "ambiguous"}
    out = asyncio.run(pool.call_tool("Bash", {"command": "rm -rf /"}))
    assert out["is_error"]
    out = asyncio.run(pool.call_tool("Read", {"file_path": "/etc/passwd"}))
    assert out["is_error"]                       # nothing outside its roots


# --- summary never blends the two facts -------------------------------------

def test_summary_reports_two_columns_and_counts_drops():
    rows = [
        {"arm": "none", "depth": 0.1, "status": "ok", "tool_calls": 1,
         "verdict": {"distinctive": "hit", "ambiguous": "wrong"}},
        {"arm": "none", "depth": 0.5, "status": "ok", "tool_calls": 1,
         "verdict": {"distinctive": "hit", "ambiguous": "hit"}},
        {"arm": "production", "depth": 0.1, "status": "dropped", "tool_calls": 0},
    ]
    s = R.summarize(rows)
    assert s["none"]["distinctive"]["k"] == 2 and s["none"]["ambiguous"]["k"] == 1
    assert s["production"]["dropped"] == 1 and s["production"]["kept"] == 0
    for arm in s.values():
        assert "recall" not in arm               # no blended number anywhere


# --- one row per (arm, session), end to end with a scripted engine -----------

def test_run_one_goes_through_compaction_and_the_loop(corpus, tmp_path, monkeypatch):
    root, files = corpus
    s = R.build_session(3, 50_000, 0.5, root=root, corpus=files)
    monkeypatch.setattr(R, "_metrics", lambda base_url: {})
    calls = {"n": 0}

    async def fake_stream(**kw):
        calls["n"] += 1
        eb = kw.get("extra_body") or {}
        if eb.get("max_tokens") == 1:             # the warm-up
            yield {"choices": [], "usage": {"prompt_tokens": 100}}
            return
        if kw.get("iteration") == 1:
            yield {"choices": [{"delta": {"tool_calls": [{
                "index": 0, "id": "t1", "type": "function",
                "function": {"name": "Grep", "arguments": json.dumps(
                    {"pattern": "open items", "path": "notes/"})}}]}}]}
            yield {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}
        else:
            yield {"choices": [{"delta": {"content":
                f"CODENAME: {s.planted.passphrase}\nPORT: {s.planted.port}"}}]}
            yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}
        yield {"choices": [], "usage": {"prompt_tokens": 1000, "completion_tokens": 5,
                                        "prompt_tokens_details": {"cached_tokens": 900}}}

    monkeypatch.setattr("app.harness.loop.stream_chat", fake_stream)
    disc = [("lloyd-mcp", [{"name": n, "description": n,
                            "inputSchema": {"type": "object", "properties": {}}}
                           for n in ("Read", "Grep", "Bash")])]
    rows = []
    for arm in ("none", "tool_clear"):
        rows.append(asyncio.run(R.run_one(
            s, arm, discovered=disc, system_prompt="sys", data_root=tmp_path / "d",
            base_url="http://stub", max_turns=4)))
    # production cannot reach either trigger at 50k and is dropped unspent
    prod = asyncio.run(R.run_one(s, "production", discovered=disc, system_prompt="sys",
                                 data_root=tmp_path / "d", base_url="http://stub"))
    assert prod["status"] == "dropped" and prod["reason"] == "cannot_fire"
    none, clear = rows
    assert none["status"] == "ok" and none["verdict"] == {"distinctive": "hit", "ambiguous": "hit"}
    assert none["session_sha256"] == clear["session_sha256"] == prod["session_sha256"]
    assert clear["fired"]["turn_start_freed"] > 0 and clear["valid"]
    assert none["tool_calls"] == 1 and none["warmup"]["warmup"] is True
    assert none["run_cache_hit"] == pytest.approx(0.9)


def test_the_late_probe_hides_the_question_in_a_tool_result(corpus):
    root, files = corpus
    early = R.build_session(5, 15_000, 0.5, root=root, corpus=files, probe="early")
    late = R.build_session(5, 15_000, 0.5, root=root, corpus=files, probe="late")
    assert early.sha256 == late.sha256            # same history either way
    assert "CODENAME:" in early.probe
    assert "codename" not in late.probe.lower()   # the user turn does not ask it
    item = next(v for k, v in late.files.items() if "open-items" in k)
    assert "CODENAME:" in item and "billing-east" in item


def test_the_summary_format_arms_are_valid_only_when_the_summary_replaced_a_block():
    """D2's eval gate: `summary_legacy` vs `summary_persisted` compare the two
    summary formats, so a run whose summarize layer did not replace a block
    (it only truncated) measured neither and is dropped."""
    summarized = {"mechanisms": ["summarize"],
                  "turn_start": {"tokens_freed": 90_000, "mechanisms": ["summarize"]}}
    truncated = {"mechanisms": ["truncate"],
                 "turn_start": {"tokens_freed": 90_000, "mechanisms": ["truncate"]}}
    for arm in ("summary_legacy", "summary_persisted"):
        assert R.valid_for_arm(arm, R.fired(summarized))
        assert not R.valid_for_arm(arm, R.fired(truncated))
    assert R.ARMS["summary_persisted"]["compaction"]["persist_summary"] is True
    assert R.ARMS["summary_legacy"]["compaction"]["persist_summary"] is False
