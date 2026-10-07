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


# --- D2 follow-up: the summary arms have to measure a summary ----------------

def test_the_conversation_shape_restates_the_facts_and_budgets_what_clearing_leaves(corpus):
    """The planted Read result is cleared at turn start like every other one,
    so in this shape the facts ride in the assistant's reply — the part a
    summary has to carry — and `target_tokens` counts only non-tool rows."""
    from app.compaction import estimate_conversation_tokens
    root, files = corpus
    s = R.build_session(9, 20_000, 0.3, root=root, corpus=files, probe="late",
                        shape="conversation")
    assert s.key.endswith("-conv") and s.meta["shape"] == "conversation"
    reply = s.messages[s.planted_index + 1]
    assert reply["role"] == "assistant"
    text = reply["content"][0]["text"]
    assert s.planted.passphrase in text
    assert f"listens on port {s.planted.port} now" in text and s.planted.old_port in text
    non_tool = estimate_conversation_tokens(
        [m for m in s.messages if m["role"] != "tool"])
    assert non_tool >= 20_000 and s.est_tokens > non_tool
    # every filler turn is one Read plus a discussion
    assert sum(1 for m in s.messages if m["role"] == "tool") == \
        sum(1 for m in s.messages if m["role"] == "user")
    # the #600 shape is untouched: its planted reply names no fact
    t = R.build_session(9, 20_000, 0.3, root=root, corpus=files)
    assert s.planted.passphrase not in t.messages[t.planted_index + 1]["content"][0]["text"]
    assert not t.key.endswith("-conv")


def test_auto_shape_is_conversation_only_for_an_all_summary_arm_list():
    assert R.resolve_shape("auto", ["summary_legacy", "summary_persisted"]) == "conversation"
    assert R.resolve_shape("auto", ["production", "summary_legacy"]) == "tool"
    assert R.resolve_shape("auto", ["none"]) == "tool"
    assert R.resolve_shape("tool", ["summary_legacy"]) == "tool"


def _turns(n: int, chars: int) -> list[dict]:
    out = []
    for i in range(n):
        out += [R._msg("user", f"q{i}"),
                {"role": "assistant", "content": [{"type": "text", "text": ""}],
                 "tool_calls": [R._tc(f"c{i}", "Read", {"file_path": "x"})]},
                R._msg("tool", "x" * chars, tool_call_id=f"c{i}"),
                R._msg("assistant", f"a{i}")]
    return out


def test_a_warmup_that_fits_is_sent_whole():
    hist = _turns(3, 300)
    msgs, info = R.fit_warmup([{"role": "system", "content": "s"}], hist, [], 100_000)
    assert info["status"] == "sent" and msgs[1:] == hist


def test_an_oversized_warmup_is_clipped_to_a_head_of_whole_turns():
    hist = _turns(10, 30_000)                     # ~10k tokens a turn by the bound
    window = 60_000
    msgs, info = R.fit_warmup([{"role": "system", "content": "s"}], hist, [], window)
    assert info["status"] == "clipped"
    kept = msgs[1:]
    assert kept == hist[:len(kept)]               # a HEAD: the prefix a cache is made of
    assert 0 < len(kept) < len(hist) and len(kept) % 4 == 0   # whole turns only
    assert hist[len(kept)]["role"] == "user"
    assert sum(R._json_tokens(m) for m in kept) <= window - R.WARMUP_RESERVE_TOKENS


def test_a_warmup_with_no_whole_turn_that_fits_is_skipped():
    msgs, info = R.fit_warmup([], _turns(2, 400_000), [], 60_000)
    assert msgs is None and info["status"] == "skipped"


def _drain(gen_fn, **kw):
    async def go():
        return [c async for c in gen_fn(**kw)]
    return asyncio.run(go())


def test_the_warmup_is_attempted_once_and_its_failure_never_reaches_the_probe():
    """The first live summary run: an oversized warm-up 400ed inside the
    loop's stream, the loop's overflow recovery retried iteration 1, and the
    warm-up — gated on an empty log — was re-sent on every attempt."""
    sent = []

    async def real(**kw):
        sent.append((kw.get("extra_body") or {}).get("max_tokens"))
        if (kw.get("extra_body") or {}).get("max_tokens") == 1:
            raise RuntimeError("HTTP 400: context overflow")
        yield {"choices": [{"delta": {"content": "ok"}}]}

    log: list = []
    done = []
    w = R._make_stream_wrapper(real, _turns(2, 100), log, lambda: done.append(1),
                               window=100_000)
    base = {"messages": [{"role": "system", "content": "s"}], "tools": [],
            "extra_body": {}}
    assert _drain(w, iteration=1, **base)          # the probe still streams
    assert _drain(w, iteration=1, **base)          # a retried iteration 1
    assert sent == [1, None, None]                 # one warm-up, ever
    warm = [x for x in log if x.get("warmup")]
    assert len(warm) == 1 and warm[0]["status"] == "error"
    assert "context overflow" in warm[0]["error"] and done == [1]


def test_a_warmup_that_cannot_fit_is_never_sent():
    sent = []

    async def real(**kw):
        sent.append((kw.get("extra_body") or {}).get("max_tokens"))
        yield {"choices": [{"delta": {"content": "ok"}}]}

    log: list = []
    w = R._make_stream_wrapper(real, _turns(2, 400_000), log, lambda: None,
                               window=60_000)
    _drain(w, iteration=1, messages=[], tools=[], extra_body={})
    assert sent == [None]
    assert log[0]["warmup"] and log[0]["status"] == "skipped"


def test_a_summary_arm_is_valid_only_for_a_summary_made_this_turn():
    made = {"turn_start": {"tokens_freed": 90_000, "mechanisms": ["summarize"],
                           "summarize_outcome": "summarized"}}
    reused = {"turn_start": {"tokens_freed": 90_000, "mechanisms": ["summarize"],
                             "summarize_outcome": "reused"}}
    fell_back = {"turn_start": {"tokens_freed": 90_000, "mechanisms": ["truncate"],
                                "summarize_outcome": "empty_summary"}}
    for arm in ("summary_legacy", "summary_persisted"):
        assert R.valid_for_arm(arm, R.fired(made))
        assert not R.valid_for_arm(arm, R.fired(reused))
        assert not R.valid_for_arm(arm, R.fired(fell_back))


def test_summary_fidelity_is_read_off_the_summary_text():
    p = _planted()
    hist = [R._msg("assistant", "[compaction summary — earlier conversation]\n\n"
                                f"codename {p.passphrase}; billing-east moved 7914 -> 7419"),
            R._msg("user", "hi")]
    text = R.summary_text(hist)
    assert R.summary_has(text, p) == {"codename": True, "port_now": True, "port_old": True}
    assert not R.fact_verbatim(hist, p)
    assert R.fact_verbatim(hist + [R._msg("assistant", p.passphrase)], p)


def _stream_must_not_run(**_kw):
    raise AssertionError("a dropped or dry row must not reach the engine")


def test_a_summary_arm_that_cannot_fire_is_dropped_before_the_engine(corpus, tmp_path,
                                                                     monkeypatch):
    root, files = corpus
    s = R.build_session(3, 30_000, 0.5, root=root, corpus=files)
    monkeypatch.setattr("app.harness.loop.stream_chat", _stream_must_not_run)
    row = asyncio.run(R.run_one(s, "summary_persisted", discovered=[], system_prompt="sys",
                                data_root=tmp_path / "d", base_url="http://stub"))
    assert row["status"] == "dropped"
    assert row["reason"] == "summary_not_fired:under_threshold"
    assert row["summarizer"]["calls"] == 0


@pytest.mark.parametrize("arm,fn", [("summary_legacy", "summarize_history"),
                                    ("summary_persisted", "summarize_incremental")])
def test_dry_run_shows_the_summary_firing_without_any_engine_traffic(
        corpus, tmp_path, monkeypatch, arm, fn):
    """A small window stands in for the recommended sizes: the conversation
    shape stays over the threshold after microcompaction, the summarize layer
    is reached, and `--dry`'s stub answers in place of the summariser."""
    root, files = corpus
    monkeypatch.setattr("app.compaction.get_context_window", lambda _m="": 100_000)
    monkeypatch.setattr("app.harness.loop.stream_chat", _stream_must_not_run)

    async def no_http(*_a, **_k):
        raise AssertionError("dry run reached the summariser endpoint")
    monkeypatch.setattr("app.compaction_llm._post_chat_completion", no_http)
    s = R.build_session(4, 60_000, 0.1, root=root, corpus=files, probe="late",
                        shape="conversation")
    row = asyncio.run(R.run_one(s, arm, discovered=[], system_prompt="sys",
                                data_root=tmp_path / "d", base_url="http://stub",
                                dry=True))
    assert row["status"] == "dry", row.get("reason")
    assert row["fired"]["summarize_outcome"] == "summarized"
    assert R.summary_fired(row["fired"])
    calls = row["summarizer"]["per_call"]
    assert calls and {c["fn"] for c in calls} == {fn}
    assert row["summary_chars"] > 0 and not row["fact_verbatim_at_start"]
    assert row["warmup"]["status"] in ("sent", "clipped", "skipped")


def test_a_summary_the_truncation_fallback_dropped_is_not_a_measurement(
        corpus, tmp_path, monkeypatch):
    """Summarised, then restored files pushed the history back over the
    threshold and drop-oldest took the summary row with it: nothing of the
    format reaches the probe, so the row is dropped before the engine."""
    root, files = corpus
    monkeypatch.setattr("app.compaction.get_context_window", lambda _m="": 70_000)
    monkeypatch.setattr("app.harness.loop.stream_chat", _stream_must_not_run)
    s = R.build_session(4, 30_000, 0.1, root=root, corpus=files, probe="late",
                        shape="conversation")
    row = asyncio.run(R.run_one(s, "summary_legacy", discovered=[], system_prompt="sys",
                                data_root=tmp_path / "d", base_url="http://stub",
                                dry=True))
    assert row["fired"]["summarize_outcome"] == "summarized"
    assert row["turn_start"]["truncated"]
    assert row["status"] == "dropped" and row["reason"] == "summary_truncated_away"


# --- P3's memory_flush arm on the same footing -------------------------------

def test_the_memory_flush_arm_is_a_summary_arm_on_the_conversation_shape():
    assert R.ARMS["memory_flush"].get("expects_summary")
    assert R.resolve_shape("auto", ["summary_legacy", "memory_flush"]) == "conversation"


def test_the_flush_is_sent_the_head_production_would_have_flushed(corpus, tmp_path,
                                                                  monkeypatch):
    """Not the whole uncompacted session — at the summary sizes that is past
    the window — but the longest head of whole turns whose compacted prompt
    is under the flush trigger; sizing it never calls a summariser."""
    root, files = corpus
    monkeypatch.setattr("app.compaction.get_context_window", lambda _m="": 100_000)

    async def no_http(*_a, **_k):
        raise AssertionError("sizing the flush head reached the summariser")
    monkeypatch.setattr("app.compaction_llm._post_chat_completion", no_http)
    s = R.build_session(4, 60_000, 0.1, root=root, corpus=files, probe="late",
                        shape="conversation")
    (tmp_path / "d" / "sessions").mkdir(parents=True)
    msgs, info = asyncio.run(R.flush_history(
        s, sid="pt-eval-x", data_root=tmp_path / "d", system_prompt="sys",
        discovered=[]))
    assert msgs is not None
    assert 0 < info["history_rows"] < info["rows_total"]
    assert s.messages[info["history_rows"]]["role"] == "user"     # whole turns
    assert info["est_prompt_tokens"] <= info["bound_tokens"]
    assert info["planted_in_history"]                             # depth 0.1 is in it


def test_run_flush_sends_the_head_it_is_given(corpus, tmp_path,
                                                                 monkeypatch):
    root, files = corpus
    s = R.build_session(4, 20_000, 0.1, root=root, corpus=files, shape="conversation")
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"messages": s.messages}))
    seen = {}

    async def fake_stream(**kw):
        seen["n"] = len(kw["messages"])
        yield {"choices": [{"delta": {"content": "Saved nothing."}}]}
        yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}

    monkeypatch.setattr("app.harness.loop.stream_chat", fake_stream)
    head = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    disc = [("lloyd-mcp", [{"name": n, "description": n,
                            "inputSchema": {"type": "object", "properties": {}}}
                           for n in ("memory_read", "memory_add", "fact_get", "fact_add")])]
    out = asyncio.run(R.run_flush(s, sid="pt-eval-y", path=path, discovered=disc,
                                  system_prompt="sys", data_root=tmp_path,
                                  base_url="http://stub", hk={}, messages=head))
    assert not out["error"], out["error"]
    assert seen["n"] == 1 + len(head) + 1          # system + head + flush prompt
    assert json.loads(path.read_text())["compaction"]["flush"]["status"] == "done"


def test_dry_run_sizes_the_flush_and_shows_the_summary_firing(corpus, tmp_path, monkeypatch):
    root, files = corpus
    monkeypatch.setattr("app.compaction.get_context_window", lambda _m="": 100_000)
    monkeypatch.setattr("app.harness.loop.stream_chat", _stream_must_not_run)

    async def no_http(*_a, **_k):
        raise AssertionError("dry run reached the summariser endpoint")
    monkeypatch.setattr("app.compaction_llm._post_chat_completion", no_http)
    s = R.build_session(4, 60_000, 0.1, root=root, corpus=files, probe="late",
                        shape="conversation")
    row = asyncio.run(R.run_one(s, "memory_flush", discovered=[], system_prompt="sys",
                                data_root=tmp_path / "d", base_url="http://stub",
                                dry=True))
    assert row["status"] == "dry", row.get("reason")
    assert row["fired"]["summarize_outcome"] == "summarized"
    assert 0 < row["flush"]["history_rows"] < row["flush"]["rows_total"]


# --- #1514 / #1481: the self-record and observation arms ----------------------

def _two_sessions(tmp_path):
    sess = tmp_path / "sessions"
    (sess / "mine.tool-results").mkdir(parents=True)
    (sess / "mine.json").write_text(json.dumps(
        {"messages": [{"role": "tool", "content": "billing-east listens on port 7185 now"}]},
        indent=2))
    (sess / "mine.tool-results" / "call_1.txt").write_text("codename `MINE-1`")
    (sess / "other.json").write_text(json.dumps(
        {"messages": [{"role": "tool", "content": "billing-east listens on port 7999 now"}]},
        indent=2))
    return sess


def test_the_pool_sees_only_its_own_session_record(tmp_path):
    """The leak this fixes: every arm's and seed's record shares the scratch
    root, each planting a different port, and a Grep used to see them all."""
    sess = _two_sessions(tmp_path)
    pool = R.EvalPool([], {}, tmp_path, tree_root=tmp_path / "tree", session_id="mine")
    out = asyncio.run(pool.call_tool("Grep", {"pattern": "billing-east"}))
    assert "7185" in out["content"] and "7999" not in out["content"]
    assert asyncio.run(pool.call_tool("Read", {"file_path": str(sess / "other.json")}))["is_error"]
    assert not asyncio.run(pool.call_tool("Read", {"file_path": str(sess / "mine.json")}))["is_error"]
    # `path` is honoured: a directory, a file, and one outside the roots.
    out = asyncio.run(pool.call_tool("Grep", {"pattern": "codename",
                                              "path": str(sess / "mine.tool-results")}))
    assert "MINE-1" in out["content"]
    out = asyncio.run(pool.call_tool("Grep", {"pattern": "billing", "path": str(sess)}))
    assert "7185" in out["content"] and "7999" not in out["content"]
    out = asyncio.run(pool.call_tool("Grep", {"pattern": "root", "path": "/etc"}))
    assert out["content"] == "No matches found"
    # Without a session id the pool behaves as it always did.
    legacy = R.EvalPool([], {}, tmp_path, tree_root=tmp_path / "tree")
    out = asyncio.run(legacy.call_tool("Grep", {"pattern": "billing-east"}))
    assert "7185" in out["content"] and "7999" in out["content"]


def test_the_pool_serves_recall_observation_and_names_the_route(tmp_path, monkeypatch):
    monkeypatch.setattr("app.harness.tool_result_spill.SESSIONS_DIR", tmp_path / "sessions")
    _two_sessions(tmp_path)
    p = _planted()
    (tmp_path / "sessions" / "mine.tool-results" / "call_planted.txt").write_text(
        f"codename `{p.passphrase}`")
    pool = R.EvalPool([], {}, tmp_path, tree_root=tmp_path / "tree", planted=p,
                      session_id="mine")
    out = asyncio.run(pool.call_tool("recall_observation", {"id": "call_planted"}))
    assert not out["is_error"] and p.passphrase in out["content"]
    assert pool.recovered_via == {"recall_observation": {"distinctive"}}
    assert asyncio.run(pool.call_tool("recall_observation", {"id": "nope"}))["is_error"]
    assert pool._route("Grep", {"path": str(tmp_path / "sessions" / "mine.json")}) \
        == "session_record"
    assert pool._route("Read", {"file_path": "x/mine.tool-results/call_1.txt"}) == "spill_file"


def test_with_recall_observation_adds_the_schema_once():
    disc = [["lloyd-mcp", [{"name": "Read"}]]]
    out = R.with_recall_observation(disc)
    names = [t["name"] for t in out[0][1]]
    assert names == ["Read", "recall_observation"]
    assert R.with_recall_observation(out)[0][1] == out[0][1]


def test_the_new_arms_reach_the_wire(corpus, tmp_path, monkeypatch):
    """Through run_one with a scripted engine: `observation` advertises the tool
    and puts a stub on the wire; `self_record` names the session record;
    `tool_clear` does neither — so the arms differ in exactly their switch."""
    root, files = corpus
    s = R.build_session(3, 50_000, 0.5, root=root, corpus=files)
    monkeypatch.setattr(R, "_metrics", lambda base_url: {})
    seen: dict[str, dict] = {}

    def fake_for(arm):
        async def fake_stream(**kw):
            eb = kw.get("extra_body") or {}
            if eb.get("max_tokens") == 1:
                yield {"choices": [], "usage": {"prompt_tokens": 100}}
                return
            wire = json.dumps(kw.get("messages"))
            seen[arm] = {"tools": [t["function"]["name"] for t in kw.get("tools") or []],
                         "stub": "[observation call_" in wire,
                         "record": f"pt-eval-c600-{s.key}-{arm}.json" in wire}
            yield {"choices": [{"delta": {"content": "CODENAME: x\nPORT: 1"}}]}
            yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}
            yield {"choices": [], "usage": {"prompt_tokens": 1000, "completion_tokens": 5}}
        return fake_stream

    disc = [("lloyd-mcp", [{"name": n, "description": n,
                            "inputSchema": {"type": "object", "properties": {}}}
                           for n in ("Read", "Grep", "Bash")])]
    monkeypatch.setenv("LLOYD_DATA", str(tmp_path / "d"))
    monkeypatch.setattr("app.harness.tool_result_spill.SESSIONS_DIR",
                        tmp_path / "d" / "sessions")
    for arm in ("tool_clear", "self_record", "observation"):
        monkeypatch.setattr("app.harness.loop.stream_chat", fake_for(arm))
        row = asyncio.run(R.run_one(s, arm, discovered=disc, system_prompt="sys",
                                    data_root=tmp_path / "d", base_url="http://stub",
                                    max_turns=3))
        assert row["fired"]["turn_start_freed"] > 0, row.get("reason")
    assert "recall_observation" not in seen["tool_clear"]["tools"]
    assert "recall_observation" not in seen["self_record"]["tools"]
    assert "recall_observation" in seen["observation"]["tools"]
    assert seen["observation"]["stub"] and not seen["tool_clear"]["stub"]
    assert seen["self_record"]["record"] and not seen["tool_clear"]["record"]


# --- #1499: the rung-4 arms ----------------------------------------------------

def test_a_rung4_arm_is_valid_only_when_rung_4_fired():
    f = {"turn_start_freed": 0, "relief_freed": 900, "relief_rungs": ["reasoning"]}
    assert not R.valid_for_arm("rung4", f)
    f["relief_rungs"] = ["reasoning", "truncate"]
    assert R.valid_for_arm("rung4", f) and R.valid_for_arm("rung4_lossy", f)
    rec = {"relief": [{"rungs": ["reasoning:2", "truncate:5"], "freed_tokens": 9,
                       "truncated_chars_freed": 40}]}
    assert R.fired(rec)["relief_rungs"] == ["reasoning", "truncate"]
    assert R.fired(rec)["truncated_chars_freed"] == 40
    # Rung 1 is held off by naming every tool the probe uses.
    assert set(R.ARMS["rung4"]["options"]["intra_turn_microcompact_non_compactable"]) \
        >= {"Read", "Grep", "Bash"}
    assert R.ARMS["rung4_lossy"].get("rung4_lossy") is True


def test_the_relief_event_carries_what_rungs_3_and_4_cut(monkeypatch):
    from app.harness import loop as L
    seen = {}
    monkeypatch.setattr(L, "_log_harness_event",
                        lambda sid, name, data, **kw: seen.update(data))

    class _O:
        session_id = "s"
        turn_id = "t"
    L._record_relief_pass(_O(), {"reason": "intra_turn", "rungs": ["truncate:3"],
                                 "freed_tokens": 10, "truncated_chars_freed": 4000,
                                 "argument_chars_freed": 12})
    assert seen["truncated_chars_freed"] == 4000 and seen["argument_chars_freed"] == 12


# --- #2028: oracle spans — which turns the answer depended on -----------------
#
# ContextBench (arXiv:2609.37725v1) grades a stateful task against an ORACLE
# TRACE: the exact set of turns that must stay in context for the task to be
# solvable. Without it a policy that sheds the span and keeps the filler reads
# as a pass whenever the model answers plausibly anyway. These tests pin the
# three things the item asks for: the oracle on the spec, retention beside
# correctness (with an honest zero denominator), and the pointer-after-content
# probe scored per turn rather than as one blended number.

def _oracle_turn_ids(s) -> set[str]:
    return {t.turn_id for t in s.oracle_turns}


def test_the_planted_turn_is_the_oracle_span_the_answer_needs(corpus):
    """Clause 1: the spec carries `oracle_turn_ids`, and it is the planted turn
    the generator already positioned at `depth` — the id is the row's own
    `tool_call_id`, which survives both compaction and
    `_prepare_messages_for_harness`, and the needle is the literal only that
    turn's content carries."""
    root, files = corpus
    s = R.build_session(7, 20_000, 0.5, root=root, corpus=files)
    assert s.oracle_turn_ids == [t.turn_id for t in s.oracle_turns]
    assert _oracle_turn_ids(s) == {R.PLANTED_CALL_ID}
    turn = s.oracle_turns[0]
    assert turn.role == "target"
    assert turn.needle == s.planted.passphrase
    # The oracle is the planted row and nothing else: the depth that placed it
    # is the depth the oracle names.
    assert s.messages[s.planted_index]["tool_call_id"] == turn.turn_id
    assert turn.needle in s.messages[s.planted_index]["content"][0]["text"]
    # A conversation-shape session keeps the same one oracle turn: the facts
    # are restated in the reply, but the reply is not a second source.
    c = R.build_session(7, 20_000, 0.5, root=root, corpus=files, shape="conversation")
    assert c.oracle_turn_ids == [R.PLANTED_CALL_ID]


def test_naming_the_oracle_draws_nothing_and_moves_no_turn(corpus):
    """Clause 1's other half: the oracle is free at generation time. A pointer
    session's prefix — planted facts, filler, every tool call id — is the
    non-pointer session's, byte for byte, so adding the probe did not consume a
    draw that would have shifted the planted values or the distractor ports."""
    root, files = corpus
    plain = R.build_session(21, 20_000, 0.5, root=root, corpus=files)
    again = R.build_session(21, 20_000, 0.5, root=root, corpus=files)
    assert plain.oracle_turn_ids == again.oracle_turn_ids == [R.PLANTED_CALL_ID]
    assert plain.sha256 == again.sha256
    ptr = R.build_session(21, 20_000, 0.5, root=root, corpus=files, pointer=True)
    assert ptr.messages[:len(plain.messages)] == plain.messages
    assert ptr.planted == plain.planted
    assert ptr.planted_index == plain.planted_index


def test_a_pointer_turn_points_at_the_span_it_does_not_restate(corpus):
    """The probe the item asks for: plant, clear, and only THEN emit "the value
    you'll need is the one in the result you dropped". The pointer is the last
    turn of the session and carries neither value, so the only way the answer
    survives is the pointer's TARGET — or a re-read."""
    root, files = corpus
    s = R.build_session(7, 20_000, 0.5, root=root, corpus=files, pointer=True)
    assert _oracle_turn_ids(s) == {R.PLANTED_CALL_ID, R.POINTER_CALL_ID}
    roles = {t.turn_id: t.role for t in s.oracle_turns}
    assert roles == {R.PLANTED_CALL_ID: "target", R.POINTER_CALL_ID: "pointer"}
    at = {m.get("tool_call_id"): n for n, m in enumerate(s.messages)
          if m.get("tool_call_id")}
    assert at[R.POINTER_CALL_ID] > at[R.PLANTED_CALL_ID]
    row = s.messages[at[R.POINTER_CALL_ID]]
    text = row["content"][0]["text"]
    p = s.planted
    assert p.passphrase not in text
    assert p.port not in text and p.old_port not in text
    assert s.oracle_needles[R.POINTER_CALL_ID] in text
    assert s.oracle_needles[R.POINTER_CALL_ID] != p.passphrase
    # It is the last turn: the content it points at has every chance to be
    # gone before the pointer exists.
    assert at[R.POINTER_CALL_ID] == max(at.values())


def test_retention_is_the_span_surviving_not_the_row_beating_the_axe(corpus):
    """Clause 4: retention is measured over the content that reaches the
    prompt. A cleared row keeps its `tool_call_id`, so row presence alone would
    score 1.0 for the very policies being tested; and a summary that NAMES the
    codename is not the planted span either."""
    root, files = corpus
    s = R.build_session(7, 20_000, 0.5, root=root, corpus=files, pointer=False)
    turns = s.oracle_turns
    intact = list(s.messages)
    assert R.oracle_survived(intact, turns) == {R.PLANTED_CALL_ID: True}

    cleared, filler_kept = [], 0
    for m in intact:
        if m.get("tool_call_id") == R.PLANTED_CALL_ID:
            m = {**m, "content": [{"type": "text",
                                   "text": "[tool output cleared to save context]"}]}
        else:
            filler_kept += 1
        cleared.append(m)
    assert filler_kept > 10                       # the filler all survived
    assert R.oracle_survived(cleared, turns) == {R.PLANTED_CALL_ID: False}
    assert R.oracle_survived([m for m in intact
                              if m.get("tool_call_id") != R.PLANTED_CALL_ID],
                             turns) == {R.PLANTED_CALL_ID: False}

    # A summary that carries the value verbatim is still not the oracle span.
    summarised = [m for m in intact if m.get("tool_call_id") != R.PLANTED_CALL_ID]
    summarised.insert(0, R._msg(
        "assistant", "[compaction summary — earlier conversation]\n\n"
        f"codename {s.planted.passphrase}; the relay port is in the notes."))
    assert R.summary_text(summarised)             # there IS a summary naming it
    assert R.oracle_survived(summarised, turns) == {R.PLANTED_CALL_ID: False}

    # A re-read of the same content in a NEW row does not count either: the
    # answer came from a fresh read, not from the span staying in context —
    # which the runner already separates with `recovered_via_tool`.
    reread = [m for m in intact if m.get("tool_call_id") != R.PLANTED_CALL_ID]
    reread.append(R._msg("tool", f"codename {s.planted.passphrase}",
                         tool_call_id="call_reread"))
    assert R.oracle_survived(reread, turns) == {R.PLANTED_CALL_ID: False}


def _span_row(arm="tool_clear", retained=(), total=(R.PLANTED_CALL_ID,),
              roles=None, verdict="hit"):
    surv = {tid: (tid in retained) for tid in total}
    return {"arm": arm, "depth": 0.5, "status": "ok", "tool_calls": 1,
            "verdict": {"distinctive": verdict, "ambiguous": verdict},
            "oracle_survived": surv,
            "oracle_roles": roles or {tid: "target" for tid in total},
            "fired": {"turn_start_freed": 0, "relief_freed": 0}}


def test_summary_reports_oracle_retention_beside_correctness_for_every_arm():
    """Clause 2: `oracle_retention` is a k/n/rate column next to
    `distinctive`/`ambiguous` for the SAME arm — correctness alone is the
    number that hides a plausible-but-wrong run."""
    rows = [_span_row(retained=(R.PLANTED_CALL_ID,)),
            _span_row(retained=(R.PLANTED_CALL_ID,))]
    arm = R.summarize(rows)["tool_clear"]
    assert arm["distinctive"]["k"] == 2 and arm["distinctive"]["n"] == 2
    o = arm["oracle_retention"]
    assert o["k"] == 2 and o["n"] == 2 and o["rate"] == 1.0
    assert set(o) >= {"k", "n", "rate", "ci95"}
    assert arm["by_depth"]["0.5"]["oracle_retention"]["n"] == 2
    assert "recall" not in arm                    # still never blended


def test_high_correctness_with_low_retention_is_visible_not_a_win():
    """Clause 4 at the reporting end: two rows that BOTH answer correctly, one
    having kept the oracle span and one having shed it with the filler. Correct
    alone says the arm is fine; the two columns disagreeing is the finding."""
    rows = [_span_row(retained=(R.PLANTED_CALL_ID,)),
            _span_row(retained=())]
    arm = R.summarize(rows)["tool_clear"]
    assert arm["distinctive"]["rate"] == 1.0      # correctness sees nothing
    assert arm["oracle_retention"]["k"] == 1 and arm["oracle_retention"]["n"] == 2
    assert arm["oracle_retention"]["rate"] == 0.5


def test_an_arm_with_no_oracle_bearing_rows_reports_n_zero_and_no_rate():
    """Clause 3: a zero denominator is not a pass. An arm whose rows never
    carried an oracle turn — or a family the run did not exercise — reports
    `n: 0` with `rate: null`, the same shape `rate()` already uses."""
    rows = [{"arm": "none", "depth": 0.1, "status": "ok", "tool_calls": 1,
             "verdict": {"distinctive": "hit", "ambiguous": "hit"}}]
    arm = R.summarize(rows)["none"]
    for col in (arm["oracle_retention"],
                arm["oracle_retention_by_role"]["target"],
                arm["oracle_retention_by_role"]["pointer"]):
        assert col["n"] == 0 and col["k"] == 0 and col["rate"] is None
        assert col["ci95"] == [None, None]
    # And the null survives the JSON the runner prints.
    assert json.loads(json.dumps(arm))["oracle_retention"]["rate"] is None


def test_the_pointer_and_its_target_are_two_separate_numbers():
    """Clause 5: keeping the pointer while losing the content it points at is
    THE failure this probe exists for, and one blended number hides it — 0.75
    reads like a mild regression either way."""
    rows = [
        _span_row(retained=(R.PLANTED_CALL_ID, R.POINTER_CALL_ID),
                  total=(R.PLANTED_CALL_ID, R.POINTER_CALL_ID),
                  roles={R.PLANTED_CALL_ID: "target", R.POINTER_CALL_ID: "pointer"}),
        _span_row(retained=(R.POINTER_CALL_ID,),
                  total=(R.PLANTED_CALL_ID, R.POINTER_CALL_ID),
                  roles={R.PLANTED_CALL_ID: "target", R.POINTER_CALL_ID: "pointer"}),
    ]
    arm = R.summarize(rows)["tool_clear"]
    by = arm["oracle_retention_by_role"]
    assert by["pointer"]["n"] == 2 and by["pointer"]["rate"] == 1.0
    assert by["target"]["n"] == 2 and by["target"]["rate"] == 0.5
    # The blended number is still printed, but it cannot be the only reading.
    assert arm["oracle_retention"]["rate"] == 0.75


def test_the_cost_column_is_the_two_path_sum_not_one_path():
    """The item's clause 4: cost is #2027's predicate — the turn-start
    projection PLUS every relief pass, the sum `valid_for_arm` already gates on
    — never one path. A row whose freeing came only from the in-turn ladder
    must still show its cost."""
    assert R.freed_total({"turn_start_freed": 7_000, "relief_freed": 3_000}) == 10_000
    rows = [_span_row(retained=(R.PLANTED_CALL_ID,),
                      total=(R.PLANTED_CALL_ID,)),
            {"arm": "tool_clear", "depth": 0.5, "status": "ok", "tool_calls": 1,
             "verdict": {"distinctive": "hit", "ambiguous": "hit"},
             "oracle_survived": {R.PLANTED_CALL_ID: False},
             "oracle_roles": {R.PLANTED_CALL_ID: "target"},
             # Freed nothing at turn start and 5,000 in turn: the dominant
             # production path (#1078), which a turn-start-only column misses.
             "fired": {"turn_start_freed": 0, "relief_freed": 5_000}}]
    arm = R.summarize(rows)["tool_clear"]
    assert arm["median_tokens_freed"] == 2_500    # median of 0 and 5,000
    rec = {"mechanisms": ["relief:intra_turn"], "turn_start": {"tokens_freed": 0},
           "relief": [{"reason": "intra_turn", "rungs": ["tool_results"],
                       "freed_tokens": 5_000}]}
    f = R.fired(rec)
    assert f["turn_start_freed"] == 0 and R.freed_total(f) == 5_000
    assert R.valid_for_arm("production", f)       # the gate uses the same sum


def test_run_one_scores_the_spans_of_the_prompt_the_answer_came_from(corpus, tmp_path,
                                                                    monkeypatch):
    """The seam: generator -> session JSON -> turn-start compaction ->
    `_prepare_messages_for_harness` -> the loop's wire prompt -> the row. Both
    oracle turns are scored, and the score is recomputed here from the prompt
    the scripted engine actually saw."""
    root, files = corpus
    s = R.build_session(3, 50_000, 0.5, root=root, corpus=files, pointer=True)
    monkeypatch.setattr(R, "_metrics", lambda base_url: {})
    seen: list[list[dict]] = []

    async def fake_stream(**kw):
        eb = kw.get("extra_body") or {}
        if eb.get("max_tokens") == 1:              # the warm-up
            yield {"choices": [], "usage": {"prompt_tokens": 100}}
            return
        seen.append(kw["messages"])
        yield {"choices": [{"delta": {"content":
            f"CODENAME: {s.planted.passphrase}\nPORT: {s.planted.port}"}}]}
        yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}
        yield {"choices": [], "usage": {"prompt_tokens": 1_000,
                                        "completion_tokens": 5,
                                        "prompt_tokens_details": {"cached_tokens": 900}}}

    monkeypatch.setattr("app.harness.loop.stream_chat", fake_stream)
    disc = [("lloyd-mcp", [{"name": n, "description": n,
                            "inputSchema": {"type": "object", "properties": {}}}
                           for n in ("Read", "Grep", "Bash")])]
    row = asyncio.run(R.run_one(s, "none", discovered=disc, system_prompt="sys",
                               data_root=tmp_path / "d", base_url="http://stub",
                               max_turns=2))
    assert row["status"] == "ok", row.get("reason")
    assert set(row["oracle_survived"]) == set(s.oracle_turn_ids)
    assert row["oracle_survived"] == {R.PLANTED_CALL_ID: True,
                                     R.POINTER_CALL_ID: True}
    assert row["oracle_survived"] == R.oracle_survived(seen[-1], s.oracle_turns)
    arm = R.summarize([row])["none"]
    assert arm["oracle_retention_by_role"]["pointer"]["rate"] == 1.0
    assert arm["oracle_retention_by_role"]["target"]["rate"] == 1.0
    assert arm["median_tokens_freed"] == 0              # `none` frees nothing


def test_the_cli_pointer_run_prints_retention_beside_correctness(tmp_path, monkeypatch,
                                                                 capsys):
    """The runner itself, end to end: argv -> sessions on disk -> the printed
    report. `--pointer` makes each task's oracle a target AND a pointer, and
    the document that gets printed carries `oracle_retention` next to
    `distinctive` for every arm with `n` beside it — including `rate: null` at
    `n: 0`, which is what a dry run (no graded rows) has to show rather than a
    rate of 0 or 1."""
    snap = tmp_path / "tools.json"
    snap.write_text(json.dumps([["lloyd-mcp", [{"name": n, "description": n,
                                                "inputSchema": {"type": "object",
                                                                "properties": {}}}
                                               for n in ("Read", "Grep", "Bash")]]]))
    root = tmp_path / "data"
    monkeypatch.setenv("LLOYD_DATA", str(root))
    rc = R.main(["--sessions", "1", "--sizes", "12000", "--depths", "0.5",
                 "--arms", "none", "--seed", "600", "--pointer", "--dry",
                 "--data-root", str(root), "--tools-snapshot", str(snap),
                 "--out", str(tmp_path / "out.json")])
    assert rc == 0
    printed = json.loads(capsys.readouterr().out.strip().split("\n{", 1)[1].rsplit("}", 1)[0].join(("{", "}")))
    arm = printed["none"]
    assert arm["distinctive"]["n"] == 0                # dry: nothing graded
    assert arm["oracle_retention"] == {"k": 0, "n": 0, "rate": None,
                                       "ci95": [None, None]}
    assert set(arm["oracle_retention_by_role"]) == {"target", "pointer"}
    assert "median_tokens_freed" in arm
    # The task on disk carries its own oracle, and the row scored both spans.
    doc = json.loads((tmp_path / "out.json").read_text())
    assert doc["sessions"][0]["oracle_roles"] == {"call_planted": "target",
                                                  "call_pointer": "pointer"}
    assert doc["rows"][0]["oracle_survived"] == {"call_planted": True,
                                                 "call_pointer": True}


# --- #2349: the sidecar arm (#2168's read-time reduction) ---------------------

def _arms_doc_entry(arm: str) -> str:
    """The module docstring's ARMS entry for `arm`: its name line plus every
    continuation line, up to the next arm's name line."""
    block = R.__doc__.split("ARMS\n----\n")[1]
    lines: list[str] = []
    for line in block.splitlines():
        if not lines:
            if line.startswith(f"  {arm}"):
                lines.append(line)
            continue
        if line.startswith("  ") and not line.startswith("   "):
            break                                   # the next arm's name line
        lines.append(line)
    return "\n".join(lines).rstrip()


def test_the_sidecar_arm_is_tool_clear_plus_the_reduction_switch():
    """#2349 clause 1: the arm exists, and against its stated baseline it differs
    by exactly one switch — the write half of #2168's reduction."""
    base, arm = R.ARMS["tool_clear"], R.ARMS["sidecar"]
    assert arm["compaction"] == base["compaction"], (
        "the read-time reduction is one variable, not a re-tuned threshold")
    assert arm["options"] == {**base["options"],
                              "microcompact_reduction_sidecar": True}
    assert {k: v for k, v in arm.items() if k != "options"} == \
        {k: v for k, v in base.items() if k != "options"}


def test_the_sidecar_arm_is_selectable_via_arms(tmp_path, monkeypatch):
    """#2349 clause 1, the CLI seam: `--arms tool_clear,sidecar` clears the
    runner's own validation and dispatches one row per arm. The bogus name is the
    control that says it was the validator that accepted `sidecar`."""
    snap = tmp_path / "tools.json"
    snap.write_text(json.dumps([["lloyd-mcp", [{"name": n, "description": n,
                                                "inputSchema": {"type": "object",
                                                                "properties": {}}}
                                               for n in ("Read", "Grep", "Bash")]]]))
    root = tmp_path / "data"
    monkeypatch.setenv("LLOYD_DATA", str(root))

    def argv(arms):
        return ["--sessions", "1", "--sizes", "12000", "--depths", "0.5",
                "--arms", arms, "--seed", "600", "--dry",
                "--data-root", str(root), "--tools-snapshot", str(snap),
                "--out", str(tmp_path / "out.json")]

    assert R.main(argv("tool_clear,sidecar")) == 0
    doc = json.loads((tmp_path / "out.json").read_text())
    assert {r["arm"] for r in doc["rows"]} == {"tool_clear", "sidecar"}
    assert all(r["status"] in ("dry", "dropped") for r in doc["rows"]), doc["rows"]
    assert R.main(argv("sidecar_not_an_arm")) == 2


def _stub_the_loop(corpus, tmp_path, monkeypatch, spy_sidecar=False):
    """`run_one` over both arms with the engine cut at `run_query`, which is also
    where the row's `RunOptions` are handed to the loop — the same instance relief
    reads. Returns the arm -> row and arm -> options maps."""
    import app.harness.loop as L
    root, files = corpus
    s = R.build_session(3, 50_000, 0.5, root=root, corpus=files)
    monkeypatch.setattr(R, "_metrics", lambda base_url: {})
    monkeypatch.setenv("LLOYD_DATA", str(tmp_path / "d"))
    monkeypatch.setattr("app.harness.tool_result_spill.SESSIONS_DIR",
                        tmp_path / "d" / "sessions")
    if spy_sidecar:
        import app.harness.microcompact as mc
        reached: list[str] = []
        real = mc.apply_reduction_sidecar

        def _spy(msgs, session_id):
            reached.append(session_id)
            return real(msgs, session_id)
        monkeypatch.setattr(mc, "apply_reduction_sidecar", _spy)

    options: dict[str, object] = {}

    async def fake_run_query(history, opts):
        options[opts.session_id] = opts
        yield {"type": "result",
               "response_text": f"CODENAME: {s.planted.passphrase}\n"
                                f"PORT: {s.planted.port}",
               "stop_reason": "stop"}
    monkeypatch.setattr(L, "run_query", fake_run_query)

    rows = {}
    for arm in ("tool_clear", "sidecar"):
        rows[arm] = asyncio.run(R.run_one(
            s, arm, discovered=[("lloyd-mcp", [{"name": n, "description": n,
                                                "inputSchema": {"type": "object",
                                                                "properties": {}}}
                                               for n in ("Read", "Grep", "Bash")])],
            system_prompt="sys", data_root=tmp_path / "d", base_url="http://stub",
            max_turns=2))
        assert rows[arm]["status"] == "ok", rows[arm].get("reason")
        assert rows[arm]["fired"]["turn_start_freed"] > 0
    by_arm = {opts.session_id.rsplit("-", 1)[-1]: opts for opts in options.values()}
    return s, rows, by_arm, reached if spy_sidecar else None


def test_the_sidecar_arm_reaches_the_read_time_reduction_and_tool_clear_does_not(
        corpus, tmp_path, monkeypatch):
    """#2349 clause 2: the arm's switch has to arrive at `load_and_compact_session`
    through `run_one`, because that parameter falls back to the RunOptions CLASS
    default when it is handed None — the options dict alone cannot open it. The
    spy is `app.harness.microcompact.apply_reduction_sidecar`, the helper
    `app.compaction` lazy-imports inside the pre-pass, so this is the read-time
    pass itself: reached once, under `sidecar`, with the row's own session id, and
    not reached at all under `tool_clear`."""
    s, rows, _opts, reached = _stub_the_loop(corpus, tmp_path, monkeypatch,
                                             spy_sidecar=True)
    assert reached == [f"pt-eval-c600-{s.key}-sidecar"], (
        "the read-time pass must be reached under `sidecar` and never under "
        f"`tool_clear`; saw {reached}")
    assert rows["sidecar"]["fired"]["turn_start_freed"] > 0, (
        "the arm still has to be a firing arm, or the row measures nothing")


def test_the_sidecar_arm_is_the_one_that_writes_the_sidecar_beside_the_session(
        corpus, tmp_path, monkeypatch):
    """#2349 clause 3: relief's writer reads the RunOptions INSTANCE, so the
    question this arm has to answer is whether the instance `run_one` actually
    handed the loop carries the switch. Driven through the real relief rung with
    nothing but that instance and 20 spilled Read rows, the `sidecar` arm's
    options produce `<sid>.microcompact-reduced.json` in the scratch sessions dir
    and `tool_clear`'s leave it without one — with relief relieving in both, so
    the difference is the writer's and not relief's."""
    import app.harness.loop as L
    from app import transcript_entries as te

    s, _rows, opts_by_arm, _ = _stub_the_loop(corpus, tmp_path, monkeypatch)
    assert opts_by_arm["sidecar"].microcompact_reduction_sidecar is True
    assert opts_by_arm["tool_clear"].microcompact_reduction_sidecar is False

    sess = tmp_path / "d" / "sessions"
    # #2348 gave the TURN-START pass its own writer of this file (the sticky
    # `cleared_call_ids` list), so a run_one row can create it before relief ever
    # runs. Start from an empty directory so the diff below measures relief's
    # writer alone, which is what this node is about.
    for stale in sess.glob("*.microcompact-reduced.json"):
        stale.unlink()
    for arm in ("tool_clear", "sidecar"):
        opts = opts_by_arm[arm]
        msgs = [{"role": "user", "content": "read these"}]
        for i in range(20):
            cid = f"call_{i:03d}"
            full = "".join(f"file {i} line {j}\n" for j in range(400))
            tc = te.build_tool_call(cid, "Read", json.dumps({"file_path": f"/f{i}.py"}))
            msgs.append({"role": "assistant", "content": "", "tool_calls": [tc]})
            msgs.append({"role": "tool", "tool_call_id": cid,
                         "content": te.shape_tool_result_for_transcript(
                             full, call_id=cid, session_id=opts.session_id,
                             tool_name="Read")})
        # The arm's trigger is a FRACTION of the window, and at the primary's real
        # 262144 the 0.2 wall sits at 42k tokens while these 20 pointer rows
        # estimate at ~12k — relief would decline on pressure that is not what this
        # test is about. Shrinking the window leaves the arm's own arithmetic
        # (0.2 trigger, 0.1 target) in charge and just moves the wall below this
        # fixture; which ids get recorded does not depend on where it is.
        monkeypatch.setattr("app.compaction.get_context_window",
                            lambda model=None: 20_000)
        before = sorted(p.name for p in sess.glob("*.microcompact-reduced.json"))
        cleared = L._intra_turn_microcompact(
            msgs, options=opts, meter=None, keep_recent=15, tool_count=20,
            iteration=20)
        after = sorted(p.name for p in sess.glob("*.microcompact-reduced.json"))
        assert cleared > 0, f"relief freed nothing under {arm}, so it proved nothing"
        wrote = [n for n in after if n not in before]
        expected = [f"{opts.session_id}.microcompact-reduced.json"]
        if arm == "sidecar":
            assert wrote == expected, (
                f"the sidecar arm relieved {cleared} rows and recorded none of "
                f"them: {wrote}")
        else:
            assert wrote == [], (
                "tool_clear wrote a sidecar, so the two arms are not one switch "
                f"apart: {wrote}")
        assert any("preview dropped" in (m.get("content") or "")
                   for m in msgs if m.get("role") == "tool"), (
            f"nothing was reduced under {arm}, so the writer had nothing to name")


def test_the_docstring_documents_the_sidecar_arm_and_its_baseline():
    """#2349 clause 4: the arm list a future operator reads has to say the arm
    exists, what it is for (#2168) and which arm it is scored against."""
    entry = _arms_doc_entry("sidecar")
    assert entry.startswith("  sidecar"), entry
    assert "#2168" in entry, entry
    assert "tool_clear" in entry, entry
    assert "microcompact_reduction_sidecar" in entry, entry
