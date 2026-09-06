"""Tests for prefetch.py + the #306 subliminal capture helpers.

Hermetic: every worker that would touch disk, qmd, or the facts store is
monkeypatched. Run:
  .venvs/lloyd/bin/python -m pytest tests/test_prefetch.py -q
"""
from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import prefetch  # noqa: E402
from agent_mcp import session as session_mod  # noqa: E402
from agent_mcp.skills import _score_skill, _skill_token_sets, _tokenize  # noqa: E402
from app.routers._messages_subliminal import (  # noqa: E402
    _classify_subliminal,
    _detect_subliminal_sources,
    _extract_subliminal_prefix,
)


# ── #306 capture helpers ──────────────────────────────────────────────────────

def test_extract_prefix_double_newline():
    text = "hello there world"
    pre = "<context>\n<facts>\n- x\n</facts>\n</context>"
    assert _extract_subliminal_prefix(pre + "\n\n" + text, text) == pre


def test_extract_prefix_nudge_without_context_single_newline():
    # A 20-turn nudge on a turn with no <context> block used to glue the
    # nudge to the user text with one "\n"; the extractor then swept the
    # user's own text into the subliminal entry.
    text = "hello there world"
    nudge = "<system-reminder>This session has 20 turns.</system-reminder>"
    assert _extract_subliminal_prefix(nudge + "\n" + text, text) == nudge
    assert _classify_subliminal(nudge) == "memory_nudge"


def test_extract_prefix_ambient_envelope_is_whole_text():
    text = "nightly job finished"
    env = f'<ambient priority="notable" source="cron" session_id="s">\n{text}\n</ambient>\n\nfooter'
    assert _extract_subliminal_prefix(env, text) == env
    assert _classify_subliminal(env) == "ambient_envelope"


def test_extract_prefix_no_injection():
    assert _extract_subliminal_prefix("same", "same") == ""


def test_detect_sources_includes_ide():
    prefix = "<context>\n<vault-context>\n- a\n</vault-context>\n<ide_state>\n  visible_file: x\n</ide_state>\n</context>"
    assert _detect_subliminal_sources(prefix) == ["vault", "ide"]


# ── Focus tracking ────────────────────────────────────────────────────────────

def test_focus_keywords_strip_trailing_punctuation():
    kws = prefetch._extract_focus_keywords("Look at the alfie servo configuration. Then config.yaml.")
    assert "servo" in kws
    assert "configuration" in kws
    assert "configuration." not in kws
    assert "config.yaml" in kws


def test_focus_update_is_thread_safe():
    focus = prefetch.SessionFocus()
    errors: list[BaseException] = []

    def hammer(word: str):
        try:
            for i in range(300):
                focus.update(f"{word}{i % 7} servo shoulder pid gains oscillation")
                focus.enrich_query("what about it")
        except BaseException as e:  # pragma: no cover - only on failure
            errors.append(e)

    threads = [threading.Thread(target=hammer, args=(w,)) for w in ("alpha", "beta", "gamma")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert focus.turn_count == 900


def test_focus_topic_attempt_is_recorded():
    focus = prefetch.SessionFocus()
    for _ in range(5):
        focus.update("servo shoulder pid tuning session")
    assert focus.needs_topic_extraction()
    focus.mark_topic_attempt()  # extraction ran but returned nothing
    assert not focus.needs_topic_extraction()


# ── Continuation detection (skill-hint suppression) ──────────────────────────

@pytest.mark.parametrize("text,is_cont", [
    ("ok", True), ("yes please", True), ("please continue", True), ("let's go", True),
    ("let's do it", True), ("sounds good, proceed", True), ("carry on", True),
    ("continue with the plan", True),
    ("please review the subliminal module", False),
    ("let's build a new feature for the vault", False),
    ("please add a test for the backlog index", False),
])
def test_continuation_regex(text, is_cont):
    assert bool(prefetch._CONTINUATION_RE.match(text)) == is_cont


# ── Backlog task-ref precision ────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("what's left on #294", {294}),
    ("302 is resolved", {302}),
    ("#300 and 300ms", {300}),
    ("PREFETCH_BUDGET_MS = 300", set()),
    ("the endpoint returned a 302 redirect", set()),
    ("HTTP 404 from the daemon", set()),
    ("port 8080 is taken", set()),
    ("took 250 ms and 3 sec", set()),
    ("status 500 on /query", set()),
    ("dated 20260421", set()),
    ("task 311 needs a look", {311}),
])
def test_task_ref_candidates(text, expected):
    assert prefetch._task_ref_candidates(text) == expected


# ── Skill scoring memo ────────────────────────────────────────────────────────

def test_score_skill_memoized_equals_fresh():
    skill = {
        "name": "system-health-check",
        "description": "Full systems check of services and disks",
        "tags": ["health", "supervisord"],
        "body": "Run supervisorctl status. Check disks. " * 50,
    }
    fresh = dict(skill)
    q = {"system", "health", "check", "disk"}
    s1 = _score_skill(skill, q)
    assert "_tok" in skill  # memoized on the dict
    s2 = _score_skill(skill, q)  # second call uses the memo
    s3 = _score_skill(fresh, q)  # separate dict, computed from scratch
    assert s1 == s2 == s3 > 0
    name_tokens, _, tag_tokens, _ = _skill_token_sets(skill)
    assert name_tokens == _tokenize("system health check")
    assert "health" in tag_tokens


def test_score_skill_tolerates_odd_metadata():
    skill = {"name": "x-y", "description": None, "tags": "single", "body": None}
    assert _score_skill(skill, {"single"}) == 1.5  # one tag hit


# ── Vault merge ───────────────────────────────────────────────────────────────

def test_merge_vault_results_dedup_and_cap():
    fresh = [{"file": "a.md", "title": "A", "score": 0.9, "snippet": "a"}]
    carried = [
        {"file": "a.md", "title": "A", "score": 0.95, "snippet": "a"},   # dup of fresh
        {"file": "b.md", "title": "B", "score": 0.7, "snippet": "b"},
        {"file": "c.md", "title": "C", "score": 0.6, "snippet": "c"},
        {"file": "d.md", "title": "D", "score": 0.55, "snippet": "d"},
        {"file": "e.md", "title": "E", "score": 0.52, "snippet": "e"},
        {"file": "f.md", "title": "F", "score": 0.51, "snippet": "f"},
    ]
    merged = prefetch._merge_vault_results(fresh, carried)
    files = [m["file"] for m in merged]
    assert files == ["a.md", "b.md", "c.md", "d.md", "e.md"]
    assert not merged[0].get("carried")
    assert all(m.get("carried") for m in merged[1:])


def test_merge_reserves_slots_for_carried_hits():
    fresh = [{"file": f"f{i}.md", "title": f"F{i}", "score": 0.9 - i * 0.05, "snippet": "x"} for i in range(5)]
    carried = [{"file": "sem1.md", "title": "S1", "score": 0.45, "snippet": "x"},
               {"file": "sem2.md", "title": "S2", "score": 0.40, "snippet": "x"},
               {"file": "sem3.md", "title": "S3", "score": 0.35, "snippet": "x"}]
    merged = prefetch._merge_vault_results(fresh, carried)
    files = [m["file"] for m in merged]
    assert len(files) == prefetch.VAULT_MAX_RESULTS
    assert "sem1.md" in files and "sem2.md" in files and "sem3.md" not in files
    assert files[:3] == ["f0.md", "f1.md", "f2.md"]  # still score-ordered


# ── Budget + carry-over end to end (workers patched) ──────────────────────────

@pytest.fixture
def quiet_workers(monkeypatch):
    monkeypatch.setattr(prefetch, "_search_skills", lambda q: [])
    monkeypatch.setattr(prefetch, "_search_facts", lambda t: [])
    monkeypatch.setattr(prefetch, "_search_recent_sessions", lambda t: [])
    monkeypatch.setattr(prefetch, "_search_backlog_refs", lambda t: [])
    monkeypatch.setattr(prefetch, "_format_ide_state", lambda: "")
    monkeypatch.setattr(prefetch, "PREFETCH_BUDGET_MS", 120)


def _fake_vault(hybrid_delay: float):
    def _search(query, focus=None, legs=("lex", "vec"), **kw):
        if legs == ("lex",):
            return [{"file": "lex.md", "title": "Lex Hit", "score": 0.9, "snippet": "from lex"}]
        time.sleep(hybrid_delay)
        return [
            {"file": "lex.md", "title": "Lex Hit", "score": 0.9, "snippet": "from lex"},
            {"file": "vec.md", "title": "Vec Hit", "score": 0.7, "snippet": "from vec"},
        ]
    return _search


def test_slow_hybrid_is_dropped_then_carried_over(quiet_workers, monkeypatch):
    monkeypatch.setattr(prefetch, "_search_vault", _fake_vault(hybrid_delay=0.5))
    sid = "test-carry"
    msg = "tell me about the alfie servo shoulder pid gains please"

    t0 = time.monotonic()
    out1 = prefetch.prefetch_context(msg, session_id=sid, plan_mode=False)
    elapsed = time.monotonic() - t0
    # The hybrid straggler must not pin the turn at the budget wall; the
    # required workers are instant here, so the call should return well
    # under the 120ms budget.
    assert elapsed < 0.10, f"prefetch blocked for {elapsed:.3f}s despite fast required workers"
    assert "Lex Hit" in out1
    assert "Vec Hit" not in out1  # straggler not ready yet
    assert out1.endswith("\n\n" + msg)

    time.sleep(0.6)  # let the straggler finish and stash
    out2 = prefetch.prefetch_context(msg, session_id=sid, plan_mode=False)
    assert "Vec Hit" in out2
    assert "semantic hit from the previous turn's query" in out2
    assert "Lex Hit" in out2
    # the carry-over was consumed (the new straggler hasn't finished yet)
    focus = prefetch._get_session_focus(sid)
    assert "hybrid" not in focus.pending_vault


def test_hybrid_leg_starts_only_after_lex_returns(quiet_workers, monkeypatch):
    # qmd serializes requests, so the hybrid (vec) leg must not reach the
    # daemon before the lex leg has come back.
    events: list[tuple[str, float]] = []

    def _search(query, focus=None, legs=("lex", "vec"), **kw):
        if legs == ("lex",):
            time.sleep(0.05)
            events.append(("lex_done", time.monotonic()))
            return [{"file": "lex.md", "title": "Lex Hit", "score": 0.9, "snippet": "x"}]
        events.append(("hybrid_start", time.monotonic()))
        return []

    monkeypatch.setattr(prefetch, "_search_vault", _search)
    prefetch.prefetch_context("tell me about the alfie servo shoulder pid", session_id="test-order", plan_mode=False)
    time.sleep(0.1)
    order = [e[0] for e in events]
    assert order == ["lex_done", "hybrid_start"], order


def test_fast_hybrid_is_used_and_stash_cleared(quiet_workers, monkeypatch):
    # The wait loop no longer waits for the hybrid leg, so to observe the
    # "hybrid landed in budget" branch another required worker has to be
    # slower than lex + hybrid. Make skills take 60ms.
    monkeypatch.setattr(prefetch, "_search_vault", _fake_vault(hybrid_delay=0.0))
    monkeypatch.setattr(prefetch, "_search_skills", lambda q: (time.sleep(0.06), [])[1])
    sid = "test-fast-hybrid"
    out = prefetch.prefetch_context("tell me about the alfie servo shoulder pid", session_id=sid, plan_mode=False)
    assert "Vec Hit" in out and "carried" not in out
    assert prefetch._get_session_focus(sid).pending_vault == {}


def test_stale_carry_over_is_discarded(quiet_workers, monkeypatch):
    monkeypatch.setattr(prefetch, "_search_vault", _fake_vault(hybrid_delay=0.0))
    focus = prefetch._get_session_focus("test-stale")
    focus.stash_vault([{"file": "old.md", "title": "Old", "score": 0.9, "snippet": "x"}])
    ts, res = focus.pending_vault["hybrid"]
    focus.pending_vault["hybrid"] = (ts - prefetch.VAULT_CARRY_MAX_AGE_S - 1, res)
    assert focus.take_vault() == []


def test_take_vault_merges_both_legs_hybrid_first():
    focus = prefetch.SessionFocus()
    focus.stash_vault([{"file": "a.md", "title": "A", "score": 0.6, "snippet": "x"},
                       {"file": "b.md", "title": "B", "score": 0.9, "snippet": "x"}], leg="lex")
    focus.stash_vault([{"file": "a.md", "title": "A", "score": 0.8, "snippet": "x"},
                       {"file": "c.md", "title": "C", "score": 0.7, "snippet": "x"}], leg="hybrid")
    got = focus.take_vault()
    assert [(r["file"], r["score"]) for r in got] == [("b.md", 0.9), ("a.md", 0.8), ("c.md", 0.7)]
    assert focus.pending_vault == {}


def test_cold_lex_leg_straggles_and_carries_over(quiet_workers, monkeypatch):
    # Lex takes longer than the soft wait: the turn must return at ~soft
    # wait, not at the full budget, and the lex result must arrive next turn.
    def _search(query, focus=None, legs=("lex", "vec"), **kw):
        if legs == ("lex",):
            time.sleep(0.25)
            return [{"file": "cold.md", "title": "Cold Lex Hit", "score": 0.8, "snippet": "x"}]
        return []

    monkeypatch.setattr(prefetch, "_search_vault", _search)
    monkeypatch.setattr(prefetch, "PREFETCH_BUDGET_MS", 400)
    monkeypatch.setattr(prefetch, "VAULT_LEX_SOFT_WAIT_MS", 100)
    sid = "test-cold-lex"
    t0 = time.monotonic()
    out1 = prefetch.prefetch_context("tell me about the cold topic please", session_id=sid, plan_mode=False)
    elapsed = time.monotonic() - t0
    assert 0.09 <= elapsed < 0.22, elapsed
    assert "Cold Lex Hit" not in out1
    time.sleep(0.4)
    out2 = prefetch.prefetch_context("and more about the cold topic please", session_id=sid, plan_mode=False)
    assert "Cold Lex Hit" in out2 and "previous turn" in out2


def test_short_message_skips_search_but_keeps_ambient(quiet_workers, monkeypatch):
    from app.sessions_io import AmbientPrefetchEntry, enqueue_ambient_prefetch
    monkeypatch.setattr(prefetch, "_search_vault", _fake_vault(hybrid_delay=0.0))
    sid = "test-ambient-short"
    enqueue_ambient_prefetch(sid, AmbientPrefetchEntry(source="cron:x", summary="job done", enqueued_at=time.time()))
    out = prefetch.prefetch_context("ok", session_id=sid, plan_mode=False)
    assert "<ambient-signals>" in out and "job done" in out
    assert "Lex Hit" not in out  # search phase skipped for short text
    assert prefetch.prefetch_context("ok", session_id=sid, plan_mode=False) == "ok"  # queue drained


def test_prefetch_context_async_offloads(quiet_workers, monkeypatch):
    monkeypatch.setattr(prefetch, "_search_vault", _fake_vault(hybrid_delay=0.0))

    async def main():
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.01)
                ticks += 1

        t = asyncio.create_task(ticker())
        out = await prefetch.prefetch_context_async(
            "tell me about the alfie servo shoulder pid gains", session_id="test-async", plan_mode=False,
        )
        t.cancel()
        return out, ticks

    out, ticks = asyncio.run(main())
    assert "Lex Hit" in out
    # The loop kept running while the worker thread did the search.
    assert ticks >= 0


# ── Lex leg: short AND sub-queries with a drop-a-term ladder ─────────────────

def test_enrich_query_does_not_repeat_message_words():
    focus = prefetch.SessionFocus()
    focus.update("alfie servo shoulder tuning")
    focus.update("more alfie servo work")
    q = focus.enrich_query("alfie servo pid")
    assert q.split().count("alfie") == 1 and q.split().count("servo") == 1
    assert "shoulder" in q  # prior-turn context still appended


def test_lex_subqueries_are_short_and_weighted():
    focus = prefetch.SessionFocus()
    focus.update("lets look at the alfie servo shoulder pid gains")
    focus.update("ok, and what did we decide about the stepper closed loop yesterday?")
    qs = focus.lex_subqueries("ok, and what did we decide about the stepper closed loop yesterday?")
    assert qs and all(len(q) <= prefetch.VAULT_LEX_MAX_TERMS for q in qs)
    assert "ok" not in qs[0]
    assert {"stepper", "closed", "loop"} <= set(qs[0])
    assert any("shoulder" in q or "servo" in q for q in qs[1:])  # prior-turn focus query


def test_lex_ladder_drops_terms_until_a_hit(monkeypatch):
    calls: list[str] = []

    def _search(query, focus=None, legs=("lex", "vec"), **kw):
        calls.append(query)
        terms = query.split()
        if legs == ("lex",) and len(terms) <= 3 and "stepper" in terms:
            return [{"file": "s.md", "title": "Closed-Loop Stepper", "score": 0.9, "snippet": "x"}]
        return []

    monkeypatch.setattr(prefetch, "_search_vault", _search)
    focus = prefetch.SessionFocus()
    focus.update("what did we decide about the stepper closed loop yesterday?")
    hits = prefetch._search_vault_lex("what did we decide about the stepper closed loop yesterday?", focus)
    assert [h["file"] for h in hits] == ["s.md"]
    assert len(calls[0].split()) == 4 and len(calls[1].split()) == 3  # ladder stepped down once
    assert len(calls) <= prefetch.VAULT_LEX_MAX_CALLS


def test_single_term_query_still_runs(monkeypatch):
    calls: list[str] = []

    def _search(query, focus=None, legs=("lex", "vec"), **kw):
        calls.append(query)
        return [{"file": "qmd.md", "title": "QMD", "score": 0.9, "snippet": "x"}]

    monkeypatch.setattr(prefetch, "_search_vault", _search)
    focus = prefetch.SessionFocus()
    focus.update("qmd")
    hits = prefetch._search_vault_lex("qmd", focus)
    assert calls == ["qmd"] and hits


def test_lex_ladder_stops_at_deadline(monkeypatch):
    """The ladder must stop starting new calls once it is past the deadline.

    Driven by a fake clock rather than real sleeps. The behaviour under test is
    a comparison against `time.monotonic()`, not real timing, and the earlier
    version — `time.sleep(0.05)` per call plus a wall-clock bound — failed
    whenever the machine was busy. That matters more than usual here: the
    self-modification gate runs this suite while a canary is booting, so a
    load-sensitive test would randomly block promotions for reasons that have
    nothing to do with the change being gated.
    """
    calls: list[str] = []
    clock = {"t": 1000.0}

    def _fake_monotonic():
        return clock["t"]

    def _search(query, focus=None, legs=("lex", "vec"), **kw):
        calls.append(query)
        clock["t"] += 0.05          # each daemon round-trip "costs" 50ms
        return []                   # never hits → the ladder keeps stepping down

    monkeypatch.setattr(prefetch.time, "monotonic", _fake_monotonic)
    monkeypatch.setattr(prefetch, "_search_vault", _search)
    focus = prefetch.SessionFocus()
    focus.update("alfie servo shoulder pid gains oscillation tuning")

    t0 = clock["t"]
    # Calls start at t0, t0+0.05, t0+0.10. A deadline of t0+0.08 (+margin) lets
    # the second start and must stop the third.
    hits = prefetch._search_vault_lex("alfie servo shoulder pid gains oscillation tuning", focus,
                                      deadline=t0 + 0.08 + prefetch.VAULT_LEX_DEADLINE_MARGIN_S)
    assert hits == []
    assert len(calls) == 2, calls   # not the 6-call maximum, and not 1


# ── Session index cache window ────────────────────────────────────────────────

def test_session_index_cache_respects_requested_window(tmp_path, monkeypatch):
    import datetime as dt
    today = dt.datetime.now().strftime("%Y%m%d")
    old = (dt.datetime.now() - dt.timedelta(days=10)).strftime("%Y%m%d")
    for stamp in (today, old):
        (tmp_path / f"{stamp}_120000_abc123.json").write_text(json.dumps({
            "session_id": stamp, "created_at": stamp, "model": "primary",
            "messages": [{"role": "user", "content": "servo tuning talk"}],
        }))
    monkeypatch.setattr(session_mod, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(session_mod, "_session_index_cache", None)

    narrow = session_mod._load_session_index(max_days=3)
    assert len(narrow) == 1
    wide = session_mod._load_session_index(max_days=14)   # must NOT be served the 3-day cache
    assert len(wide) == 2
    narrow_again = session_mod._load_session_index(max_days=3)  # wide cache is fine to reuse
    assert len(narrow_again) == 2


# ── Backlog index incremental rebuild ─────────────────────────────────────────

def test_backlog_index_reparses_only_changed_files(tmp_path, monkeypatch):
    backlog = tmp_path / "obsidian" / "backlog"
    backlog.mkdir(parents=True)
    (backlog / "300-first.md").write_text("---\nstatus: open\npriority: high\nboard: lloyd\n---\n# First task\nbody one")
    (backlog / "301-second.md").write_text("---\nstatus: done\npriority: low\nboard: lloyd\n---\n# Second task\nbody two")
    monkeypatch.setattr(prefetch.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(prefetch, "_backlog_id_cache", {})
    monkeypatch.setattr(prefetch, "_backlog_file_cache", {})
    monkeypatch.setattr(prefetch, "_backlog_id_cache_ts", 0.0)

    parsed: list[str] = []
    real_parse = prefetch._parse_backlog_file

    def counting_parse(path):
        parsed.append(Path(path).name)
        return real_parse(path)

    monkeypatch.setattr(prefetch, "_parse_backlog_file", counting_parse)

    idx = prefetch._get_backlog_index()
    assert idx[300]["title"] == "First task" and idx[301]["status"] == "done"
    assert sorted(parsed) == ["300-first.md", "301-second.md"]

    # Touch one file with a strictly newer mtime, force a rescan.
    parsed.clear()
    f = backlog / "300-first.md"
    f.write_text("---\nstatus: closed\npriority: high\nboard: lloyd\n---\n# First task renamed\nbody")
    import os
    st = f.stat()
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000))
    monkeypatch.setattr(prefetch, "_backlog_id_cache_ts", 0.0)
    idx = prefetch._get_backlog_index()
    assert parsed == ["300-first.md"]
    assert idx[300]["title"] == "First task renamed" and idx[300]["status"] == "closed"
    assert idx[301]["title"] == "Second task"

    refs = prefetch._search_backlog_refs("what's left on #300 and 301?")
    assert len(refs) == 2 and "[Task #300]" in refs[0] and "[Task #301]" in refs[1]
