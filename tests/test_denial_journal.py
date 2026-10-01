"""app/harness/denial_journal.py — one durable row per refusal, from every guard.

The claims pinned here: a PreToolUse deny is journaled under its gate's name
(a raising fail-closed gate included); the aggregator's `_refused_call` is
journaled under the guard its caller names; the session class is read off
the id; a journal that cannot be written never raises and never changes the
refusal; and scorecard row 16 counts the file without importing the app.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.harness import denial_journal as DJ  # noqa: E402
from app.harness.hooks import HookRegistry  # noqa: E402


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


@pytest.fixture
def journal(tmp_path, monkeypatch):
    path = tmp_path / "safety" / "denials.jsonl"
    monkeypatch.setenv(DJ.JOURNAL_ENV, str(path))
    return path


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def test_a_hook_deny_is_journaled_under_its_gate(journal):
    async def _safety_pretool_cb(input_data, _id, _ctx):
        return {"hookSpecificOutput": {
            "hookEventName": "PreToolUse", "permissionDecision": "deny",
            "permissionDecisionReason": "harness safety: blocked 'rm -rf on root/home/system path' on 'rm -rf ~'"}}
    hooks = HookRegistry()
    hooks.add_pre_tool_use("Bash", _safety_pretool_cb, fail_closed=True)
    out = _run(hooks.fire_pre_tool_use(session_id="20260930_120000_autocode_ab12",
                                       tool_name="Bash", tool_input={"command": "rm -rf ~"}))
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
    rows = _rows(journal)
    assert len(rows) == 1
    r = rows[0]
    assert r["guard"] == "safety" and r["where"] == "hook" and r["tool"] == "Bash"
    assert r["session_class"] == "background"
    assert r["label"] == "rm -rf on root/home/system path"
    assert r["reason"].startswith("harness safety: blocked")
    assert "at" in r and "commit" in r


def test_a_raising_fail_closed_gate_is_journaled_too(journal):
    async def _policy_pretool_cb(input_data, _id, _ctx):
        raise RuntimeError("store unavailable")
    hooks = HookRegistry()
    hooks.add_pre_tool_use(None, _policy_pretool_cb, fail_closed=True)
    out = _run(hooks.fire_pre_tool_use(session_id="20260930_120000_chat_ab12",
                                       tool_name="email_send", tool_input={}))
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
    rows = _rows(journal)
    assert len(rows) == 1 and rows[0]["guard"] == "grant" and rows[0]["tool"] == "email_send"
    assert "raised RuntimeError" in rows[0]["reason"]


def test_a_pass_and_a_fail_open_raise_write_nothing(journal):
    async def observer(input_data, _id, _ctx):
        raise RuntimeError("observer bug")

    async def passes(input_data, _id, _ctx):
        return {}
    hooks = HookRegistry()
    hooks.add_pre_tool_use(None, observer)
    hooks.add_pre_tool_use(None, passes)
    out = _run(hooks.fire_pre_tool_use(session_id="s", tool_name="Read", tool_input={}))
    assert out == {}
    assert _rows(journal) == []


def test_an_unlisted_gate_is_journaled_under_its_own_name(journal):
    async def my_custom_gate(input_data, _id, _ctx):
        return {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                       "permissionDecision": "deny",
                                       "permissionDecisionReason": "no"}}
    hooks = HookRegistry()
    hooks.add_pre_tool_use(None, my_custom_gate)
    _run(hooks.fire_pre_tool_use(session_id="x", tool_name="Write", tool_input={}))
    assert _rows(journal)[0]["guard"].endswith("my_custom_gate")


def test_the_dispatch_refusal_helper_journals_under_the_named_guard(journal):
    from agent_mcp import main as M
    res = M._refused_call("Write", "read-only session: Write is not read-only",
                          guard="tool_sandbox", session_id="bench_588_x_abcd")
    assert res.is_error
    body = json.loads(res.content[0].text)
    assert body["error"].startswith("Tool call denied:")
    rows = _rows(journal)
    assert len(rows) == 1
    assert rows[0] == {**rows[0], "guard": "tool_sandbox", "where": "dispatch",
                       "tool": "Write", "session_class": "bench"}


@pytest.mark.parametrize("sid, cls", [
    ("", "none"),
    ("20260930_120000_autocode_ab12", "background"),
    ("20260930_120000_bench_ab12", "bench"),
    ("bench_588_key_ab12", "bench"),
    ("pt-eval-7", "bench"),
    ("task:abc", "subagent"),
    ("20260930_120000_ab12", "chat"),
    ("alan-chat", "chat"),
])
def test_session_class_is_read_off_the_id(sid, cls):
    assert DJ.session_class(sid) == cls


def test_an_unwritable_journal_never_raises_and_the_refusal_stands(tmp_path, monkeypatch):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    monkeypatch.setenv(DJ.JOURNAL_ENV, str(blocker / "denials.jsonl"))  # parent is a file
    assert DJ.record(guard="safety", where="hook", session_id="s", tool="Bash",
                     reason="blocked") is False
    from agent_mcp import main as M
    res = M._refused_call("Bash", "harness safety: blocked 'x' on 'y'", guard="safety",
                          session_id="s")
    assert res.is_error


def test_rows_and_summarize_read_back_what_was_written(journal):
    DJ.record(guard="safety", where="hook", session_id="20260930_120000_autocode_1",
              tool="Bash", reason="harness safety: blocked 'a' on 'b'")
    DJ.record(guard="safety", where="dispatch", session_id="20260930_120000_1234",
              tool="Bash", reason="harness safety: blocked 'a' on 'c'")
    DJ.record(guard="grant", where="hook", session_id="20260930_120000_autocode_1",
              tool="email_send", reason="grant: no grant")
    rows = DJ.rows(journal)
    assert [r["guard"] for r in rows] == ["safety", "safety", "grant"]
    s = DJ.summarize(rows)
    assert s["total"] == 3
    assert s["by_guard"] == {"safety": 2, "grant": 1}
    assert s["by_class"] == {"background": 2, "chat": 1}
    assert s["by_where"] == {"hook": 2, "dispatch": 1}
    assert s["guard_by_class"]["safety"] == {"background": 1, "chat": 1}
    assert DJ.rows(journal.parent / "nope.jsonl") == []


def test_scorecard_row_16_counts_the_journal_without_the_app(journal, tmp_path, monkeypatch):
    import time
    from scripts.automod import scorecard as SC
    now = time.time()
    for i, (g, c) in enumerate([("safety", "background"), ("safety", "chat"),
                                ("tool_sandbox", "bench"), ("grant", "background")]):
        DJ.record(guard=g, where="hook" if i % 2 else "dispatch",
                  session_id={"background": "20260930_120000_autocode_1",
                              "chat": "20260930_120000_1234",
                              "bench": "bench_x_y_z"}[c],
                  tool="Bash", reason="harness safety: blocked 'lbl' on 'x'" if g == "safety" else "no")
    # one stale row, outside the window
    stale = {"at": "2020-01-01T00:00:00+00:00", "guard": "safety", "where": "hook",
             "session": "s", "session_class": "chat", "tool": "Bash", "label": "", "reason": "",
             "excerpt": "", "commit": ""}
    with journal.open("a") as fh:
        fh.write(json.dumps(stale) + "\n")
    ledger = tmp_path / "promotions.jsonl"
    ledger.write_text("")
    row = SC.compute(since_days=7, ledger=ledger, backlog_dir=tmp_path / "nope",
                     repo=ROOT, now=now, denials=journal)
    d = row["guard_denials"]
    assert d["recorded"] is True and d["count"] == 4
    assert d["by_guard"] == {"safety": 2, "tool_sandbox": 1, "grant": 1}
    assert d["guard_by_class"]["safety"] == {"background": 1, "chat": 1}
    assert d["top_labels"] == {"safety:lbl": 2}
    text = SC.render(row)
    line = [l for l in text.splitlines() if l.startswith("| 16 |")][0]
    assert "safety 2 (" in line and "tool_sandbox 1 (bench 1)" in line
    # a missing journal is "not recorded", not zero
    row2 = SC.compute(since_days=7, ledger=ledger, backlog_dir=tmp_path / "nope",
                      repo=ROOT, now=now, denials=tmp_path / "absent.jsonl")
    assert row2["guard_denials"]["recorded"] is False
    assert "no journal yet" in SC.render(row2)
    # the default resolver honours the env override the writer honours
    assert SC._denial_journal_default() == journal
