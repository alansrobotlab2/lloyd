"""The deterministic senses run on every turn, with no observer attached.

Stall rescue, repetition, failure payloads, the open-todo gate and the
open-round gate lived inside `install_observer` until 2026-09-24, so a turn
whose session had not opted into Inner Voice — every worker, every subagent,
most chats — got none of them. `app/harness/turn_guards.py` is where they
live now. Each test here runs on a worker-platform turn with no observer.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import usage_store
from app.harness import loop as loop_mod
from app.harness.hooks import HookRegistry
from app.harness.options import RunOptions
from app.harness.turn_guards import install_turn_guards
from app.inner_voice import guards as g

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def rows(monkeypatch):
    got: list[dict] = []
    monkeypatch.setattr(usage_store, "record_inner_voice_observation",
                        lambda **kw: (got.append(kw), len(got))[1])
    return got


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _worker(chat: list | None = None, **kw):
    hooks = HookRegistry()
    state = install_turn_guards(
        hooks, session_id=kw.pop("session_id", "20260924_120000_autocode_ab12"),
        turn_id="turn1", platform=kw.pop("platform", "worker"),
        chat_messages_handle=chat if chat is not None else [], **kw,
    )
    return hooks, state


def _terminal(text: str, iteration: int = 1) -> dict:
    return {"type": "assistant_message", "text": text, "tool_calls": [],
            "iteration": iteration}


# ---------------------------------------------------------------------------
# Through the real loop: a worker stall continues the turn
# ---------------------------------------------------------------------------


class _FakePool:
    discovered = [("lloyd-mcp", [{
        "name": "Bash", "description": "run a command",
        "inputSchema": {"type": "object", "properties": {}},
    }])]


def _chunk(**delta):
    finish = delta.pop("finish_reason", None)
    return {"choices": [{"delta": delta, "finish_reason": finish}]}


def test_a_worker_stall_continues_the_loop_with_no_observer(monkeypatch, rows):
    """The shape that motivated the move: a turn nobody watches stops on
    "Let me check…". The registry has no observer and the caller passes no
    `chat_messages_handle`, so the inject has to reach the loop's private list
    through `bind_run`."""
    requests: list[list[dict]] = []
    replies = iter(["Let me check the logs:", "The logs are clean."])

    async def fake_stream_chat(**kwargs):
        requests.append([dict(m) for m in kwargs["messages"]])
        yield _chunk(content=next(replies), finish_reason="stop")

    async def fake_build_pool(_options):
        return _FakePool()

    monkeypatch.setattr(loop_mod, "stream_chat", fake_stream_chat)
    monkeypatch.setattr(loop_mod, "_build_pool", fake_build_pool)

    hooks = HookRegistry()
    install_turn_guards(hooks, platform="worker", source="autocode")
    opts = RunOptions(model="primary", tool_call_summaries=False, hooks=hooks,
                      session_id="20260924_120000_autocode_ab12", turn_id="t1")

    async def go():
        return [e async for e in loop_mod.run_query(
            [{"role": "user", "content": "check the logs"}], opts)]

    events = _run(go())
    assert len(requests) == 2, "the stall must buy one more iteration"
    injected = requests[1][-1]
    assert injected["role"] == "user"
    assert injected["content"] == "[INNER VOICE] " + g.UNATTENDED_STALL_RESCUE_CONTENT
    assert events[-1]["type"] == "result"
    assert [(r["safeguard"], r["model"], r["session_id"]) for r in rows] == [
        ("stall_rescue", None, "20260924_120000_autocode_ab12")]


# ---------------------------------------------------------------------------
# One test per guard, worker platform, no observer
# ---------------------------------------------------------------------------


def test_stall_rescue_words_follow_the_reader(rows):
    chat_list: list = []
    hooks, _ = _worker(chat_list, platform="mission-control",
                       session_id="20260924_120000_ab12")
    _run(hooks.fire_on_event(_terminal("Let me look at the config:")))
    assert chat_list[-1]["content"].endswith(g.STALL_RESCUE_CONTENT)

    worker_list: list = []
    hooks, state = _worker(worker_list)
    state.round_open = True
    _run(hooks.fire_on_event(_terminal("Now I'll run the gate:")))
    assert worker_list[-1]["content"].endswith(g.UNATTENDED_ROUND_OPEN_CONTENT)


def test_a_delivered_answer_is_left_alone(rows):
    chat: list = []
    hooks, _ = _worker(chat)
    _run(hooks.fire_on_event(_terminal(
        "Here is the report.\n\nLet me know if you need anything else!")))
    assert chat == [] and rows == []


def test_repetition_fires_once_per_fresh_cluster(rows):
    chat: list = []
    hooks, _ = _worker(chat)
    hunt = [
        "grep -rn iv_inject_queue app/inner_voice",
        "grep -rn iv_inject_queue app/routers --include=*.py | head -40",
        'grep -rn "iv_inject_queue" workers/; echo EXIT=$?',
    ]
    for cmd in hunt:
        _run(hooks.fire_pre_tool_use(session_id="s", tool_name="Bash",
                                     tool_input={"command": cmd}))
    assert len(chat) == 1 and "iv_inject_queue" in chat[0]["content"]
    assert rows[0]["trigger"] == "pretool" and rows[0]["safeguard"] == "repetition"


def test_repetition_ignores_a_polling_tool(rows):
    chat: list = []
    hooks, _ = _worker(chat)
    for _ in range(5):
        _run(hooks.fire_pre_tool_use(session_id="s", tool_name="automod_gate_wait",
                                     tool_input={"round_id": "SM_20260924_120000"}))
    assert chat == []


def test_a_failure_payload_is_named_once_per_tool(rows):
    chat: list = []
    hooks, _ = _worker(chat)
    payload = '{"response": "\\n[stopped: max_turns]", "tools_used": ["Read"]}'
    for _ in range(2):
        _run(hooks.fire_on_event({"type": "tool_result", "name": "Task",
                                  "content": payload, "is_error": False}))
    assert len(chat) == 1 and "Task call returned without an error" in chat[0]["content"]
    assert rows[0]["safeguard"] == "failure_payload"
    # An ordinary result says nothing.
    _run(hooks.fire_on_event({"type": "tool_result", "name": "Bash",
                              "content": "ok", "is_error": False}))
    assert len(chat) == 1


def test_an_open_round_at_the_end_of_a_worker_turn_is_named(rows):
    chat: list = []
    hooks, _ = _worker(chat)
    _run(hooks.fire_on_event({"type": "tool_result", "name": "automod_start",
                              "content": "{}", "is_error": False}))
    _run(hooks.fire_on_event(_terminal("I have made the change.")))
    assert chat[-1]["content"].endswith(g.UNATTENDED_ROUND_OPEN_CONTENT)
    # Once per turn, and a closed round says nothing.
    _run(hooks.fire_on_event(_terminal("I have made the change.", iteration=2)))
    assert len(chat) == 1

    chat2: list = []
    hooks, _ = _worker(chat2)
    for name in ("automod_start", "automod_land"):
        _run(hooks.fire_on_event({"type": "tool_result", "name": name,
                                  "content": "{}", "is_error": False}))
    _run(hooks.fire_on_event(_terminal("Landed.")))
    assert chat2 == []


def test_an_open_round_on_a_chat_turn_is_the_humans_business(rows):
    chat: list = []
    hooks, _ = _worker(chat, platform="mission-control", session_id="20260924_1_ab")
    _run(hooks.fire_on_event({"type": "tool_result", "name": "automod_start",
                              "content": "{}", "is_error": False}))
    _run(hooks.fire_on_event(_terminal("Round opened; over to you.")))
    assert chat == []


def test_the_todo_gate_needs_a_list_this_turn_wrote(rows, monkeypatch, tmp_path):
    import app.paths as paths
    monkeypatch.setattr(paths, "SESSIONS_DIR", tmp_path)
    sid = "20260924_120000_autocode_ab12"
    (tmp_path / f"{sid}.json").write_text(json.dumps({"todos": [
        {"content": "write the test", "status": "completed"},
        {"content": "update the doc", "status": "in_progress"},
    ]}))
    # A list inherited from an earlier turn, never touched: no nudge.
    chat: list = []
    hooks, _ = _worker(chat, session_id=sid)
    _run(hooks.fire_on_event(_terminal("Done.")))
    assert chat == []
    # Written this turn and left open: one nudge, naming the item.
    chat2: list = []
    hooks, _ = _worker(chat2, session_id=sid)
    _run(hooks.fire_on_event({"type": "tool_result", "name": "TodoWrite",
                              "content": "ok", "is_error": False}))
    _run(hooks.fire_on_event(_terminal("Done.")))
    _run(hooks.fire_on_event(_terminal("Done.", iteration=2)))
    assert len(chat2) == 1 and "update the doc" in chat2[0]["content"]
    assert rows[-1]["safeguard"] == "todo_gate"


def test_deterministic_injects_are_capped(rows):
    chat: list = []
    hooks, state = _worker(chat)
    state.cfg["deterministic_inject_budget"] = 2
    for i in range(4):
        _run(hooks.fire_on_event(_terminal("Let me check the logs:", iteration=i)))
    assert len(chat) == 2
    assert [r["action"] for r in rows] == [
        "inject", "inject",
        "noop_deterministic_budget_exhausted", "noop_deterministic_budget_exhausted"]


def test_install_is_idempotent_and_fills_in_what_it_learns():
    hooks = HookRegistry()
    first = install_turn_guards(hooks, platform="worker")
    handle: list = []
    second = install_turn_guards(hooks, session_id="s", chat_messages_handle=handle)
    assert first is second and hooks.turn_guards is first
    assert first.session_id == "s" and first.chat_messages is handle
    assert len(hooks._pre) == 1 and len(hooks._on_event) == 1


def test_the_kill_switch(monkeypatch):
    from app.config import CONFIG
    monkeypatch.setitem(CONFIG, "inner_voice",
                        {**(CONFIG.get("inner_voice") or {}),
                         "turn_guards": {"enabled": False}})
    hooks = HookRegistry()
    assert install_turn_guards(hooks) is None
    assert hooks._pre == [] and hooks._on_event == []


# ---------------------------------------------------------------------------
# With an observer attached, the guard answers and the observer stands down
# ---------------------------------------------------------------------------


def test_the_observer_does_not_judge_an_iteration_a_guard_answered(rows, monkeypatch):
    from app.inner_voice import observer as obs_mod

    def no_llm(**_kw):
        raise AssertionError("the observer must not pay a call on a guarded stall")

    monkeypatch.setattr(obs_mod, "_post_chat_completion_with_tools", no_llm)
    monkeypatch.setattr(obs_mod, "record_inner_voice_observation",
                        lambda **kw: (rows.append(kw), len(rows))[1])
    chat: list = []
    hooks = HookRegistry()
    state = obs_mod.install_observer(
        hooks=hooks, session_id="20260924_1_ab", turn_id="t", user_request="x",
        chat_messages_handle=chat, cancel_event=asyncio.Event(),
        primary_model="primary",
    )
    _run(hooks.fire_on_event(_terminal("Let me check the logs:")))
    assert len(chat) == 1, chat
    assert state.bypass_interventions_used == 1
    assert state.decisions_this_turn[0]["action"] == "inject"
    assert [r["safeguard"] for r in rows] == ["stall_rescue", "turn_guard"]
    # One sequence for the turn, shared by both writers.
    assert [r["sequence_in_turn"] for r in rows] == [1, 2]


# ---------------------------------------------------------------------------
# Every turn path installs them
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", [
    "app/routers/messages.py",       # stream, ambient, voice (via _run_turn), sync
    "workers/sources/_common.py",    # direct worker turns
    "autonomy.py",                   # scheduled tasks
    "agent_mcp/builtin_task.py",     # Task subagents
    "app/inner_voice/observer.py",   # anything that attaches an observer
])
def test_every_turn_path_installs_the_guards(path):
    assert "install_turn_guards(" in (ROOT / path).read_text(), path


def test_the_direct_worker_options_carry_the_guards():
    from workers.sources._common import _worker_run_options

    opts = _worker_run_options(10, source="bench-mine")
    assert opts.hooks.turn_guards is not None
    assert opts.hooks.turn_guards.platform == "worker"


def test_a_tool_call_written_as_prose_is_raised_not_injected(rows, monkeypatch):
    import app.harness.turn_guards as tg
    import app.prefix_miss as pm

    toasts: list = []
    monkeypatch.setattr(pm, "_announce", lambda *a, **k: toasts.append(a) or {})
    monkeypatch.setattr(tg, "_last_fault_announce", 0.0)
    chat: list = []
    hooks, _ = _worker(chat, platform="mission-control", session_id="20260924_1_ab")
    text = 'Running it now: {"name": "Bash", "input": {"command": "sqlite3 workers.db"}}'
    _run(hooks.fire_on_event(_terminal(text)))
    _run(hooks.fire_on_event(_terminal(text, iteration=2)))
    assert chat == [], "a missing capability is not fixed by more text"
    assert [r["action"] for r in rows] == ["noop_capability_fault"] * 2
    assert len(toasts) == 1, "one toast per cooldown, not one per iteration"
    assert toasts[0][3] == "warning"
