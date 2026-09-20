"""A `Task` child receives its own background-task completions — and its parent
receives the ones that land after it closed. (#929)

`Bash(run_in_background=true)` tells the model, in its own result and in the
`run_in_background` schema text, that "a `<task_notification>` will appear on
a later turn". Completion records are queued per session id and popped by
`drain_completed_for_session`, which only the chat turn's
`RunOptions.notification_drain` ever called. A subagent's session id is its
own `task:<type>:<hex>` — a queue key no caller passed to that reader — so a
child that obeyed that sentence was waiting on a message its session could
never be handed, and the record sat in the long-lived aggregator process
until it exited.

The fix has two halves, and each is asserted at the boundary that has to
hold, not at the function that was edited:

  * **child side** — the child's `RunOptions` carries a drain over its own
    key, so a completion is spliced into the messages the child's *model* is
    sent. Everything under that runs the real `app.harness.loop.run_query`
    and the real `agent_mcp.builtin_bash`; only the vLLM engine
    (`stream_chat`) and pool construction (`_build_pool`) are faked, so the
    assertion lands on a captured request body — the way
    `tests/test_task_steering.py` asserts what a child saw.
  * **parent side** — at close, undrained records are re-keyed onto the
    parent named by `_subagent_registry.parent_scope`, the precedent
    `_tsc_runner._notify_target` already follows, and a completion arriving
    after the close is written under the parent's key from the start.
    Asserted through the parent's existing chat-turn drain — the internal
    `_BackgroundTaskDrain` handler it calls every iteration — which is what
    makes it delivery rather than a moved file.
"""

from __future__ import annotations

import asyncio
import json

import pytest

import app.harness.loop as loop_mod
from agent_mcp import _subagent_registry as SR
from agent_mcp import _task_registry as TR
from agent_mcp import builtin_bash as BB
from agent_mcp import builtin_task as T

PARENT = "sess-parent-929"
# A live child's session id is minted per run, so the end-to-end tests read it
# back through `_child_session_id()`. This constant names one for the
# unit-level checks, which never run a child at all.
CHILD = "task:general-purpose:929abcde"
_PROFILE = {"system_prompt": "sys", "max_turns": 5, "disallowed_tools": [],
            "model": "primary", "base_url": ""}


@pytest.fixture(autouse=True)
def _clean():
    SR.reset()
    TR._records.clear()
    TR._pending_by_session.clear()
    TR._close_handoff.clear()
    yield
    SR.reset()
    TR._records.clear()
    TR._pending_by_session.clear()
    TR._close_handoff.clear()


@pytest.fixture(autouse=True)
def _reset_cache():
    from app.harness import tool_search_cache
    asyncio.run(tool_search_cache.clear())
    yield
    asyncio.run(tool_search_cache.clear())


@pytest.fixture(autouse=True)
def _fake_child_profile(monkeypatch):
    """The child profile, without resolving a real model from fleet config."""
    monkeypatch.setattr(T, "_load_subagent_profile", lambda _t: dict(_PROFILE))


class _RealBashPool:
    """The child's tool surface: dispatch straight into the real Bash tool.

    The loop passes the session id it is running under — the child's `task:*`
    id — which is exactly how `_spawn_background` keys its record. Going
    through the real handler is the point: the defect was that this key had
    no reader, so a fake pool could not reproduce it.

    `queue_until` makes the timing deterministic. Waiting for the waiter
    coroutine to enqueue the completion fakes nothing — it is giving the real
    subprocess time to exit, which in production is a coin flip against the
    child's next iteration.
    """

    def __init__(self, *, queue_until: bool) -> None:
        self._discovered = [("lloyd-mcp", [{
            "name": "Bash",
            "description": "run a shell command",
            "inputSchema": {"type": "object", "properties": {
                "command": {"type": "string"},
                "run_in_background": {"type": "boolean"},
            }},
        }])]
        self._queue_until = queue_until
        self.spawned: list[str] = []

    @property
    def discovered(self):
        return self._discovered

    async def call_tool(self, name: str, args: dict, *, session_id: str = "", **_kw):
        if name != "Bash":
            return {"content": f"no such tool: {name}", "is_error": True}
        tok = TR.current_session_id.set(session_id)
        try:
            res = await BB.call_tool("Bash", args)
        finally:
            TR.current_session_id.reset(tok)
        payload = json.loads(res.content[0].text)
        task_id = payload.get("task_id", "")
        if task_id:
            self.spawned.append(task_id)
            if self._queue_until:
                await _await_queued(task_id, session_id)
        return {"content": res.content[0].text, "is_error": bool(res.is_error)}


class _StreamScript:
    """Replays scripted vLLM responses and captures every request body.

    The chunks are raw OpenAI-shaped dicts, because `stream_chat` is what is
    being replaced: the loop parses `choices[].delta` itself. Same shape
    `tests/test_task_steering.py` replays.
    """

    def __init__(self, turns) -> None:
        self.turns = list(turns)
        self.captured: list[list[dict]] = []

    def __call__(self, **kwargs):
        self.captured.append([dict(m) for m in (kwargs.get("messages") or [])])
        text, tool_calls = self.turns[min(len(self.captured) - 1,
                                         len(self.turns) - 1)]
        return self._gen(text, tool_calls)

    async def _gen(self, text: str, tool_calls: list[dict]):
        if text:
            yield {"choices": [{"delta": {"content": text}}]}
        for tc in tool_calls:
            yield {"choices": [{"delta": {"tool_calls": [{
                "index": 0, "id": tc["id"], "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc.get("arguments") or {}),
                },
            }]}}]}
        yield {"choices": [{"delta": {},
                            "finish_reason": "tool_calls" if tool_calls else "stop"}]}
        yield {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


def _bg_call(command: str, call_id: str) -> tuple[str, list[dict]]:
    """Iteration 1: the child starts a real background Bash."""
    return "", [{"id": call_id, "name": "Bash",
                 "arguments": {"command": command, "run_in_background": True}}]


ANSWER: tuple[str, list[dict]] = ("done", [])


async def _await_queued(task_id: str, session_id: str, *, limit: float = 15.0) -> None:
    """Wait until the registry has actually queued this task's completion."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + limit
    while loop.time() < deadline:
        rec = TR.get(task_id)
        if rec is not None and rec.finished_at is not None:
            async with TR._lock:
                queued = [r for r in TR._pending_by_session.get(session_id, [])
                          if r.task_id == task_id]
                if queued:
                    return queued[0]
        await asyncio.sleep(0.02)
    pytest.fail(f"background task {task_id} never reached session "
                f"{session_id!r}'s queue")


async def _run_child(script: _StreamScript, pool: _RealBashPool) -> dict:
    """Run one real `Task` child whose first move is a real background Bash.

    `bound_session` in `tests/test_task_steering.py` is the same stand-in for
    what `main.call_tool` binds from the request `_meta`.
    """
    monkey = pytest.MonkeyPatch()
    monkey.setattr(loop_mod, "_build_pool",
                   lambda options: asyncio.sleep(0, result=pool))
    monkey.setattr(loop_mod, "stream_chat", script)
    tok = TR.current_session_id.set(PARENT)
    ttok = TR.current_turn_id.set("turn-1")
    try:
        out = await T._task({
            "subagent_type": "general-purpose",
            "description": "start a build",
            "prompt": "run it in the background and tell me it started",
        })
    finally:
        TR.current_session_id.reset(tok)
        TR.current_turn_id.reset(ttok)
        monkey.undo()
    return json.loads(out)


def _notifications(bodies: list[list[dict]]) -> list[str]:
    """Every spliced `<task_notification>` message the child was shown.

    Matched on a message whose content *is* a notification, not on the
    substring anywhere in a body: the background-start tool result quotes a
    sample notification in its own payload, and that is an example of the
    promise, not delivery of it.
    """
    out = []
    for body in bodies:
        for msg in body:
            content = str(msg.get("content") or "").lstrip()
            if content.startswith("<task_notification>"):
                assert content.endswith("</task_notification>"), content
                out.append(content)
    return out


def _child_session_id() -> str:
    """The session id the run registered its child under, read back from the
    subagent registry rather than reconstructed here."""
    rows = SR.snapshot()["recent"]
    assert len(rows) == 1, rows
    return rows[0]["session_id"]


def _bare_record(task_id: str, *, session_id: str) -> TR.TaskRecord:
    """A finished record with no subprocess behind it."""
    return TR.TaskRecord(
        task_id=task_id, session_id=session_id, command="echo x",
        description=f"task {task_id}",
        output_path=TR.TASKS_DIR / f"{task_id}.log",
        process=None,  # type: ignore[arg-type]
        log_fd=-1, started_at=1000.0, finished_at=1001.0,
        exit_code=0, status="completed",
    )


# ── clause 1: the child sees its own completion ─────────────────────────────

async def test_a_child_sees_its_own_background_completion_on_a_later_iteration():
    """A child that starts a background Bash and runs one more iteration gets
    that task's id, status and exit code in its own `chat_messages`."""
    script = _StreamScript([
        _bg_call("echo child-bg-out", "c1"),
        ANSWER,
    ])
    pool = _RealBashPool(queue_until=True)

    result = await _run_child(script, pool)

    assert result.get("response") == "done", result
    assert len(pool.spawned) == 1, pool.spawned
    task_id = pool.spawned[0]
    assert len(script.captured) >= 2, script.captured
    # Iteration 1's own request body cannot carry it: the child had not
    # started the command when that request was sent.

    # Iteration 1's own request body cannot carry one: nothing was queued when
    # that request was sent.
    assert not _notifications(script.captured[:1]), script.captured[0]

    bodies = _notifications(script.captured)
    assert len(bodies) == 1, bodies
    xml = bodies[0]
    assert f"<task_id>{task_id}</task_id>" in xml, xml
    assert "<status>completed</status>" in xml, xml
    assert "<exit_code>0</exit_code>" in xml, xml
    # Spliced as a user message into the request the child's model is sent —
    # read off the captured body, not off a list someone handed us.
    assert any(m.get("role") == "user"
               and str(m.get("content") or "").lstrip()
               .startswith("<task_notification>")
               for m in script.captured[1]), script.captured[1]


def test_the_child_drain_reads_only_its_own_session_key():
    """The child's drain is a closure over ONE key. A completion belonging to
    another session must not be spliced into the child's conversation, or two
    parallel children would each be told about the other's command."""
    drain = T._child_notification_drain(CHILD)
    TR._pending_by_session.setdefault("some-other-session", []).append(
        _bare_record("bg-other", session_id="some-other-session"))
    TR._pending_by_session.setdefault(CHILD, []).append(
        _bare_record("bg-mine", session_id=CHILD))

    msgs = asyncio.run(drain())

    assert len(msgs) == 1, msgs
    assert msgs[0]["role"] == "user"
    assert "<task_id>bg-mine</task_id>" in msgs[0]["content"]
    assert TR._pending_by_session.get(CHILD, []) == [], "queue must be popped"
    assert [r.task_id for r in TR._pending_by_session["some-other-session"]] == ["bg-other"]


def test_the_child_runs_with_a_drain_and_not_the_chat_builder():
    """The child's `RunOptions` carries the drain, and it is not the chat
    turn's builder — that one also persists a `bg_task_notification` session
    message, and a subagent transcript is in-process, not a
    `~/lloyd/sessions/<id>.json`."""
    captured: dict = {}

    async def fake_run_query(messages, options):
        captured["options"] = options
        return
        yield  # pragma: no cover — keeps this an async generator

    monkey = pytest.MonkeyPatch()
    monkey.setattr(loop_mod, "run_query", fake_run_query)
    tok = TR.current_session_id.set(PARENT)
    try:
        asyncio.run(T._task({"subagent_type": "general-purpose",
                             "description": "d", "prompt": "p"}))
    finally:
        TR.current_session_id.reset(tok)
        monkey.undo()

    options = captured["options"]
    assert options.notification_drain is not None
    assert options.session_id.startswith("task:"), options.session_id
    TR._pending_by_session.setdefault(options.session_id, []).append(
        _bare_record("bg-wired", session_id=options.session_id))
    msgs = asyncio.run(options.notification_drain())
    assert [m["content"] for m in msgs] and "bg-wired" in msgs[0]["content"]

    import inspect
    assert "_build_notification_drain" not in inspect.getsource(T._task)


# ── clause 2: a completion that lands after close reaches the parent ────────

async def test_a_completion_landing_after_the_child_closed_reaches_the_parent():
    """Re-keyed via `_subagent_registry.parent_scope`, then yielded by the
    parent's existing chat-turn drain — the handler the parent's turn calls
    every iteration — so the completion reaches a reader."""
    # Long enough that `Task` has returned long before the subprocess exits:
    # the ordinary case, not an edge.
    script = _StreamScript([
        _bg_call("sleep 2; echo late", "c1"),
        ANSWER,
    ])
    pool = _RealBashPool(queue_until=False)

    await _run_child(script, pool)

    task_id = pool.spawned[0]
    child_key = _child_session_id()
    assert TR._close_handoff.get(child_key) == PARENT, dict(TR._close_handoff)

    await _await_queued(task_id, PARENT)
    tok = TR.current_session_id.set(PARENT)
    try:
        drained = json.loads(
            (await BB.call_tool("_BackgroundTaskDrain", {})).content[0].text)
    finally:
        TR.current_session_id.reset(tok)
    notes = [n for n in drained["notifications"] if n["kind"] == "bg_task"]
    assert [n["task_id"] for n in notes] == [task_id], drained
    assert "<status>completed</status>" in notes[0]["xml"]
    assert "<exit_code>0</exit_code>" in notes[0]["xml"]


async def test_a_completion_the_child_never_drained_is_moved_at_close():
    """The other half of the hand-off: a record already sitting on the child's
    key when `Task` returns goes to the parent too, rather than staying on a
    key nothing will read again."""
    rec = SR.register(subagent_type="general-purpose", description="d",
                      prompt="p", parent_session_id=PARENT,
                      parent_turn_id="turn-1", session_id=CHILD,
                      model="primary", max_turns=5)
    record = _bare_record("bg-undrained", session_id=CHILD)
    TR._pending_by_session.setdefault(CHILD, []).append(record)

    SR.finish(rec, status="completed")
    T._hand_off_background_tasks(rec)

    assert [r.task_id for r in TR._pending_by_session[PARENT]] == ["bg-undrained"]
    assert record.session_id == PARENT, "the record's own key moves with it"


# ── clause 3: nothing is left on the child's key ────────────────────────────

async def test_no_completion_is_left_queued_under_the_child_session_key():
    """The leak. `_pending_by_session` was popped only by
    `drain_completed_for_session`, and nothing ever passed a `task:*` key — so
    every Task-spawned background command kept its record, its formatted
    notification and its output path for the process lifetime."""
    script = _StreamScript([
        _bg_call("sleep 1; echo leftover", "c1"),
        ANSWER,
    ])
    pool = _RealBashPool(queue_until=False)

    await _run_child(script, pool)

    child_key = _child_session_id()
    task_id = pool.spawned[0]

    # Positive first, so the emptiness check below cannot pass on a queue that
    # was never populated: this run's own command, delivered to this run's own
    # parent.
    delivered = await _await_queued(task_id, PARENT)
    assert delivered.session_id == PARENT, delivered

    # Then the rule that made it so. Without `_hand_off_background_tasks`
    # installing it there is nothing to point at, and the assertion about the
    # child's key would be an assertion about a queue nobody filled.
    assert TR._close_handoff.get(child_key) == PARENT, dict(TR._close_handoff)

    assert child_key not in TR._pending_by_session, sorted(TR._pending_by_session)
    leftover = sorted(sid for sid in TR._pending_by_session
                      if sid.startswith("task:"))
    assert leftover == [], f"records still queued under child keys: {leftover}"
    assert [r.task_id for r in TR._pending_by_session[PARENT]] == pool.spawned


def test_a_resumed_child_clears_the_parent_route_before_it_runs():
    """A resume reuses the same `task:*` id, so the resumed run is the reader
    of that key again — the close-time route must go, or a completion the
    resumed child is told to expect would be written to its parent instead."""
    TR._close_handoff[CHILD] = PARENT
    TR.unhand_off(CHILD)
    assert CHILD not in TR._close_handoff
