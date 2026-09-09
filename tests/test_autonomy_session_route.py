"""An autonomy run leaves a transcript.

Until 2026-09-09 `run_task` called `run_query` straight from the backend
process: no session, no Inner Voice, and nothing of the run survived except its
final text in `autonomy-runs/<task>/<run_id>.md`. That is ~180 turns a day —
more LLM work than the rest of the system put together — and the failures were
what it hid worst. From the run table, all three indistinguishable from outside:

    #74 empty response after 257s (stop_reason=stop, turns=14, tool_errors=0)
    #78 empty response after 10s  (stop_reason=stop, turns=1)
    #80 timed out after 300s

#74 did fourteen iterations of real work and reported none of it. A transcript
answers "doing what?" for all three; a run record cannot.

What these tests pin is mostly not "a session is created" — it is that the
*signals `run_task` already made decisions on* survive the move to a route that
reaches the model over HTTP instead of in-process. Each one was load-bearing
before this change and would have been dropped silently:

  * `saw_tool_call` and the wall clock decide infra-vs-task on an empty answer.
    An infra failure gets a flat cooldown; a task failure spends the retry
    budget and disables the task at `max_retries`. On 2026-09-01 every task
    returned empty for eleven hours — misclassifying that disables the fleet.
  * partial text on timeout is what #80 taught: a run killed at its budget
    holding eight paragraphs must not record "(no output before timeout)".
  * the model, because #68 is pinned to `secondary` and the session helper
    hardcoded "primary".
"""
from __future__ import annotations

import pytest

from tests.test_autonomy_scheduler import aut, read_task, write_task

# `aut` is a pytest fixture borrowed from the sibling module: it builds a task
# directory, a run directory and a stub skill, which is most of what a run_task
# test needs. Named in __all__ so it reads as a deliberate re-export rather than
# an unused import.
__all__ = ["aut", "read_task", "write_task"]

# `asyncio_mode = auto` in pytest.ini, so async tests need no mark — and a
# module-level one would be applied to the three synchronous policy tests too.


class FakeSession:
    """Stands in for `run_prompt_in_session`, recording how it was called."""

    def __init__(self, **result):
        self.calls: list[dict] = []
        self.result = {"text": "did the thing", "session_id": "sess_1",
                       "stop_reason": "stop", "num_turns": 3, "errors": [],
                       "usage": {"input_tokens": 10, "output_tokens": 5},
                       "tool_errors": [], "saw_tool_call": True,
                       "structured": None, "structured_error": "", "partial": ""}
        self.result.update(result)

    async def __call__(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        return dict(self.result)


def patch_session(monkeypatch, fake):
    monkeypatch.setattr("workers.sources._common.run_prompt_in_session", fake)


def run_record(aut, task_id=1) -> str:  # noqa: F811
    return next((aut.AUTONOMY_RUNS_DIR / str(task_id)).glob("run_*.md")).read_text()


# ── The route is taken at all ────────────────────────────────────────────────

async def test_a_task_runs_in_a_session_by_default(aut, monkeypatch):
    """The fleet default. `write_task` in the sibling module pins the direct
    route explicitly, so this passes `session_backed=None` to fall through to
    config — which is the path every real task takes."""
    write_task(aut, 1, session_backed=None)
    fake = FakeSession()
    patch_session(monkeypatch, fake)

    async def must_not_run(messages, options):
        raise AssertionError("direct route used for a session-backed task")
        yield  # pragma: no cover

    monkeypatch.setattr("app.harness.run_query", must_not_run)
    result = await aut.run_task(1)

    assert result["success"] is True
    assert len(fake.calls) == 1
    assert result["session_id"] == "sess_1"


async def test_the_run_record_names_the_transcript(aut, monkeypatch):
    """The whole point of the route. A record saying a run took 257s and
    produced nothing is the start of a question; the session id answers it."""
    write_task(aut, 1, session_backed=None)
    patch_session(monkeypatch, FakeSession())
    await aut.run_task(1)

    body = run_record(aut)
    assert "sess_1" in body
    assert "session_id: sess_1" in body or "session_id: 'sess_1'" in body


async def test_opting_out_uses_the_direct_route(aut, monkeypatch):
    """A task that shells out to a script does not need an observer, and the
    escape hatch has to work or the fleet switch is a one-way door."""
    write_task(aut, 1, session_backed=False)

    async def _rq(messages, options):
        yield {"type": "text_delta", "text": "direct"}
        yield {"type": "result", "stop_reason": "stop", "usage": {}, "num_turns": 1}

    monkeypatch.setattr("app.harness.run_query", _rq)

    async def must_not_run(prompt, **kwargs):
        raise AssertionError("session route used for an opted-out task")

    patch_session(monkeypatch, must_not_run)
    result = await aut.run_task(1)

    assert result["success"] is True
    assert result["session_id"] is None


# ── Policy resolution ────────────────────────────────────────────────────────

def test_frontmatter_beats_config_in_both_directions(aut, monkeypatch):
    """Moving one task must not be a fleet change, and a fleet switch must
    still reach a task that says nothing."""
    monkeypatch.setattr(
        "app.config.CONFIG",
        {"autonomy": {"session_backed": {"enabled": True, "inner_voice": False}}},
        raising=False)

    assert aut._session_policy({}) == (True, False)
    assert aut._session_policy({"inner_voice": True}) == (True, True)
    assert aut._session_policy({"session_backed": False}) == (False, False)

    monkeypatch.setattr(
        "app.config.CONFIG",
        {"autonomy": {"session_backed": {"enabled": False, "inner_voice": True}}},
        raising=False)
    assert aut._session_policy({}) == (False, True)
    assert aut._session_policy({"session_backed": True}) == (True, True)


def test_a_blank_frontmatter_value_is_not_a_false(aut, monkeypatch):
    """YAML `inner_voice:` with nothing after it parses to None, and a task
    that states nothing must follow config rather than silently opting out."""
    monkeypatch.setattr(
        "app.config.CONFIG",
        {"autonomy": {"session_backed": {"enabled": True, "inner_voice": True}}},
        raising=False)
    assert aut._session_policy({"inner_voice": None}) == (True, True)
    assert aut._session_policy({"inner_voice": ""}) == (True, True)
    # A string that a human would read as false must not read as true just for
    # being a non-empty string.
    assert aut._session_policy({"inner_voice": "false"}) == (True, False)


def test_missing_config_falls_back_to_the_documented_defaults(aut, monkeypatch):
    """A fresh clone has no `autonomy.session_backed` block, and the default
    that ships must match what config.yaml describes: sessions on, observer off."""
    monkeypatch.setattr("app.config.CONFIG", {}, raising=False)
    assert aut._session_policy({}) == (True, False)


async def test_inner_voice_is_off_for_the_fleet_but_opt_in_per_task(aut, monkeypatch):
    """The observer runs on the primary at priority 1 and judges roughly one
    tool result in five. Defaulting it on for ~180 runs a day is load on the
    engine these tasks are already the heaviest consumer of."""
    write_task(aut, 1, session_backed=None)
    fake = FakeSession()
    patch_session(monkeypatch, fake)
    await aut.run_task(1)
    assert fake.calls[0]["inner_voice"] is False

    write_task(aut, 2, session_backed=None, inner_voice=True)
    fake2 = FakeSession()
    patch_session(monkeypatch, fake2)
    await aut.run_task(2)
    assert fake2.calls[0]["inner_voice"] is True


# ── The signals run_task decides on ──────────────────────────────────────────

async def test_the_task_model_reaches_the_session(aut, monkeypatch):
    """#68 is pinned to `secondary`. `run_prompt_in_session` hardcoded
    "primary", so routing autonomy through it without this would have been a
    silent model swap onto the slot that exists to keep the 35B's single
    tenancy off the primary's queue."""
    write_task(aut, 1, session_backed=None, model="secondary")
    fake = FakeSession()
    patch_session(monkeypatch, fake)
    monkeypatch.setattr("app.config.resolve_model_alias", lambda m: m, raising=False)
    await aut.run_task(1)

    assert fake.calls[0]["model"] == "secondary"


async def test_an_empty_answer_with_no_tool_call_is_infra_not_the_task(aut, monkeypatch):
    """`saw_tool_call` decides whether an empty response spends the task's
    retry budget. It arrives as an SSE frame now rather than a harness event,
    and losing it would make every engine outage look like a broken task —
    the 2026-09-01 shape, eleven hours of empty answers across the fleet."""
    write_task(aut, 1, session_backed=None)
    patch_session(monkeypatch, FakeSession(text="", saw_tool_call=False,
                                           num_turns=1))
    result = await aut.run_task(1)

    assert result["success"] is False
    assert result["failure_kind"] == "infra"
    # infra never escalates, so the retry budget is untouched
    assert read_task(aut, 1)["failure_count"] == 0


async def test_an_empty_answer_after_real_work_is_the_task(aut, monkeypatch):
    """The #74 shape: fourteen iterations, no report. That is a task failure,
    and it must still be counted as one."""
    write_task(aut, 1, session_backed=None)
    patch_session(monkeypatch, FakeSession(text="", saw_tool_call=True,
                                           num_turns=14))
    result = await aut.run_task(1)

    assert result["success"] is False
    assert result["failure_kind"] == "task"
    assert read_task(aut, 1)["failure_count"] == 1


async def test_tool_errors_reach_the_run_record(aut, monkeypatch):
    """A Bash step exiting non-zero is returned to the model as a tool result,
    not raised, so the run can "succeed" having failed. The direct route read
    `is_error` off the harness event; over SSE it rides the `tool_complete`
    frame."""
    write_task(aut, 1, session_backed=None)
    patch_session(monkeypatch, FakeSession(
        tool_errors=["boom: exit 1", "also broken"]))
    result = await aut.run_task(1)

    assert result["success"] is True
    assert result["meta"]["tool_errors"] == 2
    assert "boom: exit 1" in run_record(aut)


async def test_a_stream_error_frame_is_recorded_not_swallowed(aut, monkeypatch):
    """On the direct route a broken turn raises. Over SSE it is an `error`
    frame followed by silence, which would otherwise read as a clean run."""
    write_task(aut, 1, session_backed=None)
    patch_session(monkeypatch, FakeSession(errors=["engine went away"]))
    await aut.run_task(1)

    assert "engine went away" in run_record(aut)


# ── The turn that does not finish ────────────────────────────────────────────

async def test_a_timeout_keeps_the_partial_answer(aut, monkeypatch):
    """#80's lesson, one layer up. `TurnTimeout` carries what streamed before
    the kill; raising a bare error would record "(no output before timeout)"
    for a run that had in fact produced its report."""
    from workers.sources._common import TurnTimeout

    write_task(aut, 1, session_backed=None)

    async def _timeout(prompt, **kwargs):
        raise TurnTimeout("too slow", text="I found three things",
                          tool_errors=["a tool broke"], saw_tool_call=True,
                          session_id="sess_partial")

    patch_session(monkeypatch, _timeout)
    result = await aut.run_task(1)

    assert result["success"] is False
    body = run_record(aut)
    assert "Partial response before timeout" in body
    assert "I found three things" in body
    assert "a tool broke" in body
    assert "sess_partial" in body


async def test_a_drain_does_not_burn_the_retry_budget(aut, monkeypatch):
    """A landing has the backend refusing turns. That is the self-modification
    loop deploying, not this task failing — counting it would spend the retry
    budget of every task due during the window."""
    from workers.sources._common import DrainActive

    write_task(aut, 1, session_backed=None)

    async def _draining(prompt, **kwargs):
        raise DrainActive("lloyd is landing a code update")

    patch_session(monkeypatch, _draining)
    result = await aut.run_task(1)

    assert result["success"] is False
    assert result["failure_kind"] == "infra"
    assert read_task(aut, 1)["failure_count"] == 0


async def test_an_unreachable_backend_falls_back_to_the_direct_route(aut, monkeypatch):
    """The session is a better record of a run, not a precondition for one.
    Autonomy has to keep working through exactly the outages that make the
    record most interesting — and the backend hosting the transcript is the
    thing most likely to be down."""
    write_task(aut, 1, session_backed=None)

    async def _unreachable(prompt, **kwargs):
        raise RuntimeError("stream endpoint returned 500")

    patch_session(monkeypatch, _unreachable)

    ran = {}

    async def _rq(messages, options):
        ran["yes"] = True
        yield {"type": "text_delta", "text": "ran anyway"}
        yield {"type": "result", "stop_reason": "stop", "usage": {}, "num_turns": 1}

    monkeypatch.setattr("app.harness.run_query", _rq)
    result = await aut.run_task(1)

    assert ran.get("yes") is True
    assert result["success"] is True
    # Recorded, because a fallback nobody can see is a route that silently
    # stopped being taken.
    assert result["meta"]["session_fallback"].startswith("RuntimeError")


# ── The seam the fakes above stub out ────────────────────────────────────────
#
# Every test so far replaces `run_prompt_in_session` wholesale, which proves
# `run_task` reads its result correctly and proves nothing about whether the
# result is filled in correctly. These drive the real stream loop over a fake
# transport, because that is where the three new signals are actually read —
# off SSE frames rather than off harness events.

def test_the_stream_reads_the_signals_off_real_frames(monkeypatch):
    """`saw_tool_call`, `tool_errors` and the partial text come from
    `tool_start`, `tool_complete` and `text_delta`. A frame renamed on the
    server would silently zero all three, and every downstream decision reads
    as "the model did nothing"."""
    import json

    from workers.sources import _common as C

    frames = [
        "event: tool_start", 'data: {"call_id": "c1", "name": "Bash"}', "",
        "event: tool_complete",
        'data: {"call_id": "c1", "name": "Bash", "result": "boom", "is_error": true}',
        "",
        "event: tool_start", 'data: {"call_id": "c2", "name": "Read"}', "",
        "event: tool_complete",
        'data: {"call_id": "c2", "name": "Read", "result": "fine", "is_error": false}',
        "",
        "event: text_delta", 'data: {"text": "half an "}', "",
        "event: text_delta", 'data: {"text": "answer"}', "",
        "event: done",
        "data: " + json.dumps({
            "response": "the whole answer", "session_id": "s1",
            "stop_reason": "stop", "num_turns": 4,
            "stats": {"input_tokens": 12, "output_tokens": 7},
        }),
        "",
    ]

    out = _drive_stream(monkeypatch, C, frames)

    assert out["saw_tool_call"] is True
    assert out["tool_errors"] == ["boom"], "only the failing tool result"
    assert out["partial"] == "half an answer"
    assert out["text"] == "the whole answer", "done wins over the deltas"
    assert out["usage"] == {"input_tokens": 12, "output_tokens": 7}
    assert out["num_turns"] == 4


def test_a_timeout_carries_the_partial_answer_out(monkeypatch):
    """The #80 regression, at the layer that would actually cause it: a turn
    that streamed two paragraphs and never reached `done` must not be reported
    as having produced nothing."""
    from workers.sources._common import TurnTimeout
    from workers.sources import _common as C

    frames = [
        "event: tool_start", 'data: {"call_id": "c1", "name": "Bash"}', "",
        "event: tool_complete",
        'data: {"call_id": "c1", "name": "Bash", "result": "nope", "is_error": true}',
        "",
        "event: text_delta", 'data: {"text": "I found three things"}', "",
        # ...and then nothing. No `done` frame ever arrives.
    ]

    with pytest.raises(TurnTimeout) as excinfo:
        _drive_stream(monkeypatch, C, frames, timeout_seconds=0.25, hang=True)

    exc = excinfo.value
    assert exc.text == "I found three things"
    assert exc.tool_errors == ["nope"]
    assert exc.saw_tool_call is True
    assert exc.session_id


def test_the_model_reaches_both_the_session_file_and_the_payload(monkeypatch, tmp_path):
    """`new_worker_session` took a `model` the caller could not pass. A session
    written as `primary` while the turn ran on `secondary` would misreport
    every run of #68 in the session list and the usage table."""
    from workers.sources import _common as C
    import json

    monkeypatch.setattr(C, "SESSIONS_DIR", tmp_path)
    frames = ["event: done",
              'data: {"response": "ok", "session_id": "x", "stop_reason": "stop"}', ""]
    seen = _drive_stream(monkeypatch, C, frames, model="secondary",
                         return_payload=True)

    assert seen["payload"]["model"] == "secondary"
    written = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert written["model"] == "secondary"


# ── the fake transport ───────────────────────────────────────────────────────

def _drive_stream(monkeypatch, C, frames, *, timeout_seconds=30.0, hang=False,
                  model="primary", return_payload=False):
    """Run `run_prompt_in_session` against `frames`, bypassing HTTP."""
    import asyncio
    import contextlib

    captured: dict = {}

    class _Resp:
        status_code = 200

        async def aiter_lines(self):
            for line in frames:
                yield line
            if hang:
                await asyncio.sleep(3600)

        async def aread(self):  # pragma: no cover - only for error paths
            return b""

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        @contextlib.asynccontextmanager
        async def stream(self, method, url, json=None, headers=None):
            captured["payload"] = json
            yield _Resp()

    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    monkeypatch.setattr(C, "_cancel_session_turn",
                        lambda backend, sid: _true())

    async def _go():
        return await C.run_prompt_in_session(
            "go", title="t", source="autonomy-task", model=model,
            timeout_seconds=timeout_seconds, inner_voice=False)

    result = asyncio.run(_go())
    if return_payload:
        return captured
    return result


async def _true():
    return True


# ── The two halves of the wire ───────────────────────────────────────────────

def test_every_frame_the_worker_reads_is_one_the_router_emits():
    """Two hand-written lists, in two files, on either side of an HTTP hop.

    `app/routers/messages.py` names the SSE events; `run_prompt_in_session`
    branches on those names. Nothing made them agree, and drift here is silent
    in the worst direction — a renamed frame does not error, it just stops
    matching, and the signal reads as "the model never did that". CLAUDE.md's
    Browser tab has the same shape of bug written up at four lists; this is the
    same lesson at two, before it costs anything.
    """
    import inspect
    import re

    from app.routers import messages as R
    from workers.sources import _common as C

    emitted = set(re.findall(r'_emit\(turn,\s*"([a-z_]+)"', inspect.getsource(R)))
    consumed = set(re.findall(r'event == "([a-z_]+)"',
                              inspect.getsource(C.run_prompt_in_session)))

    assert consumed, "the stream loop stopped branching on event names"
    assert consumed <= emitted, (
        f"reads frames the router never sends: {sorted(consumed - emitted)}")


def test_the_router_puts_is_error_on_the_tool_complete_frame():
    """The one field this change added to the wire. It is computed a few lines
    below for persistence either way, so deleting it from the frame breaks
    nothing visible in the UI and quietly zeroes autonomy's tool-error count —
    which is what a run record uses to say a "successful" run in fact failed.
    """
    import inspect
    import re

    from app.routers import messages as R

    src = inspect.getsource(R)
    block = re.search(r'_emit\(turn,\s*"tool_complete",\s*\{(.*?)\}\)', src, re.S)
    assert block, "the tool_complete frame moved"
    # Comments stripped first. The first cut of this test matched the word
    # `is_error` in the comment explaining why the key is there, so deleting
    # the key kept it green — a test passing on its own justification.
    payload = "\n".join(line.split("#", 1)[0]
                        for line in block.group(1).splitlines())
    assert '"is_error"' in payload
