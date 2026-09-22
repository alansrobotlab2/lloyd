"""How big was that tool result? Only `raw_chars` knows (#1052).

`result_chars` is `len(truncate_tool_result(...))`, so it saturates at
`TOOL_RESULT_MAX_CHARS + len("...(truncated)")` = 2014 and cannot separate a
spilled 250 KB Grep from a merely-truncated 3 KB one — and 50k is where the
harness spills, so everything between 2,014 and 50,000 went to the model
whole and is recorded nowhere. The size survives only where it is measured
before `maybe_spill` rewrites the text, which is why it has to be threaded
out on the event and stamped onto the row.

Three process boundaries are crossed, one per group of tests:

  * harness dispatch → normalised event: the real `_execute_tool_call` with
    a stubbed MCP pool and the REAL `maybe_spill` writing into a scratch
    directory, so the spill actually happens rather than being mocked out;
  * event → chat-router row: the real `app.routers.messages._tool_pair`;
  * event → background-run row: the real `RunRecorder`, read back out of
    the session JSON it wrote, and compared against the chat row.

The rows that are rebuilt from the call log instead of from an event are
pinned at the other end: they must carry NO `raw_chars`. `0` and the 2014
cap are both numbers a later reader would trust, and neither is a
measurement of anything.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from app import transcript_entries as te
from app.harness import loop as L
from app.harness import tool_result_spill as spill_mod
from app.harness.options import RunOptions
from app.harness.tool_search import LoadedToolSet

#: 84,089 characters — over `SPILL_THRESHOLD_CHARS` (50,000) by a margin
#: wide enough that a threshold change does not quietly un-spill it.
BIG = "\n".join(f"line {i} " + "x" * 60 for i in range(1200))
CAP = te.TOOL_RESULT_MAX_CHARS + len("...(truncated)")   # 2014


def _tc(name="Bash", call_id="c1"):
    return {"id": call_id, "function": {"name": name, "arguments": "{}"},
            "_args_dict": {}, "_summary": ""}


class _Pool:
    """An MCP pool whose one tool returns whatever it is handed."""

    def __init__(self, content, is_error=False):
        self.content = content
        self.is_error = is_error

    async def call_tool(self, name, args, **kw):
        return {"content": self.content, "is_error": self.is_error}


@pytest.fixture
def scratch_spill(tmp_path, monkeypatch):
    """Let the real spill run, into a scratch tree.

    `_spill_dir` reads `SESSIONS_DIR` at call time, so this redirects the
    write without touching `maybe_spill` itself — stubbing the spill out
    would leave the length under test never having been rewritten.
    """
    monkeypatch.setattr(spill_mod, "SESSIONS_DIR", tmp_path)
    return tmp_path


async def _dispatch(content, *, session_id="20260922_100622_raw_chars"):
    return await L._execute_tool_call(
        tc=_tc(), pool=_Pool(content), options=RunOptions(model="m"),
        session_id=session_id)


# ── dispatch → event ───────────────────────────────────────────────────

def test_big_fixture_is_over_the_spill_threshold():
    # The tests below are only about spilled results if this holds; a
    # threshold change would otherwise turn them green for the wrong reason.
    assert len(BIG) > spill_mod.SPILL_THRESHOLD_CHARS


def test_a_spilled_result_reports_the_length_from_before_the_preview(scratch_spill):
    evt = asyncio.run(_dispatch(BIG))

    assert spill_mod.PERSISTED_OUTPUT_TAG in evt["content"], \
        "the result never spilled, so nothing rewrote the length"
    assert evt["raw_chars"] == len(BIG)
    assert evt["raw_chars"] > spill_mod.SPILL_THRESHOLD_CHARS
    # The point of the field: the preview is what `len(content)` would
    # have reported, and it is smaller than the answer by design.
    assert len(evt["content"]) < evt["raw_chars"]


def test_the_spilled_preview_alone_would_have_reported_the_cap(scratch_spill):
    """Why the old number was wrong, on the same bytes as the test above."""
    evt = asyncio.run(_dispatch(BIG))
    truncated = te.truncate_tool_result(evt["content"])
    assert len(truncated) == CAP          # what `result_chars` records
    assert evt["raw_chars"] > CAP         # what the model actually received


def test_an_unspilled_result_reports_its_own_length(scratch_spill):
    evt = asyncio.run(_dispatch("80% used"))
    assert spill_mod.PERSISTED_OUTPUT_TAG not in evt["content"]
    assert evt["raw_chars"] == len("80% used") == len(evt["content"])


def test_an_early_rejection_carries_the_length_of_what_the_model_was_told():
    """The factory default, so no error path has to remember to pass it."""
    evt = asyncio.run(L._pre_dispatch(
        tc=_tc("Bash"), options=RunOptions(model="m"), session_id="s",
        loaded_set=LoadedToolSet(enabled=False, catalog=[], loaded=set()),
        runtime_disallowed={"Bash"}))
    assert evt is not None and evt["is_error"]
    assert evt["raw_chars"] == len(evt["content"])


# ── event → the live chat path's row ───────────────────────────────────

def _call_row():
    return te.build_tool_call("c1", "Bash", '{"command": "df -h"}',
                              "Checking disk")


def test_the_chat_path_stamps_the_size_the_model_received(scratch_spill):
    from app.routers.messages import _tool_pair

    evt = asyncio.run(_dispatch(BIG))
    rows = _tool_pair(_call_row(),
                      result_str=te.truncate_tool_result(evt["content"]),
                      timestamp="T", iteration_stats={}, evt=evt)

    call_row, result_row = rows
    assert call_row["role"] == "assistant" and result_row["role"] == "tool"
    assert result_row["stats"]["raw_chars"] == len(BIG)
    # ...beside an unchanged `result_chars`, which stays the truncated
    # length and keeps the UI and every replay reading it unchanged.
    assert result_row["stats"]["result_chars"] == \
        len(te.truncate_tool_result(evt["content"])) <= CAP
    assert result_row["stats"]["is_error"] is False


def test_a_row_rebuilt_from_the_call_log_carries_no_raw_chars_at_all():
    """The cancel and error paths hold only `tool_results_log`.

    Absence is the only honest value: `0` would say the tool answered
    nothing and the cap would say it answered exactly 2,014 characters,
    and a replay counting either would be counting a truncation artifact.
    """
    from app.routers.messages import _tool_pair

    stored = te.truncate_tool_result("y" * 40_000)   # what the log holds
    rows = _tool_pair(_call_row(), result_str=stored, timestamp="T",
                      iteration_stats={})
    stats = rows[1]["stats"]
    assert "raw_chars" not in stats
    assert stats["result_chars"] == CAP


# ── event → the background run's row, and the two agreeing ─────────────

@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr("app.sessions_io.SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr("app.event_log.EVENT_LOGS_DIR", tmp_path / "events")
    monkeypatch.setattr("app.event_log.BLOBS_DIR", tmp_path / "events" / "blobs")
    return tmp_path


def _recorded_tool_row(store, evt):
    """Feed `evt` past the real recorder and return the tool row it wrote."""
    from app.run_recorder import record_events
    from app.sessions_io import create_session

    sid = "20260922_100622_worker_cccc"
    create_session(sid, platform="worker", model="primary", title="t",
                   source="gap-fill")

    async def _feed():
        yield {"type": "tool_call", "call_id": "c1", "name": "Bash",
               "args_json": '{"command": "df -h"}', "summary": "Checking disk"}
        yield evt
        yield {"type": "result", "stop_reason": "stop", "num_turns": 1,
               "usage": {"input_tokens": 3, "output_tokens": 1}}

    async def _run():
        async for _ in record_events(_feed(), session_id=sid, turn_id="turn1",
                                     prompt="check disk", model="primary",
                                     source="worker"):
            pass

    asyncio.run(_run())
    msgs = json.loads(
        (store / "sessions" / f"{sid}.json").read_text())["messages"]
    return next(m for m in msgs if m["role"] == "tool")


def test_the_recorder_and_the_chat_path_stamp_the_same_row_identically(
        store, scratch_spill):
    """One event, both eager writers, one shape.

    The two writers were the reason `transcript_entries` exists at all, and
    a field that landed on only one of them would be the drift that file was
    written to prevent.
    """
    from app.routers.messages import _tool_pair

    evt = asyncio.run(_dispatch(BIG))
    recorded = _recorded_tool_row(store, evt)
    chat = _tool_pair(_call_row(),
                      result_str=te.truncate_tool_result(evt["content"]),
                      timestamp="T", iteration_stats={}, evt=evt)[1]

    assert recorded["stats"]["raw_chars"] == len(BIG)
    assert recorded["stats"] == chat["stats"]
