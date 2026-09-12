"""Spilling `Write`/`Edit` bodies out of assistant tool-call arguments.

The residue no other relief rung can reach. Microcompaction clears tool
*results*; the file body the model wrote lives in the assistant message's
`tool_calls[].function.arguments` and rides in the prompt for the rest of
the turn. A round that writes eight 16k-char files has spent ~32k tokens
per iteration on text that is already on disk.
"""

from __future__ import annotations

import json

import pytest

from app.harness.microcompact import SHRUNK_ARG_MARKER, shrink_assistant_arguments


def _write_call(cid: str, path: str, body: str) -> dict:
    return {
        "id": cid, "type": "function",
        "function": {"name": "Write", "arguments": json.dumps(
            {"file_path": path, "content": body})},
    }


def _conv(n: int, body_chars: int = 20_000) -> list[dict]:
    """n Write calls, each with its result landed."""
    msgs: list[dict] = [{"role": "user", "content": "go"}]
    for i in range(n):
        msgs.append({"role": "assistant", "content": "",
                     "tool_calls": [_write_call(f"c{i}", f"/tmp/f{i}.py",
                                                "B" * body_chars)]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "ok"})
    return msgs


@pytest.fixture
def spill(monkeypatch, tmp_path):
    """Capture what got persisted instead of writing to a real session dir."""
    saved: dict[str, str] = {}

    def _persist(text, *, tool_use_id, session_id):
        p = tmp_path / f"{tool_use_id}.txt"
        p.write_text(text)
        saved[tool_use_id] = text
        return str(p)

    monkeypatch.setattr(
        "app.harness.microcompact.persist_for_compaction", _persist)
    return saved


def test_bodies_are_spilled_and_the_marker_names_the_path(spill):
    msgs = _conv(6)
    out, shrunk, freed = shrink_assistant_arguments(
        msgs, keep_recent_tools=2, min_chars=2_000, session_id="s1")
    assert shrunk == 4            # 6 calls, last 2 held back
    assert freed > 50_000

    args = json.loads(out[1]["tool_calls"][0]["function"]["arguments"])
    assert args["content"].startswith(SHRUNK_ARG_MARKER)
    assert "/tmp/f0.py" in args["content"]      # names what was written
    # The spill path is quoted so the model can Read the body back, and the
    # body really is there: a marker naming a file nobody wrote would be the
    # silent data loss this rung's refuse-on-failure rule exists to prevent.
    assert ".txt" in args["content"]
    assert spill["c0.args.content"] == "B" * 20_000


def test_the_most_recent_calls_are_left_intact(spill):
    msgs = _conv(6)
    out, _, _ = shrink_assistant_arguments(
        msgs, keep_recent_tools=2, min_chars=2_000, session_id="s1")
    # c4 and c5 are the two most recent — untouched.
    for idx, cid in ((9, "c4"), (11, "c5")):
        args = json.loads(out[idx]["tool_calls"][0]["function"]["arguments"])
        assert args["content"] == "B" * 20_000, cid


def test_a_call_whose_result_has_not_landed_is_never_touched(spill):
    """Rewriting an in-flight call would change what the model is shown it
    asked for while the tool is still running.
    """
    msgs = [{"role": "user", "content": "go"},
            {"role": "assistant", "content": "",
             "tool_calls": [_write_call("c0", "/tmp/a.py", "B" * 20_000)]}]
    out, shrunk, _ = shrink_assistant_arguments(
        msgs, keep_recent_tools=0, min_chars=2_000, session_id="s1")
    assert shrunk == 0
    assert out == msgs


def test_a_failed_spill_refuses_rather_than_destroying_the_body(monkeypatch):
    """Same rule `microcompact` applies at the equivalent point: staying over
    budget is recoverable, losing the only copy of what was written is not.
    A `Write` body is not re-derivable from a marker.
    """
    monkeypatch.setattr(
        "app.harness.microcompact.persist_for_compaction",
        lambda *a, **k: None)
    msgs = _conv(4)
    before = json.dumps(msgs)
    out, shrunk, freed = shrink_assistant_arguments(
        msgs, keep_recent_tools=0, min_chars=2_000, session_id="s1")
    assert shrunk == 0
    assert freed == 0
    assert json.dumps(out) == before


def test_no_session_id_means_no_spill_and_no_shrink():
    msgs = _conv(4)
    out, shrunk, _ = shrink_assistant_arguments(
        msgs, keep_recent_tools=0, min_chars=2_000, session_id="")
    assert shrunk == 0
    assert out == msgs


def test_rewritten_arguments_stay_valid_json(spill):
    """`loop._commit_tool_calls` replaces unparseable arguments with `{}` on
    the way in precisely because vLLM re-parses this field as history and
    400s on malformed input. A relief rung that broke it would take the turn
    down at the next request.
    """
    msgs = _conv(4)
    out, _, _ = shrink_assistant_arguments(
        msgs, keep_recent_tools=0, min_chars=2_000, session_id="s1")
    for m in out:
        for tc in m.get("tool_calls") or []:
            parsed = json.loads(tc["function"]["arguments"])
            assert isinstance(parsed, dict)


def test_small_bodies_are_below_the_floor(spill):
    msgs = _conv(4, body_chars=500)
    _, shrunk, _ = shrink_assistant_arguments(
        msgs, keep_recent_tools=0, min_chars=2_000, session_id="s1")
    assert shrunk == 0


def test_only_the_configured_tools_are_shrunk(spill):
    msgs = _conv(4)
    # Rename them to a tool not on the list.
    for m in msgs:
        for tc in m.get("tool_calls") or []:
            tc["function"]["name"] = "Bash"
    _, shrunk, _ = shrink_assistant_arguments(
        msgs, keep_recent_tools=0, min_chars=2_000, session_id="s1",
        tools=("Write", "Edit"))
    assert shrunk == 0


def test_edit_spills_both_halves(spill):
    msgs = [{"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "c0", "type": "function",
                "function": {"name": "Edit", "arguments": json.dumps({
                    "file_path": "/tmp/x.py",
                    "old_string": "O" * 20_000,
                    "new_string": "N" * 20_000})}}]},
            {"role": "tool", "tool_call_id": "c0", "content": "ok"}]
    out, shrunk, _ = shrink_assistant_arguments(
        msgs, keep_recent_tools=0, min_chars=2_000, session_id="s1")
    assert shrunk == 1
    args = json.loads(out[1]["tool_calls"][0]["function"]["arguments"])
    assert args["old_string"].startswith(SHRUNK_ARG_MARKER)
    assert args["new_string"].startswith(SHRUNK_ARG_MARKER)
    assert "c0.args.old_string" in spill
    assert "c0.args.new_string" in spill


def test_a_namespaced_tool_name_is_matched_too(spill):
    msgs = _conv(2)
    for m in msgs:
        for tc in m.get("tool_calls") or []:
            tc["function"]["name"] = "mcp__lloyd-mcp__Write"
    _, shrunk, _ = shrink_assistant_arguments(
        msgs, keep_recent_tools=0, min_chars=2_000, session_id="s1")
    assert shrunk == 2
