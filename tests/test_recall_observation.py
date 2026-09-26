"""#1481: `recall_observation(id)` — a cleared tool result, back by its id.

Clause 3: a served, read-only tool that returns the full persisted content
for an id cleared in the current session. Clause 4: a refusal for any id that
is not in the calling session's own spill set, and no path outside
`<SESSIONS_DIR>/<session_id>.tool-results/` is ever opened.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from agent_mcp import recall_observation as ro


def _cleared(tmp_path, monkeypatch, sid="sidA", content=None):
    """Clear a real result through microcompact, so the id is the one a stub names."""
    from app.compaction import estimate_conversation_tokens
    from app.harness import microcompact as mc

    monkeypatch.setattr("app.harness.tool_result_spill.SESSIONS_DIR", tmp_path)
    msgs = []
    for i in range(8):
        cid = f"call_{i}"
        msgs.append({"role": "assistant", "content": "", "tool_calls": [{
            "id": cid, "type": "function",
            "function": {"name": "Read", "arguments": json.dumps({"file_path": f"/f{i}"})}}]})
        body = content if (i == 0 and content) else f"result {i}\n" * 600
        msgs.append({"role": "tool", "tool_call_id": cid, "content": body})
    est = lambda ms: estimate_conversation_tokens(ms, "")  # noqa: E731
    out, cleared = mc.microcompact(msgs, token_budget=1, estimate_fn=est,
                                   keep_recent_tools=2, session_id=sid,
                                   legacy_count_rule=False, observation_stubs=True)
    assert cleared == 6
    return msgs, out


def _call(args, sid):
    from agent_mcp import _task_registry
    tok = _task_registry.current_session_id.set(sid)
    try:
        return asyncio.run(ro.call_tool("recall_observation", args))
    finally:
        _task_registry.current_session_id.reset(tok)


def _text(res):
    return res.content[0].text


def test_it_is_served_and_read_only():
    from agent_mcp import annotations, main

    assert ro in main.MODULES
    tools = asyncio.run(ro.list_tools())
    assert [t.name for t in tools] == ["recall_observation"]
    assert "recall_observation" in annotations.READ_ONLY
    assert annotations.annotations_for("recall_observation").read_only_hint is True


def test_it_returns_the_full_content_for_an_id_this_session_cleared(tmp_path, monkeypatch):
    planted = '{"error": "a result that happens to look like an error"}\n' + "x" * 9000
    msgs, out = _cleared(tmp_path, monkeypatch, content=planted)
    stub = out[1]["content"]
    assert stub.startswith("[observation call_0 ")
    res = _call({"id": "call_0"}, "sidA")
    assert _text(res) == planted
    # A recalled JSON-error-shaped result is still a successful call.
    assert res.is_error is False
    assert _text(_call({"id": "call_3"}, "sidA")) == msgs[7]["content"]


@pytest.mark.parametrize("bad", [
    "call_99",                         # made up
    "../sidB.tool-results/call_0",     # another session's, by path
    "/etc/passwd", "..", ".", "",
    "call_0.txt/../../sidB",
])
def test_anything_but_this_sessions_own_id_is_refused(tmp_path, monkeypatch, bad):
    _cleared(tmp_path, monkeypatch, sid="sidA")
    _cleared(tmp_path, monkeypatch, sid="sidB")
    res = _call({"id": bad}, "sidA")
    assert res.is_error is True
    assert "error" in json.loads(_text(res))


def test_another_sessions_id_is_refused_even_when_that_id_exists_there(tmp_path, monkeypatch):
    _cleared(tmp_path, monkeypatch, sid="sidB")
    (tmp_path / "sidA.tool-results").mkdir()
    res = _call({"id": "call_0"}, "sidA")       # exists only under sidB
    assert res.is_error is True
    assert _call({"id": "call_0"}, "sidB").is_error is False
    # No bound session: nothing resolves.
    assert _call({"id": "call_0"}, "").is_error is True


def test_a_symlink_out_of_the_spill_dir_is_not_followed(tmp_path, monkeypatch):
    _cleared(tmp_path, monkeypatch, sid="sidA")
    secret = tmp_path / "secret.txt"
    secret.write_text("outside")
    (tmp_path / "sidA.tool-results" / "evil.txt").symlink_to(secret)
    res = _call({"id": "evil"}, "sidA")
    assert res.is_error is True and "outside" not in _text(res)


def test_the_harness_advertises_it_only_when_stubs_are_on():
    """Off, the catalog is today's: the tool joins the turn's hidden set, which
    is both what is advertised and what dispatch refuses."""
    from app.harness import loop as L
    from app.harness.context_meter import ContextMeter
    from app.harness.options import RunOptions

    class _Pool:
        discovered = [("lloyd-mcp", [
            {"name": "Read", "description": "r", "inputSchema": {"type": "object"}},
            {"name": "recall_observation", "description": "o",
             "inputSchema": {"type": "object"}},
        ])]

    def hidden(on: bool) -> set[str]:
        opts = RunOptions(model="primary", tool_call_summaries=False,
                          intra_turn_microcompact_observation_stubs=on)
        st = asyncio.run(L._open_turn(opts, [], ContextMeter(262_144), _Pool(), 0.0))
        return st.surface_hidden

    assert "recall_observation" in hidden(False)
    assert "recall_observation" not in hidden(True)
