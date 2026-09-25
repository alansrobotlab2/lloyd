"""D11 (review 2026-09-24): `/compact` is a queued fold into the persisted record.

Until D11 the slash command read a snapshot of the session, awaited the
summariser for up to two minutes and then replaced `data["messages"]` with what
it had summarised: a row appended meanwhile was deleted, every thinking and
subliminal row in the session went with it, and nothing recorded that it had
happened. Now it is a turn of kind `compact` on the session's queue that folds
everything but the last `keep_recent_turns` into the D2 record
(`data["compaction"]`, `source: "manual"`) and never touches the messages.

`compaction.persist_summary` ships false, and the record is all a `/compact`
leaves behind — so a manual record is applied whatever the flag says, and it
stays manual through later automatic folds. That decision is pinned here.

The summariser is monkeypatched to a counter: what is pinned is what is
stored and emitted, never what a model would write.
"""
from __future__ import annotations

import json

import pytest

from app import compaction_llm
from app import compaction_state as CS
from app import sessions_io
from app.compaction import load_and_compact_session
from app.routers import messages as msg_mod

MODEL = "qwen-unknown"            # 128k default window → 76k threshold
SID = "20260924_120000_d11abc"


def _turn(i: int, chars: int = 400) -> list[dict]:
    tid = f"t{i:03d}"
    return [
        {"id": f"u{i:03d}", "role": "user", "turn_id": tid,
         "content": [{"type": "text", "text": f"U{i} " + "u" * chars}]},
        {"id": f"k{i:03d}", "role": "thinking", "content": [],
         "reasoning": f"thought {i}", "thinking": {"turn_id": tid}},
        {"id": f"a{i:03d}", "role": "assistant", "turn_id": tid,
         "content": [{"type": "text", "text": f"A{i} " + "a" * chars}]},
    ]


def _session(n_turns: int) -> list[dict]:
    rows: list[dict] = []
    for i in range(n_turns):
        rows.extend(_turn(i))
    # A subliminal row mid-session, which the old swap also discarded.
    rows.insert(1, {"id": "sub0", "role": "subliminal",
                    "content": [{"type": "text", "text": "<context>x</context>"}]})
    return rows


class Summariser:
    def __init__(self, fail: bool = False):
        self.calls: list[dict] = []
        self.fail = fail

    async def __call__(self, prior, delta, **kw):
        self.calls.append({"prior": prior, "delta": delta, **kw})
        if self.fail:
            return None
        ids = ",".join(r.get("id", "") for r in delta)
        return f"## Goal\nG{len(self.calls)}\n## Progress\nsaw {ids}"


@pytest.fixture
def env(tmp_path, monkeypatch):
    from agent_mcp import _change_ledger as ledger
    from app import event_log
    from app.config import CONFIG

    cfg = {
        "mode": "summarize", "summary_model": None, "keep_recent_turns": 2,
        "persist_summary": False, "summary_input_budget_tokens": 200_000,
        "max_folds_per_turn": 3,
        "microcompact": {"enabled": False},
        "restore": {"enabled": False},
    }
    monkeypatch.setitem(CONFIG, "compaction", cfg)
    monkeypatch.setattr(sessions_io, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(msg_mod, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(event_log, "EVENT_LOGS_DIR", tmp_path / "event_logs")
    monkeypatch.setattr(ledger, "CHANGES_ROOT", tmp_path / "changes")
    ledger.reset()
    summariser = Summariser()
    monkeypatch.setattr(compaction_llm, "summarize_incremental", summariser)

    async def _legacy(*a, **k):
        raise AssertionError("/compact must not call the regenerate path")
    monkeypatch.setattr(compaction_llm, "summarize_history", _legacy)

    path = tmp_path / f"{SID}.json"
    path.write_text(json.dumps({"session_id": SID, "model": MODEL,
                                "last_active": "2026-09-24T10:00:00",
                                "messages": _session(6)}, indent=2))
    yield {"cfg": cfg, "path": path, "sum": summariser,
           "events": tmp_path / "event_logs"}
    ledger.reset()


def _data(env) -> dict:
    return json.loads(env["path"].read_text())


def _events(env, name: str) -> list[dict]:
    p = env["events"] / f"{SID}.events.jsonl"
    if not p.exists():
        return []
    return [e for e in (json.loads(x) for x in p.read_text().splitlines() if x.strip())
            if e["event"] == name]


async def _compact(text: str = "/compact") -> list[dict]:
    """Run one `/compact` turn the way the consumer does; its events."""
    turn = msg_mod._compact_turn(SID, text, MODEL)
    q = sessions_io.SessionQueue()
    await msg_mod._run_turn(SID, turn, q)   # the dispatch at the top of _run_turn
    out = []
    while not turn.events.empty():
        out.append(turn.events.get_nowait())
    return out


async def test_compact_writes_a_manual_record_and_leaves_every_message_in_place(env):
    before = _data(env)["messages"]
    events = await _compact("/compact focus on the ports")

    data = _data(env)
    assert data["messages"] == before, "/compact rewrote the messages"
    assert sum(1 for m in data["messages"] if m["role"] == "thinking") == 6
    assert any(m["role"] == "subliminal" for m in data["messages"])
    rec = data["compaction"]
    assert rec["source"] == "manual"
    assert rec["manual_at"]
    assert rec["instructions"] == "focus on the ports"
    # keep_recent_turns=2 of 6 → the first four turns are covered.
    assert rec["covers_through_entry_id"] == "a003"
    assert env["sum"].calls[0]["instructions"] == "focus on the ports"
    # Written after `messages`, so the retention sweep's prefix read still works.
    assert list(data).index("compaction") > list(data).index("messages")

    names = [e["event"] for e in events]
    assert names[:3] == ["session", "queue_state", "compact_start"]
    done = next(e["data"] for e in events if e["event"] == "compact_done")
    for key in ("session_id", "microcompacted", "summarized", "restored_files",
                "truncated", "tokens_before", "tokens_after", "context_window"):
        assert key in done, key
    assert done["summarized"] is True
    assert done["restored_files"] == 0          # D3: /compact restores no files
    assert done["microcompacted"] == 0          # the legacy count-rule pass is gone
    assert done["tokens_after"] < done["tokens_before"]


async def test_compact_is_recorded_as_compaction_manual(env):
    await _compact("/compact keep the decisions")
    (ev,) = _events(env, "compaction.manual")
    body = ev["data"] if "data" in ev else ev
    assert body["ok"] is True
    assert body["source"] == "manual"
    assert body["instructions"] == "keep the decisions"
    assert body["summarized"] is True and body["folds"] == 1
    assert "summarize" in body["mechanisms"]
    turn = msg_mod._compaction_record.current(SID)
    assert turn is not None and turn.turn_start["summary_folds"] == 1


async def test_a_manual_record_is_applied_with_persist_summary_off(env):
    """The decision D11 makes about D2's flag: off ignores an AUTO record, but
    a `/compact` record is applied — otherwise the command would do nothing,
    since it no longer rewrites the messages."""
    assert env["cfg"]["persist_summary"] is False
    await _compact()
    comp = await load_and_compact_session(env["path"], model=MODEL)
    assert comp["summary_reused"] is True
    head = comp["history"][0]
    assert head["role"] == "assistant"
    assert head["content"][0]["text"].startswith(CS.SUMMARY_HEADER)
    assert [m.get("id") for m in comp["history"][1:]][:1] == ["u004"]

    # The same record relabelled automatic is ignored with the flag off.
    data = _data(env)
    data["compaction"]["source"] = "auto"
    data["compaction"]["manual_at"] = None
    env["path"].write_text(json.dumps(data))
    comp = await load_and_compact_session(env["path"], model=MODEL)
    assert comp["summary_reused"] is False
    assert comp["history"][0].get("id") == "u000"


async def test_a_manual_record_stays_manual_through_a_later_automatic_fold(env):
    await _compact()
    rec = CS.load_record(_data(env))
    stamp = rec["manual_at"]
    data = _data(env)
    convo = CS.conversation_rows(data["messages"])
    later = CS.build_record(summary="S2", convo=convo, boundary=len(convo),
                            files_touched=[], covered_turn_ids=[], model=MODEL,
                            prior=rec, source="auto")
    assert later["source"] == "auto"
    assert later["manual_at"] == stamp
    assert CS.is_manual(later)


async def test_a_second_compact_folds_only_what_the_first_left(env):
    await _compact()
    # Two more turns arrive; the next /compact covers the rows past the record.
    data = _data(env)
    data["messages"].extend(_turn(6) + _turn(7))
    env["path"].write_text(json.dumps(data))
    await _compact()
    second = env["sum"].calls[1]
    assert second["prior"]
    assert [r["id"] for r in second["delta"]][0] == "u004"
    assert _data(env)["compaction"]["covers_through_entry_id"] == "a005"
    assert _data(env)["compaction"]["folds"] == 2


async def test_a_failed_summary_changes_nothing_and_says_so(env):
    env["sum"].fail = True
    before = env["path"].read_text()
    events = await _compact()
    assert env["path"].read_text() == before
    err = [e for e in events if e["event"] == "error"]
    assert err and "summarization failed" in err[0]["data"]["detail"]
    assert not [e for e in events if e["event"] == "compact_done"]
    (ev,) = _events(env, "compaction.manual")
    assert (ev.get("data") or ev)["ok"] is False


async def test_nothing_older_than_the_kept_turns_is_a_done_that_summarized_nothing(env):
    data = _data(env)
    data["messages"] = _session(2)
    env["path"].write_text(json.dumps(data))
    events = await _compact()
    done = next(e["data"] for e in events if e["event"] == "compact_done")
    assert done["summarized"] is False
    assert env["sum"].calls == []
    assert "compaction" not in _data(env)


class _FakeRequest:
    def __init__(self, body: dict):
        self._body = body

    async def json(self):
        return self._body


async def test_the_endpoint_queues_compact_as_a_turn(env, monkeypatch):
    seen: list = []

    async def _fake_enqueue(session_id, turn, consumer_factory):
        seen.append((session_id, turn))
        await turn.events.put(None)
        return {"turn_id": turn.turn_id}

    monkeypatch.setattr(msg_mod, "enqueue_turn", _fake_enqueue)
    resp = await msg_mod.post_message_stream(
        _FakeRequest({"text": "/compact the plan", "session_id": SID}))
    assert resp.media_type == "text/event-stream"
    ((sid, turn),) = seen
    assert sid == SID and turn.source == "user"
    assert turn.payload["kind"] == "compact"
    assert turn.payload["instructions"] == "the plan"
    assert turn.payload["model"] == MODEL
    assert not hasattr(msg_mod, "_slash_compact_sse")
