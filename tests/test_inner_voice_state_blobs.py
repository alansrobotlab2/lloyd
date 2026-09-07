"""`/api/inner_voice/state` must hand the UI strings, never blob references.

The context strip on the Inner Voice tab renders `latest_user_request` and
the goal card's three lists straight into JSX. Those values come out of the
event log, which externalizes any field over
`event_log.DEFAULT_BLOB_THRESHOLD_BYTES` to `{"$blob": sha, "size": n}` —
and `get_state` scans with `expand_blobs=False` (correctly: it reads up to
2000 events to find one). So a worker session, whose opening prompt is
routinely 8-15 KB, put an object where the page expected a string and React
threw "Objects are not valid as a React child", blanking the whole tab.
"""
import json

import pytest

from app import event_log
from app.routers import inner_voice


@pytest.fixture
def event_log_dir(tmp_path, monkeypatch):
    logs = tmp_path / "event_logs"
    blobs = logs / "blobs"
    blobs.mkdir(parents=True)
    monkeypatch.setattr(event_log, "EVENT_LOGS_DIR", logs)
    monkeypatch.setattr(event_log, "BLOBS_DIR", blobs)
    return logs


def _write_session(sessions_dir, session_id):
    sessions_dir.mkdir(parents=True, exist_ok=True)
    (sessions_dir / f"{session_id}.json").write_text(
        json.dumps({"inner_voice": True, "inner_voice_evaluate_user_turns": True})
    )


@pytest.fixture
def sessions_dir(tmp_path, monkeypatch):
    d = tmp_path / "sessions"
    monkeypatch.setattr(inner_voice, "SESSIONS_DIR", d)
    return d


def test_long_user_request_is_externalized(event_log_dir):
    """The precondition. Without this the rest of the file proves nothing."""
    event_log.log_event("s1", "inner_voice.goal_card_extracted", {
        "user_request": "x" * 9000,
        "goal_card": {"success_criteria": ["short"]},
    })
    raw = event_log.read_events("s1")[0]
    assert isinstance(raw["data"]["user_request"], dict)
    assert set(raw["data"]["user_request"]) == {"$blob", "size"}


def test_expand_blobs_resolves_a_reference(event_log_dir):
    sha = event_log._store_blob("hello")
    assert event_log.expand_blobs({"a": {"$blob": sha, "size": 5}}) == {"a": "hello"}


def test_expand_blobs_marks_a_missing_blob(event_log_dir):
    out = event_log.expand_blobs({"$blob": "deadbeef", "size": 4})
    assert out == {"$blob_missing": "deadbeef"}


def test_read_events_expand_blobs_kwarg_still_works(event_log_dir):
    """The keyword shadows the module function inside `read_events`."""
    event_log.log_event("s2", "e", {"big": "y" * 9000})
    events = event_log.read_events("s2", expand_blobs=True)
    assert events[0]["data"]["big"] == "y" * 9000


@pytest.mark.asyncio
async def test_state_returns_a_string_for_a_blobbed_request(event_log_dir, sessions_dir):
    _write_session(sessions_dir, "s3")
    event_log.log_event("s3", "inner_voice.goal_card_extracted", {
        "user_request": "z" * 9000,
        "goal_card": {"success_criteria": ["do the thing"], "out_of_scope": []},
    }, turn_id="t1")

    state = await inner_voice.get_state(session_id="s3")

    assert isinstance(state["latest_user_request"], str)
    assert state["latest_user_request"].startswith("zzz")
    assert state["latest_goal_card"]["success_criteria"] == ["do the thing"]
    assert state["latest_turn_id"] == "t1"


@pytest.mark.asyncio
async def test_state_truncates_for_the_context_strip(event_log_dir, sessions_dir):
    _write_session(sessions_dir, "s4")
    event_log.log_event("s4", "inner_voice.goal_card_extracted", {
        "user_request": "q" * 9000, "goal_card": {},
    })
    state = await inner_voice.get_state(session_id="s4")
    text = state["latest_user_request"]
    assert len(text) == inner_voice._MAX_DISPLAY_CHARS + 1
    assert text.endswith("…")


@pytest.mark.asyncio
async def test_state_drops_a_missing_blob_rather_than_shipping_an_object(
    event_log_dir, sessions_dir
):
    """A pruned blob store must not put `{$blob_missing: ...}` into JSX."""
    _write_session(sessions_dir, "s5")
    event_log.log_event("s5", "inner_voice.goal_card_extracted", {
        "user_request": "w" * 9000, "goal_card": {},
    })
    for blob in event_log.BLOBS_DIR.glob("*.txt"):
        blob.unlink()

    state = await inner_voice.get_state(session_id="s5")
    assert state["latest_user_request"] is None


@pytest.mark.asyncio
async def test_state_survives_a_blobbed_goal_criterion(event_log_dir, sessions_dir):
    _write_session(sessions_dir, "s6")
    event_log.log_event("s6", "inner_voice.goal_card_extracted", {
        "user_request": "ask",
        "goal_card": {"success_criteria": ["short one", "L" * 9000]},
    })
    state = await inner_voice.get_state(session_id="s6")
    items = state["latest_goal_card"]["success_criteria"]
    assert all(isinstance(i, str) for i in items)
    assert "short one" in items
