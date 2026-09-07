"""`get_active_session_id` is what every ambient producer means by "the user".

On 2026-09-07 the morning brief (MockBOT meeting that night) was injected into
a backlog-triage worker session that had finished 77 seconds earlier and was
answered there, to nobody. It was the last session to receive a user-source
turn, because worker turns go through the chat path to get Inner Voice.
"""
import json
import os
import time

import pytest

from app import sessions_io as sio


def _write(d, sid, platform, age_s=0):
    p = d / f"{sid}.json"
    p.write_text(json.dumps({"session_id": sid, "platform": platform, "messages": []}))
    if age_s:
        t = time.time() - age_s
        os.utime(p, (t, t))
    return p


@pytest.fixture
def sessions(tmp_path, monkeypatch):
    monkeypatch.setattr(sio, "SESSIONS_DIR", tmp_path)
    monkeypatch.setattr(sio, "_last_user_session_id", None)
    return tmp_path


def test_a_worker_session_is_never_the_users_even_when_it_ran_last(sessions):
    _write(sessions, "human", "mission-control", age_s=600)
    _write(sessions, "bot", "worker")
    sio.set_last_user_session("bot")
    assert sio.get_active_session_id() == "human"


def test_rule_two_skips_worker_and_autonomy(sessions):
    _write(sessions, "human", "browser", age_s=900)
    _write(sessions, "task", "autonomy", age_s=300)
    _write(sessions, "bot", "worker")
    assert sio.get_active_session_id() == "human"


def test_nothing_but_machines_resolves_to_none(sessions):
    _write(sessions, "bot", "worker")
    _write(sessions, "task", "autonomy")
    sio.set_last_user_session("bot")
    assert sio.get_active_session_id() is None


def test_an_unknown_platform_stays_eligible(sessions):
    # Deny-list on purpose: a client this code has never heard of must not
    # silently lose its briefs.
    _write(sessions, "new", "some-new-client")
    assert sio.get_active_session_id() == "new"
    assert sio.is_user_session({}) is True


def test_inject_refuses_a_worker_session(tmp_path, monkeypatch):
    """409, not a 200 'skipped': `session_inject_context` reports ok=false
    and no producer records the notification as delivered."""
    import asyncio
    from fastapi import HTTPException
    from app.routers import sessions as R

    monkeypatch.setattr(R, "SESSIONS_DIR", tmp_path)
    _write(tmp_path, "bot", "worker")

    class Req:
        async def json(self):
            return {"text": "Brief + Triage", "source": "autonomy:morning-brief-and-triage"}

    with pytest.raises(HTTPException) as ei:
        asyncio.run(R.inject_ambient_turn("bot", Req()))
    assert ei.value.status_code == 409
    assert "nobody reads" in ei.value.detail
