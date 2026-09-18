"""A participant whose microphone never reaches the worker is reported.

On 2026-09-18 a voice session sat four minutes with nothing heard: the browser
joined the room and never published its mic (the SFU log showed only Lloyd's
own track). The worker logged "connected" and then nothing, which reads the
same as a quiet room. `_check_silent_participants` is the one line that
tells them apart.
"""
import asyncio
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent-services"))

livekit_worker = pytest.importorskip("livekit_worker")
WARN = livekit_worker.NO_AUDIO_WARN_S


def _bridge(present):
    async def make():
        return livekit_worker.RoomBridge(
            "lloyd-20260918_000000_mic", {"room_prefix": "lloyd-"},
            stt=None, vad_cfg={}, http_client=None)
    b = asyncio.run(make())
    b.room = SimpleNamespace(remote_participants={i: object() for i in present})
    return b


def test_a_participant_with_no_audio_is_reported_once_after_the_grace(caplog):
    b = _bridge(["user-a"])
    with caplog.at_level(logging.WARNING, logger="lloyd-agent-worker"):
        assert b._check_silent_participants(100.0) == []          # first sighting
        assert b._check_silent_participants(100.0 + WARN - 1) == []
        assert b._check_silent_participants(100.0 + WARN) == ["user-a"]
        assert b._check_silent_participants(100.0 + WARN + 30) == []  # once
    assert sum("no audio has arrived" in r.message for r in caplog.records) == 1


def test_a_participant_whose_audio_arrived_is_never_reported():
    b = _bridge(["user-a"])
    b._audio_tasks["user-a"] = object()
    b._check_silent_participants(0.0)
    assert b._check_silent_participants(WARN * 10) == []


def test_the_clock_starts_at_join_not_at_the_first_check():
    b = _bridge(["user-b"])
    b._schedule_wake_state_publish = lambda: None   # no room to publish into
    b._on_participant_connected(SimpleNamespace(identity="user-b"))
    joined = b._present_since["user-b"]
    assert b._check_silent_participants(joined + WARN) == ["user-b"]


def test_leaving_resets_the_report_so_a_rejoin_is_judged_afresh():
    b = _bridge(["user-a"])
    b._check_silent_participants(0.0)
    assert b._check_silent_participants(WARN) == ["user-a"]
    b._on_participant_disconnected(SimpleNamespace(identity="user-a"))
    assert "user-a" not in b._no_audio_warned
    b._check_silent_participants(1000.0)
    assert b._check_silent_participants(1000.0 + WARN) == ["user-a"]
