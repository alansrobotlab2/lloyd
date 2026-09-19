"""Client-side wake: the `{"type": "wake"}` data message.

The iOS app has no wake word to say — an earbud press or the Action Button is
the wake. It sends `wake` on the data channel, and the worker must treat that
exactly like an acoustic match from that participant: open the continuation
window, lock it to the sender, publish `wake_state`.

What a refactor would break silently: locking to the wrong identity (another
participant's follow-ups would pass the gate), leaving a previous speaker's
voiceprint anchor in place (the new wake has no audio, so a stale anchor would
reject the real follow-up), and the `seconds` override escaping its cap.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent-services"))

np = pytest.importorskip("numpy")
livekit_worker = pytest.importorskip("livekit_worker")


def _bridge():
    bridge = livekit_worker.RoomBridge.__new__(livekit_worker.RoomBridge)
    bridge.room_name = "lloyd-test"
    bridge.wake = livekit_worker.WakeState({"continuation_seconds": 6.0})
    bridge.tts = None
    bridge._client_meta = {}
    bridge.published = 0

    def _publish():
        bridge.published += 1

    bridge._schedule_wake_state_publish = _publish
    return bridge


def _packet(msg: dict, identity: str | None = "ios-phone"):
    participant = SimpleNamespace(identity=identity) if identity else None
    return SimpleNamespace(data=json.dumps(msg).encode(), participant=participant)


def test_wake_opens_window_locked_to_sender():
    b = _bridge()
    b._on_data_received(_packet({"type": "wake"}))
    assert b.wake.matches_lock("ios-phone")
    assert not b.wake.matches_lock("browser-tab")
    assert 5.0 < b.wake.remaining_s() <= 6.0
    assert b.published == 1


def test_wake_clears_stale_voiceprint_anchor():
    b = _bridge()
    b.wake.set_anchor(np.ones(4, dtype=np.float32), "Alan")
    b.wake.extend("browser-tab")
    b._on_data_received(_packet({"type": "wake"}))
    assert b.wake.anchor_embedding is None
    assert b.wake.anchor_name is None


def test_wake_seconds_override_is_capped():
    b = _bridge()
    b._on_data_received(_packet({"type": "wake", "seconds": 15}))
    assert 14.0 < b.wake.remaining_s() <= 15.0
    b._on_data_received(_packet({"type": "wake", "seconds": 600}))
    assert b.wake.remaining_s() <= 30.0
    # A later ordinary extension goes back to the configured window.
    b.wake.extend()
    assert b.wake.remaining_s() <= 6.0


def test_wake_without_identity_is_dropped():
    b = _bridge()
    b._on_data_received(_packet({"type": "wake"}, identity=None))
    assert not b.wake.in_continuation()
    assert b.published == 0
