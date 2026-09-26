"""The 503 a landing puts in front of chat is a contract, and the drain GET
carries what the web client needs to hold and resend a refused message.

Three readers depend on the refusal today: `run_prompt_in_session` raises
`DrainActive` on the phrase, the review grader's `_is_unavailable` matches it
and `_retry_delay` reads the retry hint out of it, and the web client keys on
the `X-Lloyd-Landing` header (never the string — the string belongs to the
parsers). These tests pin every side to `landing_refusal`, so a reword breaks
here instead of in a grader mid-landing.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from app.routers import automod as A


@pytest.fixture(autouse=True)
def _drain_off():
    A.set_drain(False)
    yield
    A.set_drain(False)


def test_landing_refusal_keeps_the_shape_every_parser_reads():
    A.set_drain(True, 42)
    exc = A.landing_refusal()
    assert exc.status_code == 503
    # The grader's unavailability matcher and its retry hint.
    from scripts.automod import review as R
    assert R._is_unavailable(f"HTTP 503: {exc.detail}")
    assert R._retry_delay(exc.detail, 0) == pytest.approx(42.0, abs=2.0)
    # The worker pool's DrainActive matcher (workers/sources/_common.py).
    assert "is landing a code update" in exc.detail.lower()
    # The web client's key: headers, never the string.
    assert exc.headers["X-Lloyd-Landing"] == "1"
    assert int(exc.headers["Retry-After"]) >= 1


def test_the_drain_get_carries_the_land_train(monkeypatch):
    monkeypatch.setattr("scripts.automod.state.read_pending", lambda: [
        {"commit": "c" * 40, "restart": True},
        {"commit": "d" * 40, "restart": False},
    ])
    monkeypatch.setattr("scripts.automod.state.flush_in_progress", lambda: None)
    body = json.loads(asyncio.run(A.get_drain()).body)
    assert body["draining"] is False and body["pending"] == 2
    assert body["restart_needed"] == 1 and body["flushing"] is False


def test_the_drain_get_answers_with_the_flag_alone_when_state_cannot(monkeypatch):
    """The web client polls this while the backend is mid-landing; the flag
    is in-process and must answer even when the state dir cannot."""
    def boom():
        raise OSError("no state dir")
    monkeypatch.setattr("scripts.automod.state.read_pending", boom)
    A.set_drain(True, 30)
    body = json.loads(asyncio.run(A.get_drain()).body)
    assert body["draining"] is True and "pending" not in body
