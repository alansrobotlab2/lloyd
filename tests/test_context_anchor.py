"""`<context>` — telling the model where it is against its own window.

The model has no view of its own prompt size. The engine reports it, the
harness knows it, and until 2026-09-12 nothing carried it across. On
2026-09-11 three autocode rounds died at the wall with finished work
uncommitted in a worktree, still Reading whole files at 241k of 262,144
tokens, because nothing had ever told them to stop.

Fires once per level, and only on a measured meter — an anchor that fired
on iteration 1, before any usage report, would be guessing.
"""

from __future__ import annotations

import asyncio

import pytest

from app.harness.context_meter import ContextMeter
from app.routers.messages import _build_state_anchor


def _anchor(meter=None, **kw):
    return _build_state_anchor("ctx-anchor-test", context_meter=meter, **kw)


def _texts(out: list[dict]) -> str:
    return "\n".join(str(m.get("content") or "") for m in out)


def _measured(window: int, fraction: float) -> ContextMeter:
    m = ContextMeter(window)
    msgs = [{"role": "user", "content": "x"}]
    m.observe_usage({"input_tokens": int(window * fraction)}, len(msgs))
    m.observe_append(msgs)
    return m


def test_silent_without_a_meter():
    """Every other caller of `_build_state_anchor` passes none, and none of
    them may start emitting a block about a number they do not have.
    """
    anchor = _anchor(meter=None)
    for i in range(1, 30):
        assert asyncio.run(anchor(i)) == []


def test_silent_before_the_first_usage_report():
    meter = ContextMeter(262_144)          # never observed
    anchor = _anchor(meter)
    assert "<context>" not in _texts(asyncio.run(anchor(1)))
    assert "<context>" not in _texts(asyncio.run(anchor(9)))


def test_silent_well_below_the_warn_fraction():
    anchor = _anchor(_measured(262_144, 0.40))
    assert "<context>" not in _texts(asyncio.run(anchor(5)))


def test_the_warn_level_fires_once_and_says_stop_reading():
    meter = _measured(262_144, 0.80)
    anchor = _anchor(meter)
    first = _texts(asyncio.run(anchor(5)))
    assert "<context>" in first
    assert "79%" in first or "80%" in first   # int() truncation at the boundary
    # What the model is actually meant to do differently.
    assert "Grep" in first
    assert "offset" in first
    assert "commit" in first.lower()
    # Once only: a warning re-sent every iteration is one the model learns
    # to skip, which is the budget anchor's rule and the same reasoning.
    assert "<context>" not in _texts(asyncio.run(anchor(6)))
    assert "<context>" not in _texts(asyncio.run(anchor(7)))


def test_the_critical_level_says_do_not_re_send_a_cut_call():
    meter = _measured(262_144, 0.93)
    anchor = _anchor(meter)
    text = _texts(asyncio.run(anchor(5)))
    assert "<context>" in text
    assert "CUT OFF" in text
    assert "do not re-send" in text.lower()
    assert "automod_gate" in text
    assert "automod_land" in text


def test_critical_suppresses_a_later_warn():
    """A turn that arrives at the wall in one jump — the common shape when a
    single big Read lands — must not then emit the gentler warning after the
    urgent one.
    """
    meter = _measured(262_144, 0.95)
    anchor = _anchor(meter)
    assert "CUT OFF" in _texts(asyncio.run(anchor(5)))
    # Pressure eases slightly (relief ran); no warn block appears.
    meter.observe_usage({"input_tokens": int(262_144 * 0.80)}, 1)
    meter.observe_append([{"role": "user", "content": "x"}])
    assert "<context>" not in _texts(asyncio.run(anchor(6)))


def test_both_levels_fire_when_pressure_builds_gradually():
    meter = _measured(262_144, 0.78)
    anchor = _anchor(meter)
    assert "<context>" in _texts(asyncio.run(anchor(5)))
    meter.observe_usage({"input_tokens": int(262_144 * 0.92)}, 1)
    meter.observe_append([{"role": "user", "content": "x"}])
    second = _texts(asyncio.run(anchor(6)))
    assert "CUT OFF" in second


def test_the_kill_switch_silences_it(monkeypatch):
    from app.config import CONFIG

    harness = dict(CONFIG.get("harness") or {})
    harness["context_anchor"] = {"enabled": False}
    monkeypatch.setitem(CONFIG, "harness", harness)
    anchor = _anchor(_measured(262_144, 0.95))
    assert "<context>" not in _texts(asyncio.run(anchor(5)))


def test_it_does_not_disturb_the_budget_or_deadline_anchors():
    """Three anchors ride one closure and each fires on its own clock. The
    budget anchor's strings are asserted by `tests/test_autonomy_budget_anchor.py`
    and the deadline's by `tests/test_worker_turn_deadline.py`; this only
    pins that adding a third did not displace them.
    """
    meter = _measured(262_144, 0.10)      # context quiet
    anchor = _anchor(meter, max_turns=10)
    out = _texts(asyncio.run(anchor(8)))   # 80% of the iteration budget
    assert "<budget>" in out
    assert "<context>" not in out


def test_a_meter_that_raises_is_not_fatal():
    """Failing open: a broken meter costs the anchor, never the turn."""

    class _Broken:
        measured = True

        @property
        def fraction(self):
            raise RuntimeError("boom")

    anchor = _anchor(_Broken())
    assert asyncio.run(anchor(5)) == []
