"""A skipped gate rung is recorded, and the skip is refused while the engine is up.

Why this file exists
--------------------
`skip_smoke` exists so the gate can run on a machine with no vLLM. Nothing
enforced that. On 2026-09-08, round SM_20260908_165950 passed the flag while
the engine was answering — the same turn then drove four bench tasks and a
20-query eval through that endpoint — and because the rung was *omitted from
the ladder* rather than recorded as skipped, the promotion record for a 66%
rewrite of the operating contract listed seven rungs with nothing to say the
eighth had been waived. A reader had to already know the eighth existed.

Two properties, and the second is the one that keeps the first honest:
a skip must be visible, and a skip must be justified.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.automod import gate as G


ROOT = Path(__file__).resolve().parent.parent


class _StubGate(G.Gate):
    """A Gate that runs no rung, so the ladder's shape can be asserted alone."""

    def __init__(self, **kw):
        super().__init__("SM_TEST_SMOKE", ROOT, "HEAD", **kw)


def test_canary_smoke_is_always_on_the_ladder(monkeypatch):
    """Omitting the rung is what made a skip invisible."""
    monkeypatch.setattr(G, "engine_reachable", lambda root, timeout=4.0: (False, "stub"))
    names = []
    g = _StubGate(skip_smoke=True)
    monkeypatch.setattr(g, "_rung", lambda name, fn: names.append(name) or False)
    g.run()
    # Short-circuits at the first rung, so assert on what was *built*, not run:
    # re-run with a _rung that records and passes everything.
    names.clear()
    monkeypatch.setattr(g, "_rung", lambda name, fn: names.append(name) or True)
    g.run()
    assert "canary_smoke" in names
    assert names.index("canary_smoke") == names.index("drill") - 1


def test_a_skipped_smoke_rung_says_so_and_carries_the_flag(monkeypatch):
    monkeypatch.setattr(G, "engine_reachable", lambda root, timeout=4.0: (False, "refused"))
    g = _StubGate(skip_smoke=True)
    monkeypatch.setattr(g, "_rung", lambda name, fn: True)
    g.run()  # resolves the skip reason
    ok, detail, data = g.rung_canary_smoke()
    assert ok is True
    assert "SKIPPED" in detail
    assert data["skipped"] is True


def test_the_skip_is_refused_while_the_engine_answers(monkeypatch):
    """The flag is for a missing engine, not for a round in a hurry."""
    monkeypatch.setattr(G, "engine_reachable",
                        lambda root, timeout=4.0: (True, "http://x/v1/models answered 200"))
    g = _StubGate(skip_smoke=True)
    monkeypatch.setattr(g, "_rung", lambda name, fn: True)
    g.run()
    assert g.skip_smoke is False
    assert "answered 200" in g._smoke_skip_refused


def test_an_unreachable_engine_keeps_the_skip(monkeypatch):
    monkeypatch.setattr(G, "engine_reachable",
                        lambda root, timeout=4.0: (False, "URLError: refused"))
    g = _StubGate(skip_smoke=True)
    monkeypatch.setattr(g, "_rung", lambda name, fn: True)
    g.run()
    assert g.skip_smoke is True
    assert g._smoke_skip_refused == ""


def test_the_ledger_event_carries_the_skip(monkeypatch):
    """`promoted` events must be filterable on it without parsing the detail string."""
    events = []
    monkeypatch.setattr(G.S, "append_event", lambda e: events.append(e))
    g = _StubGate()
    g._rung("canary_smoke", lambda: (True, "SKIPPED (engine unreachable: x)",
                                     {"skipped": True}))
    # The pytest counts, verbatim, as `rung_tests` returns them. A `skipped`
    # count of 3 here used to record the whole rung as skipped.
    g._rung("tests", lambda: (True, "2132 passed, 3 skipped",
                              {"passed": 2132, "tests_skipped": 3,
                               "collected": 2135}))
    g._rung("venv", lambda: (True, "requirements unchanged", {"skipped": True}))
    assert events[0]["skipped"] is True
    assert events[1]["skipped"] is False, "a skipped TEST is not a skipped rung"
    assert events[2]["skipped"] is True


def test_engine_reachable_reads_the_configured_endpoint(tmp_path):
    """No hardcoded port: a slot that moves must not silently read as 'down'."""
    (tmp_path / "config.yaml").write_text(
        "model:\n  default: primary\nmodels:\n  primary:\n"
        "    base_url: http://127.0.0.1:1\n", encoding="utf-8")
    up, why = G.engine_reachable(tmp_path, timeout=1.0)
    assert up is False
    assert "127.0.0.1:1" in why or "Error" in why


def test_engine_reachable_says_so_when_config_names_no_endpoint(tmp_path):
    (tmp_path / "config.yaml").write_text("model:\n  default: primary\n", encoding="utf-8")
    up, why = G.engine_reachable(tmp_path, timeout=1.0)
    assert up is False and "no base_url" in why


def test_the_mcp_tool_description_states_the_refusal():
    """The model reads this string, and it is the only place the rule is stated to it."""
    src = (ROOT / "agent_mcp" / "automod.py").read_text(encoding="utf-8")
    assert "Honoured" in src and "unreachable" in src
