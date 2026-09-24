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
    """`promoted` events must be filterable on it without parsing the detail string.

    Since 2026-09-24 a skipped `canary_smoke` or `venv` writes no ledger row
    at all (`Gate.QUIET_SKIP_RUNGS`; the report still lists it, which is where
    the smoke skip's visibility lives). `drill` is always written, so it is
    the rung that carries the flag here."""
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
    g._rung("drill", lambda: (True, "no protected paths", {"skipped": True}))
    assert [e["rung"] for e in events] == ["tests", "drill"]
    assert events[0]["skipped"] is False, "a skipped TEST is not a skipped rung"
    assert events[1]["skipped"] is True
    assert [r.name for r in g.report.rungs] == ["canary_smoke", "tests", "venv", "drill"]
    assert g.report.rungs[0].data["skipped"] is True, "the report keeps the skip"


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


# ── the smoke's structural trace (#828) ───────────────────────────────────

def _smoke_report(events, turns=2, chars=40, ok=True):
    return {"ok": ok, "errors": [] if ok else ["boom"], "events": list(events),
            "turns": turns, "response_chars": chars, "duration_s": 6.1,
            "tool_called": True, "tool_result_ok": True, "done": True,
            "sentinel_in_response": True}


_EVENTS = ["session", "thinking_delta", "thinking_delta", "tool_start",
           "tool_complete", "text_delta", "text_delta", "done"]


class _FakeCanary:
    def __init__(self, report):
        self.report = report

    def smoke(self, timeout=150.0):
        return self.report


def _gate_with_smoke(monkeypatch, tmp_path, report):
    monkeypatch.setattr(G.S, "STATE_DIR", tmp_path)
    g = _StubGate()
    g._canary = _FakeCanary(report)
    return g


def test_a_run_smoke_records_the_structural_trace(monkeypatch, tmp_path):
    g = _gate_with_smoke(monkeypatch, tmp_path, _smoke_report(_EVENTS, turns=2, chars=40))
    ok, detail, data = g.rung_canary_smoke()
    assert ok is True
    assert data["events"] == _EVENTS
    assert data["event_count"] == len(_EVENTS)
    assert data["turns"] == 2
    assert data["response_chars"] == 40
    assert data["has_trace"] is True


def test_a_failed_smoke_still_records_its_trace(monkeypatch, tmp_path):
    g = _gate_with_smoke(monkeypatch, tmp_path, _smoke_report(_EVENTS[:3], ok=False))
    ok, _, data = g.rung_canary_smoke()
    assert ok is False
    assert data["events"] == _EVENTS[:3] and data["event_count"] == 3
    assert data["has_trace"] is True
    # A failing run is not a baseline anything should be compared against.
    assert not (tmp_path / "canary_trace.json").exists()


def test_a_skipped_smoke_says_it_produced_no_trace(monkeypatch):
    monkeypatch.setattr(G, "engine_reachable", lambda root, timeout=4.0: (False, "refused"))
    g = _StubGate(skip_smoke=True)
    monkeypatch.setattr(g, "_rung", lambda name, fn: True)
    g.run()
    _, _, data = g.rung_canary_smoke()
    assert data["has_trace"] is False and data["no_trace_reason"] == "skipped"
    assert "events" not in data and "response_chars" not in data


def test_a_reused_smoke_says_it_produced_no_trace(monkeypatch):
    monkeypatch.setattr(G.S, "append_event", lambda e: None)
    g = _StubGate()
    cached = {"duration_s": 6.0, "events": _EVENTS, "event_count": 8, "turns": 2,
              "response_chars": 40, "has_trace": True, "trace_diff": {"x": 1},
              "trace_baseline": {"round_id": "SM_OLD"}}
    monkeypatch.setattr(g, "_reuse", lambda name: (dict(cached), "only tests and docs"))

    def _must_not_run():
        raise AssertionError("a reused rung must not run")
    assert g._rung("canary_smoke", _must_not_run) is True
    data = g.report.rungs[-1].data
    assert data["reused"] is True
    assert data["has_trace"] is False and data["no_trace_reason"] == "reused"
    for key in ("events", "event_count", "turns", "response_chars",
                "trace_diff", "trace_baseline"):
        assert key not in data, key


def test_a_differing_trace_passes_and_the_detail_names_each_number(monkeypatch, tmp_path):
    """Clause 4: the four assertions held, so the rung passes however the
    trace moved; the detail carries numbers, not a bare pass."""
    G.S.write_canary_trace({"round_id": "SM_PREV", "head": "abc",
                            "trace": {"events": _EVENTS[:4], "event_count": 4,
                                      "turns": 3, "response_chars": 55}},
                           path=tmp_path / "canary_trace.json")
    g = _gate_with_smoke(monkeypatch, tmp_path, _smoke_report(_EVENTS, turns=2, chars=40))
    ok, detail, data = g.rung_canary_smoke()
    assert ok is True
    assert "vs SM_PREV" in detail
    assert "events equal=no" in detail
    assert "event_count 8 (delta +4)" in detail
    assert "turns 2 (delta -1)" in detail
    assert "response_chars 40 (delta -15)" in detail
    assert data["trace_diff"]["events"]["equal"] is False
    assert data["trace_baseline"]["round_id"] == "SM_PREV"


def test_a_baseline_missing_tool_start_is_an_events_mismatch(monkeypatch, tmp_path):
    no_tool = [e for e in _EVENTS if e != "tool_start"]
    G.S.write_canary_trace({"round_id": "SM_PREV",
                            "trace": {"events": no_tool, "event_count": len(no_tool),
                                      "turns": 2, "response_chars": 40}},
                           path=tmp_path / "canary_trace.json")
    g = _gate_with_smoke(monkeypatch, tmp_path, _smoke_report(_EVENTS))
    ok, detail, data = g.rung_canary_smoke()
    assert ok is True
    ev = data["trace_diff"]["events"]
    assert ev["equal"] is False
    assert ev["extra"] == {"tool_start": 1} and ev["missing"] == {}
    assert "+1 tool_start" in detail


def test_no_previous_trace_says_so_and_still_passes(monkeypatch, tmp_path):
    g = _gate_with_smoke(monkeypatch, tmp_path, _smoke_report(_EVENTS))
    ok, detail, data = g.rung_canary_smoke()
    assert ok is True
    assert "no previous trace" in detail
    assert "event_count 8 (no baseline)" in detail
    assert data["trace_diff"]["has_baseline"] is False
    assert data["trace_baseline"] is None


def test_an_unwritable_baseline_costs_the_comparison_not_the_rung(monkeypatch, tmp_path):
    g = _gate_with_smoke(monkeypatch, tmp_path, _smoke_report(_EVENTS))

    def boom(*a, **k):
        raise OSError("read-only")
    monkeypatch.setattr(G.S, "read_canary_trace", boom)
    monkeypatch.setattr(G.S, "write_canary_trace", boom)
    ok, _, data = g.rung_canary_smoke()
    assert ok is True and data["has_trace"] is True
