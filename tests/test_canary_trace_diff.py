"""The canary smoke's structural trace: persisted on its own, compared as data (#828).

The gate's `canary_smoke` rung already drives one real agent turn per round;
its trace (event sequence, event count, turns, response length) used to be
thrown away at the rung's boundary, so nothing could say whether a build's
turn looked like the last one's. These tests pin the two halves that make it
comparable: the baseline file the gate writes and reads back, and the pure
comparison, which reports per-feature agreement and never a verdict.

Clause 5 is the committed artifact `eval/canary_trace_samples.json`: ten
repeats of the canary turn against one canary build (a real `Canary` booted
from one commit, the gate's own `smoke()`), read by the last test here.
"""
from __future__ import annotations

from pathlib import Path

from scripts.automod import canary_smoke as CS
from scripts.automod import gate as G
from scripts.automod import state as S


ROOT = Path(__file__).resolve().parent.parent

EVENTS = ["session", "thinking_delta", "tool_start", "tool_complete",
          "text_delta", "text_delta", "done"]


def _trace(events=EVENTS, turns=2, chars=40):
    return {"events": list(events), "event_count": len(events),
            "turns": turns, "response_chars": chars}


# ── clause 3: the comparison is data, never a verdict ─────────────────────

def test_identical_traces_agree_on_every_feature():
    cmp = CS.compare_traces(_trace(), _trace())
    assert cmp["has_baseline"] is True
    assert cmp["events"] == {"equal": True, "shape_equal": True,
                             "missing": {}, "extra": {}}
    for key in ("event_count", "turns", "response_chars"):
        assert cmp[key]["delta"] == 0, key


def test_the_comparison_reports_each_delta():
    cand = _trace(events=EVENTS[:-1] + ["text_delta", "done"], turns=3, chars=52)
    cmp = CS.compare_traces(cand, _trace())
    assert cmp["events"]["equal"] is False
    # A longer run of one event is the same shape.
    assert cmp["events"]["shape_equal"] is True
    assert cmp["events"]["extra"] == {"text_delta": 1}
    assert cmp["event_count"] == {"candidate": 8, "baseline": 7, "delta": 1}
    assert cmp["turns"] == {"candidate": 3, "baseline": 2, "delta": 1}
    assert cmp["response_chars"] == {"candidate": 52, "baseline": 40, "delta": 12}


def test_a_missing_tool_start_is_an_events_mismatch_both_ways():
    no_tool = [e for e in EVENTS if e != "tool_start"]
    fwd = CS.compare_traces(_trace(), _trace(events=no_tool))
    assert fwd["events"]["equal"] is False and fwd["events"]["shape_equal"] is False
    assert fwd["events"]["extra"] == {"tool_start": 1}
    back = CS.compare_traces(_trace(events=no_tool), _trace())
    assert back["events"]["missing"] == {"tool_start": 1}


def test_the_comparison_carries_no_verdict():
    for base in (_trace(), _trace(events=["done"], turns=9, chars=1), None):
        cmp = CS.compare_traces(_trace(), base)
        assert not {"ok", "pass", "passed", "verdict", "regression"} & set(cmp)


def test_no_baseline_leaves_every_comparison_empty():
    cmp = CS.compare_traces(_trace(), None)
    assert cmp["has_baseline"] is False
    assert cmp["events"]["equal"] is None
    assert cmp["event_count"] == {"candidate": 7, "baseline": None, "delta": None}
    assert cmp["turns"]["delta"] is None and cmp["response_chars"]["delta"] is None


def test_a_missing_turn_count_is_no_delta_not_a_crash():
    cmp = CS.compare_traces(_trace(turns=None), _trace())
    assert cmp["turns"] == {"candidate": None, "baseline": 2, "delta": None}
    assert "turns None (no baseline)" in CS.format_comparison(cmp)


def test_trace_of_reads_the_smoke_report():
    rep = {"events": ["session", "done"], "turns": 2, "response_chars": 9,
           "errors": [], "ok": True}
    assert CS.trace_of(rep) == {"events": ["session", "done"], "event_count": 2,
                                "turns": 2, "response_chars": 9}
    assert CS.trace_of({}) == {"events": [], "event_count": 0,
                               "turns": None, "response_chars": None}


# ── clause 2: its own baseline file, read back next round, LKG untouched ──

class _FakeCanary:
    def __init__(self, report):
        self.report = report

    def smoke(self, timeout=150.0):
        return self.report


def _report(events, turns=2, chars=40):
    return {"ok": True, "errors": [], "events": list(events), "turns": turns,
            "response_chars": chars, "duration_s": 5.0,
            "sentinel_in_response": True}


def test_the_path_resolves_at_call_time_and_is_not_the_lkg(monkeypatch, tmp_path):
    monkeypatch.setattr(S, "STATE_DIR", tmp_path)
    assert S.canary_trace_path() == tmp_path / "canary_trace.json"
    assert S.canary_trace_path().name != S.LKG_PATH.name


def test_the_gate_persists_and_reads_back_the_previous_trace(monkeypatch, tmp_path):
    monkeypatch.setattr(S, "STATE_DIR", tmp_path)
    lkg = tmp_path / "last_known_good.json"
    lkg.write_text('{"commit": "deadbeef", "schema": 1}\n', encoding="utf-8")
    monkeypatch.setattr(S, "LKG_PATH", lkg)
    before = lkg.read_bytes()

    first = G.Gate("SM_TRACE_ONE", ROOT, "HEAD")
    first._canary = _FakeCanary(_report(EVENTS, turns=2, chars=40))
    ok, _, data = first.rung_canary_smoke()
    assert ok is True and data["trace_diff"]["has_baseline"] is False

    stored = S.read_canary_trace()
    assert stored["round_id"] == "SM_TRACE_ONE"
    assert stored["trace"]["events"] == EVENTS
    assert stored["trace"]["duration_s"] == 5.0

    second = G.Gate("SM_TRACE_TWO", ROOT, "HEAD")
    second._canary = _FakeCanary(_report(EVENTS + ["text_delta"], turns=3, chars=41))
    ok, _, data = second.rung_canary_smoke()
    assert ok is True
    assert data["trace_baseline"]["round_id"] == "SM_TRACE_ONE"
    assert data["trace_diff"]["event_count"]["delta"] == 1
    assert data["trace_diff"]["turns"]["delta"] == 1
    assert data["trace_diff"]["response_chars"]["delta"] == 1
    assert S.read_canary_trace()["round_id"] == "SM_TRACE_TWO"

    assert lkg.read_bytes() == before, "the guardian is the LKG's only writer"
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "canary_trace.json", "last_known_good.json"]


def test_an_unreadable_baseline_reads_as_none(monkeypatch, tmp_path):
    monkeypatch.setattr(S, "STATE_DIR", tmp_path)
    (tmp_path / "canary_trace.json").write_text("{not json", encoding="utf-8")
    assert S.read_canary_trace() is None
    (tmp_path / "canary_trace.json").write_text("[1, 2]", encoding="utf-8")
    assert S.read_canary_trace() is None


def test_same_build_samples_are_committed_with_their_four_features():
    """Clause 5: >=10 repeats against one build, each carrying the four
    structural features and its duration, at a path git does not ignore."""
    import json
    import subprocess
    path = ROOT / "eval" / "canary_trace_samples.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = data["samples"]
    assert len(samples) >= 10
    assert data["build"] and len(data["build"]) == 40
    for s in samples:
        assert set(CS.TRACE_KEYS) <= set(s), s.keys()
        assert isinstance(s["events"], list) and s["event_count"] == len(s["events"])
        assert isinstance(s["duration_s"], (int, float))
    ignored = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "-q",
                              "eval/canary_trace_samples.json"])
    assert ignored.returncode == 1, "the samples must not be gitignored"
