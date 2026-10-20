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

import json
import re
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


# ── the ruling is recorded at the rung (#3055) ─────────────────────────────
# Whether trace agreement should ever gate — and at which per-feature floor —
# was decided on #828 (2026-10-04, re-ruled 2026-10-20): never, and no floor.
# #828 closed with that ruling on its activity log and nothing in the tree, so
# the code still carried the open question. These nodes pin the recorded
# decision, and pin the figures it rests on to the artifact they were read
# from, so a comment cannot rot into a claim the data no longer supports.

SRC = ROOT / "scripts" / "automod" / "canary_smoke.py"

# The `name_equal=n/d` figures the block states, keyed by feature name.
_STATED_FIGURES = re.compile(r"([a-z_]+)_equal=(\d+)/(\d+)")


def _ruling_block() -> str:
    """The comment block above `TRACE_KEYS`, as prose — the thing a reader of
    the module meets before a single line of the trace code.

    The `# ` markers and the wrapping are stripped: the block is written at
    ~78 columns, so a phrase asserted against the raw bytes would be a test
    that fails when someone reflows a comment and passes when someone deletes
    the ruling's second half mid-line.
    """
    lines = SRC.read_text(encoding="utf-8").splitlines()
    start = next(i for i, ln in enumerate(lines)
                 if ln.startswith("# ── the structural trace"))
    end = next(i for i, ln in enumerate(lines) if ln.startswith("TRACE_KEYS"))
    assert start < end, "the block header or TRACE_KEYS moved; the pin is blind"
    return " ".join(ln.lstrip("#").strip() for ln in lines[start:end]).strip()


def test_the_block_records_the_ruling_instead_of_the_open_question():
    """Clause 1: the question is closed, so the source must not ask it."""
    assert "human call" not in SRC.read_text(encoding="utf-8"), (
        "the open question is back: agreement is decided, not open")

    low = _ruling_block().lower()
    assert "never gates" in low, "the block does not state that agreement gates nothing"
    assert "informational" in low
    assert "no per-feature agreement floor is armed" in low, (
        "the block does not say a floor is unarmed — the half a later reader "
        "would otherwise helpfully supply")
    assert "#3055" in _ruling_block() and "#828" in _ruling_block(), (
        "the ruling must cite the item that made it and the one that recorded it")


def test_the_figures_in_the_block_are_the_figures_the_artifact_gives():
    """Clause 2: the comment's numbers are re-derived here, from the committed
    artifact, over the denominator the comment itself names.

    The block quotes four figures. A figure with no command beside it is a
    caption: it survives the artifact being regenerated and quietly keeps its
    old value. So the denominator is checked against the sample count, the
    numerator against `compare_traces` run now over consecutive pairs, and the
    build the block names against the artifact's own `build` key.
    """
    block = _ruling_block()
    stated = {m.group(1): (int(m.group(2)), int(m.group(3)))
              for m in _STATED_FIGURES.finditer(block)}
    assert set(stated) == {"events", "shape", "response_chars", "turns"}, (
        f"the block's figure set changed shape: {sorted(stated)}")

    artifact = json.loads((ROOT / "eval" / "canary_trace_samples.json").read_text(
        encoding="utf-8"))
    samples = artifact["samples"]
    comparisons = len(samples) - 1        # consecutive pairwise comparisons
    for name, (num, den) in stated.items():
        assert den == comparisons, (
            f"{name}: the block divides by {den}, the artifact holds "
            f"{len(samples)} samples = {comparisons} consecutive comparisons")
    assert artifact["build"] in block, "the block does not name the build it measured"

    diffs = [CS.compare_traces(samples[i], samples[i - 1])
             for i in range(1, len(samples))]
    agree = {
        "events": sum(d["events"]["equal"] for d in diffs),
        "shape": sum(d["events"]["shape_equal"] for d in diffs),
        "response_chars": sum(d["response_chars"]["delta"] == 0 for d in diffs),
        "turns": sum(d["turns"]["delta"] == 0 for d in diffs),
    }
    for name, (num, _) in stated.items():
        assert agree[name] == num, (
            f"{name}: the block says {num}/{comparisons} agree, the artifact "
            f"now gives {agree[name]}/{comparisons}")
    # The reason the ruling is what it is, restated as an assertion rather than
    # prose: strict event equality on one healthy build is zero.
    assert agree["events"] == 0, (
        "same-build strict equality stopped being zero — the ruling's premise "
        "moved, and #3055's block has to be re-read, not re-worded")


def test_the_docstring_states_the_rung_policy_not_just_the_return_shape():
    """Clause 3: `compare_traces` must forbid the rung failure in prose.

    `test_the_comparison_carries_no_verdict` pins the absence of an `ok` key,
    which an implementer can satisfy exactly while arming a floor in
    `gate.py` — the shape says nothing about what a caller may conclude. The
    sentence has to name the rung."""
    doc = CS.compare_traces.__doc__ or ""
    low = " ".join(doc.lower().split())
    assert "informational" in low, doc
    assert re.search(r"must never fail (?:a|the) rung", low), (
        f"the docstring does not state the rung policy: {doc}")
    assert "gate.rung_canary_smoke" in doc, (
        "the docstring must name the caller the prohibition is aimed at")
    assert "four deterministic assertions" in low
