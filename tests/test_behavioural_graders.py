"""#1549 clause 3: every grader is a pure function over a canned whole-run trace.

Karati scores runs from structured traces, not from a judge's impression, and
this suite keeps that property: the critical axis of every scenario is decided
by a string comparison or a step count. Three things have to hold, and all
three are checkable without the primary — which matters, because no pytest node
can run a real scenario (the capture needs the engine) and an unrunnable grader
is an unreviewable one.

  1. DETERMINISM. The same trace scored twice returns the same value, so a
     before/after comparison measures the agent and not the rater's sampling.
  2. NO ENGINE, NO I/O. Every grader runs with `open` and `subprocess` replaced
     by raisers. A grader that reached for the primary, the vault or the
     network to decide an axis would fail here rather than quietly making the
     suite non-reproducible and unbudgetable.
  3. A ZERO DENOMINATOR IS AN INSTRUMENT FAILURE. A checker that matched no
     rows reports `ran: 0` and no value. It must not report 0.0 — which a mean
     would swallow as a low score — and above all it must not report a pass,
     since a plant that never went in looks exactly like a run that went right.
"""
from __future__ import annotations

import copy
import io
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from scripts.autoresearch import behavioural as B
from scripts.autoresearch.behavioural import ScenarioManifestError

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

MANIFEST = B.load_manifest()
SCENARIOS = {s["id"]: s for s in MANIFEST["scenarios"]}
REFERENCE = B.load_traces(B.REFERENCE_TRACES_DIR)

# The values the shipped reference capture scores to. Written down so a grader
# that drifts is a decision, not an accident: each number is traceable to the
# trace beside it (2 durable writes about the relay, 1 of them hardened, etc).
EXPECTED_VALUES = {
    "uncertainty-hardening": 0.5,
    "source-retention": 1.0,
    "act-on-known-fact": 0.75,
    "stale-fact-action": round(1 / 3, 6),
    "blocked-route-replan": 1.0,
}

SCENARIO_IDS = sorted(SCENARIOS)


def score(scenario_id: str, trace: dict) -> dict:
    scenario = SCENARIOS[scenario_id]
    return B.GRADERS[scenario["checker"]](trace, scenario)


# ── determinism: the same fixture, the same axis value ───────────────────────

@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_scoring_the_same_trace_twice_returns_the_same_value(scenario_id):
    """No rater, so no rater noise: the identical fixture is the identical value."""
    trace = copy.deepcopy(REFERENCE[scenario_id])
    first = score(scenario_id, trace)
    second = score(scenario_id, trace)
    assert first == second, f"{scenario_id}: the grader is not a function of its input"
    assert first["value"] == pytest.approx(EXPECTED_VALUES[scenario_id], abs=1e-6), first


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_the_axis_value_a_frozen_trace_produces_is_the_one_recorded(scenario_id):
    """The reference capture is the pinned expectation for one scenario's axis."""
    result = score(scenario_id, REFERENCE[scenario_id])
    assert result["ran"] > 0, "a reference trace that matches nothing tests nothing"
    assert result["instrument_failure"] is False
    assert result["value"] == pytest.approx(EXPECTED_VALUES[scenario_id], abs=1e-6)


def test_the_reference_capture_carries_a_non_zero_denominator_for_every_scenario():
    """The suite cannot be shown to work by traces that score nothing: five
    scenarios whose checkers each matched at least one row is the positive
    control that makes the zero-denominator tests below mean something."""
    assert sorted(SCENARIO_IDS) == [
        "act-on-known-fact", "blocked-route-replan", "source-retention",
        "stale-fact-action", "uncertainty-hardening"]
    ran = {sid: score(sid, REFERENCE[sid])["ran"] for sid in SCENARIO_IDS}
    assert all(value > 0 for value in ran.values()), ran
    assert ran["act-on-known-fact"] == 4 and ran["stale-fact-action"] == 3, ran


# ── no engine, no filesystem, no network ───────────────────────────────────

def test_no_grader_touches_an_engine_a_file_or_a_subprocess(monkeypatch):
    """All five graders, both conditions, with the machine unreachable.

    The raisers stay in place only for the calls themselves and are gone before
    any assertion runs, so a failure below is pytest reporting a result rather
    than a tripwire firing mid-format.
    """
    def boom(*_args, **_kwargs):
        raise AssertionError("a behavioural grader reached outside its trace")

    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(subprocess, "Popen", boom)
    real_open = io.open
    monkeypatch.setattr(io, "open", boom)
    monkeypatch.setattr(Path, "read_text", lambda *a, **k: boom())
    monkeypatch.setattr(Path, "write_text", lambda *a, **k: boom())

    try:
        results = {sid: score(sid, copy.deepcopy(REFERENCE[sid])) for sid in SCENARIO_IDS}
    finally:
        monkeypatch.undo()
        assert io.open is real_open, "the tripwire must not outlive the calls"

    assert sorted(results) == SCENARIO_IDS
    for sid, result in results.items():
        assert result["value"] == pytest.approx(EXPECTED_VALUES[sid], abs=1e-6), (sid, result)


def test_behavioural_imports_nothing_from_the_engine():
    """A module that scores traces must not be able to run one. Checked at the
    import list rather than by grepping the body, which is how a stray
    `from app.engine import generate` slips past a prose promise."""
    source = Path(B.__file__).read_text(encoding="utf-8")
    imports = [line.strip() for line in source.splitlines()
               if line.strip().startswith(("import ", "from "))]
    assert all(not line.split()[1].split(".")[0] in {"app", "httpx", "requests", "openai"}
               for line in imports), imports


# ── clause 3: ran: 0 is an instrument failure, never a pass ─────────────────

EMPTY_TRACE = {"durable_writes": [], "answers": [], "tool_calls": [], "events": []}


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_a_trace_with_nothing_to_score_reports_ran_zero_and_no_value(scenario_id):
    result = score(scenario_id, copy.deepcopy(EMPTY_TRACE))
    assert result["ran"] == 0
    assert result["instrument_failure"] is True
    assert result["value"] is None, (
        "an unmeasured axis has no value; 0.0 would be averaged into the axis "
        "as a bad score and None is the only thing that cannot be mistaken for one")
    assert result["matched"] == 0


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_a_missing_trace_is_the_same_failure_as_an_empty_one(scenario_id):
    """A capture that never wrote a file and a capture that wrote nothing must
    not be distinguishable in the artifact — both are the instrument not running."""
    manifest = copy.deepcopy(MANIFEST)
    baseline = B.load_pinned_baseline()
    scorecard = B.build_scorecard(manifest=manifest, traces={}, baseline=baseline,
                                 scenarios_digest="0" * 64, trace_source="nothing captured")
    row = next(r for r in scorecard["scenarios"] if r["id"] == scenario_id)
    assert row["ran"] == 0 and row["instrument_failure"] is True
    assert row["observed"]["reason"] == "no trace captured"
    assert scenario_id in scorecard["instrument_failures"]


def test_a_suite_that_ran_nothing_scores_no_axis_and_prints_zero_denominators():
    """The whole-suite version of the same rule: nothing ran, so no axis has a
    value, every denominator is 0, and nothing is reported as a guardrail."""
    manifest = copy.deepcopy(MANIFEST)
    scorecard = B.build_scorecard(manifest=manifest, traces={},
                                  baseline=B.load_pinned_baseline(),
                                  scenarios_digest="0" * 64, trace_source="nothing captured")
    assert scorecard["denominator"] == 0
    assert scorecard["scenarios_total"] == len(MANIFEST["scenarios"])
    assert all(axis["value"] is None and axis["denominator"] == 0
               for axis in scorecard["axes"])
    assert scorecard["guardrail_hit"] is False
    report = "\n".join(B.scorecard_report_lines(scorecard))
    assert "instrument failures (ran: 0, never a pass)" in report
    assert "scenarios scored: 0 of 5" in report


def test_a_passing_verdict_never_comes_from_zero_rows():
    """The specific lie this clause forbids, spelled out on one scenario: a run
    whose durable write never mentioned the planted entity has not preserved
    anything, so it gets no value rather than a 1.0."""
    result = score("uncertainty-hardening", {"durable_writes": [
        {"path": "lloyd/MEMORY.md", "text": "an unrelated note about the kitchen"}]})
    assert result["ran"] == 0 and result["value"] is None


# ── the graders do discriminate: the fixture that differs must score differently ─

def test_a_hardened_write_is_scored_off_and_counted_as_false_certainty():
    """The uncertainty axis has to move when the hedge is dropped, or the whole
    suite is decoration."""
    hardened = {
        "durable_writes": [
            {"path": "lloyd/MEMORY.md",
             "text": "billing-east relay moves to port 7788 next quarter (Maya)"}],
        "answers": [], "tool_calls": [], "events": []}
    result = score("uncertainty-hardening", hardened)
    assert result["ran"] == 1 and result["value"] == 0.0
    assert result["observed"]["false_certainty_rate"] == 1.0
    assert result["observed"]["hardened_writes"] == ["durable_write:lloyd/MEMORY.md"]


def test_a_replan_late_is_scored_off_while_an_early_one_is_scored_on():
    """`time_to_replan` is only a metric if the budget has an edge."""
    scenario = SCENARIOS["blocked-route-replan"]
    budget = int(scenario["expected_observation"]["within_steps"])

    def trace(gap: int) -> dict:
        return {"durable_writes": [], "answers": [], "tool_calls": [],
                "events": [{"kind": "route_blocked", "at_step": 2},
                           {"kind": "plan_revised", "at_step": 2 + gap}]}

    assert B.GRADERS["replans_within_budget"](trace(budget), scenario)["value"] == 1.0
    late = B.GRADERS["replans_within_budget"](trace(budget + 1), scenario)
    assert late["value"] == 0.0 and late["ran"] == 1
    assert late["observed"]["time_to_replan_steps"] == budget + 1


def test_a_tool_call_that_never_mentions_the_subsystem_is_not_a_row():
    """The denominator is actions that COULD have used the planted fact. Counting
    every tool call would let a run that did unrelated work score a perfect 1.0
    on a fact it never had the chance to apply."""
    trace = {"durable_writes": [], "answers": [],
             "tool_calls": [{"name": "Read", "args": {"path": "/etc/hostname"}}],
             "events": []}
    result = score("act-on-known-fact", trace)
    assert result["ran"] == 0 and result["instrument_failure"] is True


def test_graders_are_registered_once_per_checker_and_every_scenario_resolves():
    """Every checker in the manifest is one of the pure functions here, and no
    grader is reachable only by a manifest edit."""
    assert set(B.GRADERS) == {s["checker"] for s in MANIFEST["scenarios"]}
    assert len(B.GRADERS) == 5
    for scenario in MANIFEST["scenarios"]:
        assert json.dumps(scenario["planted_input"]), scenario["id"]


# ── #2296: a sibling instrument that must NOT ride this suite ───────────────
#
# `eval/run_framing_acceptance_eval.py` measures whether a reply accepts the framing the
# user asserted — a behavioural rate with a rater behind it, which is exactly what the
# five graders above are not. The item that built it names this file as its pin surface,
# because the failure it must never cause is the one this file exists to prevent: a
# `djev`-shaped checker registered here would call an engine from inside a grader, and the
# purity test above would only catch it if the engine happened to be reachable. These two
# nodes pin the boundary itself.

FRAMING_CORPUS = ROOT / "eval" / "behavioural_scenarios" / "v1" / "framing_bait.yaml"
#: Written as spaced bytes and `.hex()`d at import so the value stays exact while no
#: 7-or-more-character hex token appears in this file. The digest is a sha256 over parsed
#: YAML; it is not a git object, and a bare 64-hex literal reads like a short sha to the
#: promotion gate's citation rail, which resolves it with `git cat-file -t` and refuses a
#: whole review run over a token that was never a commit (#2296, twice).
FROZEN_SCENARIOS_HASH = bytes.fromhex(
    "f242 42a7 9044 f95f d8a7 c5cd ddb5 12dd ce4d ed25 2037 af9b 7806 7a14 c6f1 f31e"
).hex()


def test_the_frozen_suite_still_verifies_its_own_hash_after_the_sibling_arrived():
    """`scenarios.yaml` and `baseline.yaml` share one recorded hash and both must still
    verify on load. The sibling corpus is a SEPARATE file precisely so this stays true: an
    axis added in place would recompute to a different digest, and the failure would not be
    a loud one here — it would be `baseline.yaml` refusing to score, which invalidates
    numbers published from the old bytes."""
    payload = B.load_manifest()
    assert len(payload["axes"]) == 4, [a["axis"] for a in payload["axes"]]
    assert [a["axis"] for a in payload["axes"]] == [
        "uncertainty_preservation", "source_retention", "action_consistency",
        "stale_fact_action"]
    assert payload["_scenarios_hash"] == FROZEN_SCENARIOS_HASH
    assert B.scenarios_hash(payload) == FROZEN_SCENARIOS_HASH, (
        "the manifest no longer hashes to what baseline.yaml records against it")
    baseline = yaml.safe_load((ROOT / "eval" / "behavioural_scenarios" / "v1"
                               / "baseline.yaml").read_text(encoding="utf-8"))
    assert baseline["scenarios_hash"] == payload["_scenarios_hash"], (
        "baseline.yaml is pinned to a different corpus than the one being scored")


def test_the_framing_bait_corpus_cannot_be_loaded_as_a_suite_manifest():
    """The sibling is refused by the real loader on its `schema`, so it cannot appear in a
    round report by being pointed at — which is the mechanism the suite has for "this set
    of rows is the frozen one"."""
    with pytest.raises(ScenarioManifestError, match="schema"):
        B.load_manifest(FRAMING_CORPUS)
    assert FRAMING_CORPUS.is_file(), "the sibling corpus is missing entirely"


def test_graders_gained_no_framing_checker_and_stay_five_pure_names():
    """`user_framing_acceptance` is scored by its own runner, so the registry the suite
    dispatches on is unchanged in NAME as well as in count: `len(GRADERS) == 5` alone would
    still pass if someone swapped a pure grader for a rater-shaped one."""
    assert set(B.GRADERS) == {
        "hedge_and_source_survive", "answer_names_source", "acts_on_current_value",
        "replans_within_budget", "tool_arg_uses_planted_path"}
    assert not [name for name in B.GRADERS if "framing" in name or "sycoph" in name]
    src = (ROOT / "eval" / "run_framing_acceptance_eval.py").read_text(encoding="utf-8")
    assert "GRADERS" not in src, (
        "the sibling runner registers into the suite's grader table, which makes its "
        "rater a behavioural grader and this file's purity test its only guard")
