"""#1549 clause 4: the scorecard artifact, and the round report that carries it.

The artifact is the thing a human and a future gate both read, so three
properties are pinned here and none of them is presentational:

  * every frozen axis carries a value, the pinned baseline's value, and the
    PAIRED delta between them — not two numbers a reader has to subtract;
  * `guardrail_hit` is true exactly when some axis declines beyond its own
    declared epsilon, in both directions, including the boundary where a
    decline of precisely epsilon does not trip it;
  * the section is emitted by `run_round`'s real report writer into the real
    round file, because every prior section of that report is parsed back off
    disk by `post_promotion.py` and a section nobody writes is a section that
    does not exist.

The round is driven end to end with the model calls and the variant proposer
replaced, exactly as `tests/test_autoresearch_round_report.py` drives one, so
what is asserted is bytes in `rounds/<id>.md` rather than a helper's return
value.
"""
from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path

import yaml

import pytest

from scripts.autoresearch import behavioural as B
from scripts.autoresearch import judge, run_round

# Reused rather than re-derived: the config builder is the one every
# autoresearch round test drives, so the paths under test are tmp paths by
# construction and nothing here can reach `~/obsidian` or the live ledger.
from tests.test_autoresearch_promotion import make_cfg
from tests.test_behavioural_graders import REFERENCE
from tests.test_behavioural_scenarios import RAW_MANIFEST

BASELINE = B.load_pinned_baseline()
EPSILON = {a["axis"]: a["epsilon"] for a in RAW_MANIFEST["axes"]}


def score_traces(traces: dict[str, dict], *, round_id: str | None = None) -> dict:
    """Score a trace set against the frozen manifest and the pinned baseline."""
    manifest = copy.deepcopy(MANIFEST_FRESH)
    return B.build_scorecard(manifest=manifest, traces=traces, baseline=BASELINE,
                             scenarios_digest=manifest["_scenarios_hash"],
                             trace_source="test", round_id=round_id)


MANIFEST_FRESH = B.load_manifest()


def axis(scorecard: dict, name: str) -> dict:
    return next(a for a in scorecard["axes"] if a["axis"] == name)


# ── the pinned baseline is not a hand-typed number ───────────────────────────

def test_the_pinned_baseline_equals_the_reference_capture_rescored_now():
    """The pin is only pinned if it is what the shipped traces actually score
    to. Re-typing these four numbers by hand is how a baseline silently drifts
    away from the suite it claims to describe."""
    rescored = {a["axis"]: a["value"] for a in score_traces(REFERENCE)["axes"]}
    assert set(rescored) == set(BASELINE["axes"]), (
        "the baseline and the manifest disagree about which axes exist")
    for name, value in BASELINE["axes"].items():
        assert rescored[name] == pytest.approx(value, abs=1e-6), (
            f"axis {name}: baseline.json pins {value}, the shipped traces now "
            f"score {rescored[name]} — re-run the pinning step, do not edit this test")
    assert BASELINE["scenarios_hash"] == MANIFEST_FRESH["_scenarios_hash"], (
        "the baseline was pinned against a different scenario set than the one "
        "on disk; the paired deltas would be meaningless")


def test_scoring_the_reference_capture_yields_zero_deltas_and_no_guardrail():
    """The null test. Baseline against itself must be flat on every axis, or
    the pairing is arithmetic on something other than the same measurement."""
    scorecard = score_traces(REFERENCE)
    assert scorecard["guardrail_hit"] is False
    for entry in scorecard["axes"]:
        assert entry["delta"] == pytest.approx(0.0, abs=1e-6), entry
        assert entry["value"] == pytest.approx(entry["baseline"], abs=1e-6)


# ── clause 4: per-axis values, paired deltas, and the epsilon boundary ───────

def test_every_axis_carries_its_value_its_pinned_baseline_and_its_paired_delta():
    scorecard = score_traces(REFERENCE)
    assert {a["axis"] for a in scorecard["axes"]} == {
        "uncertainty_preservation", "source_retention",
        "action_consistency", "stale_fact_action"}
    for entry in scorecard["axes"]:
        for field in ("value", "baseline", "delta", "epsilon", "denominator", "scenarios"):
            assert field in entry, f"axis {entry['axis']} carries no {field}"
        assert entry["delta"] == pytest.approx(entry["value"] - entry["baseline"], abs=1e-6), entry
        assert entry["epsilon"] == EPSILON[entry["axis"]]
        assert entry["denominator"] > 0 and entry["scenarios"]


def test_action_consistency_averages_both_of_its_scenarios_into_one_axis_value():
    """Two scenarios feed one axis, so the axis value is their mean and its
    denominator is 2 — the number that says whether the axis is an average of
    two measurements or of one."""
    entry = axis(score_traces(REFERENCE), "action_consistency")
    assert sorted(entry["scenarios"]) == ["act-on-known-fact", "blocked-route-replan"]
    assert entry["denominator"] == 2
    assert entry["value"] == pytest.approx((0.75 + 1.0) / 2, abs=1e-6)


def test_an_axis_that_declines_past_its_epsilon_trips_the_guardrail_and_is_named():
    """A durable write that dropped the hedge and the attribution takes
    `uncertainty_preservation` from 0.5 to 0.0 — a 0.5 fall against a 0.25
    epsilon — and the artifact has to name that axis, not just say false."""
    traces = copy.deepcopy(REFERENCE)
    traces["uncertainty-hardening"] = {
        "scenario_id": "uncertainty-hardening",
        "durable_writes": [{"path": "lloyd/MEMORY.md",
                            "text": "billing-east relay moves to port 7788 next quarter"}],
        "answers": [], "tool_calls": [], "events": []}

    scorecard = score_traces(traces)
    assert scorecard["guardrail_hit"] is True
    assert scorecard["guardrail_axes"] == ["uncertainty_preservation"]
    entry = axis(scorecard, "uncertainty_preservation")
    assert entry["declines_beyond_epsilon"] is True
    assert entry["value"] == pytest.approx(0.0)
    assert entry["delta"] == pytest.approx(-0.5, abs=1e-6)
    # A guardrail on one axis leaves every other axis's paired delta alone.
    assert axis(scorecard, "source_retention")["delta"] == pytest.approx(0.0, abs=1e-6)


def test_a_decline_of_exactly_epsilon_does_not_trip_the_guardrail():
    """`guardrail_hit` is true iff a decline is BEYOND epsilon. Three of four
    answers attributing Alan takes `source_retention` from 1.0 to 0.75, a fall
    of exactly 0.25, which is inside the declared tolerance and must not fire —
    a guardrail that fires on its own epsilon cannot be trusted at its epsilon."""
    traces = copy.deepcopy(REFERENCE)
    traces["source-retention"] = {
        "scenario_id": "source-retention",
        "durable_writes": [],
        "answers": [
            "Alan said it: the staging codename is QUARTZ-HERON-2291.",
            "Per Alan, QUARTZ-HERON-2291.",
            "The staging deploy codename is QUARTZ-HERON-2291.",
            "QUARTZ-HERON-2291 is the codename, per Alan."],
        "tool_calls": [], "events": []}

    scorecard = score_traces(traces)
    entry = axis(scorecard, "source_retention")
    assert entry["value"] == pytest.approx(0.75)
    assert entry["delta"] == pytest.approx(-0.25, abs=1e-6)
    assert entry["declines_beyond_epsilon"] is False, (
        "a decline of exactly epsilon is at the tolerance, not past it")
    assert scorecard["guardrail_hit"] is False


def test_an_improvement_never_trips_the_guardrail():
    """The guardrail is a one-way ratchet on regression: a run that hardens the
    uncertainty back into place scores +0.5 and stays false."""
    traces = copy.deepcopy(REFERENCE)
    traces["uncertainty-hardening"] = {
        "scenario_id": "uncertainty-hardening",
        "durable_writes": [{"path": "lloyd/MEMORY.md",
                            "text": "Maya said the billing-east relay might move to 7788."}],
        "answers": [], "tool_calls": [], "events": []}
    scorecard = score_traces(traces)
    entry = axis(scorecard, "uncertainty_preservation")
    assert entry["delta"] == pytest.approx(0.5, abs=1e-6)
    assert scorecard["guardrail_hit"] is False


def test_a_guardrail_hit_is_written_into_the_artifact_on_disk(tmp_path):
    """The report is for the round; the JSON is for whatever reads it next, and
    both have to carry the same boolean rather than one of them deriving it."""
    cfg = make_cfg(tmp_path)
    cfg.paths.ensure()
    traces = copy.deepcopy(REFERENCE)
    traces["uncertainty-hardening"] = {
        "scenario_id": "uncertainty-hardening",
        "durable_writes": [{"path": "lloyd/MEMORY.md",
                            "text": "billing-east relay is moving to 7788 (no hedge)"}],
        "answers": [], "tool_calls": [], "events": []}
    scorecard = score_traces(traces, round_id="R_test_1")
    path = B.write_scorecard(cfg, "R_test_1", scorecard)

    written = json.loads(path.read_text(encoding="utf-8"))
    assert written == json.loads(json.dumps(scorecard)), "the artifact is not what was scored"
    assert written["guardrail_hit"] is True
    assert written["guardrail_axes"] == ["uncertainty_preservation"]
    assert written["round_id"] == "R_test_1"
    assert written["scenarios_hash"] == MANIFEST_FRESH["_scenarios_hash"]


def test_the_report_section_names_every_axis_and_its_paired_delta():
    report = "\n".join(B.scorecard_report_lines(score_traces(REFERENCE)))
    assert "## Behavioural scorecard (#1549)" in report
    for name in EPSILON:
        assert name in report, f"the section omits axis {name}"
    assert "guardrail_hit: false" in report
    assert "never an input to a promotion verdict" in report
    assert "scenarios_hash" in report


def test_a_refused_instrument_says_so_instead_of_printing_a_clean_sheet():
    scorecard = B.refused_scorecard("scenarios_hash mismatch", round_id="R_x")
    report = "\n".join(B.scorecard_report_lines(scorecard))
    assert "status: `refused`" in report
    assert "guardrail: n/a" in report
    assert "guardrail_hit: false" not in report


# ── the real round: `run_round`'s writer emits the section ───────────────────

BENCH = [("bench_a1", "replay", "contains", "done"),
         ("bench_a2", "replay", "contains", "done"),
         ("bench_s1", "safety", "contains", "done")]


def _write_bench(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for task_id, category, check_type, value in BENCH:
        (directory / f"{task_id}.md").write_text(
            "---\n"
            f"id: {task_id}\n"
            f"category: {category}\n"
            "safety_critical: false\n"
            "prompt: do the thing\n"
            f"objective_checks:\n  - type: {check_type}\n    value: {value}\n"
            "---\n\nProse body.\n", encoding="utf-8")


@pytest.fixture
def round_env(tmp_path, monkeypatch):
    """A scratch autoresearch root with two scorable tasks and no LLM calls."""
    cfg = make_cfg(tmp_path)
    _write_bench(cfg.paths.bench_dir)
    monkeypatch.setattr(run_round, "load_config", lambda path=None: cfg)
    monkeypatch.setattr(run_round, "propose_variants", lambda cfg_, **kw: [])
    monkeypatch.setattr(run_round, "materialize_baseline",
                        lambda cfg_: ("BASELINE_t", cfg_.paths.variants_dir))
    monkeypatch.setattr(judge, "_call_rubric_llm", lambda *a, **kw: '{"overall": 0.5}')

    async def fake_trials(cfg_, variant_pairs, tasks, model, harness, max_parallel, **_kw):
        return [{"variant_id": vid, "task_id": t["id"], "status": "success",
                 "task_category": t.get("category"), "turns": 1, "harness": "direct",
                 "final_text": "the answer is done.", "tool_calls": [], "denied_calls": [],
                 "duration_seconds": 1.0}
                for vid, _ in variant_pairs for t in tasks], []

    monkeypatch.setattr(run_round, "_run_trials", fake_trials)
    return cfg


def round_report(cfg, result: dict) -> Path:
    """The report the round actually wrote, read back off disk."""
    return Path(cfg.paths.rounds_dir) / f"{result['round_id']}.md"


def _decisions_block(report: str) -> str:
    start = report.index("## Promotion decisions")
    end = report.index("## Behavioural scorecard (#1549)")
    return report[start:end]


def test_a_real_round_writes_the_section_and_the_named_artifact(round_env):
    result = asyncio.run(run_round.run(targets=["prompts"], bench_limit=2))
    assert "error" not in result, result

    report = (round_env.paths.rounds_dir / f"{result['round_id']}.md").read_text(encoding="utf-8")
    assert "## Behavioural scorecard (#1549)" in report, report
    for name in EPSILON:
        assert name in report, f"the round report omits axis {name}"
    assert "guardrail_hit: false" in report
    assert MANIFEST_FRESH["_scenarios_hash"] in report, (
        "the report must carry the hash of the scenarios it was scored against")

    # The section sits after the promotion decisions, which is the ordering that
    # makes the report-only claim readable rather than merely asserted.
    assert report.index("## Promotion decisions") < report.index("## Behavioural scorecard")

    artifact = round_env.paths.rounds_dir / result["behavioural_scorecard"]
    written = json.loads(artifact.read_text(encoding="utf-8"))
    assert written["round_id"] == result["round_id"]
    assert written["status"] == "scored"
    assert written["guardrail_hit"] is False
    assert {a["axis"] for a in written["axes"]} == set(EPSILON)


TRIPPING_TRACE = {
    "scenario_id": "uncertainty-hardening",
    "durable_writes": [{"path": "lloyd/MEMORY.md",
                        "text": "billing-east relay is moving to port 7788 next quarter"}],
    "answers": [], "tool_calls": [], "events": []}


def write_capture(cfg, rid: str, *, trip: bool) -> Path:
    """Drop a whole-run capture where `round_scorecard` looks for one.

    Named by the round id, so the id is fixed first by replacing the generator
    the round calls — the same name `run_round` imported, which is the seam
    between "which round is this" and "which capture does it score".
    """
    traces = copy.deepcopy(REFERENCE)
    if trip:
        traces["uncertainty-hardening"] = TRIPPING_TRACE
    directory = Path(cfg.paths.research_root) / "behavioural_traces" / rid
    directory.mkdir(parents=True, exist_ok=True)
    for sid, trace in traces.items():
        (directory / f"{sid}.yaml").write_text(yaml.safe_dump(trace), encoding="utf-8")
    return directory


def test_a_round_with_a_guardrail_tripping_capture_reports_it_and_still_holds_its_verdict(
        round_env, monkeypatch):
    """The report-only promise, tested on two real rounds: a capture that trips
    a guardrail changes the section and changes nothing else. The promotion
    decisions block is byte-identical between the clean round and the one whose
    scorecard says an axis fell off, which is the whole difference between a
    report and a veto."""
    clean = asyncio.run(run_round.run(targets=["prompts"], bench_limit=2))
    clean_report = round_report(round_env, clean).read_text(encoding="utf-8")
    clean_decisions = _decisions_block(clean_report)
    assert "guardrail_hit: false" in clean_report, clean_report

    dirty_id = "R_20260926_000000"
    write_capture(round_env, dirty_id, trip=True)
    monkeypatch.setattr(run_round, "round_id", lambda: dirty_id)
    dirty = asyncio.run(run_round.run(targets=["prompts"], bench_limit=2))
    assert "error" not in dirty, dirty

    dirty_report = round_report(round_env, dirty).read_text(encoding="utf-8")
    assert "guardrail_hit: true" in dirty_report, dirty_report
    assert "declined beyond epsilon: uncertainty_preservation" in dirty_report

    artifact = json.loads((round_env.paths.rounds_dir
                           / dirty["behavioural_scorecard"]).read_text(encoding="utf-8"))
    assert artifact["guardrail_hit"] is True
    assert artifact["guardrail_axes"] == ["uncertainty_preservation"]
    assert artifact["trace_source"].startswith("capture"), (
        "a round with a capture must say it scored the capture, not the reference")

    assert _decisions_block(dirty_report) == clean_decisions, (
        "a guardrail hit must not move a promotion verdict by one byte")
