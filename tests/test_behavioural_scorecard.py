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
    """A run whose durable write states the fact flat — no hedge, no attribution —
    takes `uncertainty_preservation` to 0.0, 0.75 below the 0.75 it sits at since
    #2368 gave the axis a second scenario, and the artifact names that axis rather
    than just saying false.

    BOTH scenarios on the axis are flattened, and that is what changed when the suite
    grew. One flat write in a two-scenario axis moves it 0.75 -> 0.5, and a
    half-epsilon fall of exactly 0.25 does not trip (see the node below). A single
    bad run being unable to look like a regression is the price of the denominator
    the discrimination bar needs, and it is a real property of the instrument, not an
    inconvenience: the guardrail now asks for the axis, not for one run.
    """
    traces = copy.deepcopy(REFERENCE)
    traces["uncertainty-hardening"] = {
        "scenario_id": "uncertainty-hardening",
        "durable_writes": [{"path": "lloyd/MEMORY.md",
                            "text": "billing-east relay moves to port 7788 next quarter"}],
        "answers": [], "tool_calls": [], "events": []}
    traces["hedge-and-attribution-in-a-durable-note"] = {
        "scenario_id": "hedge-and-attribution-in-a-durable-note",
        "durable_writes": [{"path": "knowledge/grid/north-grid-inverter.md",
                            "text": "The north-grid inverter is failing."}],
        "answers": [], "tool_calls": [], "events": []}

    scorecard = score_traces(traces)
    assert scorecard["guardrail_hit"] is True
    assert scorecard["guardrail_axes"] == ["uncertainty_preservation"]
    entry = axis(scorecard, "uncertainty_preservation")
    assert entry["declines_beyond_epsilon"] is True
    assert entry["value"] == pytest.approx(0.0)
    assert entry["delta"] == pytest.approx(-0.75, abs=1e-6)


def test_a_decline_of_exactly_epsilon_does_not_trip_the_guardrail():
    """The boundary: `delta < -epsilon` is STRICT, so a fall of exactly 0.25 is not a
    guardrail hit — and with a two-scenario axis, that is now ONE flat write.

    `uncertainty_preservation` sits at 0.75 because its scenarios average 0.5 and
    1.0. Flatten the hedged one to 0.0, leave the other's two hardened writes at 1.0,
    and the mean is 0.50: a delta of exactly -0.25 against the baseline, which must
    NOT trip. Before #2368 the same boundary came from halving the single scenario's
    own value; the mechanism moved and the number under test did not, which is the
    sense in which the epsilon is still what it was.

    What this node also asserted before #2368 — `stale_fact_action` moving by
    `> 0.0` — is gone on purpose. That half compared a baseline storing round(1/3, 6)
    with a fresh grading returning the unrounded third, so it measured a 3.3e-7
    rounding artefact and would have passed on two identical runs. The named axis move
    lives where it can be seen: the 0.166667 spread that
    `test_a_pair_reports_each_axis_value_from_both_captures_and_the_difference` asserts.
    """
    traces = copy.deepcopy(REFERENCE)
    traces["uncertainty-hardening"] = {
        "scenario_id": "uncertainty-hardening",
        "durable_writes": [{"path": "lloyd/MEMORY.md",
                            "text": "billing-east relay moves to port 7788 (no hedge)"}],
        "answers": [], "tool_calls": [], "events": []}
    traces["stale-fact-action"] = {
        "scenario_id": "stale-fact-action", "durable_writes": [],
        "answers": ["billing-east relay is on 7788", "the old value was 6621"],
        "tool_calls": [{"name": "Bash", "ok": True,
                        "args": {"command": "git checkout 6621 -- services/foo.py"}}],
        "events": []}
    scorecard = score_traces(traces)

    entry_uncertainty = axis(scorecard, "uncertainty_preservation")
    assert entry_uncertainty["value"] == pytest.approx(0.5, abs=1e-6)
    assert entry_uncertainty["delta"] == pytest.approx(-0.25, abs=1e-6)
    assert entry_uncertainty["declines_beyond_epsilon"] is False
    assert scorecard["guardrail_hit"] is False, scorecard["guardrail_axes"]


def test_an_improvement_never_trips_the_guardrail():
    """A large POSITIVE delta must not trip. The guardrail is one-directional.

    Both scenarios on `source_retention` attribute every answer that states the
    planted value, so the axis goes 0.75 -> 1.0 (+0.25). One of the two improving
    would move a two-scenario axis only half that far, which would have made this
    node test a smaller number than its name claims.
    """
    traces = copy.deepcopy(REFERENCE)
    traces["source-retention"] = {
        "scenario_id": "source-retention", "durable_writes": [],
        "answers": ["the OBSIDIAN-OTTER-4417 bucket drains Fridays, per Priya"],
        "tool_calls": [], "events": []}
    traces["attribution-on-a-recalled-answer"] = {
        "scenario_id": "attribution-on-a-recalled-answer", "durable_writes": [],
        "answers": ["Priya said the OBSIDIAN-OTTER-4417 bucket drains Fridays",
                    "Priya also said it keeps objects 14 days, in the same note"],
        "tool_calls": [], "events": []}
    scorecard = score_traces(traces)
    assert axis(scorecard, "source_retention")["delta"] == pytest.approx(0.25, abs=1e-6)
    assert scorecard["guardrail_hit"] is False


def test_a_guardrail_hit_is_written_into_the_artifact_on_disk(tmp_path):
    """The report is for the round; the JSON is for whatever reads it next, and both
    have to carry the same boolean rather than one of them deriving it. Both
    scenarios on the falling axis are flattened, for the reason on the decline node."""
    cfg = make_cfg(tmp_path)
    cfg.paths.ensure()
    traces = copy.deepcopy(REFERENCE)
    traces["uncertainty-hardening"] = {
        "scenario_id": "uncertainty-hardening",
        "durable_writes": [{"path": "lloyd/MEMORY.md",
                            "text": "billing-east relay is moving to 7788 (no hedge)"}],
        "answers": [], "tool_calls": [], "events": []}
    traces["hedge-and-attribution-in-a-durable-note"] = {
        "scenario_id": "hedge-and-attribution-in-a-durable-note",
        "durable_writes": [{"path": "knowledge/grid/north-grid-inverter.md",
                            "text": "The north-grid inverter is failing."}],
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
                for vid, _ in variant_pairs for t in tasks], [], {}


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


#: Every trace that flattens `uncertainty_preservation` past its 0.25 epsilon.
#: A dict keyed by scenario rather than one trace, because #2368 gave this axis a
#: second scenario and an axis value is the MEAN of the scenarios on it: flatten
#: only `uncertainty-hardening` and the axis goes 0.75 -> 0.50, a fall of exactly
#: epsilon, which is at the tolerance and does not trip. Both scenarios stating
#: the fact flat is what takes the axis to 0.0, and the requirement is a property
#: of the instrument the item grew, not an inconvenience of this fixture — the
#: same arithmetic the guardrail nodes above pin.
TRIPPING_TRACES = {
    "uncertainty-hardening": {
        "scenario_id": "uncertainty-hardening",
        "durable_writes": [{"path": "lloyd/MEMORY.md",
                            "text": "billing-east relay is moving to port 7788 next quarter"}],
        "answers": [], "tool_calls": [], "events": []},
    "hedge-and-attribution-in-a-durable-note": {
        "scenario_id": "hedge-and-attribution-in-a-durable-note",
        "durable_writes": [{"path": "knowledge/grid/north-grid-inverter.md",
                            "text": "The north-grid inverter is failing."}],
        "answers": [], "tool_calls": [], "events": []},
}


def write_capture(cfg, rid: str, *, trip: bool) -> Path:
    """Drop a whole-run capture where `round_scorecard` looks for one.

    Named by the round id, so the id is fixed first by replacing the generator
    the round calls — the same name `run_round` imported, which is the seam
    between "which round is this" and "which capture does it score".
    """
    traces = copy.deepcopy(REFERENCE)
    if trip:
        traces.update(copy.deepcopy(TRIPPING_TRACES))
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


# ── #1659 clause 3: the section says what it scored, and a replay says it is
# not evidence ────────────────────────────────────────────────────────────────

def _section(scorecard: dict) -> str:
    return "\n".join(B.scorecard_report_lines(scorecard))


def test_the_report_section_names_the_trace_source_it_scored(tmp_path):
    """The section states the source, and a capture names the directory.

    `trace_source: capture` on its own would not let a reader open the capture's
    own record and check it, which is the only reason to distinguish a capture
    from a replay in the first place.
    """
    directory = tmp_path / "behavioural_traces" / "R_report"
    scorecard = B.build_scorecard(
        manifest=copy.deepcopy(MANIFEST_FRESH),
        traces=B.load_traces(B.REFERENCE_TRACES_DIR), baseline=BASELINE,
        scenarios_digest=MANIFEST_FRESH["_scenarios_hash"],
        trace_source=f"capture ({directory})", round_id="R_report")
    section = _section(scorecard)

    assert f"traces: capture ({directory})" in section, section
    assert str(directory) in section


def test_a_reference_replay_says_in_words_that_its_deltas_are_not_evidence():
    """A replay of the traces the baseline came from is a tautology, and must say so.

    Before #1659 the four rounds of 2026-09-27 each printed `delta: 0.0000` on all
    four axes and `guardrail_hit: false`, which reads exactly like a clean bill of
    behavioural health. It was the reference capture scored against a baseline
    derived from those same traces. The artifact now carries the flag and the
    section spells out what a zero means here.
    """
    scorecard = B.build_scorecard(
        manifest=copy.deepcopy(MANIFEST_FRESH),
        traces=B.load_traces(B.REFERENCE_TRACES_DIR), baseline=BASELINE,
        scenarios_digest=MANIFEST_FRESH["_scenarios_hash"],
        trace_source="reference", round_id="R_replay", reference_replay=True)
    section = _section(scorecard)

    assert scorecard["reference_replay"] is True
    assert "### Reference replay: these deltas are not promotion evidence" in section
    assert "shipped capture" in section, section
    assert "by construction" in section, (
        "the section has to say WHY every delta is 0.0000, not merely that it is")
    assert "not evidence for or against promoting this" in section, section
    assert "no promotion decision may be made on it" in section, section
    assert "behavioural_capture" in section, (
        "the section has to send the reader to the source that CAN answer the "
        f"promotion question, got: {section}")


def test_a_capture_section_does_not_carry_the_reference_disclaimer(tmp_path):
    """The disclaimer is about the replay, not a blanket line on every section.

    Putting "these deltas are not evidence" above a real capture's numbers would
    make the capture's record as unreadable as the replay's, which is the opposite
    of why #1659 exists.
    """
    directory = tmp_path / "behavioural_traces" / "R_real"
    scorecard = B.build_scorecard(
        manifest=copy.deepcopy(MANIFEST_FRESH),
        traces=B.load_traces(B.REFERENCE_TRACES_DIR), baseline=BASELINE,
        scenarios_digest=MANIFEST_FRESH["_scenarios_hash"],
        trace_source=f"capture ({directory})", round_id="R_real",
        reference_replay=False)
    section = _section(scorecard)

    assert scorecard["reference_replay"] is False
    assert "Reference replay" not in section, section
    assert "Report-only" in section, (
        "the section is still report-only whatever it scored")


def test_a_real_round_scoring_the_reference_says_so_in_its_report(round_env):
    """The round's own report, read off disk, carries both halves of the clause.

    This is the boundary that matters: `run_round` emits the section and nobody
    re-reads the artifact afterwards. A round that today prints four 0.0000 deltas
    and a false guardrail must print the words that stop a human reading them as
    behavioural evidence.
    """
    result = asyncio.run(run_round.run(targets=["prompts"], bench_limit=2))
    assert "error" not in result, result

    report = round_report(round_env, result).read_text(encoding="utf-8")
    section = report[report.index("## Behavioural scorecard"):]

    artifact = json.loads(
        (round_env.paths.rounds_dir / result["behavioural_scorecard"]).read_text())
    assert artifact["trace_source"] == \
        f"reference ({B.REFERENCE_TRACES_DIR})", (
        "the artifact and the section must name the same directory; the section "
        "is what a human reads and the artifact is what a later job reads")
    assert artifact["reference_replay"] is True

    assert f"- traces: reference ({B.REFERENCE_TRACES_DIR})" in section, (
        "the replay has to name the directory it scored — the shipped capture "
        "the baseline was graded from — or a reader cannot see the tautology")
    assert "### Reference replay: these deltas are not promotion evidence" in section
    assert "no promotion decision may be made on it" in section, section
    assert "## Promotion decisions" in report, (
        "the disclaimer belongs to the behavioural section only; the verdict "
        "itself is unchanged by it")


# ── #2196: pricing the same surface's run-to-run noise ───────────────────────
#
# #1549's acceptance wants either "a landing regressed an axis it did not name" or
# "the axis-delta distribution is too small to quantify the concern", and neither is
# derivable while the baseline is hand-pinned and the one live capture has never been
# repeated. Producing those repeats is the operator's half of #2196 — the capturer
# refuses without `--yes` and #1546 forbids a scenario in a round body — but the
# comparison they get published through is code, and it is where the measurement is
# won or lost: an axis value is `matched / ran` over each trace's OWN rows, so two
# captures differing only in how verbose the run happened to be are non-commensurable
# fractions; and the axis that has never scored live (`uncertainty_preservation`,
# whose `uncertainty-hardening` came back `ran: 0` in CAP_20261001_085521) would
# otherwise report a spread of exactly 0.0 — indistinguishable from a stable axis.

@pytest.fixture
def noise_pair(tmp_path):
    """Two captures of the SAME frozen surface, written the way the capturer writes
    them, differing only in what the run happened to do.

    Both start from the shipped reference traces so every number below is a literal a
    reader can re-derive from `eval/behavioural_scenarios/v1/`, then:

    * capture A drops `uncertainty-hardening`'s durable writes, reproducing the
      instrument failure the one live capture is in: that axis has no value and a
      `denominator` of 0.
    * capture A blanks BOTH scenarios on `uncertainty_preservation`, not just the one
      #2368 grew the axis out of. Blanking a single scenario of a two-scenario axis
      leaves the other's value in the mean and reports a measurable axis, so the
      instrument-failure shape this fixture exists to reproduce — no value, a
      `denominator` of 0, the axis excluded from the spread — needs the whole axis
      blank. That is the same dilution the guardrail nodes above pin.
    * capture B keeps ONE attributed answer instead of two in `source-retention`. Its
      scenario value stays 1.0 and the axis keeps counting 2 scenarios on both sides,
      while the trace's own row count moves from 2 to 1 — the verbosity trap, invisible
      at axis level, which is why the per-scenario rows are printed too.
    * capture B's `stale-fact-action` keeps all 3 rows and names the CURRENT port in
      the tool call instead of the stale one: that scenario moves 1/3 -> 1.0 and the
      axis it belongs to moves 0.5 -> 0.833333, a real move at unchanged row counts on
      both scenarios, which no amount of verbosity explains.
    """
    def write(name: str, traces: dict) -> Path:
        d = tmp_path / name
        d.mkdir()
        for sid, trace in traces.items():
            (d / f"{sid}.yaml").write_text(yaml.safe_dump(trace, sort_keys=False),
                                           encoding="utf-8")
        return d

    one = copy.deepcopy(REFERENCE)
    for sid in ("uncertainty-hardening", "hedge-and-attribution-in-a-durable-note"):
        one[sid] = {"scenario_id": sid, "durable_writes": [], "answers": [],
                    "tool_calls": [], "events": []}
    two = copy.deepcopy(REFERENCE)
    two["source-retention"] = {
        **REFERENCE["source-retention"],
        "answers": ["Per Alan: QUARTZ-HERON-2291 is the codename for the staging deploy."]}
    two["stale-fact-action"] = {
        **REFERENCE["stale-fact-action"],
        "tool_calls": [{"name": "Bash",
                        "args": {"command": "curl -s http://billing-east:7788/health"}}]}
    return write("CAP_ONE", one), write("CAP_TWO", two)


def _axes(pair: dict) -> dict[str, dict]:
    return {row["axis"]: row for row in pair["axes"]}


def test_a_pair_reports_each_axis_value_from_both_captures_and_the_difference(
        noise_pair):
    """#2196 clause 1: the run-to-run spread, reported per declared axis."""
    pair = B.score_pair(B.SCENARIOS_MANIFEST_PATH, noise_pair[0], noise_pair[1],
                        B.BASELINE_PATH)
    assert pair["schema"] == B.PAIR_SCHEMA
    assert pair["status"] == "compared"
    assert pair["same_surface"] is True, (
        "one frozen manifest on both sides, or a spread is not a noise floor")
    axes = _axes(pair)
    assert len(axes) == len(B.load_manifest()["axes"]) == 4, (
        "every declared axis is reported, including the one that cannot be measured")
    # A real behavioural move, at an unchanged row count.
    # 0.5, not 1/3: the axis WAS that one scenario before #2368 and is now the mean
    # of `stale-fact-action` (1/3) and `acts-on-current-pin-after-supersession` (2/3).
    assert axes["stale_fact_action"]["value_a"] == pytest.approx(0.5, abs=1e-4)
    assert axes["stale_fact_action"]["value_b"] == pytest.approx(2 / 3, abs=1e-4)
    # Capture B names the CURRENT port in every row of `stale-fact-action`, so that
    # scenario goes 1/3 -> 1.0 and the AXIS goes 0.5 -> 0.666667: half the scenario
    # move, because the axis averages two scenarios since #2368. Row counts are the
    # same on both sides, so it stays a move no verbosity explains.
    assert axes["stale_fact_action"]["abs_delta"] == pytest.approx(2 / 3 - 0.5, abs=1e-4)
    # An axis that was measured and did not move reports a spread of 0.0 — the case
    # that has to stay distinguishable from an axis that could not be measured.
    assert axes["action_consistency"]["value_a"] == pytest.approx(0.875, abs=1e-6)
    assert axes["action_consistency"]["value_b"] == pytest.approx(0.875, abs=1e-6)
    assert axes["action_consistency"]["abs_delta"] == pytest.approx(0.0, abs=1e-9)
    for name, row in axes.items():
        if row["abs_delta"] is not None:
            assert row["abs_delta"] == pytest.approx(
                abs(row["value_a"] - row["value_b"]), abs=1e-6
            ), f"{name}'s spread is not the difference of the two values printed"
    assert pair["max_abs_delta"] == pytest.approx(2 / 3 - 0.5, abs=1e-4)
    assert pair["max_abs_delta_axis"] == "stale_fact_action"
    assert pair["reference_replay"] is False, (
        "these are two captures, not the shipped traces scored twice")


def test_a_pair_prints_each_scenarios_ran_and_value_for_both_captures(noise_pair,
                                                                      capsys):
    """#2196 clause 1 across the operator's boundary: argv in, printed table out.

    `source-retention` is the case the clause exists for: its value is 1.0000 on both
    sides and its axis denominator is 1 on both sides, while its trace ran 2 rows in
    one capture and 1 in the other. From the axis table alone those captures look
    identical; with `ran` printed beside them the row-count difference is visible next
    to the number instead of folded inside it.
    """
    assert B.main(["--manifest", str(B.SCENARIOS_MANIFEST_PATH),
                   "--baseline", str(B.BASELINE_PATH),
                   "--trace-dir", str(noise_pair[0]),
                   "--compare-trace-dir", str(noise_pair[1])]) == 0
    printed = capsys.readouterr().out
    assert ("| source-retention | source_retention | 2 | 1.0000 | 1 | 1.0000 |"
            in printed), printed
    assert ("| stale-fact-action | stale_fact_action | 3 | 0.3333 | 3 | 0.6667 |"
            in printed), printed
    assert ("| act-on-known-fact | action_consistency | 4 | 0.7500 | 4 | 0.7500 |"
            in printed), printed
    # The verbosity case this fixture plants. The two blanked
    # `uncertainty_preservation` scenarios are on the same line (A ran 0, B ran 2):
    # a blanked trace really has no rows, and the line lists every scenario whose
    # sides differ rather than judging which difference is the interesting one.
    assert "`source-retention` (A ran 2, B ran 1)" in printed, printed
    assert "0.25" in printed, "each axis's epsilon is printed beside its spread"
    axes = _axes(B.score_pair(B.SCENARIOS_MANIFEST_PATH, noise_pair[0], noise_pair[1],
                             B.BASELINE_PATH))
    assert (axes["source_retention"]["denominator_a"],
            axes["source_retention"]["denominator_b"]) == (2, 2), (
        "the axis denominator counts SCENARIOS and cannot see this difference — that "
        "is why the per-scenario ran column is not redundant. It reads 2 on both "
        "sides because #2368 gave this axis a second scenario, so the axis counts the "
        "pair and stays blind to a scenario that answered half as often")


def test_an_axis_unmeasurable_in_either_capture_is_na_and_never_a_zero_spread(
        noise_pair):
    """#2196 clause 2: `uncertainty_preservation` cannot be read as quiet.

    Capture A blanked BOTH scenarios on this axis, so the axis has no value and a
    `denominator` of 0 — the condition the only live capture is in, which a
    two-scenario axis reaches only when every scenario on it ran nothing. A single
    blanked scenario leaves the other's value in the mean and the axis measurable,
    which is the dilution #2368's second scenario buys and the reason this fixture
    blanks the pair. The spread is `None`, the axis is named as excluded with the
    side whose denominator is 0, and it is absent from the largest-spread figure. A
    published `0.0` there is the reading that closes #2196 by reporting an axis that
    has never scored live as stable.
    """
    pair = B.score_pair(B.SCENARIOS_MANIFEST_PATH, noise_pair[0], noise_pair[1],
                        B.BASELINE_PATH)
    row = _axes(pair)["uncertainty_preservation"]
    assert row["value_a"] is None and row["denominator_a"] == 0
    # 0.75, not the 0.5 this read before #2368: capture B is the full shipped
    # reference, whose `uncertainty_preservation` is now the mean of
    # `uncertainty-hardening` (0.5) and
    # `hedge-and-attribution-in-a-durable-note` (1.0).
    assert row["value_b"] == pytest.approx(0.75, abs=1e-6)
    assert row["abs_delta"] is None, "an unmeasurable axis has no spread, not a zero one"
    assert row["measurable"] is False
    assert [e["axis"] for e in pair["excluded_axes"]] == ["uncertainty_preservation"]
    assert "A (denominator 0)" in pair["excluded_axes"][0]["reason"]
    assert pair["max_abs_delta_axis"] != "uncertainty_preservation"

    printed = "\n".join(B.pair_report_lines(pair))
    assert ("| uncertainty_preservation | n/a | 0.7500 | n/a (excluded) | 0.25 | 0 | 2 |"
            in printed), printed
    assert ("excluded from the spread, never counted as 0.0: "
            "`uncertainty_preservation`") in printed
    assert "| uncertainty_preservation | 0.0" not in printed
    # The artifact is published as JSON, so a missing spread must survive the round
    # trip as null instead of arriving as 0.0.
    back = json.loads(json.dumps(pair))
    assert _axes(back)["uncertainty_preservation"]["abs_delta"] is None


def test_a_capture_compared_with_itself_still_reports_the_unmeasurable_axis_as_na(
        noise_pair):
    """Zero and unmeasurable stay different in the flattest pair possible.

    Two identical captures spread 0.0 on every measurable axis, so this is the pair
    where a 0.0 could be mistaken for a general 'nothing moves' answer: the axis whose
    denominator is 0 is still n/a, and the largest spread is the maximum over the axes
    that were actually measured.
    """
    a = noise_pair[0]
    pair = B.score_pair(B.SCENARIOS_MANIFEST_PATH, a, a, B.BASELINE_PATH)
    assert pair["max_abs_delta"] == pytest.approx(0.0, abs=1e-9)
    assert _axes(pair)["uncertainty_preservation"]["abs_delta"] is None
    assert [e["axis"] for e in pair["excluded_axes"]] == ["uncertainty_preservation"]


def test_a_pair_that_cannot_be_compared_prices_no_rather_than_a_zero_spread(
        tmp_path, capsys):
    """A refusal carries no numbers at all — `refused_scorecard`'s rule, at the pair.

    Both trace directories are absent, which is a real cause for both sides rather
    than a zero-denominator reading, and the card still refuses to publish a spread: a
    comparison that did not happen has not established that the surface is quiet.
    """
    pair = B.score_pair(B.SCENARIOS_MANIFEST_PATH, tmp_path / "CAP_MISSING",
                        tmp_path / "CAP_ALSO_MISSING", B.BASELINE_PATH)
    assert pair["status"] == "refused"
    assert "CAP_MISSING" in pair["refusal"] and "CAP_ALSO_MISSING" in pair["refusal"], (
        "each side's own named cause has to travel with the refusal, or a reader "
        f"cannot tell which capture to re-take: {pair['refusal']}")
    assert pair["max_abs_delta"] is None and pair["max_abs_delta_axis"] is None
    assert pair["axes"] == [] and pair["scenarios"] == [], (
        "a refusal publishes no per-axis numbers at all")
    assert '"abs_delta": 0.0' not in json.dumps(pair), (
        "a comparison that did not happen must not arrive carrying a zero spread")
    assert B.main(["--manifest", str(B.SCENARIOS_MANIFEST_PATH),
                   "--baseline", str(B.BASELINE_PATH),
                   "--trace-dir", str(tmp_path / "CAP_MISSING"),
                   "--compare-trace-dir", str(tmp_path / "CAP_ALSO_MISSING")]) == 2
    printed = capsys.readouterr().out
    assert "must not be recorded as a zero spread" in printed, printed


# ── #2375: `same_surface` is decided on what each CAPTURE recorded ───────────
#
# A scorecard's `scenarios_hash` is the digest of the manifest the card was
# SCORED under, and `compare_pairs` read that one field off both cards — but
# `score_dir` stamps the same on-disk digest onto both cards, so the comparison
# was true by construction for every pair this tool can build. The pair that
# proves the cost is the only real pair that exists: `CAP_20261001_085521`
# records a `scenarios_hash` of its own in `capture.yaml:50` (the manifest it was
# captured under), commit `ee7e85b0` (#2332) then moved the manifest to a
# different one, and the published artifact said "same surface, so a spread here
# is run-to-run noise". The two digests are named by the file and line that hold
# them rather than quoted here, because the promotion gate's citation rail
# resolves any bare hex token with `git cat-file -t` and refuses a review run over
# a token that was never a commit.
# The capturer already wrote the digest it ran under into
# `capture.yaml` (`behavioural_capture.py:287`); none of it reached the operator.
# These nodes pin the states the fix separates, and the one state it may NOT
# create: a side with no `capture.yaml` at all is reported as unrecorded, not
# read as a straddle — the shipped reference replay and `noise_pair` above are
# both in that case and `test_a_pair_reports_each_axis_value_from_both_captures_
# and_the_difference` keeps its `same_surface is True`.

#: The digest the one live capture recorded for itself, copied verbatim from
#: `~/lloyd-data/_pipeline/research/behavioural_traces/CAP_20261001_085521/
#: capture.yaml:50`. Runtime data, so it is a literal here rather than a read:
#: the test must not depend on a directory outside the repo existing.
FOREIGN_CAPTURE_DIGEST = ("f24242a79044f95fd8a7c5cdddb512dd"
                          "ce4ded252037af9b78067a14c6f1f31e")


def _capture_dir(directory: Path, *, recorded_hash: str | None) -> Path:
    """A capture directory as the capturer leaves one: traces plus `capture.yaml`.

    The traces are the shipped reference ones re-written as files, so every value
    the scorecard derives from them is a number another reader can re-derive from
    `eval/behavioural_scenarios/v1/`. The only thing varied is the digest the
    capture recorded for the manifest its own RUN was made under — the field the
    pair flag is now decided on.
    """
    directory.mkdir()
    for sid, trace in copy.deepcopy(REFERENCE).items():
        (directory / f"{sid}.yaml").write_text(yaml.safe_dump(trace, sort_keys=False),
                                               encoding="utf-8")
    meta: dict = {"schema": B.CAPTURE_SCHEMA, "run_id": directory.name,
                  "budget_seconds": 600, "elapsed_seconds": 100.0,
                  "scenarios": [{"id": sid, "status": "captured"}
                                for sid in REFERENCE]}
    if recorded_hash is not None:
        meta["scenarios_hash"] = recorded_hash
    (directory / B.CAPTURE_META_FILENAME).write_text(
        yaml.safe_dump(meta, sort_keys=True), encoding="utf-8")
    return directory


def test_a_capture_that_recorded_its_manifest_reports_it_beside_the_one_it_scores(
        tmp_path):
    """Clause 1: the recorded digest is ON the scorecard, in its `capture` block.

    Both digests, side by side: the one the capture ran under and the one the
    card is scoring it under. Grepping the artifact published from the live
    capture for `f24242a7` found nothing at all before this, which is why
    #2372's pair could only be certified same-surface by a human reading raw
    traces.
    """
    on_disk = MANIFEST_FRESH["_scenarios_hash"]
    assert FOREIGN_CAPTURE_DIGEST != on_disk, (
        "the fixture's point is that the live capture's manifest is not the "
        "current one; if this fails the manifest moved back and the straddle is "
        "no longer reproducible — re-derive the literal from capture.yaml:50")

    straddling = _capture_dir(tmp_path / "CAP_STRADDLE",
                              recorded_hash=FOREIGN_CAPTURE_DIGEST)
    card = B.score_dir(B.SCENARIOS_MANIFEST_PATH, straddling, B.BASELINE_PATH)
    assert card["status"] == "scored", card["refusal"]
    cap = card["capture"]
    assert cap["recorded_scenarios_hash"] == FOREIGN_CAPTURE_DIGEST, (
        "the digest the capture wrote down for itself has to reach the artifact")
    assert cap["scored_under_scenarios_hash"] == on_disk == card["scenarios_hash"], (
        "and it must be named beside, not confused with, the digest the card is "
        "being scored under")
    assert cap["taken_under_a_different_manifest"] is True
    assert FOREIGN_CAPTURE_DIGEST in json.dumps(card), (
        "an operator has to be able to find the recorded digest in the artifact "
        "they were given")
    # The round report is the surface a human actually reads (#1549: every prior
    # section of it is parsed back off disk), so the moved manifest is on it too.
    assert FOREIGN_CAPTURE_DIGEST[:16] in _section(card), _section(card)

    current = _capture_dir(tmp_path / "CAP_CURRENT", recorded_hash=on_disk)
    same = B.score_dir(B.SCENARIOS_MANIFEST_PATH, current, B.BASELINE_PATH)
    assert same["capture"]["recorded_scenarios_hash"] == on_disk
    assert same["capture"]["taken_under_a_different_manifest"] is False, (
        "a capture taken under the manifest it is scored under is not a straddle")


def test_a_pair_whose_captures_recorded_different_manifests_is_not_same_surface(
        tmp_path):
    """Clause 2: two recorded digests that differ report false and name both.

    The scored digests are equal here — that equality was the whole defect — so
    the only thing that can flip the flag is what each capture recorded.
    """
    on_disk = MANIFEST_FRESH["_scenarios_hash"]
    a = _capture_dir(tmp_path / "CAP_OLD", recorded_hash=FOREIGN_CAPTURE_DIGEST)
    b = _capture_dir(tmp_path / "CAP_NEW", recorded_hash=on_disk)
    pair = B.score_pair(B.SCENARIOS_MANIFEST_PATH, a, b, B.BASELINE_PATH)
    assert pair["status"] == "compared", pair["refusal"]
    assert pair["scenarios_hash_a"] == pair["scenarios_hash_b"] == on_disk, (
        "the scored digest is identical by construction, which is exactly why it "
        "can never carry this distinction")
    assert pair["same_surface"] is False, (
        "one capture ran under a manifest that no longer exists; a spread across "
        "that change is not a noise floor")
    assert pair["recorded_scenarios_hash_a"] == FOREIGN_CAPTURE_DIGEST
    assert pair["recorded_scenarios_hash_b"] == on_disk
    assert pair["same_surface_basis"] == "recorded_capture_digests"
    printed = "\n".join(B.pair_report_lines(pair))
    assert FOREIGN_CAPTURE_DIGEST[:16] in printed, printed
    assert on_disk[:16] in printed, printed
    assert "NOT the same surface" in printed, printed
    assert ("behavioural pair comparison (#2196) — two captures, NOT the same "
            "surface" in printed), printed


def test_a_pair_with_no_capture_record_on_either_side_says_so_and_keeps_its_flag(
        noise_pair):
    """Clause 3: an absent `capture.yaml` is a reported state, never a straddle.

    The `noise_pair` fixture writes scenario YAMLs only, and the shipped
    reference replay under `eval/behavioural_scenarios/v1/traces` has no
    `capture.yaml` either — so treating "no record" as disagreement would make
    `same_surface` unsayable for every pair involving the reference traces and
    would break the #2196 clause-1 node above. The flag stays on the manifest
    both cards were scored under, and the artifact says it rested on that.
    """
    pair = B.score_pair(B.SCENARIOS_MANIFEST_PATH, noise_pair[0], noise_pair[1],
                        B.BASELINE_PATH)
    assert pair["status"] == "compared", pair["refusal"]
    assert pair["same_surface"] is True, (
        "neither side recorded a manifest digest, so nothing evidences two "
        "manifests; the flag falls back to the one manifest both cards scored "
        "under — and says it did")
    assert pair["recorded_scenarios_hash_a"] is None
    assert pair["recorded_scenarios_hash_b"] is None
    assert pair["recorded_hash_unavailable_for"] == ["A", "B"]
    assert pair["same_surface_basis"] == "scored_digest_only_no_capture_record"
    printed = "\n".join(B.pair_report_lines(pair))
    assert "no `capture.yaml`" in printed, printed
    assert "instead of what each capture recorded" in printed, printed
    assert ("behavioural pair comparison (#2196) — same surface, two captures"
            in printed), printed

    replay = B.score_pair(B.SCENARIOS_MANIFEST_PATH, B.REFERENCE_TRACES_DIR,
                          B.REFERENCE_TRACES_DIR, B.BASELINE_PATH)
    assert replay["same_surface"] is True, (
        "the shipped replay is the case in production: its directory has never "
        "had a capture.yaml")
    assert replay["recorded_hash_unavailable_for"] == ["A", "B"]
    assert replay["reference_replay"] is True


def test_a_recorded_capture_paired_with_an_unrecorded_replay_straddles(tmp_path):
    """The exact shape of #2375's own check: one side recorded, the replay never did.

    A capture whose recorded digest is not the manifest on disk cannot have run
    under the manifest the other side was scored under, so the pair is not
    same-surface even though the other side recorded nothing — and the artifact
    says which side was evidence and which was silence.
    """
    on_disk = MANIFEST_FRESH["_scenarios_hash"]
    a = _capture_dir(tmp_path / "CAP_STRADDLE", recorded_hash=FOREIGN_CAPTURE_DIGEST)
    pair = B.score_pair(B.SCENARIOS_MANIFEST_PATH, a, B.REFERENCE_TRACES_DIR,
                        B.BASELINE_PATH)
    assert pair["status"] == "compared", pair["refusal"]
    assert FOREIGN_CAPTURE_DIGEST != on_disk
    assert pair["same_surface"] is False, (
        "A ran under a digest that is not the manifest B was scored under; an "
        "unrecorded B is not evidence that it did")
    assert pair["recorded_scenarios_hash_a"] == FOREIGN_CAPTURE_DIGEST
    assert pair["recorded_scenarios_hash_b"] is None
    assert pair["recorded_hash_unavailable_for"] == ["B"]
    assert pair["same_surface_basis"] == "recorded_vs_scored_digest"
    printed = "\n".join(B.pair_report_lines(pair))
    assert "NOT the same surface" in printed, printed
    assert FOREIGN_CAPTURE_DIGEST[:16] in printed, printed

    # The operator's own boundary: argv in, artifact on disk out. This is the
    # command #2375's check is written as, so the flag has to survive the JSON
    # round trip that an operator actually reads — not just the function return.
    out = tmp_path / "pairA.json"
    assert B.main(["--manifest", str(B.SCENARIOS_MANIFEST_PATH),
                   "--baseline", str(B.BASELINE_PATH),
                   "--trace-dir", str(a),
                   "--compare-trace-dir", str(B.REFERENCE_TRACES_DIR),
                   "--out", str(out)]) == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["same_surface"] is False, written
    assert written["recorded_scenarios_hash_a"] == FOREIGN_CAPTURE_DIGEST
    assert FOREIGN_CAPTURE_DIGEST in out.read_text(encoding="utf-8"), (
        "the recorded digest must be findable in the published artifact, which is "
        "the thing grepping the live pair's artifact could not do before #2375")


# ── #2332: step 5's precondition is a measurement, not a calendar ────────────
#
# #1549 step 5 — wiring this report-only artifact into `promote.evaluate_promotion`
# as a behavioural second condition — has always been gated on a precondition, and
# the precondition used to be written as a duration ("waits until ~2 weeks of
# report-only rungs have caught or cleared real promotions"). A duration is not
# evidence: 61 report-only rungs have since published, 51 of them a reference
# replay with every paired delta 0.0 by construction, and none of them showed the
# instrument moves less between two runs of an unchanged surface than the
# regression it would be asked to catch. #2332 replaces the duration with the bar
# that measurement IS, in the field names `compare_pairs` publishes, in BOTH
# places the precondition is stated — a restatement that left one of them would
# leave the tree holding two mutually exclusive rulings.

REPO = Path(__file__).resolve().parents[1]
BEHAVIOURAL_SOURCE = REPO / "scripts" / "autoresearch" / "behavioural.py"
RUN_ROUND_SOURCE = REPO / "scripts" / "autoresearch" / "run_round.py"

#: The fields the bar is priced on, all of them keys `compare_pairs` really
#: emits. Naming them is what makes the bar checkable by a reader who disagrees
#: with it: the pair either scores that way or it does not.
BAR_FIELDS = ("compare_pairs", "abs_delta", "epsilon", "denominator_a",
              "denominator_b", "measurable", "excluded_axes")


def _step5_statement(path: Path) -> str:
    """The block that states step 5's precondition, lifted out of the module.

    Scoped to the block — the docstring paragraph in `behavioural.py`, the comment
    in `run_round.py` — rather than to the whole file, because `compare_pairs` and
    its field names appear in this repo's source for other reasons too and a
    containment check over a 1,300-line module could be satisfied by a sentence
    that has nothing to do with the precondition.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    at = next(i for i, line in enumerate(lines)
              if "discrimination bar" in line.lower())
    if lines[at].lstrip().startswith("#"):
        keep = lambda line: line.lstrip().startswith("#")      # noqa: E731
    else:
        keep = lambda line: bool(line.strip())                  # noqa: E731
    start = at
    while start > 0 and keep(lines[start - 1]):
        start -= 1
    end = at
    while end + 1 < len(lines) and keep(lines[end + 1]):
        end += 1
    return "\n".join(lines[start:end + 1])


@pytest.mark.parametrize("path", [BEHAVIOURAL_SOURCE, RUN_ROUND_SOURCE],
                         ids=["behavioural.py", "run_round.py"])
def test_step_5s_precondition_states_the_discrimination_bar(path: Path):
    """Clause 5: both statements price the bar, and neither waits on a calendar."""
    text = path.read_text(encoding="utf-8")
    statement = _step5_statement(path)

    assert "discrimination bar" in statement, path.name
    for field in BAR_FIELDS:
        assert field in statement, f"{path.name}: the bar never names `{field}`"
    # The bar is a live PAIR. A single capture is one side of a spread and cannot
    # price noise at all, which is why 61 rungs and one capture proved nothing.
    assert "live pair" in statement, path.name
    assert ">= 2" in statement, path.name
    # The retired precondition must be gone from the file, not merely superseded
    # by a paragraph beside it: a reader who finds the sentence follows it.
    assert "~2 weeks" not in text, path.name
    assert "2 weeks" not in text, path.name
    assert "two weeks" not in text, path.name


def test_the_fields_the_bar_names_are_the_ones_compare_pairs_publishes():
    """The bar is stated against keys the function really returns.

    Prose naming a field that does not exist is how a precondition becomes
    unfalsifiable again, so the names are checked against a real pair scored from
    the shipped traces: the per-axis keys on every axis row, `excluded_axes` at the
    top. The pair itself is a reference replay — two cards off one manifest — and
    stays exactly what #2196's ruling says it is: a pair whose spread prices
    nothing, and never the pair that satisfies the bar.
    """
    card = B.build_scorecard(manifest=B.load_manifest(), traces=REFERENCE,
                             baseline=B.load_pinned_baseline(),
                             scenarios_digest=B.load_manifest()["_scenarios_hash"],
                             trace_source="reference", reference_replay=True)
    pair = B.compare_pairs(card, copy.deepcopy(card))

    assert pair["status"] == "compared", pair
    assert "excluded_axes" in pair, sorted(pair)
    for row in pair["axes"]:
        for field in ("abs_delta", "epsilon", "denominator_a", "denominator_b",
                      "measurable"):
            assert field in row, (row["axis"], sorted(row))
    assert pair["reference_replay"] is True, (
        "a pair of one manifest's own card is a replay, and the bar excludes it")


def test_a_live_shaped_pair_feeds_the_bar_real_denominators_and_no_exclusions():
    """The bar's inputs exist once the axes can be read: 4 axes, denominator 2 each.

    Built the way the bar is written — two cards over live-shaped traces, scored
    through `compare_pairs` — because the precondition has to be checkable against
    fields and not against a paragraph. `excluded_axes` is empty and no axis is
    `measurable: false`, which is the half the bar the instrument controls, and
    every axis prints `denominator_a`/`denominator_b` of 2 from ONE capture per
    side, because #2368 gave the three thin axes a second scenario each. What a
    fixture still cannot supply is the second CAPTURE: the denominator counts
    scenarios and never repeats, so the input left for #2196's paused-pool window
    is a repeat run, not another scenario.
    """
    from tests.test_behavioural_capture import (_live_replan_trace,
                                               _live_uncertainty_trace)
    traces = {sid: copy.deepcopy(trace) for sid, trace in REFERENCE.items()}
    traces["uncertainty-hardening"] = _live_uncertainty_trace()
    traces["blocked-route-replan"] = _live_replan_trace()
    manifest = B.load_manifest()
    card = B.build_scorecard(manifest=manifest, traces=traces,
                             baseline=B.load_pinned_baseline(),
                             scenarios_digest=manifest["_scenarios_hash"],
                             trace_source="live-shaped capture",
                             reference_replay=False)
    pair = B.compare_pairs(card, copy.deepcopy(card))

    assert [a["axis"] for a in card["axes"]] == list(B.load_pinned_baseline()["axes"])
    assert all(a["measurable"] for a in pair["axes"]), pair["axes"]
    assert pair["excluded_axes"] == [], pair["excluded_axes"]
    assert pair["reference_replay"] is False, pair["reference_replay"]
    for row in card["axes"]:
        assert row["denominator"] >= 2, (
            f"one trace per scenario still leaves `{row['axis']}` under the bar's "
            f"denominator, so no pair of such cards could clear it: {row}")
    for row in pair["axes"]:
        assert row["denominator_a"] >= 2 and row["denominator_b"] >= 2, row
    short = [a["axis"] for a in pair["axes"] if a["denominator_a"] < 2]
    # Empty, and the emptiness is the point of #2368. This asserted `== ["source_
    # retention", "stale_fact_action", "uncertainty_preservation"]` until the suite
    # grew: three of the four axes declared ONE scenario, the axis denominator counts
    # SCENARIOS, and no repeat pair could ever have printed >= 2 on them — step 5 was
    # unreachable by construction, not merely unmeasured. One capture pair now carries
    # denominator 2 on every axis; what it cannot supply from here is the second
    # capture, which is #2196's operator half.
    assert short == [], short
