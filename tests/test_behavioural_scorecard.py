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
    * capture B keeps ONE attributed answer instead of two in `source-retention`. The
      scenario value stays 1.0 and its axis denominator stays 1, while the trace's own
      row count moves from 2 to 1 — the verbosity trap, invisible at axis level.
    * capture B's `stale-fact-action` keeps all 3 rows and names the CURRENT port in
      the tool call instead of the stale one: a real 1/3 move at an unchanged row
      count, which no amount of verbosity explains.
    """
    def write(name: str, traces: dict) -> Path:
        d = tmp_path / name
        d.mkdir()
        for sid, trace in traces.items():
            (d / f"{sid}.yaml").write_text(yaml.safe_dump(trace, sort_keys=False),
                                           encoding="utf-8")
        return d

    one = copy.deepcopy(REFERENCE)
    one["uncertainty-hardening"] = {"scenario_id": "uncertainty-hardening",
                                    "durable_writes": [], "answers": [],
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
    assert axes["stale_fact_action"]["value_a"] == pytest.approx(1 / 3, abs=1e-4)
    assert axes["stale_fact_action"]["value_b"] == pytest.approx(2 / 3, abs=1e-4)
    assert axes["stale_fact_action"]["abs_delta"] == pytest.approx(1 / 3, abs=1e-4)
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
    assert pair["max_abs_delta"] == pytest.approx(1 / 3, abs=1e-4)
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
    assert "row counts differ for: `source-retention` (A ran 2, B ran 1)" in printed
    assert "0.25" in printed, "each axis's epsilon is printed beside its spread"
    axes = _axes(B.score_pair(B.SCENARIOS_MANIFEST_PATH, noise_pair[0], noise_pair[1],
                             B.BASELINE_PATH))
    assert (axes["source_retention"]["denominator_a"],
            axes["source_retention"]["denominator_b"]) == (1, 1), (
        "the axis denominator counts scenarios and cannot see this difference — which "
        "is why the per-scenario ran column is not redundant")


def test_an_axis_unmeasurable_in_either_capture_is_na_and_never_a_zero_spread(
        noise_pair):
    """#2196 clause 2: `uncertainty_preservation` cannot be read as quiet.

    Capture A has `uncertainty-hardening` at `ran: 0`, so the axis has no value and a
    `denominator` of 0 — the condition the only live capture is in. The spread is
    `None`, the axis is named as excluded with the side whose denominator is 0, and it
    is absent from the largest-spread figure. A published `0.0` there is the reading
    that closes #2196 by reporting an axis that has never scored live as stable.
    """
    pair = B.score_pair(B.SCENARIOS_MANIFEST_PATH, noise_pair[0], noise_pair[1],
                        B.BASELINE_PATH)
    row = _axes(pair)["uncertainty_preservation"]
    assert row["value_a"] is None and row["denominator_a"] == 0
    assert row["value_b"] == pytest.approx(0.5, abs=1e-6)
    assert row["abs_delta"] is None, "an unmeasurable axis has no spread, not a zero one"
    assert row["measurable"] is False
    assert [e["axis"] for e in pair["excluded_axes"]] == ["uncertainty_preservation"]
    assert "A (denominator 0)" in pair["excluded_axes"][0]["reason"]
    assert pair["max_abs_delta_axis"] != "uncertainty_preservation"

    printed = "\n".join(B.pair_report_lines(pair))
    assert ("| uncertainty_preservation | n/a | 0.5000 | n/a (excluded) | 0.25 | 0 | 1 |"
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
    """The bar's inputs exist once the axes can be read: 4 axes, both sides measured.

    Built the way the bar is written — two cards over live-shaped traces, scored
    through `compare_pairs` — because the precondition has to be checkable against
    fields and not against a paragraph. `excluded_axes` is empty and no axis is
    `measurable: false`, which is the half the bar the instrument controls; the
    denominators are still 1 per side here, because a fixture cannot produce the
    repeats the paused-pool window is owed for (#2196), and a denominator of 1 is
    exactly what keeps the bar UNMET. That is the point of stating it as
    `denominator_a`/`denominator_b` >= 2 rather than as a duration.
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
    for row in pair["axes"]:
        assert row["denominator_a"] >= 1 and row["denominator_b"] >= 1, row
    short = [a["axis"] for a in pair["axes"] if a["denominator_a"] < 2]
    # Three of the four axes are still under the bar's denominator with a whole
    # trace on each side: only `action_consistency` reaches 2, and it does so
    # because the reference `act-on-known-fact` trace carries eight action rows.
    # One capture per side cannot clear the bar, which is why it is written as
    # denominators and not as a duration — #2196's window is what moves these.
    assert short == ["source_retention", "stale_fact_action",
                     "uncertainty_preservation"], short
