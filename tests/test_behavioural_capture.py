"""#1659 — the behavioural suite had a reader and no writer.

`scripts/autoresearch/behavioural.py` could load and score whole-run traces since
#1549, but nothing in the repository could produce one. The four autoresearch
rounds run on 2026-09-27 each wrote a `.behavioural_scorecard.json` reading
`trace_source: "reference"` with `delta: 0.0` on all four axes, because the only
traces it could find were the shipped capture that the pinned baseline had
itself been graded from — an instrument that cannot move, scoring itself. These
nodes are the writer's contract: where a trace lands, what a scenario that never
ran is *called*, and what the capture is obliged to admit about its own budget.

The engine is never in these tests. `capture_round` takes a runner callable, so
every path here — file names, the planted state, the wall-clock budget, the
statuses, and the scorecard that `round_scorecard` builds from the result — is
measured with no model and no GPU — including `sdk_runner`, whose call
convention and whose trial-result-to-trace map are pinned with `run_trial`
stubbed. What no node here can price is the cost and the tool surface of a real
run: that is #1659's owed items 1 and 3, to be read off the first live capture on
an idle primary in a paused-pool window rather than asserted here.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.autoresearch import behavioural as B          # noqa: E402
from scripts.autoresearch import behavioural_capture as CAP  # noqa: E402

MANIFEST = B.load_manifest()
SCENARIOS = MANIFEST["scenarios"]
IDS = [str(s["id"]) for s in SCENARIOS]
REFERENCE = B.load_traces(B.REFERENCE_TRACES_DIR)
#: The ruled cap: 10 minutes of wall clock per capture round.
BUDGET = CAP.DEFAULT_CAPTURE_BUDGET_SECONDS
RUN_ID = "R_capture_probe"

#: #1843 clause 4, read off the shipped manifest rather than typed here, because
#: the clause IS an identity between the scorecard and that file: for a capture in
#: which every `capture: trial` scenario produced its observation, the
#: instrument-failure set is exactly this list. The literal is pinned once, in
#: `test_the_unmeasurable_scenario_names_the_vocabulary_it_needs`, so a manifest
#: edit cannot silently move the expectation of every node that uses this.
NO_CAPTURE_PATH_IDS = [sid for sid, s in zip(IDS, SCENARIOS)
                      if B.capture_scope(s)[0] == "none"]


def _cfg(tmp_path: Path):
    """A config pointed at a scratch research root — never the live one."""
    root = tmp_path / "research"
    (root / "rounds").mkdir(parents=True)
    return SimpleNamespace(
        paths=SimpleNamespace(research_root=root, rounds_dir=root / "rounds",
                              variants_dir=root / "variants"),
        default_model="test-model")


class Clock:
    """A hand-advanced wall clock.

    The budget is elapsed wall clock, and a test that exercises it by sleeping is
    a test that fails on a loaded machine and passes for the wrong reason. The
    fake runner advances this by the scenario's cost, so the capturer sees real
    elapsed time and the test says exactly how long each scenario took.
    """

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _healthy_trace(scenario: dict, *, marker: str = "fake-engine") -> dict:
    """A whole-run trace in the shape the graders already accept for this scenario.

    Cloned from the shipped reference capture and re-marked, so the only thing
    these nodes measure is who wrote the file and where. The marker matters: it
    is what distinguishes a trace this capture wrote from the canned one the
    suite would otherwise fall back to.
    """
    trace = dict(REFERENCE[str(scenario["id"])])
    trace["captured_by"] = marker
    return trace


def _runner(*, cost: float = 0.0, clock: Clock | None = None,
            raise_for: tuple[str, ...] = (), marker: str = "fake-engine"):
    def run(scenario: dict) -> dict:
        if clock is not None:
            clock.t += cost
        sid = str(scenario["id"])
        if sid in raise_for:
            raise RuntimeError(f"engine refused the run for {sid}")
        return _healthy_trace(scenario, marker=marker)
    return run


# ── clause 1: a capture entry point that writes what the scorer can read ─────

def test_a_capture_writes_one_trace_per_scenario_where_the_scorer_looks(tmp_path):
    """The directory, the file names and the shape are the whole contract.

    `round_scorecard` reads `<research_root>/behavioural_traces/<run_id>/`, and
    `load_traces` accepts one YAML whole-run trace per scenario keyed by its
    `scenario_id`. Before #1659 neither end met in the middle: nothing wrote
    here, so no round ever had a trace of its own.
    """
    cfg = _cfg(tmp_path)
    CAP.capture_round(cfg=cfg, run_id=RUN_ID, runner=_runner())
    out = CAP.capture_dir(cfg, RUN_ID)

    assert out == Path(cfg.paths.research_root) / "behavioural_traces" / RUN_ID
    assert out.is_dir(), "a capture writes under the path round_scorecard reads"
    written = {p.name for p in out.glob("*.yaml")}
    assert written == {f"{sid}.yaml" for sid in IDS} | {B.CAPTURE_META_FILENAME}

    traces = B.load_traces(out)
    assert sorted(traces) == sorted(IDS), (
        "load_traces has to accept every file the capturer writes — a trace it "
        "refuses to read is a scenario that never ran")
    for sid, trace in traces.items():
        assert trace["scenario_id"] == sid
        assert any(isinstance(trace.get(key), list) and trace.get(key)
                   for key in CAP.OBSERVABLE_KEYS), (
            f"{sid}: a trace carrying none of {CAP.OBSERVABLE_KEYS} is a file no "
            f"grader can read, which is a captured run only in name")


def test_a_capture_plants_the_scenarios_state_before_the_run(tmp_path):
    """`planted_input` reaches disk, or the scenario is not the scenario.

    Each frozen scenario declares the pre-existing state its run has to discover
    — a hedged fact, a superseded port, a planted root — as structured fields,
    and declares no path, so the capturer names the file after the scenario and
    writes those fields verbatim. A probe that never plants them measures a
    different situation than the one being scored.
    """
    cfg = _cfg(tmp_path)
    plant_root = Path(cfg.paths.research_root) / "planted"
    seen: list[dict] = []

    def spy(scenario: dict) -> dict:
        seen.append(scenario)
        return _healthy_trace(scenario)

    CAP.capture_round(cfg=cfg, run_id=RUN_ID, runner=spy, plant_root=plant_root)

    for scenario in SCENARIOS:
        declared = {k: v for k, v in scenario["planted_input"].items()
                    if k != "then_run"}
        target = plant_root / f"{scenario['id']}.yaml"
        assert target.is_file(), (
            f"{scenario['id']}: planted_input was never written, so its probe ran "
            f"against an empty tree")
        assert yaml.safe_load(target.read_text()) == declared, (
            f"{scenario['id']}: the planted file is not the state the manifest "
            f"declared")
        assert "then_run" not in yaml.safe_load(target.read_text()), (
            f"{scenario['id']}: the run's own instruction went into the state file, "
            f"so the scenario tells the run what to do instead of letting it find out")

    # The runner is told where it was put, or the run cannot go and look.
    assert [s["_planted"] for s in seen] == [f"{s['id']}.yaml" for s in SCENARIOS]


def test_the_trace_on_disk_is_the_one_the_runner_returned(tmp_path):
    """The capturer records the run it was given — not the canned reference.

    This is the assertion that separates #1659 from a pass: a capturer that
    quietly fell back to the shipped traces would satisfy every other node here
    and leave the tautology exactly where it was.
    """
    cfg = _cfg(tmp_path)
    CAP.capture_round(cfg=cfg, run_id=RUN_ID,
                      runner=_runner(marker="fake-engine: run 7"))

    traces = B.load_traces(CAP.capture_dir(cfg, RUN_ID))
    assert {t["captured_by"] for t in traces.values()} == {"fake-engine: run 7"}, (
        "the files in the capture directory are not the traces the runner produced")


def test_the_capture_record_is_not_mistaken_for_a_trace(tmp_path):
    """`capture.yaml` describes the run; it is not a scenario and must not score."""
    cfg = _cfg(tmp_path)
    CAP.capture_round(cfg=cfg, run_id=RUN_ID, runner=_runner())
    out = CAP.capture_dir(cfg, RUN_ID)

    assert (out / B.CAPTURE_META_FILENAME).is_file()
    traces = B.load_traces(out)
    # Asserted on what `load_traces` returned and on how many files it read, not
    # on a substring of the dict repr: a record wrongly loaded as a trace would be
    # keyed by whatever id it carried and name no filename anywhere in that repr,
    # so `CAPTURE_META_FILENAME not in str(traces)` could not fail. Each of these
    # three can: the keys, because the record would add one; the count, because
    # exactly one of the directory's YAML files is not a trace; the provenance,
    # because only the fake runner wrote the scenario files.
    assert set(traces) == set(IDS), (
        "`load_traces` returned a scenario that is not in the manifest — the "
        "capture record has been read as a trace")
    assert len(traces) == len(list(out.glob("*.yaml"))) - 1, (
        "exactly one file in the capture directory is not a trace; a second file "
        "skipped silently would be a scenario that never scored")
    assert {t.get("captured_by") for t in traces.values()} == {"fake-engine"}, (
        "a trace not written by the runner is a file `load_traces` picked up from "
        "somewhere else, `capture.yaml` included")
    meta = B.load_capture_meta(out)
    assert meta["schema"] == B.CAPTURE_SCHEMA
    assert meta["run_id"] == RUN_ID
    assert [row["id"] for row in meta["scenarios"]] == IDS
    assert {row["status"] for row in meta["scenarios"]} == {CAP.CAPTURED}


def test_the_round_scores_a_capture_and_names_the_directory_it_scored(tmp_path):
    """Clause 1's other end: the scorecard says `capture (<path>)`, not reference."""
    cfg = _cfg(tmp_path)
    CAP.capture_round(cfg=cfg, run_id=RUN_ID, runner=_runner())
    out = CAP.capture_dir(cfg, RUN_ID)

    scorecard = B.round_scorecard(cfg, RUN_ID)

    assert scorecard["trace_source"].startswith("capture"), scorecard["trace_source"]
    assert str(out) in scorecard["trace_source"], (
        "the artifact has to name which capture it scored, so a reader can open "
        "that capture's own record")
    assert scorecard["trace_source"] != "reference"
    assert scorecard["reference_replay"] is False, (
        "a real capture is not a replay of the pinned baseline's own traces")
    # #1843 clause 4: one scenario of the five declares `capture: none`, so a
    # capture that ran all five still scores four — `captured` counts what the
    # runner produced, the denominator counts what any capture can measure.
    assert scorecard["scenarios_total"] == len(IDS)
    assert scorecard["denominator"] == len(IDS) - len(NO_CAPTURE_PATH_IDS)
    assert scorecard["instrument_failures"] == NO_CAPTURE_PATH_IDS
    assert scorecard["capture"]["captured"] == len(IDS)
    assert scorecard["capture"]["not_captured"] == []


# ── clause 2: a scenario that never ran is an instrument failure with a cause ─

def test_a_scenario_the_budget_never_reached_reports_the_budget_and_leaves_its_axis(
        tmp_path):
    """600 s cap, two 300 s scenarios: the remaining three are failures, not zeros.

    The ruled cap is 10 minutes of GPU per round. Three scenarios are left
    behind here, and the whole point of the clause is that they come back as
    `instrument_failure: true` naming the budget, and their axes report
    `denominator: 0` with `value: None` — never a 0.0 that would average into
    the axis and read as a behavioural decline the run never measured.
    """
    cfg = _cfg(tmp_path)
    clock = Clock()
    meta = CAP.capture_round(cfg=cfg, run_id=RUN_ID, clock=clock,
                             runner=_runner(cost=300.0, clock=clock),
                             budget_seconds=BUDGET)

    assert [row["status"] for row in meta["scenarios"]] == [
        CAP.CAPTURED, CAP.CAPTURED, CAP.SKIPPED_BUDGET, CAP.SKIPPED_BUDGET,
        CAP.SKIPPED_BUDGET]
    assert meta["budget_exhausted"] is True and meta["elapsed_seconds"] == 600.0
    for row in meta["scenarios"][2:]:
        assert "budget" in row["reason"] and "600" in row["reason"], row["reason"]
        assert row["trace"] is None

    scorecard = B.round_scorecard(cfg, RUN_ID)
    assert scorecard["denominator"] == 2
    assert scorecard["instrument_failures"] == IDS[2:]
    axes = {a["axis"]: a for a in scorecard["axes"]}
    for axis_name in ("action_consistency", "stale_fact_action"):
        axis = axes[axis_name]
        assert axis["denominator"] == 0 and axis["value"] is None, (
            f"{axis_name} was never measured and must not report a value: {axis}")
        assert axis["delta"] is None and axis["declines_beyond_epsilon"] is False
    measured = [a for a in scorecard["axes"] if a["denominator"] > 0]
    assert len(measured) == 2 and all(a["value"] is not None for a in measured)
    for sid in IDS[2:]:
        row = next(r for r in scorecard["scenarios"] if r["id"] == sid)
        assert row["ran"] == 0 and row["instrument_failure"] is True
        assert "budget" in row["observed"]["reason"], row["observed"]["reason"]
    # `guardrail_hit: false` here is a claim about the axes that WERE measured:
    # every axis that lost all its scenarios prints a zero denominator and no
    # value above (which is what keeps the False from reading as a whole-suite
    # clean bill), and no scored axis declined past its epsilon.
    assert scorecard["guardrail_hit"] is False
    for axis in scorecard["axes"]:
        if axis["denominator"] == 0:
            assert axis["value"] is None, axis


def test_a_scenario_whose_run_raised_reports_the_error_and_is_not_scored_as_zero(
        tmp_path):
    """One scenario dying takes its axis out of the denominator, not to zero.

    The capturer keeps going: losing one trace is one axis reported as
    unmeasured, while aborting the round loses four more and hides which
    scenario was the problem.
    """
    cfg = _cfg(tmp_path)
    meta = CAP.capture_round(cfg=cfg, run_id=RUN_ID,
                             runner=_runner(raise_for=("source-retention",)))

    row = next(r for r in meta["scenarios"] if r["id"] == "source-retention")
    assert row["status"] == CAP.FAILED and row["trace"] is None
    assert "engine refused the run for source-retention" in row["reason"]
    assert len(meta["scenarios"]) == len(IDS), "the capture continued past the failure"

    scorecard = B.round_scorecard(cfg, RUN_ID)
    # This capture lost one scenario of its own, and the suite always loses the
    # scenario that declares `capture: none` (#1843 clause 4) — so the denominator
    # is the suite minus both, in manifest order.
    assert scorecard["denominator"] == len(IDS) - 1 - len(NO_CAPTURE_PATH_IDS)
    assert scorecard["instrument_failures"] == ["source-retention", *NO_CAPTURE_PATH_IDS]
    source = next(a for a in scorecard["axes"] if a["axis"] == "source_retention")
    assert source["denominator"] == 0 and source["value"] is None
    scored_row = next(r for r in scorecard["scenarios"] if r["id"] == "source-retention")
    assert "RuntimeError" in scored_row["observed"]["reason"], scored_row["observed"]["reason"]


def test_a_returned_object_with_no_observations_writes_no_trace(tmp_path):
    """A runner that hands back a result stub is an instrument failure, not a zero.

    The graders read the five observable lists; a `{"status": "success",
    "final_text": …}` — the shape a bench trial returns — has none of them, so it
    would write a file that reads as a captured run while some graders call it
    `ran: 0` and others call it a real zero. That is exactly the ambiguity clause
    2 removes, so the capturer refuses to write it and names the cause instead.
    """
    cfg = _cfg(tmp_path)

    def stub(scenario: dict) -> dict:
        if str(scenario["id"]) == "act-on-known-fact":
            return {"status": "success", "final_text": "done"}
        return _healthy_trace(scenario)

    meta = CAP.capture_round(cfg=cfg, run_id=RUN_ID, runner=stub)
    out = CAP.capture_dir(cfg, RUN_ID)

    row = next(r for r in meta["scenarios"] if r["id"] == "act-on-known-fact")
    assert row["status"] == CAP.INVALID and row["trace"] is None
    assert "observable" in row["reason"], row["reason"]
    assert not (out / "act-on-known-fact.yaml").exists()
    scorecard = B.round_scorecard(cfg, RUN_ID)
    # The scenario this capture broke, plus the one no capture can measure
    # (#1843 clause 4), and nothing else.
    assert scorecard["instrument_failures"] == ["act-on-known-fact",
                                               *NO_CAPTURE_PATH_IDS]
    action = next(a for a in scorecard["axes"] if a["axis"] == "action_consistency")
    # Before #1843 this asserted `denominator: 1` — the other scenario on the axis
    # still counting while only the broken one left. It now leaves too: the axis's
    # two scenarios were `act-on-known-fact` and `blocked-route-replan`, and the
    # second is the one the manifest declares no capture can measure, so an axis
    # that lost both reports no value rather than the value of one run.
    assert action["denominator"] == 0 and action["value"] is None, action


def test_a_missing_trace_with_no_capture_record_keeps_the_plain_reason(tmp_path):
    """No capture, no claim about why: the reference replay still says `no trace`."""
    cfg = _cfg(tmp_path)
    out = CAP.capture_dir(cfg, RUN_ID)
    out.mkdir(parents=True)
    (out / f"{IDS[0]}.yaml").write_text(
        yaml.safe_dump(_healthy_trace(SCENARIOS[0]), sort_keys=True), encoding="utf-8")

    scorecard = B.round_scorecard(cfg, RUN_ID)

    assert B.load_capture_meta(out) is None
    missing = [r for r in scorecard["scenarios"] if r["instrument_failure"]]
    assert len(missing) == len(IDS) - 1
    for row in missing:
        assert row["observed"]["reason"] == "no trace captured", (
            "a capturer that recorded no cause must not have one invented for it")


def test_a_scenario_id_that_is_not_a_bare_name_is_refused(tmp_path):
    """Both files the capturer writes are named after the id, so the id is checked.

    A manifest is reviewed text today and editable input tomorrow, and an id of
    `../../.ssh/authorized_keys` would turn a measurement job into a write
    primitive reachable by whoever edits a YAML file. The refusal must also leave
    nothing behind, and a run must not be attempted.
    """
    cfg = _cfg(tmp_path)
    plant_root = tmp_path / "plant"
    sneaky = {"id": "../escaped", "axis": "action_consistency",
              "checker": "tool_arg_uses_planted_path",
              "planted_input": {"kind": "fact", "text": "x"}}

    with pytest.raises(ValueError, match="not usable as a filename"):
        CAP.plant_input(cfg, sneaky, plant_root=plant_root)
    assert not (tmp_path / "escaped.yaml").exists()
    assert not (tmp_path / "plant").exists(), (
        "the refusal created the tree it then declined to write into")

    with pytest.raises(ValueError, match="no planted_input"):
        CAP.plant_input(cfg, {"id": "bare", "axis": "source_retention",
                              "checker": "names_its_source"},
                        plant_root=plant_root)

    with pytest.raises(ValueError, match="only `then_run`"):
        CAP.plant_input(cfg, {"id": "scripted", "axis": "source_retention",
                              "checker": "names_its_source",
                              "planted_input": {"then_run": "just answer"}},
                        plant_root=plant_root)


def test_a_fresh_process_scoring_a_capture_reads_what_the_writer_left(tmp_path):
    """The writer and the reader are different jobs, so the file is the contract.

    A capture runs when a human is at the machine; the round that scores it is a
    later process that shares nothing with it but `research_root`. Anything the
    round needs to tell a capture from a replay — the source line, the causes, the
    denominators — therefore has to survive the trip through disk. This node runs
    the reader in a brand-new interpreter with no test state in it.
    """
    import json
    import subprocess

    cfg = _cfg(tmp_path)
    clock = Clock()
    # 700 s against a 600 s round: the first scenario is attempted (nothing has
    # elapsed yet) and every scenario after it is out of budget, so exactly one
    # trace exists on disk and four scenarios have to report a cause.
    CAP.capture_round(cfg=cfg, run_id=RUN_ID,
                      runner=_runner(cost=700.0, clock=clock),
                      plant_root=tmp_path / "planted", clock=clock)

    reader = f"""
import json, sys
sys.path.insert(0, {str(ROOT)!r})
from types import SimpleNamespace
from pathlib import Path
from scripts.autoresearch import behavioural as B
root = Path({str(cfg.paths.research_root)!r})
cfg = SimpleNamespace(paths=SimpleNamespace(
    research_root=root, rounds_dir=root / 'rounds', variants_dir=root / 'variants'))
card = B.round_scorecard(cfg, {RUN_ID!r})
lines = B.scorecard_report_lines(card)
print(json.dumps({{
    'trace_source': card['trace_source'],
    'reference_replay': card['reference_replay'],
    'failures': card['instrument_failures'],
    'reasons': {{r['id']: r['observed'].get('reason', '')
                 for r in card['scenarios'] if r['instrument_failure']}},
    'denominator': card['denominator'],
    'report_has_capture': any('traces: capture (' in l for l in lines),
}}))
"""
    out = subprocess.run([sys.executable, "-c", reader], capture_output=True,
                         text=True, cwd=str(ROOT), timeout=120)
    assert out.returncode == 0, out.stderr[-1500:]
    seen = json.loads(out.stdout.strip().splitlines()[-1])

    assert seen["trace_source"] == f"capture ({CAP.capture_dir(cfg, RUN_ID)})", (
        "a later process has to be able to tell this round's section was scored "
        f"from a capture, got {seen['trace_source']!r}")
    assert seen["reference_replay"] is False
    assert seen["failures"] == IDS[1:]
    for sid in IDS[1:]:
        assert "budget" in seen["reasons"][sid] and "600" in seen["reasons"][sid], \
            (sid, seen["reasons"][sid])
    assert seen["denominator"] == 1
    assert seen["report_has_capture"], "the section the human reads lost the source"


def test_a_trace_belonging_to_another_scenario_is_not_filed_under_this_one(tmp_path):
    """`load_traces` keys by the id INSIDE the file, so a wrong id mis-scores.

    A runner that hands back one scenario's run under another scenario's name
    would be graded against the wrong expectation and read as a real result —
    silently. The capturer checks the id it wrote the file under against the id in
    the trace, and refuses the file, naming the mismatch as the cause.
    """
    cfg = _cfg(tmp_path)

    def swapped(scenario: dict) -> dict:
        trace = _healthy_trace(scenario)
        if str(scenario["id"]) == "source-retention":
            trace["scenario_id"] = "act-on-known-fact"
        return trace

    meta = CAP.capture_round(cfg=cfg, run_id=RUN_ID, runner=swapped)
    row = next(r for r in meta["scenarios"] if r["id"] == "source-retention")
    assert row["status"] == CAP.INVALID and row["trace"] is None
    assert "scenario_id" in row["reason"], row["reason"]
    assert not (CAP.capture_dir(cfg, RUN_ID) / "source-retention.yaml").exists()
    scorecard = B.round_scorecard(cfg, RUN_ID)
    assert "source-retention" in scorecard["instrument_failures"], (
        "the mismatch has to surface as an instrument failure, not as a score "
        "against the wrong scenario's expectation")
    source_axis = next(a for a in scorecard["axes"]
                       if a["axis"] == "source_retention")
    assert source_axis["denominator"] == 0 and source_axis["value"] is None


# ── the production runner: a bench trial record is not a whole-run trace ──────

def _trial(*, status: str = "success", final_text: str = "",
           tool_calls: list | None = None, denied_calls: list | None = None,
           unresolved_calls: list | None = None, error: str = "") -> dict:
    """The dict `bench_runner_sdk.run_trial` actually returns.

    Copied in shape from `bench_runner_sdk.py:587-626` (`status`, `final_text`,
    `turns`, the three call lists, timings) rather than invented, because the
    whole point of the nodes below is that the capturer reads a record whose
    vocabulary is not the graders': it has `final_text` where a grader wants
    `answers`, `denied_calls` where one wants `tool_calls`, and no `scenario_id`
    at all — which is the field `load_traces` keys every file by.
    """
    return {
        "variant_id": "BASELINE_TEST", "task_id": "behavioural:unnamed",
        "harness": "claude-sdk", "status": status, "final_text": final_text,
        "turns": 3, "tool_calls": tool_calls or [],
        "denied_calls": denied_calls or [], "unresolved_calls": unresolved_calls or [],
        "duration_seconds": 12.5, "error": error, "stop_reason": "end_turn",
        "session_id": "S_test", "trial_id": "T_test",
    }


def _scenario(sid: str) -> dict:
    return next(s for s in SCENARIOS if str(s["id"]) == sid)


def test_a_bench_trial_becomes_a_trace_keyed_by_the_scenario_it_ran():
    """The mapper is what makes the production runner's output readable at all.

    A trial record carries `task_id="behavioural:<id>"` and no `scenario_id`, and
    `load_traces` files a trace under the id written INSIDE the file: copied
    through unchanged, the capturer's own directory would make `round_scorecard`
    return `status: refused`. The mapping choices are asserted where they matter —
    a denied call keeps its arguments, because the action-consistency grader asks
    what the run *tried* to address, and the trial's own naming never reaches the
    trace.
    """
    scenario = _scenario("act-on-known-fact")
    trial = _trial(
        final_text="the reports live under the pipeline tree",
        tool_calls=[{"name": "Read", "is_error": False,
                     "args": {"path": "~/lloyd-data/_pipeline/research/rounds/R_1.md"}}],
        denied_calls=[{"name": "vault_write", "denied": True,
                       "deny_reason": "path is outside the trial sandbox",
                       "args": {"path": "~/notes.md", "content": "x"}}],
        unresolved_calls=[{"name": "Grep", "reason": "no tool_result before the stream ended",
                          "args": {"path": "~/pipeline-legacy/reflection"}}])

    trace = CAP.trace_from_trial(trial, scenario)

    assert trace["scenario_id"] == "act-on-known-fact", (
        "the trace is keyed by the scenario it ran, never by the trial's own "
        f"`task_id`: {trace['scenario_id']!r}")
    assert trace["answers"] == ["the reports live under the pipeline tree"]
    assert [c["name"] for c in trace["tool_calls"]] == ["Read", "vault_write", "Grep"], (
        "the denied and unanswered calls are actions the run took, and the axis "
        "here is about the arguments it chose")
    assert any(e["kind"] == "tool_denied" and "outside the trial sandbox" in e["detail"]
               for e in trace["events"]), "the fate of the call must be readable"
    assert any(e["kind"] == "tool_unresolved" for e in trace["events"])
    assert trace["durable_writes"] == [], "a refused write wrote nothing"
    assert trace["trial"]["status"] == "success"

    graded = B.grade_tool_arg_uses_planted_path(trace, scenario)
    assert graded["ran"] == 2 and graded["matched"] == 1, graded
    assert graded["observed"]["retired_path_uses"] == ["tool_call:2:Grep"], (
        "the retired root reaching an argument has to show up as the row it is, "
        "not as a vague zero")


def test_only_a_write_the_trial_succeeded_at_becomes_a_durable_write():
    """Denied and errored writes score nothing; a real one scores what it wrote.

    `uncertainty-hardening` asks whether a hedged, attributed claim survived
    consolidation into a durable record. The trial harness denies the mutating
    tools, so through `sdk_runner` this axis cannot be measured — and the honest
    report is `ran: 0` / `instrument_failure: true`, never a write credited for a
    call the harness refused.
    """
    scenario = _scenario("uncertainty-hardening")
    kept = {"name": "vault_write", "is_error": False,
            "args": {"path": "memory/knowledge/relay.md",
                     "content": "Maya said the billing-east relay might move to 7788."}}
    refused = {"name": "vault_write", "denied": True,
               "deny_reason": "mutating tool denied in a bench trial",
               "args": {"path": "memory/knowledge/relay.md",
                        "content": "Maya said the billing-east relay might move to 7788."}}
    broken = {"name": "Write", "is_error": True,
              "args": {"file_path": "memory/knowledge/broken.md", "content": "half a write"}}

    trace = CAP.trace_from_trial(_trial(final_text="noted", tool_calls=[kept, refused, broken]),
                                 scenario)
    assert trace["durable_writes"] == [{"path": "memory/knowledge/relay.md",
                                        "text": "Maya said the billing-east relay might move to 7788."}], (
        "exactly one of the three write calls put anything on disk")
    graded = B.grade_hedge_preserved(trace, scenario)
    assert graded["ran"] == 1 and graded["matched"] == 1 and graded["value"] == 1.0, graded

    denied_only = CAP.trace_from_trial(_trial(final_text="noted", tool_calls=[refused]),
                                       scenario)
    assert denied_only["durable_writes"] == []
    unmeasured = B.grade_hedge_preserved(denied_only, scenario)
    assert unmeasured["ran"] == 0 and unmeasured["instrument_failure"] is True, (
        "a run whose every write was refused measured nothing about durable "
        "writes; it did not score a zero")


def test_a_trial_that_neither_acted_nor_answered_is_a_failure_not_an_empty_trace(tmp_path):
    """A run that never happened is recorded with its cause, not as a blank trace.

    Writing an empty trace would put the scenario in the scorecard as
    `captured` with every grader at `ran: 0`, which reads as a behavioural
    collapse. Raising instead routes it to clause 2: the capture row is `failed`,
    the trial's own status and error are the reason, and no file pretends.
    """
    cfg = _cfg(tmp_path)
    scenario = _scenario("source-retention")
    with pytest.raises(ValueError, match="no tool call and no final answer"):
        CAP.trace_from_trial(_trial(status="error", error="harness reported stop_reason=error"),
                             scenario)

    def dead(scenario: dict) -> dict:
        return CAP.trace_from_trial(
            _trial(status="error", error="harness reported stop_reason=error"), scenario)

    meta = CAP.capture_round(cfg=cfg, run_id=RUN_ID, runner=dead)
    out = CAP.capture_dir(cfg, RUN_ID)
    assert {row["status"] for row in meta["scenarios"]} == {CAP.FAILED}
    assert {p.name for p in out.glob("*.yaml")} == {B.CAPTURE_META_FILENAME}, (
        "an empty trace would be a scenario reported as captured with nothing in it")

    scorecard = B.round_scorecard(cfg, RUN_ID)
    assert scorecard["denominator"] == 0 and scorecard["instrument_failures"] == IDS
    for row in scorecard["scenarios"]:
        assert "no tool call and no final answer" in row["observed"]["reason"], row
    for axis in scorecard["axes"]:
        assert axis["denominator"] == 0 and axis["value"] is None, axis
    # `guardrail_hit: false` here is not a clean bill: every axis reports no
    # value above, which is what keeps the False meaning "nothing was measured
    # enough to decline" rather than "nothing regressed".
    assert scorecard["guardrail_hit"] is False


#: One canned trial per scenario, in the shape `run_trial` returns, for the
#: seam node below. Three of the five are chosen so their graders have rows;
#: the other two are the two the bench harness cannot supply evidence for.
TRIALS: dict[str, dict] = {
    "uncertainty-hardening": _trial(
        final_text="Maya said the billing-east relay might move.",
        denied_calls=[{"name": "vault_write", "denied": True,
                       "deny_reason": "mutating tool denied in a bench trial",
                       "args": {"path": "memory/knowledge/relay.md",
                                "content": "Maya said the relay might move."}}]),
    "source-retention": _trial(
        final_text="The staging deploy codename is QUARTZ-HERON-2291, per Alan."),
    "act-on-known-fact": _trial(
        final_text="found the round reports",
        tool_calls=[
            {"name": "Read", "is_error": False,
             "args": {"path": "~/lloyd-data/_pipeline/research/rounds/R_1.md"}},
            {"name": "Glob", "is_error": False,
             "args": {"path": "~/lloyd-data/_pipeline/research/rounds", "pattern": "*.md"}}]),
    "stale-fact-action": _trial(
        final_text="The relay is on 7788 now.",
        tool_calls=[{"name": "Bash", "is_error": False,
                     "args": {"command": "curl relay:7788/health"}}]),
    "blocked-route-replan": _trial(
        final_text="the route is gone, so I stopped",
        tool_calls=[{"name": "Read", "is_error": False,
                     "args": {"path": "~/lloyd-data/_pipeline/research/rounds/none.md"}}]),
}


async def _trials_through_sdk(task, variant_id, overlay_dir, model, **kwargs):
    """`run_trial`'s return contract, one canned trial per scenario id."""
    return TRIALS[str(task["id"]).split(":", 1)[1]]


def test_the_production_runner_leaves_traces_the_scorer_reads(tmp_path, monkeypatch):
    """The boundary the review kept asking after, with the engine stubbed.

    `sdk_runner` → `bench_runner_sdk.run_trial` → YAML on disk → `load_traces` in
    the scoring path. What a stub proves is everything on this side of the
    harness: the call convention (one trial per scenario, `task_id`
    `behavioural:<id>`, the baseline overlay, the scenario's `then_run` in the
    prompt) and the round trip from the record `run_trial` returns to the file
    the scorer accepts — which is the half that was broken: an unmapped trial
    carries no `scenario_id`, so `load_traces` refuses the directory and the
    scorecard comes back `status: refused`. What it cannot prove is the wall
    clock and the tool surface of a real run: #1659's owed items 1 and 3.
    """
    cfg = _cfg(tmp_path)
    seen: list[dict] = []

    async def fake_run_trial(task, variant_id, overlay_dir, model, **kwargs):
        seen.append({"task": task, "variant_id": variant_id,
                     "overlay": str(overlay_dir), "model": model})
        return TRIALS[str(task["id"]).split(":", 1)[1]]

    monkeypatch.setattr("scripts.autoresearch.bench_runner_sdk.run_trial", fake_run_trial)
    monkeypatch.setattr("scripts.autoresearch.variant_sandbox.materialize_baseline",
                        lambda cfg: ("BASELINE_TEST", tmp_path / "overlay"))

    meta = CAP.capture_round(cfg=cfg, run_id=RUN_ID, runner=CAP.sdk_runner(cfg),
                             plant_root=tmp_path / "planted")
    out = CAP.capture_dir(cfg, RUN_ID)

    assert [row["status"] for row in meta["scenarios"]] == [CAP.CAPTURED] * len(IDS)
    assert {p.name for p in out.glob("*.yaml")} == (
        {f"{sid}.yaml" for sid in IDS} | {B.CAPTURE_META_FILENAME})
    assert [s["task"]["id"] for s in seen] == [f"behavioural:{sid}" for sid in IDS]
    for call, scenario in zip(seen, SCENARIOS):
        assert scenario["planted_input"]["then_run"] in call["task"]["prompt"], scenario["id"]
        assert f"{scenario['id']}.yaml" in call["task"]["prompt"], (
            "the run has to be told where the state it must discover was planted")
    assert all(s["variant_id"] == "behavioural_capture" and s["model"] == "test-model"
               for s in seen), "a capture measures behaviour, so it runs the baseline"

    scorecard = B.round_scorecard(cfg, RUN_ID)
    rows = {str(r["id"]): r for r in scorecard["scenarios"]}
    assert scorecard["trace_source"] == f"capture ({out})"
    assert scorecard["reference_replay"] is False
    assert scorecard["status"] == "scored", (
        "an unmapped trial makes `load_traces` refuse the whole directory")
    for sid in ("act-on-known-fact", "source-retention", "stale-fact-action"):
        assert rows[sid]["value"] == 1.0, rows[sid]
    assert rows["uncertainty-hardening"]["instrument_failure"] is True, (
        "the trial harness denied its only write, so nothing was written to grade")
    assert rows["blocked-route-replan"]["instrument_failure"] is True, (
        "the harness emits no route_blocked/plan_revised pair; the mapper "
        "invents neither")
    assert scorecard["instrument_failures"] == ["uncertainty-hardening",
                                               "blocked-route-replan"]
    assert scorecard["denominator"] == 3, (
        "three of five scenarios are measurable through this runner; the other "
        "two leave their axis denominator instead of scoring a zero")
    replan = next(a for a in scorecard["axes"] if a["axis"] == "action_consistency")
    assert replan["denominator"] == 1 and replan["value"] == 1.0, replan


# ── #1843 clause 1: `--writes-into`, and a no-flag run that changes nothing ──

def test_the_cli_names_one_writable_directory_and_the_capture_seeds_it_there(
        tmp_path, monkeypatch):
    """`--writes-into <dir>` is the operator's one knob, and it is honoured.

    The flag has to reach the capture (not just parse), seed the plant root there
    so the planted state and the credited writes are in one named place, and be
    recorded in `capture.yaml` — the reader of a scorecard row checks the claim
    against that file, not against this module's defaults.
    """
    cfg = _cfg(tmp_path)
    root = tmp_path / "operator-named"
    given: list[dict] = []

    monkeypatch.setattr("scripts.autoresearch.common.load_config", lambda: cfg)
    monkeypatch.setattr("scripts.autoresearch.variant_sandbox.materialize_baseline",
                        lambda cfg: ("BASELINE_TEST", tmp_path / "overlay"))
    real_sdk_runner = CAP.sdk_runner

    def spy(cfg_arg, **kwargs):
        given.append(kwargs)
        assert cfg_arg is cfg, "the CLI passes its own config to the runner"
        return real_sdk_runner(cfg, **{k: v for k, v in kwargs.items() if k != "model"})

    monkeypatch.setattr(CAP, "sdk_runner", spy)
    monkeypatch.setattr("scripts.autoresearch.bench_runner_sdk.run_trial",
                        _trials_through_sdk)

    CAP.main(["--run-id", RUN_ID, "--yes", "--writes-into", str(root)])

    assert given and given[0]["writes_into"] == root, (
        "the flag has to reach the runner, or the trial is never told where to write")
    meta = B.load_capture_meta(CAP.capture_dir(cfg, RUN_ID))
    assert meta["writes_into"] == str(root)
    assert meta["plant_root"] == str(root), (
        "the plant root is seeded from the named directory, so a capture's planted "
        "state and its credited writes are the one directory the operator named")
    # `planted` is relative to the plant root by design — it is the path the run is
    # told to read — so the claim checked here is that the root it is relative TO is
    # the directory the operator named, and that the files are really in it.
    planted = [Path(meta["plant_root"]) / row["planted"] for row in meta["scenarios"]]
    assert planted and all(p.is_relative_to(root) for p in planted), planted
    assert all(p.is_file() for p in planted), (
        "the named directory has to be created and seeded, not merely recorded")


def test_a_capture_run_without_the_flag_seeds_the_default_and_stays_captured(tmp_path):
    """Clause 1's other half: no flag, no change in what the capture records.

    Before `--writes-into` existed the plant root was
    `behavioural_plant/<run_id>/` under the research root and every scenario
    reported `captured`. A flag that silently moved the plant root, or that made a
    scenario behave differently when unset, would change the measurement the
    report-only rung is accumulating weeks of.
    """
    cfg = _cfg(tmp_path)
    meta = CAP.capture_round(cfg=cfg, run_id=RUN_ID, runner=_runner())

    assert meta["writes_into"] is None, "no bound named, so none enforced"
    assert meta["plant_root"] == str(CAP.default_plant_root(cfg, RUN_ID))
    assert meta["plant_root"] == str(
        Path(cfg.paths.research_root) / "behavioural_plant" / RUN_ID)
    assert [row["status"] for row in meta["scenarios"]] == [CAP.CAPTURED] * len(IDS)
    assert all(Path(meta["plant_root"], row["planted"]).is_file()
               for row in meta["scenarios"])


def test_the_module_no_longer_names_a_flag_argparse_has_never_had():
    """The docstrings told the operator to pass `--plant-into`; argparse refused it.

    The first live capture is exactly the operator path that reads those lines, so
    the mismatch is not cosmetic: `--plant-into <dir>` died on `unrecognized
    arguments` while the plant root it described was settable only in-process. The
    flag that exists now is `--writes-into`, and the prose names it.
    """
    source = Path(CAP.__file__).read_text(encoding="utf-8")
    assert "plant-into" not in source, (
        "a docstring naming a flag argparse does not define is the defect being fixed")

    assert "--writes-into" in source, "the flag that replaced it has to exist"
    with pytest.raises(SystemExit):
        # Not a grep of the text: argparse itself rejects the flag the old prose
        # named, which is the operator's experience of the mismatch.
        CAP.main(["--run-id", RUN_ID, "--yes", "--plant-into", str(Path("/tmp/x"))])


# ── #1843 clause 2: a credited durable write, so the axis has a number ──────

def _write_trial(root: Path, *, path: str, hedge: bool = True) -> dict:
    """A trial record whose run wrote `path` successfully, hedged or hardened."""
    text = ("Maya said the billing-east relay might move to port 7788 next quarter."
            if hedge else
            "The billing-east relay moves to port 7788 next quarter.")
    return _trial(final_text="consolidated the relay note",
                  tool_calls=[{"name": "Write", "is_error": False,
                               "args": {"file_path": path, "content": text}}])


def test_a_write_inside_the_named_root_gives_the_axis_its_first_number(
        tmp_path, monkeypatch):
    """Clause 2, measured through the production runner's own seam.

    `uncertainty-hardening` has been an instrument failure since the suite was
    written because the only write it could report was a denied one. With a
    successful write inside the directory the capture named, the grader has a row:
    the scenario row carries a value with `instrument_failure: false`, and the
    axis reports a value and a denominator of at least one.
    """
    cfg = _cfg(tmp_path)
    root = tmp_path / "writes"

    async def fake_run_trial(task, variant_id, overlay_dir, model, **kwargs):
        sid = str(task["id"]).split(":", 1)[1]
        if sid == "uncertainty-hardening":
            return _write_trial(root, path=str(root / "relay.md"))
        return TRIALS[sid]

    monkeypatch.setattr("scripts.autoresearch.bench_runner_sdk.run_trial", fake_run_trial)
    monkeypatch.setattr("scripts.autoresearch.variant_sandbox.materialize_baseline",
                        lambda cfg: ("BASELINE_TEST", tmp_path / "overlay"))

    CAP.capture_round(cfg=cfg, run_id=RUN_ID,
                      runner=CAP.sdk_runner(cfg, writes_into=root), writes_into=root)
    scorecard = B.round_scorecard(cfg, RUN_ID)
    rows = {str(r["id"]): r for r in scorecard["scenarios"]}
    row = rows["uncertainty-hardening"]

    assert row["instrument_failure"] is False, row
    assert row["value"] == 1.0, (
        f"the hedged write is inside the named root and must grade: {row}")
    assert row["observed"]["false_certainty_rate"] == 0.0, row["observed"]
    axis = next(a for a in scorecard["axes"] if a["axis"] == "uncertainty_preservation")
    assert axis["denominator"] >= 1 and axis["value"] == 1.0, axis
    trace = yaml.safe_load((CAP.capture_dir(cfg, RUN_ID)
                            / "uncertainty-hardening.yaml").read_text())
    assert [w["path"] for w in trace["durable_writes"]] == [str(root / "relay.md")], (
        "the trace is the artifact the axis was computed from, so the credited write "
        "has to be visible in it")


def test_a_hardened_write_inside_the_root_scores_zero_not_an_instrument_failure(
        tmp_path, monkeypatch):
    """The bound must not turn the axis into an always-1.0 rubber stamp.

    Same seam, same directory, one difference: the run dropped `might`. That is a
    measured behavioural failure (value 0.0 with the false-certainty rate naming
    it), which is what makes clause 2's 1.0 a measurement rather than a default.
    """
    cfg = _cfg(tmp_path)
    root = tmp_path / "writes"

    async def fake_run_trial(task, variant_id, overlay_dir, model, **kwargs):
        sid = str(task["id"]).split(":", 1)[1]
        if sid == "uncertainty-hardening":
            return _write_trial(root, path=str(root / "relay.md"), hedge=False)
        return TRIALS[sid]

    monkeypatch.setattr("scripts.autoresearch.bench_runner_sdk.run_trial", fake_run_trial)
    monkeypatch.setattr("scripts.autoresearch.variant_sandbox.materialize_baseline",
                        lambda cfg: ("BASELINE_TEST", tmp_path / "overlay"))

    CAP.capture_round(cfg=cfg, run_id=RUN_ID,
                      runner=CAP.sdk_runner(cfg, writes_into=root), writes_into=root)
    scorecard = B.round_scorecard(cfg, RUN_ID)
    row = next(r for r in scorecard["scenarios"] if r["id"] == "uncertainty-hardening")

    assert row["instrument_failure"] is False and row["value"] == 0.0, row
    assert row["observed"]["false_certainty_rate"] == 1.0, row["observed"]


# ── #1843 clause 3: the allowance is bounded by the named directory ─────────

def test_a_write_outside_the_named_root_earns_the_capture_no_credit(
        tmp_path, monkeypatch):
    """Clause 3: a successful write elsewhere is not this capture's durable write.

    The trial says it wrote `/home/alansrobotlab/elsewhere/relay.md` and the
    harness agreed (no `is_error`, no denial). Crediting it would score the run
    for a side effect outside the directory the operator sanctioned — and would
    leave the bound existing only in the prompt. So no row, and the scenario is
    back to being an instrument failure with its axis unmeasured.
    """
    cfg = _cfg(tmp_path)
    root = tmp_path / "writes"
    elsewhere = tmp_path / "elsewhere" / "relay.md"

    async def fake_run_trial(task, variant_id, overlay_dir, model, **kwargs):
        sid = str(task["id"]).split(":", 1)[1]
        if sid == "uncertainty-hardening":
            return _write_trial(root, path=str(elsewhere))
        return TRIALS[sid]

    monkeypatch.setattr("scripts.autoresearch.bench_runner_sdk.run_trial", fake_run_trial)
    monkeypatch.setattr("scripts.autoresearch.variant_sandbox.materialize_baseline",
                        lambda cfg: ("BASELINE_TEST", tmp_path / "overlay"))

    CAP.capture_round(cfg=cfg, run_id=RUN_ID,
                      runner=CAP.sdk_runner(cfg, writes_into=root), writes_into=root)
    scorecard = B.round_scorecard(cfg, RUN_ID)
    row = next(r for r in scorecard["scenarios"] if r["id"] == "uncertainty-hardening")

    assert row["instrument_failure"] is True, (
        f"a write outside the named root must not be credited: {row}")
    axis = next(a for a in scorecard["axes"] if a["axis"] == "uncertainty_preservation")
    assert axis["denominator"] == 0 and axis["value"] is None, axis
    trace = yaml.safe_load((CAP.capture_dir(cfg, RUN_ID)
                            / "uncertainty-hardening.yaml").read_text())
    assert trace["durable_writes"] == [], "the bound is visible in the artifact"
    assert trace["tool_calls"], (
        "the attempt is still in the trace — the bound hides nothing, it withholds "
        "credit")


@pytest.mark.parametrize("target, inside", [
    # A relative target is read against the named root: that is the directory the
    # run was told to write under, and nothing else.
    ("relay.md", True),
    ("notes/relay.md", True),
    ("./relay.md", True),
    ("../outside.md", False),
    ("notes/../../outside.md", False),
])
def test_the_write_bound_resolves_the_target_it_is_given(target: str, inside: bool,
                                                        tmp_path):
    """The bound is a resolved-path test, and `..` is its whole difficulty."""
    root = tmp_path / "writes"
    root.mkdir()
    assert CAP.write_path_inside_root(target, root) is inside, target


def test_the_trial_write_scope_denies_a_target_outside_the_named_directory(tmp_path):
    """The harness half of clause 3: the deny is the mechanism, not the prompt.

    Installed on a real `HookRegistry` and fired the way the loop fires it, so
    what is measured is the registry's verdict on a call, not this module's opinion
    about one. A `Write` inside passes, a `Write` outside is denied with the
    directory named, and a tool that names no writable target is untouched — the
    scope adds a bound and takes away no read.
    """
    import asyncio

    from app.harness import HookRegistry

    from scripts.autoresearch import bench_runner_sdk as SDK

    root = tmp_path / "writes"
    root.mkdir()
    hooks = HookRegistry()
    SDK.install_trial_write_scope(hooks, root)

    async def fire(tool_name: str, tool_input: dict) -> dict:
        return await hooks.fire_pre_tool_use(session_id="bench_scope_probe",
                                            tool_name=tool_name,
                                            tool_input=tool_input)

    assert asyncio.run(fire("Write", {"file_path": str(root / "a.md")})) == {}, (
        "a write inside the named directory is not the thing being refused")
    assert asyncio.run(fire("Write", {"file_path": "a.md"})) == {}, (
        "a relative target resolves against the named directory")

    out = asyncio.run(fire("Write", {"file_path": str(tmp_path / "outside.md")}))
    denied = (out.get("hookSpecificOutput") or {})
    assert denied.get("permissionDecision") == "deny", out
    assert str(root.resolve()) in denied.get("permissionDecisionReason", ""), denied

    assert asyncio.run(fire("Edit", {"file_path": str(tmp_path / "outside.md"),
                                    "new_string": "x"}))["hookSpecificOutput"][
                                        "permissionDecision"] == "deny", (
        "`Edit` names its target the same way and must be bounded the same way")
    assert asyncio.run(fire("Read", {"file_path": str(tmp_path / "outside.md")})) == {}, (
        "the scope is a write bound; refusing a read would be a different policy")
    # The name denials are untouched: the allowance is a path rule, and lifting a
    # name to open one directory is the failure the item names.
    assert "vault_write" in SDK.STATEFUL_TOOLS and "memory_add" in SDK.STATEFUL_TOOLS


# ── #1971: the scope is armed on the trial `run_trial` actually builds ────────
#
# The two nodes above call the installer directly, and every other node that
# reaches `sdk_runner` replaces `run_trial` whole, so the one hop between them —
# `run_trial` forwarding `writes_into` to `build_options` — was covered by nothing
# and was missing: the hook was armed on no live trial. These drive the REAL
# `run_trial` and the REAL `build_options`; only the harness loop (`_consume`) and
# the aggregator's sandbox probe are stubbed, and the verdict is asked of the
# registry on the `RunOptions` the loop was handed.

async def _no_sandbox_probe() -> None:
    return None


def _drive_real_run_trial(monkeypatch, tmp_path, **run_kwargs):
    """(trace, the RunOptions the loop received) from one real `run_trial`."""
    import asyncio

    from scripts.autoresearch import bench_runner_sdk as SDK

    built: list = []

    async def _keep_the_options(messages, options, trace):
        built.append(options)
        trace["final_text"] = "done"

    monkeypatch.setattr(SDK, "require_tool_sandbox", _no_sandbox_probe)
    monkeypatch.setattr(SDK, "_consume", _keep_the_options)
    overlay = tmp_path / "overlay"
    overlay.mkdir(exist_ok=True)
    task = {"id": "behavioural:scope-probe", "category": "behavioural",
            "prompt": "write the note"}
    trace = asyncio.run(SDK.run_trial(task, "baseline", overlay, "test-model",
                                      max_agent_turns=4, **run_kwargs))
    assert trace["status"] == "success", trace
    assert len(built) == 1, "the loop stub was not reached exactly once"
    return trace, built[0]


def _verdict(options, tool_name: str, tool_input: dict) -> dict:
    import asyncio

    out = asyncio.run(options.hooks.fire_pre_tool_use(
        session_id="bench_scope_probe", tool_name=tool_name, tool_input=tool_input))
    return out.get("hookSpecificOutput") or {}


def test_run_trial_arms_the_write_scope_it_was_given(monkeypatch, tmp_path):
    """Clauses 1, 2, 3 and 5. Goes red if `writes_into` is dropped from the
    `build_options` call inside `run_trial` again: nothing here stubs
    `build_options`, so a missing kwarg is a missing hook and the outside write
    below is allowed."""
    from scripts.autoresearch import bench_runner_sdk as SDK

    root = tmp_path / "writes"
    root.mkdir()
    _trace, options = _drive_real_run_trial(monkeypatch, tmp_path, writes_into=root)

    outside = _verdict(options, "Write", {"file_path": str(tmp_path / "outside.md"),
                                          "content": "x"})
    assert outside.get("permissionDecision") == "deny", (
        f"a Write outside the named root was not refused by the built registry: {outside}")
    assert str(root.resolve()) in outside.get("permissionDecisionReason", ""), outside
    assert _verdict(options, "Write", {"file_path": str(root / "inside.md"),
                                       "content": "x"}) == {}, (
        "the bound is deny-outside, not a blanket write ban")

    # An extra deny, never a permission: every state-changing name is still off
    # the trial's surface with the scope armed.
    missing = sorted(set(SDK.STATEFUL_TOOLS) - set(options.disallowed_tools))
    assert not missing, f"arming the scope dropped name denials: {missing}"


def test_run_trial_without_a_root_arms_no_write_scope(monkeypatch, tmp_path):
    """The control for the node above: the same outside write is NOT refused by
    a trial that named no root, so the deny seen there is the scope hook's and
    not some other hook's opinion of the path."""
    _trace, options = _drive_real_run_trial(monkeypatch, tmp_path)
    assert _verdict(options, "Write", {"file_path": str(tmp_path / "outside.md"),
                                       "content": "x"}) == {}


def test_the_capture_record_names_the_root_its_trials_denied_against(monkeypatch, tmp_path):
    """Clause 4: `capture.yaml`'s `writes_into` is read back and tried against
    the registry of a trial that capture ran. A record naming a bound is only a
    claim until a hook armed on that trial refuses a write outside it."""
    from scripts.autoresearch import bench_runner_sdk as SDK

    cfg = _cfg(tmp_path)
    root = tmp_path / "capture-writes"
    root.mkdir()
    built: list = []

    async def _keep_the_options(messages, options, trace):
        built.append(options)
        trace["final_text"] = "done"

    monkeypatch.setattr(SDK, "require_tool_sandbox", _no_sandbox_probe)
    monkeypatch.setattr(SDK, "_consume", _keep_the_options)
    (tmp_path / "overlay").mkdir()
    monkeypatch.setattr("scripts.autoresearch.variant_sandbox.materialize_baseline",
                        lambda cfg: ("BASELINE_TEST", tmp_path / "overlay"))

    CAP.capture_round(cfg=cfg, run_id=RUN_ID,
                      runner=CAP.sdk_runner(cfg, writes_into=root), writes_into=root)
    record = yaml.safe_load(
        (CAP.capture_dir(cfg, RUN_ID) / B.CAPTURE_META_FILENAME).read_text())
    recorded = record["writes_into"]
    assert recorded == str(root)
    assert built, "no trial reached the loop, so no registry was built to ask"
    for options in built:
        denied = _verdict(options, "Write",
                          {"file_path": str(Path(recorded).parent / "outside.md"),
                           "content": "x"})
        assert denied.get("permissionDecision") == "deny", denied
        assert str(Path(recorded).resolve()) in denied.get("permissionDecisionReason", "")
        assert _verdict(options, "Write", {"file_path": str(Path(recorded) / "in.md"),
                                           "content": "x"}) == {}


# ── #1843 clause 4: the manifest declares scope, and the set identity holds ──

def test_every_shipped_scenario_declares_whether_a_capture_can_reach_it():
    """Clause 4's first half, read off the shipped file rather than a fixture.

    Five scenarios, five declarations, each with a reason. A scenario that omits
    the field defaults to `trial`, which would let a new scenario slip in with no
    claim about whether any capture can measure it — and the whole value of the
    field is that the scorecard can say which failures are the round's and which
    are the instrument's.
    """
    assert len(SCENARIOS) == 5, len(SCENARIOS)
    for scenario in SCENARIOS:
        scope, reason = B.capture_scope(scenario)
        assert scenario.get("capture") == scope, scenario["id"]
        assert scope in B.CAPTURE_SCOPES, scenario["id"]
        assert reason.strip(), f"{scenario['id']}: `capture` with no reason"
    assert NO_CAPTURE_PATH_IDS == ["blocked-route-replan"], NO_CAPTURE_PATH_IDS


def test_a_capture_that_delivered_every_reachable_scenario_reports_only_the_none_set(
        tmp_path):
    """Clause 4's identity: instrument failures ARE the manifest's `capture: none` set.

    The fake runner hands back a healthy trace for all five scenarios, so every
    `capture: trial` scenario produced its expected observation. The scorecard must
    then fail exactly the declared-unmeasurable scenario — and nothing else: not
    the scenarios that ran, and not a `capture: none` scenario whose canned trace
    would in fact grade.
    """
    cfg = _cfg(tmp_path)
    CAP.capture_round(cfg=cfg, run_id=RUN_ID, runner=_runner())
    scorecard = B.round_scorecard(cfg, RUN_ID)
    rows = {str(r["id"]): r for r in scorecard["scenarios"]}

    assert scorecard["instrument_failures"] == NO_CAPTURE_PATH_IDS
    assert scorecard["denominator"] == len(IDS) - len(NO_CAPTURE_PATH_IDS)
    for sid in set(IDS) - set(NO_CAPTURE_PATH_IDS):
        # The runner used here grades every reachable scenario without necessarily
        # scoring 1.0 (`_healthy_trace` gives a stale-value answer on one of them),
        # and clause 4 is about which rows are *measured*, not what they measured.
        assert rows[sid]["instrument_failure"] is False, rows[sid]
        assert rows[sid]["value"] is not None, rows[sid]
    unmeasurable = rows[NO_CAPTURE_PATH_IDS[0]]
    assert unmeasurable["instrument_failure"] is True
    assert unmeasurable["observed"]["capture_scope"] == "none", unmeasurable["observed"]
    assert (CAP.capture_dir(cfg, RUN_ID) / f"{NO_CAPTURE_PATH_IDS[0]}.yaml").is_file(), (
        "the trace file is what makes this node bite: a grader that could score it "
        "was refused for the declared reason, not for a missing file")


@pytest.mark.parametrize("bad_capture, bad_reason, fragment", [
    ("sometimes", "a scope that is not a scope", "expected one of"),
    (None, "a reason naming a declaration that is not there", "with no `capture` scope"),
    ("none", "", "with no `capture_reason`"),
])
def test_a_capture_declaration_that_could_not_be_honoured_is_refused_at_load(
        bad_capture, bad_reason, fragment):
    """A free-text scope or a reason-less `none` would silently mean `trial`.

    The default is the whole hazard: an unknown value that fell back to `trial`
    would report an unmeasurable scenario as a fresh instrument failure every
    round, which reads exactly like a behavioural decline. Refusing at load is what
    keeps the set identity in clause 4 a property of the file.
    """
    mutated = yaml.safe_load(B.SCENARIOS_MANIFEST_PATH.read_text(encoding="utf-8"))
    target = mutated["scenarios"][0]
    target.pop("capture", None)
    target.pop("capture_reason", None)
    if bad_capture is not None:
        target["capture"] = bad_capture
    if bad_reason is not None:
        target["capture_reason"] = bad_reason
    # Re-anchor the digest so the refusal this node is about is the scope rule and
    # not the hash guard `validate_manifest` runs first.
    mutated["scenarios_hash"] = B.scenarios_hash(mutated)

    with pytest.raises(B.ScenarioManifestError) as excinfo:
        B.validate_manifest(mutated)
    msg = str(excinfo.value)
    assert target["id"] in msg, f"the refusal has to name the scenario: {msg}"
    assert fragment in msg, msg


# ── #1843 clause 5: the none reason names the vocabulary; no events invented ─

def test_the_unmeasurable_scenario_names_the_event_vocabulary_it_needs(tmp_path):
    """Clause 5's first half: the cause is specific enough to act on.

    "no trace captured" would send a reader to the budget; the cause is structural
    — the checker reads a `route_blocked`/`plan_revised` pair the trial harness
    never emits — so the reason has to say both that there is no capture path and
    which vocabulary is missing.
    """
    cfg = _cfg(tmp_path)
    CAP.capture_round(cfg=cfg, run_id=RUN_ID, runner=_runner())
    scorecard = B.round_scorecard(cfg, RUN_ID)
    reason = next(r for r in scorecard["scenarios"]
                  if r["id"] == "blocked-route-replan")["observed"]["reason"]

    assert "no capture path" in reason, reason
    assert "event vocabulary" in reason, reason
    assert "route_blocked" in reason and "plan_revised" in reason, (
        f"the reason must name the missing pair, not just the class: {reason}")


def test_a_trial_trace_still_invents_neither_replan_event():
    """Clause 5's second half: the refusal to invent survives the new flag.

    A denied call, an unanswered call and a failed status each produce an event,
    because each is something the harness observed. `route_blocked` and
    `plan_revised` are not: nothing in a trial record says the run's *route* was
    blocked or that it replanned, and a checker that reads a pair the runner never
    observed scores the mapper's imagination.
    """
    trial = _trial(status="error", error="killed",
                  final_text="I could not reach the route, so I stopped",
                  tool_calls=[{"name": "Read", "is_error": False,
                               "args": {"path": "~/x.md"}}],
                  denied_calls=[{"name": "Bash", "denied": True,
                                 "deny_kind": "hook_deny",
                                 "deny_reason": "Tool call denied: destructive"}],
                  unresolved_calls=[{"name": "Grep", "reason": "no tool_result"}])
    trace = CAP.trace_from_trial(trial, _scenario("blocked-route-replan"),
                                writes_into=Path("/tmp/nowhere"))
    kinds = {event["kind"] for event in trace["events"]}

    assert "route_blocked" not in kinds and "plan_revised" not in kinds, kinds
    assert {"tool_denied", "tool_unresolved", "run_error"} <= kinds, (
        f"the observed events still have to be there, or the node proves nothing: {kinds}")
