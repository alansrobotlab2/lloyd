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
    assert scorecard["denominator"] == len(IDS) == scorecard["scenarios_total"]
    assert scorecard["instrument_failures"] == []
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
    assert scorecard["denominator"] == len(IDS) - 1
    assert scorecard["instrument_failures"] == ["source-retention"]
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
    assert scorecard["instrument_failures"] == ["act-on-known-fact"]
    action = next(a for a in scorecard["axes"] if a["axis"] == "action_consistency")
    assert action["denominator"] == 1, (
        "the other scenario on this axis still counts; only the unmeasured one "
        "leaves the denominator")


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
