"""#2186: the repeated-sampling coverage leg — per-task pass@1 against pass@N.

WHAT THIS FILE PINS, AND WHY IT IS ITS OWN FILE
-----------------------------------------------
The bench is single-draw: `bench_runner.py:304` and `bench_runner_sdk.py:843`
each fan out one trial per (variant, task), so a task that fails is reported as a
failure with no record of whether eight draws of the same unchanged turn ever pass
it. The coverage leg draws the same (task, settings) N times and reads pass@1
against pass@N, which is the only way to tell a reliability gap from a capability
limit. The arithmetic, the verdict vocabulary, the "no record means no verdict"
rule and the provenance every record has to carry are all new surfaces with no
existing test file to live in — `test_autoresearch_judge.py` grades one trace, and
`test_autoresearch_round_report.py` drives a whole round; neither of them can see
a repeated arm.

The clause this file owns, from the item:

1. One record per objective-scored task carrying pass@1, pass@N with N stated as a
   number, a verdict drawn only from {reliable, reachable, unreachable,
   indeterminate}, and the tokens the repeated arm spent.
2. A task with no coverage record yields no verdict and is reported as
   not-evaluated rather than as a pass — the rule
   `workers/sources/automod_regression.py:35` states for the noise file.
4. Each arm's record names the sampling parameters actually sent plus the model id,
   quantization, corpus and N it is valid for, so a direct-route figure
   (temperature 0.3) and a runtime-harness figure (which sends none) cannot be read
   as one measurement.

Clauses 3 (report ordering) and 5 (report-only) live with the surfaces they grade:
`tests/test_autoresearch_round_report.py` and `tests/test_autoresearch_promotion.py`
respectively. A process-boundary test in the file that owns the boundary is what
caught #1370, not a duplicate assertion here.

The leg's own numbers, kept in the open because a test that re-derives the
formula from the module would pin nothing: pass@k is the unbiased estimator
1 - C(n-c,k)/C(n,k), `reliable` needs pass@1 >= 0.75, `reachable` needs pass@N
>= 1.0, and at k == N the estimator is exactly 1.0 when any draw passed.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from scripts.autoresearch import bench_runner, coverage_leg as cov
from scripts.autoresearch.common import load_bench_tasks

# Reused rather than re-derived: the config builder every autoresearch round test
# drives, so the artifact paths under test are tmp paths by construction and
# nothing here can reach the live `~/lloyd-data/_pipeline/research/coverage/`.
from tests.test_autoresearch_promotion import make_cfg

TASKS = [{"id": "bench_alpha_always", "prompt": "a"},
         {"id": "bench_beta_flaky", "prompt": "b"},
         {"id": "bench_gamma_never", "prompt": "c"}]

#: What the fake engine answers, per task: the three shapes a corpus has.
ALWAYS = "passed"
NEVER = "failed"


def _context(tasks=None, *, n=4, route=cov.ROUTE_DIRECT,
             sampling=None, served_model="lloyd-nova2-9b", quantization=None):
    return cov.measurement_context(
        model_alias="primary", served_model=served_model, quantization=quantization,
        corpus_tasks=tasks if tasks is not None else TASKS, n=n, route=route,
        sampling=sampling if sampling is not None
        else bench_runner.sampling_params(bench_runner.DEFAULT_MAX_TOKENS))


def _fake_run_trial(task: dict, draw_index: int) -> dict:
    """A deterministic engine: one shape per task, and a usage block per draw.

    `bench_beta_flaky` passes on draw 1 only — the draw index is what makes a
    flaky task flaky here, so the leg cannot be passing by drawing the same
    response eight times, which is what a zero-temperature arm would do.
    """
    # `FLAKY` is the shape, not the reply: the flaky task answers with the passing
    # text on draw 1 and the failing text on the other three, so a pass at N=4 can
    # only come from drawing again, which is the whole claim the leg makes.
    text = {"bench_alpha_always": ALWAYS,
            "bench_beta_flaky": ALWAYS if draw_index == 1 else NEVER,
            "bench_gamma_never": NEVER}[task["id"]]
    # The direct-route usage block, under the names `bench_runner.TOKEN_KEYS`
    # records it by — a trace keyed any other way reads as an unpriced draw.
    return {"status": "success", "final_text": text,
            "prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}


def _fake_score(task: dict, trace: dict) -> dict:
    """The grader's contract, narrowed: `objective_score` 1.0 iff the reply passed."""
    return {"objective_score": 1.0 if trace.get("final_text") == ALWAYS else 0.0,
            "composite_score": 0.5}


def _run(tmp_path, tasks=TASKS, *, n=4, **context_kw) -> dict:
    """Draw the arm over the fake engine and put its artifact on disk."""
    cfg = make_cfg(tmp_path)
    coverage = asyncio.run(cov.run_coverage(
        tasks, n=n, run_trial=_fake_run_trial, score=_fake_score,
        context=_context(tasks, n=n, **context_kw), max_parallel=3, round_id="R_TEST"))
    cov.write_coverage(cfg, coverage)
    return coverage


# ── clause 1: the record the leg produces ───────────────────────────────────


def test_the_leg_draws_n_times_and_writes_one_record_per_task(tmp_path):
    """N draws of each task, and one record each carrying the four figures.

    Three tasks, three shapes: always passes, passes once in four draws, never
    passes. `pass_at_1` is the empirical single-draw rate and `pass_at_n` is
    `pass_at_k(n, c, n)`, so `bench_beta_flaky` is the disagreement the item's
    acceptance names — a pass@1 that today reads as a failure (0.250) beside a
    pass@4 that says the task ever passes (1.000) — which is a reliability gap,
    not a capability limit.
    """
    coverage = _run(tmp_path, n=4)
    by_task = {r["task_id"]: r for r in coverage["records"]}
    assert set(by_task) == {"bench_alpha_always", "bench_beta_flaky", "bench_gamma_never"}

    for record in coverage["records"]:
        assert record["n"] == 4, "N is stated as a number on the record itself"
        assert record["verdict"] in cov.VERDICTS
        assert record["pass_at_1"] is not None and record["pass_at_n"] is not None
        assert record["tokens"]["total_tokens"] > 0

    flaky = by_task["bench_beta_flaky"]
    assert flaky["draws_run"] == 4 and flaky["draws_scored"] == 4
    assert flaky["passes"] == 1
    assert flaky["pass_at_1"] == pytest.approx(0.25)
    assert flaky["pass_at_n"] == pytest.approx(1.0)
    assert flaky["verdict"] == "reachable"
    assert by_task["bench_alpha_always"]["verdict"] == "reliable"
    assert by_task["bench_gamma_never"]["verdict"] == "unreachable"
    assert by_task["bench_gamma_never"]["pass_at_n"] == pytest.approx(0.0)


def test_the_tokens_are_the_repeated_arm_spend_not_someones_estimate(tmp_path):
    """Four draws at 120 tokens each is 480 tokens, per task, summed over tasks.

    The figure is summed from the usage blocks the draws returned. A task whose
    draws reported no usage yields `None` rather than 0: zero would say the arm
    answered for free, which is a claim about a thing that was not measured.
    """
    coverage = _run(tmp_path, n=4)
    by_task = {r["task_id"]: r for r in coverage["records"]}
    assert by_task["bench_alpha_always"]["tokens"] == {
        "prompt_tokens": 400, "completion_tokens": 80, "total_tokens": 480,
        "draws_with_usage": 4}
    assert cov.arm_totals(coverage)["total_tokens"] == 3 * 480
    assert cov.arm_totals(coverage)["draws_run"] == 12


def test_a_draw_that_reports_no_usage_leaves_the_token_figure_unreported(tmp_path):
    def no_usage(task, draw_index):
        return {"status": "success", "final_text": ALWAYS}

    coverage = asyncio.run(cov.run_coverage(
        TASKS[:1], n=2, run_trial=no_usage, score=_fake_score,
        context=_context(TASKS[:1], n=2)))
    assert coverage["records"][0]["tokens"]["total_tokens"] is None
    assert coverage["records"][0]["verdict"] == "reliable", \
        "an unpriced draw is still a draw: the verdict does not depend on usage"


def test_pass_at_k_is_the_unbiased_estimator_at_the_values_that_show_it():
    """The arithmetic the verdicts are built on, pinned at hand-checkable points.

    At k == n the estimator collapses to "did any draw pass": 1.0 with one pass in
    eight, 0.0 with none. At k == 1 it is the empirical rate. And k > n is None —
    there is no sample of eight out of three, and inventing one is how a short arm
    reports a capability limit it did not measure.
    """
    assert cov.pass_at_k(8, 1, 8) == 1.0
    assert cov.pass_at_k(8, 0, 8) == 0.0
    assert cov.pass_at_k(8, 4, 1) == pytest.approx(0.5)
    assert cov.pass_at_k(8, 2, 2) == pytest.approx(1.0 - (6 * 5) / (8 * 7))
    assert cov.pass_at_k(3, 0, 8) is None
    assert cov.pass_at_k(0, 0, 1) is None


def test_the_verdict_is_drawn_only_from_the_four_named_classes():
    """Every (draws, passes) pair maps into the vocabulary, and the four cases land
    where the item names them.

    `indeterminate` is short-sample only, plus the honest middle: it is never a
    polite "unsure" that lets an unmeasured task out of being reported.
    """
    for draws in range(0, 9):
        for passes in range(0, draws + 1):
            verdict, reason = cov.verdict_for(draws, 8, passes)
            assert verdict in cov.VERDICTS, (draws, passes, verdict)
            assert reason

    assert cov.verdict_for(8, 8, 8)[0] == "reliable"
    assert cov.verdict_for(8, 8, 6)[0] == "reliable", "0.75 of 8 is the boundary"
    assert cov.verdict_for(8, 8, 5)[0] == "reachable", "pass@1 0.625, ever passes"
    assert cov.verdict_for(8, 8, 0)[0] == "unreachable"
    assert cov.verdict_for(4, 8, 0)[0] == "indeterminate", \
        "four scored draws of eight asked for is not a capability statement"


def test_a_rubric_only_task_is_recorded_without_a_verdict_and_out_of_every_count(tmp_path):
    """Graded tasks lie at pass@k, so the leg refuses to read them.

    A draw whose `objective_score` is None was never mechanically checked (#416's
    not-rankable layer). Such a task is `mechanically_checkable: false` with no
    verdict: `pass@N` on it would be fluency coverage wearing an outcome label.
    """
    def rubric_only(task, draw_index):
        return {"status": "success", "final_text": ALWAYS, "total_tokens": 50}

    coverage = asyncio.run(cov.run_coverage(
        TASKS[:1], n=3, run_trial=rubric_only,
        score=lambda task, trace: {"objective_score": None, "composite_score": 0.8},
        context=_context(TASKS[:1], n=3)))
    record = coverage["records"][0]
    assert record["mechanically_checkable"] is False
    assert record["verdict"] == cov.NOT_EVALUATED
    assert record["pass_at_1"] is None and record["pass_at_n"] is None
    assert cov.verdict_of(coverage, record["task_id"])[0] == cov.NOT_EVALUATED


def test_a_partially_solved_draw_is_counted_as_a_partial_not_a_pass(tmp_path):
    """`objective_score` is a fraction; only 1.0 is a pass.

    The ledger's middle band is real — `bench_006` sits at 0.36 partial draws — and
    folding a 0.5 into either end would move the very number this leg exists to
    read. Partial draws get their own field so a reader can see them.
    """
    def half_solved(task, draw_index):
        return {"status": "success", "final_text": "partial", "total_tokens": 40}

    coverage = asyncio.run(cov.run_coverage(
        TASKS[:1], n=4, run_trial=half_solved,
        score=lambda task, trace: {"objective_score": 0.5, "composite_score": 0.5},
        context=_context(TASKS[:1], n=4)))
    record = coverage["records"][0]
    assert record["partials"] == 4 and record["passes"] == 0
    assert record["pass_at_n"] == pytest.approx(0.0)
    assert record["verdict"] == "unreachable", \
        "half-solved in every draw is never solved: a partial is not coverage"


def test_a_draw_that_died_is_never_a_draw_that_failed(tmp_path):
    """An engine that was down for every draw measured nothing, which is not a
    capability limit.

    Every one of eight draws errors: the record reads `not-evaluated` with the
    errored count in its reason, because `unreachable` here would tell a reader the
    model cannot do the task when the true finding is that nothing reached the
    model. A task the engine answered but never passed is the test above.
    """
    def dead_engine(task, draw_index):
        raise RuntimeError("engine reset")

    coverage = asyncio.run(cov.run_coverage(
        TASKS[:1], n=8, run_trial=dead_engine,
        score=_fake_score, context=_context(TASKS[:1], n=8)))
    record = coverage["records"][0]
    assert record["draws_run"] == 8 and record["draws_scored"] == 0
    assert record["draws_errored"] == 8
    assert record["verdict"] == cov.NOT_EVALUATED
    assert "0 of 8 draws were objectively scored (8 errored)" in record["verdict_reason"]
    # An outage is not a finding about the corpus: the leg is not cut by a table
    # in which nothing was measured.
    census = cov.summarize(coverage, ["bench_alpha_always"])
    assert census["leg_cut"] is False and census["not_evaluated_failures"] == 1


def test_fewer_scored_draws_than_the_arm_asked_for_is_indeterminate(tmp_path):
    """Four draws answered, four lost, and the four that came back never passed.

    Not `unreachable` — eight draws were asked for and only four were measured, and
    the paper's own curves are still climbing at four. And not `not-evaluated`:
    something was measured, and saying so is what `indeterminate` is for.
    """
    def half_dead(task, draw_index):
        if draw_index < 4:
            return {"status": "success", "final_text": NEVER,
                "prompt_tokens": 30, "total_tokens": 30}
        raise RuntimeError("engine reset")

    coverage = asyncio.run(cov.run_coverage(
        TASKS[:1], n=8, run_trial=half_dead,
        score=_fake_score, context=_context(TASKS[:1], n=8)))
    record = coverage["records"][0]
    assert record["draws_run"] == 8 and record["draws_scored"] == 4
    assert record["passes"] == 0
    assert record["verdict"] == "indeterminate"
    assert "only 4 of 8 draws scored" in record["verdict_reason"]


# ── clause 2: no record means no verdict, never a pass ──────────────────────


def test_a_task_with_no_record_yields_no_verdict_and_is_never_counted_a_pass(tmp_path):
    """The `automod_regression.py:35` rule applied to coverage: absence is
    "cannot evaluate", and it is reported as that."""
    coverage = _run(tmp_path, n=4)
    verdict, reason = cov.verdict_of(coverage, "bench_never_measured")
    assert verdict == cov.NOT_EVALUATED
    assert verdict not in cov.VERDICTS, "not-evaluated is the absence of a verdict"
    assert "no coverage record" in reason

    census = cov.summarize(coverage, ["bench_never_measured", "bench_beta_flaky"])
    assert census["not_evaluated_failures"] == 1
    assert census["graded_failures"] == 2
    assert census["disagreement_reachable"] == 1, "only the measured failure counts"
    assert census["disagreement_unreachable"] == 0
    assert sum(census["verdict_counts"].values()) == 1, \
        "the unmeasured task is in no class, so the class counts cannot absorb it"


def test_a_missing_or_broken_artifact_loads_as_absent_and_not_as_an_empty_leg(tmp_path):
    """A file that is not there, is torn, or is somebody else's schema is ABSENT.

    Three different ways to have no measurement, and all three must land on
    `not-evaluated`. A partial JSON file that loaded as `{}` with no records would
    render a table of nothing and read as a leg that found no coverage — the
    opposite finding, from the same keystrokes.
    """
    cfg = make_cfg(tmp_path)
    path = cov.artifact_path(cfg, model_alias="primary", route=cov.ROUTE_DIRECT, n=4)
    assert cov.load_coverage(cfg, model_alias="primary", route=cov.ROUTE_DIRECT, n=4) is None

    path.parent.mkdir(parents=True, exist_ok=True)
    for broken in ('{"schema": "lloyd-bench-coverage/v1", "records": [{"tas',
                   '{"schema": "lloyd-bench-coverage/v0", "records": []}',
                   '{"schema": "lloyd-bench-coverage/v1", "records": []}',
                   '{"schema": "lloyd-bench-coverage/v1", "model_alias": "primary", '
                   '"model_id": "x", "quantization": "unrecorded", "route": "nonsense", '
                   '"n_draws": 4, "corpus_digest": "sha256:x", "records": []}'):
        path.write_text(broken, encoding="utf-8")
        assert cov.load_coverage_path(path) is None, broken[:40]


def test_the_round_report_says_the_leg_was_not_run_rather_than_that_it_found_nothing(tmp_path):
    """No artifact, two graded failures: the counts print as not-evaluated.

    The wrong report would say `reachable: 0 · unreachable: 0`, which a reader
    takes as "the leg ran and every failure is a capability limit". That sentence
    is the difference between closing a bench task as a model limit and running
    the arm, and it is not what an absent arm measured.
    """
    cfg = make_cfg(tmp_path)
    lines = cov.report_lines(
        cfg, baseline_summary={"per_task": [
            {"task_id": "bench_gamma_never", "objective_score": 0.0},
            {"task_id": "bench_beta_flaky", "objective_score": 0.5}]},
        model_alias="primary", n=4)
    report = "\n".join(lines)
    assert "## Bench coverage (#2186)" in report
    assert "leg not run" in report
    assert "not-evaluated" in report
    assert "reachable: 0" not in report and "unreachable: 0" not in report
    assert "measures nothing" in report


# ── clause 4: what the record is valid for ─────────────────────────────────


def test_every_record_carries_the_settings_model_quantization_corpus_and_n(tmp_path):
    """A record has to be readable without its header, and the header has to be
    comparable to nothing that was not the same measurement."""
    coverage = _run(tmp_path, n=4, served_model="lloyd-nova2-9b-awq",
                    quantization="awq")
    header = coverage
    assert header["model_id"] == "lloyd-nova2-9b-awq"
    assert header["quantization"] == "awq"
    assert header["n_draws"] == 4
    assert header["corpus_digest"] == cov.corpus_digest(TASKS)
    assert header["sampling"] == {"temperature": 0.3, "max_tokens": 1500,
                                 "chat_template_kwargs": {"enable_thinking": False}}

    record = coverage["records"][0]
    for key in ("n", "route", "model_id", "quantization", "corpus_digest", "sampling"):
        assert key in record, key
    assert record["sampling"]["temperature"] == 0.3


def test_the_quantization_recorded_is_one_the_served_id_names_or_unrecorded():
    """Never a default of `none`: the box may well be running a quantised
    checkpoint and simply not have said so in its served id."""
    assert cov.quantization_from_served_id("nova2-9b-awq") == "awq"
    assert cov.quantization_from_served_id("lloyd-nova2-9b-kv-fp8") == "fp8"
    assert cov.quantization_from_served_id("lloyd-nova2-9b") == cov.QUANTIZATION_UNRECORDED
    assert cov.quantization_from_served_id(None) == cov.QUANTIZATION_UNRECORDED


def test_the_sampling_block_a_record_names_is_the_block_the_request_sends(monkeypatch):
    """The seam, across the boundary that matters: record → payload → socket.

    `bench_runner.chat_completion` posts what `sampling_params` returns and the leg
    records what `sampling_params` returns, so they agree by construction rather
    than by a transcription kept in sync by hand. Captured off the wire here: if
    someone moves the literals back into the payload builder, or records a
    temperature the request never carried, this is what fails.
    """
    sent: dict = {}

    class Resp:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"choices": [{"message": {"content": "ok"}}],
                    "usage": {"total_tokens": 1}}

    def fake_post(url, headers=None, json=None, timeout=None):
        sent.update(json)
        return Resp()

    monkeypatch.setattr(bench_runner.requests, "post", fake_post)
    monkeypatch.setattr(bench_runner, "_resolved_model_name", lambda m: "served-name")
    content, usage = bench_runner.chat_completion("primary", [{"role": "user", "content": "x"}],
                                                 max_tokens=999)

    block = bench_runner.sampling_params(999)
    assert block == {"temperature": 0.3, "max_tokens": 999,
                     "chat_template_kwargs": {"enable_thinking": False}}
    for key, value in block.items():
        assert sent[key] == value, f"{key} on the wire is not what the leg would record"


def test_a_direct_figure_and_a_runtime_harness_figure_are_not_one_measurement():
    """The item's clause 4 in one call: temperature 0.3 versus nothing sent.

    The runtime harness (`bench_runner_sdk`) sends no sampling parameters — the
    engine's server-side defaults govern — so its pass@N is a different quantity
    even over the identical task ids. `comparable` is what stops a report averaging
    the two, or letting one answer for the other.
    """
    tasks = TASKS
    direct = _context(tasks, n=8)
    harness = _context(tasks, n=8, route=cov.ROUTE_RUNTIME_HARNESS, sampling={})
    assert cov.comparable(direct, dict(direct)) is True
    assert cov.comparable(direct, harness) is False
    assert cov.comparable(direct, _context(tasks, n=4)) is False, \
        "a pass@8 and a pass@4 are not the same instrument"
    assert cov.comparable(direct, _context(tasks, n=8, served_model="other-ckpt")) is False
    assert cov.comparable(direct, None) is False
    assert harness["sampling"] == {} and harness["sampling_note"], \
        "the arm that sent nothing says so on its own header"


def test_the_leg_runs_the_corpus_and_trial_path_it_borrows_from(tmp_path):
    """No new corpus, no new trial path, no new grader.

    Two identities, both behavioural. `direct_trial_runner` must go through
    `bench_runner._run_one_sync` — the same call a bench trial goes through — so the
    leg measures the system rather than a second implementation of it. And the
    corpus it draws must be the corpus the round draws: `corpus_tasks` is asserted
    to return `load_bench_tasks`'s own list, task for task, against a real bench
    directory written here, not against the module's source text.
    """
    cfg = make_cfg(tmp_path)
    cfg.paths.bench_dir.mkdir(parents=True, exist_ok=True)
    for tid in ("bench_zulu", "bench_alpha"):
        (cfg.paths.bench_dir / f"{tid}.md").write_text(
            f"---\nid: {tid}\ncategory: replay\nprompt: do the thing\n---\n\nBody.\n",
            encoding="utf-8")
    assert [t["id"] for t in cov.corpus_tasks(cfg)] == \
        [t["id"] for t in load_bench_tasks(cfg.paths.bench_dir)], \
        "the leg's corpus is the bench's own, in the bench's own order"

    called: list = []

    def fake_run_one_sync(task, variant_id, overlay_dir, model, timeout_seconds):
        called.append((task["id"], variant_id))
        return {"status": "success", "final_text": ALWAYS, "total_tokens": 10}

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(bench_runner, "_run_one_sync", fake_run_one_sync)
    try:
        runner = cov.direct_trial_runner(model="primary")
        runner(TASKS[0], 3)
    finally:
        monkeypatch.undo()

    assert called == [("bench_alpha_always", "COVERAGE_draw3")], \
        "a draw is one bench trial of the same task, not a re-implemented request"


def test_the_artifact_round_trips_and_lives_under_the_research_root(tmp_path):
    """Written atomically to its own file, and readable back by (model, route, N).

    `sort_keys=True` round-trip matters for the automod witness: a record set that
    reorders between two runs would look like a changed measurement.
    """
    cfg = make_cfg(tmp_path)
    coverage = _run(tmp_path, n=4)
    path = cov.artifact_path(cfg, model_alias="primary", route=cov.ROUTE_DIRECT, n=4)
    assert path == cfg.paths.research_root / "coverage" / "primary-direct-n4.json"
    assert not list(path.parent.glob("*.tmp")), "the temp file was replaced, not left behind"

    loaded = cov.load_coverage(cfg, model_alias="primary", route=cov.ROUTE_DIRECT, n=4)
    assert loaded == json.loads(path.read_text(encoding="utf-8"))
    assert loaded == coverage
    assert cov.load_coverage(cfg, model_alias="primary", route=cov.ROUTE_DIRECT,
                             n=8) is None, "an N=8 report does not read an N=4 arm"


def test_the_route_a_round_gets_comes_from_its_own_harness():
    """An sdk round reads the runtime-harness arm; `auto` reads the direct arm and
    lets its runtime-half tasks show as not-evaluated rather than borrowing a
    direct figure across the boundary clause 4 exists to hold."""
    assert cov.route_for_harness("sdk") == cov.ROUTE_RUNTIME_HARNESS
    assert cov.route_for_harness("direct") == cov.ROUTE_DIRECT
    assert cov.route_for_harness("auto") == cov.ROUTE_DIRECT
    assert cov.route_for_harness(None) == cov.ROUTE_DIRECT


# ── the seams the record's own provenance depends on ─────────────────────────


def test_the_model_id_and_quantization_are_what_the_engine_answered(monkeypatch):
    """`model_id` is a probe result, so the probe is a seam and the seam is tested.

    Clause 4 says a record must name the checkpoint it is valid for. That string
    comes from `app.model_identity.probe_served_model` over HTTP, and the
    quantization is read out of it — an untested call here would let a record assert
    `awq` about a checkpoint the box is not serving, which is precisely the mistake
    the header exists to prevent. Both edges matter: an engine that answers is
    parsed, and an engine that does not is recorded as unverified rather than
    guessed.
    """
    import app.model_identity as mi

    served = "nova2-9b-cpt-sft-rl-awq"
    seen: list = []

    async def fake_probe(base_url, *, timeout=None):
        seen.append(base_url)
        return served

    monkeypatch.setattr(mi, "probe_served_model", fake_probe)
    monkeypatch.setattr(bench_runner, "_endpoint_for", lambda alias: "http://127.0.0.1:9/v1")
    assert cov.resolve_served_model("primary") == served
    assert seen == ["http://127.0.0.1:9/v1"], "asked the endpoint that serves the alias"
    assert cov.quantization_from_served_id(served) == "awq"

    context = cov.measurement_context(
        model_alias="primary", served_model=cov.resolve_served_model("primary"),
        quantization=None, corpus_tasks=TASKS, n=8, route=cov.ROUTE_DIRECT,
        sampling=bench_runner.sampling_params(1500))
    assert context["model_id"] == served
    assert context["quantization"] == "awq", "derived, not asserted from config prose"

    async def dead_probe(base_url, *, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(mi, "probe_served_model", dead_probe)
    assert cov.resolve_served_model("primary") is None
    unknown = cov.measurement_context(
        model_alias="primary", served_model=None, quantization=None,
        corpus_tasks=TASKS, n=8, route=cov.ROUTE_DIRECT,
        sampling=bench_runner.sampling_params(1500))
    assert unknown["model_id"] == "unresolved"
    assert unknown["quantization"] == cov.QUANTIZATION_UNRECORDED


def test_a_runtime_harness_report_reads_only_the_runtime_arm(tmp_path):
    """Two artifacts on disk, one per arm: a round reads its own and nothing else.

    The two arms differ in the one thing that decides the reading — the direct arm
    decoded at temperature 0.3, the runtime arm at settings the engine chose — so
    reading across the boundary would attribute a coverage figure to a decoding that
    never produced it. The direct artifact is written FIRST with passes on it, so a
    loader that picked up the wrong file could not pass this test by accident.
    """
    cfg = make_cfg(tmp_path)
    direct = _run(tmp_path, n=4)
    assert any(r["pass_at_n"] == 1.0 for r in direct["records"]), \
        "positive control: the direct arm has passes to be wrongly borrowed"

    draws_by_task = {t["id"]: [cov.draw_record(task_id=t["id"], draw_index=i,
                                               objective_score=0.0, status="success",
                                               trace={})
                               for i in range(4)] for t in TASKS}
    runtime = cov.build_coverage(
        draws_by_task, n_requested=4, round_id="R_TEST",
        context=cov.measurement_context(model_alias="primary", served_model="lloyd-nova2-9b",
                                        quantization=None, corpus_tasks=TASKS, n=4,
                                        route=cov.ROUTE_RUNTIME_HARNESS, sampling=None))
    cov.write_coverage(cfg, runtime)

    loaded = cov.load_coverage(cfg, model_alias="primary",
                               route=cov.ROUTE_RUNTIME_HARNESS, n=4)
    assert loaded == runtime and loaded != direct
    assert loaded["sampling"] == {} and loaded["sampling_note"]

    baseline = {"per_task": [{"task_id": t["id"], "objective_score": 0.0} for t in TASKS]}
    report = cov.report_lines(cfg, baseline_summary=baseline, model_alias="primary",
                              route=cov.ROUTE_RUNTIME_HARNESS, n=4)
    text = "\n".join(report)
    assert "none sent by this route" in text, "the arm that sent nothing says so"
    assert "temperature=0.3" not in text, \
        "the direct arm's sampler never appears on the runtime arm's report"
    assert cov.comparable(loaded, direct) is False
