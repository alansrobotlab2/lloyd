"""The #647 arm driver: its summary, and the seams that need a fake runner.

Three halves, deliberately unequal in cost. The pure half needs no engine: the spam
delta pairs an `all` reply with its own padded copy, the judge-independence count
compares the rubric and P x R at one threshold, and the per-cell truncation split is
arithmetic over rows. The seam half does reach outside the module, so it is tested
with a fake `run_bench_sdk` rather than a real trial: `--max-turns` has to arrive as
`run_bench_sdk`'s `max_agent_turns` and be recorded on the row it produced (#1608
clause 5), and neither end of that is visible from the summary alone. The third half
is what #2160 added — a trial the turn cap cut off and a trial the wall clock cut off
both have to reach `trials.jsonl` flagged, and `summarise` has to keep them out of the
means it reports. On 2026-09-24 two empty `all`-arm replies on bench_016 were
turn-cap-cut at the 12-turn default, were averaged in as recall 0.0, and that is what
made the phrasing arm ungradeable.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("run_find_all_arms",
                                               ROOT / "eval" / "run_find_all_arms.py")
arms = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(arms)


def _row(arm, trial, task, objective, rubric, composite, p=None, r=None, **extra):
    row = {"arm": arm, "trial": trial, "task_id": task, "objective": objective,
           "rubric": rubric, "composite": composite, "precision": p, "recall": r,
           "f1": None, "turns": 3, "rubric_status": "ok"}
    row.update(extra)
    return row


def test_spam_is_paired_with_its_own_reply_and_disagreement_is_counted():
    rows = [
        _row("all", 0, "t1", 1.0, 0.9, 0.95, 1.0, 1.0),
        _row("spam", 0, "t1", 0.5, 0.9, 0.70),
        _row("all", 1, "t1", 0.2, 0.8, 0.50, 1.0, 0.2),   # rubric says pass, P x R fail
        _row("spam", 1, "t1", 0.1, 0.8, 0.45),
        _row("one", 0, "t1", 0.1, 0.3, 0.20, 1.0, 0.1),
    ]
    s = arms.summarise(rows)
    assert s["spam"]["n"] == 2
    assert s["spam"]["objective_drop"] == 0.3
    assert s["spam"]["composite_lower"] == 2
    assert s["spam"]["rubric_not_lower"] == 2
    ji = s["judge_independence"]
    assert ji["n"] == 3 and ji["disagree_at_0.5"] == 1
    assert s["cells"]["one/t1"]["recall"] == 0.1


def test_a_cell_counts_both_truncation_causes_and_keeps_them_out_of_its_means():
    """Clause 2 (#2160): the two cut causes are counted beside `n` and excluded
    from the cell's score means.

    This is the shape that made the 2026-09-24 `all` arm unreadable. Its
    `all/bench_016` cell is two trials, both `status='success' turns=13` with an
    empty reply — 13 is one past the 12-turn default, so both were cut off
    mid-enumeration — and the cell reported `recall 0.0`, which is the number a
    find-one arm would have to beat. Its `all/bench_014` cell is one answered
    trial plus one `status='timeout' duration_s=600.0`, which is the wall clock,
    not the turn cap.

    Here the same cell carries one answered trial at recall 1.0, one at 0.5, one
    turn-cap-cut at 0.0 and one clock-cut at 0.0: `n` stays 4 because four trials
    ran, `n_truncated` is 1, `n_timeout` is 1, and every score mean is over the two
    that answered — recall 0.75, not 0.375. `n_scored` is printed beside it because a
    mean whose denominator is 2 out of 4 has to say so. `turns` is the one column the
    cut trials stay in, at (8 + 9 + 26 + 0) / 4 = 10.75: how long a trial ran is a
    fact about the run, and hiding the two most expensive ones would flatter the arm.
    """
    rows = [
        _row("all", 0, "t1", 1.0, 0.9, 0.95, 1.0, 1.0, status="success",
             turns=8, turn_budget=25, turn_truncated=False, f1=1.0),
        _row("all", 1, "t1", 0.5, 0.6, 0.55, 1.0, 0.5, status="success",
             turns=9, turn_budget=25, turn_truncated=False, f1=0.5),
        _row("all", 2, "t1", 0.0, 0.0, 0.0, 0.0, 0.0, status="success",
             turns=26, turn_budget=25, turn_truncated=True, f1=0.0),
        _row("all", 3, "t1", 0.0, 0.0, 0.0, 0.0, 0.0, status="timeout",
             turns=0, turn_budget=25, turn_truncated=False, f1=0.0),
    ]
    cell = arms.summarise(rows)["cells"]["all/t1"]
    assert (cell["n"], cell["n_truncated"], cell["n_timeout"]) == (4, 1, 1), cell
    assert cell["n_scored"] == 2, cell
    assert cell["recall"] == 0.75, cell
    assert cell["precision"] == 1.0, cell
    assert cell["f1"] == 0.75, cell
    assert cell["objective"] == 0.75, cell
    assert cell["turns"] == 10.75, cell


def test_a_cell_whose_trials_were_all_cut_reports_no_mean_rather_than_zero():
    """`n_scored` 0 reads as UNMEASURED, never as recall 0.0.

    The same rule #1510 clause 1 put on the replay probe (#2164): 0 of 0 is not a
    clean result. An `all` arm whose every trial the cap cut off has no recall
    measurement at all, and reporting 0.0 there is what made the phrasing gap look
    closed when nothing had been measured.
    """
    rows = [
        _row("all", 0, "t1", 0.0, 0.0, 0.0, 0.0, 0.0, status="success",
             turns=13, turn_budget=12, turn_truncated=True),
        _row("all", 1, "t1", 0.0, 0.0, 0.0, 0.0, 0.0, status="timeout",
             turns=0, turn_budget=12, turn_truncated=False),
    ]
    cell = arms.summarise(rows)["cells"]["all/t1"]
    assert (cell["n"], cell["n_truncated"], cell["n_timeout"], cell["n_scored"]) \
        == (2, 1, 1, 0), cell
    for metric in ("precision", "recall", "f1", "objective", "rubric", "composite"):
        assert cell[metric] is None, (metric, cell)


def test_pearson_is_none_when_undefined():
    assert arms._pearson([1.0, 1.0, 1.0], [0.1, 0.5, 0.9]) is None
    assert arms._pearson([0.0, 1.0], [0.0, 1.0]) is None
    assert arms._pearson([0.0, 0.5, 1.0], [0.0, 0.5, 1.0]) == 1.0


# ── the per-task turn budget reaches the runner (#1608 clause 5) ─────────────
#
# `run_bench_sdk` is a separate harness path: it owns the agent loop and its turn
# cap, and until now the driver never named that cap, so every arm silently ran
# under `DEFAULT_MAX_AGENT_TURNS` whatever the arm needed. That is a process seam
# the summary tests above cannot see, so the test below crosses it with a fake
# runner and reads both ends: the kwargs `run_bench_sdk` was handed, and the row
# that landed in `trials.jsonl`. The CLI string is the input, not an args object —
# flag name, type and default are part of what is pinned.

_SCORE = {"composite_score": 0.5, "objective_score": 0.5, "rubric_overall": 0.5,
          "rubric_status": "ok", "precision": 1.0, "recall": 1.0, "f1": 1.0,
          "answer_sets": []}


async def _no_sandbox():
    return None


# The four real task ids, named for the shape each one takes in the #2160 nodes: on
# 2026-09-24 bench_016's two `all` trials were turn-cap-cut with an empty reply and
# bench_014 trial 1 was clock-cut at the 600 s `--timeout`, so those two carry the
# cut shapes below and the other two answer normally.
_CUT = "bench_016_audit_skill_dead_paths"
_CLOCK = "bench_014_audit_dead_wikilinks"
_ANSWERED = "bench_017_audit_unresolved_task_skills"
_AT_CAP = "bench_015_audit_cross_entity_fact_copies"


def _install_fake_harness(monkeypatch, tmp_path, traces_by_task=None):
    """Replace the six things `main_async` reaches outside itself; return the calls.

    `traces_by_task` maps a task id to fields merged over the default trace, which is
    how a test makes the fake runner hand back a turn-cap-cut trial, a clock-cut one,
    or an ordinary answer — the shapes `summarise` has to tell apart. A value may also
    be a callable taking the kwargs the fake `run_bench_sdk` was handed, so a trial is
    written as a function of the budget the driver actually passed rather than of a
    number this file picked: that is how a capped row gets one turn past its cap,
    which is the real runner's behaviour, not a guess.
    """
    calls = []
    overrides = traces_by_task or {}

    def load_config():
        return SimpleNamespace(default_model="fake-model",
                               paths=SimpleNamespace(bench_dir=tmp_path / "bench"))

    def load_bench_tasks(_bench_dir):
        # The `one` arm rewrites a sentence out of the real prompt and asserts it is
        # there, so the fakes have to carry those sentences.
        tasks = [{"id": tid, "prompt": f"audit {tid}: report each finding once"}
                 for tid in arms.TASKS]
        for tid, (sentence, _rewrite) in arms.ONE_PHRASING.items():
            next(t for t in tasks if t["id"] == tid)["prompt"] = sentence
        return tasks

    async def run_bench_sdk(cfg, variants, tasks, model, **kwargs):
        calls.append({"model": model, "variants": variants,
                      "task_ids": [t["id"] for t in tasks], **kwargs})
        traces = []
        for t in tasks:
            trace = {"task_id": t["id"], "session_id": "bench_fake",
                     "status": "completed", "turns": 5, "tool_calls": [],
                     "denied_calls": [], "bench_probe_count": 0,
                     "duration_seconds": 1.0, "final_text": "- one finding"}
            over = overrides.get(t["id"], {})
            trace.update(over(kwargs) if callable(over) else over)
            traces.append(trace)
        return traces

    monkeypatch.setattr(arms, "load_config", load_config)
    monkeypatch.setattr(arms, "load_bench_tasks", load_bench_tasks)
    monkeypatch.setattr(arms, "require_tool_sandbox", _no_sandbox)
    monkeypatch.setattr(arms, "materialize_baseline", lambda cfg: ("BASELINE_V", tmp_path))
    monkeypatch.setattr(arms, "run_bench_sdk", run_bench_sdk)
    monkeypatch.setattr(arms, "judge_trace", lambda task, trace, rubric_model=None: dict(_SCORE))
    return calls


def test_max_turns_reaches_run_bench_sdk_and_every_trial_row(monkeypatch, tmp_path):
    """Clause 5: `--max-turns` becomes `max_agent_turns`, and the row says so."""
    calls = _install_fake_harness(monkeypatch, tmp_path)
    out = tmp_path / "arms"
    # `arms.main` owns the event loop (`asyncio.run(main_async(...))`), so this is a
    # plain synchronous call from the test.
    rc = arms.main(["--out", str(out), "--trials", "1", "--parallel", "1", "--max-turns", "7"])
    assert rc == 0
    # Both arms run: `all` over the four tasks, `one` over the two phrased ones.
    assert len(calls) == 2, [c["task_ids"] for c in calls]
    assert {c["max_agent_turns"] for c in calls} == {7}, "the cap must reach every arm"
    rows = [json.loads(line) for line in (out / "trials.jsonl").open()]
    # 4 `all` + 4 `spam` re-scores of those same replies + 2 `one`. The spam rows
    # carry the budget too: it is the cap that governed the reply they re-score.
    assert len(rows) == 10
    assert {r["turn_budget"] for r in rows} == {7}
    # Beside `turns`, which is what makes a 12-turn reply distinguishable from one
    # the cap cut off.
    assert all("turns" in r and r["turn_budget"] == 7 for r in rows)
    assert {r["arm"] for r in rows} == {"all", "one", "spam"}


def test_a_trial_cut_off_mid_answer_is_flagged_in_trials_jsonl_with_both_budgets(
        monkeypatch, tmp_path):
    """Clauses 1 and 3 (#2160): every row says whether the turn cap cut it, and both
    budgets ride on the row, so the two causes stay distinguishable in the file
    without the command line that produced it.

    Three traces, three verdicts. `_CUT` is the 2026-09-24 bench_016 shape:
    `status='success'` — a turn-cap stop is not an error to the harness — turns one
    past the cap, empty reply. `_CLOCK` is bench_014 trial 1: `status='timeout'`,
    `turns: 0`, and it must NOT be flagged turn-truncated, because blaming the wall
    clock on the turn cap sends the next reader to the wrong knob. `_ANSWERED` is the
    control that keeps the flag from being a constant.

    The capped trace's turn count is computed from the kwargs the fake runner was
    handed, so the driver's own `--max-turns` decides the flag; the CLI string, not
    an args object, is the test's input.
    """
    cap = arms.DEFAULT_MAX_AGENT_TURNS          # 12, what --max-turns defaults to
    keyed = _run_with_cuts(monkeypatch, tmp_path)

    capped = keyed[("all", _CUT)]
    assert capped["turns"] == cap + 1, capped           # 13, as 09-24 recorded it
    assert capped["status"] == "success", capped        # the cap is not an error
    assert len(capped["final_text"]) == 0, capped
    assert capped["turn_truncated"] is True, capped
    assert capped["turn_budget"] == cap, capped
    assert capped["timeout_budget"] == 900, capped      # what --timeout was passed

    clock = keyed[("all", _CLOCK)]
    assert clock["status"] == "timeout", clock
    assert clock["turns"] == 0, clock
    assert clock["turn_truncated"] is False, clock      # a clock stop is not a cap stop
    assert clock["timeout_budget"] == 900, clock
    assert clock["turn_budget"] == cap, clock

    answered = keyed[("all", _ANSWERED)]
    assert answered["turn_truncated"] is False, answered
    assert answered["timeout_budget"] == 900, answered

    # The flag is a real bool and both budgets are the run's own on every row of the
    # file — the spam re-score included, since it re-scores a reply that was already
    # cut and must inherit that trial's flag rather than lose it.
    assert all(isinstance(r["turn_truncated"], bool) for r in keyed.values()), keyed
    assert {r["timeout_budget"] for r in keyed.values()} == {900}, keyed
    assert {r["turn_budget"] for r in keyed.values()} == {cap}, keyed
    assert keyed[("spam", _CUT)]["turn_truncated"] is True, keyed[("spam", _CUT)]
    assert keyed[("one", _CUT)]["turn_truncated"] is True, keyed[("one", _CUT)]
    assert keyed[("one", _CLOCK)]["turn_truncated"] is False, keyed[("one", _CLOCK)]


def _run_with_cuts(monkeypatch, tmp_path):
    """One driver run with every truncation shape present; the rows, keyed by arm+task.

    Shared by the two #2160 nodes because writing the file and reading it back have to
    be the same run: a row excluded correctly from a summary nobody wrote is not the
    thing under test. `tmp_path/"cut"` is where the artifact lands, so the caller can
    re-read it.
    """
    def capped_trace(kwargs):
        # The real runner's capped turn count is one past the cap it was given:
        # `app/harness/loop.py` increments `num_turns` and only then compares it to
        # `options.max_turns`, which is why `turns == turn_budget` never fires.
        return {"status": "success", "turns": int(kwargs["max_agent_turns"]) + 1,
                "final_text": "", "duration_seconds": 120.0}

    def clock_cut_trace(kwargs):
        return {"status": "timeout", "turns": 0,
                "duration_seconds": float(kwargs["per_task_timeout"]) + 0.4,
                "final_text": ""}

    def at_cap_trace(kwargs):
        # Used every turn it was given and still produced a reply: no signal apart
        # from the turn count distinguishes it, so it is treated as cut too.
        return {"status": "success", "turns": int(kwargs["max_agent_turns"]),
                "final_text": "- the last finding"}

    _install_fake_harness(monkeypatch, tmp_path, traces_by_task={
        _CUT: capped_trace, _CLOCK: clock_cut_trace, _AT_CAP: at_cap_trace})
    out = tmp_path / "cut"
    # The `one` arm runs over the same two task ids it rewrites, so the cut shapes
    # appear in both arms — which is what makes an arm-vs-arm comparison honest.
    rc = arms.main(["--out", str(out), "--trials", "1", "--parallel", "1",
                    "--timeout", "900"])
    assert rc == 0, rc
    rows = [json.loads(line) for line in (out / "trials.jsonl").open()]
    keyed = {(r["arm"], r["task_id"]): r for r in rows}
    assert {("all", _CUT), ("all", _CLOCK), ("all", _ANSWERED), ("all", _AT_CAP),
            ("spam", _CUT), ("one", _CUT), ("one", _CLOCK)} <= set(keyed), sorted(keyed)
    return keyed


def test_the_written_rows_produce_the_split_the_summary_reports(monkeypatch, tmp_path):
    """The row-level flags and the cell counts are one chain, end to end.

    The pure node above proves `summarise`'s arithmetic over hand-built rows; this
    proves the file the driver actually wrote carries enough to run that arithmetic
    at all. That is the half that rots silently: if `_row` stopped stamping
    `turn_truncated`, a missing key would read as "not truncated", the cut trial
    would go straight back into the mean, and every node above would still pass.
    """
    cap = arms.DEFAULT_MAX_AGENT_TURNS
    keyed = _run_with_cuts(monkeypatch, tmp_path)
    rows = [json.loads(line)
            for line in (tmp_path / "cut" / "trials.jsonl").open()]
    cells = arms.summarise(rows)["cells"]

    cut = cells[f"all/{_CUT}"]
    assert (cut["n"], cut["n_truncated"], cut["n_timeout"], cut["n_scored"]) \
        == (1, 1, 0, 0), cut
    assert cut["recall"] is None, cut          # no mean over zero answered trials

    clocked = cells[f"all/{_CLOCK}"]
    assert (clocked["n"], clocked["n_truncated"], clocked["n_timeout"],
            clocked["n_scored"]) == (1, 0, 1, 0), clocked

    ok = cells[f"all/{_ANSWERED}"]
    assert (ok["n"], ok["n_truncated"], ok["n_timeout"], ok["n_scored"]) \
        == (1, 0, 0, 1), ok
    assert ok["recall"] == 1.0, ok

    # A trial that used every turn it was given and still answered is a judgement
    # call, and the call the clause records is "cut": the trace carries no signal
    # that it was anything else.
    at_cap = cells[f"all/{_AT_CAP}"]
    assert keyed[("all", _AT_CAP)]["turns"] == cap, keyed[("all", _AT_CAP)]
    assert (at_cap["n"], at_cap["n_truncated"], at_cap["n_scored"]) == (1, 1, 0), at_cap



def test_the_default_budget_is_the_runners_own(monkeypatch, tmp_path):
    """The default is the runner's constant, imported not copied — and readable alone.

    `--summarise` costs nothing to run and a trial does not, so the default has to
    be inspectable without booting the harness.
    """
    args = arms.build_parser().parse_args(["--out", str(tmp_path)])
    assert args.max_turns == arms.DEFAULT_MAX_AGENT_TURNS == 12
    assert arms.build_parser().parse_args(
        ["--out", str(tmp_path), "--max-turns", "3"]).max_turns == 3
    # A run that passes no --max-turns still records the cap it ran under, so the
    # budget never goes back to being an accident of the harness version.
    calls = _install_fake_harness(monkeypatch, tmp_path)
    arms.main(["--out", str(tmp_path / "d"), "--trials", "1", "--parallel", "1"])
    assert {c["max_agent_turns"] for c in calls} == {arms.DEFAULT_MAX_AGENT_TURNS}
    rows = [json.loads(line) for line in (tmp_path / "d" / "trials.jsonl").open()]
    assert {r["turn_budget"] for r in rows} == {arms.DEFAULT_MAX_AGENT_TURNS}
