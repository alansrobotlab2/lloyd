"""Behavioural-regression detection for a landed self-modification.

The measured fact this is built on: five consecutive `eval/run_eval.py` runs
against an unchanged vault produced identical values for every quality metric
(stdev 0.0000). The eval contributes no noise. What *does* move is the vault
underneath it, which is why the comparison is a paired same-data A/B rather
than a check against a number recorded at the last promotion.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import pytest

from workers.sources import automod_regression as R


ZERO_NOISE = {"metrics": {k: {"stdev": 0.0} for k in R.ARMED_METRICS}}


def base(**over):
    d = {k: 0.6 for k in R.ARMED_METRICS}
    d["errors"] = 0
    d.update(over)
    return d


# ---------------------------------------------------------------------------
# evaluate — pure
# ---------------------------------------------------------------------------

def test_identical_runs_are_not_a_regression():
    regressed, reasons, _ = R.evaluate(base(), base(), ZERO_NOISE)
    assert not regressed and reasons == []


def test_a_drop_beyond_tolerance_regresses():
    regressed, reasons, _ = R.evaluate(base(entity_hit_rate=0.45), base(), ZERO_NOISE)
    assert regressed
    assert "entity_hit_rate" in reasons[0]


def test_an_improvement_is_never_a_regression():
    regressed, _, _ = R.evaluate(base(entity_hit_rate=0.95), base(), ZERO_NOISE)
    assert not regressed


def test_float_wobble_within_the_floor_is_tolerated():
    """MIN_SIGMA exists only to absorb representation noise, not vault drift."""
    tiny = 0.6 - (R.MIN_SIGMA * R.SIGMA_MULTIPLIER / 2)
    regressed, _, _ = R.evaluate(base(entity_hit_rate=tiny), base(), ZERO_NOISE)
    assert not regressed


def test_every_armed_metric_can_fire():
    for metric in R.ARMED_METRICS:
        regressed, reasons, _ = R.evaluate(base(**{metric: 0.1}), base(), ZERO_NOISE)
        assert regressed, f"{metric} did not fire"
        assert metric in reasons[0]


def test_report_only_metrics_never_fire():
    """latency moves 562ms run to run; it must not be able to revert code."""
    cur = base(); cur["latency_ms_avg"] = 9999.0
    was = base(); was["latency_ms_avg"] = 100.0
    regressed, reasons, detail = R.evaluate(cur, was, ZERO_NOISE)
    assert not regressed, reasons
    assert detail["latency_ms_avg"]["armed"] is False


# ---------------------------------------------------------------------------
# the absolute latency budget (#1129)
# ---------------------------------------------------------------------------
#
# `test_report_only_metrics_never_fire` above is deliberately UNCHANGED by this
# change, including its default `context` (a 9,999 ms run is inside the
# paired-check ceiling, so it stays a no-verdict comparison). It is the property
# the report-only exemption existed to protect, and the acceptance keeps it green
# as written rather than editing it to fit. The two new halves — a verdict that
# fires past the ceiling, and one that still cannot reach the rollback channel —
# are pinned by the tests below instead of by rewriting that one.

# What each context has recorded, read off `eval/baselines/*.json` on
# 2026-09-18. `nightly` is `eval/run_eval.py` at production defaults against the
# live daemon (`matches_production_defaults: true`); `paired_check` is this
# module's own two-arm run against the frozen snapshot.
POST_WIDENING_MS = {
    # The step #504 bought: 708 ms nightly before it, 4,230-4,379 ms after, and
    # nobody graded it because nothing read the field.
    R.CONTEXT_NIGHTLY: 4378.8,        # nightly-20260917-20260917-060324.json
    R.CONTEXT_PAIRED_CHECK: 12863.4,  # automod-check-20260917-024810.json
}
# The worst each context has EVER produced across every artifact on disk, which
# is what the ceiling must clear for "no recorded run reads over budget" to be
# literally true. The nightly one is a 2026-09-04 outlier that predates #504.
WORST_EVER_MS = {
    R.CONTEXT_NIGHTLY: 4408.0,        # nightly-20260904-20260904-060219.json
    R.CONTEXT_PAIRED_CHECK: 12863.4,  # automod-check-20260917-024810.json
}


def test_there_are_two_named_budgets_and_every_context_has_one():
    """One ceiling for both runs would be wrong in whichever direction it moved.

    Nightly averages 4.2-4.4 s and the pinned paired check 12.0-12.9 s: the same
    queries at a different absolute cost, so a single number either calls every
    gate run over budget or tolerates any nightly step.
    """
    assert set(R.LATENCY_BUDGET_MS) == {R.CONTEXT_NIGHTLY, R.CONTEXT_PAIRED_CHECK}
    assert R.LATENCY_BUDGET_MS[R.CONTEXT_NIGHTLY] != R.LATENCY_BUDGET_MS[R.CONTEXT_PAIRED_CHECK]
    for context in WORST_EVER_MS:
        assert R.latency_budget(context) > 0, f"{context} has no budget"


def test_no_recorded_run_reads_over_budget():
    """#504's accepted 6x cost is grandfathered by construction.

    Each ceiling sits at or above the worst average its context has EVER
    recorded, so the budget cannot fire on history — it exists to catch the NEXT
    step of that class. A budget set below the current numbers would either be
    edited the first time it fired or silence the report it is for.

    The graded quantity is `WORST_EVER_MS`, the maximum across every artifact on
    disk, not the worst post-#504 run: the nightly context contains a 4,408.0 ms
    night on 2026-09-04 that predates #504 entirely, so clearing only the
    post-widening pair would leave a recorded run reading over budget.
    """
    for context, worst in WORST_EVER_MS.items():
        assert R.latency_budget(context) >= worst, (
            f"the {context} budget {R.latency_budget(context)} ms is below the "
            f"{worst} ms its own baselines already report")
        verdict = R.over_budget({"latency_ms_avg": worst}, context)
        assert verdict["over"] is False, f"{context} already reads over budget"
    for context, post in POST_WIDENING_MS.items():
        assert R.latency_budget(context) >= post
        assert not R.over_budget({"latency_ms_avg": post}, context)["over"]


def test_the_paired_check_default_tolerates_a_nightly_sized_run():
    """The two contexts are two questions, not one threshold with two names.

    6,000 ms is comfortably over anything the nightly arm has ever done and
    inside what the pinned check does routinely — so the context, not the
    number, is what makes the verdict meaningful.
    """
    over_nightly = R.over_budget({"latency_ms_avg": 6000.0}, R.CONTEXT_NIGHTLY)
    inside_paired = R.over_budget({"latency_ms_avg": 6000.0}, R.CONTEXT_PAIRED_CHECK)
    assert over_nightly["over"] is True
    assert inside_paired["over"] is False


def test_the_budget_comment_prices_itself_on_fresh_queries():
    """The 165 ms figure that justified #504's 6x step was a cached repeat.

    `agent_mcp/vault.py`'s own docstring warns that "the daemon caches query
    embeddings, so a sequential A/B measures arm order, not arm" — and the
    number that priced the widening came from exactly that. The ceiling may only
    be justified by a fresh-query measurement, and the comment has to name both
    the command and the cold-vs-warm spread it is set against, or the next
    widening re-derives a fake number again.
    """
    src = Path(R.__file__).read_text(encoding="utf-8")
    header = src[src.index("# ── the latency budget"):src.index("LATENCY_BUDGET_MS = {")]
    assert "_qmd_daemon_search" in header, "the budget names no measurement command"
    assert "3,672-4,027" in header and "119-183" in header, (
        "the budget must name the fresh-vs-cached spread it is set from "
        "(3,672-4,027 ms fresh vs 119-183 ms cached at pool 240)")
    assert "FRESH" in header and "CACHED" in header, (
        "a price in this comment that does not name its sample kind is exactly "
        "the defect that priced the widening at 55 ms")
    # Whole module, not the header slice: the sentence this retires ("the one
    # thing that still moves, by ~1.7s between a cold and warm embedding cache")
    # lived ABOVE `ARMED_METRICS`, outside the budget block, so a header-scoped
    # check would pass with the retired justification still in the file. This is
    # where that retirement is pinned — `architecture/automod.md` never contained
    # the string (`git log -S'1.7s' -- architecture/automod.md` is empty).
    assert "1.7s" not in src and "1.7 s" not in src, (
        "the asserted cold/warm band is wrong by more than 2x (measured 3.9 s "
        "cold-to-warm today); the file may not carry a band that no test can grade")
    # The two ceilings must be justifiable from the numbers above them: the
    # nightly one sits over the worst nightly artifact, and the comment says which.
    assert "worst ever" in src[src.index("LATENCY_BUDGET_MS = {"):], (
        "the budget constants no longer name the recorded run each clears")


def test_an_unknown_context_is_no_verdict_not_a_pass():
    """A ceiling nobody priced must not be able to certify a run healthy."""
    assert R.latency_budget("made_up_context") == 0.0
    assert R.over_budget({"latency_ms_avg": 999_999.0}, "made_up_context") is None


def test_a_missing_average_is_no_verdict_not_a_pass():
    verdict = R.over_budget({}, R.CONTEXT_NIGHTLY)
    assert verdict is None, "no latency measured is not latency inside budget"


def test_over_budget_never_enters_the_rollback_channel():
    """The property the report-only exemption existed to protect (#1129).

    The ceiling makes a 6x step REPORTABLE; it must not make it revertable. A
    comparison that is over budget in both contexts at once, with every armed
    metric untouched, yields the verdict and no `regressed`, no reason, and no
    new armed metric — the seven quality metrics stay the only armed ones.
    """
    cur = base(latency_ms_avg=99_000.0)
    for context in (R.CONTEXT_NIGHTLY, R.CONTEXT_PAIRED_CHECK):
        regressed, reasons, detail = R.evaluate(cur, base(), ZERO_NOISE, context)
        assert regressed is False, reasons
        assert reasons == []
        assert detail["latency_ms_avg"]["budget"]["context"] == context
        assert detail["latency_ms_avg"]["budget"]["over"] is True
    assert set(R.ARMED_METRICS) == {
        "entity_hit_rate", "entity_recall_avg", "fact_entity_recall_avg",
        "ndcg10", "mrr_doc", "doc_hit_rate", "doc_recall_avg"}, (
        "arming latency is the one thing this change must not do")


def test_an_in_budget_comparison_emits_no_verdict_at_all():
    """Inside the ceiling there is a reading and no verdict — it is not a gauge.

    `detail["latency_ms_avg"]["budget"]` still carries the ceiling the run was
    read against, because a reader has to be able to tell "measured, inside" from
    "never measured" — the same distinction this module already draws for a
    missing noise file. What must be absent is `latency_over_budget`.
    """
    cur = base(latency_ms_avg=R.latency_budget(R.CONTEXT_PAIRED_CHECK) - 1)
    _, _, detail = R.evaluate(cur, base(), ZERO_NOISE, R.CONTEXT_PAIRED_CHECK)
    assert R.OVER_BUDGET_FIELD not in detail["latency_ms_avg"], (
        "an in-budget comparison emitted an over-budget verdict")
    assert detail["latency_ms_avg"]["budget"]["over"] is False
    inside = R.over_budget({"latency_ms_avg": 100.0}, R.CONTEXT_NIGHTLY)
    assert inside["over"] is False and inside["budget_ms"] == R.latency_budget(R.CONTEXT_NIGHTLY)


def test_an_over_budget_comparison_emits_the_named_verdict():
    """The other half of the pair above: past the ceiling the verdict is there."""
    cur = base(latency_ms_avg=R.latency_budget(R.CONTEXT_NIGHTLY) + 1)
    regressed, reasons, detail = R.evaluate(cur, base(), ZERO_NOISE, R.CONTEXT_NIGHTLY)
    verdict = detail["latency_ms_avg"][R.OVER_BUDGET_FIELD]
    assert verdict["over"] is True and verdict["context"] == R.CONTEXT_NIGHTLY
    assert regressed is False and reasons == []


def test_new_eval_errors_regress():
    regressed, reasons, _ = R.evaluate(base(errors=3), base(errors=0), ZERO_NOISE)
    assert regressed and "errors" in reasons[0]


def test_pre_existing_errors_do_not_regress():
    regressed, _, _ = R.evaluate(base(errors=3), base(errors=2), ZERO_NOISE)
    assert not regressed


def test_a_measured_sigma_widens_the_tolerance():
    noisy = {"metrics": {"entity_hit_rate": {"stdev": 0.05}}}
    # 0.10 drop is inside 3σ=0.15 for a metric measured as noisy...
    regressed, _, _ = R.evaluate(base(entity_hit_rate=0.50), base(), noisy)
    assert not regressed
    # ...but fires against the zero-variance floor.
    regressed, _, _ = R.evaluate(base(entity_hit_rate=0.50), base(), ZERO_NOISE)
    assert regressed


def test_missing_metrics_are_skipped_not_assumed_good():
    regressed, reasons, detail = R.evaluate({"errors": 0}, {"errors": 0}, ZERO_NOISE)
    assert not regressed
    assert all(not d.get("armed") for d in detail.values())


# ---------------------------------------------------------------------------
# execute — the guard rails
# ---------------------------------------------------------------------------

class _FakePin:
    """Stands in for the pinned corpus. The real one snapshots 1 GB and starts
    a daemon; these tests are about `execute`'s decisions, not the pin."""
    provenance = {"index": "/fake/evalpin.sqlite", "documents": 11756}
    port = 8182

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def env_for(self, base=None, *, code_root=None):
        return {"LLOYD_CONFIG_OVERLAY": "/fake/overlay.yaml",
                "LLOYD_CODE_ROOT": str(code_root or "/fake/tree")}

    def discard(self):
        pass


@contextlib.contextmanager
def _fake_worktree(commit):
    yield Path("/fake/worktree")


def _pin_ok(monkeypatch):
    monkeypatch.setattr(R, "PinnedCorpus", lambda workdir, **kw: _FakePin())
    monkeypatch.setattr(R, "_baseline_worktree", _fake_worktree)


class _Item:
    payload: dict = {}


def _observing(**over):
    import time
    d = {"commit": "b" * 40, "parent": "a" * 40, "state": "observing",
         "landed_ts": time.time(), "changed_paths": ["app/x.py"]}
    d.update(over)
    return d


async def test_nothing_recent_to_check_is_a_noop(monkeypatch):
    import scripts.automod.state as S
    monkeypatch.setattr(S, "read_current", lambda: None)
    monkeypatch.setattr(S, "read_last_settled", lambda: None)
    out = R._execute_blocking()
    assert "skipped" in out


async def test_a_settled_promotion_is_still_checked(monkeypatch, tmp_path):
    """The window is 15 minutes and this job runs hourly at best.

    Keying on `current.json` alone meant the subject was gone before the check
    ever ran — which is why this source has never produced a single run.
    """
    import scripts.automod.state as S
    monkeypatch.setattr(S, "read_current", lambda: None)
    monkeypatch.setattr(S, "read_last_settled", lambda: _observing())
    monkeypatch.setattr(S, "read_events", lambda **k: [])
    monkeypatch.setattr(S, "append_event", lambda *a, **k: None)
    monkeypatch.setattr(R, "NOISE_PATH", tmp_path / "absent.json")
    out = R._execute_blocking()
    # Got past subject selection: it failed on the noise floor, not on
    # "nothing to check".
    assert "no measured noise floor" in out["skipped"]


async def test_the_baseline_is_the_parent_not_the_lkg(monkeypatch, tmp_path):
    """After settling, the LKG pointer IS the promoted commit.

    Comparing against it would check out the same code in both arms and be
    structurally incapable of finding anything.
    """
    import scripts.automod.state as S
    seen = {}
    noise = tmp_path / "noise.json"
    noise.write_text(json.dumps(ZERO_NOISE))
    monkeypatch.setattr(R, "NOISE_PATH", noise)
    monkeypatch.setattr(S, "read_current", lambda: _observing())
    monkeypatch.setattr(S, "read_events", lambda **k: [])
    monkeypatch.setattr(S, "append_event", lambda *a, **k: None)
    monkeypatch.setattr(S, "read_lkg", lambda: {"commit": "b" * 40})   # == promoted
    _pin_ok(monkeypatch)
    monkeypatch.setattr(R, "_baseline_worktree",
                        lambda c: seen.setdefault("baseline", c) and _fake_worktree(c)
                        or _fake_worktree(c))
    monkeypatch.setattr(R, "_run_arm", lambda *a, **k: None)
    R._execute_blocking()
    assert seen["baseline"] == "a" * 40, "must compare against the parent"


async def test_one_measurement_per_promotion(monkeypatch, tmp_path):
    """Both arms are full evals plus a worktree; the answer cannot change."""
    import scripts.automod.state as S
    monkeypatch.setattr(S, "read_current", lambda: _observing())
    monkeypatch.setattr(S, "read_last_settled", lambda: None)
    monkeypatch.setattr(S, "read_events", lambda **k: [
        {"event": "regression_check", "commit": "b" * 40}])
    out = R._execute_blocking()
    assert "already checked" in out["skipped"]


async def test_an_empty_corpus_arm_cannot_evaluate(monkeypatch, tmp_path):
    """An empty graph is not a low score, it is a measurement that did not happen.

    And it does not look like one: with the graph deleted, mrr_doc, ndcg10 and
    doc_hit_rate come back identical to a real run.
    """
    import scripts.automod.state as S
    noise = tmp_path / "noise.json"
    noise.write_text(json.dumps(ZERO_NOISE))
    monkeypatch.setattr(R, "NOISE_PATH", noise)
    monkeypatch.setattr(S, "read_current", lambda: _observing())
    monkeypatch.setattr(S, "read_events", lambda **k: [])
    monkeypatch.setattr(S, "append_event", lambda *a, **k: None)
    _pin_ok(monkeypatch)
    arms = iter([{"overall": base(), "corpus_ok": True, "corpus": {}},
                 {"overall": base(), "corpus_ok": False, "corpus": {"entities": 0}}])
    monkeypatch.setattr(R, "_run_arm", lambda *a, **k: next(arms))
    out = R._execute_blocking()
    assert "empty corpus" in out["skipped"]


async def test_a_regression_is_handed_to_the_guardian(monkeypatch, tmp_path):
    """Never rolled back inline: the rollback stops the process doing it."""
    import scripts.automod.state as S
    captured = {}
    noise = tmp_path / "noise.json"
    noise.write_text(json.dumps(ZERO_NOISE))
    monkeypatch.setattr(R, "NOISE_PATH", noise)
    monkeypatch.setattr(S, "read_current", lambda: _observing())
    monkeypatch.setattr(S, "read_events", lambda **k: [])
    monkeypatch.setattr(S, "append_event", lambda *a, **k: None)
    monkeypatch.setattr(S, "write_eval_last", lambda payload: None)
    monkeypatch.setattr(S, "request_rollback", lambda **kw: captured.update(kw) or kw)
    _pin_ok(monkeypatch)
    bad = {"overall": base(entity_hit_rate=0.1), "corpus_ok": True, "corpus": {}}
    # Three arms: a regression is looked at twice before anyone acts on it.
    arms = iter([{"overall": base(), "corpus_ok": True, "corpus": {}}, bad, dict(bad)])
    monkeypatch.setattr(R, "_run_arm", lambda *a, **k: next(arms))
    out = R._execute_blocking()
    assert out["regressed"] is True
    assert captured["trigger"] == "regression"
    assert captured["target"] == "a" * 40 and captured["commit"] == "b" * 40
    assert out["fact_side_reasons"], "a fact-layer drop must be reported as such"


async def test_an_over_budget_run_is_recorded_and_never_rolled_back(monkeypatch, tmp_path):
    """The whole point of #1129, across the seam into the state files.

    #504 landed with every rung green while nightly retrieval latency went
    708 ms → 4,230 ms, because the field was written into twenty artifacts and
    read by nothing. So the verdict has to reach the two places a reader actually
    looks — the `regression_check` ledger event and `eval_last.json`, which the
    guardian folds into the LKG record — and must not reach the guardian's
    rollback queue. Both directions are pinned here: the field is present with
    `regressed` False, and `request_rollback` is never called.
    """
    import scripts.automod.state as S
    events: list = []
    eval_last: dict = {}
    rollback: dict = {}
    noise = tmp_path / "noise.json"
    noise.write_text(json.dumps(ZERO_NOISE))
    monkeypatch.setattr(R, "NOISE_PATH", noise)
    monkeypatch.setattr(S, "read_current", lambda: _observing())
    monkeypatch.setattr(S, "read_events", lambda **k: [])
    monkeypatch.setattr(S, "append_event", lambda ev, **k: events.append(ev))
    monkeypatch.setattr(S, "write_eval_last", lambda payload: eval_last.update(payload))
    monkeypatch.setattr(S, "request_rollback", lambda **kw: rollback.update(kw) or kw)
    _pin_ok(monkeypatch)
    slow = 99_000.0     # 7x the paired-check ceiling; quality untouched
    arms = iter([_arm(20, []), _arm(20, [], latency_ms_avg=slow)])
    monkeypatch.setattr(R, "_run_arm", lambda *a, **k: next(arms))
    out = R._execute_blocking()

    assert out["regressed"] is False, out
    assert not rollback, "an over-budget latency run must never request a rollback"
    check = next(e for e in events if e.get("event") == "regression_check")
    assert check[R.OVER_BUDGET_FIELD]["over"] is True
    assert check[R.OVER_BUDGET_FIELD]["context"] == R.CONTEXT_PAIRED_CHECK
    assert check["regressed"] is False and check["reasons"] == []
    assert eval_last[R.OVER_BUDGET_FIELD]["over"] is True
    assert eval_last["regressed"] is False


async def test_an_in_budget_run_records_the_reading_and_no_verdict(monkeypatch, tmp_path):
    """`latency_budget` is written for every measurable run; the verdict is not.

    The reading has to be present even when the answer is "fine", so a reader can
    tell an inside-budget run from one where the check never ran; the named
    verdict appears only past the ceiling, so nothing has to learn a new meaning
    for "null" in `latency_over_budget`.
    """
    import scripts.automod.state as S
    events: list = []
    eval_last: dict = {}
    noise = tmp_path / "noise.json"
    noise.write_text(json.dumps(ZERO_NOISE))
    monkeypatch.setattr(R, "NOISE_PATH", noise)
    monkeypatch.setattr(S, "read_current", lambda: _observing())
    monkeypatch.setattr(S, "read_events", lambda **k: [])
    monkeypatch.setattr(S, "append_event", lambda ev, **k: events.append(ev))
    monkeypatch.setattr(S, "write_eval_last", lambda payload: eval_last.update(payload))
    monkeypatch.setattr(S, "request_rollback", lambda **kw: kw)
    _pin_ok(monkeypatch)
    inside = R.latency_budget(R.CONTEXT_PAIRED_CHECK) - 1.0
    arms = iter([_arm(20, []), _arm(20, [], latency_ms_avg=inside)])
    monkeypatch.setattr(R, "_run_arm", lambda *a, **k: next(arms))
    out = R._execute_blocking()

    assert out["regressed"] is False
    assert eval_last[R.OVER_BUDGET_FIELD] is None, (
        "an inside-budget run emitted an over-budget verdict")
    assert eval_last["latency_budget"]["over"] is False
    assert eval_last["latency_budget"]["latency_ms_avg"] == inside
    assert eval_last["regressed"] is False


async def test_a_missing_noise_file_means_cannot_evaluate(monkeypatch, tmp_path):
    """Never 'no regression'. `eval/baselines/` is gitignored and can be absent."""
    import scripts.automod.state as S
    monkeypatch.setattr(S, "read_current", lambda: _observing())
    monkeypatch.setattr(S, "read_events", lambda **k: [])
    monkeypatch.setattr(S, "append_event", lambda *a, **k: None)
    monkeypatch.setattr(R, "NOISE_PATH", tmp_path / "absent.json")
    out = R._execute_blocking()
    assert "no measured noise floor" in out["skipped"]


async def test_a_failed_paired_baseline_does_not_silently_pass(monkeypatch, tmp_path):
    import scripts.automod.state as S
    noise = tmp_path / "noise.json"
    noise.write_text(json.dumps(ZERO_NOISE))
    monkeypatch.setattr(R, "NOISE_PATH", noise)
    monkeypatch.setattr(S, "read_current", lambda: _observing())
    monkeypatch.setattr(S, "read_events", lambda **k: [])
    monkeypatch.setattr(S, "read_lkg", lambda: {"commit": "a" * 40})
    monkeypatch.setattr(S, "append_event", lambda *a, **k: None)
    _pin_ok(monkeypatch)
    monkeypatch.setattr(R, "_run_arm", lambda *a, **k: None)
    out = R._execute_blocking()
    assert "paired baseline run failed" in out["skipped"]


def test_the_recorded_noise_floor_is_what_the_code_expects():
    """If someone re-measures and the eval turns out noisy, this fails loudly."""
    if not R.NOISE_PATH.exists():
        pytest.skip("noise floor not measured on this machine")
    noise = json.loads(R.NOISE_PATH.read_text())
    for metric in R.ARMED_METRICS:
        entry = noise["metrics"].get(metric)
        if entry is None:
            continue
        assert entry["stdev"] < 0.01, (
            f"{metric} measured stdev {entry['stdev']}; it is armed with a "
            f"near-zero tolerance and would fire on noise")


# ---------------------------------------------------------------------------
# a query the daemon did not answer is not a score
# ---------------------------------------------------------------------------

def _record(qid, n_docs, error=None):
    return {"id": qid, "result_summary": {"n_docs": n_docs}, "error": error}


def test_doc_coverage_names_the_unanswered_queries():
    """An errored query is already counted by `errors`; a record from before
    `result_summary` existed reports nothing, so an old baseline cannot refuse."""
    cov = R._doc_coverage([_record("a", 20), _record("b", 0),
                           _record("c", 0, error="boom"), {"id": "legacy"}])
    assert cov["n_records"] == 4
    assert cov["empty_doc_queries"] == ["b"]


def test_a_partial_empty_arm_is_not_a_measurement():
    arm = {"n_records": 20, "empty_doc_queries": ["backlog-363"]}
    assert R.unanswered_doc_queries(arm) == ["backlog-363"]


def test_an_arm_with_every_query_empty_is_a_score():
    """The one shape a change under test can produce: retrieval itself broke."""
    assert R.unanswered_doc_queries({"n_records": 3, "empty_doc_queries": ["a", "b", "c"]}) == []
    assert R.unanswered_doc_queries({"n_records": 3, "empty_doc_queries": []}) == []
    assert R.unanswered_doc_queries({}) == []


def test_load_run_carries_doc_coverage(tmp_path):
    blob = {"summary": {"overall": base()}, "corpus_ok": True,
            "records": [_record("a", 5), _record("b", 0)]}
    (tmp_path / "automod-check-x.json").write_text(json.dumps(blob))
    run = R._load_run(tmp_path, "automod-check")
    assert run["n_records"] == 2 and run["empty_doc_queries"] == ["b"]


def _arm(n_records, empty, **over):
    return {"overall": base(**over), "corpus_ok": True, "corpus": {},
            "n_records": n_records, "empty_doc_queries": empty}


async def test_an_arm_the_daemon_did_not_answer_cannot_evaluate(monkeypatch, tmp_path):
    """2026-09-17 05:34Z: one query of twenty came back with zero documents and
    no error, the eval scored it 0, and a change to a memory-maintenance
    script was reverted for a -0.05 `doc_hit_rate`."""
    import scripts.automod.state as S
    captured = {}
    noise = tmp_path / "noise.json"
    noise.write_text(json.dumps(ZERO_NOISE))
    monkeypatch.setattr(R, "NOISE_PATH", noise)
    monkeypatch.setattr(S, "read_current", lambda: _observing())
    monkeypatch.setattr(S, "read_events", lambda **k: [])
    monkeypatch.setattr(S, "append_event", lambda *a, **k: None)
    monkeypatch.setattr(S, "request_rollback", lambda **kw: captured.update(kw) or kw)
    _pin_ok(monkeypatch)
    arms = iter([_arm(20, []), _arm(20, ["backlog-363"], doc_hit_rate=0.55)])
    monkeypatch.setattr(R, "_run_arm", lambda *a, **k: next(arms))
    out = R._execute_blocking()
    assert "did not answer" in out["skipped"] and "backlog-363" in out["skipped"]
    assert not captured, "a non-measurement must never request a rollback"


async def test_a_retriever_that_answers_nothing_is_still_a_regression(monkeypatch, tmp_path):
    import scripts.automod.state as S
    captured = {}
    noise = tmp_path / "noise.json"
    noise.write_text(json.dumps(ZERO_NOISE))
    monkeypatch.setattr(R, "NOISE_PATH", noise)
    monkeypatch.setattr(S, "read_current", lambda: _observing())
    monkeypatch.setattr(S, "read_events", lambda **k: [])
    monkeypatch.setattr(S, "append_event", lambda *a, **k: None)
    monkeypatch.setattr(S, "write_eval_last", lambda payload: None)
    monkeypatch.setattr(S, "request_rollback", lambda **kw: captured.update(kw) or kw)
    _pin_ok(monkeypatch)
    blind = _arm(3, ["a", "b", "c"], doc_hit_rate=0.0, mrr_doc=0.0)
    arms = iter([_arm(3, []), blind, dict(blind)])      # the second look is blind too
    monkeypatch.setattr(R, "_run_arm", lambda *a, **k: next(arms))
    out = R._execute_blocking()
    assert out["regressed"] is True and captured["trigger"] == "regression"


def test_measure_noise_drops_a_trial_the_daemon_did_not_answer(monkeypatch, tmp_path):
    """A flake inside the floor would widen every tolerance by the flake."""
    monkeypatch.setattr(R, "PinnedCorpus", lambda workdir, **kw: _FakePin())
    monkeypatch.setattr(R, "NOISE_PATH", tmp_path / "noise.json")
    runs = iter([_arm(2, []), _arm(2, ["q1"], doc_hit_rate=0.3), _arm(2, [])])
    monkeypatch.setattr(R, "_run_arm", lambda *a, **k: next(runs))
    noise = R.measure_noise(3)
    assert noise["metrics"]["doc_hit_rate"]["n"] == 2
    assert noise["metrics"]["doc_hit_rate"]["stdev"] == 0.0
    assert noise["dropped_trials"] == ["trial 1: q1"]


# ---------------------------------------------------------------------------
# The fact leg (#1250): a leg that read nothing is not a zero and not a score
# ---------------------------------------------------------------------------

# Measured on the live arms of 2026-09-18, copied into this fixture so the test
# names the real numbers: the last healthy arm averaged fact_entity_recall_avg
# 0.375 with 191 facts across its 20 queries; the six arms from
# ran_at 2026-09-18T18:03:25Z onward averaged 0.0 with 0 facts, `error: null` on
# every query, `errors: 0` and `corpus_ok: true`, against a corpus naming
# 315,462 facts and 26,146 entities.
HEALTHY_FACT_COV = {"n_records": 20, "n_facts_reports": 20, "n_facts_total": 191,
                    "empty_fact_queries": [], "n_fact_reads_failed_total": 0,
                    "fact_read_first_error": None}
ZEROED_FACT_COV = {"n_records": 20, "n_facts_reports": 20, "n_facts_total": 0,
                   "empty_fact_queries": [f"q{i}" for i in range(20)],
                   "n_fact_reads_failed_total": 20,
                   "fact_read_first_error": "StoreUnavailable: kg.sqlite is not readable"}
CORPUS_WITH_FACTS = {"facts": 315462, "entities": 26146, "edges_active": 53416}


def _fact_arm(coverage, *, fact_metric=0.0, corpus_ok=True, corpus=CORPUS_WITH_FACTS):
    """An arm whose ONLY defect is its fact leg: doc metrics normal, corpus
    flagged readable, no query errored. Every existing guard passes it."""
    overall = base(**{"fact_entity_recall_avg": fact_metric})
    return {"overall": overall, "corpus_ok": corpus_ok, "corpus": dict(corpus),
            "n_records": coverage["n_records"],
            "empty_doc_queries": [], "fact_coverage": dict(coverage)}


def _arms_by_label(baseline: dict, current: dict, on_call=None):
    """`_run_arm` by arm label — the check's two arms come from one function, so
    a test that wants different defects on each side dispatches on the label."""
    def run(_tree, label, _env):
        if on_call is not None:
            on_call(label)
        if "lkg" in label:
            return baseline
        return current
    return run


def _fact_check_env(monkeypatch, tmp_path):
    """The paired check with everything but its two arms stubbed (#1250 tests).

    Reuses this file's existing harness: `_pin_ok` for the pinned corpus and the
    two worktrees, `_observing()` for the promotion under watch, and a noise
    floor of stdev 0.0 — which is the MEASURED value for a pinned corpus, not a
    convenience (the module docstring: five consecutive runs on an unchanged
    vault produced identical quality metrics). Returns the list a rollback
    request would be appended to, so a test can assert the channel stayed shut.
    """
    import scripts.automod.state as S
    noise = tmp_path / "noise.json"
    noise.write_text(json.dumps(ZERO_NOISE))
    monkeypatch.setattr(R, "NOISE_PATH", noise)
    monkeypatch.setattr(S, "read_current", lambda: _observing())
    monkeypatch.setattr(S, "read_lkg", lambda: {"commit": "a" * 40})
    monkeypatch.setattr(S, "read_events", lambda **k: [])
    monkeypatch.setattr(S, "append_event", lambda *a, **k: None)
    requested: list = []
    monkeypatch.setattr(S, "request_rollback", lambda *a, **k: requested.append(a))
    _pin_ok(monkeypatch)
    return requested


def test_a_zeroed_fact_leg_is_a_skip_and_not_a_regression(monkeypatch, tmp_path):
    """The fact-leg mirror of the doc-leg refusals (#1250).

    `fact_entity_recall_avg` is ARMED, and under a pinned corpus the measured
    stdev is 0.0, so the tolerance is the MIN_SIGMA floor: 0.375 -> 0.0 is a
    regression by any arithmetic, and it becomes a ROLLBACK REASON for a commit
    that touched nothing in the fact path. That false rollback — not the "graded
    seven runs" line in the item — is the live exposure, so the claim here is
    both halves: the paired check does not report `no regressions`, and it does
    not report a regression either. It says `cannot evaluate`, no rollback is
    requested, and no verdict row is written as a success.

    The CURRENT arm is the only defective one: doc metrics normal, `corpus_ok`
    true, no query errored — so `corpus_ok`, `unanswered_doc_queries` and
    `all_queries_empty` all pass it. That is why this needed its own guard.
    """
    requested = _fact_check_env(monkeypatch, tmp_path)
    monkeypatch.setattr(R, "_run_arm", _arms_by_label(
        current=_fact_arm(ZEROED_FACT_COV),
        baseline=_fact_arm(HEALTHY_FACT_COV, fact_metric=0.375)))

    out = R._execute_blocking()
    assert out["status"] == "skipped", out
    assert out.get("regressed") is not True, out
    assert requested == [], f"a zeroed fact leg reached the rollback channel: {requested}"
    assert "current arm's fact leg read nothing" in out["skipped"], out
    assert "cannot evaluate" in out["skipped"], out
    # The evidence that makes the skip actionable. The swallow that hid this
    # destroyed the traceback, so the failed-read count and the first failure's
    # repr are the only things a person can act on.
    assert "20 failed fact reads" in out["skipped"], out
    assert "StoreUnavailable" in out["skipped"], out
    assert "315462" in out["skipped"], out


def test_a_zeroed_fact_leg_refuses_on_the_baseline_side_too(monkeypatch, tmp_path):
    """Either arm. The baseline tree is a commit checked out and measured by
    whatever code was current then, so a zeroed baseline manufactures a false
    verdict as readily as a zeroed current arm: 0.0 -> 0.375 reads as an
    improvement, and two zeroed arms read as perfect stability."""
    requested = _fact_check_env(monkeypatch, tmp_path)
    monkeypatch.setattr(R, "_run_arm", _arms_by_label(
        baseline=_fact_arm(ZEROED_FACT_COV),
        current=_fact_arm(HEALTHY_FACT_COV, fact_metric=0.375)))

    out = R._execute_blocking()
    assert out["status"] == "skipped", out
    assert out.get("regressed") is not True, out
    assert requested == [], requested
    assert "baseline arm's fact leg read nothing" in out["skipped"], out


def test_a_zeroed_fact_leg_costs_no_confirm_run(monkeypatch, tmp_path):
    """The confirm arm is not spent "confirming" a 0.0 (#1250).

    A regression has to reproduce before anyone acts on it, so the paired check
    normally runs a third arm. An arm whose fact leg read nothing is refused as a
    non-measurement, so that third run could only re-measure the same nothing —
    `_would_regress` says False and the check stops after the two arms it needs.
    """
    assert R._would_regress(_fact_arm(HEALTHY_FACT_COV, fact_metric=0.375),
                            _fact_arm(ZEROED_FACT_COV), ZERO_NOISE) is False, (
        "a zeroed fact leg told the checker a confirm arm was worth spending")

    arms: list = []
    _fact_check_env(monkeypatch, tmp_path)
    monkeypatch.setattr(R, "_run_arm", _arms_by_label(
        baseline=_fact_arm(HEALTHY_FACT_COV, fact_metric=0.375),
        current=_fact_arm(ZEROED_FACT_COV),
        on_call=arms.append))

    out = R._execute_blocking()
    assert out["status"] == "skipped", out
    assert "automod-check-confirm" not in arms, (
        f"the refusal still spent a confirm arm: {arms}")
    assert arms == ["automod-paired-lkg", "automod-check"], arms


def test_a_zeroed_fact_leg_on_the_confirm_arm_refuses_the_rollback(monkeypatch, tmp_path):
    """A fact leg that read nothing cannot CONFIRM a rollback either (#1250).

    The refusal on the two paired arms is not enough: the confirm arm is the one
    whose agreement turns a suspected regression into a rollback request, and it
    was gated on `corpus_ok is False or unanswered_doc_queries(confirm)` only.
    Neither sees this state — `corpus_ok` is the graph half of the store, and
    `unanswered_doc_queries` keys on `n_docs`, which a zeroed FACT leg leaves
    populated — so the arm reached `evaluate` and counted as the second
    independent observation of a regression whose stated cause was
    `fact_entity_recall_avg`.

    Here the DOCUMENT regression is real and its arm is scoreable, so the confirm
    run is genuinely consulted; only the confirm arm's fact leg read nothing (0
    facts on 20 queries against a corpus indexing 315,462 — the shape of the six
    arms on 2026-09-18). The check must refuse rather than roll back, and its
    reason must name the FACT leg: a skip reading "did not get an answer to every
    query" when every query got its documents sends the next reader looking at the
    retrieval daemon instead of the fact tree.
    """
    requested = _fact_check_env(monkeypatch, tmp_path)
    calls: list = []

    def run_by_label(_tree, label, _env):
        calls.append(label)
        if "lkg" in label:
            return _fact_arm(HEALTHY_FACT_COV, fact_metric=0.375)
        if "confirm" in label:
            # Documents answered, facts none: the paired arms' doc guard is green
            # on this artifact, which is exactly why it needs its own check.
            return _fact_arm(ZEROED_FACT_COV, fact_metric=0.0)
        # The candidate: an ARMED document metric regressed (doc_hit_rate 0.6 ->
        # 0.3 against a noise floor of stdev 0.0 — `base()` puts every armed
        # metric at 0.6) on a HEALTHY fact leg, so the paired refusal does not
        # fire and a confirm arm is legitimately spent.
        candidate = _fact_arm(HEALTHY_FACT_COV, fact_metric=0.375)
        candidate["overall"]["doc_hit_rate"] = 0.3
        return candidate

    monkeypatch.setattr(R, "_run_arm", run_by_label)

    out = R._execute_blocking()

    assert "automod-check-confirm" in calls, (
        f"precondition: the confirm arm never ran, so this proves nothing about it: {calls}")
    assert requested == [], (
        f"a confirm arm whose fact leg read nothing requested a rollback: {requested}")
    assert out["status"] == "skipped", (
        f"a confirm leg that read nothing became a verdict instead of a refusal: {out}")
    assert "cannot evaluate" in out["skipped"], out["skipped"]
    assert "fact leg" in out["skipped"], (
        "the refusal must name which leg died; the document leg answered every query "
        f"here: {out['skipped']}")


def test_a_fact_leg_that_returned_something_keeps_its_real_score():
    """A per-query zero is legitimate; only an all-zero leg is not (#1250).

    The healthy arm's own distribution is nineteen 10s and one 1 — zero queries
    at 0 — so the guard is total-across-queries. A run with some empty queries
    and a real total is a measurement and gets judged as one.
    """
    partial = dict(HEALTHY_FACT_COV, n_facts_total=3, empty_fact_queries=["q7", "q13"])
    assert R.empty_fact_leg(_fact_arm(partial)) is False
    assert R.empty_fact_leg(_fact_arm(HEALTHY_FACT_COV)) is False
    assert R.empty_fact_leg(_fact_arm(ZEROED_FACT_COV)) is True


def test_a_zeroed_fact_leg_over_an_empty_fact_tree_is_not_a_refusal():
    """The other half of the distinction: no facts to read IS a measurement.

    An arm over a corpus with no fact tree scores 0.0 legitimately — that is a
    fact about the corpus, and `corpus_ok` already refuses the empty-graph case.
    Refusing here would turn a real baseline into an unmeasurable one.
    """
    arm = _fact_arm(ZEROED_FACT_COV, corpus={"facts": 0, "entities": 0, "edges_active": 0})
    assert R.empty_fact_leg(arm) is False


def test_a_zeroed_fact_leg_without_a_known_corpus_is_not_a_refusal():
    """Unknown is not evidence of a non-empty tree.

    The corpus block can be missing from the arm's own artifact; with nothing to
    say the tree was populated, 0 facts is not enough to call the run a
    non-measurement — a guard that fires on silence would refuse every old
    baseline.
    """
    arm = _fact_arm(ZEROED_FACT_COV)
    arm.pop("corpus")
    assert R.empty_fact_leg(arm) is False


def test_the_fact_guard_reads_a_baseline_arm_written_by_older_code(tmp_path):
    """A baseline artifact predating the per-query counter still gets judged.

    The baseline arm is a file some other checkout wrote, so the guard may NOT
    require the `n_fact_reads_failed` keys #1250 adds to each record: it keys on
    the per-query `n_facts` that has been recorded all along. The six broken arms
    of 2026-09-18 are exactly these files — `n_facts: 0` on every query,
    `error: null`, a non-empty corpus block.
    """
    def rec(nfacts):
        return {"id": f"q{nfacts}", "error": None, "result_summary": {"n_facts": nfacts},
                "scoring": {"fact_entity_recall": 0.0}}

    blob = {"summary": {"overall": base(**{"fact_entity_recall_avg": 0.0})},
            "corpus_ok": True, "corpus": dict(CORPUS_WITH_FACTS),
            "records": [rec(0) for _ in range(4)]}
    (tmp_path / "automod-check-20260918-170645.json").write_text(json.dumps(blob))
    arm = R._load_run(tmp_path, "automod-check")
    assert arm is not None
    assert arm["fact_coverage"]["n_facts_total"] == 0
    assert arm["fact_coverage"]["n_fact_reads_failed_total"] == 0, (
        "an older artifact has no failure counter; reading its absence as "
        "failures would refuse every historical baseline")
    assert R.empty_fact_leg(arm) is True, (
        "the historical broken arm is still not recognised as a non-measurement")

    healthy = dict(blob, records=[rec(10) for _ in range(3)] + [rec(1)])
    (tmp_path / "automod-check-20260918-080311.json").write_text(json.dumps(healthy))
    ok = R._load_run(tmp_path, "automod-check")
    assert ok["fact_coverage"]["n_facts_total"] == 31
    assert R.empty_fact_leg(ok) is False


def test_evaluate_would_call_a_zeroed_fact_leg_a_rollback_reason():
    """Why the refusal has to happen before `evaluate`, not inside it.

    Pinned-corpus noise is stdev 0.0, so the tolerance is the MIN_SIGMA floor and
    0.375 -> 0.0 on an ARMED metric is a regression by any arithmetic. This is
    the false-rollback path the item names as the live exposure: an arm that read
    nothing becomes a rollback reason for an unrelated commit. The paired check
    must catch it upstream — `evaluate` is a comparator and has no idea whether
    the zero was read or measured.
    """
    assert "fact_entity_recall_avg" in R.FACT_LAYER_METRICS
    regressed, reasons, _ = R.evaluate(
        base(**{"fact_entity_recall_avg": 0.0}),
        base(**{"fact_entity_recall_avg": 0.375}), ZERO_NOISE)
    assert regressed and any("fact_entity_recall_avg" in r for r in reasons), reasons


def test_a_null_armed_metric_is_never_compared_and_never_crashes():
    """`evaluate` must survive the null clause 3 makes it read (#1250).

    Clause 3 changes what `run_eval` RECORDS: a run whose every fact leg was
    empty gets `fact_entity_recall_avg: null` where it used to get a scored 0.0.
    `evaluate` is downstream of that writer and does its comparison with
    `float(current[key])`, which on a null raises `TypeError: float() argument
    must be a string or a real number, not 'NoneType'`. In `_would_regress` the
    blanket handler swallowed that into `return False`; anywhere else it is an
    exception out of the check. Either way the check stops reporting WHY the leg
    was empty and starts reporting a crash — the same class of defect this item
    is filed under, moved from the fact read into the comparator.

    A null is treated exactly like an absent key: no verdict on that metric, and
    an explicit `not_measured` entry in the detail, so the ledger can never read
    the metric as compared-and-flat. A run that nulls one metric still gets its
    real verdict on the others.
    """
    null_current = base(**{"fact_entity_recall_avg": None})

    detail = R.evaluate(null_current, base(**{"fact_entity_recall_avg": 0.375}),
                        ZERO_NOISE)[2]
    assert detail["fact_entity_recall_avg"] == {
        "before": 0.375, "after": None, "armed": True, "not_measured": True}, detail
    assert R.evaluate(null_current, base(**{"fact_entity_recall_avg": 0.375}),
                      ZERO_NOISE)[0] is False, (
        "a metric that was not measured was reported as a regression")

    # The null is one metric's condition, not the run's: an ARMED document metric
    # that moved past the tolerance still fires with the fact metric null.
    regressed, reasons, _ = R.evaluate(
        dict(null_current, doc_hit_rate=0.3),
        base(**{"fact_entity_recall_avg": 0.375}), ZERO_NOISE)
    assert regressed and any("doc_hit_rate" in r for r in reasons), reasons

    # And a null on the BASELINE side, which is the arm written by the pinned
    # older commit — the direction that would otherwise read as an improvement
    # out of nowhere.
    assert R.evaluate(base(**{"fact_entity_recall_avg": 0.375}), null_current,
                      ZERO_NOISE)[0] is False
