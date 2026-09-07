"""Behavioural-regression detection for a landed self-modification.

The measured fact this is built on: five consecutive `eval/run_eval.py` runs
against an unchanged vault produced identical values for every quality metric
(stdev 0.0000). The eval contributes no noise. What *does* move is the vault
underneath it, which is why the comparison is a paired same-data A/B rather
than a check against a number recorded at the last promotion.
"""

from __future__ import annotations

import json

import pytest

from workers.sources import selfmod_regression as R


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

class _Item:
    payload: dict = {}


def _observing(**over):
    import time
    d = {"commit": "b" * 40, "parent": "a" * 40, "state": "observing",
         "landed_ts": time.time(), "changed_paths": ["app/x.py"]}
    d.update(over)
    return d


async def test_nothing_recent_to_check_is_a_noop(monkeypatch):
    import scripts.selfmod.state as S
    monkeypatch.setattr(S, "read_current", lambda: None)
    monkeypatch.setattr(S, "read_last_settled", lambda: None)
    out = await R.execute(_Item())
    assert "skipped" in out


async def test_a_settled_promotion_is_still_checked(monkeypatch, tmp_path):
    """The window is 15 minutes and this job runs hourly at best.

    Keying on `current.json` alone meant the subject was gone before the check
    ever ran — which is why this source has never produced a single run.
    """
    import scripts.selfmod.state as S
    monkeypatch.setattr(S, "read_current", lambda: None)
    monkeypatch.setattr(S, "read_last_settled", lambda: _observing())
    monkeypatch.setattr(S, "read_events", lambda **k: [])
    monkeypatch.setattr(S, "append_event", lambda *a, **k: None)
    monkeypatch.setattr(R, "NOISE_PATH", tmp_path / "absent.json")
    out = await R.execute(_Item())
    # Got past subject selection: it failed on the noise floor, not on
    # "nothing to check".
    assert "no measured noise floor" in out["skipped"]


async def test_the_baseline_is_the_parent_not_the_lkg(monkeypatch, tmp_path):
    """After settling, the LKG pointer IS the promoted commit.

    Comparing against it would check out the same code in both arms and be
    structurally incapable of finding anything.
    """
    import scripts.selfmod.state as S
    seen = {}
    noise = tmp_path / "noise.json"
    noise.write_text(json.dumps(ZERO_NOISE))
    monkeypatch.setattr(R, "NOISE_PATH", noise)
    monkeypatch.setattr(S, "read_current", lambda: _observing())
    monkeypatch.setattr(S, "read_events", lambda **k: [])
    monkeypatch.setattr(S, "append_event", lambda *a, **k: None)
    monkeypatch.setattr(S, "read_lkg", lambda: {"commit": "b" * 40})   # == promoted
    monkeypatch.setattr(R, "_run_eval_paired",
                        lambda commit: seen.setdefault("baseline", commit) and None)
    await R.execute(_Item())
    assert seen["baseline"] == "a" * 40, "must compare against the parent"


async def test_one_measurement_per_promotion(monkeypatch, tmp_path):
    """Both arms are full evals plus a worktree; the answer cannot change."""
    import scripts.selfmod.state as S
    monkeypatch.setattr(S, "read_current", lambda: _observing())
    monkeypatch.setattr(S, "read_last_settled", lambda: None)
    monkeypatch.setattr(S, "read_events", lambda **k: [
        {"event": "regression_check", "commit": "b" * 40}])
    out = await R.execute(_Item())
    assert "already checked" in out["skipped"]


async def test_an_empty_corpus_arm_cannot_evaluate(monkeypatch, tmp_path):
    """An empty graph is not a low score, it is a measurement that did not happen.

    And it does not look like one: with the graph deleted, mrr_doc, ndcg10 and
    doc_hit_rate come back identical to a real run.
    """
    import scripts.selfmod.state as S
    noise = tmp_path / "noise.json"
    noise.write_text(json.dumps(ZERO_NOISE))
    monkeypatch.setattr(R, "NOISE_PATH", noise)
    monkeypatch.setattr(S, "read_current", lambda: _observing())
    monkeypatch.setattr(S, "read_events", lambda **k: [])
    monkeypatch.setattr(S, "append_event", lambda *a, **k: None)
    monkeypatch.setattr(R, "_run_eval_paired", lambda commit: {
        "overall": base(), "corpus_ok": True, "corpus": {}})
    monkeypatch.setattr(R, "_run_eval", lambda label: {
        "overall": base(), "corpus_ok": False, "corpus": {"entities": 0}})
    out = await R.execute(_Item())
    assert "empty corpus" in out["skipped"]


async def test_a_regression_is_handed_to_the_guardian(monkeypatch, tmp_path):
    """Never rolled back inline: the rollback stops the process doing it."""
    import scripts.selfmod.state as S
    captured = {}
    noise = tmp_path / "noise.json"
    noise.write_text(json.dumps(ZERO_NOISE))
    monkeypatch.setattr(R, "NOISE_PATH", noise)
    monkeypatch.setattr(S, "read_current", lambda: _observing())
    monkeypatch.setattr(S, "read_events", lambda **k: [])
    monkeypatch.setattr(S, "append_event", lambda *a, **k: None)
    monkeypatch.setattr(S, "write_eval_last", lambda payload: None)
    monkeypatch.setattr(S, "request_rollback", lambda **kw: captured.update(kw) or kw)
    monkeypatch.setattr(R, "_run_eval_paired", lambda commit: {
        "overall": base(), "corpus_ok": True, "corpus": {}})
    monkeypatch.setattr(R, "_run_eval", lambda label: {
        "overall": base(entity_hit_rate=0.1), "corpus_ok": True, "corpus": {}})
    out = await R.execute(_Item())
    assert out["regressed"] is True
    assert captured["trigger"] == "regression"
    assert captured["target"] == "a" * 40 and captured["commit"] == "b" * 40
    assert out["graph_side_reasons"], "an entity-side drop must be reported as graph-side"


async def test_a_missing_noise_file_means_cannot_evaluate(monkeypatch, tmp_path):
    """Never 'no regression'. `eval/baselines/` is gitignored and can be absent."""
    import time
    import scripts.selfmod.state as S
    monkeypatch.setattr(S, "read_current", lambda: _observing())
    monkeypatch.setattr(S, "read_events", lambda **k: [])
    monkeypatch.setattr(S, "append_event", lambda *a, **k: None)
    monkeypatch.setattr(R, "NOISE_PATH", tmp_path / "absent.json")
    out = await R.execute(_Item())
    assert "no measured noise floor" in out["skipped"]


async def test_a_failed_paired_baseline_does_not_silently_pass(monkeypatch, tmp_path):
    import time
    import scripts.selfmod.state as S
    noise = tmp_path / "noise.json"
    noise.write_text(json.dumps(ZERO_NOISE))
    monkeypatch.setattr(R, "NOISE_PATH", noise)
    monkeypatch.setattr(S, "read_current", lambda: _observing())
    monkeypatch.setattr(S, "read_events", lambda **k: [])
    monkeypatch.setattr(S, "read_lkg", lambda: {"commit": "a" * 40})
    monkeypatch.setattr(S, "append_event", lambda *a, **k: None)
    monkeypatch.setattr(R, "_run_eval_paired", lambda commit: None)
    out = await R.execute(_Item())
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
