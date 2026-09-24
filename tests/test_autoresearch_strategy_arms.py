"""#1132 clauses 4-5: strategy arms at one ceiling, and the per-arm report.

The engine is a stub `requests.post`; nothing here reaches vLLM.
"""
from __future__ import annotations

import pytest
import requests

from scripts.autoresearch import bench_runner, strategy_arms as sa


class _Resp:
    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self._body


@pytest.fixture
def engine(monkeypatch):
    """Queue replies; record every request body and the url it went to."""
    import prompt_builder

    posted: list[tuple[str, dict]] = []
    replies: list = []

    def post(url, headers=None, json=None, timeout=None):
        posted.append((url, json))
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return _Resp(reply)

    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(prompt_builder, "build_system_prompt", lambda overlay_dir=None: "system")
    monkeypatch.setattr(bench_runner, "_endpoint_for",
                        lambda model: {"primary": "http://p:8096",
                                       "secondary": "http://s:8091"}[model])
    monkeypatch.setattr(bench_runner, "_resolved_model_name",
                        lambda model: {"primary": "Qwen3.8-Flash-Next",
                                       "secondary": "Qwen3.6-35B"}[model])
    return posted, replies


def _body(content, completion, prompt=100):
    return {"choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                      "total_tokens": prompt + completion}}


TASK = {"id": "bench_x", "prompt": "do the thing"}


def test_advise_runs_draft_adviser_revise_and_records_each_call(engine):
    posted, replies = engine
    replies += [_body("draft", 200), _body("fix step 2", 50), _body("final", 250)]
    trace = sa.run_arm_trial(TASK, "advise", ceiling=1000, adviser_model="secondary")

    assert trace["status"] == "success" and trace["final_text"] == "final"
    assert [c["role"] for c in trace["calls"]] == ["draft", "adviser", "revise"]
    # which engine served each role
    assert [c["served_by"] for c in trace["calls"]] == [
        "Qwen3.8-Flash-Next", "Qwen3.6-35B", "Qwen3.8-Flash-Next"]
    assert [url for url, _ in posted] == [
        "http://p:8096/v1/chat/completions", "http://s:8091/v1/chat/completions",
        "http://p:8096/v1/chat/completions"]
    # tokens per call, and the trial's sum
    assert [c["completion_tokens"] for c in trace["calls"]] == [200, 50, 250]
    assert trace["completion_tokens"] == 500 and trace["total_tokens"] == 800
    # each call may spend only the headroom left under the ceiling
    assert [body["max_tokens"] for _, body in posted] == [1000, 800, 750]
    assert trace["stopped_at_ceiling"] is False


def test_no_further_call_once_measured_completion_reaches_the_ceiling(engine):
    posted, replies = engine
    replies += [_body("long draft", 400)]  # the draft alone spends the whole ceiling
    trace = sa.run_arm_trial(TASK, "advise", ceiling=400)

    assert len(posted) == 1
    assert trace["stopped_at_ceiling"] is True
    assert trace["skipped_roles"] == ["adviser"]
    assert trace["final_text"] == "long draft"
    assert trace["completion_tokens"] == 400


def test_ceiling_reached_after_the_adviser_stops_before_the_revision(engine):
    posted, replies = engine
    replies += [_body("draft", 300), _body("advice", 120)]  # 420 >= 400
    trace = sa.run_arm_trial(TASK, "advise", ceiling=400)
    assert len(posted) == 2
    assert trace["skipped_roles"] == ["revise"]
    assert trace["final_text"] == "draft"


def test_a_call_with_no_usage_stops_the_arm_rather_than_spend_blind(engine):
    posted, replies = engine
    replies += [{"choices": [{"message": {"content": "draft"}}]}]
    trace = sa.run_arm_trial(TASK, "advise", ceiling=5000)
    assert len(posted) == 1 and trace["stopped_at_ceiling"] is True
    assert trace["calls"][0]["completion_tokens"] is None


def test_execute_is_one_call_given_the_whole_ceiling(engine):
    posted, replies = engine
    replies += [_body("answer", 700)]
    trace = sa.run_arm_trial(TASK, "execute", ceiling=3000)
    assert len(posted) == 1 and posted[0][1]["max_tokens"] == 3000
    assert trace["calls"][0]["role"] == "execute"
    assert trace["total_tokens"] == 800


def test_an_engine_error_is_an_error_trace_with_the_calls_so_far(engine):
    posted, replies = engine
    replies += [_body("draft", 100), requests.ConnectionError("down")]
    trace = sa.run_arm_trial(TASK, "advise", ceiling=1000)
    assert trace["status"] == "error" and len(trace["calls"]) == 1


def test_the_ceiling_is_required_and_positive():
    with pytest.raises(ValueError):
        sa.run_arm_trial(TASK, "advise", ceiling=0)
    with pytest.raises(ValueError):
        sa.run_arm_trial(TASK, "dream", ceiling=100)


# --- report -------------------------------------------------------------------


def _row(arm, composite, objective, total, completion):
    return {"arm": arm, "composite_score": composite, "objective_score": objective,
            "total_tokens": total, "completion_tokens": completion}


def test_report_row_per_arm_with_the_four_numbers_and_winners():
    rows = [
        _row("execute", 0.6, 1.0, 1200, 1000),
        _row("execute", 0.4, 0.5, 1200, 1000),
        _row("advise", 0.9, 1.0, 2000, 980),
        _row("advise", 0.8, 1.0, 2000, 1020),
    ]
    rep = sa.arm_report(rows, ceiling=1000)
    ex, ad = rep["arms"]["execute"], rep["arms"]["advise"]
    assert ex["mean_composite"] == pytest.approx(0.5)
    assert ex["perfect_run_rate"] == 0.5
    assert ex["mean_total_tokens"] == 1200
    assert ex["expected_tokens_to_perfect"] == pytest.approx(2400)
    assert ad["perfect_run_rate"] == 1.0 and ad["expected_tokens_to_perfect"] == 2000
    assert rep["winners"] == {"mean_composite": "advise", "perfect_run_rate": "advise",
                              "expected_tokens_to_perfect": "advise"}
    assert rep["all_within_ceiling"] is True

    text = sa.format_report(rep)
    assert "execute" in text and "advise" in text
    assert "winner, expected tokens to perfect:  advise" in text
    assert "within ±5% of the ceiling: yes" in text


def test_a_zero_perfect_rate_prints_not_computable_not_zero():
    rows = [_row("execute", 0.5, 0.5, 1000, 1000), _row("advise", 0.7, 1.0, 1500, 1000)]
    rep = sa.arm_report(rows, ceiling=1000)
    assert rep["arms"]["execute"]["expected_tokens_to_perfect"] is None
    line = next(l for l in sa.format_report(rep).splitlines() if l.startswith("execute"))
    assert sa.NOT_COMPUTABLE in line
    assert rep["winners"]["expected_tokens_to_perfect"] == "advise"


def test_an_arm_off_the_ceiling_by_more_than_five_percent_is_flagged():
    rows = [_row("execute", 0.5, 1.0, 500, 300), _row("advise", 0.5, 1.0, 1500, 1040)]
    rep = sa.arm_report(rows, ceiling=1000)
    assert rep["arms"]["execute"]["within_ceiling"] is False
    assert rep["arms"]["advise"]["within_ceiling"] is True
    assert rep["all_within_ceiling"] is False
    assert "within ±5% of the ceiling: no" in sa.format_report(rep)


def test_trial_row_carries_the_judge_and_token_fields(engine):
    posted, replies = engine
    replies += [_body("answer", 100)]
    trace = sa.run_arm_trial(TASK, "execute", ceiling=500)
    row = sa.trial_row(trace, {"composite_score": 0.7, "objective_score": 1.0,
                               "rubric_overall": 0.4})
    assert row["arm"] == "execute" and row["completion_tokens"] == 100
    assert row["calls"][0]["served_by"] == "Qwen3.8-Flash-Next"
    rep = sa.arm_report([row], ceiling=500)
    assert rep["arms"]["execute"]["perfect_run_rate"] == 1.0
