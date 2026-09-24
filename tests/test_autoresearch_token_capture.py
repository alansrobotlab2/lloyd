"""#1132: a direct-runner trial keeps the engine's usage block, end to end.

`_run_one_sync` used to read `choices[0].message.content` and drop the rest of
the response, so no ledger row could say what a trial spent and no two arms
could be compared at a matched budget. The endpoint is a stub `requests.post`;
nothing here reaches vLLM.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import requests

from scripts.autoresearch import bench_runner, run_round


class _Resp:
    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self._body


@pytest.fixture
def stub_engine(monkeypatch):
    """Answer every chat completion with the queued bodies, in order."""
    import prompt_builder

    posted: list[dict] = []
    replies: list = []

    def post(url, headers=None, json=None, timeout=None):
        posted.append(json)
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return _Resp(reply)

    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(prompt_builder, "build_system_prompt",
                        lambda overlay_dir=None: "system")
    monkeypatch.setattr(bench_runner, "_endpoint_for", lambda model: "http://stub:8096")
    monkeypatch.setattr(bench_runner, "_resolved_model_name", lambda model: "primary")
    return posted, replies


def _body(content="answer", usage=None):
    body = {"choices": [{"message": {"content": content}}]}
    if usage is not None:
        body["usage"] = usage
    return body


def _trial():
    return bench_runner._run_one_sync({"id": "bench_001", "prompt": "do it"},
                                      "BASELINE", Path("/nonexistent"), "primary", 30)


def test_a_successful_trial_carries_the_engines_three_counts(stub_engine):
    posted, replies = stub_engine
    replies.append(_body(usage={"prompt_tokens": 900, "completion_tokens": 120,
                                "total_tokens": 1020}))
    trace = _trial()
    assert trace["status"] == "success" and trace["final_text"] == "answer"
    assert (trace["prompt_tokens"], trace["completion_tokens"],
            trace["total_tokens"]) == (900, 120, 1020)
    assert len(posted) == 1


def test_counts_are_summed_over_every_call_a_trial_makes():
    """A multi-call arm reports the trial's spend, not its last call's."""
    trace = {key: None for key in bench_runner.TOKEN_KEYS}
    bench_runner.add_usage(trace, {"prompt_tokens": 900, "completion_tokens": 120,
                                   "total_tokens": 1020})
    bench_runner.add_usage(trace, {"prompt_tokens": 1100, "completion_tokens": 80,
                                   "total_tokens": 1180})
    assert (trace["prompt_tokens"], trace["completion_tokens"],
            trace["total_tokens"]) == (2000, 200, 2200)


def test_an_engine_that_reports_no_usage_leaves_the_counts_null(stub_engine):
    """Absent is not zero: a 0 would read as a trial that cost nothing."""
    _, replies = stub_engine
    replies.append(_body())
    trace = _trial()
    assert trace["status"] == "success"
    assert all(trace[key] is None for key in bench_runner.TOKEN_KEYS)


def test_a_failed_call_records_no_counts(stub_engine):
    _, replies = stub_engine
    replies.append(requests.Timeout("slow"))
    trace = _trial()
    assert trace["status"] == "timeout"
    assert all(trace[key] is None for key in bench_runner.TOKEN_KEYS)


def test_the_per_trial_ledger_row_carries_the_three_counts(stub_engine):
    """Built by CALLING the round's row writer on the runner's own trace, so a
    writer that stopped emitting the keys fails here, not in a later census."""
    _, replies = stub_engine
    replies.append(_body(usage={"prompt_tokens": 900, "completion_tokens": 120,
                                "total_tokens": 1020}))
    trace = _trial()
    score = {"composite_score": 0.8, "objective_score": 1.0, "rubric_overall": 0.6,
             "safety_critical": False, "safety_passed": True}
    row = run_round.trial_ledger_row("R_20260924_000000", trace, score)
    assert "event" not in row, "the per-trial row is the one with no event field"
    assert (row["prompt_tokens"], row["completion_tokens"],
            row["total_tokens"]) == (900, 120, 1020)


def test_the_ondemand_writer_carries_the_same_keys_null_on_an_sdk_trace():
    """The sdk trace's `usage` has harness semantics (input_tokens is the PEAK
    prompt, not a sum), so it is not restated under the engine's names."""
    from scripts.autoresearch import bench_runner_sdk

    trace = {"variant_id": "BASELINE", "task_id": "bench_010", "status": "success",
             "usage": {"input_tokens": 5000, "output_tokens": 300}}
    row = bench_runner_sdk.ledger_row_for(trace, None, "R_20260924_000000")
    assert all(row[key] is None for key in bench_runner.TOKEN_KEYS)
