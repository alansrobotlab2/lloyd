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
    from app import prompt_builder

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


# ── #2019: the re-prefill cost of a trial, and the key that joins it to a variant ──

def _usage_db(tmp_path, rows):
    """A usage.db in the live schema, holding `rows` of (session, input, cache_read)."""
    import sqlite3

    db = tmp_path / "usage.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE usage (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, "
                 "session_id TEXT, input_tokens INTEGER, cache_read INTEGER, "
                 "reprefill_tokens INTEGER)")
    conn.executemany("INSERT INTO usage(ts, session_id, input_tokens, cache_read, "
                     "reprefill_tokens) VALUES ('2026-10-01T05:32:24', ?, ?, ?, 0)", rows)
    conn.commit()
    conn.close()
    return db


def _sdk_trace(variant, session, task="bench_010"):
    return {"variant_id": variant, "task_id": task, "status": "success", "harness": "sdk",
            "session_id": session, "tool_calls": [], "denied_calls": []}


_SCORE = {"composite_score": 1.0, "objective_score": 1.0, "rubric_overall": 1.0}


def test_a_trial_row_carries_its_reprefill_cost_summed_over_its_recorded_session(
        tmp_path, monkeypatch):
    """Clause 1. The cost is input_tokens − cache_read over every usage row the
    trial's recorded session wrote — read from the store's own counts, not from
    `reprefill_tokens`, which is 0 on these rows even when 42.5k were re-prefilled."""
    from app import usage_store
    from scripts.autoresearch import bench_runner_sdk, cost

    db = _usage_db(tmp_path, [
        ("20261001_053224_bench_7a06", 42541, 22400),
        ("20261001_053224_bench_7a06", 43000, 42541),      # the turn's second call
        ("20261001_053301_bench_91c2", 42541, 0),           # a cold trial: all re-prefilled
        ("20261001_060000_chat_aaaa", 9000, 100)])           # someone else's session
    monkeypatch.setattr(usage_store, "DB_PATH", db)

    row = run_round.trial_ledger_row(
        "R_1", _sdk_trace("V_warm", "20261001_053224_bench_7a06"), _SCORE)
    assert row[cost.SESSION_FIELD] == "20261001_053224_bench_7a06"
    assert row[cost.REPREFILL_FIELD] == (42541 - 22400) + (43000 - 42541) == 20600
    cold = run_round.trial_ledger_row(
        "R_1", _sdk_trace("V_cold", "20261001_053301_bench_91c2"), _SCORE)
    assert cold[cost.REPREFILL_FIELD] == 42541

    # Both writers, one helper: the on-demand row says the same thing.
    sdk_row = bench_runner_sdk.ledger_row_for(
        _sdk_trace("V_warm", "20261001_053224_bench_7a06"), _SCORE, "R_1")
    assert {k: sdk_row[k] for k in cost.COST_LEDGER_KEYS} == \
        {k: row[k] for k in cost.COST_LEDGER_KEYS}


def test_a_missing_count_is_null_and_never_zero(tmp_path, monkeypatch):
    """Clause 1's other half: an unmeasured trial is not a free one."""
    from app import usage_store
    from scripts.autoresearch import cost

    db = _usage_db(tmp_path, [("s_half", 42541, None), ("s_half", 100, 50),
                              ("s_noinput", None, 10)])
    monkeypatch.setattr(usage_store, "DB_PATH", db)
    for session in ("s_half", "s_noinput", "s_never_recorded"):
        row = run_round.trial_ledger_row("R_1", _sdk_trace("V", session), _SCORE)
        assert row[cost.REPREFILL_FIELD] is None, session
        assert row[cost.SESSION_FIELD] == session
    assert cost.reprefill_cost([]) is None and cost.reprefill_cost(None) is None
    assert cost.reprefill_cost([{"input_tokens": 5, "cache_read": 5}]) == 0, (
        "a measured zero is a zero")
    assert cost.reprefill_cost([{"input_tokens": True, "cache_read": 0}]) is None

    # A direct trace has no recorded session: both keys are None, not a lookup of "".
    direct = run_round.trial_ledger_row("R_1", {
        "variant_id": "V", "task_id": "t", "status": "success", "harness": "direct",
        "session_id": "s_half"}, _SCORE)
    assert direct[cost.SESSION_FIELD] is None and direct[cost.REPREFILL_FIELD] is None

    # And an unreadable store costs the field, never the row.
    monkeypatch.setattr(usage_store, "DB_PATH", tmp_path / "absent" / "usage.db")
    row = run_round.trial_ledger_row("R_1", _sdk_trace("V", "s_half"), _SCORE)
    assert row[cost.REPREFILL_FIELD] is None


def test_two_variants_under_different_cache_profiles_get_different_cost_means(
        tmp_path, monkeypatch):
    """Clause 2. usage.db keys a bench row by the RECORDED session id, not by
    `bench_<variant>_<task>_<hex>`; the id carried on the row is what attributes a
    usage row to the variant that ran the trial."""
    from app import usage_store
    from scripts.autoresearch import cost

    db = _usage_db(tmp_path, [("20261001_053224_bench_7a06", 42541, 40000),
                              ("20261001_053301_bench_91c2", 42541, 0)])
    monkeypatch.setattr(usage_store, "DB_PATH", db)
    rows = [run_round.trial_ledger_row(
                "R_1", _sdk_trace("V_warm", "20261001_053224_bench_7a06"), _SCORE),
            run_round.trial_ledger_row(
                "R_1", _sdk_trace("V_cold", "20261001_053301_bench_91c2"), _SCORE)]
    assert not any(r[cost.SESSION_FIELD].startswith("bench_V_") for r in rows)
    rec = cost.round_cost_records(rows)
    assert rec["V_warm"]["cost_mean_valid"] == 2541.0
    assert rec["V_cold"]["cost_mean_valid"] == 42541.0
    assert rec["V_warm"]["cost_mean_valid"] != rec["V_cold"]["cost_mean_valid"]
