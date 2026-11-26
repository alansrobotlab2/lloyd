"""#4386: one dispatch-side wall-clock span per tool batch (#2275 re-opened).

`tool_ms_total` is a SUM of per-call spans, so several calls issued together each
book their own span and a class that overlaps them reads `engine + tool > 1.0`
(autocode 0.356 + 0.714 = 1.070 on 2026-11-19). This pins the second figure that
cannot do that: `_dispatch_batch` measures ONE span per batch, first dispatch to
last result committed to history, in the same `time.perf_counter()` frame as the
per-call spans; `run_query` hands the turn's spans out on its `result` event
(`tool_batch_ms`), and `TurnTelemetry` folds them into `tool_batch_ms_total`.

Driven through the REAL `run_query` with the scripted engine and pool
(`app/harness/tests/_replay.py`), never by reading the loop's source.
"""
from __future__ import annotations

from app.harness.options import RunOptions
from app.harness.tests import _replay as R
from app.turn_usage import TurnTelemetry


def _options(**kw) -> RunOptions:
    return RunOptions(model="m", **kw)


def _fold(events: list[dict]) -> dict:
    tel = TurnTelemetry()
    for evt in events:
        tel.note(evt)
    return tel.row()


async def test_a_three_call_batch_is_one_batch_span_and_three_call_spans(monkeypatch):
    calls = [R.tool_call(f"c{i}", "Bash", command=f"echo {i}") for i in (1, 2, 3)]
    engine = R.ReplayEngine([R.Step(tool_calls=calls), R.Step(text="done")])
    pool = R.ReplayPool(delay_by_call_id={"c1": 0.05, "c2": 0.05, "c3": 0.05})
    R.install(monkeypatch, engine, pool)

    out = await R.drive(_options())

    per_call = [e["duration_ms"] for e in R.of_type(out, "tool_result")]
    assert len(per_call) == 3, per_call
    (result,) = R.of_type(out, "result")
    spans = result["tool_batch_ms"]
    assert len(spans) == 1, f"one batch must be one span, got {spans}"
    # Same clock, and the batch contains its calls: sequential here, so its one
    # span covers all three call spans (it opens before the first dispatch and
    # closes after the last result is in history).
    assert spans[0] >= sum(per_call) - 3, (spans, per_call)
    assert spans[0] >= max(per_call)
    row = _fold(out)
    assert row["tool_batch_ms_total"] == spans[0]
    assert row["tool_ms_total"] == sum(per_call)


async def test_overlapping_calls_book_one_window_where_tool_ms_books_two(monkeypatch):
    """Two read-only calls that overlap inside one 1,000 ms window: the batch
    figure is the window, the per-call figure is the larger sum of both spans."""
    calls = [R.tool_call("r1", "Read", file_path="/a"),
             R.tool_call("r2", "Read", file_path="/b")]
    engine = R.ReplayEngine([R.Step(tool_calls=calls), R.Step(text="done")])
    pool = R.ReplayPool(delay_by_call_id={"r1": 0.6, "r2": 0.6})
    R.install(monkeypatch, engine, pool)

    out = await R.drive(_options(parallel_tool_calls_enabled=True))

    assert pool.max_inflight == 2, "the fixture must actually overlap the calls"
    row = _fold(out)
    per_call = [e["duration_ms"] for e in R.of_type(out, "tool_result")]
    assert row["tool_batch_ms_total"] <= 1000, row
    assert row["tool_ms_total"] == sum(per_call) > row["tool_batch_ms_total"], (
        row, per_call)
    assert row["tool_ms_total"] >= 1200 - 20, row


async def test_each_batch_of_a_turn_is_its_own_span(monkeypatch):
    engine = R.ReplayEngine([
        R.Step(tool_calls=[R.tool_call("a", "Bash", command="ls")]),
        R.Step(tool_calls=[R.tool_call("b", "Bash", command="pwd"),
                           R.tool_call("c", "Bash", command="id")]),
        R.Step(text="done"),
    ])
    R.install(monkeypatch, engine, R.ReplayPool())

    out = await R.drive(_options())

    (result,) = R.of_type(out, "result")
    assert len(result["tool_batch_ms"]) == 2, result["tool_batch_ms"]
    assert _fold(out)["tool_batch_ms_total"] == sum(result["tool_batch_ms"])


async def test_a_turn_with_no_timed_batch_books_null_not_zero(monkeypatch):
    """No batch span, no figure: NULL, exactly as the engine-ms half books a turn
    with no stamped iteration — a 0 would read as tools that answered instantly."""
    R.install(monkeypatch, R.ReplayEngine([R.Step(text="just prose")]), R.ReplayPool())
    out = await R.drive(_options())
    (result,) = R.of_type(out, "result")
    assert result["tool_batch_ms"] == []
    assert _fold(out)["tool_batch_ms_total"] is None
