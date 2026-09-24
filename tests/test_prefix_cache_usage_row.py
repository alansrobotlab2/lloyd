"""The turn-level usage row cannot report more cached than prompt tokens.

Backlog #859 (umbrella of #604 + #618), closing instrument for #520.

Two defects, one shared measurement:

* ``_accumulate_iteration_usage`` folded per-iteration usage into the turn
  total with mismatched units — ``input_tokens`` became the **peak** across
  iterations while ``cache_read`` was **summed** across them. Dividing one by
  the other is what every consumer does to show a cached fraction, so a
  5-iteration turn reported 5-iteration cached tokens over 1-iteration prompt
  tokens: 665 persisted ``assistant.stats`` rows on 2026-09-11 read over 100%
  cached, the worst at 8,600% (``cache_read`` 17,564,800 against
  ``input_tokens`` 202,467).
* No committed script reproduced #520's per-iteration table, so every verdict
  so far came from an ad-hoc heredoc that blended engine boots. The same query
  over a day straddling the ``2677dea`` landing returned 76-79% and over the
  next day 84-90%, and the difference was attributed to a standing defect
  (#603) rather than to the restart it was.

The fix is a unit decision, made once, in ``_accumulate_iteration_usage``:
``input_tokens`` keeps meaning the peak single prompt (every consumer in the
tree already reads it that way — ``_maybe_finalize`` refuses to fold the
finalizer's prompt for exactly that reason), ``cache_read`` becomes
peak-consistent so the pair is commensurable, and the summed figures ride
alongside under their own names.

The turn-level row is identified by the **absence** of an ``iteration`` key —
20,179 of 41,287 persisted rows on 2026-09-11 — not by ``iteration: 0``, which
is what #604 and #618 both filtered on and why each read zero offenders and
reported the store fixed.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.harness import tool_search_cache          # noqa: E402
from app.harness.loop import (                     # noqa: E402
    _accumulate_iteration_usage, run_query,
)
from app.harness.options import RunOptions         # noqa: E402
from app.turn_usage import turn_usage_row          # noqa: E402
import usage_store                                 # noqa: E402


def _script_module():
    """Load the committed measurement script by path (no package in scripts/)."""
    path = ROOT / "scripts" / "prefix_cache_iteration_table.py"
    spec = importlib.util.spec_from_file_location(
        "prefix_cache_iteration_table", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["prefix_cache_iteration_table"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# The iteration usages one long Bash-heavy turn actually produces.
#
# vLLM reports `cached_tokens <= prompt_tokens` for any single request, so a
# per-iteration row is always self-consistent; the corruption appeared only
# where iterations were folded. Prompt grows monotonically because tool results
# only append, and cache climbs with it — which is what made a SUM of cache
# outrun a MAX of prompt.
# ---------------------------------------------------------------------------
ITERATIONS: list[dict[str, int]] = [
    {"input_tokens": 10_000, "output_tokens": 220, "cache_read": 0,
     "cache_create": 10_000},
    {"input_tokens": 60_000, "output_tokens": 310, "cache_read": 55_000,
     "cache_create": 5_000},
    {"input_tokens": 120_000, "output_tokens": 140, "cache_read": 118_000,
     "cache_create": 2_000},
]
# The shape the broken fold produced: 173,000 cached over a 120,000 peak.
BROKEN_CACHE_SUM = sum(u["cache_read"] for u in ITERATIONS)


def _fold() -> dict[str, int]:
    total: dict[str, int] = {}
    for usage in ITERATIONS:
        total = _accumulate_iteration_usage(total, usage)
    return total


# ── clause 1 ────────────────────────────────────────────────────────────────
def test_fold_of_several_iterations_keeps_cache_under_the_prompt():
    """The turn row the fold emits cannot exceed 100% cached."""
    row = _fold()

    assert len(ITERATIONS) >= 2
    # Would have been 173,000 against a 120,000 peak — 144% — before #859.
    assert row["cache_read"] <= row["input_tokens"], row
    assert row["cache_read"] != BROKEN_CACHE_SUM, (
        "cache_read is still the cross-iteration SUM while input_tokens is "
        "the PEAK: the row is commensurable again only if both are peaks")


def test_fold_is_the_peak_prompt_and_the_peak_cache():
    """Both halves of the ratio are peak-of-the-same-kind figures."""
    row = _fold()
    assert row["input_tokens"] == 120_000
    assert row["cache_read"] == 118_000
    # Writes accumulate: each iteration caches its own new suffix, and nothing
    # divides by this, so it stays a SUM by design.
    assert row["cache_create"] == 17_000
    assert row["output_tokens"] == 670


def test_fold_clamps_a_row_the_engine_could_not_have_meant():
    """A cached figure above its own prompt is clamped, not published.

    The engine cannot report more cached than prompted for one request, so a
    row that does (a parser that grabbed the wrong field, a future engine)
    would otherwise land in usage.db and read as a 900% hit rate. Clamping
    bounds the published row; it does not hide the reading, which is still
    visible per iteration in the session JSON.
    """
    total = _accumulate_iteration_usage({}, {
        "input_tokens": 1_000, "output_tokens": 10, "cache_read": 1_000})
    total = _accumulate_iteration_usage(total, {
        "input_tokens": 1_200, "output_tokens": 10, "cache_read": 9_000})
    assert total["cache_read"] <= total["input_tokens"], total
    assert total["cache_read"] == 1_200, total


def test_fold_survives_an_iteration_that_reported_nothing():
    """A usage-less pass must not invent sums or divide by zero downstream."""
    total = _accumulate_iteration_usage({}, ITERATIONS[0])
    total = _accumulate_iteration_usage(total, {})
    total = _accumulate_iteration_usage(total, {"input_tokens": 0})
    assert total["prompt_tokens_sum"] == 10_000, total
    assert total["cache_read_sum"] == 0, total


# ── clause 2 ────────────────────────────────────────────────────────────────
def test_turn_row_carries_the_summed_prompt_alongside_the_peak():
    """Both units are on the row, each named for its unit.

    The peak pair answers "how big was the prompt and how much of it was
    cached"; the sum pair answers "what did the turn cost and how much of that
    was cache". Neither can exceed 100% cached, which is the whole contract: a
    hit ratio computed from this row cannot be over 100% whichever pair you
    divide.
    """
    row = _fold()

    assert row["prompt_tokens_sum"] == 190_000, row
    assert row["cache_read_sum"] == 173_000, row
    assert row["cache_read_sum"] <= row["prompt_tokens_sum"]
    assert row["cache_read"] <= row["input_tokens"]
    assert row["input_tokens"] == 120_000          # peak, unchanged meaning
    assert row["prompt_tokens_sum"] >= row["input_tokens"]


def test_usage_row_mapping_bounds_both_pairs_on_every_platform_shape():
    """`turn_usage_row` — the mapper both writers use — keeps the row bounded.

    ``app/routers/messages.py`` and ``app/run_recorder.py`` both build the
    persisted row through it, so a missing sum (a turn from before this
    change, or a finalizer-only turn) falls back to the peak rather than to a
    zero that would read as a cold cache.
    """
    row = turn_usage_row(_fold())

    assert row["input_tokens"] == 120_000
    assert row["cache_read"] == 118_000
    assert row["cache_read_sum"] <= row["prompt_tokens_sum"]
    assert row["prompt_tokens_sum"] >= row["input_tokens"]

    legacy = turn_usage_row({"input_tokens": 500, "cache_read": 480})
    assert legacy["prompt_tokens_sum"] == 500
    assert legacy["cache_read_sum"] == 480
    assert legacy["cache_read_sum"] <= legacy["prompt_tokens_sum"]

    bad = turn_usage_row({"input_tokens": 500, "cache_read": 900,
                          "prompt_tokens_sum": 600, "cache_read_sum": 4_000})
    assert bad["cache_read"] <= bad["input_tokens"], bad
    assert bad["cache_read_sum"] <= bad["prompt_tokens_sum"], bad


# ── the seam: loop → result event → turn row → usage.db → dashboard ─────────
class _FakePool:
    @property
    def discovered(self):
        return [("lloyd-mc", [{
            "name": "Bash",
            "description": "shell",
            "inputSchema": {"type": "object", "properties": {}},
        }])]

    async def call_tool(self, name, args, *, session_id="", **_kw):
        return {"content": "FAKE_RESULT", "is_error": False}


class _Script:
    """Streams a scripted turn and reports vLLM-shaped usage per request."""

    def __init__(self, turns: list[tuple[str, list[dict[str, Any]]]]):
        self.turns = turns
        self.request = 0

    def __call__(self, **_kwargs):
        idx = self.request
        self.request += 1
        return self._gen(idx, *self.turns[idx])

    async def _gen(self, idx: int, text: str, tool_calls: list[dict]):
        usage = ITERATIONS[idx]
        if text:
            yield {"choices": [{"delta": {"content": text}}]}
        for i, tc in enumerate(tool_calls):
            yield {"choices": [{"delta": {"tool_calls": [{
                "index": i, "id": tc["id"], "type": "function",
                "function": {"name": tc["name"],
                             "arguments": json.dumps(tc.get("arguments") or {})},
            }]}}]}
        yield {"choices": [{"delta": {},
                            "finish_reason": "tool_calls" if tool_calls else "stop"}]}
        # vLLM's shape on the wire: the cache counters are NESTED, and the
        # int-only merge loop in `_merge_usage` skips the dict that holds them.
        yield {"choices": [], "usage": {
            "prompt_tokens": usage["input_tokens"],
            "completion_tokens": usage["output_tokens"],
            "prompt_tokens_details": {
                "cached_tokens": usage["cache_read"],
                "created_cache_tokens": usage["cache_create"],
            },
        }}


@pytest.fixture(autouse=True)
def _reset_tool_search_cache():
    asyncio.run(tool_search_cache.clear())
    yield
    asyncio.run(tool_search_cache.clear())


async def _drain(session_id: str) -> list[dict]:
    opts = RunOptions(model="primary", session_id=session_id,
                      tool_search_enabled=False)
    return [e async for e in run_query(
        [{"role": "user", "content": "go"}], opts)]


def _run_turn(monkeypatch, session_id: str) -> dict:
    async def _build_pool(_options):
        return _FakePool()
    monkeypatch.setattr("app.harness.loop._build_pool", _build_pool)
    monkeypatch.setattr("app.harness.loop.stream_chat", _Script([
        ("a", [{"id": "c1", "name": "Bash", "arguments": {}}]),
        ("b", [{"id": "c2", "name": "Bash", "arguments": {}}]),
        ("done", []),
    ]))
    events = asyncio.run(_drain(session_id))
    result = next(e for e in events if e["type"] == "result")
    return result.get("usage") or {}


def test_a_real_three_iteration_turn_reports_a_bounded_row(monkeypatch):
    """The row the harness actually emits, not the fold called by hand.

    Crosses the seam the graph cannot see: `run_query` folds each iteration's
    nested `prompt_tokens_details` and publishes the aggregate on the final
    `result` event, which is the only thing the two writers read.
    """
    usage = _run_turn(monkeypatch, "prefix-cache-row-test")

    assert usage["input_tokens"] == 120_000, usage
    assert usage["cache_read"] <= usage["input_tokens"], usage
    assert usage["prompt_tokens_sum"] == 190_000, usage
    assert usage["cache_read_sum"] == 173_000, usage


def test_per_iteration_rows_keep_their_own_exact_usage(monkeypatch):
    """Only the aggregate changed units; per-iteration stats stay per-request.

    Those rows are what #520's per-iteration table is built from, and the
    script is worthless if the fold also rewrote them.
    """
    async def _build_pool(_options):
        return _FakePool()
    monkeypatch.setattr("app.harness.loop._build_pool", _build_pool)
    monkeypatch.setattr("app.harness.loop.stream_chat", _Script([
        ("a", [{"id": "c1", "name": "Bash", "arguments": {}}]),
        ("b", [{"id": "c2", "name": "Bash", "arguments": {}}]),
        ("done", []),
    ]))
    events = asyncio.run(_drain("per-iteration-shape"))
    assistants = [e for e in events if e["type"] == "assistant_message"]

    assert [e["iteration"] for e in assistants] == [1, 2, 3], assistants
    assert [e["usage"]["input_tokens"] for e in assistants] == [
        10_000, 60_000, 120_000]
    assert [e["usage"]["cache_read"] for e in assistants] == [
        0, 55_000, 118_000]
    for e in assistants:
        assert e["usage"]["cache_read"] <= e["usage"]["input_tokens"]


# ── clauses 3 + 4 ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("platform", ["worker", "mission-control"])
def test_a_real_writer_persists_a_row_the_dashboard_cannot_over_100(monkeypatch,
                                                                    tmp_path,
                                                                    platform):
    """Real harness events → the recorder writer → usage.db → `dashboard._usage()`.

    Three things this had to be to count as evidence:

    * **The writer runs.** `app/run_recorder.record_events` is the writer that
      worker and autonomy traffic goes through, and it calls
      `turn_usage_row` and `record_usage` itself. An earlier version of this
      test called `record_usage` with values computed in the test body, which
      asserts the test's own arithmetic and would have passed against the
      broken writer.
    * **The dashboard seam is crossed, not described.** `_usage()` does
      `import usage_store` inside the function, which is why the code graph
      reports zero inbound edges to `summary()`; calling it is the only way to
      see the figure the panel serves.
    * **The platform parameter changes something.** `usage.db` has no platform
      column (recorded as a finding on #859), so the platform-bounded half of
      clause 4 is graded where platform IS persisted — the session's own turn
      row — and each parameter writes and reads back its own platform.
    """
    from app.run_recorder import record_events
    from app.sessions_io import create_session
    import app.routers.dashboard as dashboard
    import app.sessions_io as sessions_io

    monkeypatch.setattr(usage_store, "DB_PATH", tmp_path / f"{platform}.db")
    sid = f"{platform}-session-1"

    async def _build_pool(_options):
        return _FakePool()
    monkeypatch.setattr("app.harness.loop._build_pool", _build_pool)
    monkeypatch.setattr("app.harness.loop.stream_chat", _Script([
        ("a", [{"id": "c1", "name": "Bash", "arguments": {}}]),
        ("b", [{"id": "c2", "name": "Bash", "arguments": {}}]),
        ("done", []),
    ]))
    events = asyncio.run(_drain(sid))
    create_session(sid, platform=platform, model="primary", title="t",
                   source=platform)

    async def _record():
        async def _feed():
            for evt in events:
                yield evt
        return [e async for e in record_events(
            _feed(), session_id=sid, turn_id="t1", prompt="go",
            model="primary", source=platform)]

    asyncio.run(_record())
    asyncio.run(_record())
    # Two turns of the same shape, so `last_hour` is an aggregate and not one
    # row being divided by itself.

    served = dashboard._usage()["last_hour"]
    assert served["input_tokens"] > 0, served
    assert served["cache_read"] <= served["input_tokens"], served
    assert served["cache_read"] / served["input_tokens"] <= 1.0, (
        f"{platform}: dashboard serves "
        f"{served['cache_read'] / served['input_tokens']:.1%} cached")

    doc = json.loads((sessions_io.SESSIONS_DIR / f"{sid}.json").read_text())
    assert doc["platform"] == platform, doc["platform"]
    turn_rows = [m["stats"] for m in doc["messages"]
                 if isinstance(m.get("stats"), dict)
                 and "iteration" not in m["stats"]
                 # Other entries carry a stats dict too (thinking, tool calls)
                 # with only a duration; a turn row is the one with tokens.
                 and "input_tokens" in m["stats"]]
    assert turn_rows, "the recorder persisted no turn-level row"
    for stats in turn_rows:
        assert stats["cache_read"] <= stats["input_tokens"], stats
        assert stats["cache_read_sum"] <= stats["prompt_tokens_sum"], stats

    # And the pre-fix mapping is still detectable: 173,000 summed cached over
    # a 120,000 peak is what landed here on 2026-09-11.
    assert served["cache_read"] < served["input_tokens"], served


# ── clauses 5-8: the committed script ───────────────────────────────────────
def _session(path: Path, *, platform: str, created: str,
             rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    messages = [{"role": "user", "content": "go",
                 "timestamp": created, "id": "u1"}]
    for i, stats in enumerate(rows):
        messages.append({"role": "assistant", "content": f"a{i}",
                         "timestamp": created, "id": f"a{i}", "stats": stats})
    path.write_text(json.dumps({
        "session_id": path.stem, "model": "primary",
        "created_at": created, "platform": platform,
        "messages": messages,
    }))


BOOT = datetime.fromisoformat("2026-09-11T06:00:00")
# A fixed boot epoch, so the /proc tick arithmetic in the boot-resolution test
# is arithmetic rather than a second reading of the same clock it is testing.
_BTIME = 1757000000


def _fixture_sessions(dirn: Path) -> None:
    """Two post-boot sessions (different platforms) and one pre-boot session.

    The pre-boot one carries a much better cached fraction than anything after
    the restart, so a script that failed to cut at the boot would report the
    blended number #618 was filed about rather than the post-boot truth.
    """
    _session(dirn / "after-worker.json", platform="worker",
             created="2026-09-11T07:00:00",
             rows=[
                 {"iteration": 1, "input_tokens": 10_000, "cache_read": 0},
                 {"iteration": 2, "input_tokens": 60_000, "cache_read": 42_000},
                 {"iteration": 3, "input_tokens": 90_000, "cache_read": 63_000},
                 {"iteration": 8, "input_tokens": 120_000, "cache_read": 66_000},
                 # Duplicated stats block: the same dict is written onto the
                 # assistant row and the tool-call row of one iteration.
                 {"iteration": 8, "input_tokens": 120_000, "cache_read": 66_000},
                 # The turn-level row as the fixed writer emits it: peak pair
                 # bounded, summed pair alongside. The pre-fix writer put
                 # `cache_read` here as the summed 171,000 over the 120,000
                 # peak — see test_script_counts_a_pre_fix_offending_row.
                 {"input_tokens": 120_000, "cache_read": 118_000,
                  "prompt_tokens_sum": 280_000, "cache_read_sum": 222_000},
             ])
    _session(dirn / "after-mc.json", platform="mission-control",
             created="2026-09-11T08:00:00",
             rows=[
                 {"iteration": 1, "input_tokens": 20_000, "cache_read": 16_000},
                 {"iteration": 2, "input_tokens": 40_000, "cache_read": 36_000},
                 {"iteration": 8, "input_tokens": 80_000, "cache_read": 40_000},
             ])
    _session(dirn / "before-worker.json", platform="worker",
             created="2026-09-10T20:00:00",
             rows=[
                 {"iteration": 1, "input_tokens": 10_000, "cache_read": 10_000},
                 {"iteration": 2, "input_tokens": 20_000, "cache_read": 20_000},
                 {"iteration": 8, "input_tokens": 30_000, "cache_read": 30_000},
             ])


# ── clause 5 ────────────────────────────────────────────────────────────────
def test_script_prints_the_per_iteration_table(tmp_path):
    mod = _script_module()
    dirn = tmp_path / "sessions"
    _fixture_sessions(dirn)

    report = mod.collect(dirn, boot=BOOT, boot_source="test fixture boot")
    table = mod.render(report)

    assert "iteration" in table and "cached" in table
    for iteration in (1, 2, 3, 8):
        bucket = mod.bucket_for(report.overall, iteration)
        assert bucket is not None, f"iteration {iteration} missing from table"
        assert bucket.samples >= 1
        assert bucket.prompt_tokens > 0
    b1 = mod.bucket_for(report.overall, 1)
    assert b1.prompt_tokens == 30_000            # 10k worker + 20k mc
    assert abs(b1.fraction - 16_000 / 30_000) < 1e-9


def test_script_runs_from_the_command_line(tmp_path):
    """One command, and the table lands on stdout.

    Spawned as a subprocess on purpose: the acceptance says one command, so
    what is pinned is the CLI, not the importable functions the other tests
    call directly.
    """
    dirn = tmp_path / "sessions"
    _fixture_sessions(dirn)
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "prefix_cache_iteration_table.py"),
         "--sessions-dir", str(dirn), "--boot", "2026-09-11T06:00:00"],
        capture_output=True, text=True, cwd=str(ROOT), timeout=120)
    assert out.returncode == 0, out.stderr
    assert "iteration" in out.stdout
    assert "8+" in out.stdout or "iteration 8+" in out.stdout.lower()
    assert "boot" in out.stdout.lower()


# ── clause 6 ────────────────────────────────────────────────────────────────
def test_script_states_the_boot_cut_and_what_it_excluded(tmp_path):
    mod = _script_module()
    dirn = tmp_path / "sessions"
    _fixture_sessions(dirn)

    report = mod.collect(dirn, boot=BOOT, boot_source="lloyd-backend start")
    table = mod.render(report)

    assert report.boot_cut == BOOT
    assert report.sessions_excluded == 1, report
    assert report.sessions_kept == 2, report
    assert "2026-09-11T06:00:00" in table
    assert "lloyd-backend start" in table
    assert "1" in table.split("excluded")[1][:40], (
        "the excluded-session count must be printed, not only computed")
    # The excluded session's rows are accounted for, not silently dropped.
    assert report.rows_excluded == 3, report


def test_excluding_the_boot_changes_the_answer_it_prints(tmp_path):
    """The boot cut has to be load-bearing, not decorative.

    Without it the pre-boot session's perfect 100% cache blends into the same
    table — which is precisely how a day-filtered aggregate produced three
    different answers across the 2677dea landing.
    """
    mod = _script_module()
    dirn = tmp_path / "sessions"
    _fixture_sessions(dirn)

    cut = mod.collect(dirn, boot=BOOT, boot_source="test")
    merged = mod.collect(dirn, boot=None, boot_source="no boot cut applied")

    b8_cut = mod.bucket_for(cut.overall, 8)
    b8_merged = mod.bucket_for(merged.overall, 8)
    assert b8_merged.fraction > b8_cut.fraction, (
        "the pre-boot session did not move the iteration-8 number")
    assert merged.sessions_excluded == 0


# ── clause 7 ────────────────────────────────────────────────────────────────
def test_script_reports_the_largest_fall_and_the_iteration_8_bucket(tmp_path):
    mod = _script_module()
    dirn = tmp_path / "sessions"
    _fixture_sessions(dirn)

    report = mod.collect(dirn, boot=BOOT, boot_source="test")

    # Overall iteration 2: (42k + 36k) / (60k + 40k) = 78%; iteration 3 is the
    # worker's 63k/90k = 70% → a 7.99pt fall. 8+ is (66k + 40k) / 200k = 53%.
    fall = report.largest_fall
    assert fall is not None
    assert (fall.from_iteration, fall.to_iteration) == (2, 3), fall
    assert abs(fall.fall - (0.78 - 0.70)) < 1e-9, fall

    eight = report.iteration_8_plus
    assert eight is not None
    assert eight.samples == 2, eight          # one per post-boot session
    assert abs(eight.fraction - 106_000 / 200_000) < 1e-9, eight

    table = mod.render(report)
    assert "largest fall" in table.lower()
    assert "2 -> 3" in table
    assert "8+" in table


# ── clause 8 ────────────────────────────────────────────────────────────────
def test_script_breaks_the_table_down_per_platform(tmp_path):
    mod = _script_module()
    dirn = tmp_path / "sessions"
    _fixture_sessions(dirn)

    report = mod.collect(dirn, boot=BOOT, boot_source="test")

    assert set(report.by_platform) == {"worker", "mission-control"}
    worker1 = mod.bucket_for(report.by_platform["worker"], 1)
    assert (worker1.samples, worker1.prompt_tokens) == (1, 10_000), worker1
    mc1 = mod.bucket_for(report.by_platform["mission-control"], 1)
    assert (mc1.samples, mc1.prompt_tokens) == (1, 20_000), mc1
    # Overall is the union, and per-platform falls are reported too.
    assert mod.bucket_for(report.overall, 1).samples == 2
    table = mod.render(report)
    assert "worker" in table and "mission-control" in table


# ── clause 9 ────────────────────────────────────────────────────────────────
def test_only_sessions_after_the_boot_reach_the_buckets(tmp_path):
    """Two fixtures either side of a mocked boot; only the later one counts."""
    mod = _script_module()
    dirn = tmp_path / "sessions"
    _fixture_sessions(dirn)

    report = mod.collect(dirn, boot=BOOT, boot_source="mocked boot")

    for buckets in [report.overall, *report.by_platform.values()]:
        for bucket in buckets:
            assert bucket.samples <= 2, bucket
    all1 = mod.bucket_for(report.overall, 1)
    assert all1.prompt_tokens == 30_000, (
        "the pre-boot session's 10k iteration-1 prompt leaked into the bucket")
    assert all1.cached_tokens == 16_000
    # The pre-boot session is named and counted, not silently gone.
    assert report.excluded_session_ids == ["before-worker"], report


@pytest.fixture
def _tz_pdt(monkeypatch):
    """Pin the machine's zone to UTC-7 so a frame test is not a timezone test."""
    monkeypatch.setenv("TZ", "Etc/GMT+7")        # POSIX sign is inverted
    time.tzset()
    yield
    monkeypatch.delenv("TZ", raising=False)
    time.tzset()


def test_the_boot_cut_reads_both_timestamp_frames_in_one_clock(tmp_path,
                                                               _tz_pdt):
    """`sessions/*.json` carries two timestamp frames on the same day.

    Measured 2026-09-11: the chat and background writers stamp naive local
    (`created_at: 2026-09-11T11:02:45`, file born `11:03:32 -0700`) while the
    Inner Voice writer stamps honest UTC (`created_at: 2026-09-11T17:46:04Z`,
    file born `10:51:35 -0700`). Comparing the two raw raises
    `TypeError: can't compare offset-naive and offset-aware datetimes`, and the
    obvious cure — dropping `tzinfo` — puts this session's 12:30Z start seven
    hours AFTER the 06:00 boot instead of half an hour before it, admitting a
    pre-restart session into the post-boot table. Which is #618's blend again,
    just wearing a `Z`.
    """
    mod = _script_module()
    dirn = tmp_path / "sessions"
    _session(dirn / "utc-labelled.json", platform="worker",
             created="2026-09-11T12:30:00Z",       # 05:30 local: pre-boot
             rows=[{"iteration": 1, "input_tokens": 8_000, "cache_read": 8_000}])
    _session(dirn / "naive-local.json", platform="worker",
             created="2026-09-11T07:00:00",        # naive: local, post-boot
             rows=[{"iteration": 1, "input_tokens": 9_000, "cache_read": 4_500}])

    report = mod.collect(dirn, boot=BOOT, boot_source="mocked boot")

    assert report.excluded_session_ids == ["utc-labelled"], report
    assert report.sessions_kept == 1, report
    first = mod.bucket_for(report.overall, 1)
    assert (first.prompt_tokens, first.cached_tokens) == (9_000, 4_500), first


def test_a_session_spanning_the_boot_is_cut_where_it_spanned(tmp_path):
    """A session that started before the restart cannot vote post-boot.

    Its early iterations were served by the old engine's cache; #618's rule is
    the session's FIRST message, and a long-running autonomy session is exactly
    the case that would otherwise blend the two boots.
    """
    mod = _script_module()
    dirn = tmp_path / "sessions"
    dirn.mkdir()
    _session(dirn / "spanning.json", platform="worker",
             created="2026-09-10T23:00:00",
             rows=[{"iteration": 1, "input_tokens": 5_000, "cache_read": 5_000}])
    path = dirn / "spanning.json"
    doc = json.loads(path.read_text())
    doc["messages"].append({"role": "assistant", "content": "late",
                            "timestamp": "2026-09-11T09:00:00", "id": "a9",
                            "stats": {"iteration": 2, "input_tokens": 9_000,
                                      "cache_read": 9_000}})
    path.write_text(json.dumps(doc))

    report = mod.collect(dirn, boot=BOOT, boot_source="mocked boot")

    assert report.sessions_excluded == 1
    assert report.excluded_session_ids == ["spanning"], report
    assert report.overall == [], report.overall


# ── clause 10 ───────────────────────────────────────────────────────────────
def test_rows_without_an_iteration_key_are_never_a_bucket(tmp_path):
    """The turn-level row has no `iteration` key — 20,179 of them on 2026-09-11.

    #604 and #618 both filtered on `iteration == 0`, matched nothing, and read
    as proof the store was fixed. Here the same rows are excluded from every
    bucket and printed on their own line with their own totals, so they cannot
    be mistaken for an iteration and cannot vanish either.
    """
    mod = _script_module()
    dirn = tmp_path / "sessions"
    _fixture_sessions(dirn)

    report = mod.collect(dirn, boot=BOOT, boot_source="test")

    assert mod.bucket_for(report.overall, 0) is None, (
        "the no-iteration row was counted as an iteration bucket")
    assert report.turn_rows_no_key == 1, report
    assert report.turn_rows_no_key_prompt == 120_000, report
    assert report.turn_rows_no_key_cached == 118_000, report

    table = mod.render(report)
    assert "turn-level" in table.lower()
    assert "no iteration key" in table.lower()
    # And the script shows why they must be kept out: summed cached over peak
    # prompt is the >100% reading itself.
    assert report.turn_rows_offending == 0, (
        "a post-fix turn row still reads over 100% cached")

    # An iteration:0 row is the same class of row under the old writer and is
    # reported separately too — never folded into bucket 0 of the table.
    _session(dirn / "legacy.json", platform="worker",
             created="2026-09-11T07:30:00",
             rows=[{"iteration": 0, "input_tokens": 10_000,
                    "cache_read": 9_000}])
    legacy = mod.collect(dirn, boot=BOOT, boot_source="test")
    assert mod.bucket_for(legacy.overall, 0) is None, legacy.overall
    assert legacy.turn_rows_iteration_zero == 1, legacy


def test_script_counts_a_pre_fix_offending_row_it_cannot_fix(tmp_path):
    """The instrument still reports the >100% rows it finds; it does not fix them.

    Historical rows stay in the file and in the count, so clause 3's
    "zero offenders over a day of traffic after the change" is a measurement
    the script can actually falsify.
    """
    mod = _script_module()
    dirn = tmp_path / "sessions"
    dirn.mkdir()
    _session(dirn / "prefix.json", platform="worker",
             created="2026-09-11T07:00:00",
             rows=[{"iteration": 1, "input_tokens": 10_000, "cache_read": 0},
                   {"input_tokens": 262_101, "cache_read": 19_936_000}])

    report = mod.collect(dirn, boot=BOOT, boot_source="test")

    assert report.turn_rows_offending == 1, report
    assert report.worst_offender_ratio >= 19_936_000 / 262_101, report
    assert "over 100%" in mod.render(report)


# ── boot discovery: the script's own input, not the caller's goodwill ───────
def test_boot_resolution_reads_the_supervisor_status_and_proc(tmp_path):
    """Backend AND engine units; the latest start wins, because that is when
    the prefix cache went cold — #618's `lloyd-backend start time` alone would
    miss an engine-only restart, and a vLLM restart clears the KV cache with it.
    """
    mod = _script_module()
    status = (
        "lloyd-mc:lloyd-backend           RUNNING   pid 2298786, uptime 3:02:11\n"
        "agent-llm-primary                RUNNING   pid 2340837, uptime 21:19:28\n"
        "agent-tts                        RUNNING   pid 2358013, uptime 19:51:54\n"
    )
    procs = tmp_path / "proc"
    procs.mkdir(parents=True)
    (procs / "stat").write_text(f"btime {_BTIME}\n")
    for pid, start in (("2298786", "2026-09-11 06:10:00"),
                       ("2340837", "2026-09-10 12:00:00"),
                       ("2358013", "2026-09-10 12:00:00")):
        d = procs / pid
        d.mkdir(parents=True)
        ticks = int((datetime.strptime(start, "%Y-%m-%d %H:%M:%S").timestamp()
                     - _BTIME) * 100)
        # Field 22 of /proc/<pid>/stat is starttime: `comm` is field 2, so the
        # ticks are the 20th token after it. Written out positionally rather
        # than guessed at, because a parser reading field 18 instead would pass
        # a fixture that counts from the wrong column.
        (d / "stat").write_text(" ".join(
            [pid, "(python3)", "S", *(["0"] * 18), str(ticks), "0", "0"]) + "\n")

    cut, source = mod.resolve_boot_from_proc(status, proc_root=procs)

    assert cut == datetime.fromisoformat("2026-09-11 06:10:00"), cut
    assert "lloyd-backend" in source


def test_status_parsing_tolerates_a_unit_that_is_not_running():
    mod = _script_module()
    assert mod.parse_supervisor_status(
        "lloyd-mc:lloyd-backend  STOPPED\nagent-llm-primary  RUNNING   pid 7, uptime 0:01\n"
    ) == {"agent-llm-primary": 7}
    assert mod.parse_supervisor_status("") == {}


def test_landing_cut_reads_only_promotions_that_restarted(tmp_path, monkeypatch):
    """`--boot-from-landing` cuts at a landing, not at the ledger's newest row.

    promotions.jsonl carries every loop event — gate rungs every few minutes —
    so taking the newest row's stamp would cut at the last gate run. And a
    `restart: false` promotion replaced no process, so the cache stayed warm.
    """
    mod = _script_module()
    ledger = tmp_path / "promotions.jsonl"
    ledger.write_text("\n".join(json.dumps(r) for r in [
        {"event": "promoted", "created_at": "2026-09-11T08:00:00"},
        {"event": "promoted", "created_at": "2026-09-11T09:00:00", "restart": False},
        {"event": "gate", "created_at": "2026-09-11T10:00:00"},
    ]) + "\n")
    monkeypatch.setattr(mod, "PROMOTIONS", ledger)
    assert mod.latest_landing() == datetime(2026, 9, 11, 8, 0, 0)
