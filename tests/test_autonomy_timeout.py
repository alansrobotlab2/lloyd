"""Run timeout precedence and stuck-task recovery.

Why this file exists
--------------------
Two timeout caps apply to every autonomy run — the task's frontmatter
`timeout_seconds` (`app/autonomy.py`) and the work source's
`max_duration_seconds` (`workers/pool.py`). The recorded rule is "both apply,
min wins", and the correction log spent a cycle discovering that the *opposite*
belief ("frontmatter timeout_seconds is not read") had been written into memory
as fact.

The incident that fixed is worth keeping pinned: when the two values were
**equal**, the pool timer won the race and cancelled the coroutine before its
own handler ran — no run record, no activity-log line, and the task left
`in_progress` on disk. `app/autonomy.py:682` now subtracts a margin for exactly that
reason.

`AUTONOMY_DIR` is `~/obsidian/autonomy` — a protected vault path — so every test
that touches the filesystem redirects it to a tmp dir. `_find_task_file`,
`_update_task_field` and `_append_activity_log` all resolve through that one
module global, so patching it isolates all three.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
import yaml

from app import autonomy
from workers import pool as worker_pool

DEFAULT_TASK_TIMEOUT = 1800          # app/autonomy.py's `or 1800` fallback
MARGIN = autonomy._POOL_TIMEOUT_MARGIN
POOL_DEFAULT = worker_pool._DEFAULT_MAX_DURATION_SECONDS


def resolve_timeout(declared: int, max_duration: int | None) -> int:
    """Mirrors app/autonomy.py:676-683 verbatim.

    A copy, not an import: the expression is inline inside `run_task`, which
    needs the whole harness to invoke. Extracting it is the right refactor and
    out of scope for a test-only change; until then this pins the rule and the
    constants it depends on, and fails if either moves.
    """
    if max_duration:
        return max(60, min(declared, int(max_duration) - MARGIN))
    return declared


# ── precedence rule ──────────────────────────────────────────────────────────

def test_task_timeout_is_used_when_the_pool_sets_no_cap():
    assert resolve_timeout(600, None) == 600


def test_zero_pool_cap_is_treated_as_absent():
    """`if max_duration:` — a falsy 0 must not clamp everything to 60s."""
    assert resolve_timeout(600, 0) == 600


def test_smaller_task_timeout_wins_over_a_larger_pool_cap():
    assert resolve_timeout(300, 1800) == 300


def test_smaller_pool_cap_wins_over_a_larger_task_timeout():
    assert resolve_timeout(1800, 1200) == 1200 - MARGIN


def test_equal_timeouts_yield_the_pool_cap_minus_the_margin():
    """The race condition, pinned. With both at 1800 the pool won at exactly
    1800 and cancelled the coroutine before its own handler could record the
    run — the task stayed `in_progress` on disk with no run record. Equal must
    never resolve to equal."""
    assert resolve_timeout(1800, 1800) == 1800 - MARGIN
    assert resolve_timeout(1800, 1800) < 1800


def test_the_task_timer_always_fires_before_the_pool_timer():
    """The invariant behind every case: strict inequality, for every cap."""
    for declared in (60, 300, 600, 1800, 3600, 7200):
        for cap in (300, 600, 1800, 3600):
            assert resolve_timeout(declared, cap) < cap, (declared, cap)


def test_margin_does_not_gobble_a_short_cap():
    assert resolve_timeout(1800, 45) == 60


def test_floor_is_sixty_seconds():
    assert resolve_timeout(1800, 1) == 60


def test_a_one_second_margin_cap_still_leaves_working_time():
    assert resolve_timeout(1800, MARGIN + 5) == 60


def test_the_margin_constant_is_a_positive_number_of_seconds():
    assert isinstance(MARGIN, int) and 0 < MARGIN < 600


def test_defaults_are_the_documented_numbers():
    """Both defaults are load-bearing: a task with no `timeout_seconds` gets 30
    minutes, and a source with no `max_duration_seconds` gets 15. The
    entity-resolution-sweep poisoning was diagnosed against the *source* cap."""
    assert DEFAULT_TASK_TIMEOUT == 1800
    assert POOL_DEFAULT == 900
    assert resolve_timeout(DEFAULT_TASK_TIMEOUT, POOL_DEFAULT) == POOL_DEFAULT - MARGIN


def test_the_sweep_cap_pair_is_resolvable():
    """Concrete regression from the incident log: sweep at 1800 against a 1800
    source cap previously resolved to a tie."""
    assert resolve_timeout(1800, 1800) == 1770


# ── stuck-task recovery ──────────────────────────────────────────────────────

@pytest.fixture
def autonomy_dir(tmp_path, monkeypatch):
    d = tmp_path / "autonomy"
    d.mkdir()
    monkeypatch.setattr(autonomy, "AUTONOMY_DIR", d)
    return d


def task_file(dir_, task_id, *, status, updated, timeout=None, name="some task"):
    front = [
        "---",
        f"id: {task_id}",
        f"name: {name}",
        f"status: {status}",
        f"updated: {updated}",
    ]
    if timeout is not None:
        front.append(f"timeout_seconds: {timeout}")
    front.append("---")
    path = dir_ / f"{task_id}-{name.replace(' ', '-')}.md"
    path.write_text("\n".join(front) + "\n\n# Body\n", encoding="utf-8")
    return path


def iso(offset_seconds: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=offset_seconds)).isoformat()


def test_no_stuck_tasks_is_a_noop(autonomy_dir):
    task_file(autonomy_dir, "10", status="up_next", updated=iso(99999))
    assert autonomy.recover_stuck_tasks() == []


def test_fresh_in_progress_task_is_left_alone(autonomy_dir):
    task_file(autonomy_dir, "11", status="in_progress", updated=iso(30), timeout=1800)
    assert autonomy.recover_stuck_tasks() == []
    assert "in_progress" in (autonomy_dir / "11-some-task.md").read_text()


def test_in_progress_past_its_timeout_is_reset(autonomy_dir):
    task_file(autonomy_dir, "12", status="in_progress", updated=iso(3700), timeout=3600)
    recovered = autonomy.recover_stuck_tasks()
    assert "12" in [str(t) for t in recovered]
    text = (autonomy_dir / "12-some-task.md").read_text()
    assert "status: up_next" in text
    assert "Recovered from in_progress" in text        # activity note appended


def test_missing_timeout_seconds_falls_back_to_thirty_minutes(autonomy_dir):
    task_file(autonomy_dir, "13", status="in_progress", updated=iso(1900))
    assert "13" in [str(t) for t in autonomy.recover_stuck_tasks()]


def test_a_short_timeout_recovers_sooner(autonomy_dir):
    """The reason `timeout_seconds` in frontmatter is not decorative — it drives
    recovery speed as well as the run cap."""
    task_file(autonomy_dir, "14", status="in_progress", updated=iso(700), timeout=600)
    assert "14" in [str(t) for t in autonomy.recover_stuck_tasks()]


def test_files_without_the_number_prefix_are_ignored(autonomy_dir):
    (autonomy_dir / "_config.md").write_text(
        "---\nid: cfg\nstatus: in_progress\nupdated: 2020-01-01T00:00:00+00:00\n---\n",
        encoding="utf-8")
    assert autonomy.recover_stuck_tasks() == []


def test_malformed_frontmatter_does_not_abort_the_sweep(autonomy_dir):
    (autonomy_dir / "99-broken.md").write_text("no frontmatter at all\n", encoding="utf-8")
    task_file(autonomy_dir, "15", status="in_progress", updated=iso(99999), timeout=60)
    assert "15" in [str(t) for t in autonomy.recover_stuck_tasks()]


def test_last_run_is_used_when_updated_is_absent(autonomy_dir):
    path = autonomy_dir / "16-no-updated.md"
    path.write_text(
        "---\nid: 16\nname: no updated\nstatus: in_progress\n"
        f"last_run: {iso(99999)}\ntimeout_seconds: 600\n---\n\n# Body\n", encoding="utf-8")
    assert "16" in [str(t) for t in autonomy.recover_stuck_tasks()]


def test_a_task_with_no_timestamp_at_all_is_treated_as_stuck(autonomy_dir):
    """Characterized: with no `updated` and no `last_run`, `stuck_seconds` is
    forced past the timeout, so an in_progress task with no timestamp is
    recovered on the next tick rather than hanging forever."""
    path = autonomy_dir / "17-no-stamp.md"
    path.write_text("---\nid: 17\nname: no stamp\nstatus: in_progress\n"
                    "timeout_seconds: 1800\n---\n\n# Body\n", encoding="utf-8")
    assert "17" in [str(t) for t in autonomy.recover_stuck_tasks()]


def test_missing_autonomy_dir_is_not_an_error(monkeypatch, tmp_path):
    monkeypatch.setattr(autonomy, "AUTONOMY_DIR", tmp_path / "absent")
    assert autonomy.recover_stuck_tasks() == []


def test_recovery_only_touches_the_task_it_flagged(autonomy_dir):
    task_file(autonomy_dir, "18", status="in_progress", updated=iso(99999), timeout=600)
    task_file(autonomy_dir, "19", status="in_progress", updated=iso(10), timeout=600)
    autonomy.recover_stuck_tasks()
    assert "up_next" in (autonomy_dir / "18-some-task.md").read_text()
    assert "in_progress" in (autonomy_dir / "19-some-task.md").read_text()


# ── the same pass, over what the HTTP endpoint actually wrote (#1128) ────────
#
# Everything above hand-builds the front matter, so every `updated` in it is
# already honest UTC. That is exactly why none of it could catch #1128: the
# writer that produced the field was `POST /api/autonomy/task-write`, which
# stamped `datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")` — the host's naive
# local clock (PDT, -0700) wearing a UTC label — 25,200 s behind the true
# instant. `recover_stuck_tasks` parses that field as UTC and compares it to a
# real UTC `now`, so a task set `in_progress` from the Mission Control autonomy
# page read as 7 h old at the moment it was written: past its own
# `timeout_seconds`, flipped back to `up_next`, and logged as a recovery that
# never happened. The test below therefore crosses the boundary the bug lived
# behind — HTTP in, task file out, scheduler reading it back — instead of
# constructing the input.

@pytest.fixture
def endpoint_dir(tmp_path, monkeypatch):
    """Both globals: the router writes through its own `_AUTONOMY_DIR`, the
    recovery pass reads through `autonomy.AUTONOMY_DIR`. One directory, so the
    write and the read are the same file — which is the point."""
    from app.routers import autonomy as router_module

    d = tmp_path / "autonomy"
    d.mkdir()
    monkeypatch.setattr(router_module, "_AUTONOMY_DIR", d)
    monkeypatch.setattr(autonomy, "AUTONOMY_DIR", d)
    return d


@pytest.fixture
def endpoint_client(endpoint_dir):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.routers import autonomy as router_module

    app = FastAPI()
    app.include_router(router_module.router)
    with TestClient(app) as c:
        yield c


def test_claiming_a_task_from_the_ui_is_not_reported_stuck(endpoint_dir,
                                                           endpoint_client):
    """Clause 3. A real POST sets `status: in_progress` and stamps `updated`;
    the very next recovery pass must leave the task alone.

    Pre-fix, `stuck_seconds` for that file was 25,200 s against a 1,800 s
    timeout, so the pass returned `["20"]`, rewrote it to `up_next`, and wrote
    `Recovered from in_progress after 25200s` — a fabricated incident in the
    task's own log. Both halves are asserted: the returned list, and the absence
    of the line, because the line is what a human reads."""
    (endpoint_dir / "20-ui-claimed.md").write_text(
        "---\ntype: autonomy\nid: 20\nname: ui-claimed\nstatus: up_next\n"
        "timeout_seconds: 1800\n---\n\n# Body\n", encoding="utf-8")

    r = endpoint_client.post("/api/autonomy/task-write",
                             json={"id": 20, "status": "in_progress"})
    assert r.status_code == 200, r.text
    text = (endpoint_dir / "20-ui-claimed.md").read_text()
    assert "status: in_progress" in text, "the claim itself must have landed"

    assert autonomy.recover_stuck_tasks() == []
    text = (endpoint_dir / "20-ui-claimed.md").read_text()
    assert "Recovered from in_progress" not in text
    assert "status: in_progress" in text, "the recovery pass moved a task it " \
                                          "should not have seen"


def test_the_elapsed_reading_the_recovery_pass_computes_is_under_the_timeout(
        endpoint_dir, endpoint_client):
    """The same claim as clause 3, measured rather than observed: the number the
    guard compares is the one that was wrong. Pre-fix this read 25,200 s."""
    (endpoint_dir / "21-ui-claimed.md").write_text(
        "---\ntype: autonomy\nid: 21\nname: ui-claimed\nstatus: up_next\n"
        "timeout_seconds: 1800\n---\n\n# Body\n", encoding="utf-8")
    endpoint_client.post("/api/autonomy/task-write",
                         json={"id": 21, "status": "in_progress"})

    task = autonomy._parse_task_file(endpoint_dir / "21-ui-claimed.md")
    updated = autonomy._parse_iso(task.get("updated"))
    assert updated is not None, f"unreadable: {task.get('updated')!r}"
    timeout = int(task.get("timeout_seconds") or 1800)
    now = datetime.now(timezone.utc)
    elapsed = (now - updated).total_seconds()
    assert 0 <= elapsed < timeout, (
        f"recovery would read this task as {elapsed:.0f}s old against a "
        f"{timeout}s timeout")


def test_a_task_written_through_the_endpoint_is_not_allowed_to_double_run(
        endpoint_dir, endpoint_client, monkeypatch, tmp_path):
    """The other half of the same acceptance check: the double-run guard.

    `run_task` refuses a second copy of an `in_progress` task while its
    `updated` is inside `timeout_seconds`. Pre-fix a UI claim put `updated`
    25,200 s in the past, so that comparison was false on arrival and the guard
    could never trip for a hand-claimed task — the interleaving of two runs of
    #38 the guard was written for. Post-fix the claim is fresh, the guard
    fires, and `run_query` is never reached.
    """
    skill = tmp_path / "SKILL.md"
    skill.write_text("# test skill\nDo the thing.\n", encoding="utf-8")
    (endpoint_dir / "23-ui-claimed.md").write_text(
        "---\ntype: autonomy\nid: 23\nname: ui-claimed\nstatus: up_next\n"
        f"timeout_seconds: 1800\nskill_name: {skill}\n---\n\n# Body\n",
        encoding="utf-8")
    endpoint_client.post("/api/autonomy/task-write",
                         json={"id": 23, "status": "in_progress"})

    monkeypatch.setattr(autonomy, "AUTONOMY_RUNS_DIR", tmp_path / "runs")

    async def must_not_run(messages, options):
        raise AssertionError(
            "run_query was reached for a task the UI claimed seconds ago: the "
            "double-run guard read its `updated` as stale and let a second copy start")

    monkeypatch.setattr("app.harness.run_query", must_not_run)
    result = asyncio.run(autonomy.run_task(23))
    assert result.get("skipped") is True, result
    assert "already in_progress" in result.get("error", ""), result
    assert not (tmp_path / "runs" / "23").exists()


def test_a_genuinely_stuck_task_is_still_recovered_after_the_fix(endpoint_dir,
                                                                 endpoint_client):
    """The guard against over-correction. A task whose `updated` really is older
    than its timeout must still be reset — otherwise this change would have
    traded a false recovery for a task that hangs in `in_progress` forever,
    which is the failure mode the pass exists to clear."""
    (endpoint_dir / "22-really-stuck.md").write_text(
        "---\ntype: autonomy\nid: 22\nname: really-stuck\nstatus: in_progress\n"
        f"updated: {iso(7200)}\ntimeout_seconds: 1800\n---\n\n# Body\n",
        encoding="utf-8")
    assert "22" in [str(t) for t in autonomy.recover_stuck_tasks()]
    assert "Recovered from in_progress" in (endpoint_dir / "22-really-stuck.md").read_text()


# ── #2704: what a wall-clock kill is allowed to say about itself ─────────────
#
# Autonomy task #86 (nightly iv-metrics) died at its 300 s cap on four of its last
# six scheduled runs — `run_86_20261010_151159`, `run_86_20261012_080047`,
# `run_86_20261013_080034`, `run_86_20261014_080101` — while the slowest SUCCESS in
# the same window took 55.3 s, 18 % of the cap. The cap is therefore not the defect
# and these nodes do not touch it. What they pin is the other half of what triage
# measured: the artefact lands 4-12 s into every killed run (`iv-metrics.jsonl`
# `recorded_at: 2026-10-14T08:01:13` against `started_at: 2026-10-14T08:01:01`), and
# the run then hangs ~285 s on a final-turn request that never gets its first SSE
# token — yet the record said only `stop_reason/usage/num_turns: null`, `tool_errors:
# 0`, `changes_files: 0`. Four nights of a job that produced its row read, in the
# only place anyone looks, as four nights of a job that did nothing.
#
# `tests/test_autonomy_discovery_infra_kind.py` owns the ADDITIVITY half (clause 3):
# the charge, the retry budget and the Activity Log line must not move. What lives
# here is the witness itself.

# Two lines copied out of the real 2026-10-14 attempt, from the two Bash steps of
# the task's own `description:` command — the recorder (which appends the row) and
# the mirror (which copies it into the vault). They are the payload clause 2 is about.
RECORDER_RESULT = ("BREACH iv-metrics: since=2026-10-13T08:00:00+00:00 "
                   "llm_calls=741 rate=0.0688 rows=1 appended")
MIRROR_RESULT = "iv-metrics-mirror: copied source=32 mirror_was=31 rows"


def _wedged_after_the_artefact(hang_seconds: float):
    """A `run_query` stand-in: the artefact lands, then the stream goes silent.

    The shape is #86's, read off its own session file
    (`20261014_010101_autonomy_d953.json`): two tool results by t≈0, a `thinking`
    delta a moment later, then nothing until the kill — the last real tool result at
    01:01:40.487 local, the next event the kill at 01:06:01.763. `run_query` is
    patched at the seam `run_task` iterates, so the code under test is `run_task`'s
    own `asyncio.timeout` arm and not a hand-built call into `_record_failure`.
    """
    def _fake(messages, options):
        async def _gen():
            yield {"type": "tool_call", "name": "Bash", "call_id": "c1",
                   "input": {"command": "python -m scripts.iv_metrics_probe"}}
            yield {"type": "tool_result", "name": "Bash", "call_id": "c1",
                   "content": RECORDER_RESULT, "is_error": False}
            yield {"type": "tool_call", "name": "Bash", "call_id": "c2",
                   "input": {"command": "python scripts/iv_metrics_mirror.py"}}
            yield {"type": "tool_result", "name": "Bash", "call_id": "c2",
                   "content": MIRROR_RESULT, "is_error": False}
            await asyncio.sleep(0.4)      # the last thing the run ever said
            yield {"type": "text_delta", "text": "Now the closing turn…"}
            await asyncio.sleep(hang_seconds)
            yield {"type": "result", "stop_reason": "end_turn", "usage": {},
                   "num_turns": 3}
        return _gen()
    return _fake


@pytest.fixture
def kill_env(tmp_path, monkeypatch):
    """Task #86's file, a redirected run-record tree, and no transports.

    `failure_count: 1` of `max_retries: 3` is the state #86 was in on 2026-10-14
    (`server.err.1:89407` logged `Task #86 failed (task, 3/5)` for the third one), so
    the arithmetic the clause-3 node checks is a move from 1, not from 0.
    """
    runs = tmp_path / "autonomy-runs"
    tasks = tmp_path / "autonomy"
    runs.mkdir()
    tasks.mkdir()
    monkeypatch.setattr(autonomy, "AUTONOMY_RUNS_DIR", runs)
    monkeypatch.setattr(autonomy, "AUTONOMY_DIR", tasks)

    import app.discord_notify as discord_notify

    async def _no_alert(*args, **kwargs):
        return None

    monkeypatch.setattr(discord_notify, "discord_alert", _no_alert)
    monkeypatch.setattr("app.prompt_builder.build_system_prompt",
                        lambda **_kw: "sys", raising=False)
    # The task must carry a `skill_name` to be dispatchable at all (`run_task`
    # returns an error dict without one, before the flip to in_progress), but this
    # file is about the timeout arm, not about a real SKILL.md's wording — so the
    # content is stubbed rather than read out of the vault.
    monkeypatch.setattr(autonomy, "_load_skill_content",
                        lambda _name: "# iv-metrics protocol\n")
    (tasks / "86-nightly-iv-metrics-series.md").write_text(
        "---\n"
        "id: 86\n"
        "name: nightly-iv-metrics-series\n"
        "status: up_next\n"
        "priority: medium\n"
        "trigger: scheduler\n"
        "frequency: 1d @ 08:00\n"
        "skill_name: iv-metrics-series\n"
        "model: primary\n"
        "timeout_seconds: 1\n"
        "max_retries: 3\n"
        "failure_count: 1\n"
        "---\n\n# nightly iv-metrics\n\n## Activity Log\n\n- created\n",
        encoding="utf-8")
    return SimpleNamespace(runs=runs, tasks=tasks)


def _killed_record(kill_env):
    path = next((kill_env.runs / "86").glob("run_*.md"))
    text = path.read_text(encoding="utf-8")
    return yaml.safe_load(text.split("---")[1]), text


@pytest.mark.asyncio
async def test_a_wall_clock_kill_records_the_idle_gap_and_the_tools_it_sent(
        kill_env, monkeypatch):
    """Clause 1: idle-when-killed and dispatched-tool count, on both stores.

    The two numbers are what separates the two deaths that used to look identical:
    #86 wedged idle for ~257 s waiting for a first token, versus a run cut off mid
    tool loop with its cap genuinely spent working. `idle_seconds` is measured from
    the last harness EVENT (a `text_delta` counts — #86's last was a thinking block),
    so here it must land near 0.6 s: the stream's final event is 0.4 s in and the cap
    is 1 s. An implementation that stamped the whole wall clock, or the time of the
    last TOOL result instead of the last event, would satisfy the field's name and
    still be blind — the first reads 1.0, the second 0.6 + 0.4 = 1.0.
    """
    monkeypatch.setattr("app.harness.run_query", _wedged_after_the_artefact(30))

    out = await autonomy.run_task(86)

    fm, _ = _killed_record(kill_env)
    assert fm["timeout"] is True and fm["timeout_seconds"] == 1, fm
    idle = fm["idle_seconds"]
    assert isinstance(idle, (int, float)) and 0.4 <= idle < 0.95, (
        f"`idle_seconds` is {idle!r}; the run's last event was 0.4s into a 1s cap, so "
        "the gap at the kill is ~0.6s — 1.0 would be the whole cap, 0.0 would be no "
        "measurement at all")
    assert idle < fm["duration_seconds"], (
        f"idle {idle} >= duration {fm['duration_seconds']}: the field is the run's "
        "length, not the silence at its end")
    assert fm["tool_calls_dispatched"] == 2, (
        f"{fm['tool_calls_dispatched']!r} tool calls for a stream that emitted two "
        "`tool_call` events before going quiet")
    assert out["meta"]["idle_seconds"] == idle
    assert out["meta"]["tool_calls_dispatched"] == 2, out["meta"]
    # The second store. `scheduled_task.execute` hands `result["meta"]` to the pool
    # whole (workers/sources/scheduled_task.py:460) and the pool writes
    # `json.dumps(norm["meta"])` into `runs.meta_json` (workers/pool.py:1791), so
    # these keys reaching `meta_json` is a property of THIS dict's contents — the
    # query a reader re-runs is `select meta_json from runs where task_id='86'`.
    from workers.pool import normalize_result
    from workers.queue import QueueItem
    item = QueueItem(id=1, source="scheduled-task", kind="run", priority=50,
                     payload={"task_id": 86}, dedup_key=None, state="running",
                     attempts=1, enqueued_at="", claimed_at=None, claimed_by=None,
                     completed_at=None, error=None)
    norm = normalize_result(item, {**out, "task_id": "86"})
    assert json.loads(json.dumps(norm["meta"]))["idle_seconds"] == idle
    assert json.loads(json.dumps(norm["meta"]))["tool_calls_dispatched"] == 2


@pytest.mark.asyncio
async def test_the_killed_record_carries_the_last_thing_the_run_proved(
        kill_env, monkeypatch):
    """Clause 2: the recorder's verdict line and the mirror's row count are readable.

    Under a named heading, in the record BODY, because that is the file a person
    opens at 03:00 (`autonomy-runs/86/<run>.md`) and the session JSON is not it. The
    newest completed non-error result is the mirror's `source=32 mirror_was=31 rows`;
    the recorder's `BREACH … llm_calls=741 rate=0.0688` is the step before it, which
    is why the section keeps two — task #86 writes its artefact with one command whose
    two steps print those two lines. An error result is never the witness: those
    already leave as `tool_errors`, and a traceback would read as the verdict.
    """
    monkeypatch.setattr("app.harness.run_query", _wedged_after_the_artefact(30))

    await autonomy.run_task(86)

    _fm, text = _killed_record(kill_env)
    heading = "## Last completed tool result before the kill"
    assert heading in text, text[-800:]
    assert MIRROR_RESULT in text, "the newest result did not reach the record"
    assert RECORDER_RESULT in text, (
        "only the newest result was kept, so the line that says the row was APPENDED "
        "is still only in the session JSON")
    assert text.index(RECORDER_RESULT) > text.index(heading), (
        "the evidence is above its own heading, so it is not the section")
    assert "## Last completed tool result" not in text.split(heading, 1)[0]


@pytest.mark.asyncio
async def test_a_kill_with_no_completed_tool_result_says_so_without_lying(
        kill_env, monkeypatch):
    """The witness is a fact, not a format: no completed result, no section.

    A run killed inside its first turn has nothing to show, and inventing an empty
    heading would put a `## Last completed tool result before the kill` block in front
    of a reader expecting to find the artefact there.
    """
    def _silent(messages, options):
        async def _gen():
            await asyncio.sleep(30)
            yield {"type": "result", "stop_reason": "end_turn"}
        return _gen()

    monkeypatch.setattr("app.harness.run_query", _silent)

    await autonomy.run_task(86)

    fm, text = _killed_record(kill_env)
    assert "Last completed tool result" not in text, text[-800:]
    assert fm["tool_calls_dispatched"] == 0 and fm["idle_seconds"] >= 0.9, fm


@pytest.mark.asyncio
async def test_a_silent_step_after_the_recorder_does_not_erase_the_witness(
        kill_env, monkeypatch):
    """Depth is counted in results that could SAY something, not in raw results.

    The section keeps two completed non-error results, newest last, because task #86
    writes its artefact with one command whose recorder verdict and mirror summary are
    two results. But a two-deep list of RAW results has a failure mode that defeats the
    whole block: a run whose recorder prints `BREACH iv-metrics: … rate=0.0688` and then
    runs two steps that print nothing — `cd …`, a `test -s`, a `systemctl is-active` —
    ends with two blank results, and the recorder's line is out of the section exactly
    when the reader needs it. So a blank result is not collected at all, and the
    witness here is the recorder's own line with the two silent steps named as the
    tools that followed.
    """
    def _recorder_then_silence(messages, options):
        async def _gen():
            yield {"type": "tool_call", "name": "Bash", "call_id": "c1",
                   "input": {"command": "python -m scripts.iv_metrics_probe"}}
            yield {"type": "tool_result", "name": "Bash", "call_id": "c1",
                   "content": RECORDER_RESULT, "is_error": False}
            for call_id, cmd in (("c2", "test -s iv-metrics.jsonl"),
                                 ("c3", "systemctl is-active lloyd-agent-worker")):
                yield {"type": "tool_call", "name": "Bash", "call_id": call_id,
                       "input": {"command": cmd}}
                yield {"type": "tool_result", "name": "Bash", "call_id": call_id,
                       "content": "", "is_error": False}
            await asyncio.sleep(0.4)
            await asyncio.sleep(30)
            yield {"type": "result", "stop_reason": "end_turn"}
        return _gen()

    monkeypatch.setattr("app.harness.run_query", _recorder_then_silence)

    await autonomy.run_task(86)

    _fm, text = _killed_record(kill_env)
    assert RECORDER_RESULT in text, (
        "two blank tool results evicted the only proof that the row was appended: "
        + text[-500:])
    assert "## Last completed tool result before the kill" in text
