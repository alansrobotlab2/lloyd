"""A discovery death is charged to infrastructure, never to the task (#1807).

The record that started this — `autonomy-runs/53/run_53_20260929_031704.md`:
`trigger: scheduler`, `duration_seconds: 13.0`, `exception: ToolDiscoveryError`,
`summary: 'ToolDiscoveryError: MCP discovery yielded no tools; failed server(s):
lloyd-mcp'`, `tool_errors: 0`, `changes_files: 0`, and `failure_kind: task`. A
run that touched no tool and changed no file spent the task's
`failure_count: 1` of `max_retries: 3`, because the aggregator had not finished
restarting — which is the one restart `automod_land` performs BY DESIGN (MCP
before backend; 105 `promoted` events across 09-28/29).

What these pin:
  - an exception raised by the real discovery path classifies as `infra`, and
    the run record says so while the task's `failure_count` does not move;
  - the classification is reached through the code that RAISES (`MCPPool.open()`
    via the same seam `run_query` uses), not by raising the exception by hand;
  - an exception that is not on the discovery path still classifies as `task`
    and still spends the counter at the same seam — the half that keeps this
    from becoming a way to hide real failures;
  - the fleet's health aggregation separates the two, so a task whose failures
    are all infrastructure stops reading like a broken task;
  - and the charge never reaches `status: failed`, no matter how many
    discovery deaths an outage stacks up.

What these deliberately do NOT change: the dependent chain. `_record_failure`
writes `last_attempt` and never `last_run`, and `_is_dependency_met` reads only
`last_run` — so a chain held by a discovery death stays held under either label.
That is uniform across kinds today and out of scope here;
`test_a_discovery_death_holds_a_downstream_chain_just_as_a_task_failure_does`
pins that it stayed uniform rather than quietly diverging.
"""
from __future__ import annotations

import datetime
import importlib
import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from app.harness import mcp_pool as mcp_pool_mod
from app.harness.errors import ToolDiscoveryError
from app.harness.loop import _open_pool_for_run
from app.harness.mcp_pool import DEFAULT_LLOYD_MCP_SERVERS, MCPPool
from app.harness.options import RunOptions

_RUN_SEQUENCE = [0]   # keeps each charge's record file distinct
# The aggregator's OWN config, with only the port changed — the repo rail in
# `tests/test_mcp_layer.py` forbids an inline server dict for a good reason (a
# copied `"type": "sse"` did not follow `DEFAULT_LLOYD_MCP_URL` from /sse to
# /mcp, and five callsites hung), and deriving here also proves the failure
# below is reached over the transport production actually uses.
SERVER = "lloyd-mcp"
DEAD_SERVER = {SERVER: {**DEFAULT_LLOYD_MCP_SERVERS[SERVER],
                        "url": "http://127.0.0.1:1"}}
DAY_AGO = (datetime.datetime.now(datetime.timezone.utc)
           - datetime.timedelta(days=1)).isoformat()


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Two task files and the run-record tree, all under tmp_path.

    The three module globals every writer here resolves through are redirected,
    so this touches nothing under the real vault or data root. The task starts
    one failure deep on purpose — that is #53's actual state at 03:17 on
    2026-09-29 — because `failure_count` not MOVING is the assertion, and it
    cannot be seen from a zero.
    """
    autonomy = importlib.import_module("app.autonomy")
    runs = tmp_path / "autonomy-runs"
    tasks = tmp_path / "autonomy"
    runs.mkdir()
    tasks.mkdir()
    monkeypatch.setattr(autonomy, "AUTONOMY_RUNS_DIR", runs)
    monkeypatch.setattr(autonomy, "AUTONOMY_DIR", tasks)
    # Every charge below passes `alert=False`, which is what gates the daily-note
    # line and both Discord transports, so no live note is written. Asserted, not
    # assumed: a charge that reached the note would land in the tree under
    # tmp_path, because that is where the task file the streak reader walks lives.
    # `discord_alert` is imported inside its own try block, so it is patched at
    # the source. Nothing here may reach the network.
    import app.discord_notify as discord_notify

    async def _no_alert(*args, **kwargs):
        return None

    monkeypatch.setattr(discord_notify, "discord_alert", _no_alert)

    def write(task_id: int, name: str, **fields) -> None:
        fm = {"name": name, "id": task_id, "status": "up_next",
              "priority": "medium", "trigger": "scheduler",
              "frequency": "1d @ 03:15", "model": "auto",
              "max_retries": 3, "failure_count": 1, "last_run": DAY_AGO,
              "scheduled_at": None, "depends_on": None, **fields}
        (tasks / f"{task_id}-{name}.md").write_text(
            f"---\n{yaml.dump(fm, default_flow_style=False, sort_keys=False)}---\n"
            "\n# task\n\n## Activity Log\n\n- created\n", encoding="utf-8")

    write(53, "documentation-digester")
    # A downstream nightly task: `last_run` yesterday, upstream's is yesterday
    # too, so the freshness rule is what holds it and not some other gate.
    write(54, "knowledge-health-report", depends_on=53, frequency="1d @ 05:15")
    return SimpleNamespace(autonomy=autonomy, runs=runs, tasks=tasks)


def _front_matter(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8").split("---")[1])


def task_fields(env, task_id: int) -> dict:
    """What the task file says now, read back off disk."""
    return _front_matter(next(env.tasks.glob(f"{task_id}-*.md")))


def newest_record(env, task_id: int) -> dict:
    """The front matter of the record `_record_failure` just wrote."""
    paths = sorted((env.runs / str(task_id)).glob("run_*.md"))
    assert paths, f"no run record was written for task #{task_id}"
    return _front_matter(paths[-1])


def _read_task(env, task_id: int) -> dict:
    return task_fields(env, task_id)


async def charge(env, exc: BaseException, *, task_id: int = 53,
                 duration: float = 13.0) -> dict:
    """Charge `exc` to task #53 the way `run_task`'s handler does.

    Same handler, same arguments: the kind comes from the product's own
    `_failure_kind_of` — the function `run_task` calls, not a copy of its
    expression — and the `extra` mirrors the `except Exception` block's
    (`exception`, `tool_errors`). `duration` is the 13.0 s of the real record.
    """
    autonomy = env.autonomy
    task = _read_task(env, task_id)
    now = datetime.datetime.now(datetime.timezone.utc)
    # A fresh id per charge: records are named by run id, and a test that
    # charges twice has to be able to read two records back.
    _RUN_SEQUENCE[0] += 1
    run_id = f"run_{task_id}_20260929_0317{_RUN_SEQUENCE[0]:02d}"
    return await autonomy._record_failure(
        task, task_id, run_id,
        now.isoformat(), now - datetime.timedelta(seconds=duration),
        summary=f"{type(exc).__name__}: {exc}",
        body=f"## Error\n\n```\n{type(exc).__name__}: {exc}\n```",
        kind=autonomy._failure_kind_of(exc),
        extra={"exception": type(exc).__name__, "tool_errors": 0,
               "trigger": "scheduler"})


async def _refuse_every_list_tools(monkeypatch):
    """Every `tools/list` on every HTTP server refuses, and no time is waited.

    `_discovery_pause` is the seam `MCPPool.open()` waits through between
    attempts; taking the wait out keeps this file about WHO classified the
    failure and not about the retry's pacing, which
    `tests/test_mcp_pool_discovery_failure.py` owns.
    """
    counted = {"list_tools": 0}

    @asynccontextmanager
    async def _session(self, cfg):
        # The transport itself is not the subject: a real `sse_client` against a
        # dead port raises inside its own TaskGroup before `tools/list` is ever
        # reached, which would replace the discovery failure with a connect
        # error. The session is the thing the restart makes unusable, so it is
        # handed over, and the failure is put where the restart puts it.
        yield object()

    async def _refuse(self, server_name, session):
        counted["list_tools"] += 1
        raise ConnectionRefusedError(
            "connection refused on 'POST http://127.0.0.1:1/mcp'")

    async def _no_wait(seconds):
        return None

    monkeypatch.setattr(MCPPool, "_http_session", _session, raising=True)
    monkeypatch.setattr(MCPPool, "_list_tools", _refuse, raising=True)
    monkeypatch.setattr(mcp_pool_mod, "_discovery_pause", _no_wait, raising=True)
    monkeypatch.setattr(mcp_pool_mod, "_POOL_CACHE", {})
    return counted


@pytest.mark.asyncio
async def test_a_discovery_death_from_the_real_raise_site_is_charged_to_infra(
        env, monkeypatch):
    """Clauses 3 and 5 together, because separating them is what let this ship.

    The exception is not constructed here. A dead server is handed to
    `_open_pool_for_run` — the same call `run_query` makes before a turn exists
    — and `MCPPool.open()` raises, so the type, the message, the `.servers` list
    and the traceback are all the product's. Then the exception goes through the
    same handler a scheduler dispatch sends it through. Two assertions carry the
    clause: `failure_kind: infra` on the record, and `failure_count` STILL 1 —
    the value the task already carried, not one less than it.
    """
    # If the canonical default ever stops being an HTTP transport, this file
    # would be testing the stdio branch, which raises the same error class for a
    # reason unrelated to discovery. Fail here instead of asserting nothing.
    assert DEAD_SERVER[SERVER]["type"] in mcp_pool_mod.HTTP_TRANSPORT_TYPES, (
        f"{DEAD_SERVER[SERVER]['type']!r} is not a discovery transport")
    counted = await _refuse_every_list_tools(monkeypatch)

    with pytest.raises(ToolDiscoveryError) as raised:
        await _open_pool_for_run(
            RunOptions(model="auto", base_url="http://127.0.0.1:8096",
                       mcp_servers=DEAD_SERVER))
    exc = raised.value
    # The HTTP discovery path was taken and used up — the count is the proof. A
    # config the pool mis-classifies as stdio raises the same class without ever
    # calling `tools/list`, and would satisfy every other assertion here.
    assert counted["list_tools"] == mcp_pool_mod.DISCOVERY_MAX_ATTEMPTS, (
        f"discovery was not what failed: {counted}")
    # Proof the death came from the discovery code and not from this file: one
    # frame of its traceback is `mcp_pool.py`, and the servers it names are the
    # ones this test passed in.
    frames = []
    tb = exc.__traceback__
    while tb is not None:
        frames.append(tb.tb_frame.f_code.co_filename)
        tb = tb.tb_next
    assert any(f.endswith("mcp_pool.py") for f in frames), (
        "the error must come out of the discovery code, not be raised by hand")
    assert list(exc.servers) == [SERVER]

    assert _read_task(env, 53)["failure_count"] == 1, "probe needs a non-zero start"

    await charge(env, exc)

    record = newest_record(env, 53)
    assert record["exception"] == "ToolDiscoveryError"
    assert record["failure_kind"] == "infra", (
        "a discovery death charged to the task is the bug this item is about")
    assert record["failure_count"] == 1, (
        "an infra failure must carry the count it found, not one spent against it")
    assert "MCP discovery yielded no tools" in record["summary"]

    fields = task_fields(env, 53)
    assert fields["failure_count"] == 1, "the task's counter must not move"
    assert fields["status"] == "up_next", (
        "a discovery death must never be what retires a scheduled task")
    assert fields["infra_failure_count"] == 1, (
        "the infra counter is where an outage's cost belongs (#1085)")
    assert fields["last_attempt"], "the death still has to be visible as an attempt"


@pytest.mark.asyncio
async def test_a_task_side_exception_at_the_same_seam_still_spends_the_counter(
        env):
    """The counterparty, at the identical call — so clause 3 cannot be a blanket.

    Everything above is an infrastructure story. This is a `RuntimeError` from
    inside a task's own code, charged by the same `charge()` helper: it must
    still classify as `task`, still move `failure_count` 1 → 2, and on the run
    that reaches `max_retries` still reach `status: failed`. Reclassifying
    discovery must not have bought a way for a real failure to go uncounted.
    """
    await charge(env, RuntimeError("skill script exited 1"), duration=412.0)

    fields = task_fields(env, 53)
    record = newest_record(env, 53)
    assert record["failure_kind"] == "task"
    assert record["failure_count"] == 2
    assert fields["failure_count"] == 2, "a task failure must still be counted"
    assert fields["status"] == "up_next", "one short of max_retries is not retired"

    await charge(env, RuntimeError("skill script exited 1"), duration=412.0)

    fields = task_fields(env, 53)
    assert fields["failure_count"] == 3
    assert fields["status"] == "failed", (
        "`max_retries` must still retire a task that really is broken")
    # A disabled task has no next dispatch at all: `_update_task_field` CLEARS a
    # key it is handed as None rather than writing `next_run: null`, so absence
    # is the shape of "nothing scheduled".
    assert "next_run" not in fields, fields


@pytest.mark.asyncio
async def test_discovery_deaths_cannot_reach_status_failed_however_many_stack_up(
        env):
    """An outage may be charged as often as it happens and still not disable.

    Five consecutive discovery deaths, against a task whose `max_retries` is 3 —
    the shape of a landing that stalled, or an MCP unit that came back broken and
    was dispatched at every tick meanwhile. `kind == "task"` owned the disable and
    still does; what the infra path earns instead is #1085's ceiling and a rest
    that clears itself, so nothing here needs a human to re-enable the schedule.
    """
    exc = ToolDiscoveryError(f"{SERVER}: MCP discovery yielded no tools")
    for _ in range(5):
        await charge(env, exc)

    fields = task_fields(env, 53)
    assert fields["status"] == "up_next"
    assert fields["failure_count"] == 1, "five charged deaths, zero counter movement"
    assert fields["last_run"] == DAY_AGO, (
        "`last_run` is a success field; a failure must not forge one")
    assert int(fields["infra_failure_count"]) < 5, (
        "the infra ceiling should have rested it by now, and reset the counter")
    assert fields["next_run"], "a resting task still has a time it comes back"


@pytest.mark.asyncio
async def test_the_fleet_counts_a_discovery_death_as_infra_not_as_a_task_failure(
        env, monkeypatch):
    """Clause 3's aggregation half: what `compute_health` does with the record.

    One task, three rows, built from the front matter `_record_failure` really
    wrote: a healthy 45-minute run, a 13-second discovery death, and a 900-second
    failure of the task's own. `failures` and `fail_rate` are asserted UNCHANGED
    — a death still spent its wall clock, and hiding that would be the
    `wasted_hours` blind spot #1688 was filed about — while the new split carries
    the distinction the total cannot make: two of these three rows are not the
    task's fault, and one of them is.
    """
    autonomy = env.autonomy
    await _refuse_every_list_tools(monkeypatch)
    with pytest.raises(ToolDiscoveryError) as raised:
        await _open_pool_for_run(
            RunOptions(model="auto", base_url="http://127.0.0.1:8096",
                       mcp_servers=DEAD_SERVER))
    await charge(env, raised.value)
    discovery_meta = newest_record(env, 53)
    await charge(env, RuntimeError("skill script exited 1"), duration=900.0)
    task_meta = newest_record(env, 53)

    def row(status, duration, meta):
        return {
            "task_id": 53, "run_id": f"run_53_{duration}", "status": status,
            "started_at": DAY_AGO, "completed_at": DAY_AGO,
            "duration_seconds": duration, "worker_id": "w", "exit_status": "done",
            "response_json": ("null" if status == "failed" else
                              '"the run wrote its note"'),
            # JSON, because `compute_health` reads this column with
            # `json.loads` — the shape the pool writes. YAML here would fail that
            # parse, land as an empty meta, and every row would then default to
            # `failure_kind: task`.
            "meta_json": json.dumps(meta),
        }

    rows = [
        row("success", 2700.0, {"trigger": "scheduler"}),
        # The pool stores the record's front matter as the row's meta; these are
        # those fields, read back off the file the failure path wrote above.
        row("failed", 13.0, {k: discovery_meta[k] for k in
                             ("failure_kind", "failure_count", "exception")}),
        row("failed", 900.0, {k: task_meta[k] for k in
                              ("failure_kind", "failure_count", "exception")}),
    ]
    out = autonomy.compute_health(rows, [_read_task(env, 53)], days=7)

    # The id as the aggregation itself stores it, so this lookup cannot disagree
    # with the grouping about whether an id is an int or a string.
    tid = autonomy._row_task_id(rows[0])
    entry = next(t for t in out["tasks"] if t["task_id"] == tid)
    assert entry["runs"] == 3
    assert entry["failures"] == 2, (
        "the total still counts every non-success run: a death spent its clock")
    assert entry["fail_rate"] == pytest.approx(2 / 3, abs=1e-3), (
        "the definition of fail_rate is unchanged by this item")
    assert entry["task_failures"] == 1
    assert entry["infra_failures"] == 1
    # Both failed rows' clock, because `wasted_hours` is the burn the fleet
    # actually paid and an infra death paid it too. Rounded to hundredths, which
    # is how the entry emits the field.
    assert entry["wasted_hours"] == pytest.approx(
        round((13.0 + 900.0) / 3600.0, 2), abs=0.005), (
        f"failed rows must still be charged as wasted: {entry['wasted_hours']}")
    assert entry["task_failures"] + entry["infra_failures"] == entry["failures"], (
        "the split is exhaustive by construction")
    assert out["fleet"]["task_failures"] == 1
    assert out["fleet"]["infra_failures"] == 1
    assert out["fleet"]["failures"] == 2


@pytest.mark.asyncio
async def test_a_discovery_death_holds_a_downstream_chain_just_as_a_task_failure_does(
        env):
    """The invariant the item says NOT to "fix": the dependency reading is uniform.

    `54` depends on `53`. `_record_failure` writes `last_attempt` for BOTH kinds
    and `last_run` for NEITHER, and `_is_dependency_met` reads only `last_run` —
    so a chain held by a discovery death is held exactly as one held by a task
    failure, which is how a 09-26 dead run held a downstream chain for 4 days.
    Recategorising must not have loosened it here, because `last_run` means "the
    upstream's OUTPUT is on disk" and a 13-second death produced no output —
    forwarding on it would run the digest over yesterday's input.
    """
    exc = ToolDiscoveryError(f"{SERVER}: MCP discovery yielded no tools")
    await charge(env, exc)

    autonomy = env.autonomy
    dep = _read_task(env, 54)
    held_after_infra = autonomy._is_dependency_met(
        dep, [_read_task(env, 53), dep],
        now=datetime.datetime.now(datetime.timezone.utc))
    assert task_fields(env, 53)["last_run"] == DAY_AGO, (
        "an infra death must not write the field the chain reads")

    # Same instant, same staleness, a failure of the task's own instead.
    await charge(env, RuntimeError("skill script exited 1"), duration=900.0)
    dep = _read_task(env, 54)
    held_after_task = autonomy._is_dependency_met(
        dep, [_read_task(env, 53), dep],
        now=datetime.datetime.now(datetime.timezone.utc))

    assert held_after_infra is held_after_task is False, (
        "the dependent is held by the missing output, whichever kind the death "
        "was filed under; a difference here means the two paths diverged")


# ── #2037: the same two paths, for a death the pool saw and `run_task` never did ──
#
# `scheduled_task.execute()` can raise before `run_task` is ever called — on one
# of its own imports, or on its `RuntimeError: model server … unhealthy —
# deferring task` after the 90 s health wait. Such a death reaches neither
# `run_task`'s `except` nor `_record_failure`, so the split these tests exist for
# had nothing to split: nothing was counted at all. `charge_death_without_verdict`
# is the pool's route into the SAME two paths, and what follows pins that it
# changed no policy — `_INFRA_CEILING` and `_INFRA_EXC_NAMES` are read as they
# are, not restated next to a second copy of themselves.


def _death_run_id(kind: str, n: int) -> str:
    """A run id for one charged death; the record's filename IS the run id."""
    _RUN_SEQUENCE[0] += 1
    return (f"run_scheduled-task_20261001_{_RUN_SEQUENCE[0]:06d}_{kind}{n}")


async def _charge_death(env, exc, *, task_id: int = 53, kind: str = "d",
                        n: int = 0) -> dict:
    """One death charged the way the pool charges it, and what it returned."""
    now = datetime.datetime.now(datetime.timezone.utc)
    return await env.autonomy.charge_death_without_verdict(
        task_id, exc, run_id=_death_run_id(kind, n), started_at=now.isoformat())


@pytest.mark.asyncio
async def test_a_pool_death_classified_infra_leaves_the_retry_budget_alone(env):
    """Clause 3, infra half: `ConnectionResetError` is still not the task's fault.

    The name is in `_INFRA_EXC_NAMES`, so the escaping-exception route must land
    on the infra path exactly as `run_task`'s own handler does: the infra counter
    moves, `failure_count` does not. Task #53 starts at `failure_count: 1` for
    exactly this reason — a held count and a zeroed one are the same number 0.
    """
    before = task_fields(env, 53)
    result = await _charge_death(env, ConnectionResetError(
        "[Errno 111] Connect called on a closed transport"))

    assert result["failure_kind"] == "infra"
    held = before["failure_count"]
    assert held >= 1, (
        f"the fixture holds failure_count {held}: against a zero, `unchanged` and "
        "`never charged anything` are the same number, and this node would pass "
        "on a route that silently charges nothing at all")
    after = task_fields(env, 53)
    assert after["failure_count"] == held, (
        "an infrastructure death spent the task's retry budget, which is the "
        "fleet-wide disable #1085's split exists to prevent")
    assert after["infra_failure_count"] == 1, (
        "the death was not booked as infra either, so it is invisible again")


@pytest.mark.asyncio
async def test_a_pool_death_the_classifier_cannot_name_spends_the_budget(env):
    """Clause 3, task half: an unclassifiable death now costs the task exactly 1.

    `ModuleNotFoundError` — task #74's actual 2026-09-29 death, 40 rows of it —
    is on no list, and 41 runs of it moved nothing anywhere. This is the whole
    point of the change: the genuinely unclassifiable death was the free one.
    """
    before = task_fields(env, 53)
    result = await _charge_death(env, ModuleNotFoundError(
        "No module named 'app.discord_notify'"))

    assert result["failure_kind"] == "task"
    after = task_fields(env, 53)
    assert after["failure_count"] == before["failure_count"] + 1 == 2, (
        f"failure_count went {before['failure_count']} → {after['failure_count']}")
    assert "infra_failure_count" not in after, (
        "a task death was booked against the infra counter as well")


@pytest.mark.asyncio
async def test_the_pool_death_route_reads_the_live_infra_name_set(env, monkeypatch):
    """Clause 3's "changes no classification" half, stated falsifiably.

    The name set is widened IN THE TEST to hold a name that is not on it at HEAD,
    and the pool's route has to follow. It can only follow if the route asks
    `_failure_kind_of`, which reads the module global; a route carrying its own
    copy of the list answers the old way and this fires. Restoring the set is
    monkeypatch's job, not this test's.
    """
    monkeypatch.setattr(env.autonomy, "_INFRA_EXC_NAMES",
                        frozenset({"ModuleNotFoundError"}))
    before = task_fields(env, 53)

    result = await _charge_death(env, ModuleNotFoundError("No module named 'x'"))

    assert result["failure_kind"] == "infra", (
        "the route classified from a list of its own, not from `_INFRA_EXC_NAMES`")
    assert task_fields(env, 53)["failure_count"] == before["failure_count"], (
        "widening the infra set did not move the route, so it is not the set "
        "`run_task` and this path share")


@pytest.mark.asyncio
async def test_pool_deaths_rest_a_task_only_at_the_unmodified_infra_ceiling(env):
    """The ceiling governs the new route because the new route has none of its own.

    The loop counts to `autonomy._INFRA_CEILING` and reads the constant rather
    than a number: what is asserted is that the resting state arrives on the
    LAST of those deaths and on none before it. A route that spent the retry
    budget instead, or invented its own bound, fails here — and a change to the
    constant moves this test instead of silently moving the route.
    """
    autonomy = env.autonomy
    ceiling = autonomy._INFRA_CEILING
    held = task_fields(env, 53)["failure_count"]
    assert held >= 1, (
        f"the fixture holds failure_count {held}, so the closing `the budget did "
        "not move` below would be true of a task that had none to move")

    for i in range(ceiling - 1):
        await _charge_death(env, ConnectionResetError(f"hiccup {i}"), kind="c", n=i)
        fm = task_fields(env, 53)
        assert "infra_rest_until" not in fm, (
            f"the task rested after {i + 1} of {ceiling} infra deaths")
        assert fm.get("infra_failure_count") == i + 1, (
            "the consecutive count did not accumulate, so the ceiling cannot bind")

    await _charge_death(env, ConnectionResetError("one past the ceiling"),
                        kind="c", n=99)

    fm = task_fields(env, 53)
    assert fm.get("infra_rest_until"), (
        f"{ceiling} consecutive infra deaths charged from the pool never crossed "
        "the ceiling, so this route re-dispatches an outage forever")
    assert fm.get("infra_failure_count") == 0, (
        "the crossing left a spent counter armed for a second rest")
    assert fm["failure_count"] == held, (
        f"the budget moved {held} → {fm['failure_count']} across {ceiling} infra "
        "deaths: this route reached the retry budget after all")
