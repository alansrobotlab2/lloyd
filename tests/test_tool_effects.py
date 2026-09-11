"""#544 — the effect ledger: exactly-once *effect*, not exactly-once scheduling.

Two shapes are pinned here.

The first is the one the pool makes real: an item is cancelled at
`max_duration_seconds`, recorded failed, requeued, and re-run from zero, so
every write the first attempt landed would land again. The forced-timeout test
below drives that loop for three attempts and asserts the effect happened once.

The second is the state a cancel leaves behind. `unknown` is not a synonym for
`error`: `error` means the tool answered that it failed and a retry is safe,
while `unknown` means nobody knows, and firing again is how a duplicate email
gets sent. The refusal branch is asserted to fire nothing at all.

A note on what the tests deliberately do NOT do: the ledger database is the
work queue's own file, so the fixtures point `LLOYD_EFFECT_LEDGER_DB` at the
same tmp file as `WorkQueue` rather than a second one — that shared-file
property is part of the design (the backend reads the suppression counter off
the file the aggregator writes) and a fixture that quietly splits them would
test a system that does not exist.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import sqlite3
from types import SimpleNamespace

import pytest

import agent_mcp.main as M
from agent_mcp import _tool_effects as TE
from agent_mcp import annotations as A
from app.harness import policy
from app.harness import mcp_pool
from workers.pool import WorkerPool, effect_scope_for, grant_scope_for
from workers.queue import QueueItem, WorkQueue

SCOPE = "item:scheduled-task:7"
ARGS = {"to": "someone@example.com", "subject": "digest", "body": "hello"}


class FakeWriter:
    """A side-effecting tool module: counts how many times it actually ran."""

    def __init__(self, effects: list, *, mode: str = "ok"):
        self.effects = effects
        self.mode = mode

    async def call_tool(self, name, arguments):
        self.effects.append(dict(arguments))
        if self.mode == "raise":
            raise RuntimeError("handler died mid-effect")
        if self.mode == "is_error":
            return M.CallToolResult(
                content=[M.TextContent(type="text", text=json.dumps(
                    {"error": "upstream said no"}))],
                isError=True,
            )
        return [M.TextContent(type="text", text=json.dumps({"filed": 42}))]


@pytest.fixture
def q(tmp_path) -> WorkQueue:
    return WorkQueue(tmp_path / "workers.db")


@pytest.fixture(autouse=True)
def ledger(tmp_path, monkeypatch):
    """Point the ledger at this test's own file.

    `_init_done` caches resolved paths so a process-wide test run does not
    re-create the table per call; it must be cleared when the path moves.
    """
    db = tmp_path / "workers.db"
    monkeypatch.setenv("LLOYD_EFFECT_LEDGER_DB", str(db))
    TE._init_done.clear()
    yield db
    TE._init_done.clear()


def _register(monkeypatch, mod) -> None:
    base = dict(getattr(M, "_dispatch", None) or {})
    base["fake_writer"] = mod
    monkeypatch.setattr(M, "_dispatch", base)


def _rows(db) -> list[sqlite3.Row]:
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        return list(conn.execute(
            "SELECT tool, scope, status, suppress_count FROM tool_effects"))
    finally:
        conn.close()


def _err(result) -> bool | None:
    """`isError` across the mcp 1.x/2.x split — same reason as
    `agent_mcp.main._result_is_error`: a camelCase read on a 2.x
    `CallToolResult` raises rather than returning False."""
    for attr in ("is_error", "isError"):
        value = getattr(result, attr, None)
        if value is not None:
            return bool(value)
    return None


def _texts(result) -> str:
    blocks = result.content if hasattr(result, "content") else (result or [])
    return "\n".join(b.text for b in blocks)


# ── the ledger is not on the read-only path ────────────────────────────────


async def test_a_read_only_call_never_touches_the_ledger(ledger):
    """`vault_read` twice is two reads. No row, no key, not even a file.

    The acceptance check is that the ledger is off their path entirely, so the
    assertion is about the database file's non-existence, not about a row count.
    """
    claim = await TE.claim("vault_read", {"path": "a.md"}, SCOPE)
    assert claim.key == "" and claim.may_dispatch
    assert not ledger.exists(), "a read created a ledger file"


async def test_a_repeat_expected_tool_is_not_ledgered(ledger):
    """`Bash("sleep 30 && supervisorctl status")` repeated IS the job.

    Replaying a stored observation there would hand a worker a stale answer
    dressed as a fresh one, which is worse than the duplicate this guards.
    """
    claim = await TE.claim("Bash", {"command": "ls"}, SCOPE)
    assert claim.key == "" and claim.may_dispatch


def test_the_classification_comes_from_annotations_not_a_second_list():
    assert not A.side_effecting("vault_read")
    assert not A.side_effecting("email_read")
    assert A.side_effecting("email_send")
    assert A.side_effecting("backlog_write_task")
    assert A.side_effecting("fact_add")
    # An unclassified tool is guarded, not waved through — the same default
    # `annotations_for` gives plan mode.
    assert A.side_effecting("some_tool_nobody_classified")
    # A tool may not be in both lists; REPEAT_EXPECTED would silently win.
    assert not (A.READ_ONLY & A.REPEAT_EXPECTED)
    # Acceptance clause (d): IDEMPOTENT is the classification source too. A
    # setter or a delete repeated adds no further change, so there is nothing
    # for the ledger to prevent and a replay can only be staler than the call.
    for name in ("Write", "vault_write", "graph_refresh", "email_delete",
                 "calendar_update_event", "memory_replace"):
        assert name in A.IDEMPOTENT and not A.side_effecting(name), name
    # And `Edit` is repeat-expected: its own `old_string` match is the check.
    assert "Edit" in A.REPEAT_EXPECTED and not A.side_effecting("Edit")


async def test_an_edit_that_flips_a_file_back_and_forth_is_never_replayed(
        monkeypatch, ledger, tmp_path):
    """The A→B→A→B probe that found the first cut's worst defect.

    Run against the live aggregator with a scratch ledger, the third call — an
    exact repeat of the first — was answered "Edited … (1 replacement)" while
    the file stayed at A. A replayed success is a silent revert the model
    cannot see; the handler's own "old_string not found" it can. So `Edit`
    dispatches every time and the ledger never has a row for it.
    """
    target = tmp_path / "probe.txt"
    target.write_text("colour = A\n")

    class EditLike:
        async def call_tool(self, name, arguments):
            text = target.read_text()
            if arguments["old_string"] not in text:
                return M.CallToolResult(content=[M.TextContent(
                    type="text", text=json.dumps({"error": "old_string not found"}))],
                    isError=True)
            target.write_text(text.replace(arguments["old_string"],
                                           arguments["new_string"], 1))
            return [M.TextContent(type="text", text=f"Edited {target} (1 replacement)")]

    base = dict(getattr(M, "_dispatch", None) or {})
    base["Edit"] = EditLike()
    monkeypatch.setattr(M, "_dispatch", base)
    meta = {M.META_EFFECT_SCOPE: SCOPE}
    for old, new in (("A", "B"), ("B", "A"), ("A", "B")):
        await M.call_tool("Edit", {"file_path": str(target), "old_string": f"colour = {old}",
                                   "new_string": f"colour = {new}"}, meta)
    assert target.read_text() == "colour = B\n", "the third edit was replayed, not applied"
    assert not ledger.exists() or _rows(ledger) == []


async def test_the_kill_switch_turns_the_ledger_off_without_breaking_a_call(
        monkeypatch, ledger):
    """`harness.effect_ledger.enabled: false` is the lever for the day a
    legitimate second call is refused. Off means no row and no replay — the
    call dispatches like a pre-#544 call did."""
    from app.config import CONFIG
    effects: list = []
    _register(monkeypatch, FakeWriter(effects))
    harness = dict(CONFIG.get("harness") or {})
    harness["effect_ledger"] = {"enabled": False}
    monkeypatch.setitem(CONFIG, "harness", harness)
    assert not TE.enabled()
    for _ in range(2):
        await M.call_tool("fake_writer", dict(ARGS), {M.META_EFFECT_SCOPE: SCOPE})
    assert len(effects) == 2
    assert not ledger.exists()
    assert TE.config()["retention_days"] == 14, "defaults survive a partial block"


def test_prune_keeps_unknown_rows_on_the_longer_clock(ledger, monkeypatch):
    """An `unknown` row is the only record that an effect may have landed.
    Pruning it on the settled rows' clock silently re-arms the duplicate."""
    from datetime import datetime, timedelta, timezone
    TE._ensure(TE._connect())
    old = (datetime.now(timezone.utc) - timedelta(days=20)).isoformat(timespec="seconds")
    ancient = (datetime.now(timezone.utc) - timedelta(days=40)).isoformat(timespec="seconds")
    conn = TE._connect()
    for key, status, ts in (("k_ok_old", "ok", old), ("k_unknown_old", "unknown", old),
                            ("k_unknown_ancient", "unknown", ancient),
                            ("k_ok_fresh", "ok", TE._now())):
        conn.execute("INSERT INTO tool_effects (effect_key, tool, scope, status, created_at,"
                     " updated_at) VALUES (?,?,?,?,?,?)", (key, "t", SCOPE, status, ts, ts))
    conn.close()
    out = TE.prune()
    assert out == {"settled": 1, "unknown": 1}
    left = {r[0] for r in sqlite3.connect(str(ledger)).execute(
        "SELECT effect_key FROM tool_effects")}
    assert left == {"k_unknown_old", "k_ok_fresh"}


async def test_a_suppression_is_logged_with_the_arguments_that_produced_it(
        monkeypatch, ledger, caplog):
    """A key prefix names nothing a human can grep for. The docstring promised
    the arguments; the first cut logged the prefix only."""
    effects: list = []
    _register(monkeypatch, FakeWriter(effects))
    for _ in range(2):
        await M.call_tool("fake_writer", dict(ARGS), {M.META_EFFECT_SCOPE: SCOPE})
    lines = [r.getMessage() for r in caplog.records if "suppressed duplicate" in r.getMessage()]
    assert lines and "someone@example.com" in lines[-1]


async def test_no_scope_means_no_ledger(ledger):
    """An interactive turn is unguarded in this first cut, by decision."""
    claim = await TE.claim("email_send", ARGS, "")
    assert claim.key == ""
    assert not ledger.exists()


# ── the key ────────────────────────────────────────────────────────────────


def test_the_key_survives_argument_ordering_and_changes_with_scope():
    a = TE.effect_key("email_send", {"to": "x", "body": "y"}, SCOPE)
    b = TE.effect_key("email_send", {"body": "y", "to": "x"}, SCOPE)
    assert a == b, "dict ordering must not make a fresh effect"
    assert a != TE.effect_key("email_send", {"to": "x", "body": "z"}, SCOPE)
    assert a != TE.effect_key("email_send", {"to": "x", "body": "y"},
                            "item:scheduled-task:8")
    assert a != TE.effect_key("calendar_create", {"to": "x", "body": "y"}, SCOPE)


def test_the_meta_keys_agree_on_both_sides_of_the_seam():
    """The two halves ship together and talk over HTTP: a renamed key does not
    error, it just silently un-guards every effect in the fleet."""
    assert M.META_EFFECT_SCOPE == mcp_pool.META_EFFECT_SCOPE == "lloyd/effect_scope"
    assert policy.current_effect_scope.get() == ""


# ── the ok / error / unknown state machine ─────────────────────────────────


async def test_a_second_identical_effect_is_replayed_not_refired(monkeypatch, ledger):
    effects: list = []
    _register(monkeypatch, FakeWriter(effects))

    first = await M.call_tool("fake_writer", dict(ARGS), {M.META_EFFECT_SCOPE: SCOPE})
    assert len(effects) == 1 and '"filed": 42' in _texts(first)

    second = await M.call_tool("fake_writer", dict(ARGS), {M.META_EFFECT_SCOPE: SCOPE})
    assert len(effects) == 1, "the effect fired twice inside one scope"
    body = _texts(second)
    assert '"filed": 42' in body, "the replay must carry the original result"
    assert "already fired in this scope" in body, "and say so"
    assert _err(second) is False
    assert TE.suppressed_total() == 1
    rows = _rows(ledger)
    assert [r["status"] for r in rows] == ["ok"]
    assert rows[0]["suppress_count"] == 1


async def test_a_different_call_in_the_same_scope_still_fires(monkeypatch, ledger):
    """Over-suppression is the failure that bites: one scope, many effects."""
    effects: list = []
    _register(monkeypatch, FakeWriter(effects))
    for body in ("first", "second", "third"):
        await M.call_tool("fake_writer", {**ARGS, "body": body},
                          {M.META_EFFECT_SCOPE: SCOPE})
    assert len(effects) == 3
    assert len(_rows(ledger)) == 3
    assert TE.suppressed_total() == 0


async def test_a_failed_answer_leaves_the_effect_retriable(monkeypatch, ledger):
    """`error` is the state where a retry is exactly right: the tool answered,
    and answered that it did not do the thing."""
    effects: list = []
    _register(monkeypatch, FakeWriter(effects, mode="is_error"))
    await M.call_tool("fake_writer", dict(ARGS), {M.META_EFFECT_SCOPE: SCOPE})
    assert [r["status"] for r in _rows(ledger)] == ["error"]
    await M.call_tool("fake_writer", dict(ARGS), {M.META_EFFECT_SCOPE: SCOPE})
    assert len(effects) == 2, "an answered failure must not wedge the scope"


async def test_a_handler_that_dies_leaves_the_effect_unknown(monkeypatch, ledger):
    effects: list = []
    _register(monkeypatch, FakeWriter(effects, mode="raise"))
    with pytest.raises(RuntimeError):
        await M.call_tool("fake_writer", dict(ARGS), {M.META_EFFECT_SCOPE: SCOPE})
    # `error` would license a retry to fire a second effect. The true state is
    # unknown: the handler may have written before it died.
    assert [r["status"] for r in _rows(ledger)] == ["unknown"]


async def test_an_unknown_effect_refuses_and_fires_nothing(monkeypatch, ledger):
    """The whole reason the pre-write exists.

    A prior attempt was cancelled with this effect in flight — the case Munaf
    calls out by name: the server may have written while the caller was handed
    an error. Nothing may fire until somebody looks.
    """
    effects: list = []
    _register(monkeypatch, FakeWriter(effects, mode="raise"))
    # Establish the unknown row the only way it happens in production: a
    # handler that dies.
    with pytest.raises(RuntimeError):
        await M.call_tool("fake_writer", dict(ARGS), {M.META_EFFECT_SCOPE: SCOPE})
    assert len(effects) == 1

    refused = await M.call_tool("fake_writer", dict(ARGS), {M.META_EFFECT_SCOPE: SCOPE})
    # Zero effects from the refused call: the count did not move past the one
    # the dying handler may or may not have landed.
    assert len(effects) == 1, "an unknown effect was re-fired"
    assert _err(refused) is True
    payload = json.loads(_texts(refused))
    assert payload["effect_state"] == "unknown"
    assert "do_not" in payload and "next_step" in payload
    assert "re-fire" in payload["do_not"]
    assert TE.suppressed_total() == 1


async def test_the_ledger_fails_open_when_the_database_is_unusable(monkeypatch, tmp_path):
    """A ledger that can brick every write in the fleet is a worse outage than
    the duplicate email it exists to prevent."""
    effects: list = []
    _register(monkeypatch, FakeWriter(effects))
    monkeypatch.setenv("LLOYD_EFFECT_LEDGER_DB", "/proc/cannot/write/this.db")
    out = await M.call_tool("fake_writer", dict(ARGS), {M.META_EFFECT_SCOPE: SCOPE})
    assert len(effects) == 1, "the guard blocked a call it could not record"
    assert '"filed": 42' in _texts(out)
    assert TE.suppressed_total() == 0


# ── the pool: the retry that motivated all of this ─────────────────────────


def _release_backoff(q: WorkQueue) -> None:
    """Let the item be claimable again now.

    `mark_failed` backs off 30s then 60s; that is a clock, not the mechanism
    under test, and a 90-second unit test is not worth its wall time.
    """
    with q._lock, q._connect() as conn:
        conn.execute("UPDATE queue SET not_before=NULL")
        conn.commit()


def test_the_effect_scope_is_the_queue_item_not_the_run_or_the_grant():
    item = QueueItem(id=7, source="scheduled-task", kind="k", priority=50,
                     payload={"task_id": 39}, dedup_key=None, state="queued",
                     attempts=0, enqueued_at="", claimed_at=None, claimed_by=None,
                     completed_at=None, error=None)
    assert effect_scope_for(item) == "item:scheduled-task:7"
    assert effect_scope_for(item) != grant_scope_for(item) == "autonomy-task:39"
    # The run id is minted per attempt, which is exactly why it cannot be key
    # on: attempt 2 would look like a brand-new effect.
    assert "run" not in effect_scope_for(item)


async def test_a_timed_out_retry_fires_the_effect_once(q, tmp_path, monkeypatch):
    """The reproducer the acceptance check names: a source that fires one
    side-effecting tool and then overruns `max_duration_seconds`, run through a
    real pool until the item is poisoned.

    53 `scheduled-task` items reached `attempts>=2` in the 8 days before this
    was filed, and every one of them re-ran its writes from zero.
    """
    import workers.sources as sources

    effects: list = []
    _register(monkeypatch, FakeWriter(effects))

    async def execute(item):
        # Exactly what `app/harness/loop.py` does: read the scope the pool
        # bound for this job and hand it to the aggregator in `_meta`.
        meta = {M.META_EFFECT_SCOPE: policy.current_effect_scope.get()}
        await M.call_tool("fake_writer", dict(ARGS), meta)
        await asyncio.sleep(30)     # overrun the 1s cap below

    monkeypatch.setattr(sources, "SOURCE_REGISTRY",
                        {"s": SimpleNamespace(NAME="s", execute=execute)},
                        raising=False)
    monkeypatch.setattr(sources, "get_sources_config",
                        lambda: {"s": {"max_duration_seconds": 1}}, raising=False)

    q.enqueue("s", "k")
    pool = WorkerPool(q, slots=1, poll_idle_seconds=0.01)
    await pool.start()
    try:
        for _ in range(300):
            state = q.get(1).state
            if state == "poisoned":
                break
            if state == "queued":
                _release_backoff(q)
            await asyncio.sleep(0.02)
    finally:
        await pool.stop()

    row = q.get(1)
    assert row.state == "poisoned", f"item never exhausted its attempts: {row.state}"
    assert row.attempts == 3, "the retry the acceptance check names"
    assert len(q.list_runs(limit=10)) == 3
    assert len(effects) == 1, (
        f"the effect fired {len(effects)} times across 3 attempts")
    assert [r["status"] for r in _rows(tmp_path / "workers.db")] == ["ok"]
    assert TE.suppressed_total() == 2, "attempts 2 and 3 were both refused"


async def test_a_caller_without_a_scope_gets_no_ledger(monkeypatch, ledger):
    """A caller outside the pool — a bare `run_query`, an old harness that
    sends no `_meta` scope — must not find its writes swallowed by an
    empty-scope key collision. Two identical calls with **no** scope both
    fire and leave no row.

    (The first cut of this test ran the call through a real pool — which
    binds a scope, so it tested the opposite path — and then asserted
    `_rows(...) == [] or True`.)
    """
    effects: list = []
    _register(monkeypatch, FakeWriter(effects))
    for _ in range(2):
        await M.call_tool("fake_writer", dict(ARGS), {})
        await M.call_tool("fake_writer", dict(ARGS), None)
    assert len(effects) == 4
    assert not ledger.exists()
    assert TE.suppressed_total() == 0


async def test_a_task_subagents_writes_inherit_the_items_scope(monkeypatch, ledger):
    """A `Task` re-enters `call_tool` over loopback `/mcp` from a fresh ASGI
    task, so only `_meta` crosses — and the subagent's own loop stamps `_meta`
    from `policy.current_effect_scope`. The aggregator therefore binds the
    incoming scope around the dispatch. Modelled here by a `Task` handler that
    does what the nested loop does: read the contextvar and hand it on."""
    effects: list = []
    writer = FakeWriter(effects)

    class TaskLike:
        async def call_tool(self, name, arguments):
            inner_meta = {M.META_EFFECT_SCOPE: policy.current_effect_scope.get()}
            await M.call_tool("fake_writer", dict(ARGS), inner_meta)
            return [M.TextContent(type="text", text="subagent done")]

    base = dict(getattr(M, "_dispatch", None) or {})
    base["fake_writer"] = writer
    base["Task"] = TaskLike()
    monkeypatch.setattr(M, "_dispatch", base)

    await M.call_tool("Task", {"prompt": "go"}, {M.META_EFFECT_SCOPE: SCOPE})
    await M.call_tool("Task", {"prompt": "go"}, {M.META_EFFECT_SCOPE: SCOPE})
    assert len(effects) == 1, "the subagent's second identical write fired"
    rows = _rows(ledger)
    assert [(r["tool"], r["scope"]) for r in rows] == [("fake_writer", SCOPE)]
    # And the bind is scoped to the dispatch: nothing leaks to the caller.
    assert policy.current_effect_scope.get() == ""


def test_the_router_hands_a_scope_only_to_a_non_user_session(tmp_path, monkeypatch):
    """The loopback half of the seam. `_effect_scope_for` is the one reader
    of the payload key, and it refuses a chat session: a user re-asking for
    the same email in a new turn is a legitimate second effect, and a scope
    on that turn would have it replayed from the ledger instead."""
    from pathlib import Path
    from app.routers import messages as R

    monkeypatch.setattr(R, "SESSIONS_DIR", tmp_path)
    (tmp_path / "worker.json").write_text(json.dumps({"platform": "worker"}))
    (tmp_path / "auto.json").write_text(json.dumps({"platform": "autonomy"}))
    (tmp_path / "chat.json").write_text(json.dumps({"platform": "mission-control"}))
    (tmp_path / "bare.json").write_text(json.dumps({}))

    assert R._effect_scope_for("worker", {"effect_scope": SCOPE}) == SCOPE
    assert R._effect_scope_for("auto", {"effect_scope": SCOPE}) == SCOPE
    assert R._effect_scope_for("chat", {"effect_scope": SCOPE}) == ""
    assert R._effect_scope_for("bare", {"effect_scope": SCOPE}) == ""
    assert R._effect_scope_for("missing", {"effect_scope": SCOPE}) == ""
    assert R._effect_scope_for("worker", {}) == ""
    assert "NON_USER_PLATFORMS" in inspect.getsource(R._effect_scope_for)

    # Both RunOptions builders that serve a payload set it; the loop prefers
    # the option over the contextvar. Read the file, not the live attribute
    # (see test_structured_verdict for why).
    src = Path(R.__file__).read_text()
    assert src.count("_effect_scope_for(session_id, data)") == 2
    loop_src = Path(policy.__file__).with_name("loop.py").read_text()
    assert 'getattr(options, "effect_scope", "")' in loop_src


def test_the_session_backed_worker_payload_carries_the_effect_scope():
    """Same hop as `grant_scope`, same reason, same guard: the pool binds the
    contextvar in its own task and the backend handles the POST in another.
    Without this line every session-backed source ran unledgered."""
    src = (policy.__file__.rsplit("/app/", 1)[0] + "/workers/sources/_common.py")
    text = open(src).read()
    assert '"effect_scope": current_effect_scope.get()' in text


# ── the counter someone has to be able to see ──────────────────────────────


async def test_the_suppression_counter_is_readable_without_a_database(tmp_path,
                                                                      monkeypatch):
    monkeypatch.setenv("LLOYD_EFFECT_LEDGER_DB", str(tmp_path / "nothing.db"))
    TE._init_done.clear()
    assert TE.suppressed_total() == 0
    assert not (tmp_path / "nothing.db").exists(), "reporting a zero created a store"


async def test_the_dashboard_reports_suppressions_next_to_the_workers(monkeypatch,
                                                                      ledger):
    from app.routers import dashboard

    assert dashboard._duplicate_effects_suppressed() == 0
    effects: list = []
    _register(monkeypatch, FakeWriter(effects))
    for _ in range(3):
        await M.call_tool("fake_writer", dict(ARGS), {M.META_EFFECT_SCOPE: SCOPE})
    assert len(effects) == 1
    assert dashboard._duplicate_effects_suppressed() == 2
