"""The quantity axis: a refilling per-session ceiling on mutating tool calls (#584).

Lloyd's other guards are per-call booleans — may this session touch this path at
all (#582), may it act under this grant (#534), has this exact effect already
happened (#544). None of them can see *how many of these has this session already
done*, and that is the axis the 2026-08-22 loss ran on: one nightly stage that
truncated `entity-aliases.json`, deleted `_relationships.json` and `memory-graph/`,
and rebuilt `facts/` wholesale — every single call individually legal, nothing
capping the bulk mid-run, and `_pipeline/` is gitignored so none of it came back.

Most of these tests drive the real dispatch path (`agent_mcp.main.call_tool`),
because the property under test is "the handler never ran", which only the
dispatch path can prove. The clock is injected — the window is an hour, and a
test that sleeps for an hour is a test that gets deleted. Every ceiling here is
tiny and explicit, so changing a production default is meant to show up as a diff
in `config.yaml`, not as a silent re-tuning of an expectation here.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from mcp.types import CallToolResult, TextContent

import agent_mcp.main as M
from agent_mcp import _mutation_budget as MB
from agent_mcp import _tool_effects as TE


#: The ceiling every dispatch test uses, via `budgets`. One number, quoted in the
#: assertions below.
CEILING = 3


def _ok(text: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)])


class FakeDeleter:
    """A tool that really removes one file its arguments name.

    Stands in for the 08-22 shape — a run deleting `facts/` files one legitimate
    call at a time — because no real delete tool in the fleet is safe to fire
    against a test tree. The class it is charged to is the class its behaviour
    puts it in: `register` below joins it to `_FS_DELETE`, so the gate is
    exercised through the same classification path a real tool travels.
    """

    def __init__(self, log: list) -> None:
        self.log = log

    async def call_tool(self, name: str, args: dict) -> CallToolResult:
        path = args["file_path"]
        self.log.append(path)
        os.remove(path)
        return _ok("deleted " + path)


class Recorder:
    """A tool that records that it ran, for the calls that must not."""

    def __init__(self, log: list) -> None:
        self.log = log

    async def call_tool(self, name: str, args: dict) -> CallToolResult:
        self.log.append(dict(args))
        return _ok("ran " + name)


def register(monkeypatch, name: str, mod, *, op_class: str | None = None) -> None:
    """Put a fake tool on the dispatch table, optionally in a named class."""
    base = dict(M._dispatch)
    base[name] = mod
    monkeypatch.setattr(M, "_dispatch", base)
    if op_class == "fs_delete":
        monkeypatch.setattr(MB, "_FS_DELETE", MB._FS_DELETE | {name})
    elif op_class == "fs_write":
        monkeypatch.setattr(MB, "_FS_WRITE", MB._FS_WRITE | {name})


def text_of(result) -> str:
    return "\n".join(getattr(p, "text", "") for p in result.content)


def ceilinged(result) -> bool:
    # `is_error`, not `isError`: the installed MCP SDK models the field as
    # `result_type`/`is_error` (pydantic), and reading the camelCase wire name
    # off the object raises AttributeError, which would make every "was it
    # denied" assertion below fail for the wrong reason.
    return bool(result.is_error) and "mutation ceiling" in text_of(result)


@pytest.fixture
def budgets(tmp_path, monkeypatch):
    """A temp ledger for each store, an injected clock, and tiny ceilings.

    The effect ledger is redirected too: `call_tool` claims effects, and a test
    that wrote into the live one would be editing the fleet's exactly-once memory.
    """
    db = tmp_path / "budget.db"
    monkeypatch.setenv("LLOYD_MUTATION_DB", str(db))
    monkeypatch.setenv("LLOYD_EFFECT_LEDGER_DB", str(tmp_path / "effects.db"))
    MB._init_done.clear()
    MB._last_prune = 0.0

    class Clock:
        def __init__(self) -> None:
            self.t = 1_800_000_000.0

        def __call__(self) -> float:
            return self.t

        def advance(self, seconds: float) -> None:
            self.t += seconds

    clock = Clock()
    monkeypatch.setattr(MB, "clock", clock)
    real_config = MB.config

    def fake_config():
        cfg = real_config()
        cfg["ceilings"] = {
            "interactive": {c: {"*": CEILING} for c in MB.OP_CLASSES},
            "*": {c: {"*": CEILING * 4} for c in MB.OP_CLASSES},
        }
        cfg["tripwire_ratio"] = 0.6
        return cfg
    monkeypatch.setattr(MB, "config", fake_config)
    yield SimpleNamespace(db=db, root=tmp_path, clock=clock, config=fake_config)
    MB._init_done.clear()


def meta_for(session_id: str = "", effect_scope: str = "") -> dict:
    out = {}
    if session_id:
        out[M.META_SESSION_ID] = session_id
    if effect_scope:
        out[M.META_EFFECT_SCOPE] = effect_scope
    return out


def call(name: str, args: dict, *, session_id: str = "", effect_scope: str = ""):
    return asyncio.run(M.call_tool(name, args,
                                   meta_for(session_id, effect_scope)))


def note(root: Path, name: str, body: str = "") -> str:
    """A file on disk that a test then asserts the bytes of."""
    path = root / "knowledge" / f"{name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return str(path)


INTERACTIVE = "20260915_120000_chat_a1b2"


# ── clause 1: N+1 calls execute N, and the N+1th handler never runs ─────────

def test_a_burst_beyond_the_ceiling_runs_n_and_the_next_handler_never_runs(
        tmp_path, budgets, fake_tools):
    """Clause 1, asserted on the filesystem. A deny message that fired *after*
    the handler wrote would still read like a deny, which is why the bytes are
    the evidence here and the message is not."""
    paths = [note(tmp_path, f"note{i}") for i in range(CEILING + 2)]

    for i in range(CEILING):
        result = call("Write", {"file_path": paths[i], "content": f"v{i}"},
                      session_id=INTERACTIVE)
        assert not ceilinged(result), text_of(result)

    for i in (CEILING, CEILING + 1):
        result = call("Write", {"file_path": paths[i], "content": f"v{i}"},
                      session_id=INTERACTIVE)
        assert ceilinged(result), text_of(result)

    for i in range(CEILING):
        assert Path(paths[i]).read_text() == f"v{i}", "the first N must run"
    for i in (CEILING, CEILING + 1):
        assert Path(paths[i]).read_text() == "", (
            f"the ceiling refused call {i} but the handler wrote it anyway")


def test_the_refusal_names_used_remaining_the_window_and_a_human(tmp_path, budgets):
    """Clause 1's other half: the deny has to be *corrective*, so every number in
    it is the ledger's, and it must not read like a permission the agent can
    re-word or re-grant its way past."""
    for i in range(CEILING):
        call("Write", {"file_path": note(tmp_path, f"r{i}"), "content": "x"},
             session_id=INTERACTIVE)
    text = text_of(call("Write", {"file_path": note(tmp_path, "r9"),
                                  "content": "x"}, session_id=INTERACTIVE))

    assert f"{CEILING} of {CEILING}" in text, text      # used / ceiling
    assert "remain and nothing ran" in text, text       # remaining, and no effect
    assert "minute" in text, text                       # the window, named
    assert "ask a human" in text, text                  # the second key
    assert "harness.mutation_budget" in text, text      # where the human sets it
    assert "not a permission" in text, text
    assert "Do not retry this call inside the window" in text, text


def test_a_denied_call_is_ledgered_as_denied(tmp_path, budgets):
    """`status()` is what a person calibrates a source's ceiling from (the one
    thing this round explicitly leaves to them), so a refusal has to be visible
    without reading a log line."""
    for i in range(CEILING + 2):
        call("Write", {"file_path": note(tmp_path, f"l{i}"), "content": "x"},
             session_id=INTERACTIVE)
    st = MB.status(hours=1)
    row = next((s for s in st["sources"] if s["source"] == "interactive"), None)
    assert row is not None, st
    assert row["allowed"] == CEILING and row["denied"] == 2, row


# ── clause 2: the capacity refills when the window elapses ─────────────────

def test_capacity_refills_after_the_window(tmp_path, budgets, fake_tools):
    """Clause 2, with the injected clock. A ceiling that only ratchets down is
    not a ceiling, it is an outage that arrives partway through a nightly."""
    for i in range(CEILING):
        call("Write", {"file_path": note(tmp_path, f"w{i}"), "content": "x"},
             session_id=INTERACTIVE)
    assert ceilinged(call("Write", {"file_path": note(tmp_path, "w9"),
                                    "content": "x"}, session_id=INTERACTIVE))

    budgets.clock.advance(budgets.config()["window_seconds"] + 1)

    path = note(tmp_path, "w_after")
    result = call("Write", {"file_path": path, "content": "again"},
                  session_id=INTERACTIVE)
    assert not ceilinged(result), text_of(result)
    assert Path(path).read_text() == "again"


# ── clause 3: independent budgets, and no argument moves a bucket ───────────

def test_two_sessions_draw_independent_budgets(tmp_path, budgets):
    """Clause 3, first half: the burst that exhausts one session leaves the
    other's first call running."""
    other = "20260915_120000_chat_9f8e"
    for i in range(CEILING + 1):
        call("Write", {"file_path": note(tmp_path, f"a{i}"), "content": "x"},
             session_id=INTERACTIVE)
    assert ceilinged(call("Write", {"file_path": note(tmp_path, "ax"),
                                    "content": "x"}, session_id=INTERACTIVE))

    path = note(tmp_path, "b_first")
    result = call("Write", {"file_path": path, "content": "x"}, session_id=other)
    assert not ceilinged(result), text_of(result)
    assert Path(path).read_text() == "x"


@pytest.mark.parametrize("spoof", [
    {"bypass": True},
    {"_bypass": True},
    {"skipReview": True},
    {"ceiling": 9999},
    {"mutation_ceiling": 9999},
    {"rate_limit": 9999},
    {"window_seconds": 0},
    {"op_class": "shell"},
])
def test_an_argument_named_bypass_or_ceiling_selects_nothing(tmp_path, budgets,
                                                             spoof):
    """Clause 3, second half. The decision reads `_meta` and config and treats
    the argument payload as data, so a ceiling-shaped key in the payload changes
    nothing — including `skipReview`, which is the surface the talk's own
    header-reset incident came out of."""
    for i in range(CEILING):
        call("Write", {"file_path": note(tmp_path, f"p{i}"), "content": "x"},
             session_id=INTERACTIVE)
    result = call("Write", {"file_path": note(tmp_path, "p9"), "content": "x",
                            **spoof}, session_id=INTERACTIVE)
    assert ceilinged(result), f"{spoof} was honoured: {text_of(result)[:160]}"
    assert Path(note(tmp_path, "p9")).read_text() == ""


def test_the_bypass_shaped_keys_the_gate_names_are_the_ones_the_sanitiser_drops():
    """`SanitisedKeys` is the set an argument-carrying bypass would have to be
    in; `app/harness/loop.py` strips exactly those names off the arguments before
    they are logged and replayed into history. Pin both sides so neither drifts
    into a list the other one has to guess."""
    assert {"bypass", "_session_id", "ceiling"} <= MB.SanitisedKeys
    assert "file_path" not in MB.SanitisedKeys and "path" not in MB.SanitisedKeys
    from app.harness.loop import _sanitise_tool_args
    out = _sanitise_tool_args({"file_path": "/tmp/x", "bypass": True,
                               "ceiling": 99, "_session_id": "s",
                               "content": "keep"})
    assert out == {"file_path": "/tmp/x", "content": "keep"}, out


def test_a_claimed_session_id_cannot_buy_a_fresh_or_a_looser_budget(budgets):
    """The gate's own blind spot, closed.

    `main._bound_session_id` still honours a legacy `_session_id` *argument* so a
    harness and an aggregator at different versions correlate. Fine for
    attribution, fatal for a ceiling: an id the model types would mint unlimited
    fresh budgets (`spoof-0`, `spoof-1`, …, N calls each), and one shaped like a
    nightly job (`20260915_030000_scheduledtask_ab12`) would buy the batch table.
    So an unbound call is charged to one shared key and evaluated as
    `interactive` — the tightest table, shared, not an infinite supply.
    """
    allowed = 0
    for i in range(CEILING + 6):
        result = asyncio.run(M.call_tool(
            "Write", {"file_path": note(budgets.root, f"f{i}"), "content": "x",
                      "_session_id": f"spoof-{i}"}, meta_for()))
        if not ceilinged(result):
            allowed += 1
    assert allowed == CEILING, f"the argument chose the budget: {allowed} allowed"

    assert MB.run_source("20260915_030000_scheduledtask_ab12", "",
                         bound=False) == "interactive"
    assert MB.run_source("20260915_030000_scheduledtask_ab12", "",
                         bound=True) == "scheduledtask"
    assert MB.run_source(MB.UNBOUND, "") == "interactive"


# ── clause 4: the 08-22-shaped bulk wipe halts mid-run ─────────────────────

def test_a_bulk_delete_halts_at_the_ceiling_and_the_rest_of_the_tree_survives(
        tmp_path, monkeypatch, budgets):
    """Clause 4: the same test's second case, and the one that would have stopped
    2026-08-22.

    A run under a batch source deletes a `facts/`-shaped tree one legitimate call
    at a time. `_pipeline` is the pipeline's derived state, so this is the shape
    the loss actually took — and the surviving files are asserted on disk, which
    is the only claim that matters: the point is that `facts/` still exists
    afterwards, not that a message was printed."""
    facts = tmp_path / "lloyd" / "_pipeline" / "vault-derived" / "facts"
    facts.mkdir(parents=True)
    for i in range(CEILING * 3):
        (facts / f"fact-{i}.json").write_text(json.dumps({"i": i}))
    targets = sorted(facts.glob("fact-*.json"))
    assert len(targets) == CEILING * 3

    monkeypatch.setattr(MB, "_PIPELINE_ROOT", tmp_path / "lloyd" / "_pipeline")
    log: list = []
    register(monkeypatch, "delete_fact_file", FakeDeleter(log),
             op_class="fs_delete")

    # A nightly pipeline's own session and run scope: the run that burned the
    # store would have had a source of its own, and it must still be capped.
    for path in targets:
        call("delete_fact_file", {"file_path": str(path)},
             session_id="20260915_030000_nightlypipeline_cd34",
             effect_scope="item:nightly-pipeline:9")

    assert len(log) == CEILING, (
        f"the ceiling let {len(log)} deletes through, expected {CEILING}")
    survivors = sorted(facts.glob("fact-*.json"))
    assert len(survivors) == CEILING * 3 - CEILING, (
        "files the ceiling refused are gone from disk anyway")
    assert survivors[0].read_text() == json.dumps({"i": targets.index(
        survivors[0]) if survivors[0] in targets else 0}), "contents intact"


def test_the_interactive_vault_ceiling_is_tighter_than_the_repo_one(budgets):
    """Clause 5's configuration half, against the shipped defaults: the vault is
    the unrecoverable substrate, the repo is git-tracked, so the same class gets
    different numbers per scope."""
    import app.config as appconfig
    table = appconfig.CONFIG["harness"]["mutation_budget"]
    per_class = table["ceilings"]["interactive"]
    assert per_class["fs_delete"]["vault"] == 10, per_class["fs_delete"]
    assert per_class["fs_delete"]["lloyd"] == 30, per_class["fs_delete"]
    assert per_class["fs_write"]["vault"] == 120 < per_class["fs_write"]["lloyd"]
    assert table["ceilings"].get("_default"), "a source with no table has no ceiling"
    # …and a named batch source clears the interactive number on purpose.
    batch = table["ceilings"]["nightly"]
    assert batch["fs_write"]["vault"] == 1500 > per_class["fs_write"]["vault"]
    assert "kg_write" not in batch, "an unlisted class is a bug, so it denies"


def test_an_interactive_session_and_a_worker_scope_do_not_share_a_bucket(
        tmp_path, budgets):
    """Clause 5's test half, at dispatch: the worker source's table is 4× the
    interactive one under `budgets`, so the same tool called from each cannot be
    drawing on one counter."""
    for i in range(CEILING):
        assert not ceilinged(call("Write", {"file_path": note(tmp_path, f"i{i}"),
                                            "content": "x"},
                                  session_id=INTERACTIVE))
    assert ceilinged(call("Write", {"file_path": note(tmp_path, "iX"),
                                    "content": "x"}, session_id=INTERACTIVE))

    for i in range(CEILING, CEILING * 4):
        result = call("Write", {"file_path": note(tmp_path, f"v{i}"),
                                "content": "x"},
                      session_id="20260915_040000_scheduledtask_77aa",
                      effect_scope=f"item:scheduled-task:{i}")
        assert not ceilinged(result), f"worker call {i} blocked: {text_of(result)[:160]}"


def test_one_buckets_exhaustion_does_not_burn_another_scopes(
        tmp_path, monkeypatch, budgets):
    """The ceiling is per (class, scope): a session that has used up its vault
    writes can still write to the repo, and still send mail. Per-*tool* budgets
    would have multiplied into nothing, which is the point of keying on the
    effect and not the tool name — so this pins the other direction too."""
    for i in range(CEILING + 1):
        call("Write", {"file_path": note(tmp_path, f"s{i}"), "content": "x"},
             session_id=INTERACTIVE)

    repo = tmp_path / "lloyd" / "app" / "x.py"
    repo.parent.mkdir(parents=True, exist_ok=True)
    repo.write_text("1\n")
    result = call("Write", {"file_path": str(repo), "content": "2\n"},
                  session_id=INTERACTIVE)
    assert not ceilinged(result), text_of(result)

    log: list = []
    register(monkeypatch, "delete_fact_file", FakeDeleter(log), op_class="fs_delete")
    doomed = note(tmp_path, "doomed")
    assert not ceilinged(call("delete_fact_file", {"file_path": doomed},
                              session_id=INTERACTIVE))


# ── clause 6: counters per run, and a trip wire that alerts without blocking ─

def test_a_completed_run_reports_its_counts_per_op_class(tmp_path, budgets):
    """Clause 6a: the per-run counter is written by the dispatch path and read
    off the run row, so it cannot be narration. `runs.mutations_json` is a new
    column, which is the additive migration this test also pins."""
    from workers.queue import WorkQueue

    q = WorkQueue(tmp_path / "workers.db")
    scope = "item:nightly-knowledge-write:44"
    for i in range(CEILING):
        call("Write", {"file_path": note(tmp_path, f"c{i}"), "content": "x"},
             session_id="20260915_040000_nightliewrite_1a2b",
             effect_scope=scope)
    counts = MB.run_mutations(scope)
    assert counts == {"fs_write": CEILING}, counts

    q.record_run(run_id="run-x", queue_id=1, source="nightly-knowledge-write",
                 status="success", started_at="2026-09-15T04:00:00+00:00",
                 completed_at="2026-09-15T04:05:00+00:00", duration_seconds=300.0,
                 summary="done", mutations_json=json.dumps(counts))
    row = q.list_runs(limit=5)[0]
    assert json.loads(row["mutations_json"]) == {"fs_write": CEILING}, row


async def test_the_pool_writes_the_mutations_counter_into_the_run_row(
        tmp_path, monkeypatch, budgets):
    """Clause 6b: the counter reaches the ledger through the real pool, not
    because a test passed the kwarg."""
    from workers.pool import WorkerPool
    from workers.queue import WorkQueue
    from workers import sources
    from app.harness import policy

    q = WorkQueue(tmp_path / "workers.db")
    calls: list = []
    register(monkeypatch, "recorder", Recorder(calls))

    async def execute(item):
        meta = meta_for("20260915_040000_scheduledtask_5f5f",
                        policy.current_effect_scope.get())
        for i in range(2):
            await M.call_tool("recorder", {"n": i}, meta)
        await asyncio.sleep(0.1)

    monkeypatch.setattr(sources, "SOURCE_REGISTRY",
                        {"s": SimpleNamespace(NAME="s", execute=execute)},
                        raising=False)
    monkeypatch.setattr(sources, "get_sources_config", lambda: {"s": {}},
                        raising=False)
    q.enqueue("s", "k")
    pool = WorkerPool(q, slots=1, poll_idle_seconds=0.01)
    await pool.start()
    try:
        for _ in range(200):
            if q.list_runs(limit=5):
                break
            await asyncio.sleep(0.02)
    finally:
        await pool.stop()

    runs = q.list_runs(limit=5)
    assert runs, "the pool never recorded a run"
    assert json.loads(runs[0]["mutations_json"]) == {"other": 2}, runs[0]


def test_the_run_counter_counts_effects_and_not_refusals(tmp_path, budgets):
    """A denied call is not an effect. Counting refusals would put the same
    number in the row whether the ceiling worked or the run legitimately filled
    its bucket, which is the distinction the calibration decision needs."""
    scope = "item:scheduled-task:77"
    for i in range(CEILING + 2):
        call("Write", {"file_path": note(tmp_path, f"e{i}"), "content": "x"},
             session_id="20260915_040000_scheduledtask_3c3c", effect_scope=scope)
    assert MB.run_mutations(scope) == {"fs_write": CEILING}
    assert MB.run_mutations(scope, denied=True) == {"fs_write": 2}
    assert MB.run_mutations("") == {}, "an interactive turn has no run to report"


def test_crossing_the_trip_wire_line_alerts_without_blocking(monkeypatch, budgets):
    """Clause 6c. The alert fires at 60% of the ceiling (the ratio is below 1.0
    because the ceiling stops the bucket at 1.0, so a threshold above it is an
    alert that can never fire), and the calls after it still dispatch."""
    log: list = []
    register(monkeypatch, "recorder", Recorder(log))
    source_ceiling = CEILING * 4          # the `*` table under `budgets`
    alert_at = int(0.6 * source_ceiling)  # 7

    for i in range(alert_at):
        assert not ceilinged(call("recorder", {"n": i},
                                  session_id="20260915_040000_scheduledtask_9e9e",
                                  effect_scope=f"item:scheduled-task:{i}"))
    assert MB.recent_alerts(limit=10) == [], "alerted below the line"

    for i in range(alert_at, alert_at + 2):
        assert not ceilinged(call("recorder", {"n": i},
                                  session_id="20260915_040000_scheduledtask_9e9e",
                                  effect_scope=f"item:scheduled-task:{i}")), (
            "the trip wire blocked a call, which is not what it is for")

    alerts = MB.recent_alerts(limit=10)
    assert len(alerts) == 1, f"expected one row for the hour, got {alerts}"
    row = alerts[0]
    assert row["source"] == "scheduledtask" and row["op_class"] == "other"
    assert row["count"] >= alert_at and row["ceiling"] == source_ceiling
    assert json.loads(row["sessions"]), "the per-session skew is the diagnostic"


def test_the_trip_wire_needs_a_real_majority_of_sessions_to_alert(monkeypatch,
                                                                   budgets):
    """The line is a *runaway* line. One session exhausting its own budget while
    a hundred others idle is normal shape, and paging on it is how this alert
    becomes noise — so the majority half of the conjunction is load-bearing."""
    log: list = []
    register(monkeypatch, "recorder", Recorder(log))
    ceiling = CEILING * 4
    for i in range(int(0.6 * ceiling)):
        await_ok = not ceilinged(call(
            "recorder", {"n": i},
            session_id=f"20260915_040000_scheduledtask_{i:04d}",
            effect_scope=f"item:scheduled-task:{i}"))
        assert await_ok
    assert MB.recent_alerts(limit=10) == [], (
        "alerted on a single session filling a bucket shared by many")


# ── the surrounding decisions ───────────────────────────────────────────────

def test_read_only_tools_never_draw_the_ceiling(tmp_path, budgets):
    """Two tables that must not drift: a tool `annotations.py` calls read-only is
    uncapped, and a cap that silently ate `vault_read` would throttle the fleet's
    reads before anyone noticed the number moving."""
    from agent_mcp import annotations as A
    for name in sorted(A.READ_ONLY):
        assert MB.op_class(name, {}) is None, name
    assert MB.op_class("vault_read", {"path": "knowledge/x.md"}) is None
    assert MB.op_class("Grep", {"pattern": "x"}) is None


def test_the_mutating_tools_landed_in_the_class_their_effect_puts_them(budgets):
    """The inventory, pinned. An unlisted name defaults to `other` — charged, not
    skipped — so a tool added tomorrow is capped until someone says otherwise;
    this is the test that says which class today's tools are in."""
    assert MB.op_class("vault_write", {}) == "fs_write"
    assert MB.op_class("memory_add", {}) == "fs_write"
    assert MB.op_class("Write", {}) == "fs_write"
    assert MB.op_class("fact_add", {}) == "kg_write"
    assert MB.op_class("remember", {}) == "kg_write"
    assert MB.op_class("backlog_write_task", {}) == "record_write"
    assert MB.op_class("email_send", {}) == "send"
    assert MB.op_class("calendar_create", {}) == "send"
    assert MB.op_class("fact_invalidate", {}) == "fs_delete"
    assert MB.op_class("forget", {}) == "fs_delete"
    assert MB.op_class("email_delete", {}) == "fs_delete"
    assert MB.op_class("autonomy_delete_task", {}) == "destroy"
    assert MB.op_class("Task", {}) == "spawn"
    assert MB.op_class("brand_new_uninventoried_tool", {}) == "other"


def test_a_delete_verb_inside_a_grep_pattern_is_not_a_delete(budgets):
    """`rm` anywhere in a command string would charge `grep -rn "rm " notes.md` to
    the delete ceiling, and a batch job would run out of delete budget on reads —
    the failure mode where the gate throttles exactly the wrong thing. The verb
    has to be in command position."""
    assert MB.op_class("Bash", {"command": "rm -f /tmp/x"}) == "fs_delete"
    assert MB.op_class("Bash", {"command": "cd /tmp && rm -f x"}) == "fs_delete"
    assert MB.op_class("Bash", {"command": "find . -name '*.json' -delete"}) == "fs_delete"
    assert MB.op_class("Bash", {"command": "git clean -fdx"}) == "fs_delete"
    assert MB.op_class("Bash", {"command": 'grep -rn "rm " notes.md'}) == "shell"
    assert MB.op_class("Bash", {"command": "ls | rm x"}) == "fs_delete"
    assert MB.op_class("Bash", {"command": "ls -la"}) == "shell"


def test_the_target_scope_follows_the_path_the_call_names(budgets, monkeypatch,
                                                          tmp_path):
    monkeypatch.setattr(MB, "_PIPELINE_ROOT", tmp_path / "lloyd" / "_pipeline")
    assert MB.target_scope("vault_write", {"path": "knowledge/x.md"}) == "vault"
    assert MB.target_scope("Write", {"file_path": str(tmp_path / "vault" / "a.md")}) == "vault"
    assert MB.target_scope("Write", {"file_path": str(tmp_path / "lloyd" / "_pipeline" / "facts" / "a.json")}) == "_pipeline"
    assert MB.target_scope("Write", {"file_path": str(tmp_path / "lloyd" / "app" / "x.py")}) == "lloyd"
    assert MB.target_scope("email_send", {"to": "a@b.c"}) == "none"
    assert MB.target_scope("Write", {"file_path": "/etc/passwd"}) == "other"
    # One call touching two places is charged to one bucket, named so a reader
    # never mistakes it for a per-path count.
    assert MB.target_scope("email_move_folder",
                           {"folderPath": str(tmp_path / "vault" / "a"),
                            "newParentPath": "/tmp/here"}) == "mixed"


def test_the_kill_switch_charges_nothing(tmp_path, monkeypatch, budgets):
    """The escape hatch is a config edit and an aggregator restart. It has to be
    real, because a gate that cannot be turned off is a gate nobody trusts
    enough to leave on across a nightly."""
    monkeypatch.setenv("LLOYD_MUTATION_BUDGET_DISABLED", "1")
    assert not MB.enabled()
    for i in range(CEILING * 3):
        assert not ceilinged(call("Write", {"file_path": note(tmp_path, f"k{i}"),
                                            "content": "x"},
                                  session_id=INTERACTIVE))


def test_an_unwritable_ledger_fails_open_with_the_gate_loud(tmp_path, monkeypatch,
                                                            budgets):
    """The trade, in a test. `agent_mcp.main` has no "the store would not open,
    so nothing is known" error for its callers — it answers with tool results —
    so a ceiling that hard-failed would take the fleet's writes down with the
    ledger. Availability of the fleet wins over strictness of one guard, so long
    as the degradation is logged."""
    db = Path(budgets.db)
    db.parent.mkdir(parents=True, exist_ok=True)
    db.write_text("not a database")
    for i in range(CEILING * 2):
        assert not ceilinged(call("Write", {"file_path": note(tmp_path, f"o{i}"),
                                            "content": "x"},
                                  session_id=INTERACTIVE))


def test_a_denied_call_never_reaches_the_effect_ledger(tmp_path, monkeypatch,
                                                       budgets):
    """The ordering that both this and #544 depend on: a refusal upstream of the
    claim, or a ceiling denial would poison the next legitimate retry of the same
    arguments with an `unknown` effect."""
    log: list = []
    register(monkeypatch, "recorder", Recorder(log))
    scope = "item:scheduled-task:88"
    ceiling = CEILING * 4
    for i in range(ceiling + 1):
        call("recorder", {"n": i},
             session_id="20260915_040000_scheduledtask_1b1b", effect_scope=scope)
    assert len(log) == ceiling
    assert TE.suppressed_total() == 0, "the refusal was ledgered as an effect"
