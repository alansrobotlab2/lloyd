"""Recording is universal; being watched is a per-job opt-in.

Two axes, and conflating them is the mistake this pins against. Every
background run is *recorded* — a session, a transcript, an event log, for a
few file appends. Being *observed* by the Inner Voice critic costs a
goal-extraction call plus a critique per turn on the PRIMARY at priority 1, in
front of whatever a human is typing, so it is off unless the job asked.

The autonomy half is the interesting one. `run_task` calls `run_query`
directly and installs the #534 authority gate on its own `HookRegistry`, so
the observer has to attach to that same registry rather than replacing it —
`attach_observer_for_turn` creates one only when none exists, which is what
makes passing `task_hooks` work at all. Losing the gate to gain the observer
would be a bad trade made silently.
"""

from __future__ import annotations

import asyncio
import json

import pytest

import autonomy


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr("app.sessions_io.SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr("app.event_log.EVENT_LOGS_DIR", tmp_path / "events")
    monkeypatch.setattr("app.event_log.BLOBS_DIR", tmp_path / "events" / "blobs")
    return tmp_path


def _session(store_dir):
    files = sorted((store_dir / "sessions").glob("*.json"))
    assert len(files) == 1
    return json.loads(files[0].read_text())


# ── The per-task switch ────────────────────────────────────────────────

@pytest.fixture
def fleet_off(monkeypatch):
    """`autonomy.inner_voice: false` in the live config, read through
    `app.config.CONFIG` — the loader, not a raw file read, so a canary
    resolves its own overlay rather than production's answer."""
    monkeypatch.setattr("app.config.CONFIG", {"autonomy": {"inner_voice": False}})


def test_frontmatter_beats_the_fleet_default(monkeypatch, fleet_off):
    """A single task can be watched on a fleet that is off, which is the
    whole reason the two levels exist."""
    assert autonomy._task_inner_voice({}) is False
    assert autonomy._task_inner_voice({"inner_voice": True}) is True

    monkeypatch.setattr("app.config.CONFIG", {"autonomy": {"inner_voice": True}})
    assert autonomy._task_inner_voice({}) is True
    # And off on a fleet that is on — the switch has to work both ways or it
    # is a master switch, not a default.
    assert autonomy._task_inner_voice({"inner_voice": False}) is False


def test_a_yaml_string_is_read_as_the_boolean_it_looks_like(fleet_off):
    """Task files go through a graduated parser whose regex fallback yields
    strings. `inner_voice: true` recovered that way must not read as the
    truthy string `"true"` in one path and a bool in the other."""
    assert autonomy._task_inner_voice({"inner_voice": "true"}) is True
    assert autonomy._task_inner_voice({"inner_voice": "false"}) is False
    # Anything else falls through to the fleet default rather than guessing.
    assert autonomy._task_inner_voice({"inner_voice": "maybe"}) is False


def test_the_frontmatter_key_survives_the_degraded_parser():
    """`_parse_task_file` falls back to regex field extraction when the YAML
    is broken, and only listed fields survive. A task whose file needed that
    repair must not silently lose its opt-in.

    This used to grep the field tuple out of autonomy.py's source. #1014 moved
    the list into `agent_mcp._shared.AUTONOMY_TASK_FIELDS`, the one list all
    three readers pass, so the assertion is on that constant — the object the
    extractor actually receives — plus that the scheduler still passes it."""
    from agent_mcp._shared import AUTONOMY_TASK_FIELDS

    assert "inner_voice" in AUTONOMY_TASK_FIELDS, \
        "a task file whose YAML is broken would lose its IV opt-in"
    assert "fallback_fields=AUTONOMY_TASK_FIELDS" in open(autonomy.__file__).read(), \
        "the scheduler must pass the shared constant, not a list of its own"


# ── The observer on the direct path ────────────────────────────────────

def _stub(monkeypatch, tmp_path, task, stream):
    captured: dict = {}
    monkeypatch.setattr(autonomy, "_find_task_file", lambda tid: tmp_path / "x.md")
    monkeypatch.setattr(autonomy, "_parse_task_file", lambda p: task)
    monkeypatch.setattr(autonomy, "_load_skill_content", lambda s: "SKILL")
    monkeypatch.setattr(autonomy, "_update_task_field", lambda *a, **k: None)
    monkeypatch.setattr(autonomy, "_append_activity_log", lambda *a, **k: None)
    monkeypatch.setattr(autonomy, "_get_model_env", lambda m: {})
    monkeypatch.setattr(autonomy, "_write_run_record", lambda **kw: None)

    async def _run_query(messages, options):
        captured["options"] = options
        for evt in stream:
            yield evt

    class Opts:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    import app.harness as harness
    import app.harness.mcp_pool as mcp_pool
    monkeypatch.setattr(harness, "run_query", _run_query)
    monkeypatch.setattr(harness, "RunOptions", Opts)
    monkeypatch.setattr(mcp_pool, "DEFAULT_LLOYD_MCP_SERVERS", {}, raising=False)
    monkeypatch.setattr("prompt_builder.build_system_prompt", lambda **_kw: "SYS")
    return captured


_DONE = [{"type": "text_delta", "text": "done"},
         {"type": "result", "stop_reason": "stop", "num_turns": 1, "usage": {}}]


def test_an_unwatched_task_creates_an_unwatched_session(store, monkeypatch):
    monkeypatch.setattr(autonomy, "_task_inner_voice", lambda t: False)
    captured = _stub(monkeypatch, store,
                     {"id": 12, "name": "Nightly", "skill_name": "s",
                      "status": "up_next", "timeout_seconds": 300}, _DONE)
    asyncio.run(autonomy.run_task(12))
    assert _session(store)["inner_voice"] is False
    # No observer means no cancel lever, and no lever is better than a knob
    # nothing reads.
    assert getattr(captured["options"], "cancel_event", None) is None


def test_a_watched_task_keeps_its_grant_gate_and_gains_the_observer(
        store, monkeypatch):
    """The load-bearing one. `attach_observer_for_turn` creates a registry
    only if none exists, so passing `run_task`'s own `task_hooks` adds the
    observer to the registry the #534 policy hook is already on. If it
    replaced it, an observed task would be an ungated one."""
    monkeypatch.setattr(autonomy, "_task_inner_voice", lambda t: True)
    captured = _stub(monkeypatch, store,
                     {"id": 13, "name": "Watched", "skill_name": "s",
                      "status": "up_next", "timeout_seconds": 300}, _DONE)

    seen: dict = {}

    async def _attach(**kw):
        seen.update(kw)
        # The real one installs onto `options.hooks`; what matters here is
        # which registry it was handed.
        class _State:
            closed = False
        return _State()

    monkeypatch.setattr(
        "app.routers._messages_inner_voice.attach_observer_for_turn", _attach)
    monkeypatch.setattr(
        "app.routers._messages_inner_voice.close_observer", lambda s: None)

    asyncio.run(autonomy.run_task(13))

    assert _session(store)["inner_voice"] is True
    options = captured["options"]
    # Same registry object, carrying the grant hook `run_task` installed.
    assert seen["options"] is options
    assert options.hooks is not None
    assert options.hooks._pre, "the policy hook is gone from the registry"
    # Ambient, not user: the observer's own gate fires for `ambient` when the
    # session opted in, and there is no user turn here to evaluate.
    assert seen["turn_source"] == "ambient"
    assert seen["producer_source"] == "autonomy"
    # Nobody is reading, so the two levers that exist to reach a human are
    # deliberately absent.
    assert seen.get("enqueue_ambient_callback") is None
    assert seen.get("clarify_callback") is None
    # The cancel lever reaches the loop, or it is a knob nothing reads.
    assert options.cancel_event is seen["cancel_event"]


def test_the_observer_is_closed_however_the_run_ends(store, monkeypatch):
    """A turn that ends any way other than through the `result` event leaves
    non-terminal judgments running against a dead turn."""
    monkeypatch.setattr(autonomy, "_task_inner_voice", lambda t: True)
    _stub(monkeypatch, store,
          {"id": 14, "name": "Boom", "skill_name": "s", "status": "up_next",
           "timeout_seconds": 300}, _DONE)

    closed: list = []

    async def _attach(**kw):
        class _State:
            closed = False
        return _State()

    monkeypatch.setattr(
        "app.routers._messages_inner_voice.attach_observer_for_turn", _attach)
    monkeypatch.setattr(
        "app.routers._messages_inner_voice.close_observer",
        lambda s: closed.append(s))

    async def _boom(messages, options):
        raise RuntimeError("engine went away")
        yield  # pragma: no cover — makes this an async generator

    import app.harness as harness
    monkeypatch.setattr(harness, "run_query", _boom)
    monkeypatch.setattr(autonomy, "_record_failure",
                        _fail_stub := _make_fail_stub())

    out = asyncio.run(autonomy.run_task(14))
    assert out["status"] == "failed"
    assert len(closed) == 1


def _make_fail_stub():
    async def _record(task, task_id, run_id, started_at, started_dt, **kw):
        return {"success": False, "status": "failed", "task_id": task_id}
    return _record


def test_an_observer_that_cannot_attach_does_not_stop_the_run(store, monkeypatch):
    """Watching is not the run. A broken observer costs the second opinion,
    never the work."""
    monkeypatch.setattr(autonomy, "_task_inner_voice", lambda t: True)
    _stub(monkeypatch, store,
          {"id": 15, "name": "Fine", "skill_name": "s", "status": "up_next",
           "timeout_seconds": 300}, _DONE)

    async def _explode(**kw):
        raise RuntimeError("goal extraction is down")

    monkeypatch.setattr(
        "app.routers._messages_inner_voice.attach_observer_for_turn", _explode)
    out = asyncio.run(autonomy.run_task(15))
    assert out["success"] is True


# ── The per-source switch ──────────────────────────────────────────────

def test_a_source_reads_its_switch_from_config(monkeypatch):
    from workers.sources import _common as C

    monkeypatch.setattr(
        "workers.sources.get_sources_config",
        lambda: {"autocode": {"inner_voice": True},
                 "deep-research": {"inner_voice": False},
                 "quiet": {}})
    assert C.source_inner_voice("autocode") is True
    assert C.source_inner_voice("deep-research") is False
    # #1015: a source with no key reads as the fleet default, off. It used to
    # read True, which is how board-steward came to be observed by nobody's
    # decision.
    assert C.source_inner_voice("quiet") is False
    assert C.source_inner_voice("never-heard-of-it") is False
    # The `default` keyword stays, for a caller that means to name one.
    assert C.source_inner_voice("quiet", default=True) is True


def test_an_unreadable_config_resolves_to_off(monkeypatch):
    from workers.sources import _common as C

    def _boom():
        raise RuntimeError("config import failed")

    monkeypatch.setattr("workers.sources.get_sources_config", _boom)
    assert C.source_inner_voice("autocode") is False


def test_an_unkeyed_session_source_gets_an_unobserved_session(monkeypatch, tmp_path):
    """#1015 clause 2: `inner_voice=None` on a source with no key creates the
    session with both observer flags off."""
    import asyncio
    from app import sessions_io
    from workers.sources import _common as C

    monkeypatch.setattr("workers.sources.get_sources_config",
                        lambda: {"quiet": {}})
    made = {}
    real_create = sessions_io.create_session

    class _Stop(Exception):
        pass

    def _create(session_id, **kw):
        kw["sessions_dir"] = tmp_path
        real_create(session_id, **kw)
        made["id"] = session_id
        raise _Stop()

    monkeypatch.setattr(sessions_io, "create_session", _create)
    with pytest.raises(_Stop):
        asyncio.run(C.run_prompt_in_session("hi", title="t", source="quiet",
                                            inner_voice=None, timeout_seconds=5))
    data = json.loads((tmp_path / f"{made['id']}.json").read_text())
    assert data["inner_voice"] is False
    assert data["inner_voice_evaluate_user_turns"] is False


def _session_backed_sources():
    """Every registered source whose module runs a turn through
    `run_prompt_in_session`, discovered from `workers/sources/*.py` — never a
    hand-named list, which is how a sixth observable source (board-steward)
    shipped with no key while the old test still named five."""
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "workers" / "sources"
    found = {}
    for path in sorted(root.glob("*.py")):
        if path.name.startswith("_"):
            continue
        tree = ast.parse(path.read_text())
        calls = any(isinstance(n, ast.Call)
                    and getattr(n.func, "attr", getattr(n.func, "id", None))
                    == "run_prompt_in_session"
                    for n in ast.walk(tree))
        if not calls:
            continue
        name = next((n.value.value for n in tree.body
                     if isinstance(n, ast.Assign)
                     and any(getattr(t, "id", None) == "NAME" for t in n.targets)
                     and isinstance(n.value, ast.Constant)), None)
        assert name, f"{path.name} runs a session turn but declares no NAME"
        found[name] = path
    return found


#: Session-backed sources the observer watches in the shipped config. Off for
#: all of them from cut 1 of senses-not-supervision (2026-09-12) — #874
#: abandoned at iteration 38 on an invented premise, sixteen false repetition
#: fires in a day — and back on for these two on 2026-09-25, after the causes
#: were fixed in Inner Voice rather than by switching it off: the round-id and
#: polling exemptions in the repetition guard (09-11), the terminal-only review
#: that reads the whole trajectory (R2), and the worker note and report-ask
#: rail (`observer._worker_note`, `_apply_decision_guards`).
OBSERVED_SOURCES = {"autocode", "autotriage"}


def test_only_the_named_session_backed_sources_are_observed_in_the_shipped_config():
    """Resolved through `source_inner_voice` against the real config, so a new
    session source with no key fails here rather than opting itself in.
    Recording is untouched either way: every one of these is a real session
    in the Background tab."""
    from workers.sources import _common as C

    sources = _session_backed_sources()
    assert {"autocode", "autotriage", "board-steward"} <= set(sources), sources
    for name in sources:
        assert C.source_inner_voice(name) is (name in OBSERVED_SOURCES), name


def test_no_call_site_bakes_its_own_answer_in():
    """`deep-research` passed `inner_voice=False` as a literal, which is to
    say it was not a setting. One reader, or the config is decoration."""
    for name, path in _session_backed_sources().items():
        assert "inner_voice=" not in path.read_text(), name


def test_the_switch_is_ui_mutable_without_dirtying_the_tracked_config(tmp_path,
                                                                     monkeypatch):
    """config.yaml is tracked, and a UI click that rewrote it would leave the
    live tree dirty and stop the self-modification loop — which is the whole
    reason the override file exists."""
    import app.config as cfgmod

    monkeypatch.setattr(cfgmod, "TOOL_OVERRIDES_PATH", tmp_path / "o.yaml")
    (tmp_path / "o.yaml").write_text(
        "workers:\n  sources:\n    autocode:\n      inner_voice: false\n"
        "    not-a-real-source:\n      inner_voice: true\n"
        "    autocode2:\n      enabled: true\n")
    base = {"workers": {"sources": {"autocode": {"inner_voice": True,
                                                 "enabled": True}}}}
    out = cfgmod._merge_tool_overrides(base)
    assert out["workers"]["sources"]["autocode"]["inner_voice"] is False
    # An override may not introduce a source…
    assert "not-a-real-source" not in out["workers"]["sources"]
    # …and only this one key per source is honoured.
    assert out["workers"]["sources"]["autocode"]["enabled"] is True


def test_an_override_may_not_introduce_inner_voice_on_an_unobservable_source(
        tmp_path, monkeypatch):
    """#1464: backlog-cluster's tracked block carries no `inner_voice` key on
    purpose (#1015) — it has no turn to observe, and /api/workers/health
    renders it null. The override file re-added `inner_voice: false`, which
    made the unobservable source read as one that had been switched off.
    The merge must refuse to introduce the key, and the next save must drop
    it from the file."""
    import yaml

    import app.config as cfgmod

    path = tmp_path / "o.yaml"
    monkeypatch.setattr(cfgmod, "TOOL_OVERRIDES_PATH", path)
    path.write_text(
        "workers:\n  sources:\n"
        "    autocode:\n      inner_voice: false\n"
        "    backlog-cluster:\n      inner_voice: false\n")
    base = {"workers": {"sources": {
        "autocode": {"inner_voice": True},
        "backlog-cluster": {"enabled": True, "threshold": 0.75},
    }}}
    out = cfgmod._merge_tool_overrides(base)
    assert "inner_voice" not in out["workers"]["sources"]["backlog-cluster"]
    # A source that does carry the key is still adjustable.
    assert out["workers"]["sources"]["autocode"]["inner_voice"] is False

    monkeypatch.setattr(cfgmod, "CONFIG", out)
    cfgmod.save_tool_overrides()
    saved = yaml.safe_load(path.read_text())
    assert "backlog-cluster" not in saved["workers"]["sources"]
    assert saved["workers"]["sources"]["autocode"] == {"inner_voice": False}


def test_backlog_cluster_tracked_block_has_no_inner_voice_key():
    """The half of #1015's guarantee that config.yaml owns."""
    import yaml

    from app.paths import LLOYD_HOME

    cfg = yaml.safe_load((LLOYD_HOME / "config.yaml").read_text())
    assert "inner_voice" not in cfg["workers"]["sources"]["backlog-cluster"]
