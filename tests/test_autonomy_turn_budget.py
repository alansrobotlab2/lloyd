"""Every autonomy run used to get the *global* turn cap; now a task can declare
its own, floored at that global. (#823)

Why this file exists
--------------------
`RunOptions.max_turns` for a scheduled task was `agent.max_turns` — 60 — for
every task, read once in `run_task` with nothing above it consulting the task
file. So a multi-step job that spends turns polling a background task, or one
that simply has a long artifact to write, exhausts the cap and is recorded with
an empty report: `app/harness/loop.py` breaks with `stop_reason="max_turns"` at
iteration 61, `app/harness/finalizer.py` records no verdict for that death, and
the run's remaining steps never happen.

The population is not hypothetical — 19 run records between 2026-09-10 and
2026-09-17 carry `stop_reason: max_turns` across tasks #24, #39, #40, #42, #47,
#58, #76 and #83. Three are one job, #39 Nightly Reflection — Knowledge Write:
`run_39_20260912_080016` (61 turns, 309 s), `run_39_20260913_080011` (61, 269 s)
and `run_39_20260913_081537` (61, 451 s), each `empty: true`, each followed by a
retry 10–15 minutes later that finished the same work in 30–38 turns. Each one
burned minutes of GPU to report nothing.

And the cap is binding, not merely tight: the *successful* nightly shape sits at
44–52 turns of 60 (`run_39_20260915_080029` = 52 turns, `stop_reason: stop`), so
the tail of that distribution is what crosses the line.

Why the fix is per-task and not a bigger global
-----------------------------------------------
`agent.max_turns` is not an autonomy-only knob. `app/routers/messages.py:157`
takes it as the default iteration budget for an interactive turn and
`app/routers/voice.py:127` for a voice turn, so raising 60 → 75 to spare one
nightly job also re-bounds every chat and voice turn on the primary — and one
15-minute poller (#68) was 148 of 205 runs in a recent health window, so that
cost lands on the whole fleet. `run_task` is the one site that can hand a single
job a bigger budget without spending anyone else's.

Why the declared value is FLOORED and not simply honoured
---------------------------------------------------------
Two task files already declare a number that cannot be an iteration budget:
`autonomy/48-entity-resolution-sweep.md:22` says `max_turns: 6` and
`autonomy/84-fact-improvement.md:19` says `max_turns: 4`, both added believing
the key capped something else. Honouring either as-is would take a task that
works today and give it the exact death this item exists to fix — a run stopped
at 5 or 3 iterations with nothing reported. So a declaration below the global is
raised to the global and logged at WARNING, never applied.

The tests drive the real `run_task` and assert on the `RunOptions` it built: the
cap only means anything through what the harness enforces, and a test that
called only the resolver would still pass if the call site ignored it.
`autonomy.LLOYD_HOME` — which `run_task` reads `config.yaml` through — points at
a tmp config carrying `agent.max_turns: 60`, production's value.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest

import autonomy

GLOBAL_MAX_TURNS = 60                  # config.yaml:2 `agent.max_turns`

_DONE = [{"type": "text_delta", "text": "done"},
         {"type": "result", "stop_reason": "stop", "num_turns": 1, "usage": {}}]


def _task(task_id: int, **extra) -> dict:
    """A task dict shaped like a parsed task file, carrying the fields
    `run_task` needs before it ever reaches the cap."""
    task = {"id": task_id, "name": f"Task {task_id}", "skill_name": "s",
            "status": "up_next", "timeout_seconds": 300}
    task.update(extra)
    return task


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Isolate every side effect: sessions, events, and the `config.yaml` that
    `run_task` loads the global budget out of."""
    (tmp_path / "config.yaml").write_text(
        "agent:\n  max_turns: 60\nmodel:\n  default: test-model\n")
    monkeypatch.setattr(autonomy, "LLOYD_HOME", tmp_path)
    monkeypatch.setattr("app.sessions_io.SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr("app.event_log.EVENT_LOGS_DIR", tmp_path / "events")
    monkeypatch.setattr("app.event_log.BLOBS_DIR", tmp_path / "events" / "blobs")
    return tmp_path


def _stub_run(monkeypatch, task: dict, tmp_path: Path) -> dict:
    """Drive `run_task` with the harness stubbed and return the captured
    `RunOptions` — the seam this change crosses."""
    captured: dict = {}
    monkeypatch.setattr(autonomy, "_find_task_file", lambda tid: tmp_path / "x.md")
    monkeypatch.setattr(autonomy, "_parse_task_file", lambda p: task)
    monkeypatch.setattr(autonomy, "_load_skill_content", lambda s: "SKILL")
    monkeypatch.setattr(autonomy, "_update_task_field", lambda *a, **k: None)
    monkeypatch.setattr(autonomy, "_append_activity_log", lambda *a, **k: None)
    monkeypatch.setattr(autonomy, "_get_model_env", lambda m: {})
    monkeypatch.setattr(autonomy, "_write_run_record", lambda **kw: None)
    monkeypatch.setattr(autonomy, "_task_inner_voice", lambda t: False)

    async def _run_query(messages, options):
        captured["options"] = options
        for evt in _DONE:
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


def _run_options(store, monkeypatch, task) -> object:
    captured = _stub_run(monkeypatch, task, store)
    asyncio.run(autonomy.run_task(task["id"]))
    return captured["options"]


def max_turns_warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records
            if r.name == "lloyd-autonomy" and r.levelno == logging.WARNING
            and "max_turns" in r.getMessage()]


# ── clause 1: a declared budget reaches RunOptions ──────────────────────────

def test_a_task_declaring_max_turns_90_is_built_with_90(store, monkeypatch):
    """The whole point: a job whose steps legitimately need more than 60
    iterations gets them, and only it does."""
    assert _run_options(store, monkeypatch, _task(24, max_turns=90)).max_turns == 90


def test_the_same_declared_budget_drives_the_iteration_anchor(store, monkeypatch):
    """One number for the cap and for the warning about the cap.

    `build_iteration_anchor` warns at 75 % and 90 % of the cap it is handed, so a
    run built with 90 must stay silent at iteration 45 — the iteration where a
    60-turn run hears its first warning — and speak at iteration 68 naming its
    own cap of 90. Read the cap and the anchor separately and the warning
    arrives at an iteration the run never reaches.
    """
    anchor = _run_options(store, monkeypatch, _task(24, max_turns=90)).state_anchor

    async def _fire(iteration: int):
        return [m["content"] for m in await anchor(iteration)]

    assert asyncio.run(_fire(45)) == [], "a 90-turn run warned at iteration 45"
    at_68 = asyncio.run(_fire(68))
    assert len(at_68) == 1, at_68
    assert "Iteration 68 of 90" in at_68[0]


# ── clause 2: no declaration still means the global 60 ─────────────────────

def test_a_task_with_no_max_turns_key_is_still_built_with_60(store, monkeypatch):
    """Every task on the fleet is in this branch today — none of #24, #39, #40,
    #42 or #83 declares the key — so landing must not move a single budget."""
    assert _run_options(store, monkeypatch, _task(39)).max_turns == GLOBAL_MAX_TURNS


def test_a_budget_above_the_global_is_applied_without_a_warning(store, monkeypatch,
                                                               caplog):
    """Positive control for clause 3's log assertion: a raised budget is the
    declared behaviour, so a guard that warned on every declaration would pass a
    test that only looked for a warning somewhere."""
    with caplog.at_level(logging.WARNING, logger="lloyd-autonomy"):
        options = _run_options(store, monkeypatch, _task(24, max_turns=90))
    assert options.max_turns == 90
    assert max_turns_warnings(caplog) == []


# ── clause 3: a below-global declaration is floored, and said out loud ─────

def test_the_real_max_turns_6_declaration_cannot_shrink_the_task(store, monkeypatch,
                                                                 caplog):
    """`autonomy/48-entity-resolution-sweep.md:22` declares `max_turns: 6`.

    Applied as-is that is a task stopped at iteration 7 with no report — the
    failure #823 exists to fix, manufactured by the config meant to help. The
    floor is the global, and the mismatch has to be visible because the
    declaration is still wrong and nobody is reading the run.
    """
    with caplog.at_level(logging.WARNING, logger="lloyd-autonomy"):
        options = _run_options(store, monkeypatch, _task(48, max_turns=6))
    assert options.max_turns == GLOBAL_MAX_TURNS
    warned = max_turns_warnings(caplog)
    assert len(warned) == 1, warned
    message = warned[0]
    assert "48" in message, message
    assert "max_turns=6" in message, message
    assert "60" in message, message


def test_the_real_max_turns_4_declaration_cannot_shrink_the_task(store, monkeypatch,
                                                                caplog):
    """`autonomy/84-fact-improvement.md:19` declares `max_turns: 4` — the same
    mistake one step further from the global."""
    with caplog.at_level(logging.WARNING, logger="lloyd-autonomy"):
        options = _run_options(store, monkeypatch, _task(84, max_turns=4))
    assert options.max_turns == GLOBAL_MAX_TURNS
    warned = max_turns_warnings(caplog)
    assert len(warned) == 1, warned
    message = warned[0]
    assert "84" in message, message
    assert "max_turns=4" in message, message
    assert "60" in message, message


def test_the_floor_does_not_warn_for_a_task_that_declared_nothing(caplog):
    """Second positive control: absence of a declaration is the normal case, so
    warning there too would make the real signal unreadable."""
    with caplog.at_level(logging.WARNING, logger="lloyd-autonomy"):
        got = autonomy._resolve_task_max_turns({"id": 39}, GLOBAL_MAX_TURNS)
    assert got == GLOBAL_MAX_TURNS
    assert max_turns_warnings(caplog) == []


def test_a_number_recovered_as_a_string_is_still_read_as_a_number():
    """The scheduler's degraded parser re-parses each line and usually recovers
    the int, but a quoted or commented line comes back as raw text — and the
    budget that job asked for must survive the trip either way."""
    assert autonomy._resolve_task_max_turns(
        {"id": 48, "max_turns": "6"}, GLOBAL_MAX_TURNS) == GLOBAL_MAX_TURNS
    assert autonomy._resolve_task_max_turns(
        {"id": 24, "max_turns": " 90 "}, GLOBAL_MAX_TURNS) == 90


@pytest.mark.parametrize("declared", ["", "   ", None, "ninety", "0", "-5", 0])
def test_an_unusable_declaration_falls_back_to_the_global(caplog, declared):
    """A budget that cannot be parsed must not become a budget of zero: honouring
    `max_turns: ""` as 0 would have the harness stop the run at once and record
    nothing — the same silent failure wearing a different hat."""
    with caplog.at_level(logging.WARNING, logger="lloyd-autonomy"):
        got = autonomy._resolve_task_max_turns(
            {"id": 24, "max_turns": declared}, GLOBAL_MAX_TURNS)
    assert got == GLOBAL_MAX_TURNS


# ── clause 4: the value survives the degraded frontmatter parse ────────────

def test_max_turns_is_in_the_scheduler_fallback_field_list():
    """The regex extractor recovers listed fields and only listed fields, so an
    unlisted key is silently lost from exactly the task files that are already in
    trouble. Pinned against the source text the way the `inner_voice` entry in
    the same tuple already is."""
    src = Path(autonomy.__file__).read_text()
    assert ('"skill_name", "timeout_seconds", "max_turns", "preemptible", '
            '"auto_advance",') in src


def test_a_task_file_with_broken_yaml_still_yields_its_declared_budget(store,
                                                                      monkeypatch):
    """End to end across the parser that is clause 4's whole point.

    `name: nightly: reflection: again` is the classic agent-written breakage — an
    unquoted colon — so the strict parse fails, the orphaned-tags repair cannot
    fix it, and the file comes back `_yaml_broken` with only the listed fields
    recovered. The declared 90 must still reach `RunOptions`.
    """
    path = store / "24-data-pipeline.md"
    path.write_text(
        "---\n"
        "id: 24\n"
        "name: nightly: reflection: again\n"
        "status: up_next\n"
        "skill_name: autonomy-data-pipeline\n"
        "timeout_seconds: 2400\n"
        "max_turns: 90\n"
        "---\n\n# body\n",
        encoding="utf-8")
    parsed = autonomy._parse_task_file(path)
    assert parsed is not None
    assert parsed["_yaml_broken"] is True, "fixture must exercise the fallback path"
    assert parsed["max_turns"] == 90

    monkeypatch.setattr(autonomy, "_parse_task_file", lambda p: parsed)
    captured = _stub_run(monkeypatch, parsed, store)
    asyncio.run(autonomy.run_task(24))
    assert captured["options"].max_turns == 90
