"""#750: the dispatch-time skill deliverer is installed on the Task subagent
route, under the same flag as the stream route.

`install_skill_dispatch_hook` had one call site — `post_message_stream` —
while the safety hook was installed at four, so flipping
`harness.skill_dispatch.enabled` taught the streaming chat route and nothing
else. A subagent runs exactly the repeated protocols the deliverer exists for
and gets no turn-start skills at all, so it is the one uncovered route worth
installing on; the other two record at the site why they are excluded.

The flag is read live (`skill_dispatch.enabled()` → config), so both halves
are proven by monkeypatching `_config`, never by editing `config.yaml` — that
key is a human's.
"""

from __future__ import annotations

import asyncio
import inspect
import json

import pytest

from agent_mcp import _subagent_registry as reg
from agent_mcp import builtin_task
from app.harness import skill_dispatch as sd
from app.harness.loop import _pre_dispatch


@pytest.fixture(autouse=True)
def _clean():
    reg.reset()
    sd.reset_stats()
    yield
    reg.reset()
    sd.reset_stats()


class _StubLoadedSet:
    enabled = False
    catalog: list = []
    loaded: set = set()

    def is_visible(self, name: str) -> bool:
        return True


def _subagent_options(monkeypatch):
    """Run `_task` against a fake `run_query` and return the `RunOptions` the
    subagent was built with — its `hooks` is the registry under test."""
    captured = {}

    async def _fake_run_query(messages, options):
        captured["options"] = options
        yield {"type": "result", "stop_reason": "stop", "num_turns": 1, "usage": {}}

    import app.harness.loop as loop_mod
    monkeypatch.setattr(loop_mod, "run_query", _fake_run_query)
    monkeypatch.setattr(
        builtin_task, "_load_subagent_profile",
        lambda t: {"system_prompt": "", "max_turns": 5,
                   "disallowed_tools": [], "model": "primary", "base_url": ""},
    )
    out = asyncio.run(builtin_task._task({"prompt": "probe", "description": "probe"}))
    json.loads(out)
    return captured["options"]


def _deliverers(hooks) -> int:
    return sum(1 for _m, cb in hooks._pre if cb.__name__ == "_skill_dispatch_pretool")


def _drafted_bash(options, command: str):
    return asyncio.run(_pre_dispatch(
        tc={"id": "call_sub", "function": {"name": "Bash"},
            "_args_dict": {"command": command}},
        options=options, session_id=options.session_id,
        loaded_set=_StubLoadedSet(),
    ))


def test_a_subagent_bash_call_is_held_and_answered_with_the_skill_body(monkeypatch):
    """Clause 1: with the flag on, the subagent's registry carries the
    deliverer beside the safety hook, and a matching drafted call comes back
    as a synthetic non-error result carrying the SKILL.md body instead of
    reaching the shell."""
    monkeypatch.setattr(sd, "_config", lambda: {"enabled": True})
    monkeypatch.setattr(sd, "skill_body", lambda name: f"BODY-OF[{name}]")
    options = _subagent_options(monkeypatch)
    assert options.hooks.skill_dispatch_installed is True
    assert _deliverers(options.hooks) == 1
    assert options.hooks._pre[0][1].__name__ == "_safety_pretool_cb", (
        "the safety gate is installed first; the deliverer follows it")

    evt = _drafted_bash(options, "yt-dlp https://youtu.be/x")
    assert evt is not None, "the drafted call should have been held back"
    assert evt["type"] == "tool_result"
    assert evt["is_error"] is False
    assert "NOT executed" in evt["content"]
    assert "BODY-OF[youtube-transcript]" in evt["content"]
    assert sd.stats_snapshot()["delivered"] == 1


def test_with_the_flag_off_a_subagent_call_executes_normally(monkeypatch):
    """Clause 2: the Task install passes no `force_enabled`, so an absent or
    false flag leaves the subagent exactly as default-off as every other route
    — the safety gate alone, and the drafted call proceeds to dispatch."""
    monkeypatch.setattr(sd, "_config", lambda: {})
    monkeypatch.setattr(sd, "skill_body", lambda name: f"BODY-OF[{name}]")
    options = _subagent_options(monkeypatch)
    assert options.hooks.skill_dispatch_installed is False
    assert _deliverers(options.hooks) == 0
    assert _drafted_bash(options, "yt-dlp https://youtu.be/x") is None
    assert sd.stats_snapshot()["delivered"] == 0


def test_the_task_route_installs_the_deliverer_under_the_config_flag():
    """The install sits beside the safety hook and reads the config flag: no
    `force_enabled` at the site, or the subagent would be the one route a
    human's flag does not govern."""
    src = inspect.getsource(builtin_task)
    assert src.count("install_default_safety_hook(task_hooks)") == 1
    assert src.count("install_skill_dispatch_hook(task_hooks)") == 1
    assert "force_enabled" not in src
    body = src[src.index("install_default_safety_hook(task_hooks)"):]
    assert body.index("install_skill_dispatch_hook(task_hooks)") < 600, (
        "the deliverer is installed right after the safety gate, not elsewhere")
