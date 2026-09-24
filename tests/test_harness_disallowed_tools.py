"""A disabled tool is disabled under every spelling the loop dispatches (#727).

The model is advertised bare names, but `MCPPool.call_tool` still accepts the
legacy `mcp__<server>__<tool>` form so old session JSON replays. Until #727
`loop._pre_dispatch` compared the emitted name raw, so `mcp__lloyd-mcp__Bash`
matched nothing in a deny list carrying `Bash` (the spelling config and plan
mode write) and reached a hook walk whose safety deny keys on
`tool_name == "Bash"` — a one-spelling route past the disabled-tool check and
the destructive-Bash deny, both of which the pool then dispatched. Measured
before the fix: both cases below PASSED pre-dispatch.

These run the real `_pre_dispatch` and the real safety hook; only the MCP pool
is absent, because a refusal is decided before it is reached.
"""
from __future__ import annotations

import asyncio

import pytest

from app.harness.hooks import HookRegistry
from app.harness.loop import _pre_dispatch
from app.harness.options import RunOptions
from app.harness.safety import install_default_safety_hook
from app.harness.tool_search import LoadedToolSet

LEGACY = "mcp__lloyd-mcp__Bash"


def _tc(name: str, command: str) -> dict:
    return {"id": "call_1", "function": {"name": name},
            "_args_dict": {"command": command}}


def _pre(tc: dict, **opts):
    return asyncio.run(_pre_dispatch(
        tc=tc, options=RunOptions(model="m", **opts), session_id="s",
        loaded_set=LoadedToolSet(enabled=False, catalog=[], loaded=set()),
    ))


@pytest.mark.parametrize("emitted", ["Bash", LEGACY], ids=["bare", "legacy"])
@pytest.mark.parametrize("denied", ["Bash", LEGACY], ids=["deny-bare", "deny-legacy"])
def test_a_disabled_tool_is_refused_under_either_spelling(emitted, denied):
    evt = _pre(_tc(emitted, "ls"), disallowed_tools=[denied])
    assert evt is not None, f"{emitted!r} dispatched past a deny list of [{denied!r}]"
    assert evt["is_error"] is True
    assert "disabled by configuration" in evt["content"]
    # The event records what the model emitted; call_id is the join key.
    assert evt["name"] == emitted and evt["call_id"] == "call_1"


def test_the_runtime_deny_set_is_normalised_too():
    """Plan mode refreshes the set per iteration through `runtime_disallowed`;
    it is the same gate and must read the same way."""
    evt = asyncio.run(_pre_dispatch(
        tc=_tc(LEGACY, "ls"), options=RunOptions(model="m"), session_id="s",
        loaded_set=LoadedToolSet(enabled=False, catalog=[], loaded=set()),
        runtime_disallowed={"Bash"},
    ))
    assert evt is not None and evt["is_error"] is True


def test_the_safety_deny_fires_on_the_legacy_spelling():
    hooks = HookRegistry()
    install_default_safety_hook(hooks)
    evt = _pre(_tc(LEGACY, "rm -rf ~/obsidian"), hooks=hooks)
    assert evt is not None, "destructive Bash reached dispatch under the legacy spelling"
    assert evt["is_error"] is True
    assert "harness safety" in evt["content"], evt["content"]


def test_a_harmless_legacy_call_still_passes_to_dispatch():
    """The normalisation must not turn the legacy spelling itself into a refusal:
    the pool accepts it on purpose, for replayed transcripts."""
    hooks = HookRegistry()
    install_default_safety_hook(hooks)
    assert _pre(_tc(LEGACY, "ls"), hooks=hooks) is None


def test_the_hook_walk_sees_the_bare_name():
    seen: list[str] = []

    async def _cb(input_data, _tool_use_id, _ctx):
        seen.append(input_data["tool_name"])
        return {}

    hooks = HookRegistry()
    hooks.add_pre_tool_use(None, _cb)
    assert _pre(_tc(LEGACY, "ls"), hooks=hooks) is None
    assert seen == ["Bash"]
