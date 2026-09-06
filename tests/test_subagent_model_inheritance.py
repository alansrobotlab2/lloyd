"""A Task subagent runs on the model of the turn that spawned it.

Before this, `subagents.<type>.model` was pinned to `primary` and
`base_url` fell back to `default_model_base_url()` — which also returns
the primary's endpoint. So a turn running on the secondary delegated
every subagent to the primary, and the secondary was never exercised by
the fan-out work Task exists for. Verified live on 2026-09-06: a chat
turn with `"model": "secondary"` spawned a general-purpose subagent that
answered from `:8096`.

Task runs inside the aggregator process, so the only channel for the
caller's identity is the request `_meta` (`lloyd/model`, `lloyd/base_url`),
carried by `MCPPool.call_tool` and bound to contextvars by
`agent_mcp.main.call_tool`.
"""

from __future__ import annotations

import pytest

from agent_mcp import builtin_task
from agent_mcp.builtin_task import (
    _base_url_for,
    current_parent_base_url,
    current_parent_model,
)


class _Text:
    type = "text"
    text = "ok"


class _Result:
    """Stand-in for an MCP CallToolResult."""
    isError = False
    content = [_Text()]


@pytest.fixture(autouse=True)
def _clear_parent():
    """Each test starts with no parent context bound."""
    mt = current_parent_model.set("")
    bt = current_parent_base_url.set("")
    yield
    current_parent_model.reset(mt)
    current_parent_base_url.reset(bt)


def _resolve(profile: dict) -> tuple[str, str]:
    """Mirror of the resolution in `_task`, which is inline there."""
    model = profile["model"] or current_parent_model.get("") or "primary"
    return model, (profile["base_url"] or _base_url_for(model))


# --------------------------------------------------------------- inherit --

def test_inherits_parent_model_when_profile_is_unpinned():
    current_parent_model.set("secondary")
    current_parent_base_url.set("http://127.0.0.1:8091")
    model, base_url = _resolve({"model": "", "base_url": ""})
    assert model == "secondary"
    assert base_url == "http://127.0.0.1:8091"


def test_falls_back_to_primary_outside_a_turn():
    """Task invoked with no parent context (a bare aggregator call)."""
    model, base_url = _resolve({"model": "", "base_url": ""})
    assert model == "primary"
    assert base_url


def test_explicit_profile_pin_beats_inheritance():
    """A profile that names a model still wins — the escape hatch."""
    current_parent_model.set("secondary")
    current_parent_base_url.set("http://127.0.0.1:8091")
    model, base_url = _resolve({"model": "primary", "base_url": ""})
    assert model == "primary"
    assert "8091" not in base_url


def test_explicit_base_url_beats_everything():
    current_parent_model.set("secondary")
    current_parent_base_url.set("http://127.0.0.1:8091")
    _, base_url = _resolve({"model": "", "base_url": "http://example.invalid:1234"})
    assert base_url == "http://example.invalid:1234"


# -------------------------------------------------------------- base_url --

def test_pinned_model_resolves_its_own_endpoint_not_the_default():
    """The latent bug: `model: secondary` + empty base_url hit :8096.

    `default_model_base_url()` returns the DEFAULT model's endpoint, so a
    profile pinned to a non-default alias was sent to an engine that does
    not serve that name.
    """
    model, base_url = _resolve({"model": "secondary", "base_url": ""})
    assert model == "secondary"
    assert "8091" in base_url, f"expected the secondary's port, got {base_url}"


def test_parent_url_wins_over_config_for_the_same_model():
    """The parent streams from a real endpoint; trust it over config.

    Covers a caller that overrode base_url for its own turn — the
    subagent must follow the turn, not the config default.
    """
    current_parent_model.set("secondary")
    current_parent_base_url.set("http://127.0.0.1:9999")
    assert _base_url_for("secondary") == "http://127.0.0.1:9999"


def test_unknown_alias_still_returns_something_usable():
    assert _base_url_for("no-such-model").startswith("http")


# ------------------------------------------------------------------ wire --

def test_config_profiles_are_unpinned():
    """Config must actually opt in, or the plumbing changes nothing."""
    from app.config import CONFIG

    for name, profile in (CONFIG.get("subagents") or {}).items():
        assert not (profile.get("model") or ""), (
            f"subagents.{name}.model is pinned to "
            f"{profile['model']!r}; subagents will not inherit"
        )


def test_meta_keys_agree_across_the_process_boundary():
    """The harness and the aggregator are separate processes."""
    from agent_mcp import main as agg
    from app.harness import mcp_pool

    assert mcp_pool.META_MODEL == agg.META_MODEL
    assert mcp_pool.META_BASE_URL == agg.META_BASE_URL


@pytest.mark.asyncio
async def test_pool_puts_model_in_meta(monkeypatch):
    """`MCPPool.call_tool` ships the caller's model in `_meta`, not args."""
    from app.harness.mcp_pool import MCPPool

    pool = MCPPool.__new__(MCPPool)
    pool._opened = True
    pool._tool_routes = {"Task": "lloyd-mcp"}
    pool._http_configs = {"lloyd-mcp": {}}
    pool._sessions = {}
    pool._schemas = {}

    seen: dict = {}

    async def _fake_invoke(server, bare, args, budget, meta):
        seen["args"] = args
        seen["meta"] = meta
        return _Result()

    monkeypatch.setattr(pool, "_invoke", _fake_invoke)
    await pool.call_tool(
        "Task", {"prompt": "go"},
        session_id="s1", model="secondary", base_url="http://127.0.0.1:8091",
    )

    assert seen["meta"]["lloyd/model"] == "secondary"
    assert seen["meta"]["lloyd/base_url"] == "http://127.0.0.1:8091"
    # Never in args: the MCP server validates args against the tool's
    # inputSchema before the handler runs.
    assert "model" not in seen["args"]
    assert "base_url" not in seen["args"]


@pytest.mark.asyncio
async def test_pool_omits_meta_entirely_when_nothing_to_say(monkeypatch):
    """No session and no model → `_meta` stays None, as before."""
    from app.harness.mcp_pool import MCPPool

    pool = MCPPool.__new__(MCPPool)
    pool._opened = True
    pool._tool_routes = {"Bash": "lloyd-mcp"}
    pool._http_configs = {"lloyd-mcp": {}}
    pool._sessions = {}
    pool._schemas = {}

    seen: dict = {}

    async def _fake_invoke(server, bare, args, budget, meta):
        seen["meta"] = meta
        return _Result()

    monkeypatch.setattr(pool, "_invoke", _fake_invoke)
    await pool.call_tool("Bash", {"command": "true"})
    assert seen["meta"] is None
