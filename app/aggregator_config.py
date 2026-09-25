"""One place that knows how to reach the aggregator, and with what headers.

The aggregator (`agent_mcp/main.py`) binds loopback and now refuses any request
that does not carry its boot credential (#1053). That makes every consumer need
two things it previously needed one of: the URL *and* the header. Four backend
call sites named the origin independently before this module existed —
`dashboard._MCP_STATE_URL`, `browser._MCP_NAVIGATE_URL`, and the inline
`service_url(...).rstrip('/mcp')` arithmetic in `sessions.revert_turn` and
`messages._fetch_files_changed` — so a port change was a four-file grep and a
credential change would have been another. All four derive from here now;
`services.lloyd_mcp` in config.yaml stays the single source for the origin.

Why the origin is parsed and re-joined instead of the config value being used
whole: that value carries the `/mcp` path, which is the JSON-RPC endpoint rather
than the origin the side routes (`/state`, `/changes`, `/changes/revert`,
`/browser/navigate`, `/health`) hang off. That is the reasoning
`browser._MCP_NAVIGATE_URL`'s own comment gave when it chose to hardcode; the fix
is to do that derivation once instead of once per proxy.

Two known callers stay outside this module. `app/routers/health.py` probes only
`/health`, which needs no credential and is a liveness check rather than a route
dependency, and `agent-services/guardian/policy.py:MCP_HEALTH_URL` is a detached
copy that cannot import `app` — both are on the open path by design.
"""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

from app.config import service_url

#: Every route the aggregator serves, under its canonical path. `health` is the
#: only one that needs no credential. These are the aggregator's own route table
#: (`agent_mcp/main.py`), not a deployment choice, so there is no override: a
#: knob no test pins is a knob that silently diverges from the server.
_PATHS: dict[str, str] = {
    "mcp": "/mcp",
    "health": "/health",
    "state": "/state",
    "changes": "/changes",
    "changes_revert": "/changes/revert",
    "browser_navigate": "/browser/navigate",
    # `{task_id}` and `{verb}` are filled by `subagent_route`.
    "subagent_control": "/subagents/{task_id}/{verb}",
}


def _origin() -> str:
    """scheme://host:port of the aggregator, from `services.lloyd_mcp`."""
    parts = urlsplit(service_url("lloyd_mcp", "http://127.0.0.1:8500/mcp"))
    return urlunsplit((parts.scheme, parts.netloc, "", "", ""))


def route(name: str) -> str:
    """Absolute URL of one aggregator route, e.g. `route("changes_revert")`."""
    return _origin() + _PATHS[name]


def subagent_route(task_id: str, verb: str) -> str:
    """Absolute URL of `POST /subagents/{task_id}/{verb}` (P8), id quoted."""
    from urllib.parse import quote
    return _origin() + _PATHS["subagent_control"].format(
        task_id=quote(task_id, safe=""), verb=quote(verb, safe=""))


def auth_headers_for(url: str) -> dict[str, str]:
    """Credential headers for a request to `url` (empty if it is not loopback).

    Re-exported here so a backend caller imports one module and gets the URL and
    the header from it together, which is the pairing that stops one proxy being
    updated and the other forgotten. The implementation is the server's own
    module — one definition of the header name and of where the token lives,
    shared with the process that checks it.
    """
    from agent_mcp.aggregator_auth import headers_for_url
    return headers_for_url(url)
