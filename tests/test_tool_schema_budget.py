"""The full tool schema is what every request ships, and it has a ceiling now.

`harness.tool_search.enabled` is false in `config.yaml` (and in the override
file), so `format_catalog_reminder`'s gist ceiling — the only tool-cost test
before this — guards a path that never runs. What actually rides in every
request is `build_tool_list` over the whole pool plus the `summary` parameter
`add_summary_param` grows on each tool; #361 grew that surface by ~500 tokens
while measuring the inert one (#639). 71fc9086 cut it (34.8k → 29.5k by its
own estimator) and pinned nothing, so nothing says when it grows back.

The payload is assembled the way `app/harness/loop.py` assembles it: the
aggregator's `list_tools()` in the pool's dict shape, `build_tool_list` with
the chat surface's hidden set as the disallowed set (chat is the larger
surface — worker hides nine more), then `add_summary_param`. The count is
`app.compaction.estimate_tokens` over the JSON, the estimator the item's own
numbers used; the engine renders the array into the system message through
its chat template, so this is a consistent growth gauge, not the exact bill.

Thunderbird's tools are excluded by owning module: its bridge exists only in
production, so a test that counted them would read one number there and
another in every worktree, sandbox and gate run. The exclusion makes the
figure the same everywhere the suite runs.
"""
from __future__ import annotations

import asyncio
import json

from agent_mcp import main as M
from agent_mcp.annotations import hidden_on_surface
from app.compaction import estimate_tokens
from app.harness.mcp_pool import _annotations, _input_schema
from app.harness.tool_schema import add_summary_param, build_tool_list

#: Measured 31,025 on 2026-09-24 (105 tools on the chat surface, Thunderbird
#: excluded). Raise it on purpose, with the new figure, when a tool surface
#: is deliberately grown; do not raise it to make a red run go green.
FULL_SCHEMA_TOKEN_CEILING = 33_000

#: Below this the pool is broken, not lean — an empty discovery must not pass
#: a ceiling test (see "An empty tool pool is the worst failure" in CLAUDE.md).
FULL_SCHEMA_TOKEN_FLOOR = 20_000

EXCLUDED_MODULES = ("thunderbird",)


def _module_of(name: str) -> str:
    mod = M._dispatch.get(name)
    return mod.__name__.rsplit(".", 1)[-1] if mod is not None else ""


def live_chat_schema() -> list[dict]:
    """The OpenAI `tools=[...]` payload a chat turn ships, from the live modules."""
    tools = asyncio.run(M.list_tools())
    discovered = [("lloyd-mcp", [
        {"name": t.name, "description": t.description or "",
         "inputSchema": _input_schema(t), "annotations": _annotations(t)}
        for t in tools if _module_of(t.name) not in EXCLUDED_MODULES
    ])]
    catalog = build_tool_list(discovered, set(hidden_on_surface("chat")))
    add_summary_param(catalog)
    return catalog


def schema_tokens(catalog: list[dict]) -> int:
    return estimate_tokens(json.dumps(catalog, ensure_ascii=False))


def test_the_full_schema_a_chat_turn_ships_stays_under_its_ceiling():
    catalog = live_chat_schema()
    tokens = schema_tokens(catalog)
    silent = sorted(name for name, st in M._discovery_status.items()
                    if st["tools"] == 0 and name not in EXCLUDED_MODULES)
    detail = (f"{tokens} tokens for {len(catalog)} tools (ceiling "
              f"{FULL_SCHEMA_TOKEN_CEILING}, floor {FULL_SCHEMA_TOKEN_FLOOR}); "
              f"modules that exported nothing: {silent or 'none'}")
    assert len(catalog) >= 80, detail
    assert tokens >= FULL_SCHEMA_TOKEN_FLOOR, detail
    assert tokens <= FULL_SCHEMA_TOKEN_CEILING, detail


def test_the_gauge_moves_with_the_schema():
    """A ceiling only means something if the measure it reads can exceed it."""
    catalog = live_chat_schema()
    base = schema_tokens(catalog)
    padded = catalog + [{
        "type": "function",
        "function": {"name": f"padding_{i}", "description": "x" * 4000,
                     "parameters": {"type": "object", "properties": {}}},
    } for i in range(4)]
    assert schema_tokens(padded) > FULL_SCHEMA_TOKEN_CEILING > base
