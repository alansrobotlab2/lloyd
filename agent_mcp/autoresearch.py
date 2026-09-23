#!/usr/bin/env python3
"""Lloyd MCP server: Autoresearch — Karpathy-style parallel hypothesis/eval loop.

Tools:
    autoresearch_status     — list recent rounds or inspect one
    autoresearch_rollback   — restore canonical prompts from a snapshot

Five more were retired on 2026-09-23 as operator verbs no turn called: a round
is scheduled by the `autoresearch` worker source or run by hand with
`python -m scripts.autoresearch.run_round`; the bench is markdown under the
bench dir and the ledger is JSONL, both read with Read/Grep; and the manual
promote was a rescue path that promoted a variant on a stubbed passing score.
`autoresearch_rollback` stays because `post_promotion.py` writes that exact
call into the report a human reads after a promotion.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from mcp.types import Tool

from agent_mcp._shared import text_result

logger = logging.getLogger("lloyd-autoresearch-mcp")

# Keep imports lazy inside handlers — the round orchestrator pulls in claude_agent_sdk,
# requests, etc., which we'd rather not load until the tool is actually called.


async def list_tools():
    return [
        Tool(
            name="autoresearch_status",
            description=(
                "List recent autoresearch rounds with their status and outcome, or "
                "pass round_id to inspect one round's variants and scores."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "round_id": {"type": "string", "description": "Inspect this round (omit to list)."},
                    "limit": {"type": "integer", "description": "Max rounds to list (default 10)."},
                },
            },
        ),
        Tool(
            name="autoresearch_rollback",
            description=(
                "Restore canonical prompt files from a snapshot directory (by timestamp). "
                "The restore goes through the vault landing route: validated against the "
                "prompt surface and the real loaders, committed on the vault's main with the "
                "snapshot ts in the message, and ledgered as a `vault_land` event. A restore "
                "that fails validation is refused and the tree is left at HEAD. `vault_commit` "
                "carries the new sha; `no_change` is true when the restored bytes already "
                "match HEAD."
            ),
            inputSchema={
                "type": "object",
                "properties": {"snapshot_ts": {"type": "string", "description": "Directory name under research/snapshots/."}},
                "required": ["snapshot_ts"],
            },
        ),
    ]


async def call_tool(name: str, arguments: dict):
    if name == "autoresearch_status":
        return text_result(_handle_status(arguments))
    if name == "autoresearch_rollback":
        return text_result(_handle_rollback(arguments))
    return text_result(json.dumps({"error": f"unknown tool: {name}"}))


def _load_cfg():
    from scripts.autoresearch.common import load_config
    return load_config()


def _handle_status(params: dict) -> str:
    cfg = _load_cfg()
    rid = params.get("round_id")
    if rid:
        path = cfg.paths.rounds_dir / f"{rid}.md"
        if not path.exists():
            return json.dumps({"error": f"round {rid} not found"})
        return json.dumps({"round_id": rid, "summary": path.read_text(encoding="utf-8")})

    limit = int(params.get("limit") or 10)
    rounds = sorted(cfg.paths.rounds_dir.glob("R_*.md"), reverse=True)[:limit]

    queue_items: list[dict] = []
    recent_runs: list[dict] = []
    try:
        from workers.queue import configured_db_path, get_queue
        q = get_queue(configured_db_path())
        queue_items = [
            i.to_dict() for i in q.list_items(source="autoresearch", limit=10)
        ]
        recent_runs = q.list_runs(source="autoresearch", limit=5)
    except (RuntimeError, ImportError, OSError):
        pass

    return json.dumps({
        "rounds": [
            {"round_id": p.stem, "path": str(p),
             "mtime": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat()}
            for p in rounds
        ],
        "queue": queue_items,
        "recent_runs": recent_runs,
    })


def _handle_rollback(params: dict) -> str:
    cfg = _load_cfg()
    ts = params.get("snapshot_ts", "")
    if not ts:
        return json.dumps({"error": "snapshot_ts is required"})
    from scripts.autoresearch.promote import rollback as _rollback
    return json.dumps(_rollback(cfg, ts))
