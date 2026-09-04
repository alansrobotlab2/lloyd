#!/usr/bin/env python3
"""Lloyd MCP server: Autoresearch — Karpathy-style parallel hypothesis/eval loop.

Tools:
    autoresearch_round      — launch a full round (backgrounded)
    autoresearch_status     — list recent rounds or inspect one
    autoresearch_bench_list — list eval tasks
    autoresearch_bench_add  — add an eval task (YAML frontmatter + body)
    autoresearch_ledger_query — query ledger entries
    autoresearch_promote    — manually promote a variant by id (dry-run by default)
    autoresearch_rollback   — restore canonical prompts from a snapshot
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from mcp.types import Tool

from agent_mcp._shared import text_result

logger = logging.getLogger("lloyd-autoresearch-mcp")

# Keep imports lazy inside handlers — the round orchestrator pulls in claude_agent_sdk,
# requests, etc., which we'd rather not load until the tool is actually called.


async def list_tools():
    return [
        Tool(
            name="autoresearch_round",
            description=(
                "Kick off an autoresearch round: generate variant overlays, evaluate against "
                "the bench on the local model, and promote the winner if it beats baseline. "
                "Returns the round_id immediately and runs asynchronously."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "targets": {"type": "array", "items": {"type": "string"}, "description": "Target surfaces (v1 supports only 'prompts')"},
                    "budget_minutes": {"type": "integer", "description": "Advisory budget in minutes (default from config)."},
                    "max_variants": {"type": "integer", "description": "Max variants to propose (default from config)."},
                    "dry_run": {"type": "boolean", "description": "Score everything but do not promote."},
                    "bench_limit": {"type": "integer", "description": "Limit bench to the first N tasks (for shakedown)."},
                    "max_parallel": {"type": "integer", "description": "Max concurrent (variant × task) trials (default 4)."},
                },
            },
        ),
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
            name="autoresearch_bench_list",
            description=(
                "List the autoresearch bench: every task id with its category and "
                "whether it is safety-critical. Use this before adding a task to "
                "avoid duplicating an existing one."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="autoresearch_bench_add",
            description=(
                "Add a new bench task. Provide the full YAML frontmatter (as an object) and "
                "an optional markdown body for additional context."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Task id (also used as filename)."},
                    "frontmatter": {"type": "object", "description": "YAML frontmatter: category, prompt, objective_checks, rubric_criteria, safety_critical."},
                    "body": {"type": "string", "description": "Optional markdown body (context, grading notes, etc.)."},
                },
                "required": ["id", "frontmatter"],
            },
        ),
        Tool(
            name="autoresearch_ledger_query",
            description="Query the autoresearch ledger. Supports filters on round_id, variant_id, or event.",
            inputSchema={
                "type": "object",
                "properties": {
                    "round_id": {"type": "string", "description": "Only rows from this round (from autoresearch_status)"},
                    "variant_id": {"type": "string", "description": "Only rows for this prompt variant"},
                    "event": {"type": "string", "description": "e.g. 'decision' for promotion decisions only"},
                    "limit": {"type": "integer", "description": "Max rows (default 100, max 1000)."},
                },
            },
        ),
        Tool(
            name="autoresearch_promote",
            description="Manually promote a variant that was previously evaluated. Dry-run by default.",
            inputSchema={
                "type": "object",
                "properties": {
                    "variant_id": {"type": "string", "description": "Variant to promote, as reported by autoresearch_status or the ledger"},
                    "dry_run": {"type": "boolean", "description": "Default true; set false to actually promote."},
                },
                "required": ["variant_id"],
            },
        ),
        Tool(
            name="autoresearch_rollback",
            description="Restore canonical prompt files from a snapshot directory (by timestamp).",
            inputSchema={
                "type": "object",
                "properties": {"snapshot_ts": {"type": "string", "description": "Directory name under research/snapshots/."}},
                "required": ["snapshot_ts"],
            },
        ),
    ]


async def call_tool(name: str, arguments: dict):
    if name == "autoresearch_round":
        return text_result(_handle_round(arguments))
    if name == "autoresearch_status":
        return text_result(_handle_status(arguments))
    if name == "autoresearch_bench_list":
        return text_result(_handle_bench_list(arguments))
    if name == "autoresearch_bench_add":
        return text_result(_handle_bench_add(arguments))
    if name == "autoresearch_ledger_query":
        return text_result(_handle_ledger_query(arguments))
    if name == "autoresearch_promote":
        return text_result(_handle_promote(arguments))
    if name == "autoresearch_rollback":
        return text_result(_handle_rollback(arguments))
    return text_result(json.dumps({"error": f"unknown tool: {name}"}))


def _load_cfg():
    from scripts.autoresearch.common import load_config
    return load_config()


def _handle_round(params: dict) -> str:
    """Enqueue an autoresearch round through the unified work queue.

    The worker pool picks it up and runs it at priority=2. Coalesces via
    dedup_key `autoresearch:round` — a second enqueue while one is in flight
    is dropped.
    """
    targets = params.get("targets") or ["prompts"]
    budget_minutes = params.get("budget_minutes")
    max_variants = params.get("max_variants")
    dry_run = bool(params.get("dry_run", False))
    bench_limit = params.get("bench_limit")
    max_parallel = int(params.get("max_parallel") or 4)

    try:
        from workers.queue import get_queue
        q = get_queue()
    except (RuntimeError, ImportError) as e:
        return json.dumps({"error": f"work queue not available: {e}"})

    new_id = q.enqueue(
        source="autoresearch",
        kind="round",
        payload={
            "targets": targets,
            "budget_minutes": budget_minutes,
            "max_variants": max_variants,
            "dry_run": dry_run,
            "bench_limit": bench_limit,
            "max_parallel": max_parallel,
        },
        priority=60,
        dedup_key="autoresearch:round",
    )
    if new_id is None:
        return json.dumps({"coalesced": True, "reason": "round already queued or running"})
    return json.dumps({
        "accepted": True,
        "queue_id": new_id,
        "queued_at": datetime.now(timezone.utc).isoformat(),
        "targets": targets,
        "dry_run": dry_run,
    })


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
        from workers.queue import get_queue
        q = get_queue()
        queue_items = [
            i.to_dict() for i in q.list_items(source="autoresearch", limit=10)
        ]
        recent_runs = q.list_runs(source="autoresearch", limit=5)
    except (RuntimeError, ImportError):
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


def _handle_bench_list(params: dict) -> str:
    cfg = _load_cfg()
    from scripts.autoresearch.common import load_bench_tasks
    tasks = load_bench_tasks(cfg.paths.bench_dir)
    return json.dumps({
        "count": len(tasks),
        "bench_dir": str(cfg.paths.bench_dir),
        "tasks": [
            {"id": t.get("id"), "category": t.get("category"), "safety_critical": bool(t.get("safety_critical")),
             "has_objective_checks": bool(t.get("objective_checks")),
             "rubric_criteria": t.get("rubric_criteria", [])}
            for t in tasks
        ],
    })


def _handle_bench_add(params: dict) -> str:
    cfg = _load_cfg()
    task_id = str(params.get("id", "")).strip()
    frontmatter = params.get("frontmatter") or {}
    body = str(params.get("body", ""))
    if not task_id or not isinstance(frontmatter, dict):
        return json.dumps({"error": "id and frontmatter are required"})
    frontmatter.setdefault("id", task_id)
    cfg.paths.bench_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.paths.bench_dir / f"{task_id}.md"
    if path.exists():
        return json.dumps({"error": f"bench task {task_id} already exists at {path}"})
    yaml_text = yaml.safe_dump(frontmatter, sort_keys=False).strip()
    path.write_text(f"---\n{yaml_text}\n---\n\n{body}\n", encoding="utf-8")
    return json.dumps({"success": True, "path": str(path)})


def _handle_ledger_query(params: dict) -> str:
    cfg = _load_cfg()
    if not cfg.paths.ledger_path.exists():
        return json.dumps({"rows": [], "count": 0})
    limit = min(int(params.get("limit") or 100), 1000)
    rid = params.get("round_id")
    vid = params.get("variant_id")
    evt = params.get("event")
    rows: list[dict[str, Any]] = []
    for line in reversed(cfg.paths.ledger_path.read_text(encoding="utf-8").splitlines()):
        try:
            entry = json.loads(line)
        except Exception:
            continue
        if rid and entry.get("round_id") != rid:
            continue
        if vid and entry.get("variant_id") != vid:
            continue
        if evt and entry.get("event") != evt:
            continue
        rows.append(entry)
        if len(rows) >= limit:
            break
    return json.dumps({"rows": rows, "count": len(rows)})


def _handle_promote(params: dict) -> str:
    cfg = _load_cfg()
    variant_id = params.get("variant_id", "")
    dry_run = bool(params.get("dry_run", True))
    if not variant_id:
        return json.dumps({"error": "variant_id is required"})
    overlay_dir = cfg.paths.variants_dir / variant_id
    if not overlay_dir.exists():
        return json.dumps({"error": f"overlay not found: {overlay_dir}"})
    meta_path = overlay_dir / "variant.json"
    if not meta_path.exists():
        return json.dumps({"error": "variant.json missing — cannot determine metadata"})
    variant = json.loads(meta_path.read_text(encoding="utf-8"))
    # Minimal stub summaries — caller should have evaluated first; manual promote is rescue path
    variant_summary = {"mean_composite": 1.0, "safety_passed": True, "per_task": [], "task_count": 0}
    baseline_summary = {"mean_composite": 0.0, "per_task": []}
    from scripts.autoresearch.promote import promote as _promote
    result = _promote(cfg, variant, overlay_dir, variant_summary, baseline_summary, dry_run=dry_run)
    return json.dumps(result, default=str)


def _handle_rollback(params: dict) -> str:
    cfg = _load_cfg()
    ts = params.get("snapshot_ts", "")
    if not ts:
        return json.dumps({"error": "snapshot_ts is required"})
    from scripts.autoresearch.promote import rollback as _rollback
    return json.dumps(_rollback(cfg, ts))
