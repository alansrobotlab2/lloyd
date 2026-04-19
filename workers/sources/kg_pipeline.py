"""kg-pipeline source — the knowledge-graph mining chain.

Previously fired by qmd-watcher via POST /api/autonomy/vault-change. Now polls
vault mtime each tick and enqueues a single `chain` item when the vault is
newer than the last successful pipeline run. Coalescing is free (UNIQUE dedup
key `kg-pipeline:chain`).

Chain: data-pipeline → conversation-relation-linking → entity-resolution-sweep
(30-min throttle preserved) → groundskeeper-loop. Executes on primary 122B at
priority=2 through the priority proxy.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from workers.queue import WorkQueue, QueueItem

logger = logging.getLogger("lloyd-workers.kg_pipeline")

NAME = "kg-pipeline"
DEFAULT_PRIORITY = 40
DEDUP_KEY = "kg-pipeline:chain"

VAULT_DIR = Path.home() / "obsidian"
STEPS: list[tuple[str, str]] = [
    ("data-pipeline",
     str(Path.home() / "obsidian/skills/autonomy-data-pipeline/SKILL.md")),
    ("conversation-relation-linking",
     str(Path.home() / "obsidian/skills/conversation-relation-linking/SKILL.md")),
    ("entity-resolution-sweep",
     str(Path.home() / "obsidian/skills/entity-resolution-sweep/SKILL.md")),
    ("groundskeeper-loop",
     str(Path.home() / "obsidian/skills/groundskeeper-loop/SKILL.md")),
]
ENTITY_RESOLUTION_MIN_INTERVAL_SEC = 1800  # preserved from old realtime pipeline


def _max_vault_mtime() -> float:
    """Scan ~/obsidian/**/*.md for the newest mtime. Cheap enough at ~1000s of files."""
    newest = 0.0
    if not VAULT_DIR.exists():
        return newest
    for p in VAULT_DIR.rglob("*.md"):
        try:
            m = p.stat().st_mtime
            if m > newest:
                newest = m
        except OSError:
            continue
    return newest


async def enqueue_if_due(queue: WorkQueue, src_cfg: dict) -> None:
    last_mtime_str = queue.wm_get(NAME, "last_vault_mtime")
    last_mtime = float(last_mtime_str) if last_mtime_str else 0.0
    current_mtime = _max_vault_mtime()
    if current_mtime <= last_mtime:
        return

    # Cooldown — don't fire more often than cooldown_seconds apart.
    cooldown = int(src_cfg.get("cooldown_seconds", 900))
    last_completed = queue.wm_get(NAME, "last_completed_at")
    if last_completed:
        last_dt = datetime.fromisoformat(last_completed)
        elapsed = (datetime.now(timezone.utc) - last_dt).total_seconds()
        if elapsed < cooldown:
            return

    new_id = queue.enqueue(
        source=NAME,
        kind="chain",
        payload={"triggered_mtime": current_mtime},
        priority=int(src_cfg.get("priority", DEFAULT_PRIORITY)),
        dedup_key=DEDUP_KEY,
    )
    if new_id is not None:
        logger.info("Enqueued kg-pipeline chain id=%d (vault_mtime=%.0f)", new_id, current_mtime)


async def execute(item: QueueItem) -> dict[str, Any]:
    triggered_mtime = float(item.payload.get("triggered_mtime", 0.0))
    results = []
    summaries = []

    for step_name, skill_path in STEPS:
        # Entity-resolution-sweep throttle.
        if step_name == "entity-resolution-sweep":
            last_run_str = _load_realtime_state().get(step_name, {}).get("last_run")
            if last_run_str:
                last_run = _parse_iso(last_run_str)
                if last_run:
                    elapsed = (datetime.now(timezone.utc) - last_run).total_seconds()
                    if elapsed < ENTITY_RESOLUTION_MIN_INTERVAL_SEC:
                        logger.info("kg-pipeline: skipping %s (ran %.0fs ago)", step_name, elapsed)
                        results.append({"step": step_name, "skipped": "recent_run"})
                        continue

        result = await asyncio.get_event_loop().run_in_executor(
            None, _run_step, step_name, skill_path
        )
        results.append(result)
        if result.get("success"):
            summaries.append(f"{step_name}:ok")
        else:
            summaries.append(f"{step_name}:FAIL")

    # Persist watermarks.
    from workers.queue import get_queue
    q = get_queue()
    q.wm_set(NAME, "last_vault_mtime", f"{triggered_mtime:.6f}")
    q.wm_set(NAME, "last_completed_at", datetime.now(timezone.utc).isoformat())

    return {
        "summary": " | ".join(summaries),
        "response": json.dumps(results, ensure_ascii=False, default=str)[:50000],
    }


# ── Step executor — routes to primary model via priority proxy ────────────

REALTIME_STATE_PATH = Path.home() / "lloyd" / "autonomy-runs" / "realtime-state.json"


def _load_realtime_state() -> dict:
    if not REALTIME_STATE_PATH.exists():
        return {}
    try:
        return json.loads(REALTIME_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_realtime_state(state: dict) -> None:
    REALTIME_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    REALTIME_STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _parse_iso(s: str):
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _run_step(step_name: str, skill_path: str) -> dict:
    """Execute one KG skill on the primary model at priority=2.

    Mirrors the shape of autonomy._run_realtime_step but without the
    secondary-model routing — everything goes to primary via the
    priority proxy (8096 → 8097).
    """
    from autonomy import (
        _load_skill_content,
        _get_model_env,
        _to_bg_url,
        _write_run_record,  # reuse for parity with existing autonomy-runs/
    )
    from prompt_builder import build_system_prompt
    from claude_agent_sdk import query as sdk_query, ClaudeAgentOptions
    import os
    import yaml
    import uuid
    from app.paths import LLOYD_HOME

    skill_content = _load_skill_content(skill_path)
    if not skill_content:
        return {"success": False, "step": step_name, "error": f"skill not found: {skill_path}"}

    silent_hint = (
        "[SYSTEM: If you have a meaningful status report or findings, "
        "send them. Only respond with exactly \"[SILENT]\" when there is "
        "genuinely nothing new to report.]\n\n"
    )
    prompt = (
        f"{silent_hint}"
        f"[SYSTEM: KG pipeline step: {step_name}. Follow the skill instructions below.]\n\n"
        f"{skill_content}"
    )

    # Always primary — no more secondary routing for KG.
    model = "primary"
    model_env = _to_bg_url(_get_model_env(model))
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    run_id = f"run_rt-{step_name}_{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"

    logger.info("KG step: %s (model=%s)", step_name, model)

    try:
        system_prompt = build_system_prompt()
        config = yaml.safe_load((LLOYD_HOME / "config.yaml").read_text()) or {}
        mcp_servers = {}
        disallowed_tools = list(config.get("tools", {}).get("disabled_builtin", []))
        for name, cfg in config.get("mcp_servers", {}).items():
            if not cfg.get("enabled", True):
                continue
            server_type = cfg.get("type", "stdio")
            if server_type in ("sse", "http"):
                mcp_servers[name] = {"type": server_type, "url": cfg["url"]}
            else:
                mcp_servers[name] = {"command": cfg.get("command", "python"),
                                     "args": cfg.get("args", [])}
            for tool_name in cfg.get("disabled_tools", []):
                disallowed_tools.append(f"mcp__{name}__{tool_name}")

        options = ClaudeAgentOptions(
            model=model,
            system_prompt=system_prompt,
            max_turns=config.get("agent", {}).get("max_turns", 60),
            permission_mode="bypassPermissions",
            mcp_servers=mcp_servers,
            disallowed_tools=disallowed_tools,
        )

        old_env = {}
        for k, v in model_env.items():
            old_env[k] = os.environ.get(k)
            os.environ[k] = v

        try:
            final_response = ""

            async def _run():
                nonlocal final_response
                async for message in sdk_query(prompt=prompt, options=options):
                    if hasattr(message, "content"):
                        for block in message.content:
                            if hasattr(block, "text"):
                                final_response += block.text
            asyncio.run(_run())
        finally:
            for k, old_val in old_env.items():
                if old_val is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = old_val

        if not final_response:
            final_response = "(No response)"

        completed_at = datetime.now(timezone.utc).isoformat()
        duration = (datetime.now(timezone.utc) - now).total_seconds()

        # Preserve existing file layout for continuity.
        runs_dir = Path.home() / "lloyd" / "autonomy-runs" / f"rt-{step_name}"
        runs_dir.mkdir(parents=True, exist_ok=True)
        fm = {
            "run_id": run_id, "step": step_name, "status": "success",
            "started_at": now_iso, "completed_at": completed_at,
            "duration_seconds": round(duration, 1),
            "summary": final_response[:200],
        }
        (runs_dir / f"{run_id}.md").write_text(
            f"---\n{yaml.dump(fm, default_flow_style=False)}---\n\n## Response\n\n{final_response}",
            encoding="utf-8",
        )

        state = _load_realtime_state()
        state[step_name] = {
            "last_run": completed_at, "run_id": run_id,
            "status": "success", "duration_seconds": round(duration, 1),
        }
        _save_realtime_state(state)

        return {"success": True, "step": step_name, "run_id": run_id,
                "duration_seconds": round(duration, 1),
                "response_preview": final_response[:300]}
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        logger.error("KG step %s failed: %s", step_name, error_msg)
        state = _load_realtime_state()
        state[step_name] = {"last_run": now_iso, "status": "failed", "error": error_msg[:200]}
        _save_realtime_state(state)
        return {"success": False, "step": step_name, "error": error_msg}
