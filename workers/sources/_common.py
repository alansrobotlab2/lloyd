"""Shared helpers for knowledge-acquisition sources.

All these sources follow the same pattern:
  1. enqueue_if_due scans some watermark / input and enqueues items
  2. execute builds a prompt for the primary model via the priority proxy
  3. response lands under ~/obsidian/pending-research/{source}/{yyyy-mm-dd}/
"""

from __future__ import annotations

import asyncio
import logging
import os
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.paths import LLOYD_HOME

logger = logging.getLogger("lloyd-workers.common")

STAGING_ROOT = Path.home() / "obsidian" / "pending-research"


def staging_dir(source: str) -> Path:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    d = STAGING_ROOT / source / today
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_staging_note(
    source: str,
    slug: str,
    body: str,
    confidence: float = 0.5,
    rationale: str = "",
    source_refs: Optional[list[str]] = None,
) -> Path:
    """Write a structured note under pending-research/{source}/{date}/{slug}.md."""
    d = staging_dir(source)
    # Avoid collisions within the same minute.
    ts = datetime.now(timezone.utc).strftime("%H%M%S")
    path = d / f"{ts}-{slug}.md"
    fm = {
        "source": source,
        "confidence": round(confidence, 2),
        "review_status": "pending",
        "rationale": rationale,
        "source_refs": source_refs or [],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    content = f"---\n{yaml.dump(fm, default_flow_style=False, allow_unicode=True)}---\n\n{body}\n"
    path.write_text(content, encoding="utf-8")
    return path


async def run_prompt_on_primary(prompt: str, max_turns: int = 20) -> str:
    """Dispatch a prompt to the primary model through the priority proxy.

    Runs in executor (Claude Agent SDK uses asyncio.run internally and we call
    it from async context — so shell it out to a thread).
    """
    def _run() -> str:
        from claude_agent_sdk import query as sdk_query, ClaudeAgentOptions
        from prompt_builder import build_system_prompt
        from autonomy import _get_model_env, _to_bg_url

        system_prompt = build_system_prompt()
        cfg = yaml.safe_load((LLOYD_HOME / "config.yaml").read_text()) or {}

        mcp_servers = {}
        disallowed = list(cfg.get("tools", {}).get("disabled_builtin", []))
        for name, sc in cfg.get("mcp_servers", {}).items():
            if not sc.get("enabled", True):
                continue
            stype = sc.get("type", "stdio")
            if stype in ("sse", "http"):
                mcp_servers[name] = {"type": stype, "url": sc["url"]}
            else:
                mcp_servers[name] = {"command": sc.get("command", "python"),
                                     "args": sc.get("args", [])}
            for tname in sc.get("disabled_tools", []):
                disallowed.append(f"mcp__{name}__{tname}")

        options = ClaudeAgentOptions(
            model="primary",
            system_prompt=system_prompt,
            max_turns=max_turns,
            permission_mode="bypassPermissions",
            mcp_servers=mcp_servers,
            disallowed_tools=disallowed,
        )

        model_env = _to_bg_url(_get_model_env("primary"))
        old = {}
        for k, v in model_env.items():
            old[k] = os.environ.get(k)
            os.environ[k] = v

        try:
            final = ""

            async def _q():
                nonlocal final
                async for msg in sdk_query(prompt=prompt, options=options):
                    if hasattr(msg, "content"):
                        for block in msg.content:
                            if hasattr(block, "text"):
                                final += block.text

            asyncio.run(_q())
            return final or ""
        finally:
            for k, old_val in old.items():
                if old_val is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = old_val

    return await asyncio.get_event_loop().run_in_executor(None, _run)
