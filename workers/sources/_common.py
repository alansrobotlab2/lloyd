"""Shared helpers for knowledge-acquisition sources.

All these sources follow the same pattern:
  1. enqueue_if_due scans some watermark / input and enqueues items
  2. execute builds a prompt for the primary model via the priority proxy
  3. response lands under ~/obsidian/pending-research/{source}/{yyyy-mm-dd}/
"""

from __future__ import annotations

import logging
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
    """Dispatch a prompt to the primary model through the priority proxy."""
    from app.harness import run_query, RunOptions
    from app.harness.mcp_pool import DEFAULT_LLOYD_MCP_URL
    from prompt_builder import build_system_prompt
    from autonomy import _get_model_env, _to_bg_url

    system_prompt = build_system_prompt()
    cfg = yaml.safe_load((LLOYD_HOME / "config.yaml").read_text()) or {}

    disallowed = list(cfg.get("tools", {}).get("disabled_builtin", []))
    for name, sc in cfg.get("mcp_servers", {}).items():
        for tname in sc.get("disabled_tools", []):
            disallowed.append(f"mcp__{name}__{tname}")

    model_env = _to_bg_url(_get_model_env("primary"))

    options = RunOptions(
        model="primary",
        base_url=model_env.get("ANTHROPIC_BASE_URL", "http://127.0.0.1:8096"),
        system_prompt=system_prompt,
        max_turns=max_turns,
        permission_mode="bypassPermissions",
        mcp_servers={"lloyd-mcp": {"type": "sse", "url": DEFAULT_LLOYD_MCP_URL}},
        disallowed_tools=disallowed,
        env=model_env,
    )

    messages = [{"role": "user", "content": prompt}]
    final = ""
    async for evt in run_query(messages, options):
        if evt["type"] == "text_delta":
            final += evt.get("text", "")
    return final
