"""Shared helpers for knowledge-acquisition sources.

All these sources follow the same pattern:
  1. enqueue_if_due scans some watermark / input and enqueues items
  2. execute builds a prompt for the primary model at low vLLM priority (1)
     so interactive chat can preempt it.
  3. response lands under ~/obsidian/pending-research/{source}/{yyyy-mm-dd}/
"""

from __future__ import annotations

import logging
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.paths import LLOYD_HOME, VAULT_PENDING_RESEARCH_DIR as STAGING_ROOT

logger = logging.getLogger("lloyd-workers.common")


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
    """Dispatch a prompt to the primary model at low vLLM priority."""
    from app.harness import run_query, RunOptions
    from app.harness.mcp_pool import DEFAULT_LLOYD_MCP_SERVERS
    from prompt_builder import build_system_prompt
    from autonomy import _get_model_env

    # The landing drain applies to worker turns too. The promoter idles the
    # backend and then restarts it; a worker job that starts in that gap is
    # killed mid-flight, and the connection errors it logs on the way down
    # land inside the observation window and are blamed on the promotion. That
    # is the exact shape of the 2026-09-06 20:14 false positive.
    try:
        from app.routers.selfmod import drain_active, drain_remaining
        if drain_active():
            raise RuntimeError(
                f"lloyd is landing a code update; not starting a worker turn "
                f"(retry in {drain_remaining():.0f}s)")
    except ImportError:
        pass

    system_prompt = build_system_prompt()
    cfg = yaml.safe_load((LLOYD_HOME / "config.yaml").read_text()) or {}

    disallowed: list[str] = []
    for name, sc in cfg.get("mcp_servers", {}).items():
        for tname in sc.get("disabled_tools", []):
            disallowed.append(f"mcp__{name}__{tname}")

    # A worker job may not drive the self-modification loop. These tools were
    # advertised to every worker prompt, `domain-research` included — and that
    # one reads arbitrary web pages into its context, so the machinery that
    # rewrites production sat one prompt injection away from a source whose
    # entire job is ingesting untrusted text. The backlog triage worker was
    # told not to start a round IN ITS PROMPT, which is not a control.
    for tname in ("selfmod_start", "selfmod_gate", "selfmod_land",
                  "selfmod_abort", "selfmod_rollback"):
        disallowed.append(tname)
        disallowed.append(f"mcp__lloyd-mcp__{tname}")

    model_env = _get_model_env("primary")

    options = RunOptions(
        model="primary",
        base_url=model_env.get("ANTHROPIC_BASE_URL", "http://127.0.0.1:8096"),
        system_prompt=system_prompt,
        max_turns=max_turns,
        permission_mode="bypassPermissions",
        mcp_servers=DEFAULT_LLOYD_MCP_SERVERS,
        disallowed_tools=disallowed,
        env=model_env,
        priority=1,
    )

    messages = [{"role": "user", "content": prompt}]
    final = ""
    async for evt in run_query(messages, options):
        if evt["type"] == "text_delta":
            final += evt.get("text", "")
    return final
