"""Discord webhook helper for autonomy task-completion notifications."""

import logging
import os
import re
from datetime import datetime

from app.config import CONFIG


logger = logging.getLogger("lloyd-server")


def _discord_token() -> str:
    """Read DISCORD_BOT_TOKEN from config, expanding ${VAR} placeholders."""
    raw = CONFIG.get("discord", {}).get("token", "")
    if not isinstance(raw, str):
        return ""
    return re.sub(r"\$\{([^}]+)\}", lambda m: os.environ.get(m.group(1), ""), raw)


async def _discord_notify_task_complete(task_id: int, task_name: str, response_preview: str) -> None:
    """Post an autonomy task-completion embed to the Discord home channel."""
    home_channel = CONFIG.get("discord", {}).get("home_channel")
    token = _discord_token()
    if not home_channel or not token:
        return
    if not response_preview or response_preview.strip() == "[SILENT]":
        return
    embed = {
        "title": f"Task Complete: {task_name}",
        "description": response_preview[:2000],
        "color": 5763719,  # green
        "timestamp": datetime.utcnow().isoformat(),
        "footer": {"text": f"Task #{task_id}"},
    }
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            await client.post(
                f"https://discord.com/api/v10/channels/{home_channel}/messages",
                headers={"Authorization": f"Bot {token}", "Content-Type": "application/json"},
                json={"embeds": [embed]},
            )
    except Exception as e:
        logger.warning("Discord notify failed: %s", e)
