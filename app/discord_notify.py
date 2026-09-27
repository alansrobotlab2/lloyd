"""Discord webhook helper for autonomy task-completion notifications.

`discord_alert` is the terminal route for several independent callers — the
scheduler's five alarms in `workers/sources/scheduled_task.py::_alert` (model-server
outage, unparseable task files, their recovery, the due-ness stall and the next_run
stall), `autonomy._record_failure`'s infra-ceiling crossing, and the retry-budget
disable — and an alarm about "autonomy scheduler may be stalled" could burn four
nightly cycles while every trace of it sat in a rotating log file.

This module owns the one decision those callers should not each re-derive: whether
the Discord transport is configured at all and, since #1592, where an alarm goes when
it is not. The config object is the boot-merged one (`config.yaml` +
`data/tool_overrides.yaml` + the process env, merged once at import), read at CALL
time from inside the function rather than copied into a module-level constant: a
token that is unset here can exist later without a code change, and a second, lazier
source of truth would go stale on that change.
"""

import logging
import os
import re
from datetime import datetime

from app.config import CONFIG
from app.silent_sentinel import is_silent_response


logger = logging.getLogger("lloyd-server")


def _discord_token() -> str:
    """Read DISCORD_BOT_TOKEN from config, expanding ${VAR} placeholders."""
    raw = CONFIG.get("discord", {}).get("token", "")
    if not isinstance(raw, str):
        return ""
    return re.sub(r"\$\{([^}]+)\}", lambda m: os.environ.get(m.group(1), ""), raw)


def _missing_transport_halves() -> list[str]:
    """Which of the two things a Discord post needs are absent right now.

    Named separately rather than as one "not configured", because #1592 puts this
    list on a surface a person reads and they are two different repairs: a channel is
    an id to fill in, a token is a credential the box currently declines to hold.
    """
    missing = []
    if not CONFIG.get("discord", {}).get("home_channel"):
        missing.append("discord.home_channel is unset")
    if not _discord_token():
        missing.append("the bot token is empty")
    return missing


async def _discord_notify_task_complete(task_id: int, task_name: str, response_preview: str) -> None:
    """Post an autonomy task-completion embed to the Discord home channel."""
    home_channel = CONFIG.get("discord", {}).get("home_channel")
    token = _discord_token()
    if not home_channel or not token:
        return
    if not response_preview or is_silent_response(response_preview):
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


def _survive_the_dropped_alert(message: str, title: str) -> None:
    """Put an alarm Discord refused onto today's daily note instead.

    The drop branch used to be a `logger.warning` and a `return`, which on this box
    is not a degraded path but the only one: `config.yaml` carries
    `home_channel: null` and an empty token, there is no `data/tool_overrides.yaml`,
    and `tests/test_autonomy_failure_alert.py` reads that null off the disk and fails
    a round that configures it, because leaving it null is Alan's decision to make and
    not a code change's to make for him. So an alarm here reached a rotating log and no
    person — the 2026-09-27 alarm that #68 was 258 h past its own `next_run` is the
    instance on record, and it is the same class of silence that cost #42 four nightly
    cycles.

    The destination is the daily note, the machine-owned surface `_append_fast_failure_alert`
    already writes alarms to for #1209 and which demonstrably lands on this box. The
    warning beside the write stays on the log: the acceptance for #1592 is that the
    WARNING keeps appearing AND a matching note line appears, not that one replaces the
    other. And nothing raised: the callers are a scheduler tick and a run's failure
    path, and the destination being unavailable is never a reason to change what the
    run or the next tick does.

    `app.autonomy` is imported inside the function for the same reason `httpx` is:
    this module is imported from the MCP server and the backend boot path, where a
    cycle through the autonomy module at import time would be a boot-time failure. If
    that import is ever the thing that stops alerts reaching the note, it surfaces as
    the warning below — which is the same shape as the failure it replaces, and is
    stated here rather than hidden so the next reader sees the trade.
    """
    missing = _missing_transport_halves()
    try:
        from app.autonomy import append_daily_alert_line
        appended = append_daily_alert_line(
            f"**Scheduler alert not delivered** ({'; '.join(missing)}), so it is "
            f"written here instead — {title}: {message[:1000]}")
        if not appended:
            logger.warning("discord_alert: the daily note refused the alert too; "
                           "it exists only in this log line")
    except Exception as e:  # noqa: BLE001 — never the caller's problem
        logger.warning("discord_alert: the daily-note fallback failed: %s", e)


async def discord_alert(message: str, title: str = "⚠️ Autonomy health alert") -> None:
    """Post a plain operational alert (e.g. scheduler stall, unparseable tasks).

    When the transport is unconfigured the alert goes to today's daily note instead of
    evaporating into the log — see `_survive_the_dropped_alert` for why that branch is
    this box's only branch, and for what the note line has to carry.
    """
    home_channel = CONFIG.get("discord", {}).get("home_channel")
    token = _discord_token()
    if not home_channel or not token:
        logger.warning("discord_alert (no channel/token configured): %s", message)
        _survive_the_dropped_alert(message, title)
        return
    embed = {
        "title": title,
        "description": message[:2000],
        "color": 15548997,  # red
        "timestamp": datetime.utcnow().isoformat(),
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
        logger.warning("Discord alert failed: %s", e)
