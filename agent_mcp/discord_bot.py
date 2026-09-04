"""
mcp_server/discord_bot.py
Discord bot (gateway inbound) + MCP outbound tools for Lloyd.

Bot:
  - Receives messages via discord.py gateway, routes to /api/message/stream (SSE)
  - Streams response back into Discord with live edits
  - Auto-threads guild conversations
  - Slash commands: /ask /reset /status /stop /model /sethome
  - Reactions: 👀 in-progress, ✅/❌ result

MCP tools (use Discord REST API, no bot client required):
  - discord_send(channel_id, content)
  - discord_send_embed(channel_id, title, description, fields?, color?)
  - discord_list_channels(guild_id?)
  - discord_get_home_channel()
"""

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

from agent_mcp._shared import make_http_client, text_result
from app.config import service_url
import yaml

logger = logging.getLogger("lloyd-discord")

LLOYD_HOME = Path.home() / "lloyd"
SESSIONS_DIR = LLOYD_HOME / "sessions"
LLOYD_BACKEND = service_url("backend", "http://127.0.0.1:8080")
DISCORD_API = "https://discord.com/api/v10"

# Tools blocked for non-owner Discord users. All entries are bare MCP
# tool names — the harness advertises tools without a namespace prefix.
NON_OWNER_DISALLOWED = [
    # Memory — block writes, allow reads
    "fact_add",
    "fact_resolve",
    "vault_write",
    # Autonomy (all)
    "autonomy_tasks",
    "autonomy_write_task",
    "autonomy_get_task",
    "autonomy_delete_task",
    "autonomy_config",
    "autonomy_run_task",
    # Mission control (all)
    "chat_list_sessions",
    "chat_get_session",
    # Backlog (all)
    "backlog_boards",
    "backlog_tasks",
    "backlog_get_task",
    "backlog_write_task",
    # Browser (all)
    "browser_navigate",
    "browser_snapshot",
    "browser_click",
    "browser_type",
    "browser_scroll",
    "browser_press",
    "browser_tabs",
    "browser_screenshot",
    "browser_evaluate",
    "browser_fill",
    "browser_wait",
    "browser_select",
    "browser_drag",
    "browser_cookies",
    # Built-in shell/file tools
    "Bash",
    "Read",
    "Write",
    "Edit",
    "Glob",
    "Grep",
    "TodoWrite",
]

# ── Global bot state (set during start_bot_task) ──────────────────────────────

_bot = None          # discord.Client instance
_bot_config: dict = {}
_bot_task = None     # asyncio.Task wrapping bot.start()
_active_tasks: dict = {}  # session_id -> asyncio.Task (for /stop)


# ── Config helpers ────────────────────────────────────────────────────────────

def _expand_env(value: str) -> str:
    """Expand ${VAR} placeholders from environment."""
    if not isinstance(value, str):
        return value
    return re.sub(r"\$\{([^}]+)\}", lambda m: os.environ.get(m.group(1), ""), value)


def _load_discord_config() -> dict:
    cfg_path = LLOYD_HOME / "config.yaml"
    if not cfg_path.exists():
        return {}
    raw = yaml.safe_load(cfg_path.read_text()) or {}
    disc = dict(raw.get("discord", {}))
    for key in ("token", "owner_id"):
        if isinstance(disc.get(key), str):
            disc[key] = _expand_env(disc[key])
    return disc


def _save_home_channel(channel_id: str) -> None:
    cfg_path = LLOYD_HOME / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    if "discord" not in cfg:
        cfg["discord"] = {}
    cfg["discord"]["home_channel"] = channel_id
    cfg_path.write_text(yaml.dump(cfg, default_flow_style=False, allow_unicode=True))
    _bot_config["home_channel"] = channel_id


# ── Session file helpers ──────────────────────────────────────────────────────

def _session_path(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}.json"


def _get_thread_id(session_id: str) -> Optional[int]:
    path = _session_path(session_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        tid = data.get("discord_thread_id")
        return int(tid) if tid else None
    except Exception:
        return None


def _store_thread_id(session_id: str, thread_id: int) -> None:
    path = _session_path(session_id)
    try:
        data = json.loads(path.read_text()) if path.exists() else {"session_id": session_id, "platform": "discord"}
        data["discord_thread_id"] = thread_id
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))
    except Exception as e:
        logger.error("Failed to store thread_id for %s: %s", session_id, e)


def _delete_session(session_id: str) -> None:
    path = _session_path(session_id)
    if path.exists():
        path.unlink()


# ── Discord REST API helpers (used by MCP tools + autonomy notify) ─────────────

async def discord_rest_send(channel_id: str, content: str = "", embed: Optional[dict] = None) -> dict:
    """Send a message to a Discord channel via REST API."""
    token = _bot_config.get("token", "")
    if not token:
        return {"error": "No Discord token configured"}
    payload: dict = {}
    if content:
        payload["content"] = content[:2000]
    if embed:
        payload["embeds"] = [embed]
    if not payload:
        return {"error": "No content or embed provided"}
    try:
        async with make_http_client(timeout=15.0) as client:
            resp = await client.post(
                f"{DISCORD_API}/channels/{channel_id}/messages",
                headers={"Authorization": f"Bot {token}", "Content-Type": "application/json"},
                json=payload,
            )
            return resp.json()
    except Exception as e:
        return {"error": str(e)}


# ── MCP tool interface (called by mcp_server/main.py) ─────────────────────────

async def list_tools():
    from mcp.types import Tool
    return [
        Tool(
            name="discord_send",
            description="Send a plain text message to a Discord channel by id. The message is posted immediately and cannot be recalled.",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel_id": {"type": "string", "description": "Discord channel ID"},
                    "content": {"type": "string", "description": "Message text (max 2000 chars)"},
                },
                "required": ["channel_id", "content"],
            },
        ),
        Tool(
            name="discord_send_embed",
            description="Send a rich embed (title, description, coloured sidebar, fields) to a Discord channel by id. Posted immediately and cannot be recalled.",
            inputSchema={
                "type": "object",
                "properties": {
                    "channel_id": {"type": "string", "description": "Discord channel ID"},
                    "title": {"type": "string", "description": "Embed title"},
                    "description": {"type": "string", "description": "Embed body text (max 4096 chars)"},
                    "fields": {
                        "type": "array",
                        "description": "Optional list of {name, value, inline?} field objects",
                        "items": {"type": "object"},
                    },
                    "color": {
                        "type": "integer",
                        "description": "Embed colour as decimal integer (default 5763719 = green)",
                    },
                },
                "required": ["channel_id", "title", "description"],
            },
        ),
        Tool(
            name="discord_list_channels",
            description="List the text channels in a Discord guild, with their ids and names. Use the id with discord_send.",
            inputSchema={
                "type": "object",
                "properties": {
                    "guild_id": {"type": "string", "description": "Guild ID (optional; uses first available guild)"},
                },
                "required": [],
            },
        ),
        Tool(
            name="discord_get_home_channel",
            description="Get the configured Discord home channel ID for autonomous notifications.",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
    ]


async def call_tool(name: str, arguments: dict):

    try:
        if name == "discord_send":
            result = await discord_rest_send(
                channel_id=str(arguments.get("channel_id", "")),
                content=str(arguments.get("content", "")),
            )
            return text_result(json.dumps(result))

        elif name == "discord_send_embed":
            embed: dict = {
                "title": str(arguments.get("title", ""))[:256],
                "description": str(arguments.get("description", ""))[:4096],
                "color": int(arguments.get("color", 5763719)),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            if arguments.get("fields"):
                embed["fields"] = arguments["fields"]
            result = await discord_rest_send(
                channel_id=str(arguments.get("channel_id", "")),
                embed=embed,
            )
            return text_result(json.dumps(result))

        elif name == "discord_list_channels":
            if not _bot:
                return text_result(json.dumps({"error": "Bot not running"}))
            channels = []
            gid = arguments.get("guild_id")
            guilds = [_bot.get_guild(int(gid))] if gid else list(_bot.guilds)
            for guild in guilds:
                if not guild:
                    continue
                for ch in guild.channels:
                    channels.append({
                        "id": str(ch.id),
                        "name": ch.name,
                        "guild": guild.name,
                        "type": str(ch.type),
                    })
            return text_result(json.dumps({"channels": channels}))

        elif name == "discord_get_home_channel":
            return text_result(json.dumps({
                "home_channel": _bot_config.get("home_channel"),
            }))

        else:
            return text_result(json.dumps({"error": f"Unknown tool: {name}"}))

    except Exception as e:
        logger.error("discord_bot call_tool(%s) error: %s", name, e)
        return text_result(json.dumps({"error": str(e)}))


# ── Discord bot (gateway) ─────────────────────────────────────────────────────

def _build_bot():
    """Build the discord.py bot instance with all event handlers and commands."""
    try:
        import discord
        from discord import app_commands
    except ImportError:
        logger.error("discord.py not installed — pip install 'discord.py>=2.3.0'")
        return None

    intents = discord.Intents.default()
    intents.message_content = True
    intents.members = True

    bot = discord.Client(intents=intents)
    tree = app_commands.CommandTree(bot)

    # ── Tier helpers ───────────────────────────────────────────────────────────

    def _is_owner(uid: str) -> bool:
        return str(uid) == str(_bot_config.get("owner_id", ""))

    def _is_allowed(uid: str) -> bool:
        if _is_owner(uid):
            return True
        return str(uid) in [str(u) for u in (_bot_config.get("allowed_users") or [])]

    def _tier(uid: str) -> tuple[list[str], str]:
        """Return (extra_disallowed, permission_mode) for user."""
        if _is_owner(uid):
            return [], "bypassPermissions"
        return list(NON_OWNER_DISALLOWED), "default"

    # ── Session ID helpers ─────────────────────────────────────────────────────

    def _sid_from_message(msg) -> str:
        if isinstance(msg.channel, discord.DMChannel):
            return f"discord:dm:{msg.author.id}"
        if isinstance(msg.channel, discord.Thread):
            parent_id = msg.channel.parent_id or msg.channel.id
            return f"discord:{parent_id}:{msg.author.id}"
        return f"discord:{msg.channel.id}:{msg.author.id}"

    def _sid_from_interaction(interaction) -> str:
        if interaction.guild is None:
            return f"discord:dm:{interaction.user.id}"
        return f"discord:{interaction.channel_id}:{interaction.user.id}"

    # ── Streaming helper ───────────────────────────────────────────────────────

    async def _stream_to_discord(
        trigger_msg,
        text: str,
        session_id: str,
        reply_channel,
    ) -> None:
        """POST to Lloyd backend and stream SSE response into Discord."""
        extra_disallowed, permission_mode = _tier(str(trigger_msg.author.id))
        do_reactions = _bot_config.get("reactions", True)

        # 👀 reaction — processing
        if do_reactions:
            try:
                await trigger_msg.add_reaction("👀")
            except Exception:
                pass

        # Initial placeholder
        reply_msg = None
        try:
            reply_msg = await reply_channel.send("…")
        except Exception as e:
            logger.error("Failed to send placeholder: %s", e)
            return

        # Register task for /stop support
        _active_tasks[session_id] = asyncio.current_task()

        accumulated = ""
        last_edit = 0.0
        success = False

        try:
            async with make_http_client(timeout=None) as client:
                async with client.stream(
                    "POST",
                    f"{LLOYD_BACKEND}/api/message/stream",
                    json={
                        "text": text,
                        "session_id": session_id,
                        "extra_disallowed": extra_disallowed,
                        "permission_mode": permission_mode,
                    },
                ) as resp:
                    event_type = None
                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if line.startswith("event:"):
                            event_type = line[6:].strip()
                        elif line.startswith("data:"):
                            raw = line[5:].strip()
                            if not raw or not event_type:
                                continue
                            try:
                                data = json.loads(raw)
                            except json.JSONDecodeError:
                                continue

                            if event_type == "text_delta":
                                accumulated += data.get("text", "")
                                now = time.monotonic()
                                if now - last_edit >= 1.0 and accumulated:
                                    try:
                                        preview = accumulated[:1990] + ("…" if len(accumulated) > 1990 else "")
                                        await reply_msg.edit(content=preview)
                                        last_edit = now
                                    except Exception:
                                        pass

                            elif event_type == "done":
                                final = data.get("response", accumulated).strip()
                                if final:
                                    await _deliver_final(reply_channel, reply_msg, final)
                                else:
                                    try:
                                        await reply_msg.edit(content="*(no response)*")
                                    except Exception:
                                        pass
                                success = True
                                break

        except asyncio.CancelledError:
            try:
                await reply_msg.edit(content="*(stopped)*")
            except Exception:
                pass
            raise
        except Exception as e:
            logger.error("Stream error for session %s: %s", session_id, e)
            try:
                await reply_msg.edit(content=f"*(error: {e})*")
            except Exception:
                pass
        finally:
            _active_tasks.pop(session_id, None)
            if do_reactions:
                try:
                    await trigger_msg.remove_reaction("👀", bot.user)
                except Exception:
                    pass
                try:
                    await trigger_msg.add_reaction("✅" if success else "❌")
                except Exception:
                    pass

    async def _deliver_final(channel, placeholder_msg, final: str) -> None:
        """Edit/replace placeholder with final response, splitting at 2000 chars."""
        if len(final) <= 2000:
            try:
                await placeholder_msg.edit(content=final)
                return
            except Exception:
                pass
        # Long response — delete placeholder and send in chunks
        try:
            await placeholder_msg.delete()
        except Exception:
            pass
        for i in range(0, len(final), 2000):
            try:
                await channel.send(final[i : i + 2000])
            except Exception as e:
                logger.error("Failed to send chunk: %s", e)
                break

    # ── Thread helpers ─────────────────────────────────────────────────────────

    async def _get_or_create_thread(msg, session_id: str):
        """Return the thread for this session, creating one if needed."""
        if not _bot_config.get("auto_thread", True):
            return None
        import discord as _discord
        if isinstance(msg.channel, (_discord.DMChannel, _discord.Thread)):
            return None

        existing_tid = _get_thread_id(session_id)
        if existing_tid:
            thread = bot.get_channel(existing_tid)
            if thread and not getattr(thread, "archived", True):
                return thread

        try:
            archive_mins = int(_bot_config.get("auto_thread_archive_duration", 1440))
            thread = await msg.create_thread(
                name=f"Lloyd · {msg.author.display_name}",
                auto_archive_duration=archive_mins,
            )
            _store_thread_id(session_id, thread.id)
            return thread
        except Exception as e:
            logger.error("Thread creation failed: %s", e)
            return None

    # ── on_ready ──────────────────────────────────────────────────────────────

    @bot.event
    async def on_ready():
        guild_names = ", ".join(g.name for g in bot.guilds) or "(none)"
        logger.info("Discord bot ready: %s | Guilds: %s", bot.user, guild_names)
        try:
            await tree.sync()
            logger.info("Discord slash commands synced globally")
        except Exception as e:
            logger.error("Slash command sync failed: %s", e)

    # ── on_message ────────────────────────────────────────────────────────────

    @bot.event
    async def on_message(message):
        import discord as _discord
        if message.author == bot.user:
            return
        if not _is_allowed(str(message.author.id)):
            return

        is_dm = isinstance(message.channel, _discord.DMChannel)
        is_thread = isinstance(message.channel, _discord.Thread)
        free_channels = [str(c) for c in (_bot_config.get("free_response_channels") or [])]
        require_mention = _bot_config.get("require_mention", True)

        # In non-DM, non-thread channels check for mention
        if not is_dm and not is_thread:
            in_free = str(message.channel.id) in free_channels
            mentioned = bot.user in (message.mentions or [])
            if require_mention and not in_free and not mentioned:
                return

        # Strip bot mention
        text = message.content or ""
        if bot.user:
            text = text.replace(f"<@{bot.user.id}>", "").replace(f"<@!{bot.user.id}>", "").strip()
        if not text:
            return

        session_id = _sid_from_message(message)

        # Determine reply channel
        if is_thread:
            # Already in thread — store thread→session mapping if not set
            parent_id = message.channel.parent_id or message.channel.id
            base_session = f"discord:{parent_id}:{message.author.id}"
            _store_thread_id(base_session, message.channel.id)
            reply_channel = message.channel
        elif is_dm:
            reply_channel = message.channel
        else:
            thread = await _get_or_create_thread(message, session_id)
            reply_channel = thread if thread else message.channel

        asyncio.create_task(_stream_to_discord(message, text, session_id, reply_channel))

    # ── Slash commands ─────────────────────────────────────────────────────────

    @tree.command(name="ask", description="Send a message to Lloyd")
    @app_commands.describe(message="Your message to Lloyd")
    async def cmd_ask(interaction, message: str):
        if not _is_allowed(str(interaction.user.id)):
            await interaction.response.send_message("Not authorized.", ephemeral=True)
            return
        await interaction.response.defer()
        session_id = _sid_from_interaction(interaction)
        extra_disallowed, permission_mode = _tier(str(interaction.user.id))
        asyncio.create_task(
            _stream_interaction(interaction, message, session_id, extra_disallowed, permission_mode)
        )

    @tree.command(name="reset", description="Clear your current Lloyd session")
    async def cmd_reset(interaction):
        if not _is_allowed(str(interaction.user.id)):
            await interaction.response.send_message("Not authorized.", ephemeral=True)
            return
        session_id = _sid_from_interaction(interaction)
        # Archive thread if one exists
        tid = _get_thread_id(session_id)
        if tid:
            import discord as _discord
            thread = bot.get_channel(tid)
            if thread and isinstance(thread, _discord.Thread):
                try:
                    await thread.edit(archived=True)
                except Exception:
                    pass
        _delete_session(session_id)
        await interaction.response.send_message("Session cleared.", ephemeral=True)

    @tree.command(name="status", description="Show Lloyd's current status")
    async def cmd_status(interaction):
        if not _is_allowed(str(interaction.user.id)):
            await interaction.response.send_message("Not authorized.", ephemeral=True)
            return
        session_id = _sid_from_interaction(interaction)
        running = session_id in _active_tasks
        home = _bot_config.get("home_channel") or "not set"
        await interaction.response.send_message(
            f"**Lloyd Status**\n"
            f"Session: `{session_id}`\n"
            f"Running: {'yes' if running else 'no'}\n"
            f"Home channel: {home}",
            ephemeral=True,
        )

    @tree.command(name="stop", description="Interrupt the current Lloyd run")
    async def cmd_stop(interaction):
        if not _is_allowed(str(interaction.user.id)):
            await interaction.response.send_message("Not authorized.", ephemeral=True)
            return
        session_id = _sid_from_interaction(interaction)
        task = _active_tasks.get(session_id)
        if task and not task.done():
            task.cancel()
            await interaction.response.send_message("Stopped.", ephemeral=True)
        else:
            await interaction.response.send_message("Nothing running.", ephemeral=True)

    @tree.command(name="model", description="Switch Lloyd's model (owner only)")
    @app_commands.describe(model_name="Model name or alias (e.g. sonnet, primary)")
    async def cmd_model(interaction, model_name: str):
        if not _is_owner(str(interaction.user.id)):
            await interaction.response.send_message("Owner only.", ephemeral=True)
            return
        session_id = _sid_from_interaction(interaction)
        try:
            async with make_http_client(timeout=10.0) as client:
                await client.post(
                    f"{LLOYD_BACKEND}/api/model/switch",
                    json={"model": model_name, "session_id": session_id},
                )
        except Exception as e:
            await interaction.response.send_message(f"Failed: {e}", ephemeral=True)
            return
        await interaction.response.send_message(f"Model switched to `{model_name}`.", ephemeral=True)

    @tree.command(name="sethome", description="Set this channel as Lloyd's home for notifications (owner only)")
    async def cmd_sethome(interaction):
        if not _is_owner(str(interaction.user.id)):
            await interaction.response.send_message("Owner only.", ephemeral=True)
            return
        channel_id = str(interaction.channel_id)
        try:
            _save_home_channel(channel_id)
            await interaction.response.send_message(
                f"Home channel set to <#{channel_id}>.", ephemeral=True
            )
        except Exception as e:
            await interaction.response.send_message(f"Failed to save: {e}", ephemeral=True)

    # ── Interaction streaming (for /ask command) ───────────────────────────────

    async def _stream_interaction(interaction, text: str, session_id: str,
                                   extra_disallowed: list, permission_mode: str) -> None:
        accumulated = ""
        last_edit = 0.0
        followup_msg = None

        try:
            async with make_http_client(timeout=None) as client:
                async with client.stream(
                    "POST",
                    f"{LLOYD_BACKEND}/api/message/stream",
                    json={
                        "text": text,
                        "session_id": session_id,
                        "extra_disallowed": extra_disallowed,
                        "permission_mode": permission_mode,
                    },
                ) as resp:
                    event_type = None
                    async for line in resp.aiter_lines():
                        line = line.strip()
                        if line.startswith("event:"):
                            event_type = line[6:].strip()
                        elif line.startswith("data:"):
                            raw = line[5:].strip()
                            if not raw or not event_type:
                                continue
                            try:
                                data = json.loads(raw)
                            except json.JSONDecodeError:
                                continue

                            if event_type == "text_delta":
                                accumulated += data.get("text", "")
                                now = time.monotonic()
                                if now - last_edit >= 1.0 and accumulated:
                                    preview = accumulated[:1990] + ("…" if len(accumulated) > 1990 else "")
                                    try:
                                        if followup_msg is None:
                                            followup_msg = await interaction.followup.send(preview)
                                        else:
                                            await followup_msg.edit(content=preview)
                                        last_edit = now
                                    except Exception:
                                        pass

                            elif event_type == "done":
                                final = data.get("response", accumulated).strip()
                                if not final:
                                    final = "*(no response)*"
                                chunks = [final[i : i + 2000] for i in range(0, len(final), 2000)]
                                for i, chunk in enumerate(chunks):
                                    if i == 0:
                                        if followup_msg:
                                            await followup_msg.edit(content=chunk)
                                        else:
                                            await interaction.followup.send(chunk)
                                    else:
                                        await interaction.followup.send(chunk)
                                break

        except Exception as e:
            logger.error("Interaction stream error: %s", e)
            try:
                await interaction.followup.send(f"*(error: {e})*")
            except Exception:
                pass

    return bot


# ── Bot lifecycle ─────────────────────────────────────────────────────────────

async def start_bot_task() -> None:
    """Start the Discord bot as a background asyncio task. Called from MCP server startup."""
    global _bot, _bot_task, _bot_config

    _bot_config = _load_discord_config()
    token = _bot_config.get("token", "")

    if not token:
        logger.info("Discord bot disabled — DISCORD_BOT_TOKEN not set")
        return

    _bot = _build_bot()
    if not _bot:
        return

    logger.info("Starting Discord bot...")
    _bot_task = asyncio.create_task(_run_bot(token))


async def _run_bot(token: str) -> None:
    """Run the bot, logging any fatal errors."""
    try:
        await _bot.start(token)
    except Exception as e:
        logger.error("Discord bot exited with error: %s", e)


async def stop_bot() -> None:
    """Gracefully close the bot connection."""
    if _bot and not _bot.is_closed():
        await _bot.close()
    if _bot_task:
        _bot_task.cancel()
