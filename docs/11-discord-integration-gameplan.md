# Discord Integration: Analysis & Implementation Gameplan

> **Date**: 2026-04-04  
> **Status**: Planning  
> **References**: `~/Projects/openclaw/extensions/discord/`, `~/Projects/hermes-agent/gateway/platforms/discord.py`

---

## 1. Problem Statement

Lloyd currently has no Discord presence. All interaction is via the web UI. A Discord bot would let Lloyd be a first-class participant in servers — receiving messages, running tasks, and delivering autonomous output — without requiring users to open a browser.

Both reference projects have mature Discord integrations. This document analyzes them and defines an implementation path for Lloyd.

---

## 2. Reference Implementation Analysis

### 2.1 Hermes-Agent (`~/Projects/hermes-agent`)

**Stack**: Python, `discord.py` 2.7+, asyncio, platform adapter pattern

**Architecture**:
```
Discord Gateway
    │
    ▼
DiscordAdapter(BasePlatformAdapter)
    │
    ├── on_message → _handle_message → session routing → agent run → send()
    ├── Slash commands (/ask, /reset, /model, /stop, /status, ...)
    ├── Reactions (👀 in-progress, ✅/❌ result)
    └── Auto-threading (optional, per channel config)
```

**Session model**: Sessions keyed on `(platform, chat_id, user_id, thread_id)`. Each user in a shared channel has isolated context (`group_sessions_per_user`). Sessions persist to disk across restarts.

**Config** (env + YAML):
```bash
DISCORD_BOT_TOKEN=...
DISCORD_ALLOWED_USERS=284102345871466496,9876543210
DISCORD_REQUIRE_MENTION=true
DISCORD_FREE_RESPONSE_CHANNELS=chan1,chan2   # no mention needed
DISCORD_REACTIONS=true
DISCORD_HOME_CHANNEL=123456789012345678     # cron delivery target
```

**Voice support**: Full pipeline — RTP packet capture, SSRC→user_id mapping, Opus→PCM decode, silence detection, STT (Whisper/Groq/local). Complex; skip for initial implementation.

**Strengths**: Python-native (matches Lloyd's stack), clean adapter pattern, well-tested slash commands, proven session isolation model.

---

### 2.2 OpenClaw (`~/Projects/openclaw`)

**Stack**: TypeScript, Discord.js via `@buape/carbon`, monorepo plugin system

**Architecture**:
```
OpenClaw Plugin Host
    │
    ▼
Discord Channel Plugin
    │
    ├── inbound: DiscordMonitorProvider (gateway events)
    ├── outbound: Discord action handlers (send, embed, react, moderate)
    ├── Slash commands (via SDK abstraction)
    ├── Button-based exec approvals
    └── Per-thread session bindings with TTL
```

**Strengths**: Richer interactive components (buttons, modals, select menus), multi-account support, sophisticated approval/permission system.

**Relevance to Lloyd**: High-quality reference for interactive components and permission patterns — but TypeScript and OpenClaw-specific, so not directly portable.

---

## 3. Design Decisions for Lloyd

### 3.1 Framework

**Choice: `discord.py` (Python)**

Lloyd's backend is Python/FastAPI. Using `discord.py` keeps everything in one language and one venv. The hermes integration is a directly portable reference.

### 3.2 Integration Point

Lloyd uses the Claude Agent SDK's `query()` call (`server.py`). The Discord bot will be a **separate asyncio task** running inside the same FastAPI process, or optionally as a dedicated MCP server. The simpler approach: run the bot alongside FastAPI using `asyncio` task group on startup.

Preferred approach: **`mcp-servers/discord.py`** — a standalone MCP server that exposes tools for sending Discord messages, while also running the inbound bot listener internally. This gives Lloyd the ability to proactively post to Discord, and lets the bot route inbound messages through the standard `/api/message/stream` HTTP endpoint (loopback to itself).

```
Discord Gateway
    │
    ▼
mcp-servers/discord.py  (discord.py bot, asyncio)
    │
    ├── INBOUND: on_message / slash commands
    │       └──► POST http://localhost:8080/api/message/stream  (SSE)
    │               └──► streams agent response back to Discord
    │
    └── OUTBOUND: MCP tools exposed to Lloyd
            ├── discord_send(channel_id, message)
            ├── discord_send_embed(channel_id, title, description, fields)
            └── discord_get_channels()
```

This pattern means:
- Lloyd can be asked "post a summary to #updates" and it calls `discord_send` as a tool
- Inbound Discord messages arrive as normal agent sessions via HTTP

### 3.3 Session Strategy

Adopt hermes's session key pattern: `discord:{channel_id}:{user_id}` for guild channels, `discord:dm:{user_id}` for DMs. Stored in Lloyd's existing session directory (`sessions/<session_id>.json`).

**Per-user isolation** in shared channels: enabled by default (matches hermes behavior).

### 3.4 Authentication & Access Control

- Bot token in `.env` or `config.yaml` secrets block
- `owner_id` in `config.yaml` — the owner's Discord user ID (single value)
- `allowed_users` list — other users permitted to interact at all (reduced tier)
- Optional `require_mention` flag (default `true` for guild channels, `false` for DMs)

### 3.5 Capability Tiers

Lloyd's tool and permission surface in Discord is **caller-dependent**. The owner gets the full agent; everyone else gets a constrained one.

**Owner** (`owner_id` in config) — full capabilities regardless of whether it's a DM or a guild channel:
- All MCP tools enabled (file system, memory, autonomy, browser, thunderbird, etc.)
- All built-in Claude tools enabled (Bash, Read, Write, Edit, etc.)
- Full `bypassPermissions` mode (same as web UI sessions)
- `/model`, `/personality`, `/reset`, `/stop`, `/status` slash commands

**Others** (`allowed_users`) — reduced surface:
- Memory: read-only — `fact_get`, `fact_profile`, `fact_check`, `vault_get`, `vault_overview`, `vault_search`, `vault_recall` allowed; **`fact_add`, `fact_resolve`, `vault_write` blocked**
- `http_tools`: all tools allowed (fetch, search — no side effects)
- `pipeline`: read/status tools allowed
- All other MCP servers blocked entirely (autonomy, mission_control, backlog, subliminal, thunderbird, browser)
- No Claude built-in file/shell tools (Bash, Read, Write, Edit, Glob, Grep)
- `default` permission mode (not bypass)
- Only `/ask`, `/reset`, `/status` slash commands available

The write-capable memory tools are blocked because they mutate Lloyd's persistent knowledge base (`~/obsidian/`). An untrusted Discord user should be able to ask Lloyd to recall facts or search the vault, but not plant or alter facts.

The capability tier is resolved when the message is received and passed to `/api/message/stream` as a session parameter. The backend constructs a `disallowed_tools` list before calling `query()`.

**Disallowed tools list for non-owner tier:**
```python
NON_OWNER_DISALLOWED = [
    # Memory — write tools only; reads remain available
    "mcp__memory__fact_add",
    "mcp__memory__fact_resolve",
    "mcp__memory__vault_write",
    # Blocked MCP servers entirely
    "mcp__autonomy__*",
    "mcp__mission_control__*",
    "mcp__backlog__*",
    "mcp__subliminal__*",
    "mcp__thunderbird__*",
    "mcp__browser__*",
    # Built-in Claude tools
    "Bash",
    "Read",
    "Write",
    "Edit",
    "Glob",
    "Grep",
    "TodoWrite",
]
```

**Config representation**:
```yaml
discord:
  owner_id: "284102345871466496"
  allowed_users: []              # other permitted users (reduced tier)
  # users not in either list are silently ignored
```

**How enforcement works (Option A):**

The bot resolves the tier at message-receive time, builds the disallowed list locally, and POSTs it directly to the server as `extra_disallowed`. The server merges it with the globally disabled tools from `config.yaml` before building `ClaudeAgentOptions`. `permission_mode` travels with it for the same reason — a guest with `bypassPermissions` would still bypass prompts for whatever tools they can reach.

Bot side (`mcp_server/discord.py`):
```python
owner_id = config["discord"]["owner_id"]
allowed_users = config["discord"]["allowed_users"]

if str(message.author.id) == owner_id:
    extra_disallowed = []
    permission_mode = "bypassPermissions"
elif str(message.author.id) in allowed_users:
    extra_disallowed = NON_OWNER_DISALLOWED
    permission_mode = "default"
else:
    return  # not in either list — silently ignore

await httpx.post("http://localhost:8080/api/message/stream", json={
    "text": message.content,
    "session_id": session_key,
    "extra_disallowed": extra_disallowed,
    "permission_mode": permission_mode,
})
```

Server side (`server.py`) — two lines change in `post_message_stream`:
```python
extra_disallowed = data.get("extra_disallowed", [])
permission_mode = data.get("permission_mode") or CONFIG.get("agent", {}).get("permission_mode", "bypassPermissions")

options = ClaudeAgentOptions(
    ...
    permission_mode=permission_mode,
    disallowed_tools=_get_disallowed_tools() + extra_disallowed,
    ...
)
```

The same change applies to the synchronous `/api/message` endpoint (line ~565) to keep both consistent.

This is safe because the bot is a local process — nothing external can reach `localhost:8080` to forge a permissive `extra_disallowed=[]`. The trust boundary is the bot's tier check, which runs before the HTTP call.

### 3.6 Commands

Minimal initial slash command set (matches hermes):

| Command | Action |
|---------|--------|
| `/ask <message>` | Send message to Lloyd |
| `/reset` | Clear current session |
| `/status` | Show running/idle state |
| `/stop` | Interrupt current agent run |
| `/model <name>` | Switch model (choices from `config.yaml`) |

Regular `@mention` or message in a free-response channel routes to `/ask` implicitly.

### 3.7 Streaming Responses

The existing `/api/message/stream` SSE endpoint streams agent responses as chunks. The Discord bot will:

1. Post an initial "typing..." indicator (or use `channel.typing()`)
2. Accumulate SSE chunks into a string buffer
3. Edit the message in-place as content arrives (Discord edit API)
4. On completion, add ✅ or ❌ reaction

Discord has a 2000-char message limit — long responses will be split into multiple messages or sent as a file attachment.

---

## 4. Implementation Phases

### Phase 1 — Core Bot (DMs + basic guild)

**Goal**: Lloyd responds to Discord messages and slash commands.

1. Add `discord.py` to `requirements.txt`
2. Create `mcp-servers/discord.py`:
   - `DiscordBot` class with `discord.Client` and intents (message content, members, guilds)
   - `on_ready` handler — log connected guilds, register slash commands
   - `on_message` handler — allowlist check, session key build, POST to `/api/message/stream`, stream response back
   - `@discord.app_commands.command` for `/ask`, `/reset`, `/status`, `/stop`
3. Add `discord` section to `config.yaml`:
   ```yaml
   discord:
     token: "${DISCORD_BOT_TOKEN}"
     owner_id: "${DISCORD_OWNER_ID}"
     allowed_users: []
     require_mention: true
     free_response_channels: []
     reactions: true
     home_channel: null
   ```
4. Register discord MCP server in `config.yaml` under `mcp_servers`
5. Start bot task in `server.py` via `asyncio` lifespan event (or launch as a separate supervisor process)
6. Add `DISCORD_BOT_TOKEN` to `.env`

**Test**: DM the bot `@Lloyd hello` → gets a response. `/reset` clears the session.

---

### Phase 2 — Streaming + Reactions

**Goal**: Responses stream into Discord in real-time.

1. Implement SSE consumer in the bot: read chunks from `/api/message/stream`, buffer text
2. On first chunk: send an initial Discord message, save `message_id`
3. On subsequent chunks: edit the message with accumulated text (throttled to ~1 edit/sec to avoid rate limits)
4. On `done` event: finalize message, add ✅ or ❌ reaction based on success
5. Add 👀 reaction when processing begins
6. Handle Discord 2000-char limit: split into follow-up messages

**Test**: Long agent response shows streaming updates live in Discord.

---

### Phase 3 — Auto-threading

**Goal**: Each conversation gets its own thread to keep channels clean.

1. On first message in a guild channel, create a thread from that message
2. All subsequent turns in the same session go into that thread
3. Store `discord_thread_id` in the session JSON
4. On session reset (`/reset`), archive the old thread and create a new one on next message
5. Add `auto_thread: true` to discord config (default true for guilds, false for DMs)

**Test**: Multiple users in `#general` each get isolated threads with their Lloyd conversations.

---

### Phase 4 — Outbound MCP Tools

**Goal**: Lloyd can proactively post to Discord as a tool action.

1. Expose MCP tools from `mcp-servers/discord.py`:
   - `discord_send(channel_id, content)` — post text to a channel
   - `discord_send_embed(channel_id, title, description, fields, color)` — rich embed
   - `discord_list_channels(guild_id?)` — list available channels
   - `discord_get_home_channel()` — return configured home channel ID
2. Register in `config.yaml` under `mcp_servers`
3. Test: ask Lloyd "post a daily summary to #updates" → it calls `discord_send`

**Test**: Lloyd's autonomy tasks can deliver output to Discord without user prompting.

---

### Phase 5 — Home Channel & Cron Delivery

**Goal**: Autonomous tasks (from `autonomy.py`) can push notifications to a configured Discord channel.

1. Add `home_channel` to discord config
2. Add `/sethome` slash command — sets the current channel as home, writes to `config.yaml`
3. Modify `autonomy.py` task runner to optionally call `discord_send` MCP tool when a task completes, using the home channel
4. Notification format: embed with task name, result summary, and timestamp

**Test**: A scheduled task completes and a summary appears in #bot-updates without user intervention.

---

### Phase 6 — Per-User Memory Stores (Future)

**Goal**: Each `allowed_user` gets their own isolated memory space so Lloyd can build context about them over time without that context bleeding into the owner's vault or other users' stores.

**Problem with the current model**: The memory MCP server has one vault root (`~/obsidian/`) and one facts store (`~/obsidian/memory/_pipeline/facts/`). If a guest user's facts were written there, they'd sit alongside the owner's facts with no isolation. The read-only restriction in Phase 1 avoids this problem by not writing at all — but it also means Lloyd can't learn anything persistent about guest users.

**Proposed approach**: Each Discord user gets a subdirectory within a `discord_users/` namespace:

```
~/obsidian/discord_users/
    <discord_user_id>/
        facts/          # their entity facts (same schema as main facts store)
        notes/          # free-form notes Lloyd saves about them
```

The bot passes the user's Discord ID to the server alongside the message. The server (or a new `discord_memory` MCP server variant) scopes all reads and writes to that user's subdirectory.

**Options for implementation**:

- **Option A — Parameterized memory server**: The existing `memory.py` MCP server is extended to accept a `--vault-root` and `--facts-root` argument at launch. The server spawns a separate instance per active Discord user session, pointed at their personal subdirectory. Clean isolation, no code duplication — but more subprocess overhead.

- **Option B — Separate `discord_memory` MCP server**: A thin wrapper that re-uses the same logic but hard-codes paths under `~/obsidian/discord_users/<user_id>/`. The bot passes `user_id` as a startup arg; the server is spawned on-demand per session. More code, but more explicit.

- **Option C — Namespace prefix in existing server**: Add an optional `user_scope` parameter to `fact_add`, `vault_write`, etc. that prefixes all paths with `discord_users/<user_id>/`. Simpler but mixes owner and guest data in the same server instance.

Option A is likely cleanest — the memory server is already self-contained and the path constants at the top are easy to parameterize.

**What changes**:
- Guest users get write access back to their personal store (`fact_add`, `vault_write` unblocked — but scoped)
- Owner sessions continue using the full vault, unchanged
- `vault_search` and `vault_recall` for guest sessions search only their personal store (not the owner's vault)
- Lloyd can greet returning users by name, remember preferences, track ongoing topics

**Prerequisites**: Phases 1–3 must be complete. Per-user memory is only useful once threading gives each user a persistent session identity.

---

## 5. Config Schema (Final State)

```yaml
discord:
  token: "${DISCORD_BOT_TOKEN}"       # or inline (not recommended)
  owner_id: "284102345871466496"      # full capabilities — your Discord user ID
  allowed_users: []                   # other permitted users (reduced capabilities)
  require_mention: true               # guild channels require @mention
  free_response_channels:            # channel IDs — no mention needed
    - "123456789012345678"
  reactions: true                     # 👀/✅/❌ reaction feedback
  auto_thread: true                   # create thread per conversation in guilds
  home_channel: null                  # set via /sethome or config directly
  auto_thread_archive_duration: 1440 # minutes before thread auto-archives (24h)
```

---

## 6. File Layout

```
~/lloyd/
├── mcp-servers/
│   └── discord.py          # Bot client + MCP tool server (new)
├── config.yaml             # add discord section + mcp_servers entry
├── .env                    # add DISCORD_BOT_TOKEN
├── server.py               # add lifespan hook to start bot (or delegate to supervisord)
└── docs/
    └── 11-discord-integration-gameplan.md
```

If the bot is resource-intensive (voice phase), it can be broken out as a separate supervisord service (`lloyd-mc:lloyd-discord`).

---

## 7. Discord Developer Setup Checklist

Before implementation:

- [ ] Create a Discord application at [discord.com/developers/applications](https://discord.com/developers/applications)
- [ ] Create a Bot under the application, copy the token to `.env`
- [ ] Enable **Privileged Intents**: Message Content Intent, Server Members Intent
- [ ] Generate an invite URL with scopes: `bot`, `applications.commands`
- [ ] Permissions: Send Messages, Read Message History, Add Reactions, Create Public Threads, Manage Threads, Embed Links, Attach Files
- [ ] Invite bot to test server

---

## 8. Deferred / Out of Scope

| Feature | Reason deferred |
|---------|-----------------|
| Voice channels (TTS/STT) | High complexity; not needed for text-first use cases |
| Multi-guild multi-account | Single bot token covers all servers |
| Button-based exec approvals | Claude SDK permission mode handles this already |
| DAVE E2EE voice decryption | Voice is deferred entirely |
| Pairing/invite codes for DMs | Allowlist approach is sufficient |
