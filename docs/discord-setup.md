# Lloyd on Discord — Setup & User Guide

## Overview

Lloyd is available on Discord via DMs. He uses two access tiers:

- **Alan (owner)**: Full Lloyd — memory, vault, tasks, code, everything
- **Friends (approved)**: Conversational Lloyd — chat, jokes, web search, general knowledge. No access to personal info, vault, files, or tasks.

Unknown users must be approved by Alan before Lloyd will respond.

## Bot Setup (one-time)

### 1. Create the Discord Application

1. Go to https://discord.com/developers/applications
2. Click **New Application** — name it "Lloyd" (or whatever you prefer)
3. Go to the **Bot** tab in the left sidebar
4. Click **Reset Token** — copy it immediately (only shown once)
5. Scroll to **Privileged Gateway Intents** — enable **Message Content Intent** — Save

### 2. Generate the Invite Link

1. Go to **OAuth2** → **URL Generator**
2. Under **Scopes**, check `bot`
3. Under **Bot Permissions**, check:
   - Send Messages
   - Read Message History
   - Add Reactions
   - Use External Emojis
4. Copy the generated URL at the bottom

### 3. Create a Discord Server

Lloyd needs a shared server with users to enable DMs — Discord doesn't allow DMing a bot without one.

1. In Discord, click the **+** button in the server list
2. Choose **Create My Own** — name it anything (e.g., "Lloyd")
3. Open the invite URL from step 2 in your browser
4. Select your new server and authorize

Lloyd won't respond in server channels (they're disabled). The server exists solely as a bridge for DMs.

### 4. Configure OpenClaw

Add the bot token and your Discord user ID to `openclaw.json` under `channels.discord`. See the `channels.discord` section in the config for the full structure.

To get your Discord user ID:
1. Discord → **User Settings** → **Advanced** → enable **Developer Mode**
2. Right-click your own name anywhere → **Copy User ID**

### 5. Restart the Gateway

```bash
distrobox-enter lloyd -- bash -c "kill $(lsof -ti :18789) 2>/dev/null; sleep 2"
distrobox-enter lloyd -- systemctl --user start openclaw-gateway.service
```

Check logs for `[discord] logged in to discord as <bot-id>` to confirm success.

## How to DM Lloyd

1. Open your Discord server where Lloyd is a member
2. Find Lloyd in the **member list** on the right side
3. **Right-click** Lloyd → **Message** (or click his name → **Send Message**)

This opens a DM channel. You do NOT need to send a friend request — bots can't accept those. DMs work through shared server membership.

## How to Give Friends Access

### Step 1: Invite friends to the server

Share your server invite link:
- Server settings → **Invites** → **Create Invite**
- Or right-click any channel → **Invite People**

### Step 2: Friends DM Lloyd

Once in the server, they:
1. Find Lloyd in the member list
2. Right-click → **Message**
3. Send any message

### Step 3: Pairing approval

When a new user DMs Lloyd:
1. They receive a **pairing code** (Lloyd won't respond to their messages yet)
2. Lloyd notifies Alan in his DM with the user's Discord name and ID
3. Alan approves with: `/pair approve <code>`
4. The friend can now chat with Lloyd freely

After approval, the friend is permanently allowlisted — no need to re-pair.

## What Friends Can Do

- Chat naturally with Lloyd (same personality, humor, opinions)
- Ask general knowledge questions
- Request web searches on factual topics
- Participate in group DMs with Lloyd

## What Friends Cannot Do

- Access Alan's personal info (schedule, projects, files, notes, moods)
- Direct Lloyd to run tasks, code, or system commands
- Read vault notes, daily logs, or the task board
- Get Lloyd to forward messages between people

If a friend asks about Alan's personal stuff, Lloyd deflects: "That's Alan's business — ask him directly."

This is enforced at two levels:
1. **Hard tool boundary** — the social agent only has `http_search` and `message` tools. No vault, file, bash, or task tools exist in its context.
2. **System prompt rules** — explicit instructions to not share personal information.

## Group DMs

Group DMs are enabled. Lloyd follows prompt-based engagement rules in groups:
- Responds when @mentioned
- Responds when he can add genuine value
- Responds to correct misinformation
- Stays quiet during casual banter that doesn't need him

## Architecture Notes

Lloyd uses two agents for Discord:
- **`main`** — routes Alan's DMs (full tool access, Opus model)
- **`social`** — routes all other DMs (restricted tools, Sonnet model)

Routing is controlled by `bindings` in `openclaw.json`. Alan's DM matches a specific peer binding; everyone else falls through to the social agent.

Both agents share the same workspace (`~/obsidian/agents/lloyd`) and read `SOUL.md` for personality. The social agent skips `USER.md`, `MEMORY.md`, daily notes, and `HEARTBEAT.md`.
