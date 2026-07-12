---
segment: architecture
type: reference
tags: [architecture]

---

# OpenClaw Usage Tracking Architecture

Research into how token usage,costs,and provider quotas are tracked across the OpenClaw system.

## 1. Where Usage Data Is Stored

### Session Transcripts (primary source)
- **Location**: `~/.openclaw/agents/{agentId}/sessions/*.jsonl`
- **Format**: JSONL — one JSON object per line,per message
- Each assistant message includes a `usage` object:
  ```json
  {
    "type": "message",
    "timestamp": "...",
    "message": {
      "role": "assistant",
      "usage": {
        "input": 1234,
        "output": 567,
        "cacheRead": 890,
        "cacheWrite": 100,
        "totalTokens": 2791,
        "cost": { "input": 0.003,"output": 0.008,"total": 0.011 }
      },
      "model": "claude-sonnet-4-6",
      "provider": "anthropic"
    }
  }
  ```
- This is the **only persistent source of per-request token data** for gateway sessions
- No dedicated usage database exists — all aggregation is done by scanning JSONL files at query time

### CC-Instances Logs (Claude Code / Agent SDK subagents)
- **Location**: `~/.openclaw/logs/cc-instances/{instanceId}.jsonl` + `{instanceId}.summary.json`
- **Written by**: `agent-orchestrator` plugin (`query-consumer.ts`)
- JSONL captures: start,tool_use,text,session_init,result,complete events
- Summary JSON captures: id,type,status,task,costUsd,turns,budgetUsd,elapsedMs,resultPreview
- **Cost tracking**: Two mechanisms:
  1. **Per-turn estimation**: `(input_tokens / 1M) * 3 + (output_tokens / 1M) * 15` (hardcoded Sonnet pricing)
  2. **SDK-reported total**: When the Agent SDK's `result` message includes `total_cost_usd`,it overwrites the running estimate
- Logs are persistent on disk but **not aggregated into any database or rolled-up summary**

### Session Usage Update (live state)
- `persistSessionUsageUpdate()` writes accumulated usage to the session store (sessions.json)
- Tracks: total tokens,model used,provider used,context window utilization
- Called after each assistant response during the auto-reply loop

### No Dedicated Usage Database
- `~/.openclaw/data/` only contains `backlog.db` (task tracking)
- There is **no SQLite/JSON database dedicated to usage or cost tracking**
- All usage queries require scanning raw session JSONL files

## 2. Mission Control Usage Tab

### Stats Endpoint (`/api/mc/stats`)
- Aggregates token usage across all sessions by scanning `agents/main/sessions/*.jsonl`
- Returns: `totalInput`,`totalOutput`,`totalCacheRead`,`totalSessions`
- Uses a 5-second cache to avoid repeated file scans
- **Token counts only** — no cost aggregation at this level

### Usage Chart Endpoint (`/api/mc/usage-chart`)
- Scans session JSONL files and buckets usage by time windows
- Supports 24h (1hr buckets),7d (6hr buckets),30d (daily buckets)
- Returns time-series: `{ ts,input,output,cacheRead }` per bucket
- **Token counts only** — no cost data in the chart

### Sessions Endpoint (`/api/mc/sessions`)
- Lists sessions with per-session token totals (input,output,cacheRead,messageCount)
- Generates summaries lazily via local LLM (Qwen3.5-35B-A3B at localhost:8091)
- **No cost data** — only token counts

### Agent Call Log (`/api/mc/agent-call-log`)
- Reads the most recent session JSONL for a specific agent
- Extracts per-LLM-call entries with: model,provider,inputTokens,outputTokens,cost
- Also lists tool calls with duration and error status
- **Has cost data** via `usage.cost.total` from assistant messages

### CC-Instances Endpoints (`/api/mc/cc-instances`,`/api/mc/cc-instance-log`)
- Served by the `agent-orchestrator` plugin
- Lists all cc_orchestrate/cc_spawn instances with their costUsd,turns,budgetUsd
- Instance log falls back to the JSONL files in `~/.openclaw/logs/cc-instances/`
- **Has cost data** — includes `costUsd` per instance

## 3. Gateway Per-Request Token Tracking

### Flow
1. Provider API returns response with usage metadata (format varies by provider)
2. `normalizeUsage()` normalizes across provider formats (Anthropic `input_tokens`/`output_tokens`,OpenAI `prompt_tokens`/`completion_tokens`,etc.)
3. Usage is attached to the assistant message and written to session JSONL
4. `persistSessionUsageUpdate()` updates session store with accumulated totals
5. Cost is estimated using `resolveModelCostConfig()` → per-model pricing from config
6. `estimateUsageCost()` computes: `(input * rate) + (output * rate) + (cacheRead * rate) + (cacheWrite * rate)`

### Provider Usage Snapshots (quota windows)
- Separate from per-request tracking — these are **provider-reported quota snapshots**
- `loadProviderUsageSummary()` fetches from provider APIs:
  - Anthropic,GitHub Copilot,Gemini CLI,MiniMax,OpenAI Codex,z.ai
- Shows usage windows (e.g.,"80% of daily limit used,resets at X")
- Requires OAuth tokens or API keys per provider
- Displayed in `/status`,`openclaw status --usage`,`openclaw channels list`

### Cost