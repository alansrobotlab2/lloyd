---
tags:
  - lloyd
  - architecture
type: reference
segment: projects
---

# OpenClaw Usage Tracking Architecture

Research into how token usage, costs, and provider quotas are tracked across the OpenClaw system.

## 1. Where Usage Data Is Stored

### Session Transcripts (primary source)
- **Location**: `~/.openclaw/agents/{agentId}/sessions/*.jsonl`
- **Format**: JSONL — one JSON object per line, per message
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
        "cost": { "input": 0.003, "output": 0.008, "total": 0.011 }
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
- JSONL captures: start, tool_use, text, session_init, result, complete events
- Summary JSON captures: id, type, status, task, costUsd, turns, budgetUsd, elapsedMs, resultPreview
- **Cost tracking**: Two mechanisms:
  1. **Per-turn estimation**: `(input_tokens / 1M) * 3 + (output_tokens / 1M) * 15` (hardcoded Sonnet pricing)
  2. **SDK-reported total**: When the Agent SDK's `result` message includes `total_cost_usd`, it overwrites the running estimate
- Logs are persistent on disk but **not aggregated into any database or rolled-up summary**

### Session Usage Update (live state)
- `persistSessionUsageUpdate()` writes accumulated usage to the session store (sessions.json)
- Tracks: total tokens, model used, provider used, context window utilization
- Called after each assistant response during the auto-reply loop

### No Dedicated Usage Database
- `~/.openclaw/data/` only contains `backlog.db` (task tracking)
- There is **no SQLite/JSON database dedicated to usage or cost tracking**
- All usage queries require scanning raw session JSONL files

## 2. Mission Control Usage Tab

### Stats Endpoint (`/api/mc/stats`)
- Aggregates token usage across all sessions by scanning `agents/main/sessions/*.jsonl`
- Returns: `totalInput`, `totalOutput`, `totalCacheRead`, `totalSessions`
- Uses a 5-second cache to avoid repeated file scans
- **Token counts only** — no cost aggregation at this level

### Usage Chart Endpoint (`/api/mc/usage-chart`)
- Scans session JSONL files and buckets usage by time windows
- Supports 24h (1hr buckets), 7d (6hr buckets), 30d (daily buckets)
- Returns time-series: `{ ts, input, output, cacheRead }` per bucket
- **Token counts only** — no cost data in the chart

### Sessions Endpoint (`/api/mc/sessions`)
- Lists sessions with per-session token totals (input, output, cacheRead, messageCount)
- Generates summaries lazily via local LLM (Qwen3.5-35B-A3B at localhost:8091)
- **No cost data** — only token counts

### Agent Call Log (`/api/mc/agent-call-log`)
- Reads the most recent session JSONL for a specific agent
- Extracts per-LLM-call entries with: model, provider, inputTokens, outputTokens, cost
- Also lists tool calls with duration and error status
- **Has cost data** via `usage.cost.total` from assistant messages

### CC-Instances Endpoints (`/api/mc/cc-instances`, `/api/mc/cc-instance-log`)
- Served by the `agent-orchestrator` plugin
- Lists all cc_orchestrate/cc_spawn instances with their costUsd, turns, budgetUsd
- Instance log falls back to the JSONL files in `~/.openclaw/logs/cc-instances/`
- **Has cost data** — includes `costUsd` per instance

## 3. Gateway Per-Request Token Tracking

### Flow
1. Provider API returns response with usage metadata (format varies by provider)
2. `normalizeUsage()` normalizes across provider formats (Anthropic `input_tokens`/`output_tokens`, OpenAI `prompt_tokens`/`completion_tokens`, etc.)
3. Usage is attached to the assistant message and written to session JSONL
4. `persistSessionUsageUpdate()` updates session store with accumulated totals
5. Cost is estimated using `resolveModelCostConfig()` → per-model pricing from config
6. `estimateUsageCost()` computes: `(input * rate) + (output * rate) + (cacheRead * rate) + (cacheWrite * rate)`

### Provider Usage Snapshots (quota windows)
- Separate from per-request tracking — these are **provider-reported quota snapshots**
- `loadProviderUsageSummary()` fetches from provider APIs:
  - Anthropic, GitHub Copilot, Gemini CLI, MiniMax, OpenAI Codex, z.ai
- Shows usage windows (e.g., "80% of daily limit used, resets at X")
- Requires OAuth tokens or API keys per provider
- Displayed in `/status`, `openclaw status --usage`, `openclaw channels list`

### Cost Configuration
- `resolveModelCostConfig()` returns per-model pricing: `{ input, output, cacheRead, cacheWrite }` (rates per token)
- Configured in `models.providers.*.models.*.cost` in openclaw.json
- Used by `/usage full` footer and `/status` display
- Only shown for API-key auth (OAuth hides dollar costs)

## 4. What Happens When cc_orchestrate/cc_spawn Complete

### During Execution
- `agent-orchestrator/query-consumer.ts` consumes the Agent SDK async generator
- Each `response` event increments `instance.costUsd` using hardcoded Sonnet pricing
- Each event is logged to the instance JSONL file

### On Completion
- SDK `result` message may include `total_cost_usd` → overwrites the estimate
- `summary.json` is written with final costUsd, turns, elapsedMs, resultPreview
- JSONL gets a final `result` event and `complete` event with cost data
- A completion notification is sent to the parent (Lloyd) session
- **Cost IS persisted** to the summary.json and JSONL files
- **Cost is NOT rolled into** the gateway's session usage totals — it's a separate tracking silo

### What the Notification Contains
```
**[Agent Complete]** `{id}` (pipeline: {name})
Status: ✓ complete | Cost: $0.18 | Turns: 14
Task: {truncated task description}
Result: {truncated result}
```

## 5. Gap Analysis: What's Tracked vs What's Missing

### What IS Tracked
| Data | Where | Format |
|------|-------|--------|
| Per-request tokens (gateway sessions) | Session JSONL files | input, output, cacheRead per message |
| Per-request cost (gateway sessions) | Session JSONL files | Estimated via model pricing config |
| Provider quota windows | Live API calls | Percentage used, reset times |
| CC-instance cost (subagents) | `logs/cc-instances/*.summary.json` | total costUsd per instance |
| CC-instance event log | `logs/cc-instances/*.jsonl` | tool calls, text, timing |
| Session-level totals | `persistSessionUsageUpdate()` + sessions.json | Accumulated per session |

### What's MISSING

1. **No unified cost database**: Gateway session costs and cc-instance costs live in completely separate silos. There's no single query that answers "how much did I spend today across everything?"

2. **No cc-instance cost aggregation in Mission Control**: The stats/usage-chart endpoints only scan gateway session JSONL files. CC-instance costs (which can be substantial — $0.18-2.00+ per orchestrate call) are invisible in the main usage dashboard.

3. **Hardcoded pricing in agent-orchestrator**: CC-instance cost estimation uses `$3/M input + $15/M output` (Sonnet 4.6 pricing) regardless of actual model used. When Opus agents run, costs are underestimated ~5x until the SDK reports final cost.

4. **No per-subagent breakdown within cc_orchestrate**: When a coordinator spawns coder/reviewer/tester, the total cost is tracked at the instance level, but there's no breakdown showing which agent consumed what portion.

5. **No historical cost trending for cc-instances**: Summary JSONs accumulate on disk but nothing aggregates them into daily/weekly/monthly totals.

6. **Mission Control usage chart ignores cc-instances**: The time-series chart only shows gateway token usage. A heavy cc_orchestrate run ($2+) wouldn't appear in the usage chart at all.

7. **No cost alerts or budget tracking across sessions**: Each cc-instance has its own budget cap, but there's no system-wide daily/monthly budget or alerting.

### Recommended Improvements (ordered by impact)

1. **Unified usage summary**: Add a function that scans both session JSONLs and cc-instance summaries to produce a combined daily cost rollup
2. **Include cc-instance costs in Mission Control charts**: Extend the usage-chart endpoint to also scan `logs/cc-instances/*.summary.json`
3. **Use actual model pricing for cc-instances**: Look up the model from the Agent SDK response rather than hardcoding Sonnet rates
4. **Daily cost rollup file**: Write a daily aggregate to `~/.openclaw/data/usage/YYYY-MM-DD.json` for fast historical queries
5. **System-wide budget alerts**: Add configurable daily/monthly spend limits with notifications
