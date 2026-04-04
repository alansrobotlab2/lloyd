# Token Stats Footer — Per-Response Metadata

## Context

Each `ResultMessage` from the Claude Agent SDK carries full usage data: input/output tokens, cache tokens (create+read), `total_cost_usd`, `duration_ms`, `duration_api_ms`, and `num_turns`. This is already persisted to `usage.db` but **never sent to the frontend**. The `done` SSE event only carries `{response, session_id}`.

For **local models** (Qwen via `ANTHROPIC_BASE_URL`), `total_cost_usd` is `null` and cache fields are typically 0. Duration and token counts are still present. The plan must handle both cases gracefully.

Goal: display a compact stats line beneath each assistant response bubble, rendered after `onDone` fires.

---

## What Data Is Available

From `ResultMessage` (server.py line ~483):

| Field | Local | Anthropic |
|---|---|---|
| `usage.input_tokens` | yes | yes |
| `usage.output_tokens` | yes | yes |
| `usage.cache_creation_input_tokens` | 0 | yes |
| `usage.cache_read_input_tokens` | 0 | yes |
| `total_cost_usd` | null | yes |
| `duration_ms` | yes | yes |
| `duration_api_ms` | yes | yes |
| `num_turns` | yes | yes |

Additionally we know `model` from request context.

---

## Critical Files

- `server.py` — `stream_message()` generator (~line 483): enrich the `done` SSE payload
- `web/src/api.ts` — `streamMessage()` callbacks: pass stats through `onDone`
- `web/src/components/ChatPanel.tsx` — render the stats footer on assistant messages

---

## Step 1 — `server.py`: Enrich the `done` event

In the `ResultMessage` branch (~line 540), replace:
```python
yield f"event: done\ndata: {json.dumps({'response': result_text, 'session_id': session_id})}\n\n"
```
with:
```python
usage = getattr(message, "usage", None) or {}
done_payload = {
    "response": result_text,
    "session_id": session_id,
    "stats": {
        "input_tokens":  usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "cache_create":  usage.get("cache_creation_input_tokens", 0),
        "cache_read":    usage.get("cache_read_input_tokens", 0),
        "cost_usd":      getattr(message, "total_cost_usd", None),
        "duration_ms":   getattr(message, "duration_ms", None),
        "num_turns":     getattr(message, "num_turns", None),
        "model":         model,
    }
}
yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"
```

Same change applies to the non-streaming `/api/message` endpoint (`ResultMessage` branch ~line 654) — add `stats` to its JSON response body.

---

## Step 2 — `api.ts`: Thread stats through `onDone`

### New type
```typescript
export interface TurnStats {
  input_tokens: number
  output_tokens: number
  cache_create: number
  cache_read: number
  cost_usd: number | null
  duration_ms: number | null
  num_turns: number | null
  model: string
}
```

### Update `streamMessage` callback signature
```typescript
onDone?: (response: string, sessionId: string, stats?: TurnStats) => void
```

### Parse stats in the `done` case
```typescript
case 'done':
  callbacks.onDone?.(payload.response, payload.session_id, payload.stats)
  break
```

---

## Step 3 — `ChatPanel.tsx`: Attach stats to the assistant message

### Extend `MessageEntry` (or local state type) with optional `stats`
Add `stats?: TurnStats` to the `MessageEntry` interface in `api.ts`.

### Store stats on the message in `onDone`
```typescript
onDone: (response, _sid, stats) => {
  // Update the streaming assistant message to attach stats
  setMessages(prev => prev.map(m =>
    m.id === assistantMsgId
      ? { ...m, stats }
      : m
  ))
  // ... existing setThinking/setSending/focus ...
}
```
If no streaming deltas arrived and a new message is created in `onDone`, include `stats` on that message too.

### Render the stats footer below each assistant message bubble

Below the `dangerouslySetInnerHTML` markdown block (after line ~566), add:
```tsx
{msg.stats && (
  <div className="mt-2 flex flex-wrap gap-x-3 gap-y-0.5 text-[10px] text-slate-500 font-mono">
    <span title="Input tokens">↑{msg.stats.input_tokens.toLocaleString()}</span>
    <span title="Output tokens">↓{msg.stats.output_tokens.toLocaleString()}</span>
    {msg.stats.cache_read > 0 && (
      <span title="Cache read tokens" className="text-emerald-600">
        ⚡{msg.stats.cache_read.toLocaleString()}
      </span>
    )}
    {msg.stats.cache_create > 0 && (
      <span title="Cache write tokens" className="text-amber-600">
        ✎{msg.stats.cache_create.toLocaleString()}
      </span>
    )}
    {msg.stats.cost_usd != null && msg.stats.cost_usd > 0 && (
      <span title="Cost">${msg.stats.cost_usd.toFixed(4)}</span>
    )}
    {msg.stats.duration_ms != null && (
      <span title="Turn duration">{(msg.stats.duration_ms / 1000).toFixed(1)}s</span>
    )}
    {msg.stats.num_turns != null && msg.stats.num_turns > 1 && (
      <span title="Agent turns">{msg.stats.num_turns}t</span>
    )}
  </div>
)}
```

**Visual intent**: single muted row, barely visible at rest. Icons/arrows give quick scanning cues. Cache tokens highlighted in color since they directly affect cost. Cost line hidden entirely for local models (`cost_usd === null`). Duration always shown as it's useful for both local and Anthropic.

---

## Step 4 — Persist stats in session messages (optional, recommended)

`_append_messages` writes the final assistant message dict to `sessions/<id>.json`. Currently no stats field is included. To make stats survive a page reload:

When building the final assistant message dict in the `ResultMessage` block (~line 533):
```python
stats_dict = {
    "input_tokens":  usage.get("input_tokens", 0),
    "output_tokens": usage.get("output_tokens", 0),
    "cache_create":  usage.get("cache_creation_input_tokens", 0),
    "cache_read":    usage.get("cache_read_input_tokens", 0),
    "cost_usd":      getattr(message, "total_cost_usd", None),
    "duration_ms":   getattr(message, "duration_ms", None),
    "num_turns":     getattr(message, "num_turns", None),
    "model":         model,
}
tail.append({
    "id": uuid.uuid4().hex[:8],
    "role": "assistant",
    "content": [{"type": "text", "text": result_text}],
    "timestamp": end_ts,
    "stats": stats_dict,   # ← add this
})
```

Then in `loadMessages` (backend `/api/messages/<key>`), the stats are returned as-is and ChatPanel hydrates them on load. The `MessageEntry` type already handles the optional `stats` field once added.

---

## Edge Cases

### Local model — no cost
`cost_usd` will be `null`. The cost span is conditionally rendered only when `cost_usd != null && cost_usd > 0`, so nothing breaks.

### SDK returns 0 tokens (error path or mid-stream disconnect)
The fallback path in server.py (~line 542) that persists content without a `ResultMessage` will not emit a `done` event with stats — it yields its own `done` with just `{response, session_id}`. The `stats` field will be `undefined` in `onDone`, and the stats row simply won't render. This is correct behavior.

### `num_turns = 1` (single-turn, no tools)
The turns counter is hidden when `num_turns <= 1` to keep the footer clean for simple responses.

### Loading historical messages
Stats are persisted to `sessions/<id>.json` in Step 4. On reload, ChatPanel maps `msg.stats` directly. No additional API changes needed.

---

## Implementation Order

1. `server.py` — enrich `done` payload in both stream and non-stream handlers
2. `api.ts` — add `TurnStats` type, update `onDone` signature, parse stats from payload; add `stats` field to `MessageEntry`
3. `ChatPanel.tsx` — attach stats in `onDone`, render stats footer
4. `server.py` (persistence) — write `stats` dict into the final assistant message in `_append_messages`

Steps 1–3 are the visible deliverable; step 4 is polish for reload persistence.

## Verification

- Send a message using a local model → stats row shows tokens + duration, no cost
- Send a message using Anthropic → stats row shows tokens + cost + duration
- Multi-turn tool call → `num_turns` shows `> 1`, cache tokens appear if prompt caching active
- Reload page → stats persist on historical messages
- Mid-stream disconnect (stop button) → no stats row on the `[Interrupted]` message (correct)
