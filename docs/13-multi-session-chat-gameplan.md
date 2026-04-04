# Multi-Session Chat, Stop Button & Thinking/Working Indicator

## Context

The Mission Control chat tab currently supports only one active session at a time. Switching sessions remounts ChatPanel (via the `key` prop), destroying all in-progress streaming state. There is no way to interrupt an active LLM call, and the "Thinking..." indicator gives no feedback about what the agent is actually doing.

This plan adds:
1. **Multiple simultaneous sessions** — slot-based rendering keeps background sessions alive; SessionsPanel shows streaming indicators
2. **Stop button** — aborts the active fetch/stream mid-call; backend already handles client disconnects gracefully
3. **Thinking/Working indicator** — shows "Working: `<tool_name>`..." when a tool is running, "Thinking..." otherwise

---

## Critical Files

- `web/src/api.ts` — `streamMessage()` AbortController and callbacks
- `web/src/components/ChatPanel.tsx` — streaming state, indicators, stop button
- `web/src/components/Layout.tsx` — session slot management
- `web/src/components/SessionsPanel.tsx` — streaming dot indicators

---

## Step 1 — `api.ts`: Add `onAborted` callback

`AbortError` is currently silently swallowed; neither `onDone` nor `onError` fires, leaving ChatPanel stuck with `thinking=true` forever after a stop.

Change `streamMessage`'s callbacks type and catch:
```typescript
onAborted?: () => void   // add to callbacks interface

// catch:
if (err.name === 'AbortError') callbacks.onAborted?.()
else callbacks.onError?.(err.message)
```

---

## Step 2 — `ChatPanel.tsx`: Stop button, Working indicator, streaming callbacks

### New props
```typescript
visible?: boolean                    // hide/show (default true)
onThinkingChange?: (thinking: boolean, toolName: string | null) => void
```

### New internal state/refs
```typescript
const abortControllerRef = useRef<AbortController | null>(null)
const [activeToolName, setActiveToolName] = useState<string | null>(null)
```

### `settled` flag pattern in `handleSubmit`
Prevents double-firing if `onDone` and `onAborted` race:
```typescript
let settled = false
// wrap onDone, onError, onAborted each with: if (settled) return; settled = true
```

### Wire AbortController
```typescript
const controller = api.streamMessage(text, ..., {
  ...callbacks,
  onToolStart: (callId, name, args) => {
    setActiveToolName(name)
    // ...existing logic...
  },
  onToolComplete: (callId, name, result) => {
    setMessages(prev => {
      const updated = prev.map(/* existing map */)
      const stillPending = updated.find(m => m.role === 'tool' && m.content[0]?.text === '⏳ Running...')
      if (!stillPending) setActiveToolName(null)
      return updated
    })
  },
  onDone: (response, sid) => {
    if (settled) return; settled = true
    abortControllerRef.current = null
    setActiveToolName(null)
    // ...existing logic...
  },
  onError: (detail) => {
    if (settled) return; settled = true
    abortControllerRef.current = null
    setActiveToolName(null)
    // ...existing logic...
  },
  onAborted: () => {
    if (settled) return; settled = true
    abortControllerRef.current = null
    setActiveToolName(null)
    setThinking(false)
    setSending(false)
    setMessages(prev => [...prev, {
      id: `msg_${Date.now()}_interrupted`,
      role: 'assistant',
      content: [{ type: 'text', text: '[Interrupted]' }],
      timestamp: new Date().toISOString(),
    }])
    inputRef.current?.focus()
  },
})
abortControllerRef.current = controller
```

### Polling guard for hidden idle panels
```typescript
if (!sessionKey || sending || thinking) return
if (visible === false) return   // don't poll hidden idle panels
```

### Fire `onThinkingChange`
```typescript
useEffect(() => {
  onThinkingChange?.(thinking, thinking ? activeToolName : null)
}, [thinking, activeToolName])
```

### Stop button (replace send button area)
Add `Square` to lucide-react import.
```tsx
{(sending || thinking) ? (
  <button
    type="button"
    onClick={() => abortControllerRef.current?.abort()}
    className="bg-red-600/20 hover:bg-red-600/30 border border-red-500/40 text-red-400 rounded-lg px-3 transition-colors"
    title="Stop"
  >
    <Square className="w-4 h-4" />
  </button>
) : (
  <button type="submit" disabled={!input.trim()} className="bg-brand-600 ...">
    <Send className="w-4 h-4" />
  </button>
)}
```

### Working indicator
```tsx
{thinking && (
  <div className="flex gap-3">
    <div className="w-7 h-7 rounded-full flex-shrink-0 mt-0.5 overflow-hidden">
      <img src="/lloyd.jpg" alt="Lloyd" className="w-full h-full object-cover" />
    </div>
    <div className="bg-surface-2 border border-surface-3/50 px-3.5 py-2.5 rounded-xl">
      <div className="flex items-center gap-2 text-sm text-slate-400">
        <Loader2 className="w-4 h-4 animate-spin text-brand-400" />
        {activeToolName
          ? <span>Working: <span className="font-mono text-brand-300">{activeToolName}</span>...</span>
          : <span>Thinking...</span>
        }
      </div>
    </div>
  </div>
)}
```

---

## Step 3 — `Layout.tsx`: Slot-based multi-session

### Core idea

Replace the `key`-remount pattern with an array of slots. Each slot maps to a stable `slotId` (React key) and a `sessionKey`. The ChatPanel for each slot is always mounted; non-visible slots are hidden with Tailwind's `hidden` class (`display: none`), which preserves React state including in-progress streams.

### Replace old session state

Remove: `chatSessionKey`, `activeSessionKey`, `isNewSession`, `sessionRefreshTrigger`

Add:
```typescript
interface Slot {
  slotId: string
  sessionKey: string | null   // null = blank new-session panel
  model: string
}

const [slots, setSlots] = useState<Slot[]>([])
const [visibleSlotId, setVisibleSlotId] = useState<string | null>(null)
const [activeSessions, setActiveSessions] = useState<Set<string>>(new Set())

const visibleSlot = slots.find(s => s.slotId === visibleSlotId) ?? null
const currentModel = visibleSlot?.model ?? ''
const nextSlotId = () => `slot_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`
```

### Auto-load on mount
```typescript
useEffect(() => {
  if (page !== 'chat' || slots.length > 0) return
  api.listSessions().then(result => {
    const id = nextSlotId()
    setSlots([{ slotId: id, sessionKey: result.sessions?.[0]?.session_key ?? null, model: '' }])
    setVisibleSlotId(id)
  }).catch(() => {
    const id = nextSlotId()
    setSlots([{ slotId: id, sessionKey: null, model: '' }])
    setVisibleSlotId(id)
  })
}, [page, slots.length])
```

### `handleNewSession`
```typescript
const handleNewSession = () => {
  // Reuse existing blank slot if one exists
  const existingBlank = slots.find(s => s.sessionKey === null)
  if (existingBlank) { setVisibleSlotId(existingBlank.slotId); return }
  const id = nextSlotId()
  setSlots(prev => [...prev, { slotId: id, sessionKey: null, model: currentModel }])
  setVisibleSlotId(id)
}
```

### `handleOpenSession` — uses functional `setSlots` to avoid stale closures
```typescript
const handleOpenSession = useCallback((sessionKey: string) => {
  setSlots(prev => {
    const existing = prev.find(s => s.sessionKey === sessionKey)
    if (existing) {
      setVisibleSlotId(existing.slotId)
      return prev
    }
    const id = nextSlotId()
    setVisibleSlotId(id)
    return [...prev, { slotId: id, sessionKey, model: '' }]
  })
  setPage('chat')
}, [])
```

### Per-slot callbacks
```typescript
const handleActiveSessionChange = (slotId: string, key: string | null) => {
  setSlots(prev => prev.map(s => s.slotId === slotId ? { ...s, sessionKey: key } : s))
}

const handleThinkingChange = (
  slotId: string,
  sessionKey: string | null,
  thinking: boolean,
  _toolName: string | null,
) => {
  if (!sessionKey) return
  setActiveSessions(prev => {
    const next = new Set(prev)
    thinking ? next.add(sessionKey) : next.delete(sessionKey)
    return next
  })
}

const handleModelSwitch = (slotId: string, model: string) => {
  setSlots(prev => prev.map(s => s.slotId === slotId ? { ...s, model } : s))
}
```

### Slot render (replace single `<ChatPanel>`)
```tsx
<div className="flex-1 flex flex-col min-h-0 overflow-hidden relative">
  {slots.map(slot => (
    <div
      key={slot.slotId}
      className={`absolute inset-0 flex flex-col ${slot.slotId === visibleSlotId ? '' : 'hidden'}`}
    >
      <ChatPanel
        requestedSessionKey={slot.sessionKey}
        onSessionLoaded={() => {}}
        onActiveSessionChange={(key) => handleActiveSessionChange(slot.slotId, key)}
        onThinkingChange={(thinking, toolName) =>
          handleThinkingChange(slot.slotId, slot.sessionKey, thinking, toolName)
        }
        onModelSwitch={(model) => handleModelSwitch(slot.slotId, model)}
        currentSessionKey={slot.sessionKey}
        showAgentDetails={showAgentDetails}
        pendingModel={slot.model || models[0]?.name}
        visible={slot.slotId === visibleSlotId}
      />
    </div>
  ))}
</div>
```

### SessionsPanel — pass `activeSessions` and a computed refresh trigger
```typescript
const sessionsPanelRefreshTrigger = slots.map(s => s.sessionKey ?? 'null').join(',')

<SessionsPanel
  onSwitchSession={(key) => handleOpenSession(key)}
  currentSessionKey={visibleSlot?.sessionKey ?? null}
  activeSessions={activeSessions}
  refreshTrigger={sessionsPanelRefreshTrigger}
/>
```

### URL `?session=` param
Replace `setChatSessionKey(sessionKey)` with `handleOpenSession(sessionKey)`.

### Model dropdown
Reads `currentModel` (derived from `visibleSlot?.model`). On select:
```typescript
await api.switchModel(model.name, visibleSlot.sessionKey)
handleModelSwitch(visibleSlotId!, model.name)
```

---

## Step 4 — `SessionsPanel.tsx`: Streaming dot indicator

### New prop
```typescript
activeSessions?: Set<string>
```

### Indicator in session list
```tsx
{sessions.map((session) => {
  const key = session.session_key || session.id
  const isActive = activeSessions?.has(key) ?? false
  const isCurrent = currentSessionKey === key
  return (
    <button key={key} data-selected={isCurrent ? "true" : undefined} ...>
      <div className="flex items-center justify-between mb-1">
        <div className="text-[10px] text-slate-400 truncate flex-1">
          {session.preview || 'No preview'}
        </div>
        {isActive && (
          <span className="w-1.5 h-1.5 rounded-full bg-brand-400 animate-pulse flex-shrink-0 ml-1.5" />
        )}
      </div>
      <div className="flex items-center gap-1 text-[10px] text-slate-500">
        <Clock className="w-2.5 h-2.5" />
        <span>{session.last_active}</span>
      </div>
    </button>
  )
})}
```

---

## Edge Cases

### Stale closure when blank slot gets real sessionKey
When a blank slot's first message resolves and `handleActiveSessionChange` updates `slots`, the `handleOpenSession` callback may still hold a stale `slots` reference. Using the functional form of `setSlots(prev => ...)` in `handleOpenSession` reads the latest state, preventing duplicate slot creation.

### Race: `onDone` and `onAborted` both fire
The `settled` flag in `handleSubmit` ensures only one cleanup path runs.

### Duplicate blank slots
`handleNewSession` checks for an existing blank slot (`sessionKey === null`) and reuses it rather than creating another.

### Multiple slots polling
Hidden idle slots skip polling via the `if (visible === false) return` guard. Hidden streaming slots (thinking=true) continue polling as a fallback — the guard only fires when `!thinking`, which is checked before `visible`.

---

## Implementation Order

1. `api.ts` — `onAborted` (standalone, no dependencies)
2. `ChatPanel.tsx` — all changes (stop button + indicator testable with existing Layout)
3. `Layout.tsx` — slot refactor
4. `SessionsPanel.tsx` — streaming dot

## Verification

- Send a long task → see "Working: Bash..." → click stop → "[Interrupted]" message, input re-enables
- Open session A, send long task, switch to session B, send another message — both streams run in parallel; switch back to A to see continued progress
- Pulsing brand dot appears in SessionsPanel sidebar next to background streaming sessions
- "New Session" opens a blank slot without affecting any running session
