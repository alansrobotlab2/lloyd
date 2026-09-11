import { useState, useRef, useEffect, useCallback, useMemo, memo } from 'react'
import {
  Send, User, Loader2, Brain, MessageCircle, ChevronRight,
  Wrench, Square, Sparkles,
} from 'lucide-react'
import { Streamdown } from 'streamdown'
import { api, type MessageEntry as ApiMessage, type ModelInfo, type TurnStats, type QueueState, type InnerVoiceObservation, type ChangedFile, type RevertResult } from '../api'
import TodoList from './TodoList'
import PlanHeader from './PlanHeader'
import GoalHeader from './GoalHeader'
import ObservationBubble from './ObservationBubble'
import { actionStyle, parseObservationTime } from './innerVoiceStyles'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import {
  Collapsible, CollapsibleContent, CollapsibleTrigger,
} from '@/components/ui/collapsible'
import {
  Command, CommandEmpty, CommandGroup, CommandItem, CommandList,
} from '@/components/ui/command'

// ── helpers ────────────────────────────────────────────────────────────

const timeStr = (iso: string) =>
  new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })

// How long the model spent reasoning, for the collapsed thinking header.
// Whole seconds throughout. The tenths were noise on a header that ticks
// while it streams — a digit that changes five times a second reads as
// motion, not as information. Rounding is applied before the minute split
// so 59.6s renders as `1m 00s` rather than `60s`.
const thinkDuration = (ms: number): string => {
  if (!(ms > 0)) return ''
  const total = Math.round(ms / 1000)
  if (total < 60) return `${total}s`
  return `${Math.floor(total / 60)}m ${String(total % 60).padStart(2, '0')}s`
}

type ToolCallRef = { name: string; args: string; summary: string }

// Live thinking rows.
//
// A phase opens a provisional row on its *first delta*, not on
// `thinking_done`: the harness only emits that once the iteration's stream
// has ended, which is after the iteration's own text, so a row created there
// would sort below the answer it preceded. Opening early puts it in the
// right place and lets the duration tick; `thinking_done` then closes it in
// place with the harness's own measurement.
//
// This lives in one factory because the two stream handlers below (the goal
// path and the main path) are near-duplicates, and drifting apart is the
// standing hazard in this file.
function makeThinkingTracker(apply: (fn: (prev: ApiMessage[]) => ApiMessage[]) => void) {
  let liveId: string | null = null
  let opened = 0
  return {
    /** Every reasoning delta. Opens the row on the first of each phase. */
    onDelta() {
      if (liveId) return
      const id = `msg_think_live_${Date.now()}_${opened++}`
      liveId = id
      apply(prev => [...prev, {
        id,
        role: 'thinking' as const,
        content: [],
        timestamp: new Date().toISOString(),
        thinking: { chars: 0, iteration: 0, turn_id: '', live: true },
      }])
    },
    /** Close the phase.
     *
     *  Returns whether the backend is persisting a `role="thinking"` row for
     *  it, which it signals by sending `seq`. When it is not — the
     *  `harness.thinking_trace` kill switch is off — the provisional row is
     *  withdrawn and the caller falls back to hanging the reasoning off the
     *  assistant bubble, so what you see live and what you see on reload
     *  agree under either setting.
     */
    onDone(fullText: string, durationMs?: number, seq?: number, iteration?: number): boolean {
      const id = liveId
      liveId = null
      const persisted = seq !== undefined
      if (!id) return persisted
      if (!persisted) {
        apply(prev => prev.filter(m => m.id !== id))
        return false
      }
      apply(prev => prev.map(m => m.id === id ? {
        ...m,
        reasoning: fullText,
        reasoning_ms: durationMs ?? 0,
        thinking: { chars: fullText.length, iteration: iteration ?? 0, turn_id: '' },
      } : m))
      return true
    },
    /** Drop any row still open — the turn ended or was cancelled mid-phase. */
    discard() {
      const id = liveId
      liveId = null
      if (id) apply(prev => prev.filter(m => m.id !== id))
    },
  }
}

// Preserve per-message object identity across polling refreshes so memoized
// rows don't re-render when nothing actually changed.
const mergeMessages = (prev: ApiMessage[], next: ApiMessage[]): ApiMessage[] => {
  const prevById = new Map(prev.map(m => [m.id, m]))
  let different = prev.length !== next.length
  const merged = next.map((n, i) => {
    const p = prevById.get(n.id) || prev[i]
    if (p && p.id === n.id) {
      const pt = p.content?.map(c => c.text).join('') || ''
      const nt = n.content?.map(c => c.text).join('') || ''
      const sameStats = (p.stats == null && n.stats == null) ||
        (p.stats && n.stats && JSON.stringify(p.stats) === JSON.stringify(n.stats))
      // A thinking row's identity is entirely in these fields — its
      // content is empty — so without the `thinking` comparison a live
      // row would never be repainted by the persisted one.
      const sameThinking = (p.thinking == null && n.thinking == null) ||
        (p.thinking && n.thinking && JSON.stringify(p.thinking) === JSON.stringify(n.thinking))
      if (pt === nt && sameStats && sameThinking && p.reasoning === n.reasoning
          && p.reasoning_ms === n.reasoning_ms) {
        return p
      }
    }
    different = true
    return n
  })
  return different ? merged : prev
}

/** A tool call the transcript has asked for but not answered, or null.
 *
 *  Read off the transcript rather than off an event stream, because the
 *  transcript is the one record that exists for a turn this browser did not
 *  start: a call is open when an assistant row carries it and no `role:
 *  'tool'` row answers it (or the answer is still the `⏳ Running...`
 *  placeholder the stream path writes — see onToolStart). The newest open
 *  call wins; a model that fans out three tools is working on the last.
 *
 *  Do NOT read a null here as "not running a tool". The harness persists a
 *  call's row only when its result arrives, so while a tool executes the
 *  polling reader sees nothing open — measured on a live session at 4s, 25s
 *  and 45s into one 50s call, zero unanswered `tool_calls` each time. This
 *  finds calls the log left unanswered; it does not watch the harness work. */
const pendingToolName = (msgs: ApiMessage[]): string | null => {
  const settled = new Set<string>()
  for (const m of msgs) {
    if (m.role !== 'tool' || !m.tool_call_id) continue
    const text = m.content?.[0]?.text ?? ''
    if (!text.startsWith('\u23F3 Running')) settled.add(m.tool_call_id)
  }
  for (let i = msgs.length - 1; i >= 0; i--) {
    const calls = msgs[i].tool_calls
    if (!calls?.length) continue
    for (let j = calls.length - 1; j >= 0; j--) {
      const c = calls[j]
      if (!settled.has(c.id) && !settled.has(c.call_id)) return c.function?.name ?? null
    }
    // Everything on the newest tool-carrying row is answered: the turn is
    // past that call, so an older row cannot hold the live one.
    return null
  }
  return null
}

// ── memoized message row ───────────────────────────────────────────────

interface MessageRowProps {
  msg: ApiMessage
  showAgentDetails: boolean
  isMobile: boolean
  toolCallIndex: Map<string, ToolCallRef>
  forceLeftAlign?: boolean
  /** Compact mode: drop avatars, full-width bubbles. Used by the right
   *  chat sidebar to reclaim horizontal space in narrow layouts. */
  compact?: boolean
  /** The session this row belongs to. Only the changed-files footer needs
   *  it — reverting is addressed by (session, turn). */
  sessionId?: string
}

/** "changed 2 files: a.py, b.tsx · revert", with the undo behind a confirm.
 *
 *  `~/lloyd` is production: a saved file is a deploy, and until the change
 *  ledger existed a turn that edited three files said nothing about which
 *  three. The revert is per-file on the server, and a file something else
 *  wrote since is REFUSED rather than clobbered — so the outcome has to be
 *  rendered per file, not as a single success. */
function ChangedFilesFooter({ sessionId, turnId, files }: {
  sessionId: string
  turnId: string
  files: ChangedFile[]
}) {
  const [results, setResults] = useState<RevertResult[] | null>(null)
  const [busy, setBusy] = useState(false)

  const outcomes = useMemo(() => {
    const map = new Map<string, RevertResult>()
    for (const r of results ?? []) map.set(r.path, r)
    return map
  }, [results])

  const allReverted = files.length > 0 && files.every(
    f => f.reverted_at != null || ['restored', 'deleted'].includes(
      outcomes.get(f.path)?.status ?? ''))

  const doRevert = useCallback(async () => {
    if (!window.confirm(
      `Put back ${files.length} file${files.length === 1 ? '' : 's'} this turn wrote?\n\n`
      + files.map(f => f.path).join('\n')
      + `\n\nA file that changed since will be refused, not overwritten.`
    )) return
    setBusy(true)
    try {
      const res = await api.revertTurn(sessionId, turnId)
      setResults(res.results)
    } catch (e) {
      setResults(files.map(f => ({
        path: f.path, op: f.op, status: 'refused', reason: String(e),
      })))
    } finally {
      setBusy(false)
    }
  }, [sessionId, turnId, files])

  return (
    <span className="flex flex-wrap items-center gap-x-1.5">
      <span className={allReverted ? 'text-muted-foreground/60' : 'text-amber-600'}>
        changed {files.length} file{files.length === 1 ? '' : 's'}:
      </span>
      {files.map(f => {
        const outcome = outcomes.get(f.path)
        const done = f.reverted_at != null
          || ['restored', 'deleted'].includes(outcome?.status ?? '')
        return (
          <span key={f.path} title={f.path} className={done ? 'line-through opacity-60' : ''}>
            {f.path.split('/').pop()}
            {outcome?.status === 'refused' && (
              <span className="text-destructive ml-1">
                (refused: {outcome.reason ?? 'changed since'})
              </span>
            )}
          </span>
        )
      })}
      {allReverted ? (
        <span className="text-muted-foreground/60">(reverted)</span>
      ) : (
        <button
          type="button"
          onClick={doRevert}
          disabled={busy}
          className="underline underline-offset-2 hover:text-foreground disabled:opacity-50"
        >
          {busy ? 'reverting…' : 'revert'}
        </button>
      )}
    </span>
  )
}

// One reasoning phase, rendered where it happened rather than collapsed
// into the answer at the end. The harness emits one `thinking_done` per
// agent-loop iteration, so a long tool-using turn shows its thinking
// interleaved with the tool bubbles.
//
// A thinking row carries no content blocks — its text is in `reasoning` —
// so this cannot be folded into MessageRow's normal path, which drops any
// message with no renderable text.
function ThinkingRow({ msg, isMobile, compact, forceLeftAlign }: {
  msg: ApiMessage
  isMobile: boolean
  compact: boolean
  forceLeftAlign: boolean
}) {
  const live = msg.thinking?.live === true
  // While the phase is streaming the harness has not reported a duration
  // yet, so tick locally off the row's own timestamp; `thinking_done`
  // replaces this row with the measured one. Only live rows pay for the
  // interval.
  const [tick, setTick] = useState(0)
  useEffect(() => {
    if (!live) return
    const id = setInterval(() => setTick(t => t + 1), 500)
    return () => clearInterval(id)
  }, [live])

  const ms = live
    ? Math.max(0, Date.now() - Date.parse(msg.timestamp))
    : (msg.reasoning_ms ?? 0)
  void tick   // the interval exists to re-read the clock above

  const chars = msg.thinking?.chars ?? msg.reasoning?.length ?? 0
  const shown = thinkDuration(ms)
  const label = live
    ? (shown ? `Thinking for ${shown}` : 'Thinking')
    : (shown ? `Thought for ${shown}` : 'Thought')

  const header = (
    <>
      <Brain className={cn('w-3 h-3 shrink-0', live && 'animate-pulse')} />
      <span className="font-semibold shrink-0">{label}</span>
      {!live && chars > 0 && (
        <span className="text-muted-foreground/80 font-normal truncate min-w-0">
          · {chars.toLocaleString()} chars
        </span>
      )}
    </>
  )

  return (
    <div className="flex gap-3">
      {!compact && !forceLeftAlign && !isMobile && (
        <div className="w-7 h-7 rounded-full flex-shrink-0 mt-0.5 overflow-hidden hidden sm:flex">
          <div className="w-full h-full bg-purple-900/40 flex items-center justify-center">
            <Brain className="w-3.5 h-3.5 text-purple-300" />
          </div>
        </div>
      )}
      <div className="flex-1 min-w-0">
        <div className="rounded-xl border px-2.5 py-1.5 bg-purple-950/25 border-purple-500/20 text-foreground">
          {/* A live row has nothing to expand yet, so it renders as a plain
              header — a disclosure triangle that opens on an empty body
              reads as a bug. */}
          {live || !msg.reasoning ? (
            <div className="flex items-center gap-1.5 text-xs text-purple-400 w-full min-w-0">
              {header}
            </div>
          ) : (
            <Collapsible>
              <CollapsibleTrigger className="group cursor-pointer flex items-center gap-1.5 text-xs text-purple-400 hover:text-purple-300 transition-colors w-full min-w-0 text-left">
                <ChevronRight className="w-3 h-3 shrink-0 transition-transform group-data-[state=open]:rotate-90" />
                {header}
              </CollapsibleTrigger>
              <CollapsibleContent className="mt-2 p-3 bg-purple-900/10 border border-purple-500/10 rounded text-xs text-foreground/90 whitespace-pre-wrap max-h-96 overflow-y-auto">
                {msg.reasoning}
              </CollapsibleContent>
            </Collapsible>
          )}
        </div>
      </div>
    </div>
  )
}

const MessageRow = memo(function MessageRow({
  msg,
  showAgentDetails,
  isMobile,
  toolCallIndex,
  forceLeftAlign = false,
  compact = false,
  sessionId = '',
}: MessageRowProps) {
  // Above the content guard on purpose: a thinking row's text lives in
  // `reasoning`, not in a content block, so it has nothing for the guard
  // below to find and would be dropped before it rendered.
  //
  // Visibility is the agent-details flag alone, exactly like a tool row.
  // Not `thinkEnabled`: that toggle asks the *model* for extended thinking
  // on the next turn, so keying display on it made a reader's view of an
  // old turn depend on how the composer happens to be set right now, and
  // left one transcript rendering two different ways in two panels.
  if (msg.role === 'thinking') {
    if (!showAgentDetails) return null
    return (
      <ThinkingRow
        msg={msg}
        isMobile={isMobile}
        compact={compact}
        forceLeftAlign={forceLeftAlign}
      />
    )
  }

  const hasContent = msg.content?.some(c => c.text?.trim())
  if (!hasContent) return null

  const isError = msg.role === 'tool' && msg.stats?.is_error === true
  const hideToolMessage = !showAgentDetails && msg.role === 'tool' && !isError
  const hideSubliminal = !showAgentDetails && msg.role === 'subliminal'
  if (hideToolMessage || hideSubliminal) return null

  const textJoined = useMemo(
    () => msg.content.map(c => c.text).join('\n'),
    [msg.content]
  )

  const isUser = msg.role === 'user'
  const isAssistant = msg.role === 'assistant'
  const isTool = msg.role === 'tool'
  const isSubliminal = msg.role === 'subliminal'

  return (
    <div className={cn('flex gap-3', !compact && !forceLeftAlign && isUser && 'justify-end')}>
      {!compact && !forceLeftAlign && !isUser && !isMobile && (
        <div className="w-7 h-7 rounded-full flex-shrink-0 mt-0.5 overflow-hidden hidden sm:flex">
          {isTool ? (
            <div className="w-full h-full bg-secondary flex items-center justify-center">
              <Wrench className="w-3.5 h-3.5 text-muted-foreground" />
            </div>
          ) : isSubliminal ? (
            <div className="w-full h-full bg-purple-900/40 flex items-center justify-center">
              <Sparkles className="w-3.5 h-3.5 text-purple-300" />
            </div>
          ) : (
            <img src="/lloyd.jpg" alt="Lloyd" className="w-full h-full object-cover" />
          )}
        </div>
      )}
      <div className={cn(
        compact || forceLeftAlign
          ? 'flex-1 min-w-0'
          : `max-w-[80%] ${isUser ? 'min-w-0' : 'flex-1 min-w-0'}`,
      )}>
        <div className={cn(
          'rounded-xl border',
          isTool || isSubliminal ? 'px-2.5 py-1.5' : 'px-3.5 py-2.5',
          isUser
            ? 'bg-primary/15 border-primary/30 text-foreground'
            : isTool && isError
            ? 'bg-destructive/10 border-destructive/40 text-foreground'
            : isTool
            ? 'bg-secondary/40 border-border text-foreground'
            : isSubliminal
            ? 'bg-purple-950/30 border-purple-500/20 text-foreground'
            : 'bg-card border-border text-foreground',
        )}>
          <div className={cn('prose-chat leading-relaxed', isMobile ? 'text-[15px]' : 'text-[13px]')}>
            {isAssistant ? (
              <>
                {msg.reasoning && showAgentDetails && (
                  <Collapsible className="mb-3">
                    <CollapsibleTrigger className="group cursor-pointer flex items-center gap-1 text-xs text-purple-400 hover:text-purple-300 transition-colors">
                      <Brain className="w-3 h-3" />
                      <ChevronRight className="w-3 h-3 transition-transform group-data-[state=open]:rotate-90" />
                      <span className="font-semibold">Thinking</span>
                      <span className="text-muted-foreground/80 font-normal ml-1">
                        ({[thinkDuration(msg.reasoning_ms ?? 0),
                           `${msg.reasoning.length.toLocaleString()} chars`]
                          .filter(Boolean).join(' · ')})
                      </span>
                    </CollapsibleTrigger>
                    <CollapsibleContent className="mt-2 p-3 bg-purple-900/10 border border-purple-500/10 rounded text-xs text-foreground/90 whitespace-pre-wrap max-h-96 overflow-y-auto">
                      {msg.reasoning}
                    </CollapsibleContent>
                  </Collapsible>
                )}
                <Streamdown parseIncompleteMarkdown>{textJoined}</Streamdown>
              </>
            ) : isTool ? (() => {
              const tc = msg.tool_call_id ? toolCallIndex.get(msg.tool_call_id) : undefined
              const toolName = tc?.name || 'Tool'
              const toolSummary = tc?.summary || ''
              const toolArgs = tc?.args || '{}'
              let argsDisplay = toolArgs
              try {
                // `summary` rides in the arguments so the model sees its own
                // captions when this call is replayed as history, but the tool
                // never received it — this block shows what was dispatched, and
                // the header already shows the caption. Dropped only when the
                // header is rendering it, so a tool with a real `summary`
                // parameter of its own still shows it here.
                const parsed = JSON.parse(toolArgs)
                if (toolSummary && parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
                  delete (parsed as Record<string, unknown>).summary
                }
                argsDisplay = JSON.stringify(parsed, null, 2)
              } catch { /* keep raw */ }
              const responseText = textJoined
              return (
                <Collapsible>
                  <CollapsibleTrigger className={cn(
                    'group cursor-pointer flex items-center gap-1.5 text-xs transition-colors w-full min-w-0 text-left',
                    isError
                      ? 'text-destructive hover:text-destructive/80'
                      : 'text-muted-foreground hover:text-foreground',
                  )}>
                    <ChevronRight className="w-3 h-3 shrink-0 transition-transform group-data-[state=open]:rotate-90" />
                    <Wrench className="w-3 h-3 shrink-0" />
                    <span className={cn(
                      'font-mono font-semibold shrink-0',
                      isError ? 'text-destructive' : 'text-foreground',
                    )}>{toolName}</span>
                    {toolSummary && (
                      <span
                        title={toolSummary}
                        className={cn(
                          'font-normal truncate min-w-0',
                          isError ? 'text-destructive/80' : 'text-muted-foreground/90',
                        )}
                      >&mdash; {toolSummary}</span>
                    )}
                  </CollapsibleTrigger>
                  <CollapsibleContent className="mt-2 space-y-2">
                    {argsDisplay !== '{}' && (
                      <div>
                        <div className="text-[10px] uppercase tracking-wide text-muted-foreground mb-1">Arguments</div>
                        <pre className="p-2 bg-muted/40 rounded text-xs text-foreground/90 overflow-x-auto whitespace-pre-wrap font-mono">
                          {argsDisplay}
                        </pre>
                      </div>
                    )}
                    <div>
                      <div className="text-[10px] uppercase tracking-wide text-muted-foreground mb-1">Response</div>
                      <pre className="p-2 bg-muted/40 rounded text-xs text-foreground/90 overflow-x-auto whitespace-pre-wrap font-mono max-h-48 overflow-y-auto">
                        {responseText || '⏳ Running...'}
                      </pre>
                    </div>
                  </CollapsibleContent>
                </Collapsible>
              )
            })() : isSubliminal ? (() => {
              const subl = msg.subliminal
              const kind = subl?.kind || 'other'
              const sources = subl?.sources || []
              const chars = subl?.chars ?? textJoined.length
              return (
                <Collapsible>
                  <CollapsibleTrigger className="group cursor-pointer flex items-center gap-1.5 text-xs text-purple-400/80 hover:text-purple-300 transition-colors w-full">
                    <ChevronRight className="w-3 h-3 shrink-0 transition-transform group-data-[state=open]:rotate-90" />
                    <Sparkles className="w-3 h-3 shrink-0" />
                    <span className="font-semibold uppercase tracking-wide">Subliminal</span>
                    <span className="font-mono font-normal text-purple-500/80">{kind}</span>
                    {sources.length > 0 && (
                      <span className="font-mono font-normal text-muted-foreground truncate">
                        {sources.join(', ')}
                      </span>
                    )}
                    <span className="font-mono font-normal text-muted-foreground/70 ml-auto">
                      {chars.toLocaleString()} chars
                    </span>
                  </CollapsibleTrigger>
                  <CollapsibleContent className="mt-2">
                    <div className="text-[10px] uppercase tracking-wide text-muted-foreground mb-1">Injected context</div>
                    <pre className="p-2 bg-muted/40 rounded text-xs text-foreground/90 overflow-x-auto whitespace-pre-wrap font-mono max-h-96 overflow-y-auto">
                      {textJoined}
                    </pre>
                  </CollapsibleContent>
                </Collapsible>
              )
            })() : (
              <Streamdown parseIncompleteMarkdown>{textJoined}</Streamdown>
            )}
          </div>
        </div>
        <div className="mt-1.5 text-[10px] text-muted-foreground/70 font-mono flex flex-wrap items-center gap-x-2.5 gap-y-0.5">
          <span>{timeStr(msg.timestamp)}</span>
          {msg.stats && isAssistant && (() => {
            const s = msg.stats as TurnStats
            const peak = s.peak_input_tokens ?? s.input_tokens
            const pct = (peak / 262144 * 100).toFixed(1)
            return (<>
              <span>ctx: {peak.toLocaleString()} ({pct}%)</span>
              {s.cache_read > 0 && <span className="text-emerald-700">cache↑: {s.cache_read.toLocaleString()}</span>}
              {s.cache_create > 0 && <span className="text-amber-700">cache✎: {s.cache_create.toLocaleString()}</span>}
              {s.duration_ms != null && <span>time: {(s.duration_ms / 1000).toFixed(1)}s</span>}
              {s.num_turns != null && s.num_turns > 1 && <span>turns: {s.num_turns}</span>}
              {s.files_changed && s.files_changed.files.length > 0 && sessionId && (
                <ChangedFilesFooter
                  sessionId={sessionId}
                  turnId={s.files_changed.turn_id}
                  files={s.files_changed.files}
                />
              )}
            </>)
          })()}
          {msg.context_tokens != null && msg.context_tokens > 0 && isTool && (() => {
            const pct = (msg.context_tokens / 262144 * 100).toFixed(1)
            return <span>ctx: {msg.context_tokens.toLocaleString()} ({pct}%)</span>
          })()}
        </div>
      </div>
      {!forceLeftAlign && isUser && !isMobile && (
        <div className="w-7 h-7 rounded-full bg-secondary flex items-center justify-center flex-shrink-0 mt-0.5 hidden sm:flex">
          <User className="w-3.5 h-3.5 text-muted-foreground" />
        </div>
      )}
    </div>
  )
})

interface ChatPanelProps {
  requestedSessionKey?: string | null
  onSessionLoaded?: () => void
  onActiveSessionChange?: (key: string | null) => void
  onModelSwitch?: (modelName: string) => void
  showAgentDetails?: boolean
  currentSessionKey?: string | null
  pendingModel?: string
  visible?: boolean
  onThinkingChange?: (thinking: boolean, toolName: string | null) => void
  isMobile?: boolean
  // Inner Voice timeline mode: primary actions on the left of a vertical line,
  // IV observations on the right, ordered chronologically.
  timelineRight?: InnerVoiceObservation[]
  /** Compact rendering: drop avatars, full-width bubbles. Used by the
   *  right chat sidebar to reclaim horizontal space. */
  compact?: boolean
}

const SLASH_COMMANDS: Array<{ name: string; desc: string; alias?: string }> = [
  { name: 'new', desc: 'Start a new session' },
  { name: 'clear', desc: 'Clear screen and start new session' },
  { name: 'history', desc: 'Show conversation history' },
  { name: 'retry', desc: 'Retry the last message' },
  { name: 'undo', desc: 'Remove last exchange' },
  { name: 'title', desc: 'Set session title' },
  { name: 'compress', desc: 'Compress conversation context' },
  { name: 'stop', desc: 'Kill background processes' },
  { name: 'background', desc: 'Run prompt in background', alias: 'bg' },
  { name: 'btw', desc: 'Ephemeral side question' },
  { name: 'queue', desc: 'Queue prompt for next turn', alias: 'q' },
  { name: 'think', desc: 'Toggle extended thinking (on/off)' },
  { name: 'profile', desc: 'Show active profile' },
  { name: 'config', desc: 'Show configuration' },
  { name: 'provider', desc: 'Show available providers' },
  { name: 'prompt', desc: 'View/set system prompt' },
  { name: 'personality', desc: 'Set predefined personality' },
  { name: 'statusbar', desc: 'Toggle status bar', alias: 'sb' },
  { name: 'verbose', desc: 'Toggle verbose mode' },
  { name: 'yolo', desc: 'Toggle YOLO mode' },
  { name: 'reasoning', desc: 'Manage reasoning display' },
  { name: 'skin', desc: 'Show/change theme' },
  { name: 'voice', desc: 'Toggle voice mode' },
  { name: 'tools', desc: 'Manage tools' },
  { name: 'toolsets', desc: 'List toolsets' },
  { name: 'skills', desc: 'Search/manage skills' },
  { name: 'goal', desc: 'Set a persistent goal (e.g. /goal write a haiku to /tmp/h.txt)' },
  { name: 'clear-goal', desc: 'Clear the current persistent goal' },
  { name: 'cron', desc: 'Manage scheduled tasks' },
  { name: 'reload-mcp', desc: 'Reload MCP servers', alias: 'reload_mcp' },
  { name: 'browser', desc: 'Connect browser tools' },
  { name: 'plugins', desc: 'List plugins' },
  { name: 'commands', desc: 'Browse all commands' },
  { name: 'help', desc: 'Show available commands' },
  { name: 'usage', desc: 'Show token usage' },
  { name: 'insights', desc: 'Show usage insights' },
  { name: 'platforms', desc: 'Show gateway status', alias: 'gateway' },
  { name: 'paste', desc: 'Check clipboard for image' },
  { name: 'update', desc: 'Update Lloyd' },
  { name: 'quit', desc: 'Exit CLI', alias: 'exit q' },
  { name: 'model', desc: 'Switch or list models', alias: 'switch' },
]


export default function ChatPanel({
  requestedSessionKey,
  onSessionLoaded,
  onActiveSessionChange,
  onModelSwitch,
  showAgentDetails = false,
  currentSessionKey = null,
  pendingModel,
  visible = true,
  onThinkingChange,
  isMobile = false,
  timelineRight,
  compact = false,
}: ChatPanelProps = {}) {
  const [sessionKey, setSessionKey] = useState<string | null>(null)
  const [messages, setMessages] = useState<ApiMessage[]>([])
  const [input, setInput] = useState('')
  const [thinking, setThinking] = useState(false)
  const [sending, setSending] = useState(false)
  const [showCommands, setShowCommands] = useState(false)
  const [models, setModels] = useState<ModelInfo[]>([])
  const [activeToolName, setActiveToolName] = useState<string | null>(null)
  const [thinkEnabled, setThinkEnabled] = useState<boolean>(() => {
    return localStorage.getItem('mc_think_enabled') === '1'
  })
  const [queueState, setQueueState] = useState<QueueState | null>(null)
  // Bumped on session change + every TodoWrite/EnterPlanMode/ExitPlanMode tool result.
  const [todoRefreshKey, setTodoRefreshKey] = useState(0)

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const messagesContainerRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const isNearBottom = useRef<boolean>(true)
  const abortControllerRef = useRef<AbortController | null>(null)
  const clientId = useRef<string>(localStorage.getItem('mc_client_id') || `client_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`)

  useEffect(() => {
    localStorage.setItem('mc_client_id', clientId.current)
  }, [])

  useEffect(() => {
    api.getModels().then(result => {
      if (result.models) setModels(result.models)
    }).catch(err => {
      console.warn('Failed to load models:', err)
    })
  }, [])

  useEffect(() => {
    const activeKey = currentSessionKey || requestedSessionKey
    if (activeKey) {
      loadMessages(activeKey, onSessionLoaded)
    } else if (!activeKey && messages.length > 0) {
      setMessages([])
      setSessionKey(null)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentSessionKey, requestedSessionKey])

  const handleScroll = useCallback(() => {
    const el = messagesContainerRef.current
    if (!el) return
    isNearBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80
  }, [])

  useEffect(() => {
    const el = messagesContainerRef.current
    if (el) {
      el.addEventListener('scroll', handleScroll)
      return () => el.removeEventListener('scroll', handleScroll)
    }
  }, [handleScroll])

  useEffect(() => {
    onThinkingChange?.(thinking, thinking ? activeToolName : null)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [thinking, activeToolName])

  // Idle polling — refresh messages while not actively streaming, and adopt
  // a turn this component did not start.
  //
  // `thinking` / `sending` / `activeToolName` / `queueState` all ride the SSE
  // stream that *this* component opened. When the turn comes from somewhere
  // else — the Chrome side panel's service worker fires its own kickoff POST
  // and abandons the response, an ambient injection, a second tab — those
  // events are never seen here. Before this, the only thing that could have
  // told us the harness was mid-turn was the single getSessionStatus call at
  // the end of loadMessages, which races the POST and usually loses, and the
  // poll that does re-probe status is gated on `thinking` already being true.
  // Closed loop: the side panel rendered a transcript appearing out of
  // nowhere with a live input and a Send button, which read as three missing
  // features. Asking /status on every tick is what breaks it.
  useEffect(() => {
    if (!sessionKey || sending || thinking) return
    if (visible === false) return
    const load = async () => {
      try {
        // A failed status probe must not cost us the message refresh, and a
        // dead backend must not clear a busy state we have no news about.
        const [result, status] = await Promise.all([
          api.loadMessages(sessionKey),
          api.getSessionStatus(sessionKey).catch(() => null),
        ])
        let next: ApiMessage[] | null = null
        if (result.messages) {
          next = result.messages as ApiMessage[]
          setMessages(prev => mergeMessages(prev, next!))
        }
        if (status?.streaming) {
          if (next) {
            const tool = pendingToolName(next)
            if (tool) setActiveToolName(tool)
          }
          setThinking(true)
          setSending(true)
          api.getSessionQueue(sessionKey).then(setQueueState).catch(() => { /* ignore */ })
        }
      } catch (err) {
        console.error('Failed to load messages:', err)
      }
    }
    load()
    const interval = setInterval(load, 5000)
    return () => clearInterval(interval)
  }, [sessionKey, thinking, sending, visible])

  // Restored-stream polling — when we attached to an in-flight backend turn
  // without a local AbortController, watch /status until it settles.
  //
  // On such a turn the indicator says "Thinking..." and not "Working: <tool>",
  // and that is the backend's doing, not a gap here: a call's `tool_calls`
  // row is persisted only when its result arrives. Snapshots of a live
  // session taken 4s, 25s and 45s into one 50s tool call all read zero
  // unanswered `tool_calls` — while the tool runs, the transcript tail is the
  // previous result plus a provisional `thinking` row. Nothing pollable names
  // the running tool, so the adopted path only ever reports the generic
  // state. Only the component that opened the SSE stream gets the name, from
  // `tool_start`, and it skips this effect.
  //
  // `pendingToolName` is therefore write-only-if-found here: it can surface a
  // call the log left unanswered, and must never erase a name the stream put
  // there. Clearing happens when the turn actually ends.
  useEffect(() => {
    if (!sessionKey || !thinking || abortControllerRef.current) return
    const poll = async () => {
      try {
        const status = await api.getSessionStatus(sessionKey)
        const result = await api.loadMessages(sessionKey)
        if (result.messages) {
          const next = result.messages as ApiMessage[]
          setMessages(prev => mergeMessages(prev, next))
          if (status.streaming) {
            const tool = pendingToolName(next)
            if (tool) setActiveToolName(tool)
          }
        }
        if (!status.streaming) {
          setThinking(false)
          setSending(false)
          setActiveToolName(null)
          setQueueState(null)
        }
      } catch { /* ignore */ }
    }
    poll()
    const interval = setInterval(poll, 3000)
    return () => clearInterval(interval)
  }, [sessionKey, thinking])

  // Auto-scroll on new activity. 'auto' during streaming so smooth-scroll
  // animations don't pile up and jank the main thread. Follows by default;
  // pauses only when the user has explicitly scrolled away from the bottom
  // (handleScroll keeps isNearBottom in sync). Watches timelineRight too so
  // Inner Voice observation polls also nudge the view.
  useEffect(() => {
    const hasContent = messages.length > 0 || (timelineRight && timelineRight.length > 0)
    if (!hasContent) return
    const el = messagesContainerRef.current
    if (!el) return
    if (thinking || isNearBottom.current) {
      messagesEndRef.current?.scrollIntoView({ behavior: thinking ? 'auto' : 'smooth' })
    }
  }, [messages, timelineRight, thinking])

  const loadMessages = async (key: string, onLoaded?: () => void) => {
    if (!key) return
    try {
      const result = await api.loadMessages(key)
      if (result.messages) {
        const next = result.messages as ApiMessage[]
        setMessages(prev => mergeMessages(prev, next))
        setSessionKey(key)
        onActiveSessionChange?.(key)
        if (result.model) onModelSwitch?.(result.model)
        onLoaded?.()
        try {
          const status = await api.getSessionStatus(key)
          if (status.streaming) {
            setThinking(true)
            setSending(true)
          } else {
            setThinking(false)
            setSending(false)
          }
        } catch {
          setThinking(false)
          setSending(false)
        }
        isNearBottom.current = true
        setTimeout(() => {
          messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
        }, 100)
      }
    } catch (err) {
      console.error('Failed to load messages:', err)
    }
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = e.target.value
    setInput(value)

    // Auto-execute "/model <alias>" once it's a complete word.
    const match = value.match(/^\/(model|switch)\s+(\w+)$/)
    if (match) {
      const modelAlias = match[2]
      const targetModel = models.find(m => m.alias === modelAlias || m.name === modelAlias)
      if (targetModel) {
        handleModelSwitch(targetModel.name)
        setInput('')
        setShowCommands(false)
        return
      }
    }

    if (value.startsWith('/')) {
      setShowCommands(true)
    } else {
      setShowCommands(false)
    }
  }

  const handleModelSwitch = async (modelName: string) => {
    const activeSession = currentSessionKey
    if (!activeSession) {
      try {
        const result = await api.switchModel(modelName)
        if (result.success) {
          onModelSwitch?.(modelName)
          setMessages(prev => [...prev, {
            id: `msg_${Date.now()}_switch`,
            role: 'assistant',
            content: [{ type: 'text', text: `Switched to **${modelName}** (will apply to new sessions)` }],
            timestamp: new Date().toISOString(),
          }])
        }
      } catch (err) {
        console.error('Failed to switch model:', err)
        setMessages(prev => [...prev, {
          id: `msg_${Date.now()}_err`,
          role: 'tool',
          content: [{ type: 'text', text: `Error switching model: ${err}` }],
          timestamp: new Date().toISOString(),
        }])
      }
      return
    }
    try {
      const result = await api.switchModel(modelName, activeSession)
      if (result.success) {
        onModelSwitch?.(modelName)
        setMessages(prev => [...prev, {
          id: `msg_${Date.now()}_switch`,
          role: 'assistant',
          content: [{ type: 'text', text: `Switched to **${modelName}** for this session` }],
          timestamp: new Date().toISOString(),
        }])
      }
    } catch (err) {
      console.error('Failed to switch model:', err)
      setMessages(prev => [...prev, {
        id: `msg_${Date.now()}_err`,
        role: 'tool',
        content: [{ type: 'text', text: `Error switching model: ${err}` }],
        timestamp: new Date().toISOString(),
      }])
    }
  }

  const handleCommandSelect = async (cmd: string) => {
    if (cmd === 'model') {
      try {
        const result = await api.getModels()
        const modelText = result.models?.length
          ? result.models.map(m =>
              `**/${m.alias}** - ${m.name}\n   ${m.provider} (context: ${m.context_length})`
            ).join('\n\n')
          : 'No models available'
        setMessages(prev => [...prev, {
          id: `msg_${Date.now()}_model`,
          role: 'assistant',
          content: [{ type: 'text', text: `Available models:\n\n${modelText}\n\nType **/model <alias>** to switch (e.g., /model primary)` }],
          timestamp: new Date().toISOString(),
        }])
        setInput('')
        setShowCommands(false)
        inputRef.current?.focus()
        return
      } catch (err) {
        console.error('Failed to get models:', err)
      }
    }
    setInput('/' + cmd + ' ')
    setShowCommands(false)
    inputRef.current?.focus()
  }

  // tool_call_id → { name, args } index for O(1) tool-row lookup.
  const toolCallIndex = useMemo(() => {
    const map = new Map<string, ToolCallRef>()
    for (const m of messages) {
      if (m.role === 'assistant' && m.tool_calls) {
        for (const tc of m.tool_calls) {
          const callId = tc.call_id || tc.id
          if (callId) {
            map.set(callId, {
              name: tc.function?.name || '',
              args: tc.function?.arguments || '{}',
              summary: tc.summary || '',
            })
          }
        }
      }
    }
    return map
  }, [messages])

  type TimelineItem =
    | { kind: 'msg'; ts: number; key: string; msg: ApiMessage }
    | { kind: 'obs'; ts: number; key: string; obs: InnerVoiceObservation }

  const timeline = useMemo<TimelineItem[] | null>(() => {
    if (!timelineRight) return null
    const items: TimelineItem[] = [
      ...messages.map(m => ({ kind: 'msg' as const, ts: Date.parse(m.timestamp), key: m.id, msg: m })),
      ...timelineRight.map(o => ({ kind: 'obs' as const, ts: parseObservationTime(o.created_at), key: `obs_${o.id}`, obs: o })),
    ]
    items.sort((a, b) => a.ts - b.ts)
    return items
  }, [messages, timelineRight])

  // Build the visible command list.
  const filteredCommands = useMemo(() => {
    const filter = input.startsWith('/') ? input.slice(1).toLowerCase() : ''
    if (filter.startsWith('model ') || filter.startsWith('switch ')) {
      const modelArg = filter.split(' ')[1] || ''
      return models
        .filter(m => m.alias.includes(modelArg) || m.name.includes(modelArg))
        .map(m => ({ name: `model ${m.alias}`, desc: m.name, alias: m.alias }))
        .slice(0, 8)
    }
    if (!filter) return SLASH_COMMANDS.slice(0, 8)
    return SLASH_COMMANDS
      .filter(cmd => cmd.name.includes(filter) || (cmd.alias && cmd.alias.includes(filter)))
      .slice(0, 8)
  }, [input, models])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    const text = input.trim()
    if (!text || sending || thinking) return

    if (text === '/new' || text === '/reset') {
      setMessages([])
      setInput('')
      setSending(false)
      setThinking(false)
      setSessionKey(null)
      onActiveSessionChange?.(null)
      localStorage.removeItem('mc_session_id')
      return
    }

    if (text === '/clear-goal' || text === '/cleargoal') {
      if (!sessionKey) {
        setMessages(prev => [...prev, {
          id: `msg_${Date.now()}_goal`,
          role: 'assistant',
          content: [{ type: 'text', text: '/clear-goal requires an active session.' }],
          timestamp: new Date().toISOString(),
        }])
        setInput('')
        return
      }
      try {
        await api.clearSessionGoal(sessionKey)
        setTodoRefreshKey(k => k + 1)
        setMessages(prev => [...prev, {
          id: `msg_${Date.now()}_goal`,
          role: 'assistant',
          content: [{ type: 'text', text: '🎯 Goal cleared.' }],
          timestamp: new Date().toISOString(),
        }])
      } catch (err) {
        setMessages(prev => [...prev, {
          id: `msg_${Date.now()}_goal`,
          role: 'assistant',
          content: [{ type: 'text', text: `Failed to clear goal: ${err instanceof Error ? err.message : String(err)}` }],
          timestamp: new Date().toISOString(),
        }])
      }
      setInput('')
      return
    }

    if (text.startsWith('/goal')) {
      const goalText = text.slice('/goal'.length).trim()
      if (!goalText) {
        setMessages(prev => [...prev, {
          id: `msg_${Date.now()}_goal`,
          role: 'assistant',
          content: [{ type: 'text', text: 'Usage: `/goal <verifiable end condition>` — e.g. `/goal save a haiku about supervisord to /tmp/h.txt`. The inner voice will check after each turn and loop until the goal is met.' }],
          timestamp: new Date().toISOString(),
        }])
        setInput('')
        return
      }
      // Need an active session to attach the goal to. If we don't have one,
      // streamMessage will create it on the first turn; in that case post the
      // goal after onSession fires.
      const submitGoal = async (sid: string) => {
        try {
          await api.setSessionGoal(sid, goalText)
          setTodoRefreshKey(k => k + 1)
        } catch (err) {
          console.warn('setSessionGoal failed:', err)
        }
      }
      if (sessionKey) {
        await submitGoal(sessionKey)
      }
      // Fall through: send the goal text as the first user message so the
      // loop kicks off immediately, matching Claude Code's /goal behavior.
      setInput('')
      setSending(true)
      setMessages(prev => [...prev, {
        id: `msg_${Date.now()}`,
        role: 'user',
        content: [{ type: 'text', text: goalText }],
        timestamp: new Date().toISOString(),
      }])
      setThinking(true)
      // Track whether the goal was already submitted (when we had a session
      // up front) or whether we need to submit after the session is created.
      let goalPosted = !!sessionKey
      let assistantMsgIdG: string | null = null
      let segmentCounterG = 0
      let streamingStartedG = false
      let settledG = false
      let accumulatedThinkingG = ''
      // Live duration: the harness only reports it on `thinking_done`,
      // which lands *after* the iteration's text, so the panel would show
      // no time for the whole stream. Measure the delta timestamps the
      // same way (first chunk to last) until the real number arrives.
      let thinkStartG = 0
      let thinkLastG = 0
      let thinkMsG = 0
      const thinkTrackerG = makeThinkingTracker(fn => setMessages(fn))
      const thinkingMsG = () => thinkMsG || (thinkStartG ? thinkLastG - thinkStartG : 0)
      let pendingDeltaG = ''
      let rafIdG: number | null = null
      const flushDeltaG = () => {
        rafIdG = null
        const delta = pendingDeltaG
        if (!delta || !assistantMsgIdG) return
        pendingDeltaG = ''
        const cid = assistantMsgIdG
        setMessages(prev => prev.map(m =>
          m.id === cid
            ? { ...m, content: [{ type: 'text' as const, text: m.content[0].text + delta }] }
            : m
        ))
      }
      const scheduleFlushG = () => { if (rafIdG === null) rafIdG = requestAnimationFrame(flushDeltaG) }
      abortControllerRef.current = api.streamMessage(goalText, clientId.current, sessionKey || undefined, {
        onSession: async (sid) => {
          if (!sessionKey) {
            setSessionKey(sid)
            localStorage.setItem('mc_session_id', sid)
            onActiveSessionChange?.(sid)
          }
          if (!goalPosted) {
            await submitGoal(sid)
            goalPosted = true
          }
        },
        onQueueState: (s) => setQueueState(s),
        onToolStart: (callId, name, args, contextTokens, summary) => {
          setActiveToolName(name)
          assistantMsgIdG = null
          accumulatedThinkingG = ''
          thinkStartG = 0; thinkLastG = 0; thinkMsG = 0
          setMessages(prev => [
            ...prev,
            { id: `msg_${callId}_tc`, role: 'assistant', content: [{ type: 'text', text: '' }], tool_calls: [{ id: callId, call_id: callId, type: 'function', function: { name, arguments: args }, summary }], timestamp: new Date().toISOString() },
            { id: `msg_${callId}_result`, role: 'tool', content: [{ type: 'text', text: '⏳ Running...' }], tool_call_id: callId, context_tokens: contextTokens, timestamp: new Date().toISOString() },
          ])
        },
        onToolComplete: (callId, _name, result) => {
          setMessages(prev => {
            const upd = prev.map(m => m.id === `msg_${callId}_result` ? { ...m, content: [{ type: 'text' as const, text: result }] } : m)
            const stillPending = upd.find(m => m.role === 'tool' && m.content[0]?.text === '⏳ Running...')
            if (!stillPending) setActiveToolName(null)
            return upd
          })
          if (_name === 'TodoWrite' || _name === 'EnterPlanMode' || _name === 'ExitPlanMode' || _name === 'SetGoal' || _name === 'ClearGoal') {
            setTodoRefreshKey(k => k + 1)
          }
        },
        onThinkingDelta: (delta) => {
          thinkLastG = Date.now()
          if (!thinkStartG) thinkStartG = thinkLastG
          accumulatedThinkingG += delta
          thinkTrackerG.onDelta()
        },
        onThinkingDone: (fullText, durationMs, seq, iteration) => {
          const text = fullText || accumulatedThinkingG
          if (thinkTrackerG.onDone(text, durationMs, seq, iteration)) {
            // The phase has its own row now. Leaving it on the accumulator
            // would also hang it off the next assistant bubble, rendering
            // the same thinking twice.
            accumulatedThinkingG = ''
            thinkStartG = 0; thinkLastG = 0; thinkMsG = 0
          } else {
            // Trace off: no row is coming, so the reasoning belongs on the
            // assistant bubble as it always did. Attached here rather than
            // on the text delta because that fires *before* thinking_done
            // and would show a phase the reload would not have.
            accumulatedThinkingG = text
            thinkMsG = durationMs || thinkMsG
            const aid = assistantMsgIdG
            const ms = thinkingMsG()
            if (aid) {
              setMessages(prev => prev.map(m => m.id === aid
                ? { ...m, reasoning: text, ...(ms ? { reasoning_ms: ms } : {}) }
                : m))
            }
          }
        },
        onTextDelta: (delta) => {
          streamingStartedG = true
          if (assistantMsgIdG === null) {
            segmentCounterG += 1
            const nid = `msg_${Date.now()}_resp_${segmentCounterG}`
            assistantMsgIdG = nid
            setMessages(prev => [...prev, { id: nid, role: 'assistant', content: [{ type: 'text', text: delta }], timestamp: new Date().toISOString() }])
          } else {
            pendingDeltaG += delta
            scheduleFlushG()
          }
        },
        onDone: () => {
          if (settledG) return
          settledG = true
          thinkTrackerG.discard()
          // Flush any final delta before clearing state.
          if (pendingDeltaG) flushDeltaG()
          setSending(false)
          setThinking(false)
          setTodoRefreshKey(k => k + 1)
          // Suppress unused-var warnings for stream gating fields kept for parity with the main handler.
          void streamingStartedG
        },
        onError: (detail) => {
          thinkTrackerG.discard()
          setMessages(prev => [...prev, { id: `msg_${Date.now()}_err`, role: 'assistant', content: [{ type: 'text', text: `Error: ${detail}` }], timestamp: new Date().toISOString() }])
          setSending(false)
          setThinking(false)
        },
        onAborted: () => { thinkTrackerG.discard(); setSending(false); setThinking(false) },
      })
      return
    }

    if (text.startsWith('/think')) {
      const arg = text.split(/\s+/)[1]?.toLowerCase() || ''
      let next: boolean
      if (arg === 'on') next = true
      else if (arg === 'off') next = false
      else if (!arg) next = !thinkEnabled
      else {
        setMessages(prev => [...prev, {
          id: `msg_${Date.now()}_think`,
          role: 'assistant',
          content: [{ type: 'text', text: `Invalid argument **"${arg}"**. Use: /think, /think on, /think off` }],
          timestamp: new Date().toISOString(),
        }])
        setInput('')
        return
      }
      setThinkEnabled(next)
      localStorage.setItem('mc_think_enabled', next ? '1' : '0')
      setMessages(prev => [...prev, {
        id: `msg_${Date.now()}_think`,
        role: 'assistant',
        content: [{ type: 'text', text: next ? '🧠 Extended thinking **on**' : '🧠 Extended thinking **off**' }],
        timestamp: new Date().toISOString(),
      }])
      setInput('')
      return
    }

    setInput('')
    setSending(true)

    setMessages(prev => [...prev, {
      id: `msg_${Date.now()}`,
      role: 'user',
      content: [{ type: 'text', text }],
      timestamp: new Date().toISOString(),
    }])

    setThinking(true)

    let assistantMsgId: string | null = null
    let segmentCounter = 0
    let streamingStarted = false
    let settled = false
    let accumulatedThinking = ''
    // See the goal path above: `thinking_done` (which carries the
    // harness's own measurement) arrives after the iteration's text, so
    // the live panel measures the delta timestamps until it lands.
    let thinkStart = 0
    let thinkLast = 0
    let thinkMs = 0
    const thinkTracker = makeThinkingTracker(fn => setMessages(fn))
    const thinkingMs = () => thinkMs || (thinkStart ? thinkLast - thinkStart : 0)

    // RAF-batched delta flush — per-token setState on long sessions kills the
    // main thread; coalesce into a single update per animation frame instead.
    let pendingDelta = ''
    let rafId: number | null = null
    const flushDelta = () => {
      rafId = null
      const delta = pendingDelta
      if (!delta || !assistantMsgId) return
      pendingDelta = ''
      const currentId = assistantMsgId
      setMessages(prev => prev.map(m =>
        m.id === currentId
          ? {
              ...m,
              content: [{ type: 'text' as const, text: m.content[0].text + delta }],
            }
          : m
      ))
    }
    const scheduleFlush = () => {
      if (rafId === null) rafId = requestAnimationFrame(flushDelta)
    }

    const controller = api.streamMessage(text, clientId.current, sessionKey || undefined, {
      onSession: (sid) => {
        if (!sessionKey) {
          setSessionKey(sid)
          localStorage.setItem('mc_session_id', sid)
          onActiveSessionChange?.(sid)
        }
      },
      onQueueState: (state) => setQueueState(state),
      onToolStart: (callId, name, args, contextTokens, summary) => {
        setActiveToolName(name)
        assistantMsgId = null
        accumulatedThinking = ''
        thinkStart = 0; thinkLast = 0; thinkMs = 0
        setMessages(prev => [
          ...prev,
          {
            id: `msg_${callId}_tc`,
            role: 'assistant' as const,
            content: [{ type: 'text' as const, text: '' }],
            // `args` is already the JSON string the model emitted. It used
            // to be re-stringified here, which double-encoded it and made
            // the live Arguments block render an escaped blob until the
            // turn finished and the persisted row replaced it.
            tool_calls: [{ id: callId, call_id: callId, type: 'function', function: { name, arguments: args }, summary }],
            timestamp: new Date().toISOString(),
          },
          {
            id: `msg_${callId}_result`,
            role: 'tool' as const,
            content: [{ type: 'text' as const, text: '⏳ Running...' }],
            tool_call_id: callId,
            context_tokens: contextTokens,
            timestamp: new Date().toISOString(),
          },
        ])
      },
      onToolComplete: (callId, _name, result) => {
        setMessages(prev => {
          const updated = prev.map(m =>
            m.id === `msg_${callId}_result`
              ? { ...m, content: [{ type: 'text' as const, text: result }] }
              : m
          )
          const stillPending = updated.find(m => m.role === 'tool' && m.content[0]?.text === '⏳ Running...')
          if (!stillPending) setActiveToolName(null)
          return updated
        })
        if (
          _name === 'TodoWrite' || _name === 'EnterPlanMode' || _name === 'ExitPlanMode'
          || _name === 'SetGoal' || _name === 'ClearGoal'
        ) setTodoRefreshKey(k => k + 1)
      },
      onThinkingDelta: (delta) => {
        thinkLast = Date.now()
        if (!thinkStart) thinkStart = thinkLast
        accumulatedThinking += delta
        thinkTracker.onDelta()
      },
      onThinkingDone: (fullText, durationMs, seq, iteration) => {
        const text = fullText || accumulatedThinking
        if (thinkTracker.onDone(text, durationMs, seq, iteration)) {
          // The phase has its own row now. Leaving it on the accumulator
          // would also hang it off the next assistant bubble, rendering
          // the same thinking twice.
          accumulatedThinking = ''
          thinkStart = 0; thinkLast = 0; thinkMs = 0
        } else {
          // Trace off: see the goal path above.
          accumulatedThinking = text
          thinkMs = durationMs || thinkMs
          const aid = assistantMsgId
          const ms = thinkingMs()
          if (aid) {
            setMessages(prev => prev.map(m => m.id === aid
              ? { ...m, reasoning: text, ...(ms ? { reasoning_ms: ms } : {}) }
              : m))
          }
        }
      },
      onTextDelta: (delta) => {
        streamingStarted = true
        if (assistantMsgId === null) {
          segmentCounter += 1
          const newId = `msg_${Date.now()}_resp_${segmentCounter}`
          assistantMsgId = newId
          setMessages(prev => [...prev, {
            id: newId,
            role: 'assistant' as const,
            content: [{ type: 'text' as const, text: delta }],
            timestamp: new Date().toISOString(),
          }])
        } else {
          pendingDelta += delta
          scheduleFlush()
        }
      },
      onDone: (response, _sid, stats, reasoning, _cancelled, reasoningMs) => {
        if (settled) return
        settled = true
        thinkTracker.discard()
        if (rafId !== null) { cancelAnimationFrame(rafId); rafId = null }
        const pendingFinal = pendingDelta
        pendingDelta = ''
        const finalReasoning = accumulatedThinking || reasoning || ''
        const finalReasoningMs = reasoningMs || thinkingMs()
        if (!streamingStarted && response) {
          const fallbackId = `msg_${Date.now()}_resp_final`
          setMessages(prev => [...prev, {
            id: fallbackId,
            role: 'assistant' as const,
            content: [{ type: 'text' as const, text: response }],
            timestamp: new Date().toISOString(),
            stats,
            ...(finalReasoning
              ? {
                  reasoning: finalReasoning,
                  ...(finalReasoningMs ? { reasoning_ms: finalReasoningMs } : {}),
                }
              : {}),
          }])
        } else if (assistantMsgId && (stats || finalReasoning || pendingFinal)) {
          const lastId = assistantMsgId
          setMessages(prev => prev.map(m =>
            m.id === lastId
              ? {
                  ...m,
                  ...(pendingFinal ? { content: [{ type: 'text' as const, text: m.content[0].text + pendingFinal }] } : {}),
                  ...(stats ? { stats } : {}),
                  ...(finalReasoning
                    ? {
                        reasoning: finalReasoning,
                        ...(finalReasoningMs ? { reasoning_ms: finalReasoningMs } : {}),
                      }
                    : {}),
                }
              : m
          ))
        }
        abortControllerRef.current = null
        setActiveToolName(null)
        setThinking(false)
        setSending(false)
        setQueueState(null)
        inputRef.current?.focus()
      },
      onError: (detail) => {
        if (settled) return
        settled = true
        thinkTracker.discard()
        if (rafId !== null) { cancelAnimationFrame(rafId); rafId = null }
        pendingDelta = ''
        setMessages(prev => [...prev, {
          id: `msg_${Date.now()}_err`,
          role: 'tool' as const,
          content: [{ type: 'text' as const, text: `Error: ${detail}` }],
          timestamp: new Date().toISOString(),
        }])
        abortControllerRef.current = null
        setActiveToolName(null)
        setThinking(false)
        setSending(false)
        setQueueState(null)
        inputRef.current?.focus()
      },
      onAborted: () => {
        if (settled) return
        settled = true
        thinkTracker.discard()
        if (rafId !== null) { cancelAnimationFrame(rafId); rafId = null }
        pendingDelta = ''
        abortControllerRef.current = null
        setActiveToolName(null)
        setThinking(false)
        setSending(false)
        setQueueState(null)
        setMessages(prev => [...prev, {
          id: `msg_${Date.now()}_interrupted`,
          role: 'assistant' as const,
          content: [{ type: 'text' as const, text: '*[Interrupted]*' }],
          timestamp: new Date().toISOString(),
        }])
        inputRef.current?.focus()
      },
    }, !sessionKey ? pendingModel : undefined, thinkEnabled ? 'on' : undefined)
    abortControllerRef.current = controller
  }

  const handleStop = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      if (sessionKey) {
        api.cancelSession(sessionKey, { drainPending: true }).catch(() => {})
      }
    } else if (sessionKey) {
      api.cancelSession(sessionKey, { drainPending: true }).then(() => {
        setThinking(false)
        setSending(false)
        setMessages(prev => [...prev, {
          id: `msg_${Date.now()}_interrupted`,
          role: 'assistant' as const,
          content: [{ type: 'text' as const, text: '*[Cancelled]*' }],
          timestamp: new Date().toISOString(),
        }])
      }).catch(() => {
        setThinking(false)
        setSending(false)
      })
    }
  }

  const toggleThink = () => {
    const next = !thinkEnabled
    setThinkEnabled(next)
    localStorage.setItem('mc_think_enabled', next ? '1' : '0')
  }

  const thinkingIndicatorBody = (
    <div className="bg-card border border-border px-3.5 py-2.5 rounded-xl">
      <div className="flex items-center gap-2 text-[13px] text-muted-foreground">
        <Loader2 className="w-4 h-4 animate-spin text-primary" />
        {queueState?.current?.source === 'ambient'
          ? <span><span className="text-amber-400">Ambient context</span> — Lloyd is processing background input...</span>
          : activeToolName
            ? <span>Working: <span className="font-mono text-primary">{activeToolName}</span>...</span>
            : <span>Thinking...</span>
        }
        {queueState && queueState.depth > 0 && (
          <Badge variant="secondary" className="ml-2 font-mono">
            queue: {queueState.pending_user}u + {queueState.pending_ambient}a
          </Badge>
        )}
      </div>
    </div>
  )

  return (
    <div className="flex flex-col h-full">
      {/* Messages */}
      <main
        ref={messagesContainerRef}
        className={cn('flex-1 overflow-y-auto p-4', timeline === null && 'space-y-4')}
      >
        {messages.length === 0 && (
          <div className="flex items-center justify-center h-full">
            <div className="text-center text-muted-foreground">
              <MessageCircle className="w-10 h-10 mx-auto mb-3 opacity-30" />
              <p className="text-sm">Welcome to Lloyd Mission Control</p>
              <p className="text-xs mt-1 text-muted-foreground/80">Send a message to get started</p>
            </div>
          </div>
        )}

        {timeline === null && messages.map((msg) => (
          <MessageRow
            key={msg.id}
            msg={msg}
            showAgentDetails={showAgentDetails}
            isMobile={isMobile}
            toolCallIndex={toolCallIndex}
            compact={compact}
            sessionId={sessionKey ?? ''}
          />
        ))}

        {timeline === null && thinking && (
          <div className="flex gap-3">
            {!compact && (
              <div className="w-7 h-7 rounded-full flex-shrink-0 mt-0.5 overflow-hidden hidden sm:flex">
                <img src="/lloyd.jpg" alt="Lloyd" className="w-full h-full object-cover" />
              </div>
            )}
            {thinkingIndicatorBody}
          </div>
        )}

        {timeline !== null && (
          <div className="flex flex-col">
            {timeline.map(item => {
              if (item.kind === 'msg') {
                const m = item.msg
                if (m.role === 'thinking') {
                  // Carries no content blocks, so it has to clear the
                  // guard below before it reaches MessageRow.
                  if (!showAgentDetails) return null
                } else {
                  const hasContent = m.content?.some(c => c.text?.trim())
                  if (!hasContent) return null
                  const isError = m.role === 'tool' && m.stats?.is_error === true
                  if (!showAgentDetails && m.role === 'tool' && !isError) return null
                  if (!showAgentDetails && m.role === 'subliminal') return null
                }
              }
              return (
                <div key={item.key} className="grid grid-cols-[1fr_28px_1fr] gap-x-3 py-1.5">
                  <div className="flex justify-end min-w-0">
                    {item.kind === 'msg' && (
                      <div className="w-full max-w-[94%] min-w-0">
                        <MessageRow
                          msg={item.msg}
                          showAgentDetails={showAgentDetails}
                          isMobile={isMobile}
                          toolCallIndex={toolCallIndex}
                          forceLeftAlign
                        />
                      </div>
                    )}
                  </div>
                  <div className="relative flex justify-center">
                    <div className="absolute inset-y-0 left-1/2 -translate-x-1/2 border-l border-border/60" />
                    <div className={cn(
                      'relative mt-3 w-2 h-2 rounded-full ring-2 ring-card',
                      item.kind === 'obs'
                        ? (item.obs.trigger === 'result' && item.obs.action === 'noop'
                            ? 'bg-emerald-400'
                            : actionStyle(item.obs.action).dot)
                        : item.msg.role === 'thinking'
                        ? 'bg-purple-400'
                        : 'bg-primary',
                    )} />
                  </div>
                  <div className="flex justify-start min-w-0">
                    {item.kind === 'obs' && (
                      <div className="w-full max-w-[94%] min-w-0">
                        <ObservationBubble obs={item.obs} />
                      </div>
                    )}
                  </div>
                </div>
              )
            })}
            {thinking && (
              <div className="grid grid-cols-[1fr_28px_1fr] gap-x-3 py-1.5">
                <div className="flex justify-end min-w-0">
                  <div className="w-full max-w-[94%] min-w-0">
                    {thinkingIndicatorBody}
                  </div>
                </div>
                <div className="relative flex justify-center">
                  <div className="absolute inset-y-0 left-1/2 -translate-x-1/2 border-l border-border/60" />
                  <div className="relative mt-3 w-2 h-2 rounded-full ring-2 ring-card bg-primary animate-pulse" />
                </div>
                <div />
              </div>
            )}
          </div>
        )}

        <div ref={messagesEndRef} />
      </main>

      <GoalHeader
        sessionId={sessionKey}
        refreshKey={todoRefreshKey}
        onCleared={() => setTodoRefreshKey(k => k + 1)}
      />
      <PlanHeader
        sessionId={sessionKey}
        refreshKey={todoRefreshKey}
        onExitPlanMode={() => setTodoRefreshKey(k => k + 1)}
      />
      <TodoList sessionId={sessionKey} refreshKey={todoRefreshKey} />

      {/* Input */}
      <footer className="p-3 border-t border-border relative">
        <form onSubmit={handleSubmit} className="flex gap-2 items-center">
          <Input
            ref={inputRef}
            type="text"
            value={input}
            onChange={handleInputChange}
            onKeyDown={(e) => {
              // Let cmdk handle navigation keys when the palette is open by
              // forwarding them to the hidden Command instance.
              if (showCommands && (e.key === 'ArrowDown' || e.key === 'ArrowUp')) {
                // Command palette handles its own focus via data-attrs; we
                // only need to prevent the input cursor moving.
                return
              }
              if (showCommands && e.key === 'Escape') {
                e.preventDefault()
                setShowCommands(false)
              }
            }}
            placeholder={thinkEnabled ? 'Talk to Lloyd... (thinking on)' : 'Talk to Lloyd... (use / for commands)'}
            className={cn('flex-1 h-[38px] bg-card text-foreground', isMobile && 'text-base')}
            disabled={sending || thinking}
          />
          {/* Think on/off toggle */}
          <Button
            type="button"
            variant="ghost"
            size="icon"
            onClick={toggleThink}
            title={`Extended thinking: ${thinkEnabled ? 'on (click to turn off)' : 'off (click to turn on)'}`}
            className={cn(
              'h-[38px] w-[38px] shrink-0 border',
              thinkEnabled
                ? 'bg-purple-600/20 border-purple-500/30 text-purple-400 hover:bg-purple-600/30 hover:text-purple-300'
                : 'bg-card border-border text-muted-foreground hover:bg-accent',
            )}
          >
            <Brain className="w-4 h-4" />
          </Button>
          {/* Submit / Stop */}
          {(sending || thinking) ? (
            <Button
              type="button"
              variant="destructive"
              size="icon"
              onClick={handleStop}
              title="Stop"
              className="h-[38px] w-[38px] shrink-0"
            >
              <Square className="w-4 h-4" />
            </Button>
          ) : (
            <Button
              type="submit"
              size="icon"
              disabled={!input.trim()}
              title="Send"
              className="h-[38px] w-[38px] shrink-0"
            >
              <Send className="w-4 h-4" />
            </Button>
          )}
        </form>

        {/* Slash command palette — cmdk handles keyboard nav (↑↓/Enter)
            internally; we listen for Escape on the Input and close. */}
        {showCommands && filteredCommands.length > 0 && (
          <div className="absolute bottom-full left-0 right-0 mb-1 mx-3 border border-border rounded-lg shadow-lg max-h-60 overflow-hidden z-50 bg-popover">
            <Command shouldFilter={false} loop>
              <CommandList>
                <CommandEmpty>No matches.</CommandEmpty>
                <CommandGroup>
                  {filteredCommands.map(cmd => (
                    <CommandItem
                      key={cmd.name}
                      value={cmd.name}
                      onSelect={() => handleCommandSelect(cmd.name.split(' ')[0] === 'model' ? cmd.name : cmd.name)}
                    >
                      <span className="font-mono text-primary">/{cmd.name}</span>
                      {cmd.alias && (
                        <span className="text-xs text-muted-foreground">({cmd.alias.split(' ')[0]})</span>
                      )}
                      <span className="ml-auto text-xs text-muted-foreground truncate max-w-[200px]">
                        {cmd.desc}
                      </span>
                    </CommandItem>
                  ))}
                </CommandGroup>
              </CommandList>
            </Command>
          </div>
        )}
      </footer>
    </div>
  )
}
