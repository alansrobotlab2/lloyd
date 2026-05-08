import { useState, useRef, useEffect, useCallback, useMemo, memo } from 'react'
import {
  Send, User, Loader2, Brain, MessageCircle, ChevronRight,
  Wrench, Square, Sparkles,
} from 'lucide-react'
import { Streamdown } from 'streamdown'
import { api, type MessageEntry as ApiMessage, type ModelInfo, type TurnStats, type QueueState, type InnerVoiceObservation } from '../api'
import TodoList from './TodoList'
import PlanHeader from './PlanHeader'
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

type ToolCallRef = { name: string; args: string }

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
      if (pt === nt && sameStats && p.reasoning === n.reasoning) {
        return p
      }
    }
    different = true
    return n
  })
  return different ? merged : prev
}

// ── memoized message row ───────────────────────────────────────────────

interface MessageRowProps {
  msg: ApiMessage
  showAgentDetails: boolean
  thinkEnabled: boolean
  isMobile: boolean
  toolCallIndex: Map<string, ToolCallRef>
  forceLeftAlign?: boolean
  /** Compact mode: drop avatars, full-width bubbles. Used by the right
   *  chat sidebar to reclaim horizontal space in narrow layouts. */
  compact?: boolean
}

const MessageRow = memo(function MessageRow({
  msg,
  showAgentDetails,
  thinkEnabled,
  isMobile,
  toolCallIndex,
  forceLeftAlign = false,
  compact = false,
}: MessageRowProps) {
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
                {msg.reasoning && (showAgentDetails || thinkEnabled) && (
                  <Collapsible className="mb-3">
                    <CollapsibleTrigger className="group cursor-pointer flex items-center gap-1 text-xs text-purple-400 hover:text-purple-300 transition-colors">
                      <Brain className="w-3 h-3" />
                      <ChevronRight className="w-3 h-3 transition-transform group-data-[state=open]:rotate-90" />
                      <span className="font-semibold">Thinking</span>
                      <span className="text-muted-foreground/80 font-normal ml-1">
                        ({msg.reasoning.length.toLocaleString()} chars)
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
              const toolName = tc?.name || ''
              const toolArgs = tc?.args || '{}'
              let argsDisplay = toolArgs
              try {
                argsDisplay = JSON.stringify(JSON.parse(toolArgs), null, 2)
              } catch { /* keep raw */ }
              const responseText = textJoined
              return (
                <Collapsible>
                  <CollapsibleTrigger className={cn(
                    'group cursor-pointer flex items-center gap-1.5 text-xs transition-colors',
                    isError
                      ? 'text-destructive hover:text-destructive/80'
                      : 'text-muted-foreground hover:text-foreground',
                  )}>
                    <ChevronRight className="w-3 h-3 shrink-0 transition-transform group-data-[state=open]:rotate-90" />
                    <Wrench className="w-3 h-3 shrink-0" />
                    <span className="font-semibold uppercase tracking-wide">Tool</span>
                    {toolName && (
                      <span className={cn(
                        'font-mono font-normal truncate',
                        isError ? 'text-destructive/80' : 'text-muted-foreground/80',
                      )}>{toolName}</span>
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

  // Idle polling — refresh messages while not actively streaming.
  useEffect(() => {
    if (!sessionKey || sending || thinking) return
    if (visible === false) return
    const load = async () => {
      try {
        const result = await api.loadMessages(sessionKey)
        if (result.messages) {
          const next = result.messages as ApiMessage[]
          setMessages(prev => mergeMessages(prev, next))
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
  useEffect(() => {
    if (!sessionKey || !thinking || abortControllerRef.current) return
    const poll = async () => {
      try {
        const status = await api.getSessionStatus(sessionKey)
        if (!status.streaming) {
          setThinking(false)
          setSending(false)
          const result = await api.loadMessages(sessionKey)
          if (result.messages) {
            const next = result.messages as ApiMessage[]
            setMessages(prev => mergeMessages(prev, next))
          }
        } else {
          const result = await api.loadMessages(sessionKey)
          if (result.messages) {
            const next = result.messages as ApiMessage[]
            setMessages(prev => mergeMessages(prev, next))
          }
        }
      } catch { /* ignore */ }
    }
    poll()
    const interval = setInterval(poll, 3000)
    return () => clearInterval(interval)
  }, [sessionKey, thinking])

  // Auto-scroll on new messages. 'auto' during streaming so smooth-scroll
  // animations don't pile up and jank the main thread.
  useEffect(() => {
    if (messages.length === 0) return
    const el = messagesContainerRef.current
    if (!el) return
    if (thinking) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'auto' })
    } else {
      const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 200
      if (nearBottom) {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
      }
    }
  }, [messages, thinking])

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
      const currentThinking = accumulatedThinking
      setMessages(prev => prev.map(m =>
        m.id === currentId
          ? {
              ...m,
              content: [{ type: 'text' as const, text: m.content[0].text + delta }],
              ...(currentThinking && !m.reasoning ? { reasoning: currentThinking } : {}),
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
      onToolStart: (callId, name, args, contextTokens) => {
        setActiveToolName(name)
        assistantMsgId = null
        accumulatedThinking = ''
        setMessages(prev => [
          ...prev,
          {
            id: `msg_${callId}_tc`,
            role: 'assistant' as const,
            content: [{ type: 'text' as const, text: '' }],
            tool_calls: [{ id: callId, call_id: callId, type: 'function', function: { name, arguments: JSON.stringify(args) } }],
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
        ) setTodoRefreshKey(k => k + 1)
      },
      onThinkingDelta: (delta) => { accumulatedThinking += delta },
      onThinkingDone: (fullText) => {
        accumulatedThinking = fullText || accumulatedThinking
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
            ...(accumulatedThinking ? { reasoning: accumulatedThinking } : {}),
          }])
        } else {
          pendingDelta += delta
          scheduleFlush()
        }
      },
      onDone: (response, _sid, stats, reasoning) => {
        if (settled) return
        settled = true
        if (rafId !== null) { cancelAnimationFrame(rafId); rafId = null }
        const pendingFinal = pendingDelta
        pendingDelta = ''
        const finalReasoning = accumulatedThinking || reasoning || ''
        if (!streamingStarted && response) {
          const fallbackId = `msg_${Date.now()}_resp_final`
          setMessages(prev => [...prev, {
            id: fallbackId,
            role: 'assistant' as const,
            content: [{ type: 'text' as const, text: response }],
            timestamp: new Date().toISOString(),
            stats,
            ...(finalReasoning ? { reasoning: finalReasoning } : {}),
          }])
        } else if (assistantMsgId && (stats || finalReasoning || pendingFinal)) {
          const lastId = assistantMsgId
          setMessages(prev => prev.map(m =>
            m.id === lastId
              ? {
                  ...m,
                  ...(pendingFinal ? { content: [{ type: 'text' as const, text: m.content[0].text + pendingFinal }] } : {}),
                  ...(stats ? { stats } : {}),
                  ...(finalReasoning ? { reasoning: finalReasoning } : {}),
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
            thinkEnabled={thinkEnabled}
            isMobile={isMobile}
            toolCallIndex={toolCallIndex}
            compact={compact}
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
                const hasContent = m.content?.some(c => c.text?.trim())
                if (!hasContent) return null
                const isError = m.role === 'tool' && m.stats?.is_error === true
                if (!showAgentDetails && m.role === 'tool' && !isError) return null
                if (!showAgentDetails && m.role === 'subliminal') return null
              }
              return (
                <div key={item.key} className="grid grid-cols-[1fr_28px_1fr] gap-x-3 py-1.5">
                  <div className="flex justify-end min-w-0">
                    {item.kind === 'msg' && (
                      <div className="w-full max-w-[94%] min-w-0">
                        <MessageRow
                          msg={item.msg}
                          showAgentDetails={showAgentDetails}
                          thinkEnabled={thinkEnabled}
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
