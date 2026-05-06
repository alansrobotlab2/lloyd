import { useState, useRef, useEffect, useCallback, useMemo, memo } from 'react'
import { Send, User, Loader2, Brain, MessageCircle, ChevronRight, Wrench, Square, Sparkles } from 'lucide-react'
import { marked } from 'marked'
import { api, type MessageEntry as ApiMessage, type ModelInfo, type TurnStats, type QueueState, type InnerVoiceObservation } from '../api'
import TodoList from './TodoList'
import ObservationBubble from './ObservationBubble'
import { actionStyle, parseObservationTime } from './innerVoiceStyles'

// Configure marked
marked.setOptions({ breaks: true, gfm: true })

// ------------ perf helpers ------------

const timeStr = (iso: string) => {
  return new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

type ToolCallRef = { name: string; args: string }

// Preserve per-message object identity across polling refreshes so memoized rows
// don't re-render when nothing actually changed. Compares by id + joined-text
// length + stats/reasoning shallow-ish equality.
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
        return p // preserve identity → row memo hit
      }
    }
    different = true
    return n
  })
  return different ? merged : prev
}

// ------------ memoized message row ------------

interface MessageRowProps {
  msg: ApiMessage
  showAgentDetails: boolean
  thinkLevel: string
  isMobile: boolean
  toolCallIndex: Map<string, ToolCallRef>
  forceLeftAlign?: boolean
}

const MessageRow = memo(function MessageRow({
  msg,
  showAgentDetails,
  thinkLevel,
  isMobile,
  toolCallIndex,
  forceLeftAlign = false,
}: MessageRowProps) {
  // Skip messages with empty content
  const hasContent = msg.content?.some(c => c.text?.trim())
  if (!hasContent) return null

  // Hide tool/subliminal messages when details hidden, but always show error messages
  const isError = msg.role === 'tool' && msg.content?.some(c => c.text?.startsWith('Error:'))
  const hideToolMessage = !showAgentDetails && msg.role === 'tool' && !isError
  const hideSubliminal = !showAgentDetails && msg.role === 'subliminal'
  if (hideToolMessage || hideSubliminal) return null

  const textJoined = useMemo(
    () => msg.content.map(c => c.text).join('\n'),
    [msg.content]
  )

  // Parse markdown once per text change, not every render
  const parsedHtml = useMemo(() => {
    if (msg.role === 'tool' || msg.role === 'subliminal') return '' // render as pre, no markdown
    return marked.parse(textJoined) as string
  }, [textJoined, msg.role])

  return (
    <div className={`flex gap-3 ${!forceLeftAlign && msg.role === 'user' ? 'justify-end' : ''}`}>
      {!forceLeftAlign && msg.role !== 'user' && !isMobile && (
        <div className="w-7 h-7 rounded-full flex-shrink-0 mt-0.5 overflow-hidden hidden sm:flex">
          {msg.role === 'tool'
            ? <div className="w-full h-full bg-slate-700 flex items-center justify-center"><Wrench className="w-3.5 h-3.5 text-slate-400" /></div>
            : msg.role === 'subliminal'
            ? <div className="w-full h-full bg-purple-900/40 flex items-center justify-center"><Sparkles className="w-3.5 h-3.5 text-purple-300" /></div>
            : <img src="/lloyd.jpg" alt="Lloyd" className="w-full h-full object-cover" />
          }
        </div>
      )}
      <div className={`${forceLeftAlign ? 'flex-1 min-w-0' : `max-w-[80%] ${msg.role === 'user' ? 'min-w-0' : 'flex-1 min-w-0'}`}`}>
        <div
          className={`rounded-xl ${
            msg.role === 'tool' || msg.role === 'subliminal' ? 'px-2.5 py-1.5' : 'px-3.5 py-2.5'
          } ${
            msg.role === 'user'
              ? 'bg-brand-600/30 border border-brand-500/40 text-white'
              : msg.role === 'tool'
              ? 'bg-slate-800/40 border border-slate-600/30 text-slate-200'
              : msg.role === 'subliminal'
              ? 'bg-purple-950/30 border border-purple-500/20 text-slate-200'
              : 'bg-surface-2 border border-surface-3/50 text-slate-200'
          }`}
        >
          <div className="prose-chat text-sm leading-relaxed">
            {msg.role === 'assistant' ? (
              <>
                {msg.reasoning && (showAgentDetails || thinkLevel !== 'off') && (
                  <details className="group mb-3">
                    <summary className="cursor-pointer list-none flex items-center gap-1 text-xs text-purple-400 hover:text-purple-300 transition-colors">
                      <Brain className="w-3 h-3" />
                      <ChevronRight className="w-3 h-3 group-open:rotate-90 transition-transform" />
                      <span className="font-semibold">Thinking</span>
                      <span className="text-slate-500 font-normal ml-1">({msg.reasoning.length.toLocaleString()} chars)</span>
                    </summary>
                    <div className="mt-2 p-3 bg-purple-900/10 border border-purple-500/10 rounded text-xs text-slate-300 whitespace-pre-wrap max-h-96 overflow-y-auto">
                      {msg.reasoning}
                    </div>
                  </details>
                )}
                <div dangerouslySetInnerHTML={{ __html: parsedHtml }} />
              </>
            ) : msg.role === 'tool' ? (() => {
              // Tool call rendering — single-line summary, click to expand args+response
              const tc = msg.tool_call_id ? toolCallIndex.get(msg.tool_call_id) : undefined
              const toolName = tc?.name || ''
              const toolArgs = tc?.args || '{}'

              let argsDisplay = toolArgs
              try {
                argsDisplay = JSON.stringify(JSON.parse(toolArgs), null, 2)
              } catch {
                // Keep raw string if not valid JSON
              }

              const responseText = textJoined
              return (
                <details className="group">
                  <summary className="cursor-pointer list-none flex items-center gap-1.5 text-xs text-slate-400 hover:text-slate-300 transition-colors">
                    <ChevronRight className="w-3 h-3 group-open:rotate-90 transition-transform shrink-0" />
                    <Wrench className="w-3 h-3 shrink-0" />
                    <span className="font-semibold uppercase tracking-wide">Tool</span>
                    {toolName && (
                      <span className="font-mono font-normal text-slate-500 truncate">{toolName}</span>
                    )}
                  </summary>
                  <div className="mt-2 space-y-2">
                    {argsDisplay !== '{}' && (
                      <div>
                        <div className="text-[10px] uppercase tracking-wide text-slate-500 mb-1">Arguments</div>
                        <pre className="p-2 bg-surface-3/30 rounded text-xs text-slate-300 overflow-x-auto whitespace-pre-wrap font-mono">
                          {argsDisplay}
                        </pre>
                      </div>
                    )}
                    <div>
                      <div className="text-[10px] uppercase tracking-wide text-slate-500 mb-1">Response</div>
                      <pre className="p-2 bg-surface-3/30 rounded text-xs text-slate-300 overflow-x-auto whitespace-pre-wrap font-mono max-h-48 overflow-y-auto">
                        {responseText || '⏳ Running...'}
                      </pre>
                    </div>
                  </div>
                </details>
              )
            })() : msg.role === 'subliminal' ? (() => {
              // Subliminal rendering — ephemeral context injected into the
              // SDK prompt but not part of the conversation. Mirrors the
              // tool renderer shape; metadata comes from msg.subliminal.
              const subl = msg.subliminal
              const kind = subl?.kind || 'other'
              const sources = subl?.sources || []
              const chars = subl?.chars ?? textJoined.length
              return (
                <details className="group">
                  <summary className="cursor-pointer list-none flex items-center gap-1.5 text-xs text-purple-400/80 hover:text-purple-300 transition-colors">
                    <ChevronRight className="w-3 h-3 group-open:rotate-90 transition-transform shrink-0" />
                    <Sparkles className="w-3 h-3 shrink-0" />
                    <span className="font-semibold uppercase tracking-wide">Subliminal</span>
                    <span className="font-mono font-normal text-purple-500/80">{kind}</span>
                    {sources.length > 0 && (
                      <span className="font-mono font-normal text-slate-500 truncate">
                        {sources.join(', ')}
                      </span>
                    )}
                    <span className="font-mono font-normal text-slate-600 ml-auto">
                      {chars.toLocaleString()} chars
                    </span>
                  </summary>
                  <div className="mt-2">
                    <div className="text-[10px] uppercase tracking-wide text-slate-500 mb-1">Injected context</div>
                    <pre className="p-2 bg-surface-3/30 rounded text-xs text-slate-300 overflow-x-auto whitespace-pre-wrap font-mono max-h-96 overflow-y-auto">
                      {textJoined}
                    </pre>
                  </div>
                </details>
              )
            })() : (
              <div dangerouslySetInnerHTML={{ __html: parsedHtml }} />
            )}
          </div>
        </div>
        <div className="mt-1.5 text-[10px] text-slate-600 font-mono flex flex-wrap items-center gap-x-2.5 gap-y-0.5">
          <span>{timeStr(msg.timestamp)}</span>
          {msg.stats && msg.role === 'assistant' && (() => {
            const s = msg.stats as TurnStats
            const peak = s.peak_input_tokens ?? s.input_tokens
            const pct = (peak / 262144 * 100).toFixed(1)
            return (<>
              <span className="text-slate-600">ctx: {peak.toLocaleString()} ({pct}%)</span>
              {s.cache_read > 0 && (
                <span className="text-emerald-700">cache↑: {s.cache_read.toLocaleString()}</span>
              )}
              {s.cache_create > 0 && (
                <span className="text-amber-700">cache✎: {s.cache_create.toLocaleString()}</span>
              )}
              {s.duration_ms != null && (
                <span className="text-slate-600">time: {(s.duration_ms / 1000).toFixed(1)}s</span>
              )}
              {s.num_turns != null && s.num_turns > 1 && (
                <span className="text-slate-600">turns: {s.num_turns}</span>
              )}
            </>)
          })()}
          {msg.context_tokens != null && msg.context_tokens > 0 && msg.role === 'tool' && (() => {
            const pct = (msg.context_tokens / 262144 * 100).toFixed(1)
            return <span className="text-slate-600">ctx: {msg.context_tokens.toLocaleString()} ({pct}%)</span>
          })()}
        </div>
      </div>
      {!forceLeftAlign && msg.role === 'user' && !isMobile && (
        <div className="w-7 h-7 rounded-full bg-slate-700 flex items-center justify-center flex-shrink-0 mt-0.5 hidden sm:flex">
          <User className="w-3.5 h-3.5 text-slate-300" />
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
  // When provided, switch to centerline-timeline mode: primary actions on
  // the left of a vertical line, IV observations on the right, ordered
  // chronologically. Used by the Inner Voice page.
  timelineRight?: InnerVoiceObservation[]
}

// Mock slash commands (will be replaced with actual backend fetch)
const SLASH_COMMANDS = [
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
  { name: 'think', desc: 'Toggle extended thinking (off/low/medium/high/xhigh/max)' },
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
}: ChatPanelProps = {}) {
  const [sessionKey, setSessionKey] = useState<string | null>(null)
  const [messages, setMessages] = useState<ApiMessage[]>([])
  const [input, setInput] = useState('')
  const [thinking, setThinking] = useState(false)
  const [sending, setSending] = useState(false)
  const [showCommands, setShowCommands] = useState(false)
  const [commandFilter, setCommandFilter] = useState('')
  const [selectedCommandIndex, setSelectedCommandIndex] = useState(0)
  const [models, setModels] = useState<ModelInfo[]>([])
  const [activeToolName, setActiveToolName] = useState<string | null>(null)
  const [thinkLevel, setThinkLevel] = useState<string>(() => localStorage.getItem('mc_think_level') || 'off')
  const [queueState, setQueueState] = useState<QueueState | null>(null)
  // Bumped to force <TodoList> to re-fetch from /api/sessions/<id>/todos.
  // Bumped on session change (initial load) and on every TodoWrite tool
  // result so the panel reflects the model's latest checklist.
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

  // Load models
  useEffect(() => {
    api.getModels().then(result => {
      if (result.models) {
        setModels(result.models)
      }
    }).catch(err => {
      console.warn('Failed to load models:', err)
    })
  }, [])

  // Load messages when currentSessionKey changes (from parent)
  useEffect(() => {
    const activeKey = currentSessionKey || requestedSessionKey
    if (activeKey) {
      loadMessages(activeKey, onSessionLoaded)
    } else if (!activeKey && messages.length > 0) {
      // Clear messages if no session is active
      setMessages([])
      setSessionKey(null)
    }
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

  // Notify parent of thinking/tool state changes
  useEffect(() => {
    onThinkingChange?.(thinking, thinking ? activeToolName : null)
  }, [thinking, activeToolName])

  // Poll messages for current session (skip while streaming or sending, or hidden idle)
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

  // Poll status + messages for restored streaming sessions (no local AbortController).
  // Detects when the backend finishes so the cancel button goes away and messages refresh.
  useEffect(() => {
    if (!sessionKey || !thinking || abortControllerRef.current) return
    const poll = async () => {
      try {
        const status = await api.getSessionStatus(sessionKey)
        if (!status.streaming) {
          setThinking(false)
          setSending(false)
          // Refresh messages to pick up final response
          const result = await api.loadMessages(sessionKey)
          if (result.messages) {
            const next = result.messages as ApiMessage[]
            setMessages(prev => mergeMessages(prev, next))
          }
        } else {
          // Still streaming — refresh messages so new tool calls / text appear
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

  // Scroll to bottom when messages change. Use 'auto' (instant) during streaming
  // so smooth-scroll animations don't pile up and jank the main thread.
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
        setSessionKey(key) // Update local state so polling works
        onActiveSessionChange?.(key) // Notify parent of active session
        if (result.model) onModelSwitch?.(result.model) // Sync model dropdown to session's model
        onLoaded?.() // Clear requestedSessionKey only after activeSessionKey is set
        // Check if this session is still actively streaming on the backend
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
          // Status endpoint unavailable — assume idle
          setThinking(false)
          setSending(false)
        }
        // Scroll to bottom after loading
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
    
    // Check if we have a complete model switch command like "/model primary"
    const match = value.match(/^\/(model|switch)\s+(\w+)$/)
    if (match) {
      // Auto-execute the model switch
      const modelAlias = match[2]
      const targetModel = models.find(m => 
        m.alias === modelAlias || m.name === modelAlias
      )
      if (targetModel) {
        handleModelSwitch(targetModel.name)
        setInput('')
        setShowCommands(false)
        return
      }
    }
    
    // Show command dropdown when input starts with /
    if (value.startsWith('/')) {
      const filter = value.slice(1).toLowerCase()
      setCommandFilter(filter)
      setShowCommands(true)
    } else {
      setShowCommands(false)
    }
  }

  const handleModelSwitch = async (modelName: string) => {
    // Use the session key from props (always up-to-date from parent)
    const activeSession = currentSessionKey
    if (!activeSession) {
      // No active session, just update global config
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
      // Switch model for current session
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
    // Special handling for /model command
    if (cmd === 'model') {
      // Show models in chat
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

  // Build tool_call_id → { name, args } index once per messages change so
  // memoized tool rows can look up their call info in O(1) instead of scanning
  // the whole message array per render.
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

  // Chronological merge of primary messages and IV observations. Returns
  // null when the panel is not in timeline mode (preserving the default
  // right-aligned chat rendering for the regular Chat page).
  const timeline = useMemo<TimelineItem[] | null>(() => {
    if (!timelineRight) return null
    const items: TimelineItem[] = [
      ...messages.map(m => ({ kind: 'msg' as const, ts: Date.parse(m.timestamp), key: m.id, msg: m })),
      ...timelineRight.map(o => ({ kind: 'obs' as const, ts: parseObservationTime(o.created_at), key: `obs_${o.id}`, obs: o })),
    ]
    items.sort((a, b) => a.ts - b.ts)
    return items
  }, [messages, timelineRight])

  const filteredCommands = useMemo(() => {
    if (!showCommands || !commandFilter) {
      setSelectedCommandIndex(0)
      return SLASH_COMMANDS.slice(0, 8)
    }
    // Check if we're typing a model switch command like "/model primary"
    if (commandFilter.startsWith('model ') || commandFilter.startsWith('switch ')) {
      // Show model names as completions
      const modelArg = commandFilter.split(' ')[1] || ''
      return models.filter(m => 
        m.alias.includes(modelArg) || m.name.includes(modelArg)
      ).map(m => ({
        name: `model ${m.alias}`,
        desc: m.name,
        alias: m.alias
      })).slice(0, 8)
    }
    const filtered = SLASH_COMMANDS.filter(cmd => 
      cmd.name.includes(commandFilter) || 
      (cmd.alias && cmd.alias.includes(commandFilter))
    ).slice(0, 8)
    setSelectedCommandIndex(0)
    return filtered
  }, [showCommands, commandFilter, models])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    const text = input.trim()
    if (!text || sending || thinking) return

    // Handle /new command locally
    if (text === '/new' || text === '/reset') {
      setMessages([])
      setInput('')
      setSending(false)
      setThinking(false)
      // Also clear session key to force new session on next message
      setSessionKey(null)
      onActiveSessionChange?.(null)
      localStorage.removeItem('mc_session_id')
      return
    }

    // Handle /think command locally
    if (text.startsWith('/think')) {
      const arg = text.split(/\s+/)[1]?.toLowerCase() || ''
      const validLevels = ['off', 'low', 'medium', 'high', 'xhigh', 'max']
      let newLevel: string
      if (arg && validLevels.includes(arg)) {
        newLevel = arg
      } else if (!arg) {
        // Toggle: off → high, anything else → off
        newLevel = thinkLevel === 'off' ? 'high' : 'off'
      } else {
        setMessages(prev => [...prev, {
          id: `msg_${Date.now()}_think`,
          role: 'assistant',
          content: [{ type: 'text', text: `Invalid think level **"${arg}"**. Valid: off, low, medium, high, xhigh, max` }],
          timestamp: new Date().toISOString(),
        }])
        setInput('')
        return
      }
      setThinkLevel(newLevel)
      localStorage.setItem('mc_think_level', newLevel)
      setMessages(prev => [...prev, {
        id: `msg_${Date.now()}_think`,
        role: 'assistant',
        content: [{ type: 'text', text: newLevel === 'off' ? '🧠 Extended thinking **off**' : `🧠 Extended thinking set to **${newLevel}**` }],
        timestamp: new Date().toISOString(),
      }])
      setInput('')
      return
    }

    setInput('')
    setSending(true)

    // Add user message
    setMessages(prev => [...prev, {
      id: `msg_${Date.now()}`,
      role: 'user',
      content: [{ type: 'text', text }],
      timestamp: new Date().toISOString(),
    }])

    setThinking(true)

    // Current streaming assistant bubble — reset on each tool_start so text
    // emitted between tool calls lands in its own bubble.
    let assistantMsgId: string | null = null
    let segmentCounter = 0
    let streamingStarted = false
    let settled = false
    let accumulatedThinking = ''

    // Coalesce text deltas — per-token setState was O(n) per message array and
    // killed the main thread on long sessions. Buffer deltas and flush once per
    // animation frame instead.
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
      onQueueState: (state) => {
        setQueueState(state)
      },
      onToolStart: (callId, name, args, contextTokens) => {
        setActiveToolName(name)
        // Close the current text segment so the next text_delta starts a new bubble.
        assistantMsgId = null
        accumulatedThinking = ''
        // Add an assistant message with the tool_call, then a pending tool result message
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
        if (_name === 'TodoWrite') setTodoRefreshKey(k => k + 1)
      },
      onThinkingDelta: (delta) => {
        accumulatedThinking += delta
      },
      onThinkingDone: (fullText) => {
        // Finalize thinking — use full text from server (more reliable than accumulated deltas)
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
        // Cancel any queued delta frame and merge its payload into the final update
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
    }, !sessionKey ? pendingModel : undefined, thinkLevel !== 'off' ? thinkLevel : undefined)
    abortControllerRef.current = controller
  }

  const thinkingIndicatorBody = (
    <div className="bg-surface-2 border border-surface-3/50 px-3.5 py-2.5 rounded-xl">
      <div className="flex items-center gap-2 text-sm text-slate-400">
        <Loader2 className="w-4 h-4 animate-spin text-brand-400" />
        {queueState?.current?.source === 'ambient'
          ? <span><span className="text-amber-400">Ambient context</span> — Lloyd is processing background input...</span>
          : activeToolName
            ? <span>Working: <span className="font-mono text-brand-300">{activeToolName}</span>...</span>
            : <span>Thinking...</span>
        }
        {queueState && queueState.depth > 0 && (
          <span className="ml-2 text-xs text-slate-500 font-mono">
            (queue: {queueState.pending_user}u + {queueState.pending_ambient}a)
          </span>
        )}
      </div>
    </div>
  )

  return (
    <div className="flex flex-col h-full">
      {/* Messages */}
      <main ref={messagesContainerRef} className={`flex-1 overflow-y-auto p-4 ${timeline === null ? 'space-y-4' : ''}`}>
        {messages.length === 0 && (
          <div className="flex items-center justify-center h-full">
            <div className="text-center text-slate-500">
              <MessageCircle className="w-10 h-10 mx-auto mb-3 opacity-30" />
              <p className="text-sm">Welcome to Lloyd Mission Control</p>
              <p className="text-xs mt-1 text-slate-600">Send a message to get started</p>
            </div>
          </div>
        )}

        {timeline === null && messages.map((msg) => (
          <MessageRow
            key={msg.id}
            msg={msg}
            showAgentDetails={showAgentDetails}
            thinkLevel={thinkLevel}
            isMobile={isMobile}
            toolCallIndex={toolCallIndex}
          />
        ))}

        {timeline === null && thinking && (
          <div className="flex gap-3">
            <div className="w-7 h-7 rounded-full flex-shrink-0 mt-0.5 overflow-hidden hidden sm:flex">
              <img src="/lloyd.jpg" alt="Lloyd" className="w-full h-full object-cover" />
            </div>
            {thinkingIndicatorBody}
          </div>
        )}

        {timeline !== null && (
          <div className="flex flex-col">
            {timeline.map(item => {
              // Mirror MessageRow's visibility rules — drop items that would
              // render nothing so the centerline doesn't sprout orphan dots.
              // The most common case: assistant rows that carry tool_calls
              // but no text (the tool result row alongside renders the bubble).
              if (item.kind === 'msg') {
                const m = item.msg
                const hasContent = m.content?.some(c => c.text?.trim())
                if (!hasContent) return null
                const isError = m.role === 'tool' && m.content?.some(c => c.text?.startsWith('Error:'))
                if (!showAgentDetails && m.role === 'tool' && !isError) return null
                if (!showAgentDetails && m.role === 'subliminal') return null
              }
              return (
              <div key={item.key} className="grid grid-cols-[1fr_28px_1fr] gap-x-3 py-1.5">
                {/* Left half — primary actions */}
                <div className="flex justify-end min-w-0">
                  {item.kind === 'msg' && (
                    <div className="w-full max-w-[94%] min-w-0">
                      <MessageRow
                        msg={item.msg}
                        showAgentDetails={showAgentDetails}
                        thinkLevel={thinkLevel}
                        isMobile={isMobile}
                        toolCallIndex={toolCallIndex}
                        forceLeftAlign
                      />
                    </div>
                  )}
                </div>
                {/* Centerline column with dot */}
                <div className="relative flex justify-center">
                  <div className="absolute inset-y-0 left-1/2 -translate-x-1/2 border-l border-surface-3/40" />
                  <div className={`relative mt-3 w-2 h-2 rounded-full ring-2 ring-surface-1 ${
                    item.kind === 'obs'
                      ? (item.obs.trigger === 'result' && item.obs.action === 'noop'
                          ? 'bg-emerald-400'
                          : actionStyle(item.obs.action).dot)
                      : 'bg-brand-400'
                  }`} />
                </div>
                {/* Right half — IV observations */}
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
                  <div className="absolute inset-y-0 left-1/2 -translate-x-1/2 border-l border-surface-3/40" />
                  <div className="relative mt-3 w-2 h-2 rounded-full ring-2 ring-surface-1 bg-brand-400 animate-pulse" />
                </div>
                <div />
              </div>
            )}
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </main>

      <TodoList sessionId={sessionKey} refreshKey={todoRefreshKey} />

      {/* Input */}
      <footer className="p-3 border-t border-surface-3/50 relative">
        <form onSubmit={handleSubmit} className="flex gap-2 items-center">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={handleInputChange}
            onKeyDown={(e) => {
              if (!showCommands) return

              if (e.key === 'ArrowDown') {
                e.preventDefault()
                setSelectedCommandIndex((prev) =>
                  prev < filteredCommands.length - 1 ? prev + 1 : prev
                )
              } else if (e.key === 'ArrowUp') {
                e.preventDefault()
                setSelectedCommandIndex((prev) => prev > 0 ? prev - 1 : 0)
              } else if (e.key === 'Enter') {
                e.preventDefault()
                if (filteredCommands.length > 0) {
                  handleCommandSelect(filteredCommands[selectedCommandIndex].name)
                }
              } else if (e.key === 'Escape') {
                setShowCommands(false)
              }
            }}
            placeholder={thinkLevel !== 'off' ? `Talk to Lloyd... (thinking: ${thinkLevel})` : 'Talk to Lloyd... (use / for commands)'}
            className="flex-1 bg-surface-2 text-sm text-slate-200 rounded-lg px-3.5 py-2.5 border border-surface-3/50 outline-none focus:border-brand-500/50 placeholder:text-slate-500 transition-colors disabled:opacity-50"
            disabled={sending || thinking}
          />
          {/* Think level toggle — always visible, cycles off→low→medium→high→xhigh→max→off */}
          {(() => {
            const levels = ['off', 'low', 'medium', 'high', 'xhigh', 'max'] as const
            const idx = levels.indexOf(thinkLevel as typeof levels[number])
            const nextLevel = levels[(idx + 1) % levels.length]
            const isActive = thinkLevel !== 'off'
            return (
              <button
                type="button"
                onClick={() => {
                  setThinkLevel(nextLevel)
                  localStorage.setItem('mc_think_level', nextLevel)
                }}
                className={`w-[38px] h-[38px] flex items-center justify-center rounded-lg shrink-0 transition-colors ${
                  isActive
                    ? 'bg-purple-600/20 border border-purple-500/30 text-purple-400 hover:bg-purple-600/30'
                    : 'bg-surface-2 border border-surface-3/50 text-slate-500 hover:text-slate-400 hover:bg-surface-3/30'
                }`}
                title={`Extended thinking: ${thinkLevel} (click → ${nextLevel})`}
              >
                <Brain className="w-4 h-4" />
              </button>
            )
          })()}
          {/* Submit / Cancel */}
          {(sending || thinking) ? (
            <button
              type="button"
              onClick={() => {
                if (abortControllerRef.current) {
                  // Local stream — abort the fetch directly. Also call the
                  // backend cancel API so the harness sees cancel_event and
                  // Inner Voice stops observing/injecting; drain any queued
                  // ambient turns IV may have enqueued for the cancelled work.
                  abortControllerRef.current.abort()
                  if (sessionKey) {
                    api.cancelSession(sessionKey, { drainPending: true }).catch(() => {})
                  }
                } else if (sessionKey) {
                  // Restored session with no local stream — cancel via API
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
                    // Cancel failed — force reset UI state anyway
                    setThinking(false)
                    setSending(false)
                  })
                }
              }}
              className="w-[38px] h-[38px] flex items-center justify-center bg-red-600/20 hover:bg-red-600/30 border border-red-500/40 text-red-400 rounded-lg shrink-0 transition-colors"
              title="Stop"
            >
              <Square className="w-4 h-4" />
            </button>
          ) : (
            <button
              type="submit"
              disabled={!input.trim()}
              className="w-[38px] h-[38px] flex items-center justify-center bg-brand-600 hover:bg-brand-500 disabled:opacity-40 disabled:cursor-not-allowed text-white rounded-lg shrink-0 transition-colors"
            >
              <Send className="w-4 h-4" />
            </button>
          )}
        </form>
        
        {/* Command dropdown */}
        {showCommands && filteredCommands.length > 0 && (
          <div className="absolute bottom-full left-0 right-0 mb-1 bg-surface-1 border border-surface-3/50 rounded-lg shadow-lg max-h-60 overflow-y-auto z-50">
            {filteredCommands.map((cmd, idx) => (
              <button
                key={cmd.name}
                onClick={() => handleCommandSelect(cmd.name)}
                className={`w-full text-left px-3 py-2 text-sm transition-colors flex items-center justify-between ${
                  idx === selectedCommandIndex 
                    ? 'bg-brand-500/20 text-brand-400' 
                    : 'hover:bg-surface-2 text-slate-200'
                }`}
              >
                <div className="flex items-center gap-2">
                  <span className="font-mono text-brand-400">/{cmd.name}</span>
                  {cmd.alias && <span className="text-xs text-slate-500">({cmd.alias.split(' ')[0]})</span>}
                </div>
                <span className="text-xs text-slate-400 truncate max-w-[200px]">{cmd.desc}</span>
              </button>
            ))}
          </div>
        )}
      </footer>
      {/* Session list moved to sidebar SessionsPanel */}
    </div>
  )
}
