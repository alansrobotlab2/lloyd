import { useEffect, useRef, useState, useCallback, useMemo } from "react";
import { Send, User, Plus, Loader2, Brain, MessageCircle, StopCircle, Wrench } from "lucide-react";
import { marked } from "marked";
import { api, type MessageEntry, type CommandInfo } from "../api";
import SlashCommandPicker from "./SlashCommandPicker";

// Configure marked for safe, sane defaults
marked.setOptions({ breaks: true, gfm: true });

function extractText(content: Array<{ type: string; text?: string }>): string {
  return content
    .filter((c) => c.type === "text" && c.text)
    .map((c) => c.text!)
    .join("\n")
    .replace(/<summary>[\s\S]*?<\/summary>/g, "")
    .trim();
}

/** Detect pure system-injected messages with no real user text */
function isContextMessage(msg: MessageEntry): boolean {
  if (msg.role !== "user") return false;
  const text = extractText(msg.content);
  if (/^\[cron:/.test(text)) return true;
  // Strip all gateway-injected wrappers — if nothing remains, it's pure context
  const stripped = stripInjectedContext(text);
  return stripped.length === 0;
}

function isSystemMessage(msg: MessageEntry): boolean {
  if (msg.role !== "user") return false;
  const text = extractText(msg.content);
  return /^\[System Message\]/.test(text);
}

function timeStr(ts: string): string {
  return new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

/** Strip all gateway-injected context wrappers to find the real user text */
function stripInjectedContext(text: string): string {
  return text
    .replace(/<daily_notes>[\s\S]*?<\/daily_notes>\s*/i, "")
    .replace(/<active_mode>\w*<\/active_mode>\s*/i, "")
    .replace(/<memory_context>[\s\S]*?<\/memory_context>\s*/i, "")
    .replace(/\[(?:[A-Z][a-z]{2} )?\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?[^\]]*\]\s*/, "")
    .replace(/A new session was started via\b[\s\S]*/i, "")
    .trim();
}

/** Strip injected prefixes from user messages for display */
function stripDatetimePrefix(text: string): string {
  return stripInjectedContext(text);
}

interface ChatPanelProps {
  requestedSessionId?: string | null;
  onSessionLoaded?: () => void;
}

export default function ChatPanel({ requestedSessionId, onSessionLoaded }: ChatPanelProps = {}) {
  const [messages, setMessages] = useState<MessageEntry[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sessions, setSessions] = useState<Array<{ sessionId: string; lastActivity: string; model: string; summary?: string }>>([]);
  const [awaitingReset, setAwaitingReset] = useState(false);
  // Show the "Thinking..." bubble while waiting for an assistant reply
  const [thinking, setThinking] = useState(false);
  // Granular activity state while thinking: "thinking" (LLM reasoning) vs "working" (tool call)
  const [activityType, setActivityType] = useState<"thinking" | "working" | null>(null);
  const [activityDetail, setActivityDetail] = useState<string | null>(null);
  const [showToolCalls, setShowToolCalls] = useState(() => {
    try { return localStorage.getItem("mc-agent-details-visible") === "true"; } catch { return false; }
  });
  const [expandedThinking, setExpandedThinking] = useState<Record<string, boolean>>({});
  // Slash command picker state
  const [commandsList, setCommandsList] = useState<CommandInfo[]>([]);
  const [showPicker, setShowPicker] = useState(false);
  const [pickerIndex, setPickerIndex] = useState(0);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const isNearBottom = useRef(true);
  const initialLoadDone = useRef(false);
  // Track the timestamp of the last user-sent message so we can detect new replies
  const lastUserSendTs = useRef(0);
  const scrollToBottomOnLoad = useRef(true);
  const thinkingRestored = useRef(false);
  const workingStartRef = useRef<number>(0);
  const workingTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isWorkingRef = useRef(false);

  // Derive current model and thinking status from available state
  const currentModel = useMemo(() => {
    // Prefer the most recent assistant message's model (most current)
    const lastAssistant = [...messages].reverse().find((m) => m.role === "assistant" && m.model);
    if (lastAssistant?.model) return lastAssistant.model;
    // Fall back to session-level model from sessions list
    return sessions.find((s) => s.sessionId === sessionId)?.model || null;
  }, [messages, sessions, sessionId]);

  const thinkingEnabled = useMemo(() => {
    const lastAssistant = [...messages].reverse().find((m) => m.role === "assistant");
    return lastAssistant?.hasThinking ?? false;
  }, [messages]);

  // Switch to a session requested by another page (e.g. Sessions table)
  useEffect(() => {
    if (requestedSessionId && requestedSessionId !== sessionId) {
      initialLoadDone.current = true; // prevent initial-load from overriding
      setSessionId(requestedSessionId);
      setThinking(false);
      setAwaitingReset(false);
      onSessionLoaded?.();
    }
  }, [requestedSessionId]);

  const refreshSessions = useCallback(() => {
    return api.sessions().then((d) => {
      setSessions(d.sessions);
      return d.sessions;
    }).catch((err) => {
      console.error(err);
      return [] as Array<{ sessionId: string; lastActivity: string; model: string; summary?: string }>;
    });
  }, []);

  // Initial load — auto-select most recent session
  useEffect(() => {
    refreshSessions().then((list) => {
      if (!initialLoadDone.current && list.length > 0) {
        initialLoadDone.current = true;
        setSessionId(list[0].sessionId);
      }
    });
  }, [refreshSessions]);

  // Reset scroll-to-bottom and thinking-restore flags when session changes
  useEffect(() => {
    scrollToBottomOnLoad.current = true;
    thinkingRestored.current = false;
  }, [sessionId]);

  // Fetch slash commands on mount
  useEffect(() => {
    api.commands().then((d) => setCommandsList(d.commands)).catch(() => {});
  }, []);

  // Compute filter text and filtered command list for picker
  const slashFilter = showPicker && input.startsWith("/") ? input.slice(1).split(" ")[0] : "";
  const filteredCommands = useMemo(() => {
    if (!showPicker) return [];
    return commandsList.filter((cmd) => {
      if (!slashFilter) return true;
      const f = slashFilter.toLowerCase();
      return cmd.name.toLowerCase().includes(f) || cmd.description.toLowerCase().includes(f);
    });
  }, [commandsList, slashFilter, showPicker]);

  // Poll messages for the selected session
  useEffect(() => {
    if (!sessionId) return;
    const load = () =>
      api.sessionMessages(sessionId, showToolCalls)
        .then((d) => {
          // Don't replace local messages with fewer server messages while
          // thinking — protects the optimistic user message from flickering
          // away before the server has written it to disk.
          setMessages((prev) => {
            if (thinking && d.messages.length < prev.length) return prev;
            return d.messages;
          });

          // Restore thinking indicator on re-mount if agent is still processing
          if (!thinking && !thinkingRestored.current && d.messages.length > 0) {
            thinkingRestored.current = true;
            const lastMsg = d.messages[d.messages.length - 1];
            if (lastMsg.role === "user") {
              api.agentStatus().then((status) => {
                if (status.mainAgent.state !== "idle") {
                  lastUserSendTs.current = new Date(lastMsg.timestamp).getTime();
                  setThinking(true);
                }
              }).catch(() => {});
            }
          }

          // Clear awaitingReset once any messages appear in the new session
          if (d.messages.length > 0 && awaitingReset) {
            setAwaitingReset(false);
            refreshSessions();
          }

          // Clear thinking indicator once the agent is idle AND we have
          // a new assistant message with real text content. This keeps the
          // bubble visible during multi-response processing (tool calls
          // between text replies). Only check agent status when the newest
          // assistant message has text — intermediate messages with only
          // tool call blocks mean the agent is still mid-turn.
          if (thinking && lastUserSendTs.current > 0) {
            const postSendAssistant = d.messages.filter(
              (m) => m.role === "assistant" && new Date(m.timestamp).getTime() > lastUserSendTs.current,
            );
            if (postSendAssistant.length > 0) {
              const newest = postSendAssistant[postSendAssistant.length - 1];
              const hasTextContent = newest.content?.some(
                (c) => c.type === "text" && c.text && c.text.trim().length > 0,
              );
              if (hasTextContent) {
                api.agentStatus().then((status) => {
                  if (status.mainAgent.state === "idle") {
                    setThinking(false);
                  }
                }).catch(() => {
                  // Can't reach agent-status — fall back to clearing immediately
                  setThinking(false);
                });
              }
            }
          }
        })
        .catch(() => {
          // 404 is expected while the session file doesn't exist yet (after /new).
          // Clear stale messages so old session content doesn't linger.
          setMessages([]);
        });
    load();
    const interval = setInterval(load, thinking || awaitingReset ? 1_500 : 3_000);
    return () => clearInterval(interval);
  }, [sessionId, awaitingReset, thinking, refreshSessions, showToolCalls]);

  // Track whether user is scrolled near the bottom
  const handleScroll = useCallback(() => {
    const el = messagesContainerRef.current;
    if (!el) return;
    const threshold = 80;
    isNearBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < threshold;
  }, []);

  // Auto-scroll only when user is near the bottom
  useEffect(() => {
    if (scrollToBottomOnLoad.current && messages.length > 0) {
      scrollToBottomOnLoad.current = false;
      const el = messagesContainerRef.current;
      if (el) el.scrollTop = el.scrollHeight;
    } else if (isNearBottom.current) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, thinking]);

  // Poll agent activity type while thinking is active
  useEffect(() => {
    if (!thinking) {
      setActivityType(null);
      setActivityDetail(null);
      if (workingTimeoutRef.current) {
        clearTimeout(workingTimeoutRef.current);
        workingTimeoutRef.current = null;
      }
      isWorkingRef.current = false;
      return;
    }
    const poll = () => {
      api.agentStatus().then((status) => {
        if (status.activity.type === "tool_call") {
          if (!isWorkingRef.current) {
            isWorkingRef.current = true;
            workingStartRef.current = Date.now();
          }
          // Cancel any pending transition away from working
          if (workingTimeoutRef.current) {
            clearTimeout(workingTimeoutRef.current);
            workingTimeoutRef.current = null;
          }
          setActivityType("working");
          setActivityDetail(status.activity.detail);
        } else {
          if (isWorkingRef.current) {
            const elapsed = Date.now() - workingStartRef.current;
            const remaining = 1000 - elapsed;
            if (remaining > 0 && !workingTimeoutRef.current) {
              // Hold the "working" display until 1s minimum is met
              const capturedType = status.activity.type;
              const capturedDetail = status.activity.detail;
              workingTimeoutRef.current = setTimeout(() => {
                workingTimeoutRef.current = null;
                isWorkingRef.current = false;
                if (capturedType === "llm_thinking") {
                  setActivityType("thinking");
                  setActivityDetail(capturedDetail);
                } else {
                  setActivityType(null);
                  setActivityDetail(null);
                }
              }, remaining);
              return; // Keep displaying "working" until timeout fires
            }
            isWorkingRef.current = false;
          }
          if (status.activity.type === "llm_thinking") {
            setActivityType("thinking");
            setActivityDetail(status.activity.detail);
          }
          // idle case is handled by the existing code that clears `thinking`
        }
      }).catch(() => {});
    };
    poll();
    const interval = setInterval(poll, 500);
    return () => {
      clearInterval(interval);
      if (workingTimeoutRef.current) {
        clearTimeout(workingTimeoutRef.current);
        workingTimeoutRef.current = null;
      }
    };
  }, [thinking]);

  const handleSend = async () => {
    if (!input.trim() || sending) return;
    const text = input.trim();

    setSending(true);
    setInput("");

    // Optimistic user message
    const optimisticMsg: MessageEntry = {
      id: `pending-${Date.now()}`,
      timestamp: new Date().toISOString(),
      role: "user",
      content: [{ type: "text", text }],
    };
    setMessages((prev) => [...prev, optimisticMsg]);
    lastUserSendTs.current = Date.now();

    try {
      await api.chat(text);
      // Show thinking indicator while agent processes
      setThinking(true);
    } catch (err) {
      console.error("Chat send failed:", err);
      setMessages((prev) => prev.filter((m) => m.id !== optimisticMsg.id));
      setInput(text);
    } finally {
      setSending(false);
    }
  };


  const handleStop = async () => {
    try {
      await api.chatAbort();
      setThinking(false);
    } catch (err) {
      console.error("Stop failed:", err);
    }
  };

  const handleNew = async () => {
    if (sending || awaitingReset) return;

    setMessages([]);
    setSessionId(null);        // halt polling for old session immediately
    setAwaitingReset(true);
    setThinking(false);
    setSending(true);

    try {
      const result = await api.sessionReset();
      if (result.sessionId) {
        setSessionId(result.sessionId);
        // Optimistic dropdown entry — replaced by real data once messages appear
        setSessions((prev) => [
          { sessionId: result.sessionId!, lastActivity: new Date().toISOString(), model: "", summary: "New session" },
          ...prev,
        ]);
        // Show thinking bubble while waiting for the greeting response
        lastUserSendTs.current = Date.now();
        setThinking(true);
      }
    } catch (err) {
      console.error("New chat failed:", err);
      if (sessions.length > 0) {
        setSessionId(sessions[0].sessionId);
      }
    } finally {
      setSending(false);
      setAwaitingReset(false);
    }
  };

  return (
    <div className="flex flex-col h-full p-6 space-y-4 overflow-hidden">
      {/* Page header — matches other tabs */}
      <div className="flex items-center justify-between flex-shrink-0">
        <div className="flex items-center gap-3">
          <MessageCircle className="w-5 h-5 text-brand-400" />
          <h2 className="text-lg font-semibold">Chat</h2>
          <select
            value={sessionId || ""}
            onChange={(e) => {
              const val = e.target.value;
              if (!val) return;
              setSessionId(val);
              setAwaitingReset(false);
              setThinking(false);
            }}
            className="text-xs text-slate-400 bg-transparent border-none outline-none cursor-pointer"
          >
            {awaitingReset && (
              <option value={sessionId || ""} className="bg-surface-2">
                New session...
              </option>
            )}
            {sessions.map((s) => (
              <option key={s.sessionId} value={s.sessionId} className="bg-surface-2">
                {s.summary || s.sessionId.slice(0, 8) + "..."} ({s.model || "unknown"})
              </option>
            ))}
          </select>
          {/* Persistent status: model + thinking */}
          {(currentModel || messages.length > 0) && (
            <div className="flex items-center gap-2 ml-2 pl-2 border-l border-surface-3/50">
              {currentModel && (
                <span className="text-[11px] text-slate-500 font-mono tracking-tight">
                  {currentModel}
                </span>
              )}
              <span
                className={`flex items-center gap-1 text-[11px] font-medium ${
                  thinkingEnabled ? "text-purple-400" : "text-slate-600"
                }`}
                title={thinkingEnabled ? "Extended thinking is active" : "Extended thinking is off"}
              >
                <Brain className="w-3 h-3" />
                {thinkingEnabled ? "On" : "Off"}
              </span>
            </div>
          )}
        </div>
        <div className="flex items-center gap-1.5">
          <button
            onClick={() => setShowToolCalls((v) => {
              const next = !v;
              try { localStorage.setItem("mc-agent-details-visible", String(next)); } catch {}
              return next;
            })}
            title={showToolCalls ? "Hide agent details" : "Show agent details"}
            className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium transition-colors ${
              showToolCalls
                ? "text-brand-400 bg-brand-500/10"
                : "text-slate-400 hover:text-brand-400 hover:bg-brand-500/10"
            }`}
          >
            <Brain className="w-3.5 h-3.5" />
            Agent Details
          </button>
          <button
            onClick={handleNew}
            disabled={sending || awaitingReset}
            title="New conversation"
            className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium text-slate-400 hover:text-brand-400 hover:bg-brand-500/10 transition-colors disabled:opacity-40"
          >
            <Plus className="w-3.5 h-3.5" />
            New
          </button>
        </div>
      </div>

      {/* Messages card */}
      <div className="flex-1 flex flex-col min-h-0 bg-surface-1 rounded-xl border border-surface-3/50 overflow-hidden">
        <div ref={messagesContainerRef} onScroll={handleScroll} className="flex-1 overflow-y-auto p-4 space-y-4 min-h-0">
          {messages.length === 0 && !thinking && (
            <div className="flex flex-col items-center justify-center h-full text-slate-500">
              <img src="/api/mc/agent-avatar?id=lloyd" alt="Lloyd" className="w-12 h-12 rounded-full object-cover mb-3 opacity-50" />
              {awaitingReset ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin mb-2 text-brand-400" />
                  <p className="text-sm">Starting new session...</p>
                </>
              ) : (
                <p className="text-sm">No messages yet</p>
              )}
            </div>
          )}
          {messages
            .filter((msg) => {
              if (!showToolCalls) {
                if (msg.role === "toolResult") return false;
                if (msg.role === "assistant" && !msg.content?.some((c) => c.type === "text" && c.text)) return false;
                if (isContextMessage(msg)) return false;
                if (isSystemMessage(msg)) return false;
              }
              return true;
            })
            .map((msg) => {
              // Tool result messages — compact inline row
              if (msg.role === "toolResult") {
                const preview = extractText(msg.content);
                return (
                  <div key={msg.id} className="flex gap-3">
                    <div className="w-7 flex-shrink-0" /> {/* spacer to align under assistant */}
                    <div className={`max-w-[80%] rounded-lg px-3 py-1.5 text-xs font-mono ${msg.isError ? "bg-red-500/10 border border-red-500/20 text-red-400" : "bg-surface-2/50 border border-surface-3/30 text-slate-500"}`}>
                      <span className="text-slate-400 mr-1.5">&rarr;</span>
                      <span className={msg.isError ? "text-red-300" : "text-slate-400"}>{msg.toolName}</span>
                      {preview && <span className="ml-1.5 text-slate-600">{preview.slice(0, 200)}{preview.length > 200 ? "..." : ""}</span>}
                    </div>
                  </div>
                );
              }

              const text = extractText(msg.content);
              const isCtx = isContextMessage(msg);
              const toolCalls = showToolCalls ? msg.content.filter((c) => c.type === "toolCall") : [];
              const thinkingBlocks = showToolCalls ? msg.content.filter((c) => c.type === "thinking") : [];

              // Context/prefill messages — muted, compact style
              if (isCtx) {
                return (
                  <div key={msg.id} className="flex gap-3">
                    <div className="w-7 flex-shrink-0" />
                    <div className="max-w-[90%] rounded-lg px-3 py-2 bg-surface-2/30 border border-surface-3/20">
                      <div className="text-[10px] font-mono text-slate-600 uppercase tracking-wide mb-1">context prefill</div>
                      <div className="text-xs text-slate-500 leading-relaxed max-h-40 overflow-y-auto prose-chat"
                        dangerouslySetInnerHTML={{ __html: marked.parse(text) as string }}
                      />
                      <div className="text-[10px] text-slate-600 mt-1">{timeStr(msg.timestamp)}</div>
                    </div>
                  </div>
                );
              }

              return (
                <div
                  key={msg.id}
                  className={`flex gap-3 ${msg.role === "user" ? "justify-end" : ""}`}
                >
                  {msg.role === "assistant" && (
                    <img src="/api/mc/agent-avatar?id=lloyd" alt="Lloyd" className="w-7 h-7 rounded-full object-cover flex-shrink-0 mt-0.5" />
                  )}
                  <div
                    className={`max-w-[80%] ${
                      msg.role === "user"
                        ? "bg-brand-600/20 border-brand-500/30"
                        : "bg-surface-2 border-surface-3/50"
                    } rounded-xl px-3.5 py-2.5 border`}
                  >
                    {thinkingBlocks.length > 0 && (
                      <div className={`space-y-1 ${text || toolCalls.length > 0 ? "mb-2 pb-2 border-b border-purple-500/20" : ""}`}>
                        {thinkingBlocks.map((tb, i) => {
                          const key = `${msg.id}-t${i}`;
                          const isExpanded = expandedThinking[key] ?? false;
                          return (
                            <div key={key} className="rounded-md border border-purple-500/20 bg-purple-500/5 overflow-hidden">
                              <button
                                onClick={() => setExpandedThinking((prev) => ({ ...prev, [key]: !isExpanded }))}
                                className="w-full flex items-center gap-1.5 px-2.5 py-1.5 text-left text-xs text-purple-400 hover:bg-purple-500/10 transition-colors"
                              >
                                <Brain className="w-3 h-3 flex-shrink-0" />
                                <span className="font-medium">Thinking</span>
                                <span className="ml-auto text-purple-500/60">{isExpanded ? "\u25B2" : "\u25BC"}</span>
                              </button>
                              {isExpanded && (
                                <div className="px-2.5 pb-2 text-[11px] text-purple-300/70 italic font-mono leading-relaxed whitespace-pre-wrap border-t border-purple-500/15">
                                  {tb.thinking}
                                </div>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    )}
                    {text && (
                      <div
                        className="text-sm leading-relaxed prose-chat"
                        dangerouslySetInnerHTML={{ __html: marked.parse(msg.role === "user" ? stripDatetimePrefix(text) : text) as string }}
                      />
                    )}
                    {toolCalls.length > 0 && (
                      <div className={`space-y-1 ${text ? "mt-2 pt-2 border-t border-surface-3/30" : ""}`}>
                        {toolCalls.map((tc, i) => {
                          const argsStr = tc.arguments ? JSON.stringify(tc.arguments) : "";
                          const truncArgs = argsStr.length > 80 ? argsStr.slice(0, 80) + "..." : argsStr;
                          return (
                            <div key={tc.id || i} className="flex items-center gap-1.5 text-xs font-mono text-slate-500">
                              <Wrench className="w-3 h-3 text-slate-600 flex-shrink-0" />
                              <span className="text-slate-400">{tc.name}</span>
                              {truncArgs && <span className="text-slate-600 truncate">{truncArgs}</span>}
                            </div>
                          );
                        })}
                      </div>
                    )}
                    <div className="flex items-center gap-2 mt-1.5 text-[10px] text-slate-500">
                      <span>{timeStr(msg.timestamp)}</span>
                      {msg.model && <span>{msg.model}</span>}
                      {msg.hasThinking && <Brain className="w-3 h-3 text-purple-400" />}
                      {msg.usage && (
                        <span>{msg.usage.totalTokens} tok</span>
                      )}
                      {msg.toolCallCount != null && msg.toolCallCount > 0 && (
                        <span>{msg.toolCallCount} calls</span>
                      )}
                      {msg.durationMs != null && (
                        <span>{(msg.durationMs / 1000).toFixed(1)}s</span>
                      )}
                    </div>
                  </div>
                  {msg.role === "user" && (
                    <div className="w-7 h-7 rounded-full bg-slate-700 flex items-center justify-center flex-shrink-0 mt-0.5">
                      <User className="w-3.5 h-3.5 text-slate-300" />
                    </div>
                  )}
                </div>
              );
            })}

          {/* Thinking / Working indicator */}
          {thinking && (
            <div className="flex gap-3">
              <img src="/api/mc/agent-avatar?id=lloyd" alt="Lloyd" className="w-7 h-7 rounded-full object-cover flex-shrink-0 mt-0.5" />
              <div className="bg-surface-2 border-surface-3/50 rounded-xl px-3.5 py-2.5 border">
                <div className="flex items-center gap-2 text-sm text-slate-400">
                  {activityType === "working" ? (
                    <>
                      <Wrench className="w-4 h-4 animate-pulse text-amber-400" />
                      <span className="animate-pulse">Working{activityDetail ? <span className="text-slate-500 ml-1 text-xs font-mono">· {activityDetail}</span> : ""}...</span>
                    </>
                  ) : (
                    <>
                      <Brain className="w-4 h-4 animate-pulse text-brand-400" />
                      <span className="animate-pulse">Thinking...</span>
                    </>
                  )}
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input — inside the card at the bottom */}
        <div className="p-3 border-t border-surface-3/50">
          <div className="relative">
            {showPicker && filteredCommands.length > 0 && (
              <SlashCommandPicker
                commands={filteredCommands}
                filter={slashFilter}
                selectedIndex={pickerIndex}
                onSelect={(cmd) => {
                  setShowPicker(false);
                  setInput(cmd.acceptsArgs ? `/${cmd.name} ` : `/${cmd.name}`);
                }}
                onHover={setPickerIndex}
              />
            )}
            <div className="flex gap-2">
              <input
                type="text"
                value={input}
                onChange={(e) => {
                  const val = e.target.value;
                  setInput(val);
                  if (val.startsWith("/") && !val.includes(" ")) {
                    setShowPicker(true);
                    setPickerIndex(0);
                  } else {
                    setShowPicker(false);
                  }
                }}
                onKeyDown={(e) => {
                  if (showPicker && filteredCommands.length > 0) {
                    if (e.key === "ArrowDown") {
                      e.preventDefault();
                      setPickerIndex((p) => (p + 1) % filteredCommands.length);
                      return;
                    }
                    if (e.key === "ArrowUp") {
                      e.preventDefault();
                      setPickerIndex((p) => (p - 1 + filteredCommands.length) % filteredCommands.length);
                      return;
                    }
                    if (e.key === "Enter" || e.key === "Tab") {
                      e.preventDefault();
                      const cmd = filteredCommands[pickerIndex];
                      setShowPicker(false);
                      setInput(cmd.acceptsArgs ? `/${cmd.name} ` : `/${cmd.name}`);
                      return;
                    }
                    if (e.key === "Escape") {
                      e.preventDefault();
                      setShowPicker(false);
                      return;
                    }
                  }
                  if (e.key === "Enter" && !e.shiftKey) handleSend();
                }}
                onBlur={() => {
                  setTimeout(() => setShowPicker(false), 150);
                }}
                placeholder="Talk to Lloyd... (type / for commands)"
                disabled={awaitingReset}
                className="flex-1 bg-surface-2 text-sm text-slate-200 rounded-lg px-3.5 py-2.5 border border-surface-3/50 outline-none focus:border-brand-500/50 placeholder:text-slate-500 transition-colors disabled:opacity-50"
              />
              <button
                onClick={handleSend}
                disabled={!input.trim() || sending || awaitingReset}
                className="bg-brand-600 hover:bg-brand-500 disabled:opacity-40 disabled:cursor-not-allowed text-white rounded-lg px-3 transition-colors"
              >
                <Send className="w-4 h-4" />
              </button>
              <button
                onClick={handleStop}
                disabled={!thinking}
                title="Stop"
                className="bg-red-600 hover:bg-red-500 disabled:opacity-40 disabled:cursor-not-allowed text-white rounded-lg px-3 transition-colors"
              >
                <StopCircle className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
