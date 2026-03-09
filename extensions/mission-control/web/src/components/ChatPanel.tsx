import { useEffect, useRef, useState, useCallback, useMemo } from "react";
import { Send, User, Loader2, Brain, Hash, MessageCircle, StopCircle, Wrench, Plus, PanelLeft, Globe, Lock, Smartphone, MessageSquare } from "lucide-react";
import { marked } from "marked";
import { api, type MessageEntry, type CommandInfo } from "../api";
import { sanitizeHtml } from "../utils/sanitize";
import SlashCommandPicker from "./SlashCommandPicker";
import VoicePanel from "./VoicePanel";
import { useVoiceContext } from "../contexts/VoiceContext";
import { getLastTtsPlayedAt, setLastTtsPlayedAt } from "../hooks/useVoiceStream";

// Configure marked for safe, sane defaults
marked.setOptions({ breaks: true, gfm: true });

function extractSummary(content: Array<{ type: string; text?: string }>): string | null {
  const fullText = content
    .filter((c) => c.type === "text" && c.text)
    .map((c) => c.text!)
    .join("\n");
  const match = fullText.match(/<summary>([\s\S]*?)<\/summary>/);
  return match ? match[1].trim() : null;
}

function extractText(content: Array<{ type: string; text?: string }>, keepSummary = false): string {
  let text = content
    .filter((c) => c.type === "text" && c.text)
    .map((c) => c.text!)
    .join("\n");
  if (!keepSummary) {
    text = text.replace(/<summary>[\s\S]*?<\/summary>/g, "");
  }
  return text.trim();
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
  return /^\[System Message\]|^System:/.test(text);
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

function formatSessionKey(key: string): string {
  const parts = key.split(":");
  const name = parts[parts.length - 1];
  if (name === "main") return "Main";
  return name.length > 12 ? name.slice(0, 12) + "..." : name;
}

function timeAgo(iso: string): string {
  const d = Date.now() - new Date(iso).getTime();
  if (d < 60000) return "just now";
  if (d < 3600000) return `${Math.floor(d / 60000)}m ago`;
  if (d < 86400000) return `${Math.floor(d / 3600000)}h ago`;
  return `${Math.floor(d / 86400000)}d ago`;
}

interface ChatPanelProps {
  requestedSessionKey?: string | null;
  onSessionLoaded?: () => void;
}

export default function ChatPanel({ requestedSessionKey, onSessionLoaded }: ChatPanelProps = {}) {
  // ── Multi-session state ─────────────────────────────────────────
  const [sessionKey, setSessionKey] = useState<string | null>(null);
  const [messages, setMessages] = useState<MessageEntry[]>([]);
  const [input, setInput] = useState("");
  const [thinking, setThinking] = useState(false);
  const [activityType, setActivityType] = useState<"thinking" | "working" | null>(null);
  const [activityDetail, setActivityDetail] = useState<string | null>(null);
  const [awaitingReset, setAwaitingReset] = useState(false);
  const [lastUserSendTs, setLastUserSendTs] = useState(0);
  // ── Shared state ─────────────────────────────────────────────────
  const [sending, setSending] = useState(false);
  const [sessions, setSessions] = useState<Array<{ sessionKey: string; lastActivity: string; model: string; summary?: string; source?: string; peer?: string }>>([]);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [showToolCalls, setShowToolCalls] = useState(() => {
    try { return localStorage.getItem("mc-agent-details-visible") === "true"; } catch { return false; }
  });
  const [expandedThinking, setExpandedThinking] = useState<Record<string, boolean>>({});
  // Slash command picker state
  const [commandsList, setCommandsList] = useState<CommandInfo[]>([]);
  const [showPicker, setShowPicker] = useState(false);
  const [pickerIndex, setPickerIndex] = useState(0);

  // ── Refs ────────────────────────────────────────────────────────────
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const isNearBottom = useRef(true);
  const initialLoadDone = useRef(false);
  const scrollToBottomOnLoad = useRef(true);
  const thinkingRestored = useRef(false);
  const workingStartRef = useRef<number>(0);
  const workingTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const isWorkingRef = useRef(false);
  const { voiceEnabled, ttsEnabled } = useVoiceContext();
  const spokenMsgIds = useRef<Set<string>>(new Set());

  // ── Derived state ──────────────────────────────────────────────────
  const currentModel = useMemo(() => {
    const lastAssistant = [...messages].reverse().find((m) => m.role === "assistant" && m.model);
    if (lastAssistant?.model) return lastAssistant.model;
    return sessions.find((s) => s.sessionKey === sessionKey)?.model || null;
  }, [messages, sessions, sessionKey]);

  const thinkingEnabled = useMemo(() => {
    const lastAssistant = [...messages].reverse().find((m) => m.role === "assistant");
    return lastAssistant?.hasThinking ?? false;
  }, [messages]);

  // ── Switch to a session requested by another page ──────────────────
  useEffect(() => {
    if (requestedSessionKey) {
      initialLoadDone.current = true;
      switchSession(requestedSessionKey);
      onSessionLoaded?.();
    }
  }, [requestedSessionKey]);

  const switchSession = useCallback((key: string) => {
    setSessionKey(key);
    setMessages([]);
    setInput("");
    setThinking(false);
    setActivityType(null);
    setActivityDetail(null);
    setAwaitingReset(false);
    setLastUserSendTs(0);
    scrollToBottomOnLoad.current = true;
    thinkingRestored.current = false;
  }, []);

  const refreshSessions = useCallback(() => {
    return api.sessions().then((d) => {
      const mapped = d.sessions.map((s) => ({
        sessionKey: s.sessionKey,
        lastActivity: s.lastActivity,
        model: s.model,
        summary: s.summary,
        source: s.source,
        peer: s.peer,
      }));
      setSessions((prev) => {
        // Keep optimistic entries not yet in the API response
        const apiKeys = new Set(mapped.map((s) => s.sessionKey));
        const optimistic = prev.filter((s) => !apiKeys.has(s.sessionKey));
        return [...optimistic, ...mapped];
      });
      return mapped;
    }).catch((err) => {
      console.error(err);
      return [] as Array<{ sessionKey: string; lastActivity: string; model: string; summary?: string }>;
    });
  }, []);

  // Initial load — auto-open most recent session
  useEffect(() => {
    refreshSessions().then((list) => {
      if (!initialLoadDone.current && list.length > 0) {
        initialLoadDone.current = true;
        setSessionKey(list[0].sessionKey);
      }
    });
  }, [refreshSessions]);

  // Periodic session list refresh (catches gateway restarts, new sessions from other sources)
  useEffect(() => {
    if (thinking) return;
    const interval = setInterval(() => { refreshSessions(); }, 15_000);
    return () => clearInterval(interval);
  }, [thinking, refreshSessions]);

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

  // Poll messages for the current session
  useEffect(() => {
    if (!sessionKey) return;
    const currentKey = sessionKey; // capture for closure stability
    const load = () =>
      api.sessionMessages(currentKey, showToolCalls)
        .then((d) => {
          setMessages((prev) => {
            // Guard: don't replace with fewer messages while thinking
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
                  setLastUserSendTs(new Date(lastMsg.timestamp).getTime());
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
          // a new assistant message with real text content.
          if (thinking && lastUserSendTs > 0) {
            const postSendAssistant = d.messages.filter(
              (m) => m.role === "assistant" && new Date(m.timestamp).getTime() > lastUserSendTs,
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
                  setThinking(false);
                });
              }
            }
          }
        })
        .catch(() => {
          setMessages([]);
          refreshSessions().then((list) => {
            if (list.length > 0 && !list.some((s) => s.sessionKey === currentKey)) {
              switchSession(list[0].sessionKey);
            }
          });
        });
    load();
    const interval = setInterval(load, thinking || awaitingReset ? 1_500 : 3_000);
    return () => clearInterval(interval);
  }, [sessionKey, awaitingReset, thinking, refreshSessions, showToolCalls, lastUserSendTs]);

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
              return;
            }
            isWorkingRef.current = false;
          }
          if (status.activity.type === "llm_thinking") {
            setActivityType("thinking");
            setActivityDetail(status.activity.detail);
          }
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

  // ── TTS fallback: extract <summary> tags and play audio ──────────
  useEffect(() => {
    if (!ttsEnabled || thinking || messages.length === 0) return;
    const last = messages[messages.length - 1];
    if (last.role !== "assistant") return;
    if (spokenMsgIds.current.has(last.id)) return;
    const summary = extractSummary(last.content);
    if (!summary) return;
    // Skip if any TTS path already played recently (dedup)
    if (Date.now() - getLastTtsPlayedAt() < 15_000) {
      spokenMsgIds.current.add(last.id);
      return;
    }
    spokenMsgIds.current.add(last.id);
    setLastTtsPlayedAt(Date.now());
    fetch("/api/mc/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: summary }),
    })
      .then((res) => {
        if (!res.ok) throw new Error(`TTS returned ${res.status}`);
        return res.blob();
      })
      .then((blob) => {
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        audio.onended = () => URL.revokeObjectURL(url);
        audio.onerror = () => URL.revokeObjectURL(url);
        audio.play().catch((err) => console.error("TTS playback failed:", err));
      })
      .catch((err) => console.error("TTS fetch failed:", err));
  }, [messages, thinking, ttsEnabled]);

  const handleSend = async () => {
    if (!input.trim() || sending || !sessionKey) return;
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
    setLastUserSendTs(Date.now());

    try {
      await api.chat(text, sessionKey);
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
    if (!sessionKey) return;
    try {
      await api.chatAbort(sessionKey);
      setThinking(false);
    } catch (err) {
      console.error("Stop failed:", err);
    }
  };

  const handleNew = async () => {
    if (sending) return;

    setSending(true);
    setAwaitingReset(true);
    setThinking(true);
    setMessages([]);
    setInput("");
    setActivityType(null);
    setActivityDetail(null);
    scrollToBottomOnLoad.current = true;

    try {
      const result = await api.sessionNew();
      if (result.sessionKey) {
        setSessionKey(result.sessionKey);
        setLastUserSendTs(Date.now());
        setSessions((prev) => [
          { sessionKey: result.sessionKey, lastActivity: new Date().toISOString(), model: "", summary: "New session" },
          ...prev,
        ]);
      }
    } catch (err) {
      console.error("New chat failed:", err);
      setAwaitingReset(false);
      setThinking(false);
    } finally {
      setSending(false);
    }
  };

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Page header */}
      <div className="flex items-center justify-between flex-shrink-0 px-6 pt-6 pb-2">
        <div className="flex items-center gap-3">
          <button
            onClick={() => setSidebarOpen((v) => !v)}
            title={sidebarOpen ? "Hide sessions" : "Show sessions"}
            className="text-slate-400 hover:text-brand-400 transition-colors"
          >
            <PanelLeft className="w-5 h-5" />
          </button>
          <MessageCircle className="w-5 h-5 text-brand-400" />
          <h2 className="text-lg font-semibold">Chat</h2>
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
            onClick={handleNew}
            disabled={sending}
            className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium text-slate-400 hover:text-brand-400 hover:bg-brand-500/10 transition-colors disabled:opacity-40"
          >
            <Plus className="w-3.5 h-3.5" />
            New Session
          </button>
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
        </div>
      </div>

      {/* Main content: sidebar + chat */}
      <div className="flex-1 flex min-h-0 mx-6 mb-6 overflow-hidden">
        {/* Session sidebar */}
        {sidebarOpen && (
          <div className="w-56 flex-shrink-0 bg-surface-1 rounded-l-xl border border-r-0 border-surface-3/50 flex flex-col overflow-hidden">
            <div className="p-3 border-b border-surface-3/50 flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wide">Sessions</span>
              <button
                onClick={handleNew}
                disabled={sending}
                title="New session"
                className="text-slate-400 hover:text-brand-400 transition-colors disabled:opacity-40"
              >
                <Plus className="w-4 h-4" />
              </button>
            </div>
            <div className="flex-1 overflow-y-auto">
              {[...sessions].sort((a, b) => new Date(b.lastActivity).getTime() - new Date(a.lastActivity).getTime()).map((s) => {
                const sourceIcons: Record<string, React.ElementType> = { webchat: Globe, discord: Hash, telegram: Send, signal: Lock, whatsapp: Smartphone, other: MessageSquare };
                const SourceIcon = sourceIcons[s.source || "other"] || MessageSquare;
                const rawLabel = (() => {
                  if (!s.source || s.source === "webchat") return "alan";
                  if (s.peer === "912913439209443358") return "alan";
                  return s.peer || s.source;
                })();
                const label = rawLabel.toLowerCase();
                return (
                  <button
                    key={s.sessionKey}
                    onClick={() => switchSession(s.sessionKey)}
                    className={`w-full text-left px-3 py-2.5 text-sm border-b border-surface-3/20 transition-colors ${
                      sessionKey === s.sessionKey
                        ? "bg-brand-500/10 text-brand-400"
                        : "text-slate-400 hover:bg-surface-2"
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <div className="truncate text-xs font-medium">
                        <SourceIcon className="w-3.5 h-3.5 inline-block mr-1 -mt-0.5" />{label}
                      </div>
                      <span className="text-[10px] text-slate-600 whitespace-nowrap shrink-0">{timeAgo(s.lastActivity)}</span>
                    </div>
                    {(s.summary || formatSessionKey(s.sessionKey) !== label) && (
                      <div className="truncate text-[10px] text-slate-500 mt-0.5">
                        {s.summary || formatSessionKey(s.sessionKey)}
                      </div>
                    )}
                  </button>
                );
              })}
              {sessions.length === 0 && (
                <div className="p-3 text-xs text-slate-600 text-center">No sessions</div>
              )}
            </div>
          </div>
        )}

        {/* Messages card */}
        <div className={`flex-1 flex flex-col min-w-0 bg-surface-1 border border-surface-3/50 overflow-hidden ${sidebarOpen ? "rounded-r-xl" : "rounded-xl"}`}>
          <VoicePanel />
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
                  const preview = extractText(msg.content, showToolCalls);
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

                const text = extractText(msg.content, showToolCalls);
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
                          dangerouslySetInnerHTML={{ __html: sanitizeHtml(text) }}
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
                          dangerouslySetInnerHTML={{ __html: sanitizeHtml(msg.role === "user" ? stripDatetimePrefix(text) : text) }}
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
                  placeholder={sessionKey ? "Talk to Lloyd... (type / for commands)" : "Select a session"}
                  disabled={awaitingReset || !sessionKey}
                  className="flex-1 bg-surface-2 text-sm text-slate-200 rounded-lg px-3.5 py-2.5 border border-surface-3/50 outline-none focus:border-brand-500/50 placeholder:text-slate-500 transition-colors disabled:opacity-50"
                />
                <button
                  onClick={handleSend}
                  disabled={!input.trim() || sending || awaitingReset || !sessionKey}
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
    </div>
  );
}
