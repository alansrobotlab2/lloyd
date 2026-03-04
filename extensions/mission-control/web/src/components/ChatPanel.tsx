import { useEffect, useRef, useState, useCallback } from "react";
import { Send, User, Plus, Loader2, Brain, MessageCircle, StopCircle } from "lucide-react";
import { marked } from "marked";
import { api, type MessageEntry } from "../api";

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

function timeStr(ts: string): string {
  return new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
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
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const isNearBottom = useRef(true);
  const initialLoadDone = useRef(false);
  // Track the timestamp of the last user-sent message so we can detect new replies
  const lastUserSendTs = useRef(0);

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

  // Poll messages for the selected session
  useEffect(() => {
    if (!sessionId) return;
    const load = () =>
      api.sessionMessages(sessionId)
        .then((d) => {
          // Don't replace local messages with fewer server messages while
          // thinking — protects the optimistic user message from flickering
          // away before the server has written it to disk.
          setMessages((prev) => {
            if (thinking && d.messages.length < prev.length) return prev;
            return d.messages;
          });

          // Clear awaitingReset once any messages appear in the new session
          if (d.messages.length > 0 && awaitingReset) {
            setAwaitingReset(false);
            refreshSessions();
          }

          // Clear thinking indicator once the agent is idle AND we have
          // a new assistant message. This keeps the bubble visible during
          // multi-response processing (tool calls between text replies).
          if (thinking && lastUserSendTs.current > 0) {
            const hasNewAssistant = d.messages.some(
              (m) => m.role === "assistant" && new Date(m.timestamp).getTime() > lastUserSendTs.current,
            );
            if (hasNewAssistant) {
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
        })
        .catch(() => {
          // 404 is expected while the session file doesn't exist yet (after /new).
          // Clear stale messages so old session content doesn't linger.
          setMessages([]);
        });
    load();
    const interval = setInterval(load, thinking || awaitingReset ? 1_500 : 3_000);
    return () => clearInterval(interval);
  }, [sessionId, awaitingReset, thinking, refreshSessions]);

  // Track whether user is scrolled near the bottom
  const handleScroll = useCallback(() => {
    const el = messagesContainerRef.current;
    if (!el) return;
    const threshold = 80;
    isNearBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < threshold;
  }, []);

  // Auto-scroll only when user is near the bottom
  useEffect(() => {
    if (isNearBottom.current) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, thinking]);

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
        </div>
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
          {messages.map((msg) => (
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
                <div
                  className="text-sm leading-relaxed prose-chat"
                  dangerouslySetInnerHTML={{ __html: marked.parse(extractText(msg.content)) as string }}
                />
                <div className="flex items-center gap-2 mt-1.5 text-[10px] text-slate-500">
                  <span>{timeStr(msg.timestamp)}</span>
                  {msg.model && <span>{msg.model}</span>}
                  {msg.usage && (
                    <span>
                      {msg.usage.totalTokens} tok
                    </span>
                  )}
                </div>
              </div>
              {msg.role === "user" && (
                <div className="w-7 h-7 rounded-full bg-slate-700 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <User className="w-3.5 h-3.5 text-slate-300" />
                </div>
              )}
            </div>
          ))}

          {/* Thinking indicator */}
          {thinking && (
            <div className="flex gap-3">
              <img src="/api/mc/agent-avatar?id=lloyd" alt="Lloyd" className="w-7 h-7 rounded-full object-cover flex-shrink-0 mt-0.5" />
              <div className="bg-surface-2 border-surface-3/50 rounded-xl px-3.5 py-2.5 border">
                <div className="flex items-center gap-2 text-sm text-slate-400">
                  <Brain className="w-4 h-4 animate-pulse text-brand-400" />
                  <span className="animate-pulse">Thinking...</span>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input — inside the card at the bottom */}
        <div className="p-3 border-t border-surface-3/50">
          <div className="flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleSend()}
              placeholder="Talk to Lloyd..."
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
  );
}
