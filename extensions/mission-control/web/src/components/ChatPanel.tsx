import { useEffect, useRef, useState, useCallback, useMemo } from "react";
import { Send, User, Loader2, Brain, MessageCircle, StopCircle, Wrench } from "lucide-react";
import { marked } from "marked";
import { api, type MessageEntry, type CommandInfo } from "../api";
import SlashCommandPicker from "./SlashCommandPicker";
import SessionTabBar from "./SessionTabBar";
import { useSessionTabs } from "./hooks/useSessionTabs";

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
  // ── Tab state ──────────────────────────────────────────────────────
  const {
    tabs, activeTabId, activeTab,
    addTab, closeTab, setActiveTab, updateTab, findTabBySession,
  } = useSessionTabs();

  // ── Shared (cross-tab) state ───────────────────────────────────────
  const [sending, setSending] = useState(false);
  const [sessions, setSessions] = useState<Array<{ sessionId: string; lastActivity: string; model: string; summary?: string }>>([]);
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
  const prevActiveTabIdRef = useRef<string | null>(null);

  // ── Convenience aliases from active tab ────────────────────────────
  const messages = activeTab?.messages ?? [];
  const input = activeTab?.input ?? "";
  const thinking = activeTab?.thinking ?? false;
  const activityType = activeTab?.activityType ?? null;
  const activityDetail = activeTab?.activityDetail ?? null;
  const awaitingReset = activeTab?.awaitingReset ?? false;
  const sessionId = activeTab?.sessionKey ?? null;

  // ── Derived state ──────────────────────────────────────────────────
  const currentModel = useMemo(() => {
    const lastAssistant = [...messages].reverse().find((m) => m.role === "assistant" && m.model);
    if (lastAssistant?.model) return lastAssistant.model;
    return sessions.find((s) => s.sessionId === sessionId)?.model || null;
  }, [messages, sessions, sessionId]);

  const thinkingEnabled = useMemo(() => {
    const lastAssistant = [...messages].reverse().find((m) => m.role === "assistant");
    return lastAssistant?.hasThinking ?? false;
  }, [messages]);

  // ── Helper: update active tab state ────────────────────────────────
  const updateActive = useCallback(
    (updates: Parameters<typeof updateTab>[1]) => {
      if (activeTabId) updateTab(activeTabId, updates);
    },
    [activeTabId, updateTab],
  );

  // Wrapper for setInput that updates active tab
  const setInput = useCallback(
    (val: string) => updateActive({ input: val }),
    [updateActive],
  );

  // ── Switch to a session requested by another page ──────────────────
  useEffect(() => {
    if (requestedSessionId) {
      initialLoadDone.current = true;
      const sessionInfo = sessions.find((s) => s.sessionId === requestedSessionId);
      const label = sessionInfo?.summary || requestedSessionId.slice(0, 8) + "...";
      addTab(requestedSessionId, label);
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

  // Initial load — auto-open most recent session as first tab
  useEffect(() => {
    refreshSessions().then((list) => {
      if (!initialLoadDone.current && list.length > 0) {
        initialLoadDone.current = true;
        // If tabs were restored from localStorage, don't auto-open
        if (tabs.length > 0) return;
        const s = list[0];
        addTab(s.sessionId, s.summary || s.sessionId.slice(0, 8) + "...");
      }
    });
  }, [refreshSessions]);

  // Update tab labels when sessions list refreshes (e.g., after summary arrives)
  useEffect(() => {
    if (sessions.length === 0) return;
    for (const tab of tabs) {
      const sessionInfo = sessions.find((s) => s.sessionId === tab.sessionKey);
      if (sessionInfo) {
        const newLabel = sessionInfo.summary || sessionInfo.sessionId.slice(0, 8) + "...";
        if (newLabel !== tab.label) {
          updateTab(tab.id, { label: newLabel });
        }
      }
    }
  }, [sessions]);

  // Save scroll position when switching tabs, restore on activate
  useEffect(() => {
    const prevId = prevActiveTabIdRef.current;
    if (prevId && prevId !== activeTabId) {
      // Save scroll position for the tab we're leaving
      const el = messagesContainerRef.current;
      if (el) {
        updateTab(prevId, { scrollPosition: el.scrollTop });
      }
    }
    prevActiveTabIdRef.current = activeTabId;

    // Reset scroll flags for new tab
    scrollToBottomOnLoad.current = true;
    thinkingRestored.current = false;

    // Restore scroll position for the tab we're switching to (next frame)
    if (activeTab) {
      requestAnimationFrame(() => {
        const el = messagesContainerRef.current;
        if (el && activeTab.scrollPosition > 0) {
          el.scrollTop = activeTab.scrollPosition;
          scrollToBottomOnLoad.current = false;
        }
      });
    }
  }, [activeTabId]);

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

  // Poll messages for the active tab's session
  useEffect(() => {
    if (!sessionId || !activeTabId) return;
    const tabId = activeTabId; // capture for closure stability
    const load = () =>
      api.sessionMessages(sessionId, showToolCalls)
        .then((d) => {
          updateTab(tabId, {
            messages: (() => {
              // Use functional access to current tab state for the guard
              const tab = tabs.find((t) => t.id === tabId);
              if (tab?.thinking && d.messages.length < tab.messages.length) return tab.messages;
              return d.messages;
            })(),
          });

          // Restore thinking indicator on re-mount if agent is still processing
          const tab = tabs.find((t) => t.id === tabId);
          if (!tab?.thinking && !thinkingRestored.current && d.messages.length > 0) {
            thinkingRestored.current = true;
            const lastMsg = d.messages[d.messages.length - 1];
            if (lastMsg.role === "user") {
              api.agentStatus().then((status) => {
                if (status.mainAgent.state !== "idle") {
                  updateTab(tabId, {
                    lastUserSendTs: new Date(lastMsg.timestamp).getTime(),
                    thinking: true,
                  });
                }
              }).catch(() => {});
            }
          }

          // Clear awaitingReset once any messages appear in the new session
          if (d.messages.length > 0 && tab?.awaitingReset) {
            updateTab(tabId, { awaitingReset: false });
            refreshSessions();
          }

          // Clear thinking indicator once the agent is idle AND we have
          // a new assistant message with real text content.
          if (tab?.thinking && tab.lastUserSendTs > 0) {
            const postSendAssistant = d.messages.filter(
              (m) => m.role === "assistant" && new Date(m.timestamp).getTime() > tab.lastUserSendTs,
            );
            if (postSendAssistant.length > 0) {
              const newest = postSendAssistant[postSendAssistant.length - 1];
              const hasTextContent = newest.content?.some(
                (c) => c.type === "text" && c.text && c.text.trim().length > 0,
              );
              if (hasTextContent) {
                api.agentStatus().then((status) => {
                  if (status.mainAgent.state === "idle") {
                    updateTab(tabId, { thinking: false });
                  }
                }).catch(() => {
                  updateTab(tabId, { thinking: false });
                });
              }
            }
          }
        })
        .catch(() => {
          updateTab(tabId, { messages: [] });
        });
    load();
    const interval = setInterval(load, thinking || awaitingReset ? 1_500 : 3_000);
    return () => clearInterval(interval);
  }, [sessionId, awaitingReset, thinking, refreshSessions, showToolCalls, activeTabId]);

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
    if (!thinking || !activeTabId) {
      if (!thinking) {
        updateActive({ activityType: null, activityDetail: null });
      }
      if (workingTimeoutRef.current) {
        clearTimeout(workingTimeoutRef.current);
        workingTimeoutRef.current = null;
      }
      isWorkingRef.current = false;
      return;
    }
    const tabId = activeTabId;
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
          updateTab(tabId, { activityType: "working", activityDetail: status.activity.detail });
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
                  updateTab(tabId, { activityType: "thinking", activityDetail: capturedDetail });
                } else {
                  updateTab(tabId, { activityType: null, activityDetail: null });
                }
              }, remaining);
              return;
            }
            isWorkingRef.current = false;
          }
          if (status.activity.type === "llm_thinking") {
            updateTab(tabId, { activityType: "thinking", activityDetail: status.activity.detail });
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
  }, [thinking, activeTabId]);

  const handleSend = async () => {
    if (!input.trim() || sending || !activeTabId) return;
    const text = input.trim();
    const tabId = activeTabId;

    setSending(true);
    updateTab(tabId, { input: "" });

    // Optimistic user message
    const optimisticMsg: MessageEntry = {
      id: `pending-${Date.now()}`,
      timestamp: new Date().toISOString(),
      role: "user",
      content: [{ type: "text", text }],
    };
    const currentTab = tabs.find((t) => t.id === tabId);
    updateTab(tabId, {
      messages: [...(currentTab?.messages ?? []), optimisticMsg],
      lastUserSendTs: Date.now(),
    });

    try {
      await api.chat(text);
      updateTab(tabId, { thinking: true });
    } catch (err) {
      console.error("Chat send failed:", err);
      const tab = tabs.find((t) => t.id === tabId);
      updateTab(tabId, {
        messages: (tab?.messages ?? []).filter((m) => m.id !== optimisticMsg.id),
        input: text,
      });
    } finally {
      setSending(false);
    }
  };

  const handleStop = async () => {
    if (!activeTabId) return;
    try {
      await api.chatAbort();
      updateTab(activeTabId, { thinking: false });
    } catch (err) {
      console.error("Stop failed:", err);
    }
  };

  const handleNew = async () => {
    if (sending) return;

    // Create a temporary tab immediately
    const tempKey = `pending-${Date.now()}`;
    const tabId = addTab(tempKey, "New session...");

    updateTab(tabId, { awaitingReset: true });
    setSending(true);

    try {
      const result = await api.sessionNew();
      if (result.sessionId) {
        updateTab(tabId, {
          sessionKey: result.sessionId,
          label: "New session",
          awaitingReset: true,
          lastUserSendTs: Date.now(),
          thinking: true,
        });
        setSessions((prev) => [
          { sessionId: result.sessionId!, lastActivity: new Date().toISOString(), model: "", summary: "New session" },
          ...prev,
        ]);
      }
    } catch (err) {
      console.error("New chat failed:", err);
      closeTab(tabId);
    } finally {
      setSending(false);
    }
  };

  const handleTabSelect = useCallback(
    (tabId: string) => {
      // Save scroll position before switching
      if (activeTabId) {
        const el = messagesContainerRef.current;
        if (el) {
          updateTab(activeTabId, { scrollPosition: el.scrollTop });
        }
      }
      setActiveTab(tabId);
    },
    [activeTabId, setActiveTab, updateTab],
  );

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Page header */}
      <div className="flex items-center justify-between flex-shrink-0 px-6 pt-6 pb-2">
        <div className="flex items-center gap-3">
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

      {/* Tab bar */}
      <div className="px-6">
        <SessionTabBar
          tabs={tabs}
          activeTabId={activeTabId}
          onSelect={handleTabSelect}
          onClose={closeTab}
          onNew={handleNew}
        />
      </div>

      {/* Messages card */}
      <div className="flex-1 flex flex-col min-h-0 mx-6 mb-6 bg-surface-1 rounded-b-xl border border-t-0 border-surface-3/50 overflow-hidden">
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
                disabled={awaitingReset || !activeTab}
                className="flex-1 bg-surface-2 text-sm text-slate-200 rounded-lg px-3.5 py-2.5 border border-surface-3/50 outline-none focus:border-brand-500/50 placeholder:text-slate-500 transition-colors disabled:opacity-50"
              />
              <button
                onClick={handleSend}
                disabled={!input.trim() || sending || awaitingReset || !activeTab}
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
