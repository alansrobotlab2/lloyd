import { useState, useCallback, useMemo, useRef, useEffect } from "react";
import type { MessageEntry } from "../../api";

export interface TabState {
  id: string;
  sessionKey: string;
  label: string;
  messages: MessageEntry[];
  input: string;
  thinking: boolean;
  activityType: "thinking" | "working" | null;
  activityDetail: string | null;
  awaitingReset: boolean;
  scrollPosition: number;
  lastUserSendTs: number;
}

export interface UseSessionTabsReturn {
  tabs: TabState[];
  activeTabId: string | null;
  activeTab: TabState | null;
  addTab: (sessionKey: string, label?: string) => string;
  closeTab: (tabId: string) => void;
  setActiveTab: (tabId: string) => void;
  updateTab: (tabId: string, updates: Partial<TabState>) => void;
  findTabBySession: (sessionKey: string) => TabState | undefined;
}

const STORAGE_KEY = "mc-session-tabs";

interface PersistedTabState {
  sessionKey: string;
  label: string;
}

interface PersistedState {
  tabs: PersistedTabState[];
  activeSessionKey: string | null;
}

function loadPersistedState(): PersistedState | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed?.tabs)) return parsed;
    return null;
  } catch {
    return null;
  }
}

function savePersistedState(tabs: TabState[], activeTabId: string | null) {
  try {
    const activeTab = tabs.find((t) => t.id === activeTabId);
    const state: PersistedState = {
      tabs: tabs
        .filter((t) => !t.sessionKey.startsWith("pending-"))
        .map((t) => ({ sessionKey: t.sessionKey, label: t.label })),
      activeSessionKey: activeTab?.sessionKey ?? null,
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch {}
}

let tabCounter = 0;

// Compute initial state from localStorage at module level
const _initState = loadPersistedState();
const _initTabs: TabState[] = (_initState?.tabs ?? []).map((p): TabState => ({
  id: `tab-${++tabCounter}`,
  sessionKey: p.sessionKey,
  label: p.label,
  messages: [],
  input: "",
  thinking: false,
  activityType: null,
  activityDetail: null,
  awaitingReset: false,
  scrollPosition: 0,
  lastUserSendTs: 0,
}));
const _initActiveId = _initState?.activeSessionKey
  ? _initTabs.find((t) => t.sessionKey === _initState.activeSessionKey)?.id ?? _initTabs[0]?.id ?? null
  : _initTabs[0]?.id ?? null;

// Live cache survives component unmount/remount cycles (e.g. tab navigation)
// so draft input text is not lost. localStorage only persists sessionKey+label.
let _liveTabs: TabState[] = _initTabs;
let _liveActiveTabId: string | null = _initActiveId;

export function useSessionTabs(): UseSessionTabsReturn {
  const [tabs, setTabs] = useState<TabState[]>(_liveTabs);
  const [activeTabId, setActiveTabId] = useState<string | null>(_liveActiveTabId);

  const activeTab = useMemo(
    () => tabs.find((t) => t.id === activeTabId) ?? null,
    [tabs, activeTabId],
  );

  const findTabBySession = useCallback(
    (sessionKey: string) => tabs.find((t) => t.sessionKey === sessionKey),
    [tabs],
  );

  const addTab = useCallback(
    (sessionKey: string, label?: string): string => {
      // If a tab with the same sessionKey already exists, just focus it
      const existing = tabs.find((t) => t.sessionKey === sessionKey);
      if (existing) {
        setActiveTabId(existing.id);
        return existing.id;
      }

      const id = `tab-${++tabCounter}`;
      const newTab: TabState = {
        id,
        sessionKey,
        label: label || sessionKey.slice(0, 8) + "...",
        messages: [],
        input: "",
        thinking: false,
        activityType: null,
        activityDetail: null,
        awaitingReset: false,
        scrollPosition: 0,
        lastUserSendTs: 0,
      };
      setTabs((prev) => [...prev, newTab]);
      setActiveTabId(id);
      return id;
    },
    [tabs],
  );

  const closeTab = useCallback(
    (tabId: string) => {
      setTabs((prev) => {
        const idx = prev.findIndex((t) => t.id === tabId);
        if (idx === -1) return prev;
        const next = prev.filter((t) => t.id !== tabId);

        // If the closed tab was active, activate the previous or first remaining
        if (activeTabId === tabId && next.length > 0) {
          const newIdx = Math.min(idx, next.length - 1);
          setActiveTabId(next[newIdx].id);
        } else if (next.length === 0) {
          setActiveTabId(null);
        }

        return next;
      });
    },
    [activeTabId],
  );

  const updateTab = useCallback(
    (tabId: string, updates: Partial<TabState>) => {
      setTabs((prev) =>
        prev.map((t) => (t.id === tabId ? { ...t, ...updates } : t)),
      );
    },
    [],
  );

  // Persist tab state to localStorage
  useEffect(() => {
    savePersistedState(tabs, activeTabId);
  }, [tabs, activeTabId]);

  // Keep module-level cache in sync so remounts recover draft input
  useEffect(() => {
    _liveTabs = tabs;
    _liveActiveTabId = activeTabId;
  });

  return {
    tabs,
    activeTabId,
    activeTab,
    addTab,
    closeTab,
    setActiveTab: setActiveTabId,
    updateTab,
    findTabBySession,
  };
}
