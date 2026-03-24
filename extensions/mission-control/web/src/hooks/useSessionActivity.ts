import { useState, useEffect, useCallback, useRef } from "react";

interface UseSessionActivityReturn {
  hasActivity: (sessionKey: string) => boolean;
  markSeen: (sessionKey: string) => void;
}

const STORAGE_KEY = "mc-session-lastSeen";

function loadLastSeen(): Record<string, number> {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    return stored ? JSON.parse(stored) : {};
  } catch {
    return {};
  }
}

function saveLastSeen(lastSeen: Record<string, number>) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(lastSeen));
  } catch {
    // Ignore storage errors
  }
}

export function useSessionActivity(
  sessions: Array<{ sessionKey: string; lastActivity: string }>,
  enabled: boolean = true,
  currentSessionKey: string | null = null
): UseSessionActivityReturn {
  const [lastSeen, setLastSeen] = useState<Record<string, number>>(() => loadLastSeen());
  const [tick, setTick] = useState(0);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (!enabled) {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      return;
    }
    setTick(t => t + 1);
    intervalRef.current = setInterval(() => {
      setTick(t => t + 1);
    }, 2000);
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [enabled]);

  useEffect(() => {
    const now = Date.now();
    const newLastSeen = { ...lastSeen };
    let changed = false;
    for (const session of sessions) {
      if (!(session.sessionKey in newLastSeen)) {
        newLastSeen[session.sessionKey] = now;
        changed = true;
      }
    }
    if (changed) {
      setLastSeen(newLastSeen);
      saveLastSeen(newLastSeen);
    }
  }, [sessions]);

  const hasActivity = useCallback(
    (sessionKey: string): boolean => {
      const session = sessions.find((s) => s.sessionKey === sessionKey);
      if (!session) return false;
      const lastActivityTs = new Date(session.lastActivity).getTime();
      const wasSeen = lastSeen[sessionKey] || 0;
      if (sessionKey === currentSessionKey) return false;
      return lastActivityTs > wasSeen;
    },
    [sessions, lastSeen, currentSessionKey]
  );

  const markSeen = useCallback(
    (sessionKey: string) => {
      const newLastSeen = { ...lastSeen, [sessionKey]: Date.now() };
      setLastSeen(newLastSeen);
      saveLastSeen(newLastSeen);
    },
    [lastSeen]
  );

  return { hasActivity, markSeen };
}

export default useSessionActivity;
