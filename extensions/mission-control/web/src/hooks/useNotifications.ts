import { useState, useEffect, useCallback } from "react";
import type { ToastNotification } from "../components/Toast";

const SSE_URL = "/api/mc/notification-stream";
const FETCH_URL = "/api/mc/notifications";

// Seen IDs to deduplicate between initial fetch and SSE
const seenIds = new Set<string>();

export function useNotifications() {
  const [notifications, setNotifications] = useState<ToastNotification[]>([]);

  const addNotification = useCallback((n: ToastNotification) => {
    if (seenIds.has(n.id)) return;
    seenIds.add(n.id);
    setNotifications((prev) => [...prev, n]);
  }, []);

  const dismiss = useCallback((id: string) => {
    setNotifications((prev) => prev.filter((n) => n.id !== id));
  }, []);

  // Fetch any notifications we may have missed on mount
  useEffect(() => {
    fetch(FETCH_URL, { credentials: "include" })
      .then((r) => r.json())
      .then((data: { notifications?: ToastNotification[] }) => {
        if (Array.isArray(data.notifications)) {
          for (const n of data.notifications) addNotification(n);
        }
      })
      .catch(() => { /* silently ignore — SSE will catch live ones */ });
  }, [addNotification]);

  // Connect to SSE stream for real-time notifications
  useEffect(() => {
    let es: EventSource | null = null;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;

    function connect() {
      es = new EventSource(SSE_URL);

      es.addEventListener("notification", (event: MessageEvent) => {
        try {
          const n = JSON.parse(event.data) as ToastNotification;
          addNotification(n);
        } catch { /* skip malformed */ }
      });

      es.onerror = () => {
        es?.close();
        es = null;
        // Retry after 5s
        retryTimer = setTimeout(connect, 5000);
      };
    }

    connect();

    return () => {
      if (retryTimer) clearTimeout(retryTimer);
      es?.close();
    };
  }, [addNotification]);

  return { notifications, dismiss };
}
