import React, { useEffect, useState } from "react";

export type NotificationLevel = "info" | "success" | "warning" | "error";

export interface ToastNotification {
  id: string;
  text: string;
  level: NotificationLevel;
  timestamp: number;
}

interface ToastItemProps {
  notification: ToastNotification;
  onDismiss: (id: string) => void;
}

const LEVEL_STYLES: Record<NotificationLevel, string> = {
  success: "border-green-500/60 bg-green-950/80 text-green-200",
  error:   "border-red-500/60 bg-red-950/80 text-red-200",
  warning: "border-yellow-500/60 bg-yellow-950/80 text-yellow-200",
  info:    "border-blue-500/60 bg-blue-950/80 text-blue-200",
};

const LEVEL_ICON_COLOR: Record<NotificationLevel, string> = {
  success: "text-green-400",
  error:   "text-red-400",
  warning: "text-yellow-400",
  info:    "text-blue-400",
};

const AUTO_DISMISS_MS = 8000;

function ToastItem({ notification, onDismiss }: ToastItemProps) {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    // Animate in
    const enterTimer = setTimeout(() => setVisible(true), 10);
    // Auto-dismiss
    const dismissTimer = setTimeout(() => {
      setVisible(false);
      setTimeout(() => onDismiss(notification.id), 300);
    }, AUTO_DISMISS_MS);

    return () => {
      clearTimeout(enterTimer);
      clearTimeout(dismissTimer);
    };
  }, [notification.id, onDismiss]);

  const handleClose = () => {
    setVisible(false);
    setTimeout(() => onDismiss(notification.id), 300);
  };

  return (
    <div
      className={`
        flex items-start gap-3 px-4 py-3 rounded-lg border shadow-lg backdrop-blur-sm
        transition-all duration-300 max-w-sm w-full
        ${LEVEL_STYLES[notification.level]}
        ${visible ? "opacity-100 translate-x-0" : "opacity-0 translate-x-4"}
      `}
    >
      <div className={`flex-1 text-sm leading-snug ${LEVEL_ICON_COLOR[notification.level]}`}>
        {notification.text}
      </div>
      <button
        onClick={handleClose}
        className="flex-shrink-0 text-white/40 hover:text-white/80 transition-colors mt-0.5"
        aria-label="Dismiss"
      >
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
    </div>
  );
}

interface ToastContainerProps {
  notifications: ToastNotification[];
  onDismiss: (id: string) => void;
}

export function ToastContainer({ notifications, onDismiss }: ToastContainerProps) {
  if (notifications.length === 0) return null;

  return (
    <div className="fixed top-4 right-4 z-50 flex flex-col gap-2 pointer-events-none">
      {notifications.map((n) => (
        <div key={n.id} className="pointer-events-auto">
          <ToastItem notification={n} onDismiss={onDismiss} />
        </div>
      ))}
    </div>
  );
}
