import { useEffect, useRef, useState } from "react"
import ChatPanel from "@/components/ChatPanel"

interface FocusMessage {
  type: "focus-session"
  windowId: number
  tabId: number
  sessionKey: string | null
  url?: string
  title?: string
  canCheck?: boolean
}

interface PanelState {
  sessionKey: string | null
  url: string | null
  title: string | null
  canCheck: boolean
}

const INITIAL: PanelState = {
  sessionKey: null,
  url: null,
  title: null,
  canCheck: false,
}

const CHECK_LABEL = "Check it out, Lloyd"

export default function SidePanelApp() {
  const [state, setState] = useState<PanelState>(INITIAL)
  const [pending, setPending] = useState(false)
  const windowIdRef = useRef<number | null>(null)
  const portRef = useRef<chrome.runtime.Port | null>(null)
  const pendingTimerRef = useRef<number | null>(null)

  function settlePending() {
    setPending(false)
    if (pendingTimerRef.current) {
      window.clearTimeout(pendingTimerRef.current)
      pendingTimerRef.current = null
    }
  }

  useEffect(() => {
    let cancelled = false

    async function connect() {
      const win = await chrome.windows.getCurrent()
      const [tab] = await chrome.tabs.query({
        active: true,
        windowId: win.id,
      })
      if (cancelled) return

      windowIdRef.current = win.id ?? null
      const port = chrome.runtime.connect({ name: "lloyd-sidepanel" })
      portRef.current = port

      port.onMessage.addListener((raw: unknown) => {
        if (!raw || typeof raw !== "object") return
        const msg = raw as FocusMessage
        if (msg.type !== "focus-session") return
        if (msg.windowId !== windowIdRef.current) return
        setState({
          sessionKey: msg.sessionKey,
          url: msg.url ?? null,
          title: msg.title ?? null,
          canCheck: Boolean(msg.canCheck),
        })
        // Any focus push from the SW settles a pending check request.
        settlePending()
      })

      port.onDisconnect.addListener(() => {
        portRef.current = null
        // The SW was likely killed; reconnect on the next tick.
        setTimeout(connect, 250)
      })

      port.postMessage({
        type: "panel-ready",
        windowId: win.id,
        tabId: tab?.id,
      })
    }

    connect()
    return () => {
      cancelled = true
      portRef.current?.disconnect()
      portRef.current = null
      if (pendingTimerRef.current) window.clearTimeout(pendingTimerRef.current)
    }
  }, [])

  async function requestCheck() {
    if (pending || !windowIdRef.current) return
    setPending(true)
    // Safety net: if the SW never answers (dead, port down), un-stick.
    pendingTimerRef.current = window.setTimeout(() => setPending(false), 10000)
    try {
      const win = await chrome.windows.getCurrent()
      const [tab] = await chrome.tabs.query({
        active: true,
        windowId: win.id,
      })
      portRef.current?.postMessage({
        type: "request-session",
        windowId: win.id,
        tabId: tab?.id,
      })
    } catch {
      settlePending()
    }
  }

  return (
    <div className="flex h-screen flex-col bg-background text-foreground">
      <div className="border-b border-border bg-card/40 px-3 py-2">
        <CheckButton
          canCheck={state.canCheck}
          pending={pending}
          onRequest={requestCheck}
        />
      </div>
      {state.sessionKey ? (
        <>
          <PageHeader url={state.url} title={state.title} />
          <div className="min-h-0 flex-1">
            <ChatPanel requestedSessionKey={state.sessionKey} compact />
          </div>
        </>
      ) : (
        <EmptyState canCheck={state.canCheck} />
      )}
    </div>
  )
}

function CheckButton({
  canCheck,
  pending,
  onRequest,
}: {
  canCheck: boolean
  pending: boolean
  onRequest: () => void
}) {
  return (
    <button
      type="button"
      onClick={onRequest}
      disabled={pending || !canCheck}
      className="w-full rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90 disabled:cursor-not-allowed disabled:opacity-50"
    >
      {pending ? "Checking\u2026" : CHECK_LABEL}
    </button>
  )
}

function EmptyState({ canCheck }: { canCheck: boolean }) {
  return (
    <div className="flex flex-1 items-center justify-center p-6 text-center">
      <div className="space-y-1">
        <div className="text-lg font-medium">
          {canCheck ? "Ready when you are" : "No checkable page here"}
        </div>
        <div className="text-sm text-muted-foreground">
          {canCheck
            ? "Lloyd hasn't looked at this page yet. Hit the button above to pull the contents and get the highlights."
            : "Open a regular web page or YouTube video, then ask me to check it out."}
        </div>
      </div>
    </div>
  )
}

function PageHeader({
  url,
  title,
}: {
  url: string | null
  title: string | null
}) {
  if (!url) return null
  let host = ""
  try {
    host = new URL(url).hostname
  } catch {
    /* keep empty */
  }
  return (
    <div className="border-b border-border px-3 py-1.5">
      <div className="truncate text-sm font-medium">{title || host || url}</div>
      {host && title ? (
        <div className="truncate text-xs text-muted-foreground">{host}</div>
      ) : null}
    </div>
  )
}
