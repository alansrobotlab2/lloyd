import { useEffect, useRef, useState } from "react"
import ChatPanel from "@/components/ChatPanel"

interface FocusMessage {
  type: "focus-session"
  windowId: number
  tabId: number
  sessionKey: string | null
  url?: string
  title?: string
}

interface PanelState {
  sessionKey: string | null
  url: string | null
  title: string | null
}

const INITIAL: PanelState = { sessionKey: null, url: null, title: null }

export default function SidePanelApp() {
  const [state, setState] = useState<PanelState>(INITIAL)
  const windowIdRef = useRef<number | null>(null)
  const portRef = useRef<chrome.runtime.Port | null>(null)

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
        })
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
    }
  }, [])

  if (!state.sessionKey) {
    return <EmptyState />
  }

  return (
    <div className="flex h-screen flex-col bg-background text-foreground">
      <PageHeader url={state.url} title={state.title} />
      <div className="min-h-0 flex-1">
        <ChatPanel
          requestedSessionKey={state.sessionKey}
          compact
        />
      </div>
    </div>
  )
}

function EmptyState() {
  return (
    <div className="flex h-screen items-center justify-center bg-background p-6 text-center text-foreground">
      <div className="space-y-2">
        <div className="text-lg font-medium">Lloyd is watching this tab</div>
        <div className="text-sm text-muted-foreground">
          Open a webpage and I'll fetch it and give you the highlights.
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
    <div className="border-b border-border bg-card/40 px-3 py-2">
      <div className="truncate text-sm font-medium">
        {title || host || url}
      </div>
      {host && title ? (
        <div className="truncate text-xs text-muted-foreground">{host}</div>
      ) : null}
    </div>
  )
}
