import { useCallback, useEffect, useRef, useState } from 'react'
import { Camera, RefreshCw, ExternalLink, Globe, Crosshair } from 'lucide-react'
import { api, type BrowserFrame } from '../../api'
import { useReportMcFocus } from '../../contexts/McUiContext'
import { Button } from '../ui/button'
import { cn } from '@/lib/utils'

// Mirrors the agent's live browser session (#278).
//
// The agent drives a headless Chromium and POSTs a frame — JPEG viewport plus
// the annotated accessibility snapshot — after every browser_* tool call. We
// subscribe to /api/browser/state and render the newest one, so watching a
// browse session costs the user a tab instead of a second browser window on
// the box's display.
//
// Deliberately read-only: the overlay shows which regions the agent's `eN`
// refs point at, but clicking them does nothing. There is no backend route
// that accepts an injected click, and the tool surface has no "click pixel
// (x,y)" — refs come from an a11y tree we no longer hold. Driving the page is
// the agent's job; this page's job is to let you see what it is doing.
const REFS_WITHOUT_GEOMETRY = 0

function RefBadge({ label }: { label: string }) {
  return (
    <span className="absolute -top-2 left-0 -translate-y-full rounded bg-amber-500 px-1 py-0 text-[10px] font-semibold leading-4 text-black shadow">
      {label}
    </span>
  )
}

export default function BrowserPage() {
  const [frame, setFrame] = useState<BrowserFrame | null>(null)
  const [connected, setConnected] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [natural, setNatural] = useState({ w: 0, h: 0 })
  const [showOverlay, setShowOverlay] = useState(true)
  const imgRef = useRef<HTMLImageElement>(null)
  const bodyRef = useRef<HTMLDivElement>(null)
  const [containerW, setContainerW] = useState(0)

  useReportMcFocus('browser', frame?.url ? { kind: 'url', id: frame.url } : null)

  // Newest frame wins; a burst of tool calls shouldn't queue a slideshow.
  const seq = useRef(0)
  const applyFrame = useCallback((next: BrowserFrame) => {
    seq.current += 1
    setFrame(next)
    setError(null)
  }, [])

  useEffect(() => {
    let es: EventSource | null = null
    let closed = false
    let retry: number | null = null

    const connect = async () => {
      if (closed) return
      try {
        const current = await api.getBrowserFrame()
        if (current?.active) applyFrame(current)
      } catch {
        // Cold backend or the route isn't up yet — the stream will still
        // deliver the next push.
      }
      if (closed) return
      setLoading(false)
      es = new EventSource('/api/browser/state')
      es.addEventListener('hello', () => setConnected(true))
      es.addEventListener('state', (ev: MessageEvent) => {
        try {
          const data = JSON.parse(ev.data) as BrowserFrame
          if (data && typeof data === 'object') applyFrame(data)
        } catch {
          // Ignore malformed payloads.
        }
      })
      es.onerror = () => {
        setConnected(false)
        if (es) {
          es.close()
          es = null
        }
        if (!closed) retry = window.setTimeout(connect, 2000)
      }
    }

    void connect()
    return () => {
      closed = true
      if (retry !== null) window.clearTimeout(retry)
      if (es) es.close()
    }
  }, [applyFrame])

  // Track the frame area so ref boxes can be scaled from screenshot pixels
  // down to whatever the <img> is actually rendered at.
  useEffect(() => {
    const el = bodyRef.current
    if (!el) return
    const ro = new ResizeObserver(entries => {
      for (const e of entries) setContainerW(e.contentRect.width)
    })
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const scale = natural.w > 0 && containerW > 0 ? containerW / natural.w : 0
  const boxes = frame?.refs?.filter(r => typeof r.x === 'number' && typeof r.y === 'number') ?? []
  const hiddenRefs = (frame?.refs?.length ?? 0) - boxes.length - REFS_WITHOUT_GEOMETRY

  return (
    <div className="flex h-full min-h-0 flex-col bg-background">
      <header className="flex items-center gap-2 border-b border-border/60 px-4 py-2">
        <Globe className="h-4 w-4 text-muted-foreground" />
        <span className="text-sm font-semibold text-foreground">Browser</span>
        <span
          className={cn(
            'rounded-full px-2 py-0.5 text-[10px] font-medium',
            connected ? 'bg-emerald-500/15 text-emerald-400' : 'bg-muted text-muted-foreground',
          )}
        >
          {connected ? 'streaming' : 'offline'}
        </span>
        {frame?.tool && (
          <span className="truncate font-mono text-[11px] text-muted-foreground">{frame.tool}</span>
        )}
        <div className="ml-auto flex items-center gap-1">
          <Button
            variant="ghost"
            size="sm"
            className="h-7 gap-1 text-xs"
            onClick={() => setShowOverlay(v => !v)}
            disabled={boxes.length === 0}
          >
            <Crosshair className="h-3.5 w-3.5" />
            refs
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="h-7 gap-1 text-xs"
            onClick={() => {
              setLoading(true)
              void api.getBrowserFrame().then(f => {
                if (f?.active) applyFrame(f)
                setLoading(false)
              }).catch(() => setLoading(false))
            }}
          >
            <RefreshCw className={cn('h-3.5 w-3.5', loading && 'animate-spin')} />
            refresh
          </Button>
          {frame?.url && (
            <a
              href={frame.url}
              target="_blank"
              rel="noreferrer noopener"
              className="flex h-7 items-center gap-1 rounded px-2 text-xs text-muted-foreground hover:text-foreground"
            >
              <ExternalLink className="h-3.5 w-3.5" />
              open
            </a>
          )}
        </div>
      </header>

      <div className="truncate border-b border-border/40 px-4 py-1 font-mono text-[11px] text-muted-foreground">
        {frame?.url ?? 'no browser session yet'}
      </div>

      <div ref={bodyRef} className="min-h-0 flex-1 overflow-auto p-4">
        {frame?.screenshot_b64 ? (
          <div className="relative inline-block max-w-full">
            <img
              ref={imgRef}
              alt={frame.title || 'agent browser screenshot'}
              src={`data:${frame.mime || 'image/jpeg'};base64,${frame.screenshot_b64}`}
              onLoad={e => {
                const el = e.currentTarget
                setNatural({ w: el.naturalWidth, h: el.naturalHeight })
              }}
              className="block max-w-full rounded border border-border/60"
            />
            {showOverlay && scale > 0 && (
              <div className="pointer-events-none absolute inset-0">
                {boxes.map(r => (
                  <div
                    key={r.ref}
                    className="absolute rounded border-2 border-amber-400/80 bg-amber-400/10"
                    style={{
                      left: (r.x ?? 0) * scale,
                      top: (r.y ?? 0) * scale,
                      width: Math.max(8, (r.w ?? 0) * scale),
                      height: Math.max(8, (r.h ?? 0) * scale),
                    }}
                  >
                    <RefBadge label={r.ref} />
                  </div>
                ))}
              </div>
            )}
          </div>
        ) : (
          <div className="flex h-64 flex-col items-center justify-center gap-2 rounded border border-dashed border-border/60 text-sm text-muted-foreground">
            <Camera className="h-6 w-6" />
            <span>Waiting for the agent to open a page.</span>
            <span className="text-xs">
              Every browser_* tool call pushes a frame here automatically.
            </span>
          </div>
        )}
        {error && <div className="mt-3 text-xs text-destructive">{error}</div>}
        {hiddenRefs > 0 && (
          <div className="mt-2 text-[11px] text-muted-foreground">
            {hiddenRefs} refs have no on-screen geometry (offscreen or not rendered).
          </div>
        )}
      </div>

      {frame?.snapshot && (
        <details className="max-h-1/3 shrink-0 overflow-auto border-t border-border/60 px-4 py-2">
          <summary className="cursor-pointer text-xs font-medium text-muted-foreground">
            accessibility snapshot
          </summary>
          <pre className="mt-2 whitespace-pre-wrap font-mono text-[11px] text-muted-foreground">
            {frame.snapshot}
          </pre>
        </details>
      )}
    </div>
  )
}
