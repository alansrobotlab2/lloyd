import { useCallback, useEffect, useRef, useState } from 'react'
import {
  ArrowRight, Camera, RefreshCw, ExternalLink, Globe, Crosshair, Loader2,
  Eye, EyeOff, Keyboard, Send,
} from 'lucide-react'
import { api, type BrowserFrame, type BrowserInputEvent } from '../../api'
import { useReportMcFocus } from '../../contexts/McUiContext'
import { Button } from '../ui/button'
import { cn } from '@/lib/utils'
import { playwrightKey } from '@/lib/browserKeys'

// Mirrors the agent's live browser session (#278).
//
// The agent drives a headless Chromium and POSTs a frame — JPEG viewport plus
// the annotated accessibility snapshot — after every browser_* tool call. We
// subscribe to /api/browser/state and render the newest one, so watching a
// browse session costs the user a tab instead of a second browser window on
// the box's display.
//
// The URL bar is the first control. It POSTs to /api/browser/navigate,
// which the backend proxies to the aggregator — Playwright runs in that
// process, so a control action has to cross the seam. It is not an MCP call:
// the user typing a URL is not the agent using a tool, and dispatching it as
// one would write a browser_navigate into the transcript nobody made.
//
// The frame itself is the second control (POST /api/browser/input, the same
// seam): a click on the screenshot is a click at that viewport pixel, and while
// the frame has focus the keyboard, paste and the wheel go to the page. That is
// what lets a human answer a login the agent cannot — so what is typed here is
// a credential, and nothing on this path logs it. The "type into page" row is
// the same `type` event for a phone, whose keyboard a focused <div> never opens.
//
// The overlay shows which regions the agent's `eN` refs point at; it is not a
// control — refs come from an a11y tree we no longer hold by render time.
const REFS_WITHOUT_GEOMETRY = 0

// Printable keystrokes arrive one keydown at a time; a short window turns a
// typed word into one request and one frame instead of one per letter.
const TYPE_FLUSH_MS = 120
const SCROLL_FLUSH_MS = 120

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
  const [urlInput, setUrlInput] = useState('')
  const [urlDirty, setUrlDirty] = useState(false)
  const [navigating, setNavigating] = useState(false)
  const [navError, setNavError] = useState<string | null>(null)
  const urlRef = useRef<HTMLInputElement>(null)
  const imgRef = useRef<HTMLImageElement>(null)
  const bodyRef = useRef<HTMLDivElement>(null)
  const [containerW, setContainerW] = useState(0)
  const frameRef = useRef<HTMLDivElement>(null)
  const [frameFocused, setFrameFocused] = useState(false)
  const [inFlight, setInFlight] = useState(0)
  const [inputError, setInputError] = useState<string | null>(null)
  const [sendText, setSendText] = useState('')
  const [revealText, setRevealText] = useState(false)
  // Events go out strictly in order: a password typed then Enter must not
  // race into Enter-then-password.
  const queue = useRef<Promise<void>>(Promise.resolve())
  const textBuf = useRef('')
  const textTimer = useRef<number | null>(null)
  const scrollAcc = useRef({ x: 0, y: 0, dx: 0, dy: 0 })
  const scrollTimer = useRef<number | null>(null)
  // The frame wrapper mounts with the first screenshot; the wheel listener
  // has to be attached to it then, not at page mount.
  const hasFrame = Boolean(frame?.screenshot_b64)

  useReportMcFocus('browser', frame?.url ? { kind: 'url', id: frame.url } : null)

  // Newest frame wins; a burst of tool calls shouldn't queue a slideshow.
  const seq = useRef(0)
  const applyFrame = useCallback((next: BrowserFrame) => {
    seq.current += 1
    setFrame(next)
    setError(null)
  }, [])

  // A frame lands after every browser_* tool call, so binding the input
  // straight to frame.url would wipe whatever the user is halfway through
  // typing the moment the agent navigates. Re-seed only while the bar is
  // clean; what they typed survives until they submit it or press Escape.
  useEffect(() => {
    if (!urlDirty) setUrlInput(frame?.url ?? '')
  }, [frame?.url, urlDirty])

  const submitUrl = useCallback(async (e: React.FormEvent) => {
    e.preventDefault()
    const target = urlInput.trim()
    if (!target || navigating) return
    setNavigating(true)
    setNavError(null)
    try {
      const res = await api.browserNavigate(target)
      // A bad host or a timeout comes back 200 with `error` — the request
      // was fine, the page wasn't.
      if (res.error) setNavError(res.error)
      // Clean again on success, so the frame this navigation pushes re-seeds
      // the bar with where we actually landed: redirects, added scheme and
      // trailing-slash normalisation included.
      else setUrlDirty(false)
    } catch (err) {
      setNavError(err instanceof Error ? err.message : String(err))
    } finally {
      setNavigating(false)
    }
  }, [urlInput, navigating])

  const enqueue = useCallback((event: BrowserInputEvent) => {
    setInFlight(n => n + 1)
    queue.current = queue.current
      .then(async () => {
        try {
          const res = await api.browserInput(event)
          setInputError(res.error ?? null)
        } catch (err) {
          setInputError(err instanceof Error ? err.message : String(err))
        }
      })
      .finally(() => setInFlight(n => n - 1))
  }, [])

  const flushText = useCallback(() => {
    if (textTimer.current !== null) {
      window.clearTimeout(textTimer.current)
      textTimer.current = null
    }
    const text = textBuf.current
    textBuf.current = ''
    if (text) enqueue({ kind: 'type', text })
  }, [enqueue])

  const queueText = useCallback((text: string) => {
    textBuf.current += text
    if (textTimer.current !== null) window.clearTimeout(textTimer.current)
    textTimer.current = window.setTimeout(flushText, TYPE_FLUSH_MS)
  }, [flushText])

  const sendKey = useCallback((key: string) => {
    flushText()
    enqueue({ kind: 'key', key })
  }, [enqueue, flushText])

  // Screen pixels of the rendered <img> → the frame's own pixels, which are
  // the agent browser's viewport pixels.
  const toFrame = useCallback((clientX: number, clientY: number) => {
    const img = imgRef.current
    if (!img || !img.naturalWidth) return null
    const r = img.getBoundingClientRect()
    if (r.width === 0 || r.height === 0) return null
    return {
      x: Math.round((clientX - r.left) * (img.naturalWidth / r.width)),
      y: Math.round((clientY - r.top) * (img.naturalHeight / r.height)),
    }
  }, [])

  const onFrameClick = useCallback((e: React.MouseEvent, button: 'left' | 'right' | 'middle') => {
    const p = toFrame(e.clientX, e.clientY)
    if (!p) return
    flushText()
    enqueue({ kind: 'click', ...p, button, count: Math.min(3, Math.max(1, e.detail || 1)) })
  }, [enqueue, flushText, toFrame])

  const onFrameKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.nativeEvent.isComposing) return
    if (['Shift', 'Control', 'Alt', 'Meta', 'AltGraph', 'CapsLock'].includes(e.key)) return
    // Paste arrives as its own event with the clipboard text; let it.
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'v') return
    const altGraph = e.getModifierState?.('AltGraph')
    if (e.key.length === 1 && (altGraph || (!e.ctrlKey && !e.metaKey))) {
      e.preventDefault()
      queueText(e.key)
      return
    }
    const key = playwrightKey(e)
    if (!key) return
    // Tab must not move focus out of the frame, Backspace must not navigate
    // Mission Control, Ctrl+R must reload the page in there, not this one.
    e.preventDefault()
    sendKey(key)
  }, [queueText, sendKey])

  const onFramePaste = useCallback((e: React.ClipboardEvent) => {
    const text = e.clipboardData.getData('text/plain')
    if (!text) return
    e.preventDefault()
    queueText(text)
    flushText()
  }, [flushText, queueText])

  // React's onWheel is passive, so it cannot stop Mission Control scrolling
  // under the page that is being scrolled. A native listener can.
  useEffect(() => {
    const el = frameRef.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      const p = toFrame(e.clientX, e.clientY)
      if (!p) return
      e.preventDefault()
      const unit = e.deltaMode === 1 ? 40 : e.deltaMode === 2 ? 800 : 1
      const acc = scrollAcc.current
      acc.x = p.x
      acc.y = p.y
      acc.dx += e.deltaX * unit
      acc.dy += e.deltaY * unit
      if (scrollTimer.current !== null) return
      scrollTimer.current = window.setTimeout(() => {
        scrollTimer.current = null
        const { x, y, dx, dy } = scrollAcc.current
        scrollAcc.current = { x, y, dx: 0, dy: 0 }
        if (dx || dy) enqueue({ kind: 'scroll', x, y, dx: Math.round(dx), dy: Math.round(dy) })
      }, SCROLL_FLUSH_MS)
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [enqueue, toFrame, hasFrame])

  useEffect(() => () => {
    if (textTimer.current !== null) window.clearTimeout(textTimer.current)
    if (scrollTimer.current !== null) window.clearTimeout(scrollTimer.current)
  }, [])

  const submitSendText = useCallback((e: React.FormEvent) => {
    e.preventDefault()
    if (!sendText) return
    flushText()
    enqueue({ kind: 'type', text: sendText })
    setSendText('')
  }, [enqueue, flushText, sendText])

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
        {frameFocused && (
          <span className="flex items-center gap-1 rounded-full bg-sky-500/15 px-2 py-0.5 text-[10px] font-medium text-sky-400">
            <Keyboard className="h-3 w-3" />
            keys go to the page
          </span>
        )}
        {inFlight > 0 && <Loader2 className="h-3.5 w-3.5 animate-spin text-muted-foreground" />}
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

      <form onSubmit={submitUrl} className="flex items-center gap-2 border-b border-border/40 px-4 py-1.5">
        <input
          ref={urlRef}
          value={urlInput}
          onChange={e => {
            setUrlInput(e.target.value)
            setUrlDirty(true)
          }}
          onKeyDown={e => {
            if (e.key !== 'Escape') return
            e.preventDefault()
            setUrlDirty(false)
            setUrlInput(frame?.url ?? '')
            urlRef.current?.blur()
          }}
          spellCheck={false}
          autoComplete="off"
          autoCorrect="off"
          placeholder="Type a URL and press Enter"
          aria-label="Browser address"
          className="min-w-0 flex-1 rounded border border-border/60 bg-muted/30 px-2 py-1 font-mono text-[11px] text-foreground outline-none transition-colors placeholder:text-muted-foreground focus:border-ring"
        />
        <Button
          type="submit"
          variant="ghost"
          size="sm"
          className="h-7 shrink-0 gap-1 text-xs"
          disabled={navigating || !urlInput.trim()}
        >
          {navigating
            ? <Loader2 className="h-3.5 w-3.5 animate-spin" />
            : <ArrowRight className="h-3.5 w-3.5" />}
          go
        </Button>
      </form>

      {navError && (
        <div className="border-b border-border/40 px-4 py-1 text-[11px] text-destructive">
          {navError}
        </div>
      )}

      {frame?.screenshot_b64 && (
        <form onSubmit={submitSendText} className="flex items-center gap-1 border-b border-border/40 px-4 py-1.5">
          <input
            value={sendText}
            onChange={e => setSendText(e.target.value)}
            type={revealText ? 'text' : 'password'}
            spellCheck={false}
            autoComplete="off"
            autoCorrect="off"
            autoCapitalize="off"
            placeholder="Type into the page (click a field first)"
            aria-label="Text to type into the page"
            className="min-w-0 flex-1 rounded border border-border/60 bg-muted/30 px-2 py-1 font-mono text-[11px] text-foreground outline-none transition-colors placeholder:text-muted-foreground focus:border-ring"
          />
          <Button type="button" variant="ghost" size="sm" className="h-7 w-7 shrink-0 p-0"
            aria-label={revealText ? 'Hide text' : 'Show text'}
            onClick={() => setRevealText(v => !v)}>
            {revealText ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
          </Button>
          <Button type="submit" variant="ghost" size="sm" className="h-7 shrink-0 gap-1 text-xs" disabled={!sendText}>
            <Send className="h-3.5 w-3.5" />
            type
          </Button>
          {(['Tab', 'Enter', 'Backspace'] as const).map(k => (
            <Button key={k} type="button" variant="ghost" size="sm" className="h-7 shrink-0 px-2 font-mono text-[11px]"
              onClick={() => sendKey(k)}>
              {k === 'Backspace' ? '⌫' : k}
            </Button>
          ))}
        </form>
      )}

      {inputError && (
        <div className="border-b border-border/40 px-4 py-1 text-[11px] text-destructive">
          {inputError}
        </div>
      )}

      <div ref={bodyRef} className="min-h-0 flex-1 overflow-auto p-4">
        {frame?.screenshot_b64 ? (
          <div
            ref={frameRef}
            tabIndex={0}
            role="application"
            aria-label="Agent browser — click to interact; keys go to the page while focused"
            onKeyDown={onFrameKeyDown}
            onPaste={onFramePaste}
            onFocus={() => setFrameFocused(true)}
            onBlur={() => { flushText(); setFrameFocused(false) }}
            className={cn(
              'relative inline-block max-w-full cursor-pointer rounded outline-none',
              frameFocused && 'ring-2 ring-sky-500/70',
            )}
          >
            <img
              ref={imgRef}
              draggable={false}
              onClick={e => onFrameClick(e, 'left')}
              onAuxClick={e => { if (e.button === 1) { e.preventDefault(); onFrameClick(e, 'middle') } }}
              onContextMenu={e => { e.preventDefault(); onFrameClick(e, 'right') }}
              alt={frame.title || 'agent browser screenshot'}
              src={`data:${frame.mime || 'image/jpeg'};base64,${frame.screenshot_b64}`}
              onLoad={e => {
                const el = e.currentTarget
                setNatural({ w: el.naturalWidth, h: el.naturalHeight })
              }}
              className="block max-w-full select-none rounded border border-border/60"
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
              Every browser_* tool call pushes a frame here automatically — or type a URL above.
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
