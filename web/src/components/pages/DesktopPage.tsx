import { useCallback, useEffect, useRef, useState } from 'react'
import { Hand, Monitor, ShieldCheck, ShieldOff, Timer } from 'lucide-react'
import { api, type DesktopFrame, type DesktopLease } from '../../api'
import { useReportMcFocus } from '../../contexts/McUiContext'
import { Button } from '../ui/button'
import { cn } from '@/lib/utils'
import { formatRemaining } from '../../lib/desktopLease'

// Desktop computer use: what Lloyd last captured of the real desktop, and the
// lease that lets him act on it (app/desktop_lease.py).
//
// The frame is pushed by desktop_capture / desktop_act after each capture,
// the same pipeline the Browser tab uses. Unlike the browser, a capture here
// carries element geometry, so the overlay shows exactly what each element
// index the model was given points at.
//
// The lease toggle is the only control, and it is deliberately the only way a
// lease is granted: no tool may call the route (safety.check_bash_command and
// the aggregator refuse it). Moving the mouse also takes the seat back — the
// tripwire revokes the lease at Lloyd's next action.

const GRANT_CHOICES = [10, 30, 60]


export default function DesktopPage() {
  const [frame, setFrame] = useState<DesktopFrame | null>(null)
  const [lease, setLease] = useState<DesktopLease | null>(null)
  const [connected, setConnected] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showOverlay, setShowOverlay] = useState(true)
  const [containerW, setContainerW] = useState(0)
  const bodyRef = useRef<HTMLDivElement>(null)

  useReportMcFocus('desktop', null)

  useEffect(() => {
    let es: EventSource | null = null
    let stopped = false
    let retry: ReturnType<typeof setTimeout> | null = null
    api.getDesktopFrame().then(f => {
      if (f.lease) setLease(f.lease)
      if (f.active) setFrame(f)
    }).catch(() => { /* the stream will fill it */ })
    const open = () => {
      es = new EventSource('/api/desktop/state')
      es.onopen = () => setConnected(true)
      es.addEventListener('state', (e) => {
        try { setFrame(JSON.parse((e as MessageEvent).data)) } catch { /* skip */ }
      })
      es.addEventListener('lease', (e) => {
        try { setLease(JSON.parse((e as MessageEvent).data)) } catch { /* skip */ }
      })
      es.onerror = () => {
        setConnected(false)
        es?.close()
        if (!stopped) retry = setTimeout(open, 3000)
      }
    }
    open()
    return () => { stopped = true; es?.close(); if (retry) clearTimeout(retry) }
  }, [])

  useEffect(() => {
    const el = bodyRef.current
    if (!el) return
    const ro = new ResizeObserver(() => setContainerW(el.clientWidth))
    ro.observe(el)
    setContainerW(el.clientWidth)
    return () => ro.disconnect()
  }, [])

  const setLeaseOp = useCallback(async (op: 'grant' | 'revoke', minutes?: number) => {
    setBusy(true)
    setError(null)
    try {
      setLease(await api.setDesktopLease(op, minutes))
    } catch (e) {
      setError(String(e))
    } finally {
      setBusy(false)
    }
  }, [])

  const agentHolds = !!lease?.agent_holds
  const w = frame?.width || 0
  const h = frame?.height || 0
  const scale = w && containerW ? Math.min(1, containerW / w) : 1

  return (
    <div className="flex h-full flex-col">
      <div className="flex flex-wrap items-center gap-2 border-b border-border px-4 py-2">
        <Monitor className="h-4 w-4 text-muted-foreground" />
        <span className="font-semibold">Desktop</span>
        <span className={cn('text-xs', connected ? 'text-emerald-500' : 'text-muted-foreground')}>
          {connected ? 'live' : 'reconnecting…'}
        </span>
        <div className="ml-auto flex items-center gap-2">
          {agentHolds ? (
            <>
              <span className="flex items-center gap-1 rounded bg-amber-500/15 px-2 py-1 text-xs text-amber-500">
                <ShieldCheck className="h-3.5 w-3.5" />
                Lloyd has the desktop
                {lease?.remaining_s != null && (
                  <span className="ml-1 flex items-center gap-0.5 font-mono">
                    <Timer className="h-3 w-3" />{formatRemaining(lease.remaining_s)}
                  </span>
                )}
              </span>
              <Button size="sm" variant="destructive" disabled={busy}
                      onClick={() => setLeaseOp('revoke')}>
                <Hand className="mr-1 h-3.5 w-3.5" />Take back
              </Button>
            </>
          ) : (
            <>
              <span className="flex items-center gap-1 text-xs text-muted-foreground">
                <ShieldOff className="h-3.5 w-3.5" />
                You have the desktop{lease?.reason && lease.reason !== 'granted' ? ` (${lease.reason})` : ''}
              </span>
              {GRANT_CHOICES.map(m => (
                <Button key={m} size="sm" variant="outline" disabled={busy}
                        onClick={() => setLeaseOp('grant', m)}>
                  Let Lloyd act {m}m
                </Button>
              ))}
            </>
          )}
          <Button size="sm" variant="ghost" onClick={() => setShowOverlay(v => !v)}>
            {showOverlay ? 'Hide' : 'Show'} elements
          </Button>
        </div>
      </div>
      {error && <div className="px-4 py-1 text-xs text-destructive">{error}</div>}
      <div className="px-4 py-1 text-xs text-muted-foreground">
        {frame?.window?.title
          ? <>Last capture: <span className="text-foreground">{frame.window.class}</span> — {frame.window.title}
              {frame.tool ? ` · by ${frame.tool}` : ''}
              {frame.ts ? ` · ${new Date(frame.ts * 1000).toLocaleTimeString()}` : ''}</>
          : frame?.kind === 'screen' ? 'Last capture: whole screen' : 'Lloyd has not looked at the desktop yet.'}
      </div>
      <div ref={bodyRef} className="relative flex-1 overflow-auto px-4 pb-4">
        {frame?.image_b64 ? (
          <div className="relative" style={{ width: w * scale, height: h * scale }}>
            <img src={`data:${frame.mime || 'image/png'};base64,${frame.image_b64}`}
                 alt="Lloyd's last desktop capture" width={w * scale} height={h * scale}
                 className="rounded border border-border" />
            {showOverlay && (frame.elements || []).map(el => el.bounds && (
              <div key={el.index} title={`#${el.index} ${el.role} ${el.name}`}
                   className="absolute border border-amber-400/70 bg-amber-400/5"
                   style={{ left: el.bounds[0] * scale, top: el.bounds[1] * scale,
                            width: el.bounds[2] * scale, height: el.bounds[3] * scale }}>
                <span className="absolute -top-3.5 left-0 rounded bg-amber-500 px-0.5 text-[9px] font-semibold leading-3 text-black">
                  {el.index}
                </span>
              </div>
            ))}
          </div>
        ) : frame?.summary ? (
          <pre className="whitespace-pre-wrap text-xs">{frame.summary}</pre>
        ) : null}
      </div>
    </div>
  )
}
