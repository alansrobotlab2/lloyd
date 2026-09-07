import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  ChevronDown, ChevronRight, Folder, FolderOpen, ArrowUp, Home, HardDrive,
  Loader2, AlertCircle, Check, EyeOff, Eye,
} from 'lucide-react'
import { api } from '../../api'
import { cn } from '@/lib/utils'

/**
 * Directory selector modal.
 *
 * Browses the server filesystem (directories only) through GET /api/ide/list,
 * lazily per expanded folder, and hands one absolute path back to the caller
 * via `onPick`. Keyboard: ↑/↓ move, → expand, ← collapse/jump to parent,
 * Enter select, Esc close. The path bar takes any absolute path plus
 * `~`/`$VARS` (expanded server-side); Enter there navigates rather than
 * selecting.
 */

export interface DirectoryPickerProps {
  open: boolean
  onClose: () => void
  /** Where to start browsing. Absolute, or `~/…`. Empty → the server's home. */
  initialPath?: string | null
  /** Receives the confirmed absolute directory path. */
  onPick: (absPath: string) => void
  title?: string
  confirmLabel?: string
}

interface DirState {
  loading: boolean
  error: string | null
  /** Subdirectory names, as served (dirs first, case-insensitive). */
  dirs: string[] | null
  truncated: boolean
}

const EMPTY_DIR: DirState = { loading: false, error: null, dirs: null, truncated: false }
// Per-folder display cap. The API caps listings at 5000; neither is a
// problem you want to scroll through, so cut the render short too.
const MAX_SHOWN = 400

function joinPath(parent: string, name: string) {
  return parent.endsWith('/') ? parent + name : parent + '/' + name
}

function parentOf(path: string) {
  const norm = path.replace(/\/+$/, '')
  const idx = norm.lastIndexOf('/')
  if (idx <= 0) return '/'
  return norm.slice(0, idx)
}

function basename(path: string) {
  const norm = path.replace(/\/+$/, '')
  const idx = norm.lastIndexOf('/')
  return idx >= 0 ? norm.slice(idx + 1) : norm
}

/** Ancestor chain from '/' down to `path` inclusive, for the breadcrumb. */
function ancestry(path: string): string[] {
  const out: string[] = []
  let cur = path.replace(/\/+$/, '') || '/'
  for (;;) {
    out.unshift(cur)
    if (cur === '/') break
    cur = parentOf(cur)
  }
  return out
}

interface Row {
  path: string
  name: string
  depth: number
  expanded: boolean
  loading: boolean
  error: string | null
  hiddenCount: number
  omitted: number
  truncated: boolean
}

export default function DirectoryPicker({
  open, onClose, initialPath, onPick,
  title = 'Choose a directory', confirmLabel = 'Open',
}: DirectoryPickerProps) {
  const [root, setRoot] = useState<string | null>(null)
  const [pathInput, setPathInput] = useState('')
  const [expanded, setExpanded] = useState<Record<string, boolean>>({})
  const [cache, setCache] = useState<Record<string, DirState>>({})
  const [showHidden, setShowHidden] = useState(false)
  const [rootError, setRootError] = useState<string | null>(null)
  const [highlight, setHighlight] = useState(0)
  // Path to focus once its row appears (used by "up one level" and breadcrumb
  // jumps so the previous location stays visible after re-rooting).
  const [reveal, setReveal] = useState<string | null>(null)
  const pathRef = useRef<HTMLInputElement>(null)
  const rowRefs = useRef<Record<string, HTMLButtonElement | null>>({})

  const loadDir = useCallback((dir: string) => {
    setCache(prev => ({ ...prev, [dir]: { ...EMPTY_DIR, loading: true } }))
    api.ideList(dir)
      .then(r => setCache(prev => ({
        ...prev,
        [dir]: {
          loading: false,
          error: null,
          dirs: r.entries.filter(e => e.isDir).map(e => e.name),
          truncated: r.truncated,
        },
      })))
      .catch(e => setCache(prev => ({
        ...prev,
        [dir]: {
          loading: false,
          error: e instanceof Error ? e.message : String(e),
          dirs: null,
          truncated: false,
        },
      })))
  }, [])

  /**
   * Re-root at `dir`, which must already be the absolute path the server
   * resolved. Expands `toReveal`'s chain beneath it so the caller's previous
   * location is still on screen after re-rooting.
   */
  const route = useCallback((dir: string, toReveal?: string) => {
    setRoot(dir)
    setPathInput(dir)
    setHighlight(0)
    loadDir(dir)
    const next: Record<string, boolean> = { [dir]: true }
    if (toReveal && toReveal !== dir && (toReveal + '/').startsWith(dir + '/')) {
      const rest = toReveal.slice(dir.length).replace(/^\/+/, '')
      let cur = dir
      for (const part of rest.split('/').filter(Boolean)) {
        cur = joinPath(cur, part)
        next[cur] = true
        loadDir(cur)
      }
    }
    setExpanded(prev => ({ ...prev, ...next }))
    if (toReveal) setReveal(toReveal)
  }, [loadDir])

  /**
   * Navigate to a user-supplied path. `~` and `$VARS` are expanded
   * server-side, so fetch once and adopt the canonical path the response
   * carries — everything downstream (rows, reveal, `onPick`) then deals only
   * in absolute paths. A failed fetch still routes, so the row shows the
   * error inline rather than the dialog silently doing nothing.
   */
  const navigate = useCallback((target: string, toReveal?: string) => {
    const trimmed = target.trim()
    if (!trimmed) return
    setRootError(null)
    api.ideList(trimmed)
      .then(r => {
        let reveal = toReveal
        if (reveal && r.path !== trimmed && reveal.startsWith(trimmed)) {
          reveal = joinPath(r.path, reveal.slice(trimmed.length).replace(/^\/+/, ''))
        }
        route(r.path, reveal)
      })
      .catch(e => {
        setRootError(e instanceof Error ? e.message : String(e))
        route(trimmed, toReveal)
      })
  }, [route])

  // Seed on open from initialPath, else from the server's home dir
  // (`~` is expanded server-side; navigate adopts the absolute form).
  useEffect(() => {
    if (!open) return
    setCache({})
    setExpanded({})
    setReveal(null)
    setHighlight(0)
    const seed = (initialPath ?? '').trim()
    setRoot(null)
    setPathInput(seed)
    setRootError(null)
    navigate(seed || '~')
    const t = setTimeout(() => pathRef.current?.focus(), 0)
    return () => clearTimeout(t)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  const toggle = useCallback((dir: string) => {
    setExpanded(prev => {
      if (!prev[dir] && !cache[dir]?.dirs && !cache[dir]?.loading) loadDir(dir)
      return { ...prev, [dir]: !prev[dir] }
    })
  }, [cache, loadDir])

  const rows = useMemo(() => {
    const out: Row[] = []
    if (!root) return out
    const walk = (path: string, depth: number) => {
      const st = cache[path] ?? EMPTY_DIR
      const all = st.dirs ?? []
      const visible = showHidden ? all : all.filter(d => !d.startsWith('.'))
      out.push({
        path,
        name: basename(path) || path,
        depth,
        expanded: !!expanded[path],
        loading: st.loading,
        error: st.error,
        hiddenCount: all.length - visible.length,
        omitted: Math.max(0, visible.length - MAX_SHOWN),
        truncated: st.truncated,
      })
      if (expanded[path]) {
        visible.slice(0, MAX_SHOWN).forEach(name => walk(joinPath(path, name), depth + 1))
      }
    }
    walk(root, 0)
    return out
  }, [root, cache, expanded, showHidden])

  // Clamp the highlight to the row list, then honour a pending reveal.
  useEffect(() => {
    setHighlight(h => Math.min(h, Math.max(rows.length - 1, 0)))
    if (reveal) {
      const idx = rows.findIndex(r => r.path === reveal)
      if (idx >= 0) {
        setHighlight(idx)
        setReveal(null)
      }
    }
  }, [rows, reveal])

  // Single source of truth for "which directory would be returned".
  const selection = rows[highlight]?.path ?? root ?? ''

  const crumbs = useMemo(
    () => ancestry(selection || root || '/'),
    [selection, root],
  )

  const confirm = useCallback((path: string) => {
    const target = path.trim()
    if (!target) return
    onPick(target)
    onClose()
  }, [onPick, onClose])

  // Modal-wide keys. The path bar owns its own Enter (navigate, not select).
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') { e.preventDefault(); onClose(); return }
      if (e.target === pathRef.current) return
      if (e.key === 'ArrowDown') {
        e.preventDefault()
        setHighlight(i => Math.min(i + 1, rows.length - 1))
      } else if (e.key === 'ArrowUp') {
        e.preventDefault()
        setHighlight(i => Math.max(i - 1, 0))
      } else if (e.key === 'ArrowRight') {
        e.preventDefault()
        const r = rows[highlight]
        if (r && !r.expanded) toggle(r.path)
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault()
        const r = rows[highlight]
        if (!r) return
        if (r.expanded) { toggle(r.path); return }
        const idx = rows.findIndex(x => x.path === parentOf(r.path))
        if (idx >= 0) setHighlight(idx)
      } else if (e.key === 'Enter') {
        e.preventDefault()
        confirm(rows[highlight]?.path ?? '')
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, rows, highlight, toggle, confirm, onClose])

  // Keep the highlighted row on screen during arrow navigation.
  useEffect(() => {
    const r = rows[highlight]
    if (r) rowRefs.current[r.path]?.scrollIntoView({ block: 'nearest' })
  }, [highlight, rows])

  if (!open) return null

  const goParent = () => {
    const here = selection || root
    if (!here) return
    const up = parentOf(here)
    if (up === here) return
    navigate(up, here)
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-black/50 pt-24"
      onClick={onClose}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label={title}
        className="w-full max-w-2xl bg-card/95 backdrop-blur-sm border border-border rounded-lg shadow-2xl flex flex-col overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-1.5 px-3 py-2 border-b border-border">
          <FolderOpen className="w-4 h-4 text-primary flex-shrink-0" />
          <span className="text-sm font-semibold text-foreground">{title}</span>
          <div className="flex-1" />
          <button
            type="button"
            onClick={goParent}
            title="Up one level"
            className="p-1 rounded hover:bg-accent text-muted-foreground"
          >
            <ArrowUp className="w-3.5 h-3.5" />
          </button>
          <button
            type="button"
            onClick={() => navigate('~')}
            title="Home"
            className="p-1 rounded hover:bg-accent text-muted-foreground"
          >
            <Home className="w-3.5 h-3.5" />
          </button>
          <button
            type="button"
            onClick={() => navigate('/')}
            title="Filesystem root"
            className="p-1 rounded hover:bg-accent text-muted-foreground"
          >
            <HardDrive className="w-3.5 h-3.5" />
          </button>
          <button
            type="button"
            onClick={() => setShowHidden(h => !h)}
            title={showHidden ? 'Hide folders starting with "."' : 'Show folders starting with "."'}
            className="p-1 rounded hover:bg-accent text-muted-foreground"
          >
            {showHidden ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
          </button>
        </div>

        {/* Path bar — type or paste a path, Enter navigates there. */}
        <div className="flex items-center gap-2 px-3 py-2 border-b border-border">
          <input
            ref={pathRef}
            type="text"
            value={pathInput}
            onChange={(e) => { setPathInput(e.target.value); setRootError(null) }}
            onKeyDown={(e) => {
              if (e.key === 'Enter') { e.preventDefault(); navigate(pathInput) }
            }}
            placeholder="/absolute/path  or  ~/relative"
            spellCheck={false}
            className="flex-1 bg-transparent text-xs font-mono focus:outline-none"
          />
        </div>

        {/* Breadcrumb along the selected path. */}
        {crumbs.length > 1 && (
          <div className="flex items-center gap-1 px-3 py-1.5 border-b border-border text-[11px] overflow-x-auto">
            {crumbs.map((p, i) => (
              <span key={p} className="flex items-center gap-1 flex-shrink-0">
                {i > 0 && <span className="text-muted-foreground/60">/</span>}
                <button
                  type="button"
                  onClick={() => navigate(p, selection || undefined)}
                  className={cn(
                    'px-1 rounded hover:bg-accent truncate',
                    p === selection ? 'text-primary' : 'text-muted-foreground',
                  )}
                >
                  {basename(p) || '/'}
                </button>
              </span>
            ))}
          </div>
        )}

        {/* Directory tree. */}
        <div className="max-h-[45vh] min-h-[160px] overflow-y-auto py-1">
          {rootError && (
            <div className="flex items-start gap-2 text-xs text-red-400 px-3 py-2">
              <AlertCircle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
              <div className="break-words">{rootError}</div>
            </div>
          )}
          {!root && !rootError && (
            <div className="flex items-center gap-2 text-xs text-muted-foreground italic px-3 py-2">
              <Loader2 className="w-3.5 h-3.5 animate-spin" /> Loading…
            </div>
          )}
          {rows.map((r, i) => (
            <div key={r.path}>
              <button
                type="button"
                ref={(el) => { rowRefs.current[r.path] = el }}
                onClick={() => { setHighlight(i); toggle(r.path) }}
                onDoubleClick={() => confirm(r.path)}
                onMouseEnter={() => setHighlight(i)}
                className={cn(
                  'w-full flex items-center gap-1 px-2 py-0.5 text-xs truncate',
                  i === highlight ? 'bg-primary/15 text-primary' : 'text-foreground/85 hover:bg-accent',
                )}
                style={{ paddingLeft: `${r.depth * 12 + 8}px` }}
              >
                {r.loading
                  ? <Loader2 className="w-3 h-3 flex-shrink-0 animate-spin text-muted-foreground" />
                  : r.expanded
                    ? <ChevronDown className="w-3 h-3 flex-shrink-0 text-muted-foreground" />
                    : <ChevronRight className="w-3 h-3 flex-shrink-0 text-muted-foreground" />}
                {r.expanded
                  ? <FolderOpen className="w-3.5 h-3.5 flex-shrink-0 text-primary" />
                  : <Folder className="w-3.5 h-3.5 flex-shrink-0 text-primary/80" />}
                <span className="truncate">{r.name}</span>
                {r.hiddenCount > 0 && (
                  <span className="text-[10px] text-muted-foreground/70 flex-shrink-0">
                    +{r.hiddenCount} hidden
                  </span>
                )}
                {r.omitted > 0 && (
                  <span className="text-[10px] text-muted-foreground/70 flex-shrink-0">
                    +{r.omitted} more
                  </span>
                )}
              </button>
              {r.expanded && r.error && (
                <div
                  className="flex items-center gap-1 text-[10px] text-red-400 px-2 py-0.5"
                  style={{ paddingLeft: `${(r.depth + 1) * 12 + 8}px` }}
                >
                  <AlertCircle className="w-3 h-3 flex-shrink-0" />
                  <span className="truncate">{r.error}</span>
                </div>
              )}
              {r.expanded && r.truncated && (
                <div
                  className="text-[10px] text-muted-foreground/70 italic px-2 py-0.5"
                  style={{ paddingLeft: `${(r.depth + 1) * 12 + 8}px` }}
                >
                  listing truncated by the server
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Footer — exactly what gets handed back to the caller. */}
        <div className="flex items-center gap-2 px-3 py-2 border-t border-border">
          <div className="flex-1 min-w-0 text-xs text-muted-foreground truncate font-mono" title={selection}>
            {selection || 'no directory selected'}
          </div>
          <button
            type="button"
            onClick={onClose}
            className="text-xs px-3 py-1.5 rounded hover:bg-accent text-muted-foreground"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={() => confirm(selection)}
            disabled={!selection}
            className="text-xs px-3 py-1.5 rounded bg-primary/20 text-primary hover:bg-primary/30 flex items-center gap-1.5 disabled:opacity-50"
          >
            <Check className="w-3.5 h-3.5" />
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  )
}
