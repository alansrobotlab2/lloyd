import { useEffect, useMemo, useRef, useState } from 'react'
import { File, Loader2 } from 'lucide-react'
import { api } from '../../api'
import { cn } from '@/lib/utils'

interface QuickOpenProps {
  open: boolean
  onClose: () => void
  rootPath: string | null
  onPick: (path: string) => void
}

interface CacheEntry {
  fetchedAt: number
  files: string[]
  truncated: boolean
}

const _cache = new Map<string, CacheEntry>()
const TTL_MS = 30_000

// Tiny fuzzy-rank: counts ordered char matches; tie-break by shorter path.
// Good enough for a few-thousand-file workspace; not VSCode-grade.
function fuzzyScore(query: string, target: string): number {
  if (!query) return 0.5
  const q = query.toLowerCase()
  const t = target.toLowerCase()
  let qi = 0
  let score = 0
  let lastMatch = -1
  for (let ti = 0; ti < t.length && qi < q.length; ti++) {
    if (t[ti] === q[qi]) {
      // Bonus for consecutive matches and word-start matches.
      if (lastMatch === ti - 1) score += 2
      else score += 1
      if (ti === 0 || /[\/\-_.]/.test(t[ti - 1])) score += 1
      lastMatch = ti
      qi++
    }
  }
  if (qi < q.length) return -1     // not all chars matched
  return score - target.length * 0.01
}

export default function QuickOpen({ open, onClose, rootPath, onPick }: QuickOpenProps) {
  const [query, setQuery] = useState('')
  const [files, setFiles] = useState<string[]>([])
  const [loading, setLoading] = useState(false)
  const [highlightIdx, setHighlightIdx] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)

  // Fetch (or use cache) when the modal opens.
  useEffect(() => {
    if (!open || !rootPath) return
    setQuery('')
    setHighlightIdx(0)
    const cached = _cache.get(rootPath)
    if (cached && Date.now() - cached.fetchedAt < TTL_MS) {
      setFiles(cached.files)
      return
    }
    setLoading(true)
    api.ideGlob(rootPath, 4000)
      .then(r => {
        _cache.set(rootPath, { fetchedAt: Date.now(), files: r.files, truncated: r.truncated })
        setFiles(r.files)
      })
      .catch(() => setFiles([]))
      .finally(() => setLoading(false))
  }, [open, rootPath])

  // Focus the input when shown.
  useEffect(() => {
    if (open) {
      const t = setTimeout(() => inputRef.current?.focus(), 0)
      return () => clearTimeout(t)
    }
  }, [open])

  const ranked = useMemo(() => {
    if (!rootPath) return [] as Array<{ path: string; rel: string; score: number }>
    const root = rootPath.endsWith('/') ? rootPath : rootPath + '/'
    const items = files.map(p => ({
      path: p,
      rel: p.startsWith(root) ? p.slice(root.length) : p,
      score: 0,
    }))
    if (!query) {
      // No query → show first 50 alphabetically (already sorted from API).
      return items.slice(0, 50)
    }
    const scored = items
      .map(it => ({ ...it, score: fuzzyScore(query, it.rel) }))
      .filter(it => it.score >= 0)
    scored.sort((a, b) => b.score - a.score)
    return scored.slice(0, 80)
  }, [files, query, rootPath])

  // Reset highlight when results change.
  useEffect(() => { setHighlightIdx(0) }, [ranked.length, query])

  if (!open) return null

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') { onClose(); return }
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setHighlightIdx(i => Math.min(i + 1, ranked.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setHighlightIdx(i => Math.max(i - 1, 0))
    } else if (e.key === 'Enter') {
      e.preventDefault()
      const pick = ranked[highlightIdx]
      if (pick) {
        onPick(pick.path)
        onClose()
      }
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center bg-black/50 pt-24"
      onClick={onClose}
    >
      <div
        className="w-full max-w-xl bg-card border border-border rounded-lg shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center gap-2 px-3 py-2 border-b border-border">
          {loading
            ? <Loader2 className="w-4 h-4 animate-spin text-muted-foreground" />
            : <File className="w-4 h-4 text-muted-foreground" />}
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Type to search files…"
            className="flex-1 bg-transparent text-sm focus:outline-none"
          />
        </div>
        <div className="max-h-80 overflow-y-auto py-1">
          {!loading && ranked.length === 0 && (
            <div className="text-xs text-muted-foreground italic px-3 py-2">
              No files match.
            </div>
          )}
          {ranked.map((it, i) => (
            <button
              key={it.path}
              onClick={() => { onPick(it.path); onClose() }}
              onMouseEnter={() => setHighlightIdx(i)}
              className={cn(
                'w-full text-left px-3 py-1.5 text-xs flex items-center gap-2 truncate',
                i === highlightIdx ? 'bg-primary/15 text-primary' : 'text-foreground/85 hover:bg-accent',
              )}
            >
              <File className="w-3.5 h-3.5 flex-shrink-0 text-muted-foreground" />
              <span className="truncate">{it.rel}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
