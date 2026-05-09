import { useEffect, useMemo, useRef, useState } from 'react'
import { Command as CommandIcon } from 'lucide-react'
import { cn } from '@/lib/utils'

export interface PaletteCommand {
  id: string
  label: string
  description?: string
  keybinding?: string
  run: () => void | Promise<void>
}

interface CommandPaletteProps {
  open: boolean
  onClose: () => void
  commands: PaletteCommand[]
}

function fuzzyMatch(query: string, target: string): number {
  if (!query) return 0.5
  const q = query.toLowerCase()
  const t = target.toLowerCase()
  let qi = 0
  let score = 0
  for (let ti = 0; ti < t.length && qi < q.length; ti++) {
    if (t[ti] === q[qi]) {
      score += (qi === 0 ? 2 : 1)
      qi++
    }
  }
  return qi < q.length ? -1 : score
}

export default function CommandPalette({ open, onClose, commands }: CommandPaletteProps) {
  const [query, setQuery] = useState('')
  const [highlightIdx, setHighlightIdx] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (open) {
      setQuery('')
      setHighlightIdx(0)
      const t = setTimeout(() => inputRef.current?.focus(), 0)
      return () => clearTimeout(t)
    }
  }, [open])

  const ranked = useMemo(() => {
    if (!query) return commands
    return commands
      .map(c => ({ c, score: fuzzyMatch(query, c.label + ' ' + (c.description ?? '')) }))
      .filter(x => x.score >= 0)
      .sort((a, b) => b.score - a.score)
      .map(x => x.c)
  }, [commands, query])

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
        onClose()
        void pick.run()
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
          <CommandIcon className="w-4 h-4 text-muted-foreground" />
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Type a command…"
            className="flex-1 bg-transparent text-sm focus:outline-none"
          />
        </div>
        <div className="max-h-80 overflow-y-auto py-1">
          {ranked.length === 0 && (
            <div className="text-xs text-muted-foreground italic px-3 py-2">
              No commands match.
            </div>
          )}
          {ranked.map((c, i) => (
            <button
              key={c.id}
              onClick={() => { onClose(); void c.run() }}
              onMouseEnter={() => setHighlightIdx(i)}
              className={cn(
                'w-full text-left px-3 py-2 text-xs flex items-center gap-3',
                i === highlightIdx ? 'bg-primary/15 text-primary' : 'text-foreground/85 hover:bg-accent',
              )}
            >
              <div className="flex-1 min-w-0">
                <div className="truncate">{c.label}</div>
                {c.description && (
                  <div className="text-[10px] text-muted-foreground truncate">{c.description}</div>
                )}
              </div>
              {c.keybinding && (
                <div className="text-[10px] text-muted-foreground font-mono">{c.keybinding}</div>
              )}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
