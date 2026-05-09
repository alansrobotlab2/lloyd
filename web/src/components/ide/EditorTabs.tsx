import { X, Circle } from 'lucide-react'
import { cn } from '@/lib/utils'
import type { OpenFile } from '../../contexts/IdeContext'

interface EditorTabsProps {
  files: OpenFile[]
  activePath: string | null
  isDirty: (path: string) => boolean
  onSelect: (path: string) => void
  onClose: (path: string) => void
}

function basename(path: string) {
  const idx = path.lastIndexOf('/')
  return idx >= 0 ? path.slice(idx + 1) : path
}

export default function EditorTabs({ files, activePath, isDirty, onSelect, onClose }: EditorTabsProps) {
  if (files.length === 0) {
    return (
      <div className="flex items-center text-xs text-muted-foreground border-b border-border h-9 px-3 italic flex-shrink-0">
        No file open. Click one in the tree to open it.
      </div>
    )
  }

  return (
    <div className="flex items-center border-b border-border h-9 overflow-x-auto flex-shrink-0">
      {files.map(file => {
        const active = file.path === activePath
        const dirty = isDirty(file.path)
        return (
          <div
            key={file.path}
            className={cn(
              'flex items-center gap-1.5 px-3 py-1.5 text-xs border-r border-border cursor-pointer flex-shrink-0',
              active
                ? 'bg-card text-foreground'
                : 'bg-background text-muted-foreground hover:text-foreground hover:bg-accent/50',
            )}
            onClick={() => onSelect(file.path)}
            title={file.path}
          >
            <span className="truncate max-w-[160px]">{basename(file.path)}</span>
            {dirty
              ? <Circle className="w-2.5 h-2.5 fill-current text-primary flex-shrink-0" />
              : null}
            <button
              type="button"
              onClick={e => {
                e.stopPropagation()
                onClose(file.path)
              }}
              className="ml-0.5 -mr-1 p-0.5 rounded hover:bg-foreground/10"
              aria-label={`Close ${basename(file.path)}`}
            >
              <X className="w-3 h-3" />
            </button>
          </div>
        )
      })}
    </div>
  )
}
