import { useCallback, useEffect, useState } from 'react'
import { ChevronRight, ChevronDown, Folder, FolderOpen, File, AlertCircle } from 'lucide-react'
import { api, type IdeEntry } from '../../api'
import { cn } from '@/lib/utils'

interface FileTreeProps {
  rootPath: string | null
  selectedPath: string | null
  onFileClick: (absPath: string) => void
}

interface NodeProps {
  path: string
  name: string
  depth: number
  selectedPath: string | null
  onFileClick: (absPath: string) => void
}

interface DirState {
  loading: boolean
  error: string | null
  entries: IdeEntry[] | null
}

function joinPath(parent: string, name: string) {
  if (parent.endsWith('/')) return parent + name
  return parent + '/' + name
}

function DirNode({ path, name, depth, selectedPath, onFileClick }: NodeProps) {
  const [open, setOpen] = useState(false)
  const [state, setState] = useState<DirState>({ loading: false, error: null, entries: null })

  const load = useCallback(() => {
    setState({ loading: true, error: null, entries: null })
    api.ideList(path)
      .then(r => setState({ loading: false, error: null, entries: r.entries }))
      .catch(e => setState({
        loading: false,
        error: e instanceof Error ? e.message : String(e),
        entries: null,
      }))
  }, [path])

  const toggle = () => {
    if (!open && state.entries === null && !state.loading) load()
    setOpen(o => !o)
  }

  return (
    <div>
      <button
        type="button"
        onClick={toggle}
        className={cn(
          'w-full flex items-center gap-1 px-2 py-0.5 text-xs text-foreground/90 hover:bg-accent rounded',
          'truncate',
        )}
        style={{ paddingLeft: `${depth * 12 + 8}px` }}
      >
        {open
          ? <ChevronDown className="w-3 h-3 flex-shrink-0 text-muted-foreground" />
          : <ChevronRight className="w-3 h-3 flex-shrink-0 text-muted-foreground" />}
        {open
          ? <FolderOpen className="w-3.5 h-3.5 flex-shrink-0 text-primary" />
          : <Folder className="w-3.5 h-3.5 flex-shrink-0 text-primary/80" />}
        <span className="truncate">{name}</span>
      </button>
      {open && (
        <div>
          {state.loading && (
            <div
              className="text-[10px] text-muted-foreground px-2 py-0.5 italic"
              style={{ paddingLeft: `${(depth + 1) * 12 + 8}px` }}
            >
              loading…
            </div>
          )}
          {state.error && (
            <div
              className="flex items-center gap-1 text-[10px] text-red-400 px-2 py-0.5"
              style={{ paddingLeft: `${(depth + 1) * 12 + 8}px` }}
            >
              <AlertCircle className="w-3 h-3" />
              <span className="truncate">{state.error}</span>
            </div>
          )}
          {state.entries && state.entries.map(entry => (
            entry.isDir ? (
              <DirNode
                key={entry.name}
                path={joinPath(path, entry.name)}
                name={entry.name}
                depth={depth + 1}
                selectedPath={selectedPath}
                onFileClick={onFileClick}
              />
            ) : (
              <FileNode
                key={entry.name}
                path={joinPath(path, entry.name)}
                name={entry.name}
                depth={depth + 1}
                selectedPath={selectedPath}
                onFileClick={onFileClick}
              />
            )
          ))}
        </div>
      )}
    </div>
  )
}

function FileNode({ path, name, depth, selectedPath, onFileClick }: NodeProps) {
  const isSelected = selectedPath === path
  return (
    <button
      type="button"
      onClick={() => onFileClick(path)}
      className={cn(
        'w-full flex items-center gap-1 px-2 py-0.5 text-xs rounded truncate',
        isSelected
          ? 'bg-primary/15 text-primary'
          : 'text-foreground/80 hover:bg-accent',
      )}
      style={{ paddingLeft: `${depth * 12 + 8 + 12}px` }}
    >
      <File className="w-3.5 h-3.5 flex-shrink-0 text-muted-foreground" />
      <span className="truncate">{name}</span>
    </button>
  )
}

export default function FileTree({ rootPath, selectedPath, onFileClick }: FileTreeProps) {
  const [rootState, setRootState] = useState<DirState>({ loading: false, error: null, entries: null })

  useEffect(() => {
    if (!rootPath) {
      setRootState({ loading: false, error: null, entries: null })
      return
    }
    setRootState({ loading: true, error: null, entries: null })
    api.ideList(rootPath)
      .then(r => setRootState({ loading: false, error: null, entries: r.entries }))
      .catch(e => setRootState({
        loading: false,
        error: e instanceof Error ? e.message : String(e),
        entries: null,
      }))
  }, [rootPath])

  if (!rootPath) {
    return (
      <div className="text-xs text-muted-foreground p-3">
        Open a folder to browse files.
      </div>
    )
  }

  if (rootState.loading) {
    return <div className="text-xs text-muted-foreground p-3 italic">Loading {rootPath}…</div>
  }

  if (rootState.error) {
    return (
      <div className="flex items-start gap-2 text-xs text-red-400 p-3">
        <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
        <div className="break-words">{rootState.error}</div>
      </div>
    )
  }

  return (
    <div className="text-xs">
      {rootState.entries?.map(entry => (
        entry.isDir ? (
          <DirNode
            key={entry.name}
            path={joinPath(rootPath, entry.name)}
            name={entry.name}
            depth={0}
            selectedPath={selectedPath}
            onFileClick={onFileClick}
          />
        ) : (
          <FileNode
            key={entry.name}
            path={joinPath(rootPath, entry.name)}
            name={entry.name}
            depth={0}
            selectedPath={selectedPath}
            onFileClick={onFileClick}
          />
        )
      ))}
    </div>
  )
}
