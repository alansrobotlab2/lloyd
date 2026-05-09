import { useCallback, useEffect, useRef, useState } from 'react'
import { ChevronRight, ChevronDown, Folder, FolderOpen, File, AlertCircle, FilePlus, FolderPlus, Edit3, Trash2 } from 'lucide-react'
import { api, type IdeEntry } from '../../api'
import { cn } from '@/lib/utils'

interface FileTreeProps {
  rootPath: string | null
  selectedPath: string | null
  onFileClick: (absPath: string) => void
  // Bumped externally to force a re-fetch of every open dir (used after
  // an inotify event might have invalidated the listing).
  refreshKey?: number
}

interface NodeProps {
  path: string
  name: string
  depth: number
  selectedPath: string | null
  onFileClick: (absPath: string) => void
  onContextMenu: (e: React.MouseEvent, path: string, isDir: boolean) => void
  refreshKey?: number
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

function dirOf(path: string): string {
  const idx = path.lastIndexOf('/')
  return idx > 0 ? path.slice(0, idx) : path
}

function basename(path: string): string {
  const idx = path.lastIndexOf('/')
  return idx >= 0 ? path.slice(idx + 1) : path
}

function DirNode({ path, name, depth, selectedPath, onFileClick, onContextMenu, refreshKey }: NodeProps) {
  const [open, setOpen] = useState(false)
  const [state, setState] = useState<DirState>({ loading: false, error: null, entries: null })

  // Two flavours of fetch:
  //   load()           — initial open; clears entries first so a spinner shows.
  //   refreshSilent()  — re-fetch in the background; keep prior entries on
  //                      screen until the new payload arrives so the tree
  //                      doesn't flash.
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

  const refreshSilent = useCallback(() => {
    api.ideList(path)
      .then(r => setState({ loading: false, error: null, entries: r.entries }))
      .catch(() => { /* keep prior entries on transient failure */ })
  }, [path])

  // External refresh — only re-fetch if this dir is open. Use the silent
  // variant so the tree doesn't flash through a "loading" state.
  useEffect(() => {
    if (open && refreshKey !== undefined) refreshSilent()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refreshKey])

  const toggle = () => {
    if (!open && state.entries === null && !state.loading) load()
    setOpen(o => !o)
  }

  return (
    <div>
      <button
        type="button"
        onClick={toggle}
        onContextMenu={(e) => onContextMenu(e, path, true)}
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
                onContextMenu={onContextMenu}
                refreshKey={refreshKey}
              />
            ) : (
              <FileNode
                key={entry.name}
                path={joinPath(path, entry.name)}
                name={entry.name}
                depth={depth + 1}
                selectedPath={selectedPath}
                onFileClick={onFileClick}
                onContextMenu={onContextMenu}
                refreshKey={refreshKey}
              />
            )
          ))}
        </div>
      )}
    </div>
  )
}

function FileNode({ path, name, depth, selectedPath, onFileClick, onContextMenu }: NodeProps) {
  const isSelected = selectedPath === path
  return (
    <button
      type="button"
      onClick={() => onFileClick(path)}
      onContextMenu={(e) => onContextMenu(e, path, false)}
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

export default function FileTree({ rootPath, selectedPath, onFileClick, refreshKey }: FileTreeProps) {
  const [rootState, setRootState] = useState<DirState>({ loading: false, error: null, entries: null })

  // Right-click menu state.
  const [menu, setMenu] = useState<{
    x: number; y: number; path: string; isDir: boolean
  } | null>(null)

  // Modal state for create/rename prompts.
  const [modal, setModal] = useState<{
    title: string
    initialValue: string
    onSubmit: (value: string) => Promise<void>
  } | null>(null)
  const [modalValue, setModalValue] = useState('')
  const [modalError, setModalError] = useState<string | null>(null)

  // Local refresh trigger — bumped after CRUD mutations to invalidate
  // open dir caches.
  const [localRefresh, setLocalRefresh] = useState(0)

  // Initial load: clear entries so a spinner shows on first listing.
  const initialLoad = useCallback((target: string) => {
    setRootState({ loading: true, error: null, entries: null })
    api.ideList(target)
      .then(r => setRootState({ loading: false, error: null, entries: r.entries }))
      .catch(e => setRootState({
        loading: false,
        error: e instanceof Error ? e.message : String(e),
        entries: null,
      }))
  }, [])

  // Background refresh: keep current entries on screen during the fetch.
  const refreshSilent = useCallback((target: string) => {
    api.ideList(target)
      .then(r => setRootState({ loading: false, error: null, entries: r.entries }))
      .catch(() => { /* swallow — keep prior entries */ })
  }, [])

  // Effect 1: rootPath changes → fresh load (with spinner).
  useEffect(() => {
    if (!rootPath) {
      setRootState({ loading: false, error: null, entries: null })
      return
    }
    initialLoad(rootPath)
  }, [rootPath, initialLoad])

  // Effect 2: external refreshKey bump → silent re-fetch.
  // Skip the very first run so we don't double-fetch on mount.
  const initialRefreshKey = useRef(refreshKey)
  useEffect(() => {
    if (refreshKey === initialRefreshKey.current) return
    if (!rootPath) return
    refreshSilent(rootPath)
  }, [refreshKey, rootPath, refreshSilent])

  // Close menu on outside click / scroll.
  useEffect(() => {
    if (!menu) return
    const close = () => setMenu(null)
    window.addEventListener('click', close)
    window.addEventListener('scroll', close, true)
    window.addEventListener('keydown', close)
    return () => {
      window.removeEventListener('click', close)
      window.removeEventListener('scroll', close, true)
      window.removeEventListener('keydown', close)
    }
  }, [menu])

  const openContextMenu = useCallback((e: React.MouseEvent, path: string, isDir: boolean) => {
    e.preventDefault()
    e.stopPropagation()
    setMenu({ x: e.clientX, y: e.clientY, path, isDir })
  }, [])

  const handleNewFile = (parent: string) => {
    setMenu(null)
    setModalError(null)
    setModalValue('')
    setModal({
      title: `New file in ${basename(parent) || parent}`,
      initialValue: '',
      onSubmit: async (name) => {
        const target = joinPath(parent, name)
        await api.ideCreate(target, 'file')
        setLocalRefresh(x => x + 1)
        onFileClick(target)
      },
    })
  }

  const handleNewFolder = (parent: string) => {
    setMenu(null)
    setModalError(null)
    setModalValue('')
    setModal({
      title: `New folder in ${basename(parent) || parent}`,
      initialValue: '',
      onSubmit: async (name) => {
        await api.ideCreate(joinPath(parent, name), 'dir')
        setLocalRefresh(x => x + 1)
      },
    })
  }

  const handleRename = (path: string) => {
    setMenu(null)
    setModalError(null)
    setModalValue(basename(path))
    setModal({
      title: `Rename ${basename(path)}`,
      initialValue: basename(path),
      onSubmit: async (newName) => {
        const newPath = joinPath(dirOf(path), newName)
        await api.ideRename(path, newPath)
        setLocalRefresh(x => x + 1)
      },
    })
  }

  const handleDelete = async (path: string) => {
    setMenu(null)
    if (!window.confirm(`Delete ${path}? This cannot be undone.`)) return
    try {
      await api.ideDelete(path)
      setLocalRefresh(x => x + 1)
    } catch (e) {
      window.alert(`Delete failed: ${e instanceof Error ? e.message : String(e)}`)
    }
  }

  const submitModal = async () => {
    if (!modal) return
    const v = modalValue.trim()
    if (!v) {
      setModalError('Name is required')
      return
    }
    try {
      await modal.onSubmit(v)
      setModal(null)
    } catch (e) {
      setModalError(e instanceof Error ? e.message : String(e))
    }
  }

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
    <div
      className="text-xs"
      onContextMenu={(e) => {
        // Right-click on tree background = context menu rooted at the workspace.
        if (e.target === e.currentTarget) {
          openContextMenu(e, rootPath, true)
        }
      }}
    >
      {rootState.entries?.map(entry => (
        entry.isDir ? (
          <DirNode
            key={entry.name}
            path={joinPath(rootPath, entry.name)}
            name={entry.name}
            depth={0}
            selectedPath={selectedPath}
            onFileClick={onFileClick}
            onContextMenu={openContextMenu}
            refreshKey={localRefresh}
          />
        ) : (
          <FileNode
            key={entry.name}
            path={joinPath(rootPath, entry.name)}
            name={entry.name}
            depth={0}
            selectedPath={selectedPath}
            onFileClick={onFileClick}
            onContextMenu={openContextMenu}
            refreshKey={localRefresh}
          />
        )
      ))}

      {/* Context menu */}
      {menu && (
        <div
          onClick={(e) => e.stopPropagation()}
          className="fixed z-50 bg-popover border border-border rounded-md shadow-lg min-w-[180px] py-1 text-xs"
          style={{ left: menu.x, top: menu.y }}
        >
          {menu.isDir && (
            <>
              <MenuItem icon={<FilePlus className="w-3.5 h-3.5" />} onClick={() => handleNewFile(menu.path)}>New file…</MenuItem>
              <MenuItem icon={<FolderPlus className="w-3.5 h-3.5" />} onClick={() => handleNewFolder(menu.path)}>New folder…</MenuItem>
              <Separator />
            </>
          )}
          <MenuItem icon={<Edit3 className="w-3.5 h-3.5" />} onClick={() => handleRename(menu.path)}>Rename…</MenuItem>
          <MenuItem icon={<Trash2 className="w-3.5 h-3.5" />} onClick={() => handleDelete(menu.path)} danger>
            Delete
          </MenuItem>
        </div>
      )}

      {/* Rename / new-file modal */}
      {modal && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
          onClick={() => setModal(null)}
        >
          <div
            className="bg-card border border-border rounded-lg shadow-xl p-4 w-96"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="text-sm font-semibold mb-3">{modal.title}</div>
            <input
              autoFocus
              type="text"
              value={modalValue}
              onChange={(e) => { setModalValue(e.target.value); setModalError(null) }}
              onKeyDown={(e) => {
                if (e.key === 'Enter') void submitModal()
                if (e.key === 'Escape') setModal(null)
              }}
              className="w-full bg-input border border-border rounded px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary/40"
            />
            {modalError && (
              <div className="text-xs text-red-400 mt-2">{modalError}</div>
            )}
            <div className="flex justify-end gap-2 mt-4">
              <button
                onClick={() => setModal(null)}
                className="text-xs px-3 py-1.5 rounded hover:bg-accent text-muted-foreground"
              >Cancel</button>
              <button
                onClick={() => { void submitModal() }}
                className="text-xs px-3 py-1.5 rounded bg-primary/20 text-primary hover:bg-primary/30"
              >OK</button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function MenuItem({
  icon, onClick, children, danger,
}: {
  icon: React.ReactNode
  onClick: () => void
  children: React.ReactNode
  danger?: boolean
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        'w-full flex items-center gap-2 px-3 py-1.5 hover:bg-accent text-left',
        danger && 'text-red-400 hover:text-red-300',
      )}
    >
      {icon}
      <span>{children}</span>
    </button>
  )
}

function Separator() {
  return <div className="border-t border-border my-1" />
}
