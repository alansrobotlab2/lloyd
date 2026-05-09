import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react'
import { api, type IdeFileResponse } from '../api'
import type { editor as Mon } from 'monaco-editor'
import { animateDiff, type AnimatedDiffController } from '../components/ide/AnimatedDiff'

const STORAGE_KEY = 'lloyd_ide_state_v1'

export interface OpenFile {
  path: string
  // mtime when the file was loaded — POSTed back on save as the conflict guard.
  loadedMtime: number
  // Original text loaded from disk — diff against `content` to compute dirty.
  loadedContent: string
  content: string
  binary: boolean
  tooLarge: boolean
  // Loaded async on first open; null while pending.
  loading: boolean
  loadError: string | null
  // Set when inotify reports the underlying file vanished.
  deletedOnDisk?: boolean
}

export interface IncomingConflict {
  path: string
  // True if the file was deleted, not modified.
  deleted: boolean
}

interface IdeContextValue {
  openFolder: string | null
  setOpenFolder: (path: string | null) => void

  openFiles: OpenFile[]
  activeFile: string | null
  setActive: (path: string | null) => void

  openFile: (path: string) => void
  closeTab: (path: string) => void
  setContent: (path: string, content: string) => void
  isDirty: (path: string) => boolean

  saveActive: () => Promise<void>
  saving: boolean
  saveError: string | null

  // Editor registration — MonacoHost calls these on mount/unmount so the
  // context can drive animations and skip-cancellations from outside the
  // editor component.
  registerEditor: (editor: Mon.IStandaloneCodeEditor | null,
                   monaco: typeof import('monaco-editor') | null) => void

  // file_changed plumbing
  applyIncomingChange: (path: string) => Promise<void>     // force-reload & animate
  dismissConflict: (path: string) => void                   // user kept their edits
  conflictByPath: Record<string, IncomingConflict>

  animatingPath: string | null
  skipAnimation: () => void

  // Hook called by the IdePage SSE listener — you can call this with the
  // path that just changed on disk and the context decides what to do
  // (silent reload / animate / set conflict).
  handleFileChanged: (path: string, deleted: boolean) => Promise<void>
}

const IdeContext = createContext<IdeContextValue | null>(null)

interface PersistedShape {
  openFolder: string | null
  openTabs: string[]
  activeFile: string | null
}

function loadPersisted(): PersistedShape {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return { openFolder: null, openTabs: [], activeFile: null }
    const parsed = JSON.parse(raw)
    return {
      openFolder: typeof parsed.openFolder === 'string' ? parsed.openFolder : null,
      openTabs: Array.isArray(parsed.openTabs)
        ? parsed.openTabs.filter((t: unknown): t is string => typeof t === 'string')
        : [],
      activeFile: typeof parsed.activeFile === 'string' ? parsed.activeFile : null,
    }
  } catch {
    return { openFolder: null, openTabs: [], activeFile: null }
  }
}

function persist(folder: string | null, tabs: string[], active: string | null) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({
      openFolder: folder,
      openTabs: tabs,
      activeFile: active,
    }))
  } catch {
    // Quota exceeded etc. — best-effort.
  }
}

function fileFromResponse(resp: IdeFileResponse): OpenFile {
  return {
    path: resp.path,
    loadedMtime: resp.mtime,
    loadedContent: resp.content ?? '',
    content: resp.content ?? '',
    binary: resp.binary,
    tooLarge: resp.too_large,
    loading: false,
    loadError: null,
    deletedOnDisk: false,
  }
}

export function IdeProvider({ children }: { children: React.ReactNode }) {
  const initial = useRef<PersistedShape>(loadPersisted())
  const [openFolder, setOpenFolderState] = useState<string | null>(initial.current.openFolder)
  const [openFiles, setOpenFiles] = useState<OpenFile[]>([])
  const [activeFile, setActiveFile] = useState<string | null>(initial.current.activeFile)
  const [saving, setSaving] = useState(false)
  const [saveError, setSaveError] = useState<string | null>(null)
  const [conflictByPath, setConflictByPath] = useState<Record<string, IncomingConflict>>({})
  const [animatingPath, setAnimatingPath] = useState<string | null>(null)

  const editorHandle = useRef<{
    editor: Mon.IStandaloneCodeEditor
    monaco: typeof import('monaco-editor')
  } | null>(null)
  const animationCtrl = useRef<AnimatedDiffController | null>(null)

  // Always-current ref to openFiles so async handlers don't capture stale state.
  const openFilesRef = useRef<OpenFile[]>(openFiles)
  useEffect(() => { openFilesRef.current = openFiles }, [openFiles])
  const activeFileRef = useRef<string | null>(activeFile)
  useEffect(() => { activeFileRef.current = activeFile }, [activeFile])

  // On first mount, hydrate persisted tabs by fetching contents.
  const hydratedRef = useRef(false)
  useEffect(() => {
    if (hydratedRef.current) return
    hydratedRef.current = true
    const tabs = initial.current.openTabs
    if (tabs.length === 0) return
    setOpenFiles(tabs.map(p => ({
      path: p,
      loadedMtime: 0,
      loadedContent: '',
      content: '',
      binary: false,
      tooLarge: false,
      loading: true,
      loadError: null,
    })))
    tabs.forEach(p => {
      api.ideRead(p)
        .then(resp => {
          setOpenFiles(prev => prev.map(f => f.path === p ? fileFromResponse(resp) : f))
        })
        .catch(e => {
          setOpenFiles(prev => prev.map(f => f.path === p
            ? { ...f, loading: false, loadError: e instanceof Error ? e.message : String(e) }
            : f))
        })
    })
  }, [])

  // Persist tab list + folder + active on any change.
  useEffect(() => {
    const tabs = openFiles.map(f => f.path)
    persist(openFolder, tabs, activeFile)
  }, [openFolder, openFiles, activeFile])

  const setOpenFolder = useCallback((path: string | null) => {
    setOpenFolderState(path)
  }, [])

  const openFile = useCallback((path: string) => {
    setOpenFiles(prev => {
      if (prev.some(f => f.path === path)) {
        return prev
      }
      return [...prev, {
        path,
        loadedMtime: 0,
        loadedContent: '',
        content: '',
        binary: false,
        tooLarge: false,
        loading: true,
        loadError: null,
      }]
    })
    setActiveFile(path)
    api.ideRead(path)
      .then(resp => {
        setOpenFiles(prev => prev.map(f => f.path === path ? fileFromResponse(resp) : f))
      })
      .catch(e => {
        setOpenFiles(prev => prev.map(f => f.path === path
          ? { ...f, loading: false, loadError: e instanceof Error ? e.message : String(e) }
          : f))
      })
  }, [])

  const closeTab = useCallback((path: string) => {
    setOpenFiles(prev => {
      const idx = prev.findIndex(f => f.path === path)
      if (idx < 0) return prev
      const next = prev.filter(f => f.path !== path)
      setActiveFile(active => {
        if (active !== path) return active
        if (next.length === 0) return null
        const newActive = next[Math.min(idx, next.length - 1)]
        return newActive?.path ?? null
      })
      return next
    })
    setConflictByPath(prev => {
      if (!(path in prev)) return prev
      const { [path]: _, ...rest } = prev
      return rest
    })
  }, [])

  const setActive = useCallback((path: string | null) => {
    setActiveFile(path)
  }, [])

  const setContent = useCallback((path: string, content: string) => {
    setOpenFiles(prev => prev.map(f => f.path === path ? { ...f, content } : f))
  }, [])

  const isDirty = useCallback((path: string) => {
    const f = openFiles.find(x => x.path === path)
    if (!f) return false
    if (f.loading) return false
    return f.content !== f.loadedContent
  }, [openFiles])

  const saveActive = useCallback(async () => {
    if (!activeFile) return
    const f = openFiles.find(x => x.path === activeFile)
    if (!f) return
    if (f.binary || f.tooLarge || f.loading) return
    if (f.content === f.loadedContent) return
    setSaving(true)
    setSaveError(null)
    try {
      const resp = await api.ideWrite(f.path, f.content, f.loadedMtime || undefined)
      setOpenFiles(prev => prev.map(x => x.path === f.path
        ? { ...x, loadedContent: x.content, loadedMtime: resp.mtime }
        : x))
    } catch (e) {
      setSaveError(e instanceof Error ? e.message : String(e))
    } finally {
      setSaving(false)
    }
  }, [activeFile, openFiles])

  const registerEditor = useCallback((
    editor: Mon.IStandaloneCodeEditor | null,
    monaco: typeof import('monaco-editor') | null,
  ) => {
    if (editor && monaco) {
      editorHandle.current = { editor, monaco }
    } else {
      editorHandle.current = null
    }
  }, [])

  const dismissConflict = useCallback((path: string) => {
    setConflictByPath(prev => {
      if (!(path in prev)) return prev
      const { [path]: _, ...rest } = prev
      return rest
    })
  }, [])

  const skipAnimation = useCallback(() => {
    animationCtrl.current?.cancel()
  }, [])

  // Force-reload `path` from disk, replacing whatever was in the buffer
  // (used by the conflict banner's Reload button, and by handleFileChanged
  // for clean tabs). When the path is currently active, animate the diff.
  const applyIncomingChange = useCallback(async (path: string) => {
    let resp: IdeFileResponse
    try {
      resp = await api.ideRead(path)
    } catch (e) {
      // If the read fails (e.g. file gone), mark deleted-on-disk.
      setOpenFiles(prev => prev.map(f => f.path === path
        ? { ...f, deletedOnDisk: true, loadError: e instanceof Error ? e.message : String(e) }
        : f))
      return
    }
    const next = fileFromResponse(resp)
    const isActive = activeFileRef.current === path
    const handle = editorHandle.current
    dismissConflict(path)

    if (isActive && handle) {
      // Animate the change, then commit final state to React when done.
      animationCtrl.current?.cancel()
      setAnimatingPath(path)
      const ctrl = animateDiff(handle.editor, handle.monaco, next.content, {
        onDone: () => {
          setAnimatingPath(prev => (prev === path ? null : prev))
          // Sync the React buffer to the editor's final value (Monaco
          // performed the edits directly on the model during animation).
          setOpenFiles(prev => prev.map(f => f.path === path ? next : f))
        },
      })
      animationCtrl.current = ctrl
    } else {
      // Inactive (or no editor handle yet) — silent replace.
      setOpenFiles(prev => prev.map(f => f.path === path ? next : f))
    }
  }, [dismissConflict])

  const handleFileChanged = useCallback(async (path: string, deleted: boolean) => {
    const files = openFilesRef.current
    const f = files.find(x => x.path === path)
    if (!f) return   // not open, nothing to do

    if (deleted) {
      setOpenFiles(prev => prev.map(x => x.path === path
        ? { ...x, deletedOnDisk: true }
        : x))
      return
    }

    // Ignore self-induced events: if the buffer is clean AND the on-disk
    // mtime appears equal to what we already have, it was probably our
    // own save round-tripping back.
    const dirty = !f.loading && f.content !== f.loadedContent
    if (dirty) {
      setConflictByPath(prev => ({ ...prev, [path]: { path, deleted: false } }))
      return
    }
    // Clean tab → reload (and animate if it's the active one).
    await applyIncomingChange(path)
  }, [applyIncomingChange])

  const value = useMemo<IdeContextValue>(() => ({
    openFolder,
    setOpenFolder,
    openFiles,
    activeFile,
    setActive,
    openFile,
    closeTab,
    setContent,
    isDirty,
    saveActive,
    saving,
    saveError,
    registerEditor,
    applyIncomingChange,
    dismissConflict,
    conflictByPath,
    animatingPath,
    skipAnimation,
    handleFileChanged,
  }), [openFolder, setOpenFolder, openFiles, activeFile, setActive,
       openFile, closeTab, setContent, isDirty, saveActive, saving, saveError,
       registerEditor, applyIncomingChange, dismissConflict, conflictByPath,
       animatingPath, skipAnimation, handleFileChanged])

  return <IdeContext.Provider value={value}>{children}</IdeContext.Provider>
}

export function useIde(): IdeContextValue {
  const ctx = useContext(IdeContext)
  if (!ctx) throw new Error('useIde must be used inside <IdeProvider>')
  return ctx
}
