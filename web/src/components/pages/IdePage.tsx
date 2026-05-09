import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { FileCode2, FolderOpen, AlertCircle, Save, Loader2, RefreshCw, X, Search, Command as CommandIcon } from 'lucide-react'
import FileTree from '../ide/FileTree'
import EditorTabs from '../ide/EditorTabs'
import MonacoHost from '../ide/MonacoHost'
import QuickOpen from '../ide/QuickOpen'
import CommandPalette, { type PaletteCommand } from '../ide/CommandPalette'
import { IdeProvider, useIde } from '../../contexts/IdeContext'
import { useMcUi, type IdeState } from '../../contexts/McUiContext'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { cn } from '@/lib/utils'

// Strip a folder prefix from a file path for the open_tabs mirror so the
// agent sees relative-ish paths when they live inside the open folder.
function shortenForMirror(path: string, folder: string | null): string {
  if (!folder) return path
  const f = folder.endsWith('/') ? folder : folder + '/'
  if (path === folder) return path
  if (path.startsWith(f)) return path.slice(f.length)
  return path
}

function IdePageInner() {
  const {
    openFolder, setOpenFolder,
    openFiles, activeFile, setActive,
    openFile, closeTab, setContent, isDirty,
    saveActive, saving, saveError,
    registerEditor,
    handleFileChanged,
    applyIncomingChange,
    dismissConflict,
    conflictByPath,
    animatingPath,
    skipAnimation,
  } = useIde()

  const { reportIdeState, pendingIdeAction, pendingFileChange, currentTab } = useMcUi()
  const [folderInput, setFolderInput] = useState(openFolder ?? '')
  const [quickOpen, setQuickOpen] = useState(false)
  const [paletteOpen, setPaletteOpen] = useState(false)
  const [treeRefreshKey, setTreeRefreshKey] = useState(0)

  // Keep the input synced if the folder changes from outside (e.g. agent).
  useEffect(() => {
    setFolderInput(openFolder ?? '')
  }, [openFolder])

  // Mirror IDE state to the backend (consumed by mc_get_state and prefetch).
  // Only the relative-ish file names go in open_tabs to keep the injected
  // <ide_state> block compact.
  useEffect(() => {
    const next: IdeState = {}
    if (openFolder) next.open_folder = openFolder
    if (activeFile) next.visible_file = shortenForMirror(activeFile, openFolder)
    if (openFiles.length > 0) {
      next.open_tabs = openFiles.map(f => shortenForMirror(f.path, openFolder))
    }
    if (!next.open_folder && !next.visible_file && !next.open_tabs?.length) {
      reportIdeState(null)
    } else {
      reportIdeState(next)
    }
  }, [openFolder, activeFile, openFiles, reportIdeState])

  // Apply pending agent-issued IDE actions (open_folder, close_tab).
  // open_file is handled by the pendingFocus channel below.
  const lastAppliedIdeSeq = useRef<number>(0)
  useEffect(() => {
    if (!pendingIdeAction) return
    if (pendingIdeAction.seq === lastAppliedIdeSeq.current) return
    lastAppliedIdeSeq.current = pendingIdeAction.seq
    if (pendingIdeAction.kind === 'open_folder') {
      setOpenFolder(pendingIdeAction.path)
      setFolderInput(pendingIdeAction.path)
    } else if (pendingIdeAction.kind === 'close_tab') {
      closeTab(pendingIdeAction.path)
    }
  }, [pendingIdeAction, setOpenFolder, closeTab])

  // pendingFocus for the IDE tab (from mc_navigate / ide_open_file) → open file.
  const { pendingFocus, consumePendingFocus } = useMcUi()
  useEffect(() => {
    if (pendingFocus && pendingFocus.tab === 'ide') {
      const id = consumePendingFocus('ide')
      if (id) openFile(id)
    }
  }, [pendingFocus, consumePendingFocus, openFile])

  // file_changed events from inotify → IdeContext decides what to do.
  // Tree refresh is debounced (~600ms) so a burst of writes (save, git
  // pull, Lloyd's Edit-then-Edit-then-Edit) collapses into one fetch.
  // The fetch is silent in FileTree so it never flashes the user.
  const lastAppliedFileSeq = useRef<number>(0)
  const treeRefreshTimer = useRef<number | null>(null)
  useEffect(() => {
    if (!pendingFileChange) return
    if (pendingFileChange.seq === lastAppliedFileSeq.current) return
    lastAppliedFileSeq.current = pendingFileChange.seq
    void handleFileChanged(pendingFileChange.path, pendingFileChange.deleted)
    if (treeRefreshTimer.current !== null) {
      window.clearTimeout(treeRefreshTimer.current)
    }
    treeRefreshTimer.current = window.setTimeout(() => {
      setTreeRefreshKey(x => x + 1)
      treeRefreshTimer.current = null
    }, 600)
  }, [pendingFileChange, handleFileChanged])
  useEffect(() => {
    return () => {
      if (treeRefreshTimer.current !== null) {
        window.clearTimeout(treeRefreshTimer.current)
      }
    }
  }, [])

  // Cmd/Ctrl+P → Quick Open. Cmd/Ctrl+Shift+P → Command Palette.
  // Only active when the IDE tab is the visible one.
  useEffect(() => {
    if (currentTab !== 'ide') return
    const onKey = (e: KeyboardEvent) => {
      const mod = e.metaKey || e.ctrlKey
      if (!mod) return
      if (e.key === 'p' || e.key === 'P') {
        e.preventDefault()
        if (e.shiftKey) {
          setPaletteOpen(true)
          setQuickOpen(false)
        } else {
          setQuickOpen(true)
          setPaletteOpen(false)
        }
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [currentTab])

  const handleOpenFolder = () => {
    const trimmed = folderInput.trim()
    if (!trimmed) return
    setOpenFolder(trimmed)
  }

  const activeOpenFile = useMemo(
    () => openFiles.find(f => f.path === activeFile) ?? null,
    [openFiles, activeFile],
  )

  const handleEditorChange = useCallback((value: string) => {
    if (activeFile) setContent(activeFile, value)
  }, [activeFile, setContent])

  // Command palette command list — derived from current state.
  const paletteCommands: PaletteCommand[] = useMemo(() => {
    const cmds: PaletteCommand[] = [
      {
        id: 'ide.quickOpen',
        label: 'Quick Open File…',
        description: 'Find a file by fuzzy name',
        keybinding: 'Ctrl/Cmd+P',
        run: () => setQuickOpen(true),
      },
      {
        id: 'ide.save',
        label: 'Save',
        description: 'Save the active file',
        keybinding: 'Ctrl/Cmd+S',
        run: () => { void saveActive() },
      },
      {
        id: 'ide.closeTab',
        label: 'Close Tab',
        description: 'Close the active editor tab',
        run: () => { if (activeFile) closeTab(activeFile) },
      },
      {
        id: 'ide.refreshTree',
        label: 'Refresh File Tree',
        run: () => setTreeRefreshKey(x => x + 1),
      },
    ]
    return cmds
  }, [activeFile, closeTab, saveActive])

  return (
    <div className="flex flex-col h-full min-h-0 overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-3 px-6 pt-6 pb-2 flex-shrink-0">
        <FileCode2 className="w-5 h-5 text-primary" />
        <h2 className="text-lg font-semibold text-foreground">IDE</h2>
        <div className="flex-1 flex items-center gap-2 max-w-xl">
          <FolderOpen className="w-4 h-4 text-muted-foreground" />
          <Input
            value={folderInput}
            onChange={e => setFolderInput(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter') handleOpenFolder() }}
            placeholder="/absolute/path/to/folder"
            className="text-xs h-8"
          />
          <Button size="sm" variant="ghost" onClick={handleOpenFolder} className="text-xs gap-1.5">
            Open
          </Button>
        </div>
        <div className="flex items-center gap-2">
          {saving && <Loader2 className="w-4 h-4 animate-spin text-muted-foreground" />}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setQuickOpen(true)}
            disabled={!openFolder}
            className="text-xs gap-1.5"
            title="Quick open file (Ctrl/Cmd+P)"
          >
            <Search className="w-3.5 h-3.5" />
            <span className="hidden lg:inline">Open File</span>
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setPaletteOpen(true)}
            className="text-xs gap-1.5"
            title="Command palette (Ctrl/Cmd+Shift+P)"
          >
            <CommandIcon className="w-3.5 h-3.5" />
            <span className="hidden lg:inline">Commands</span>
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => { void saveActive() }}
            disabled={!activeOpenFile || !isDirty(activeOpenFile.path) || saving}
            className="text-xs gap-1.5"
          >
            <Save className="w-3.5 h-3.5" />
            Save
          </Button>
        </div>
      </div>

      {saveError && (
        <div className="mx-6 mb-2 flex items-start gap-2 text-xs text-red-400 bg-red-500/10 border border-red-500/30 rounded p-2 flex-shrink-0">
          <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
          <div className="break-words">{saveError}</div>
        </div>
      )}

      {/* Conflict banner — Lloyd (or someone else) modified the active
          file on disk while the user has unsaved edits. */}
      {activeFile && conflictByPath[activeFile] && (
        <div className="mx-6 mb-2 flex items-center gap-3 text-xs text-amber-200 bg-amber-500/10 border border-amber-500/40 rounded p-2 flex-shrink-0">
          <AlertCircle className="w-4 h-4 flex-shrink-0 text-amber-300" />
          <div className="flex-1 break-words">
            {conflictByPath[activeFile].deleted
              ? 'This file was deleted on disk. Save will recreate it.'
              : 'Lloyd changed this file on disk and you have unsaved edits.'}
          </div>
          {!conflictByPath[activeFile].deleted && (
            <Button
              size="sm"
              variant="ghost"
              onClick={() => { void applyIncomingChange(activeFile) }}
              className="text-xs gap-1.5 text-amber-100 hover:text-amber-50 hover:bg-amber-500/20"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              Reload
            </Button>
          )}
          <Button
            size="sm"
            variant="ghost"
            onClick={() => dismissConflict(activeFile)}
            className="text-xs gap-1.5 text-amber-100 hover:text-amber-50 hover:bg-amber-500/20"
          >
            <X className="w-3.5 h-3.5" />
            Keep mine
          </Button>
        </div>
      )}

      {/* Animation in progress — give the user an escape hatch. */}
      {animatingPath && (
        <div className="mx-6 mb-2 flex items-center gap-3 text-xs text-primary bg-primary/10 border border-primary/30 rounded p-2 flex-shrink-0">
          <Loader2 className="w-4 h-4 flex-shrink-0 animate-spin" />
          <div className="flex-1">Animating Lloyd's changes…</div>
          <Button
            size="sm"
            variant="ghost"
            onClick={skipAnimation}
            className="text-xs gap-1.5 text-primary hover:text-primary hover:bg-primary/20"
          >
            Skip
          </Button>
        </div>
      )}

      <QuickOpen
        open={quickOpen}
        onClose={() => setQuickOpen(false)}
        rootPath={openFolder}
        onPick={openFile}
      />
      <CommandPalette
        open={paletteOpen}
        onClose={() => setPaletteOpen(false)}
        commands={paletteCommands}
      />

      {/* Body */}
      <div className="flex-1 flex min-h-0 overflow-hidden mx-6 mb-6 border border-border rounded-xl bg-card">
        {/* Tree */}
        <div className="w-64 border-r border-border flex-shrink-0 overflow-y-auto py-2">
          <FileTree
            rootPath={openFolder}
            selectedPath={activeFile}
            onFileClick={openFile}
            refreshKey={treeRefreshKey}
          />
        </div>

        {/* Editor area */}
        <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
          <EditorTabs
            files={openFiles}
            activePath={activeFile}
            isDirty={isDirty}
            onSelect={setActive}
            onClose={closeTab}
          />
          <div className="flex-1 min-h-0 overflow-hidden bg-[#1e1e1e]">
            {activeOpenFile ? (
              activeOpenFile.loading ? (
                <div className="flex items-center justify-center h-full text-xs text-muted-foreground italic">
                  Loading {activeOpenFile.path}…
                </div>
              ) : activeOpenFile.loadError ? (
                <div className="flex items-start gap-2 p-4 text-xs text-red-400">
                  <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
                  <div className="break-words">{activeOpenFile.loadError}</div>
                </div>
              ) : activeOpenFile.binary ? (
                <div className="p-4 text-xs text-muted-foreground italic">
                  Binary file — not displayed.
                </div>
              ) : activeOpenFile.tooLarge ? (
                <div className="p-4 text-xs text-muted-foreground italic">
                  File too large to display ({activeOpenFile.path}).
                </div>
              ) : (
                <MonacoHost
                  path={activeOpenFile.path}
                  content={activeOpenFile.content}
                  onChange={handleEditorChange}
                  onSaveShortcut={() => { void saveActive() }}
                  openFolder={openFolder}
                  onEditorReady={registerEditor}
                />
              )
            ) : (
              <div className={cn(
                'flex flex-col items-center justify-center h-full gap-2 text-muted-foreground',
              )}>
                <FileCode2 className="w-12 h-12 opacity-30" />
                <div className="text-sm italic">No file selected</div>
                <div className="text-xs">Click a file in the tree, or ask Lloyd to open one.</div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default function IdePage() {
  return (
    <IdeProvider>
      <IdePageInner />
    </IdeProvider>
  )
}
