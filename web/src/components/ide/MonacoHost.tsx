import { useEffect, useRef, useState } from 'react'
import Editor, { loader, type OnMount, type OnChange } from '@monaco-editor/react'
import type { editor } from 'monaco-editor'

// Use the codingame VSCode-flavored Monaco build. The vite alias
// (vite.config.ts) routes every `monaco-editor` import to this same
// package, so the language client and our editor share one runtime.
import * as monacoNs from '@codingame/monaco-vscode-editor-api'
loader.config({ monaco: monacoNs as unknown as typeof import('monaco-editor') })

import { ensureVscodeBootstrap } from './vscodeBootstrap'
import { ensureLspClient, lspLanguageFor } from './LspClient'
import { registerAiProviders, attachAiActionCommand } from './AiProviders'
import { attachGitGutter, type GitGutterController } from './GitGutter'

const LANG_BY_EXT: Record<string, string> = {
  ts: 'typescript', tsx: 'typescript',
  js: 'javascript', jsx: 'javascript', mjs: 'javascript', cjs: 'javascript',
  py: 'python',
  rs: 'rust',
  go: 'go',
  c: 'c', h: 'c',
  cpp: 'cpp', cc: 'cpp', hpp: 'cpp', hh: 'cpp',
  java: 'java',
  rb: 'ruby',
  sh: 'shell', bash: 'shell', zsh: 'shell',
  md: 'markdown', mdx: 'markdown',
  json: 'json',
  yaml: 'yaml', yml: 'yaml',
  toml: 'plaintext',
  html: 'html', htm: 'html',
  css: 'css', scss: 'scss',
  sql: 'sql',
  xml: 'xml',
}

function languageFor(path: string): string {
  const dot = path.lastIndexOf('.')
  if (dot < 0) return 'plaintext'
  const ext = path.slice(dot + 1).toLowerCase()
  return LANG_BY_EXT[ext] ?? 'plaintext'
}

interface MonacoHostProps {
  path: string
  content: string
  onChange: (value: string) => void
  onSaveShortcut: () => void
  readOnly?: boolean
  // The IDE's currently-open folder. Used as the workspace root for
  // any LSP servers we start.
  openFolder?: string | null
  // Called when the underlying Monaco editor mounts/unmounts. Used by
  // IdeContext to drive the animated-diff playback from outside the
  // editor component.
  onEditorReady?: (
    editor: editor.IStandaloneCodeEditor | null,
    monaco: typeof import('monaco-editor') | null,
  ) => void
}

// Single Monaco instance, model-swap on path change. Mounting a fresh
// Monaco per tab burns memory and recompiles workers; swapping the model
// is what VSCode does internally.
export default function MonacoHost({
  path,
  content,
  onChange,
  onSaveShortcut,
  readOnly,
  openFolder,
  onEditorReady,
}: MonacoHostProps) {
  const editorRef = useRef<editor.IStandaloneCodeEditor | null>(null)
  const monacoRef = useRef<typeof import('monaco-editor') | null>(null)
  // Track last path so we know when to flush content into the model.
  const lastPathRef = useRef<string | null>(null)

  const gitGutterRef = useRef<GitGutterController | null>(null)

  // Block the first editor mount until the codingame VSCode runtime is
  // up. Without this, MonacoLanguageClient.start() races the language
  // registry and providers silently no-op (see "Default api is not ready
  // yet, do not forget to import 'vscode/localExtensionHost'…").
  const [bootstrapReady, setBootstrapReady] = useState(false)
  useEffect(() => {
    let cancelled = false
    ensureVscodeBootstrap()
      .then(() => { if (!cancelled) setBootstrapReady(true) })
      .catch(e => console.warn('[MonacoHost] vscode bootstrap failed:', e))
    return () => { cancelled = true }
  }, [])

  const handleMount: OnMount = (editor, monaco) => {
    editorRef.current = editor
    monacoRef.current = monaco
    monaco.editor.setTheme('vs-dark')
    // Cmd/Ctrl+S → save.
    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, () => {
      onSaveShortcut()
    })
    // Register AI providers (hover, code actions, inline completions) once.
    registerAiProviders(monaco)
    // Bind the AI action runner to this editor so the lightbulb can invoke it.
    attachAiActionCommand(editor, monaco)
    onEditorReady?.(editor, monaco)
  }

  // Cleanup on unmount: drop the editor handle from the context so
  // animations never run on a stale instance.
  useEffect(() => {
    return () => {
      onEditorReady?.(null, null)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // When the *path* changes we want a model swap so each tab keeps its
  // own undo history. @monaco-editor/react does this automatically via
  // the `path` prop — Monaco creates one model per (path, value) pair
  // and reuses on re-mount. We just have to keep the prop stable.
  useEffect(() => {
    lastPathRef.current = path
  }, [path])

  // Lazily start an LSP client for the file's language whenever the
  // editor receives a path that we know how to handle. Gated on the
  // VSCode bootstrap being complete; otherwise the language client
  // races the runtime. Failures are non-fatal — the editor falls back
  // to vanilla Monaco.
  useEffect(() => {
    if (!bootstrapReady) return
    const lang = lspLanguageFor(path)
    if (!lang || !openFolder) return
    void ensureLspClient(openFolder, lang)
  }, [path, openFolder, bootstrapReady])

  // Git gutter: attach when the path changes, dispose on prior path.
  useEffect(() => {
    const editor = editorRef.current
    const monaco = monacoRef.current
    if (!editor || !monaco) return
    if (gitGutterRef.current) {
      gitGutterRef.current.dispose()
      gitGutterRef.current = null
    }
    gitGutterRef.current = attachGitGutter(editor, monaco, path)
    return () => {
      gitGutterRef.current?.dispose()
      gitGutterRef.current = null
    }
  }, [path])

  const handleChange: OnChange = (value) => {
    onChange(value ?? '')
  }

  // Build a proper file:// URI for the model. @monaco-editor/react passes
  // the `path` prop straight into monaco.Uri.parse, which does the wrong
  // thing with naked absolute paths (treats them as schemeless URIs).
  // The LSP server (pyright/tsserver) needs file:// URIs to resolve
  // imports against the workspace.
  const monacoPath = path.startsWith('/') ? `file://${path}` : path

  if (!bootstrapReady) {
    return (
      <div className="flex items-center justify-center h-full text-xs text-muted-foreground italic">
        Initializing IDE…
      </div>
    )
  }

  return (
    <Editor
      path={monacoPath}
      defaultLanguage={languageFor(path)}
      language={languageFor(path)}
      value={content}
      theme="vs-dark"
      onMount={handleMount}
      onChange={handleChange}
      options={{
        minimap: { enabled: true },
        fontSize: 13,
        lineHeight: 20,
        readOnly: readOnly ?? false,
        scrollBeyondLastLine: false,
        renderWhitespace: 'selection',
        wordWrap: 'off',
        automaticLayout: true,
        tabSize: 2,
        smoothScrolling: true,
      }}
    />
  )
}
