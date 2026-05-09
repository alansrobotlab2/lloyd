import { useEffect, useRef } from 'react'
import Editor, { loader, type OnMount, type OnChange } from '@monaco-editor/react'
import type { editor } from 'monaco-editor'

// Use the locally-installed monaco package rather than fetching from a CDN
// so the IDE keeps working without internet (the rest of Lloyd does).
import * as monacoNs from 'monaco-editor'
loader.config({ monaco: monacoNs })

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
  onEditorReady,
}: MonacoHostProps) {
  const editorRef = useRef<editor.IStandaloneCodeEditor | null>(null)
  const monacoRef = useRef<typeof import('monaco-editor') | null>(null)
  // Track last path so we know when to flush content into the model.
  const lastPathRef = useRef<string | null>(null)

  const handleMount: OnMount = (editor, monaco) => {
    editorRef.current = editor
    monacoRef.current = monaco
    monaco.editor.setTheme('vs-dark')
    // Cmd/Ctrl+S → save.
    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, () => {
      onSaveShortcut()
    })
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

  const handleChange: OnChange = (value) => {
    onChange(value ?? '')
  }

  return (
    <Editor
      path={path}
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
