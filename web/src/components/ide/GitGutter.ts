/**
 * Git gutter — fetches `git diff` hunks for a path and applies Monaco
 * line decorations on the gutter (M/A/D color bars).
 */

import { api, type GitHunk } from '../../api'
import type { editor as Mon } from 'monaco-editor'

export interface GitGutterController {
  refresh(): Promise<void>
  dispose(): void
}

const ADD_CLASS = 'ide-git-add'
const MOD_CLASS = 'ide-git-mod'
const DEL_CLASS = 'ide-git-del'

export function attachGitGutter(
  editor: Mon.IStandaloneCodeEditor,
  monaco: typeof import('monaco-editor'),
  path: string,
): GitGutterController {
  let collection: Mon.IEditorDecorationsCollection | null = null
  let disposed = false

  const apply = (hunks: GitHunk[]) => {
    if (disposed) return
    const decos: Mon.IModelDeltaDecoration[] = []
    for (const h of hunks) {
      const start = Math.max(1, h.new_start)
      // Pure deletions get marked at the *line above* (where the deletion appears).
      if (h.type === 'delete') {
        decos.push({
          range: new monaco.Range(Math.max(1, start), 1, Math.max(1, start), 1),
          options: {
            isWholeLine: false,
            linesDecorationsClassName: DEL_CLASS,
          },
        })
        continue
      }
      const end = h.new_count > 0 ? start + h.new_count - 1 : start
      const klass = h.type === 'add' ? ADD_CLASS : MOD_CLASS
      decos.push({
        range: new monaco.Range(start, 1, end, 1),
        options: {
          isWholeLine: false,
          linesDecorationsClassName: klass,
        },
      })
    }
    if (collection) collection.clear()
    collection = editor.createDecorationsCollection(decos)
  }

  const refresh = async () => {
    try {
      const r = await api.ideGitDiff(path)
      apply(r.hunks)
    } catch {
      // Non-fatal — leave gutter empty.
    }
  }

  // Refresh on mount + on user save (Ctrl/Cmd+S triggers a save in IdeContext;
  // file_changed events come back over SSE and IdePage pings us via refresh()
  // explicitly when needed).
  void refresh()

  return {
    refresh,
    dispose() {
      disposed = true
      if (collection) collection.clear()
      collection = null
    },
  }
}
