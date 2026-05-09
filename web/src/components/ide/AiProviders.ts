/**
 * AI providers — register Monaco hover, code-action, and inline-completion
 * providers backed by Lloyd. Each provider is registered globally per
 * language and is gated lightly so it doesn't interfere with the LSP
 * providers (which take precedence for everything they handle).
 *
 * Hover is Cmd-hover only — bare hover would be too chatty and conflict
 * with LSP hover.
 */

import { api } from '../../api'
import type { editor as Mon, IRange, IPosition, languages as MonLangs } from 'monaco-editor'

let _registered = false
let _activeMonaco: typeof import('monaco-editor') | null = null

// Cmd/Ctrl pressed flag — set on the document, gates AI hover.
let _modifierDown = false

function attachModifierTracking() {
  const onDown = (e: KeyboardEvent) => {
    if (e.key === 'Meta' || e.key === 'Control') _modifierDown = true
  }
  const onUp = (e: KeyboardEvent) => {
    if (e.key === 'Meta' || e.key === 'Control') _modifierDown = false
  }
  const onBlur = () => { _modifierDown = false }
  window.addEventListener('keydown', onDown)
  window.addEventListener('keyup', onUp)
  window.addEventListener('blur', onBlur)
}

const PYTHON_LANGS = ['python']
const JS_LANGS = ['typescript', 'javascript', 'typescriptreact', 'javascriptreact']
const ALL_LANGS = [...PYTHON_LANGS, ...JS_LANGS]

interface CompletionEntry {
  prefix: string
  text: string
}
let _lastCompletion: CompletionEntry | null = null

export function registerAiProviders(monaco: typeof import('monaco-editor')) {
  if (_registered && _activeMonaco === monaco) return
  _registered = true
  _activeMonaco = monaco
  attachModifierTracking()

  // ── Hover: Cmd/Ctrl-hover triggers an AI explanation ──────────────
  for (const lang of ALL_LANGS) {
    monaco.languages.registerHoverProvider(lang, {
      async provideHover(model, position) {
        if (!_modifierDown) return null
        const word = model.getWordAtPosition(position)
        if (!word) return null
        const symbol = word.word
        const code = model.getValue()
        try {
          const r = await api.ideAiHover({
            path: model.uri.fsPath,
            code,
            symbol,
            line: position.lineNumber,
            language: lang,
          })
          if (!r.markdown) return null
          return {
            range: new monaco.Range(
              position.lineNumber, word.startColumn,
              position.lineNumber, word.endColumn,
            ),
            contents: [
              { value: `**Lloyd:** ${symbol}` },
              { value: r.markdown },
            ],
          }
        } catch {
          return null
        }
      },
    })
  }

  // ── Code actions: explain / add docstring / add type hints / modernize ──
  for (const lang of ALL_LANGS) {
    monaco.languages.registerCodeActionProvider(lang, {
      provideCodeActions(model, range) {
        const actions: MonLangs.CodeAction[] = []
        const isPython = PYTHON_LANGS.includes(lang)
        const titles: Array<[string, string]> = [
          ['explain', 'Lloyd: Explain selection'],
          ['docstring', 'Lloyd: Add docstring'],
        ]
        if (isPython) {
          titles.push(['type_hints', 'Lloyd: Add type hints'])
        }
        titles.push(['modernize', 'Lloyd: Modernize'])

        for (const [action, title] of titles) {
          actions.push({
            title,
            kind: 'refactor.lloyd',
            // We don't apply a workspace edit synchronously here — the
            // command runs the AI call and applies the result. This keeps
            // the lightbulb decisions cheap (no network).
            command: {
              id: 'lloyd.runAiAction',
              title,
              arguments: [{ action, language: lang, range: range as IRange, modelUri: model.uri.toString() }],
            },
            isPreferred: action === 'explain',
          })
        }
        return { actions, dispose: () => {} }
      },
    })
  }

  // The lightbulb's command is registered on the editor instance — see
  // attachAiActionCommand below. We can't register a global command
  // here without an editor reference.

  // ── Inline completions: ghost-text on cursor change ────────────
  for (const lang of ALL_LANGS) {
    monaco.languages.registerInlineCompletionsProvider(lang, {
      async provideInlineCompletions(model: Mon.ITextModel, position: IPosition) {
        if (model.getLineCount() > 5000) return { items: [] }
        const offset = model.getOffsetAt(position)
        const all = model.getValue()
        const prefix = all.slice(0, offset)
        const suffix = all.slice(offset)
        // Reuse the previous completion if the user just typed its first chars.
        if (_lastCompletion && prefix.startsWith(_lastCompletion.prefix)) {
          const typedSince = prefix.slice(_lastCompletion.prefix.length)
          if (_lastCompletion.text.startsWith(typedSince)) {
            const remaining = _lastCompletion.text.slice(typedSince.length)
            if (remaining) {
              return {
                items: [{
                  insertText: remaining,
                  range: new monaco.Range(
                    position.lineNumber, position.column,
                    position.lineNumber, position.column,
                  ),
                }],
                enableForwardStability: true,
              }
            }
          }
        }
        return new Promise(resolve => {
          let acc = ''
          const ctrl = new AbortController()
          const timeout = setTimeout(() => ctrl.abort(), 6000)
          api.ideAiComplete({ prefix, suffix, language: lang }, (chunk) => {
            acc += chunk
            // Stop early when a sensible boundary appears so latency is bounded.
            if (acc.length > 240 || acc.includes('\n\n')) {
              ctrl.abort()
            }
          }, ctrl.signal)
            .catch(() => { /* swallow */ })
            .finally(() => {
              clearTimeout(timeout)
              const text = acc.split('\n\n')[0].trim()
              if (!text) {
                resolve({ items: [] })
                return
              }
              _lastCompletion = { prefix, text }
              resolve({
                items: [{
                  insertText: text,
                  range: new monaco.Range(
                    position.lineNumber, position.column,
                    position.lineNumber, position.column,
                  ),
                }],
                enableForwardStability: true,
              })
            })
        })
      },
      freeInlineCompletions() { /* no-op */ },
      disposeInlineCompletions() { /* no-op */ },
    } as unknown as MonLangs.InlineCompletionsProvider)
  }
  return
}

/**
 * Register the `lloyd.runAiAction` command on a specific editor instance.
 * Called from MonacoHost on mount so the code-action lightbulb knows what
 * to invoke.
 */
export function attachAiActionCommand(
  editor: Mon.IStandaloneCodeEditor,
  monaco: typeof import('monaco-editor'),
): void {
  // Monaco editor commands are registered with `addCommand` for keybindings
  // but for code-action invocations we need `registerCommand` on the global
  // CommandsRegistry. monaco-editor exposes this via `editor.editor.registerCommand`
  // in some versions; falling back to `addCommand` keyed to a fake keybinding
  // works because code actions invoke commands by id.
  // The simplest cross-version path: use `editor.addAction(...)` registered with
  // the same id so the command surface picks it up.
  editor.addAction({
    id: 'lloyd.runAiAction',
    label: 'Lloyd AI action',
    run: async (_ed, ...args) => {
      // The monaco code-action layer passes the command's `arguments`
      // through to addAction's run. We accept both shapes.
      const arg = args[0] as {
        action: string
        language: string
        range: IRange
        modelUri: string
      } | undefined
      if (!arg) return
      const model = monaco.editor.getModels().find(m => m.uri.toString() === arg.modelUri)
      if (!model) return
      const fullCode = model.getValue()
      const rangeText = model.getValueInRange(arg.range) || fullCode
      try {
        const r = await api.ideAiAction({
          path: model.uri.fsPath,
          code: fullCode,
          range_code: rangeText,
          action: arg.action,
          language: arg.language,
        })
        if (r.edit) {
          // Replace either the selection or the full file depending on action.
          const replaceRange = arg.action === 'modernize' || arg.action === 'type_hints'
            ? model.getFullModelRange()
            : arg.range
          editor.executeEdits('lloyd-ai-action', [{
            range: replaceRange,
            text: r.edit,
            forceMoveMarkers: true,
          }])
        } else if (r.result) {
          // Surface the explain text in a popover-style hover at the
          // top-of-selection. Easier: just open a Monaco "showCommand"
          // dialog. For v1 we use window.alert to avoid a heavier UI.
          // (You can later route this into a proper side panel.)
          window.alert(`Lloyd:\n\n${r.result}`)
        }
      } catch (e) {
        window.alert(`Lloyd action failed: ${e instanceof Error ? e.message : String(e)}`)
      }
    },
  })
}
