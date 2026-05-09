/**
 * One-time bootstrap of the codingame VSCode runtime.
 *
 * Wraps `MonacoVscodeApiWrapper` from monaco-languageclient — which
 * internally handles all the gnarly setup that broke last time we tried
 * to call `initialize()` directly:
 *   • imports `vscode/localExtensionHost` so the language client's
 *     DidOpenTextDocumentFeature has a usable vscode api
 *   • configures Monaco's web workers
 *   • loads default themes + language extensions
 *   • registers the file/extension service overrides
 *
 * Idempotent. Returns a Promise that resolves when the runtime is ready;
 * any caller that mounts a Monaco editor or starts a language client
 * must `await` it first.
 */

import { MonacoVscodeApiWrapper } from 'monaco-languageclient/vscodeApiWrapper'
import { configureDefaultWorkerFactory } from 'monaco-languageclient/workerFactory'

// Default theme + language grammar extensions. These register the
// TextMate grammars that drive syntax highlighting.
import '@codingame/monaco-vscode-theme-defaults-default-extension'
import '@codingame/monaco-vscode-python-default-extension'
import '@codingame/monaco-vscode-typescript-basics-default-extension'
import '@codingame/monaco-vscode-javascript-default-extension'
import '@codingame/monaco-vscode-yaml-default-extension'
import '@codingame/monaco-vscode-shellscript-default-extension'
import '@codingame/monaco-vscode-markdown-basics-default-extension'
import '@codingame/monaco-vscode-rust-default-extension'
import '@codingame/monaco-vscode-go-default-extension'
import '@codingame/monaco-vscode-html-default-extension'
import '@codingame/monaco-vscode-css-default-extension'
import '@codingame/monaco-vscode-json-default-extension'
import '@codingame/monaco-vscode-xml-default-extension'

let _readyPromise: Promise<MonacoVscodeApiWrapper> | null = null
let _wrapper: MonacoVscodeApiWrapper | null = null

export function ensureVscodeBootstrap(): Promise<MonacoVscodeApiWrapper> {
  if (_readyPromise) return _readyPromise
  _readyPromise = (async () => {
    const wrapper = new MonacoVscodeApiWrapper({
      $type: 'extended',
      // EditorService = the lightweight views config; we mount editors
      // via @monaco-editor/react ourselves rather than through their
      // workbench. This keeps our React tree as the source of truth.
      viewsConfig: { $type: 'EditorService' },
      monacoWorkerFactory: configureDefaultWorkerFactory,
      userConfiguration: {
        json: JSON.stringify({
          'workbench.colorTheme': 'Default Dark Modern',
          'editor.fontSize': 13,
          'editor.minimap.enabled': true,
          'editor.tabSize': 2,
          'editor.scrollBeyondLastLine': false,
          'editor.wordBasedSuggestions': 'off',
          'editor.semanticHighlighting.enabled': true,
        }),
      },
    })
    await wrapper.start({ caller: 'lloyd-ide' })
    _wrapper = wrapper
    return wrapper
  })().catch(e => {
    console.warn('[vscodeBootstrap] failed:', e)
    _readyPromise = null
    throw e
  })
  return _readyPromise
}

export function getVscodeWrapper(): MonacoVscodeApiWrapper | null {
  return _wrapper
}
