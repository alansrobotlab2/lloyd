/**
 * LspClient — connects Monaco's language client to a backend LSP proxy
 * via WebSocket. One client per (workspace, language). Lazily started
 * the first time a supported file is mounted in the editor.
 */

import { MonacoLanguageClient } from 'monaco-languageclient'
import { CloseAction, ErrorAction, type MessageTransports } from 'vscode-languageclient'
import { toSocket, WebSocketMessageReader, WebSocketMessageWriter } from 'vscode-ws-jsonrpc'
import { URI } from 'vscode-uri'

export type SupportedLanguage = 'python' | 'typescript'

const LANG_BY_EXT: Record<string, SupportedLanguage | undefined> = {
  py: 'python', pyi: 'python',
  ts: 'typescript', tsx: 'typescript',
  js: 'typescript', jsx: 'typescript',
  mjs: 'typescript', cjs: 'typescript',
}

const SERVER_DOC_LANGUAGES: Record<SupportedLanguage, string[]> = {
  python: ['python'],
  typescript: ['typescript', 'javascript', 'typescriptreact', 'javascriptreact'],
}

export function lspLanguageFor(path: string): SupportedLanguage | null {
  const dot = path.lastIndexOf('.')
  if (dot < 0) return null
  const ext = path.slice(dot + 1).toLowerCase()
  return LANG_BY_EXT[ext] ?? null
}

interface ClientEntry {
  workspace: string
  language: SupportedLanguage
  client: MonacoLanguageClient
  socket: WebSocket
}

const _clients = new Map<string, ClientEntry>()

function clientKey(workspace: string, language: SupportedLanguage): string {
  return `${workspace}::${language}`
}

/**
 * Ensure an LSP client for (workspace, language) is running. Returns the
 * client when it's started (or already running). Idempotent.
 *
 * Failures (server unreachable, language unsupported on backend) are
 * non-fatal — the editor continues without language intelligence.
 */
export async function ensureLspClient(
  workspace: string,
  language: SupportedLanguage,
): Promise<MonacoLanguageClient | null> {
  const key = clientKey(workspace, language)
  const existing = _clients.get(key)
  if (existing) return existing.client

  // Construct the WebSocket URL. Vite proxies /api → backend, so a
  // relative ws:// works regardless of cert/hostname juggling.
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  const url = `${protocol}//${window.location.host}/api/lsp/${language}?workspace=${encodeURIComponent(workspace)}`

  console.info('[lsp] connecting', { language, workspace, url })
  const socket = new WebSocket(url)

  return new Promise<MonacoLanguageClient | null>(resolve => {
    let settled = false

    socket.onopen = () => {
      console.info('[lsp] socket open', language)
      try {
        const transport = toSocket(socket)
        const reader = new WebSocketMessageReader(transport)
        const writer = new WebSocketMessageWriter(transport)
        const transports: MessageTransports = { reader, writer }

        const client = new MonacoLanguageClient({
          name: `Lloyd ${language} LS`,
          clientOptions: {
            documentSelector: SERVER_DOC_LANGUAGES[language].map(l => ({ language: l })),
            workspaceFolder: {
              uri: workspaceUri(workspace),
              name: workspace.split('/').pop() || workspace,
              index: 0,
            },
            errorHandler: {
              error: (e) => { console.warn('[lsp] error', language, e); return { action: ErrorAction.Continue } },
              closed: () => { console.warn('[lsp] closed', language); return { action: CloseAction.DoNotRestart } },
            },
          },
          messageTransports: transports,
        })

        client.start()
          .then(() => {
            console.info('[lsp] client started', language)
            _clients.set(key, { workspace, language, client, socket })
            if (!settled) { settled = true; resolve(client) }
          })
          .catch(e => {
            console.warn('[lsp] start failed:', e)
            if (!settled) { settled = true; resolve(null) }
          })
      } catch (e) {
        console.warn('[lsp] setup failed:', e)
        if (!settled) { settled = true; resolve(null) }
      }
    }

    socket.onerror = (e) => {
      console.warn('[lsp] socket error:', e)
      if (!settled) { settled = true; resolve(null) }
    }

    socket.onclose = (e) => {
      console.info('[lsp] socket close', language, e.code, e.reason)
      _clients.delete(key)
      if (!settled) { settled = true; resolve(null) }
    }
  })
}

/**
 * Stop and forget all LSP clients (called on workspace change).
 */
export function stopAllLspClients(): void {
  for (const entry of _clients.values()) {
    try { entry.client.stop() } catch { /* swallow */ }
    try { entry.socket.close() } catch { /* swallow */ }
  }
  _clients.clear()
}

/**
 * Stop only the client(s) bound to a specific workspace.
 */
export function stopLspClientsForWorkspace(workspace: string): void {
  for (const [key, entry] of _clients.entries()) {
    if (entry.workspace === workspace) {
      try { entry.client.stop() } catch { /* swallow */ }
      try { entry.socket.close() } catch { /* swallow */ }
      _clients.delete(key)
    }
  }
}

function workspaceUri(workspace: string): URI {
  return URI.file(workspace)
}
