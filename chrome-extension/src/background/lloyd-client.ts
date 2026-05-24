// Minimal HTTP client for the Lloyd backend, used by the service worker
// only. The side panel uses the full `web/src/api.ts` (with all its
// session helpers); the SW only needs three operations.
//
// All calls hit http://127.0.0.1:8080 directly — the FastAPI mTLS
// middleware at server.py:76-113 skips loopback, so no client cert is
// required.

const API_BASE = "http://127.0.0.1:8080/api"

export async function createBrowserSession(): Promise<string> {
  const res = await fetch(`${API_BASE}/sessions/create`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ platform: "browser", inner_voice: true }),
  })
  if (!res.ok) {
    throw new Error(`createBrowserSession: ${res.status} ${await res.text()}`)
  }
  const data = await res.json()
  return data.session_key as string
}

export async function patchBrowserMetadata(
  sessionKey: string,
  patch: { url?: string; title?: string },
): Promise<void> {
  await fetch(
    `${API_BASE}/sessions/${encodeURIComponent(sessionKey)}/metadata`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    },
  )
}

// Fire a real user-turn against the session and abandon the SSE response
// immediately. Lloyd's /api/message/stream explicitly survives client
// disconnect (messages.py:10, 924-935) — the consumer keeps running on
// the server even though we never read the stream. We just need the POST
// to land.
export async function fireKickoff(
  sessionKey: string,
  text: string,
  clientId: string,
): Promise<void> {
  const res = await fetch(`${API_BASE}/message/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text,
      session_id: sessionKey,
      client_id: clientId,
    }),
  })
  // Close the body without reading it. The backend's generator is
  // cancelled on disconnect but the underlying turn continues.
  try {
    await res.body?.cancel()
  } catch {
    /* ignore */
  }
}
