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
  if (!res.ok) {
    // A rejected POST enqueues nothing, so there is no turn and the panel
    // would sit on a session with nothing ever arriving. Throw rather than
    // resolve quietly: the caller logs it, and a 503 from the self-mod drain
    // window is at least findable in the service worker console.
    throw new Error(`fireKickoff: ${res.status} ${await res.text()}`)
  }
  // Close the body without reading it. The backend's generator is
  // cancelled on disconnect but the underlying turn continues. Resolving
  // here is also the caller's barrier — the turn is enqueued by the time
  // the response headers arrive, so awaiting this call before showing the
  // panel the session is what lets the in-progress indicator appear.
  try {
    await res.body?.cancel()
  } catch {
    /* ignore */
  }
}
