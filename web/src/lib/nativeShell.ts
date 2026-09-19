// Bridge to the lloyd-ios native shell (a WKWebView around this app).
//
// Inside the shell, voice is owned natively: the app joins the LiveKit room
// itself so it survives the screen locking and can use the earbud mic, and
// wakes Lloyd from an earbud / Action Button press. The web app must not join
// the room too (two mics in one room), and tells the shell which chat session
// is current so both talk to the same one.

type NativeMessage = { type: 'session'; sessionId: string | null }

interface WebkitBridge {
  messageHandlers?: { lloyd?: { postMessage: (msg: NativeMessage) => void } }
}

function bridge() {
  if (typeof window === 'undefined') return undefined
  return (window as unknown as { webkit?: WebkitBridge }).webkit?.messageHandlers?.lloyd
}

export const isNativeShell = bridge() !== undefined

export function postToNative(msg: NativeMessage): void {
  try {
    bridge()?.postMessage(msg)
  } catch {
    // Shell went away mid-navigation — nothing to tell.
  }
}
