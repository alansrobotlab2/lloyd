import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from 'react'
import VoiceRoom, { type VoiceRoomState } from '../components/VoiceRoom'

/**
 * Global voice mode context. One VoiceRoom is mounted at the app level when
 * `enabled` is true and `sessionId` is set; its state is pushed into the
 * context so Sidebar / ChatPanel / VoicePreview can read the same RTC
 * connection.
 *
 * IMPORTANT — the outer JSX tree must stay stable across enabled/disabled.
 * Earlier versions wrapped `children` in `<VoiceRoom>` when enabled and
 * left it bare when disabled; toggling voice changed the tree's root and
 * caused React to remount every descendant (Layout, page state, in-flight
 * chat, the chat sidebar's open/closed flag — all reset). The current
 * structure keeps `<Provider>` as the only outer wrapper and renders the
 * VoiceRoom as a sibling to `children`, so toggling voice no longer
 * disturbs the tree.
 */

export interface VoiceModeContextValue {
  enabled: boolean
  sessionId: string | null
  room: VoiceRoomState | null
  setEnabled: (next: boolean) => void
  setSessionId: (id: string | null) => void
  /** Convenience: enable + bind to a specific session in one call. */
  engage: (sessionId: string) => void
  /** Convenience: turn off and forget the session. */
  disengage: () => void
}

const VoiceModeContext = createContext<VoiceModeContextValue | null>(null)

export function useVoiceMode(): VoiceModeContextValue {
  const ctx = useContext(VoiceModeContext)
  if (!ctx) throw new Error('useVoiceMode must be used inside <VoiceModeProvider>')
  return ctx
}

/** Tiny consumer that syncs the VoiceRoom's render-prop state up to the
 *  provider via a setter. Renders nothing — it's just a state pump so the
 *  provider can hold the room reference and surface it through context. */
function VoiceRoomStateBridge({
  room,
  setRoom,
}: {
  room: VoiceRoomState
  setRoom: (s: VoiceRoomState | null) => void
}) {
  useEffect(() => {
    setRoom(room)
    return () => setRoom(null)
  }, [room, setRoom])
  return null
}

export function VoiceModeProvider({ children }: { children: ReactNode }) {
  const [enabled, setEnabled] = useState(false)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [room, setRoom] = useState<VoiceRoomState | null>(null)

  const engage = useCallback((id: string) => {
    setSessionId(id)
    setEnabled(true)
  }, [])

  const disengage = useCallback(() => {
    setEnabled(false)
  }, [])

  return (
    <VoiceModeContext.Provider
      value={{ enabled, sessionId, room, setEnabled, setSessionId, engage, disengage }}
    >
      {/* Sibling, not ancestor: keeps the outer tree stable across
          engaged/disengaged so children don't remount. */}
      {enabled && sessionId && (
        <VoiceRoom sessionId={sessionId} key={sessionId}>
          {(roomState) => <VoiceRoomStateBridge room={roomState} setRoom={setRoom} />}
        </VoiceRoom>
      )}
      {children}
    </VoiceModeContext.Provider>
  )
}
