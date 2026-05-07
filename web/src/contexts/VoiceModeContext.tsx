import { createContext, useCallback, useContext, useState, type ReactNode } from 'react'
import VoiceRoom, { type VoiceRoomState } from '../components/VoiceRoom'

/**
 * Global voice mode context. One VoiceRoom is mounted at the app level when
 * `enabled` is true and `sessionId` is set; its render-prop state is pushed
 * into the context so Sidebar / ChatPanel / VoicePreview can all read the
 * same RTC connection.
 *
 * Layout owns the `sessionId` decision — when the user toggles voice on
 * while on the chat page, Layout points the context at the visible chat
 * slot's session_key. Switching chat tabs while voice is enabled
 * reconnects to the new session.
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

export function VoiceModeProvider({ children }: { children: ReactNode }) {
  const [enabled, setEnabled] = useState(false)
  const [sessionId, setSessionId] = useState<string | null>(null)

  const engage = useCallback((id: string) => {
    setSessionId(id)
    setEnabled(true)
  }, [])

  const disengage = useCallback(() => {
    setEnabled(false)
  }, [])

  // Render the VoiceRoom only when both enabled and sessionId are set.
  // The render-prop pattern lets us forward its full state into the
  // context without VoiceRoom needing to know about the context.
  if (enabled && sessionId) {
    return (
      <VoiceRoom sessionId={sessionId} key={sessionId}>
        {(room) => (
          <VoiceModeContext.Provider
            value={{ enabled, sessionId, room, setEnabled, setSessionId, engage, disengage }}
          >
            {children}
          </VoiceModeContext.Provider>
        )}
      </VoiceRoom>
    )
  }

  return (
    <VoiceModeContext.Provider
      value={{ enabled, sessionId, room: null, setEnabled, setSessionId, engage, disengage }}
    >
      {children}
    </VoiceModeContext.Provider>
  )
}
