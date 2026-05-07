import { useEffect, useRef, useState } from 'react'
import { Room, RoomEvent, Track, type LocalAudioTrack } from 'livekit-client'
import { RoomContext } from '@livekit/components-react'
import type { AgentState } from '@livekit/components-react'
import { api } from '../api'

export type ConnectionStatus = 'idle' | 'connecting' | 'connected' | 'failed'

export interface VoiceRoomState {
  room: Room
  status: ConnectionStatus
  /** True once the local mic track is published. */
  micPublished: boolean
  /** Local mic audio track once published — feeds the aura visualizer. */
  localAudioTrack: LocalAudioTrack | undefined
  /** Best-effort agent state mapped from connection lifecycle. */
  agentState: AgentState
  error: string | null
  /** Manually reconnect / retry. */
  reconnect: () => void
}

interface VoiceRoomProps {
  sessionId: string
  /** Auto-publish the user's microphone after connecting. Default true. */
  publishMic?: boolean
  /** Render-prop receiving the room state. */
  children: (state: VoiceRoomState) => React.ReactNode
}

/**
 * Phase 3 LiveKit room wrapper.
 *
 * Mints a token for `sessionId`, connects to the LiveKit server, optionally
 * publishes the mic, and provides the room via `<RoomContext>` so any
 * `@livekit/components-react` hooks (`useTrackVolume`, `useVoiceAssistant`,
 * etc.) inside `children` work.
 *
 * The `agentState` returned is a coarse mapping of the connection lifecycle
 * to the `AgentState` enum so the aura visualizer animates appropriately
 * even before the actual server-side agent is talking back.
 */
export default function VoiceRoom({
  sessionId,
  publishMic = true,
  children,
}: VoiceRoomProps) {
  const roomRef = useRef<Room | null>(null)
  if (roomRef.current === null) roomRef.current = new Room()
  const room = roomRef.current

  const [status, setStatus] = useState<ConnectionStatus>('idle')
  const [micPublished, setMicPublished] = useState(false)
  const [localAudioTrack, setLocalAudioTrack] = useState<LocalAudioTrack | undefined>(undefined)
  const [error, setError] = useState<string | null>(null)
  const [reconnectKey, setReconnectKey] = useState(0)

  useEffect(() => {
    let cancelled = false

    const onDisconnected = () => {
      if (cancelled) return
      setStatus('idle')
      setMicPublished(false)
      setLocalAudioTrack(undefined)
    }
    const onReconnecting = () => {
      if (cancelled) return
      setStatus('connecting')
    }
    const onReconnected = () => {
      if (cancelled) return
      setStatus('connected')
    }

    room.on(RoomEvent.Disconnected, onDisconnected)
    room.on(RoomEvent.Reconnecting, onReconnecting)
    room.on(RoomEvent.Reconnected, onReconnected)

    const connect = async () => {
      try {
        setStatus('connecting')
        setError(null)
        const t = await api.livekitToken(sessionId)
        if (cancelled) return
        await room.connect(t.url, t.token, { autoSubscribe: true })
        if (cancelled) return
        setStatus('connected')

        if (publishMic) {
          try {
            await room.localParticipant.setMicrophoneEnabled(true)
            if (cancelled) return
            const pub = room.localParticipant.getTrackPublication(Track.Source.Microphone)
            const track = pub?.audioTrack as LocalAudioTrack | undefined
            setLocalAudioTrack(track)
            setMicPublished(!!track)
          } catch (e) {
            // Mic permission denied or no device — keep the room alive but
            // surface the error so the UI can prompt the user.
            console.warn('mic publish failed:', e)
            setError(e instanceof Error ? e.message : String(e))
          }
        }
      } catch (e) {
        if (cancelled) return
        console.error('LiveKit connect failed:', e)
        setStatus('failed')
        setError(e instanceof Error ? e.message : String(e))
      }
    }

    connect()

    return () => {
      cancelled = true
      room.off(RoomEvent.Disconnected, onDisconnected)
      room.off(RoomEvent.Reconnecting, onReconnecting)
      room.off(RoomEvent.Reconnected, onReconnected)
      // Disconnect, but keep the Room instance for re-use on reconnectKey bump.
      room.disconnect().catch(() => {})
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId, publishMic, reconnectKey])

  const agentState: AgentState =
    status === 'connecting'
      ? 'connecting'
      : status === 'failed'
      ? 'disconnected'
      : status === 'connected' && micPublished
      ? 'listening'
      : 'idle'

  return (
    <RoomContext.Provider value={room}>
      {children({
        room,
        status,
        micPublished,
        localAudioTrack,
        agentState,
        error,
        reconnect: () => setReconnectKey(k => k + 1),
      })}
    </RoomContext.Provider>
  )
}
