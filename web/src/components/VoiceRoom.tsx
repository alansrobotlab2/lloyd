import { useEffect, useRef, useState } from 'react'
import {
  Room,
  RoomEvent,
  Track,
  DataPacket_Kind,
  type LocalAudioTrack,
  type RemoteAudioTrack,
  type RemoteTrack,
  type RemoteTrackPublication,
  type RemoteParticipant,
} from 'livekit-client'
import { RoomContext, RoomAudioRenderer, useTrackVolume } from '@livekit/components-react'
import type { AgentState } from '@livekit/components-react'
import { api } from '../api'

const AGENT_IDENTITY_PREFIX = 'lloyd-agent'
// Volume threshold (0..1 from useTrackVolume) above which we treat the agent
// track as "actively speaking". 0.01 is below noise floor; 0.04 catches even
// quiet TTS passages.
const AGENT_SPEAKING_THRESHOLD = 0.02

export type ConnectionStatus = 'idle' | 'connecting' | 'connected' | 'failed'

export interface VoiceRoomState {
  room: Room
  status: ConnectionStatus
  /** True once the local mic track is published. */
  micPublished: boolean
  /** True if the local mic is muted (e.g. half-duplex during agent TTS). */
  micMuted: boolean
  /** True while the agent track is producing audio above threshold. */
  agentSpeaking: boolean
  /** Local mic audio track once published — useful for input-side meters. */
  localAudioTrack: LocalAudioTrack | undefined
  /** Agent-published audio track (Lloyd's voice). Drives the aura visualizer
   *  in Phase 5A onward. Undefined until the worker publishes its TTS track. */
  agentAudioTrack: RemoteAudioTrack | undefined
  /** Best-effort agent state from connection lifecycle + agent track presence. */
  agentState: AgentState
  error: string | null
  /** Manually reconnect / retry. */
  reconnect: () => void
  /** Cancel the current agent turn + clear the worker's TTS queue. */
  interrupt: () => Promise<void>
}

interface VoiceRoomProps {
  sessionId: string
  /** Auto-publish the user's microphone after connecting. Default true. */
  publishMic?: boolean
  /** Mute the local mic publication while Lloyd is speaking, to kill the
   *  speakers→mic echo loop that otherwise produces fake transcripts.
   *  Default true. Disable if you have a clean headset and want full-duplex. */
  halfDuplex?: boolean
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
  halfDuplex = true,
  children,
}: VoiceRoomProps) {
  const roomRef = useRef<Room | null>(null)
  if (roomRef.current === null) roomRef.current = new Room()
  const room = roomRef.current

  const [status, setStatus] = useState<ConnectionStatus>('idle')
  const [micPublished, setMicPublished] = useState(false)
  const [micMuted, setMicMuted] = useState(false)
  const [localAudioTrack, setLocalAudioTrack] = useState<LocalAudioTrack | undefined>(undefined)
  const [agentAudioTrack, setAgentAudioTrack] = useState<RemoteAudioTrack | undefined>(undefined)
  const [error, setError] = useState<string | null>(null)
  const [reconnectKey, setReconnectKey] = useState(0)

  // Watch agent track volume — drives the "speaking" detection that
  // toggles half-duplex mute and the visualizer state.
  const agentVolume = useTrackVolume(agentAudioTrack, {
    fftSize: 256,
    smoothingTimeConstant: 0.4,
  })
  const agentSpeaking = agentVolume > AGENT_SPEAKING_THRESHOLD

  // Half-duplex: while the agent's track has audio above threshold, mute
  // the local mic so the speakers don't feed back into the ASR pipeline.
  // Restored automatically once the agent stops.
  useEffect(() => {
    if (!halfDuplex || !micPublished) return
    const targetMuted = agentSpeaking
    if (targetMuted === micMuted) return
    let cancelled = false
    ;(async () => {
      try {
        await room.localParticipant.setMicrophoneEnabled(!targetMuted)
        if (!cancelled) setMicMuted(targetMuted)
      } catch (e) {
        console.warn('half-duplex mic toggle failed:', e)
      }
    })()
    return () => { cancelled = true }
  }, [halfDuplex, micPublished, agentSpeaking, micMuted, room])

  useEffect(() => {
    let cancelled = false

    const onDisconnected = () => {
      if (cancelled) return
      setStatus('idle')
      setMicPublished(false)
      setLocalAudioTrack(undefined)
      setAgentAudioTrack(undefined)
    }
    const onReconnecting = () => {
      if (cancelled) return
      setStatus('connecting')
    }
    const onReconnected = () => {
      if (cancelled) return
      setStatus('connected')
    }

    // Surface the agent's published audio track once it appears in the room.
    const onTrackSubscribed = (
      track: RemoteTrack,
      _publication: RemoteTrackPublication,
      participant: RemoteParticipant,
    ) => {
      if (cancelled) return
      if (track.kind !== Track.Kind.Audio) return
      if (!participant.identity.startsWith(AGENT_IDENTITY_PREFIX)) return
      setAgentAudioTrack(track as RemoteAudioTrack)
    }
    const onTrackUnsubscribed = (track: RemoteTrack) => {
      if (cancelled) return
      if (track.kind !== Track.Kind.Audio) return
      setAgentAudioTrack(prev => (prev && prev.sid === track.sid ? undefined : prev))
    }

    room.on(RoomEvent.Disconnected, onDisconnected)
    room.on(RoomEvent.Reconnecting, onReconnecting)
    room.on(RoomEvent.Reconnected, onReconnected)
    room.on(RoomEvent.TrackSubscribed, onTrackSubscribed)
    room.on(RoomEvent.TrackUnsubscribed, onTrackUnsubscribed)

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
          // Secure-context check: browsers only expose getUserMedia on
          // localhost or HTTPS. Detect it explicitly so the user sees a
          // useful message instead of "Cannot read properties of undefined".
          if (typeof navigator === 'undefined' || !navigator.mediaDevices?.getUserMedia) {
            const host = typeof window !== 'undefined' ? window.location.host : '?'
            const isSecure = typeof window !== 'undefined' && window.isSecureContext
            const hint = isSecure
              ? 'Browser does not expose mediaDevices on this origin.'
              : `Browser blocks getUserMedia on http://${host}. Load mc via http://localhost:5173/ or set up HTTPS.`
            console.warn('[VoiceRoom]', hint)
            setError(hint)
          } else {
            try {
              // Explicit AEC + noise suppression + AGC. Defaults are
              // usually true but spelling them out keeps half-duplex
              // mute and echo cancellation cooperating predictably.
              await room.localParticipant.setMicrophoneEnabled(true, {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
              })
              if (cancelled) return
              const pub = room.localParticipant.getTrackPublication(Track.Source.Microphone)
              const track = pub?.audioTrack as LocalAudioTrack | undefined
              setLocalAudioTrack(track)
              setMicPublished(!!track)
            } catch (e) {
              // Permission denied, no device, or revoked — keep the room
              // alive but surface the error to the UI.
              console.warn('mic publish failed:', e)
              setError(e instanceof Error ? e.message : String(e))
            }
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
      room.off(RoomEvent.TrackSubscribed, onTrackSubscribed)
      room.off(RoomEvent.TrackUnsubscribed, onTrackUnsubscribed)
      // Disconnect, but keep the Room instance for re-use on reconnectKey bump.
      room.disconnect().catch(() => {})
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId, publishMic, reconnectKey])

  // 'speaking' only when the agent track is actually producing audio
  // (volume above threshold). Otherwise 'listening' if user mic is on,
  // matching the LiveKit AgentState convention.
  const agentState: AgentState =
    status === 'connecting'
      ? 'connecting'
      : status === 'failed'
      ? 'disconnected'
      : agentSpeaking
      ? 'speaking'
      : status === 'connected' && (micPublished || agentAudioTrack)
      ? 'listening'
      : 'idle'

  const interrupt = async () => {
    // 1. Tell the worker to stop TTS immediately, via LiveKit data channel.
    try {
      const data = new TextEncoder().encode(JSON.stringify({ type: 'interrupt' }))
      await room.localParticipant.publishData(data, { reliable: true, kind: DataPacket_Kind.RELIABLE })
    } catch (e) {
      console.warn('interrupt: data send failed', e)
    }
    // 2. Cancel the in-flight harness turn (drains any queued ambient too).
    try {
      await fetch(
        `/api/sessions/${encodeURIComponent(sessionId)}/cancel?drain_pending=true`,
        { method: 'POST' },
      )
    } catch (e) {
      console.warn('interrupt: cancel POST failed', e)
    }
  }

  return (
    <RoomContext.Provider value={room}>
      {children({
        room,
        status,
        micPublished,
        micMuted,
        agentSpeaking,
        localAudioTrack,
        agentAudioTrack,
        agentState,
        error,
        reconnect: () => setReconnectKey(k => k + 1),
        interrupt,
      })}
      {/* Plays all remote audio tracks (Lloyd's TTS) through the speakers.
          Hidden — it just needs to be mounted somewhere inside RoomContext. */}
      <RoomAudioRenderer />
    </RoomContext.Provider>
  )
}
