import { createContext, useContext, useState, useEffect, useCallback, type ReactNode } from "react";
import { useVoiceStream, getPersistedMicEnabled } from "../hooks/useVoiceStream";

interface VoiceContextValue {
  isConnected: boolean;
  isListening: boolean;
  isSpeaking: boolean;
  wakewordDetected: boolean;
  pipelineState: string;
  transcripts: Array<{ text: string; speaker: string; is_continuity: boolean; timestamp: number }>;
  startMic: (sessionKey?: string) => Promise<void>;
  stopMic: () => void;
  clearTranscripts: () => void;
  wsAvailable: boolean;
  statusLoaded: boolean;
  voiceEnabled: boolean;
  handleVoiceToggle: () => Promise<void>;
  ttsEnabled: boolean;
  handleTtsToggle: () => void;
  latestTranscript: string | undefined;
  transcriptVisible: boolean;
  stateColor: string;
  stateText: string;
  updateSessionKey: (sessionKey: string) => void;
}

const VoiceContext = createContext<VoiceContextValue | null>(null);

export function useVoiceContext(): VoiceContextValue {
  const ctx = useContext(VoiceContext);
  if (!ctx) throw new Error("useVoiceContext must be used within VoiceProvider");
  return ctx;
}

export function VoiceProvider({ children, activeSessionKey }: { children: ReactNode; activeSessionKey?: string | null }) {
  const [wsPort, setWsPort] = useState(8095);
  const [wsAvailable, setWsAvailable] = useState(false);
  const [statusLoaded, setStatusLoaded] = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(() => {
    try { return localStorage.getItem("voice-power-enabled") === "true"; } catch { return false; }
  });
  const [ttsEnabled, setTtsEnabled] = useState(() => {
    try { return localStorage.getItem("tts-enabled") === "true"; } catch { return false; }
  });
  const [transcriptVisible, setTranscriptVisible] = useState(true);

  const {
    isConnected, isListening, isSpeaking, wakewordDetected,
    pipelineState, transcripts, startMic, stopMic, clearTranscripts, updateSessionKey,
  } = useVoiceStream(wsPort);

  // Poll voice status
  useEffect(() => {
    const fetchStatus = () =>
      fetch("/api/mc/voice-status", { credentials: "include" as RequestCredentials })
        .then(r => r.json())
        .then(data => {
          setWsAvailable(data.ws_active);
          if (data.ws_port) setWsPort(data.ws_port);
          setVoiceEnabled(data.voice_enabled);
          setStatusLoaded(true);
          try { localStorage.setItem("voice-power-enabled", String(data.voice_enabled)); } catch {}
          if (!data.ws_active) setTimeout(fetchStatus, 2000);
        })
        .catch(() => setWsAvailable(false));
    fetchStatus();
    const id = setInterval(fetchStatus, 10000);
    return () => clearInterval(id);
  }, []);

  // Auto-start mic after refresh if voice pipeline is active, enabled, AND mic was previously on
  useEffect(() => {
    if (wsAvailable && voiceEnabled && !isListening && getPersistedMicEnabled()) {
      try { startMic(activeSessionKey || undefined); } catch { /* getUserMedia may need user gesture */ }
    }
  }, [wsAvailable, voiceEnabled]);

  // Sync session key to the voice pipeline whenever it changes while mic is active
  useEffect(() => {
    if (activeSessionKey && isListening) {
      updateSessionKey(activeSessionKey);
    }
  }, [activeSessionKey, isListening, updateSessionKey]);

  // Voice toggle handler
  const handleVoiceToggle = useCallback(async () => {
    try {
      const resp = await fetch("/api/mc/voice-toggle", { method: "POST", credentials: "include" as RequestCredentials });
      const data = await resp.json();
      if (data.voice_enabled !== undefined) {
        setVoiceEnabled(data.voice_enabled);
        try { localStorage.setItem("voice-power-enabled", String(data.voice_enabled)); } catch {}
      }
    } catch {
      // ignore
    }
  }, []);

  const handleTtsToggle = useCallback(async () => {
    const next = !ttsEnabled;
    setTtsEnabled(next);
    try { localStorage.setItem("tts-enabled", String(next)); } catch {}
  }, [ttsEnabled]);

  // Latest transcript + 5s fade
  const latestTranscript = transcripts[transcripts.length - 1]?.text;
  useEffect(() => {
    if (!latestTranscript) return;
    setTranscriptVisible(true);
    const timer = setTimeout(() => setTranscriptVisible(false), 5000);
    return () => clearTimeout(timer);
  }, [latestTranscript]);

  // Computed state color and text
  const stateColor = pipelineState === "LISTENING" ? "text-green-400" :
                     pipelineState === "PROCESSING" ? "text-yellow-400" :
                     pipelineState === "SPEAKING" ? "text-orange-400" :
                     pipelineState === "AWAITING_WAKEWORD" ? "text-blue-400" :
                     pipelineState === "ACTIVE_LISTEN" ? "text-cyan-400" : "text-slate-500";
  const stateText = pipelineState === "LISTENING" ? "Listening..." :
                    pipelineState === "PROCESSING" ? "Processing..." :
                    pipelineState === "SPEAKING" ? "Speaking..." :
                    pipelineState === "AWAITING_WAKEWORD" ? "Awaiting wake word..." :
                    pipelineState === "ACTIVE_LISTEN" ? "Listening for follow-up..." : "Idle";

  return (
    <VoiceContext.Provider value={{
      isConnected, isListening, isSpeaking, wakewordDetected, pipelineState,
      transcripts, startMic, stopMic, clearTranscripts, updateSessionKey,
      wsAvailable, statusLoaded, voiceEnabled, handleVoiceToggle,
      ttsEnabled, handleTtsToggle,
      latestTranscript, transcriptVisible, stateColor, stateText,
    }}>
      {children}
    </VoiceContext.Provider>
  );
}
