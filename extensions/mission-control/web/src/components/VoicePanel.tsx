import { useState, useEffect } from "react";
import { Mic, MicOff, Power, Radio, Volume2 } from "lucide-react";
import { useVoiceStream } from "../hooks/useVoiceStream";

interface VoicePanelProps {
  onVoiceActive?: (active: boolean) => void;
  onVoiceEnabled?: (enabled: boolean) => void;
}

export default function VoicePanel({ onVoiceActive, onVoiceEnabled }: VoicePanelProps) {
  const [wsPort, setWsPort] = useState(8095);
  const [expanded, setExpanded] = useState(false);
  const [transcriptVisible, setTranscriptVisible] = useState(true);
  const { isConnected, isListening, isSpeaking, wakewordDetected, pipelineState, transcripts, startMic, stopMic, clearTranscripts } = useVoiceStream(wsPort);

  // Check if voice WS mode is available
  const [wsAvailable, setWsAvailable] = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(false);
  useEffect(() => {
    fetch("/api/mc/voice-status", { credentials: "include" as RequestCredentials })
      .then(r => r.json())
      .then(data => {
        setWsAvailable(data.ws_active);
        if (data.ws_port) setWsPort(data.ws_port);
        setVoiceEnabled(data.voice_enabled);
      })
      .catch(() => setWsAvailable(false));
    const id = setInterval(() => {
      fetch("/api/mc/voice-status", { credentials: "include" as RequestCredentials })
        .then(r => r.json())
        .then(data => {
          setWsAvailable(data.ws_active);
          setVoiceEnabled(data.voice_enabled);
        })
        .catch(() => setWsAvailable(false));
    }, 10000);
    return () => clearInterval(id);
  }, []);

  const handleVoiceToggle = async () => {
    try {
      const resp = await fetch("/api/mc/voice-toggle", { method: "POST", credentials: "include" as RequestCredentials });
      const data = await resp.json();
      if (data.voice_enabled !== undefined) setVoiceEnabled(data.voice_enabled);
    } catch {
      // ignore
    }
  };

  useEffect(() => {
    onVoiceActive?.(isListening);
  }, [isListening, onVoiceActive]);

  useEffect(() => {
    onVoiceEnabled?.(voiceEnabled);
  }, [voiceEnabled, onVoiceEnabled]);

  const latestTranscript = transcripts[transcripts.length - 1]?.text;
  useEffect(() => {
    if (!latestTranscript) return;
    setTranscriptVisible(true);
    const timer = setTimeout(() => setTranscriptVisible(false), 5000);
    return () => clearTimeout(timer);
  }, [latestTranscript]);

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
    <div className="border-b border-surface-3/30">
      {/* Compact bar */}
      <div className="flex items-center gap-2 px-3 py-1.5">
        <button
          onClick={wsAvailable ? () => (isListening ? stopMic() : startMic()) : undefined}
          className={`p-1.5 rounded-lg transition-colors ${
            !wsAvailable
              ? "bg-surface-2 text-slate-600 cursor-not-allowed opacity-50"
              : isListening
                ? "bg-green-600/20 text-green-400 hover:bg-green-600/30"
                : "bg-red-600/20 text-red-400 hover:bg-red-600/30"
          }`}
          title={!wsAvailable ? "Voice service unavailable" : isListening ? "Stop mic" : "Start mic"}
          disabled={!wsAvailable}
        >
          {isListening ? <Mic className="w-4 h-4" /> : <MicOff className="w-4 h-4" />}
        </button>

        <button
          onClick={handleVoiceToggle}
          className={`p-1.5 rounded-lg transition-colors ${
            voiceEnabled
              ? "bg-green-600/20 text-green-400 hover:bg-green-600/30"
              : "bg-surface-2 text-slate-500 hover:text-slate-300 hover:bg-surface-3"
          }`}
          title={voiceEnabled ? "Disable voice" : "Enable voice"}
        >
          <Power className="w-4 h-4" />
        </button>

        <span className={`text-[10px] font-mono ${wsAvailable ? stateColor : "text-slate-600"}`}>
          {wsAvailable ? stateText : "Voice offline"}
        </span>

        {isConnected && (
          <Radio className="w-3 h-3 text-green-500" />
        )}

        {isSpeaking && (
          <Volume2 className="w-3.5 h-3.5 text-blue-400 animate-pulse" />
        )}

        {wakewordDetected && (
          <span className="text-[10px] text-green-400 animate-pulse font-medium">
            Wake word!
          </span>
        )}

        {transcripts.length > 0 && (
          <span className={`text-xs text-slate-400 truncate flex-1 ml-2 transition-opacity duration-1000 ${transcriptVisible ? "opacity-100" : "opacity-0"}`}>
            {transcripts[transcripts.length - 1]?.text}
          </span>
        )}

        <button
          onClick={() => setExpanded(!expanded)}
          className="text-slate-500 hover:text-slate-300 text-[10px] ml-auto"
        >
          {expanded ? "Hide" : "Show"} transcripts
        </button>
      </div>

      {/* Expanded transcript feed */}
      {expanded && transcripts.length > 0 && (
        <div className="px-3 pb-2 max-h-32 overflow-y-auto space-y-1">
          {transcripts.map((t, i) => (
            <div key={i} className="text-xs flex gap-2">
              <span className="text-slate-500 font-mono">{t.speaker}</span>
              <span className={`${t.is_continuity ? "text-slate-400" : "text-slate-200"}`}>
                {t.text}
              </span>
            </div>
          ))}
          <button onClick={clearTranscripts} className="text-[10px] text-slate-600 hover:text-slate-400">
            Clear
          </button>
        </div>
      )}
    </div>
  );
}
