import { useState, useEffect } from "react";
import { Mic, MicOff, Radio, Volume2 } from "lucide-react";
import { useVoiceStream } from "../hooks/useVoiceStream";

interface VoicePanelProps {
  onVoiceActive?: (active: boolean) => void;
}

export default function VoicePanel({ onVoiceActive }: VoicePanelProps) {
  const [wsPort, setWsPort] = useState(8095);
  const [expanded, setExpanded] = useState(false);
  const { isConnected, isListening, isSpeaking, pipelineState, transcripts, startMic, stopMic, clearTranscripts } = useVoiceStream(wsPort);

  // Check if voice WS mode is available
  const [wsAvailable, setWsAvailable] = useState(false);
  useEffect(() => {
    fetch("/api/mc/voice-status")
      .then(r => r.json())
      .then(data => {
        setWsAvailable(data.ws_active);
        if (data.ws_port) setWsPort(data.ws_port);
      })
      .catch(() => setWsAvailable(false));
    const id = setInterval(() => {
      fetch("/api/mc/voice-status")
        .then(r => r.json())
        .then(data => setWsAvailable(data.ws_active))
        .catch(() => setWsAvailable(false));
    }, 10000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    onVoiceActive?.(isListening);
  }, [isListening, onVoiceActive]);

  if (!wsAvailable) return null;

  const stateColor = pipelineState === "LISTENING" ? "text-green-400" :
                     pipelineState === "PROCESSING" ? "text-yellow-400" : "text-slate-500";

  return (
    <div className="border-b border-surface-3/30">
      {/* Compact bar */}
      <div className="flex items-center gap-2 px-3 py-1.5">
        <button
          onClick={() => isListening ? stopMic() : startMic()}
          className={`p-1.5 rounded-lg transition-colors ${
            isListening
              ? "bg-red-600/20 text-red-400 hover:bg-red-600/30"
              : "bg-surface-2 text-slate-400 hover:text-slate-200 hover:bg-surface-3"
          }`}
          title={isListening ? "Stop mic" : "Start mic"}
        >
          {isListening ? <MicOff className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
        </button>

        <span className={`text-[10px] font-mono ${stateColor}`}>
          {pipelineState}
        </span>

        {isConnected && (
          <Radio className="w-3 h-3 text-green-500" />
        )}

        {isSpeaking && (
          <Volume2 className="w-3.5 h-3.5 text-blue-400 animate-pulse" />
        )}

        {transcripts.length > 0 && (
          <span className="text-xs text-slate-400 truncate flex-1 ml-2">
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
