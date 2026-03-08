import { useState } from "react";
import { useVoiceContext } from "../contexts/VoiceContext";

export default function VoicePanel() {
  const [expanded, setExpanded] = useState(false);
  const { transcripts, clearTranscripts, wsAvailable } = useVoiceContext();

  if (!wsAvailable) return null;

  return (
    <div className="border-b border-surface-3/30">
      <div className="flex items-center gap-2 px-3 py-1">
        <button
          onClick={() => setExpanded(!expanded)}
          className="text-slate-500 hover:text-slate-300 text-[10px] ml-auto"
        >
          {expanded ? "Hide" : "Show"} transcripts
        </button>
      </div>

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
