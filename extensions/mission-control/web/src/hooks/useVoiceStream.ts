import { useState, useRef, useCallback, useEffect } from "react";

interface Transcript {
  text: string;
  speaker: string;
  is_continuity: boolean;
  timestamp: number;
}

interface UseVoiceStreamReturn {
  isConnected: boolean;
  isListening: boolean;
  isSpeaking: boolean;
  wakewordDetected: boolean;
  pipelineState: string;
  transcripts: Transcript[];
  startMic: () => Promise<void>;
  stopMic: () => void;
  clearTranscripts: () => void;
}

export function useVoiceStream(_wsPort: number = 8095): UseVoiceStreamReturn {
  const [isConnected, setIsConnected] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [pipelineState, setPipelineState] = useState("IDLE");
  const [transcripts, setTranscripts] = useState<Transcript[]>([]);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [wakewordDetected, setWakewordDetected] = useState(false);

  const eventSourceRef = useRef<EventSource | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const workletNodeRef = useRef<ScriptProcessorNode | null>(null);
  const playbackCtxRef = useRef<AudioContext | null>(null);
  const nextPlayTimeRef = useRef(0);
  const ttsSampleRateRef = useRef(24000);

  // Connect SSE
  const connectSse = useCallback(() => {
    if (eventSourceRef.current) return;
    const es = new EventSource("/api/mc/voice-stream", { withCredentials: true });
    eventSourceRef.current = es;

    es.addEventListener("connected", (e) => {
      const data = JSON.parse((e as MessageEvent).data);
      setIsConnected(data.connected);
    });

    es.addEventListener("state", (e) => {
      const data = JSON.parse((e as MessageEvent).data);
      setPipelineState(data.state);
    });

    es.addEventListener("transcript", (e) => {
      const msg = JSON.parse((e as MessageEvent).data);
      setTranscripts(prev => [...prev.slice(-49), {
        text: msg.text,
        speaker: msg.speaker,
        is_continuity: msg.is_continuity,
        timestamp: Date.now(),
      }]);
    });

    es.addEventListener("tts_start", (e) => {
      const msg = JSON.parse((e as MessageEvent).data);
      ttsSampleRateRef.current = msg.sample_rate || 24000;
      if (!playbackCtxRef.current || playbackCtxRef.current.state === "closed") {
        playbackCtxRef.current = new AudioContext({ sampleRate: msg.sample_rate || 24000 });
      }
      nextPlayTimeRef.current = 0;
      setIsSpeaking(true);
    });

    es.addEventListener("tts_end", () => {
      const ctx = playbackCtxRef.current;
      if (ctx) {
        const remaining = Math.max(0, (nextPlayTimeRef.current - ctx.currentTime) * 1000);
        setTimeout(() => setIsSpeaking(false), remaining + 50);
      } else {
        setIsSpeaking(false);
      }
    });

    es.addEventListener("tts_audio", (e) => {
      const ctx = playbackCtxRef.current;
      if (!ctx) return;
      // Decode base64 to Float32Array
      const b64 = (e as MessageEvent).data;
      const binary = atob(b64);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      const float32 = new Float32Array(bytes.buffer);

      const buffer = ctx.createBuffer(1, float32.length, ttsSampleRateRef.current);
      buffer.getChannelData(0).set(float32);
      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(ctx.destination);
      const startTime = Math.max(ctx.currentTime, nextPlayTimeRef.current);
      source.start(startTime);
      nextPlayTimeRef.current = startTime + buffer.duration;
    });

    es.addEventListener("wakeword", () => {
      setWakewordDetected(true);
      setTimeout(() => setWakewordDetected(false), 2000);
    });

    es.onerror = () => {
      setIsConnected(false);
      // EventSource will auto-reconnect
    };

  }, []);

  // Resample Float32 from native rate to 16kHz
  function resampleTo16k(input: Float32Array, inputRate: number): Int16Array {
    const ratio = inputRate / 16000;
    const outputLen = Math.floor(input.length / ratio);
    const output = new Int16Array(outputLen);
    for (let i = 0; i < outputLen; i++) {
      const srcIdx = i * ratio;
      const idx = Math.floor(srcIdx);
      const frac = srcIdx - idx;
      const s0 = input[idx] || 0;
      const s1 = input[idx + 1] || 0;
      const sample = s0 + frac * (s1 - s0);
      output[i] = Math.max(-32768, Math.min(32767, Math.round(sample * 32767)));
    }
    return output;
  }

  const startMic = useCallback(async () => {
    // Connect SSE if not connected
    connectSse();

    // Send start command
    await fetch("/api/mc/voice-cmd", {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ type: "start" }),
    });

    // Get mic
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, sampleRate: { ideal: 48000 } }
    });
    mediaStreamRef.current = stream;

    const audioCtx = new AudioContext();
    audioContextRef.current = audioCtx;
    const source = audioCtx.createMediaStreamSource(stream);
    const nativeRate = audioCtx.sampleRate;

    const processor = audioCtx.createScriptProcessor(4096, 1, 1);
    workletNodeRef.current = processor;

    processor.onaudioprocess = (e) => {
      const inputData = e.inputBuffer.getChannelData(0);
      const resampled = resampleTo16k(inputData, nativeRate);
      // Send audio via POST
      fetch("/api/mc/voice-send", {
        method: "POST",
        credentials: "include",
        body: resampled.buffer.slice(0) as ArrayBuffer,
      }).catch(() => {}); // fire-and-forget
    };

    source.connect(processor);
    processor.connect(audioCtx.destination);

    setIsListening(true);
  }, [connectSse]);

  const stopMic = useCallback(() => {
    // Send stop command
    fetch("/api/mc/voice-cmd", {
      method: "POST",
      credentials: "include",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ type: "stop" }),
    }).catch(() => {});

    if (workletNodeRef.current) {
      workletNodeRef.current.disconnect();
      workletNodeRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(t => t.stop());
      mediaStreamRef.current = null;
    }

    setIsListening(false);
  }, []);

  const clearTranscripts = useCallback(() => setTranscripts([]), []);

  // Connect SSE on mount for TTS playback (independent of mic)
  useEffect(() => {
    connectSse();
  }, [connectSse]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopMic();
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
      if (playbackCtxRef.current) {
        playbackCtxRef.current.close();
        playbackCtxRef.current = null;
      }
    };
  }, [stopMic]);

  return { isConnected, isListening, isSpeaking, wakewordDetected, pipelineState, transcripts, startMic, stopMic, clearTranscripts };
}
