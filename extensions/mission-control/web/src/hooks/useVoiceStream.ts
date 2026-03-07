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
  pipelineState: string;
  transcripts: Transcript[];
  startMic: () => Promise<void>;
  stopMic: () => void;
  clearTranscripts: () => void;
}

export function useVoiceStream(wsPort: number = 8095): UseVoiceStreamReturn {
  const [isConnected, setIsConnected] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [pipelineState, setPipelineState] = useState("IDLE");
  const [transcripts, setTranscripts] = useState<Transcript[]>([]);

  const [isSpeaking, setIsSpeaking] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const workletNodeRef = useRef<ScriptProcessorNode | null>(null);
  const playbackCtxRef = useRef<AudioContext | null>(null);
  const nextPlayTimeRef = useRef(0);
  const ttsSampleRateRef = useRef(24000);

  // Connect WebSocket
  const connectWs = useCallback(() => {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.hostname;
    const ws = new WebSocket(`${protocol}//${host}:${wsPort}`);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => setIsConnected(true);
    ws.onclose = () => {
      setIsConnected(false);
      setIsListening(false);
    };
    ws.onmessage = (ev) => {
      if (ev.data instanceof ArrayBuffer) {
        // TTS audio chunk - Float32 PCM
        const ctx = playbackCtxRef.current;
        if (!ctx) return;
        const float32 = new Float32Array(ev.data);
        const buffer = ctx.createBuffer(1, float32.length, ttsSampleRateRef.current);
        buffer.getChannelData(0).set(float32);
        const source = ctx.createBufferSource();
        source.buffer = buffer;
        source.connect(ctx.destination);
        // Schedule for gapless playback
        const startTime = Math.max(ctx.currentTime, nextPlayTimeRef.current);
        source.start(startTime);
        nextPlayTimeRef.current = startTime + buffer.duration;
        return;
      }
      if (typeof ev.data === "string") {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === "state") {
            setPipelineState(msg.state);
          } else if (msg.type === "transcript") {
            setTranscripts(prev => [...prev.slice(-49), {
              text: msg.text,
              speaker: msg.speaker,
              is_continuity: msg.is_continuity,
              timestamp: Date.now(),
            }]);
          } else if (msg.type === "tts_start") {
            ttsSampleRateRef.current = msg.sample_rate || 24000;
            // Create or reuse AudioContext for playback
            if (!playbackCtxRef.current || playbackCtxRef.current.state === "closed") {
              playbackCtxRef.current = new AudioContext({ sampleRate: msg.sample_rate || 24000 });
            }
            nextPlayTimeRef.current = 0;
            setIsSpeaking(true);
          } else if (msg.type === "tts_end") {
            // Wait for scheduled audio to finish, then clear speaking state
            const ctx = playbackCtxRef.current;
            if (ctx) {
              const remaining = Math.max(0, (nextPlayTimeRef.current - ctx.currentTime) * 1000);
              setTimeout(() => setIsSpeaking(false), remaining + 50);
            } else {
              setIsSpeaking(false);
            }
          }
        } catch {}
      }
    };

    wsRef.current = ws;
    return ws;
  }, [wsPort]);

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
      // Float32 [-1,1] -> Int16
      output[i] = Math.max(-32768, Math.min(32767, Math.round(sample * 32767)));
    }
    return output;
  }

  const startMic = useCallback(async () => {
    // Connect WS if not connected
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      connectWs();
      // Wait for connection
      await new Promise<void>((resolve, reject) => {
        const ws = wsRef.current!;
        const onOpen = () => { ws.removeEventListener("open", onOpen); resolve(); };
        const onError = () => { ws.removeEventListener("error", onError); reject(new Error("WS connect failed")); };
        if (ws.readyState === WebSocket.OPEN) { resolve(); return; }
        ws.addEventListener("open", onOpen);
        ws.addEventListener("error", onError);
      });
    }

    // Get mic
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, sampleRate: { ideal: 48000 } }
    });
    mediaStreamRef.current = stream;

    const audioCtx = new AudioContext();
    audioContextRef.current = audioCtx;
    const source = audioCtx.createMediaStreamSource(stream);
    const nativeRate = audioCtx.sampleRate;

    // Use ScriptProcessorNode (deprecated but simple and widely supported)
    // Buffer size: 4096 samples at native rate
    const processor = audioCtx.createScriptProcessor(4096, 1, 1);
    workletNodeRef.current = processor;

    processor.onaudioprocess = (e) => {
      const ws = wsRef.current;
      if (!ws || ws.readyState !== WebSocket.OPEN) return;

      const inputData = e.inputBuffer.getChannelData(0);
      const resampled = resampleTo16k(inputData, nativeRate);
      ws.send(resampled.buffer);
    };

    source.connect(processor);
    processor.connect(audioCtx.destination); // required for ScriptProcessor to fire

    // Send start command
    wsRef.current!.send(JSON.stringify({ type: "start" }));
    setIsListening(true);
  }, [connectWs]);

  const stopMic = useCallback(() => {
    // Send stop command
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "stop" }));
    }

    // Stop audio processing
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

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopMic();
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      if (playbackCtxRef.current) {
        playbackCtxRef.current.close();
        playbackCtxRef.current = null;
      }
    };
  }, [stopMic]);

  return { isConnected, isListening, isSpeaking, pipelineState, transcripts, startMic, stopMic, clearTranscripts };
}
