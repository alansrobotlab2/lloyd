#!/usr/bin/env python3
"""Orpheus TTS server for Lloyd voice assistant.

Same HTTP API as other TTS servers: POST /v1/tts streams raw PCM int16 mono at 24kHz.
Uses unsloth/orpheus-3b-0.1-ft (3B Llama-based model) via vLLM + SNAC decoder.

Voices: tara, leah, jess, leo, dan, mia, zac, zoe

Emotion/expression tags are embedded directly in text:

  Non-verbal sounds (production-ready):
    <laugh>   <chuckle>  <sigh>   <cough>
    <sniffle> <groan>    <yawn>   <gasp>

  Emotional tone tags (experimental — results vary by voice):
    <happy>   <sad>      <angry>  <excited>
    <fearful> <surprised> <disgusted> <neutral>

  Example: "That's hilarious! <laugh> I can't believe you said that."
  Example: "I'm so nervous about this. <sigh> Here we go."

Usage: bash scripts/start-orpheus-tts.sh

Direct usage:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \\
    ~/.venvs/orpheus/bin/python tts/orpheus/orpheus_server.py \\
        --voice dan

Architecture note:
  vLLM 0.16.0 AsyncLLMEngine requires all .generate() calls to share ONE event loop —
  if asyncio.run() is called per-request (as orpheus_tts does), the engine marks itself
  dead after the first call. Fix: dedicated background thread with a persistent event
  loop; all engine.generate() calls dispatched via run_coroutine_threadsafe().
"""

import argparse
import asyncio
import queue
import threading
import time
import uuid

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="Orpheus TTS")

# Globals set at startup
_model = None
_model_lock = threading.Lock()
_sample_rate = 24000
_model_name = "orpheus-3b-0.1-ft"
_default_voice = "dan"

# Persistent event loop for vLLM engine calls — avoids EngineDeadError from
# closing the loop between asyncio.run() calls (vLLM 0.16.0 v1 API requirement)
_vllm_loop: asyncio.AbstractEventLoop | None = None

AVAILABLE_VOICES = {"tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"}

# Non-verbal sound tags — production-ready, consistently supported
EXPRESSION_TAGS = ["laugh", "chuckle", "sigh", "cough", "sniffle", "groan", "yawn", "gasp"]

# Emotional tone tags — experimental, results vary by voice
EMOTION_TAGS = ["happy", "sad", "angry", "excited", "fearful", "surprised", "disgusted", "neutral"]


def _start_vllm_loop():
    global _vllm_loop
    _vllm_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_vllm_loop)
    _vllm_loop.run_forever()


def _generate_tokens(prompt: str, voice: str, temperature: float, top_p: float,
                     max_tokens: int, repetition_penalty: float):
    """Generate audio tokens via the persistent vLLM event loop.

    Yields token strings progressively (same interface as generate_tokens_sync).
    Uses run_coroutine_threadsafe so all calls share _vllm_loop — engine stays alive.
    """
    from vllm import SamplingParams

    token_queue: queue.Queue = queue.Queue()

    prompt_str = _model._format_prompt(prompt, voice)
    params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop_token_ids=[49158],
        repetition_penalty=repetition_penalty,
    )
    request_id = str(uuid.uuid4())

    async def producer():
        try:
            async for result in _model.engine.generate(
                prompt=prompt_str,
                sampling_params=params,
                request_id=request_id,
            ):
                token_queue.put(result.outputs[0].text)
        except Exception as e:
            token_queue.put(e)
        finally:
            token_queue.put(None)

    future = asyncio.run_coroutine_threadsafe(producer(), _vllm_loop)

    while True:
        item = token_queue.get()
        if item is None:
            break
        if isinstance(item, Exception):
            raise item
        yield item

    future.result()  # propagate any exception from the coroutine


class TTSRequest(BaseModel):
    text: str
    voice: str = ""     # empty = use server default
    speed: float = 1.0


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": _model_name,
        "sample_rate": _sample_rate,
        "default_voice": _default_voice,
        "voices": sorted(AVAILABLE_VOICES),
        "expression_tags": EXPRESSION_TAGS,
        "emotion_tags": EMOTION_TAGS,
    }


@app.post("/v1/tts")
def tts(req: TTSRequest):
    text = req.text.strip()
    if not text:
        return JSONResponse({"error": "empty text"}, status_code=400)

    voice = req.voice.strip() or _default_voice
    if voice not in AVAILABLE_VOICES:
        return JSONResponse(
            {"error": f"unknown voice {voice!r}, available: {sorted(AVAILABLE_VOICES)}"},
            status_code=400,
        )

    def generate():
        from orpheus_tts.decoder import tokens_decoder_sync

        t0 = time.time()
        total_samples = 0
        first_chunk = True

        with _model_lock:
            token_gen = _generate_tokens(
                prompt=text,
                voice=voice,
                temperature=0.6,
                top_p=0.8,
                max_tokens=2000,
                repetition_penalty=1.1,
            )

            for audio_bytes in tokens_decoder_sync(token_gen):
                if first_chunk:
                    print(f"  TTS first chunk: {time.time() - t0:.3f}s", flush=True)
                    first_chunk = False

                if not audio_bytes:
                    continue

                if abs(req.speed - 1.0) > 0.01:
                    from scipy.signal import resample as scipy_resample
                    samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                    target_len = int(len(samples) / req.speed)
                    if target_len > 0:
                        resampled = scipy_resample(samples, target_len)
                        audio_bytes = resampled.clip(-32768, 32767).astype(np.int16).tobytes()

                total_samples += len(audio_bytes) // 2  # int16 = 2 bytes
                yield audio_bytes

        elapsed = time.time() - t0
        duration = total_samples / _sample_rate
        rtf = elapsed / duration if duration > 0 else float("inf")
        print(
            f"  TTS [{voice}]: {len(text)}ch -> {duration:.1f}s in {elapsed:.2f}s (RTF {rtf:.2f}x)",
            flush=True,
        )

    return StreamingResponse(
        generate(),
        media_type="application/octet-stream",
        headers={"X-Sample-Rate": str(_sample_rate)},
    )


def warmup(voice: str):
    from orpheus_tts.decoder import tokens_decoder_sync

    print(f"  Warming up (voice={voice!r})...", flush=True)
    t0 = time.time()
    token_gen = _generate_tokens(
        prompt="Hello, how are you doing today?",
        voice=voice,
        temperature=0.6,
        top_p=0.8,
        max_tokens=200,
        repetition_penalty=1.1,
    )
    for _ in tokens_decoder_sync(token_gen):
        pass
    print(f"  Warmup done in {time.time() - t0:.1f}s", flush=True)


def main():
    global _model, _model_name, _default_voice

    parser = argparse.ArgumentParser(description="Orpheus TTS Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument(
        "--model",
        default="unsloth/orpheus-3b-0.1-ft",
        help="HuggingFace repo id for Orpheus model (default: unsloth mirror, "
             "canopylabs/orpheus-3b-0.1-ft requires HF terms acceptance)",
    )
    parser.add_argument(
        "--voice",
        default="dan",
        choices=sorted(AVAILABLE_VOICES),
        help="Default voice (default: dan)",
    )
    parser.add_argument("--no-warmup", action="store_true", help="Skip warmup synthesis")
    args = parser.parse_args()

    _default_voice = args.voice
    _model_name = args.model.split("/")[-1]

    # Start the persistent vLLM event loop before loading the model
    loop_thread = threading.Thread(target=_start_vllm_loop, daemon=True)
    loop_thread.start()
    # Wait until the loop is running
    while _vllm_loop is None or not _vllm_loop.is_running():
        time.sleep(0.01)

    from orpheus_tts import OrpheusModel

    print("=== Orpheus TTS Server ===")
    print(f"  Loading {args.model!r} (bfloat16)...")
    t0 = time.time()
    _model = OrpheusModel(
        model_name=args.model,
        dtype="bfloat16",
    )
    print(f"  Model loaded in {time.time() - t0:.1f}s")
    print(f"  Default voice: {_default_voice}  |  Sample rate: {_sample_rate}Hz")
    print(f"  Available voices: {', '.join(sorted(AVAILABLE_VOICES))}")

    if not args.no_warmup:
        warmup(args.voice)

    print(f"  Listening on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
