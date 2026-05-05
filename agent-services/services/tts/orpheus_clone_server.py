#!/usr/bin/env python3
"""Orpheus TTS Clone server for Lloyd voice assistant.

Same HTTP API as other TTS servers: POST /v1/tts streams raw PCM int16 mono at 24kHz.
Uses unsloth/orpheus-3b-0.1-ft (3B Llama-based model) via vLLM + SNAC decoder.

Zero-shot voice cloning: encodes multiple reference audio clips with SNAC, stacks them
as conversation turns so the model mimics the reference speaker.

Prompt format (as token IDs, multi-turn conversation):
  Turns 1..N (references): [START_OF_HUMAN] ref_transcript [END_OF_TEXT, END_OF_HUMAN, START_OF_AI] ref_audio_tokens [END_OF_SPEECH, END_OF_AI]
  Final turn (target):     [START_OF_HUMAN] target_text [END_OF_TEXT, END_OF_HUMAN, START_OF_AI]
  Model generates audio tokens for target text in the reference voice.

Usage: bash bin/start-orpheus-clone.sh

Architecture note:
  vLLM 0.16.0 AsyncLLMEngine requires all .generate() calls to share ONE event loop --
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
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="Orpheus TTS Clone")

# Globals set at startup
_engine = None           # vLLM AsyncLLMEngine
_snac_model = None       # SNAC model for encoding reference audio
_tokenizer = None        # Orpheus tokenizer
_ref_pairs: list[tuple[list[int], str]] = []  # [(audio_tokens, transcript), ...]
_ref_dir = ""            # Directory containing reference audio files
_model_lock = threading.Lock()
_sample_rate = 24000
_model_name = "orpheus-3b-0.1-ft"

# Persistent event loop for vLLM engine calls
_vllm_loop: asyncio.AbstractEventLoop | None = None

# Special token IDs
START_OF_HUMAN = 128259
END_OF_HUMAN = 128260
END_OF_TEXT = 128009
START_OF_AI = 128261
END_OF_AI = 128262
START_OF_SPEECH = 128257
END_OF_SPEECH = 128258

# SNAC codec offsets for interleaving 3 codebook levels into 7-token groups
AUDIO_TOKENS_START = 128266


def _start_vllm_loop():
    global _vllm_loop
    _vllm_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_vllm_loop)
    _vllm_loop.run_forever()


def encode_reference_audio(audio_path: str) -> list[int]:
    """Encode a reference audio file into interleaved SNAC tokens."""
    import librosa

    print(f"  Encoding reference audio: {audio_path}", flush=True)
    t0 = time.time()

    audio, sr = librosa.load(audio_path, sr=24000, mono=True)
    audio_tensor = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0).to("cuda")

    with torch.inference_mode():
        codes = _snac_model.encode(audio_tensor)

    # codes is a list of 3 tensors: [level0, level1, level2]
    # level0: N frames, level1: 2N frames, level2: 4N frames
    # Interleave into 7-code groups with offsets
    all_codes = []
    n_frames = codes[0].shape[1]
    for i in range(n_frames):
        all_codes.append(codes[0][0, i].item() + AUDIO_TOKENS_START)
        all_codes.append(codes[1][0, 2 * i].item() + AUDIO_TOKENS_START + 4096)
        all_codes.append(codes[2][0, 4 * i].item() + AUDIO_TOKENS_START + 8192)
        all_codes.append(codes[2][0, 4 * i + 1].item() + AUDIO_TOKENS_START + 12288)
        all_codes.append(codes[1][0, 2 * i + 1].item() + AUDIO_TOKENS_START + 16384)
        all_codes.append(codes[2][0, 4 * i + 2].item() + AUDIO_TOKENS_START + 20480)
        all_codes.append(codes[2][0, 4 * i + 3].item() + AUDIO_TOKENS_START + 24576)

    elapsed = time.time() - t0
    duration = len(audio) / 24000
    print(
        f"  Reference encoded: {duration:.1f}s audio -> {len(all_codes)} tokens in {elapsed:.2f}s",
        flush=True,
    )
    return all_codes


def build_clone_prompt(text: str, ref_pairs: list[tuple[list[int], str]]):
    """Build multi-turn prompt for voice cloning.

    Reference turns: each (transcript, audio_tokens) pair teaches the voice.
    Final turn: target text — model generates audio in the reference voice.
    """
    from vllm import TokensPrompt

    prompt_ids = []

    # Stack all reference turns
    for audio_tokens, transcript in ref_pairs:
        ref_text_tokens = _tokenizer.encode(transcript, add_special_tokens=False)
        prompt_ids += (
            [START_OF_HUMAN] + ref_text_tokens
            + [END_OF_TEXT, END_OF_HUMAN, START_OF_AI]
            + audio_tokens
            + [END_OF_SPEECH, END_OF_AI]
        )

    # Final turn: target text
    target_text_tokens = _tokenizer.encode(text, add_special_tokens=False)
    prompt_ids += (
        [START_OF_HUMAN] + target_text_tokens
        + [END_OF_TEXT, END_OF_HUMAN, START_OF_AI]
    )

    print(f"  Prompt: {len(ref_pairs)} ref turns, {len(prompt_ids)} total tokens", flush=True)
    return TokensPrompt(prompt_token_ids=prompt_ids)


def _generate_tokens(text: str, temperature: float, top_p: float,
                     max_tokens: int, repetition_penalty: float):
    """Generate audio tokens via the persistent vLLM event loop.

    Yields token strings progressively (same interface as generate_tokens_sync).
    Uses run_coroutine_threadsafe so all calls share _vllm_loop -- engine stays alive.
    """
    from vllm import SamplingParams

    token_queue: queue.Queue = queue.Queue()

    tokens_prompt = build_clone_prompt(text, _ref_pairs)
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
            async for result in _engine.generate(
                prompt=tokens_prompt,
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
    speed: float = 1.0


class SetReferenceRequest(BaseModel):
    path: str


@app.get("/health")
def health():
    total_tokens = sum(len(tokens) for tokens, _ in _ref_pairs)
    return {
        "status": "ok",
        "model": _model_name,
        "sample_rate": _sample_rate,
        "reference_dir": _ref_dir,
        "ref_pairs": len(_ref_pairs),
        "ref_total_tokens": total_tokens,
    }


@app.post("/v1/tts")
def tts(req: TTSRequest):
    text = req.text.strip()
    if not text:
        return JSONResponse({"error": "empty text"}, status_code=400)

    if not _ref_pairs:
        return JSONResponse({"error": "no reference audio loaded"}, status_code=400)

    def generate():
        from orpheus_tts.decoder import tokens_decoder_sync

        t0 = time.time()
        total_samples = 0
        first_chunk = True

        with _model_lock:
            token_gen = _generate_tokens(
                text=text,
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
            f"  TTS [clone]: {len(text)}ch -> {duration:.1f}s in {elapsed:.2f}s (RTF {rtf:.2f}x)",
            flush=True,
        )

    return StreamingResponse(
        generate(),
        media_type="application/octet-stream",
        headers={"X-Sample-Rate": str(_sample_rate)},
    )


def _load_transcript(audio_path: str) -> str:
    """Load the transcript from the .lab file alongside the audio file."""
    import os
    lab_path = os.path.splitext(audio_path)[0] + ".lab"
    if os.path.isfile(lab_path):
        text = open(lab_path).read().strip()
        print(f"  Loaded transcript ({len(text)} chars) from {lab_path}", flush=True)
        return text
    print(f"  WARNING: no .lab transcript found at {lab_path}", flush=True)
    return ""


@app.post("/v1/set_reference")
def set_reference(req: SetReferenceRequest):
    """Load reference pairs from a directory (path to dir) or single file."""
    global _ref_pairs, _ref_dir

    import os
    path = req.path.strip()
    if not path:
        return JSONResponse({"error": "empty path"}, status_code=400)

    path = os.path.expanduser(path)

    try:
        with _model_lock:
            if os.path.isdir(path):
                _ref_pairs = _load_reference_dir(path)
                _ref_dir = path
            elif os.path.isfile(path):
                tokens = encode_reference_audio(path)
                transcript = _load_transcript(path)
                _ref_pairs = [(tokens, transcript)]
                _ref_dir = os.path.dirname(path)
            else:
                return JSONResponse({"error": f"not found: {path!r}"}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    total_tokens = sum(len(t) for t, _ in _ref_pairs)
    return {
        "status": "ok",
        "reference_dir": _ref_dir,
        "ref_pairs": len(_ref_pairs),
        "ref_total_tokens": total_tokens,
    }


def _load_reference_dir(ref_dir: str) -> list[tuple[list[int], str]]:
    """Load all .wav files in ref_dir that have matching .lab transcripts."""
    import glob
    import os

    wav_files = sorted(glob.glob(os.path.join(ref_dir, "*.wav")))
    pairs = []
    for wav_path in wav_files:
        lab_path = os.path.splitext(wav_path)[0] + ".lab"
        if not os.path.isfile(lab_path):
            print(f"  Skipping {os.path.basename(wav_path)} — no .lab file", flush=True)
            continue
        tokens = encode_reference_audio(wav_path)
        transcript = _load_transcript(wav_path)
        pairs.append((tokens, transcript))

    print(f"  Loaded {len(pairs)} reference pairs from {ref_dir}", flush=True)
    total_tokens = sum(len(t) for t, _ in pairs)
    print(f"  Total reference audio tokens: {total_tokens}", flush=True)
    return pairs


def warmup():
    from orpheus_tts.decoder import tokens_decoder_sync

    print("  Warming up...", flush=True)
    t0 = time.time()
    token_gen = _generate_tokens(
        text="Hello, how are you doing today?",
        temperature=0.6,
        top_p=0.8,
        max_tokens=200,
        repetition_penalty=1.1,
    )
    for _ in tokens_decoder_sync(token_gen):
        pass
    print(f"  Warmup done in {time.time() - t0:.1f}s", flush=True)


def main():
    global _engine, _snac_model, _tokenizer, _ref_pairs, _ref_dir, _model_name

    parser = argparse.ArgumentParser(description="Orpheus TTS Clone Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8097)
    parser.add_argument(
        "--model",
        default="unsloth/orpheus-3b-0.1-ft",
        help="HuggingFace repo id for Orpheus model",
    )
    parser.add_argument(
        "--reference-dir",
        default="~/lloyd/agent-services/references/cullen/",
        help="Directory containing reference .wav + .lab pairs for voice cloning",
    )
    parser.add_argument("--no-warmup", action="store_true", help="Skip warmup synthesis")
    args = parser.parse_args()

    import os
    args.reference_dir = os.path.expanduser(args.reference_dir)
    _model_name = args.model.split("/")[-1]

    # 1. Start the persistent vLLM event loop
    loop_thread = threading.Thread(target=_start_vllm_loop, daemon=True)
    loop_thread.start()
    while _vllm_loop is None or not _vllm_loop.is_running():
        time.sleep(0.01)

    print("=== Orpheus TTS Clone Server ===")

    # 2. Load SNAC model for encoding reference audio
    from snac import SNAC
    print("  Loading SNAC encoder...", flush=True)
    t0 = time.time()
    _snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().to("cuda")
    print(f"  SNAC encoder loaded in {time.time() - t0:.1f}s", flush=True)

    # 3. Load tokenizer
    from transformers import AutoTokenizer
    _tokenizer = AutoTokenizer.from_pretrained(args.model)

    # 4. Create vLLM AsyncLLMEngine
    from vllm import AsyncLLMEngine, AsyncEngineArgs
    print(f"  Loading {args.model!r} (bfloat16)...", flush=True)
    t0 = time.time()
    engine_args = AsyncEngineArgs(
        model=args.model,
        dtype="bfloat16",
    )
    _engine = AsyncLLMEngine.from_engine_args(engine_args)
    print(f"  Model loaded in {time.time() - t0:.1f}s", flush=True)

    # 5. Load reference audio pairs from directory
    if os.path.isdir(args.reference_dir):
        _ref_pairs = _load_reference_dir(args.reference_dir)
        _ref_dir = args.reference_dir
    else:
        print(f"  WARNING: reference dir not found: {args.reference_dir}", flush=True)
        print("  Use POST /v1/set_reference to load references before synthesis.", flush=True)

    print(f"  Sample rate: {_sample_rate}Hz")

    # 6. Optional warmup
    if not args.no_warmup and _ref_pairs:
        warmup()

    # 7. Start server
    print(f"  Listening on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
