#!/usr/bin/env python3
"""CosyVoice3 TTS server for Lloyd voice assistant.

Thin FastAPI wrapper around CosyVoice3's inference_zero_shot().
Streams raw PCM int16 mono audio at the model's native sample rate.

Usage (via start script):
    bash scripts/start-cosyvoice-tts.sh

Direct usage:
    PYTHONPATH=/path/to/CosyVoice:/path/to/CosyVoice/third_party/Matcha-TTS \
    python cosyvoice_server.py --model-dir /path/to/Fun-CosyVoice3-0.5B \
        --prompt-wav references/ronan/ronan_001.wav \
        --prompt-text "transcript of the reference audio"
"""

import argparse
import threading
import time

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="CosyVoice3 TTS")

# Globals set at startup
_model = None
_model_lock = threading.Lock()
_sample_rate = None
_model_name = None
_spk_id = "lloyd"  # pre-registered speaker ID
_initial_token_hop_len = 13  # reset before each request (13 → ~0.5s first chunk)

PROMPT_PREFIX = "You are a helpful assistant.<|endofprompt|>"


class TTSRequest(BaseModel):
    text: str
    speed: float = 1.0


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": _model_name,
        "sample_rate": _sample_rate,
    }


@app.post("/v1/tts")
def tts(req: TTSRequest):
    text = req.text.strip()
    if not text:
        return JSONResponse({"error": "empty text"}, status_code=400)

    def generate():
        t0 = time.time()
        total_samples = 0
        with _model_lock:
            # Reset token_hop_len — CosyVoice3 ratchets it up during streaming
            _model.model.token_hop_len = _initial_token_hop_len
            for chunk in _model.inference_zero_shot(
                text,
                "",  # prompt_text unused when zero_shot_spk_id is set
                None,  # prompt_wav unused when zero_shot_spk_id is set
                zero_shot_spk_id=_spk_id,
                stream=True,
                speed=req.speed,
            ):
                audio = chunk["tts_speech"].numpy().flatten()
                pcm = (audio * 32768.0).clip(-32768, 32767).astype(np.int16)
                total_samples += len(pcm)
                yield pcm.tobytes()
        elapsed = time.time() - t0
        duration = total_samples / _sample_rate if _sample_rate else 0
        rtf = elapsed / duration if duration > 0 else float("inf")
        print(f"  TTS: {len(text)}ch -> {duration:.1f}s audio in {elapsed:.2f}s (RTF {rtf:.2f}x)")

    return StreamingResponse(
        generate(),
        media_type="application/octet-stream",
        headers={"X-Sample-Rate": str(_sample_rate)},
    )


def warmup(text: str = "Hello, how are you today?"):
    """Run a streaming synthesis to warm up all code paths."""
    print("  Warming up model (streaming)...")
    t0 = time.time()
    _model.model.token_hop_len = _initial_token_hop_len
    for chunk in _model.inference_zero_shot(
        text, "", None, zero_shot_spk_id=_spk_id, stream=True, speed=1.0
    ):
        pass
    print(f"  Warmup done in {time.time() - t0:.1f}s")


def main():
    global _model, _sample_rate, _model_name, _initial_token_hop_len

    parser = argparse.ArgumentParser(description="CosyVoice3 TTS Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--model-dir", required=True,
                        help="Path to CosyVoice3 model directory")
    parser.add_argument("--prompt-wav", required=True,
                        help="Path to reference voice WAV file")
    parser.add_argument("--prompt-text", required=True,
                        help="Transcript of the reference audio")
    parser.add_argument("--no-warmup", action="store_true",
                        help="Skip warmup synthesis")
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16 for faster inference")
    parser.add_argument("--vllm", action="store_true",
                        help="Use vLLM for LLM acceleration (~4x faster token generation)")
    parser.add_argument("--trt", action="store_true",
                        help="Use TensorRT for flow decoder acceleration (auto-builds on first run)")
    parser.add_argument("--flow-steps", type=int, default=6,
                        help="Number of diffusion steps for flow decoder (default: 6, upstream: 10)")
    parser.add_argument("--cfg-rate", type=float, default=0.0,
                        help="Classifier-free guidance rate (0.0=disabled, 0.7=upstream default)")
    parser.add_argument("--token-hop", type=int, default=13,
                        help="Initial token hop length (default: 13, upstream: 25)")
    args = parser.parse_args()

    _initial_token_hop_len = args.token_hop

    from cosyvoice.cli.cosyvoice import AutoModel

    # Reduce polling interval from 100ms to 5ms in the streaming loop.
    # CosyVoice3's tts() method uses time.sleep(0.1) to poll for ready tokens;
    # monkey-patching just that module's time.sleep avoids editing upstream code.
    import cosyvoice.cli.model as _cv_model_mod
    _orig_sleep = _cv_model_mod.time.sleep
    _cv_model_mod.time.sleep = lambda s: _orig_sleep(0.005 if s == 0.1 else s)

    # Disable torch.cuda.empty_cache() called after every request in model.py.
    # For a persistent single-user server, keeping allocations warm is better.
    import torch
    _cv_model_mod.torch.cuda.empty_cache = lambda: None

    # Force ONNX speech tokenizer to CPU — onnxruntime-gpu 1.23.2 lacks Blackwell
    # (SM 12.0) kernels. This only affects the one-time speaker pre-registration,
    # not the inference hot path (which is pure PyTorch + TRT).
    import onnxruntime
    _orig_ort_session = onnxruntime.InferenceSession

    class _CpuFallbackSession(_orig_ort_session):
        def __init__(self, *a, providers=None, **kw):
            if providers and "CUDAExecutionProvider" in providers:
                providers = ["CPUExecutionProvider"]
            super().__init__(*a, providers=providers, **kw)

    onnxruntime.InferenceSession = _CpuFallbackSession

    print("=== CosyVoice3 TTS Server ===")
    print(f"  Loading model from {args.model_dir}...")
    _model = AutoModel(model_dir=args.model_dir, fp16=args.fp16,
                       load_vllm=args.vllm)
    _sample_rate = _model.sample_rate
    _model_name = "Fun-CosyVoice3-0.5B-2512_RL"

    # Restore original ORT session class after model loading
    onnxruntime.InferenceSession = _orig_ort_session

    # Apply diffusion step count override (upstream hardcodes n_timesteps=10)
    if args.flow_steps != 10:
        _orig_flow_inference = _model.model.flow.inference
        _flow_steps = args.flow_steps
        import functools

        @functools.wraps(_orig_flow_inference)
        def _patched_flow_inference(*a, **kw):
            # The flow.inference() calls decoder(n_timesteps=10) — we intercept
            # the decoder's forward to override n_timesteps
            _orig_decoder_fwd = _model.model.flow.decoder.forward

            @functools.wraps(_orig_decoder_fwd)
            def _patched_decoder_fwd(*da, **dkw):
                dkw['n_timesteps'] = _flow_steps
                return _orig_decoder_fwd(*da, **dkw)

            _model.model.flow.decoder.forward = _patched_decoder_fwd
            try:
                return _orig_flow_inference(*a, **kw)
            finally:
                _model.model.flow.decoder.forward = _orig_decoder_fwd

        _model.model.flow.inference = _patched_flow_inference
        print(f"  Flow diffusion steps: {args.flow_steps} (upstream: 10)")

    # Apply CFG rate override
    if args.cfg_rate != 0.7:
        _model.model.flow.decoder.inference_cfg_rate = args.cfg_rate
        print(f"  CFG rate: {args.cfg_rate} (upstream: 0.7)")

    if args.trt:
        import os
        # ONNX source is always fp32; FP16 is a TRT builder flag applied at plan build time.
        # Plan file name reflects the precision to avoid mixing up cached plans.
        fp16_label = "fp16" if args.fp16 else "fp32"
        plan_path = os.path.join(args.model_dir, f"flow.decoder.estimator.{fp16_label}.mygpu.plan")
        onnx_path = os.path.join(args.model_dir, "flow.decoder.estimator.fp32.onnx")
        print(f"  Loading TensorRT flow decoder ({fp16_label.upper()})...")
        t_trt = time.time()
        _model.model.load_trt(plan_path, onnx_path, trt_concurrent=1, fp16=args.fp16)
        print(f"  TensorRT loaded in {time.time() - t_trt:.1f}s")

    print(f"  Model loaded. Sample rate: {_sample_rate}")
    print(f"  Reference voice: {args.prompt_wav}")
    print(f"  Token hop: {_initial_token_hop_len} (upstream: 25)")

    # Pre-register speaker embedding to avoid re-processing reference audio
    # on every request (saves ~0.5-1s per call for long reference wavs)
    full_prompt_text = f"{PROMPT_PREFIX}{args.prompt_text}"
    print(f"  Pre-registering speaker '{_spk_id}'...")
    t0 = time.time()
    _model.add_zero_shot_spk(full_prompt_text, args.prompt_wav, _spk_id)
    print(f"  Speaker registered in {time.time() - t0:.1f}s")

    if not args.no_warmup:
        warmup()

    print(f"  Listening on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
