#!/usr/bin/env python
"""Register a cloned-voice profile on the vLLM-Omni TTS server, and prove it speaks.

vLLM-Omni keeps uploaded voices as `.safetensors` under its
`SPEAKER_SAMPLES_DIR` (production: `~/lloyd-data/tts/speakers`, set by
`agent-services/bin/start-qwen3-tts-omni.sh`). That directory is runtime
data: lose it and the server boots happily and answers 4xx for the voice
until this is run again. The asset is the profile directory the vendored
server also reads (`ref.wav` + `meta.json` with `ref_text`, ICL cloning),
so one source of truth feeds both engines.

The name defaults to `clone:<profile>` — the exact string the worker and the
guardian send as `voice` (`livekit.tts.voice`), so switching engines changed
`api_url` and nothing else.

    .venvs/lloyd/bin/python scripts/voice/register_tts_voice.py            # dave_cullen on :8093
    .venvs/lloyd/bin/python scripts/voice/register_tts_voice.py --profile other --url http://127.0.0.1:8093

Standard library only, so it runs from any venv.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROFILES = ROOT / "agent-services" / "services" / "tts" / "qwen3-tts" / "voice_library" / "profiles"


def multipart(fields: dict[str, str], files: dict[str, tuple[str, bytes, str]]) -> tuple[bytes, str]:
    boundary = "----lloyd" + uuid.uuid4().hex
    out = bytearray()
    for k, v in fields.items():
        out += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
    for k, (fname, data, ctype) in files.items():
        out += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"; "
                f"filename=\"{fname}\"\r\nContent-Type: {ctype}\r\n\r\n").encode()
        out += data + b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={boundary}"


def post(url: str, data: bytes, ctype: str, timeout: float) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=data, headers={"Content-Type": ctype}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def wait_healthy(url: str, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8093")
    ap.add_argument("--profile", default="dave_cullen")
    ap.add_argument("--profiles-dir", default=str(PROFILES))
    ap.add_argument("--name", default=None, help="voice name on the server (default clone:<profile>)")
    ap.add_argument("--consent", default=None,
                    help="the server's required consent id; default names the profile's provenance file")
    ap.add_argument("--wait", type=float, default=0.0, help="seconds to wait for /health first")
    a = ap.parse_args()

    url = a.url.rstrip("/")
    prof = Path(a.profiles_dir).expanduser() / a.profile
    meta = json.loads((prof / "meta.json").read_text())
    ref = prof / meta.get("ref_audio_filename", "ref.wav")
    ref_text = meta.get("ref_text") or (prof / "ref.lab").read_text().strip()
    name = a.name or f"clone:{a.profile}"
    consent = a.consent or f"{a.profile}: see {prof / 'provenance.json'}"
    if a.wait and not wait_healthy(url, a.wait):
        print(f"{url}/health did not answer within {a.wait:.0f}s", file=sys.stderr)
        return 2

    body, ctype = multipart(
        {"consent": consent, "name": name, "ref_text": ref_text},
        {"audio_sample": (ref.name, ref.read_bytes(), "audio/wav")},
    )
    status, resp = post(f"{url}/v1/audio/voices", body, ctype, 120)
    if status >= 300:
        print(f"upload failed: HTTP {status} {resp[:300]!r}", file=sys.stderr)
        return 1
    print(f"registered {name!r} on {url} ({len(ref.read_bytes())} bytes of reference, ICL)")

    # The round trip is the proof: a bad name on the old server returned 200
    # with no audio, so count bytes, never the status alone.
    req = json.dumps({"model": "qwen3-tts", "input": "The voice is registered and speaking.",
                      "voice": name, "response_format": "pcm", "stream": True,
                      "stream_format": "audio", "speed": 1.0}).encode()
    t0 = time.monotonic()
    status, pcm = post(f"{url}/v1/audio/speech", req, "application/json", 120)
    dt = time.monotonic() - t0
    if status >= 300 or len(pcm) < 24000:
        print(f"round trip failed: HTTP {status}, {len(pcm)} bytes", file=sys.stderr)
        return 1
    print(f"round trip: HTTP {status}, {len(pcm) / 2 / 24000:.2f}s of audio in {dt:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
