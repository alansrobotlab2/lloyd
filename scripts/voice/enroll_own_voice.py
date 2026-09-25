#!/usr/bin/env python
"""Enrol Lloyd's own cloned voice as a speaker profile, so the worker can tell
his echo from a person.

With the browser's half-duplex mute off (2026-09-24), a speaker-to-mic path
that the echo canceller does not fully cancel brings Lloyd's own words back as
an "utterance". The worker drops a segment whose voiceprint matches the
`livekit.voiceprint.own_voice_profile` profile while he is speaking or within
2 s of stopping (livekit_worker.py, "Lloyd's own voice").

The profile is the mean of several renders from the live TTS server in the
production voice: measured with CAM++, a 5-render mean scores held-out
windows of the clone at a median 0.75 against 0.54 for a single render, while
the highest room utterance scored 0.25 (architecture/voice.md, "Who is
speaking"). Re-run after a voice change or a voiceprint backend switch —
profiles are per backend.

    python scripts/voice/enroll_own_voice.py            # into the live profiles dir
    python scripts/voice/enroll_own_voice.py --dry-run  # render + score, save nothing
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import urllib.request
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "agent-services"))

# Ordinary replies of mixed length and prosody: a statement, a question, a
# number, a list, the phrases a tool turn says.
SENTENCES = [
    "The build finished with two warnings, both in the test suite.",
    "Your next meeting is at three o'clock with the design team.",
    "Do you want me to restart the backend now, or wait until the round lands?",
    "It's sixty-eight degrees and sunny, with a light breeze from the west.",
    "I found three open items on the backlog, and the oldest is from last Tuesday.",
    "One moment. Checking supervisor status now.",
    "The nightly reflection chain is stalled because the handoff file is missing.",
    "Sure. I've added milk, eggs and coffee to the shopping list.",
]


def render(text: str, cfg: dict) -> tuple[np.ndarray, int]:
    tts = cfg["livekit"]["tts"]
    url = (tts.get("api_url") or "http://127.0.0.1:8090").rstrip("/") + "/v1/audio/speech"
    req = urllib.request.Request(url, method="POST", headers={"Content-Type": "application/json"},
                                 data=json.dumps({"model": tts.get("model", "qwen3-tts"),
                                                  "input": text, "voice": tts["voice"],
                                                  "response_format": "wav"}).encode())
    with urllib.request.urlopen(req, timeout=300) as r:
        data = r.read()
    with wave.open(io.BytesIO(data)) as w:
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        if w.getnchannels() > 1:
            pcm = pcm.reshape(-1, w.getnchannels()).mean(axis=1).astype(np.int16)
    return pcm, sr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--profiles-dir", default=None)
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    vp = cfg["livekit"]["voiceprint"]
    name = vp.get("own_voice_profile", "lloyd-voice")
    data_root = os.environ.get("LLOYD_DATA") or str(Path.home() / "lloyd-data")
    profiles_dir = args.profiles_dir or str(vp["profiles_dir"]).replace("${LLOYD_DATA}", data_root)

    from speaker_id import SpeakerIdentifier
    sid = SpeakerIdentifier(
        profiles_dir=profiles_dir, threshold=float(vp.get("profile_threshold", 0.4)),
        backend=vp.get("backend", "campplus"),
        model_path=(str(ROOT / vp["model_path"]) if vp.get("model_path") else None),
        num_threads=int(vp.get("num_threads", 2)))

    print(f"rendering {len(SENTENCES)} sentences in {cfg['livekit']['tts']['voice']}")
    clips = [render(t, cfg) for t in SENTENCES]
    enrol, held = clips[:5], clips[5:]
    embs = sid.embed_many(enrol)
    ref = embs.mean(axis=0)
    ref /= np.linalg.norm(ref) or 1.0
    scores = [float(np.dot(sid.extract_embedding(a, sr), ref)) for a, sr in held]
    print(f"held-out renders against the profile: {', '.join(f'{s:.2f}' for s in scores)}"
          f"  (threshold {sid.threshold:.2f})")
    if min(scores) < sid.threshold:
        print("WARNING: a held-out render scores under the profile threshold — the "
              "profile would miss some of Lloyd's own speech")
    if args.dry_run:
        print("dry run: nothing saved")
        return 0
    path = sid.enroll_reference(name, clips)
    print(f"enrolled {name!r} from {len(clips)} renders -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
