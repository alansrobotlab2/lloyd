#!/usr/bin/env python
"""Synthesise the corpus `hotword_eval.py` reads: sentences that contain a
hotword, and sound-alikes that must not come out as one, in six Qwen3-TTS
built-in voices (none of which the ASR model was trained on).

Built-in voices load a different checkpoint than the clone voice mode uses;
the last request switches the server back to `clone:dave_cullen`. Needs the
TTS server on :8090 and ~4 minutes. Existing clips are kept.

    python scripts/voice/synth_hotword_corpus.py
"""
import json, time, urllib.request, wave
from pathlib import Path

import numpy as np

OUT = Path.home() / ".cache/lloyd-voice-eval/hotwords"
OUT.mkdir(parents=True, exist_ok=True)

TARGETS = [
    ("How many items are in the backlog right now?", "backlog"),
    ("Move the top backlog item to done.", "backlog"),
    ("What's the oldest thing on the backlog?", "backlog"),
    ("Tell Lisa I'll be home by six.", "Lisa"),
    ("Did Lisa call this afternoon?", "Lisa"),
    ("Remind me to pick up Emilio after practice.", "Emilio"),
    ("Emilio has a dentist appointment tomorrow.", "Emilio"),
    ("Did Alfie eat his breakfast?", "Alfie"),
    ("Take Alfie for a walk at noon.", "Alfie"),
    ("Is Stompy charged up?", "Stompy"),
    ("Stompy's left leg servo is making noise.", "Stompy"),
    ("Gracie needs to go to the vet on Friday.", "Gracie"),
    ("Has anyone fed Gracie today?", "Gracie"),
    ("What's the latest on the Groot model?", "gr00t"),
    ("Alan's meeting moved to three o'clock.", "Alan"),
    ("Ask Alan if he wants coffee.", "Alan"),
    ("Lloyd, what's the weather like?", "Lloyd"),
    ("How is the autocode loop doing?", "autocode"),
    ("Did autotriage close anything overnight?", "autotriage"),
    ("Open Mission Control on the big screen.", "Mission Control"),
    ("Turn on Inner Voice for this session.", "Inner Voice"),
    ("Is the LiveKit server running?", "LiveKit"),
]
DISTRACTORS = [
    "The back lot is full of old cars.",
    "Please lease the truck for one more year.",
    "The alpha version ships next week.",
    "Emily is coming over for dinner.",
    "He was stomping around upstairs all night.",
    "That was very gracious of her.",
    "The route was closed for repairs.",
    "Floyd came over yesterday.",
    "Auto correct changed my text again.",
    "The mission went exactly according to plan.",
    "I heard a voice inside the house.",
    "The live cat video had a million views.",
]
VOICES = ["Ryan", "Aiden", "Dylan", "Eric", "Serena", "Vivian"]


def synth(text, voice):
    req = urllib.request.Request(
        "http://127.0.0.1:8090/v1/audio/speech",
        data=json.dumps({"model": "qwen3-tts", "input": text, "voice": voice,
                         "response_format": "pcm", "stream": True, "speed": 1.0}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        return np.frombuffer(r.read(), dtype=np.int16)


manifest = []
n = 0
for voice in VOICES:
    items = [("t", i, t, h) for i, (t, h) in enumerate(TARGETS)] + \
            [("d", i, t, None) for i, t in enumerate(DISTRACTORS)]
    for kind, i, text, hw in items:
        path = OUT / f"{kind}_{voice}_{i:02d}.wav"
        if not path.exists():
            a = synth(text, voice)
            with wave.open(str(path), "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(a.tobytes())
            n += 1
        manifest.append({"path": str(path), "kind": kind, "voice": voice, "text": text, "hotword": hw})
    print(voice, "done", time.strftime("%T"), flush=True)
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
synth("Done.", "clone:dave_cullen")
print("clone model restored; synthesised", n, flush=True)
