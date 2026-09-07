#!/usr/bin/env python3
"""Project `livekit.tts` from config.yaml into the guardian's state dir.

The guardian speaks in the same cloned voice as voice mode, and there must be
exactly one place that decides what that voice is. config.yaml is it — but the
guardian runs on the system interpreter with no yaml module and, more to the
point, must not read the repo at all for anything on its critical path.

So the settings are *pushed* to it instead of pulled: this runs at stage time
(from guardian-stage.sh, best-effort) and writes a small JSON file that
`speak.py` overlays on its own defaults. If it never runs, or the repo is
mid-rewrite, speak.py falls back to those defaults and still sounds right —
this only keeps the two from drifting after a voice change.

Deliberately not importing `app.config`: it expands ${VAR} secrets and pulls
in the backend's dependency tree, and none of `livekit.tts` is a secret.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(os.environ.get("LLOYD_REPO", "/home/alansrobotlab/lloyd"))
DEST = Path(os.environ.get(
    "LLOYD_GUARDIAN_STATE", Path.home() / ".local/state/lloyd-guardian")) / "voice.json"

# Only these reach the guardian. `speed` and `shaping` are the two that were
# hard-won (see CLAUDE.md "Voice output"); the rest identify the endpoint.
KEYS = ("api_url", "model", "voice", "speed", "sample_rate", "tail_silence_ms")


def build(tts: dict, guardian_voice: dict | None = None) -> dict:
    out = {k: tts[k] for k in KEYS if k in tts}
    shaping = tts.get("shaping") or {}
    if "presence_eq" in shaping:
        out["presence_eq"] = bool(shaping["presence_eq"])
    if shaping.get("shelves"):
        out["shelves"] = shaping["shelves"]
    # Quiet hours come from `guardian.voice`, not `livekit.tts`, and the split
    # is load-bearing: livekit_worker reads livekit.tts, and a conversation
    # that goes mute at 23:00 because an alert policy leaked into it would be
    # a genuine bug. Only alerts are gated by the clock.
    qh = (guardian_voice or {}).get("quiet_hours")
    if isinstance(qh, dict):
        out["quiet_hours"] = qh
    return out


def main() -> int:
    try:
        import yaml
    except ImportError:
        print("sync-voice-config: no yaml module — leaving speak.py on its defaults",
              file=sys.stderr)
        return 0
    try:
        cfg = yaml.safe_load((REPO / "config.yaml").read_text(encoding="utf-8")) or {}
        tts = ((cfg.get("livekit") or {}).get("tts") or {})
        if not tts:
            print("sync-voice-config: livekit.tts is empty — nothing to sync",
                  file=sys.stderr)
            return 0
        guardian_voice = ((cfg.get("guardian") or {}).get("voice") or {})
        payload = build(tts, guardian_voice)
        DEST.parent.mkdir(parents=True, exist_ok=True)
        tmp = DEST.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, DEST)          # never leave speak.py a half-written file
        print(f"sync-voice-config: wrote {DEST} ({payload.get('voice')}, "
              f"speed {payload.get('speed')})", file=sys.stderr)
        return 0
    except Exception as exc:
        print(f"sync-voice-config: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 0                       # never block staging


if __name__ == "__main__":
    raise SystemExit(main())
