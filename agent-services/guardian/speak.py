"""Speak an alert aloud in Lloyd's cloned voice — the guardian's sixth channel.

Why this file lives in `agent-services/guardian/` rather than beside
`tts_shaping.py`: `guardian-stage.sh` stages `agent-services/guardian/*.py`
and nothing else into the pinned snapshot. A module the guardian imports
must be in this directory or it simply will not exist at runtime, and the
failure would appear only in production.

Three properties, in priority order:

1. **It must never delay the loop.** The unit sets `WatchdogSec=90` against a
   5s tick. Synthesis takes seconds and playback takes as long as the
   sentence, so `dispatch` spawns a detached worker and returns immediately.
   It reports *dispatched*, never *heard* — the honest thing a fire-and-forget
   channel can claim.
2. **It must never raise.** Same contract as every other channel in
   `notify.py`.
3. **Shaping is a soft dependency.** The presence EQ and the WSOLA speed live
   in `agent-services/tts_shaping.py`, which needs numpy and sits outside the
   snapshot. When it imports, the alert sounds like Lloyd; when it does not,
   the alert is still spoken, just flat and at 1.0. The guardian's own failure
   domain never widens, because the worker is a separate process and the other
   five channels have already fired before it starts.

Suppression is on **disk**, not in memory, because the two producers are
different processes: the long-lived guardian daemon and the 15-minute
`lloyd-guardian-nag` oneshot. `guardian.py`'s in-memory `_alert_seen` cannot
see the nag, so without a shared record a persistent BROKEN state would say
the same sentence out loud four times an hour forever, which is how you teach
someone to unplug the speakers. See `policy.VOICE_REPEAT_SECONDS`.

Failures are logged to `voice.log` in the guardian state dir. An alerting
channel that fails silently is the exact anti-pattern `notify.py`'s own
docstring is about.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

try:                                    # policy is staged beside us
    import policy as _policy
    _REPO = _policy.REPO
except Exception:                       # standalone / test import
    _REPO = "/home/alansrobotlab/lloyd"

# Mirrors config.yaml `livekit.tts`. These are defaults, not the source of
# truth: `agent-services/bin/sync-voice-config.py` writes voice.json from
# config.yaml at stage time so the two cannot drift. They exist so a guardian
# with no synced config still speaks in the right voice.
DEFAULTS: dict = {
    "api_url": "http://127.0.0.1:8090",
    "model": "qwen3-tts",
    "voice": "clone:dave_cullen",
    "speed": 1.22,
    "sample_rate": 24000,
    "tail_silence_ms": 250,
    "presence_eq": True,
    "shelves": [
        {"freq": 1800, "gain_db": 2.75, "q": 1.0},
        {"freq": 8000, "gain_db": 4.0, "q": 0.7},
    ],
    "shaping_module": f"{_REPO}/agent-services/tts_shaping.py",
    "synth_timeout": 60.0,
    "max_chars": 240,
    # Spoken alerts only. This must never reach livekit_worker: voice mode has
    # to stay conversational at any hour, and a Lloyd who goes mute mid-answer
    # at 23:00 is a bug, not a courtesy. That is why the setting lives under
    # `guardian.voice` in config.yaml rather than `livekit.tts`.
    "quiet_hours": {"enabled": True, "start": 23, "end": 7, "allow_critical": False},
}

VENV_PYTHON = f"{_REPO}/.venvs/lloyd/bin/python"

CONFIG_NAME = "voice.json"
SPOKEN_NAME = "voice_spoken.json"
LOG_NAME = "voice.log"
_LOG_CAP = 256 * 1024


def voice_enabled() -> bool:
    """Master mute, checked at dispatch time.

    `LLOYD_VOICE_ALERTS=0` keeps every other channel and drops only the
    speech. Two callers need this and neither is hypothetical: the test suite,
    which must never make noise on a developer's machine, and anyone who wants
    the toasts without the talking at two in the morning.
    """
    return str(os.environ.get("LLOYD_VOICE_ALERTS", "1")).strip().lower() not in (
        "0", "false", "no", "off", "")


# ── config ────────────────────────────────────────────────────────────
def load_config(state_dir: Path) -> dict:
    """DEFAULTS overlaid with voice.json, which never has to exist."""
    cfg = dict(DEFAULTS)
    try:
        raw = (Path(state_dir) / CONFIG_NAME).read_text(encoding="utf-8")
        loaded = json.loads(raw)
        if isinstance(loaded, dict):
            for k, v in loaded.items():
                if v is None:
                    continue
                # One level of merge: {"quiet_hours": {"start": 22}} must move
                # the hour without silently dropping `enabled` and `end`.
                if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                    cfg[k] = {**cfg[k], **v}
                else:
                    cfg[k] = v
    except Exception:
        pass
    return cfg


def _log(state_dir: Path, message: str) -> None:
    try:
        p = Path(state_dir) / LOG_NAME
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists() and p.stat().st_size > _LOG_CAP:
            p.write_text("", encoding="utf-8")
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {message}\n")
    except Exception:
        pass


# ── what to actually say ──────────────────────────────────────────────
# A hash, not merely a long run of hex-legal characters: it must contain at
# least one a-f. Without that lookahead "SM_20260906_a1b2" loses its date and
# every 7-digit number in a body is silently eaten.
_HEX = re.compile(r"\b(?=[0-9a-f]*[a-f])[0-9a-f]{7,40}\b", re.I)
_PATH = re.compile(r"[~\w.-]*(?:/[\w.-]+)+/?")
_EMPTY_BRACKET = re.compile(r"\(\s*\)|\[\s*\]|\{\s*\}")
_WS = re.compile(r"\s+")
_FILLER_TAIL = re.compile(r"\b(?:to|from|into|and|the|at|on|of|with|by)\s*$", re.I)


def speakable(text: str) -> str:
    """Turn alert prose into something worth hearing out loud.

    Commit hashes are the main offender: "Rolled back 1a2b3c4d → 5e6f7a8b"
    read aloud is sixteen letters of noise, and the hash is in the toast, the
    ledger and the journal for anyone who needs it. Arrows, backticks, paths
    and newlines get the same treatment for the same reason — this is the one
    channel that cannot be skimmed.
    """
    t = text.replace("→", " to ").replace("->", " to ")
    t = t.replace("`", " ").replace("*", " ").replace("_", " ")
    t = _HEX.sub("", t)
    t = _PATH.sub(" ", t)                           # filesystem paths
    t = _EMPTY_BRACKET.sub(" ", t)                  # "(1a2b3c4d)" -> "()" -> gone
    t = _WS.sub(" ", t)
    t = re.sub(r"\s+([.,;:!?])", r"\1", t)         # "good ." after a hash left
    t = t.strip(" .,;:")
    # Removing a hash strips the noun but leaves the preposition that pointed
    # at it: "Rolled back 1a2b3c4d -> 5e6f7a8b" collapses to "Rolled back to",
    # which sounds like the sentence was cut off mid-word. Drop the dangler.
    prev = None
    while prev != t:
        prev = t
        t = _FILLER_TAIL.sub("", t).strip(" .,;:")
    return t


def utterance_for(level: str, title: str, body: str = "", *,
                  max_chars: int = 240) -> str:
    """One or two sentences. Long enough to know what happened, short enough
    that nobody is waiting for it to finish."""
    lead = "Critical." if level == "critical" else "Guardian alert."
    head = speakable(title or "Something needs attention")
    parts = [f"{lead} {head}."]
    rest = speakable((body or "").split("\n")[0])
    if rest and len(rest) > 12:
        parts.append(f"{rest}.")
    out = " ".join(parts)
    if len(out) > max_chars:
        out = out[:max_chars].rsplit(" ", 1)[0] + "."
    return out


def in_quiet_hours(cfg: dict, now: float | None = None) -> bool:
    """True when the hour says to withhold speech.

    Only the *sound* is withheld. The toast, journal, ledger, vault note and
    backlog task all still fire, so nothing is lost — the record is intact and
    waiting in the morning. That is what makes this safe to default on: it
    drops an accompaniment, never an alert.

    Local time, because the point is what hour it is in the room. A window
    that wraps midnight is the normal case (23 to 7), so `start > end` is
    handled, and `start == end` means no window rather than a full day of
    silence — the reading of an empty range that cannot mute the box forever.
    """
    qh = cfg.get("quiet_hours") or {}
    if not qh.get("enabled", False):
        return False
    try:
        start, end = int(qh["start"]), int(qh["end"])
    except (KeyError, TypeError, ValueError):
        return False
    if start == end:
        return False
    hour = time.localtime(now).tm_hour
    return start <= hour < end if start < end else (hour >= start or hour < end)


# ── suppression, shared across processes ──────────────────────────────
def should_speak(key: str, state_dir: Path, window: float,
                 now: float | None = None) -> bool:
    """True at most once per `window` for a given key, across processes.

    Records the decision under an exclusive lock *before* speaking, so the
    daemon and the nag oneshot racing on the same tick produce one utterance
    rather than two overlapping ones.
    """
    now = time.time() if now is None else now
    try:
        d = Path(state_dir)
        d.mkdir(parents=True, exist_ok=True)
        path = d / SPOKEN_NAME
        with open(path, "a+", encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            fh.seek(0)
            try:
                seen = json.loads(fh.read() or "{}")
                if not isinstance(seen, dict):
                    seen = {}
            except Exception:
                seen = {}
            last = seen.get(key)
            if isinstance(last, (int, float)) and now - last < window:
                return False
            seen[key] = now
            # Keep the file from growing without bound; anything outside the
            # window can no longer suppress anything.
            seen = {k: v for k, v in seen.items()
                    if isinstance(v, (int, float)) and now - v < max(window * 4, 86400)}
            fh.seek(0)
            fh.truncate()
            fh.write(json.dumps(seen))
        return True
    except Exception:
        # A broken suppression file must not mute the channel.
        return True


# ── synthesis ─────────────────────────────────────────────────────────
def synthesize(text: str, cfg: dict) -> bytes | None:
    """Raw s16le PCM from the Qwen3-TTS server, unshaped.

    Requests `stream: true` and `speed: 1.0` to match `livekit_worker`
    exactly: that is the path the voice was tuned against, and the server
    silently drops `speed` there anyway — we apply it ourselves in `shape`.
    """
    payload = json.dumps({
        "model": cfg["model"],
        "input": text,
        "voice": cfg["voice"],
        "response_format": "pcm",
        "stream": True,
        "speed": 1.0,
    }).encode()
    req = urllib.request.Request(
        f"{str(cfg['api_url']).rstrip('/')}/v1/audio/speech",
        data=payload, headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=float(cfg["synth_timeout"])) as resp:
        if not (200 <= resp.status < 300):
            return None
        return resp.read()


def _load_shaping_module(cfg: dict):
    """Import tts_shaping by absolute path.

    By path rather than by name because the worker runs with the snapshot dir
    on sys.path and the repo lives somewhere else entirely; putting the whole
    repo on sys.path to reach one module is a bigger door than this needs.
    """
    try:
        import importlib.util
        path = Path(cfg["shaping_module"])
        if not path.is_file():
            return None
        spec = importlib.util.spec_from_file_location("lloyd_tts_shaping", path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


def shape(pcm: bytes, cfg: dict, state_dir: Path | None = None) -> bytes:
    """Presence EQ + WSOLA speed, degrading in tiers rather than all at once.

    The EQ needs scipy; the stretch needs only numpy. A worker that fell back
    to the system interpreter can therefore still fix the *pace* even where it
    cannot fix the presence band, which is worth more than either-or.

    Which tier ran is written to voice.log. A shaping fallback that reports
    nothing is indistinguishable from shaping that worked — that is exactly
    how the first cut of this module shipped unshaped audio while looking
    healthy, and the log line is what caught it.
    """
    mod = _load_shaping_module(cfg)
    if mod is None:
        _log(state_dir, "shaping module unavailable — speaking unshaped")
        return pcm
    tiers = [bool(cfg["presence_eq"])]
    if tiers[0]:
        tiers.append(False)          # scipy missing: keep the pace, lose the EQ
    last: Exception | None = None
    for presence in tiers:
        try:
            shaper = mod.OutputShaper(
                sample_rate=int(cfg["sample_rate"]),
                speed=float(cfg["speed"]),
                presence_eq=presence,
                shelves=cfg["shelves"],
            )
            if not shaper.enabled:
                return pcm
            out = shaper.process(pcm) + shaper.flush()
            if presence != bool(cfg["presence_eq"]):
                _log(state_dir, f"shaping degraded to: {shaper.describe()}")
            return out
        except Exception as exc:
            last = exc
    _log(state_dir, f"shaping failed ({type(last).__name__}: {last}) — speaking unshaped")
    return pcm


def _with_tail(pcm: bytes, cfg: dict) -> bytes:
    ms = int(cfg.get("tail_silence_ms") or 0)
    if ms <= 0:
        return pcm
    return pcm + b"\x00" * (int(cfg["sample_rate"]) * ms // 1000 * 2)


# ── playback ──────────────────────────────────────────────────────────
def _player(path: str) -> list[str] | None:
    for cmd in (["paplay", path],
                ["pw-play", path],
                ["aplay", "-q", path],
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path]):
        if shutil.which(cmd[0]):
            return cmd
    return None


def play(pcm: bytes, cfg: dict) -> bool:
    sr = int(cfg["sample_rate"])
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="lloyd-guardian-")
    os.close(fd)
    try:
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm)
        cmd = _player(path)
        if cmd is None:
            return False
        budget = len(pcm) / float(2 * sr) + 15.0
        proc = subprocess.run(cmd, capture_output=True, timeout=budget)
        return proc.returncode == 0
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# ── the two entry points ──────────────────────────────────────────────
def _worker_python() -> str:
    """The lloyd venv interpreter when it is usable, otherwise our own.

    The detached worker is the one place the guardian's stdlib-only rule does
    not apply, and it is worth being explicit about why. That rule exists so
    the watchdog cannot be taken down by the thing it watches. This child runs
    *after* all five reliable channels have already fired, nothing waits on
    its exit, and its failure cannot reach the loop — so spending the venv
    here buys the presence EQ (scipy, which the system interpreter does not
    have) at no cost to the property the rule protects.

    A wrecked venv therefore costs a duller voice, never an alert.
    """
    venv = Path(VENV_PYTHON)
    try:
        if venv.is_file() and os.access(venv, os.X_OK):
            return str(venv)
    except OSError:
        pass
    return sys.executable


def speak_now(text: str, cfg: dict, state_dir: Path) -> bool:
    """Synthesise and play, blocking. This is what the detached worker runs."""
    try:
        pcm = synthesize(text, cfg)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        _log(state_dir, f"synth failed ({type(exc).__name__}: {exc}) for {text[:60]!r}")
        return False
    if not pcm:
        _log(state_dir, f"synth returned nothing for {text[:60]!r}")
        return False
    try:
        audio = _with_tail(shape(pcm, cfg, state_dir), cfg)
        ok = play(audio, cfg)
        if not ok:
            _log(state_dir, "playback failed — no usable player, or it exited nonzero")
        return ok
    except Exception as exc:
        _log(state_dir, f"playback error ({type(exc).__name__}: {exc})")
        return False


def dispatch(level: str, title: str, body: str, state_dir: Path, *,
             window: float, key: str | None = None) -> bool:
    """Suppression-check, then hand off to a detached worker.

    Returns whether an utterance was *dispatched*. False means suppressed or
    un-spawnable, never "the speaker did not work" — nobody is waiting long
    enough to find that out. See `voice.log` for what happened next.
    """
    if not voice_enabled():
        return False
    try:
        state_dir = Path(state_dir)
        cfg = load_config(state_dir)
        # Checked BEFORE should_speak, which records as it decides. Recording a
        # quiet-hours drop would spend the hourly slot on an utterance nobody
        # heard, so the 08:00 repeat of a 03:00 alert would stay silent too.
        if in_quiet_hours(cfg) and not (
                level == "critical" and cfg["quiet_hours"].get("allow_critical")):
            _log(state_dir, f"quiet hours — withholding speech for {title!r}")
            return False
        if not should_speak(key or title or "alert", state_dir, window):
            return False
        text = utterance_for(level, title, body, max_chars=int(cfg["max_chars"]))
        subprocess.Popen(
            [_worker_python(), str(Path(__file__).resolve()),
             "--state-dir", str(state_dir), "--text", text],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True,
        )
        return True
    except Exception as exc:
        _log(Path(state_dir), f"dispatch failed ({type(exc).__name__}: {exc})")
        return False


def main(argv: list[str] | None = None) -> int:
    import argparse
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    ap = argparse.ArgumentParser(description="Speak one line in Lloyd's voice.")
    ap.add_argument("--state-dir", required=True)
    ap.add_argument("--text", required=True)
    args = ap.parse_args(argv)
    state_dir = Path(args.state_dir)
    return 0 if speak_now(args.text, load_config(state_dir), state_dir) else 1


if __name__ == "__main__":
    raise SystemExit(main())
