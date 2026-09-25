#!/usr/bin/env python
"""Talk to a build of Lloyd through a real LiveKit room, and time it.

Everything below the event boundary has offline tests; this is the one check
that crosses every seam at once — WebRTC in, the hearing pipeline, the gate,
`/api/voice/inject` streaming a real turn on the live primary, clause-by-clause
TTS, WebRTC out — and measures what a person would: how long after they stop
talking Lloyd starts.

Nothing it does touches live state:

  * the build under test runs as an automod **canary** (scripts/automod/
    canary.py) on its own ports, under a scratch HOME — its own sessions dir,
    workers.db and vault, workers and autonomy off, and a supervisord socket
    that does not exist, so it cannot reach the live engines' process control;
  * its worker joins rooms on its own prefix (`e2e-`), so the live worker
    never hears the test, and writes its diagnostics under that scratch HOME;
  * the test participant's voice is synthesised by the live TTS server in the
    cloned voice, so the server never swaps models mid-run.

Shared, read-only in effect: the LiveKit SFU, the primary engine (one short
turn per exchange — pause the worker pool first, as for any bench) and the TTS
server.

    python scripts/voice/e2e_voice.py            # the build at this checkout's HEAD
    python scripts/voice/e2e_voice.py --keep     # leave the rig up afterwards
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agent-services"))

LIVE_PY = Path.home() / "lloyd" / ".venvs" / "lloyd" / "bin" / "python"
UNTRACKED_MODELS = ("smart-turn", "parakeet-tdt-v3", "nemo-streaming-480ms", "campplus")
BACKEND_PORT, MCP_PORT = 18180, 18600
PREFIX = "e2e-"
SR = 48000

# (label, text, expect-reply regex or None for "must stay silent", pause after)
#
# Since 2026-09-24 the wake word opens a CONVERSATION (90 s), so the third line
# lands inside it but past the 6 s follow-up window: the addressee judge
# (djev) must reject a remark to nobody.
SCRIPT = [
    ("wake + question", "Hey Lloyd, what is six times seven?", r"42|forty.?two", 1.0),
    ("follow-up, no wake word", "And what is that divided by two?", r"21|twenty.?one", 9.0),
    ("conversation, not addressed", "The weather is really nice today.", None, 1.0),
    ("bare wake", "Hey Lloyd.", "", 1.2),
    ("question after bare wake", "What is the capital of France?", r"paris", 1.0),
]

# Talking over a reply: (label, opener, interjection, seconds into the reply,
# expectation). `barge` — the reply must stop and the interjection must be
# answered; `backchannel` — the reply must carry on and nothing is injected.
# The interjection lands after the barge-in warm-up (3 s of reply).
OVERLAPS = [
    ("barge-in mid-reply", "Hey Lloyd, count slowly from one to thirty, one number per sentence.",
     "Actually, what is two plus two?", 4.0, ("barge", r"\b4\b|four")),
    ("backchannel mid-reply", "Hey Lloyd, count slowly from one to twenty, one number per sentence.",
     "Yeah.", 4.0, ("backchannel", None)),
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── the rig ──────────────────────────────────────────────────────────────

def build_rig(rig: Path) -> Path:
    """A detached worktree of HEAD at <rig>/home/lloyd, where the canary's HOME
    isolation needs it, with the untracked model directories linked in."""
    wt = rig / "home" / "lloyd"
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()
    if not wt.exists():
        wt.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "-C", str(ROOT), "worktree", "add", "--detach",
                        str(wt), head], check=True, capture_output=True)
    else:
        # A rig left from an earlier run must test THIS commit, not that one.
        subprocess.run(["git", "-C", str(wt), "checkout", "-q", "--detach", head],
                       check=True, capture_output=True)
    # The worker reads its LiveKit keys from `<tree>/.env`, which is untracked:
    # a fresh rig has none and the worker exits before joining anything.
    env_src, env_dst = Path.home() / "lloyd" / ".env", wt / ".env"
    if env_src.exists() and not env_dst.exists():
        env_dst.symlink_to(env_src)
    for name in UNTRACKED_MODELS:
        src = Path.home() / "lloyd" / "agent-services" / "models" / name
        dst = wt / "agent-services" / "models" / name
        if src.exists() and not dst.exists():
            dst.symlink_to(src)
    return wt


def start_worker(rig: Path, wt: Path) -> subprocess.Popen:
    from scripts.automod.canary_config import canary_data_root
    env = dict(os.environ)
    env.update({
        "HOME": str(rig / "home"),
        # The canary's data root, so the worker and the backend it talks to agree.
        "LLOYD_DATA": str(canary_data_root(rig)),
        "LLOYD_BACKEND_URL": f"http://127.0.0.1:{BACKEND_PORT}",
        "LLOYD_LIVEKIT_ROOM_PREFIX": PREFIX,
        "PYTHONUNBUFFERED": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "LLOYD_VOICE_ALERTS": "0",
    })
    (rig / "logs").mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [str(LIVE_PY), str(wt / "agent-services" / "livekit_worker.py")],
        cwd=str(wt), env=env, start_new_session=True,
        stdout=open(rig / "logs" / "worker.log", "wb"), stderr=subprocess.STDOUT)


def synth(text: str) -> np.ndarray:
    """The live TTS server, in the cloned voice (so it never swaps models),
    resampled to 48 kHz int16 as a browser would send it."""
    from voice.resample import StreamResampler, to_int16

    req = urllib.request.Request(
        "http://127.0.0.1:8090/v1/audio/speech",
        data=json.dumps({"model": "qwen3-tts", "input": text,
                         "voice": "clone:dave_cullen", "response_format": "pcm",
                         "stream": True, "speed": 1.0}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        pcm = np.frombuffer(r.read(), dtype=np.int16)
    rs = StreamResampler(24000, out_rate=SR)
    return to_int16(np.concatenate([rs.push(pcm, 24000), rs.flush()]))


# ── the participant ──────────────────────────────────────────────────────

class Participant:
    """Joins the room, keeps a continuous mic stream going (room tone between
    phrases — the VAD needs real silence to close an utterance), and records
    Lloyd's track with arrival times."""

    def __init__(self, room_name: str, cfg: dict):
        self.room_name = room_name
        self.cfg = cfg
        self.speech: list[np.ndarray] = []
        self.spoke_until = 0.0
        self.heard: list[tuple[float, float]] = []   # (arrival, rms) per frame
        self.heard_pcm: list[tuple[float, np.ndarray]] = []
        self.wake_events: list[tuple[float, str]] = []
        self.agent_track_ready = asyncio.Event()

    async def connect(self):
        from livekit import api as lkapi
        from livekit import rtc

        lk = self.cfg["livekit"]
        token = (lkapi.AccessToken(lk["api_key"], lk["api_secret"])
                 .with_identity("e2e-tester").with_name("e2e")
                 .with_grants(lkapi.VideoGrants(
                     room=self.room_name, room_join=True, can_publish=True,
                     can_subscribe=True, can_publish_data=True))
                 .to_jwt())
        self.room = rtc.Room()

        @self.room.on("track_subscribed")
        def _sub(track, pub, participant):
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                asyncio.ensure_future(self._record(track))

        @self.room.on("data_received")
        def _data(packet):
            try:
                msg = json.loads(packet.data.decode())
            except Exception:
                return
            if msg.get("type") == "wake_state":
                self.wake_events.append((time.monotonic(), msg.get("state", "")))

        await self.room.connect(lk["url"], token)
        self.source = rtc.AudioSource(SR, 1)
        track = rtc.LocalAudioTrack.create_audio_track("mic", self.source)
        opts = rtc.TrackPublishOptions()
        opts.source = rtc.TrackSource.SOURCE_MICROPHONE
        await self.room.local_participant.publish_track(track, opts)
        self._mic = asyncio.ensure_future(self._mic_loop())

    async def _mic_loop(self):
        from livekit import rtc

        rng = np.random.default_rng(0)
        frame = SR // 100
        buf = np.zeros(0, dtype=np.int16)
        t_next = time.monotonic()
        while True:
            if buf.size < frame and self.speech:
                buf = np.concatenate([buf, self.speech.pop(0)])
            if buf.size >= frame:
                chunk, buf = buf[:frame], buf[frame:]
                if buf.size == 0 and not self.speech:
                    self.spoke_until = time.monotonic() + 0.01
            else:
                chunk = rng.normal(0, 40, frame).astype(np.int16)  # ~-58 dBFS
            await self.source.capture_frame(rtc.AudioFrame(
                data=chunk.tobytes(), sample_rate=SR, num_channels=1,
                samples_per_channel=frame))
            t_next += 0.01
            await asyncio.sleep(max(0.0, t_next - time.monotonic()))

    async def _record(self, track):
        from livekit import rtc

        self.agent_track_ready.set()
        async for evt in rtc.AudioStream(track, sample_rate=SR, num_channels=1):
            x = np.frombuffer(evt.frame.data, dtype=np.int16)
            now = time.monotonic()
            self.heard.append((now, float(np.sqrt(np.mean((x / 32768.0) ** 2)))))
            self.heard_pcm.append((now, x.copy()))

    async def say(self, pcm: np.ndarray) -> float:
        # Whole 10 ms frames only, or the tail sits in the mic buffer forever
        # and this never returns.
        frame = SR // 100
        pcm = np.concatenate([pcm, np.zeros((-pcm.size) % frame, dtype=np.int16)])
        self.spoke_until = float("inf")
        self.speech.append(pcm)
        while self.spoke_until == float("inf"):
            await asyncio.sleep(0.01)
        return self.spoke_until

    async def reply_after(self, t0: float, wait: float = 45.0, gap: float = 1.6):
        """(first sound, end, pcm) of Lloyd's reply after t0, or None."""
        deadline = t0 + wait
        first = None
        while time.monotonic() < deadline:
            loud = [t for t, r in self.heard if t > t0 and r > 0.01]
            if loud:
                first = loud[0]
                last = loud[-1]
                if time.monotonic() - last > gap:
                    pcm = np.concatenate([x for t, x in self.heard_pcm
                                          if first - 0.05 <= t <= last + 0.05])
                    return first, last, pcm
            elif first is None and time.monotonic() > deadline:
                break
            await asyncio.sleep(0.1)
        return None

    async def speaking_until_quiet(self, t0: float, wait: float = 60.0,
                                   gap: float = 1.6) -> Optional[float]:
        """The last loud frame after t0, once Lloyd has been quiet `gap` s."""
        deadline = t0 + wait
        while time.monotonic() < deadline:
            loud = [t for t, r in self.heard if t > t0 and r > 0.01]
            if loud and time.monotonic() - loud[-1] > gap:
                return loud[-1]
            await asyncio.sleep(0.1)
        return None

    async def close(self):
        self._mic.cancel()
        await self.room.disconnect()


# ── the run ──────────────────────────────────────────────────────────────

async def converse(cfg: dict, session: str, clips: dict) -> list[dict]:
    from voice import asr as voice_asr
    from voice.resample import StreamResampler

    rec = voice_asr.build_recognizer({
        "backend": "parakeet",
        "model_dir": str(Path.home() / "lloyd/agent-services/models/parakeet-tdt-v3")})
    rec.load()
    p = Participant(PREFIX + session, cfg)
    await p.connect()
    log(f"joined {PREFIX + session}; waiting for the worker")
    await asyncio.wait_for(p.agent_track_ready.wait(), timeout=60)
    await asyncio.sleep(3.0)          # let the wake model's 400 ms warmup pass
    results = []
    for label, text, expect, pause in SCRIPT:
        t_end = await p.say(clips[text])
        log(f"said {text!r}")
        wake_at = None
        # A bare wake expects silence but opens a 6 s window, so the next line
        # has to land inside it: wait out a reply briefly, not for 10 s.
        quiet_wait = 2.5 if expect == "" else 10.0
        reply = await p.reply_after(t_end, wait=45.0 if expect else quiet_wait)
        for t, st in p.wake_events:
            if t > t_end - 3 and st == "listening":
                wake_at = t - t_end
                break
        row = {"label": label, "said": text, "wake_ui_s": wake_at}
        if reply is None:
            row.update(replied=False, ok=expect is None or expect == "")
        else:
            first, last, pcm = reply
            rs = StreamResampler(SR)
            heard = rec.transcribe(np.concatenate([rs.push(pcm, SR), rs.flush()])).text
            row.update(replied=True, latency_s=first - t_end,
                       reply_s=last - first, heard=heard,
                       ok=bool(expect) and re.search(expect, heard, re.I) is not None)
        results.append(row)
        log(f"  -> {row}")
        await asyncio.sleep(pause)
    for label, opener, inter, into, (kind, expect) in OVERLAPS:
        await asyncio.sleep(3.0)
        t_end = await p.say(clips[opener])
        log(f"said {opener!r}")
        start = await p.reply_after(t_end, wait=45.0, gap=0.3)
        row = {"label": label, "said": inter, "wake_ui_s": None}
        if start is None:
            row.update(replied=False, ok=False)
            results.append(row)
            log(f"  -> {row}")
            continue
        first = start[0]
        await asyncio.sleep(max(0.0, first + into - time.monotonic()))
        t_inter = await p.say(clips[inter])
        log(f"  said {inter!r} {t_inter - first:.1f}s into the reply")
        if kind == "barge":
            # The counting has to stop: nothing loud for a while after the
            # interjection, other than the answer to it.
            quiet_at = None
            while time.monotonic() < t_inter + 20:
                await asyncio.sleep(0.1)
                loud = [t for t, r in p.heard if t > t_inter and r > 0.01]
                # the first silence of >= 0.5 s after the interjection is the stop
                prev = t_inter
                for t in loud:
                    if t - prev >= 0.5:
                        quiet_at = prev
                        break
                    prev = t
                if quiet_at is not None or (loud == [] and time.monotonic() - t_inter > 3):
                    quiet_at = quiet_at or t_inter
                    break
            stopped = None if quiet_at is None else quiet_at - t_inter
            answer = await p.reply_after(quiet_at or t_inter, wait=45.0)
            heard = ""
            if answer is not None:
                rs = StreamResampler(SR)
                heard = rec.transcribe(np.concatenate([rs.push(answer[2], SR),
                                                       rs.flush()])).text
            row.update(replied=answer is not None, stop_s=stopped,
                       latency_s=(answer[0] - t_inter) if answer else None,
                       reply_s=(answer[1] - answer[0]) if answer else None,
                       heard=heard,
                       ok=stopped is not None and stopped < 1.5 and
                       re.search(expect, heard, re.I) is not None)
        else:
            end = await p.speaking_until_quiet(t_inter, wait=60.0)
            carried = None if end is None else end - t_inter
            row.update(replied=True, carried_on_s=carried, reply_s=carried,
                       latency_s=None, heard="",
                       ok=carried is not None and carried > 3.0)
        results.append(row)
        log(f"  -> {row}")
    await p.close()
    return results


def latency_lines(worker_log: Path) -> list[str]:
    """The worker's per-turn `[latency]` breakdowns (voice/timeline.py)."""
    try:
        text = worker_log.read_text(errors="replace")
    except OSError:
        return []
    return [ln.split("[latency] ", 1)[1] for ln in text.splitlines() if "[latency] " in ln]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rig", default=str(Path.home() / ".cache" / "lloyd-voice-e2e"))
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    import yaml
    from scripts.automod.canary import Canary

    rig = Path(args.rig)
    wt = build_rig(rig)
    raw = yaml.safe_load((wt / "config.yaml").read_text())
    env_file = Path.home() / "lloyd" / ".env"
    envs = dict(line.split("=", 1) for line in env_file.read_text().splitlines()
                if "=" in line and not line.lstrip().startswith("#"))
    raw["livekit"]["api_key"] = envs["LIVEKIT_API_KEY"].strip().strip("'\"")
    raw["livekit"]["api_secret"] = envs["LIVEKIT_API_SECRET"].strip().strip("'\"")

    log("synthesising the participant's lines")
    clips = {text: synth(text) for _, text, _, _ in SCRIPT}
    for _, opener, inter, _, _ in OVERLAPS:
        clips[opener] = synth(opener)
        clips[inter] = synth(inter)

    canary = Canary(rig, wt, python=LIVE_PY, backend_port=BACKEND_PORT,
                    mcp_port=MCP_PORT, autorestart=False)
    worker = None
    session = time.strftime("%Y%m%d_%H%M%S") + "_e2ev"
    try:
        log("booting the canary backend + aggregator")
        canary.start()
        probe = canary.probe(timeout=240)
        if not probe["ok"]:
            log(f"canary unhealthy: {probe['errors']}")
            return 2
        log(f"canary up ({probe.get('internal_tools')} tools); starting the worker")
        worker = start_worker(rig, wt)
        results = asyncio.run(converse(raw, session, clips))
    finally:
        if worker is not None:
            worker.terminate()
            try:
                worker.wait(timeout=15)
            except subprocess.TimeoutExpired:
                worker.kill()
        if not args.keep:
            canary.stop()

    from scripts.automod.canary_config import canary_data_root
    # The canary backend may keep its sessions under its own data root or under
    # the rig tree's `.lloyd-data` (app.paths' non-production fallback).
    candidates = [canary_data_root(rig) / "sessions" / f"{session}.json",
                  wt / ".lloyd-data" / "sessions" / f"{session}.json"]
    found = next((c for c in candidates if c.exists()), None)
    msgs = json.loads(found.read_text()).get("messages", []) if found else []
    users = [c.get("text", "") for m in msgs if m.get("role") == "user"
             for c in (m.get("content") or []) if c.get("type") == "text"]
    print("\n=== injected user turns ===")
    for u in users:
        print(f"  {u!r}")
    print("\n=== exchanges ===")
    print(f"{'':32} {'wake→UI':>8} {'voice→voice':>12} {'reply':>7}  ok  heard")
    for r in results:
        wake = "-" if r["wake_ui_s"] is None else "%+.2fs" % r["wake_ui_s"]
        lat = "%.2fs" % r["latency_s"] if r.get("latency_s") is not None else "-"
        dur = "%.1fs" % r["reply_s"] if r.get("reply_s") is not None else "-"
        ok = "Y" if r["ok"] else "N"
        extra = ""
        if "stop_s" in r:
            extra = f" [stopped {r['stop_s']:.2f}s after]" if r["stop_s"] is not None \
                else " [never stopped]"
        if "carried_on_s" in r:
            extra = f" [carried on {r['carried_on_s'] or 0:.1f}s]"
        print(f"{r['label']:32} {wake:>8} {lat:>12} {dur:>7}  {ok}   "
              f"{r.get('heard', '')[:60]!r}{extra}")
    print("\n=== latency breakdown (worker [latency] lines, stage+ms) ===")
    for line in latency_lines(rig / "logs" / "worker.log"):
        print(f"  {line}")
    print(f"\nworker log: {rig / 'logs' / 'worker.log'}")
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
