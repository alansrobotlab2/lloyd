"""Lloyd LiveKit agent worker — Phase 3 echo/observe skeleton.

Joins every `${room_prefix}*` room that has a participant and reports
inbound audio levels. Acts as a connectivity proof for the RTC pipeline:
no STT, no TTS, no harness integration yet — that's Phase 4 / 5A.

The worker is "always-on": once it's running under supervisord, it polls
the LiveKit RoomService every 2s, connects to any matching room that
has participants but no agent, and disconnects from rooms that no
longer need it.

Run via:
  python agent-services/livekit_worker.py

Or under supervisord (see agent-services/supervisor/conf.d/agent-livekit-worker.conf).

Logs each subscribed audio track's RMS every ~2s, so you can see in the
log that audio frames are actually arriving from the browser.
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import signal
import sys
import time
from pathlib import Path

import yaml
from livekit import api as lkapi
from livekit import rtc


LOG = logging.getLogger("lloyd-agent-worker")

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config.yaml"

POLL_INTERVAL = 2.0          # seconds between RoomService polls
RMS_LOG_INTERVAL = 2.0        # seconds between per-track RMS logs
DEFAULT_ROOM_PREFIX = "lloyd-"


def _load_livekit_cfg() -> dict:
    with CONFIG_PATH.open() as f:
        cfg = yaml.safe_load(f) or {}
    lk = cfg.get("livekit") or {}
    if not lk.get("url") or not lk.get("api_key") or not lk.get("api_secret"):
        raise SystemExit("config.yaml: livekit.{url,api_key,api_secret} are required")
    return lk


def _http_url(ws_url: str) -> str:
    """LiveKit's RoomService API uses http(s); the rtc client uses ws(s)."""
    if ws_url.startswith("ws://"):
        return "http://" + ws_url[len("ws://"):]
    if ws_url.startswith("wss://"):
        return "https://" + ws_url[len("wss://"):]
    return ws_url


class RoomBridge:
    """One-room connection: subscribes to remote audio, logs RMS."""

    def __init__(self, room_name: str, lk_cfg: dict) -> None:
        self.room_name = room_name
        self.lk_cfg = lk_cfg
        self.room = rtc.Room()
        self._tasks: list[asyncio.Task] = []

    async def connect(self) -> None:
        token = (
            lkapi.AccessToken(self.lk_cfg["api_key"], self.lk_cfg["api_secret"])
            .with_identity(self.lk_cfg.get("agent_identity", "lloyd-agent"))
            .with_name("Lloyd")
            .with_grants(
                lkapi.VideoGrants(
                    room=self.room_name,
                    room_join=True,
                    can_publish=True,
                    can_subscribe=True,
                    can_publish_data=True,
                )
            )
            .to_jwt()
        )

        self.room.on("track_subscribed", self._on_track_subscribed)
        self.room.on("participant_disconnected", self._on_participant_disconnected)

        LOG.info("[%s] connecting…", self.room_name)
        await self.room.connect(self.lk_cfg["url"], token)
        LOG.info("[%s] connected as %s", self.room_name, self.room.local_participant.identity)

    async def disconnect(self) -> None:
        for t in self._tasks:
            t.cancel()
        await self.room.disconnect()
        LOG.info("[%s] disconnected", self.room_name)

    def _on_track_subscribed(self, track, publication, participant) -> None:  # noqa: ARG002
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        LOG.info("[%s] audio track from %s — listening for frames", self.room_name, participant.identity)
        task = asyncio.create_task(self._consume_audio(track, participant.identity))
        self._tasks.append(task)

    def _on_participant_disconnected(self, participant) -> None:
        LOG.info("[%s] participant left: %s", self.room_name, participant.identity)

    async def _consume_audio(self, track, identity: str) -> None:
        stream = rtc.AudioStream(track)
        last_log = 0.0
        sum_sq = 0.0
        sample_count = 0
        try:
            async for evt in stream:
                frame = evt.frame
                # int16 PCM samples → RMS for log readout.
                # frame.data is a `bytes`-like memoryview of int16 little-endian.
                samples = memoryview(frame.data).cast("h")
                for s in samples:
                    sum_sq += float(s) * float(s)
                sample_count += len(samples)
                now = time.monotonic()
                if now - last_log >= RMS_LOG_INTERVAL and sample_count > 0:
                    rms = math.sqrt(sum_sq / sample_count)
                    db = 20 * math.log10(rms / 32768.0) if rms > 0 else -120.0
                    LOG.info("[%s] audio rms %.0f (%.1f dBFS) over %d samples from %s",
                             self.room_name, rms, db, sample_count, identity)
                    sum_sq = 0.0
                    sample_count = 0
                    last_log = now
        except asyncio.CancelledError:
            pass
        finally:
            await stream.aclose()


class WorkerManager:
    """Polls RoomService and maintains one RoomBridge per active matching room."""

    def __init__(self, lk_cfg: dict) -> None:
        self.lk_cfg = lk_cfg
        self.room_prefix = lk_cfg.get("room_prefix", DEFAULT_ROOM_PREFIX)
        self.bridges: dict[str, RoomBridge] = {}
        self._stopping = asyncio.Event()
        self._http_url = _http_url(lk_cfg["url"])

    async def run(self) -> None:
        LOG.info("worker starting; polling %s every %.1fs for rooms with prefix %r",
                 self._http_url, POLL_INTERVAL, self.room_prefix)
        try:
            while not self._stopping.is_set():
                try:
                    await self._tick()
                except Exception as e:  # broad catch — keep the worker alive
                    LOG.warning("tick failed: %s", e)
                try:
                    await asyncio.wait_for(self._stopping.wait(), timeout=POLL_INTERVAL)
                except asyncio.TimeoutError:
                    pass
        finally:
            await self._teardown()

    async def _tick(self) -> None:
        async with lkapi.LiveKitAPI(self._http_url, self.lk_cfg["api_key"], self.lk_cfg["api_secret"]) as svc:
            rooms = (await svc.room.list_rooms(lkapi.ListRoomsRequest())).rooms

        active: dict[str, int] = {}
        for r in rooms:
            if not r.name.startswith(self.room_prefix):
                continue
            # `num_participants` includes the worker once it's joined; trigger
            # off whether *any* non-agent participant is present.
            non_agent = max(0, r.num_participants - (1 if r.name in self.bridges else 0))
            if non_agent > 0:
                active[r.name] = non_agent

        # Connect to new rooms.
        for room_name in active:
            if room_name in self.bridges:
                continue
            bridge = RoomBridge(room_name, self.lk_cfg)
            try:
                await bridge.connect()
                self.bridges[room_name] = bridge
            except Exception as e:
                LOG.warning("[%s] connect failed: %s", room_name, e)

        # Disconnect from rooms with no remote participants.
        for room_name in list(self.bridges):
            if room_name not in active:
                bridge = self.bridges.pop(room_name)
                try:
                    await bridge.disconnect()
                except Exception:
                    pass

    async def _teardown(self) -> None:
        for room_name, bridge in list(self.bridges.items()):
            try:
                await bridge.disconnect()
            except Exception:
                pass
        self.bridges.clear()

    def request_stop(self) -> None:
        self._stopping.set()


async def _amain() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    lk_cfg = _load_livekit_cfg()
    manager = WorkerManager(lk_cfg)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, manager.request_stop)
        except NotImplementedError:
            pass

    await manager.run()


if __name__ == "__main__":
    os.chdir(REPO_ROOT)
    asyncio.run(_amain())
