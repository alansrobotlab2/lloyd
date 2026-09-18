"""Running the hearing pipeline off the event loop.

The pipeline is pure CPU — resample, Silero every 32 ms, openWakeWord every
80 ms — and measures about 6% of one core per participant here. That is small,
but it arrives as a spike every eighth frame, and the loop it would land on is
the same one streaming TTS frames into LiveKit on a 100 ms cadence. A 3 ms stall
in the wrong place is an audible gap in Lloyd's voice.

`asyncio.to_thread` per frame is the obvious fix and the wrong one: LiveKit
delivers 10 ms frames, so that is 100 thread handoffs per second per
participant, each with its own future. Batching frames to amortise that trades
the cost straight back into wake-word latency, which is the thing being fixed.

So: one thread per audio stream, fed by a queue, posting events back with
`call_soon_threadsafe`. Frames are never dropped — the queue is unbounded
because the consumer is 15x faster than real time, and a bounded queue that
dropped audio would reintroduce exactly the silent gaps this work is removing.
"""
from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from typing import Callable, Optional

import numpy as np

from .pipeline import HearingEvent, HearingPipeline

LOG = logging.getLogger("lloyd-agent-worker.runner")

_STOP = object()


class HearingThread:
    """Owns one `HearingPipeline` and the thread that drives it."""

    def __init__(
        self,
        pipeline: HearingPipeline,
        on_event: Callable[[HearingEvent], None],
        loop: Optional[asyncio.AbstractEventLoop] = None,
        name: str = "hearing",
    ) -> None:
        self.pipeline = pipeline
        self._on_event = on_event
        self._loop = loop or asyncio.get_running_loop()
        self._q: "queue.Queue" = queue.Queue()
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._started = False
        self._stopped = False
        #: Wall-clock seconds spent inside the pipeline, for the diag line.
        self.cpu_seconds = 0.0
        self.frames_in = 0

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread.start()

    def push(self, samples: np.ndarray, sample_rate: int) -> None:
        """Called from the audio consumer on the event loop. Never blocks."""
        if self._stopped:
            return
        self.frames_in += 1
        self._q.put((samples, sample_rate))

    def stop(self, timeout: float = 2.0) -> None:
        if not self._started or self._stopped:
            self._stopped = True
            return
        self._stopped = True
        self._q.put(_STOP)
        self._thread.join(timeout=timeout)

    def _emit(self, event: HearingEvent) -> None:
        try:
            self._loop.call_soon_threadsafe(self._on_event, event)
        except RuntimeError:
            # Loop is closing; the room is going away with it.
            pass

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is _STOP:
                break
            samples, sample_rate = item
            t0 = time.monotonic()
            try:
                events = self.pipeline.feed(samples, sample_rate)
            except Exception as e:
                LOG.warning("hearing pipeline raised: %s", e, exc_info=True)
                continue
            finally:
                self.cpu_seconds += time.monotonic() - t0
            for ev in events:
                self._emit(ev)
        # A participant who leaves mid-sentence still gets that sentence heard.
        try:
            tail = self.pipeline.close()
        except Exception:
            tail = None
        if tail is not None:
            self._emit(HearingEvent("utterance", utterance=tail, sample=tail.end_sample))
