"""One reaper pass at a time.

`reap_abandoned_rounds` runs from housekeeping, the held-loop retry, the boot
settle and every turn end, each in its own thread. Its rescues check a marker,
spawn, then write the marker, so two overlapping passes could start one landing
twice. A second caller stands down instead of queueing.
"""
from __future__ import annotations

import threading

from scripts.automod import state as S
from workers.sources import autocode as I


def test_a_second_reaper_pass_returns_at_once_while_one_is_running(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    reads = []

    def slow_read(*a, **k):
        reads.append(1)
        entered.set()
        release.wait(timeout=10)
        return []
    monkeypatch.setattr(S, "read_events", slow_read)

    first = threading.Thread(target=I.reap_abandoned_rounds)
    first.start()
    try:
        assert entered.wait(timeout=10), "the first pass never started"
        assert I.reap_abandoned_rounds() == []
        assert reads == [1], "the second pass ran beside the first"
    finally:
        release.set()
        first.join(timeout=10)
    # The lock is released when the pass ends: the next caller runs.
    assert I.reap_abandoned_rounds() == [] and reads == [1, 1]
