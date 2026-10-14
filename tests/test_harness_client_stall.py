"""Pre-first-line silence on a streamed request: witnessed, still legal (#2704).

`harness.stream_chunk_timeout_seconds` (60 s on this box, `config.yaml:517`) exists for a
mid-generation wedge: vLLM under memory pressure stops producing deltas but keeps the
HTTP connection open, so an idle read is abnormal *once tokens have started*. The
exemption before the first token is deliberate and stays in force — a cold prefill emits
no bytes, and the secondary slot runs llama.cpp with `--parallel 1`, so a queued request
legitimately sits silent for as long as the request ahead of it takes.

What that exemption costs is invisibility. On 2026-10-14 autonomy task #86 finished its
work in 12 s, then its final-turn request never got a first token while task #39 held the
engine; the run was killed by its own 300 s wall clock at 08:06:01 local and the fleet
learned nothing about why. Three lines made that undiagnosable, and the first two are
what this file pins:

    timeout = httpx.Timeout(timeout_s, read=None, connect=10.0)          # client.py:187
    raw = await _next_line(lines, cancel_wait,
                           chunk_timeout_s if lines_seen else 0.0)       # client.py:258

set httpx's READ timeout to None, so `timeout_s` (600 s default,
`app/harness/options.py:77`) bounds connect/write/pool only, and hand `_next_line` a
deadline of 0.0 — no deadline — until a line has arrived. `grep -rn "chunk_timeout"
tests/*.py` returned 0 hits before this file: neither the 60 s bound nor the exemption
was pinned by anything.

The change pinned here is a WARNING and nothing more: naming the model, the URL and the
seconds waited, without aborting the request or raising. Aborting would convert a slow
night into a failed run, which is the opposite of what a job queued behind another job
needs — and the owed ruling on whether such a stall should be charged to the transport
rather than to the task is explicitly not this round's to make.
"""

import asyncio
import json
import logging
import re

import pytest

from app.harness.client import stream_chat
from app.harness.errors import StreamStalledError

MODEL = "qwen3.6-27b"
BASE_URL = "http://127.0.0.1:8001"
URL = "http://127.0.0.1:8001/v1/chat/completions"
LOGGER = "lloyd-harness-client"
# The threshold this file exercises. The shipped value is 60 s
# (`harness.stream_chunk_timeout_seconds`), which no test may wait out.
THRESHOLD_S = 0.2


def _sse(delta=None, finish=None):
    chunk = {"choices": [{"delta": ({"content": delta} if delta else {}),
                          "finish_reason": finish}]}
    return "data: " + json.dumps(chunk)


class _Response:
    """An SSE body whose first line arrives `hold_s` after the request opened.

    Lines after the first are immediate: the subject is the arrival of line one, and a
    later gap would exercise the mid-chunk bound instead.
    """

    def __init__(self, lines=(), hold_s=0.0):
        self._lines = list(lines)
        self._hold_s = hold_s
        self.status_code = 200

    async def aiter_lines(self):
        first = True
        for line in self._lines:
            if first:
                first = False
                if self._hold_s:
                    await asyncio.sleep(self._hold_s)
            yield line

    async def aread(self):
        return b""

    @property
    def request(self):
        return None


def _client_for(response):
    class _StreamCtx:
        async def __aenter__(self):
            return response

        async def __aexit__(self, *exc):
            return False

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def stream(self, method, url, headers=None, json=None):
            return _StreamCtx()

    return _Client


def _install(monkeypatch, response):
    """Stand in for `httpx.AsyncClient` so no socket and no engine is involved.

    The same seam `tests/test_component_manifest_send_sites.py:729` patches, for the
    same reason: everything under test is in `stream_chat`'s own read loop.
    """
    monkeypatch.setattr("app.harness.client.httpx.AsyncClient", _client_for(response))


async def _drain(**over):
    kwargs = dict(base_url=BASE_URL, model=MODEL,
                  messages=[{"role": "user", "content": "hi"}],
                  tools=None, extra_body=None, cancel_event=None, timeout_s=30.0,
                  chunk_timeout_s=THRESHOLD_S)
    kwargs.update(over)
    return [chunk async for chunk in stream_chat(**kwargs)]


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records
            if r.name == LOGGER and "FIRST SSE line" in r.getMessage()]


def _seconds_in(message: str) -> float:
    """The `after N.Ns` figure out of the witness line."""
    match = re.search(r"after (\d+(?:\.\d+)?)s", message)
    assert match, message
    return float(match.group(1))


@pytest.mark.asyncio
async def test_silence_before_the_first_line_warns_naming_model_url_and_seconds(
        monkeypatch, caplog):
    """Clause 4, the positive half: the wait is announced, not merely tolerated.

    A 0.7 s silence against a 0.2 s threshold must produce one WARNING naming the model,
    the URL and the seconds waited — and the stream must still deliver its chunks: the
    witness observes, it does not abort. That line is exactly what the 08:00 window on
    2026-10-14 did not have: 699 lines in `server.err.1`, none naming session `d953`,
    because time-to-first-line had neither a clock nor a voice.
    """
    _install(monkeypatch, _Response([_sse(delta="ok"), _sse(finish="stop"),
                                     "data: [DONE]"], hold_s=0.7))
    caplog.set_level(logging.WARNING, logger=LOGGER)

    chunks = await _drain()

    deltas = [c["choices"][0]["delta"].get("content") for c in chunks]
    assert deltas == ["ok", None], (
        f"a pre-first-line silence past the threshold came back as {chunks!r}: the "
        "request must NOT be aborted — a job queued behind another job is slow, not "
        "dead, and charging it to the task is the owed ruling, not this change")
    warnings = _warnings(caplog)
    assert len(warnings) == 1, _warnings(caplog)
    message = warnings[0]
    assert MODEL in message, message
    assert URL in message, message
    seconds = _seconds_in(message)
    assert THRESHOLD_S <= seconds < 0.7, (
        f"the WARNING reported {seconds}s against a {THRESHOLD_S}s threshold and a "
        "0.7s wait: it must name the seconds ACTUALLY waited. The configured bound is "
        "grep-able from config.yaml; the number a reader can act on at 03:00 is how "
        "long this engine has been silent.")


@pytest.mark.asyncio
async def test_a_stream_that_answers_inside_the_threshold_stays_quiet(monkeypatch,
                                                                      caplog):
    """The witness must not degrade into a per-request log line.

    Ordinary prefill is silent for a second or two and the fleet runs tens of thousands
    of turns; a warning that fired on normal waits would bury the one that means
    something, which is why the mid-chunk detector only ever speaks past its bound.
    """
    _install(monkeypatch, _Response([_sse(delta="fine"), _sse(finish="stop"),
                                     "data: [DONE]"], hold_s=0.01))
    caplog.set_level(logging.WARNING, logger=LOGGER)

    chunks = await _drain()

    assert [c["choices"][0]["delta"].get("content") for c in chunks] == ["fine", None]
    assert _warnings(caplog) == [], caplog.text


@pytest.mark.asyncio
async def test_the_first_line_disarms_the_witness_and_the_mid_chunk_bound_still_bites(
        monkeypatch, caplog):
    """The exemption keeps its exact shape: unbounded before line one, bounded after it.

    Tokens started and then stopped — the wedge `stream_chunk_timeout_seconds` was added
    for, pinned by `tests/test_harness_streaming.py:55`
    `test_stream_stall_guard_catches_a_wedge_without_a_disconnect`. The new witness must
    not swallow that, and must not speak either: this node holds line one fast and the
    following gap long, and expects `StreamStalledError` with no FIRST-line warning. That
    pair is what says the two detectors did not trade places.
    """
    class _GapResponse(_Response):
        async def aiter_lines(self):
            yield _sse(delta="first")
            await asyncio.sleep(1.0)          # > the 0.2s chunk bound
            yield _sse(delta="never arrives in time")

    _install(monkeypatch, _GapResponse())
    caplog.set_level(logging.WARNING, logger=LOGGER)

    with pytest.raises(StreamStalledError):
        await _drain()

    assert _warnings(caplog) == [], (
        "the pre-first-line witness fired on a stream whose first line had already "
        "arrived, which means the disarm is not wired and the two detectors overlap")


@pytest.mark.asyncio
async def test_a_request_that_dies_before_the_first_line_leaves_no_late_warning(
        monkeypatch, caplog):
    """The witness is cancelled with the request, so a dead stream cannot speak later.

    Status came back 200, the witness was armed, and then the body died on its first
    read — so this walks the `finally` that disarms it, not the status branch (which
    never arms it at all). An un-cancelled task waking past this test's end would print
    a WARNING about a request that finished seconds earlier, and attributing a silence
    to a dead request is the exact false alarm this file exists to prevent. Hence the
    wait past the threshold before the log is read.
    """
    import httpx

    class _DeadResponse(_Response):
        def aiter_lines(self):
            # A plain method handing back an async iterator, which is what
            # `aiter_lines()` is on a real httpx response; the ReadError surfaces on
            # the first `__anext__`, which is the first thing the loop does.
            async def _die():
                raise httpx.ReadError("connection reset before the first byte")
                yield ""  # unreachable — makes `_die` an async generator function
            return _die()

    _install(monkeypatch, _DeadResponse())
    caplog.set_level(logging.WARNING, logger=LOGGER)

    with pytest.raises(httpx.TransportError):
        await _drain()

    await asyncio.sleep(THRESHOLD_S * 2)   # any late witness has now had its turn
    assert _warnings(caplog) == [], caplog.text


@pytest.mark.asyncio
async def test_turning_the_chunk_timeout_off_turns_the_witness_off_too(monkeypatch,
                                                                       caplog):
    """One knob, one promise: `chunk_timeout_s: 0` means no stall clock at all.

    The flag's own escape hatch exists because prefill was once misread as a wedge, and a
    fleet that turned the detector off has not asked for a second one it cannot disable.
    """
    _install(monkeypatch, _Response([_sse(delta="still fine"), _sse(finish="stop"),
                                     "data: [DONE]"], hold_s=0.4))
    caplog.set_level(logging.WARNING, logger=LOGGER)

    chunks = await _drain(chunk_timeout_s=0.0)

    assert [c["choices"][0]["delta"].get("content") for c in chunks] == ["still fine",
                                                                        None]
    assert _warnings(caplog) == [], caplog.text
