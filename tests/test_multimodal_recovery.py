"""One multimodal rejection latches images off for the rest of the run (#1419).

The engine refuses an image it cannot take with a 400, which the client raises
as ``MultimodalRejectedError`` and the loop answers by stripping every image ref
and retrying. Until this change the counter behind that retry was
``multimodal_recoveries = 0`` per ``run_query`` with ``if >= 1: raise`` — one
recovery per *turn* — while the 400 is per *request*, and
``_tool_history_message`` attached ``_image_refs`` to every later result whose
image ref routed ``native``. A ``desktop_capture`` after the first recovery put
an image back on the wire, the next 400 hit the ``raise``, and the exception
escaped ``run_query`` and ended the turn mid-work. A desktop turn takes more
than one capture by definition, so any model slot lying about
``supports_vision`` — ``LANGUAGE_MODEL_ONLY=1`` behind ``supports_vision: true``,
or an image-count 400 from the engine — killed the turn at its second
screenshot.

The scripted engine here refuses a request by *reading what was sent*: it calls
``wire_messages`` on the exact message list the loop holds and refuses iff that
list carries an ``image_url`` part. It never inspects the latch — a latch that
merely sets a flag while images keep going out, or one that stops the retry but
not the attachment, fails these tests.

The seam between the raise and the recovery is real HTTP. The fast tests below
drive ``run_query`` with ``app.harness.client.stream_chat`` replaced by a
callable that reuses the client's own wire step and raises what the client
raises (``_RefusesImages``), which is honest about what it covers: the
classification of a 400 into ``MultimodalRejectedError`` runs inside
``client.stream_chat`` and is not in that path.
``test_the_retry_over_http_reaches_the_engine_without_image_parts`` therefore
runs the whole way — a loopback engine answering over a socket, the client's
400 classification, the recovery, and the retry as a second real request.

Run through the real ``run_query``: the loop owns the assistant/tool bookkeeping
and the finalizer, and a scripted turn has to satisfy its guards (the summary
ratchet, the echo guard) exactly as production does — a stub of only the client
is what lets these assert anything about the turn rather than about a
reimplementation of it.
"""

import asyncio
import base64
import json
import logging
import struct

import pytest

import app.harness.loop as loop_mod
import app.harness.tool_images as ti
from app.harness.errors import MultimodalRejectedError
from app.harness.options import RunOptions

PNG_HEAD = (b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR"
            + struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0))
PNG_TAIL = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", 0xB085279)


def _png_bytes(n: int = 0) -> bytes:
    """A real 1x1 PNG with a payload that differs per call.

    Real because the loop routes the bytes through the production path —
    ``shape_tool_images`` writes the file, ``wire_messages`` sniffs it and
    base64s it — so the magic has to be there. Distinct per call because
    ``shape_tool_images`` dedupes identical bytes from the same tool and
    session, and a deduped ref would be dropped for a reason that has nothing
    to do with the latch.
    """
    idat = struct.pack(">I", 12) + b"IDAT" + f"capture-{n:04d}xxxx".encode() \
        + struct.pack(">I", 0) + b"IE" + b"ND"
    return PNG_HEAD + idat + PNG_TAIL


# The engine's own words. `looks_like_multimodal_rejection` is a substring test
# on this body, and clause 3 wants the body in the log line, so the string is
# fixed here rather than invented per test.
ENGINE_400 = ("This model does not support image input. Set "
              "LANGUAGE_MODEL_ONLY=0 to serve vision requests.")


@pytest.fixture
def blind_engine_cfg(monkeypatch, tmp_path):
    """A slot that claims vision and cannot take it — the trigger of #1419.

    `supports_vision: True` is what routes a capture `native`, so every
    screenshot the turn takes is put on the wire to a model that will refuse it.
    Patched at `tool_images._config`, the one place the routing decision reads
    config, so the decision under test is the production one.
    """
    state = {
        "harness": {"images": {"enabled": True, "route": "auto", "aux_model": ""}},
        "models": {"primary": {"supports_vision": True}},
    }
    monkeypatch.setattr(ti, "_config", lambda: state)
    monkeypatch.setattr(ti, "SESSIONS_DIR", tmp_path)
    ti.DEDUP.reset()
    return state


def _seeded_history(tmp_path):
    """History with one accepted screenshot already in it, as a desktop turn has.

    The tool message is the exact shape ``_tool_history_message`` produces: text
    content plus a private ``_image_refs`` key, which is the only form that
    reaches the wire. The ref points at a real file, because that is what a
    healthy turn's send does with it — reads the bytes at request time.
    """
    path = tmp_path / "seed.png"
    path.write_bytes(_png_bytes(0))
    return [
        {"role": "user", "content": "open the dialog"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "seed", "type": "function", "function": {
                "name": "desktop_capture",
                "arguments": json.dumps({"summary": "first look"})}}]},
        {"role": "tool", "tool_call_id": "seed", "content": "SEED element list",
         "_image_refs": [{"path": str(path), "sha256": "seed", "mime": "image/png",
                          "width": 1, "height": 1, "bytes": path.stat().st_size}]},
    ]


# A real `desktop_capture` call carries the `summary` its schema asks for;
# without one the summary ratchet rewrites the tool result and these assertions
# would be reading the ratchet's text rather than the latch's.
CAPTURE = {"name": "desktop_capture", "arguments": {"summary": "reading the dialog"}}


class _FakePool:
    """Pool for the scripted turn: real capture bytes, no MCP connection.

    The result is ``mcp_pool._flatten_result``'s shape, so the loop's next real
    step — ``shape_tool_images``, which persists the image and asks
    ``resolve_image_route`` for its route — runs rather than being stubbed out.
    The scripted result therefore arrives as base64 like any capture's does, and
    only the desktop session's lease and window manager are missing.
    """

    def __init__(self):
        self.calls: list[str] = []

    async def call_tool(self, name: str, args: dict, *, session_id: str = "", **_kw):
        n = len(self.calls) + 1        # CAPTURE-0 is the seeded one in history
        self.calls.append(name)
        return {
            "content": f"CAPTURE-{n} element list: [1] Sign In button",
            "structured_content": None,
            "is_error": False,
            "images": [{"data": base64.b64encode(_png_bytes(n)).decode(),
                        "mime_type": "image/png"}],
        }

    @property
    def discovered(self):
        """Advertise the capture tool, so dispatch finds it where it looks."""
        return [("lloyd-mcp", [{"name": "desktop_capture",
                                "description": "capture a window",
                                "inputSchema": {"type": "object",
                                                "properties": {}}}])]


def _image_parts(messages) -> dict[int, int]:
    """{message index: image_url parts} for one wired payload.

    Per message on purpose: a run that evicted the seed would show zero total
    parts while the post-flip capture still shipped one, so the aggregate is not
    the assertion clause 1 is — the clause names every message after the
    rejection.
    """
    out = {}
    for i, m in enumerate(messages):
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, list):
            n = sum(1 for p in c
                    if isinstance(p, dict) and p.get("type") == "image_url")
            if n:
                out[i] = n
    return out


def _parts_on(messages, tool_call_id: str) -> int:
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "tool" \
                and m.get("tool_call_id") == tool_call_id:
            return _image_parts([m]).get(0, 0)
    return 0


class _RefusesImages:
    """A text-only engine wearing the client's send path.

    ``stream_chat`` is replaced by this, and it reuses the client's own wire
    step (``wire_messages``) on the list the loop holds, then refuses iff an
    ``image_url`` part is in it — the test therefore never reads the latch, only
    whether an image reached the payload. It is still not the client: the
    classification of a 400 into ``MultimodalRejectedError`` lives in
    ``client.stream_chat``, which is why the HTTP test below exists.
    """

    def __init__(self, replies: list[tuple[str, list[dict]]], *,
                 refuse_images: bool = True):
        self.replies = replies
        self.refuse_images = refuse_images
        self.refusals = 0
        self.history: list[list[dict]] = []    # as the loop holds it
        self.captured: list[list[dict]] = []   # as it goes out, refs wired in

    def __call__(self, **kwargs):
        # Two views of every request: the list the loop built, which is where the
        # latch acts, and the wired payload, which is where the engine decides.
        # `wire_messages` copies the ref-bearing message and drops the private
        # key, so asserting refs on the payload alone would prove nothing.
        self.history.append([dict(m) for m in (kwargs.get("messages") or [])])
        wired = ti.wire_messages(list(kwargs.get("messages") or []))
        self.captured.append([dict(m) for m in wired])
        if self.refuse_images and ti.payload_has_images(wired):
            self.refusals += 1
            return self._refuse()
        text, tool_calls = self.replies[len(self.captured) - self.refusals - 1]
        return self._reply(text, tool_calls)

    async def _refuse(self):
        # An async generator that never yields: `stream_chat` is consumed with
        # `async for`, and the client's real 400 branch raises before its first
        # line is read — the same shape, so the loop sees an exception from
        # inside the stream, not from the call that set it up.
        raise MultimodalRejectedError(f"vLLM returned 400: {ENGINE_400}")
        yield  # pragma: no cover - unreachable, makes this an async generator

    async def _reply(self, text: str, tool_calls: list[dict]):
        chunks = [{"choices": [{"index": 0,
                                 "delta": {"role": "assistant", "content": text},
                                 "finish_reason": None}]}]
        for i, tc in enumerate(tool_calls):
            chunks.append({"choices": [{"index": 0, "delta": {"tool_calls": [{
                "index": i, "id": tc["id"], "type": "function",
                "function": {"name": tc["name"],
                             "arguments": json.dumps(tc["arguments"])}}]},
                "finish_reason": None}]})
        chunks.append({"choices": [{"index": 0, "delta": {},
                                    "finish_reason": ("tool_calls" if tool_calls
                                                      else "stop")}]})
        for c in chunks:
            yield c


def _sse_chunks(text: str, tool_calls: list[dict]):
    """OpenAI-format SSE lines for one reply, as vLLM frames them."""
    chunks = [{"choices": [{"index": 0,
                             "delta": {"role": "assistant", "content": text},
                             "finish_reason": None}]}]
    for i, tc in enumerate(tool_calls):
        chunks.append({"choices": [{"index": 0, "delta": {"tool_calls": [{
            "index": i, "id": tc["id"], "type": "function",
            "function": {"name": tc["name"],
                         "arguments": json.dumps(tc["arguments"])}}]},
            "finish_reason": None}]})
    chunks.append({"choices": [{"index": 0, "delta": {},
                                "finish_reason": ("tool_calls" if tool_calls
                                                  else "stop")}]})
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
    return body + "data: [DONE]\n\n"


def _options(base_url: str = "http://127.0.0.1:1") -> RunOptions:
    return RunOptions(model="primary", base_url=base_url, api_key="test-key",
                      system_prompt="You are Lloyd.", max_turns=8,
                      session_id="mm-recovery", tool_search_enabled=False)


async def _drain(gen):
    return [e async for e in gen]


def _patch_pool(monkeypatch, pool: "_FakePool"):
    """Swap the MCP pool the loop opens, the same seam the overflow test uses.

    `_build_pool` rather than `get_or_open_pool` because the loop awaits the
    former, and `tool_search_enabled=False` because the scripted turn has no
    tool catalog to search — it is here for the image path, not the schema path.
    """
    async def _build_pool(_options):
        return pool
    monkeypatch.setattr("app.harness.loop._build_pool", _build_pool)


def _run(monkeypatch, script, history, *, base_url="http://127.0.0.1:1"):
    """Drive one scripted run to completion; returns (events, pool).

    A re-raised ``MultimodalRejectedError`` propagates out of here, which is how
    the pre-fix shape shows up: the turn does not end on a stop reason, the call
    raises.
    """
    pool = _FakePool()
    _patch_pool(monkeypatch, pool)
    monkeypatch.setattr("app.harness.loop.stream_chat", script)
    events = asyncio.run(_drain(
        loop_mod.run_query(history, _options(base_url=base_url))))
    return events, pool


def test_a_second_capture_after_a_rejection_never_reaches_the_wire(
        monkeypatch, blind_engine_cfg, tmp_path):
    """Clauses 1 and 2 together: no image parts on any later request, and the
    run gets to make those later requests at all.

    The scripted turn answers the refusal with a second capture, so the bug's
    shape is intact: request 2 is the one that used to carry an image and die.
    The engine decides by reading its own payload, so a latch that only sets a
    flag, or that strips history but keeps attaching new refs, fails here.
    """
    script = _RefusesImages([          # refused once (the seed), then answered
        ("looking", [dict(CAPTURE, id="c1")]),
        ("again", [dict(CAPTURE, id="c2")]),
        ("done", []),
    ])
    events, _ = _run(monkeypatch, script, _seeded_history(tmp_path))

    assert script.refusals == 1, \
        "a second refusal means the post-flip requests still carried images"
    # Control: the request that was refused really did carry the seed on the
    # wire, so the zeros below mean "the latch held", not "no images existed".
    assert _parts_on(script.captured[0], "seed") == 1
    # Clause 2: the turn has to be alive long enough to ask more than once
    # after the refusal. At base this is where the exception arrives instead.
    assert len(script.captured) >= 3, (
        f"only {len(script.captured)} request(s): the turn died on the second "
        "refusal rather than continuing past the rejection")

    for n, msgs in enumerate(script.captured):
        if n == 0:
            continue        # the refused request: pre-flip by definition
        assert _image_parts(msgs) == {}, \
            f"request {n + 1} after the rejection still carries image parts"
        # ...and the private key the parts are built from is gone from the
        # history the loop keeps, so no later request can revive them.
        for m in script.history[n]:
            assert "_image_refs" not in m, (
                f"request {n + 1}: a tool message still carried _image_refs "
                "after the rejection")

    result = [e for e in events if e["type"] == "result"][-1]
    assert result["stop_reason"] in ("stop", "end_turn"), result["stop_reason"]


def test_the_suppressed_capture_tells_the_model_its_text_survived(
        monkeypatch, blind_engine_cfg, tmp_path):
    """`allow_images=False` is a lost picture, not a lost observation.

    The capture's element list and the path its file went to are what the turn
    drives on once it cannot see; silently swallowing the screenshot would turn
    a visible degradation into a model that reports on pixels it never got.
    """
    script = _RefusesImages([
        ("looking", [dict(CAPTURE, id="c1")]),
        ("done", []),
    ])
    events, _ = _run(monkeypatch, script, _seeded_history(tmp_path))
    tool_msgs = [m for msgs in script.captured[1:] for m in msgs
                 if m.get("role") == "tool" and m["tool_call_id"] == "c1"]
    assert tool_msgs, "the second capture never reached history at all"
    content = tool_msgs[-1]["content"]
    assert "CAPTURE-1 element list" in content
    assert ".png" in content, content


def test_the_flip_is_logged_once_with_the_advice_and_the_400_body(
        monkeypatch, blind_engine_cfg, tmp_path, caplog):
    """Clause 3: exactly one line for the whole turn, naming the config fix and
    the engine's own words.

    The body earns its place: `looks_like_multimodal_rejection` is a bare
    substring test, so any 400 whose text merely contains "image" is booked as a
    vision refusal — vLLM's per-prompt image-count refusal among them. Once the
    latch makes that class non-fatal, the body is the only trace a
    misclassification leaves. And the line has to stay greppable for
    `supports_vision: false`: that is the advice that fixes the slot for good.
    """
    # DEBUG, not WARNING: a per-capture line added later would most likely be an
    # info, and this has to fail if one appears.
    caplog.set_level(logging.DEBUG, logger="lloyd-harness-loop")
    script = _RefusesImages([
        ("looking", [dict(CAPTURE, id="c1")]),
        ("again", [dict(CAPTURE, id="c2")]),
        ("done", []),
    ])
    events, _ = _run(monkeypatch, script, _seeded_history(tmp_path))

    vision_lines = [r.getMessage() for r in caplog.records
                    if "supports_vision" in r.getMessage()]
    assert len(vision_lines) == 1, [str(x) for x in caplog.records]
    assert ENGINE_400 in vision_lines[0], vision_lines[0]
    assert "primary" in vision_lines[0], "the advice has to name the lying slot"
    assert "image parts stay off for the rest of this turn" in vision_lines[0]
    # The log line is not the only surface: the UI and the session row see the
    # flip exactly once too, then go quiet about it.
    raws = [e for e in events if e["type"] == "stream_raw"
            and "multimodal_rejected" in (e.get("error") or "")]
    assert len(raws) == 1, [r.get("error") for r in raws]


def test_a_turn_that_is_never_refused_still_sends_both_captures(
        monkeypatch, blind_engine_cfg, tmp_path):
    """Clause 4's driven half, and the proof that the per-message assertion is
    measuring what it claims.

    Identical script and config, one difference: the engine can see. The seeded
    screenshot and the fresh capture must both reach the wire — the latch's
    default-open value is what keeps a healthy desktop turn whole, so it is
    tested rather than assumed. The capture's own message is asserted
    individually: a run that evicted the seed would show a part count that says
    nothing about whether the new capture got through.
    """
    script = _RefusesImages([
        ("looking", [dict(CAPTURE, id="c1")]),
        ("done", []),
    ], refuse_images=False)
    events, _ = _run(monkeypatch, script, _seeded_history(tmp_path))

    assert script.refusals == 0
    assert _parts_on(script.captured[0], "seed") == 1
    assert _parts_on(script.captured[1], "c1") == 1, \
        "a healthy turn dropped its screenshot"
    assert sum(_image_parts(script.captured[1]).values()) == 2
    result = [e for e in events if e["type"] == "result"][-1]
    assert result["stop_reason"] in ("stop", "end_turn"), result["stop_reason"]


def test_the_retry_over_http_reaches_the_engine_without_image_parts(
        monkeypatch, blind_engine_cfg, tmp_path):
    """The whole seam over a socket, client classification included.

    The fast tests substitute a callable for ``client.stream_chat``, so the code
    that turns a 400 into ``MultimodalRejectedError`` is outside them. Here the
    engine is an aiohttp server on loopback that answers 400 with the real
    engine's body, and the loop retries on a second real request. That makes the
    claim this test makes the one that matters: the refusal the loop latches on
    is a refusal the *engine* actually sent, and the retry that must be clean is
    a payload the *engine* actually received.
    """
    from aiohttp import web

    seen: list[dict] = []
    replies = [("looking", [dict(CAPTURE, id="c1")]), ("signed in", [])]

    async def handler(request):
        body = await request.json()
        seen.append(body)
        if ti.payload_has_images(body.get("messages") or []):
            return web.json_response({"error": {"message": ENGINE_400}}, status=400)
        n_clean = sum(1 for b in seen
                      if not ti.payload_has_images(b.get("messages") or []))
        text, tool_calls = replies[min(n_clean - 1, len(replies) - 1)]
        return web.Response(text=_sse_chunks(text, tool_calls),
                            content_type="text/event-stream")

    async def _main():
        app = web.Application()
        app.router.add_post("/v1/chat/completions", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]
        try:
            return [e async for e in loop_mod.run_query(
                _seeded_history(tmp_path),
                _options(base_url=f"http://127.0.0.1:{port}"))]
        finally:
            await runner.cleanup()

    _patch_pool(monkeypatch, _FakePool())
    for var in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy",
                "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    events = asyncio.run(_main())

    assert len(seen) == 3, [ti.payload_has_images(b.get("messages") or [])
                            for b in seen]
    assert ti.payload_has_images(seen[0]["messages"]), \
        "request 1 was refused for a reason other than an image"
    for n, body in enumerate(seen[1:], start=2):
        assert _image_parts(body["messages"]) == {}, \
            f"request {n} reached the engine over HTTP with image parts"
    raws = [e for e in events if e["type"] == "stream_raw"
            and "multimodal_rejected" in (e.get("error") or "")]
    assert len(raws) == 1 and ENGINE_400 in raws[0]["error"], \
        "the client did not classify the 400 as a vision rejection"
    result = [e for e in events if e["type"] == "result"][-1]
    assert result["stop_reason"] in ("stop", "end_turn"), result["stop_reason"]
