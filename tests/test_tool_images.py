"""Screenshots a tool returns: persisted, referenced, routed per model.

Pins app/harness/tool_images.py and its seams — mcp_pool's flattening, the
loop's history message, the transcript row, the history rebuild, and the wire.
The invariant that matters most: no image part ever leaves toward a model
alias whose ``supports_vision`` is not literally ``true``.
"""

from __future__ import annotations

import asyncio
import base64
import json
import struct
import zlib

import pytest

from app.harness import tool_images as ti


def _png(w: int = 4, h: int = 3, seed: int = 0) -> bytes:
    raw = b"".join(b"\x00" + bytes([seed % 256]) * (w * 3) for _ in range(h))

    def chunk(t: bytes, d: bytes) -> bytes:
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


@pytest.fixture
def cfg(monkeypatch, tmp_path):
    state = {
        "harness": {"images": {}},
        "models": {
            "primary": {"alias": "primary", "base_url": "http://x"},
            "seer": {"alias": "seer", "base_url": "http://y", "supports_vision": True},
            "liar": {"alias": "liar", "supports_vision": "yes"},
        },
    }
    monkeypatch.setattr(ti, "_config", lambda: state)
    monkeypatch.setattr(ti, "SESSIONS_DIR", tmp_path)
    ti.DEDUP.reset()
    return state


# ── route ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("route,model,aux,expected", [
    ("auto", "primary", "", "drop"),
    ("auto", "seer", "", "native"),
    ("auto", "liar", "", "drop"),          # "yes" is not literally true
    ("native", "primary", "", "drop"),     # config asking cannot override the slot
    ("native", "primary", "seer", "aux"),
    ("auto", "primary", "seer", "aux"),
    ("aux", "seer", "seer", "aux"),
    ("aux", "primary", "primary", "drop"),  # an aux that cannot see is no aux
    ("drop", "seer", "seer", "drop"),
])
def test_route_matrix(cfg, route, model, aux, expected):
    cfg["harness"]["images"] = {"route": route, "aux_model": aux}
    assert ti.resolve_image_route(model) == expected


def test_native_only_when_supports_vision_is_literally_true(cfg):
    for v in (None, False, "true", 1, "yes"):
        cfg["models"]["primary"]["supports_vision"] = v
        assert ti.resolve_image_route("primary") != "native", v
    cfg["models"]["primary"]["supports_vision"] = True
    assert ti.resolve_image_route("primary") == "native"


# ── persist ───────────────────────────────────────────────────────────

def test_persist_writes_beside_text_spills_and_reads_dims(cfg, tmp_path):
    data = base64.b64encode(_png(7, 5)).decode()
    refs = ti.persist_tool_images([{"data": data, "mime_type": "image/png"}],
                                  session_id="s1", call_id="c/1",
                                  text=json.dumps({"orig_width": 14, "scale": 0.5}))
    (ref,) = refs
    assert ref["path"].startswith(str(tmp_path / "s1.tool-results"))
    assert ref["name"] == "c_1.img0.png"
    assert (ref["width"], ref["height"]) == (7, 5)
    assert ref["orig_width"] == 14 and ref["scale"] == 0.5
    assert "data" not in ref


def test_jpeg_dims():
    # Minimal SOF0 header: FFD8, APP0 skipped, SOF0 with h=20, w=30.
    jpg = (b"\xff\xd8" + b"\xff\xe0" + struct.pack(">H", 4) + b"ab"
           + b"\xff\xc0" + struct.pack(">HBHH", 11, 8, 20, 30) + b"\x00" * 6)
    assert ti._dims(jpg) == (30, 20)


# ── shaping per route ─────────────────────────────────────────────────

def _shape(model, png=None, call="c1", tool="desktop_capture"):
    data = base64.b64encode(png or _png()).decode()
    return asyncio.run(ti.shape_tool_images(
        images=[{"data": data, "mime_type": "image/png"}], content="summary",
        session_id="s1", call_id=call, tool_name=tool, model=model))


def test_native_route_returns_history_refs(cfg):
    text, refs, hist = _shape("seer")
    assert text == "summary"
    assert refs[0]["route"] == "native" and hist and hist[0]["sha256"]


def test_drop_route_says_so_and_sends_nothing(cfg):
    text, refs, hist = _shape("primary")
    assert hist == [] and refs[0]["route"] == "drop"
    assert "no vision route" in text


def test_aux_route_describes_and_sends_no_image(cfg, monkeypatch):
    cfg["harness"]["images"] = {"aux_model": "seer"}
    seen = {}

    async def fake(ref, summary, *, model_alias, session_id=""):
        seen["alias"], seen["summary"] = model_alias, summary
        return "A window with a Sign In button."
    monkeypatch.setattr(ti, "describe_image", fake)
    text, refs, hist = _shape("primary")
    assert hist == [] and refs[0]["described"] is True
    assert "Sign In" in text and seen == {"alias": "seer", "summary": "summary"}


def test_dedup_omits_twice_then_resends(cfg):
    outs = [_shape("seer", call=f"c{i}") for i in range(4)]
    sent = [bool(h) for _, _, h in outs]
    assert sent == [True, False, False, True]
    assert "screen unchanged" in outs[1][0]
    assert outs[1][1][0]["deduped_from"] == "c0.img0.png"


def test_dedup_is_per_tool_and_session(cfg):
    _shape("seer", call="a")
    assert _shape("seer", call="b", tool="browser_screenshot")[2]


# ── wire ──────────────────────────────────────────────────────────────

def test_wire_materialises_refs_and_passes_the_rest_through(cfg):
    _, _, hist = _shape("seer")
    plain = {"role": "user", "content": "hi"}
    tool = {"role": "tool", "tool_call_id": "c1", "content": "summary",
            "_image_refs": hist}
    msgs = [plain, tool]
    out = ti.wire_messages(msgs)
    assert out[0] is plain
    assert "_image_refs" not in out[1]
    parts = out[1]["content"]
    assert parts[0] == {"type": "text", "text": "summary"}
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "_image_refs" in tool   # the live history is untouched
    assert ti.payload_has_images(out) and not ti.payload_has_images([plain])


def test_a_list_without_refs_is_the_same_object(cfg):
    msgs = [{"role": "user", "content": "x"}]
    assert ti.wire_messages(msgs) is msgs


# ── eviction ──────────────────────────────────────────────────────────

def _history(n):
    msgs = [{"role": "system", "content": "sys", "_image_refs": [{"bytes": 1}]}]
    for i in range(n):
        msgs.append({"role": "tool", "content": f"r{i}",
                     "_image_refs": [{"bytes": 10, "sha256": str(i)}]})
    return msgs


def test_no_eviction_below_the_cap(cfg):
    msgs = _history(19)
    before = json.dumps(msgs)
    assert ti.enforce_outbound_cap(msgs) == 0
    assert json.dumps(msgs) == before


def test_crossing_the_cap_evicts_a_batch_of_the_oldest(cfg):
    msgs = _history(21)
    assert ti.enforce_outbound_cap(msgs, protect_from=len(msgs) - 2) == 8
    assert "_image_refs" in msgs[0]                         # index 0 untouched
    assert all("_image_refs" not in m for m in msgs[1:9])
    assert ti.EVICTED_PLACEHOLDER in msgs[1]["content"]
    assert all("_image_refs" in m for m in msgs[9:])


def test_byte_cap_also_triggers(cfg):
    cfg["harness"]["images"] = {"max_bytes": 25}
    msgs = _history(3)
    assert ti.enforce_outbound_cap(msgs) >= 1


def test_protected_batch_is_never_evicted(cfg):
    msgs = _history(30)
    ti.enforce_outbound_cap(msgs, protect_from=5)
    assert all("_image_refs" in m for m in msgs[5:])


def test_keep_newest(cfg):
    msgs = _history(6)
    assert ti.keep_newest(msgs[1:], 3) == 3
    assert [bool(m.get("_image_refs")) for m in msgs[1:]] == [False] * 3 + [True] * 3


# ── the seams ─────────────────────────────────────────────────────────

def test_mcp_pool_carries_images_out_and_omits_the_key_when_none():
    from mcp.types import CallToolResult, ImageContent, TextContent
    from app.harness.mcp_pool import _flatten_result
    both = _flatten_result(CallToolResult(content=[
        TextContent(type="text", text="hello"),
        ImageContent(type="image", data="QUJD", mimeType="image/jpeg")]))
    assert both["content"] == "hello"
    assert both["images"] == [{"data": "QUJD", "mime_type": "image/jpeg"}]
    only = _flatten_result(CallToolResult(content=[TextContent(type="text", text="x")]))
    assert only == {"content": "x", "is_error": False}


def test_tool_history_message_keeps_only_native_unrepeated_refs():
    from app.harness.loop import _tool_history_message
    evt = {"content": "t", "images": [
        {"path": "/a", "sha256": "1", "route": "native", "name": "a"},
        {"path": "/b", "sha256": "2", "route": "native", "deduped_from": "a"},
        {"path": "/c", "sha256": "3", "route": "drop"},
    ]}
    msg = _tool_history_message("c1", evt)
    assert msg["content"] == "t"
    assert [r["path"] for r in msg["_image_refs"]] == ["/a"]
    assert "_image_refs" not in _tool_history_message("c2", {"content": "t"})


def test_an_unrejected_turn_still_puts_native_refs_on_the_wire(cfg):
    """#1419's non-regression: with no rejection in the run, nothing changed.

    The latch is a parameter defaulted to open, so a turn that was never
    refused must still ride refs into history and still have them become
    ``image_url`` parts at send time — otherwise the fix for the second
    rejection would simply have been "never send screenshots".
    """
    from app.harness.loop import _tool_history_message
    _, refs, hist = _shape("seer")
    msg = _tool_history_message("c9", {"content": "capture", "images": refs})
    assert [r["path"] for r in msg["_image_refs"]] == [r["path"] for r in hist]
    assert msg["content"] == "capture"          # no suppression note
    out = ti.wire_messages([{"role": "user", "content": "hi"}, msg])
    assert ti.payload_has_images(out)
    assert out[1]["content"][1]["type"] == "image_url"
    assert "_image_refs" not in out[1]          # the private key never ships


def test_a_latched_turn_keeps_the_text_and_drops_the_refs(cfg):
    """The same event one rejection later: text and path, no refs, no parts."""
    from app.harness.loop import _tool_history_message
    _, refs, _hist = _shape("seer")
    msg = _tool_history_message("c9", {"content": "capture", "images": refs},
                               allow_images=False)
    assert "_image_refs" not in msg
    assert "capture" in msg["content"] and ".png" in msg["content"]
    out = ti.wire_messages([{"role": "user", "content": "hi"}, msg])
    assert not ti.payload_has_images(out)


def test_the_session_row_carries_refs_never_bytes():
    from app.transcript_entries import build_tool_result_entry
    row = build_tool_result_entry("c1", "t", timestamp="T", images=[
        {"path": "/p", "sha256": "s", "data": "SHOULD-NOT-SURVIVE"}])
    assert row["images"] == [{"path": "/p", "sha256": "s"}]
    assert "SHOULD-NOT-SURVIVE" not in json.dumps(row)
    assert "images" not in build_tool_result_entry("c1", "t", timestamp="T")


def test_rebuild_attaches_refs_only_on_the_native_route(cfg, tmp_path):
    from app.routers._messages_harness_adapter import _prepare_messages_for_harness
    f = tmp_path / "x.png"
    f.write_bytes(_png())
    history = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {"role": "tool", "tool_call_id": "c1",
         "content": [{"type": "text", "text": "t"}],
         "images": [{"path": str(f), "sha256": "s", "mime": "image/png"}]},
        {"role": "tool", "tool_call_id": "c2",
         "content": [{"type": "text", "text": "t"}],
         "images": [{"path": str(f), "sha256": "s", "evicted": True}]},
    ]
    blind = asyncio.run(_prepare_messages_for_harness(history, model="primary"))
    assert all("_image_refs" not in m for m in blind)
    seeing = asyncio.run(_prepare_messages_for_harness(history, model="seer"))
    assert seeing[1]["_image_refs"][0]["path"] == str(f)
    assert "_image_refs" not in seeing[2]
    assert isinstance(seeing[1]["content"], str)


def test_microcompact_clearing_drops_refs():
    from app.harness.microcompact import _replace_tool_content
    out = _replace_tool_content({"role": "tool", "content": "x",
                                 "_image_refs": [{}], "images": [{"path": "/p"}]},
                                "[cleared]")
    assert "_image_refs" not in out
    assert out["images"][0]["evicted"] is True


def test_token_estimate_counts_refs(cfg):
    from app.compaction import estimate_message_tokens
    base = estimate_message_tokens({"role": "tool", "content": "abc"})
    with_img = estimate_message_tokens({"role": "tool", "content": "abc",
                                        "_image_refs": [{}, {}]})
    assert with_img - base == 2 * 1500


def test_multimodal_rejection_detection():
    assert ti.looks_like_multimodal_rejection(
        "At most 0 image(s) may be provided in one prompt")
    assert not ti.looks_like_multimodal_rejection("maximum context length exceeded")
