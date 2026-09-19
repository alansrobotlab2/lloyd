"""The prefetch component is registered by the live stream handler (#581 clause 1).

The finding this file exists to close: `note_prefetch` is called at
`app/routers/messages.py:1877`, inside `post_message_stream`, and the injected
context block is the one component type in the manifest that no other module can
supply — `prompt_builder` registers the system-prompt half, the router registers
the retrieved half, and only the router knows what `prefetch_context_async`
returned for this turn. Before this file, nothing in the diff drove that handler:
the registry→line half was pinned in `tests/test_component_manifest.py` and the
router→registry half was pinned nowhere, so the component the item cares most
about — the one that moves when retrieval moves rather than when the user does —
was unproven before landing.

The seam is driven, not inspected. `tests/unit/test_skill_dispatch.py:344` reads
`inspect.getsource(post_message_stream)`; a source string cannot show that the
call runs, that it runs after the prefetch, or that its argument is the prefetched
text. Here the real handler coroutine is awaited with a real request body, the
real `stream_chat` then sends a real request for the same session against the
stub engine at the HTTP seam, and the assertion is on the line that lands in the
store. Everything patched below is either I/O the test has no business doing
(a vault search on the prefetch path, the session-meta write, the turn enqueue)
or an unrelated RunOptions input — each named at the patch site, and none of them
is the thing under test.

Against HEAD before the diff this file fails at import of
`app.component_manifest`, like every other manifest test.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from app import component_manifest as cm  # noqa: E402


def _helpers():
    """The stub engine from `tests/test_component_manifest.py`, loaded by path.

    Reused rather than re-implemented: the HTTP-seam stub there is what makes
    `stream_chat` build its real URL, body and headers with only the socket
    faked, and two copies of that stub would drift into two different fakes of
    the same process boundary.
    """
    path = REPO / "tests" / "test_component_manifest.py"
    spec = importlib.util.spec_from_file_location("_cm_helpers", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["_cm_helpers"] = module
    spec.loader.exec_module(module)
    return module


H = _helpers()

#: The injected block, in the shape `prefetch_context_async` actually returns it
#: — a `<context>` wrapper around the retrieved notes. The sentinel appears only
#: here: the same test that proves it reaches the manifest also proves the store
#: holds its digest and not its text.
PREFETCH_SENTINEL = (
    "<context>\n<vault-context>\n- PREFETCH-SEAM-SENTINEL note "
    "about content-addressed manifests\n</vault-context>\n</context>"
)
#: What the user typed. Kept distinct from the sentinel so a reader can see that
#: the block hash and the message hash name two different things that can move —
#: the exact distinction the separate prefetch hash buys.
USER_TEXT = "what changed between these two runs"


@pytest.fixture(autouse=True)
def _store(tmp_path, monkeypatch):
    monkeypatch.setenv("LLOYD_MANIFEST_STORE", str(tmp_path / "manifest-store"))
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    cm.reset_stats()
    cm._reset_registry()
    cm._reset_template_cache()
    cm._reset_tools_memo()
    yield
    cm.flush(timeout=5.0)
    cm._reset_registry()
    cm._reset_template_cache()
    cm._reset_tools_memo()


class _Request:
    """Only `await request.json()` of FastAPI's `Request` — all the handler uses."""

    def __init__(self, body: dict):
        self._body = body

    async def json(self):
        return self._body


def _returned(value):
    async def _coro(*a, **k):
        return value
    return _coro


def _capture(sink: list):
    async def _enqueue(session_id, turn, *, consumer_factory=None):
        sink.append(turn)
        return turn
    return _enqueue


async def _drive_handler(messages, *, session_id: str, monkeypatch,
                         enqueued: list) -> None:
    """Await the real `post_message_stream` for one chat turn, up to the enqueue.

    Patched, and why each one is out of scope here:

    * `prefetch_context_async` — the vault search itself. Its *return value* is
      the input to the seam, so the test hands it a known value and asserts what
      the handler does with it. Running the real search would only make the
      expected digest unknowable.
    * `build_system_prompt` — the system half of the prompt, whose registration
      through the `prompt_builder` hook is pinned by
      `tests/test_component_manifest.py::test_the_component_dict_prompt_builder_built_is_the_one_recorded`.
      Running it here would read the live vault into a test about the prefetch.
    * `_save_session_meta`, `enqueue_turn`, `set_last_user_session` — the writes
      and the queue handoff that would start a real agent turn on this machine.
      Everything above them, including `note_prefetch`, is the real code.
    * `_get_mcp_servers` / `_get_disallowed_tools` / `_get_harness_kwargs` —
      RunOptions inputs from `app.mcp_discovery` with nothing to do with the
      manifest; the real ones reach for the MCP discovery cache.
    * `drain_active` — the self-mod landing gate at the top of the handler. It
      reads automod state; it is not what is being proved, and an in-flight
      landing would otherwise answer 503 before the seam is reached.
    """
    import app.routers.automod as automod

    monkeypatch.setattr(messages, "prefetch_context_async",
                        _returned(PREFETCH_SENTINEL))
    monkeypatch.setattr(messages, "build_system_prompt",
                        lambda *a, **k: "SEAM SYSTEM PROMPT")
    monkeypatch.setattr(messages, "_save_session_meta", _returned(None))
    monkeypatch.setattr(messages, "enqueue_turn", _capture(enqueued))
    monkeypatch.setattr(messages, "set_last_user_session", lambda sid: None)
    monkeypatch.setattr(messages, "_get_mcp_servers", lambda: {})
    monkeypatch.setattr(messages, "_get_disallowed_tools", lambda *a, **k: [])
    monkeypatch.setattr(messages, "_get_harness_kwargs", lambda: {})
    monkeypatch.setattr(automod, "drain_active", lambda: False)

    await messages.post_message_stream(_Request(
        {"text": USER_TEXT, "session_id": session_id}))


def _lines(store_root: Path) -> list[dict]:
    out = []
    manifests = store_root / "manifests"
    if manifests.is_dir():
        for path in sorted(manifests.glob("*.ndjson")):
            out += [json.loads(raw) for raw in
                    path.read_text(encoding="utf-8").splitlines() if raw.strip()]
    return out


def _send_one(session_id: str, monkeypatch) -> list[dict]:
    """One real `stream_chat` request for `session_id`, through the stub engine.

    The stub is the same HTTP-seam double `tests/test_component_manifest.py`
    drives the loop with: it stands in for `httpx.AsyncClient` only, so
    `stream_chat` builds its own URL, headers and payload and then records the
    request the way it records every other one. The returned payload list is the
    positive control — one line in the store alongside one payload here means the
    line came from this request and not from a leftover.
    """
    monkeypatch.setattr("app.harness.client.httpx.AsyncClient", H._StubClient)
    H._StubClient.payloads = []
    asyncio.run(H._drain_stream(base_url="http://stub:8096", model="stub-model",
                                messages=[{"role": "user", "content": USER_TEXT}],
                                tools=None, session_id=session_id))
    assert len(H._StubClient.payloads) == 1, (
        "the request did not cross the seam, so a manifest line here would be "
        "evidence of nothing")
    cm.flush(timeout=8.0)
    return _lines(cm.store_root())


def _handler_registered_the_block(session_id: str, monkeypatch) -> list:
    """Drive the handler and prove it reached the enqueue, returning what it queued."""
    import app.routers.messages as messages

    assert cm.components_for(session_id) == {}, (
        "the registry already holds this session, so the assertion after the "
        "drive could pass on someone else's write")
    enqueued: list = []
    asyncio.run(_drive_handler(messages, session_id=session_id,
                              monkeypatch=monkeypatch, enqueued=enqueued))
    assert enqueued, (
        "the handler never reached the enqueue, so it may have returned before "
        "note_prefetch rather than after it")
    return enqueued


def test_the_stream_handler_registers_the_injected_block_for_its_turn(monkeypatch):
    """`note_prefetch` runs on the live handler, with the block it retrieved.

    The claim is narrow and it is the one the finding says is unproven: after a
    real `post_message_stream` call, `components_for(session_id)["prefetch"]` is
    the prefetched text. Asserted against the sentinel and not mere truthiness,
    because a weaker test would pass on an empty registration — which is what a
    turn with no retrieval gets, and which must not be confused with the block.
    """
    session_id = "20260919_121314_seamchat"
    _handler_registered_the_block(session_id, monkeypatch)
    entry = cm.components_for(session_id)
    assert entry.get("prefetch") == PREFETCH_SENTINEL, (
        "post_message_stream did not register the prefetched block for the "
        f"session it was about to run: {sorted(entry)}")


def test_the_injected_block_reaches_the_manifest_as_its_own_component(monkeypatch):
    """Router → registry → `stream_chat` → store, in one run, with no stub in between.

    This is the whole chain the finding asked for. After the handler has run for
    the session, the real `stream_chat` sends for that same session against the
    stub engine, and the line in the store must carry a `prefetch` component
    whose digest is exactly `sha256(PREFETCH_SENTINEL)` — so the hash on disk is
    the block the router retrieved, not a re-derivation of it. It also pins the
    two fields that make the line attributable: `session_id`, and `source` for a
    chat id, which is what separates a user's turn from a worker's in a diff.
    """
    session_id = "20260919_121315_seamchat"
    _handler_registered_the_block(session_id, monkeypatch)
    assert _lines(cm.store_root()) == [], (
        "a manifest line existed before the request was sent, so the count below "
        "proves nothing about where this one came from")

    rows = _send_one(session_id, monkeypatch)
    assert cm.stats()["write_errors"] == 0, cm.stats()
    assert len(rows) == 1, rows
    row = rows[0]
    assert row["session_id"] == session_id
    assert row["source"] == "chat", (
        "a chat session's line must tag `chat`, which is what separates it from a "
        "worker's in the diff — the id has three underscore-separated parts")
    assert row["prefetch"] == {
        "sha256": cm.digest_text(PREFETCH_SENTINEL),
        "bytes": len(PREFETCH_SENTINEL.encode("utf-8")),
    }, row.get("prefetch")
    # And the confidentiality half, for this component specifically: the block is
    # vault text, so the store must carry its digest and its length and neither
    # one word of it.
    written = "\n".join(json.dumps(r) for r in rows)
    assert "PREFETCH-SEAM-SENTINEL" not in written, (
        "the injected block's text landed in the manifest store")
    assert USER_TEXT not in written, "the user's message landed in it too"


def test_a_turn_with_no_injected_block_records_no_prefetch_component(monkeypatch):
    """The negative, so the field above cannot pass on a stale registry entry.

    An empty prefetch — which is what a turn whose retrieval found nothing gets —
    must add no `prefetch` key at all rather than a hash of `""`. A digest of the
    empty string would make every no-retrieval turn look like it shared one
    identical component, and a diff would blame it for a miss it did not cause.
    """
    session_id = "20260919_121316_seamchat"
    _handler_registered_the_block(session_id, monkeypatch)
    # The handler registered the sentinel above; this test is about the empty
    # case, so replace it the way the handler would have, through the same call.
    cm.note_prefetch(session_id, "")
    assert cm.components_for(session_id).get("prefetch") == ""

    rows = _send_one(session_id, monkeypatch)
    assert len(rows) == 1, rows
    assert "prefetch" not in rows[0], rows[0].get("prefetch")
    assert rows[0]["messages"][0]["sha256"], rows[0]
    assert cm.digest_text("") not in json.dumps(rows[0]), (
        "an empty prefetch was recorded as a component with the empty-string "
        f"digest: {rows[0].get('prefetch')}")
