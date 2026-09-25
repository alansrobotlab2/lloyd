"""Every model request is recorded as a content-addressed component manifest (#581).

Clauses 1, 3 and 5 of #581, each pinned at the seam rather than beside the
function that does the work:

* **one line per request, off the render path** — a real three-iteration turn
  driven through `run_query` against a stub engine, asserting the digests and
  sizes the line has to carry (`test_three_iterations_emit_exactly_one_manifest_line_each`);
* **a broken store costs nothing** — the append point raises `OSError` and the
  turn still completes, with `stats()["write_errors"]` the only thing that
  noticed (`test_a_broken_writer_still_completes_the_turn`);
* **digests and byte counts only** — a sentinel inside a component appears in no
  written line, and the resolved store sits outside every git tree
  (`test_no_written_line_carries_component_text`,
  `test_the_store_resolves_outside_every_git_tree`).

Against HEAD before this diff every test in this file fails at import:
`app/component_manifest.py` does not exist, which is the same fact the item's
acceptance check measured as zero hits for
`component_manifest|request_manifest|prompt_diff` across `app/`, `scripts/` and
`tests/`.

What this file does NOT claim, because the item does not ask for it: no
component bytes are retained, so a manifest addresses a request it cannot
replay, and `rebuild_request` is deliberately absent. The docstring on
`app/component_manifest.py` records why and who retired the consumer.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import threading
from pathlib import Path

import pytest

from app import component_manifest as cm
from app.harness import tool_search_cache
from app.harness.client import stream_chat
from app.harness.loop import run_query
from app.harness.options import RunOptions

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

TOOLS = [{"type": "function", "function": {"name": "Bash",
                                           "parameters": {"type": "object"}}}]


# ── the engine, at the HTTP seam ────────────────────────────────────────────
#
# Stands in for `httpx.AsyncClient` — the boundary every send site in the harness
# crosses — so the code under test builds its real URL, real payload and real
# headers and only the socket is fake. `.stream()` answers the harness's
# streaming path; `.post()` the four non-streaming sites. Nothing here talks to a
# GPU, and nothing here is a substitute for the client: a change to how
# `stream_chat` builds its request body trips over this stub, as it should.

def _sse(*payloads: dict) -> list[str]:
    return [f"data: {json.dumps(p)}" for p in payloads] + ["data: [DONE]"]


def _said(text: str) -> dict:
    return {"choices": [{"delta": {"content": text}}]}


def _finished(reason: str) -> dict:
    return {"choices": [{"delta": {}, "finish_reason": reason}]}


def _used(prompt: int, completion: int) -> dict:
    return {"choices": [], "usage": {"prompt_tokens": prompt,
                                     "completion_tokens": completion}}


def _overlay(tmp_path: Path) -> Path:
    """A scratch overlay dir: an explicit SOUL.md, no vault read required."""
    overlay = tmp_path / "prompts"
    overlay.mkdir(parents=True, exist_ok=True)
    (overlay / "SOUL.md").write_text(
        "# SOUL\n\nSOUL-SENTINEL identity body for the manifest test.\n",
        encoding="utf-8")
    return overlay


def _note_real_components(session_id: str, tmp_path: Path) -> dict:
    """Have `prompt_builder` build and record a real component dict.

    An `overlay_dir` carrying its own `SOUL.md`, a `goal`, and an explicit
    `session_state="system_head"` are the three documented knobs that make the
    component set deterministic without touching the live vault or config.yaml:
    `SOUL.md`, `goal` and `harness_hints` then appear in a known order on every
    platform. The layout is pinned rather than read from
    `harness.prompt_layout.session_state` because production moved it to
    `system_tail` on 2026-09-25 (P1 step 1), which renders the goal as a trailing
    `session_state` component after `harness_hints` — a config rollout, not a
    manifest property, and the test went red on main the day it shipped (#1513). `memories` and `skills_index` are asserted nowhere
    in this file on purpose — whether they load depends on what is under the
    vault, and that is not what these clauses are about.
    """
    import prompt_builder

    prompt_builder.build_system_prompt(session_id=session_id,
                                       overlay_dir=_overlay(tmp_path),
                                       goal={"text": "record every model request"},
                                       session_state="system_head")
    comps = cm.components_for(session_id).get("components") or {}
    assert {"SOUL.md", "goal", "harness_hints"} <= set(comps), sorted(comps)
    return comps


def _tool_call(id_: str, name: str) -> dict:
    """One `tool_calls` delta, in the shape the parser in `client.py` expects."""
    return {"choices": [{"delta": {"tool_calls": [{
        "index": 0, "id": id_, "type": "function",
        "function": {"name": name, "arguments": json.dumps({"command": "true"})}}]}}]}


class _Resp:
    def __init__(self, body: dict | None = None, status: int = 200):
        self.status_code = status
        self._body = body or {}
        self.request = None

    def json(self):
        return self._body

    @property
    def text(self):
        return json.dumps(self._body)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _StreamResp(_Resp):
    def __init__(self, lines: list[str]):
        super().__init__()
        self._lines = lines

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self) -> bytes:
        return b""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _PostCM:
    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *exc):
        return False


class Engine:
    """Records every payload that crossed the seam; answers with `script`."""

    posted: list[dict] = []
    streamed: list[dict] = []
    stream_calls = 0
    script: list[list[str]] = []
    on_request = None          # called with each payload, at send time

    def __init__(self, *a, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def stream(self, method, url, headers=None, json=None):
        Engine.stream_calls += 1
        Engine.streamed.append(json)
        if Engine.on_request:
            Engine.on_request(json)
        idx = len(Engine.streamed) - 1
        lines = Engine.script[idx] if idx < len(Engine.script) else [_said("x"), _finished("stop"), _used(1, 1)]
        return _StreamResp(lines)

    async def post(self, url, headers=None, json=None, timeout=None):
        Engine.posted.append(json)
        if Engine.on_request:
            Engine.on_request(json)
        content = json.get("_stub_content", '{"verdict": "confirmed"}')
        return _Resp({"choices": [{"message": {"content": content},
                                   "finish_reason": "stop"}], "usage": {}})


@pytest.fixture(autouse=True)
def _store(tmp_path, monkeypatch):
    """A store of our own, and clean module state around every test.

    Both halves matter: the module-level writer thread, session registry,
    tools-array memo and counters outlive a test, and an inherited tools memo
    would make one test's request read as another's identical generation.
    """
    monkeypatch.setenv("LLOYD_MANIFEST_STORE", str(tmp_path / "manifest-store"))
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    Engine.posted, Engine.streamed, Engine.script = [], [], []
    Engine.stream_calls, Engine.on_request = 0, None
    cm.reset_stats()
    cm._reset_registry()
    cm._reset_template_cache()
    cm._reset_tools_memo()
    yield tmp_path / "manifest-store"
    cm.flush(timeout=5.0)
    cm.reset_stats()
    cm._reset_registry()
    cm._reset_template_cache()
    cm._reset_tools_memo()


@pytest.fixture(autouse=True)
def _no_shared_tool_cache():
    asyncio.run(tool_search_cache.clear())
    yield
    asyncio.run(tool_search_cache.clear())


class _LinesStream:
    """A response body that is a fixed list of byte lines, no socket involved."""

    status_code = 200

    def __init__(self, lines):
        self._lines = list(lines)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    def __aiter__(self):
        async def _gen():
            for line in self._lines:
                yield line
        return _gen()

    def aiter_lines(self):
        async def _gen():
            for line in self._lines:
                yield line.decode("utf-8") if isinstance(line, bytes) else line
        return _gen()


class _StubClient:
    """An engine that accepts anything and returns one empty completion.

    For the tests about *what was sent* rather than what came back: it keeps every
    payload it was handed, so a test can assert the requests it is judging actually
    happened instead of passing against an empty store.
    """

    payloads: "list[dict]" = []

    def __init__(self, **_kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    def stream(self, _method, _url, *, json=None, **_kw):  # noqa: ANN001
        _StubClient.payloads.append(json)
        return _LinesStream([b"data: [DONE]\n\n"])


def _lines(store: Path) -> list[dict]:
    """Manifest lines on disk, oldest first. Always call after `cm.flush()`."""
    root = store / "manifests"
    if not root.is_dir():
        return []
    out: list[dict] = []
    for path in sorted(root.glob("*.ndjson")):
        out += [json.loads(r) for r in path.read_text(encoding="utf-8").splitlines() if r.strip()]
    return out


class _FakePool:
    @property
    def discovered(self):
        return [("lloyd-mcp", [{"name": "Bash", "description": "shell",
                                "inputSchema": {"type": "object", "properties": {}}}]
                 )]

    async def call_tool(self, name, args, *, session_id="", **_kw):
        return {"content": "FAKE_RESULT", "is_error": False}


def _drive_turn(monkeypatch, *, session_id="manifest-loop", visible_tools=None):
    """Run one real three-request turn; return the events and the manifest lines.

    `run_query` is the real loop and `app/harness/client.py::stream_chat` is the
    real client, so the manifest is written by the code that ships and the
    iteration index is the loop's own, not a fixture's. `visible_tools` is what
    the caller passed as `RunOptions.visible_tools_capture`, so the array the
    loop sent can be compared with the array the caller believes it sent.
    """
    pool = _FakePool()

    async def _build_pool(_options):
        return pool

    monkeypatch.setattr("app.harness.loop._build_pool", _build_pool)
    monkeypatch.setattr("app.harness.client.httpx.AsyncClient", Engine)
    monkeypatch.setattr("app.harness.finalizer.httpx.AsyncClient", Engine)
    Engine.script = [
        _sse(_said("a"), _tool_call("c1", "Bash"), _finished("tool_calls"), _used(10, 2)),
        _sse(_said("b"), _tool_call("c2", "Bash"), _finished("tool_calls"), _used(12, 2)),
        _sse(_said("done"), _finished("stop"), _used(14, 3)),
    ]
    options = RunOptions(model="primary", session_id=session_id,
                         tool_search_enabled=False, system_prompt="SYSTEM V1")
    if visible_tools is not None:
        options.visible_tools_capture = visible_tools

    async def _drain():
        return [e async for e in run_query(
            [{"role": "user", "content": "go"}], options)]

    events = asyncio.run(_drain())
    cm.flush(timeout=5.0)
    return events


def test_three_iterations_emit_exactly_one_manifest_line_each(monkeypatch, tmp_path):
    """Clause 1: one line per request, with everything the diff needs to name a part.

    Three iterations means three requests and therefore exactly three lines — a
    fourth would double-count a request and a fifth would mean the finalizer's
    two payload spellings were both recorded (they are not; only the one that
    reached the wire is). `params` carries the sampling keys and nothing else: never `messages`, never
    `tools`, whose identity belongs to the `tools` block — so a tool-set change
    can never be reported as "`params` moved".
    """
    comps = _note_real_components("manifest-loop", tmp_path)
    _drive_turn(monkeypatch)
    lines = _lines(Path(os.environ["LLOYD_MANIFEST_STORE"]))
    assert len(lines) == 3, lines
    assert all(ln["model"] == "primary" for ln in lines)
    # The provider/chat-template identifier: both fields on every line, and the
    # provider half names the engine slot the request was addressed to.
    assert all("provider" in ln and "chat_template" in ln for ln in lines), lines[0].keys()
    assert all(ln["provider"]["slot"] == "primary" for ln in lines), lines[0]["provider"]
    assert [ln["iteration"] for ln in lines] == [1, 2, 3]
    assert len({ln["request_id"] for ln in lines}) == 3
    assert all(ln["session_id"] == "manifest-loop" for ln in lines)

    for line in lines:
        assert [c["name"] for c in line["components"]] == list(comps)
        assert all(c["sha256"].startswith("sha256:") and c["bytes"] > 0
                   for c in line["components"])
        assert line["messages"][0]["index"] == 0
        assert all("sha256" in m and "bytes" in m for m in line["messages"])
        assert "messages" not in line["params"] and "tools" not in line["params"]
        assert line["tools"]["count"] == len(TOOLS)
    # One turn, so one component set. Three different digest tuples would mean
    # the registry was rebuilt mid-turn — the false positive the diff would then
    # blame on the prompt.
    assert len({tuple(c["sha256"] for c in ln["components"]) for ln in lines}) == 1


def test_the_component_list_keeps_position_order_through_the_file(monkeypatch, tmp_path):
    """The ordered list survives the JSON round trip, which is what makes a diff readable.

    `prompt_builder`'s insertion order is the prompt's own order — identity
    first, the harness's own operating rules last — and the diff prints positions
    in it, so a set or a sorted dict at the write site would silently turn "the
    second component changed" into "some component changed".
    """
    comps = _note_real_components("order-test", tmp_path)
    assert list(comps)[0] == "SOUL.md", list(comps)
    assert list(comps)[-1] == "harness_hints", list(comps)

    _drive_turn(monkeypatch, session_id="order-test")
    line = _lines(Path(os.environ["LLOYD_MANIFEST_STORE"]))[0]
    # The order survives the round trip through JSON, which is what makes
    # "in position order" a fact about the file and not about a sort.
    assert [c["name"] for c in line["components"]] == list(comps)


def test_the_tools_array_is_hashed_once_and_each_definition_separately(monkeypatch):
    """The thing #520 can only ablate today: is the tool set itself stable?

    A single array digest proves the set as a whole; the per-definition digests
    are what say *which* tool moved. The expensive part is memoized per
    tools-array generation, so all three lines share one digest while the array
    is unchanged.
    """
    _drive_turn(monkeypatch)
    lines = _lines(Path(os.environ["LLOYD_MANIFEST_STORE"]))
    for line in lines:
        assert line["tools"]["sha256"].startswith("sha256:")
        assert line["tools"]["count"] >= 1
        assert [d["name"] for d in line["tools"]["definitions"]]
        assert all(d["sha256"].startswith("sha256:") and d["bytes"] > 0
                   for d in line["tools"]["definitions"])
    assert len({line["tools"]["sha256"] for line in lines}) == 1


def test_a_moved_tool_definition_is_visible_as_that_definition_and_not_the_array(monkeypatch):
    """The diff granularity the whole item exists for, on real lines.

    One definition's description changed; the array digest necessarily moved
    with it, and the record still has to say which definition did — otherwise
    the answer to "did the model get a different tool?" is only "yes".
    """
    monkeypatch.setattr("app.harness.client.httpx.AsyncClient", _StubClient)
    _StubClient.payloads = []
    one = dict(TOOLS[0])
    two = json.loads(json.dumps(one))
    two["function"]["description"] = "now with an extra sentence"
    for array in ([one], [one], [two]):
        asyncio.run(_drain_stream(base_url="http://stub:8096", model="primary",
                                  messages=[{"role": "user", "content": "go"}],
                                  tools=array))
    cm.flush(timeout=5.0)
    assert len(_StubClient.payloads) == 3, "the three requests did not happen"
    first, second, third = _lines(Path(os.environ["LLOYD_MANIFEST_STORE"]))
    assert first["tools"]["sha256"] == second["tools"]["sha256"]
    assert first["tools"]["sha256"] != third["tools"]["sha256"]
    named = {d["name"]: d["sha256"] for d in first["tools"]["definitions"]}
    moved = {d["name"]: d["sha256"] for d in third["tools"]["definitions"]}
    assert named["Bash"] != moved["Bash"]


async def _drain_stream(**kwargs):
    """One real `stream_chat` call against the stub engine, with no loop between.

    The three extra kwargs are part of `stream_chat`'s signature — a call that
    omitted them would be a call the harness never makes.
    """
    kwargs.setdefault("extra_body", {})
    kwargs.setdefault("cancel_event", None)
    kwargs.setdefault("timeout_s", 5.0)
    return [e async for e in stream_chat(**kwargs)]


def test_a_broken_writer_still_completes_the_turn(monkeypatch):
    """Clause 3: the store may die; the turn may not.

    Every append raises `OSError`, as it would on a full disk, a revoked mount
    or a read-only home. The turn must still issue all three requests and finish
    — `asyncio.run` above would have propagated anything that escaped — and the
    only trace may be the counter. `lines_written` staying at zero is what says
    the swallow happened at the write and not after a partial line, which would
    leave a truncated JSON row for the next reader of `prompt-diff` to choke on.
    """
    def _boom(_root, _text):
        raise OSError("simulated full disk")

    monkeypatch.setattr(cm, "_append_line", _boom)
    events = _drive_turn(monkeypatch)
    assert Engine.stream_calls == 3, "a manifest failure shortened the turn"
    assert not any("is_error" in str(e) and "manifest" in str(e).lower()
                   for e in events)
    store = Path(os.environ["LLOYD_MANIFEST_STORE"])
    assert _lines(store) == []
    stats = cm.stats()
    assert stats["recorded"] == 3, stats
    assert stats["write_errors"] == 3, stats
    assert stats["lines_written"] == 0, stats
    assert stats["hash_errors"] == 0, stats


def test_a_broken_hasher_is_counted_and_never_reaches_the_caller(monkeypatch):
    """The other failure mode: hashing, not writing. Same contract, new counter."""
    def _boom(_obj):
        raise ValueError("simulated serializer failure")

    monkeypatch.setattr(cm, "canonical_json", _boom)
    assert cm.record_request(base_url="http://stub:8096", model="primary",
                             payload={"model": "primary",
                                      "messages": [{"role": "user", "content": "go"}]},
                             session_id="s") is None
    stats = cm.stats()
    assert stats["hash_errors"] == 1, stats
    assert stats["recorded"] == 0, stats


def test_no_manifest_write_happens_inside_the_request_path(monkeypatch):
    """Off the render path, proved by thread identity and by timing.

    TTFT on the primary is 1.87 s; a manifest that cost syscalls inside the token
    stream would spend part of that. Two things are pinned here: at the instant
    each request crossed the seam no append had happened yet, and every append
    that did happen ran on the writer thread, never on the one driving the turn.
    """
    seen_at_send: list[int] = []
    appends: list[str] = []
    writer_threads: set[str] = set()

    def _on_request(_payload):
        seen_at_send.append(len(appends))

    def _record(_root, text):
        writer_threads.add(threading.current_thread().name)
        appends.append(text)

    Engine.on_request = _on_request
    monkeypatch.setattr(cm, "_append_line", _record)
    _drive_turn(monkeypatch)
    cm.flush(timeout=5.0)

    assert seen_at_send == [0, 0, 0], seen_at_send
    assert len(appends) == 3
    assert writer_threads == {"component-manifest"}, writer_threads
    assert "MainThread" not in writer_threads


def test_no_written_line_carries_component_text(tmp_path, monkeypatch):
    """Clause 5: the store holds hashes and sizes, and nothing else.

    Sentinels stand for the three kinds of thing this thing could otherwise
    become: a system prompt, an email body in a tool result, and a parameter that
    is not content. The system-prompt sentinel is not in the payload — it is a
    prompt_builder component that rides into the *rendered* prompt only — so an
    implementation that hashed or stored the whole request would pass every
    payload-shaped check and fail this one.
    """
    sys_sent = "SOUL-SENTINEL-8f2c1a7d"
    user_sent = "EMAIL-BODY-4b9de0c3"
    param_sent = "TEMPERATURE-VALUE-77e1"
    tools = [{"type": "function", "function": {
        "name": "Bash", "description": "shell",
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string", "default": user_sent}}}}}]

    cm.note_components("sentinel-test", {
        "SOUL.md": f"# Identity\n{sys_sent}\n",
        "memories": "# MEMORY.md\n\nsecond component body\n",
    })
    cm.record_request(base_url="http://stub:8096", model="m", session_id="sentinel-test",
                      iteration=1, send_site="tests/test_component_manifest.py",
                      payload={"model": "m",
                               "messages": [{"role": "user", "content": user_sent}],
                               "tools": tools, "temperature": param_sent})
    cm.flush(timeout=5.0)

    store = Path(os.environ["LLOYD_MANIFEST_STORE"])
    written = "\n".join(p.read_text(encoding="utf-8")
                        for p in store.rglob("*") if p.is_file())
    assert written, "the store wrote nothing, so this test would pass vacuously"
    for sentinel in (sys_sent, user_sent):
        assert sentinel not in written
    assert "EMAIL-BODY" not in written and "SOUL-SENTINEL" not in written

    line = _lines(store)[0]
    # The non-content half is kept verbatim, so it has to be *provably* the
    # non-content half: nothing message- or tool-shaped may appear in it.
    assert set(line["params"]) <= {"model", "stream", "stream_options", "temperature",
                                   "tools", "tool_choice", "max_tokens", "priority",
                                   "chat_template_kwargs"}, sorted(line["params"])
    # `params` is the sampling half: `tools` never appears in it, because the
    # array's identity lives in the `tools` block — mirroring it would make the
    # diff report two parts moving for one change.
    assert "tools" not in line["params"], sorted(line["params"])
    assert "messages" not in line["params"], sorted(line["params"])
    assert line["params"]["temperature"] == param_sent
    assert "messages" not in line["params"] and "payload" not in line["params"]
    assert "tools" not in line["params"], (
        "this payload sent no tools array; `params` is the non-content-bearing "
        "keys that were, so an absent array must not appear as a key")

    # And no bytes anywhere on disk for a second copy of the components.
    assert not (store / "blobs").exists()


def test_the_store_resolves_outside_every_git_tree(tmp_path):
    """Clause 5, mechanically: the default store has no git tree among its ancestors.

    The positive control is the repo itself in the same call — if the walk ever
    stopped reporting trees at all, the assertion below would pass for the wrong
    reason. A store inside the checkout would be committed one `git add -A` away
    by the next job that sweeps the tree, and a manifest line is a fingerprint of
    vault text, mail and user messages.
    """
    control = cm.git_tree_containing(REPO / "app" / "component_manifest.py")
    assert Path(control).resolve() == REPO, (
        "the git-tree walk found no tree even for the checkout itself — every "
        "assertion below would now pass vacuously")

    resolved = cm.store_root().resolve()
    assert cm.git_tree_containing(resolved) == "", (
        f"manifest store {resolved} sits inside git tree "
        f"{cm.git_tree_containing(resolved)}")

    # Same answer for a store inside this very repo's worktree, the case a
    # well-meaning operator would choose by hand.
    assert cm.git_tree_containing(REPO / "data" / "manifests") == str(REPO)
    # And a path with no tree at all answers "" for the same reason, not because
    # the walk failed: /tmp has no .git and neither does the fixture directory.
    assert cm.git_tree_containing(tmp_path) == ""


def test_the_provider_slot_names_the_engine_a_request_is_addressed_to(monkeypatch):
    """The identifier's value comes from the same table requests are built from.

    `models.<name>.base_url` is the key, so a two-engine box resolves to two
    different answers — which is the point: #551 is an item about measuring the
    secondary on the work it is routed to, and a manifest that cannot say which
    engine it addressed has nothing to measure. A base_url with no configured
    engine answers `unrecorded`, never a guess at the default, because a wrong
    engine label is worse than an empty one.
    """
    from app.config import CONFIG

    monkeypatch.setitem(CONFIG, "models", {
        "primary": {"base_url": "http://127.0.0.1:8096/v1", "engine": "vllm"},
        "secondary": {"base_url": "http://127.0.0.1:8097/v1", "engine": "llama.cpp"},
    })
    assert cm.provider_for("http://127.0.0.1:8096/v1")["slot"] == "primary"
    assert cm.provider_for("http://127.0.0.1:8097/v1/")["slot"] == "secondary"
    stranger = cm.provider_for("http://elsewhere:9999/v1")
    assert stranger["slot"] == cm._UNRECORDED, stranger
    assert stranger["base_url"] == "http://elsewhere:9999/v1", (
        "the base_url is the fallback identity, so an unresolved slot is still "
        "attributable to an endpoint")


def test_the_prefetch_block_is_hashed_as_its_own_component():
    """`<context>` is injected by the chat router, not by `prompt_builder`.

    It is a prefix sitting between the system prompt and the user message, so it
    breaks the cached prefix on its own and has to be nameable on its own: a
    component list that folded it into the system prompt would blame the wrong
    part every time a background injection landed mid-session.
    """
    cm.note_components("pf", {"SOUL.md": "identity"},
                       prefetch_text="<context>\n- mail summary\n</context>")
    assert cm.record_request(
        base_url="http://stub:8096", model="m", session_id="pf", iteration=1,
        send_site="tests/test_component_manifest.py",
        payload={"model": "m", "messages": [{"role": "user", "content": "go"}]})
    cm.flush(timeout=5.0)
    line = _lines(Path(os.environ["LLOYD_MANIFEST_STORE"]))[0]
    assert line["prefetch"]["sha256"].startswith("sha256:")
    assert line["prefetch"]["bytes"] > 0
    assert [c["name"] for c in line["components"]] == ["SOUL.md"]


def test_a_component_dict_reaches_the_line_only_through_prompt_builder(monkeypatch, tmp_path):
    """The one boundary with no call chain across it, so it is pinned explicitly.

    `prompt_builder.build_system_prompt` writes into the registry as a side
    effect and `app/harness/client.py` reads it from inside the loop's task. A
    refactor that renames the hook, moves the call below the return, or keys the
    registry on something other than `session_id` leaves both sides green — and
    silently leaves the manifest with no component list, which is exactly the
    record #520 needs. This test drives the two sides for real: prompt_builder
    builds, the loop sends, the line carries the components prompt_builder named.
    """
    import prompt_builder

    # One call, no call between: it returns the rendered prompt AND hands the
    # same named dict to the registry, so each recorded body must be found in
    # the rendered string byte for byte. That is the assertion that fails if the
    # hook moves, the key changes, or the dict is rebuilt rather than recorded.
    rendered = prompt_builder.build_system_prompt(
        session_id="seam-test", overlay_dir=_overlay(tmp_path),
        goal={"text": "record every model request"})
    registry = cm.components_for("seam-test").get("components") or {}
    assert registry, "prompt_builder handed nothing to the registry"
    for name, body in registry.items():
        assert body and body in rendered, f"{name} was recorded but is not in the prompt"

    _drive_turn(monkeypatch, session_id="seam-test")
    line = _lines(Path(os.environ["LLOYD_MANIFEST_STORE"]))[0]
    got = {c["name"]: c for c in line["components"]}
    assert set(got) == set(registry), sorted(got)
    for name, body in registry.items():
        assert got[name]["sha256"] == cm.digest_text(body), name
        assert got[name]["bytes"] == cm._size(body), name


def test_a_session_prompt_builder_never_built_is_reported_unrecorded(monkeypatch, tmp_path):
    """The registry may come up empty; what it must never do is guess.

    A caller that skips `build_system_prompt` gets
    `components_captured: "unrecorded"` and no component list — not a copy of
    some other session's prompt, which would be a wrong answer delivered as a
    real one. This is the shape every non-chat worker has until it opts in, and
    the reason the field exists at all.
    """
    import prompt_builder

    prompt_builder.build_system_prompt(session_id="other-session",
                                       overlay_dir=_overlay(tmp_path))
    assert cm.components_for("other-session").get("components"), (
        "prompt_builder recorded nothing at all, so this test would pass vacuously")
    _drive_turn(monkeypatch, session_id="nobody-built-this")
    line = _lines(Path(os.environ["LLOYD_MANIFEST_STORE"]))[0]
    assert line["components"] == []
    assert line["components_captured"] == cm._UNRECORDED

def test_the_chat_template_field_says_why_when_no_file_is_configured():
    """The absent half, asserted exactly rather than accepted either way.

    A version the manifest cannot name is a variable that moved without appearing
    in the diff, so the absence must be *reported*, not guessed. This is what
    shipped `config.yaml` yields — it has no `harness.component_manifest` section,
    so the found path below is the only way that branch is ever driven, and this
    test pins the other branch by its exact shape: `unrecorded`, plus a source that
    names the two keys that would fix it.
    """
    got = cm.chat_template("qwen3-32b", "http://127.0.0.1:8096/v1")

    assert got == {"id": "unrecorded",
                   "source": "no chat template file found; set "
                             "harness.component_manifest.chat_template_path or "
                             "model_dir"}, got


def test_a_configured_chat_template_is_recorded_by_digest(tmp_path, monkeypatch):
    """The found path by explicit path: the digest must be the file's, byte for byte.

    Two different templates can render the same messages into different prompts, so
    two turns whose components hash identically can still differ in what the engine
    actually saw — the template is the only such variable, and replaying against a
    different one silently compares two different prompts. A digest rather than a
    path string: a path proves the file was reachable, not that it was unchanged.

    Driven here because nothing else drives it — shipped `config.yaml` has no
    `harness.component_manifest` section, so every other test in this file takes the
    absent branch.
    """
    from app.config import CONFIG
    template = tmp_path / "chat_template.jinja"
    body = "{% for message in messages %}{{ message.content }}{% endfor %}"
    template.write_text(body, encoding="utf-8")
    monkeypatch.setitem(CONFIG, "harness", {
        "component_manifest": {"chat_template_path": str(template)}})
    cm._template_cache.clear()

    got = cm.chat_template("primary", "http://127.0.0.1:8096/v1")

    assert got["id"] == f"sha256:{hashlib.sha256(body.encode()).hexdigest()}", got
    assert got["source"] == str(template), got


def test_a_chat_template_found_under_model_dir_is_digested_in_candidate_order(
        tmp_path, monkeypatch):
    """The found path by directory: the candidate list is the search, so pin its order.

    `model_dir` is how an operator points the record at a served checkpoint, and the
    resolver tries `chat_template.jinja`, `.json`, `.jsonl` in that order. The order
    matters when a checkpoint ships more than one — the manifest records whichever it
    read first, and a reader comparing two turns has to know which file that was. The
    digest and the source must therefore name the *chosen* file, not the directory.
    """
    from app.config import CONFIG
    (tmp_path / "chat_template.json").write_text('{"unused": true}', encoding="utf-8")
    chosen = tmp_path / "chat_template.jinja"
    body = "jinja-body"
    chosen.write_text(body, encoding="utf-8")
    monkeypatch.setitem(CONFIG, "harness", {"component_manifest":
                                            {"model_dir": str(tmp_path)}})
    cm._template_cache.clear()

    got = cm.chat_template("primary", "http://127.0.0.1:8096/v1")

    assert got["id"] == f"sha256:{hashlib.sha256(body.encode()).hexdigest()}", got
    assert got["source"] == str(chosen), got


def test_the_recorded_chat_template_reaches_the_manifest_line(tmp_path, monkeypatch):
    """The digest is not just computed, it is written on the line — end to end.

    `chat_template()` returning a digest is half of it; clause 1 requires the field
    on every manifest line. One real recorded request here, with the template found
    through config, and the line has to carry that same digest: a line whose
    `chat_template.id` disagrees with the resolver's answer would send a reader to
    the wrong file when they go to replay it.
    """
    from app.config import CONFIG
    template = tmp_path / "chat_template.jinja"
    body = "replayed-template-body"
    template.write_text(body, encoding="utf-8")
    monkeypatch.setitem(CONFIG, "harness", {
        "component_manifest": {"chat_template_path": str(template)}})
    cm._template_cache.clear()

    line = cm.record_request(base_url="http://127.0.0.1:8096", model="primary",
                             session_id="20260919_120000_templ_s",
                             iteration=1,
                             send_site="tests/test_component_manifest.py",
                             payload={"model": "primary", "temperature": 0.7,
                                      "max_tokens": 8192, "stream": True,
                                      "messages": [{"role": "user",
                                                    "content": "go"}]})
    cm.flush(timeout=8.0)

    assert line["chat_template"] == {
        "id": f"sha256:{hashlib.sha256(body.encode()).hexdigest()}",
        "source": str(template)}, line["chat_template"]
