"""Clause 2 (#581): every non-streaming send site writes its own manifest line.

The triage named the non-streaming sites, and an earlier reading of this item
wrongly included two call sites that turn out to be Lloyd-to-Lloyd API calls
rather than model sends. These four are the real ones:

  app/harness/finalizer.py         the post-capture structured restate
  app/compaction_llm.py            the history summarizer
  app/inner_voice/observer.py      the second opinion on each turn event
  app/secondary_models.py          the routed background jobs

plus `app/harness/client.py::stream_chat`, which is every streaming turn and is
covered in `tests/test_component_manifest.py`.

Each test drives the module's own send function against a stub engine — no real
model, no token spend — and reads back what landed in the store.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import component_manifest as cm  # noqa: E402
from app import secondary_models  # noqa: E402
from app.compaction_llm import _post_chat_completion  # noqa: E402
from app.harness.finalizer import run_finalizer  # noqa: E402
from app.harness.loop import run_query  # noqa: E402
from app.harness.options import RunOptions  # noqa: E402
from app.inner_voice.observer import (  # noqa: E402
    _post_chat_completion_with_tools,
)

SCHEMA = {"type": "object", "properties": {"summary": {"type": "string"}}}
TOOLS = [{"type": "function",
          "function": {"name": "Bash", "description": "Run a command",
                       "parameters": {"type": "object"}}}]

SITES = {
    "finalizer": "app/harness/finalizer.py::run_finalizer",
    "compaction": "app/compaction_llm.py::_post_chat_completion",
    "observer": ("app/inner_voice/observer.py"
                 "::_post_chat_completion_with_tools"),
    "secondary": "app/secondary_models.py",
}


@pytest.fixture(autouse=True)
def _store(tmp_path, monkeypatch):
    """Own store, clean module state: the writer thread and memos are process-wide."""
    monkeypatch.setenv("LLOYD_MANIFEST_STORE", str(tmp_path / "manifest-store"))
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    cm.reset_stats()
    cm._reset_registry()
    cm._reset_template_cache()
    cm._reset_tools_memo()
    yield
    cm.flush(timeout=5.0)
    cm.reset_stats()
    cm._reset_registry()
    cm._reset_template_cache()
    cm._reset_tools_memo()


def _lines() -> list[dict]:
    root = Path(os.environ["LLOYD_MANIFEST_STORE"]) / "manifests"
    out: list[dict] = []
    if root.is_dir():
        for path in sorted(root.glob("*.ndjson")):
            out += [json.loads(r) for r in
                    path.read_text(encoding="utf-8").splitlines() if r.strip()]
    return out


def _body(content: str = "{\"summary\": \"restated\"}") -> dict:
    return {"choices": [{"message": {"content": content},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 2}}


class _Resp:
    def __init__(self, payload: dict, status: int = 200):
        self.status_code = status
        self.text = json.dumps(payload)
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _PostClient:
    """Stands in for `httpx.AsyncClient`, recording every payload it is handed."""

    posted: "list[dict]" = []
    responses: "list[_Resp]" = []

    def __init__(self, **_kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def post(self, _url, json=None, **_kw):
        _PostClient.posted.append(json)
        return _PostClient.responses.pop(0) if _PostClient.responses else _Resp(_body())


def _fake_urlopen(holder: list):
    """A `urllib.request.urlopen` stand-in for secondary_models' sync posts."""

    class _R:
        status = 200

        def __init__(self, request, timeout=None):
            holder.append(json.loads(request.data.decode("utf-8")))

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {
                "content": '{"title":"a name","tags":[],"summary":"s","facts":[],'
                           '"candidates":[],"focus":[]}'}}]}).encode()

    return _R


@pytest.fixture
def stub_engine(monkeypatch):
    """Every async engine call goes to `_PostClient`, every sync one to urlopen."""
    _PostClient.posted, _PostClient.responses = [], []
    monkeypatch.setattr("app.harness.finalizer.httpx.AsyncClient", _PostClient)
    holder: list = []
    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen(holder))
    return holder


def test_all_four_non_streaming_send_sites_write_their_own_line(stub_engine,
                                                       monkeypatch):
    """Clause 2: each site emits its own manifest line, tagged with its source.

    A manifest on the streaming path only would describe chat and say nothing about
    the three sites that reach the primary outside a turn plus the routed jobs — and
    those are the requests whose cost #551 is about.
    """
    # The engine table the sites resolve against: the same base_urls the real config
    # carries, so a slot named below is a slot a real request would be named too.
    from app.config import CONFIG
    monkeypatch.setitem(CONFIG, "models", {
        "primary": {"base_url": "http://127.0.0.1:8096", "engine": "vllm"},
        "secondary": {"base_url": "http://127.0.0.1:8097", "engine": "llama.cpp"},
    })

    posted_before = 0

    asyncio.run(run_finalizer(base_url="http://127.0.0.1:8096", model="fin-model",
                              chat_messages=[{"role": "user", "content": "go"}],
                              tools=TOOLS, schema=SCHEMA, session_id="sess-fin"))
    assert len(_PostClient.posted) > posted_before, "finalizer sent nothing"
    asyncio.run(_post_chat_completion("http://127.0.0.1:8096", "cmp-model",
                                      [{"role": "user", "content": "summarize"}],
                                      max_tokens=200, timeout_seconds=5.0))
    asyncio.run(_post_chat_completion_with_tools(
        base_url="http://127.0.0.1:8096", model_name="obs-model",
        system_prompt="judge this", user_prompt="the event", tools=TOOLS,
        max_tokens=200, timeout_seconds=5.0))
    secondary_models._sync_secondary_title("a transcript worth naming", timeout=5.0)
    cm.flush(timeout=8.0)

    lines = _lines()
    seen = {line["send_site"] for line in lines}
    for key, site in SITES.items():
        assert site in seen, f"{key} wrote no manifest line; saw {sorted(seen)}"
    for line in lines:
        assert line["model"], f"{line['send_site']} recorded no model id"
        assert line["request_id"], f"{line['send_site']} recorded no request id"
        assert line["messages"] and all(
            m["sha256"].startswith("sha256:") for m in line["messages"]), line["send_site"]
        assert "provider" in line and "chat_template" in line, line["send_site"]
        # The slot must actually resolve, not merely exist: a line whose provider
        # reads `unrecorded` says only that the writer ran. These four sites are
        # driven with the base_url their real callers pass — config's
        # `models.*.base_url`, no `/v1`, which each site appends itself — so a slot
        # that resolves to `unrecorded` here means the site handed the manifest a
        # URL other than the one it is about to POST to, and a mis-resolved slot is
        # precisely the error this field exists to make visible.
        # `title` is a job the router pins to the primary, so all four of these
        # requests go to the same engine under this config; the expectation is read
        # from the routing table rather than hard-coded, so flipping the routing
        # moves the expectation instead of making this assertion wrong.
        want = ("primary" if "title" in secondary_models.JOBS_ON_PRIMARY
                else "secondary") if line["send_site"] == SITES["secondary"] else "primary"
        assert line["provider"]["slot"] == want, (
            f"{line['send_site']} recorded slot {line['provider']['slot']!r} for a "
            f"request to the {want}; source {line['provider']['source']}")
        # Every line says who sent it, and says it from the session id on that
        # same line rather than from the module that sent it: `compaction` and
        # `observer` run for both a chat and a worker, so the send site cannot
        # answer the question the diff asks ("did a user turn or a background job
        # take this cache miss?"). Both shapes are pinned by name below.
        assert line["source"] == cm.source_of_session(line["session_id"]), (
            f"{line['send_site']} tagged {line['source']!r} for session "
            f"{line['session_id']!r}")


def test_the_source_tag_separates_a_chat_turn_from_a_worker_run(stub_engine):
    """The two shapes of `source`, through a send site rather than the helper alone.

    A session id has three underscore-separated parts for a chat and four for a
    background run, and the tag is what lets a manifest be read per-source. Both
    branches go through the real finalizer here, because the field is only worth
    pinning where a line is actually written — a unit test of
    `source_of_session` would pass on a site that forgot to record it.
    """
    asyncio.run(run_finalizer(base_url="http://127.0.0.1:8096", model="m",
                              chat_messages=[{"role": "user", "content": "go"}],
                              tools=TOOLS, schema=SCHEMA,
                              session_id="20260919_120001_mission-control"))
    asyncio.run(run_finalizer(base_url="http://127.0.0.1:8096", model="m",
                              chat_messages=[{"role": "user", "content": "go"}],
                              tools=TOOLS, schema=SCHEMA,
                              session_id="20260919_120001_autocode_9f2a"))
    cm.flush(timeout=8.0)

    lines = [ln for ln in _lines() if ln["send_site"] == SITES["finalizer"]]
    assert len(lines) == 2, [ln["session_id"] for ln in lines]
    by_session = {ln["session_id"]: ln for ln in lines}
    assert len(by_session) == 2, "the two session ids collapsed into one line"
    assert by_session["20260919_120001_mission-control"]["source"] == "chat", (
        "a three-part chat id must tag `chat`, which is how a reader filters the "
        "user's own turns out of the store")
    assert by_session["20260919_120001_autocode_9f2a"]["source"] == "autocode", (
        "a four-part id must tag the worker source named by its third part, which "
        "is what attributes a background request to the job that made it")


def test_the_finalizer_line_carries_the_tools_array_the_loop_handed_it(stub_engine):
    """Clause 2's checkable half: the `tools` hash equals the array it was given.

    The finalizer's own comment measures that the shared prefix collapses when
    `tools` is omitted, so its prefix behaviour is entirely a question about this
    array — and `RunOptions.visible_tools_capture` is how the loop hands over the
    exact list it sent, mid-turn changes included. The manifest must hash that
    array, not a rebuild of it.
    """
    captured: list = list(TOOLS)

    asyncio.run(run_finalizer(base_url="http://127.0.0.1:8096", model="m",
                              chat_messages=[{"role": "user", "content": "go"}],
                              tools=captured, schema=SCHEMA, session_id="sess-fin"))
    cm.flush(timeout=8.0)
    assert len(_PostClient.posted) >= 1, _PostClient.posted
    lines = [ln for ln in _lines() if ln["send_site"] == SITES["finalizer"]]
    assert len(lines) == 1, [ln["send_site"] for ln in _lines()]
    assert lines[0]["tools"]["sha256"] == cm.digest_obj(captured)
    assert lines[0]["tools"]["definitions"][0]["name"] == "Bash"
    assert lines[0]["session_id"] == "sess-fin", (
        "without the session this request cannot be diffed against the turn it "
        "finalized, which is the comparison the item exists to make")
    assert captured == TOOLS, "the array the caller holds must be the one hashed"


def test_a_send_still_goes_out_when_the_manifest_writer_is_broken(stub_engine,
                                                                  monkeypatch):
    """A manifest failure may not eat a real request, at any site.

    `_post_chat_completion` returns the completion text on a real response, so a
    return value is proof the send happened; the counters are the module's own and
    are what an operator reads.
    """
    _PostClient.responses = [_Resp(_body("SUMMARY")), _Resp(_body())]

    def _boom(*_a, **_kw):
        raise OSError("device full")

    monkeypatch.setattr(cm, "_append_line", _boom)
    summary = asyncio.run(_post_chat_completion(
        "http://127.0.0.1:8096", "cmp-model",
        [{"role": "user", "content": "summarize"}],
        max_tokens=200, timeout_seconds=5.0))
    assert summary, "the compaction request failed because the manifest did"

    obj, err, _usage = asyncio.run(run_finalizer(
        base_url="http://127.0.0.1:8096", model="m",
        chat_messages=[{"role": "user", "content": "go"}],
        tools=TOOLS, schema=SCHEMA, session_id="sess-fin"))
    assert obj is not None, err
    assert _PostClient.posted, "the finalizer sent nothing"
    cm.flush(timeout=8.0)
    assert cm.stats()["recorded"] >= 2, cm.stats()
    assert cm.stats()["write_errors"] >= 2, cm.stats()
    assert _lines() == [], "a failed append must not leave a partial line behind"


def test_a_routed_job_names_the_engine_it_was_answered_by(stub_engine, monkeypatch):
    """The routed jobs and the chat engine must be tellable apart in the store.

    #551 is an item about measuring the secondary on the work it is routed to, and
    it starts from this store: a line that cannot say which engine answered cannot
    answer it. The expectation is derived from the routing config rather than
    hard-coded, so a flipped job does not make this test lie.
    """
    from app.config import CONFIG

    monkeypatch.setitem(CONFIG, "models", {
        "primary": {"base_url": "http://127.0.0.1:8096", "engine": "vllm"},
        "secondary": {"base_url": "http://127.0.0.1:8097", "engine": "llama.cpp"},
    })
    expected = ("primary" if "title" in secondary_models.JOBS_ON_PRIMARY
                else "secondary")
    secondary_models._sync_secondary_title("a transcript worth naming", timeout=5.0)
    cm.flush(timeout=8.0)
    assert stub_engine, "no HTTP request was attempted at all"
    lines = [ln for ln in _lines() if ln["send_site"] == SITES["secondary"]]
    assert len(lines) == 1, lines
    assert lines[0]["provider"]["slot"] == expected, lines[0]["provider"]


# ── the array the caller captures is the array the finalizer sends ───────────

def _loop_module_helpers():
    """The stub engine and pool from the streaming test, reused rather than cloned.

    `test_component_manifest.py` already owns a stub that speaks both the SSE
    streaming protocol and the finalizer's POST; two stubs for one protocol pair is
    how one of them drifts. Test modules sit next to each other without a package,
    hence the load-by-path.
    """
    import importlib.util

    path = Path(__file__).with_name("test_component_manifest.py")
    spec = importlib.util.spec_from_file_location("_cm_stream_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_finalizer_tools_hash_is_the_array_the_caller_captured(monkeypatch):
    """Clause 2's second half, across the loop/finalizer seam.

    The caller reads the advertised set back through `RunOptions.visible_tools_capture`
    (#529) and the loop passes that same set to the finalizer — which is the whole
    point, because `finalizer.py:125-137` measured that the shared prefix is only 41
    tokens when the array is omitted. So the `tools` digest on the finalizer's manifest
    line must equal the digest of what the caller captured; otherwise the record
    describes an array nobody sent and a finalizer prefix break gets misattributed to a
    component that did not move.
    """
    import asyncio

    import app.harness.finalizer as fin_module

    helpers = _loop_module_helpers()
    engine = helpers.Engine
    engine.posted, engine.streamed, engine.script = [], [], []
    engine.stream_calls, engine.on_request = 0, None
    # The finalizer reaches httpx through its own module-level import, so the
    # module object is what gets replaced, not the attribute on the real httpx.
    class _HttpxShim:
        AsyncClient = engine
        Timeout = staticmethod(lambda *a, **k: None)

    monkeypatch.setattr(fin_module, "httpx", _HttpxShim)
    monkeypatch.setattr("app.harness.client.httpx.AsyncClient", engine)

    def _sse(*objs):
        return ["data: " + json.dumps(o) + "\n\n" for o in objs] + ["data: [DONE]\n\n"]

    engine.script = [
        _sse({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1",
                    "type": "function", "function": {"name": "Bash",
                                                     "arguments": "{}"}}]}}],
              "usage": {"prompt_tokens": 5, "completion_tokens": 1}},
             {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
        _sse({"choices": [{"delta": {"content": "all done"}}],
              "usage": {"prompt_tokens": 6, "completion_tokens": 2}},
             {"choices": [{"delta": {}, "finish_reason": "stop"}]}),
    ]
    captured: list = []
    options = RunOptions(model="primary", session_id="fin-seam",
                         final_schema=SCHEMA, visible_tools_capture=captured)
    messages = [{"role": "system", "content": "SYS"},
                {"role": "user", "content": "go"}]

    async def _drive():
        async for _ in run_query(messages, options):
            pass

    asyncio.run(_drive())
    cm.flush(timeout=8.0)
    assert captured, "the loop captured no visible tools, so nothing is being compared"
    fin = [ln for ln in _lines()
           if ln["send_site"] == "app/harness/finalizer.py::run_finalizer"]
    assert len(fin) >= 1, [ln["send_site"] for ln in _lines()]
    assert fin[-1]["tools"]["sha256"] == cm.digest_obj(captured), (
        "the finalizer's recorded tools hash is not the array the caller captured")
