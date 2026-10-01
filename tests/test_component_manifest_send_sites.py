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
import collections
import hashlib
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


# ── #1782: the two sites that injected a prompt and reported nothing ─────────
#
# Both nodes drive the real send function against the same stubs the four-site
# test above uses, and both compare the digest on the line against `hashlib` over
# the system message the stub captured — the bytes that were about to go on the
# wire. Deriving the expectation from the manifest module's own helper would let
# the two drift together, which is the one thing a digest test must not allow.


def _sha(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _system_message(payload: dict) -> str:
    for msg in payload.get("messages") or []:
        if msg.get("role") == "system":
            return str(msg.get("content") or "")
    raise AssertionError(f"the stub captured no system message: {payload}")


def test_the_observer_site_records_its_own_system_prompt(stub_engine, monkeypatch):
    """#1782 clause 4: the observer line is no longer `unrecorded`.

    2,346 of the day's 3,837 unrecorded lines on 2026-09-28 — 61% of them — came
    from this one function, every one of them carrying a real system prompt that
    the manifest had no way to name: the component registry is keyed on sessions,
    and an inner-voice judgement has no session. The site now describes what it
    composed, so the line names the component and the diff can ask which prompt
    moved.
    """
    system = "judge this event against the goal card"
    asyncio.run(_post_chat_completion_with_tools(
        base_url="http://127.0.0.1:8096", model_name="obs-model",
        system_prompt=system, user_prompt="the event", tools=TOOLS,
        max_tokens=200, timeout_seconds=5.0))
    cm.flush(timeout=8.0)

    lines = [ln for ln in _lines()
             if ln["send_site"] == SITES["observer"]]
    assert len(lines) == 1, lines
    line = lines[0]
    assert line["components_captured"] != "unrecorded", line
    assert line["components_captured"] == cm._SEND_SITE, (
        "a site that describes itself must not be reported as a turn-boundary "
        f"handoff; got {line['components_captured']!r}")
    by_name = {c["name"]: c for c in line["components"]}
    assert set(by_name) == {"system_prompt"}, line["components"]
    assert by_name["system_prompt"]["sha256"] == _sha(_system_message(_PostClient.posted[-1])), (
        "the digest on the line is not the system prompt the stub was sent")
    assert by_name["system_prompt"]["bytes"] == len(system.encode("utf-8"))


def test_the_routed_jobs_record_their_system_prompt(stub_engine, monkeypatch):
    """#1782 clause 5: the five secondary jobs are no longer 100% unrecorded.

    The smaller half of the same defect — 245 lines on 2026-09-28, all of them
    unrecorded, because the five routed jobs are handed a transcript and have no
    session id at any level of their call chain: `_sync_secondary_title(transcript)`
    is the whole signature. Threading a session through four call sites to fill a
    session-keyed table with rows no session owns would be the wrong fix; naming
    the prompt at the post is the right one.
    """
    from app.config import CONFIG
    monkeypatch.setitem(CONFIG, "models", {
        "primary": {"base_url": "http://127.0.0.1:8096", "engine": "vllm"},
        "secondary": {"base_url": "http://127.0.0.1:8097", "engine": "llama.cpp"},
    })
    before = len(_lines())
    title = secondary_models._sync_secondary_title("a transcript worth naming",
                                                   timeout=5.0)
    cm.flush(timeout=8.0)

    assert title, "the job did not run, so the line below proves nothing"
    lines = [ln for ln in _lines()[before:]
             if ln["send_site"] == SITES["secondary"]]
    assert len(lines) == 1, lines
    line = lines[0]
    assert line["components_captured"] != "unrecorded", (
        f"the job's own line still reads unrecorded: {line['components_captured']!r}")
    assert line["components_captured"] == cm._SEND_SITE, line
    by_name = {c["name"]: c for c in line["components"]}
    assert set(by_name) == {"system_prompt"}, line["components"]
    assert by_name["system_prompt"]["sha256"] == _sha(secondary_models._TITLE_SYSTEM), (
        "the recorded digest is not the title job's own system prompt")
    assert by_name["system_prompt"]["bytes"] == len(
        secondary_models._TITLE_SYSTEM.encode("utf-8"))


def test_a_digest_pair_handed_over_inline_reads_as_the_same_row_as_the_text():
    """#1880 clause 3: one reader, two shapes, this path included.

    `note_components` digests the components at note time now, so what the
    registry hands `_build_line` is a `{sha256, bytes}` pair; a send site has no
    entry to be read from and still hands over the text it is injecting on this
    call. The reader has to serve both without the row showing which arrived —
    which is the same dual shape the prefetch block has been through since
    #1782, and the reason this file's four sites keep passing after a change
    made only to the registry.

    Both shapes go through `record_request`, the function the sites named at the
    top of this file actually call, carrying the observer's own `send_site`
    string, and the arrays are compared as they landed in the store. The
    expectation is `hashlib` over the literal by way of `_sha`, never the
    module's own digest helper.
    """
    text = "judge this event against the goal card\n"
    pair = {"sha256": _sha(text), "bytes": len(text.encode("utf-8"))}

    assert cm.record_request(base_url="http://127.0.0.1:8096", model="dual-text",
                             payload={"model": "m", "messages": [
                                 {"role": "user", "content": "go"}]},
                             send_site=SITES["observer"],
                             components={"system_prompt": text})
    assert cm.record_request(base_url="http://127.0.0.1:8096", model="dual-pair",
                             payload={"model": "m", "messages": [
                                 {"role": "user", "content": "go"}]},
                             send_site=SITES["observer"],
                             components={"system_prompt": pair})
    cm.flush(timeout=8.0)

    by_model = {ln["model"]: ln for ln in _lines()
                if ln["send_site"] == SITES["observer"]}
    assert {"dual-text", "dual-pair"} <= set(by_model), sorted(by_model)
    assert by_model["dual-text"]["components"] == by_model["dual-pair"]["components"], (
        "the row depends on which shape arrived: "
        f"{by_model['dual-text']['components']} vs {by_model['dual-pair']['components']}")
    assert by_model["dual-text"]["components"] == [
        {"name": "system_prompt", "sha256": _sha(text),
         "bytes": len(text.encode("utf-8"))}], by_model["dual-text"]["components"]
    for model in ("dual-text", "dual-pair"):
        assert by_model[model]["components_captured"] == cm._SEND_SITE, (
            f"{model}: a dict handed over inline is this call's own answer, "
            "however it was digested")


# ── #1879: the senders whose system prompt was built with no session key ─────
#
# Every node above reaches the manifest through a site that describes itself.
# The residual this section closes is the opposite shape, and it is the larger
# one: 359 of the 2,626 `app/harness/client.py::stream_chat` lines written on
# 2026-09-30 — 13.7% of that send site's lines — read `components_captured:
# "unrecorded"`, and the 29 distinct session ids carrying those 359 lines
# emitted no `turn_start` line whatsoever that day (13 autonomy, 7 benchmine, 8
# bench, 1 sessiondistill). Those are exactly the sessions the nightly
# prompt-diff consumer reviews and cannot see. The 29 ids and their 359 lines are
# committed at `backlog/data/2026-09-30.stream-chat-residual.ndjson`;
# `scripts/maintenance/components_unrecorded_tally.py <day>` regenerates the same
# report for any day the live store still holds, which is the check the item
# originally named as a /tmp script.
#
# One guard is the cause: `note_components` runs only inside the
# `if session_id:` block at `app/prompt_builder.py:586`, and the four callers
# below built their system prompt without passing the session id they already
# held — so nothing was ever keyed for the session that was about to send. The
# three sites that hold a session id are now threaded, and the two autoresearch
# bench sites, whose send is a bare `requests.post` with no session at any level
# of their chain, describe their prompt at the post the way `observer.py` and
# `secondary_models.py` do.
#
# The threaded nodes are driven end to end on purpose. The registry handoff
# between `prompt_builder` (which writes) and `client.py::stream_chat` (which
# reads) is the one boundary in this mechanism with no call chain across it, so
# a node that looked only at the registry would keep passing if the hook were
# renamed, and a node that called `record_request` by hand would never show a
# caller that forgot to thread its id.


STREAM_SITE = "app/harness/client.py::stream_chat"
BENCH_SEND_SITE = "scripts/autoresearch/bench_runner.py::chat_completion"


def _sent_system_message(payload: dict) -> str:
    """The system message the stub engine was handed — the bytes about to go out."""
    for msg in (payload or {}).get("messages") or []:
        if msg.get("role") == "system":
            return str(msg.get("content") or "")
    raise AssertionError(f"the stub captured no system message: {payload}")


def _stream_lines(session_id: str) -> list[dict]:
    return [ln for ln in _lines()
            if ln["send_site"] == STREAM_SITE and ln["session_id"] == session_id]


def _first_stream_line(session_id: str) -> dict:
    lines = _stream_lines(session_id)
    assert lines, (
        f"no {STREAM_SITE} line for session {session_id!r} in "
        f"{[ln['send_site'] for ln in _lines()]}")
    return min(lines, key=lambda ln: (ln["iteration"] is None,
                                      ln["iteration"] or 0))


def _assert_records_the_sent_prompt(line: dict, system_text: str) -> None:
    """The line names the components, and each digest is the sent bytes.

    The expectation is `hashlib` over the system message the stub captured,
    re-sliced by the byte counts the line itself carries — never
    `cm.digest_text`, which would pass while both sides drifted. A caller that
    appends its own text after `build_system_prompt` returns (the worker path
    adds the denied-tools block, the SDK path a planted policy) is allowed: what
    is pinned is that every recorded digest is the digest of the next bytes of
    the prompt on the wire, in order, from the first byte.
    """
    assert line["components_captured"] == cm._TURN_START, (
        f"{line['send_site']} captured {line['components_captured']!r}, not "
        f"{cm._TURN_START!r}: the session's components never reached the registry")
    assert line["components"], "the line carries no component list at all"
    raw = system_text.encode("utf-8")
    pos = 0
    for row in line["components"]:
        assert row["sha256"].startswith("sha256:") and row["bytes"] > 0, row
        piece = raw[pos:pos + row["bytes"]]
        assert piece and ("sha256:" + hashlib.sha256(piece).hexdigest()
                          == row["sha256"]), (
            f"{row['name']}: the recorded digest is not the digest of the bytes "
            f"the engine was sent at offset {pos}")
        pos += row["bytes"] + len(b"\n\n")          # components joined by "\n\n"
    assert pos - 2 <= len(raw), (
        f"the recorded components account for {pos - 2} bytes, more than the "
        f"{len(raw)} bytes actually sent")


_DONE_LINES = [
    "data: " + json.dumps({"choices": [{"delta": {"content": "all done"}}]}) + "\n\n",
    "data: " + json.dumps({"choices": [{"delta": {},
                                        "finish_reason": "stop"}]}) + "\n\n",
    "data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": 11,
                                                    "completion_tokens": 2}}) + "\n\n",
    "data: [DONE]\n\n",
]


class _StreamResp:
    def __init__(self, lines: list[str]):
        self.status_code = 200
        self._lines = lines

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self) -> bytes:
        return b""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False


class _Engine:
    """`httpx.AsyncClient` at the socket: records every payload, streams `script`.

    `stream_chat` builds its own URL, headers and body against this, so the
    payload in `.streamed` is the one that would have gone to the engine, and the
    manifest line written beside it is the code under test's own answer.
    """

    streamed: "list[dict]" = []
    posted: "list[dict]" = []
    script: "list[list[str]]" = []

    def __init__(self, **_kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    def stream(self, _method, _url, headers=None, json=None, **_kw):
        _Engine.streamed.append(json)
        idx = len(_Engine.streamed) - 1
        return _StreamResp(_Engine.script[idx] if idx < len(_Engine.script)
                           else _DONE_LINES)

    async def post(self, _url, json=None, **_kw):
        _Engine.posted.append(json)
        return _Resp(_body())


class _Pool:
    @property
    def discovered(self):
        return [("lloyd-mcp", [{"name": "Bash", "description": "shell",
                                "inputSchema": {"type": "object",
                                                "properties": {}}}])]

    async def call_tool(self, name, args, *, session_id="", **_kw):
        return {"content": "FAKE_RESULT", "is_error": False}


def _install_engine_stub(monkeypatch, turns: int = 1):
    """Real loop, real `stream_chat`, fake socket. Returns the recording engine.

    `.streamed` is the positive control every node below reads: one payload there
    beside one line in the store is what makes the line evidence rather than a
    leftover.
    """
    _Engine.posted, _Engine.streamed = [], []
    _Engine.script = [list(_DONE_LINES) for _ in range(turns)]

    pool = _Pool()

    async def _build_pool(_options):
        return pool

    monkeypatch.setattr("app.harness.loop._build_pool", _build_pool)
    monkeypatch.setattr("app.harness.client.httpx.AsyncClient", _Engine)
    monkeypatch.setattr("app.harness.finalizer.httpx.AsyncClient", _Engine)
    return _Engine


def _drive_options(engine, options, prompt: str = "go") -> dict:
    """Send `options` through the real loop and return the payload the engine saw."""
    from app.harness.loop import run_query as real_run_query

    async def _drain():
        return [e async for e in real_run_query(
            [{"role": "user", "content": prompt}], options)]

    asyncio.run(_drain())
    cm.flush(timeout=8.0)
    assert engine.streamed, (
        "no request crossed the seam, so a manifest line here would be evidence "
        "of nothing")
    return engine.streamed[-1]


# ── the autonomy path (clause 1) ────────────────────────────────────────────

@pytest.fixture
def autonomy_env(tmp_path, monkeypatch):
    """Isolate what `autonomy.run_task` touches outside the turn it builds.

    The same seam set `tests/test_autonomy_turn_budget.py` drives the real
    `run_task` with: task files, session and event directories, and the
    `config.yaml` the global budget is read out of. Left real here because the
    clause is about what `run_task` itself does with the id it mints: the real
    prompt builder, the real loop, the real `stream_chat`.
    """
    from app import autonomy

    (tmp_path / "config.yaml").write_text(
        "agent:\n  max_turns: 60\nmodel:\n  default: primary\n")
    monkeypatch.setattr(autonomy, "LLOYD_HOME", tmp_path)
    monkeypatch.setattr("app.sessions_io.SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr("app.event_log.EVENT_LOGS_DIR", tmp_path / "events")
    monkeypatch.setattr("app.event_log.BLOBS_DIR", tmp_path / "events" / "blobs")
    monkeypatch.setenv("LLOYD_GRANT_DB", str(tmp_path / "grants.db"))
    monkeypatch.setattr(autonomy, "_find_task_file", lambda tid: tmp_path / "x.md")
    monkeypatch.setattr(autonomy, "_parse_task_file", lambda p: {
        "id": 24, "name": "Task 24", "skill_name": "s", "status": "up_next",
        "timeout_seconds": 300, "description": "record the prompt"})
    monkeypatch.setattr(autonomy, "_load_skill_content", lambda s: "SKILL")
    monkeypatch.setattr(autonomy, "_update_task_field", lambda *a, **k: None)
    monkeypatch.setattr(autonomy, "_append_activity_log", lambda *a, **k: None)
    monkeypatch.setattr(autonomy, "_get_model_env", lambda m: {})
    monkeypatch.setattr(autonomy, "_write_run_record", lambda **kw: None)
    monkeypatch.setattr(autonomy, "_task_inner_voice", lambda t: False)
    monkeypatch.setattr("app.mcp_discovery._get_disallowed_tools",
                        lambda *a, **k: [])
    monkeypatch.setattr("app.mcp_discovery._get_harness_kwargs", lambda: {})
    monkeypatch.setattr("app.harness.mcp_pool.DEFAULT_LLOYD_MCP_SERVERS", {},
                        raising=False)
    return tmp_path


def test_an_autonomy_run_records_turn_start_for_the_session_it_mints(
        autonomy_env, monkeypatch):
    """Clause 1: `app/autonomy.py:4011` passes the run's `session_id`.

    `run_task` mints the id at `app/autonomy.py:3920` and already hands it to
    `RunOptions`, so the streaming line has always carried a session — it was
    the prompt build that never learned of it, and 13 of the 29 unrecorded
    sessions on 2026-09-30 were autonomy runs. This drives `run_task` for real
    and reads its first `stream_chat` line back off the store.
    """
    from app import autonomy
    import app.harness as harness

    engine = _install_engine_stub(monkeypatch)
    captured: dict = {}
    from app.harness.loop import run_query as real_run_query

    async def _run_query(messages, options):
        captured["options"] = options
        async for evt in real_run_query(messages, options):
            yield evt

    monkeypatch.setattr(harness, "run_query", _run_query)
    asyncio.run(autonomy.run_task(24))
    cm.flush(timeout=8.0)

    options = captured["options"]
    assert getattr(options, "session_id", ""), (
        "the run built its options with no session id, so there is no session to "
        "attribute a manifest line to")
    assert engine.streamed, (
        "the autonomy turn never crossed the seam, so its manifest line would be "
        "evidence of nothing")
    first = _first_stream_line(options.session_id)
    _assert_records_the_sent_prompt(
        first, _sent_system_message(engine.streamed[0]))
    assert cm.registry_occupancy() == 1, (
        "the run's prompt was noted for more than one session key")


# ── the worker path (clause 2) ──────────────────────────────────────────────

def _worker_env(monkeypatch, tmp_path):
    """Isolate the files a background turn writes, and nothing else.

    The one helper every in-process worker turn builds its options through is
    `_worker_run_options`, so this drives the real path once and the fix covers
    each source that shares it — the measured residual's `benchmine`, `bench`
    and `sessiondistill` sessions among them, including the `sessiondisti…`
    session the filing's own source list did not name.
    """
    from app import autonomy

    monkeypatch.setattr("app.sessions_io.SESSIONS_DIR", tmp_path / "sessions")
    monkeypatch.setattr("app.event_log.EVENT_LOGS_DIR", tmp_path / "events")
    monkeypatch.setattr("app.event_log.BLOBS_DIR", tmp_path / "events" / "blobs")
    monkeypatch.setenv("LLOYD_GRANT_DB", str(tmp_path / "grants.db"))
    monkeypatch.setattr(autonomy, "_get_model_env", lambda m: {})
    monkeypatch.setattr("app.routers.automod.drain_active", lambda: False)
    return tmp_path


def test_a_worker_turn_records_turn_start_for_the_session_it_mints(monkeypatch, tmp_path):
    """Clause 2: `_worker_run_options` passes the `session_id` it is handed.

    `run_prompt_on_primary` mints the session id, passes it to the options
    builder for the scratchpad anchor and the budget anchor, and then sends — and
    the prompt build beside those still had no id, which is how 8 benchmine and 1
    sessiondistill sessions landed in the 2026-09-30 residual. One kwarg on that
    one call covers every source that builds through the helper.
    """
    from workers.sources import _common as wc

    _worker_env(monkeypatch, tmp_path)
    engine = _install_engine_stub(monkeypatch)

    result = asyncio.run(wc.run_prompt_on_primary(
        "prove the prompt was recorded", max_turns=2, source="benchmine"))
    cm.flush(timeout=8.0)

    assert result.session_id, "the worker turn minted no session to attribute"
    assert engine.streamed, (
        "the worker turn never crossed the seam, so its manifest line would be "
        "evidence of nothing")
    first = _first_stream_line(result.session_id)
    _assert_records_the_sent_prompt(first, _sent_system_message(engine.streamed[0]))
    assert cm.registry_occupancy() == 1, (
        "the worker's prompt was noted for more than one session key")


# ── the autoresearch harness-routed path (clause 3) ─────────────────────────

def test_an_autoresearch_trial_records_turn_start_for_its_session(
        monkeypatch, tmp_path):
    """Clause 3: `build_options` passes the `session_id` it already takes.

    The SDK runner takes the trial's session id as a keyword argument and hands
    it to `RunOptions` two lines below the prompt build, so the trial's lines
    always named a session whose prompt the manifest had never seen — 8 `bench`
    sessions of the 359 unrecorded lines. The overlay dir is the runner's own
    knob for the prompt, so the build is deterministic without reading the vault.
    """
    from scripts.autoresearch import bench_runner_sdk as sdk

    monkeypatch.setattr("app.mcp_discovery._get_disallowed_tools",
                        lambda *a, **k: [])
    monkeypatch.setattr("app.mcp_discovery._get_harness_kwargs", lambda: {})
    monkeypatch.setattr("app.mcp_discovery._get_mcp_servers", lambda: {})
    engine = _install_engine_stub(monkeypatch)

    sid = "20260930_070000_bench_1879aa"
    options = sdk.build_options(
        model="primary", overlay_dir=_overlay_dir(tmp_path), session_id=sid,
        max_agent_turns=2, sandbox_stateful_tools=False)
    assert options.session_id == sid, "the trial's options lost the id"

    _drive_options(engine, options, prompt="answer the bench task")
    first = _first_stream_line(sid)
    _assert_records_the_sent_prompt(first, _sent_system_message(engine.streamed[0]))
    assert cm.registry_occupancy() == 1, (
        "the trial's prompt was noted for more than one session key")


def _overlay_dir(tmp_path: Path) -> Path:
    """An overlay carrying its own SOUL.md, so no vault read is needed to build."""
    overlay = tmp_path / "prompts"
    overlay.mkdir(parents=True, exist_ok=True)
    (overlay / "SOUL.md").write_text(
        "# SOUL\n\nSOUL-SENTINEL identity body for the bench manifest test.\n",
        encoding="utf-8")
    return overlay


# ── the two direct autoresearch sites (clause 4) ────────────────────────────

def _assert_names_the_sent_prompt(line: dict, system_text: str) -> None:
    """The inline path: one component, named at the send, digesting the sent text.

    A site with no session key cannot use the registry, so its answer is handed
    over inline as the payload is built — the `observer.py` / `secondary_models.py`
    shape. The expected digest is `hashlib` over the system message the fake
    socket captured, never `cm.digest_text`.
    """
    assert line["components_captured"] == cm._SEND_SITE, (
        f"captured {line['components_captured']!r}, not {cm._SEND_SITE!r}")
    body = system_text.encode("utf-8")
    assert line["components"] == [{"name": "system_prompt",
                                   "sha256": "sha256:" + hashlib.sha256(body).hexdigest(),
                                   "bytes": len(body)}], line["components"]


def _stub_bench_socket(monkeypatch):
    """Stand in for the bench runner's `requests.post` and record the payloads."""
    import requests

    posted: list[dict] = []

    def _post(url, headers=None, json=None, timeout=None, **_kw):
        posted.append(json)
        return _Resp({"choices": [{"message": {"content": "bench answer"},
                                   "finish_reason": "stop"}],
                      "usage": {"completion_tokens": 4}})

    monkeypatch.setattr(requests, "post", _post)
    return posted


def test_a_direct_bench_trial_names_the_prompt_it_sends(monkeypatch, tmp_path):
    """Clause 4a: `scripts/autoresearch/bench_runner.py:163` has no session key.

    Its send is `chat_completion`'s bare `requests.post`, three levels above any
    session concept, so the prompt is named at the send instead of threaded down.
    Two facts are pinned: the line says which prompt went out, and the registry
    holds nothing — the inline path digests and forgets, so no raw context text
    stays resident. (This send does not pass through `stream_chat`, so it cannot
    move that site's residual; it is here for provenance symmetry, which is what
    the triage ruled these two sites down to.)
    """
    from scripts.autoresearch import bench_runner as br

    posted = _stub_bench_socket(monkeypatch)
    monkeypatch.setattr(br, "_endpoint_for", lambda m: "http://stub:8096")
    monkeypatch.setattr(br, "_resolved_model_name", lambda m: "stub-model")
    cm.reset_stats()

    trace = br._run_one_sync({"id": "t1", "prompt": "say the thing"},
                            "variant-a", _overlay_dir(tmp_path), "primary", 30)
    cm.flush(timeout=8.0)

    assert trace["status"] == "success", trace["error"]
    assert len(posted) == 1, posted
    lines = [ln for ln in _lines() if ln["send_site"] == BENCH_SEND_SITE]
    assert len(lines) == 1, (
        f"expected exactly one manifest line for the bench send, got {len(lines)}")
    _assert_names_the_sent_prompt(lines[0], _sent_system_message(posted[0]))
    assert cm.registry_occupancy() == 0, (
        "the bench send left its prompt resident in the registry; the inline "
        "path digests and forgets")
    assert "SOUL-SENTINEL" not in json.dumps(lines[0]), (
        "the written line carries raw context text, not a digest")


def test_a_strategy_arm_trial_names_the_prompt_of_every_call_it_sends(
        monkeypatch, tmp_path):
    """Clause 4b: `scripts/autoresearch/strategy_arms.py:193`, same send, same rule.

    The `advise` arm is the one that tests the fallback rather than letting it
    coast: it spends three calls through the shared `chat_completion` and they do
    **not** send the same prompt — the draft and the revision carry the
    overlay-built system prompt, the adviser call carries `ADVISER_SYSTEM`. So each
    manifest line is compared with the payload at its own index, and the adviser's
    digest is required to differ from the draft's: a fallback that digested one
    prompt for the whole trial would pass a node that compared every line to the
    first payload, and fails this one.
    """
    from scripts.autoresearch import bench_runner as br
    from scripts.autoresearch import strategy_arms as sa

    posted = _stub_bench_socket(monkeypatch)
    monkeypatch.setattr(br, "_endpoint_for", lambda m: "http://stub:8096")
    monkeypatch.setattr(br, "_resolved_model_name", lambda m: "stub-model")
    cm.reset_stats()

    trace = sa.run_arm_trial({"id": "t2", "prompt": "say the thing"}, "advise",
                             ceiling=2000, model="primary",
                             overlay_dir=_overlay_dir(tmp_path))
    cm.flush(timeout=8.0)

    assert trace["status"] == "success", trace["error"]
    assert len(posted) == 3, (
        f"the advise arm is three calls (draft, adviser, revise); the stub saw "
        f"{len(posted)}, so the pairing below would prove nothing")
    lines = [ln for ln in _lines() if ln["send_site"] == BENCH_SEND_SITE]
    assert len(lines) == len(posted), (
        f"{len(lines)} manifest lines for {len(posted)} bench calls")
    for line, payload in zip(lines, posted):
        _assert_names_the_sent_prompt(line, _sent_system_message(payload))
    digests = [ln["components"][0]["sha256"] for ln in lines]
    assert digests[0] == digests[2], (
        "the draft and the revision send the same system prompt, so their lines "
        "must agree on one digest")
    assert digests[1] != digests[0], (
        "the adviser call got the draft's digest: the fallback named the trial, "
        "not the request")
    assert digests[1] == "sha256:" + hashlib.sha256(
        sa.ADVISER_SYSTEM.encode("utf-8")).hexdigest(), (
        "the adviser line does not digest the prompt the adviser actually received")
    assert cm.registry_occupancy() == 0, (
        "the arm left its prompt resident in the registry")


# ── the chat and voice builders are unchanged (clause 5) ────────────────────

def test_the_chat_and_voice_builders_still_hand_their_session_to_the_prompt(
        monkeypatch, tmp_path):
    """Clause 5: `app/routers/turn_options.py:203` and `:211` still pass it.

    The two builders that were always correct are pinned from both sides, because
    this round edits the builder they call: the id reaches `build_system_prompt`
    for a spoken turn as well as a typed one, and a prompt built the way they
    build it really does come back as `turn_start` capture once it is sent.
    """
    from app.prompt_builder import build_system_prompt as real_build
    from app.routers import turn_options as topts

    seen: list[dict] = []

    def _record(**kwargs):
        seen.append(kwargs)
        return "CHAT SYSTEM PROMPT"

    monkeypatch.setattr(topts, "build_system_prompt", _record)
    monkeypatch.setattr(topts, "_memory_snapshot",
                        _FrozenMemories())
    monkeypatch.setattr(topts, "_prompt_layout", _Layout())
    monkeypatch.setattr(topts, "_get_model_env", lambda m: {})
    monkeypatch.setattr(topts, "_get_harness_kwargs", lambda: {})
    monkeypatch.setattr(topts, "_get_mcp_servers", lambda: {})
    monkeypatch.setattr(topts, "_get_disallowed_tools", lambda *a, **k: [])
    _stub_messages_helpers(monkeypatch)
    monkeypatch.setattr("app.routers.voice._voice_extra_body", lambda: {})

    for kind in ("stream", "voice"):
        seen.clear()
        build = topts.build_turn_options(
            topts.SessionSnapshot(session_id="chat-1879",
                                  path=tmp_path / "chat-1879.json"),
            {}, kind, text="say the thing")
        assert seen, f"kind={kind}: the builder never built a system prompt"
        assert seen[0].get("session_id") == "chat-1879", (
            f"kind={kind}: build_system_prompt got {seen[0].get('session_id')!r}")
        assert build.system_prompt == "CHAT SYSTEM PROMPT"
        assert build.options.system_prompt == "CHAT SYSTEM PROMPT"

    # And the capture behaviour itself: the real builder, called the way the chat
    # path calls it, then the real send — still turn_start, still with digests.
    from app import prompt_builder

    prompt_builder.build_system_prompt(session_id="chat-1879",
                                       overlay_dir=_overlay_dir(tmp_path))
    engine = _install_engine_stub(monkeypatch)
    _drive_options(engine, RunOptions(model="primary", session_id="chat-1879",
                                      tool_search_enabled=False,
                                      system_prompt=real_build(
                                          session_id="chat-1879",
                                          overlay_dir=_overlay_dir(tmp_path))))
    first = _first_stream_line("chat-1879")
    _assert_records_the_sent_prompt(first, _sent_system_message(engine.streamed[0]))


class _FrozenMemories:
    def frozen_memories(self, session_id, platform=""):
        return "", ""


class _Layout:
    def mem_kwargs(self, frozen_mem):
        return {}

    def turn_tail(self, todos, plan, goal, memory_note):
        return ""


def _stub_messages_helpers(monkeypatch):
    """Neutralise the authority/reviewer helpers, which are not this clause.

    They live on `app.routers.messages` and read grants, session metadata and
    the action-reviewer config — all of them choices about what a turn may do,
    none about how its prompt is recorded. Patching them keeps the assertion
    about `build_system_prompt`'s arguments, which is what the clause names.
    """
    from app.routers import messages as m

    monkeypatch.setattr(m, "_authority_scope_for",
                        lambda *a, **k: "", raising=False)
    monkeypatch.setattr(m, "_ban_grant_minting", lambda *a, **k: None,
                        raising=False)
    monkeypatch.setattr(m, "_ban_automod_for_workers", lambda *a, **k: None,
                        raising=False)
    monkeypatch.setattr(m, "_install_action_review", lambda *a, **k: None,
                        raising=False)
    monkeypatch.setattr(m, "_turn_budget", lambda *a, **k: 60, raising=False)
    monkeypatch.setattr(m, "_clamp_priority", lambda p, **k: 0, raising=False)
    monkeypatch.setattr(m, "_tool_surface", lambda p: "chat", raising=False)
    monkeypatch.setattr(m, "_final_schema_for", lambda *a, **k: None,
                        raising=False)
    monkeypatch.setattr(m, "_effect_scope_for", lambda *a, **k: "",
                        raising=False)


# ── the check itself, and the bytes it was measured on (clause 6) ─────────────

WITNESS = "backlog/data/2026-09-30.stream-chat-residual.ndjson"


def _vault_root() -> Path:
    """`LLOYD_OBSIDIAN_VAULT` if set, else `~/obsidian` — the same resolution
    `tests/board_presence.py` uses, so this node asks about the vault the rest of
    the suite is asking about."""
    raw = os.environ.get("LLOYD_OBSIDIAN_VAULT")
    return Path(raw).expanduser() if raw else Path.home() / "obsidian"


def test_the_committed_witness_bytes_rederive_the_numbers_the_item_quotes():
    """Clause 6: the figures #1879 quotes are a sum of rows that are on disk.

    The item's numbers were measured on a live, append-only, 14-day-retained store,
    which is why they had no history. They are now committed as a projection — one
    row per unrecorded session, one per send site — and this node re-derives every
    quoted figure from those bytes, so an edit to the witness that changes a number
    goes red. The day file itself is 107 MB of chat digests and prompt-part names
    and cannot go in a repo; #1880 projected for the same reason.
    """
    path = _vault_root() / WITNESS
    assert path.is_file(), (
        f"{path} is missing: the bytes #1879's acceptance was measured on are gone, "
        "and the item's figures have no witness")
    rows = [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]
    sessions = [r for r in rows if r["kind"] == "session"]
    sites = [r for r in rows if r["kind"] == "send_site"]

    assert len(rows) == 33, (
        f"`wc -l < {WITNESS}` must be the figure the item quotes, got {len(rows)}")
    assert len(sessions) == 29, (
        f"the residual was 29 distinct session ids, the witness has {len(sessions)}")
    assert sum(r["unrecorded_lines"] for r in sessions) == 359, (
        "the per-session rows no longer sum to the 359 lines the item quotes")
    assert all(r["turn_start_lines"] == 0 for r in sessions), (
        "a session in the witness emitted a turn_start line, which is the "
        "cross-check that made the residual unthreaded-site traffic rather than "
        "registry eviction")

    by_site = {r["send_site"]: r for r in sites}
    sc = by_site[STREAM_SITE]
    assert (sc["lines"], sc["unrecorded_lines"]) == (2626, 359), sc
    assert round(100.0 * sc["unrecorded_lines"] / sc["lines"], 1) == 13.7, (
        "the residual percentage the item quotes is not the one these bytes say")
    assert all(r["unrecorded_lines"] == 0 for site, r in by_site.items()
               if site != STREAM_SITE), (
        "a site that passes components= inline is carrying unrecorded lines in the "
        "witness, so 0% for the components sites is no longer what was measured")

    mix = collections.Counter(
        r["session_id"].split("_")[2] if len(r["session_id"].split("_")) >= 4
        else "chat"
        for r in sessions)
    assert dict(mix) == {"autonomy": 13, "bench": 8, "benchmine": 7,
                         "sessiondisti": 1}, (
        f"the source mix over the 29 ids is {dict(mix)}; the triage's line said 14 "
        "autonomy / 8 benchmine / 8 bench / 1 sessiondisti, which sums to 31")


def _load_tally():
    """Import `scripts/maintenance/components_unrecorded_tally.py` by path."""
    import importlib.util

    script = (Path(__file__).resolve().parent.parent
              / "scripts" / "maintenance" / "components_unrecorded_tally.py")
    spec = importlib.util.spec_from_file_location("cm_tally", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_residual_check_splits_an_absent_session_file_from_an_unthreaded_site(
        tmp_path, capsys):
    """The check has to answer the question the residual ruling turns on.

    A manifest line carries its own `session_id` — so asking the manifest whether
    that id "appears in the day" answers yes by construction and the file-absent
    bucket can never fill. This drives the script over a scratch store whose
    sessions dir knows three of four residual sessions, and requires the buckets to
    come out distinct.

    #1984: the fourth session has no file and nothing else, and that is ALL the
    check may say about it. It used to be named a `restart_boundary` on absence
    alone, which is a cause nobody had checked against any restart; it is
    `session_file_absent`, and `restart_boundary` stays empty without a
    process-death discriminant.
    """
    tally_mod = _load_tally()
    store = tmp_path / "manifests"
    store.mkdir()
    sessions = tmp_path / "sessions"
    sessions.mkdir()

    def _row(sid, captured, site=STREAM_SITE):
        return json.dumps({"ts": "2026-09-30T01:00:00+00:00", "session_id": sid,
                           "send_site": site, "components_captured": captured})

    known = ["20260929_010000_autonomy_aaaa", "20260929_010001_autonomy_bbbb",
             "20260929_010002_bench_cccc"]
    for sid in known:
        (sessions / f"{sid}.json").write_text("{}", encoding="utf-8")
    lines = ([_row(known[0], "unrecorded")] * 2 + [_row(known[1], "unrecorded")]
             + [_row(known[2], "turn_start"), _row(known[2], "unrecorded")]
             + [_row("20260929_010003_ghost_ffff", "unrecorded")]
             + [_row("", "turn_start", site="app/secondary_models.py")])
    (store / "2026-09-30.ndjson").write_text("\n".join(lines) + "\n",
                                             encoding="utf-8")

    rc = tally_mod.report(str(store), ["2026-09-30"], json_out=False,
                          sessions_dir=str(sessions))
    printed = capsys.readouterr().out
    assert rc == 0, printed
    assert "SESSION STORE UNREADABLE" not in printed, printed

    rep = tally_mod.tally(tally_mod.day_rows(str(store), "2026-09-30"),
                          session_ids=tally_mod.known_sessions(str(sessions)))
    assert rep["stream_chat"] == {"lines": 6, "unrecorded": 5,
                                  "pct": 100.0 * 5 / 6}, rep["stream_chat"]
    assert rep["attribution"] == {
        "registry_eviction": [known[2]],
        "trial_traffic": [],
        "unthreaded_send_site": sorted(known[:2]),
        "restart_boundary": [],
        "session_file_absent": ["20260929_010003_ghost_ffff"],
    }, rep["attribution"]
    # Every bucket line carries the day's unrecorded-session denominator (4 here).
    for kind in tally_mod.KINDS:
        want = f"   {kind}: {len(rep['attribution'][kind])}/4 unrecorded sessions"
        assert any(ln.startswith(want) for ln in printed.splitlines()), (kind, printed)

    # The zero-denominator rail: a window with lines and no stream_chat line must
    # not come back looking like a clean day.
    (store / "2026-09-29.ndjson").write_text(
        _row("", "turn_start", site="app/secondary_models.py") + "\n",
        encoding="utf-8")
    capsys.readouterr()
    rc = tally_mod.report(str(store), ["2026-09-29", "2026-01-01"], json_out=False,
                          sessions_dir=str(sessions))
    empty = capsys.readouterr().out
    assert rc == 2, (
        "a day with no stream_chat line and a day with no file both returned 0 — "
        "the check can report a clean result over nothing")
    assert "NO stream_chat LINE" in empty, empty


def _tally_fixture(tmp_path, rows, known=()):
    store = tmp_path / "manifests"
    store.mkdir()
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    for sid in known:
        (sessions / f"{sid}.json").write_text("{}", encoding="utf-8")
    (store / "2026-10-01.ndjson").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return store, sessions


def _unrec(sid, ts, **extra):
    return {"ts": ts, "session_id": sid, "send_site": STREAM_SITE,
            "components_captured": "unrecorded", **extra}


def test_a_residual_bench_session_with_no_session_file_does_not_land_in_restart_boundary(
        tmp_path, capsys):
    """#1984 clauses 1, 2 and 4, over a scratch store and a scratch sessions dir.

    The live shape on 2026-10-01: fifteen `bench` trials, no session file for any of
    them, printed as `restart_boundary: 15`. Trial traffic is tested BEFORE the
    session store, on the row's `source` field first and the id slug only where the
    row carries no `source`, and the bucket reports its burst window.
    """
    tally_mod = _load_tally()
    by_field = "20260930_200049_worker_aaaa"       # slug says nothing; the row does
    by_slug = "20260930_200050_benchmine_bbbb"     # no `source` on the row at all
    with_file = "20260930_200051_bench_cccc"       # a trial that DOES have a file
    not_trial = "20260930_200052_autonomy_dddd"    # source present and not a trial
    rows = [
        _unrec(by_field, "1790823649.5", source="bench"),
        _unrec(by_field, 1790823700.0, source="bench"),
        _unrec(by_slug, 1790823973.5),
        _unrec(with_file, 1790823800.0, source="bench"),
        _unrec(not_trial, 1790823900.0, source="autonomy"),
    ]
    store, sessions = _tally_fixture(tmp_path, rows, known=[with_file])

    rc = tally_mod.report(str(store), ["2026-10-01"], json_out=False,
                          sessions_dir=str(sessions))
    printed = capsys.readouterr().out
    assert rc == 0, printed
    rep = tally_mod.tally(tally_mod.day_rows(str(store), "2026-10-01"),
                          session_ids=tally_mod.known_sessions(str(sessions)))

    assert rep["attribution"]["trial_traffic"] == sorted([by_field, by_slug, with_file])
    assert rep["attribution"]["restart_boundary"] == []
    assert rep["attribution"]["session_file_absent"] == [not_trial]
    assert rep["attribution"]["unthreaded_send_site"] == []

    # Clause 2: the session count and the burst window, first and last `ts`.
    assert rep["trial_burst"] == {"sessions": 3, "lines": 4,
                                  "first_ts": 1790823649.5, "last_ts": 1790823973.5}
    trial_line = next(ln for ln in printed.splitlines()
                      if ln.strip().startswith("trial_traffic:"))
    assert "3/4 unrecorded sessions" in trial_line, trial_line
    assert "4 lines" in trial_line and "burst " in trial_line, trial_line
    assert tally_mod._stamp(1790823649.5) in trial_line
    assert tally_mod._stamp(1790823973.5) in trial_line

    # Clause 4: no bucket line without the denominator.
    bucket_lines = [ln for ln in printed.splitlines()
                    if ln.strip().split(":")[0] in tally_mod.KINDS]
    assert len(bucket_lines) == len(tally_mod.KINDS), printed
    for ln in bucket_lines:
        assert "/4 unrecorded sessions" in ln, ln


def test_restart_boundary_needs_a_process_death_discriminant(tmp_path, capsys):
    """#1984 clause 3: absence alone is `session_file_absent`; a restart time that
    falls between the session being minted and its first unrecorded line is what
    makes it a restart boundary — and a restart outside that span does not."""
    from datetime import datetime

    tally_mod = _load_tally()
    sid = "20260930_200052_autonomy_dddd"
    minted = datetime.strptime("20260930200052", "%Y%m%d%H%M%S").timestamp()
    assert tally_mod._minted_at(sid) == minted
    rows = [_unrec(sid, minted + 600, source="autonomy")]
    store, sessions = _tally_fixture(tmp_path, rows)
    day = tally_mod.day_rows(str(store), "2026-10-01")

    alone = tally_mod.tally(day, session_ids=set())
    assert alone["attribution"]["session_file_absent"] == [sid]
    assert alone["attribution"]["restart_boundary"] == []

    spanned = tally_mod.tally(day, session_ids=set(), restarts=(minted + 300,))
    assert spanned["attribution"]["restart_boundary"] == [sid]
    assert spanned["attribution"]["session_file_absent"] == []

    for outside in (minted - 300, minted + 900):
        rep = tally_mod.tally(day, session_ids=set(), restarts=(outside,))
        assert rep["attribution"]["restart_boundary"] == [], outside
        assert rep["attribution"]["session_file_absent"] == [sid]

    # A session that HAS a file is never a restart boundary, restart or no restart.
    has_file = tally_mod.tally(day, session_ids={sid}, restarts=(minted + 300,))
    assert has_file["attribution"]["unthreaded_send_site"] == [sid]
    assert has_file["attribution"]["restart_boundary"] == []

    # And the report says why the bucket is empty when no restart was supplied,
    # then fills it through the CLI flag.
    assert tally_mod.main(["2026-10-01", "--store", str(store),
                           "--sessions", str(sessions)]) == 0
    assert "no --restart-at given" in capsys.readouterr().out
    assert tally_mod.main(["2026-10-01", "--store", str(store), "--sessions",
                           str(sessions), "--restart-at", str(minted + 300)]) == 0
    out = capsys.readouterr().out
    assert "restart_boundary: 1/1 unrecorded sessions" in out, out
    assert "no --restart-at given" not in out


def test_the_tally_docstring_calls_the_missing_file_cause_unconfirmed():
    """The script must not commit the sin it fixes: it may name
    `bench_runner_sdk.py`'s swallowed `create_session` as a candidate for a trial's
    missing session file, never as the cause — no runtime witness exists."""
    doc = " ".join((_load_tally().__doc__ or "").split())
    assert "bench_runner_sdk.py" in doc
    assert "CANDIDATE, unconfirmed" in doc
    assert "the known cause" not in doc

